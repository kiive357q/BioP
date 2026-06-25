"""
阶段一：NCDE因果世界模型主训练脚本 (training/train_phase1_ncde.py)

【模块定位】BioP因果世界模型的阶段一训练入口
【设计理念】工业级混合精度训练 + 物理约束 + 因果去偏

【训练特性】
- 混合精度训练 (AMP): torch.cuda.amp.autocast + GradScaler
- 梯度防爆: clip_grad_norm_(max_norm=1.0)
- 早停机制: EarlyStopping with patience=10
- 物理约束: PINN损失函数
- 因果去偏: IPW加权

【版本】V2.0-Phase4-TrainingPipeline
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloaders import WastewaterDataset, create_rolling_windows_tensor
from data.cubic_spline_interp import DifferentiableCubicSpline
from data.ipw_confounder import IPWLossWeighter
from models.ncde_solver import NCDEFunction, NCDESolver
from models.sindy_library import SINDyLibrary
from models.pinn_loss import BioPCompositeLoss, PINNLossConfig, PhysicsConstants


class NCDEWrapper(nn.Module):
    """NCDE模型封装器，适配训练接口"""
    
    def __init__(self, ncde_solver: NCDESolver, time_points: Optional[Tensor] = None):
        super().__init__()
        self.ncde_solver = ncde_solver
        self.register_buffer('time_points', time_points if time_points is not None else torch.linspace(0, 1, 20))
    
    def forward(self, states: Tensor) -> Tensor:
        """前向传播
        
        Args:
            states: [batch, seq_len, state_dim] 输入状态序列
            
        Returns:
            trajectory: [batch, seq_len, state_dim] 预测轨迹
        """
        batch_size, seq_len, state_dim = states.shape
        
        initial_state = states[:, 0, :]
        
        time_points = self.time_points[:seq_len].to(states.device)
        
        trajectory, _ = self.ncde_solver(
            initial_state=initial_state,
            time_points=time_points,
            return_hidden=False
        )
        
        # NCDESolver返回 [time, batch, dim]，需要转换为 [batch, time, dim]
        if trajectory.dim() == 3 and trajectory.shape[0] != batch_size:
            trajectory = trajectory.permute(1, 0, 2)
        
        # 确保维度匹配
        trajectory = trajectory[:, :, :state_dim]
        
        if trajectory.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - trajectory.shape[1], state_dim,
                device=trajectory.device, dtype=trajectory.dtype
            )
            trajectory = torch.cat([trajectory, padding], dim=1)
        elif trajectory.shape[1] > seq_len:
            trajectory = trajectory[:, :seq_len, :]
        
        return trajectory


@dataclass
class TrainingConfig:
    """训练配置"""
    model_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_path: str = "./data/processed"
    
    state_dim: int = 20
    hidden_dim: int = 128
    output_dim: int = 20
    n_layers: int = 4
    
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    epochs: int = 100
    val_split: float = 0.2
    
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    
    use_amp: bool = True
    use_ipw: bool = True
    use_physics_loss: bool = True
    
    save_interval: int = 5
    log_interval: int = 10


class EarlyStopping:
    """
    早停机制
    
    【策略】监控验证损失，连续patience个epoch没有改善则停止训练
    
    Args:
        patience: 容忍epoch数
        min_delta: 最小改善量
        mode: 'min'或'max'
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: optim.Optimizer
    ) -> bool:
        """
        判断是否应该早停
        
        Args:
            val_loss: 当前验证损失
            model: 模型（用于保存最佳状态）
            optimizer: 优化器
            
        Returns:
            True if should early stop, False otherwise
        """
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_model_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            self.counter = 0
        
        return False


class MetricsTracker:
    """
    训练指标追踪器
    
    【功能】记录和汇总每个epoch的训练/验证指标
    """
    
    def __init__(self) -> None:
        self.train_losses = []
        self.val_losses = []
        self.train_data_losses = []
        self.train_physics_losses = []
        self.val_data_losses = []
        self.val_physics_losses = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_data_loss: float,
        train_physics_loss: float,
        val_data_loss: float,
        val_physics_loss: float,
        learning_rate: float,
        epoch_time: float
    ) -> None:
        """更新指标"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_data_losses.append(train_data_loss)
        self.train_physics_losses.append(train_physics_loss)
        self.val_data_losses.append(val_data_loss)
        self.val_physics_losses.append(val_physics_loss)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'best_val_loss': min(self.val_losses) if self.val_losses else float('inf'),
            'best_epoch': np.argmin(self.val_losses) + 1 if self.val_losses else 0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'total_epochs': len(self.train_losses),
            'total_training_time': sum(self.epoch_times)
        }
    
    def save(self, filepath: str) -> None:
        """保存指标到文件"""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump({
                'train_losses': [convert(x) for x in self.train_losses],
                'val_losses': [convert(x) for x in self.val_losses],
                'train_data_losses': [convert(x) for x in self.train_data_losses],
                'train_physics_losses': [convert(x) for x in self.train_physics_losses],
                'val_data_losses': [convert(x) for x in self.val_data_losses],
                'val_physics_losses': [convert(x) for x in self.val_physics_losses],
                'learning_rates': [convert(x) for x in self.learning_rates],
                'epoch_times': [convert(x) for x in self.epoch_times],
                'summary': {k: convert(v) for k, v in self.get_summary().items()}
            }, f, indent=2)


def setup_data_loaders(
    config: TrainingConfig,
    device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    设置数据加载器
    
    【流水线】
    1. 加载原始SCADA数据
    2. 生成滚动窗口张量
    3. 划分训练/验证/测试集
    4. 创建DataLoader
    
    Args:
        config: 训练配置
        device: 计算设备
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_path = Path(config.data_path)
    
    import pandas as pd
    import json
    
    preprocessed_files = list(data_path.glob("preprocessed_phase_*.csv"))
    preprocessed_files.extend(list(data_path.parent.glob("preprocessed_phase_*.csv")))
    
    csv_search_paths = [data_path, data_path.parent]
    
    csv_files = []
    for search_path in csv_search_paths:
        csv_files = list(search_path.glob("*.csv"))
        if csv_files:
            data_path = search_path
            break
    
    if preprocessed_files:
        import pandas as pd
        print(f"[数据加载] 使用预处理数据: {preprocessed_files[0].name}")
        
        df = pd.read_csv(preprocessed_files[0], parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 获取所有数值列
        numeric_cols = [c for c in df.columns if c != 'date' and not c.startswith('Unnamed') 
                       and not c.startswith('process_phase')]
        
        # 只选择主要特征和关键派生特征
        main_features = ['T1_O2', 'T1_NH4', 'T1_PO4', 'IN_Q', 'TEMPERATURE', 'METAL_Q']
        available_main = [c for c in main_features if c in numeric_cols]
        
        # 添加最重要的派生特征（差分和滞后）
        derived = [c for c in numeric_cols if '_diff' in c or ('_lag' in c and 'lag1' in c)]
        available_derived = derived[:10]  # 限制数量
        
        all_features = available_main + available_derived
        all_features = [c for c in all_features if c in numeric_cols]
        
        data_array = df[all_features].values
        
        n_features = len(all_features)
        print(f"[数据加载] 特征数: {n_features}, 数据点: {len(df):,}")
        print(f"[数据加载] 特征: {all_features[:6]}...")
        
        config.state_dim = n_features
        config.hidden_dim = max(128, n_features * 8)
        
        all_data = torch.tensor(data_array, dtype=torch.float32)
        
        window_size = 50
        seq_len = 50
        pred_horizon = 50
        
        split_1 = int(len(all_data) * 0.7)
        split_2 = int(len(all_data) * 0.85)
        
        train_data = all_data[:split_1]
        val_data = all_data[split_1:split_2]
        test_data = all_data[split_2:]
        
        train_dataset = WastewaterDataset(train_data, seq_len=seq_len, pred_horizon=pred_horizon)
        val_dataset = WastewaterDataset(val_data, seq_len=seq_len, pred_horizon=pred_horizon)
        test_dataset = WastewaterDataset(test_data, seq_len=seq_len, pred_horizon=pred_horizon)
        
    elif csv_files:
        csv_path = csv_files[0]
        print(f"[数据加载] 使用 CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)
        df = df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df

        numeric_cols = [c for c in df.columns if c != 'date' and not c.startswith('Unnamed')
                        and not c.startswith('process_phase')]
        main_features = ['T1_O2', 'T1_NH4', 'T1_PO4', 'IN_Q', 'TEMPERATURE', 'METAL_Q']
        available_main = [c for c in main_features if c in numeric_cols]
        derived = [c for c in numeric_cols if '_diff' in c or ('_lag' in c and 'lag1' in c)]
        available_derived = derived[:10]
        all_features = [c for c in (available_main + available_derived) if c in numeric_cols]

        if not all_features:
            all_features = numeric_cols[: min(10, len(numeric_cols))]

        data_array = df[all_features].values.astype(np.float32)

        n_features = len(all_features)
        config.state_dim = n_features
        config.hidden_dim = max(128, n_features * 8)

        all_data = torch.tensor(data_array, dtype=torch.float32)

        window_size = 50
        seq_len = 50
        pred_horizon = 50

        split_1 = int(len(all_data) * 0.7)
        split_2 = int(len(all_data) * 0.85)

        train_data = all_data[:split_1]
        val_data = all_data[split_1:split_2]
        test_data = all_data[split_2:]

        train_dataset = WastewaterDataset(train_data, seq_len=seq_len, pred_horizon=pred_horizon)
        val_dataset = WastewaterDataset(val_data, seq_len=seq_len, pred_horizon=pred_horizon)
        test_dataset = WastewaterDataset(test_data, seq_len=seq_len, pred_horizon=pred_horizon)
    elif data_path.exists() and list(data_path.glob("*.pt")):
        print(f"[数据加载] 从 {data_path} 加载预处理数据")
        
        train_data = torch.load(data_path / "train_data.pt")
        val_data = torch.load(data_path / "val_data.pt")
        test_data = torch.load(data_path / "test_data.pt")
        
        window_size = train_data.shape[1]
        
        train_dataset = WastewaterDataset(train_data, window_size=window_size)
        val_dataset = WastewaterDataset(val_data, window_size=window_size)
        test_dataset = WastewaterDataset(test_data, window_size=window_size)
    else:
        print(f"[数据生成] 生成模拟训练数据")
        
        n_timesteps = 5000
        n_features = config.state_dim
        
        all_data = torch.randn(n_timesteps, n_features)
        
        seq_len = 20
        pred_horizon = 20
        
        split_1 = int(n_timesteps * 0.7)
        split_2 = int(n_timesteps * 0.85)
        
        train_data = all_data[:split_1]
        val_data = all_data[split_1:split_2]
        test_data = all_data[split_2:]
        
        train_dataset = WastewaterDataset(train_data, seq_len=seq_len, pred_horizon=pred_horizon)
        val_dataset = WastewaterDataset(val_data, seq_len=seq_len, pred_horizon=pred_horizon)
        test_dataset = WastewaterDataset(test_data, seq_len=seq_len, pred_horizon=pred_horizon)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[数据加载] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def setup_model(
    config: TrainingConfig,
    device: torch.device
) -> Tuple[NCDEWrapper, SINDyLibrary, BioPCompositeLoss]:
    """
    设置模型组件
    
    【组件】
    1. NeuralCDE: 神经控制微分方程
    2. SINDyLibrary: 稀疏特征库
    3. CompositeLoss: 复合损失函数
    
    Args:
        config: 训练配置
        device: 计算设备
        
    Returns:
        (model, sindy_library, criterion)
    """
    print(f"[模型初始化] 构建NCDE模型")
    
    vector_field = NCDEFunction(
        hidden_dim=config.hidden_dim,
        hidden_layers=config.n_layers,
        hidden_width=config.hidden_dim
    )
    
    model = NCDESolver(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        solver="dopri5"
    ).to(device)
    
    model = NCDEWrapper(model)
    
    sindy_library = SINDyLibrary(
        n_state_variables=config.state_dim,
        poly_order=3
    )
    
    pinn_config = PINNLossConfig(
        data_loss_weight=1.0,
        physics_loss_weight=0.5 if config.use_physics_loss else 0.0,
        ipw_loss_weight=0.3 if config.use_ipw else 0.0
    )
    
    physics_constants = PhysicsConstants()
    
    criterion = BioPCompositeLoss(
        config=pinn_config,
        physics_constants=physics_constants,
        device=device
    )
    
    ipw_weighter = None
    if config.use_ipw:
        ipw_weighter = IPWLossWeighter(
            weighting_strategy="standard",
            weight_ceiling=10.0,
            epsilon=1e-4
        )
    
    print(f"[模型初始化] 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, sindy_library, criterion


def train_epoch(
    model: NCDEWrapper,
    train_loader: DataLoader,
    criterion: BioPCompositeLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
    use_amp: bool = True
) -> Tuple[float, float, float]:
    """
    训练一个epoch
    
    【混合精度训练流程】
    1. 前向传播 (autocast)
    2. 计算损失
    3. 反向传播 (scaler.scale)
    4. 梯度裁剪
    5. 参数更新 (scaler.step)
    6. scaler更新
    
    Args:
        model: NCDE模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 梯度缩放器
        device: 计算设备
        config: 训练配置
        use_amp: 是否使用混合精度
        
    Returns:
        (total_loss, data_loss, physics_loss)
    """
    model.train()
    
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    n_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if isinstance(batch_data, (list, tuple)):
            states, targets = batch_data
            states = states.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dim() == 2:
                targets = targets.unsqueeze(1)
            targets = targets.repeat(1, states.shape[1] // targets.shape[1] + 1, 1)[:, :states.shape[1], :]
            inputs = None
        else:
            states = batch_data['states'].to(device, non_blocking=True)
            targets = batch_data['targets'].to(device, non_blocking=True)
            inputs = batch_data.get('inputs', None)
            if inputs is not None:
                inputs = inputs.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast():
                pred_states = model(states)
                loss, loss_dict = criterion(
                    pred_states, targets, inputs, time_delta=1.0
                )
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_states = model(states)
            loss, loss_dict = criterion(
                pred_states, targets, inputs, time_delta=1.0
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm
            )
            optimizer.step()
        
        total_loss_sum += loss.item()
        data_loss_sum += loss_dict['mse_loss'].item()
        physics_loss_sum += loss_dict['physics_loss'].item()
        n_batches += 1
        
        if batch_idx % config.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item():.6f}, "
                  f"Data={loss_dict['mse_loss'].item():.6f}, "
                  f"Physics={loss_dict['physics_loss'].item():.6f}")
    
    avg_total_loss = total_loss_sum / n_batches
    avg_data_loss = data_loss_sum / n_batches
    avg_physics_loss = physics_loss_sum / n_batches
    
    return avg_total_loss, avg_data_loss, avg_physics_loss


def validate(
    model: NCDEWrapper,
    val_loader: DataLoader,
    criterion: BioPCompositeLoss,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float, float]:
    """
    验证模型
    
    Args:
        model: NCDE模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        use_amp: 是否使用混合精度
        
    Returns:
        (total_loss, data_loss, physics_loss)
    """
    model.eval()
    
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, (list, tuple)):
                states, targets = batch_data
                states = states.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dim() == 2:
                    targets = targets.unsqueeze(1)
                targets = targets.repeat(1, states.shape[1] // targets.shape[1] + 1, 1)[:, :states.shape[1], :]
                inputs = None
            else:
                states = batch_data['states'].to(device, non_blocking=True)
                targets = batch_data['targets'].to(device, non_blocking=True)
                inputs = batch_data.get('inputs', None)
                if inputs is not None:
                    inputs = inputs.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    pred_states = model(states)
                    loss, loss_dict = criterion(
                        pred_states, targets, inputs, time_delta=1.0
                    )
            else:
                pred_states = model(states)
                loss, loss_dict = criterion(
                    pred_states, targets, inputs, time_delta=1.0
                )
            
            total_loss_sum += loss.item()
            data_loss_sum += loss_dict['mse_loss'].item()
            physics_loss_sum += loss_dict['physics_loss'].item()
            n_batches += 1
    
    avg_total_loss = total_loss_sum / n_batches
    avg_data_loss = data_loss_sum / n_batches
    avg_physics_loss = physics_loss_sum / n_batches
    
    return avg_total_loss, avg_data_loss, avg_physics_loss


def train(
    config: TrainingConfig,
    device: torch.device
) -> Dict[str, Any]:
    """
    主训练函数
    
    【流程】
    1. 设置数据加载器
    2. 设置模型和优化器
    3. 训练循环 + 早停
    4. 保存最佳模型
    
    Args:
        config: 训练配置
        device: 计算设备
        
    Returns:
        training_metrics: 训练指标字典
    """
    print("=" * 70)
    print("BioP因果世界模型 - 阶段一训练")
    print("=" * 70)
    
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"train_{timestamp}.log")
    
    train_loader, val_loader, test_loader = setup_data_loaders(config, device)
    
    model, sindy_library, criterion = setup_model(config, device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    scaler = GradScaler() if config.use_amp else None
    
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=1e-5,
        mode='min'
    )
    
    metrics_tracker = MetricsTracker()
    
    best_val_loss = float('inf')
    
    print(f"\n【训练配置】")
    print(f"设备: {device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print(f"混合精度: {config.use_amp}")
    print(f"物理损失: {config.use_physics_loss}")
    print(f"IPW去偏: {config.use_ipw}")
    print(f"梯度裁剪: {config.gradient_clip_norm}")
    print(f"早停耐心: {config.early_stopping_patience}")
    
    print(f"\n【开始训练】")
    
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.perf_counter()
        
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, config.use_amp
        )
        
        val_loss, val_data_loss, val_physics_loss = validate(
            model, val_loader, criterion, device, config.use_amp
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.perf_counter() - epoch_start_time
        
        metrics_tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_data_loss=train_data_loss,
            train_physics_loss=train_physics_loss,
            val_data_loss=val_data_loss,
            val_physics_loss=val_physics_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time
        )
        
        print(f"\nEpoch {epoch}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  [Train] Total={train_loss:.6f}, Data={train_data_loss:.6f}, Physics={train_physics_loss:.6f}")
        print(f"  [Val]   Total={val_loss:.6f}, Data={val_data_loss:.6f}, Physics={val_physics_loss:.6f}")
        print(f"  [LR]    {current_lr:.2e}")
        
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.model_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(config)
            }, checkpoint_path)
            print(f"  [保存] Checkpoint -> {checkpoint_path}")
        
        if early_stopping(val_loss, model, optimizer):
            print(f"\n[早停] 验证损失连续{config.early_stopping_patience}个epoch未改善")
            print(f"[早停] 最佳epoch: {np.argmin(metrics_tracker.val_losses) + 1}")
            print(f"[早停] 最佳验证损失: {min(metrics_tracker.val_losses):.6f}")
            break
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.model_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': early_stopping.best_model_state['model'],
                'optimizer_state_dict': early_stopping.best_model_state['optimizer'],
                'val_loss': val_loss
            }, best_model_path)
            print(f"  [最佳] New best model saved!")
    
    summary = metrics_tracker.get_summary()
    print(f"\n【训练完成】")
    print(f"最佳验证损失: {summary['best_val_loss']:.6f}")
    print(f"最佳epoch: {summary['best_epoch']}")
    print(f"总训练时间: {summary['total_training_time']:.1f}s")
    
    metrics_path = os.path.join(config.log_dir, f"metrics_{timestamp}.json")
    metrics_tracker.save(metrics_path)
    print(f"[保存] 训练指标 -> {metrics_path}")
    
    return summary


def parse_args() -> TrainingConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BioP因果世界模型阶段一训练"
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/processed',
        help='数据路径'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./checkpoints',
        help='模型保存路径'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='日志保存路径'
    )
    
    parser.add_argument(
        '--state_dim',
        type=int,
        default=20,
        help='状态维度'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='隐藏层维度'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='网络层数'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='批大小'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练epoch数'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学习率'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='权重衰减'
    )
    
    parser.add_argument(
        '--no_amp',
        action='store_true',
        help='禁用混合精度训练'
    )
    parser.add_argument(
        '--no_physics_loss',
        action='store_true',
        help='禁用物理损失'
    )
    parser.add_argument(
        '--no_ipw',
        action='store_true',
        help='禁用IPW去偏'
    )
    
    parser.add_argument(
        '--gradient_clip',
        type=float,
        default=1.0,
        help='梯度裁剪阈值'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='早停耐心值'
    )
    
    parser.add_argument(
        '--save_interval',
        type=int,
        default=5,
        help='模型保存间隔'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='日志打印间隔'
    )
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_path=args.data_path,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        use_physics_loss=not args.no_physics_loss,
        use_ipw=not args.no_ipw,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )
    
    return config


if __name__ == "__main__":
    """【测试入口】主训练脚本"""
    print("=" * 70)
    print("BioP因果世界模型 - 阶段一训练脚本")
    print("=" * 70)
    
    config = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[显存] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n[启动训练]")
    print(f"Dry-run 测试: epochs={config.epochs}")
    
    summary = train(config, device)
    
    print(f"\n" + "=" * 70)
    print("✓ 训练完成")
    print("=" * 70)
    
    print(f"\n【训练摘要】")
    print(f"最佳验证损失: {summary['best_val_loss']:.6f}")
    print(f"最佳epoch: {summary['best_epoch']}")
    print(f"总epochs: {summary['total_epochs']}")
    print(f"总训练时间: {summary['total_training_time']:.1f}s")
    
    print(f"\n【使用说明】")
    print(f"完整训练: python training/train_phase1_ncde.py --epochs 100")
    print(f"无物理损失: python training/train_phase1_ncde.py --no_physics_loss")
    print(f"大批量: python training/train_phase1_ncde.py --batch_size 512")
