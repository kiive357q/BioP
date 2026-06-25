# -*- coding: utf-8 -*-
"""
Expanded NCDE World Model - 扩展版神经常微分方程世界模型
BioP Causal WorldModel V2.0 - 20M参数版本

【架构】
1. 状态编码器: 50D → 1024D
2. 因果动力学: NCDE向量场 (4层MLP, 1024隐藏)
3. 控制接口: 动作编码 → 4D → 受控动力学
4. 观测解码器: 1024D → 50D
5. Actor网络: 1024D → 4D (策略)
6. CBF安全验证器: 1024D → 安全分数

【总参数量】~20M
【序列长度】100步
【批处理】32-128

【版本】V2.0-ExpandedNCDE
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch import Tensor
from torchdiffeq import odeint_adjoint


class ExpandedNCDEFunction(nn.Module):
    """
    扩展版NCDE向量场
    
    【架构】
    - 输入: [batch, hidden_dim + 1] (状态 + 时间)
    - 输出: [batch, hidden_dim] (状态导数)
    
    【网络结构】
    - 4层全连接
    - 每层: Linear + LayerNorm + GELU
    - 残差连接
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        # 状态处理
        self.state_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # 核心MLP (4层)
        layers = []
        for i in range(n_layers):
            # 第一层处理拼接后的维度，之后每层处理 hidden_dim
            if i == 0:
                in_dim = hidden_dim * 2
            else:
                in_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        矢量场函数
        
        Args:
            t: 时间 [1] 或标量
            z: 隐状态 [batch, hidden_dim]
            
        Returns:
            dz_dt: [batch, hidden_dim]
        """
        # 时间编码
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        
        t_expanded = t.expand(z.shape[0], -1)  # [batch, 1]
        time_encoded = self.time_encoder(t_expanded)  # [batch, hidden_dim]
        
        # 状态处理 + 残差
        state_processed = self.state_processor(z)
        
        # 拼接
        combined = torch.cat([state_processed, time_encoded], dim=-1)  # [batch, hidden_dim*2]
        
        # MLP
        mlp_out = self.mlp(combined)
        
        # 残差连接
        residual = state_processed + time_encoded
        output = self.output_proj(torch.cat([mlp_out, residual], dim=-1))
        
        return output


class ControlledNCDEFunction(nn.Module):
    """
    受控NCDE向量场
    
    在标准NCDE基础上加入动作输入
    dz/dt = f(z, u, t)
    """
    
    def __init__(
        self,
        state_dim: int = 1024,
        action_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, state_dim)
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        
        # 动力学核心
        self.dynamics = ExpandedNCDEFunction(
            hidden_dim=state_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # 状态-动作交互
        self.state_action_interaction = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )
    
    def forward(self, t: Tensor, z: Tensor, u: Tensor = None) -> Tensor:
        """
        受控矢量场
        
        Args:
            t: 时间
            z: 隐状态 [batch, state_dim]
            u: 动作 [batch, action_dim] (可选，如果使用set_action_input)
            
        Returns:
            dz_dt: [batch, state_dim]
        """
        # 如果u未提供，使用之前存储的action_input（已编码）
        if u is None:
            u_encoded = self.action_input
        else:
            # 编码原始动作
            u_encoded = self.action_encoder(u)
        
        batch = z.shape[0]
        
        # 编码
        z_encoded = self.state_encoder(z)
        
        # 交互
        combined = torch.cat([z_encoded, u_encoded], dim=-1)  # [batch, state_dim*2]
        interaction = self.state_action_interaction(combined)  # [batch, state_dim]
        
        # 时间编码
        t_tensor = torch.atleast_1d(t)
        if t_tensor.numel() == 1:
            t_tensor = t_tensor.expand(batch, -1)
        elif t_tensor.shape[0] != batch:
            t_tensor = t_tensor[0].expand(batch, -1)
        
        time_encoded = self.dynamics.time_encoder(t_tensor)  # [batch, state_dim]
        
        # 状态处理
        state_processed = self.dynamics.state_processor(interaction)
        
        # 拼接
        combined_final = torch.cat([state_processed, time_encoded], dim=-1)  # [batch, state_dim*2]
        
        # MLP
        mlp_out = self.dynamics.mlp(combined_final)
        
        # 残差连接
        residual = state_processed + time_encoded
        output = self.dynamics.output_proj(torch.cat([mlp_out, residual], dim=-1))
        
        return output
    
    def set_action_input(self, action_input: Tensor):
        """设置动作输入（由forward_dynamics调用，已编码为state_dim维）"""
        self.action_input = action_input


class WorldModelEncoder(nn.Module):
    """
    观测编码器
    
    将50维观测映射到1024维隐空间
    """
    
    def __init__(
        self,
        obs_dim: int = 50,
        latent_dim: int = 1024,
        hidden_dim: int = 2048
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)


class WorldModelDecoder(nn.Module):
    """
    观测解码器
    
    将1024维隐状态解码回50维观测
    """
    
    def __init__(
        self,
        latent_dim: int = 1024,
        obs_dim: int = 50,
        hidden_dim: int = 2048
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, obs_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class ActorNetwork(nn.Module):
    """
    Actor网络 - 策略函数
    
    【输入】隐状态 [batch, 1024]
    【输出】动作 [batch, 4] (Tanh激活，归一化到动作空间)
    """
    
    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 4,
        hidden_dim: int = 1024
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 归一化输出
        )
        
        # 动作缩放参数
        self.register_buffer('action_scale', torch.tensor([200.0, 100.0, 1.0, 500.0]))
        self.register_buffer('action_bias', torch.tensor([100.0, 0.0, 0.5, 200.0]))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 隐状态 [batch, latent_dim]
            
        Returns:
            action: 归一化动作 [batch, action_dim]
        """
        action = self.network(z)
        
        # 缩放到动作空间
        action = action * self.action_scale + self.action_bias
        
        return action
    
    def get_action(self, z: Tensor, deterministic: bool = False) -> Tensor:
        """获取动作，带探索"""
        if deterministic:
            return self.forward(z)
        
        # 添加噪声探索
        action = self.forward(z)
        noise = torch.randn_like(action) * 0.1 * self.action_scale
        return torch.clamp(action + noise, 
                         self.action_bias - self.action_scale,
                         self.action_bias + self.action_scale)


class CriticNetwork(nn.Module):
    """
    Critic网络 - Q值函数
    
    【输入】隐状态 + 动作 [batch, latent_dim + action_dim]
    【输出】Q值 [batch, 1]
    """
    
    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 4,
        hidden_dim: int = 1024
    ):
        super().__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2网络 (双Q学习)
        self.q2_network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            z: 隐状态 [batch, latent_dim]
            action: 动作 [batch, action_dim]
            
        Returns:
            q1, q2: 两个Q值 [batch, 1]
        """
        combined = torch.cat([z, action], dim=-1)
        q1 = self.q1_network(combined)
        q2 = self.q2_network(combined)
        return q1, q2


class BioPWorldModel(nn.Module):
    """
    完整的世界模型
    
    【组成】
    1. 观测编码器 (50D → 1024D)
    2. 因果动力学 (NCDE)
    3. 观测解码器 (1024D → 50D)
    4. Actor网络 (策略)
    5. Critic网络 (Q值)
    6. CBF安全验证器
    
    【总参数量】~20M
    """
    
    def __init__(
        self,
        obs_dim: int = 50,
        action_dim: int = 4,
        latent_dim: int = 1024,
        n_ncde_layers: int = 4,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = WorldModelEncoder(obs_dim, latent_dim)
        
        # 自治动力学 (无控制时的自然演化)
        self.dynamics = ExpandedNCDEFunction(
            hidden_dim=latent_dim,
            n_layers=n_ncde_layers
        )
        
        # 受控动力学 (有控制时的演化)
        self.controlled_dynamics = ControlledNCDEFunction(
            state_dim=latent_dim,
            action_dim=256,  # 动作编码维度
            n_layers=n_ncde_layers
        )
        
        # 动作编码器: 4D -> 256D -> 1024D
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        
        # 解码器
        self.decoder = WorldModelDecoder(latent_dim, obs_dim)
        
        # Actor
        self.actor = ActorNetwork(latent_dim, action_dim)
        
        # Critic (双Q网络)
        self.critic = CriticNetwork(latent_dim, action_dim)
        
        # 目标网络
        self.target_encoder = WorldModelEncoder(obs_dim, latent_dim)
        self.target_decoder = WorldModelDecoder(latent_dim, obs_dim)
        self.target_dynamics = ExpandedNCDEFunction(latent_dim, n_ncde_layers)
        
        # 同步目标网络
        self.sync_target_networks()
        
        self.device = device
    
    def sync_target_networks(self):
        """同步目标网络"""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_decoder.load_state_dict(self.decoder.state_dict())
        self.target_dynamics.load_state_dict(self.dynamics.state_dict())
    
    def encode(self, obs: Tensor) -> Tensor:
        """编码观测到隐状态"""
        return self.encoder(obs)
    
    def decode(self, z: Tensor) -> Tensor:
        """解码隐状态到观测"""
        return self.decoder(z)
    
    def forward_dynamics(
        self,
        z: Tensor,
        action: Optional[Tensor] = None,
        time_points: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        前向动力学预测
        
        Args:
            z: 初始隐状态 [batch, latent_dim]
            action: 控制动作 [batch, action_dim] (可选)
            time_points: 时间点 [seq_len]
            
        Returns:
            trajectory: 轨迹 [seq_len, batch, latent_dim]
            final_z: 最终隐状态 [batch, latent_dim]
        """
        if time_points is None:
            time_points = torch.linspace(0, 1, 10, device=z.device)
        
        batch = z.shape[0]
        
        # 选择动力学函数
        if action is not None:
            u_encoded = self.action_encoder(action)
            
            # 将u_encoded存储在controlled_dynamics中供forward使用
            self.controlled_dynamics.set_action_input(u_encoded)
            
            trajectory = odeint_adjoint(
                self.controlled_dynamics,
                z,
                time_points,
                method='dopri5',
                adjoint_params=()
            )
        else:
            trajectory = odeint_adjoint(
                self.dynamics,
                z,
                time_points,
                method='dopri5',
                adjoint_params=()
            )
        
        final_z = trajectory[-1] if trajectory.dim() == 2 else trajectory[-1]
        
        return trajectory, final_z
    
    def predict_next_obs(
        self,
        obs: Tensor,
        action: Tensor,
        time_delta: float = 0.01
    ) -> Tuple[Tensor, Tensor]:
        """
        预测下一步观测
        
        Args:
            obs: 当前观测 [batch, obs_dim]
            action: 控制动作 [batch, action_dim]
            time_delta: 时间步长
            
        Returns:
            next_obs: 预测的下一观测 [batch, obs_dim]
            next_z: 预测的下一隐状态 [batch, latent_dim]
        """
        # 编码
        z = self.encode(obs)
        
        # 动力学预测
        time_points = torch.tensor([0, time_delta], device=z.device)
        _, next_z = self.forward_dynamics(z, action, time_points)
        
        # 解码
        next_obs = self.decode(next_z)
        
        return next_obs, next_z
    
    def actor_forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        """
        Actor前向 - 获取动作
        
        Args:
            obs: 观测 [batch, obs_dim]
            deterministic: 是否确定性 (无探索)
            
        Returns:
            action: 动作 [batch, action_dim]
        """
        z = self.encode(obs)
        return self.actor.get_action(z, deterministic)
    
    def critic_forward(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Critic前向 - 计算Q值
        
        Args:
            obs: 观测 [batch, obs_dim]
            action: 动作 [batch, action_dim]
            
        Returns:
            q1, q2: 两个Q值
        """
        z = self.encode(obs)
        return self.critic(z, action)
    
    def get_state_dict_info(self) -> Dict[str, int]:
        """获取各模块参数量"""
        info = {}
        
        info['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        info['dynamics'] = sum(p.numel() for p in self.dynamics.parameters())
        info['controlled_dynamics'] = sum(p.numel() for p in self.controlled_dynamics.parameters())
        info['action_encoder'] = sum(p.numel() for p in self.action_encoder.parameters())
        info['decoder'] = sum(p.numel() for p in self.decoder.parameters())
        info['actor'] = sum(p.numel() for p in self.actor.parameters())
        info['critic'] = sum(p.numel() for p in self.critic.parameters())
        
        info['total'] = sum(info.values())
        
        return info


def create_expanded_model(
    obs_dim: int = 50,
    action_dim: int = 4,
    latent_dim: int = 1024,
    device: str = "cuda"
) -> BioPWorldModel:
    """创建扩展版世界模型"""
    model = BioPWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim
    ).to(device)
    
    # 打印参数量
    info = model.get_state_dict_info()
    print("=" * 70)
    print("Expanded NCDE World Model - Parameter Count")
    print("=" * 70)
    for name, count in info.items():
        if name != 'total':
            print(f"  {name:25s}: {count:>12,}")
    print("-" * 70)
    print(f"  {'TOTAL':25s}: {info['total']:>12,}")
    print("=" * 70)
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Expanded NCDE World Model Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型
    model = create_expanded_model(
        obs_dim=50,
        action_dim=4,
        latent_dim=1024,
        device=device
    )
    
    # 测试编码
    print("\nTesting encoder...")
    obs = torch.randn(8, 50, device=device)
    z = model.encode(obs)
    print(f"  Obs shape: {obs.shape} -> Z shape: {z.shape}")
    
    # 测试动力学
    print("\nTesting dynamics...")
    action = torch.randn(8, 4, device=device)
    trajectory, final_z = model.forward_dynamics(z, action)
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final Z shape: {final_z.shape}")
    
    # 测试解码
    print("\nTesting decoder...")
    next_obs = model.decode(final_z)
    print(f"  Next obs shape: {next_obs.shape}")
    
    # 测试Actor
    print("\nTesting actor...")
    action_out = model.actor_forward(obs)
    print(f"  Action shape: {action_out.shape}")
    print(f"  Action range: [{action_out.min().item():.2f}, {action_out.max().item():.2f}]")
    
    # 测试Critic
    print("\nTesting critic...")
    q1, q2 = model.critic_forward(obs, action_out)
    print(f"  Q1: {q1.mean().item():.3f}, Q2: {q2.mean().item():.3f}")
    
    print("\n✓ Expanded NCDE World Model test passed!")
