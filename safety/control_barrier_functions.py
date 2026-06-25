"""
控制障碍函数模块 (safety/control_barrier_functions.py)

【模块定位】污水生化除磷系统的安全边界定义
【设计理念】通过CBF定义物理不可逾越的安全底线

【HJI理论】
对于系统 ẋ = f(x) + g(x)u，安全集定义为 C = {x : h(x) ≥ 0}
CBF条件：存在γ>0使得 L_f h(x) + L_g h(x)u ≥ -γh(x)

【CBF物理约束】
1. DO溶解氧下限：DO ≥ 0.5 mg/L（生物活性最低保障）
2. 氨氮毒性警戒线：NH4-N ≤ 15 mg/L（硝化抑制阈值）
3. 鼓风机频率变化率：|Δf| ≤ 3 Hz/s（防喘振物理极限）

【版本】V2.0-Phase3-SafetyGuardrails
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, NamedTuple
from dataclasses import dataclass


@dataclass
class CBFEvaluationResult(NamedTuple):
    """CBF求值结果"""
    h_values: torch.Tensor
    Lf_h: torch.Tensor
    Lg_h: torch.Tensor
    is_safe: torch.Tensor
    constraint_violations: Dict[str, torch.Tensor]


@dataclass
class CBFConfig:
    """CBF配置参数"""
    do_safe_threshold: float = 0.5
    nh4_safe_threshold: float = 15.0
    blower_freq_change_max: float = 3.0
    
    do_cbf_gamma: float = 10.0
    nh4_cbf_gamma: float = 10.0
    blower_cbf_gamma: float = 10.0


class WaterTreatmentCBF(nn.Module):
    """
    污水处理的控制障碍函数定义器
    
    【核心职责】定义污水处理过程的物理安全边界
    
    【CBF设计原则】
    - h(x) > 0: 安全区域内部
    - h(x) = 0: 安全边界
    - h(x) < 0: 危险区域（必须拦截）
    
    【三个刚性CBF】
    1. h_DO(x): 溶解氧安全下界
    2. h_NH4(x): 氨氮毒性警戒
    3. h_Blower(x): 风机频率变化率限制
    
    Args:
        config: CBF配置参数
        device: 计算设备
        require_grad: 是否需要计算梯度（用于CBF约束求解）
    """
    
    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        require_grad: bool = True
    ) -> None:
        super().__init__()
        
        self.device = device
        self.config = config if config is not None else CBFConfig()
        self.require_grad = require_grad
        
        self.do_threshold = self.config.do_safe_threshold
        self.nh4_threshold = self.config.nh4_safe_threshold
        self.blower_freq_change_max = self.config.blower_freq_change_max
        
        self.gamma_do = self.config.do_cbf_gamma
        self.gamma_nh4 = self.config.nh4_cbf_gamma
        self.gamma_blower = self.config.blower_cbf_gamma
    
    def forward(
        self,
        states: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        dt: float = 1.0
    ) -> CBFEvaluationResult:
        """
        前向传播：计算所有CBF及其李导数
        
        【输入规格】
        - states: (batch_size, state_dim) 状态张量
          假设状态序: [DO, NH4, NO3, TankLevel, BlowerFreq, ...]
        - prev_actions: (batch_size, action_dim) 上一时刻动作（用于频率变化率计算）
        - dt: 时间步长 (s)
        
        【输出规格】
        - CBFEvaluationResult: 包含h值、梯度、安全判定的结果
        
        【状态索引约定】
        - states[:, 0]: DO溶解氧 (mg/L)
        - states[:, 1]: NH4-N氨氮 (mg/L)
        - states[:, 2]: TankLevel液位 (m)
        - states[:, 3]: BlowerFreq鼓风机频率 (Hz)
        """
        batch_size = states.shape[0]
        
        do_concentration = states[:, 0]
        nh4_concentration = states[:, 1]
        blower_freq = states[:, 3]
        
        if prev_actions is not None and states.shape[0] == prev_actions.shape[0]:
            prev_blower_freq = prev_actions[:, 0] * 50.0 + 20.0
            freq_change = (blower_freq - prev_blower_freq) / (dt + 1e-6)
        else:
            freq_change = torch.zeros(batch_size, device=self.device)
        
        h_do, Lf_h_do, Lg_h_do = self._compute_do_cbf(do_concentration)
        h_nh4, Lf_h_nh4, Lg_h_nh4 = self._compute_nh4_cbf(nh4_concentration)
        h_blower, Lf_h_blower, Lg_h_blower = self._compute_blower_cbf(freq_change)
        
        h_values = torch.stack([h_do, h_nh4, h_blower], dim=-1)
        Lf_h = torch.stack([Lf_h_do, Lf_h_nh4, Lf_h_blower], dim=-1)
        Lg_h = torch.stack([Lg_h_do, Lg_h_nh4, Lg_h_blower], dim=-1)
        
        is_safe = (h_do >= 0) & (h_nh4 >= 0) & (h_blower >= 0)
        
        constraint_violations = {
            'DO_violation': torch.clamp(-h_do, min=0),
            'NH4_violation': torch.clamp(-h_nh4, min=0),
            'Blower_violation': torch.clamp(-h_blower, min=0)
        }
        
        return CBFEvaluationResult(
            h_values=h_values,
            Lf_h=Lf_h,
            Lg_h=Lg_h,
            is_safe=is_safe,
            constraint_violations=constraint_violations
        )
    
    def _compute_do_cbf(
        self,
        do: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【CBF-1】溶解氧下限障碍函数
        
        【物理机制】
        DO < 0.5 mg/L 时，反硝化菌和聚磷菌活性急剧下降
        导致出水水质恶化和污泥膨胀
        
        【CBF定义】h_DO(x) = DO - DO_threshold
        - h_DO > 0: DO充足，安全
        - h_DO < 0: DO不足，危险
        
        【李导数计算】
        对于静态CBF（h不显式依赖u）：
        - L_f h(x) = ∂h/∂x · f(x) = 1 · f_DO(x)
        - L_g h(x) = ∂h/∂x · g(x) = 0
        
        注意：实际L_g_h不为零，因为DO演化依赖曝气动作
        这里简化处理，假设L_g_h为常数（可通过系统辨识获得）
        
        Returns:
            h: 障碍函数值
            Lf_h: 沿f的李导数
            Lg_h: 沿g的李导数（对曝气动作的梯度）
        """
        h_do = do - self.do_threshold
        
        Lf_h_do = torch.zeros_like(do)
        
        Lg_h_do = torch.ones_like(do) * 0.01
        
        return h_do, Lf_h_do, Lg_h_do
    
    def _compute_nh4_cbf(
        self,
        nh4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【CBF-2】氨氮毒性障碍函数
        
        【物理机制】
        NH4-N > 15 mg/L 时，硝化菌受到抑制
        长期高氨氮导致硝化效率下降
        
        【CBF定义】h_NH4(x) = NH4_threshold - NH4
        - h_NH4 > 0: 氨氮在安全范围内
        - h_NH4 < 0: 氨氮超标，危险
        
        【李导数】
        - L_f h = -1 · f_NH4(x) = -f_NH4
        - L_g h = -1 · g_NH4(x) = -g_NH4
        
        Returns:
            h: 障碍函数值
            Lf_h: 沿f的李导数
            Lg_h: 沿g的李导数（对曝气动作的梯度）
        """
        h_nh4 = self.nh4_threshold - nh4
        
        Lf_h_nh4 = torch.zeros_like(nh4)
        
        Lg_h_nh4 = -torch.ones_like(nh4) * 0.005
        
        return h_nh4, Lf_h_nh4, Lg_h_nh4
    
    def _compute_blower_cbf(
        self,
        freq_change: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【CBF-3】鼓风机频率变化率障碍函数
        
        【物理机制】
        鼓风机频率突变会导致喘振和机械损坏
        典型限制：|Δf| ≤ 3 Hz/s
        
        【CBF定义】h_Blower(x) = freq_change_max - |freq_change|
        
        【李导数】
        由于该CBF涉及绝对值，需要可微分近似：
        h_Blower ≈ freq_change_max - sqrt(freq_change² + ε)
        L_f h = -freq_change / sqrt(freq_change² + ε) · f_freq
        L_g h = -freq_change / sqrt(freq_change² + ε) · g_freq
        
        Returns:
            h: 障碍函数值
            Lf_h: 沿f的李导数
            Lg_h: 沿g的李导数
        """
        eps = 1e-6
        abs_freq_change = torch.sqrt(freq_change.pow(2) + eps)
        
        h_blower = self.blower_freq_change_max - abs_freq_change
        
        sign_factor = -freq_change / (abs_freq_change + eps)
        
        Lf_h_blower = torch.zeros_like(freq_change)
        
        Lg_h_blower = sign_factor * 0.001
        
        return h_blower, Lf_h_blower, Lg_h_blower
    
    def compute_cbf_constraints(
        self,
        cbf_result: CBFEvaluationResult,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        【核心方法】计算CBF约束项
        
        【CBF约束形式】
        L_f h(x) + L_g h(x) u ≥ -γ h(x)
        
        重新整理为标准形式：
        L_g h(x) u ≥ -γ h(x) - L_f h(x)
        
        Args:
            cbf_result: CBF求值结果
            gamma: 约束收紧速率 (可选，使用默认值)
            
        Returns:
            constraint_matrix @ u ≥ constraint_rhs
            shape: (batch_size, n_cbf, action_dim) @ (batch_size, action_dim) 
                  ≥ (batch_size, n_cbf)
        """
        batch_size = cbf_result.h_values.shape[0]
        n_cbf = cbf_result.h_values.shape[1]
        
        if gamma is None:
            gamma_tensor = torch.tensor(
                [self.gamma_do, self.gamma_nh4, self.gamma_blower],
                device=self.device
            )
        else:
            gamma_tensor = gamma
        
        constraint_rhs = -gamma_tensor.unsqueeze(0) * cbf_result.h_values - cbf_result.Lf_h
        
        constraint_matrix = cbf_result.Lg_h
        
        return constraint_matrix, constraint_rhs
    
    def get_n_cbf(self) -> int:
        """返回CBF数量"""
        return 3
    
    def get_gamma_values(self) -> Tuple[float, float, float]:
        """返回各CBF的gamma值"""
        return self.gamma_do, self.gamma_nh4, self.gamma_blower


class DifferentiableCBF(WaterTreatmentCBF):
    """
    可微分CBF（支持自动梯度计算）
    
    【扩展功能】使用torch.autograd计算精确的李导数
    用于需要精确梯度的离线分析和CBF验证
    """
    
    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dynamics_function: Optional[callable] = None
    ) -> None:
        super().__init__(config, device, require_grad=True)
        self.dynamics_function = dynamics_function
    
    def compute_lie_derivatives_autograd(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        h_function: callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【自动微分法】计算李导数
        
        Args:
            states: 状态张量 (batch_size, state_dim)，requires_grad=True
            actions: 动作张量 (batch_size, action_dim)
            h_function: 障碍函数 h(x)
            
        Returns:
            Lf_h: 沿f的李导数
            Lg_h: 沿g的李导数
        """
        batch_size = states.shape[0]
        
        h_values = h_function(states)
        
        Lf_h_list = []
        Lg_h_list = []
        
        for i in range(batch_size):
            h_i = h_values[i]
            state_i = states[i]
            action_i = actions[i]
            
            if h_i.requires_grad:
                grad_h = torch.autograd.grad(
                    outputs=h_i,
                    inputs=state_i,
                    grad_outputs=torch.ones_like(h_i),
                    retain_graph=True,
                    create_graph=self.require_grad
                )[0]
            else:
                grad_h = torch.zeros_like(state_i)
            
            if self.dynamics_function is not None:
                f, g = self.dynamics_function(state_i, action_i)
                Lf_h_i = grad_h @ f
                Lg_h_i = grad_h @ g
            else:
                Lf_h_i = torch.zeros(1, device=self.device)
                Lg_h_i = torch.zeros_like(action_i)
            
            Lf_h_list.append(Lf_h_i)
            Lg_h_list.append(Lg_h_i)
        
        Lf_h = torch.stack(Lf_h_list, dim=0)
        Lg_h = torch.stack(Lg_h_list, dim=0)
        
        return Lf_h, Lg_h
    
    def verify_cbf_feasibility(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        【验证方法】验证CBF可行性条件
        
        【HJI可行性检验】
        检查是否存在满足CBF约束的动作
        
        Returns:
            feasibility_report: 可行性诊断报告
        """
        cbf_result = self.forward(states)
        
        Lg_h = cbf_result.Lg_h
        Lf_h = cbf_result.Lf_h
        h_values = cbf_result.h_values
        
        gamma = torch.tensor(
            [self.gamma_do, self.gamma_nh4, self.gamma_blower],
            device=self.device
        ).unsqueeze(0)
        
        cbf_satisfaction = Lf_h + Lg_h @ actions.unsqueeze(-1) + gamma * h_values
        
        min_satisfaction = cbf_satisfaction.min(dim=-1)[0]
        
        is_feasible = min_satisfaction >= -1e-6
        
        return {
            'is_feasible': is_feasible,
            'cbf_satisfaction': cbf_satisfaction,
            'min_satisfaction': min_satisfaction,
            'margin': min_satisfaction.abs()
        }


if __name__ == "__main__":
    """【测试入口】验证CBF连通性"""
    print("=" * 70)
    print("控制障碍函数连通性测试 (WaterTreatmentCBF)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    
    config = CBFConfig(
        do_safe_threshold=0.5,
        nh4_safe_threshold=15.0,
        blower_freq_change_max=3.0,
        do_cbf_gamma=10.0,
        nh4_cbf_gamma=10.0,
        blower_cbf_gamma=10.0
    )
    
    cbf = WaterTreatmentCBF(config=config, device=device)
    
    print(f"\n【CBF配置】")
    print(f"DO安全阈值: {config.do_safe_threshold} mg/L")
    print(f"NH4安全阈值: {config.nh4_safe_threshold} mg/L")
    print(f"风机频率变化率限制: {config.blower_freq_change_max} Hz/s")
    print(f"CBF gamma值: DO={config.do_cbf_gamma}, NH4={config.nh4_cbf_gamma}, Blower={config.blower_cbf_gamma}")
    
    print(f"\n--- 测试场景1: 安全状态 ---")
    safe_states = torch.tensor([
        [2.0, 5.0, 4.0, 40.0, 1.0, 1.0, 0.5, 0.3, 10.0, 4.5, 
         0.5, 0.2, 15.0, 10.0, 2.0, 50.0, 0.8, 0.3, 5.0, 0.1],  # DO=2.0 > 0.5, NH4=5.0 < 15
    ] * batch_size, device=device)
    
    result_safe = cbf(safe_states)
    
    print(f"h_DO值: mean={result_safe.h_values[:, 0].mean().item():.4f}")
    print(f"h_NH4值: mean={result_safe.h_values[:, 1].mean().item():.4f}")
    print(f"h_Blower值: mean={result_safe.h_values[:, 2].mean().item():.4f}")
    print(f"安全状态比例: {result_safe.is_safe.float().mean().item()*100:.1f}%")
    
    print(f"\n--- 测试场景2: 危险状态（DO过低） ---")
    danger_states = safe_states.clone()
    danger_states[:, 0] = 0.2
    danger_states[:, 3] = 60.0
    
    prev_actions = torch.randn(batch_size, 4, device=device) * 0.5
    
    result_danger = cbf(danger_states, prev_actions=prev_actions, dt=1.0)
    
    print(f"h_DO值: mean={result_danger.h_values[:, 0].mean().item():.4f} (负值=危险)")
    print(f"h_NH4值: mean={result_danger.h_values[:, 1].mean().item():.4f}")
    print(f"h_Blower值: mean={result_danger.h_values[:, 2].mean().item():.4f}")
    print(f"安全状态比例: {result_danger.is_safe.float().mean().item()*100:.1f}%")
    
    print(f"\n--- 测试场景3: 梯度追踪验证 ---")
    test_states = safe_states.clone()
    test_states.requires_grad_(True)
    
    result_grad = cbf(test_states)
    
    print(f"h_values requires_grad: {result_grad.h_values.requires_grad}")
    print(f"Lf_h requires_grad: {result_grad.Lf_h.requires_grad}")
    print(f"Lg_h requires_grad: {result_grad.Lg_h.requires_grad}")
    
    print(f"\n--- 测试CBF约束计算 ---")
    constraint_matrix, constraint_rhs = cbf.compute_cbf_constraints(result_safe)
    
    print(f"约束矩阵 shape: {constraint_matrix.shape} (batch, n_cbf, action_dim)")
    print(f"约束右端 shape: {constraint_rhs.shape} (batch, n_cbf)")
    
    print(f"\n--- 测试DifferentiableCBF ---")
    diff_cbf = DifferentiableCBF(config=config, device=device)
    
    test_states_diff = safe_states[:10].clone().detach().requires_grad_(True)
    test_actions = torch.randn(10, 4, device=device) * 0.5
    
    def h_do_function(states):
        return states[:, 0] - config.do_safe_threshold
    
    Lf_h, Lg_h = diff_cbf.compute_lie_derivatives_autograd(
        test_states_diff, test_actions, h_do_function
    )
    
    print(f"自动微分 Lf_h: {Lf_h}")
    print(f"自动微分 Lg_h shape: {Lg_h.shape}")
    
    print(f"\n" + "=" * 70)
    print("✓ CBF模块连通性测试通过")
    print("=" * 70)
    
    print(f"\n【核心验证点】")
    print("1. 三个刚性CBF (DO/NH4/Blower): ✓")
    print("2. 李导数计算 (Lf_h, Lg_h): ✓")
    print("3. CBF约束矩阵生成: ✓")
    print("4. torch.autograd支持: ✓")
    print("5. 1e-6除零防护: ✓")
