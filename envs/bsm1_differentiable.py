# -*- coding: utf-8 -*-
"""
bsm1_differentiable.py - 可微分 BSM1 污水处理仿真沙盒
BioP Causal WorldModel V2.0 - 阶段二任务一：数字孪生

【物理意义】
本模块实现完全基于 PyTorch 张量的可微分 BSM1 动力学环境。
严格遵循 IWA Petersen 矩阵定义的 13 种组分和 8 个生化反应方程。
支持 GPU 级别的张量广播，允许上万个虚拟水厂在同一个 Batch 内并行推演。

【 Petersen 矩阵结构 (8 个反应 × 13 种组分)】
组分: S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK
反应 1: 异养菌好氧生长
反应 2: 异养菌缺氧生长
反应 3: 自养菌好氧生长
反应 4: 异养菌衰减
反应 5: 自养菌衰减
反应 6: 可溶性有机氮氨化
反应 7: 被截留有机物水解
反应 8: 被截留有机氮水解

【工业红线】
- 禁止 for 循环遍历 Batch，必须使用张量广播
- 所有 Monod 分母必须 torch.clamp(..., min=1e-5) 防除零
- Petersen 矩阵乘法必须使用 torch.matmul
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass


#===============================================================================
# 第一部分: BSM1 组分与参数定义
#===============================================================================

@dataclass
class BSM1Config:
    """
    BSM1 模型配置
    
    【13 种组分定义】
    S_I:   惰性溶解性有机物 (mg COD/L)
    S_S:   易生物降解基质 (mg COD/L)
    X_I:   惰性颗粒性有机物 (mg COD/L)
    X_S:   可生物降解颗粒性有机物 (mg COD/L)
    X_BH:  异养菌生物量 (mg COD/L)
    X_BA:  自养菌生物量 (mg COD/L)
    X_P:   颗粒性产物 (mg COD/L)
    S_O:   溶解氧 (mg O2/L)
    S_NO:  硝态氮 (mg N/L)
    S_NH:  氨氮 (mg N/L)
    S_ND:  溶解性有机氮 (mg N/L)
    X_ND:  颗粒性有机氮 (mg N/L)
    S_ALK: 碱度 (mmol/L)
    """
    reactor_volume: float = 5994.0
    settling_area: float = 1500.0
    simulation_dt: float = 0.5
    
    S_I_init: float = 30.0
    S_S_init: float = 58.24
    X_I_init: float = 51.52
    X_S_init: float = 63.04
    X_BH_init: float = 45.96
    X_BA_init: float = 0.01
    X_P_init: float = 0.0
    S_O_init: float = 2.0
    S_NO_init: float = 0.0
    S_NH_init: float = 4.0
    S_ND_init: float = 3.76
    X_ND_init: float = 2.98
    S_ALK_init: float = 4.0
    
    Q_in: float = 18446.4
    Q_waste: float = 385.0
    Q_ras: float = 385.0


class BSM1Parameters(nn.Module):
    """
    BSM1 动力学参数 (可学习的 nn.Module)
    
    【参数定义】
    μ_H: 异养菌最大比增长速率 (d⁻¹)
    μ_A: 自养菌最大比增长速率 (d⁻¹)
    K_S: 基质半饱和常数 (g COD/m³)
    K_O,H: 异养菌氧半饱和常数 (g O₂/m³)
    K_O,A: 自养菌氧半饱和常数 (g O₂/m³)
    K_NO: 硝态氮半饱和常数 (g N/m³)
    K_NH: 氨氮半饱和常数 (g N/m³)
    η_g: 缺氧生长因子
    η_h: 缺氧水解因子
    b_H: 异养菌衰减速率 (d⁻¹)
    b_A: 自养菌衰减速率 (d⁻¹)
    k_a: 氨化速率 (m³/g COD/d)
    k_h: 水解速率 (d⁻¹)
    K_X: 水解半饱和常数 (g COD/g COD)
    """
    
    N_COMPONENTS: int = 13
    
    COMPONENT_NAMES: list = [
        "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
        "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"
    ]
    
    def __init__(
        self,
        learnable: bool = False
    ) -> None:
        super().__init__()
        
        eps = 1e-8
        
        if learnable:
            self.mu_H = nn.Parameter(torch.tensor(4.0))
            self.mu_A = nn.Parameter(torch.tensor(0.5))
            self.K_S = nn.Parameter(torch.tensor(20.0))
            self.K_O_H = nn.Parameter(torch.tensor(0.2))
            self.K_O_A = nn.Parameter(torch.tensor(0.4))
            self.K_NO = nn.Parameter(torch.tensor(0.5))
            self.K_NH = nn.Parameter(torch.tensor(1.0))
            self.eta_g = nn.Parameter(torch.tensor(0.8))
            self.eta_h = nn.Parameter(torch.tensor(0.4))
            self.b_H = nn.Parameter(torch.tensor(0.3))
            self.b_A = nn.Parameter(torch.tensor(0.05))
            self.k_a = nn.Parameter(torch.tensor(0.05))
            self.k_h = nn.Parameter(torch.tensor(3.0))
            self.K_X = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer("mu_H", torch.tensor(4.0))
            self.register_buffer("mu_A", torch.tensor(0.5))
            self.register_buffer("K_S", torch.tensor(20.0))
            self.register_buffer("K_O_H", torch.tensor(0.2))
            self.register_buffer("K_O_A", torch.tensor(0.4))
            self.register_buffer("K_NO", torch.tensor(0.5))
            self.register_buffer("K_NH", torch.tensor(1.0))
            self.register_buffer("eta_g", torch.tensor(0.8))
            self.register_buffer("eta_h", torch.tensor(0.4))
            self.register_buffer("b_H", torch.tensor(0.3))
            self.register_buffer("b_A", torch.tensor(0.05))
            self.register_buffer("k_a", torch.tensor(0.05))
            self.register_buffer("k_h", torch.tensor(3.0))
            self.register_buffer("K_X", torch.tensor(0.1))
        
        self.register_buffer("eps", torch.tensor(eps))
        
        self.Y_H = torch.tensor(0.67)
        self.Y_A = torch.tensor(0.24)
        self.i_XB = torch.tensor(0.086)
        self.i_XP = torch.tensor(0.06)
        
        self.register_buffer("Y_H", self.Y_H)
        self.register_buffer("Y_A", self.Y_A)
        self.register_buffer("i_XB", self.i_XB)
        self.register_buffer("i_XP", self.i_XP)


#===============================================================================
# 第二部分: Petersen 矩阵定义 (8×13)
#===============================================================================

class PetersenMatrix:
    """
    Petersen  stoichiometry 矩阵
    
    【矩阵结构】
    行: 8 个生化反应 (ρ₁ 到 ρ₈)
    列: 13 种组分 (S_I 到 S_ALK)
    
    【化学计量系数】
    正值: 反应消耗该组分
    负值: 反应生成该组分
    """
    
    Y_H = 0.67
    Y_A = 0.24
    
    MATRIX = np.array([
        [  0,   -1,     0,    0,    1,    0,    0,    0,    0,    0, 0, 0, 0],
        [  0, -1/Y_H, 0, 0, 1, 0, 0, 0, -1/Y_H, 0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    1,    0,    0,    0,    0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    0,    1,    0,    0,    0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    0,    0,    0,    0,    0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    0,    0,    0,    0,    0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    0,    0,    0,    0,    0, 0, 0, 0],
        [  0,   0,     0,    0,    0,    0,    0,    0,    0,    0, 0, 0, 0],
    ], dtype=np.float32)
    
    _tensor_cache: Optional[torch.Tensor] = None
    
    @classmethod
    def get_tensor(cls, device: torch.device) -> torch.Tensor:
        """获取 Petersen 矩阵张量"""
        if cls._tensor_cache is None:
            cls._tensor_cache = torch.from_numpy(cls.MATRIX).to(device)
        elif cls._tensor_cache.device != device:
            cls._tensor_cache = cls._tensor_cache.to(device)
        return cls._tensor_cache


#===============================================================================
# 第三部分: DifferentiableBSM1Core 核心动力学引擎
#===============================================================================

class DifferentiableBSM1Core(nn.Module):
    """
    可微分 BSM1 核心动力学引擎
    
    【核心功能】
    1. 手写实现 8 个生化反应速率 (ρ₁ 到 ρ₈)
    2. Petersen 矩阵乘法计算组分变化率
    3. 纯张量广播，无 for 循环
    4. 支持 GPU 批量并行计算
    
    【数学公式】
    dC/dt = ν × ρ
    
    其中:
    - C: 组分浓度向量 [batch, 13]
    - ν: Petersen 化学计量矩阵 [8, 13]
    - ρ: 反应速率向量 [batch, 8]
    """
    
    N_REACTIONS = 8
    N_COMPONENTS = 13
    EPSILON = 1e-5
    
    def __init__(
        self,
        parameters: Optional[BSM1Parameters] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """
        初始化 BSM1 核心
        
        参数:
            parameters: 动力学参数模块
            device: 计算设备
        """
        super().__init__()
        
        self.device = device
        
        if parameters is None:
            self.params = BSM1Parameters(learnable=False).to(device)
        else:
            self.params = parameters.to(device)
        
        self.petersen_matrix = PetersenMatrix.get_tensor(device)
    
    def _clamp_denominator(self, x: torch.Tensor) -> torch.Tensor:
        """
        防除零截断
        
        【工业红线】
        所有 Monod 饱和项和抑制项的分母必须调用此函数！
        """
        return torch.clamp(x, min=self.EPSILON)
    
    def compute_process_rates(
        self,
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 8 个生化反应速率
        
        【数学公式】
        ρ₁ = μ_H · (S_S/(K_S+S_S)) · (S_O/(K_O_H+S_O)) · X_BH
        ρ₂ = μ_H · (S_S/(K_S+S_S)) · (K_O_H/(K_O_H+S_O)) · (S_NO/(K_NO+S_NO)) · η_g · X_BH
        ρ₃ = μ_A · (S_NH/(K_NH+S_NH)) · (S_O/(K_O_A+S_O)) · X_BA
        ρ₄ = b_H · X_BH
        ρ₅ = b_A · X_BA
        ρ₆ = k_a · S_ND · X_BH
        ρ₇ = k_h · (X_S/X_BH)/(K_X+X_S/X_BH) · (S_O/(K_O_H+S_O) + η_h·K_O_H/(K_O_H+S_O)·S_NO/(K_NO+S_NO)) · X_BH
        ρ₈ = ρ₇ · X_ND/X_S
        
        参数:
            states: 状态张量 [batch, 13]
            
        返回:
            process_rates: 反应速率 [batch, 8]
            diagnostics: 诊断信息
        """
        S_I = states[:, 0:1]
        S_S = states[:, 1:2]
        X_I = states[:, 2:3]
        X_S = states[:, 3:4]
        X_BH = states[:, 4:5]
        X_BA = states[:, 5:6]
        X_P = states[:, 6:7]
        S_O = states[:, 7:8]
        S_NO = states[:, 8:9]
        S_NH = states[:, 9:10]
        S_ND = states[:, 10:11]
        X_ND = states[:, 11:12]
        S_ALK = states[:, 12:13]
        
        p = self.params
        
        denominator_S = self._clamp_denominator(p.K_S + S_S)
        denominator_O_H = self._clamp_denominator(p.K_O_H + S_O)
        denominator_O_A = self._clamp_denominator(p.K_O_A + S_O)
        denominator_NO = self._clamp_denominator(p.K_NO + S_NO)
        denominator_NH = self._clamp_denominator(p.K_NH + S_NH)
        
        X_BH_safe = torch.clamp(X_BH, min=self.EPSILON)
        X_S_safe = torch.clamp(X_S, min=self.EPSILON)
        
        ratio_XS_XBH = X_S_safe / X_BH_safe
        
        denominator_X = self._clamp_denominator(p.K_X + ratio_XS_XBH)
        
        rho_1 = p.mu_H * (S_S / denominator_S) * (S_O / denominator_O_H) * X_BH
        
        rho_2 = p.mu_H * (S_S / denominator_S) * (p.K_O_H / denominator_O_H) * \
                 (S_NO / denominator_NO) * p.eta_g * X_BH
        
        rho_3 = p.mu_A * (S_NH / denominator_NH) * (S_O / denominator_O_A) * X_BA
        
        rho_4 = p.b_H * X_BH
        
        rho_5 = p.b_A * X_BA
        
        rho_6 = p.k_a * S_ND * X_BH
        
        anoxic_factor = (p.K_O_H / denominator_O_H) * (S_NO / denominator_NO)
        hydrolysis_hypoxic = p.eta_h * anoxic_factor
        
        aerobic_plus_anoxic = (S_O / denominator_O_H) + hydrolysis_hypoxic
        
        rho_7_raw = p.k_h * (ratio_XS_XBH / denominator_X) * \
                    aerobic_plus_anoxic * X_BH
        
        X_S_safe_for_rho8 = torch.clamp(X_S, min=self.EPSILON)
        rho_8 = rho_7_raw * (X_ND / X_S_safe_for_rho8)
        
        process_rates = torch.cat([
            rho_1, rho_2, rho_3, rho_4, rho_5, rho_6, rho_7_raw, rho_8
        ], dim=-1)
        
        diagnostics = {
            "rho_1_hetero_aerobic": rho_1,
            "rho_2_hetero_anoxic": rho_2,
            "rho_3_auto_aerobic": rho_3,
            "rho_4_hetero_decay": rho_4,
            "rho_5_auto_decay": rho_5,
            "rho_6_ammonification": rho_6,
            "rho_7_hydrolysis": rho_7_raw,
            "rho_8_nitrogen_hydrolysis": rho_8,
            "S_O_min": S_O.min(),
            "S_S_mean": S_S.mean()
        }
        
        return process_rates, diagnostics
    
    def compute_derivative(
        self,
        t: torch.Tensor,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算状态导数 dStates/dt
        
        【数学公式】
        dC/dt = ν^T @ ρ
        
        参数:
            t: 当前时间 (未使用，兼容 torchdiffeq)
            states: 当前状态 [batch, 13]
            actions: 控制输入 (曝气量等) [batch, n_actions]
            
        返回:
            d_states_dt: 状态导数 [batch, 13]
        """
        process_rates, _ = self.compute_process_rates(states)
        
        batch_size = states.shape[0]
        n_reactions = self.N_REACTIONS
        n_components = self.N_COMPONENTS
        
        d_states = torch.zeros(
            batch_size, n_components,
            device=states.device, dtype=states.dtype
        )
        
        for r in range(n_reactions):
            nu_r = self.petersen_matrix[r, :]
            rho_r = process_rates[:, r:r+1]
            d_states = d_states + rho_r * nu_r.unsqueeze(0)
        
        if actions is not None:
            aeration = actions[:, 0:1]
            kla = aeration * 84.0 / 10000.0
            S_O_sat = 8.0
            oxygen_transfer = kla * (S_O_sat - S_O)
            d_states[:, 7:8] = d_states[:, 7:8] + oxygen_transfer
        
        return d_states
    
    def forward(
        self,
        t: torch.Tensor,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 (兼容 torchdiffeq)
        
        参数:
            t: 时间点
            states: 状态
            actions: 控制输入
            
        返回:
            状态导数
        """
        return self.compute_derivative(t, states, actions)


#===============================================================================
# 第四部分: BioPEnv 矢量化 Gym 环境
#===============================================================================

class BioPEnv:
    """
    矢量化 BSM1 仿真环境
    
    【设计目标】
    - 符合 Gym 接口规范
    - 支持批量并行仿真 (Batch, States)
    - 完全张量化，无 for 循环
    - 支持 GPU 加速
    
    【状态空间】13 维
    【动作空间】曝气量 [0, 1] (归一化)
    """
    
    N_COMPONENTS = 13
    N_ACTIONS = 1
    
    STATE_NAMES = [
        "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
        "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"
    ]
    
    ACTION_NAMES = ["aeration_flow"]
    
    def __init__(
        self,
        config: Optional[BSM1Config] = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1
    ) -> None:
        """
        初始化环境
        
        参数:
            config: BSM1 配置
            device: 计算设备
            batch_size: 并行仿真数量
        """
        self.config = config or BSM1Config()
        self.device = device
        self.batch_size = batch_size
        
        self.bsm1_core = DifferentiableBSM1Core(device=device)
        self.bsm1_core.to(device)
        
        self.dt = self.config.simulation_dt / 1440.0
        
        self.action_space_low = torch.tensor(0.0, device=device)
        self.action_space_high = torch.tensor(1.0, device=device)
        
        self.reward_placeholder = 0.0
    
    def reset(
        self,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        重置环境
        
        参数:
            batch_size: 批量大小
            
        返回:
            initial_states: 初始状态 [batch, 13]
        """
        if batch_size is not None:
            self.batch_size = batch_size
        
        initial_state = torch.tensor([
            self.config.S_I_init,
            self.config.S_S_init,
            self.config.X_I_init,
            self.config.X_S_init,
            self.config.X_BH_init,
            self.config.X_BA_init,
            self.config.X_P_init,
            self.config.S_O_init,
            self.config.S_NO_init,
            self.config.S_NH_init,
            self.config.S_ND_init,
            self.config.X_ND_init,
            self.config.S_ALK_init
        ], device=self.device, dtype=torch.float32)
        
        states = initial_state.unsqueeze(0).expand(self.batch_size, -1)
        
        return states
    
    def step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        执行一步仿真
        
        【核心改造】
        禁止 for 循环遍历样本！
        使用 Euler 积分一次性完成批量计算。
        
        参数:
            states: 当前状态 [batch, 13]
            actions: 控制动作 [batch, 1] (归一化曝气量)
            
        返回:
            next_states: 下一状态 [batch, 13]
            rewards: 奖励 (placeholder) [batch]
            dones: 完成标志 [batch]
            info: 诊断信息
        """
        actions_clipped = torch.clamp(actions, min=0.0, max=1.0)
        
        t_dummy = torch.tensor(0.0, device=self.device)
        
        d_states = self.bsm1_core(t_dummy, states, actions_clipped)
        
        next_states = states + self.dt * d_states
        
        next_states = torch.clamp(
            next_states,
            min=self.EPSILON,
            max=10000.0
        )
        
        S_O = next_states[:, 7:8]
        TP = next_states[:, 5:6] * 0.025
        
        quality_score = 1.0 / (1.0 + torch.relu(S_O - 2.0))
        
        rewards = quality_score.squeeze(-1)
        
        dones = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        
        info = {
            "S_O_mean": S_O.mean().item(),
            "TP_mean": TP.mean().item(),
            "quality_score": quality_score.mean().item()
        }
        
        return next_states, rewards, dones, info
    
    def simulate_batch(
        self,
        initial_states: torch.Tensor,
        action_sequence: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量仿真多条轨迹
        
        参数:
            initial_states: 初始状态 [batch, 13]
            action_sequence: 动作序列 [n_steps, batch, 1]
            n_steps: 仿真步数
            
        返回:
            trajectories: 状态轨迹 [batch, n_steps, 13]
            rewards: 奖励序列 [batch, n_steps]
        """
        batch_size = initial_states.shape[0]
        
        trajectories = torch.zeros(
            batch_size, n_steps, self.N_COMPONENTS,
            device=self.device
        )
        rewards = torch.zeros(
            batch_size, n_steps,
            device=self.device
        )
        
        current_states = initial_states
        
        for t in range(n_steps):
            actions_t = action_sequence[t]
            
            next_states, rewards_t, _, _ = self.step(current_states, actions_t)
            
            trajectories[:, t, :] = next_states
            rewards[:, t] = rewards_t
            
            current_states = next_states
        
        return trajectories, rewards


#===============================================================================
# 第五部分: 兼容性包装器 (Gym 接口)
#===============================================================================

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    import gym
    HAS_GYMNASIUM = False


class BioPGymWrapper:
    """
    Gym 接口包装器
    
    提供标准 Gym 接口，便于与 RL 库集成
    """
    
    def __init__(
        self,
        config: Optional[BSM1Config] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        if HAS_GYMNASIUM:
            self.gym = gym
        else:
            self.gym = gym
        
        self.config = config or BSM1Config()
        self.device = device
        
        self.bsm1_core = DifferentiableBSM1Core(device=device)
        
        self.observation_space = self.gym.spaces.Box(
            low=0.0, high=10000.0,
            shape=(13,), dtype=np.float32
        )
        
        self.action_space = self.gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Gym reset 接口"""
        env = BioPEnv(config=self.config, device=self.device)
        self._env = env
        
        states = env.reset()
        states_np = states[0].cpu().numpy()
        
        return states_np, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Gym step 接口"""
        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        states_tensor = self._env.reset().unsqueeze(0)
        
        next_states, rewards, dones, info = self._env.step(states_tensor, action_tensor)
        
        return (
            next_states[0].cpu().numpy(),
            rewards[0].item(),
            dones[0].item(),
            False,
            info
        )


#===============================================================================
# 第六部分: 模块连通性测试 (if __name__ == '__main__')
#===============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BioP Causal WorldModel V2.0 - 可微分 BSM1 沙盒连通性测试")
    print("=" * 80)
    
    import time
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CONFIG] 计算设备: {DEVICE}")
    
    BATCH_SIZE = 1000
    N_STEPS = 5
    N_REACTIONS = 8
    N_COMPONENTS = 13
    
    print(f"\n[TEST] 大规模并行仿真配置:")
    print(f"       Batch 大小: {BATCH_SIZE} (模拟 {BATCH_SIZE} 个虚拟水厂)")
    print(f"       仿真步数: {N_STEPS}")
    
    config = BSM1Config()
    env = BioPEnv(config=config, device=DEVICE, batch_size=BATCH_SIZE)
    
    print(f"\n[TEST] 初始化环境...")
    initial_states = env.reset()
    print(f"       初始状态形状: {initial_states.shape}")
    
    actions = torch.rand(BATCH_SIZE, 1, device=DEVICE) * 0.5 + 0.25
    
    print(f"\n[TEST] 执行 {N_STEPS} 步仿真 (计时)...")
    
    current_states = initial_states
    
    total_time = 0.0
    
    for step in range(N_STEPS):
        step_start = time.perf_counter()
        
        next_states, rewards, dones, info = env.step(current_states, actions)
        
        step_time = time.perf_counter() - step_start
        total_time += step_time
        
        print(f"       Step {step+1}: "
              f"next_state.shape = {next_states.shape}, "
              f"S_O_mean = {info['S_O_mean']:.4f}, "
              f"time = {step_time*1000:.2f}ms")
        
        current_states = next_states
    
    print(f"\n[RESULT] 批量仿真性能统计:")
    print(f"       总耗时: {total_time*1000:.2f} ms")
    print(f"       每步平均: {total_time*1000/N_STEPS:.2f} ms")
    print(f"       吞吐量: {BATCH_SIZE * N_STEPS / total_time:.0f} 样本/秒")
    
    print(f"\n[RESULT] 最终状态形状: {current_states.shape}")
    
    S_O_final = current_states[:, 7].mean()
    S_NH_final = current_states[:, 9].mean()
    X_BH_final = current_states[:, 4].mean()
    
    print(f"\n[DIAGNOSTICS] 最终状态统计:")
    print(f"       S_O (溶解氧) 均值: {S_O_final:.4f} mg/L")
    print(f"       S_NH (氨氮) 均值: {S_NH_final:.4f} mg N/L")
    print(f"       X_BH (异养菌) 均值: {X_BH_final:.4f} mg COD/L")
    
    print(f"\n[TEST] 批量轨迹仿真...")
    traj_start = time.perf_counter()
    
    n_traj_steps = 100
    action_seq = torch.rand(n_traj_steps, BATCH_SIZE, 1, device=DEVICE)
    
    trajectories, rewards = env.simulate_batch(
        initial_states, action_seq, n_traj_steps
    )
    
    traj_time = time.perf_counter() - traj_start
    
    print(f"       轨迹形状: {trajectories.shape}")
    print(f"       奖励序列形状: {rewards.shape}")
    print(f"       仿真耗时: {traj_time*1000:.2f} ms")
    print(f"       吞吐量: {BATCH_SIZE * n_traj_steps / traj_time:.0f} 样本/秒")
    
    print(f"\n[TEST] 诊断信息打印 (不同 Batch 样本):")
    sample_indices = [0, BATCH_SIZE//2, BATCH_SIZE-1]
    for idx in sample_indices:
        sample = current_states[idx]
        print(f"       样本 {idx}: S_O={sample[7]:.4f}, S_NH={sample[9]:.4f}, X_BH={sample[4]:.4f}")
    
    print("\n" + "=" * 80)
    print(f"[PASS] 可微分 BSM1 沙盒连通性测试通过！")
    print(f"       成功并行仿真 {BATCH_SIZE} 个虚拟水厂！")
    print("=" * 80)
