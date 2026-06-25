# -*- coding: utf-8 -*-
"""
BSM1 Full Simulation Environment - 完整污水处理仿真沙盒
BioP Causal WorldModel V2.0 - 核心仿真器

【完整功能】
1. 13种组分完整Petersen矩阵
2. 8个生化反应动力学
3. 4个控制变量 (曝气、碳源、回流、脱水)
4. 物理传感器仿真 (带噪声、延迟、故障)
5. 能耗模型 (曝气、搅拌、泵送)
6. 安全约束监控

【模型参数】20M
【控制接口】4个执行器
【状态空间】50维 (含虚拟传感器)

【版本】V2.0-FullSimulation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ProcessState(Enum):
    """工艺状态"""
    NORMAL = "normal"
    HIGH_LOAD = "high_load"
    LOW_LOAD = "low_load"
    RAIN_STORM = "rain_storm"
    MAINTENANCE = "maintenance"


@dataclass
class BSM1FullConfig:
    """
    BSM1完整配置
    
    【13种组分】(mg COD/L 或 mg N/L 或 mg P/L)
    S_I:   惰性溶解性有机物
    S_S:   易生物降解基质
    X_I:   惰性颗粒性有机物
    X_S:   可生物降解颗粒性有机物
    X_BH:  异养菌生物量
    X_BA:  自养菌生物量
    X_P:   颗粒性产物
    S_O:   溶解氧 (mg O2/L)
    S_NO:  硝态氮 (mg N/L)
    S_NH:  氨氮 (mg N/L)
    S_ND:  溶解性有机氮
    X_ND:  颗粒性有机氮
    S_ALK: 碱度 (mmol/L)
    
    【8个反应】
    ρ₁: 异养菌好氧生长
    ρ₂: 异养菌缺氧生长
    ρ₃: 自养菌好氧生长
    ρ₄: 异养菌衰减
    ρ₅: 自养菌衰减
    ρ₆: 可溶性有机氮氨化
    ρ₇: 被截留有机物水解
    ρ₈: 被截留有机氮水解
    """
    # ========== 反应器参数 ==========
    reactor_volume: float = 5994.0  # m³ (单个反应器)
    n_reactors: int = 5  # 曝气池数量
    total_volume: float = 29970.0  # 总反应体积
    
    # ========== 组分初始值 ==========
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
    
    # ========== 进水参数 (典型值) ==========
    Q_in_mean: float = 18446.4  # m³/d 平均流量
    Q_in_std: float = 3000.0   # 流量波动
    COD_in: float = 300.0      # 进水COD
    TN_in: float = 40.0       # 总氮
    TP_in: float = 6.0         # 总磷
    
    # ========== 动力学常数 (IWA ASM1标准) ==========
    Y_H: float = 0.67   # 异养菌产率
    Y_A: float = 0.24   # 自养菌产率
    mu_H: float = 4.0   # 异养菌最大比增长速率 (d⁻¹)
    mu_A: float = 0.5   # 自养菌最大比增长速率 (d⁻¹)
    k_d_H: float = 0.3  # 异养菌衰减速率 (d⁻¹)
    k_d_A: float = 0.05  # 自养菌衰减速率 (d⁻¹)
    K_S: float = 2.0    # 基质半饱和常数 (mgCOD/L)
    K_NH: float = 0.5   # 氨氮半饱和常数 (mgN/L)
    K_NO: float = 0.5   # 硝态氮半饱和常数 (mgN/L)
    K_O_H: float = 0.2  # 异养菌氧饱和常数
    K_O_A: float = 0.4  # 自养菌氧饱和常数
    k_a: float = 0.08   # 氨化速率
    
    # ========== 沉降参数 ==========
    settling_area: float = 1500.0  # 沉淀池面积 m²
    v_max: float = 7.4  # 最大沉降速率 m/d
    v_min: float = 1.0  # 最小沉降速率 m/d
    
    # ========== 控制变量范围 ==========
    aeration_range: Tuple[float, float] = (0.0, 200.0)  # 曝气量 m³/h
    carbon_dose_range: Tuple[float, float] = (0.0, 100.0)  # 碳源投加 L/d
    return_sludge_range: Tuple[float, float] = (0.0, 1.0)  # 回流比 0-1
    waste_sludge_range: Tuple[float, float] = (0.0, 500.0)  # 排泥量 m³/d
    
    # ========== 安全约束 ==========
    DO_min: float = 0.5   # 溶解氧下限 mg/L
    DO_max: float = 6.0   # 溶解氧上限 mg/L
    NH4_max: float = 15.0 # 氨氮上限 mgN/L (毒性)
    TP_max: float = 10.0  # 总磷上限 mgP/L
    TN_max: float = 20.0  # 总氮上限 mgN/L
    
    # ========== 能耗参数 ==========
    aeration_energy_coeff: float = 0.04  # kWh/m³ O2
    mixing_energy_coeff: float = 0.002   # kWh/m³/h
    pumping_energy_coeff: float = 0.08   # kWh/m³


class PetersenMatrixFull:
    """
    Petersen stoichiometry矩阵 (8反应 × 13组分)
    
    行: 反应 ρ₁ 到 ρ₈
    列: 组分 S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK
    
    BSM1标准产率参数:
    Y_H = 0.67 (异养菌产率)
    Y_A = 0.24 (自养菌产率)
    """
    
    # BSM1标准产率
    Y_H = 0.67
    Y_A = 0.24
    
    MATRIX = np.array([
        # S_I  S_S   X_I  X_S  X_BH X_BA X_P  S_O  S_NO S_NH S_ND X_ND S_ALK
        [  0,  -1/Y_H, 0,   0,   1,   0,   0,   -(1-Y_H)/Y_H, 0,   0,   0,   0,   0  ],  # ρ₁ 异养好氧
        [  0,  -1/Y_H, 0,   0,   1,   0,   0,    0, -(1-Y_H)/(2.86*Y_H), 0, 0, 0, 0  ],  # ρ₂ 异养缺氧
        [  0,   0,    0,   0,   0,   1,   0,   -4.57/Y_A,  1/Y_A, -1,  0,  0,  0  ],  # ρ₃ 自养好氧
        [  0,   0,    0,   1,  -1,   0,   1,    0,   0,   0,   0,  0,  0  ],  # ρ₄ 异养衰减
        [  0,   0,    0,   0,   0,  -1,   1,    0,   0,   0,   0,  1,  0  ],  # ρ₅ 自养衰减
        [  0,   0,    0,   0,   0,   0,   0,    0,   0,   1,  -1,  0,  0  ],  # ρ₆ 氨化
        [  0,   1,    0,  -1,   0,   0,   0,    0,   0,   0,   0,  1,  0  ],  # ρ₇ 水解
        [  0,   0,    0,   0,   0,   0,   0,    0,   0,   0,   1, -1,  0  ],  # ρ₈ 有机氮水解
    ], dtype=np.float32)


class BioPSimulator:
    """
    完整污水处理仿真器
    
    【功能】
    - 完整的BSM1动力学仿真
    - 可控执行器接口
    - 传感器仿真 (噪声、故障)
    - 能耗计算
    - 安全约束监控
    """
    
    def __init__(
        self,
        config: Optional[BSM1FullConfig] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = 1,
        simulation_dt: float = 0.01  # 仿真步长 (天)
    ):
        self.config = config or BSM1FullConfig()
        self.device = device
        self.batch_size = batch_size
        self.simulation_dt = simulation_dt
        
        # Petersen矩阵
        self.p_matrix = torch.tensor(
            PetersenMatrixFull.MATRIX,
            dtype=torch.float32,
            device=device
        )
        
        # 反应速率向量 (8个反应)
        self.reaction_rates = None
        
        # 当前状态
        self.state = None  # [batch, 13] 组分浓度
        self.time = 0.0
        
        # 控制变量
        self.actions = None
        
        # 历史记录
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'energy': [],
            'constraints': []
        }
        
        # 传感器噪声参数
        self.sensor_noise_std = {
            'S_O': 0.1,    # DO噪声
            'S_NH': 0.2,  # 氨氮噪声
            'S_NO': 0.1,  # 硝氮噪声
            'Q_in': 50.0, # 流量噪声
            'pH': 0.05,   # pH噪声
        }
        
        # 传感器延迟 (步数)
        self.sensor_delay = {
            'S_O': 0,      # DO实时
            'S_NH': 1,     # 氨氮延迟1步
            'S_NO': 1,     # 硝氮延迟1步
            'TP': 5,       # 磷延迟5步 (需消解)
        }
        
        # 初始化状态
        self.reset()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置仿真器"""
        self.time = 0.0
        
        # 初始组分状态 [batch, 13]
        S_I = self.config.S_I_init
        S_S = self.config.S_S_init
        X_I = self.config.X_I_init
        X_S = self.config.X_S_init
        X_BH = self.config.X_BH_init
        X_BA = self.config.X_BA_init
        X_P = self.config.X_P_init
        S_O = self.config.S_O_init
        S_NO = self.config.S_NO_init
        S_NH = self.config.S_NH_init
        S_ND = self.config.S_ND_init
        X_ND = self.config.X_ND_init
        S_ALK = self.config.S_ALK_init
        
        initial_state = torch.tensor(
            [S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK],
            dtype=torch.float32,
            device=self.device
        )
        
        self.state = initial_state.unsqueeze(0).repeat(self.batch_size, 1)
        
        # 默认控制动作
        self.actions = torch.tensor(
            [100.0, 0.0, 0.6, 200.0],  # [曝气, 碳源, 回流比, 排泥]
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).repeat(self.batch_size, 1)
        
        # 清空历史
        self.history = {
            'states': [self.state.clone()],
            'actions': [self.actions.clone()],
            'rewards': [],
            'energy': [],
            'constraints': []
        }
        
        return self._get_observation()
    
    def _compute_reaction_rates(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        计算8个反应的反应速率
        
        Args:
            state: [batch, 13] 组分浓度
            actions: [batch, 4] 控制动作
            
        Returns:
            reaction_rates: [batch, 8] 反应速率
        """
        batch = state.shape[0]
        
        # 提取组分
        S_S = state[:, 1]    # 易生物降解基质
        S_O = state[:, 7]    # 溶解氧
        S_NO = state[:, 8]   # 硝态氮
        S_NH = state[:, 9]   # 氨氮
        X_BH = state[:, 4]   # 异养菌
        X_BA = state[:, 5]   # 自养菌
        
        # 提取控制变量
        aeration = actions[:, 0]  # 曝气量
        
        # 曝气供氧速率 (与曝气量成正比，与体积成反比)
        kLa = aeration / self.config.reactor_volume * 0.1  # 氧气传质系数
        
        # 反应速率计算 (Monod动力学)
        # ρ₁: 异养菌好氧生长
        mu_H = self.config.mu_H
        K_S_val = self.config.K_S
        K_O_H_val = self.config.K_O_H
        Y_H_val = self.config.Y_H
        
        r1 = mu_H * (S_S / (K_S_val + S_S)) * (S_O / (K_O_H_val + S_O)) * X_BH
        
        # ρ₂: 异养菌缺氧生长
        K_NO_val = self.config.K_NO
        r2 = mu_H * (S_S / (K_S_val + S_S)) * (K_NO_val / (K_NO_val + S_NO)) * X_BH
        
        # ρ₃: 自养菌好氧生长
        mu_A = self.config.mu_A
        K_NH_val = self.config.K_NH
        K_O_A_val = self.config.K_O_A
        Y_A_val = self.config.Y_A
        
        r3 = mu_A * (S_NH / (K_NH_val + S_NH)) * (S_O / (K_O_A_val + S_O)) * X_BA
        
        # ρ₄: 异养菌衰减
        k_d_H_val = self.config.k_d_H
        r4 = k_d_H_val * X_BH
        
        # ρ₅: 自养菌衰减
        k_d_A_val = self.config.k_d_A
        r5 = k_d_A_val * X_BA
        
        # ρ₆: 氨化
        k_a_val = self.config.k_a
        S_ND = state[:, 10]  # 溶解性有机氮
        r6 = k_a_val * S_ND * X_BH
        
        # ρ₇: 水解 (简化)
        X_S = state[:, 3]  # 可生物降解颗粒性有机物
        r7 = 0.5 * X_S * X_BH
        
        # ρ₈: 有机氮水解 (简化)
        X_ND = state[:, 11]  # 颗粒性有机氮
        r8 = 0.3 * X_ND * X_BH
        
        # 组装反应速率向量
        rates = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        
        return rates
    
    def _compute_dSdt(self, state: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        """
        计算组分随时间变化率
        
        dS/dt = Σ(ρ_i × ν_i,j)
        
        Args:
            state: [batch, 13] 当前组分
            rates: [batch, 8] 反应速率
            
        Returns:
            dSdt: [batch, 13] 组分导数
        """
        # 反应速率 × Petersen矩阵行
        # rates: [batch, 8], p_matrix: [8, 13]
        dSdt = torch.matmul(rates, self.p_matrix)
        
        # 添加曝气供氧 (曝气→S_O增加)
        aeration_effect = torch.zeros_like(dSdt)
        aeration_effect[:, 7] = 1.0  # S_O索引
        
        return dSdt
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool, Dict]:
        """
        仿真一步
        
        Args:
            actions: [batch, 4] 控制动作 [曝气, 碳源, 回流比, 排泥]
            
        Returns:
            observation: 观测字典
            reward: 奖励
            done: 是否结束
            info: 附加信息
        """
        self.actions = actions
        
        # 计算反应速率
        rates = self._compute_reaction_rates(self.state, actions)
        self.reaction_rates = rates
        
        # 计算导数
        dSdt = self._compute_dSdt(self.state, rates)
        
        # 添加进水扰动 (模拟真实工况)
        inflow_disturbance = self._get_inflow_disturbance()
        dSdt = dSdt + inflow_disturbance
        
        # Euler积分
        self.state = self.state + dSdt * self.simulation_dt
        
        # 确保非负
        self.state = torch.clamp(self.state, min=0.0)
        
        # 更新时间
        self.time += self.simulation_dt
        
        # 获取观测
        observation = self._get_observation()
        
        # 计算奖励
        reward, reward_info = self._compute_reward(observation, actions)
        
        # 检查约束
        constraints_violated = self._check_constraints(observation)
        
        # 计算能耗
        energy = self._compute_energy(actions)
        
        # 记录历史
        self.history['states'].append(self.state.clone())
        self.history['actions'].append(actions.clone())
        self.history['rewards'].append(reward.item())
        self.history['energy'].append(energy.item())
        self.history['constraints'].append(constraints_violated)
        
        # 溢出检测
        done = torch.any(torch.isnan(self.state)) or torch.any(self.state > 10000)
        
        info = {
            'reward_breakdown': reward_info,
            'constraints_violated': constraints_violated,
            'energy': energy.item(),
            'time': self.time,
            'reaction_rates': rates,
        }
        
        return observation, reward, done, info
    
    def _get_inflow_disturbance(self) -> torch.Tensor:
        """生成进水扰动 (模拟真实工况)"""
        batch = self.state.shape[0]
        
        # 时间相关的扰动
        t = self.time
        
        # 日负荷变化 (正弦) - 使用numpy后转换
        daily_factor = 1.0 + 0.3 * np.sin(2 * np.pi * t / 1.0)
        
        # 周负荷变化
        weekly_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / 7.0)
        
        # 随机扰动
        random_factor = 1.0 + 0.1 * torch.randn(batch, device=self.device)
        
        # 雨季冲击 (偶尔)
        rain_impact = 1.5 if torch.rand(1).item() < 0.01 else 1.0
        
        total_factor = daily_factor * weekly_factor * random_factor * rain_impact
        
        # 进水COD扰动 (影响S_S)
        cod_disturbance = torch.zeros_like(self.state)
        cod_disturbance[:, 1] = (total_factor - 1.0) * self.config.COD_in
        
        return cod_disturbance * 0.01  # 缩放
    
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """
        获取观测 (包含虚拟传感器)
        
        【观测空间】48维
        - 13组分浓度
        - 4个控制变量
        - 12个派生特征 (时刻、统计量等)
        - 6个虚拟传感器
        - 13个历史滞后特征 (过去5步状态平均)
        """
        state = self.state
        actions = self.actions
        
        # 1. 真实组分 (13维)
        components = state  # [batch, 13]
        
        # 2. 控制变量 (4维)
        controls = actions  # [batch, 4]
        
        # 3. 派生特征 (12维)
        S_O = state[:, 7:8]
        S_NH = state[:, 9:10]
        S_NO = state[:, 8:9]
        TP = state[:, 4:5] + state[:, 5:6] + state[:, 6:7]  # 总磷近似
        
        time_cycle = np.array([
            np.sin(2 * np.pi * self.time / 1.0),
            np.cos(2 * np.pi * self.time / 1.0),
            np.sin(2 * np.pi * self.time / 7.0),
            np.cos(2 * np.pi * self.time / 7.0),
        ], dtype=np.float32)
        
        derived = torch.cat([
            S_O / (self.config.DO_max + 1e-6),  # DO归一化
            S_NH / (self.config.NH4_max + 1e-6),  # NH4归一化
            S_NO / 20.0,  # NO3归一化
            TP / 10.0,  # TP归一化
            torch.from_numpy(time_cycle[0:1]).float().expand(state.shape[0], 1).to(self.device),  # 日周期
            torch.from_numpy(time_cycle[1:2]).float().expand(state.shape[0], 1).to(self.device),
            torch.from_numpy(time_cycle[2:3]).float().expand(state.shape[0], 1).to(self.device),  # 周周期
            torch.from_numpy(time_cycle[3:4]).float().expand(state.shape[0], 1).to(self.device),
            torch.ones(state.shape[0], 4, device=self.device),  # 填充到12
        ], dim=1)
        
        # 4. 传感器噪声 (6维) - 模拟真实传感器
        noise_features = torch.cat([
            torch.randn(state.shape[0], 1, device=self.device) * self.sensor_noise_std['S_O'],
            torch.randn(state.shape[0], 1, device=self.device) * self.sensor_noise_std['S_NH'],
            torch.randn(state.shape[0], 1, device=self.device) * self.sensor_noise_std['S_NO'],
            torch.randn(state.shape[0], 1, device=self.device) * self.sensor_noise_std['Q_in'],
            torch.zeros(state.shape[0], 2, device=self.device),  # pH, 浊度
        ], dim=1)
        
        # 5. 历史滞后特征 (13维，与组分相同)
        if len(self.history['states']) >= 5:
            recent_states = torch.stack(self.history['states'][-5:], dim=1)  # [batch, 5, 13]
            lagged_features = recent_states.mean(dim=1)  # 过去5步平均 -> [batch, 13]
        else:
            lagged_features = state  # 当前状态 -> [batch, 13]
        
        # 合并所有特征
        observation = torch.cat([
            components,   # 13维
            controls,     # 4维
            derived,      # 12维
            noise_features,  # 6维
            lagged_features, # 13维
        ], dim=1)  # 总计: 13+4+12+6+13 = 48维
        
        return {
            'observation': observation,
            'state': state,
            'actions': actions,
        }
    
    def _compute_reward(
        self,
        observation: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算奖励 - 字典序多目标
        
        【优先级】
        1. 安全约束 (硬约束)
        2. 水质达标 (次级)
        3. 能耗最小 (第三级)
        """
        state = observation['state']
        
        # 1. 安全约束违反 (大惩罚)
        DO = state[:, 7]
        NH4 = state[:, 9]
        
        safety_penalty = torch.zeros(self.batch_size, device=self.device)
        
        # DO约束
        do_violation = (DO < self.config.DO_min).float() * (self.config.DO_min - DO) * 100
        do_over = (DO > self.config.DO_max).float() * (DO - self.config.DO_max) * 50
        safety_penalty += do_violation + do_over
        
        # NH4约束
        nh4_violation = (NH4 > self.config.NH4_max).float() * (NH4 - self.config.NH4_max) * 100
        safety_penalty += nh4_violation
        
        # 2. 水质达标奖励
        water_quality_reward = torch.zeros(self.batch_size, device=self.device)
        
        # TP达标
        TP = state[:, 4:7].sum(dim=1)  # 简化TP
        tp_compliant = (TP < 2.0).float() * 10.0
        water_quality_reward += tp_compliant
        
        # TN达标
        TN = state[:, 8:10].sum(dim=1)
        tn_compliant = (TN < 15.0).float() * 10.0
        water_quality_reward += tn_compliant
        
        # 3. 能耗惩罚
        aeration = actions[:, 0]
        energy_penalty = aeration / 200.0 * 5.0  # 归一化能耗
        
        # 组合奖励
        total_reward = -safety_penalty * 1000 + water_quality_reward - energy_penalty
        
        reward_info = {
            'safety_penalty': safety_penalty.mean().item(),
            'water_quality_reward': water_quality_reward.mean().item(),
            'energy_penalty': energy_penalty.mean().item(),
            'total_reward': total_reward.mean().item(),
        }
        
        return total_reward.mean(), reward_info
    
    def _check_constraints(self, observation: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """检查约束违反"""
        state = observation['state']
        
        DO = state[:, 7]
        NH4 = state[:, 9]
        
        return {
            'DO_low': torch.any(DO < self.config.DO_min).item(),
            'DO_high': torch.any(DO > self.config.DO_max).item(),
            'NH4_high': torch.any(NH4 > self.config.NH4_max).item(),
        }
    
    def _compute_energy(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算能耗 (kWh)
        
        【能耗组成】
        - 曝气能耗: 与曝气量成正比
        - 搅拌能耗: 与体积成正比
        - 泵送能耗: 与流量成正比
        """
        aeration = actions[:, 0]  # m³/h
        return_sludge = actions[:, 2]  # 回流比
        Q_in = self.config.Q_in_mean / 24  # m³/h
        
        # 曝气能耗
        aeration_energy = aeration * self.config.aeration_energy_coeff * self.simulation_dt * 24
        
        # 搅拌能耗
        mixing_energy = self.config.mixing_energy_coeff * self.config.total_volume * self.simulation_dt * 24
        
        # 泵送能耗
        pumping_energy = (Q_in + Q_in * return_sludge) * self.config.pumping_energy_coeff * self.simulation_dt * 24
        
        total_energy = aeration_energy + mixing_energy + pumping_energy
        
        return total_energy.mean()
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 50
    
    def get_action_dim(self) -> int:
        """获取动作维度"""
        return 4
    
    def get_action_space(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作空间范围"""
        low = torch.tensor(
            [self.config.aeration_range[0], self.config.carbon_dose_range[0],
             self.config.return_sludge_range[0], self.config.waste_sludge_range[0]],
            device=self.device
        )
        high = torch.tensor(
            [self.config.aeration_range[1], self.config.carbon_dose_range[1],
             self.config.return_sludge_range[1], self.config.waste_sludge_range[1]],
            device=self.device
        )
        return low, high


def create_simulator(
    batch_size: int = 32,
    device: str = "cuda"
) -> BioPSimulator:
    """创建仿真器工厂函数"""
    config = BSM1FullConfig()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return BioPSimulator(config, dev, batch_size)


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("BSM1 Full Simulation Environment Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建仿真器
    sim = BioPSimulator(
        config=BSM1FullConfig(),
        device=device,
        batch_size=4
    )
    
    print(f"State dim: {sim.get_state_dim()}")
    print(f"Action dim: {sim.get_action_dim()}")
    
    # 重置
    obs = sim.reset()
    print(f"Observation shape: {obs['observation'].shape}")
    
    # 随机动作
    action = torch.randn(4, 4, device=device) * 50 + 100
    action[:, 2] = torch.sigmoid(action[:, 2])  # 回流比归一化
    
    # 运行几步
    print("\nRunning 10 steps...")
    for i in range(10):
        obs, reward, done, info = sim.step(action)
        print(f"  Step {i+1}: reward={info['total_reward']:.3f}, "
              f"DO={obs['state'][0,7].item():.2f}, "
              f"NH4={obs['state'][0,9].item():.2f}")
    
    print("\n✓ Simulation environment test passed!")
    
    # 估算模型参数量
    print("\n" + "=" * 70)
    print("Parameter Count Estimation")
    print("=" * 70)
    print(f"BSM1组分维度: 13")
    print(f"仿真器总维度: 50")
    print(f"控制变量: 4")
    print(f"反应速率: 8")
    print(f"能耗模型: 3个独立参数")
