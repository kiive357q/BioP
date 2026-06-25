"""
环境奖励函数模块 (envs/reward_functions.py)

【模块定位】污水生化除磷数字孪生的工业级奖励函数
【设计理念】绝对字典序优化：安全 > 水质合规 > 能耗优化

物理安全边界（第一优先级）：
- DO溶解氧浓度：过低导致反硝化抑制（< 0.2 mg/L），过高导致氧化浪费（> 4.0 mg/L）
- 液位安全：防止溢流（Level < 5.0 m）
- 流量稳定性：防止水力冲击（ΔFlow < 0.5 m³/h）

水质合规边界（第二优先级）：
- TP总磷：工业排放标准 ≤ 0.5 mg/L（GB 21900-2008）
- TN总氮：工业排放标准 ≤ 15 mg/L
- NH4-N氨氮：生物毒性阈值 ≤ 10 mg/L

能耗优化边界（第三优先级）：
- 曝气量：鼓风机能耗占运行成本60-70%
- 化学除磷药剂：PAC/PAM投加量最小化

【版本】V2.0-Phase2-Task2
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, NamedTuple


class RewardVector(NamedTuple):
    """奖励向量命名元组，保证类型安全"""
    safety: torch.Tensor
    compliance: torch.Tensor
    energy: torch.Tensor
    
    def sum(self) -> torch.Tensor:
        """【禁止使用】仅用于调试，返回标量和"""
        return self.safety + self.compliance + self.energy


class IndustrialVectorReward(nn.Module):
    """
    工业级向量化奖励计算器
    
    【核心职责】将多目标优化问题转化为可微分的张量运算
    【物理约束】所有阈值均为硬边界，违反即触发惩罚
    
    Args:
        device: torch.device for tensor operations
        batch_size: 并行仿真批次大小（默认1000，支持万级水厂并行）
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = 1000
    ) -> None:
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        
        # ============ 第一优先级：物理安全阈值 ============
        # 【安全红线】DO溶解氧临界值，过低触发反硝化失败
        self.do_min_threshold = 0.2        # mg/L
        self.do_max_threshold = 4.0         # mg/L（氧化浪费警戒）
        
        # 【安全红线】液位安全边界，防止溢流事故
        self.level_max_threshold = 5.0     # m
        self.level_safe_margin = 0.5        # m（安全裕度）
        
        # 【安全红线】水力冲击容忍度
        self.flow_change_max = 0.5          # m³/h
        
        # ============ 第二优先级：水质合规阈值 ============
        # 【法规红线】TP总磷（GB 21900-2008表3标准）
        self.tp合规阈值 = 0.5               # mg/L
        self.tp_预警阈值 = 0.3              # mg/L（接近超标预警）
        
        # 【法规红线】TN总氮
        self.tn合规阈值 = 15.0              # mg/L
        self.tn_预警阈值 = 10.0             # mg/L
        
        # 【生物毒性】NH4-N氨氮
        self.nh4合规阈值 = 10.0             # mg/L
        self.nh4_预警阈值 = 8.0             # mg/L
        
        # ============ 第三优先级：能耗优化参数 ============
        # 曝气能耗系数（与DO控制直接相关）
        self.aeration_energy_base = 0.4     # kWh/m³（基准曝气能耗）
        self.oxygen_transfer_efficiency = 0.85  # 氧转移效率
        
        # 药剂能耗系数（化学除磷PAC）
        self.chemical_dose_base = 20.0     # mg/L（基准PAC投加量）
        self.chemical_cost_factor = 0.05   # $/mg（药剂单价系数）
        
        # 鼓风机功率特性（典型离心风机曲线）
        self.blower_power_coef_a = 0.3      # 幂律系数a (P ∝ Q^a)
        self.blower_power_coef_b = 0.5      # 幂律系数b
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        info_dict: Optional[dict] = None
    ) -> RewardVector:
        """
        前向传播：计算三维奖励向量
        
        【输入规格】
        - states: (batch_size, state_dim) 状态张量
        - actions: (batch_size, action_dim) 动作张量
        - info_dict: 包含生化指标的字典（可选）
        
        【输出规格】
        - RewardVector: (safety, compliance, energy) 三维奖励
        
        【物理约定】奖励值范围 [-100, +10]
        """
        batch_size = states.shape[0]
        
        # 解析状态变量（根据BSM1标准状态序）
        # 索引对应：DO=0, Level=1, Q_in=2, TP=3, TN=4, NH4=5, NO3=6, ...
        do_concentration = states[:, 0]           # 溶解氧 mg/L
        tank_level = states[:, 1]                   # 液位 m
        influent_flow = states[:, 2]                # 进水流量 m³/h
        
        # 解析动作变量（曝气量、药剂投加量）
        aeration_rate = actions[:, 0]               # 曝气量 m³/h
        chemical_dose = actions[:, 1]               # PAC投加量 mg/L
        
        # ============ 第一级：物理安全奖励 ============
        safety_reward = self._compute_safety_reward(
            do_concentration, tank_level, influent_flow, aeration_rate
        )
        
        # ============ 第二级：水质合规奖励 ============
        compliance_reward = self._compute_compliance_reward(
            states, info_dict
        )
        
        # ============ 第三级：能耗优化奖励 ============
        energy_reward = self._compute_energy_reward(
            do_concentration, aeration_rate, chemical_dose
        )
        
        return RewardVector(
            safety=safety_reward,
            compliance=compliance_reward,
            energy=energy_reward
        )
    
    def _compute_safety_reward(
        self,
        do: torch.Tensor,
        level: torch.Tensor,
        flow: torch.Tensor,
        aeration: torch.Tensor
    ) -> torch.Tensor:
        """
        【第一优先级】物理安全奖励计算
        
        【物理机制】
        1. DO过低（<0.2mg/L）：反硝化菌受抑制，TN累积风险
        2. DO过高（>4.0mg/L）：能源浪费，且抑制生物除磷
        3. 液位超限（>5.0m）：溢流事故风险
        4. 水力冲击（ΔFlow>0.5m³/h）：污泥流失风险
        
        【奖励曲线】
        - 完全安全：+10
        - 轻微偏离：-10 ~ 0
        - 严重偏离：-50 ~ -100
        """
        batch_size = do.shape[0]
        safety_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # ---------- 溶解氧安全判定 ----------
        # 临界下限惩罚：反硝化失效风险
        do_low_penalty = torch.where(
            do < self.do_min_threshold,
            -100.0 * torch.exp(self.do_min_threshold - do),  # 指数惩罚
            torch.zeros(batch_size, device=self.device)
        )
        
        # 临界上限惩罚：氧化浪费
        do_high_penalty = torch.where(
            do > self.do_max_threshold,
            -50.0 * (do - self.do_max_threshold),  # 线性惩罚
            torch.zeros(batch_size, device=self.device)
        )
        
        # 安全区间奖励（鼓励维持在合理范围）
        do_safe_mask = (do >= self.do_min_threshold) & (do <= self.do_max_threshold)
        do_safe_bonus = torch.where(
            do_safe_mask,
            5.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- 液位安全判定 ----------
        # 【工业红线】溢流风险
        level_penalty = torch.where(
            level > self.level_max_threshold,
            -100.0 * torch.clamp(level - self.level_max_threshold, min=0, max=1.0),
            torch.zeros(batch_size, device=self.device)
        )
        
        # 安全裕度内奖励
        level_safe_mask = level < (self.level_max_threshold - self.level_safe_margin)
        level_safe_bonus = torch.where(
            level_safe_mask,
            3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- 水力冲击判定 ----------
        # 流量变化率惩罚（简化计算，实际需对比上一时刻）
        flow_change_penalty = torch.where(
            flow > self.flow_change_max,
            -20.0 * torch.log1p(flow - self.flow_change_max),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- 综合安全奖励 ----------
        safety_reward = (
            do_low_penalty + 
            do_high_penalty + 
            do_safe_bonus + 
            level_penalty + 
            level_safe_bonus +
            flow_change_penalty
        )
        
        # 裁剪到物理合理范围
        safety_reward = torch.clamp(safety_reward, min=-100.0, max=10.0)
        
        return safety_reward
    
    def _compute_compliance_reward(
        self,
        states: torch.Tensor,
        info_dict: Optional[dict]
    ) -> torch.Tensor:
        """
        【第二优先级】水质合规奖励计算
        
        【物理机制】
        1. TP总磷：>0.5mg/L违反GB 21900-2008表3标准
        2. TN总氮：>15mg/L违反排放标准
        3. NH4-N：>10mg/L具有生物毒性
        
        【奖励策略】
        - 合规区间：+10（鼓励维持）
        - 预警区间：-5 ~ 0（提前干预）
        - 超标区间：-50（硬惩罚）
        """
        batch_size = states.shape[0]
        compliance_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # 解析水质指标（索引可能因状态维度配置而异）
        if states.shape[1] >= 7:
            tp_concentration = states[:, 3]    # TP mg/L
            tn_concentration = states[:, 4]    # TN mg/L
            nh4_concentration = states[:, 5]  # NH4-N mg/L
        else:
            # 默认值（实际应从info_dict或传感器获取）
            tp_concentration = torch.full((batch_size,), 0.3, device=self.device)
            tn_concentration = torch.full((batch_size,), 12.0, device=self.device)
            nh4_concentration = torch.full((batch_size,), 5.0, device=self.device)
        
        # ---------- TP总磷合规判定 ----------
        # 超标惩罚（违反GB标准）
        tp_exceed_penalty = torch.where(
            tp_concentration > self.tp合规阈值,
            -50.0 * torch.clamp(tp_concentration - self.tp合规阈值, max=1.0),
            torch.zeros(batch_size, device=self.device)
        )
        
        # 预警区间惩罚
        tp_warning_penalty = torch.where(
            (tp_concentration >= self.tp_预警阈值) & (tp_concentration <= self.tp合规阈值),
            -5.0 * (tp_concentration - self.tp_预警阈值) / (self.tp合规阈值 - self.tp_预警阈值),
            torch.zeros(batch_size, device=self.device)
        )
        
        # 达标奖励（鼓励维持低水平）
        tp_compliance_bonus = torch.where(
            tp_concentration < self.tp_预警阈值,
            3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- TN总氮合规判定 ----------
        tn_exceed_penalty = torch.where(
            tn_concentration > self.tn合规阈值,
            -50.0 * torch.clamp(tn_concentration - self.tn合规阈值, max=5.0),
            torch.zeros(batch_size, device=self.device)
        )
        
        tn_warning_penalty = torch.where(
            (tn_concentration >= self.tn_预警阈值) & (tn_concentration <= self.tn合规阈值),
            -5.0 * (tn_concentration - self.tn_预警阈值) / (self.tn合规阈值 - self.tn_预警阈值),
            torch.zeros(batch_size, device=self.device)
        )
        
        tn_compliance_bonus = torch.where(
            tn_concentration < self.tn_预警阈值,
            3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- NH4-N氨氮毒性判定 ----------
        nh4_toxic_penalty = torch.where(
            nh4_concentration > self.nh4合规阈值,
            -50.0 * torch.clamp(nh4_concentration - self.nh4合规阈值, max=5.0),
            torch.zeros(batch_size, device=self.device)
        )
        
        nh4_warning_penalty = torch.where(
            (nh4_concentration >= self.nh4_预警阈值) & (nh4_concentration <= self.nh4合规阈值),
            -5.0 * (nh4_concentration - self.nh4_预警阈值) / (self.nh4合规阈值 - self.nh4_预警阈值),
            torch.zeros(batch_size, device=self.device)
        )
        
        # ---------- 综合水质合规奖励 ----------
        compliance_reward = (
            tp_exceed_penalty + tp_warning_penalty + tp_compliance_bonus +
            tn_exceed_penalty + tn_warning_penalty + tn_compliance_bonus +
            nh4_toxic_penalty + nh4_warning_penalty
        )
        
        compliance_reward = torch.clamp(compliance_reward, min=-100.0, max=10.0)
        
        return compliance_reward
    
    def _compute_energy_reward(
        self,
        do: torch.Tensor,
        aeration: torch.Tensor,
        chemical: torch.Tensor
    ) -> torch.Tensor:
        """
        【第三优先级】能耗优化奖励计算
        
        【物理机制】
        1. 曝气能耗：与DO浓度正相关，幂律特性 P ∝ Q^0.5
        2. 药剂成本：PAC投加量与TP去除直接相关
        
        【优化目标】在满足安全和水质约束下，最小化能耗
        
        【奖励策略】
        - 基准能耗：0奖励
        - 高于基准：负奖励（惩罚）
        - 低于基准：正奖励（鼓励）
        """
        batch_size = do.shape[0]
        energy_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # ---------- 曝气能耗计算 ----------
        # 幂律风机功耗模型：P ∝ Q^a，a≈0.5（离心风机特性）
        normalized_aeration = aeration / (aeration.mean() + 1e-6)
        aeration_power = torch.pow(
            normalized_aeration + 1e-6,
            self.blower_power_coef_a
        )
        
        # 曝气量惩罚（鼓励在满足DO前提下降低曝气）
        aeration_penalty = -10.0 * (aeration_power - 1.0)
        aeration_penalty = torch.clamp(aeration_penalty, min=-20.0, max=5.0)
        
        # ---------- 化学药剂成本计算 ----------
        normalized_chemical = chemical / (self.chemical_dose_base + 1e-6)
        chemical_cost = -5.0 * (normalized_chemical - 0.8)  # 基准为0.8倍
        
        # ---------- 综合能耗奖励 ----------
        energy_reward = aeration_penalty + chemical_cost
        energy_reward = torch.clamp(energy_reward, min=-30.0, max=5.0)
        
        return energy_reward
    
    def compute_lexicographic_mask(
        self,
        reward_vector: RewardVector
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【核心方法】计算字典序优化掩码
        
        【功能】确定哪些目标满足阈值，允许对下一级进行梯度更新
        
        【物理约定】
        - safety_mask=True：允许优化能耗
        - compliance_mask=True：允许优化能耗
        - 所有掩码为False：冻结所有非安全相关参数
        
        Args:
            reward_vector: 当前奖励向量
            
        Returns:
            (safety_satisfied, compliance_satisfied, energy_satisfied) 布尔掩码
        """
        safety_threshold = 0.0   # 安全奖励阈值
        compliance_threshold = 0.0 # 合规奖励阈值
        energy_threshold = -10.0  # 能耗奖励阈值（允许一定浪费）
        
        safety_satisfied = reward_vector.safety >= safety_threshold
        compliance_satisfied = reward_vector.compliance >= compliance_threshold
        energy_satisfied = reward_vector.energy >= energy_threshold
        
        return safety_satisfied, compliance_satisfied, energy_satisfied


def compute_reward_vector(
    states: torch.Tensor,
    actions: torch.Tensor,
    info_dict: Optional[dict] = None,
    device: torch.device = torch.device("cpu")
) -> RewardVector:
    """
    【便捷函数】计算奖励向量（无需实例化类）
    
    Args:
        states: 状态张量 (batch_size, state_dim)
        actions: 动作张量 (batch_size, action_dim)
        info_dict: 额外信息字典
        device: 计算设备
        
    Returns:
        RewardVector: 三维奖励向量
    """
    reward_calculator = IndustrialVectorReward(device=device)
    return reward_calculator(states, actions, info_dict)


if __name__ == "__main__":
    """【测试入口】验证奖励函数连通性"""
    print("=" * 60)
    print("工业奖励函数连通性测试 (IndustrialVectorReward)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    
    # Mock随机状态张量（BSM1标准状态序）
    # 索引: DO=0, Level=1, Q_in=2, TP=3, TN=4, NH4=5, NO3=6, ...
    states = torch.randn(batch_size, 20, device=device)
    states[:, 0] = torch.clamp(states[:, 0], min=0.1, max=5.0)  # DO: 0.1~5.0 mg/L
    states[:, 1] = torch.clamp(states[:, 1], min=3.0, max=5.5)   # Level: 3.0~5.5 m
    states[:, 2] = torch.clamp(states[:, 2], min=0.1, max=1.0)  # Flow: 0.1~1.0 m³/h
    states[:, 3] = torch.clamp(states[:, 3], min=0.1, max=1.0)   # TP: 0.1~1.0 mg/L
    states[:, 4] = torch.clamp(states[:, 4], min=5.0, max=20.0) # TN: 5~20 mg/L
    states[:, 5] = torch.clamp(states[:, 5], min=1.0, max=15.0)  # NH4: 1~15 mg/L
    
    # Mock随机动作张量（曝气量、药剂投加）
    actions = torch.randn(batch_size, 4, device=device)
    actions[:, 0] = torch.clamp(actions[:, 0], min=50.0, max=200.0)  # Aeration: 50~200 m³/h
    actions[:, 1] = torch.clamp(actions[:, 1], min=5.0, max=50.0)   # Chemical: 5~50 mg/L
    
    # 实例化奖励计算器
    reward_calculator = IndustrialVectorReward(device=device, batch_size=batch_size)
    
    # 前向传播计算奖励
    reward_vector = reward_calculator(states, actions)
    
    # 计算字典序掩码
    safety_mask, compliance_mask, energy_mask = reward_calculator.compute_lexicographic_mask(reward_vector)
    
    # 输出统计信息
    print(f"\n【批次规模】Batch={batch_size}")
    print(f"\n--- 奖励向量统计 ---")
    print(f"安全奖励  (Safety)  : mean={reward_vector.safety.mean().item():.2f}, "
          f"std={reward_vector.safety.std().item():.2f}, "
          f"min={reward_vector.safety.min().item():.2f}, "
          f"max={reward_vector.safety.max().item():.2f}")
    print(f"合规奖励  (Compliance): mean={reward_vector.compliance.mean().item():.2f}, "
          f"std={reward_vector.compliance.std().item():.2f}, "
          f"min={reward_vector.compliance.min().item():.2f}, "
          f"max={reward_vector.compliance.max().item():.2f}")
    print(f"能耗奖励  (Energy)  : mean={reward_vector.energy.mean().item():.2f}, "
          f"std={reward_vector.energy.std().item():.2f}, "
          f"min={reward_vector.energy.min().item():.2f}, "
          f"max={reward_vector.energy.max().item():.2f}")
    
    print(f"\n--- 字典序掩码统计 ---")
    print(f"安全达标率 (Safety Satisfied)  : {safety_mask.sum().item()}/{batch_size} "
          f"({100*safety_mask.sum().item()/batch_size:.1f}%)")
    print(f"合规达标率 (Compliance Satisfied): {compliance_mask.sum().item()}/{batch_size} "
          f"({100*compliance_mask.sum().item()/batch_size:.1f}%)")
    print(f"能耗达标率 (Energy Satisfied)  : {energy_mask.sum().item()}/{batch_size} "
          f"({100*energy_mask.sum().item()/batch_size:.1f}%)")
    
    # 梯度追踪验证
    states.requires_grad_(True)
    actions.requires_grad_(True)
    reward_vector = reward_calculator(states, actions)
    
    print(f"\n--- 梯度追踪验证 ---")
    print(f"Safety gradient requires_grad: {reward_vector.safety.requires_grad}")
    print(f"Compliance gradient requires_grad: {reward_vector.compliance.requires_grad}")
    print(f"Energy gradient requires_grad: {reward_vector.energy.requires_grad}")
    
    print("\n" + "=" * 60)
    print("✓ 奖励函数模块连通性测试通过")
    print("=" * 60)
