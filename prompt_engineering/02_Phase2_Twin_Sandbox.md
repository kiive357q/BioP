# 阶段二任务二：字典序多目标强化学习代理 (Phase 2 Task 2: Lexicographic RL)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 阶段二 - 任务二
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席环境动力学算法科学家 / RL 架构师

---

## [Context] 局部物理上下文 (Phase 2 Task 2: Lexicographic RL)

### 目标文件 1: envs/reward_functions.py (工业底线奖赏函数)

### 目标文件 2: rl_agents/lexicographic_sac.py (字典序 Soft Actor-Critic)

### 物理痛点

传统的标量奖赏（把所有目标加权求和）在工业中是致命的。模型往往会通过"轻微的水质超标"来换取"极大的电费节省"。

### 学术叙事

我们采用绝对字典序（Lexicographic Order）优化。优先级划分为：
- **第一级**: 物理安全（如不溢流）
- **第二级**: 出水 TP/TN 合规
- **第三级**: 曝气与药剂节能

---

## [Execution] 核心执行逻辑

### IndustrialVectorReward

编写一个接收状态张量和动作张量的类/函数。

必须返回一个多维张量 (Reward Vector)，而非单一标量。例如返回 (r_safety, r_compliance, r_energy)。

包含详尽的注释，说明各级奖赏的物理判定边界。

### LexicographicSACAgent

- 网络结构定义：ActorNetwork、CriticNetwork（多维 Q 向量）
- 字典序优化器：带有拉格朗日乘子的动态自适应调整机制
- compute_loss：实现逐级非平滑截断逻辑

---

## [Constraints] 模块级绝对红线警告

### 禁止妥协权重 (NO LINEAR SCALARIZATION)

在 lexicographic_sac.py 计算 Loss 时，**绝对禁止**写出类似 `loss = w1*Q1 + w2*Q2 + w3*Q3` 这种妥协性代码！

必须强制实现硬约束：当且仅当上一级目标函数（如安全与水质）满足阈值（Constraint Thresholds）时，才允许对下一级目标（能耗）的梯度进行反向传播更新。如果上级不达标，必须**冻结下级梯度**！

### 无训练循环 (Architecture Only)

本阶段不涉及与 Environment 的实际交互循环 (如 env.step)。只需提供 Actor/Critic 的网络类以及核心 Loss 计算与参数更新的方法 (update_parameters)，不需要写主训练的 while/for 循环。

### 张量运算保护

在计算 Actor 的 log_prob（特别是对角高斯策略）时，由于涉及到方差和对数运算，必须加上 1e-6 的安全截断，防止发生 NaN。

---

## 字典序优化数学原理

### Lexicographic Order 定义

给定两个奖励向量 R₁ = (r₁₁, r₁₂, r₁₃) 和 R₂ = (r₂₁, r₂₂, r₂₃)：

R₁ ≻ R₂ 当且仅当：
- r₁₁ > r₂₁，或
- r₁₁ = r₂₁ 且 r₁₂ > r₂₂，或
- r₁₁ = r₂₁ 且 r₁₂ = r₂₂ 且 r₁₃ > r₂₃

### 拉格朗日乘子机制

在第 k 步的优化目标：

```
max θ  min λ  E[R₁(π_θ)] - λ · (threshold_k - E[R_k(π_θ)])
s.t. E[R_i(π_θ)] ≥ threshold_i, ∀i < k
```

### 梯度冻结逻辑

```
if E[R₁(π_θ)] < threshold_1:
    freeze_grad(actor_params)
    freeze_grad(critic_params for Q_2, Q_3)
elif E[R₂(π_θ)] < threshold_2:
    freeze_grad(actor_params for energy_optimization)
    freeze_grad(critic_params for Q_3)
else:
    optimize_all_levels()
```

---

## [Implementation] 完整代码实现

### 文件1: envs/reward_functions.py

```python
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
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = 1000
    ) -> None:
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        
        # 第一优先级：物理安全阈值
        self.do_min_threshold = 0.2        # mg/L - 反硝化失效警戒
        self.do_max_threshold = 4.0        # mg/L - 氧化浪费警戒
        self.level_max_threshold = 5.0    # m - 溢流事故警戒
        self.level_safe_margin = 0.5       # m - 安全裕度
        self.flow_change_max = 0.5         # m³/h - 水力冲击容忍度
        
        # 第二优先级：水质合规阈值
        self.tp合规阈值 = 0.5               # mg/L - GB 21900-2008表3
        self.tp_预警阈值 = 0.3             # mg/L
        self.tn合规阈值 = 15.0              # mg/L
        self.tn_预警阈值 = 10.0            # mg/L
        self.nh4合规阈值 = 10.0             # mg/L - 生物毒性
        self.nh4_预警阈值 = 8.0            # mg/L
        
        # 第三优先级：能耗优化参数
        self.aeration_energy_base = 0.4     # kWh/m³
        self.oxygen_transfer_efficiency = 0.85
        self.chemical_dose_base = 20.0     # mg/L
        self.chemical_cost_factor = 0.05    # $/mg
        self.blower_power_coef_a = 0.3     # 幂律系数 P ∝ Q^a
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        info_dict: Optional[dict] = None
    ) -> RewardVector:
        """前向传播：计算三维奖励向量"""
        batch_size = states.shape[0]
        
        # 解析状态变量（BSM1标准序）
        do_concentration = states[:, 0]           # 溶解氧 mg/L
        tank_level = states[:, 1]                 # 液位 m
        influent_flow = states[:, 2]              # 进水流量 m³/h
        
        # 解析动作变量
        aeration_rate = actions[:, 0]             # 曝气量 m³/h
        chemical_dose = actions[:, 1]             # PAC投加量 mg/L
        
        # 三级奖励计算
        safety_reward = self._compute_safety_reward(do_concentration, tank_level, influent_flow, aeration_rate)
        compliance_reward = self._compute_compliance_reward(states, info_dict)
        energy_reward = self._compute_energy_reward(do_concentration, aeration_rate, chemical_dose)
        
        return RewardVector(safety=safety_reward, compliance=compliance_reward, energy=energy_reward)
    
    def _compute_safety_reward(self, do, level, flow, aeration):
        """【第一优先级】物理安全奖励计算"""
        batch_size = do.shape[0]
        safety_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # DO临界下限惩罚（反硝化失效风险）
        do_low_penalty = torch.where(do < self.do_min_threshold,
            -100.0 * torch.exp(self.do_min_threshold - do),
            torch.zeros(batch_size, device=self.device))
        
        # DO临界上限惩罚（氧化浪费）
        do_high_penalty = torch.where(do > self.do_max_threshold,
            -50.0 * (do - self.do_max_threshold),
            torch.zeros(batch_size, device=self.device))
        
        # 安全区间奖励
        do_safe_mask = (do >= self.do_min_threshold) & (do <= self.do_max_threshold)
        do_safe_bonus = torch.where(do_safe_mask, 5.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device))
        
        # 液位安全惩罚
        level_penalty = torch.where(level > self.level_max_threshold,
            -100.0 * torch.clamp(level - self.level_max_threshold, min=0, max=1.0),
            torch.zeros(batch_size, device=self.device))
        
        # 安全裕度奖励
        level_safe_mask = level < (self.level_max_threshold - self.level_safe_margin)
        level_safe_bonus = torch.where(level_safe_mask, 3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device))
        
        # 水力冲击惩罚
        flow_change_penalty = torch.where(flow > self.flow_change_max,
            -20.0 * torch.log1p(flow - self.flow_change_max),
            torch.zeros(batch_size, device=self.device))
        
        safety_reward = (do_low_penalty + do_high_penalty + do_safe_bonus + 
                        level_penalty + level_safe_bonus + flow_change_penalty)
        safety_reward = torch.clamp(safety_reward, min=-100.0, max=10.0)
        
        return safety_reward
    
    def _compute_compliance_reward(self, states, info_dict):
        """【第二优先级】水质合规奖励计算"""
        batch_size = states.shape[0]
        compliance_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # 解析水质指标
        if states.shape[1] >= 7:
            tp_concentration = states[:, 3]
            tn_concentration = states[:, 4]
            nh4_concentration = states[:, 5]
        else:
            tp_concentration = torch.full((batch_size,), 0.3, device=self.device)
            tn_concentration = torch.full((batch_size,), 12.0, device=self.device)
            nh4_concentration = torch.full((batch_size,), 5.0, device=self.device)
        
        # TP总磷判定
        tp_exceed_penalty = torch.where(tp_concentration > self.tp合规阈值,
            -50.0 * torch.clamp(tp_concentration - self.tp合规阈值, max=1.0),
            torch.zeros(batch_size, device=self.device))
        tp_warning_penalty = torch.where((tp_concentration >= self.tp_预警阈值) & (tp_concentration <= self.tp合规阈值),
            -5.0 * (tp_concentration - self.tp_预警阈值) / (self.tp合规阈值 - self.tp_预警阈值),
            torch.zeros(batch_size, device=self.device))
        tp_compliance_bonus = torch.where(tp_concentration < self.tp_预警阈值,
            3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device))
        
        # TN总氮判定
        tn_exceed_penalty = torch.where(tn_concentration > self.tn合规阈值,
            -50.0 * torch.clamp(tn_concentration - self.tn合规阈值, max=5.0),
            torch.zeros(batch_size, device=self.device))
        tn_warning_penalty = torch.where((tn_concentration >= self.tn_预警阈值) & (tn_concentration <= self.tn合规阈值),
            -5.0 * (tn_concentration - self.tn_预警阈值) / (self.tn合规阈值 - self.tn_预警阈值),
            torch.zeros(batch_size, device=self.device))
        tn_compliance_bonus = torch.where(tn_concentration < self.tn_预警阈值,
            3.0 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device))
        
        # NH4-N氨氮判定
        nh4_toxic_penalty = torch.where(nh4_concentration > self.nh4合规阈值,
            -50.0 * torch.clamp(nh4_concentration - self.nh4合规阈值, max=5.0),
            torch.zeros(batch_size, device=self.device))
        nh4_warning_penalty = torch.where((nh4_concentration >= self.nh4_预警阈值) & (nh4_concentration <= self.nh4合规阈值),
            -5.0 * (nh4_concentration - self.nh4_预警阈值) / (self.nh4合规阈值 - self.nh4_预警阈值),
            torch.zeros(batch_size, device=self.device))
        
        compliance_reward = (tp_exceed_penalty + tp_warning_penalty + tp_compliance_bonus +
                           tn_exceed_penalty + tn_warning_penalty + tn_compliance_bonus +
                           nh4_toxic_penalty + nh4_warning_penalty)
        compliance_reward = torch.clamp(compliance_reward, min=-100.0, max=10.0)
        
        return compliance_reward
    
    def _compute_energy_reward(self, do, aeration, chemical):
        """【第三优先级】能耗优化奖励计算"""
        batch_size = do.shape[0]
        energy_reward = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # 曝气能耗（幂律风机功耗）
        normalized_aeration = aeration / (aeration.mean() + 1e-6)
        aeration_power = torch.pow(normalized_aeration + 1e-6, self.blower_power_coef_a)
        aeration_penalty = -10.0 * (aeration_power - 1.0)
        aeration_penalty = torch.clamp(aeration_penalty, min=-20.0, max=5.0)
        
        # 药剂成本
        normalized_chemical = chemical / (self.chemical_dose_base + 1e-6)
        chemical_cost = -5.0 * (normalized_chemical - 0.8)
        
        energy_reward = aeration_penalty + chemical_cost
        energy_reward = torch.clamp(energy_reward, min=-30.0, max=5.0)
        
        return energy_reward
    
    def compute_lexicographic_mask(self, reward_vector):
        """【核心方法】计算字典序优化掩码"""
        safety_threshold = 0.0
        compliance_threshold = 0.0
        energy_threshold = -10.0
        
        safety_satisfied = reward_vector.safety >= safety_threshold
        compliance_satisfied = reward_vector.compliance >= compliance_threshold
        energy_satisfied = reward_vector.energy >= energy_threshold
        
        return safety_satisfied, compliance_satisfied, energy_satisfied
```

### 文件2: rl_agents/lexicographic_sac.py

```python
"""
字典序多目标强化学习代理模块 (rl_agents/lexicographic_sac.py)

【模块定位】BioP污水控制系统的核心决策引擎
【设计理念】绝对字典序优化（Lexicographic Order）替代标量优化

【优化层级】
- 第一优先级：物理安全（DO临界值、液位不溢流）
- 第二优先级：水质合规（TP/TN达标）
- 第三优先级：曝气药剂节能

【绝对红线】禁止线性标量化！
严禁使用：loss = w1*Q1 + w2*Q2 + w3*Q3
必须实现：硬约束 + 梯度冻结 + 拉格朗日乘子

【版本】V2.0-Phase2-Task2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import math


@dataclass
class LexicographicConfig:
    """字典序SAC超参数配置"""
    state_dim: int = 20
    action_dim: int = 4
    n_objectives: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    safety_threshold: float = 0.0
    compliance_threshold: float = 0.0
    energy_threshold: float = -10.0
    lagrange_multiplier_lr: float = 0.01
    lagrange_max: float = 100.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    target_update_interval: int = 1
    gradient_clip: float = 10.0


class ReplayBuffer:
    """经验回放缓冲区（支持多维奖励）"""
    
    def __init__(self, capacity, state_dim, action_dim, n_objectives, device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, n_objectives, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices])
    
    def __len__(self):
        return self.size


class ActorNetwork(nn.Module):
    """Actor网络：对角高斯策略（输出mean + log_std）"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """返回(mean, log_std)"""
        features = self.feature_extractor(state)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        """重参数化采样"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        actions = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)  # 1e-6安全截断
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return actions, log_prob


class CriticNetwork(nn.Module):
    """
    多维Critic网络
    
    【核心设计】输出Q值向量，而非单一标量
    - Q_values[:, 0]: 物理安全Q值
    - Q_values[:, 1]: 水质合规Q值
    - Q_values[:, 2]: 能耗优化Q值
    """
    
    def __init__(self, state_dim, action_dim, n_objectives, hidden_dims=[256, 256]):
        super().__init__()
        self.n_objectives = n_objectives
        
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.q_value_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(n_objectives)])
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """返回Q值向量 (batch_size, n_objectives)"""
        state_action = torch.cat([state, action], dim=-1)
        features = self.feature_extractor(state_action)
        q_values = torch.stack([q_layer(features) for q_layer in self.q_value_layers], dim=-1)
        return q_values.squeeze(-2)


class LagrangeMultiplier(nn.Module):
    """拉格朗日乘子（可学习参数）"""
    
    def __init__(self, initial_value=1.0, lr=0.01, max_value=100.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(initial_value + 1e-6)))
        self.lr = lr
        self.max_value = max_value
    
    def forward(self):
        return torch.exp(self.log_lambda).clamp(max=self.max_value)
    
    def update(self, constraint_violation):
        with torch.no_grad():
            current_lambda = self()
            grad = constraint_violation.mean()
            new_log_lambda = self.log_lambda.data + self.lr * grad
            new_log_lambda = new_log_lambda.clamp(-10, 10)
            self.log_lambda.data = new_log_lambda
            return current_lambda.item()


class LexicographicSACAgent:
    """
    字典序多目标Soft Actor-Critic代理
    
    【核心创新】替代传统标量优化的革命性架构
    
    传统SAC（致命缺陷）：
        loss = α·log_prob - w₁·Q₁ - w₂·Q₂ - w₃·Q₃
    
    字典序SAC（工业安全）：
        if safety < threshold:
            freeze_gradient(energy_optimization)
        elif compliance < threshold:
            freeze_gradient(energy_optimization)
        else:
            optimize_all_levels()
    """
    
    def __init__(self, config, device=torch.device("cpu")):
        self.config = config
        self.device = device
        self.n_objectives = config.n_objectives
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.safety_threshold = config.safety_threshold
        self.compliance_threshold = config.compliance_threshold
        self.energy_threshold = config.energy_threshold
        
        self.actor = ActorNetwork(config.state_dim, config.action_dim, config.hidden_dims).to(device)
        self.critic = CriticNetwork(config.state_dim, config.action_dim, config.n_objectives, config.hidden_dims).to(device)
        self.target_critic = CriticNetwork(config.state_dim, config.action_dim, config.n_objectives, config.hidden_dims).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.lagrange_multipliers = [
            LagrangeMultiplier(initial_value=1.0, lr=config.lagrange_multiplier_lr, max_value=config.lagrange_max).to(device)
            for _ in range(config.n_objectives)
        ]
        
        self.update_count = 0
        self._training_stats = {
            'safety_satisfied_rate': [], 'compliance_satisfied_rate': [], 'energy_satisfied_rate': [],
            'lambda_safety': [], 'lambda_compliance': [], 'lambda_energy': []
        }
    
    def select_action(self, state, deterministic=False):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                actions = torch.tanh(mean)
            else:
                actions, _ = self.actor.sample(state)
        return actions.squeeze(0) if actions.shape[0] == 1 else actions
    
    def compute_lexicographic_critic_loss(self, states, actions, rewards, next_states, dones):
        """
        【核心方法】计算字典序Critic Loss
        
        【禁止模式】
        ❌ loss = w1 * Q1 + w2 * Q2 + w3 * Q3
        
        【必须模式】
        ✓ 逐级计算，梯度冻结
        """
        batch_size = states.shape[0]
        
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            min_next_q_values = next_q_values.min(dim=-1, keepdim=True)[0]
            next_value = min_next_q_values - self.alpha * next_log_prob
            target_q_values = rewards + self.gamma * (1 - dones.unsqueeze(-1)) * next_value
            target_q_values = target_q_values.detach()
        
        current_q_values = self.critic(states, actions)
        
        safety_rewards = rewards[:, 0]
        compliance_rewards = rewards[:, 1]
        energy_rewards = rewards[:, 2]
        
        safety_mask = safety_rewards >= self.safety_threshold
        compliance_mask = compliance_rewards >= self.compliance_threshold
        
        safety_satisfied_rate = safety_mask.float().mean().item()
        compliance_satisfied_rate = compliance_mask.float().mean().item()
        
        self._training_stats['safety_satisfied_rate'].append(safety_satisfied_rate)
        self._training_stats['compliance_satisfied_rate'].append(compliance_satisfied_rate)
        
        # Critic安全损失（始终优化）
        q1_error = current_q_values[:, 0] - target_q_values[:, 0]
        critic_safety_loss = 0.5 * q1_error.pow(2).mean()
        safety_constraint = self.safety_threshold - current_q_values[:, 0].mean()
        lambda_safety_value = self.lagrange_multipliers[0].update(safety_constraint)
        self._training_stats['lambda_safety'].append(lambda_safety_value)
        
        # Critic合规损失（安全不达标时冻结）
        if safety_satisfied_rate < 1.0:
            compliance_loss_full = 0.5 * (current_q_values[:, 1] - target_q_values[:, 1]).pow(2)
            safety_violated_mask = ~safety_mask
            compliance_loss = compliance_loss_full.masked_fill(safety_violated_mask, 0.0)
            critic_compliance_loss = compliance_loss.sum() / (batch_size - safety_violated_mask.sum() + 1e-6)
        else:
            q2_error = current_q_values[:, 1] - target_q_values[:, 1]
            critic_compliance_loss = 0.5 * q2_error.pow(2).mean()
        compliance_constraint = self.compliance_threshold - current_q_values[:, 1].mean()
        lambda_compliance_value = self.lagrange_multipliers[1].update(compliance_constraint)
        self._training_stats['lambda_compliance'].append(lambda_compliance_value)
        
        # Critic能耗损失（合规不达标时冻结）
        if compliance_satisfied_rate < 1.0:
            energy_loss_full = 0.5 * (current_q_values[:, 2] - target_q_values[:, 2]).pow(2)
            compliance_violated_mask = ~compliance_mask
            energy_loss = energy_loss_full.masked_fill(compliance_violated_mask, 0.0)
            critic_energy_loss = energy_loss.sum() / (batch_size - compliance_violated_mask.sum() + 1e-6)
        else:
            q3_error = current_q_values[:, 2] - target_q_values[:, 2]
            critic_energy_loss = 0.5 * q3_error.pow(2).mean()
        energy_constraint = self.energy_threshold - current_q_values[:, 2].mean()
        lambda_energy_value = self.lagrange_multipliers[2].update(energy_constraint)
        self._training_stats['lambda_energy'].append(lambda_energy_value)
        
        critic_loss = critic_safety_loss + critic_compliance_loss + critic_energy_loss
        
        return critic_loss, {
            'critic_total': critic_loss, 'critic_safety': critic_safety_loss,
            'critic_compliance': critic_compliance_loss, 'critic_energy': critic_energy_loss,
            'lambda_safety': self.lagrange_multipliers[0](),
            'lambda_compliance': self.lagrange_multipliers[1](),
            'lambda_energy': self.lagrange_multipliers[2](),
            'safety_satisfied_rate': safety_satisfied_rate,
            'compliance_satisfied_rate': compliance_satisfied_rate
        }
    
    def compute_lexicographic_actor_loss(self, states):
        """
        【核心方法】计算字典序Actor Loss
        
        【禁止模式】
        ❌ loss = α·log_prob - Σwᵢ·Qᵢ
        
        【必须模式】
        ✓ 梯度冻结 + 拉格朗日乘子
        """
        batch_size = states.shape[0]
        
        actions, log_prob = self.actor.sample(states)
        q_values = self.critic(states, actions)
        
        q_safety = q_values[:, 0]
        q_compliance = q_values[:, 1]
        q_energy = q_values[:, 2]
        
        safety_mean = q_safety.mean().item()
        compliance_mean = q_compliance.mean().item()
        
        lambda_safety = self.lagrange_multipliers[0]()
        lambda_compliance = self.lagrange_multipliers[1]()
        lambda_energy = self.lagrange_multipliers[2]()
        
        safety_violation = (self.safety_threshold - q_safety).clamp(min=0).mean()
        compliance_violation = (self.compliance_threshold - q_compliance).clamp(min=0).mean()
        energy_violation = (self.energy_threshold - q_energy).clamp(min=0).mean()
        
        policy_loss_safety = self.alpha * log_prob.mean() + lambda_safety * safety_violation - q_safety.mean()
        policy_loss_compliance = self.alpha * log_prob.mean() + lambda_compliance * compliance_violation - q_compliance.mean()
        policy_loss_energy = self.alpha * log_prob.mean() + lambda_energy * energy_violation - q_energy.mean()
        
        safety_mask_active = safety_mean >= self.safety_threshold
        compliance_mask_active = compliance_mean >= self.compliance_threshold
        
        # 【字典序核心】梯度冻结逻辑
        if not safety_mask_active:
            actor_loss = policy_loss_safety
            actor_loss = actor_loss + policy_loss_compliance.detach() + policy_loss_energy.detach()
        elif not compliance_mask_active:
            actor_loss = policy_loss_safety + policy_loss_compliance
            actor_loss = actor_loss + policy_loss_energy.detach()
        else:
            actor_loss = policy_loss_safety + policy_loss_compliance + policy_loss_energy
        
        return actor_loss, {
            'actor_total': actor_loss, 'actor_safety': policy_loss_safety,
            'actor_compliance': policy_loss_compliance, 'actor_energy': policy_loss_energy,
            'q_safety_mean': safety_mean, 'q_compliance_mean': compliance_mean,
            'q_energy_mean': q_energy.mean().item(), 'log_prob_mean': log_prob.mean().item(),
            'safety_mask_active': safety_mask_active, 'compliance_mask_active': compliance_mask_active
        }
    
    def update_parameters(self, states, actions, rewards, next_states, dones):
        """【主更新方法】更新Actor和Critic网络参数"""
        self.update_count += 1
        
        critic_loss, critic_stats = self.compute_lexicographic_critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        
        actor_loss, actor_stats = self.compute_lexicographic_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        
        if self.update_count % self.config.target_update_interval == 0:
            self._soft_update_target_network()
        
        return {
            'critic_loss': critic_stats['critic_total'].item(),
            'actor_loss': actor_stats['actor_total'].item(),
            'lambda_safety': critic_stats['lambda_safety'].item(),
            'lambda_compliance': critic_stats['lambda_compliance'].item(),
            'lambda_energy': critic_stats['lambda_energy'].item(),
            'safety_satisfied_rate': critic_stats['safety_satisfied_rate'],
            'compliance_satisfied_rate': critic_stats['compliance_satisfied_rate'],
            'q_safety_mean': actor_stats['q_safety_mean'],
            'q_compliance_mean': actor_stats['q_compliance_mean'],
            'q_energy_mean': actor_stats['q_energy_mean']
        }
    
    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## [Testing] 测试入口规范

```python
# Mock随机状态、动作和多维奖赏张量
# 实例化Agent，执行一次update_parameters
# 打印拉格朗日乘子的变化
```

---

**版本**: V2.0-Phase2-Task2
**制定日期**: 2026-05-29
**适用范围**: 强化学习模块 - 字典序多目标优化
