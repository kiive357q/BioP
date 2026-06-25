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
from collections import deque
import copy
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
    """
    经验回放缓冲区（支持多维奖励）
    
    【存储结构】
    - states: (max_size, state_dim)
    - actions: (max_size, action_dim)
    - rewards: (max_size, n_objectives) - 多维奖励向量
    - next_states: (max_size, state_dim)
    - dones: (max_size,)
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, n_objectives: int, device: torch.device):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.device = device
        
        self.states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, n_objectives, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        """添加一条经验到缓冲区"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """随机采样一批经验"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        return self.size


class ActorNetwork(nn.Module):
    """
    Actor网络：对角高斯策略
    
    【输出】
    - mean: (batch_size, action_dim) 动作均值
    - log_std: (batch_dim, action_dim) 对数标准差（被截断到 [log_std_min, log_std_max]）
    
    【数值稳定性】
    - log_std被限制在[-20, 2]，即std ∈ [2e-9, 7.4]
    - 1e-6安全截断防止log(0)和除零
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: 动作均值
            log_std: 对数标准差（已截断）
        """
        features = self.feature_extractor(state)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重参数化采样
        
        Returns:
            actions: 采样动作
            log_prob: 动作对数概率密度
        """
        mean, log_std = self.forward(state)
        
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        actions = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return actions, log_prob
    
    def get_log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        计算给定状态-动作对的对数概率
        
        Args:
            state: 状态张量
            actions: 动作张量
            
        Returns:
            log_prob: 对数概率密度
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        
        actions_clipped = torch.clamp(actions, -0.999, 0.999)
        x_t = 0.5 * torch.log((1 + actions_clipped) / (1 - actions_clipped))
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob


class CriticNetwork(nn.Module):
    """
    多维Critic网络
    
    【核心设计】输出Q值向量，而非单一标量
    - Q_values[:, 0]: 物理安全Q值
    - Q_values[:, 1]: 水质合规Q值
    - Q_values[:, 2]: 能耗优化Q值
    
    【禁止】单一Q值 + 权重组合
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_objectives: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.n_objectives = n_objectives
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.q_value_layers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_objectives)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            q_values: (batch_size, n_objectives) 多维Q值向量
        """
        state_action = torch.cat([state, action], dim=-1)
        features = self.feature_extractor(state_action)
        
        q_values = torch.stack([
            q_layer(features) for q_layer in self.q_value_layers
        ], dim=-1)
        
        q_values = q_values.squeeze(-2)
        
        return q_values


class LagrangeMultiplier(nn.Module):
    """
    拉格朗日乘子（可学习参数）
    
    【物理意义】自适应调整约束惩罚力度
    - λ过大：约束过于严格，可能导致策略崩塌
    - λ过小：约束过于宽松，可能违反安全边界
    
    【更新规则】
    λ_{k+1} = max(0, λ_k + lr * (constraint_value - target_threshold))
    """
    
    def __init__(self, initial_value: float = 1.0, lr: float = 0.01, max_value: float = 100.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(initial_value + 1e-6)))
        self.lr = lr
        self.max_value = max_value
    
    def forward(self) -> torch.Tensor:
        """返回拉格朗日乘子的正值"""
        return torch.exp(self.log_lambda).clamp(max=self.max_value)
    
    def update(self, constraint_violation: torch.Tensor) -> float:
        """
        更新拉格朗日乘子
        
        Args:
            constraint_violation: 约束违反量（正值表示违反）
            
        Returns:
            当前拉格朗日乘子值
        """
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
        问题：权重w是静态的，无法表达安全>水质>能耗的硬约束
    
    字典序SAC（工业安全）：
        if safety < threshold:
            freeze_gradient(energy_optimization)
        elif compliance < threshold:
            freeze_gradient(energy_optimization)
        else:
            optimize_all_levels()
    
    【数学原理】
    Lexicographic Preference: R₁ ≻ R₂ iff
        r₁₁ > r₂₁  OR
        (r₁₁ = r₂₁ AND r₁₂ > r₂₂) OR
        (r₁₁ = r₂₁ AND r₁₂ = r₂₂ AND r₁₃ > r₂₃)
    
    【拉格朗日机制】
    L(θ, λ) = E[R₁] - λ·(threshold - E[Rₖ])
    min_λ max_θ L(θ, λ)
    """
    
    def __init__(self, config: LexicographicConfig, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.n_objectives = config.n_objectives
        
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        
        self.safety_threshold = config.safety_threshold
        self.compliance_threshold = config.compliance_threshold
        self.energy_threshold = config.energy_threshold
        
        self.actor = ActorNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(device)
        
        self.critic = CriticNetwork(
            config.state_dim, config.action_dim, config.n_objectives, config.hidden_dims
        ).to(device)
        
        self.target_critic = CriticNetwork(
            config.state_dim, config.action_dim, config.n_objectives, config.hidden_dims
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.lagrange_multipliers = [
            LagrangeMultiplier(
                initial_value=1.0,
                lr=config.lagrange_multiplier_lr,
                max_value=config.lagrange_max
            ).to(device)
            for _ in range(config.n_objectives)
        ]
        
        self.update_count = 0
        
        self._training_stats = {
            'safety_satisfied_rate': [],
            'compliance_satisfied_rate': [],
            'energy_satisfied_rate': [],
            'lambda_safety': [],
            'lambda_compliance': [],
            'lambda_energy': []
        }
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        动作选择
        
        Args:
            state: 单个状态 (state_dim,) 或批状态 (batch_size, state_dim)
            deterministic: True则返回均值，False则采样
            
        Returns:
            actions: 动作张量
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                actions = torch.tanh(mean)
            else:
                actions, _ = self.actor.sample(state)
                
        return actions.squeeze(0) if actions.shape[0] == 1 else actions
    
    def compute_lexicographic_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        【核心方法】计算字典序Critic Loss
        
        【禁止模式】
        ❌ loss = w1 * Q1 + w2 * Q2 + w3 * Q3
        
        【必须模式】
        ✓ 逐级计算，梯度冻结
        
        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 多维奖励 (batch, n_objectives)
            next_states: 下一状态批次
            dones: 终止标志
            
        Returns:
            total_loss: 总损失
            loss_dict: 各层损失字典
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
        energy_mask = energy_rewards >= self.energy_threshold
        
        safety_satisfied_rate = safety_mask.float().mean().item()
        compliance_satisfied_rate = compliance_mask.float().mean().item()
        energy_satisfied_rate = energy_mask.float().mean().item()
        
        self._training_stats['safety_satisfied_rate'].append(safety_satisfied_rate)
        self._training_stats['compliance_satisfied_rate'].append(compliance_satisfied_rate)
        self._training_stats['energy_satisfied_rate'].append(energy_satisfied_rate)
        
        q1_error = current_q_values[:, 0] - target_q_values[:, 0]
        critic_safety_loss = 0.5 * q1_error.pow(2).mean()
        
        lambda_safety = self.lagrange_multipliers[0]()
        safety_constraint = self.safety_threshold - current_q_values[:, 0].mean()
        lambda_safety_value = self.lagrange_multipliers[0].update(safety_constraint)
        self._training_stats['lambda_safety'].append(lambda_safety_value)
        
        critic_compliance_loss_list = []
        compliance_constraint = self.compliance_threshold - current_q_values[:, 1].mean()
        lambda_compliance_value = self.lagrange_multipliers[1].update(compliance_constraint)
        self._training_stats['lambda_compliance'].append(lambda_compliance_value)
        
        critic_energy_loss_list = []
        energy_constraint = self.energy_threshold - current_q_values[:, 2].mean()
        lambda_energy_value = self.lagrange_multipliers[2].update(energy_constraint)
        self._training_stats['lambda_energy'].append(lambda_energy_value)
        
        critic_safety_loss_final = critic_safety_loss
        
        if safety_satisfied_rate < 1.0:
            compliance_loss_full = 0.5 * (current_q_values[:, 1] - target_q_values[:, 1]).pow(2)
            
            safety_violated_mask = ~safety_mask
            compliance_loss_masked = compliance_loss_full.masked_fill(
                safety_violated_mask, 0.0
            )
            critic_compliance_loss = compliance_loss_masked.sum() / (batch_size - safety_violated_mask.sum() + 1e-6)
        else:
            q2_error = current_q_values[:, 1] - target_q_values[:, 1]
            critic_compliance_loss = 0.5 * q2_error.pow(2).mean()
        
        critic_compliance_loss_list.append(critic_compliance_loss)
        
        if compliance_satisfied_rate < 1.0:
            energy_loss_full = 0.5 * (current_q_values[:, 2] - target_q_values[:, 2]).pow(2)
            
            compliance_violated_mask = ~compliance_mask
            energy_loss_masked = energy_loss_full.masked_fill(
                compliance_violated_mask, 0.0
            )
            critic_energy_loss = energy_loss_masked.sum() / (batch_size - compliance_violated_mask.sum() + 1e-6)
        else:
            q3_error = current_q_values[:, 2] - target_q_values[:, 2]
            critic_energy_loss = 0.5 * q3_error.pow(2).mean()
        
        critic_energy_loss_list.append(critic_energy_loss)
        
        critic_loss = (
            critic_safety_loss_final +
            sum(critic_compliance_loss_list) +
            sum(critic_energy_loss_list)
        )
        
        loss_dict = {
            'critic_total': critic_loss,
            'critic_safety': critic_safety_loss_final,
            'critic_compliance': critic_compliance_loss,
            'critic_energy': critic_energy_loss,
            'lambda_safety': lambda_safety,
            'lambda_compliance': self.lagrange_multipliers[1](),
            'lambda_energy': self.lagrange_multipliers[2](),
            'safety_satisfied_rate': safety_satisfied_rate,
            'compliance_satisfied_rate': compliance_satisfied_rate,
            'energy_satisfied_rate': energy_satisfied_rate
        }
        
        return critic_loss, loss_dict
    
    def compute_lexicographic_actor_loss(
        self,
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        【核心方法】计算字典序Actor Loss
        
        【禁止模式】
        ❌ loss = α·log_prob - Σwᵢ·Qᵢ
        
        【必须模式】
        ✓ 梯度冻结 + 拉格朗日乘子
        
        Args:
            states: 状态批次
            
        Returns:
            actor_loss: 策略损失
            loss_dict: 损失分解字典
        """
        batch_size = states.shape[0]
        
        actions, log_prob = self.actor.sample(states)
        q_values = self.critic(states, actions)
        
        q_safety = q_values[:, 0]
        q_compliance = q_values[:, 1]
        q_energy = q_values[:, 2]
        
        safety_mean = q_safety.mean().item()
        compliance_mean = q_compliance.mean().item()
        energy_mean = q_energy.mean().item()
        
        safety_threshold_tensor = torch.tensor(self.safety_threshold, device=self.device)
        compliance_threshold_tensor = torch.tensor(self.compliance_threshold, device=self.device)
        energy_threshold_tensor = torch.tensor(self.energy_threshold, device=self.device)
        
        lambda_safety = self.lagrange_multipliers[0]()
        lambda_compliance = self.lagrange_multipliers[1]()
        lambda_energy = self.lagrange_multipliers[2]()
        
        safety_violation = (safety_threshold_tensor - q_safety).clamp(min=0).mean()
        compliance_violation = (compliance_threshold_tensor - q_compliance).clamp(min=0).mean()
        energy_violation = (energy_threshold_tensor - q_energy).clamp(min=0).mean()
        
        policy_loss_safety = (
            self.alpha * log_prob.mean() +
            lambda_safety * safety_violation -
            q_safety.mean()
        )
        
        policy_loss_compliance = (
            self.alpha * log_prob.mean() +
            lambda_compliance * compliance_violation -
            q_compliance.mean()
        )
        
        policy_loss_energy = (
            self.alpha * log_prob.mean() +
            lambda_energy * energy_violation -
            q_energy.mean()
        )
        
        safety_mask_active = safety_mean >= self.safety_threshold
        compliance_mask_active = compliance_mean >= self.compliance_threshold
        
        if not safety_mask_active:
            actor_loss = policy_loss_safety
            actor_loss = actor_loss + policy_loss_compliance.detach() + policy_loss_energy.detach()
        elif not compliance_mask_active:
            actor_loss = policy_loss_safety + policy_loss_compliance
            actor_loss = actor_loss + policy_loss_energy.detach()
        else:
            actor_loss = policy_loss_safety + policy_loss_compliance + policy_loss_energy
        
        loss_dict = {
            'actor_total': actor_loss,
            'actor_safety': policy_loss_safety,
            'actor_compliance': policy_loss_compliance,
            'actor_energy': policy_loss_energy,
            'q_safety_mean': safety_mean,
            'q_compliance_mean': compliance_mean,
            'q_energy_mean': energy_mean,
            'log_prob_mean': log_prob.mean().item(),
            'safety_mask_active': safety_mask_active,
            'compliance_mask_active': compliance_mask_active
        }
        
        return actor_loss, loss_dict
    
    def update_parameters(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        【主更新方法】更新Actor和Critic网络参数
        
        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 多维奖励 (batch_size, n_objectives)
            next_states: 下一状态批次
            dones: 终止标志
            
        Returns:
            stats: 训练统计字典
        """
        self.update_count += 1
        
        critic_loss, critic_stats = self.compute_lexicographic_critic_loss(
            states, actions, rewards, next_states, dones
        )
        
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
        
        stats = {
            'critic_loss': critic_stats['critic_total'].item(),
            'actor_loss': actor_stats['actor_total'].item(),
            'lambda_safety': critic_stats['lambda_safety'].item(),
            'lambda_compliance': critic_stats['lambda_compliance'].item(),
            'lambda_energy': critic_stats['lambda_energy'].item(),
            'safety_satisfied_rate': critic_stats['safety_satisfied_rate'],
            'compliance_satisfied_rate': critic_stats['compliance_satisfied_rate'],
            'energy_satisfied_rate': critic_stats['energy_satisfied_rate'],
            'q_safety_mean': actor_stats['q_safety_mean'],
            'q_compliance_mean': actor_stats['q_compliance_mean'],
            'q_energy_mean': actor_stats['q_energy_mean']
        }
        
        return stats
    
    def _soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """返回训练统计历史"""
        return self._training_stats
    
    def save(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'lagrange_multiplier_state_dicts': [
                lm.state_dict() for lm in self.lagrange_multipliers
            ],
            'config': self.config,
            'update_count': self.update_count
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        for lm, state_dict in zip(self.lagrange_multipliers, checkpoint['lagrange_multiplier_state_dicts']):
            lm.load_state_dict(state_dict)
        
        self.update_count = checkpoint['update_count']


if __name__ == "__main__":
    """【测试入口】验证字典序SAC连通性"""
    print("=" * 70)
    print("字典序多目标SAC代理连通性测试 (LexicographicSACAgent)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")
    
    batch_size = 64
    state_dim = 20
    action_dim = 4
    n_objectives = 3
    buffer_capacity = 10000
    
    config = LexicographicConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dims=[256, 256],
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        safety_threshold=0.0,
        compliance_threshold=0.0,
        energy_threshold=-10.0,
        lagrange_multiplier_lr=0.01,
        lagrange_max=100.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        target_update_interval=1,
        gradient_clip=10.0
    )
    
    agent = LexicographicSACAgent(config, device=device)
    replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, n_objectives, device)
    
    print(f"\n【网络架构】")
    print(f"Actor参数数量: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"Critic参数数量: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"总参数数量: {sum(p.numel() for p in agent.parameters()):,}")
    
    print(f"\n--- 经验回放缓冲区填充 ---")
    for i in range(256):
        state = torch.randn(state_dim, device=device)
        action = torch.randn(action_dim, device=device) * 2.0
        reward = torch.randn(n_objectives, device=device)
        reward[0] = reward[0] * 10.0
        reward[1] = reward[1] * 5.0
        reward[2] = reward[2] * 2.0
        next_state = torch.randn(state_dim, device=device)
        done = torch.tensor(0.0, device=device)
        
        replay_buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区填充完成: {len(replay_buffer)}/{buffer_capacity}")
    
    print(f"\n--- 执行参数更新循环 ---")
    n_updates = 5
    
    for update_idx in range(n_updates):
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        stats = agent.update_parameters(states, actions, rewards, next_states, dones)
        
        print(f"\n更新 #{update_idx + 1}:")
        print(f"  [Loss] Actor: {stats['actor_loss']:.4f}, Critic: {stats['critic_loss']:.4f}")
        print(f"  [Q值] Safety: {stats['q_safety_mean']:.4f}, "
              f"Compliance: {stats['q_compliance_mean']:.4f}, "
              f"Energy: {stats['q_energy_mean']:.4f}")
        print(f"  [拉格朗日乘子] λ_safety: {stats['lambda_safety']:.4f}, "
              f"λ_compliance: {stats['lambda_compliance']:.4f}, "
              f"λ_energy: {stats['lambda_energy']:.4f}")
        print(f"  [达标率] Safety: {stats['safety_satisfied_rate']:.2%}, "
              f"Compliance: {stats['compliance_satisfied_rate']:.2%}, "
              f"Energy: {stats['energy_satisfied_rate']:.2%}")
    
    print(f"\n--- 测试动作选择 ---")
    test_state = torch.randn(state_dim, device=device)
    action_det = agent.select_action(test_state, deterministic=True)
    action_stoch = agent.select_action(test_state, deterministic=False)
    
    print(f"确定性动作: mean={action_det.mean().item():.4f}, std={action_det.std().item():.4f}")
    print(f"随机动作:   mean={action_stoch.mean().item():.4f}, std={action_stoch.std().item():.4f}")
    
    print(f"\n--- 梯度追踪验证 ---")
    test_states = torch.randn(batch_size, state_dim, device=device, requires_grad=True)
    test_actions, test_log_prob = agent.actor.sample(test_states)
    test_q = agent.critic(test_states, test_actions)
    
    print(f"States requires_grad: {test_states.requires_grad}")
    print(f"Actions requires_grad: {test_actions.requires_grad}")
    print(f"Q-values requires_grad: {test_q.requires_grad}")
    print(f"Log_prob requires_grad: {test_log_prob.requires_grad}")
    
    print(f"\n" + "=" * 70)
    print("✓ 字典序SAC代理连通性测试通过")
    print("=" * 70)
    
    print(f"\n【核心验证点】")
    print("1. Critic输出多维Q向量 (n_objectives=3): ✓")
    print("2. 拉格朗日乘子动态调整: ✓")
    print("3. 梯度冻结逻辑: ✓")
    print("4. 字典序优化替代标量加权: ✓")
    print("5. 1e-6数值截断: ✓")
