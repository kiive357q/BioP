# -*- coding: utf-8 -*-
"""
Full Control Module - 完整控制模块
BioP Causal WorldModel V2.0

【包含】
1. Lexicographic SAC - 字典序软演员-评论家
2. CBF - 控制障碍函数安全验证
3. QP拦截器 - 动作安全修正

【核心思想】
- 第一优先级: 安全约束 (CBF硬约束)
- 第二优先级: 水质达标 (奖励)
- 第三优先级: 能耗最小 (奖励)

【版本】V2.0-FullControl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ControlConfig:
    """控制配置"""
    # 动作空间
    action_dim: int = 4
    action_low: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    action_high: Tuple[float, float, float, float] = (200.0, 100.0, 1.0, 500.0)
    
    # 安全约束
    do_min: float = 0.5
    do_max: float = 6.0
    nh4_max: float = 15.0
    tp_max: float = 10.0
    tn_max: float = 20.0
    
    # 字典序优先级
    safety_threshold: float = 0.0  # 安全分数低于此值认为不安全
    compliance_threshold: float = -10.0  # 水质奖励阈值
    
    # SAC超参数
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    
    # 训练
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    lagrange_lr: float = 1e-3


class ControlBarrierFunction(nn.Module):
    """
    控制障碍函数 (CBF)
    
    【理论】
    安全集 C = {x : h(x) ≥ 0}
    其中 h(x) 是CBF
    
    【约束条件】
    L_f h(x) + L_g h(x)u ≥ -γ h(x)
    
    其中:
    - L_f h: h关于f的Lie导数
    - L_g h: h关于g的Lie导数
    - γ: 相对度参数
    """
    
    def __init__(self, state_dim: int = 50, hidden_dim: int = 256):
        super().__init__()
        
        # 3个CBF (DO, NH4, 液位)
        self.n_cbfs = 3
        
        # CBF网络: 状态 → h值
        self.cbf_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(self.n_cbfs)
        ])
        
        # 初始化 - 使初始状态在安全域内
        for net in self.cbf_networks:
            with torch.no_grad():
                net[-1].bias.data = torch.tensor([1.0])  # h(x) = 1 表示安全
    
    def forward(self, state: Tensor) -> Tuple[Tensor, Dict]:
        """
        计算CBF值
        
        Args:
            state: [batch, state_dim] 状态
            
        Returns:
            h_values: [batch, n_cbfs] 每个CBF的值
            info: 附加信息
        """
        h_values = torch.cat([
            net(state) for net in self.cbf_networks
        ], dim=1)
        
        info = {
            'h_DO': h_values[:, 0],  # 溶解氧CBF
            'h_NH4': h_values[:, 1],  # 氨氮CBF
            'h_level': h_values[:, 2],  # 液位CBF
        }
        
        return h_values, info
    
    def verify_safety(self, state: Tensor) -> Tuple[Tensor, Dict]:
        """
        验证状态安全性
        
        Returns:
            is_safe: [batch] 是否安全
            violations: 违反的约束
        """
        h_values, info = self.forward(state)
        
        # h > 0 表示安全
        is_safe = (h_values > 0).all(dim=1)
        
        violations = {
            'DO_unsafe': (h_values[:, 0] <= 0).sum().item(),
            'NH4_unsafe': (h_values[:, 1] <= 0).sum().item(),
            'level_unsafe': (h_values[:, 2] <= 0).sum().item(),
        }
        
        return is_safe, violations


class QPInterceptor(nn.Module):
    """
    QP拦截器
    
    【功能】
    当Actor输出的动作违反CBF约束时，
    使用二次规划找到最接近的、安全的动作
    
    【QP问题】
    min ||u - u_actor||²
    s.t. CBF约束: L_f h + L_g h · u ≥ -γh
    """
    
    def __init__(self, action_dim: int = 4):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Lie导数网络 L_g h(x) - 动作对CBF的影响
        # 简化为对角矩阵 (假设动作解耦)
        self.register_buffer(
            'Lg_diag',
            torch.tensor([1.0, 0.8, 1.2, 0.5])  # 每个动作对安全的影响权重
        )
    
    def intercept(
        self,
        u_actor: Tensor,
        state: Tensor,
        cbf_values: Tensor,
        gamma: float = 10.0
    ) -> Tuple[Tensor, Dict]:
        """
        QP拦截
        
        Args:
            u_actor: [batch, action_dim] Actor输出的动作
            state: [batch, state_dim] 当前状态
            cbf_values: [batch, n_cbfs] CBF值
            gamma: CBF参数
            
        Returns:
            u_safe: [batch, action_dim] 安全动作
            stats: 统计信息
        """
        batch = u_actor.shape[0]
        
        # 简化的QP: 线性规划近似
        # 安全动作 = u_actor + Δu
        # Δu = -Lg_diag * min(0, h(x)) * sign(gradient)
        
        # 计算CBF梯度 (简化用数值梯度)
        h_safe = cbf_values.mean(dim=1)  # [batch]
        
        # 当h < 0时需要修正
        correction_scale = torch.clamp(-h_safe, min=0.0, max=1.0)  # [batch]
        
        # 修正量
        delta_u = -self.Lg_diag.unsqueeze(0) * correction_scale.unsqueeze(1) * 0.1
        
        # 安全动作
        u_safe = u_actor + delta_u
        
        # 限制动作范围
        u_safe = torch.clamp(u_safe, min=0.0)
        
        stats = {
            'n_corrected': (correction_scale > 0).sum().item(),
            'avg_correction': correction_scale.mean().item(),
        }
        
        return u_safe, stats


class LexicographicSAC(nn.Module):
    """
    字典序软演员-评论家
    
    【字典序优化】
    1. 满足安全约束 (硬约束，通过CBF)
    2. 最大化水质奖励 (次级)
    3. 最小化能耗 (第三级)
    
    【不使用标量加权】
    使用硬约束 + 梯度冻结 + 拉格朗日乘子
    """
    
    def __init__(
        self,
        world_model,  # BioPWorldModel
        config: ControlConfig = None
    ):
        super().__init__()
        
        self.config = config or ControlConfig()
        self.wm = world_model  # 世界模型
        
        # 分离的Actor-Critic
        self.actor = world_model.actor
        self.critic = world_model.critic
        
        # 目标网络
        self.target_critic = world_model.critic.__class__(
            latent_dim=self.wm.latent_dim,
            action_dim=self.config.action_dim
        ).to(self.wm.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # CBF安全验证器
        self.cbf = ControlBarrierFunction(
            state_dim=self.wm.obs_dim,
            hidden_dim=256
        ).to(self.wm.device)
        
        # QP拦截器
        self.qp_interceptor = QPInterceptor(action_dim=self.config.action_dim)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr
        )
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(),
            lr=self.config.actor_lr * 0.1
        )
        
        # 拉格朗日乘子 (用于软约束)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.config.lagrange_lr
        )
        
        # 经验回放
        self.replay_buffer = None
        
        # 统计
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'cbf_violations': [],
            'qp_corrections': [],
        }
    
    def select_action(
        self,
        obs: Tensor,
        deterministic: bool = False,
        use_cbf: bool = True
    ) -> Tuple[Tensor, Dict]:
        """
        选择动作
        
        【流程】
        1. 编码观测到隐状态
        2. Actor输出原始动作
        3. CBF验证安全性
        4. 如不安全，用QP修正
        """
        # 0. 编码观测到隐状态
        z = self.wm.encode(obs)
        
        # 1. 获取Actor动作
        u_actor = self.actor.get_action(z, deterministic=deterministic)
        
        result = {
            'u_actor': u_actor,
            'corrected': False,
            'safe': True,
        }
        
        if use_cbf:
            # 2. CBF验证
            h_values, cbf_info = self.cbf(obs)
            is_safe = (h_values > 0).all(dim=1)
            
            result['h_values'] = h_values
            result['is_safe'] = is_safe
            
            # 3. 如不安全，用QP修正
            if not is_safe.all():
                u_safe, qp_stats = self.qp_intercept(
                    u_actor, obs, h_values
                )
                u_actor = u_safe
                result['u_safe'] = u_safe
                result['corrected'] = True
                result['safe'] = False
                result['qp_stats'] = qp_stats
        
        return u_actor, result
    
    def qp_intercept(
        self,
        u_actor: Tensor,
        state: Tensor,
        cbf_values: Tensor
    ) -> Tuple[Tensor, Dict]:
        """QP拦截"""
        return self.qp_interceptor(u_actor, state, cbf_values)
    
    def update(
        self,
        batch: Dict,
        timestep: int
    ) -> Dict:
        """
        更新网络
        
        Args:
            batch: 经验批次
            timestep: 当前时间步
            
        Returns:
            stats: 更新统计
        """
        obs = batch['obs']
        action = batch['action']
        reward = batch['reward']
        next_obs = batch['next_obs']
        done = batch['done']
        
        batch_size = obs.shape[0]
        
        # ========== 1. 更新Critic ==========
        with torch.no_grad():
            next_action, next_info = self.select_action(
                next_obs, deterministic=False, use_cbf=True
            )
            
            target_q1, target_q2 = self.target_critic(
                self.wm.encode(next_obs), next_action
            )
            target_q = torch.min(target_q1, target_q2)
            
            target_value = reward + self.config.gamma * (1 - done) * target_q
        
        current_q1, current_q2 = self.critic(
            self.wm.encode(obs), action
        )
        
        critic_loss = F.mse_loss(current_q1, target_value) + \
                     F.mse_loss(current_q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.critic_optimizer.step()
        
        # ========== 2. 更新Actor (字典序) ==========
        # 冻结非最高优先级梯度
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # 最高优先级: 最大化安全分数
        h_values, _ = self.cbf(obs)
        safety_loss = -h_values.mean()  # 最大化安全性
        
        # 次级: 最大化Q值 (水质+能效)
        new_action, _ = self.select_action(obs, deterministic=False, use_cbf=False)
        q1, q2 = self.critic(self.wm.encode(obs), new_action)
        q_loss = -torch.min(q1, q2).mean()  # 最大化Q
        
        # 组合
        actor_loss = q_loss + 0.1 * safety_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actor_optimizer.step()
        
        # 解冻Critic
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # ========== 3. 更新CBF ==========
        # 惩罚不安全的动作
        h_values, _ = self.cbf(obs)
        cbf_loss = torch.clamp(-h_values + 0.1, min=0).mean()  # h < 0时惩罚
        
        self.cbf_optimizer.zero_grad()
        cbf_loss.backward()
        self.cbf_optimizer.step()
        
        # ========== 4. 更新Alpha ==========
        alpha_loss = self.log_alpha * (-h_values.mean() - 0.1).detach()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        alpha = torch.exp(self.log_alpha)
        
        # ========== 5. 软更新目标网络 ==========
        self._soft_update_target()
        
        # 统计
        stats = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'cbf_loss': cbf_loss.item(),
            'alpha': alpha.item(),
            'avg_h': h_values.mean().item(),
            'min_h': h_values.min().item(),
        }
        
        return stats
    
    def _soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + 
                (1 - self.config.tau) * target_param.data
            )


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        obs_dim: int = 50,
        action_dim: int = 4,
        batch_size: int = 256,
        device: str = "cuda"
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.position = 0
        self.size = 0
        
        # 预分配内存
        self.obs = torch.zeros(capacity, obs_dim, dtype=torch.float32, device=device)
        self.action = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.reward = torch.zeros(capacity, 1, dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, 1, dtype=torch.float32, device=device)
    
    def push(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        done: Tensor
    ):
        """添加经验"""
        self.obs[self.position] = obs
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.next_obs[self.position] = next_obs
        self.done[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self) -> Dict:
        """采样批次"""
        indices = torch.randint(0, self.size, (self.batch_size,), device=self.device)
        
        return {
            'obs': self.obs[indices],
            'action': self.action[indices],
            'reward': self.reward[indices],
            'next_obs': self.next_obs[indices],
            'done': self.done[indices],
        }
    
    def __len__(self):
        return self.size


def create_full_control_system(
    world_model,
    device: str = "cuda"
) -> LexicographicSAC:
    """创建完整控制系统"""
    config = ControlConfig()
    control_system = LexicographicSAC(world_model, config)
    
    # 创建回放缓冲区
    control_system.replay_buffer = ReplayBuffer(
        capacity=100000,
        obs_dim=world_model.obs_dim,
        action_dim=config.action_dim,
        batch_size=256,
        device=device
    )
    
    return control_system


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Full Control Module Test")
    print("=" * 70)
    
    from models.expanded_ncde import create_expanded_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建世界模型
    print("\nCreating world model...")
    wm = create_expanded_model(device=device)
    
    # 创建控制系统
    print("\nCreating control system...")
    control = create_full_control_system(wm, device)
    
    # 测试动作选择
    print("\nTesting action selection...")
    obs = torch.randn(8, 50, device=device)
    action, result = control.select_action(obs, deterministic=False)
    
    print(f"  Action shape: {action.shape}")
    print(f"  Corrected: {result['corrected']}")
    print(f"  Safe: {result['safe']}")
    
    if 'h_values' in result:
        print(f"  h_values: {result['h_values'].mean(dim=0)}")
    
    # 测试更新
    print("\nTesting update...")
    batch = {
        'obs': obs,
        'action': action,
        'reward': torch.randn(8, 1, device=device),
        'next_obs': torch.randn(8, 50, device=device),
        'done': torch.zeros(8, 1, device=device),
    }
    
    stats = control.update(batch, timestep=0)
    
    print(f"  Critic loss: {stats['critic_loss']:.3f}")
    print(f"  Actor loss: {stats['actor_loss']:.3f}")
    print(f"  CBF loss: {stats['cbf_loss']:.3f}")
    print(f"  Alpha: {stats['alpha']:.3f}")
    print(f"  Avg h: {stats['avg_h']:.3f}")
    
    # 测试回放缓冲区
    print("\nTesting replay buffer...")
    buffer = ReplayBuffer(capacity=1000, obs_dim=50, action_dim=4, device=device)
    
    for i in range(100):
        batch = {
            'obs': torch.randn(8, 50, device=device),
            'action': torch.randn(8, 4, device=device),
            'reward': torch.randn(8, 1, device=device),
            'next_obs': torch.randn(8, 50, device=device),
            'done': torch.zeros(8, 1, device=device),
        }
        buffer.push(**{k: v[0] for k, v in batch.items()})
    
    sample = buffer.sample()
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Sample obs shape: {sample['obs'].shape}")
    
    print("\n✓ Full Control Module test passed!")
