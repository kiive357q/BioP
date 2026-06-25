# -*- coding: utf-8 -*-
# hierarchical_actor_critic.py
# 层级 Actor-Critic，分层策略架构
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Dict
from torch import Tensor


class HierarchicalActor(nn.Module):
    """
    层级演员网络
    
    【架构】
    Meta-Controller → Primitive Controller
    1. Meta-Controller 选择子目标
    2. Primitive Controller 实现子目标
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        
        self.meta_controller = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Tanh()
        )
        
        self.primitive_controller = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def select_goal(self, state: Tensor) -> Tensor:
        """Meta-Controller: 选择子目标"""
        return self.meta_controller(state)
    
    def select_action(
        self,
        state: Tensor,
        goal: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Primitive Controller: 基于目标选择动作"""
        state_goal = torch.cat([state, goal], dim=-1)
        action = self.primitive_controller(state_goal)
        return action, None


class HierarchicalCritic(nn.Module):
    """层级评论家网络"""
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        
        self.goal_critic = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.primitive_critic = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def evaluate_goal(self, state: Tensor, goal: Tensor) -> Tensor:
        """评估子目标的价值"""
        return self.goal_critic(torch.cat([state, goal], dim=-1))
    
    def evaluate_action(
        self,
        state: Tensor,
        goal: Tensor,
        action: Tensor
    ) -> Tensor:
        """评估动作的价值"""
        return self.primitive_critic(
            torch.cat([state, goal, action], dim=-1)
        )


class HierarchicalActorCritic(nn.Module):
    """
    层级 Actor-Critic 算法
    
    【训练流程】
    1. Meta-Controller 接收稀疏奖励，更新频率低
    2. Primitive Controller 接收密集奖励，更新频率高
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        
        self.actor = HierarchicalActor(state_dim, goal_dim, action_dim, hidden_dim)
        self.critic = HierarchicalCritic(state_dim, goal_dim, action_dim, hidden_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
    def update_primitive(
        self,
        state: Tensor,
        goal: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor
    ) -> Dict[str, Tensor]:
        """更新 Primitive Controller"""
        with torch.no_grad():
            next_action, _ = self.actor.select_action(next_state, goal)
            next_value = self.critic.evaluate_action(next_state, goal, next_action)
            target_q = reward + 0.99 * (1 - done) * next_value
        
        current_q = self.critic.evaluate_action(state, goal, action)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return {"critic_loss": critic_loss}
    
    def update_meta(
        self,
        state: Tensor,
        goal: Tensor,
        goal_reward: Tensor
    ) -> Dict[str, Tensor]:
        """更新 Meta-Controller"""
        goal_value = self.critic.evaluate_goal(state, goal)
        
        actor_loss = -goal_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {"actor_loss": actor_loss}
