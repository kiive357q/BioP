# -*- coding: utf-8 -*-
# priority_replay.py
# 优先级经验回放，异常工况样本加权
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
import random


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    
    【核心思想】
    优先回放 TD 误差大的样本，加速学习:
    
    P(i) ∝ |TD_error(i)|^α + ε
    
    其中 α 控制优先级程度，ε 防止除零
    
    【工业增强】
    - 异常工况样本自动加权
    - 物质守恒违规样本优先回放
    - 安全临界样本高优先级
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        epsilon: float = 1e-4,
        device: str = "cuda"
    ) -> None:
        """
        初始化优先级缓冲区
        
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta 增量
            epsilon: 优先级最小值
            device: 计算设备
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.device = device
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        self.safety_priority_boost = 3.0
        self.anomaly_detection_threshold = 2.0
    
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
        td_error: Optional[float] = None,
        is_safety_critical: bool = False
    ) -> None:
        """
        添加经验
        
        参数:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 完成标志
            td_error: TD 误差
            is_safety_critical: 是否为安全关键样本
        """
        transition = (state, action, reward, next_state, done)
        
        if td_error is None:
            priority = self.epsilon
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        
        if is_safety_critical:
            priority *= self.safety_priority_boost
        
        if self._is_anomaly(state, reward):
            priority *= 2.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def _is_anomaly(
        self,
        state: torch.Tensor,
        reward: torch.Tensor
    ) -> bool:
        """检测异常工况"""
        if torch.isnan(state).any() or torch.isinf(state).any():
            return True
        
        if torch.abs(reward) > self.anomaly_detection_threshold:
            return True
        
        return False
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        采样批次
        
        【重要性采样权重】
        w(i) = (P(i) · N)^(-β) / max_w
        
        参数:
            batch_size: 批次大小
            
        返回:
            批次数据
        """
        if len(self.buffer) < batch_size:
            return None
        
        priorities_np = np.array(self.priorities)
        probabilities = priorities_np / priorities_np.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )
        
        samples = [self.buffer[idx] for idx in indices]
        
        states = torch.stack([s[0] for s in samples]).to(self.device)
        actions = torch.stack([s[1] for s in samples]).to(self.device)
        rewards = torch.stack([s[2] for s in samples]).to(self.device)
        next_states = torch.stack([s[3] for s in samples]).to(self.device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32).to(self.device)
        
        weights = []
        for idx in indices:
            p = priorities_np[idx]
            w = (p * len(self.buffer)) ** (-self.beta)
            weights.append(w)
        
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(
        self,
        indices: List[int],
        td_errors: List[float]
    ) -> None:
        """
        更新优先级
        
        参数:
            indices: 样本索引
            td_errors: 新的 TD 误差
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 100) -> bool:
        """检查缓冲区是否准备就绪"""
        return len(self.buffer) >= min_size
