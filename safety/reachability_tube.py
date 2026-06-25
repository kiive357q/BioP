# -*- coding: utf-8 -*-
# reachability_tube.py
# 可达性管计算，约束鲁棒优化
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor
import numpy as np


class ReachabilityTube(nn.Module):
    """
    可达性管计算
    
    【物理意义】
    可达性管描述了系统从初始集出发，在所有可能扰动下
    能够到达的状态集合:
    
    Ξ(t) = {x(t) | x(0) ∈ X_0, d(t) ∈ D}
    
    【工业应用】
    - 鲁棒控制器设计
    - 不确定性分析
    - 安全验证
    """
    
    def __init__(
        self,
        state_dim: int,
        disturbance_dim: int,
        n_samples: int = 1000
    ) -> None:
        """
        初始化可达性管计算器
        
        参数:
            state_dim: 状态维度
            disturbance_dim: 扰动维度
            n_samples: 蒙特卡洛采样数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.disturbance_dim = disturbance_dim
        self.n_samples = n_samples
    
    def compute_tube_monte_carlo(
        self,
        initial_set: Tensor,
        dynamics_fn: callable,
        disturbance_distribution: dict,
        time_horizon: float,
        dt: float
    ) -> Tuple[Tensor, Tensor]:
        """
        蒙特卡洛法计算可达管
        
        参数:
            initial_set: 初始集合采样
            dynamics_fn: 动力学函数
            disturbance_distribution: 扰动分布参数
            time_horizon: 时间范围
            dt: 时间步长
            
        返回:
            tube_mean: 期望轨迹
            tube_bounds: 不确定性边界 (3σ)
        """
        n_steps = int(time_horizon / dt)
        
        trajectories = []
        
        for _ in range(self.n_samples):
            x = initial_set[torch.randint(len(initial_set), (1,))]
            trajectory = [x]
            
            for t in range(n_steps):
                disturbance = self._sample_disturbance(disturbance_distribution)
                u = torch.zeros(self.state_dim)
                
                dx = dynamics_fn(x, u, disturbance)
                x = x + dt * dx
                
                trajectory.append(x)
            
            trajectories.append(torch.cat(trajectory, dim=0))
        
        trajectories_tensor = torch.stack(trajectories, dim=0)
        
        tube_mean = trajectories_tensor.mean(dim=0)
        tube_std = trajectories_tensor.std(dim=0)
        tube_bounds = 3 * tube_std
        
        return tube_mean, tube_bounds
    
    def _sample_disturbance(
        self,
        distribution: dict
    ) -> Tensor:
        """采样扰动"""
        mean = distribution.get("mean", torch.zeros(self.disturbance_dim))
        std = distribution.get("std", torch.ones(self.disturbance_dim))
        
        return torch.randn(self.disturbance_dim) * std + mean
    
    def compute_lipschitz_tube(
        self,
        state_bounds: Tensor,
        L: float,
        time_horizon: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Lipschitz 界估计可达管
        
        【数学公式】
        ||x(t) - x̂(t)|| ≤ ||x(0) - x̂(0)|| · exp(L·t)
        
        参数:
            state_bounds: 状态边界
            L: Lipschitz 常数
            time_horizon: 时间范围
            
        返回:
            center: 中心轨迹
            radius: 不确定性半径
        """
        center = (state_bounds[:, 0] + state_bounds[:, 1]) / 2
        initial_radius = (state_bounds[:, 1] - state_bounds[:, 0]) / 2
        
        t = torch.linspace(0, time_horizon, 100)
        radius = initial_radius * torch.exp(L * t)
        
        return center, radius


def verify_safety_tube(
    reachability_tube: Tuple[Tensor, Tensor],
    safety_set: Tensor,
    state_dim: int
) -> Tuple[bool, Tensor]:
    """
    验证安全管
    
    检查可达性管是否完全在安全集内
    
    参数:
        reachability_tube: (center, bounds) 元组
        safety_set: 安全集合定义
        state_dim: 状态维度
        
    返回:
        is_safe: 是否安全
        violation_region: 违规区域
    """
    center, bounds = reachability_tube
    
    tube_lower = center - bounds
    tube_upper = center + bounds
    
    safe_lower = safety_set["lower"]
    safe_upper = safety_set["upper"]
    
    lower_violation = torch.clamp(safe_lower - tube_lower, min=0)
    upper_violation = torch.clamp(tube_upper - safe_upper, min=0)
    
    violation = lower_violation + upper_violation
    is_safe = (violation == 0).all()
    
    return bool(is_safe.item()), violation
