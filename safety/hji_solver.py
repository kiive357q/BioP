# -*- coding: utf-8 -*-
# hji_solver.py
# Hamilton-Jacobi-Isaacs 求解器，计算安全可达集
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from torch import Tensor
import numpy as np


class HJISolver(nn.Module):
    """
    Hamilton-Jacobi-Isaacs (HJI) 求解器
    
    【物理意义】
    HJI 方程用于计算安全可达集 (Backward Reachability):
    
    ∂V/∂t + H(x, ∇V) = 0
    
    其中 Hamiltonian H = min_u max_d L(x, u, d) · ∇V
    
    【工业应用】
    - 计算避免危险区域的最小安全控制器
    - 验证 CBF 约束的安全性
    - 处理对抗性扰动 (如传感器噪声)
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        disturbance_dim: int = 0,
        grid_resolution: int = 101,
        state_bounds: Optional[Tuple[Tensor, Tensor]] = None
    ) -> None:
        """
        初始化 HJI 求解器
        
        参数:
            state_dim: 状态维度
            control_dim: 控制维度
            disturbance_dim: 扰动维度
            grid_resolution: 网格分辨率
            state_bounds: 状态边界 (lower, upper)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        
        self.grid_resolution = grid_resolution
        
        if state_bounds is None:
            lower = torch.zeros(state_dim)
            upper = torch.ones(state_dim)
            state_bounds = (lower, upper)
        
        self.register_buffer("state_lower", state_bounds[0])
        self.register_buffer("state_upper", state_bounds[1])
    
    def compute_hamiltonian(
        self,
        x: Tensor,
        grad_V: Tensor,
        dynamics_fn: Callable,
        control_bounds: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        计算 Hamiltonian
        
        H = min_u max_d L(x, u, d) · grad_V
        
        参数:
            x: 当前状态
            grad_V: 值函数梯度
            dynamics_fn: 系统动力学函数
            control_bounds: 控制边界
            
        返回:
            Hamiltonian 值
        """
        u_min, u_max = control_bounds
        
        batch_size = x.shape[0]
        device = x.device
        
        u_candidates = torch.linspace(
            u_min[0].item(), u_max[0].item(), 10,
            device=device
        ).reshape(1, -1, 1)
        
        f = dynamics_fn(x.unsqueeze(1), u_candidates)
        
        hamiltonians = (f * grad_V.unsqueeze(1)).sum(dim=-1)
        
        hamiltonian = hamiltonians.max(dim=1)[0]
        
        return hamiltonian
    
    def solve_backward_reachability(
        self,
        target_set: Tensor,
        dynamics_fn: Callable,
        control_bounds: Tuple[Tensor, Tensor],
        time_horizon: float = 10.0,
        n_time_steps: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        求解向后可达集
        
        【数学公式】
        V(x, T) = 0 if x ∈ target_set
        V(x, 0) = min_{u∈U} max_{d∈D} L(x, u, d)
        
        参数:
            target_set: 目标集指示函数
            dynamics_fn: 动力学函数
            control_bounds: 控制边界
            time_horizon: 时间范围
            n_time_steps: 时间步数
            
        返回:
            value_function: 值函数
            reachable_set: 可达集
        """
        dt = time_horizon / n_time_steps
        
        V = torch.zeros(
            self.grid_resolution ** self.state_dim,
            device=self.state_lower.device
        )
        
        for t in reversed(range(n_time_steps)):
            grad_V = torch.autograd.grad(
                V.sum(), V, create_graph=True
            )[0]
            
            H = self.compute_hamiltonian(
                self._grid_to_state(V), grad_V,
                dynamics_fn, control_bounds
            )
            
            V = V - dt * H
        
        V = torch.clamp(V, min=0)
        
        reachable_set = (V <= 0).float()
        
        return V, reachable_set
    
    def _grid_to_state(self, V: Tensor) -> Tensor:
        """将网格索引转换为状态"""
        batch_size = V.shape[0]
        state_dim = self.state_dim
        
        indices = torch.arange(
            V.shape[-1],
            device=V.device
        ).float()
        
        state_range = self.state_upper - self.state_lower
        
        coordinates = (
            self.state_lower.unsqueeze(0) + 
            (indices / (self.grid_resolution - 1)).unsqueeze(0) * state_range.unsqueeze(0)
        )
        
        return coordinates[:batch_size] if batch_size <= len(coordinates) else coordinates


def compute_safety_index(
    state: Tensor,
    unsafe_set: Tensor,
    dynamics_fn: Callable,
    barrier_fn: Optional[Callable] = None
) -> Tuple[Tensor, Tensor]:
    """
    计算安全指标 (CBF 相关)
    
    参数:
        state: 当前状态
        unsafe_set: 不安全区域指示函数
        dynamics_fn: 动力学函数
        barrier_fn: 屏障函数 (可选)
        
    返回:
        safety_index: 安全指标 (越大越安全)
        is_safe: 是否安全
    """
    if barrier_fn is not None:
        safety_index = barrier_fn(state)
    else:
        distance_to_unsafe = -unsafe_set(state)
        safety_index = torch.exp(-distance_to_unsafe)
    
    is_safe = safety_index > 0
    
    return safety_index, is_safe
