# -*- coding: utf-8 -*-
# sindy_library.py
# SINDy 稀疏辨识库，构建动态模式特征矩阵
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import itertools


class SINDyLibrary(nn.Module):
    """
    SINDy (Sparse Identification of Nonlinear Dynamics) 特征库
    
    【物理意义】
    SINDy 通过构建丰富的候选函数库 Θ(x)，
    稀疏求解系数 Ξ 来识别潜在的动态方程:
    
    dx/dt = Θ(x) · Ξ
    
    其中特征库包含:
    - 多项式基: {1, x, x², x³, ...}
    - 三角函数: {sin(x), cos(x), sin(2x), cos(2x), ...}
    - 交叉项: {x₁x₂, x₁x₃, ...}
    
    【工业应用】
    用于识别污水处理过程中关键的动力学模式，
    如磷的吸附/解吸、聚羟基脂肪酸酯 (PHAs) 的生成等
    """
    
    def __init__(
        self,
        n_state_variables: int,
        poly_order: int = 3,
        include_sin: bool = True,
        include_cos: bool = True,
        include_interactions: bool = True,
        interaction_order: int = 2
    ) -> None:
        """
        初始化 SINDy 特征库
        
        参数:
            n_state_variables: 状态变量数量
            poly_order: 多项式最高阶数
            include_sin: 是否包含正弦项
            include_cos: 是否包含余弦项
            include_interactions: 是否包含交叉项
            interaction_order: 交叉项最高阶数
        """
        super().__init__()
        
        self.n_state = n_state_variables
        self.poly_order = poly_order
        self.include_sin = include_sin
        self.include_cos = include_cos
        self.include_interactions = include_interactions
        self.interaction_order = interaction_order
        
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)
        
        self.register_buffer(
            "coefficient_mask",
            torch.ones(self.n_features, n_state_variables)
        )
    
    def _build_feature_names(self) -> List[str]:
        """构建特征名称列表"""
        names = []
        
        names.append("1")
        
        for order in range(1, self.poly_order + 1):
            for combo in itertools.combinations_with_replacement(
                range(self.n_state), order
            ):
                if len(combo) == 1:
                    names.append(f"x{combo[0]}^{order}")
                else:
                    parts = [f"x{c}" for c in combo]
                    names.append("_".join(parts))
        
        if self.include_sin:
            for i in range(self.n_state):
                names.append(f"sin(x{i})")
        
        if self.include_cos:
            for i in range(self.n_state):
                names.append(f"cos(x{i})")
        
        if self.include_interactions:
            for order in range(2, self.interaction_order + 1):
                for combo in itertools.combinations(
                    range(self.n_state), order
                ):
                    parts = [f"x{c}" for c in combo]
                    names.append("_".join(parts))
        
        return names
    
    def compute_polynomial_features(
        self,
        x: torch.Tensor,
        order: int
    ) -> List[torch.Tensor]:
        """
        计算多项式特征
        
        参数:
            x: 状态 [batch, n_state]
            order: 多项式阶数
            
        返回:
            多项式特征列表
        """
        features = []
        
        if order >= 1:
            features.append(x)
        
        if order >= 2:
            for i in range(self.n_state):
                features.append(x[:, i:i+1] ** 2)
        
        if order >= 3:
            for i in range(self.n_state):
                features.append(x[:, i:i+1] ** 3)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        构建 SINDy 特征矩阵
        
        参数:
            x: 状态向量 [batch, n_state] 或 [batch, seq_len, n_state]
            
        返回:
            特征矩阵 Θ(x) [batch, n_features] 或 [batch, seq_len, n_features]
        """
        is_3d = x.dim() == 3
        squeeze_output = False
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        batch_size, seq_len, n_state = x.shape
        
        feature_list = []
        
        feature_list.append(torch.ones(batch_size, seq_len, 1, device=x.device))
        
        for order in range(1, self.poly_order + 1):
            if order == 1:
                poly_features = x
            else:
                poly_features = self._compute_polynomial_terms(x, order)
            feature_list.append(poly_features)
        
        if self.include_sin:
            sin_features = torch.sin(x)
            feature_list.append(sin_features)
        
        if self.include_cos:
            cos_features = torch.cos(x)
            feature_list.append(cos_features)
        
        if self.include_interactions:
            interaction_features = self._compute_interaction_terms(x)
            feature_list.append(interaction_features)
        
        theta = torch.cat(feature_list, dim=-1)
        
        if squeeze_output:
            theta = theta.squeeze(1)
        
        return theta
    
    def _compute_polynomial_terms(
        self,
        x: torch.Tensor,
        order: int
    ) -> torch.Tensor:
        """计算指定阶数的多项式项"""
        batch_size, seq_len, n_state = x.shape
        
        if order == 2:
            terms = []
            for i in range(n_state):
                terms.append(x[:, :, i:i+1] ** 2)
            return torch.cat(terms, dim=-1)
        
        elif order == 3:
            terms = []
            for i in range(n_state):
                terms.append(x[:, :, i:i+1] ** 3)
            return torch.cat(terms, dim=-1)
        
        return torch.zeros(batch_size, seq_len, 0, device=x.device)
    
    def _compute_interaction_terms(self, x: torch.Tensor) -> torch.Tensor:
        """计算交叉项"""
        batch_size, seq_len, n_state = x.shape
        
        interaction_terms = []
        
        for i in range(n_state):
            for j in range(i + 1, n_state):
                interaction_terms.append(x[:, :, i:i+1] * x[:, :, j:j+1])
        
        if interaction_terms:
            return torch.cat(interaction_terms, dim=-1)
        else:
            return torch.zeros(batch_size, seq_len, 0, device=x.device)


class SINDyOptimizer(nn.Module):
    """
    SINDy 稀疏优化器
    
    【数学公式】
    min ||Ξ||₁  s.t. ||dx/dt - Θ(x)·Ξ||₂ < ε
    
    使用阈交 (Thresholded) OLS 或 SR3 算法
    """
    
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        threshold: float = 0.1,
        optimizer_type: str = "thresholded_ols"
    ) -> None:
        """
        初始化 SINDy 优化器
        
        参数:
            n_features: 特征数量
            n_targets: 目标数量
            threshold: 稀疏化阈值
            optimizer_type: 优化类型
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_targets = n_targets
        self.threshold = threshold
        self.optimizer_type = optimizer_type
        
        self.coefficients = nn.Parameter(
            torch.zeros(n_features, n_targets)
        )
    
    def forward(
        self,
        theta: torch.Tensor,
        dx_dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 计算系数和解码动态
        
        参数:
            theta: 特征矩阵 [batch, n_features] 或 [batch, seq, n_features]
            dx_dt: 状态导数 [batch, n_targets] 或 [batch, seq, n_targets]
            
        返回:
            predicted_dynamics: 预测的动态
            coefficients: 当前系数
        """
        if theta.dim() == 2:
            predicted_dynamics = theta @ self.coefficients
        else:
            batch_size, seq_len, n_features = theta.shape
            theta_flat = theta.reshape(-1, n_features)
            pred_flat = theta_flat @ self.coefficients
            predicted_dynamics = pred_flat.reshape(batch_size, seq_len, -1)
        
        return predicted_dynamics, self.coefficients
    
    def apply_threshold(self) -> None:
        """
        应用阈值进行稀疏化
        
        【核心算法】阈交最小二乘法:
        Ξ_ij = 0 if |Ξ_ij| < threshold
        """
        with torch.no_grad():
            self.coefficients.copy_(
                torch.where(
                    torch.abs(self.coefficients) < self.threshold,
                    torch.zeros_like(self.coefficients),
                    self.coefficients
                )
            )
    
    def sparsify(self, training_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        稀疏化训练
        
        参数:
            training_data: (theta, dx_dt) 元组
        """
        theta, dx_dt = training_data
        
        for iteration in range(10):
            predicted, _ = self.forward(theta, dx_dt)
            
            loss = torch.nn.functional.mse_loss(predicted, dx_dt)
            
            loss.backward()
            
            with torch.no_grad():
                self.coefficients.grad *= (torch.abs(self.coefficients) >= self.threshold).float()
            
            self.apply_threshold()


def sindy_fit(
    theta: torch.Tensor,
    dx_dt: torch.Tensor,
    threshold: float = 0.01,
    iterations: int = 10
) -> torch.Tensor:
    """
    SINDy 拟合函数
    
    【便捷接口】
    
    参数:
        theta: 特征矩阵
        dx_dt: 状态导数
        threshold: 稀疏化阈值
        iterations: 迭代次数
        
    返回:
        稀疏系数矩阵 Ξ
    """
    n_features = theta.shape[-1]
    n_targets = dx_dt.shape[-1]
    
    optimizer = SINDyOptimizer(n_features, n_targets, threshold)
    
    for _ in range(iterations):
        predicted, _ = optimizer.forward(theta, dx_dt)
        loss = torch.nn.functional.mse_loss(predicted, dx_dt)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.apply_threshold()
    
    return optimizer.coefficients.detach()
