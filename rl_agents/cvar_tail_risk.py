# -*- coding: utf-8 -*-
# cvar_tail_risk.py
# CVaR 尾端风险度量，极端工况鲁棒性
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
import numpy as np


class CVaRTailRisk:
    """
    CVaR (Conditional Value at Risk) 尾端风险度量
    
    【物理意义】
    CVaR 衡量最坏情况下的期望损失:
    
    CVaR_α = E[Loss | Loss ≥ VaR_α]
    
    【工业应用】
    - 极端天气工况的风险评估
    - 传感器故障场景的鲁棒性
    - 污泥膨胀异常的损失评估
    """
    
    def __init__(self, alpha: float = 0.05, n_samples: int = 100) -> None:
        """
        初始化 CVaR 计算器
        
        参数:
            alpha: 风险水平 (默认 5% 表示关注最差的 5%)
            n_samples: 蒙特卡洛采样数
        """
        self.alpha = alpha
        self.n_samples = n_samples
    
    def compute_cvar(
        self,
        losses: Tensor,
        weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        计算 CVaR
        
        参数:
            losses: 损失样本 [n_samples]
            weights: 样本权重
            
        返回:
            cvar: CVaR 值
            var: VaR 值
        """
        sorted_losses, indices = torch.sort(losses)
        
        n_threshold = max(1, int(self.alpha * len(losses)))
        
        var = sorted_losses[-n_threshold]
        
        tail_losses = losses[losses >= var]
        cvar = tail_losses.mean()
        
        return cvar, var
    
    def compute_weighted_cvar(
        self,
        losses: Tensor,
        weights: Tensor
    ) -> Tensor:
        """
        计算加权 CVaR
        
        参数:
            losses: 损失
            weights: 样本权重
            
        返回:
            加权 CVaR
        """
        sorted_indices = torch.argsort(losses)
        
        n_threshold = max(1, int(self.alpha * len(losses)))
        threshold_idx = len(losses) - n_threshold
        
        tail_indices = sorted_indices[threshold_idx:]
        
        tail_weights = weights[tail_indices]
        tail_losses = losses[tail_indices]
        
        weighted_cvar = (tail_weights * tail_losses).sum() / (tail_weights.sum() + 1e-8)
        
        return weighted_cvar


class CVaRLoss(nn.Module):
    """
    CVaR 风险厌恶损失
    
    在标准损失函数中添加 CVaR 正则化项:
    L_total = L_task + λ · CVaR(L_task)
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        lambda_cvar: float = 0.1
    ) -> None:
        super().__init__()
        
        self.alpha = alpha
        self.lambda_cvar = lambda_cvar
        self.cvar_calculator = CVaRTailRisk(alpha)
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        return_cvar: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        计算带 CVaR 正则化的损失
        
        参数:
            predictions: 预测值
            targets: 目标值
            return_cvar: 是否返回 CVaR 值
            
        返回:
            total_loss: 总损失
            cvar: CVaR (可选)
        """
        mse_loss = (predictions - targets) ** 2
        
        batch_cvar, _ = self.cvar_calculator.compute_cvar(
            mse_loss.mean(dim=-1) if mse_loss.dim() > 1 else mse_loss
        )
        
        total_loss = mse_loss.mean() + self.lambda_cvar * batch_cvar
        
        if return_cvar:
            return total_loss, batch_cvar
        return total_loss, None


def cvar_sac_loss(
    q_values: Tensor,
    alpha: float = 0.05,
    beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    CVaR-SAC 损失函数
    
    【论文】CVaR-SAC: A Robust Reinforcement Learning Algorithm
    
    参数:
        q_values: Q 值样本 [batch, n_samples]
        alpha: CVaR 风险水平
        beta: 风险厌恶系数
        
    返回:
        cvar_loss: CVaR 损失
        cvar_value: CVaR 值
    """
    batch_size, n_samples = q_values.shape
    
    sorted_q, _ = torch.sort(q_values, dim=1, descending=True)
    
    n_quantile = max(1, int(alpha * n_samples))
    
    cvar_q = sorted_q[:, :n_quantile].mean(dim=1)
    
    var_q = sorted_q[:, n_quantile:].mean(dim=1)
    
    cvar_loss = -(1 - beta) * var_q - beta * cvar_q
    
    return cvar_loss.mean(), cvar_q.mean()
