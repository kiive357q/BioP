# -*- coding: utf-8 -*-
# koopman_operator.py
# Koopman 算子学习，实现线性嵌入空间的控制
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor


class KoopmanEncoder(nn.Module):
    """
    Koopman 编码器
    
    【物理意义】
    Koopman 算子理论指出，存在一个无穷维线性算子 K
    作用于观测函数空间，使得非线性系统可以线性表示:
    
    K ∘ g(x) = g(f(x))
    
    其中 g: X → C 是观测函数，f 是系统动力学
    
    【工业应用】
    将非线性污水处理动力学映射到线性 Koopman 空间，
    实现线性模型预测控制 (LMPC)
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3
    ) -> None:
        """
        初始化 Koopman 编码器
        
        参数:
            state_dim: 原始状态维度
            latent_dim: Koopman 潜空间维度
            hidden_dim: 隐藏层维度
            n_layers: 网络层数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers - 1)],
            nn.Linear(hidden_dim, latent_dim * 2)
        )
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码: 原始空间 → Koopman 空间
        
        参数:
            x: 原始状态 [batch, state_dim]
            
        返回:
            z: 潜状态 [batch, latent_dim]
            log_var: 对数方差 (用于 VAE-style 正则化)
        """
        output = self.network(x)
        
        z = output[:, :self.latent_dim]
        log_var = output[:, self.latent_dim:]
        
        return z, log_var


class KoopmanDecoder(nn.Module):
    """
    Koopman 解码器
    
    将 Koopman 空间的线性演化结果解码回原始状态空间
    """
    
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3
    ) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        layers = []
        prev_dim = latent_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        解码: Koopman 空间 → 原始空间
        
        参数:
            z: 潜状态 [batch, latent_dim]
            
        返回:
            原始状态 [batch, state_dim]
        """
        return self.network(z)


class KoopmanOperator(nn.Module):
    """
    Koopman 算子学习器
    
    【核心功能】
    1. 学习离散 Koopman 算子 K
    2. 支持线性动力学预测
    3. 集成控制输入 (Koopman with Control)
    
    【数学公式】
    状态更新: z_{t+1} = K · z_t + L · u_t
    观测重建: x_t = D(z_t)
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        初始化 Koopman 算子
        
        参数:
            state_dim: 状态维度
            control_dim: 控制维度
            latent_dim: 潜空间维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim)
        self.decoder = KoopmanDecoder(latent_dim, state_dim, hidden_dim)
        
        self.K = nn.Parameter(
            torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        )
        
        self.L = nn.Parameter(
            torch.randn(latent_dim, control_dim) * 0.01
        )
    
    def encode(self, x: Tensor) -> Tensor:
        """编码状态"""
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z: Tensor) -> Tensor:
        """解码状态"""
        return self.decoder(z)
    
    def linear_predict(
        self,
        z: Tensor,
        u: Optional[Tensor] = None
    ) -> Tensor:
        """
        线性预测 (Koopman 空间)
        
        参数:
            z: 当前潜状态
            u: 当前控制输入
            
        返回:
            下一潜状态
        """
        z_next = z @ self.K.T
        
        if u is not None:
            z_next = z_next + u @ self.L.T
        
        return z_next
    
    def forward(
        self,
        x: Tensor,
        u: Optional[Tensor] = None,
        steps: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        参数:
            x: 初始状态 [batch, state_dim]
            u: 控制序列 [batch, steps, control_dim] (可选)
            steps: 预测步数
            
        返回:
            predictions: 状态预测序列
            latent_trajectory: 潜空间轨迹
        """
        z = self.encode(x)
        
        predictions = []
        latent_trajectory = [z]
        
        for step in range(steps):
            u_step = u[:, step, :] if u is not None else None
            z = self.linear_predict(z, u_step)
            
            x_pred = self.decode(z)
            predictions.append(x_pred)
            latent_trajectory.append(z)
        
        predictions = torch.stack(predictions, dim=1)
        latent_trajectory = torch.stack(latent_trajectory, dim=1)
        
        return predictions, latent_trajectory
    
    def compute_koopman_loss(
        self,
        x_sequence: Tensor,
        u_sequence: Optional[Tensor] = None
    ) -> Tuple[Tensor, dict]:
        """
        计算 Koopman 学习损失
        
        参数:
            x_sequence: 状态序列 [batch, seq_len, state_dim]
            u_sequence: 控制序列 [batch, seq_len-1, control_dim]
            
        返回:
            total_loss: 总损失
            loss_dict: 损失分量字典
        """
        x0 = x_sequence[:, 0, :]
        z0 = self.encode(x0)
        
        z_pred_list = [z0]
        x_pred_list = []
        
        for t in range(x_sequence.shape[1] - 1):
            z_t = z_pred_list[-1]
            u_t = u_sequence[:, t, :] if u_sequence is not None else None
            
            z_next = self.linear_predict(z_t, u_t)
            z_pred_list.append(z_next)
            
            x_pred = self.decode(z_next)
            x_pred_list.append(x_pred)
        
        z_trajectory = torch.stack(z_pred_list[1:], dim=1)
        x_predictions = torch.stack(x_pred_list, dim=1)
        
        reconstruction_loss = torch.nn.functional.mse_loss(
            x_predictions, x_sequence[:, 1:, :]
        )
        
        z_true = self.encode(x_sequence[:, 1:, :].reshape(-1, x_sequence.shape[-1]))
        z_true = z_true.reshape_as(z_trajectory)
        
        latent_loss = torch.nn.functional.mse_loss(z_trajectory, z_true)
        
        koopman_regularization = torch.mean(
            torch.abs(self.K @ self.K.T - torch.eye(self.K.shape[0], device=self.K.device))
        )
        
        total_loss = (
            1.0 * reconstruction_loss +
            0.5 * latent_loss +
            0.01 * koopman_regularization
        )
        
        loss_dict = {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "latent": latent_loss,
            "koopman_reg": koopman_regularization
        }
        
        return total_loss, loss_dict
    
    def to_onnx(self, x_sample: Tensor, u_sample: Optional[Tensor] = None) -> None:
        """
        导出为 ONNX 格式
        
        参数:
            x_sample: 示例输入
            u_sample: 示例控制输入
        """
        self.eval()
        
        torch.onnx.export(
            self.encoder,
            x_sample,
            "koopman_encoder.onnx",
            input_names=["state"],
            output_names=["latent", "log_var"]
        )
        
        K_np = self.K.detach().cpu().numpy()
        L_np = self.L.detach().cpu().numpy()
        
        torch.save({
            "K": K_np,
            "L": L_np,
            "decoder_state_dict": self.decoder.state_dict()
        }, "koopman_operator.pt")
