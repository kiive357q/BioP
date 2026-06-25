# -*- coding: utf-8 -*-
# state_observers.py
# 软测量观测器：在线推断不可测状态 (X_OHO, X_AOO)
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class SoftSensorObserver(nn.Module):
    """
    软测量观测器
    
    【物理意义】
    污水处理过程中，许多关键状态变量（如生物量浓度）
    无法在线测量，只能通过软测量方法推断。
    
    软测量利用可测量的辅助变量 (DO, pH, OUR) 推断不可测变量。
    
    数学模型:
        X_hidden = f(X_measured, u, θ)
        其中 X_hidden 为隐状态，X_measured 为可测状态
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 4,
        n_layers: int = 3
    ) -> None:
        """
        初始化软测量观测器
        
        参数:
            input_dim: 可测变量维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (隐状态维度)
            n_layers: LSTM 层数
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
    
    def forward(
        self,
        measured_sequence: torch.Tensor,
        initial_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        参数:
            measured_sequence: 可测变量序列 [batch, seq_len, input_dim]
            initial_hidden: 初始隐藏状态
            
        返回:
            inferred_states: 推断的隐状态 [batch, seq_len, output_dim]
            hidden_state: 最终隐藏状态
        """
        if initial_hidden is None:
            lstm_out, hidden_state = self.lstm(measured_sequence)
        else:
            lstm_out, hidden_state = self.lstm(
                measured_sequence, initial_hidden
            )
        
        inferred_states = self.fc(lstm_out)
        
        return inferred_states, hidden_state


class ExtendedKalmanFilter(nn.Module):
    """
    扩展卡尔曼滤波器 (EKF)
    
    【物理意义】
    用于非线性系统的状态估计，结合过程模型和测量数据。
    适用于 DO、磷浓度等可在线测量的软测量校正。
    
    数学公式:
        预测步骤:
        x̂_k|k-1 = f(x̂_k-1, u_k-1)
        P_k|k-1 = F_k P_k-1 F_k^T + Q
        
        更新步骤:
        K_k = P_k|k-1 H_k^T (H_k P_k|k-1 H_k^T + R)^-1
        x̂_k = x̂_k|k-1 + K_k (z_k - h(x̂_k|k-1))
        P_k = (I - K_k H_k) P_k|k-1
    """
    
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ) -> None:
        """
        初始化 EKF
        
        参数:
            state_dim: 状态维度
            measurement_dim: 测量维度
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        self.register_buffer(
            "Q",
            torch.eye(state_dim) * process_noise
        )
        self.register_buffer(
            "R",
            torch.eye(measurement_dim) * measurement_noise
        )
        
        self.register_buffer("P", torch.eye(state_dim))
        self.register_buffer("x_hat", torch.zeros(state_dim, 1))
    
    def predict(
        self,
        state_transition: torch.Tensor,
        control_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测步骤
        
        参数:
            state_transition: 状态转移矩阵或函数
            control_input: 控制输入
            
        返回:
            预测状态
        """
        if control_input is not None:
            x_pred = state_transition @ self.x_hat + control_input
        else:
            x_pred = state_transition @ self.x_hat
        
        self.x_hat = x_pred
        
        return x_pred.squeeze(-1)
    
    def update(
        self,
        measurement: torch.Tensor,
        measurement_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新步骤
        
        参数:
            measurement: 测量值
            measurement_matrix: 测量矩阵
            
        返回:
            校正状态和卡尔曼增益
        """
        z = measurement.unsqueeze(-1) if measurement.dim() == 1 else measurement
        
        innovation = z - measurement_matrix @ self.x_hat
        
        S = measurement_matrix @ self.P @ measurement_matrix.T + self.R
        
        K = self.P @ measurement_matrix.T @ torch.linalg.inv(S)
        
        self.x_hat = self.x_hat + K @ innovation
        
        self.P = (torch.eye(self.state_dim) - K @ measurement_matrix) @ self.P
        
        return self.x_hat.squeeze(-1), K
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self.P = torch.eye(self.state_dim)
        self.x_hat = torch.zeros(self.state_dim, 1)


class BiomassObserver(nn.Module):
    """
    生物量观测器
    
    专门用于推断异养菌 (X_OHO) 和自养菌 (X_AOO) 浓度
    
    物理模型:
        dX_OHO/dt = μ_OHO · X_OHO - b_OHO · X_OHO
        dX_AOO/dt = μ_AOO · X_AOO - b_AOO · X_AOO
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.register_buffer("Y_H", torch.tensor(0.67))
        self.register_buffer("Y_A", torch.tensor(0.24))
        self.register_buffer("b_H", torch.tensor(0.3))
        self.register_buffer("b_A", torch.tensor(0.05))
        
        self.soft_sensor = SoftSensorObserver(
            input_dim=5,
            hidden_dim=64,
            output_dim=2
        )
    
    def forward(
        self,
        do_sequence: torch.Tensor,
        ammonia_sequence: torch.Tensor,
        nitrate_sequence: torch.Tensor,
        our_sequence: torch.Tensor,
        temperature_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        推断生物量浓度
        
        参数:
            do_sequence: 溶解氧序列 [batch, seq_len]
            ammonia_sequence: 氨氮序列
            nitrate_sequence: 硝态氮序列
            our_sequence: 氧利用率序列
            temperature_sequence: 温度序列
            
        返回:
            生物量推断结果字典
        """
        measured = torch.stack([
            do_sequence,
            ammonia_sequence,
            nitrate_sequence,
            our_sequence,
            temperature_sequence
        ], dim=-1)
        
        inferred, _ = self.soft_sensor(measured)
        
        x_ohp = inferred[:, :, 0:1]
        x_aob = inferred[:, :, 1:2]
        
        return {
            "X_OHO": x_ohp,
            "X_AOO": x_aob,
            "X_BH": x_ohp,
            "X_BA": x_aob
        }


def create_state_observer(
    state_dim: int,
    measurement_dim: int,
    device: str = "cuda"
) -> ExtendedKalmanFilter:
    """
    创建状态观测器工厂函数
    
    参数:
        state_dim: 状态维度
        measurement_dim: 测量维度
        device: 计算设备
        
    返回:
        EKF 观测器实例
    """
    observer = ExtendedKalmanFilter(
        state_dim=state_dim,
        measurement_dim=measurement_dim
    )
    return observer.to(device)
