# -*- coding: utf-8 -*-
# attention_decay.py
# 时序注意力衰减机制，捕获长期依赖
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch import Tensor


class TemporalAttentionDecay(nn.Module):
    """
    时序注意力衰减机制
    
    【物理意义】
    污水处理过程中，当前状态对历史信息的依赖呈指数衰减:
    
    α(t, τ) = exp(-λ · (t - τ)) / Z(t)
    
    其中 λ 是衰减率，Z(t) 是归一化常数
    
    【应用场景】
    - 冲击负荷检测：最近的历史数据权重更高
    - 季节性模式：长期趋势捕获
    - 传感器故障检测：异常值抑制
    """
    
    def __init__(
        self,
        hidden_dim: int,
        decay_rate: float = 0.1,
        learnable_decay: bool = True,
        max_seq_len: int = 512
    ) -> None:
        """
        初始化时序注意力衰减
        
        参数:
            hidden_dim: 隐藏层维度
            decay_rate: 基础衰减率
            learnable_decay: 是否学习衰减率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        if learnable_decay:
            self.log_decay = nn.Parameter(torch.tensor(math.log(decay_rate)))
        else:
            self.register_buffer("decay", torch.tensor(decay_rate))
        
        self.attention_proj = nn.Linear(hidden_dim, 1, bias=False)
    
    @property
    def decay_rate(self) -> Tensor:
        """获取当前衰减率"""
        if hasattr(self, "log_decay"):
            return torch.exp(self.log_decay.clamp(-10, 2))
        return self.decay
    
    def compute_decay_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tensor:
        """
        计算时间衰减掩码
        
        【数学公式】
        decay_mask[t, τ] = exp(-decay · (t - τ)) for τ <= t
                         = 0 for τ > t
        
        参数:
            seq_len: 序列长度
            device: 计算设备
            
        返回:
            衰减掩码矩阵 [seq_len, seq_len]
        """
        positions = torch.arange(seq_len, device=device).float()
        
        time_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        time_diff = torch.clamp(time_diff, min=0)
        
        decay = self.decay_rate.item() if isinstance(self.decay_rate, Tensor) else self.decay_rate
        decay_mask = torch.exp(-decay * time_diff)
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        decay_mask = decay_mask * causal_mask
        
        return decay_mask
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        参数:
            query: 查询向量 [batch, seq_len, hidden_dim]
            key: 键向量 (如果为 None，则使用 query)
            value: 值向量 (如果为 None，则使用 query)
            mask: 注意力掩码
            
        返回:
            output: 注意力加权输出
            attention_weights: 注意力权重
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        q = self.attention_proj(query)
        k = self.attention_proj(key)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        decay_mask = self.compute_decay_mask(seq_len, query.device)
        
        scores = scores + torch.log(decay_mask.unsqueeze(0) + 1e-8)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttentionDecay(nn.Module):
    """
    多头时序注意力衰减
    
    【功能】
    多个并行的注意力头，捕获不同时间尺度的依赖关系
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        decay_rate: float = 0.1,
        dropout: float = 0.1
    ) -> None:
        """
        初始化多头注意力
        
        参数:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量
            decay_rate: 衰减率
            dropout: Dropout 比例
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.attention_heads = nn.ModuleList([
            TemporalAttentionDecay(
                hidden_dim=self.head_dim,
                decay_rate=decay_rate * (1.0 + i * 0.1)
            )
            for i in range(num_heads)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        多头注意力前向传播
        
        参数:
            query: 查询 [batch, seq_len, hidden_dim]
            key: 键 (可选)
            value: 值 (可选)
            mask: 掩码
            
        返回:
            output: 输出
            all_attention_weights: 所有头的注意力权重
        """
        batch_size, seq_len, _ = query.shape
        
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim) if key is not None else None
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim) if value is not None else None
        
        outputs = []
        attention_weights_list = []
        
        for i, attn_head in enumerate(self.attention_heads):
            q_i = query[:, i, :, :]
            k_i = key[:, i, :, :] if key is not None else None
            v_i = value[:, i, :, :] if value is not None else None
            
            out, attn = attn_head(q_i, k_i, v_i, mask)
            outputs.append(out)
            attention_weights_list.append(attn)
        
        output = torch.stack(outputs, dim=1)
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        
        output = self.dropout(self.output_proj(output))
        
        all_attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return output, all_attention_weights


class ExponentiallyDecayedLSTM(nn.Module):
    """
    指数衰减 LSTM
    
    【物理意义】
    将时序注意力衰减机制融入 LSTM 单元，
    替代标准的遗忘门机制
    
    数学公式:
        f_t = σ(W_f · [h_{t-1} * exp(-λ·Δt), x_t] + b_f)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        decay_rate: float = 0.05
    ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_transform = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        self.decay_proj = nn.Linear(1, hidden_dim)
    
    def forward(
        self,
        x: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
        time_delta: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        LSTM 单步前向传播
        
        参数:
            x: 输入 [batch, input_dim]
            h_prev: 上一隐状态 [batch, hidden_dim]
            c_prev: 上一细胞状态 [batch, hidden_dim]
            time_delta: 时间间隔 [batch, 1] (可选)
            
        返回:
            h_new: 新隐状态
            c_new: 新细胞状态
        """
        combined = torch.cat([h_prev, x], dim=-1)
        
        i_t = torch.sigmoid(self.input_gate(combined))
        
        if time_delta is not None:
            decay_factor = torch.exp(-self.decay_rate * time_delta)
            decay_factor = self.decay_proj(decay_factor)
            decay_factor = torch.sigmoid(decay_factor)
        else:
            decay_factor = torch.sigmoid(
                torch.ones_like(h_prev) * self.decay_rate
            )
        
        f_t = decay_factor * torch.sigmoid(self.forget_gate(combined))
        
        c_tilde = torch.tanh(self.cell_transform(combined))
        c_new = f_t * c_prev + i_t * c_tilde
        
        o_t = torch.sigmoid(self.output_gate(combined))
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new
