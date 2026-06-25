# -*- coding: utf-8 -*-
# train_phase2_koopman.py
# 阶段二训练：Koopman 算子迁移学习
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import argparse
from pathlib import Path
from datetime import datetime


class KoopmanTrainer:
    """
    Koopman 算子阶段二训练器
    
    【训练目标】
    1. 迁移学习 NCDE 学到的动力学
    2. 学习 Koopman 线性嵌入
    3. 实现线性空间控制
    """
    
    def __init__(
        self,
        state_dim: int = 13,
        control_dim: int = 4,
        latent_dim: int = 64,
        device: str = "cuda"
    ) -> None:
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        
        from models.koopman_operator import KoopmanOperator
        
        self.koopman = KoopmanOperator(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.koopman.parameters(), lr=1e-3)
        
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
    
    def train_epoch(self, dataloader, epoch: int):
        """训练一个 epoch"""
        self.koopman.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for state_seq, action_seq in dataloader:
            state_seq = state_seq.to(self.device)
            action_seq = action_seq.to(self.device)
            
            loss, loss_dict = self.koopman.compute_koopman_loss(
                state_seq, action_seq
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.koopman.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {"train_loss": total_loss / n_batches}
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        torch.save({
            "koopman_state_dict": self.koopman.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "timestamp": datetime.now().isoformat()
        }, self.checkpoint_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--load_from_phase1", type=str, default=None)
    args = parser.parse_args()
    
    trainer = KoopmanTrainer()
    print("[INFO] 阶段二训练器初始化完成")


if __name__ == "__main__":
    main()
