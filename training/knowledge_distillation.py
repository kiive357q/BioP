# -*- coding: utf-8 -*-
# knowledge_distillation.py
# 知识蒸馏：大规模模型压缩至边缘部署
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path


class KnowledgeDistiller:
    """
    知识蒸馏器
    
    【功能】
    将大模型知识迁移到小模型，适配边缘部署
    
    【蒸馏损失】
    L_total = α · L_student + (1-α) · L_kd
    
    其中:
    L_kd = KL(p_teacher || p_student) · T²
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.3
    ) -> None:
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_distillation_loss(
        self,
        inputs: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算蒸馏损失
        
        参数:
            inputs: 输入数据
            hard_labels: 硬标签
            
        返回:
            总损失和损失字典
        """
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = self.student(inputs)
        
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        
        kd_loss = nn.functional.kl_div(
            student_log_probs,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        student_loss = nn.functional.cross_entropy(student_logits, hard_labels)
        
        total_loss = self.alpha * student_loss + (1 - self.alpha) * kd_loss
        
        return total_loss, {
            "student_loss": student_loss,
            "kd_loss": kd_loss,
            "total_loss": total_loss
        }
    
    def distill(
        self,
        dataloader,
        epochs: int = 100,
        output_dir: str = "checkpoints/distilled"
    ) -> None:
        """执行蒸馏"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            self.student.train()
            
            total_loss = 0.0
            for inputs, labels in dataloader:
                loss, _ = self.compute_distillation_loss(inputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")
        
        torch.save(self.student.state_dict(), output_path / "distilled_model.pt")
        print(f"[INFO] 蒸馏完成，模型保存至 {output_path}")


def main():
    """主函数"""
    print("[INFO] 知识蒸馏模块初始化完成")


if __name__ == "__main__":
    main()
