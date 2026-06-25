# 阶段四：物理约束训练流水线 (Phase 4: Physics-Informed Training)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 阶段四 - 训练流水线
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席环境动力学算法科学家 / AI训练架构师

---

## [Context] 局部物理上下文 (Phase 4: Physics-Informed Training)

### 目标文件 1: models/pinn_loss.py (物理信息神经网络损失函数)

### 目标文件 2: training/train_phase1_ncde.py (因果世界模型阶段一主训练脚本)

### 核心意图

我们要训练阶段一封装好的 ncde_solver.py。但是，在常规的预测误差 (MSE) 之外，为了确保模型满足《Water Research》等顶刊的要求，我们必须引入 PINN 损失函数，强制要求模型的预测结果符合 C/N/P（碳氮磷）质量守恒定律。

### 工程标准

训练脚本必须支持混合精度训练 (AMP, Automatic Mixed Precision) 以加速计算，并集成动态梯度裁剪与学习率衰减。

---

## [Execution] 核心执行逻辑

### MassBalanceLoss

继承 nn.Module。

核心数学重构: 编写 forward(pred_states, inputs, time_delta) 方法。利用生化反应常数，计算输入碳源/氮源与输出、污泥消耗之间的差值惩罚项（Residuals）。

即使无法做到绝对精确的质量守恒计算，也必须用代码写出"稳态下氨氮减少量与硝酸盐增加量成正比"这种物理学强关联的张量正则化惩罚项（Regularization Term）。

### BioPCompositeLoss

将数据驱动损失（Data Loss, 如 MSE）、逆概率因果去偏权重（IPW Weights，由 ipw_confounder.py 提供）以及物理损失（Physics Loss）进行自适应加权组合。

### 主训练循环 (train_phase1_ncde.py)

组件导入: 实例化 WastewaterDataset/DataLoader, SINDyMonodLibrary, NCDE_Odeint, BioPCompositeLoss。

混合精度: 必须使用 torch.cuda.amp.autocast() 和 GradScaler 编写高效的前向/反向传播逻辑，以应对数万规模 Batch Size 的极端吞吐。

梯度防爆: 在 scaler.step(optimizer) 之前，强制调用 torch.nn.utils.clip_grad_norm_ 将梯度截断在合理的阈值（如 1.0）。

早停与日志: 实现基于 Validation Loss 的 Early Stopping 机制，并将每 Epoch 的 Loss（拆分为 Data Loss 和 Physics Loss 分别记录）打印出来。

---

## [Constraints] 模块级绝对红线警告

### 杜绝内存泄漏 (Dataloader to Device)

在训练循环中，从 DataLoader 取出数据送到 GPU (data.to(device)) 后，绝对禁止保存计算图历史的 Tensor 到外部列表中（如累加 total_loss 时，必须使用 loss.item()）。

### 严苛的测试验证

train_phase1_ncde.py 不仅是一段脚本，必须在文件底部设置标准化的入口 if __name__ == '__main__'，使用 argparse 解析终端命令，允许通过 --epochs 1 进行 dry-run 测试。

---

## PINN损失函数设计

### 质量守恒原理

对于ASM1模型，碳/氮/磷质量守恒要求：
- 进水总氮 = 出水总氮 + 污泥去除氮
- 进水COD = 出水COD + 污泥增长COD
- 稳态下：ΔNH4 与 ΔNO3 成正比（硝化-反硝化平衡）

### 复合损失函数

$$\mathcal{L}_{total} = \lambda_{data} \mathcal{L}_{data} + \lambda_{physics} \mathcal{L}_{physics} + \lambda_{ipw} \mathcal{L}_{ipw}$$

---

## 训练工程标准

### 混合精度训练 (AMP)

- 使用 torch.cuda.amp.autocast() 进行前向传播
- 使用 GradScaler 管理反向传播缩放
- 减少显存占用，支持更大Batch Size

### 梯度裁剪

- clip_grad_norm_(parameters, max_norm=1.0)
- 防止梯度爆炸

### 早停机制

- patience=10 epochs
- monitor validation loss
- restore best model weights

---

**版本**: V2.0-Phase4-TrainingPipeline
**制定日期**: 2026-05-29
**适用范围**: 训练流水线模块 - PINN损失 + 混合精度训练
