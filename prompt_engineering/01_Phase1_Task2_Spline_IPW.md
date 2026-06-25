# 阶段一任务二：物理流形重建与因果去偏 (Phase 1 Task 2: Manifold & IPW)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 阶段一 - 任务二
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席环境动力学算法科学家

---

## [Context] 局部物理上下文 (Phase 1 Task 2: Manifold & IPW)

### 目标文件 1: data/cubic_spline_interp.py (自然三次样条插值算子)

### 目标文件 2: data/ipw_confounder.py (逆概率加权因果解耦)

### 物理痛点

我们的 2 分钟级 SCADA 数据极其缺乏高频的进水 COD（碳源）和 MLSS（污泥浓度）。

### 学术叙事 (Grey-Box Fusion)

我们不能用黑盒瞎猜。我们要求将化验室低频数据（假设每日 1 次）作为物理刚性锚点（Hard Anchors），利用 cubic_spline_interp.py 在两个锚点间进行流形插值；同时，考虑到低频数据与高频 DO 之间存在的因果混淆（Confounding Bias），我们需要 ipw_confounder.py 来计算倾向得分（Propensity Score）以动态调整重构权重。

---

## [Execution] 核心执行逻辑

### 可微分三次样条算子 DifferentiableCubicSpline

不要直接调用缺乏梯度追踪的 scipy.interpolate，必须使用纯 PyTorch 张量编写或封装一个支持 autograd（反向传播）的自然三次样条插值类。

输入: 形状为 (Batch, Seq_Len, Features) 的高频张量，以及表示低频锚点的稀疏张量掩码（Mask）。

输出: 填补完毕的平滑时序张量。

### 倾向得分匹配网络 PropensityScoreNetwork

继承 nn.Module，构建一个轻量级 MLP（如 3 层），输入特征张量，输出采取某一特定动作/状态的概率 P(A|X)。

### 动态 IPW 损失加权器 IPWLossWeighter

计算逆概率权重：W = 1 / P(A|X)。

提供一个接口 apply_weights(base_loss, weights)，用于在后续的 NCDE 训练中修正损失函数。

---

## [Constraints] 模块级绝对红线警告 (VIOLATION = FAILURE)

### 反向梯度爆炸防御 (IPW Critical)

在 ipw_confounder.py 计算 1 / P(A|X) 时，如果分母 P(A|X) 极小，会导致张量梯度瞬间爆炸（NaN）！你必须强制在分母引入截断机制，例如 `weights = 1.0 / torch.clamp(propensity_scores, min=1e-4)`，绝不允许出现裸除！

### 伴随灵敏度兼容性

DifferentiableCubicSpline 的所有内部矩阵求解（如解三对角矩阵方程组）必须使用 PyTorch 原生的 `torch.linalg.solve` 或 `torch.bmm`，绝对禁止将其转化为 Numpy 运算，否则将彻底破坏后续神经常微分方程 (NCDE) 的伴随灵敏度图追踪（Adjoint Sensitivity Method）！

### 零占位符与测试入口

两个脚本都必须在底部包含 `if __name__ == "__main__":`，各自 mock 几个高维张量（例如 Batch=32, Seq_Len=120, Features=11），实例化类，执行一次前向传播，并打印出输出的 Tensor Shape 字典。

---

## 核心算法数学原理

### 三次样条插值

给定 n 个控制点 (x_i, y_i)，自然三次样条 S(x) 满足：

1. **插值条件**: S(x_i) = y_i, ∀i ∈ {0, ..., n-1}
2. **自然边界**: S''(x_0) = S''(x_{n-1}) = 0
3. **C2 连续性**: S'(x) 和 S''(x) 在节点处连续

### 三对角矩阵求解 (Thomas Algorithm)

对于 n 个区间，需要求解三对角系统：
```
| b0 c0  0  ...  0 | | m0 |   | d0 |
| a1 b1 c1 ...  0 | | m1 |   | d1 |
|  0 a2 b2 c2 ... | | m2 | = | d2 |
| ...            | |...|   |... |
|  0  ...  an bn | | mn |   | dn |
```

使用 PyTorch 的 `torch.linalg.solve` 进行 GPU 加速求解。

### 逆概率加权 (IPW)

倾向得分: e(x) = P(A=1 | X=x)

IPW 权重: W = 1 / e(x) (处理组) 或 W = 1 / (1-e(x)) (对照组)

---

## 变量物理定义

| 变量 | 物理意义 | 单位 |
|------|---------|------|
| COD | 化学需氧量 | mg/L |
| MLSS | 混合液悬浮固体浓度 | mg/L |
| T1_O2 | 曝气池溶解氧 | mg/L |
| Propensity Score | 倾向得分 | [0, 1] |
| IPW Weight | 逆概率权重 | ≥ 1 |

---

**版本**: V1.0-Phase1-Task2
**制定日期**: 2026-05-29
**适用范围**: 数据工程模块 - 物理流形重建与因果去偏
