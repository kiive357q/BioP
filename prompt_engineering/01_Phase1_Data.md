# 阶段一：工业数据工程提示词增强 (Phase 1: Data Manifold)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 阶段一 - 工业数据中枢
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席环境动力学算法科学家

---

## [Context] 局部物理上下文 (Phase 1: Data Manifold)

**目标文件**: data/dataloaders.py

**数据源**: 丹麦威立雅 Agtrup 水厂 2 分钟级高频 SCADA 数据（无缺失值）。

**特征维度**: date, IN_METAL_Q, T1_O2, METAL_Q, TEMPERATURE, IN_Q, MAX_CF, PROCESSPHASE_INLET, PROCESSPHASE_OUTLET, T1_NH4, T1_PO4

**核心难点**: 在进入底层神经常微分方程 (NCDE) 前，必须利用张量化操作将 52.5 万行数据极速转化为滑动时间窗口 (Rolling Windows)。

---

## [Execution] 核心执行逻辑

### 工业级时序数据集 WastewaterDataset(Dataset)

接收参数：处理好的 Tensor，seq_len (历史滑动窗口长度, 如 120 步), pred_horizon (预测步长, 如 30 步)。

__getitem__ 必须极速返回 (X, Y)，绝对禁止在其中包含任何逻辑判定或清洗代码。

### 具有物理感知能力的归一化器 CausalDataProcessor

- 不允许直接 import sklearn.preprocessing，必须使用纯 PyTorch Tensor 算子自行编写一个类。
- 包含 fit, transform, inverse_transform 方法。
- 重点：inverse_transform 必须能够单独对目标变量（如 T1_NH4, T1_PO4）进行反解，因为后续 RL 控制器必须使用真实的 mg/L 物理浓度进行决策。

### 数据清洗与 DataLoader 构建器 get_industrial_dataloaders

- 读取 CSV 文件，按 7:2:1 (Train/Val/Test) 进行严格的时间序列切分。
- Train 集在打包成 DataLoader 时允许 shuffle=True，Val/Test 必须为 False。
- 返回三个 DataLoader 对象及其实例化的 CausalDataProcessor。

### 模块连通性测试

- 必须在脚本末尾提供 if __name__ == '__main__':
- 生成 1000 行 mock 数据，实例化流水线，并打印 Train DataLoader 第一个 Batch 的 X.shape 和 Y.shape。

---

## [Constraints] 模块级绝对红线警告 (VIOLATION = FAILURE)

### 防除零刚性熔断 (Critical)

在 get_industrial_dataloaders 读取数据并转为张量时，**必须立即针对 T1_O2 (溶解氧) 列执行硬截断**。必须写出类似 `torch.clamp(o2_tensor, min=1e-5)` 的逻辑，确保绝对不可能出现 0.0，以防止在 NCDE 的广义 Monod 抑制项中引发除零崩溃！

### 纯张量化无循环

无论是在处理滑动窗口还是在归一化阶段，**绝对禁止使用原生的 Python for 循环一行行遍历 Pandas DataFrame**。必须使用 PyTorch 的张量广播 (Broadcasting) 或 unfold 操作进行升维切片。

### 零占位符

- 禁止任何 pass 或 # 此处实现逻辑
- 必须输出完全可运行的代码
- 必须包含详细的中文物理意义注释与 Type Hints

---

## 特征变量物理定义

| 特征名 | 物理意义 | 单位 | 典型范围 |
|--------|---------|------|---------|
| date | 时间戳 | - | - |
| IN_METAL_Q | 进水金属量 | - | - |
| T1_O2 | 曝气池溶解氧浓度 | mg/L | 0.5 - 8.0 |
| METAL_Q | 金属量 | - | - |
| TEMPERATURE | 水温 | °C | 8 - 20 |
| IN_Q | 进水流量 | m³/d | 15000 - 22000 |
| MAX_CF | 最大化学需氧量 | - | - |
| PROCESSPHASE_INLET | 进水工艺阶段 | - | - |
| PROCESSPHASE_OUTLET | 出水工艺阶段 | - | - |
| T1_NH4 | 曝气池氨氮浓度 | mg N/L | 0.5 - 10.0 |
| T1_PO4 | 曝气池磷酸盐浓度 | mg P/L | 1.0 - 10.0 |

### 关键物理约束

1. **T1_O2 (溶解氧)**: 在 Monod 抑制项中作为分母，必须严格截断至 > 1e-5
2. **T1_NH4, T1_PO4**: 出水水质目标变量，必须在 inverse_transform 中保留物理单位
3. **物质守恒**: 归一化后必须可逆，保持 C/N/P 物质流

---

## 数据集统计特征

- **数据来源**: 丹麦威立雅 Agtrup 水厂 SCADA 系统
- **采样频率**: 2 分钟/采样点
- **数据行数**: 约 52.5 万行
- **时间跨度**: 2023年8月 (Aug 2023)
- **数据质量**: 无缺失值

### 时间序列切分

```
Train: Val: Test = 7:2:1
     时间顺序不可打乱！
```

### 滑动窗口参数

- **seq_len (历史窗口)**: 120 步 = 240 分钟 = 4 小时
- **pred_horizon (预测窗口)**: 30 步 = 60 分钟 = 1 小时

---

## 模块接口规范

### WastewaterDataset

```python
class WastewaterDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int = 120,
        pred_horizon: int = 30
    ) -> None: ...
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

### CausalDataProcessor

```python
class CausalDataProcessor:
    def fit(self, data: torch.Tensor) -> "CausalDataProcessor": ...
    
    def transform(self, data: torch.Tensor) -> torch.Tensor: ...
    
    def inverse_transform(
        self,
        data: torch.Tensor,
        target_indices: Optional[list] = None
    ) -> torch.Tensor: ...
```

### get_industrial_dataloaders

```python
def get_industrial_dataloaders(
    csv_path: str,
    seq_len: int = 120,
    pred_horizon: int = 30,
    batch_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader, CausalDataProcessor]: ...
```

---

**版本**: V1.0-Phase1
**制定日期**: 2026-05-29
**适用范围**: 数据工程模块 - 阶段一
