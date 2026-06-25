# 里程碑零：发车前检验与张量连通性验证 (Milestone 0: Pre-flight Validation)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 里程碑零 - 发车前检验
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席科学家 / 顶刊独立审查委员会主席

---

## [Context] 局部物理上下文 (Milestone 0: Pre-flight Validation)

### 目标文件 1: data/data_validation.py (数据张量与流形连续性体检)

### 目标文件 2: training/integration_dryrun.py (全链路张量连通图、梯度流穿透与显存健康度测试)

### 深层物理与工程痛点

**梯度断崖与图断裂**: 多个模块（DataLoader -> SINDy -> NCDE -> BSM1 -> RL -> CBF 安全约束）由不同的算子拼接。只要有一个 Numpy 操作混入，或者张量 View 复制不当，反向传播的计算图（Autograd Graph）就会被悄无声息地切断。

**刚性方程 (Stiff ODE) 灾难**: BSM1 生化反应是一个典型的刚性系统（曝气扩散在秒级，微生物生长在天级）。未经校验的极小值（如 DO 或底物浓度接近 0）在积分器中会引发雅可比矩阵（Jacobian）条件数激增，几个 Epoch 后就会得到一堆 NaN。

**质量守恒违背**: 如果 SINDy 提取的字典或插值函数在隐空间中创造了"凭空出现的质量"（例如负浓度的氨氮，或者出水总磷大于进水），这将触碰顶刊的绝对学术死刑。

### 学术叙事与消融实验基准

顶刊极其看重"消融实验 (Ablation)"和"物理守恒证明"。我们的校验脚本不仅是找 Bug，更是从严格的数学分析上证明：在随机初始化下，模型的梯度方向能正确反映物理规律。

---

## [Execution] 核心执行逻辑

### IndustrialDataValidator

**探针 A (物理刚性边界)**: 检查 T1_O2 张量全集，assert torch.min(o2_tensor) >= 1e-5，若失败则直接抛出"除零截断失效"的致命错误。

**探针 B (龙格现象拦截)**: 针对三次样条插值后的数据，检查是否存在异常的振荡。如果插值后两个化验锚点之间的极大值超过了局部锚点均值的 300%，必须抛出"流形重建产生龙格现象"警告。

**探针 C (因果倾向得分饱和)**: 统计并打印逆概率权重 (IPW) 的分布方差。如果倾向得分极度贴近 0 或 1，导致权重出现10^4以上的数量级，需立刻抛出红色异常。

### SystemIntegrationDryRun

**正向流形连通性与内存泄漏测试**: 依次穿过 SINDy -> NCDE -> BioPEnv -> RL -> CBF，动态显存追踪。

**逆向梯度穿透性测试**: 虚构复合物理损失，执行 loss.backward()，检查所有可学习参数的梯度健康度。

---

## [Constraints] 模块级绝对红线警告

### 极端严苛的断言驱动 (Assertion-Driven Development)

所有的检验脚本绝不是用来打印出几行好看的日志安慰自己的，而是用来随时随地让程序崩溃的。你必须写满 assert 语句，只要有一处不符合张量维度对齐、物理守恒或梯度健康，立马 raise AssertionError。

### 自动化测试报告生成 (CI/CD Style)

integration_dryrun.py 运行结束后，必须在控制台输出一段类似大厂 DevOps 管道的 ASCII 格式"全链路连通性及梯度健康度测试报告"。

---

## 三大探针设计

### 探针A：物理刚性边界

```python
assert torch.min(o2_tensor) >= 1e-5, "【顶刊红线】DO浓度存在接近零值，除零截断失效"
assert torch.min(concentration_tensors) >= 0.0, "【顶刊红线】负浓度检测，违反物理实在性"
```

### 探针B：龙格现象拦截

```python
overshoot_ratio = local_max / local_mean
assert overshoot_ratio <= 3.0, "【顶刊红线】样条插值产生龙格振荡，过冲比={overshoot_ratio}"
```

### 探针C：因果倾向得分饱和

```python
ipw_weights = 1.0 / propensity_scores
assert ipw_weights.max() < 1e4, "【顶刊红线】IPW权重溢出，数量级={ipw_weights.max()}"
```

---

## 梯度健康度检测

### 梯度断裂检测

```python
assert param.grad is not None, "【顶刊红线】梯度为None，计算图断裂"
```

### NaN/Inf检测

```python
assert not torch.isnan(param.grad).any(), "【顶刊红线】梯度存在NaN，刚性方程崩溃"
assert not torch.isinf(param.grad).any(), "【顶刊红线】梯度存在Inf，除零检测失效"
```

### 梯度消失检测

```python
assert param.grad.norm() > 1e-7, "【顶刊红线】梯度范数趋近零，伴随灵敏度失效"
```

---

## 测试报告格式

```
============================================================
  全链路连通性及梯度健康度测试报告
  BioP Causal WorldModel V2.0 - Milestone 0
============================================================

[环境初始化]        [PASS] / [FAIL]
  - PyTorch版本: X.X.X
  - CUDA可用: Yes/No
  - GPU型号: XXXX

[模块导入]          [PASS] / [FAIL]
  - SINDyMonodLibrary
  - NeuralCDE
  - BioPEnv
  - LexicographicSACAgent
  - QPActionInterceptor

[数据加载]          [PASS] / [FAIL]
  - 探针A: 物理边界检查
  - 探针B: 龙格现象检查
  - 探针C: IPW饱和检查

[前向传播链]        [PASS] / [FAIL]
  - SINDy特征扩展
  - NCDE积分推演
  - BioPEnv step
  - RL动作生成
  - CBF安全拦截

[显存健康]         [PASS] / [FAIL]
  - 5次连续推演显存增量

[梯度穿透性]       [PASS] / [FAIL]
  - 可学习参数梯度检测
  - NaN/Inf检测
  - 梯度范数检测

============================================================
  测试结论: [全部通过] / [存在故障，需修复后重测]
============================================================
```

---

**版本**: V2.0-Milestone0-PreflightCheck
**制定日期**: 2026-05-29
**适用范围**: 发车前检验 - 全链路验证与梯度健康度测试
