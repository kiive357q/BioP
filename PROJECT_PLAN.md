# BioP Causal WorldModel V2.0 · 分任务完美计划书

> **项目定位**：污水生化除磷工艺 · 世界模型驱动的动态强化学习控制框架
> **目标参数量**：~30M（latent_dim=896, hidden_dim=896）
> **核心 KPI**：NCDE 训练 50 epoch 无 NaN + 稳定收敛 + TP 预测误差 ≤ 10%
> **当前进度**：~65%（工程骨架扎实，训练尚未闭环）
> **总体工期**：约 8 周（可并行压缩至 4 周）
> **版本**：v1.0 · 2026-06-26

---

## 一、总体路线图（5 大阶段 + 验收门禁）

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Phase 0: 参数定标 & 代码健康 (1 周)                                                  │
│  ├── Task 0.1 锁定 30M 配置 (latent=896, hidden=896)                                   │
│  ├── Task 0.2 清理残留问题 & 冒烟测试                                                  │
│  └── Task 0.3 建立单元测试基线                                                        │
│  ▼  门禁：python tools/estimate_params.py 输出 ≈30M；pytest 通过                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: NCDE 世界模型训练收敛 (2 周)                                                │
│  ├── Task 1.1 数值稳定性加固 (dtype/梯度/PINN warmup)                                 │
│  ├── Task 1.2 小样本训练验证 (50 epoch, loss < 0.1 无 NaN)                            │
│  ├── Task 1.3 全量训练 (500 epoch) + Checkpoint                                       │
│  └── Task 1.4 物质守恒与动力学一致性验证                                             │
│  ▼  门禁：Val Loss < 0.05；TP 预测 NRMSE < 10%；质量守恒误差 < 1e-4                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 2: Koopman + SINDy 因果结构学习 (1.5 周)                                        │
│  ├── Task 2.1 Koopman 算子迁移学习                                                    │
│  ├── Task 2.2 SINDy 稀疏动力学识别                                                    │
│  └── Task 2.3 因果结构图与 IPW 去偏                                                  │
│  ▼  门禁：Koopman 预测稳定；SINDy 项稀疏（≤ 12 项）；IPW 权重分布合理                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 3: BSM1 闭环 RL 强化学习 (2 周)                                                │
│  ├── Task 3.1 字典序 SAC 接入 BSM1 环境                                              │
│  ├── Task 3.2 安全约束 (CBF + QP 拦截器) 集成                                         │
│  ├── Task 3.3 多目标优化（TP 达标 + 能耗下降）                                         │
│  └── Task 3.4 知识蒸馏压缩                                                            │
│  ▼  门禁：TP 稳定 < 0.5 mg/L；曝气能耗下降 ≥ 15%；CBF 零违规                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 4: 部署与上线 (1.5 周)                                                          │
│  ├── Task 4.1 ONNX 导出与验证                                                         │
│  ├── Task 4.2 Rust 边缘推理引擎编译                                                    │
│  ├── Task 4.3 OPC-UA 桥接与影子模式                                                   │
│  └── Task 4.4 灰度上线 + 监控                                                         │
│  ▼  门禁：ONNX Runtime 推理误差 < 1e-5；边缘延迟 < 50ms；影子模式稳定 7 天             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Phase 0：参数定标 & 代码健康（1 周）

### Task 0.1 锁定 30M 参数预算 ⭐（最高优先级）

**目标**：把默认 `latent_dim=1024/hidden_dim=1024`（41M）调整为 `latent_dim=896/hidden_dim=896`（≈32M，符合 30M 预算）

**涉及文件**：
- 修改：[configs/model_config.yaml](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/configs/model_config.yaml)
- 修改：[models/expanded_ncde.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/expanded_ncde.py)（`create_expanded_model` 默认参数）
- 验证：[tools/estimate_params.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/tools/estimate_params.py)

**步骤**：
1. 在 `configs/model_config.yaml` 新增 `world_model` 段：
   ```yaml
   world_model:
     latent_dim: 896
     hidden_dim: 896
     n_ncde_layers: 4
     action_encoder_dim: 256
   ```
2. 修改 `models/expanded_ncde.py` 中 `create_expanded_model` 默认参数为 `latent_dim=896, hidden_dim=896`
3. 运行 `python tools/estimate_params.py` 验证 ≈ 32M
4. 运行 `python -c "from models.expanded_ncde import create_expanded_model; m=create_expanded_model(); print(sum(p.numel() for p in m.parameters())/1e6, 'M')"`

**验收标准**：
- [ ] 总参数量 28M–33M
- [ ] 默认配置下无手动传参即可正确实例化

---

### Task 0.2 残留问题清理 & 冒烟测试

**涉及文件**：
- [models/ncde_solver.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/ncde_solver.py)
- [envs/bsm1_full_simulation.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/envs/bsm1_full_simulation.py)
- [data/dataloaders.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/data/dataloaders.py)

**步骤**：
1. 逐个运行冒烟测试：
   ```bash
   python -c "from models.ncde_solver import NCDESolver; s=NCDESolver(); print('OK')"
   python -c "from envs.bsm1_full_simulation import BioPSimulator, BSM1FullConfig; s=BioPSimulator(BSM1FullConfig(), batch_size=2); print(s.step(s.action_space.sample()))"
   python -c "from data.dataloaders import WastewaterDataset; ds=WastewaterDataset('data/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023(1).csv'); print(len(ds))"
   ```
2. 修复剩余 RuntimeError/警告
3. 输出 `tests/smoke/test_core_modules.py` 统一冒烟测试脚本

**验收标准**：
- [ ] 三个核心模块冒烟测试 100% 通过
- [ ] 无警告（或仅有可忽略的 UserWarning）

---

### Task 0.3 单元测试基线建设

**涉及文件**（新增）：
- `tests/smoke/test_core_modules.py`
- `tests/numerical/test_nan_guard.py`
- `tests/numerical/test_mass_conservation.py`
- `tests/numerical/test_gradient_clipping.py`

**关键测试用例**：
```python
def test_no_nan_after_100_steps():
    # 前向传播 100 步无 NaN
    ...

def test_mass_conservation_residual():
    # C/N/P 三相物质守恒残差 < 1e-4
    ...

def test_gradient_clipping_effective():
    # 超阈值梯度被裁剪到 max_norm
    ...
```

**验收标准**：
- [ ] `pytest tests/ -v` 通过率 100%
- [ ] `pytest tests/ --cov=models --cov-report=term` 覆盖率 ≥ 70%

---

## 三、Phase 1：NCDE 世界模型训练收敛（2 周）

### ⚠️ 核心阻断项，后续所有阶段依赖此阶段成功

### Task 1.1 数值稳定性加固

**问题**：原训练出现 NaN，根因包括 dtype 漂移、物理损失权重过大、学习率过高

**涉及文件**：
- [training/train_phase1_ncde.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/training/train_phase1_ncde.py)
- [models/pinn_loss.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/pinn_loss.py)
- [configs/model_config.yaml](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/configs/model_config.yaml)

**修改要点**：
1. **强制 dtype 一致**：`model.to(torch.float32)` + AMP autocast 限定 `dtype=torch.float32`
2. **物理损失 warmup**：前 20 epoch `mass_conservation_weight=0.1`，线性升温到 1.0
3. **学习率下调**：从 `1e-3` 降到 `3e-4`，配合 CosineAnnealingLR
4. **梯度裁剪**：`max_norm=0.5`（原 1.0 太松）
5. **输入归一化**：按 z-score 归一化，均值/方差固化到 buffer

**验收标准**：
- [ ] 配置完成后连续 10 epoch 训练无 NaN
- [ ] 梯度范数稳定 ≤ 0.5

---

### Task 1.2 小样本快速验证（50 epoch）

**命令**：
```bash
python -m training.train_phase1_ncde \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --log_dir logs \
    --checkpoint_dir checkpoints \
    --warmup_epochs 20 \
    --clip_value 0.5
```

**验证项**：
- 每 epoch 打印 `train_loss / val_loss / grad_norm / nan_count`
- 输出 `logs/training_phase1_smoke.csv` 追踪曲线
- 训练结束后运行 `scripts/evaluate_ncde.py`（需新建）：
  - TP NRMSE < 10%
  - 动力学一致性误差（预测 vs 真实下一状态）< 5%

**门禁**：
- [ ] 50 epoch 零 NaN
- [ ] Val Loss 从初始值下降 ≥ 50%
- [ ] TP NRMSE < 10%

---

### Task 1.3 全量训练（500 epoch）

**步骤**：
1. 在 Task 1.2 通过基础上扩展到 500 epoch
2. 每 20 epoch 保存 checkpoint：`checkpoints/phase1_epoch_{k}.pt`
3. 保存最优模型：`checkpoints/phase1_best.pt`（按 Val Loss）
4. 保存最终模型：`checkpoints/phase1_final.pt`
5. 生成训练曲线报告：`logs/training_phase1_full_report.md`

**验收标准**：
- [ ] 全程 500 epoch 零 NaN
- [ ] 最终 Val Loss < 0.05
- [ ] 最优 Checkpoint 优于初始 ≥ 3x
- [ ] 生成可视化训练曲线（loss / TP 预测 / 物质守恒）

---

### Task 1.4 物质守恒与动力学一致性验证

**涉及文件**：
- [models/pinn_loss.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/pinn_loss.py)
- 新建：`scripts/verify_physics.py`

**验证项**：
```python
def test_mass_conservation():
    # 对 100 个随机初态跑 50 步预测
    # C_total / N_total / P_total 相对变化 < 1e-4
    ...

def test_kinetic_consistency():
    # 预测动力学方程右端与真实观测 MSE < 5e-3
    ...

def test_long_horizon_stability():
    # 100 步 rollout 无爆炸、无稳态偏移
    ...
```

**门禁**：
- [ ] 物质守恒误差 < 1e-4
- [ ] 动力学一致性 MSE < 5e-3
- [ ] 100 步预测稳定

---

## 四、Phase 2：Koopman + SINDy 因果结构学习（1.5 周）

### Task 2.1 Koopman 算子迁移学习

**涉及文件**：
- [models/koopman_operator.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/koopman_operator.py)
- [training/train_phase2_koopman.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/training/train_phase2_koopman.py)

**步骤**：
1. 载入 `checkpoints/phase1_best.pt` 作为预训练编码器
2. 冻结 NCDE 主干，仅训练 Koopman 线性算子 `K`
3. 对 `K` 施加谱约束（特征值在单位圆内，保证稳定性）
4. 训练 300 epoch，保存 `checkpoints/phase2_koopman.pt`

**验收标准**：
- [ ] Koopman 预测 50 步 NRMSE < 5%
- [ ] `K` 的特征值半径 < 0.99

---

### Task 2.2 SINDy 稀疏动力学识别

**涉及文件**：
- [models/sindy_library.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/models/sindy_library.py)

**步骤**：
1. 用 Koopman 编码后的 latent 序列构造 SINDy 候选库
2. 阈值迭代 10 次，保留项数 ≤ 12
3. 输出解析动力学方程到 `models/sindy_equation.py`
4. 与 NCDE 对比验证预测精度

**验收标准**：
- [ ] 非零项数 ≤ 12
- [ ] 预测精度损失 < 1%（相比完整 NCDE）
- [ ] 生成可解释的动力学方程

---

### Task 2.3 因果结构图与 IPW 去偏

**涉及文件**：
- [data/ipw_confounder.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/data/ipw_confounder.py)

**步骤**：
1. 基于 SINDy 结果画因果图 `docs/causal_graph.svg`
2. 计算 IPW 权重（倾向得分匹配）
3. 对比加/不加 IPW 的下游 RL 表现

**验收标准**：
- [ ] 因果图有清晰物理语义
- [ ] IPW 覆盖率 ≥ 95%
- [ ] 加权后偏差下降 ≥ 20%

---

## 五、Phase 3：BSM1 闭环 RL 强化学习（2 周）

### Task 3.1 字典序 SAC 接入 BSM1

**涉及文件**：
- [rl_agents/lexicographic_sac.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/rl_agents/lexicographic_sac.py)
- [envs/bsm1_full_simulation.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/envs/bsm1_full_simulation.py)

**步骤**：
1. 用训练好的世界模型作为 `world_model` 接入 RL
2. 字典序目标：① TP 达标 → ② TN 达标 → ③ 能耗最小
3. 训练 1M steps，每 10k step 评估一次
4. 保存 `checkpoints/rl_lexicographic.pt`

**验收标准**：
- [ ] TP 稳定 < 0.5 mg/L（95% 时间）
- [ ] TN 稳定 < 15 mg/L（95% 时间）
- [ ] 曝气能耗下降 ≥ 15%（相比固定策略）

---

### Task 3.2 安全约束集成

**涉及文件**：
- [safety/control_barrier_functions.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/safety/control_barrier_functions.py)
- [safety/qp_interceptor.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/safety/qp_interceptor.py)
- [safety/reachability_tube.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/safety/reachability_tube.py)

**步骤**：
1. 定义安全集：`DO ∈ [0.5, 8.0]`, `TP < 10`, `TN < 20`
2. 设计 CBF 函数 `B(x)` 满足 `Ḃ ≥ -γB`
3. QP 拦截器以 1kHz 频率运行
4. 红蓝对抗（最坏情况进水扰动）

**验收标准**：
- [ ] CBF 违规率 = 0%
- [ ] QP 平均求解时间 < 1ms
- [ ] 对抗测试 100 轮全通过

---

### Task 3.3 多目标优化与 Pareto 前沿

**步骤**：
1. 扫描权重 `λ ∈ [0.1, 1.0]`，生成 Pareto 前沿
2. 选择工业最优点（TP 优先前提下能耗最低）
3. 固化策略为 `policy_optimal.pt`

**验收标准**：
- [ ] Pareto 前沿清晰、单调
- [ ] 最优策略可复现（5 次训练方差 < 5%）

---

### Task 3.4 知识蒸馏压缩

**涉及文件**：
- [training/knowledge_distillation.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/training/knowledge_distillation.py)

**步骤**：
1. Teacher：完整 30M 模型
2. Student：压缩到 25%（≈7.5M）
3. 蒸馏损失：`L = T² * CE(logits/T, targets/T) + α * L_feature`
4. 保存 `checkpoints/distilled/student.pt`

**验收标准**：
- [ ] 压缩比 ≥ 4x
- [ ] 精度损失 < 2%
- [ ] 推理速度提升 ≥ 2x

---

## 六、Phase 4：部署与上线（1.5 周）

### Task 4.1 ONNX 导出与验证

**涉及文件**：
- [training/export_onnx.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/training/export_onnx.py)

**步骤**：
```bash
python -m training.export_onnx \
    --checkpoint checkpoints/distilled/student.pt \
    --output_dir onnx_exports \
    --opset_version 14
python -c "import onnx; onnx.checker.check_model(onnx.load('onnx_exports/biop_worldmodel.onnx'))"
```

**验收标准**：
- [ ] ONNX 模型结构合法
- [ ] ONNX vs PyTorch 推理误差 < 1e-5
- [ ] ONNX Runtime 延迟 < 50ms（CPU）

---

### Task 4.2 Rust 边缘推理引擎

**涉及文件**：
- [edge_deployment/Cargo.toml](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/edge_deployment/Cargo.toml)
- [edge_deployment/src/inference_engine.rs](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/edge_deployment/src/inference_engine.rs)

**步骤**：
```bash
cd edge_deployment && cargo build --release
```

**验收标准**：
- [ ] 编译通过，无 warning
- [ ] 单次推理延迟 < 50ms
- [ ] 内存占用 < 200MB

---

### Task 4.3 OPC-UA 桥接与影子模式

**涉及文件**：
- [edge_deployment/src/opcua_bridge.rs](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/edge_deployment/src/opcua_bridge.rs)
- [edge_deployment/src/shadow_mode.rs](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/edge_deployment/src/shadow_mode.rs)

**步骤**：
1. 配置 OPC-UA endpoint
2. 影子模式：AI 建议 vs 现场 DCS 建议并行记录，不直接执行
3. 记录至少 7 天数据
4. 偏差分析报告

**验收标准**：
- [ ] OPC-UA 连接稳定（断线率 < 0.1%）
- [ ] 影子模式 AI 建议与人工偏差 ≤ 10%
- [ ] 无异常报警

---

### Task 4.4 灰度上线与监控

**步骤**：
1. 10% → 50% → 100% 流量灰度
2. 关键指标监控：TP / 能耗 / CBF 违规率 / 模型延迟
3. 回滚机制：一键切回 DCS
4. 生成上线报告 `reports/go_live_report.md`

**验收标准**：
- [ ] 灰度各阶段无 P0 事故
- [ ] 全量上线 7 天稳定
- [ ] 指标达成项目 KPI

---

## 七、风险管理矩阵

| 风险 | 影响 | 概率 | 缓解方案 |
|------|------|------|----------|
| 训练长时间不收敛 | 高 | 中 | 降 LR / 去 PINN warmup / 换小模型 |
| 物理损失与数据损失冲突 | 高 | 中 | 分阶段权重调度 + Pareto 搜索 |
| 数据量不足 | 中 | 中 | 合成数据 + 数据增强 + 迁移学习 |
| RL 奖励稀疏 | 中 | 中 | 奖励塑形 + Hindsight Experience Replay |
| 边缘设备算力不足 | 中 | 低 | 蒸馏 + 量化 + ONNX Runtime Mobile |
| 现场网络抖动 | 低 | 中 | 本地缓存 + 断线重连 + 降级模式 |

---

## 八、交付物清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 项目全景说明 | [PROJECT_OVERVIEW.md](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/PROJECT_OVERVIEW.md) | ✅ 已完成 |
| 分任务计划书 | [PROJECT_PLAN.md](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/PROJECT_PLAN.md) | ✅ 本文件 |
| 参数估算工具 | [tools/estimate_params.py](file:///e:/泽浮蠃/PWMAI/BioP_Causal_WorldModel_V2/tools/estimate_params.py) | ✅ 已完成 |
| 30M 配置锁版 | `configs/model_config.yaml`（Phase 0） | ⏳ 待执行 |
| 训练收敛报告 | `logs/training_phase1_full_report.md`（Phase 1） | ⏳ 待执行 |
| 因果结构图 | `docs/causal_graph.svg`（Phase 2） | ⏳ 待执行 |
| ONNX 模型 | `onnx_exports/biop_worldmodel.onnx`（Phase 4） | ⏳ 待执行 |
| 边缘控制器 | `edge_deployment/target/release/biop_edge_controller`（Phase 4） | ⏳ 待执行 |
| 上线报告 | `reports/go_live_report.md`（Phase 4） | ⏳ 待执行 |

---

## 九、立即行动清单（按顺序执行）

1. **[今天]** 运行 `python tools/estimate_params.py` 确认当前参数规模
2. **[今天]** 修改 `configs/model_config.yaml` 锁定 896/896 配置
3. **[明天]** 修改 `models/expanded_ncde.py` 默认参数
4. **[明天]** 运行冒烟测试验证
5. **[本周]** 启动 Phase 1 小样本训练（Task 1.2）
6. **[连续 2 周]** 每日监控 `logs/training_phase1_smoke.csv`，发现 NaN 立即回滚
7. **[第 3 周起]** Phase 2、3、4 按门禁推进

---

## 十、成功标志（Definition of Done）

当且仅当以下全部达成时，项目视为成功：

- ✅ **模型参数** ≈ 30M（±10%）
- ✅ **NCDE 收敛** Val Loss < 0.05，零 NaN
- ✅ **TP 预测** NRMSE < 10%
- ✅ **物质守恒** 误差 < 1e-4
- ✅ **TP 达标** 出水 TP < 0.5 mg/L（95% 时间）
- ✅ **能耗下降** ≥ 15%
- ✅ **CBF 零违规** 红蓝对抗通过
- ✅ **边缘延迟** < 50ms
- ✅ **影子模式稳定** 7 天
- ✅ **代码仓库** GitHub 完整可复现
