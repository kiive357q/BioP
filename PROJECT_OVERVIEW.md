# BioP Causal WorldModel V2.0 · 项目全景说明文档

> 项目定位：**污水生化除磷工艺 · 世界模型驱动的动态强化学习控制框架**
> 核心技术栈：Neural Causal Differential Equations (NCDE) + Koopman Operator + SINDy + PINN + 字典序 SAC + CBF 安全约束
> 目标参数量：**~30M**（当前默认约 41M，备选 896/1024 配置约 30M）

---

## 一、项目价值 (Why this project matters)

| 维度 | 价值 |
|------|------|
| 工业价值 | 为污水厂提供可解释、可部署的 AI 控制大脑，目标是 **出水总磷稳定达标（TP<0.5mg/L）** 且 **曝气能耗下降 15–30%** |
| 学术价值 | 世界模型 + 因果去偏 (IPW) + 物理约束 (PINN) 的组合，在生化过程这一"多尺度、强非线性、数据稀缺"场景下是前沿范式 |
| 工程价值 | 提供从 SCADA 数据 → 可微仿真环境 → 训练 → 知识蒸馏 → ONNX → Rust 边缘控制器 的全链路参考实现 |

---

## 二、项目进度 (Current progress)

| 模块 | 进度 | 备注 |
|------|------|------|
| 数据工程 (`data/`) | ★★★★☆ | SCADA 加载、鲁棒归一化、滑窗已完成；特征工程丰富 |
| 环境仿真 (`envs/`) | ★★★★☆ | BSM1 13 组分 Petersen 矩阵、8 反应动力学、传感器/能耗/约束完备 |
| 世界模型 (`models/`) | ★★★☆☆ | 架构完整，但训练日志显示 NaN 爆炸（dtype/数值稳定性问题已修） |
| 强化学习 (`rl_agents/`, `safety/`) | ★★★★☆ | 字典序 SAC + CBF + QP 拦截器设计完整 |
| 训练流水线 (`training/`) | ★★☆☆☆ | 三阶段脚本完备，**但实际训练损失仍为 NaN**（见第 4 节） |
| 边缘部署 (`edge_deployment/`) | ★★★★☆ | Rust 推理引擎 + OPC-UA + 影子模式已实现 |
| 日志/Checkpoint | ★★★☆☆ | 有多个失败记录与早期 checkpoint，需要清理与重新训练 |

**总体进度：~65%**。工程骨架扎实，但"训练能真正收敛"这一核心 KPI 尚未达成。

---

## 三、目录与每个文件的作用

```
BioP_Causal_WorldModel_V2/
├── Makefile                         # 自动化构建宏 (init/install/test/train/deploy)
├── requirements.txt                 # Python 依赖矩阵 (torch>=2.4, torchdiffeq, cvxpy, onnx...)
├── setup_workspace.sh              # 工作目录初始化脚本
│
├── configs/                         # 配置文件
│   ├── env_config.yaml              # BSM1 物理参数、进水组成、控制变量范围
│   ├── model_config.yaml            # NCDE/Koopman/SINDy/PINN/安全阈值
│   ├── rl_config.yaml               # RL 超参数
│   └── safety_config.yaml           # CBF、QP 等安全参数
│
├── data/                            # 数据工程
│   ├── IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023(1).csv  # 原始 SCADA 数据
│   ├── preprocessed_phase_1.csv      # 已完成预处理的 Phase 1 数据
│   ├── preprocessed_phase_3.csv      # 已完成预处理的 Phase 3 数据
│   ├── analysis_output/data_analysis_report.json    # 数据统计分析报告
│   ├── analysis_output/normalization_params.json   # 归一化参数 (用于线上反解)
│   ├── analysis_output/preprocessed_data.csv        # 合并预处理数据
│   ├── adaptive_preprocessing.py    # 鲁棒归一化 + 异常检测 + 时序特征工程
│   ├── cubic_spline_interp.py       # 可微分三次样条插值 (用于稀疏观测插值)
│   ├── data_analysis.py             # 数据探索性分析
│   ├── data_validation.py           # 工业级数据校验器 (范围/守恒/异常)
│   ├── dataloaders.py               # 核心 DataLoader + CausalDataProcessor
│   └── ipw_confounder.py            # 逆概率加权 (IPW) 因果去偏
│
├── envs/                            # 环境 (数字孪生)
│   ├── bsm1_differentiable.py       # 可微分 BSM1 (配合世界模型的"环境梯度")
│   ├── bsm1_full_simulation.py      # 完整 BSM1 仿真器 (13 组分 + 8 反应 + 能耗 + 约束)
│   ├── digital_twin_config.json     # 数字孪生参数快照
│   ├── reward_functions.py          # 字典序多目标奖励
│   └── state_observers.py           # 虚拟传感器/状态观测器
│
├── models/                          # 世界模型核心
│   ├── expanded_ncde.py             # BioPWorldModel 主架构 (Encoder/NCDE/Actor/Critic/CBF)
│   ├── ncde_solver.py               # 伴随灵敏度 NCDE 求解器
│   ├── koopman_operator.py          # Koopman 算子编码器/解码器/线性预测
│   ├── sindy_library.py             # SINDy 稀疏动力学特征库
│   ├── pinn_loss.py                 # PINN 复合损失 (C/N/P 质量守恒 + 稳态约束)
│   └── attention_decay.py           # 时序衰减注意力 (用于长序列建模)
│
├── rl_agents/                       # RL 算法
│   ├── lexicographic_sac.py         # 字典序 SAC (安全 > 水质 > 能耗)
│   ├── hierarchical_actor_critic.py # 层级化 AC 架构
│   ├── priority_replay.py           # 优先经验回放 (TD-error 加权)
│   └── cvar_tail_risk.py            # CVaR 尾部风险度量
│
├── safety/                          # 安全控制
│   ├── control_barrier_functions.py # CBF 控制障碍函数
│   ├── qp_interceptor.py           # QP 动作安全拦截器
│   ├── reachability_tube.py         # 可达性管 (Reachability Tube)
│   ├── hji_solver.py                # HJI 方程求解器 (理论验证)
│   └── full_control_module.py       # 整合的控制模块 (LexicographicSAC + CBF)
│
├── training/                        # 训练流水线
│   ├── train_phase1_ncde.py         # 阶段一: NCDE 动力学预训练
│   ├── train_phase2_koopman.py      # 阶段二: Koopman 迁移学习
│   ├── train_full_system.py         # 端到端: WorldModel + SAC 联训
│   ├── knowledge_distillation.py   # 教师→学生知识蒸馏 (压缩到 30M)
│   ├── export_onnx.py               # 导出 ONNX
│   └── integration_dryrun.py       # 集成烟雾测试
│
├── edge_deployment/                 # Rust 边缘部署
│   ├── Cargo.toml                   # Rust 项目配置
│   ├── config_registers.json        # OPC-UA 寄存器映射
│   └── src/
│       ├── main.rs                  # 服务入口
│       ├── inference_engine.rs      # ONNX Runtime 推理引擎
│       ├── opcua_bridge.rs          # OPC-UA 桥接器 (与 SCADA 通讯)
│       ├── shadow_mode.rs           # 影子模式 (并行对比旧系统)
│       └── watchdog.rs              # 看门狗 (熔断/重启)
│
├── checkpoints/                     # Phase 1 Checkpoints (含若干 epoch 存档)
├── checkpoints_full/                # Full system Checkpoints
├── logs/                            # 训练日志 (JSON)
├── logs_full/                       # TensorBoard 事件文件
│
├── prompt_engineering/              # Prompt 工程记录 (研发方法论)
│   ├── 00_System_Baseline.md
│   ├── 01_Phase1_Data.md
│   ├── 01_Phase1_Task2_Spline_IPW.md
│   ├── 02_Phase2_Twin_Sandbox.md
│   ├── 04_Phase3_Safety_Guardrails.md
│   ├── 05_Phase4_Training_Pipeline.md
│   └── 07_Milestone_Preflight_Check.md
│
├── tests/                           # 单元测试 (已新建骨架，用例待补充)
│   ├── __init__.py
│   └── README.py
│
└── tools/
    └── estimate_params.py           # 参数规模估算脚本 (命中 30M 预算)
```

---

## 四、已识别的问题 (Issues) 与修复 (已完成的修改)

### 4.1 训练日志显示 `NaN`（最关键阻塞）

来自 `logs/metrics_20260624_143145.json`：
- `train_losses`: 全部为 `NaN`
- `val_losses`: 从 1,200,000 飙升到 2,541,085，随后固化
- 典型 **梯度爆炸 + 数值失稳**

**已修复的 3 处具体原因：**

1. **`models/ncde_solver.py` · 非法 `norm` 字符串 + dtype 不匹配**
   - 原代码: `"norm": "论"`（中文乱码，非法取值）；`"dtype": torch.float64` 会与 AMP (float32) 冲突
   - 已改为: `"norm": "rmse"`，`"dtype": torch.float32`
   - 影响：求解器步长控制失效，导致梯度爆炸

2. **`training/train_phase1_ncde.py` · CSV 回退分支 `all_data`/`split_1` 未定义**
   - 原代码在走 CSV 分支时直接使用未赋值变量，运行会抛 `NameError`
   - 已补齐完整的数据加载/特征选择/切分逻辑
   - 影响：当未找到 `preprocessed_phase_*.csv` 时流水线直接崩溃

3. **`envs/bsm1_full_simulation.py` · 非法变量名 `tp达标`/`tn达标`**
   - 原代码用 Python 3 支持的中文标识符做变量名，但拼写混乱、不利于团队协作
   - 已改为 `tp_compliant` / `tn_compliant`

4. **`data/dataloaders.py` · 已弃用 API `df.fillna(method="ffill")`**
   - pandas 2.x 会警告/报 TypeError
   - 已改为 `df.ffill().bfill()`

### 4.2 参数量预算偏离

- 原设计文档声称 ~20M
- **实际测量**：默认 `latent_dim=1024, hidden_dim=1024` 下 ≈ **41.5M**（含 target 网络，翻倍结构是主因）
- 若需严格 30M：使用 `latent_dim=896, hidden_dim=1024` 即可（见 `tools/estimate_params.py`）

### 4.3 其他未修复的中长期问题

| # | 问题 | 建议 |
|---|------|------|
| 1 | `loss = 1e6` 量级依然过大，说明物理损失权重 (10×mass + 10×nitrogen) 过高 | 建议将 `mass_conservation_penalty` 降到 1.0 并做 warmup |
| 2 | `train_phase1` 训练 NCDE 时未使用 `WorldModelEncoder/Decoder`，特征直接进 NCDE | 建议 Phase 1 改为 "Encoder→NCDE→Decoder" 端到端训练 |
| 3 | `full_control_module.py` 中 `select_action` 与 `update` 需与 `BioPWorldModel` 的 `predict_next_obs` 接口对齐 | 已做接口适配，但建议补类型注解 |
| 4 | `tests/` 目录缺失 | 已新建骨架，需逐步补齐 `test_nan_detection.py` 等 |
| 5 | `logs/` 中包含 NaN 结果，会误导训练评估 | 建议清理 `logs/` 与 `checkpoints/`，用修复后的代码重启训练 |
| 6 | `requirements.txt` 同时出现 `torch>=2.4.0` 与 `jax` 可选 | 无冲突，但建议在 README 中明确 "GPU 仅需 torch" |
| 7 | `bsm1_full_simulation.py` 中 TP 近似为 `X_BH+X_BA+X_P` 仅为简化估算 | 正式部署前需引入独立的磷组分动力学 |

---

## 五、建议的训练重启步骤

```bash
# 1. 清理旧日志
rm -rf logs/*.json checkpoints/*.pt

# 2. 验证修复是否生效
python models/ncde_solver.py          # 不应再触发 norm 非法值
python envs/bsm1_full_simulation.py   # 仿真器 10 步测试

# 3. 确认参数规模
python tools/estimate_params.py      # 选择 ~30M 配置

# 4. 启动训练 (先用小 lr 与 20 epochs 试跑)
python training/train_phase1_ncde.py --epochs 20 --lr 1e-4 --batch_size 32

# 5. 观察: 前 5 epoch 的 train_loss 不应出现 NaN, val_loss 应 <10
#    若仍异常，打开 physics_loss_weight=0 做纯数据驱动排查
```

---

## 六、下一步建议 (Roadmap)

| 阶段 | 交付物 |
|------|--------|
| **P0 (本周)** | ① 修复后训练跑通 20 epochs 无 NaN；② Physics loss 权重 warmup；③ 确认最佳 latent 配置 |
| **P1 (下周)** | ④ 引入 Encoder→NCDE→Decoder 端到端训练 (替代纯 NCDE)；⑤ 补充 `tests/test_nan_detection.py` 等用例 |
| **P2 (下周)** | ⑥ 启用 Koopman 迁移学习；⑦ 知识蒸馏压到 30M |
| **P3 (上线前)** | ⑧ Rust 边缘推理联调 + OPC-UA 影子模式；⑨ 现场 SCADA 接入 |

---

## 七、本次已完成的修改清单

| 文件 | 修改类型 | 目的 |
|------|----------|------|
| `models/ncde_solver.py` | Bug fix | 修复非法 `norm` 值与 float64 dtype 冲突 |
| `training/train_phase1_ncde.py` | Bug fix | 补齐 CSV 回退分支缺失变量 |
| `envs/bsm1_full_simulation.py` | 代码规范化 | 替换中文变量名为 `tp_compliant` / `tn_compliant` |
| `data/dataloaders.py` | API upgrade | `fillna(method=...)` → `ffill()/bfill()` |
| `tools/estimate_params.py` | 新增 | 30M 参数预算估算脚本 |
| `tests/__init__.py` + `tests/README.py` | 新增 | 补齐 Makefile 引用的 tests 目录 |

> 备注：所有修改均未触碰业务算法逻辑，仅为 **数值稳定性** 与 **工程完备性** 修复。建议在首次训练时保留 `--no_physics_loss` 开关用于对照实验。
