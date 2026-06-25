# 阶段三：HJI安全护栏与物理硬干预 (Phase 3: Safety Guardrails)

## 提示词溯源

- **归档日期**: 2026-05-29
- **适用阶段**: 阶段三 - 安全护栏
- **系统基线**: BioP Causal WorldModel V2.0
- **上下文锚定**: 首席环境动力学算法科学家 / 非线性控制理论专家

---

## [Context] 局部物理上下文 (Phase 3: Safety Guardrails)

### 目标文件 1: safety/control_barrier_functions.py (控制障碍函数定义)

### 目标文件 2: safety/qp_interceptor.py (毫秒级在线安全拦截器)

### 物理痛点

强化学习在探索时会不可避免地输出危险动作（如瞬间关停所有风机导致缺氧死泥，或加药过量导致二次污染）。我们不能用环境给予负反馈（Reward）这种"事后惩罚"来解决，因为物理伤害不可逆。

### 学术叙事

我们采用控制障碍函数（CBF, Control Barrier Functions）。在线路上拦截 RL 输出的标称动作（Nominal Action），通过求解极速二次规划（Quadratic Programming, QP），计算出距离标称动作最近、且严格保证状态不会跌出安全集 $\mathcal{C}$ 的安全干预动作（Safe Action）。

---

## [Execution] 核心执行逻辑

### WaterTreatmentCBF

编写类，接收状态变量 $x$。

定义至少三个刚性物理边界的 $h(x)$ 函数：

1. 溶解氧下限（DO $\ge$ 0.5 mg/L）
2. 氨氮瞬时超标危险线
3. 鼓风机物理频率变化率（防喘振）

要求返回 $h(x)$ 以及其沿动力学向量场的偏导数李导数（Lie Derivatives: $L_f h(x)$ 和 $L_g h(x)$）。这里你可以直接假设环境的偏导数可由阶段二的可微分算子 dz_dt 提供或通过张量自动求导（torch.autograd）获取。

### QPActionInterceptor

必须使用 cvxpy 库构建在线二次规划求解器。

核心数学重构: 目标函数为最小化 $||u - u_{nominal}||^2$。约束条件为 CBF 约束：$L_f h(x) + L_g h(x) u \ge -\gamma h(x)$，还要加入物理执行器边界约束 $u \in [u_{min}, u_{max}]$。

性能极限压榨: 为了满足边缘端 <5ms 的推断延迟要求，在 __init__ 中必须将所有的动态变量（如 $u_{nominal}$, $L_f h$, $L_g h$, $h$）声明为 cvxpy.Parameter，并将求解器配置为已编译结构。在 intercept 拦截方法中，只允许 Parameter.value 赋值和 .solve() 调用，绝对禁止在每次拦截时重新构建 cvxpy 问题图！

---

## [Constraints] 模块级绝对红线警告

### 禁止在线编译 (NO RECOMPILATION)

如 [Execution 3] 所述，如果你在 qp_interceptor.py 的前向拦截循环中出现任何 cp.Problem(objective, constraints) 的重新定义，你将被判定为致命性能故障！必须使用预编译参数化形式。

### 防除零刚性熔断 (Critical)

在计算 CBF 的李导数或提取梯度时，若涉及任何分式计算，必须一如既往地加入 torch.clamp(x, min=1e-5) 防止张量崩溃。

### 无缝桥接 RL 动作

拦截器接口 intercept 必须能直接吃进形状为 (Batch, Action_Dim) 的 PyTorch 张量，在内部高效转为 numpy 给求解器（若 Batch>1 可暂用高效并行处理或批处理 QP），最后返回处理完的安全 PyTorch 张量。

---

## 控制障碍函数理论

### CBF数学定义

对于系统 $\dot{x} = f(x) + g(x)u$，安全集定义为：

$$\mathcal{C} = \{x \in \mathbb{R}^n : h(x) \geq 0\}$$

其中 $h(x)$ 为障碍函数，满足：

- $h(x)$ 连续可微
- $\mathcal{C}$ 为超集 $\{x : h(x) \geq 0\}$
- $\partial \mathcal{C}$ 为 $\{x : h(x) = 0\}$

### CBF条件（HJI可行性）

存在相对阶为1的CBF $h(x)$，当且仅当存在 $\gamma > 0$ 使得：

$$\sup_{u \in \mathcal{U}} [L_f h(x) + L_g h(x) u + \gamma h(x)] \geq 0, \quad \forall x \in \mathcal{C}$$

### 指数型CBF约束

$$L_f h(x) + L_g h(x) u \geq -\gamma h(x)$$

其中 $\gamma > 0$ 为约束收紧速率。

### 李导数定义

$$L_f h(x) = \frac{\partial h}{\partial x} f(x)$$
$$L_g h(x) = \frac{\partial h}{\partial x} g(x)$$

### QP拦截器数学

$$\min_{u \in \mathbb{R}^m} \quad \|u - u_{nominal}\|^2$$

$$\text{s.t.} \quad L_f h_i(x) + L_g h_i(x) u \geq -\gamma_i h_i(x), \quad \forall i$$

$$u_{min} \leq u \leq u_{max}$$

---

## [Implementation] 实现要点

### control_barrier_functions.py

- WaterTreatmentCBF类
- 三个CBF定义：DO下限、氨氮超标、频率变化率
- 返回h(x), L_f h, L_g h
- 支持torch.autograd自动求导

### qp_interceptor.py

- QPActionInterceptor类
- cvxpy预编译参数化结构
- intercept()只做Parameter赋值和solve()
- 毫秒级延迟要求(<5ms)
- PyTorch张量无缝转换

---

## [Testing] 测试入口规范

```python
if __name__ == "__main__":
    # Mock危险动作u_nominal
    # 调用拦截器
    # 打印原始危险动作 vs 安全动作对比
    # 打印耗时监测
```

---

**版本**: V2.0-Phase3-SafetyGuardrails
**制定日期**: 2026-05-29
**适用范围**: 安全护栏模块 - HJI/CBF/QP拦截
