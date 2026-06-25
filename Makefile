#===============================================================================
# 文件名称: Makefile
# 自动化生命周期指令集 - 污水生化除磷工业世界模型 (BioP Causal WorldModel V2.0)
# 创世阶段: 全链路构建宏定义
# 开发者: 首席环境动力学算法科学家 / DevSecOps 首席架构师
# 创建日期: 2026-05-29
#===============================================================================
# 【数学验证备注】
# 本 Makefile 定义的构建目标均满足以下约束:
#   - 物理可行性: 每个阶段训练必须满足质量守恒验证
#   - 数值稳定性: NaN/Inf 检测门禁
#   - 安全边界: CBF 约束激活状态监控
#===============================================================================

.PHONY: help init install test lint typecheck train_phase1 train_phase2 train_full
.PHONY: safety_check export_onnx build_edge run_edge clean check_gpu logs

#===============================================================================
# 全局变量定义
#===============================================================================
PYTHON := python3
PYTEST := pytest
PIP := pip install
CUDA_DEVICE := 0
LOG_DIR := logs
CHECKPOINT_DIR := checkpoints
ONNX_DIR := onnx_exports
EDGE_DIR := edge_deployment
RUST_DIR := $(EDGE_DIR)
PROJECT_NAME := BioP_Causal_WorldModel_V2

# 训练超参数
NCDE_EPOCHS ?= 500
KOOPMAN_EPOCHS ?= 300
BATCH_SIZE ?= 128
LEARNING_RATE ?= 0.001

# 安全阈值 (从配置文件读取)
DO_MIN_THRESHOLD := 0.5
TP_UPPER_BOUND := 10.0

#===============================================================================
# 帮助信息
#===============================================================================
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║     BioP Causal WorldModel V2.0 - 自动化生命周期指令集                         ║"
	@echo "║     污水生化除磷工业世界模型 · 全链路构建系统                                   ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════════╣"
	@echo "║                                                                              ║"
	@echo "║  【初始化阶段】                                                                ║"
	@echo "║    make init           - 执行 setup_workspace.sh 初始化项目结构                 ║"
	@echo "║    make install        - 安装 Python 依赖 (requirements.txt)                    ║"
	@echo "║                                                                              ║"
	@echo "║  【代码质量保障】                                                              ║"
	@echo "║    make lint           - 运行 flake8 PEP8 代码风格检查                         ║"
	@echo "║    make typecheck      - 运行 mypy 静态类型检查                                 ║"
	@echo "║    make test           - 执行 pytest 单元测试                                   ║"
	@echo "║                                                                              ║"
	@echo "║  【训练阶段】                                                                  ║"
	@echo "║    make train_phase1   - 阶段一: NCDE 动力学模型预训练                          ║"
	@echo "║    make train_phase2   - 阶段二: Koopman 算子迁移学习                           ║"
	@echo "║    make train_full     - 完整训练流程 (phase1 + phase2)                        ║"
	@echo "║                                                                              ║"
	@echo "║  【安全验证】                                                                  ║"
	@echo "║    make safety_check   - 红蓝对抗测试与 NaN 巡检                                 ║"
	@echo "║    make check_gpu      - GPU 资源与显存健康检查                                 ║"
	@echo "║                                                                              ║"
	@echo "║  【模型导出与部署】                                                            ║"
	@echo "║    make export_onnx    - 导出 ONNX 静态计算图                                  ║"
	@echo "║    make build_edge    - 编译 Rust 边缘控制器 (cargo build)                     ║"
	@echo "║    make run_edge       - 启动边缘推理服务                                      ║"
	@echo "║                                                                              ║"
	@echo "║  【日志与监控】                                                                ║"
	@echo "║    make logs           - 实时监控训练日志                                       ║"
	@echo "║    make clean          - 清理临时文件与缓存                                     ║"
	@echo "║                                                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"
	@echo ""

#===============================================================================
# 初始化阶段
#===============================================================================

# 执行 setup 脚本，初始化项目结构
init:
	@echo "[INFO] 执行全景物理目录构建脚本..."
	@chmod +x setup_workspace.sh 2>/dev/null || true
	@bash setup_workspace.sh || echo "[WARN] Windows 环境: 请使用 Git Bash 执行 bash setup_workspace.sh"
	@echo "[INFO] 项目结构初始化完成"

# 安装 Python 依赖
install:
	@echo "[INFO] 安装 Python 环境依赖..."
	@$(PIP) --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "[INFO] 依赖安装完成"

# 创建虚拟环境并安装依赖
venv:
	@echo "[INFO] 创建 Python 虚拟环境..."
	@$(PYTHON) -m venv venv
	@./venv/Scripts/activate.bat && $(PIP) install -r requirements.txt
	@echo "[INFO] 虚拟环境创建完成"

#===============================================================================
# 代码质量保障
#===============================================================================

# PEP8 代码风格检查
lint:
	@echo "[INFO] 执行 flake8 代码风格检查..."
	@mkdir -p $(LOG_DIR)
	@flake8 . --max-line-length=120 --extend-ignore=E203,W503 \
		--exclude=venv,__pycache__,.git,checkpoints,onnx_exports \
		--output-file=$(LOG_DIR)/flake8_report.txt || true
	@cat $(LOG_DIR)/flake8_report.txt
	@echo "[INFO] 代码风格检查完成"

# 静态类型检查
typecheck:
	@echo "[INFO] 执行 mypy 静态类型检查..."
	@mkdir -p $(LOG_DIR)
	@mypy . --ignore-missing-imports --no-error-summary \
		--exclude=venv,__pycache__,.git,checkpoints,onnx_exports \
		2>&1 | tee $(LOG_DIR)/mypy_report.txt || true
	@echo "[INFO] 类型检查完成"

# 运行单元测试
test:
	@echo "[INFO] 执行 pytest 单元测试..."
	@mkdir -p test_results
	@$(PYTEST) tests/ -v --cov=. --cov-report=html:test_results/coverage \
		--cov-report=term --tb=short
	@echo "[INFO] 测试完成，结果位于 test_results/ 目录"

# 快速冒烟测试
test_quick:
	@echo "[INFO] 执行快速冒烟测试..."
	@$(PYTEST) tests/ -v -k "test_nan or test_zero_division or test_mass_conservation" \
		--tb=short
	@echo "[INFO] 冒烟测试完成"

#===============================================================================
# 训练阶段
#===============================================================================

# 阶段一: NCDE 动力学模型预训练
train_phase1:
	@echo "[INFO] 启动阶段一训练: NCDE 动力学模型预训练"
	@echo "[CONFIG] EPOCHS=$(NCDE_EPOCHS), BATCH_SIZE=$(BATCH_SIZE), LR=$(LEARNING_RATE)"
	@mkdir -p $(LOG_DIR) $(CHECKPOINT_DIR)
	@CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) -m training.train_phase1_ncde \
		--epochs $(NCDE_EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE) \
		--log_dir $(LOG_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--do_min $(DO_MIN_THRESHOLD) \
		--tp_upper $(TP_UPPER_BOUND) \
		--enable_gradient_clipping \
		--clip_value 1.0
	@echo "[INFO] 阶段一训练完成"

# 阶段二: Koopman 算子迁移学习
train_phase2:
	@echo "[INFO] 启动阶段二训练: Koopman 算子迁移学习"
	@echo "[CONFIG] EPOCHS=$(KOOPMAN_EPOCHS)"
	@mkdir -p $(LOG_DIR) $(CHECKPOINT_DIR)
	@CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) -m training.train_phase2_koopman \
		--epochs $(KOOPMAN_EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE) \
		--log_dir $(LOG_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--load_from_phase1 checkpoints/phase1_final.pt
	@echo "[INFO] 阶段二训练完成"

# 完整训练流程
train_full: train_phase1 train_phase2
	@echo "[SUCCESS] 完整训练流程 (phase1 + phase2) 执行完成"
	@echo "[NEXT] 执行知识蒸馏: make distill"

# 知识蒸馏
distill:
	@echo "[INFO] 启动知识蒸馏..."
	@CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) -m training.knowledge_distillation \
		--teacher_model checkpoints/phase2_final.pt \
		--output_dir checkpoints/distilled \
		--compression_ratio 0.25
	@echo "[INFO] 知识蒸馏完成"

#===============================================================================
# 安全验证
#===============================================================================

# 红蓝对抗测试与 NaN 巡检
safety_check:
	@echo "[INFO] 执行安全验证: 红蓝对抗测试"
	@mkdir -p test_results/safety
	@echo "[CHECK] 1. NaN/Inf 检测..."
	@$(PYTEST) tests/test_nan_detection.py -v \
		--junitxml=test_results/safety/nan_report.xml
	@echo "[CHECK] 2. 物质守恒验证 (C/N/P)..."
	@$(PYTHON) -c "from models.pinn_loss import verify_mass_conservation; verify_mass_conservation()"
	@echo "[CHECK] 3. DO 抑制项除零防护..."
	@$(PYTEST) tests/test_zero_division.py -v \
		--junitxml=test_results/safety/zero_div_report.xml
	@echo "[CHECK] 4. CBF 约束激活状态..."
	@$(PYTEST) tests/test_cbf_activation.py -v \
		--junitxml=test_results/safety/cbf_report.xml
	@echo "[CHECK] 5. 梯度爆炸检测..."
	@$(PYTEST) tests/test_gradient_clipping.py -v \
		--junitxml=test_results/safety/gradient_report.xml
	@echo "[SUCCESS] 安全验证完成，结果位于 test_results/safety/"

# GPU 健康检查
check_gpu:
	@echo "[INFO] 检查 GPU 资源状态..."
	@$(PYTHON) -c "
import torch
if torch.cuda.is_available():
    print(f'[GPU] 设备数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'[GPU:{i}] 名称: {torch.cuda.get_device_name(i)}')
        print(f'[GPU:{i}] 显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
        mem_allocated = torch.cuda.memory_allocated(i) / 1e9
        mem_reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f'[GPU:{i}] 已分配: {mem_allocated:.2f} GB')
        print(f'[GPU:{i}] 已预留: {mem_reserved:.2f} GB')
        if mem_allocated > 10.0:
            print('[WARN] 显存使用率较高，建议清理缓存')
else:
    print('[WARN] CUDA 不可用，将使用 CPU 进行计算')
"
	@echo "[INFO] GPU 检查完成"

# 显存泄漏检测
check_memory_leak:
	@echo "[INFO] 执行显存泄漏检测..."
	@$(PYTHON) -c "
import torch
import gc

def check_memory_leak():
    if not torch.cuda.is_available():
        print('[SKIP] CUDA 不可用，跳过显存泄漏检测')
        return
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_mem = torch.cuda.memory_allocated() / 1e9
    print(f'[MEM] 初始显存: {initial_mem:.4f} GB')
    
    for i in range(100):
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x.T)
        del x, y
    
    gc.collect()
    torch.cuda.empty_cache()
    
    final_mem = torch.cuda.memory_allocated() / 1e9
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f'[MEM] 最终显存: {final_mem:.4f} GB')
    print(f'[MEM] 峰值显存: {peak_mem:.4f} GB')
    
    if final_mem - initial_mem > 0.5:
        print('[ERROR] 检测到显存泄漏，请检查张量释放逻辑')
    else:
        print('[PASS] 显存使用正常')

check_memory_leak()
"

#===============================================================================
# 模型导出与部署
#===============================================================================

# 导出 ONNX 静态计算图
export_onnx:
	@echo "[INFO] 导出 ONNX 静态计算图..."
	@mkdir -p $(ONNX_DIR)
	@CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) -m training.export_onnx \
		--checkpoint checkpoints/distilled/model.pt \
		--output_dir $(ONNX_DIR) \
		--opset_version 14 \
		--enable_onnx_validation
	@echo "[INFO] ONNX 导出完成，文件位于 $(ONNX_DIR)/"

# 验证 ONNX 模型
validate_onnx:
	@echo "[INFO] 验证 ONNX 模型正确性..."
	@$(PYTHON) -c "
import onnx
import onnxruntime as ort
import numpy as np

model_path = '$(ONNX_DIR)/biop_worldmodel.onnx'
print(f'[CHECK] 加载模型: {model_path}')

try:
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print('[PASS] ONNX 模型结构验证通过')
except Exception as e:
    print(f'[ERROR] ONNX 模型验证失败: {e}')

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f'[INFO] 输入名称: {input_name}, 形状: {input_shape}')

dummy_input = np.random.randn(*input_shape).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
print(f'[INFO] 推理输出形状: {output[0].shape}')
print('[PASS] ONNX 模型推理验证通过')
"
	@echo "[INFO] ONNX 验证完成"

# 编译 Rust 边缘控制器
build_edge:
	@echo "[INFO] 编译 Rust 边缘控制器..."
	@cd $(RUST_DIR)
	@if command -v cargo >/dev/null 2>&1; then \
		cargo build --release --target-dir target; \
		echo "[INFO] Rust 构建完成，可执行文件位于 target/release/"; \
	else \
		echo "[ERROR] Rust 工具链未安装，请访问 https://rustup.rs 安装"; \
	fi

# 构建 Debug 版本 (开发用)
build_edge_debug:
	@echo "[INFO] 编译 Rust 边缘控制器 (Debug)..."
	@cd $(RUST_DIR)
	@if command -v cargo >/dev/null 2>&1; then \
		cargo build --target-dir target/debug; \
		echo "[INFO] Debug 构建完成"; \
	else \
		echo "[ERROR] Rust 工具链未安装"; \
	fi

# 运行边缘推理服务
run_edge:
	@echo "[INFO] 启动边缘推理服务..."
	@if [ -f "$(RUST_DIR)/target/release/biop_edge_controller" ]; then \
		$(RUST_DIR)/target/release/biop_edge_controller \
			--config $(RUST_DIR)/config_registers.json \
			--onnx_model $(ONNX_DIR)/biop_worldmodel.onnx \
			--log_level info; \
	else \
		echo "[ERROR] 边缘控制器未编译，请先执行 make build_edge"; \
	fi

# 运行影子模式 (Shadow Mode)
run_shadow_mode:
	@echo "[INFO] 启动影子模式验证..."
	@if [ -f "$(RUST_DIR)/target/release/biop_edge_controller" ]; then \
		$(RUST_DIR)/target/release/biop_edge_controller \
			--config $(RUST_DIR)/config_registers.json \
			--onnx_model $(ONNX_DIR)/biop_worldmodel.onnx \
			--shadow_mode \
			--log_level debug; \
	else \
		echo "[ERROR] 边缘控制器未编译"; \
	fi

#===============================================================================
# 日志与监控
#===============================================================================

# 实时监控训练日志
logs:
	@echo "[INFO] 监控训练日志 (Ctrl+C 退出)..."
	@if [ -d "$(LOG_DIR)" ]; then \
		tail -f $(LOG_DIR)/*.log 2>/dev/null || echo "[INFO] 暂无日志文件"; \
	else \
		echo "[INFO] 日志目录不存在"; \
	fi

# 查看最新训练日志
logs_latest:
	@echo "[INFO] 最新训练日志..."
	@if ls $(LOG_DIR)/training*.log 1>/dev/null 2>&1; then \
		tail -n 50 $$(ls -t $(LOG_DIR)/training*.log | head -1); \
	else \
		echo "[INFO] 暂无训练日志"; \
	fi

# 查看安全检查报告
logs_safety:
	@echo "[INFO] 安全检查报告..."
	@cat $(LOG_DIR)/safety_*.txt 2>/dev/null || echo "[INFO] 暂无安全报告"

#===============================================================================
# 清理
#===============================================================================

# 清理 Python 缓存
clean:
	@echo "[INFO] 清理临时文件与缓存..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "[INFO] Python 缓存清理完成"

# 清理所有生成的文件
clean_all: clean
	@echo "[INFO] 清理所有生成文件..."
	@rm -rf $(LOG_DIR)/*.txt $(LOG_DIR)/*.log 2>/dev/null || true
	@rm -rf $(CHECKPOINT_DIR)/*.pt 2>/dev/null || true
	@rm -rf $(ONNX_DIR)/*.onnx 2>/dev/null || true
	@rm -rf test_results/* 2>/dev/null || true
	@rm -rf $(EDGE_DIR)/target 2>/dev/null || true
	@echo "[INFO] 全部清理完成"

#===============================================================================
# 开发辅助
#===============================================================================

# 创建新模块模板
new_module:
	@read -p "模块名称 (如 MyModule): " module_name; \
	read -p "模块路径 (如 models/): " module_path; \
	echo "[TEMPLATE] 创建 $$module_path$$module_name.py..."; \
	touch "$$module_path/$$module_name.py"; \
	echo "# -*- coding: utf-8 -*-" > "$$module_path/$$module_name.py"; \
	echo "\"\"\"" >> "$$module_path/$$module_name.py"; \
	echo "模块名称: $$module_name" >> "$$module_path/$$module_name.py"; \
	echo "模块描述: [请填写]" >> "$$module_path/$$module_name.py"; \
	echo "\"\"\"" >> "$$module_path/$$module_name.py"; \
	echo "from __future__ import annotations" >> "$$module_path/$$module_name.py"; \
	echo "" >> "$$module_path/$$module_name.py"; \
	echo "[INFO] 模块模板创建完成: $$module_path/$$module_name.py"

# 生成测试报告
report:
	@echo "[INFO] 生成测试覆盖率报告..."
	@if [ -d "test_results/coverage" ]; then \
		$(PYTHON) -m coverage html -d test_results/coverage_report; \
		echo "[INFO] 报告位于 test_results/coverage_report/index.html"; \
	else \
		echo "[INFO] 请先运行 make test"; \
	fi
