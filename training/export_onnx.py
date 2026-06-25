# -*- coding: utf-8 -*-
# export_onnx.py
# ONNX 静态计算图导出，支持 Rust 推理引擎
# 创建时间: 2026-05-29
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from typing import Optional


class ONNXExporter:
    """
    ONNX 导出器
    
    【功能】
    1. 验证模型可导出性
    2. 导出静态计算图
    3. 验证导出结果正确性
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "onnx_exports"
    ) -> None:
        self.model = model
        self.model.eval()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        input_shape: tuple,
        output_name: str = "biop_worldmodel.onnx",
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None
    ) -> Path:
        """
        导出 ONNX 模型
        
        参数:
            input_shape: 输入形状 (batch, seq_len, features)
            output_name: 输出文件名
            opset_version: ONNX opset 版本
            dynamic_axes: 动态轴配置
            
        返回:
            导出文件路径
        """
        example_input = torch.randn(*input_shape)
        
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            self.model,
            example_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_modules_as_functions=False
        )
        
        print(f"[INFO] ONNX 模型已导出: {output_path}")
        
        self.validate(output_path, example_input)
        
        return output_path
    
    def validate(self, onnx_path: Path, example_input: torch.Tensor) -> bool:
        """
        验证 ONNX 模型
        
        参数:
            onnx_path: ONNX 文件路径
            example_input: 示例输入
            
        返回:
            验证是否通过
        """
        import onnx
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("[PASS] ONNX 模型结构验证通过")
        except Exception as e:
            print(f"[FAIL] ONNX 验证失败: {e}")
            return False
        
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(onnx_path))
            
            torch_output = self.model(example_input)
            
            ort_input = {session.get_inputs()[0].name: example_input.numpy()}
            ort_output = session.run(None, ort_input)[0]
            
            if isinstance(torch_output, tuple):
                torch_output = torch_output[0]
            
            max_diff = abs(torch_output.detach().numpy() - ort_output).max()
            print(f"[INFO] PyTorch-ONNX 最大差异: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("[PASS] PyTorch-ONNX 推理一致性验证通过")
                return True
            else:
                print("[WARN] PyTorch-ONNX 推理差异较大")
                return False
        
        except ImportError:
            print("[WARN] onnxruntime 未安装，跳过推理验证")
            return True


def export_model(
    checkpoint_path: str,
    output_dir: str = "onnx_exports",
    input_shape: tuple = (1, 1440, 13)
) -> None:
    """
    从检查点导出模型
    
    参数:
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        input_shape: 输入形状
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "ncde_state_dict" in checkpoint:
        from models.ncde_solver import NCDESolver
        model = NCDESolver(state_dim=13, hidden_dim=256)
        model.load_state_dict(checkpoint["ncde_state_dict"])
    elif "koopman_state_dict" in checkpoint:
        from models.koopman_operator import KoopmanOperator
        model = KoopmanOperator(state_dim=13, control_dim=4, latent_dim=64)
        model.load_state_dict(checkpoint["koopman_state_dict"])
    else:
        raise ValueError("无法识别的检查点格式")
    
    exporter = ONNXExporter(model, output_dir)
    exporter.export(input_shape)


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="onnx_exports")
    parser.add_argument("--opset_version", type=int, default=14)
    parser.add_argument("--enable_validation", action="store_true")
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
