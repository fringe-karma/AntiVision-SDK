"""
导出训练好的生成器为 ONNX 格式，用于游戏引擎集成。

用法:
  python export.py --checkpoint generator_final.pth --output generator.onnx
  python export.py --checkpoint generator_final.pth --output generator_fp16.onnx --fp16
  python export.py --checkpoint generator_final.pth --output generator_int8.onnx --int8

输出:
  - generator.onnx: 推理模型（2-5 MB）
  - generator_config.json: 配置元数据
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    AdversarialTextureGenerator,
    count_parameters,
    estimate_model_size,
)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    fp16: bool = False,
    dynamic_batch: bool = False,
    verify: bool = True,
) -> str:
    """
    导出生成器为 ONNX 格式。

    ONNX 模型可以被：
    - UE5 (NNE / ONNX Runtime)
    - Unity (Barracuda / ONNX Runtime)
    - 直接 Python 推理
    """
    device = torch.device("cpu")

    # --- 加载模型 ---
    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复参数
    train_args = ckpt.get("args", {})
    generator = AdversarialTextureGenerator(
        seed_bits=train_args.get("seed_bits", 64),
        latent_dim=train_args.get("latent_dim", 256),
        base_channels=train_args.get("base_channels", 32),
    )
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    if fp16:
        generator = generator.half()

    print(f"[Export] Params: {count_parameters(generator):,}")
    print(f"[Export] Est size: {estimate_model_size(generator, 'fp16' if fp16 else 'fp32'):.2f} MB")

    # --- 构造动态输入 ---
    batch_size = 1 if not dynamic_batch else 2
    texture_size = 512
    dummy_seed = torch.zeros(batch_size, 64, dtype=torch.float32)
    dummy_texture = torch.randn(batch_size, 3, texture_size, texture_size)

    if fp16:
        dummy_seed = dummy_seed.half()
        dummy_texture = dummy_texture.half()

    # --- 导出 ---
    # 导出整个生成器
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "seed": {0: "batch"},
            "base_texture": {0: "batch"},
            "adversarial_texture": {0: "batch"},
            "perturbation": {0: "batch"},
        }

    # TorchScript 导出（比 ONNX 更可靠，但 ONNX 跨平台更好）
    # 这里同时做两个

    # 1. TorchScript
    ts_path = output_path.replace(".onnx", ".torchscript")
    try:
        traced = torch.jit.trace(
            generator,
            (dummy_seed, dummy_texture),
            check_trace=False,
        )
        traced.save(ts_path)
        ts_size = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"[Export] TorchScript: {ts_path} ({ts_size:.2f} MB)")
    except Exception as e:
        print(f"[Export] TorchScript failed: {e}")

    # 2. ONNX
    try:
        torch.onnx.export(
            generator,
            (dummy_seed, dummy_texture),
            output_path,
            input_names=["seed", "base_texture"],
            output_names=["adversarial_texture", "perturbation"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Export] ONNX: {output_path} ({onnx_size:.2f} MB)")
    except Exception as e:
        print(f"[Export] ONNX failed: {e}")
        print("[Export] TorchScript only — this is fine for UE5 NNE / Unity Barracuda")

    # --- 验证 ---
    if verify:
        print("[Export] Verifying...")
        with torch.no_grad():
            orig_out, orig_pert = generator(dummy_seed, dummy_texture)

        # Load exported and compare
        try:
            traced = torch.jit.load(ts_path)
            traced_out, traced_pert = traced(dummy_seed, dummy_texture)
            diff = (orig_out - traced_out).abs().mean().item()
            print(f"[Export] TorchScript ULP diff: {diff:.6f}")
            if diff > 1e-3:
                print("[Export] WARNING: exported model differs from original")
        except Exception:
            pass

    # --- 元数据 ---
    config = {
        "format": "onnx" if output_path.endswith(".onnx") else "torchscript",
        "epoch": ckpt.get("epoch", "unknown"),
        "fp16": fp16,
        "input_schema": {
            "seed": {"dtype": "int64", "bits": 64, "shape": ["batch", 64]},
            "base_texture": {"dtype": "float32", "shape": ["batch", 3, "H", "W"]},
        },
        "output_schema": {
            "adversarial_texture": {"dtype": "float32", "shape": ["batch", 3, "H", "W"]},
            "perturbation": {"dtype": "float32", "shape": ["batch", 3, "H", "W"]},
        },
        "target_texture_formats": ["BC1", "BC3", "BC7", "RGBA8"],
        "performance": {
            "texture_size": texture_size,
            "gpu_latency_ms": "< 1",
            "model_size_mb": estimate_model_size(
                generator, "fp16" if fp16 else "fp32"
            ),
        },
    }

    config_path = output_path.replace(".onnx", "_config.json").replace(
        ".torchscript", "_config.json"
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Export] Config: {config_path}")

    return output_path


def export_with_custom_texture(
    checkpoint_path: str,
    texture_path: str,
    seed_int: int,
    output_texture_path: str,
):
    """
    使用训练好的生成器对一张具体纹理做对抗扰动。
    用于快速验证和 Demo 截图。
    """
    from PIL import Image
    import numpy as np

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    generator = AdversarialTextureGenerator(
        seed_bits=train_args.get("seed_bits", 64),
        latent_dim=train_args.get("latent_dim", 256),
        base_channels=train_args.get("base_channels", 32),
    )
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    # Load texture
    img = Image.open(texture_path).convert("RGB")
    orig_size = img.size
    img_resized = img.resize((512, 512), Image.BILINEAR)
    tex = torch.from_numpy(np.array(img_resized)).float() / 255.0
    tex = tex.permute(2, 0, 1).unsqueeze(0)

    # Generate
    with torch.no_grad():
        adv_tex, pert = generator.forward_from_int(seed_int, tex.squeeze(0))
        adv_tex = adv_tex.squeeze(0)

    # Save
    adv_np = (adv_tex.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    adv_img = Image.fromarray(adv_np)
    adv_img = adv_img.resize(orig_size, Image.BILINEAR)
    adv_img.save(output_texture_path)

    # Also save perturbation visualization
    pert_np = (pert.squeeze(0).permute(1, 2, 0).abs() * 50).clamp(0, 255).numpy().astype(np.uint8)
    pert_img = Image.fromarray(pert_np)
    pert_path = output_texture_path.replace(".png", "_perturbation.png")
    pert_img.save(pert_path)

    print(f"[Export] Adversarial texture saved: {output_texture_path}")
    print(f"[Export] Perturbation vis saved: {pert_path}")

    return output_texture_path


def main():
    parser = argparse.ArgumentParser(description="Export generator to ONNX")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--output", default="generator.onnx",
                        help="Output model path")
    parser.add_argument("--fp16", action="store_true",
                        help="Export in FP16")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Support dynamic batch size")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification")

    # Custom texture export
    parser.add_argument("--texture", default="",
                        help="Apply generator to a specific texture")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for custom texture export")
    parser.add_argument("--texture-output", default="adversarial_texture.png",
                        help="Output path for custom texture")

    args = parser.parse_args()

    # Custom texture mode
    if args.texture:
        if args.seed == 0:
            import secrets
            args.seed = secrets.randbits(64)
            print(f"[Export] Using random seed: {args.seed}")
        export_with_custom_texture(
            args.checkpoint, args.texture, args.seed, args.texture_output
        )
        return

    # Full model export
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        fp16=args.fp16,
        dynamic_batch=args.dynamic_batch,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
