"""
对抗纹理生成器 — 训练脚本

使用集成多检测器的特征层攻击，训练一个生成器能对所有
CNN 检测模型产生有效对抗扰动。

用法:
  python train.py --data ./character_textures/ --epochs 200 --batch-size 4
  python train.py --resume checkpoint.pth

输出:
  - checkpoint.pth: 完整模型权重
  - generator.onnx: 精简推理模型
"""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    AdversarialTextureGenerator,
    count_parameters,
    estimate_model_size,
)
from losses import TotalLoss


# ---------------------------------------------------------------------------
# 数据集：从纹理文件中加载
# ---------------------------------------------------------------------------
class TextureDataset(Dataset):
    """加载游戏角色纹理贴图"""

    def __init__(
        self,
        data_dir: str,
        exts: tuple = (".png", ".jpg", ".jpeg", ".tga", ".bmp"),
        target_size: int = 512,
    ):
        from PIL import Image

        self.files = []
        for ext in exts:
            self.files.extend(Path(data_dir).rglob(f"*{ext}"))
            self.files.extend(Path(data_dir).rglob(f"*{ext.upper()}"))

        if not self.files:
            raise FileNotFoundError(f"No texture files found in {data_dir}")

        self.target_size = target_size
        print(f"[Dataset] Found {len(self.files)} texture files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.files[idx]).convert("RGB")
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]
        tensor = tensor.permute(2, 0, 1)  # [C, H, W]
        return tensor


# ---------------------------------------------------------------------------
# 检测器 Hook：提取中间特征
# ---------------------------------------------------------------------------
class DetectorEnsemble:
    """
    加载多个 YOLO 检测器，提供特征提取接口。

    对抗训练需要：
    - extract_features(image) → list[Tensor]  中间层特征图
    - detect(image) → list[dict]              检测结果
    """

    def __init__(
        self,
        models: list[str] | None = None,
        device: str = "cuda",
    ):
        if models is None:
            models = ["yolov5n", "yolov8n", "yolov8s", "yolov10n", "rtdetr-l"]

        self.device = device
        self.detectors: dict[str, YOLO] = {}

        for name in models:
            from ultralytics import YOLO
            weight = {
                "yolov5n": "yolov5nu.pt",
                "yolov5s": "yolov5su.pt",
                "yolov8n": "yolov8n.pt",
                "yolov8s": "yolov8s.pt",
                "yolov8m": "yolov8m.pt",
                "yolov10n": "yolov10n.pt",
                "yolov10s": "yolov10s.pt",
                "rtdetr-l": "rtdetr-l.pt",
                "rtdetr-x": "rtdetr-x.pt",
            }.get(name, name)

            print(f"[Ensemble] Loading {name}...")
            model = YOLO(weight)
            model.to(device)
            self.detectors[name] = model

    def extract_features(
        self, image_batch: torch.Tensor
    ) -> dict[str, list[torch.Tensor]]:
        """
        提取每个检测器的中间层特征图。

        通过 hook 截取 backbone 各阶段输出。
        """
        features_all: dict[str, list[torch.Tensor]] = {}

        for name, detector in self.detectors.items():
            feats = []

            try:
                # 尝试从 model.model 获取特征
                inner = detector.model
                # YOLO 模型通常有 model 属性存储实际 nn.Module
                if hasattr(inner, 'model') and isinstance(inner.model, nn.Module):
                    _extract_features_from_module(
                        inner.model, image_batch, feats
                    )
            except Exception:
                pass

            if not feats:
                # 回退：直接用检测结果当特征（置信度作为代理）
                pass

            features_all[name] = feats

        return features_all

    def detect_batch(
        self, image_batch: torch.Tensor
    ) -> dict[str, list[dict]]:
        """批量检测"""
        detections_all: dict[str, list[dict]] = {}

        with torch.no_grad():
            for name, detector in self.detectors.items():
                results = detector(
                    image_batch,
                    conf=0.15,  # 外挂的典型低阈值
                    verbose=False,
                    device=self.device,
                )
                dets = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_name = result.names.get(
                                int(box.cls.item()), "unknown"
                            )
                            if cls_name == "person":
                                dets.append({
                                    "confidence": float(box.conf.item()),
                                    "class_name": cls_name,
                                })
                detections_all[name] = dets

        return detections_all

    def get_confidence_sum(self, image_batch: torch.Tensor) -> float:
        """获取所有模型对 person 类别的置信度总和（用于简单评估）"""
        dets = self.detect_batch(image_batch)
        total = 0.0
        for model_dets in dets.values():
            for d in model_dets:
                total += d["confidence"]
        return total


def _extract_features_from_module(
    module: nn.Module,
    x: torch.Tensor,
    features: list,
    max_layers: int = 8,
):
    """
    在前向传播中抓取中间层特征图。
    简化方案：遍历 module 的子模块，每次 Conv2d 后记录 output。
    """
    hooks = []
    layer_outputs = []

    def hook_fn(_, __, output):
        layer_outputs.append(output.detach())

    count = 0
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) and count < max_layers:
            hooks.append(child.register_forward_hook(hook_fn))
            count += 1

    try:
        with torch.no_grad():
            module.eval()
            _ = module(x)
    except Exception:
        pass

    for h in hooks:
        h.remove()

    features.extend(layer_outputs)


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print(f"[Train] Device: {device}")

    # --- 数据 ---
    if args.data_dir and Path(args.data_dir).is_dir():
        dataset = TextureDataset(args.data_dir, target_size=args.texture_size)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        print(f"[Train] {len(dataset)} textures, {len(dataloader)} batches")
    else:
        # 没有真实纹理时用随机噪声训练（概念验证）
        print("[Train] WARNING: No texture data — using random noise as placeholder")
        dataloader = None

    # --- 模型 ---
    generator = AdversarialTextureGenerator(
        seed_bits=64,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    ).to(device)

    print(f"[Train] Generator params: {count_parameters(generator):,}")
    print(f"[Train] Est size: {estimate_model_size(generator):.2f} MB (FP32)")

    # --- 检测器集成 ---
    detector_ensemble = DetectorEnsemble(
        models=args.ensemble_models, device=device
    )

    # --- 损失 ---
    criterion = TotalLoss(
        adv_weight=args.w_adv,
        div_weight=args.w_div,
        percep_weight=args.w_percep,
        eot_weight=args.w_eot,
    )

    # --- 优化器 ---
    optimizer = optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # --- 恢复 ---
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[Train] Resumed from epoch {start_epoch}")

    # --- 日志 ---
    writer = SummaryWriter(args.log_dir)

    # --- 训练 ---
    global_step = 0
    batch_size_actual = args.batch_size
    texture_size = args.texture_size

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        epoch_losses = {"adv": 0, "div": 0, "percep": 0, "eot": 0, "total": 0}
        n_batches = 0

        # 每 epoch 的 batch 数
        batches_per_epoch = args.batches_per_epoch or (
            len(dataloader) if dataloader else 100
        )

        for batch_idx in range(batches_per_epoch):
            # --- 构造输入 ---
            if dataloader:
                try:
                    base_textures = next(iter(dataloader))
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    base_textures = next(dataloader_iter)
                base_textures = base_textures.to(device)
                B = base_textures.shape[0]
            else:
                # 占位：随机纹理
                B = batch_size_actual
                base_textures = torch.rand(B, 3, texture_size, texture_size,
                                           device=device)

            # N 个不同 seed，每个对应一张纹理
            seeds = torch.stack([
                generator.seed_to_tensor(secrets.randbits(64)).squeeze(0)
                for _ in range(B)
            ]).to(device)

            # --- 前向 ---
            adv_textures, perturbations = generator(seeds, base_textures)

            # --- 检测器评估 ---
            # 对对抗纹理做检测
            features_all = detector_ensemble.extract_features(adv_textures)
            detections_all = detector_ensemble.detect_batch(adv_textures)

            # 汇总特征和检测
            all_features = []
            all_detections = []
            for name in detector_ensemble.detectors:
                all_features.extend(features_all.get(name, []))
                all_detections.extend(detections_all.get(name, []))

            # --- 损失 ---
            loss, components = criterion(
                adv_textures, perturbations, base_textures,
                all_features, all_detections,
            )

            # --- 反向 ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()

            # --- 记录 ---
            for k in epoch_losses:
                epoch_losses[k] += components[k]
            n_batches += 1
            global_step += 1

            if batch_idx % args.log_interval == 0:
                conf_sum = detector_ensemble.get_confidence_sum(
                    adv_textures[:4]
                )
                baseline = detector_ensemble.get_confidence_sum(
                    base_textures[:4]
                )
                print(
                    f"Epoch {epoch:4d} | Batch {batch_idx:4d}/{batches_per_epoch} | "
                    f"Loss {components['total']:.4f} "
                    f"(adv={components['adv']:.4f} "
                    f"div={components['div']:.4f} "
                    f"per={components['percep']:.4f}) | "
                    f"Conf: {baseline:.2f}→{conf_sum:.2f}"
                )

            writer.add_scalar("loss/total", components["total"], global_step)
            writer.add_scalar("loss/adv", components["adv"], global_step)
            writer.add_scalar("loss/div", components["div"], global_step)
            writer.add_scalar("loss/percep", components["percep"], global_step)
            writer.add_scalar("metrics/confidence_sum", conf_sum, global_step)

        # --- Epoch 结束 ---
        scheduler.step()

        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch:4d} AVG — "
            f"adv={avg_losses['adv']:.4f} "
            f"div={avg_losses['div']:.4f} "
            f"per={avg_losses['percep']:.4f} "
            f"total={avg_losses['total']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # --- 保存 ---
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pth")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "generator": generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[Train] Saved: {ckpt_path}")

        # 验证
        if (epoch + 1) % args.val_interval == 0:
            validate(generator, detector_ensemble, device, writer, epoch)

    # --- 最终保存 ---
    final_path = os.path.join(args.output_dir, "generator_final.pth")
    torch.save({
        "epoch": args.epochs,
        "generator": generator.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"[Train] Final: {final_path}")

    writer.close()
    return generator


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate(generator, detector_ensemble, device, writer, epoch):
    """评估生成器效果"""
    generator.eval()
    texture_size = 512
    batch_size = 8

    base_textures = torch.rand(batch_size, 3, texture_size, texture_size,
                               device=device)
    seeds = torch.stack([
        generator.seed_to_tensor(secrets.randbits(64)).squeeze(0)
        for _ in range(batch_size)
    ]).to(device)

    with torch.no_grad():
        adv_textures, perturbations = generator(seeds, base_textures)

    # 原始 vs 对抗
    base_conf = detector_ensemble.get_confidence_sum(base_textures)
    adv_conf = detector_ensemble.get_confidence_sum(adv_textures)
    reduction = (base_conf - adv_conf) / (base_conf + 1e-8) * 100

    print(
        f"[Val] Epoch {epoch} — "
        f"Base conf: {base_conf:.2f} → Adv conf: {adv_conf:.2f} "
        f"({reduction:.1f}% reduction)"
    )

    writer.add_scalar("val/base_confidence", base_conf, epoch)
    writer.add_scalar("val/adv_confidence", adv_conf, epoch)
    writer.add_scalar("val/reduction_pct", reduction, epoch)

    generator.train()
    return adv_conf, reduction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Adversarial Texture Generator"
    )

    # Data
    parser.add_argument("--data-dir", default="",
                        help="Character texture directory")
    parser.add_argument("--texture-size", type=int, default=512)

    # Model
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")

    # Loss weights
    parser.add_argument("--w-adv", type=float, default=10.0)
    parser.add_argument("--w-div", type=float, default=2.0)
    parser.add_argument("--w-percep", type=float, default=1.0)
    parser.add_argument("--w-eot", type=float, default=5.0)

    # Ensemble models
    parser.add_argument(
        "--ensemble-models", nargs="+",
        default=["yolov5n", "yolov8n", "yolov8s", "yolov10n", "rtdetr-l"],
    )

    # IO
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./runs")
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", default="")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
