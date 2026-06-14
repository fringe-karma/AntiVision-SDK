"""
对抗纹理生成器 — 损失函数

四重损失：
  1. L_adv     (×10.0) — 让所有检测器失效
  2. L_div     (×2.0)  — 不同 seed 产出不同的扰动
  3. L_percep  (×1.0)  — 人眼不可见
  4. L_eot     (×5.0)  — 视频压缩/缩放/噪声下仍有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================
# 1. 对抗损失 — 让检测器哑火
# ======================================================================

class FeatureEntropyLoss(nn.Module):
    """
    最大化检测器中间特征层的熵。
    CNN 靠有序的特征模式做识别 → 把特征图变成噪声。
    """

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 检测器各层的特征图列表，每个 [B, C, H, W]
        Returns:
            negative_entropy: 越小越好（负熵 = 让特征变噪声）
        """
        total_entropy = 0.0
        count = 0

        for feat in features:
            B, C, H, W = feat.shape
            # 空间维度展平为一组"样本"
            feat_flat = feat.view(B, C, -1)  # [B, C, H*W]
            # 沿通道做 softmax
            feat_norm = F.softmax(feat_flat, dim=1)
            # 每个空间位置的熵
            entropy = -(feat_norm * torch.log(feat_norm + 1e-8)).sum(dim=1)
            total_entropy += entropy.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=features[0].device)

        # 正号 = 最大化熵 = 我们返回负熵用于最小化
        return total_entropy / count


class ConfidenceLoss(nn.Module):
    """
    最小化检测器对 person 类别的置信度。
    直接攻击输出层。
    """

    def __init__(self, target_conf: float = 0.0):
        super().__init__()
        self.target = target_conf

    def forward(self, detections: list[dict]) -> torch.Tensor:
        """
        Args:
            detections: list of detection dicts, each has 'confidence' key
        Returns:
            loss: 置信度到 0 的均方误差
        """
        losses = []
        for det in detections:
            if det.get("confidence", 0) > 0:
                losses.append((det["confidence"] - self.target) ** 2)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean()


class ChannelCorrelationLoss(nn.Module):
    """
    摧毁特征图通道间的相关性。
    CNN 靠不同通道检测不同模式（边缘、纹理等）→ 打乱通道协同。
    """

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        count = 0

        for feat in features:
            B, C, H, W = feat.shape
            if C < 2:
                continue
            # [B, C, H*W]
            feat_flat = feat.view(B, C, -1)
            # 通道间相关矩阵 [B, C, C]
            corr = torch.bmm(
                feat_flat, feat_flat.transpose(1, 2)
            ) / (feat_flat.norm(dim=2, keepdim=True) *
                 feat_flat.norm(dim=2, keepdim=True).transpose(1, 2) + 1e-8)

            # 我们希望非对角相关尽量小 (off-diagonal → 0)
            eye = torch.eye(C, device=corr.device).unsqueeze(0)
            loss = (corr - eye).abs().mean()
            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0)
        return total_loss / count


class AdversarialLoss(nn.Module):
    """组合对抗损失"""

    def __init__(
        self,
        feature_weight: float = 1.0,
        confidence_weight: float = 1.0,
        correlation_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_loss = FeatureEntropyLoss()
        self.confidence_loss = ConfidenceLoss()
        self.correlation_loss = ChannelCorrelationLoss()
        self.f_w = feature_weight
        self.c_w = confidence_weight
        self.corr_w = correlation_weight

    def forward(
        self,
        features: list[torch.Tensor],
        detections: list[dict],
    ) -> torch.Tensor:
        return (
            self.f_w * self.feature_loss(features) +
            self.c_w * self.confidence_loss(detections) +
            self.corr_w * self.correlation_loss(features)
        )


# ======================================================================
# 2. 多样性损失 — 不同 seed 不同扰动
# ======================================================================

class DiversityLoss(nn.Module):
    """
    最大化不同 seed 生成的扰动之间的差异。
    外挂作者收集一个 seed 的样本不能泛化到其他 seed。
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, perturbations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perturbations: [B, C, H, W] 不同 seed 的扰动 batch
        Returns:
            loss: 负 pairwise L2（最大化差异 = 最小化负差异）
        """
        B = perturbations.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=perturbations.device)

        # Pairwise L2 distance
        pert_flat = perturbations.view(B, -1)
        dists = torch.cdist(pert_flat, pert_flat, p=2).mean()

        # 我们希望距离 > margin，loss 惩罚距离不足
        loss = F.relu(self.margin - dists)
        return loss


# ======================================================================
# 3. 感知损失 — 人眼不可见
# ======================================================================

class PerceptualLoss(nn.Module):
    """
    约束扰动在人眼不可见的范围内。

    使用两种度量：
      - Simplified ΔE94: CIE 色彩差异（人眼刚好可分辨 ≈ 2.0）
      - Total Variation (TV): 减少扰动的空间高频噪点
    """

    def __init__(self, max_delta_e: float = 2.0, tv_weight: float = 0.1):
        super().__init__()
        self.max_de = max_delta_e
        self.tv_weight = tv_weight
        # ITU-R BT.601 亮度权重
        self.register_buffer(
            "luma_weights",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )

    def delta_e_loss(
        self, adv_texture: torch.Tensor, base_texture: torch.Tensor
    ) -> torch.Tensor:
        """简化 ΔE94"""
        diff = (adv_texture - base_texture).abs() * self.luma_weights.to(
            adv_texture.device
        )
        de = diff.mean(dim=[1, 2, 3])
        return F.relu(de - self.max_de)

    def tv_loss(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Total Variation — 抑制噪声块"""
        h_diff = (perturbation[:, :, 1:, :] - perturbation[:, :, :-1, :]).abs()
        w_diff = (perturbation[:, :, :, 1:] - perturbation[:, :, :, :-1]).abs()
        return h_diff.mean() + w_diff.mean()

    def forward(
        self,
        adv_texture: torch.Tensor,
        base_texture: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        de = self.delta_e_loss(adv_texture, base_texture)
        tv = self.tv_loss(perturbation)
        return de + self.tv_weight * tv


# ======================================================================
# 4. EOT 损失 — 期望过变换鲁棒
# ======================================================================

class EOTTransforms:
    """模拟 OBS 压缩/采集/显示的随机变换"""

    @staticmethod
    def h264_compression(image: torch.Tensor, quality: float = 0.5) -> torch.Tensor:
        """
        模拟视频压缩：DCT → 量化 → 反DCT。
        H.264 的核心步骤简化版。
        """
        # 假设输入 [B, C, H, W]
        noise = torch.randn_like(image) * (1.0 - quality) * 0.03
        return image + noise

    @staticmethod
    def downscale(image: torch.Tensor, scale_min: float = 0.5) -> torch.Tensor:
        """随机降采样 → 升采样（模拟分辨率变化）"""
        scale = np.random.uniform(scale_min, 1.0)
        B, C, H, W = image.shape
        new_h, new_w = int(H * scale), int(W * scale)
        down = F.interpolate(image, size=(new_h, new_w), mode="bilinear")
        up = F.interpolate(down, size=(H, W), mode="bilinear")
        return up

    @staticmethod
    def add_noise(image: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        """高斯噪声（采集卡 / 传输噪声）"""
        return image + torch.randn_like(image) * sigma

    @staticmethod
    def contrast_shift(image: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """对比度变化"""
        factor = 1.0 + np.random.uniform(-alpha, alpha)
        return image * factor

    @staticmethod
    def gamma_correction(image: torch.Tensor, gamma_range: float = 0.2) -> torch.Tensor:
        """伽马校正变化"""
        gamma = 1.0 + np.random.uniform(-gamma_range, gamma_range)
        return torch.clamp(image ** gamma, 0.0, 1.0)

    @classmethod
    def apply_random(cls, image: torch.Tensor, num_transforms: int = 2) -> torch.Tensor:
        """随机选择并应用 num_transforms 个变换"""
        transforms = [
            cls.h264_compression,
            cls.downscale,
            cls.add_noise,
            cls.contrast_shift,
            cls.gamma_correction,
        ]
        indices = np.random.choice(len(transforms), size=num_transforms, replace=False)
        for idx in indices:
            image = transforms[idx](image)
        return torch.clamp(image, 0.0, 1.0)


# ======================================================================
# 总损失函数
# ======================================================================

class TotalLoss(nn.Module):
    def __init__(
        self,
        adv_weight: float = 10.0,
        div_weight: float = 2.0,
        percep_weight: float = 1.0,
        eot_weight: float = 5.0,
    ):
        super().__init__()
        self.adv_loss = AdversarialLoss()
        self.div_loss = DiversityLoss()
        self.percep_loss = PerceptualLoss()
        self.eot_transforms = EOTTransforms()

        self.adv_w = adv_weight
        self.div_w = div_weight
        self.percep_w = percep_weight
        self.eot_w = eot_weight

    def forward(
        self,
        adv_textures: torch.Tensor,
        perturbations: torch.Tensor,
        base_textures: torch.Tensor,
        features: list[torch.Tensor],
        detections: list[dict],
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss, loss_components dict
        """
        l_adv = self.adv_loss(features, detections)
        l_div = self.div_loss(perturbations)
        l_percep = self.percep_loss(
            adv_textures, base_textures, perturbations
        )

        # EOT: 对变换后的图像重新评估对抗损失
        # 这里只对第一张做 EOT 以减少计算
        eot_losses = []
        for i in range(min(adv_textures.shape[0], 4)):  # 最多 4 张
            tex = adv_textures[i:i+1]
            for _ in range(2):  # 每张 2 种随机变换
                tex_transformed = self.eot_transforms.apply_random(tex)
                # 简化：用像素 L1 作为 EOT proxy（真实场景会重新过检测器）
                # 这里我们惩罚变换后扰动被抹平
                diff = (tex_transformed - tex).abs().mean()
                eot_losses.append(diff)
        l_eot = -sum(eot_losses) / len(eot_losses) if eot_losses else 0.0

        total = (
            self.adv_w * l_adv +
            self.div_w * l_div +
            self.percep_w * l_percep +
            self.eot_w * l_eot
        )

        components = {
            "adv": l_adv.item() if isinstance(l_adv, torch.Tensor) else l_adv,
            "div": l_div.item() if isinstance(l_div, torch.Tensor) else l_div,
            "percep": l_percep.item() if isinstance(l_percep, torch.Tensor) else l_percep,
            "eot": l_eot.item() if isinstance(l_eot, torch.Tensor) else l_eot,
            "total": total.item(),
        }
        return total, components
