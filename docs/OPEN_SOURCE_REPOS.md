---
title: "开源仓库分析"
date: 2026-05-31
status: reference
tags:
  - anti-vision
  - research
  - references
---

# GitHub 开源项目清单：对抗纹理 / 3D 可微渲染 / 目标检测攻击

最后更新：2026-05-31

---

## 🥇 第一梯队：最直接可用的

### 1. RAUCA — UV-map 对抗伪装（ICML 2024）⭐ 最推荐
| | |
|---|---|
| **仓库** | [github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack) |
| **为什么选它** | UV 贴图攻击 + 神经渲染器 NRP + 6 种检测器验证 |
| **对抗目标** | YOLOv3/v5、Faster R-CNN、SSD、DETR 等 |
| **渲染器** | Neural Renderer Plus (NRP) — 自己实现的可微神经渲染器 |
| **特点** | 多天气数据集集成、光照/环境特征捕获、模拟和真实世界双重验证 |
| **可直接用** | ✅ 有完整训练 pipeline + 评估 |

**与你的目标距离：** 它是做车辆伪装的。你需要把车辆模型换成游戏角色模型，检测器换成 YOLOv8/v10。核心 Pipeline 完全可复用。

---

### 2. FCA — 全车身对抗伪装（AAAI 2022）
| | |
|---|---|
| **仓库** | [github.com/idrl-lab/Full-coverage-camouflage-adversarial-attack](https://github.com/idrl-lab/Full-coverage-camouflage-adversarial-attack) |
| **项目页** | [idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/](https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/) |
| **对抗目标** | YOLO 系列检测器 |
| **方法** | 可微渲染 + EOT（多视角/多距离/部分遮挡） |
| **损失** | objectness loss + IoU loss + class loss + smoothness loss |
| **特点** | 第一个全车身覆盖对抗纹理工作，被后续论文广泛引用 |
| **可直接用** | ✅ 有完整代码 |

---

### 3. REVAMP — 通用对抗攻击平台（ICLR 2024）
| | |
|---|---|
| **仓库** | [github.com/poloclub/revamp](https://github.com/poloclub/revamp) |
| **是什么** | Python 库，自动化模拟对任意 3D 物体的对抗攻击 |
| **对抗目标** | 分类器 + 检测器（Faster R-CNN） |
| **方法** | 可微渲染 + PGD（L2/L∞）纹理扰动 |
| **特点** | 可配置场景/相机/光照/攻击参数 |
| **可直接用** | ✅ 库级封装，API 友好 |

**与你的目标距离：** 它目前主要针对 Faster R-CNN，需要加 YOLO 支持。但它的可微渲染管线是现成的。

---

## 🥈 第二梯队：需要改造但高度相关

### 4. AT3D — 对抗纹理 3D 网格（CVPR 2023 Highlight）
| | |
|---|---|
| **仓库** | [github.com/thu-ml/AT3D](https://github.com/thu-ml/AT3D) |
| **对抗目标** | 人脸识别（不是目标检测） |
| **方法** | 可微渲染 + EOT + 3DMM 纹理扰动 |
| **特点** | 清华 ML 组出品，CVPR 2023 Highlight |
| **可取什么** | EOT 实现、可微渲染的纹理梯度反传、平滑损失 |
| **需要改造** | 从人脸识别模型改成目标检测模型 |

---

### 5. Complicit-Splat — 3DGS 对抗攻击（CVPR 2025 Workshop）
| | |
|---|---|
| **仓库** | [github.com/poloclub/complicit-splat](https://github.com/poloclub/complicit-splat) |
| **对抗目标** | Faster R-CNN 检测器 |
| **方法** | CLOAK + DAGGER — 隐藏对抗纹理只在特定视角可见 |
| **特点** | 基于 3D Gaussian Splatting（2024 年新兴的 3D 表示） |
| **可取什么** | 多视角攻击策略、检测器梯度反传 |

---

## 🥉 第三梯队：相关但不直接

### 6. UPC — 通用物理伪装攻击（CVPR 2020）
| 仓库 | 搜索 GitHub `Universal-Physical-Camouflage` |
|------|------------------------------------------|
| 目标 | RPN + 检测器分类头联合攻击 |
| 可取 | 跨类别通用伪装纹理生成 |

### 7. DAS — 双注意力抑制攻击
| 仓库 | 搜索 GitHub `Dual-Attention-Suppression` |
|------|------------------------------------------|
| 目标 | 攻击检测器的 attention map |
| 可取 | 部分覆盖纹理（非全车） |

---

## 🔧 辅助工具：可微渲染器

| 工具 | 仓库 | 用途 |
|------|------|------|
| **PyTorch3D** | [github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | Facebook 的可微渲染库，RAUCA/FCA/REVAMP 都基于它 |
| **Mitsuba 3** | [github.com/mitsuba-renderer/mitsuba3](https://github.com/mitsuba-renderer/mitsuba3) | 物理可微渲染器，更真实但更慢 |
| **Nvdiffrast** | [github.com/NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) | NVIDIA 的可微光栅化器，速度快 |
| **Kaolin** | [github.com/NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) | NVIDIA 3D 深度学习库，含可微渲染 |

---

## 📋 你应该先克隆哪个

按优先级排序：

```bash
# 1. RAUCA — 最完整的 pipeline，多检测器验证
git clone https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack.git

# 2. FCA — 全车身伪装，经典方法
git clone https://github.com/idrl-lab/Full-coverage-camouflage-adversarial-attack.git

# 3. REVAMP — 库级封装，容易上手
git clone https://github.com/poloclub/revamp.git

# 4. FCA 项目页面（论文 + 补充材料）
# https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/
```

---

## 🎯 改造计划

拿 RAUCA 为例：

```
RAUCA 现在做的事：
  车辆 3D 模型 → UV 纹理 → NRP 可微渲染 → YOLOv3/v5 检测 → 梯度反传 → 更新纹理

你要改成：
  角色 3D 模型 → UV 纹理 → NRP/UE5 可微渲染 → YOLOv8/v10 检测 → 梯度反传 → 更新纹理
                                                            ↑
                                                    换成外挂实际用的检测器

额外需要加的：
  + 多角色支持（不同角色不同 UV）
  + 生成器网络（seed → 纹理，不是固定纹理）
  + EOT 中加入 OBS 视频压缩模拟
  + 评估跨模型黑盒迁移（YOLOv5/v8/v10/RT-DETR）
```

---

## ⚠️ 已知的坑

1. **RAUCA 和 FCA 都针对自动驾驶场景的车辆**——背景是道路/城市，不是 FPS 地图。需要更换场景数据集。
2. **NRP（RAUCA 的渲染器）是自定义的**，与 PyTorch3D 不完全兼容。可能需要替换。
3. **大模型（RT-DETR）的梯度计算很慢**——训练时可能只在 YOLOv8 上做白盒，然后在其他模型上测黑盒迁移。
4. **UE5 渲染 → 神经网络近似 → 梯度反传 这条路（TACO 的做法）比纯 PyTorch3D 更难复现**——TACO 没有公开代码，只有论文描述。

---

## 相关文档
- [[LEARNING_ROADMAP]]
- [[STAGE_SUMMARY]]
