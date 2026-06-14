---
title: "学习路线图"
date: 2026-05-31
status: reference
tags:
  - anti-vision
  - research
  - learning
---

# 路线 B 学习路线：对抗纹理 · 3D 可微渲染 · 目标检测攻击

## 学习顺序与优先级

---

## 📚 阶段 0：对抗样本基础（如果还没学过）

### 必读：开山作
| # | 论文 | 年份 | 取什么 | arXiv |
|---|------|------|--------|-------|
| 0.1 | Goodfellow et al. — **Explaining and Harnessing Adversarial Examples** | ICLR 2015 | FGSM 攻击原理，为什么 CNN 容易被骗 | [1412.6572](https://arxiv.org/abs/1412.6572) |
| 0.2 | Madry et al. — **Towards Deep Learning Models Resistant to Adversarial Attacks** | ICLR 2018 | PGD 攻击（FGSM 的迭代加强版），对抗训练基础 | [1706.06083](https://arxiv.org/abs/1706.06083) |

**学习目标：** 理解"对抗样本"是什么、梯度上升为什么有效、为什么像素级别的微小改动能摧毁 CNN 分类。

**预计时间：** 2-3 天

---

## 📚 阶段 1：EOT — 让对抗样本在物理世界存活

### 核心论文
| # | 论文 | 年份 | 取什么 | arXiv |
|---|------|------|--------|-------|
| 1.1 | Athalye et al. — **Synthesizing Robust Adversarial Examples** (EOT) | ICML 2018 | 如何让对抗样本对旋转/缩放/光照变化鲁棒——这就是那只著名的"3D 对抗乌龟"的论文 | [1707.07397](https://arxiv.org/abs/1707.07397) |
| 1.2 | Athalye et al. — **Obfuscated Gradients Give a False Sense of Security** | ICML 2018 | 为什么很多防御是假的，EOT 怎么打破它们 | [1802.00420](https://arxiv.org/abs/1802.00420) |

**学习目标：** 理解 EOT（Expectation Over Transformation）——蒙特卡洛采样过随机变换估计真实梯度。这是所有 3D 对抗纹理工作的数学基石。

**关键公式：**
$$\nabla_x L = \mathbb{E}_{t \sim T} [\nabla_x L(h_\theta(t(x)), c)]$$

在多个随机变换下取梯度平均 = 对变换分布鲁棒的对抗样本。

**预计时间：** 3-5 天

---

## 📚 阶段 2：3D 可微渲染 — 把梯度从 2D 搬到 3D

### 核心论文
| # | 论文 | 年份 | 取什么 | arXiv |
|---|------|------|--------|-------|
| 2.1 | **RenderBender: Adversarial Attacks Using Differentiable Rendering — A Survey** | IJCAI 2025 | 整个领域的全景地图，覆盖纹理/光照/Mesh/NeRF 攻击 | [2411.09749](https://arxiv.org/abs/2411.09749) |
| 2.2 | Xiao et al. — **MeshAdv: Adversarial Meshes for Visual Recognition** | CVPR 2019 | 第一个用可微渲染做 3D 对抗攻击的工作（著名的"对抗乌龟"） | [1810.05206](https://arxiv.org/abs/1810.05206) |

**补充参考书：**
- PyTorch3D 官方文档 + 教程（https://pytorch3d.org/）—— 可微渲染器的实际使用
- Mitsuba 3 文档（https://mitsuba.readthedocs.io/）—— 物理可微渲染器

**学习目标：** 理解"可微渲染器"是什么——它把 3D 场景的渲染过程变成可导的，这样可以把检测器的梯度反向传播到纹理像素上。

**预计时间：** 5-7 天（含动手跑 PyTorch3D 教程）

---

## 📚 阶段 3：3D 纹理对抗攻击 — 你最需要的

### 核心论文（按重要度排序）

| # | 论文 | 年份 | 取什么 | arXiv |
|---|------|------|--------|-------|
| 3.1 ⭐ | **TACO: Adversarial Camouflage Optimization on Trucks** | 2024 | **最重要的参考实现**——UE5 + 可微渲染 + YOLOv8，AP@0.5 打到 0.0099 | [2410.21443](https://arxiv.org/abs/2410.21443) |
| 3.2 ⭐ | Hu/Chu et al. — **AdvCaT: Clothing Textures Evade Person Detectors** | CVPR 2023 | **攻击人检测**（不是车），Voronoi 参数化纹理 + 真实布料打印验证 | [2307.01778](https://arxiv.org/abs/2307.01778) |
| 3.3 | Duan et al. — **Adversarial Texture Optimization for 3D Object Detection** | ECCV 2022 | 全车覆盖对抗纹理，多视角 EOT | 搜索 arXiv |
| 3.4 | Wang et al. — **DAS: Dual Attention Suppression Attack** | 2021 | 攻击检测器的 attention map，用部分覆盖纹理 | 搜索 arXiv |
| 3.5 | Zhong et al. — **Frequency Domain Adversarial Attacks** | CVPR 2022 | 频域对抗攻击——我们之前讨论的 DCT 路线的学术版本 | 搜索 arXiv |

**学习目标：** 完整理解"3D 纹理 → 可微渲染 → 多视角 EOT → 检测器失效"的流水线。

**预计时间：** 7-10 天（含读 TACO 代码）

---

## 📚 阶段 4：代码复现

### 有公开代码的仓库

| 论文 | 代码位置 |
|------|---------|
| TACO | 搜 GitHub `TACO adversarial camouflage`，或 MDPI 文章附带的 supplementary |
| AdvCaT | GitHub `chuwd19` (作者) |
| RenderBender Survey | 综述整理了大量仓库链接 |
| PyTorch3D 官方教程 | `github.com/facebookresearch/pytorch3d` |

### 你需要的可微渲染栈

```
选项 A（学术路线）：
  PyTorch3D + Mitsuba 3 = 最灵活，但和 UE5 不直通

选项 B（TACO 路线）：
  UE5 渲染 → Photorealistic Rendering Network (PRN) 代理 → PyTorch 梯度
  UE5 负责真实光照/阴影，PRN 负责可微性

选项 C（简化路线——推荐先试这个）：
  UE5 截图（多视角）→ 离线训练对抗纹理 → 验证
  不完全可微，但足够做概念验证
```

**预计时间：** 10-14 天

---

## 📖 参考书

| 书 | 用途 |
|----|------|
| **Deep Learning** (Goodfellow, Bengio, Courville) | 第 6-9 章（CNN）、第 13 章（对抗样本）—— MIT 出版社免费在线 |
| **Computer Vision: Algorithms and Applications** (Szeliski) | 第 5 章（特征检测）、第 14 章（识别） |
| **Physically Based Rendering** (Pharr, Jakob, Humphreys) | 渲染管线基础知识 |
| **PyTorch3D 官方教程** | 动手学可微渲染 |

---

## 🗺️ 推荐学习顺序图

```
Week 1-2:  阶段 0-1 → 看懂为什么 CNN 能被像素级改动摧毁
Week 3-4:  阶段 2   → 跑通 PyTorch3D 教程，理解可微渲染
Week 5-6:  阶段 3   → 精读 TACO + AdvCaT，画出他们的 Pipeline
Week 7-8:  阶段 4   → 复现 TACO，把卡车换成游戏角色
```

---

## 🔑 每个阶段你要能回答的问题

| 阶段 | 要能回答 |
|------|---------|
| 0 | 为什么 CNN 对 $L_\infty < \epsilon$ 的扰动脆弱？ |
| 1 | EOT 的蒙特卡洛采样为什么能打破随机变换防御？ |
| 2 | 可微渲染器的梯度怎么从像素级传到纹理 UV 坐标上？ |
| 3 | TACO 的 Photorealistic Rendering Network 是什么？为什么需要它？ |
| 4 | 你能在自己的 GPU 上跑通 TACO 的 Pipeline 吗？对 YOLOv8 的 AP 打到多少？ |

---

## 🎯 出口：能做什么

完成学习后，你应该能：

1. **解释**对抗纹理为什么能让 AI 检测器失效（给投资人/CTO 讲的技术故事）
2. **复现** TACO 在大卡车上对 YOLOv8 的攻击效果
3. **改造** TACO 的 Pipeline 到游戏角色上（敌人角色 → 对抗纹理 → OBS 采集 → YOLO 检测失效）
4. **设计** 生成器训练方案（无限种子 → 每局不同纹理）

---

## 相关文档
- [[OPEN_SOURCE_REPOS]]
- [[STAGE_SUMMARY]]
- [[TECHNICAL_DESIGN]]
