---
title: "项目方案 v2.0（已归档）"
date: 2026-06-01
status: archived
tags:
  - anti-vision
  - planning
  - archived
---

# 项目 v2 修订计划

**日期：** 2026-06-01  
**状态：** 基于 v1 实验结果重新规划

---

## 一、v1 复盘：什么成了，什么没成

### 成了的

| 里程碑 | 怎么证的 |
|--------|---------|
| 3D 角色模型可用 | OBJ 解析器零外部依赖，25K 顶点 52K 三角面 |
| 多材质 3D 渲染 | nvdiffrast 成功渲染了 Arm/Chest/Helmet/Legs/Mask |
| YOLO 能检测 3D 角色 | 多材质渲染 + 随机背景 → 16 视角稳定检测到 person |
| 3D 纹理攻击方向正确 | NES 一度将置信度从 1.06 打到 0.03（降 97%） |

### 没成的

| 问题 | 根因 |
|------|------|
| NES 梯度攻击稳不住 | 5×512×512 = 130 万个像素，10 个随机采样 → 信号噪声比太低 |
| nvdiffrast 不适合梯度攻击 | CUDA kernel 不保留 autograd 计算图，必须用零阶优化 |
| 2D 图像攻击无效 | PGD 精确梯度只降 YOLO 7.6%，跨模型迁移失效 |
| 生成器训练从未开始 | 可微链路缺一环，无法做种子→纹理的端到端训练 |

### 核心结论

> **缺的不是步数、不是算力、不是更好的 NES 调参。缺的是"可微渲染→可微检测器→梯度反传→纹理更新"这条端到端链路。**

---

## 二、v2 技术路线

### 根本改变：从 nvdiffrast 换到 PyTorch3D

| | v1（nvdiffrast） | v2（PyTorch3D） |
|------|---------------|----------------|
| 渲染方式 | CUDA kernel | 纯 PyTorch |
| 保留 autograd 计算图 | ❌ | ✅ |
| 能精确梯度反传 | ❌ 必须零阶优化 | ✅ autograd 原生支持 |
| 推理速度 | 快 | 较慢但可接受 |
| 安装 | Windows 无官方支持 | pip 装（虽然也没有 Windows） |

**PyTorch3D 的局限：** 同样没有 Windows pip 包。但云 GPU (Linux) 上直接 `conda install -c pytorch3d pytorch3d`。

### 新 Pipeline

```
角色 OBJ + UV + Base Color 纹理（已有）
     │
     ▼
PyTorch3D 网格加载 + 纹理贴图
     │
     ▼
PyTorch3D 可微渲染器（多视角，随机相机/光照/背景）
     │
     ▼
合成 2D 检测画面
     │
     ▼
可微检测器（torchvision Faster R-CNN / DETR）← 白盒攻击
     │
     ▼
对抗损失 ← autograd.backward()
     │
     ▼
梯度更新纹理像素（PGD / Adam，精确梯度，非随机采样）
     │
     ▼
黑盒验证：YOLOv5/v8/v10/RT-DETR ← 跨模型迁移测试
```

### 为什么这次能行

1. **梯度是精确的。** 不是随机猜 10 个方向取平均——是 130 万像素上每个像素都有一阶导数。
2. **学术界已验证。** CVC-Lab/3D_ADV_Mesh_pytorch3d 和 TAMU-VITA/3D_Adversarial_Logo 都打通了这条链路，我们做的是适配，不是发明。
3. **我们的资产是现成的。** OBJ 模型、纹理、YOLO 测试脚本、渲染管线骨架都有了。

---

## 三、可用的开源基础

### 主仓库：CVC-Lab/3D_ADV_Mesh_pytorch3d

| | |
|---|---|
| 仓库 | `github.com/CVC-Lab/3D_ADV_Mesh_pytorch3d` |
| 做什么 | PyTorch3D 渲染 SMPL 人体 → YOLO/Faster R-CNN 检测 → 对抗纹理 |
| 怎么用 | **直接改：SMPL 模型 → 我们的古代战士 OBJ** |
| 缺什么 | 需要适配 YOLOv8/v10（它用的是旧版 YOLO） |

### 辅助参考：TAMU-VITA/3D_Adversarial_Logo

| | |
|---|---|
| 仓库 | `github.com/TAMU-VITA/3D_Adversarial_Logo` |
| 做什么 | 第一个"3D 对抗 Logo 隐身穿衣"攻击人检测的工作 |
| 怎么用 | 参考它的可微渲染管线 + YOLO 损失函数设计 |
| 效果参考 | YOLOv2/v3 攻击成功率 86-91% |

### 最新参考：AdvReal (2025)

| | |
|---|---|
| 仓库 | `github.com/Huangyh98/AdvReal` |
| 做什么 | 2D+3D 联合对抗补丁，YOLOv12 成功率 70.13% |
| 怎么用 | 参考它的非刚性表面建模 + EOT 鲁棒性设计 |

### 工具库：PADetBench

| | |
|---|---|
| 仓库 | `github.com/JiaweiLian/PADetBench` |
| 做什么 | 23 种攻击 × 48 种检测器的基准测试 |
| 怎么用 | 用它评估我们的对抗纹理在多种检测器上的效果 |

---

## 四、v2 开发计划

### Phase 1：克隆 + 跑通参考代码（第 1 周）

- [ ] 克隆 CVC-Lab/3D_ADV_Mesh_pytorch3d
- [ ] 在云 GPU (Linux) 上跑通它的 demo——确保 PyTorch3D + YOLO 全链路工作
- [ ] 克隆 TAMU-VITA/3D_Adversarial_Logo，跑通它的 demo
- [ ] 验证"可微渲染→检测器→梯度反传→纹理更新"确凿可行

**出口：** 在云 GPU 上看到检测置信度随攻击步骤下降。

### Phase 2：替换模型 + 升级检测器（第 2-3 周）

- [ ] 把 SMPL 人体模型替换为我们的古代战士 OBJ（Arm/Chest/Helmet/Legs/Mask 分别贴纹理）
- [ ] 把旧版 YOLO 升级为 YOLOv8/v10/RT-DETR
- [ ] 实现白盒攻击（torchvision Faster R-CNN）+ 黑盒验证（YOLO）的双轨评估
- [ ] 多种子 EOT——不同随机种子生成不同纹理，测迁移性

**出口：** 对抗纹理对 YOLOv8 黑盒置信度降低 ≥ 50%。

### Phase 3：生成器训练（第 4-5 周）

- [ ] 将"优化一张纹理"升级为"训练生成器 G(seed, base_tex) → 对抗纹理"
- [ ] 网络架构：U-Net 或轻量残差网络（2-5 MB 目标）
- [ ] 训练目标：对任意 seed ∈ [0, 2^64)，G 产出的纹理都能让检测器失效
- [ ] 导出 ONNX

**出口：** generator.onnx 能在 1ms 内从种子生成有效对抗纹理。

### Phase 4：产品化（第 6 周）

- [ ] UE5 插件集成（NNE / ONNX Runtime）
- [ ] 服务端 Seed 管理组件
- [ ] Demo 视频：OBS 采集画面中敌人检测置信度对比
- [ ] 技术白皮书 v2

---

## 五、与 v1 的差异总结

| | v1 | v2 |
|------|-----|-----|
| 可微渲染器 | nvdiffrast | **PyTorch3D** |
| 梯度方法 | NES 零阶优化 | **autograd 一阶精确梯度** |
| 检测器（白盒） | 无（全靠黑盒 YOLO） | **torchvision Faster R-CNN / DETR** |
| 检测器（黑盒验证） | YOLOv8 | YOLOv5/v8/v10/RT-DETR |
| 基础代码 | 从零写 | **基于 3D_ADV_Mesh_pytorch3d** |
| 人体模型 | SMPL 占位 | **古代战士 OBJ（已有）** |
| 安装 | Windows 不可用 | Linux 云 GPU（确定可用） |

---

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| PyTorch3D 遇到 autograd 内存爆炸 | 中 | 降低网格面数、减少视角数、用梯度检查点 |
| 可微检测器（Faster R-CNN）的梯度不稳定 | 中 | 用 DETR（Transformer 架构，梯度更平滑）|
| 白盒有效但黑盒不迁移 | 低 | 多检测器集成训练（已有论文证明跨架构迁移可行）|
| PyTorch3D Windows 安装问题 | 不存在 | 全程在云 GPU (Linux) 上开发 |

---

## 相关文档
- [[FINAL_PLAN]]
- [[STAGE_SUMMARY]]
