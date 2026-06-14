---
title: "AntiVision SDK — 首页"
date: 2026-06-05
status: active
tags:
  - anti-vision
  - index
---

# AntiVision SDK

> 让 AI 视觉外挂变得不可靠——FPS 游戏反 AI 采集的引擎中间件。

**状态：** ✅ MVP 核心完成（2026-06-04）

---

## 做了什么

对抗纹理生成器系统——在游戏角色皮肤上注入人眼看不见的扰动，让 AI 外挂的检测器有时认人、有时不认。

**不是"让 AI 瞎"，是"让 AI 不可靠"。**

---

## 成果

三个生成器训练完成：

| 生成器 | 目标 | 效果 |
|------|------|------|
| **G_A** (13MB) | YOLOv8全家族 | v8s 检测归零 |
| **G_B** (13MB) | YOLOv5+v10广域 | v8s/v10s 归零 |
| **G_C** (13MB) | RT-DETR | RT-DETR -62% |

架构：多生成器轮换 + 行为驱动追猎。种子→纹理 < 1ms。

---

## 项目文件

| 目录/文件 | 用途 |
|------|------|
| `generator_train.py` | **生成器训练脚本**（核心） |
| `projgrad_attack.py` | **单纹理梯度攻击框架**（核心） |
| `ancient_character/` | 3D角色模型 + 纹理 |
| `models/` | 检测模型缓存（9个YOLO + DETR + RetinaNet + FRCNN） |
| `docs/` | 项目文档（提案/设计/审计/商业计划） |
| `src/generator/` | 生成器网络定义 + 损失函数 |
| `src/server/` | 服务端 Seed 管理 |
| `src/runtime/` | UE5/Unity 插件骨架 |

---

## 下一步

1. v5s 加固（最后未覆盖的 CNN 模型）
2. ONNX 导出 → UE5 插件 Demo
3. Demo 视频 + 客户接触

---

## 相关文档
- [[PROPOSAL]]
- [[FINAL_PLAN]]
- [[HONEST_STATUS]]
- [[ONE_PAGER]]
- [[WHITEPAPER]]
