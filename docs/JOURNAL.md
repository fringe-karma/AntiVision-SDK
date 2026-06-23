---
title: "项目日志：从 v13 到知乎到五次复现"
date: 2026-06-21
status: active
tags:
  - anti-vision
  - journal
  - retrospective
---

# AntiVision 项目日志

## 已完成

### 五次独立复现

在 Mixamo Quantum Soldier 上做了 5 次独立 G_A 训练 + 24 视角验证。结果一致：**YOLOv8n 置信度从 0.79 降到 0.15，降幅 81%。YOLOv5s 从 0.22 降到 0.13，降幅 39%。**

### 十一条实验记录

- 古代战士 v13 单材质梯度投影 — 100% 成功
- Quantum Soldier 上 FRCNN 检测头梯度投影 — 失败（FRCNN 看不人）
- SPSA 像素级盲搜 — 失败（维度灾难）
- DCT SPSA — 失败（块间不协调）
- FRCNN 特征层 + COMBINED — 部分成功（单材质有效，拼起来打架）
- COMBINED 联合梯度投影（六材质）— **有效，五次复现**
- YOLO backbone 梯度 — NaN 爆炸
- 其余：COMBINED + 多视角 EOT / YOLO Backbone Hook / 古代战士全材质

### 公开产物

- GitHub 仓库：`https://github.com/fringe-karma/AntiVision-SDK`
- 知乎文章：[一个高中生，把 AI 视觉外挂逼到了墙角](https://zhuanlan.zhihu.com/p/105813622)
- 评论区收获：一名光子 TA 的脉脉接触、多条技术批评、多次社区讨论

## 已知局限性

- **完全依赖 YOLO 默认预训练权重**：未在自训权重上验证
- **FRCNN 检测头必须能看到渲染人物**：古代战士能，量子兵不能
- **渲染器差距**：nvdiffrast 渲染经不起 4K 缩放测试、DLSS 未测
- **攻击迁移性脆弱**：换 backbone 或微调后大概率失效
- **CNN 偏向假设在 ViT 等架构上未检验**

## 下一步

1. 用 YOLOv8n 微调版测迁移性
2. 重建 OBS 压缩实验
3. 准备 UE 验证环境
4. Get 第一个真实客户角色模型
5. 写第二篇知乎文章：五次复现 + 实验惨败记录
