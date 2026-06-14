# AntiVision SDK

> 让 OBS AI 视觉自瞄外挂锁不死人 — 在角色纹理上做手脚，蒙住 AI 的眼睛。

**一个高中生做的 3D 对抗纹理攻击实验，四次独立复现，YOLOv8n 检测置信度降 81%。**

---

## 做了什么

在 3D 角色模型的纹理贴图上叠加微小扰动（每像素 ≤ 10/255，人眼不可见），使得 CNN 检测器的特征提取被破坏，无法稳定识别人物。

**不是"让 AI 瞎"，是"让 AI 不可靠"。**

外挂只要连续几帧检测不到人，自瞄就废了。

---

## 实验数据

在 Mixamo Quantum Soldier（战术士兵角色）上进行了 **4 次独立实验**，结果一致：

| 模型 | 外挂社区使用率 | 原始置信度 | 对抗后置信度 | 下降幅度 |
|------|-------------|-----------|------------|---------|
| **YOLOv8n** | ~15% | 0.79 | 0.15 | **-81%** |
| **YOLOv5s** | ~15% | 0.22 | 0.13 | **-39%** |

- 验证方式：50 个随机相机角度，7 个 YOLO 模型交叉验证
- 渲染器：nvdiffrast（离线 3D 渲染）
- 攻击方法：Faster R-CNN 检测头梯度投影 + 多视角 EOT + 六材质联合优化

---

## 技术原理

1. **梯度投影 (Gradient Projection)**：FRCNN 检测头在渲染画面上算梯度 → UV 映射反推 → 纹理空间更新
2. **EOT (Expectation Over Transformation)**：每步采样 6 个随机相机角度，防止扰动过拟合到单一视角
3. **联合优化**：5-6 个材质纹理同时更新，避免单材质独立优化互相抵消

三篇核心论文支撑：[FGSM (Goodfellow 2015)](https://arxiv.org/abs/1412.6572) · [Texture Bias (Geirhos 2019)](https://arxiv.org/abs/1811.12231) · [EOT (Athalye 2018)](https://arxiv.org/abs/1707.07397)

---

## 项目文件

| 文件 | 说明 |
|------|------|
| `projgrad_attack.py` | 梯度投影攻击框架（v13 验证成功版） |
| `joint_combined.py` | 联合优化攻击脚本（当前最强版本） |
| `generator_train.py` | 种子→对抗纹理 Generator 训练框架 |
| `docs/` | 完整项目文档（提案/技术设计/审计/商业计划） |
| `src/generator/` | Generator 网络定义 + 损失函数 |
| `src/server/` | 服务端 Seed 管理组件 |
| `src/runtime/` | UE5/Unity 插件骨架（未完成） |

---

## 实验历程

11 种方法，200+ 次 GPU 实验，最终定位到一条稳定可复现的攻击链路。

详细记录见 [docs/STAGE_SUMMARY.md](docs/STAGE_SUMMARY.md) 和 [docs/HONEST_STATUS.md](docs/HONEST_STATUS.md)。

---

## 文档导航

- [项目提案](docs/PROPOSAL.md)
- [完整项目方案 v3](docs/FINAL_PLAN.md)
- [诚实状态报告](docs/HONEST_STATUS.md)
- [技术白皮书](docs/WHITEPAPER.md)
- [技术设计文档](docs/TECHNICAL_DESIGN.md)
- [对抗性审计（红队视角）](docs/ADVERSARIAL_AUDIT.md)
- [商业计划](docs/BUSINESS_PLAN.md)
- [一句话讲清楚](docs/ONE_PAGER.md)
- [学习路线图](docs/LEARNING_ROADMAP.md)

---

## 待完成

- G_B（v5+v10 族）联合优化训练
- G_D（v5s 专攻）联合优化训练
- Generator 种子网络在量子兵上重新训练 → ONNX 导出
- 拿到真实游戏角色资产后的适配验证
- OBS 双机推流验证、UE5 Demo

---

## 联系

如果你在游戏公司做反作弊、技术美术或安全，对这个方向感兴趣：

给一个角色模型，72 小时出适配数据。或单纯聊一聊也行。

**Email**：2182212637@qq.com  
**GitHub**：https://github.com/fringe-karma/AntiVision-SDK

---

*AntiVision — 让 AI 视觉外挂在这款游戏里用不了。*
