---
title: "阶段性总结"
date: 2026-06-02
status: complete
tags:
  - anti-vision
  - status
  - retrospective
---

# 阶段性总结：AntiVision SDK

**日期：** 2026-06-02  
**状态：** 核心假设验证通过 ✅

---

## 一句话总结

**在 3D 角色纹理上注入人眼不可见的对抗扰动，让 YOLOv8 检测器完全检测不到人。经过 13 个版本迭代，首次达成 100% 置信度降低。**

---

## 项目起源

2026年4-5月，《三角洲行动》爆发OBS AI视觉吸附外挂危机——作弊者用OBS采集游戏画面，通过YOLO AI模型识别敌人位置，自动锁头。外挂不注入、不读内存、不改文件，传统反作弊(ACE)无从检测。单周封禁10万账号，大量玩家退游。

我们的方案：在游戏角色皮肤上注入对抗纹理，让AI视觉外挂"看"不到人。

---

## 实验结果总览

| 版本 | 方法 | 搜索空间 | 梯度 | 最佳效果 | 最终效果 | 结论 |
|------|------|---------|------|---------|---------|------|
| v1 | 静态DCT频域扰动 | 整图频谱 | 无 | — | 0.6% | ❌ 盲目噪声无效 |
| v2 | 2D图像NES | 2.6M像素 | 随机采样 | — | 3.5% | ❌ 空间太大 |
| v3 | 2D图像PGD | 2.6M像素 | 精确梯度 | — | 7.6% | ❌ 黑盒不迁移 |
| v4 | 胶囊体代理 | 简单几何体 | — | — | — | ❌ YOLO不认 |
| v5 | 多纹理3D NES | 5×262K像素 | 随机10方向 | 降97% | 33% | ⚠️ 方向对了但稳不住 |
| v6 | 多纹理NES 500步 | 5×262K像素 | 随机15方向 | 降97% | -207% | ❌ NES收敛失败 |
| v8 | 纯torch光栅化 | 262K像素 | autograd | — | — | ❌ 画质不够 |
| v9 | DCT系数NES | 200系数×5 | 随机20方向 | 正在跑 | — | 待验证 |
| v10 | nvdiffrast混合渲染 | 262K像素 | bilinear近似 | — | — | ❌ backward炸了 |
| v11 | grid_sample逐面 | 500面×262K像素 | autograd | — | — | ❌ 画质太差 |
| **v12** | **梯度投影** | **262K像素** | **手工反投** | **多次100%** | **-177%** | ⚠️ 4视角过拟合 |
| **v13** | **梯度投影 6视角 400步** | **262K像素** | **手工反投** | **多次100%** | **+100%** ✅ | **首次成功** |

---

## 核心发现

### 1. 什么有用

| 发现 | 证据 |
|------|------|
| **3D纹理空间攻击 > 2D图像攻击** | v5(33%) vs v3(7.6%) |
| **多材质分别渲染 > 单一纹理** | v5(33%) vs v4(0%) |
| **梯度投影 > NES随机采样** | v13(100%) vs v6(-207%) |
| **多视角EOT > 单视角过拟合** | v13(6视角100%) vs v12(4视角负值) |
| **Faster R-CNN代理 > YOLO直接攻击** | 精确梯度 + 黑盒迁移可行 |

### 2. 什么没用

| 发现 | 证据 |
|------|------|
| **2D像素空间攻击YOLO几乎无效** | v1(0.6%), v2(3.5%), v3(7.6%) |
| **NES在130万像素上收敛不了** | v6训练中97%→最终-207% |
| **PyTorch3D在云GPU上无法稳定安装** | 3次尝试，2次ABI冲突 |
| **纯grid_sample渲染画质不够** | v8,v11 YOLO检测不到人 |
| **自定义autograd Function嵌套会炸** | v10 RuntimeError |

### 3. 学术支撑

| 来源 | 核心贡献 | 与我们的关系 |
|------|---------|------------|
| **TACO (2024)** | UE5+可微渲染+YOLOv8卡车隐身 | 证明了"3D纹理+梯度攻击+检测器"这条路能走通 |
| **UV-Attack (ICLR 2025)** | NeRF UV映射+EoPT姿态变换+92.7% ASR | 提供了姿态变化的训练方案 |
| **AdvReal (2025)** | 2D+3D联合框架+NRSM非刚性建模 | 提供了非刚性表面建模参考 |
| **AdvCaT (CVPR 2023)** | Voronoi纹理+拓扑投影+实物验证 | 证明了物理世界纹理攻击可行 |
| **3D_Adversarial_Logo (2020)** | 首个3D人体对抗Logo攻击 | 方法论奠基 |
| **RenderBender综述 (2024)** | 28+篇可微渲染攻击系统分类 | 领域全景地图 |

---

## 技术路线确认

### 可行的攻击链路 (v13验证)

```
nvdiffrast渲染(高手质量) → Faster R-CNN检测(可微代理)
→ loss.backward() → 画面梯度 → UV映射 → 手工反投 → 纹理梯度
→ 更新纹理 → 多视角EOT平均 → YOLO黑盒验证
```

### 关键技术参数

| 参数 | 值 |
|------|-----|
| 纹理分辨率 | 512×512 = 262K像素 |
| 扰动预算 | ε=0.039 (10/255, 人眼不可见) |
| 多视角EOT | 6 views/step |
| 训练步数 | 400 steps |
| 代理检测器 | torchvision Faster R-CNN ResNet50 |
| 黑盒目标 | YOLOv8n |
| 渲染器 | nvdiffrast (forward) + 梯度投影 (backward) |
| 云GPU | RTX 4090 24GB |
| 单次训练时间 | ~10分钟 |

---

## 未解决的问题

| 问题 | 严重程度 | 下一步 |
|------|---------|--------|
| 训练过程不稳定（间歇性失效） | 中 | 更多视角EOT + 姿态变化 |
| 只在16个视角验证 | 中 | 50+视角大规模验证 |
| 只测了YOLOv8n | 中 | 加YOLOv5/v10/RT-DETR |
| 只攻击了单一纹理(Arm) | 低 | 扩展到全部5个材质 |
| 训练结果可能受随机视角影响 | 中 | 多轮独立训练取平均 |
| 没有生成器(G) | 高 | 训练seed→texture网络 |
| 没有UE5/Unity集成 | 高 | 引擎插件开发 |

---

## 已克隆的开源仓库

| 仓库 | 用途 | 状态 |
|------|------|------|
| `CVC-Lab/3D_ADV_Mesh_pytorch3d` | PyTorch3D人体对抗纹理 | ✅ 已读，待适配 |
| `TAMU-VITA/3D_Adversarial_Logo` | 首个3D人体Logo攻击 | ✅ 已读，YOLO太旧 |
| `PolyLiYJ/UV-Attack` | ICLR 2025 NeRF UV攻击 | ✅ 已读，代码可参考 |
| `TRLou/PGA` | ICCV 2025 3DGS伪装 | ✅ 已读 |
| `Huangyh98/AdvReal` | 2025 2D+3D联合框架 | ✅ 已读，PyTorch3D依赖 |
| `zhicheng2T0/Full-Distance-Attack` | NeurIPS 2024全距离攻击 | ✅ 已读 |
| `Wwangb/BadPatch` | 扩散模型对抗补丁 | ✅ 已读 |
| `SeRAlab/RAUCA` | ICML 2024车辆伪装 | ❌ Linux+特定数据集 |
| `idrl-lab/FCA` | AAAI 2022全车身伪装 | ❌ 同上 |

---

## 项目文档清单

| 文档 | 内容 |
|------|------|
| `docs/PROPOSAL.md` | 项目提案 |
| `docs/TECHNICAL_DESIGN.md` | 技术设计(含Seed白盒保护) |
| `docs/ADVERSARIAL_AUDIT.md` | 红队审计(5种攻击途径) |
| `docs/BUSINESS_PLAN.md` | 商业计划 |
| `docs/LEARNING_ROADMAP.md` | 学习路线(4阶段) |
| `docs/OPEN_SOURCE_REPOS.md` | 开源仓库分析 |
| `docs/PLAN_v2.md` | 修订版项目计划 |
| `docs/STAGE_SUMMARY.md` | 本文档 |

---

## 核心代码文件

| 文件 | 功能 | 状态 |
|------|------|------|
| `src/generator/model.py` | 对抗纹理生成器网络 | 骨架就绪，待训练 |
| `src/generator/losses.py` | 四重损失函数 | 设计完成 |
| `src/generator/train.py` | 生成器训练脚本 | 设计完成 |
| `src/generator/export.py` | ONNX导出 | 设计完成 |
| `experiments/obs_aimbot_sim/simulator.py` | OBS外挂模拟器 | ✅ 可运行 |
| `experiments/benchmarks/evaluate.py` | 评估框架 | ✅ 可运行 |
| `experiments/benchmarks/gradient_attack.py` | NES梯度攻击(v1-v6) | ✅ 可运行 |
| `experiments/benchmarks/quick_test.py` | DCT快速验证 | ❌ 已废弃 |
| `projgrad_attack.py` | **梯度投影攻击(v13)** | ✅ **首次成功** |
| `proxy_attack.py` | PGD代理攻击 | ✅ 可运行 |
| `gridras_attack.py` | grid_sample光栅化(v11) | ❌ 画质不足 |
| `softras_attack.py` | nvdiffrast混合渲染(v10) | ❌ backward炸了 |
| `dct_nes_attack.py` | DCT系数NES(v9) | ⚠️ 待验证 |
| `torch_attack.py` | 纯torch光栅化(v8) | ❌ 画质不足 |
| `p3d_final_attack.py` | PyTorch3D攻击 | ❌ ABI不兼容 |
| `nvdiffrast_attack.py` | nvdiffrast 3D攻击(v7) | ✅ 可运行 |
| `nes_tex_attack.py` | 纹理空间NES(v5) | ✅ 可运行(33%) |
| `v5_attack.py`, `v6_attack.py` | 多纹理NES攻击 | ✅ 可运行 |
| `src/server/seed_manager.py` | 服务端Seed管理 | ✅ 完成 |
| `src/runtime/unreal/*` | UE5插件框架 | 骨架就绪 |
| `src/runtime/unity/*` | Unity插件框架 | 骨架就绪 |
| `nvdiffrast_attack_v3.py` | 完整人体代理(v3) | ❌ 几何体不像人 |
| `full_pipeline.py` | OBJ→渲染→检测一条龙 | ✅ 可运行 |
| `cloud_train.py` | 云GPU训练脚本 | ✅ 可运行 |
| `render_v4.py` | 多材质渲染(v4) | ✅ 可运行 |

---

## 资金消耗

| 日期 | 用途 | 平台 | GPU | 约耗 |
|------|------|------|-----|------|
| 6/1 | v1-v6 实验 | 青椒云 | 4090 × 3台 | ~¥30 |
| 6/2 | v8-v13 实验 | 青椒云 | 4090 × 2台 | ~¥20 |
| **合计** | | | | **~¥50** |

---

## 下一步优先级

### 🔴 立即（本周）
- [ ] **v13 结果 50 视角验证**——确认不是随机视角的偶然结果
- [ ] **跨模型测试**——YOLOv5, YOLOv10, RT-DETR 黑盒迁移
- [ ] **多轮独立训练**——3轮取平均，确认可复现

### 🟡 短期（2-3周）
- [ ] **扩展到全部 5 个材质纹理**
- [ ] **增加训练稳定性**——更多视角EOT、姿态变化
- [ ] **生成器网络训练**——从"优化一张纹理"升级到"seed→对抗纹理"

### 🟢 中期（1-2月）
- [ ] **ONNX 导出**
- [ ] **UE5 插件集成 demo**
- [ ] **技术白皮书 + Demo 视频**

---

## 诚实的自评

**做到了什么：** 从零到一验证了核心假设——3D纹理对抗攻击能让YOLO检测器完全失效。方向正确，方法可行，学术界有支撑。

**还不够：** 只在一个纹理、一个检测器、16个视角上验证了一次。需要大规模复现确认不是偶然，然后才是生成器和产品化。

**最大的收获：** 知道哪条路能走（梯度投影）、哪条路走不通（2D攻击、NES、PyTorch3D在云GPU上），省下了未来几周的试错成本。

---

## 相关文档
- [[HONEST_STATUS]]
- [[FINAL_PLAN]]
- [[WHITEPAPER]]
- [[LEARNING_ROADMAP]]
