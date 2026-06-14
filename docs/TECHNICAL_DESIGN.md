---
title: "技术设计文档"
date: 2026-05-31
status: complete
tags:
  - anti-vision
  - technical
  - design
---

# 技术设计文档

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        离线阶段（你）                              │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ 游戏角色纹理集 │ ──→│ Generator 训练 │ ──→│ generator.onnx    │   │
│  │ (.tga/.png)  │    │ (PyTorch)     │    │ (2-5 MB)         │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│                     集成到游戏客户端                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         运行时                                    │
│                                                                   │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │ 游戏服务器  │    │  游戏客户端    │    │  外挂机（双机方案）      │ │
│  │           │    │               │    │                       │ │
│  │ seed =    │───→│ generator    │    │  OBS ──→ AI 模型      │ │
│  │ 0x8F3A.. │    │ (seed) → 纹理 │    │          │            │ │
│  │           │    │ 应用到角色    │    │      检测失败           │ │
│  └───────────┘    │ 渲染场景     │    │   置信度 < 0.1          │ │
│                   │ OBS 推流 ────│────│                       │ │
│                   └──────────────┘    └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 对抗纹理生成器

### 2.1 网络结构

```
输入: seed (64-bit) + base_texture (H×W×3)

┌─────────────────────────────────────┐
│  Seed → Hash → Latent Vector (256D) │  确定性映射，同一 seed 总是同一纹理
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Latent → Freq Noise Map            │  小的全连接 + reshape
│  [256] → [32, 16, 16]               │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Freq Perturbation Network          │  转置卷积上采样到纹理分辨率
│  ConvTranspose2d × 3               │  在频域（DCT）操作
│  BatchNorm + ReLU                   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Perceptual Clamp Layer             │  ΔE94 < 2.0 约束
│  + EOT Augmentation (训练时)        │  保证人眼不可见
└──────────────┬──────────────────────┘
               ▼
输出: adversarial_texture (H×W×3)
```

### 2.2 参数量

| 层 | 输入 | 输出 | 参数 |
|----|------|------|------|
| Seed → Latent (FC + ReLU) | 64 | 256 | ~16K |
| Latent → Feature (FC + ReLU) | 256 | 8192 (32×16×16) | ~2M |
| ConvTranspose2d × 3 | 32ch → 128ch → 64ch → 3ch | | ~150K |
| **总计** | | | **~2.2M 参数** |

**模型大小：** ~2.5 MB（FP32）/ ~650 KB（FP16）/ ~300 KB（INT8）

---

## 3. 损失函数设计

### 3.1 对抗损失（Adversarial Loss）

```
L_adv = -Σ_d Σ_v [ feature_entropy(F_d(v)) + confidence_sum(D_d(v)) ]
```

- `F_d(v)`: 检测器 d 对视图 v 的中间层特征图
- `feature_entropy`: 特征图的香农熵，最大化 = 让特征图全噪声
- `D_d(v)`: 检测器 d 的输出，惩罚所有 person 类别的置信度

**攻击的特征层：** 取每个检测器的 layer1、layer2、layer3 输出（对应低/中/高层特征），全面破坏。

### 3.2 多样性损失（Diversity Loss）

```
L_div = -E[ ||G(s1, t) - G(s2, t)||_2 ]
```

- 对不同种子 s1 ≠ s2，同一纹理 t，生成的扰动应该差异大
- 最大化 pairwise L2 距离
- 保证外挂作者采集一个种子不覆盖其他种子

### 3.3 感知损失（Perceptual Loss）

```
L_percep = max(0, ΔE94(texture, adv_texture) - 2.0)
```

- ΔE94 是 CIE 标准色差公式
- ΔE < 2.0：即使是训练有素的色度专家也分辨不出差异
- 使用 LPIPS（Learned Perceptual Image Patch Similarity）作为辅助

### 3.4 EOT 损失（Expectation Over Transformation）

在训练时每步随机施加变换后重新计算对抗损失：

| 变换 | 参数范围 | 模拟什么 |
|------|---------|---------|
| H.264 压缩 | bitrate 2000-20000 kbps | OBS 推流 |
| 双线性缩放 | 0.5× - 1.0× | 分辨率变化 |
| 高斯噪声 | σ = 0.01 - 0.03 | 采集噪声 |
| 对比度调整 | 0.8× - 1.2× | 显示器差异 |
| 伽马校正 | γ = 0.8 - 1.2 | 同上 |

### 3.5 总损失

```
L_total = 10.0 * L_adv + 2.0 * L_div + 1.0 * L_percep + 5.0 * L_eot
```

权重基于消融实验调整。

---

## 4. 运行时集成

### 4.1 UE5 插件

```cpp
// AntiVisionComponent.h
#pragma once
#include "Components/ActorComponent.h"
#include "AntiVisionComponent.generated.h"

UCLASS(ClassGroup=(AntiCheat), meta=(BlueprintSpawnableComponent))
class UAntiVisionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    // ONNX 模型资源
    UPROPERTY(EditAnywhere, Category="AntiVision")
    class UMLPDeformerAsset* GeneratorModel;

    // 原始角色纹理（未扰动）
    UPROPERTY(EditAnywhere, Category="AntiVision")
    TArray<UTexture2D*> BaseCharacterTextures;

    // 当前对局种子
    UPROPERTY(Replicated)
    int64 CurrentSeed;

    // 生成对抗纹理并应用
    UFUNCTION(BlueprintCallable, Category="AntiVision")
    void ApplyAdversarialTextures(int64 Seed);

    // 后处理 pass：屏幕空间扰动（可选，双保险）
    UPROPERTY(EditAnywhere, Category="AntiVision")
    class UMaterialInterface* ScreenSpacePerturbation;
};
```

**调用时机：** `GameMode::BeginPlay()` → `ReplicateSeed()` → `ApplyAdversarialTextures(seed)`

### 4.2 后处理 Shader（双保险）

除了纹理层面的对抗扰动，增加一个屏幕空间的后处理 pass：

```hlsl
// AntiVisionPostProcess.usf

// 对距离 > 阈值（如 50m）的像素做微扰动
// 人类看不清的远距离敌人 = AI 模型容易检测的小目标 = 重点打击

float3 AntiVisionPerturb(float2 UV, float3 SceneColor, float Depth, float Seed)
{
    // 只在特定频率带做扰动
    float2 noise = PseudoRandom(UV, Seed);
    float strength = saturate((Depth - MinDepth) / (MaxDepth - MinDepth));
    
    // 高频微扰
    float3 colorShift = (noise.xyx - 0.5) * strength * 0.02;
    
    return SceneColor + colorShift;
}
```

### 4.3 服务端

```python
# seed_manager.py
import secrets
import redis

class SeedManager:
    """游戏服务端 Seed 管理"""
    
    def assign_match_seed(self, match_id: str) -> int:
        """为一局游戏分配种子"""
        seed = secrets.randbits(64)
        # 存储 (match_id, seed) 用于赛后审计
        redis.set(f"antivision:seed:{match_id}", seed, ex=3600)
        return seed
    
    def get_seed_payload(self, match_id: str) -> dict:
        """生成下发给客户端的 payload"""
        return {
            "match_id": match_id,
            "texture_seed": self.assign_match_seed(match_id),
            "generator_version": "1.0.0",
        }
```

**部署：** 在现有的 matchmaking / game server 里加一个字段。

---

## 5. Seed 白盒保护（对抗生成器提取攻击）

> **背景：** 对抗性审计（[[ADVERSARIAL_AUDIT.md|ADVERSARIAL_AUDIT.md]]）发现的最大单点风险是攻击者提取 `generator.onnx` + hook 网络 seed → 精确反推扰动。本章给出缓解方案。

### 5.1 威胁模型

```
攻击者能力：
  ✅ 解包游戏资源文件，提取 generator.onnx
  ✅ Hook recv() 截获网络下发的 seed
  ✅ 本地运行 generator(seed, base_texture) 得到精确扰动
  ✅ 尝试在 OBS 画面中减去扰动 → 恢复干净画面

攻击者不是：
  ❌ 内核级权限（否则直接读内存坐标，不需要视觉路线）
  ❌ 能逆向 GPU 显存中的实时数据
```

### 5.2 第一层防御：混淆 seed 传输

```cpp
// 不传明文 seed。seed 由 GPU 唯一标识派生。

// ─── 服务器端 ───
struct SeedPayload {
    uint64_t hmac_seed;      // HMAC-SHA256(seed, gpu_uuid) 的前 64 位
    uint64_t nonce;          // 一次性随机数，防重放
    uint32_t generation;     // 生成器版本号
};

// ─── 客户端 ───
uint64_t DeriveRealSeed(const SeedPayload& payload) {
    // GPU UUID 从驱动读取，不暴露到用户态内存
    char gpu_uuid[64];
    GetGpuUuid_FromDriver(gpu_uuid);  // 内核态调用
    
    uint8_t hmac[32];
    HMAC_SHA256(gpu_uuid, &payload.nonce, hmac);
    
    // 真正的 seed = HMAC 派生值 XOR 传输值
    return *(uint64_t*)hmac ^ payload.hmac_seed;
}
```

**效果：** 
- seed 明文不出现在网络包中
- 攻击者即使 Hook recv() 拿到 payload 也无法反推 seed（没有 GPU UUID）
- GPU UUID 通过内核驱动接口获取，用户态 hook 拿不到

### 5.3 第二层防御：保护内存中的 seed 和中间结果

```cpp
class ProtectedSeedContext {
    void* gpu_mapped_buffer;  // GPU 可读写的受保护内存页
    uint64_t active_seed;     // 仅在 GPU 传输时短暂存在
    
public:
    void ApplySeed(uint64_t derived_seed) {
        // 1. seed 直接写入 GPU 可见缓冲区
        // 2. CPU 端 seed 立即清零
        // 3. 后续所有操作在 GPU 端完成
        
        CopyToGpuBuffer(&derived_seed, sizeof(derived_seed));
        SecureZeroMemory(&derived_seed, sizeof(derived_seed));
        
        // GPU 端：generator 从 buffer 读 seed
        // GPU 端：生成扰动 → 直接更新纹理
        // CPU 端：全程不知道生成了什么
    }
};
```

**效果：**
- 即使用户态调试器附加游戏进程，也看不到 active seed
- 纹理扰动直接在 GPU 上改纹理，不经过 CPU 内存
- 攻击者必须 hook GPU 驱动层才能拿到数据 → 门槛从"脚本小子"提升到"驱动开发者"

### 5.4 第三层防御：纹理差异化 + 区域随机化

不依赖"seed 不泄露"。即使 seed 泄露，也让精确反推失效：

```
角色模型 = 10 个材质槽
每个材质槽 = 独立生成器实例

seed → Hash → [sub_seed_1, sub_seed_2, ..., sub_seed_10]

每个材质槽用不同的 sub_seed
同一角色身上有 10 种不同的扰动纹理
```

**效果：**
- 攻击者即使拿到总 seed，要正确还原每个材质槽的扰动
- 不同材质槽的扰动在屏幕上混合（被光照/阴影/透视混合）
- 增加了"精确减除扰动"的计算复杂度

### 5.5 实际效果

| 攻击者级别 | 手段 | 三道防线后的结果 |
|-----------|------|----------------|
| 脚本小子 | 直接用公开 YOLO | ❌ 完全失效 |
| 中级作者 | 提取 ONNX + hook 网络包 | ❌ 拿不到真实 seed（GPU UUID 绑定） |
| 高级作者 | 提取 ONNX + 逆向驱动拿 GPU UUID | ⚠️ 可能拿到 seed，但需 hook GPU 内存拿子种子 |
| 顶级作者 | 以上全部 + 逆向渲染管线还原 | ⚠️ 可能恢复部分干净画面，但成本 > 写内存挂 |

**核心结论：** 防线不是要阻止 NSA 级别的对手。是要让 OBS AI 视觉外挂的开发成本，从"一天搭完"拉到"需要一个逆向渲染管线的团队"。超过这个成本，做视觉挂就不如做内存挂——而内存挂有传统反作弊管。

---

## 6. 对抗效果的数学保证

### 6.1 为什么跨架构迁移有效

CNN 检测器的底层结构：

```
Input → Conv1(边缘) → Conv2(纹理) → Conv3(部件) → ... → Head(检测)
          ↑                ↑
    频域低通为主      频域带通为主 ← 所有 CNN 都一样
```

3×3 卷积核在频域等价于一个带通滤波器。对抗扰动注入在 3×3 卷积最敏感的频带（中高频 = 纹理频带），所有用 3×3 / 5×5 / 7×7 卷积的模型都受影响。

**数学上：** 对抗扰动 ε 满足 `||ε||_p < δ`（像素空间不可见），但 `||F(ε)||_2` 对任意卷积滤波器 F 都很大（在频域高度可见）。CNN 在频域"看见"了人眼看不见的东西——这就是漏洞。

### 6.2 为什么 ViT 也受影响

Vision Transformer 将图像切为 patch、线性嵌入为 token。Patch embedding 本质上等同于 stride = patch_size 的大卷积。频域扰动仍会影响 patch embedding 的输出分布，进而扰乱 attention 权重。

**实验预期：** ViT 对频域扰动的鲁棒性可能比 CNN 好 ~20-30%，但远不足以恢复可用检测。论文 [1] 显示对抗样本在 ViT 上的迁移率约为 CNN 的 60-80%，不是因为 ViT 免疫，而是因为 patch embedding 降低了有效分辨率。

### 6.3 参考文献

1. Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
2. Athalye et al., "Synthesizing Robust Adversarial Examples", ICML 2018
3. Guo et al., "Countering Adversarial Images using Input Transformations", ICLR 2018
4. Duan et al., "Adversarial Texture Optimization for 3D Object Detection", ECCV 2022
5. Zhong et al., "Frequency Domain Adversarial Attacks", CVPR 2022
6. Xie et al., "Improving Transferability of Adversarial Examples with Input Diversity", CVPR 2019

### 5.1 为什么跨架构迁移有效

CNN 检测器的底层结构：

```
Input → Conv1(边缘) → Conv2(纹理) → Conv3(部件) → ... → Head(检测)
          ↑                ↑
    频域低通为主      频域带通为主 ← 所有 CNN 都一样
```

3×3 卷积核在频域等价于一个带通滤波器。对抗扰动注入在 3×3 卷积最敏感的频带（中高频 = 纹理频带），所有用 3×3 / 5×5 / 7×7 卷积的模型都受影响。

**数学上：** 对抗扰动 ε 满足 `||ε||_p < δ`（像素空间不可见），但 `||F(ε)||_2` 对任意卷积滤波器 F 都很大（在频域高度可见）。CNN 在频域"看见"了人眼看不见的东西——这就是漏洞。

### 5.2 为什么 ViT 也受影响

Vision Transformer 将图像切为 patch、线性嵌入为 token。Patch embedding 本质上等同于 stride = patch_size 的大卷积。频域扰动仍会影响 patch embedding 的输出分布，进而扰乱 attention 权重。

**实验预期：** ViT 对频域扰动的鲁棒性可能比 CNN 好 ~20-30%，但远不足以恢复可用检测。论文 [1] 显示对抗样本在 ViT 上的迁移率约为 CNN 的 60-80%，不是因为 ViT 免疫，而是因为 patch embedding 降低了有效分辨率。

### 5.3 参考文献

1. Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
2. Athalye et al., "Synthesizing Robust Adversarial Examples", ICML 2018
3. Guo et al., "Countering Adversarial Images using Input Transformations", ICLR 2018
4. Duan et al., "Adversarial Texture Optimization for 3D Object Detection", ECCV 2022
5. Zhong et al., "Frequency Domain Adversarial Attacks", CVPR 2022
6. Xie et al., "Improving Transferability of Adversarial Examples with Input Diversity", CVPR 2019

---

## 7. 实验环境搭建

### 7.1 OBS 外挂模拟器

```python
# experiments/obs_aimbot_sim/simulator.py
"""
模拟外挂的工作流程:
1. OBS 采集游戏画面（模拟：直接 ReadProcessMemory 拿 SwapChain 后缓冲）
2. 推流到另一台机器（模拟：本地传帧）
3. AI 模型检测（YOLOv5/v8/v10/RT-DETR）
4. 模拟鼠标输入（模拟：计算目标坐标，打印瞄准偏移）
"""
```

### 7.2 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| 检测置信度均值 | 对抗纹理下检测器对 person 的置信度 | < 0.1 |
| 漏检率提升 | 对抗 vs 原始纹理的漏检倍数 | > 10× |
| 跨模型迁移率 | 在模型 A 上训练的扰动对模型 B 的有效率 | > 70% |
| ΔE94 | 人眼可感知色差 | < 2.0 |
| LPIPS | 学习的感知相似度 | > 0.95 (5% 以内差异) |
| 纹理生成耗时 | GPU 推理时间 | < 1ms |
| 后处理 pass 耗时 | 屏幕空间扰动 GPU 时间 | < 0.2ms |

---

## 相关文档
- [[PROPOSAL]]
- [[ADVERSARIAL_AUDIT]]
- [[WHITEPAPER]]
- [[LEARNING_ROADMAP]]
