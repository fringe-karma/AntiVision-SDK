// AntiVisionComponent.h
// 接入到任意 Character/Actor 上的对抗纹理组件。
//
// 使用方式:
//   1. 将此 Component 挂到 GameMode/GameState 上
//   2. 在 BeginMatch() 时调用 ApplyAdversarialTextures(Seed)
//   3. 组件自动接管所有 Character 的 SkeletalMesh 材质纹理
//
// Copyright AntiVision SDK. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"
#include "AntiVisionComponent.generated.h"

// Forward declarations
class UNNERuntimeORT;
class UNeuralNetwork;
struct FNNEModelRaw;

/**
 * 对抗纹理策略
 */
UENUM(BlueprintType)
enum class EAntiVisionStrategy : uint8
{
    /** 替换角色纹理贴图（推荐，对采集完全不可见） */
    TextureReplacement    UMETA(DisplayName = "Texture Replacement"),

    /** 屏幕空间后处理扰动（双保险） */
    ScreenSpacePerturbation UMETA(DisplayName = "Screen Space Perturbation"),

    /** 两者同时启用 */
    Both                  UMETA(DisplayName = "Both"),
};

/**
 * 对抗纹理生成组件
 *
 * 在 GameMode::BeginPlay 时调用 SetActiveSeed()，组件会：
 *   1. 加载 ONNX 生成器模型
 *   2. 用 seed 生成对抗纹理
 *   3. 替换所有受影响角色身上的纹理
 */
UCLASS(
    ClassGroup = (AntiCheat),
    meta = (BlueprintSpawnableComponent, DisplayName = "AntiVision Component"),
    Blueprintable
)
class ANTIVISION_API UAntiVisionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UAntiVisionComponent();

    // ------------------------------------------------------------------
    // 配置（编辑器内设置）
    // ------------------------------------------------------------------

    /** 对抗策略 */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AntiVision|Config")
    EAntiVisionStrategy Strategy = EAntiVisionStrategy::TextureReplacement;

    /** ONNX 生成器模型资源（.onnx 文件） */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AntiVision|Model")
    TSoftObjectPtr<UObject> GeneratorModelAsset;

    /** 哪些 Class 的角色需要对抗纹理保护 */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AntiVision|Config")
    TArray<TSubclassOf<AActor>> ProtectedActorClasses;

    /** 要替换的材质参数名（通常为 "BaseColorMap" 或 "DiffuseTexture"） */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AntiVision|Config")
    FName BaseTextureParameterName = FName("BaseColorMap");

    /** 后处理材质实例（用于屏幕空间扰动） */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AntiVision|PostProcess",
              meta = (EditCondition = "Strategy != EAntiVisionStrategy::TextureReplacement"))
    TSoftObjectPtr<UMaterialInterface> ScreenSpacePerturbationMaterial;

    // ------------------------------------------------------------------
    // Blueprint API
    // ------------------------------------------------------------------

    /**
     * 设置当前对局的对抗纹理种子。
     * 由服务端在 match start 时下发。
     */
    UFUNCTION(BlueprintCallable, Category = "AntiVision")
    void SetActiveSeed(int64 Seed);

    /**
     * 立即对所有受保护角色应用对抗纹理。
     * 如果种子未变化则跳过（避免重复生成）。
     */
    UFUNCTION(BlueprintCallable, Category = "AntiVision")
    void ApplyAdversarialTextures();

    /**
     * 为指定角色单独生成对抗纹理。
     */
    UFUNCTION(BlueprintCallable, Category = "AntiVision")
    void ProtectActor(AActor* TargetActor);

    /**
     * 检查生成器是否已加载完毕。
     */
    UFUNCTION(BlueprintCallable, Category = "AntiVision")
    bool IsGeneratorReady() const { return bGeneratorLoaded; }

    /**
     * 获取当前种子（调试用）。
     */
    UFUNCTION(BlueprintCallable, Category = "AntiVision|Debug")
    int64 GetActiveSeed() const { return ActiveSeed; }

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    // --- 内部状态 ---
    int64 ActiveSeed = 0;
    int64 LastAppliedSeed = -1;
    bool bGeneratorLoaded = false;
    bool bSeedPending = false;

    // ONNX Runtime 推理实例
    TSharedPtr<UNNERuntimeORT> NNERuntime;

    // 缓存的对抗纹理（Texture2D）
    UPROPERTY()
    TMap<FName, TObjectPtr<UTexture2D>> CachedAdversarialTextures;

    // 缓存的原始纹理（用于恢复）
    UPROPERTY()
    TMap<FName, TObjectPtr<UTexture2D>> OriginalTextures;

    // --- 内部方法 ---

    /** 加载并初始化 ONNX 生成器 */
    bool InitializeGenerator();

    /** 运行生成器推理 */
    bool RunGeneratorInference(
        int64 Seed,
        const TArray<uint8>& BaseTextureData,
        int32 Width,
        int32 Height,
        TArray<uint8>& OutAdversarialData
    );

    /** 扫描并收集所有需要保护的角色 */
    TArray<AActor*> CollectProtectedActors();

    /** 替换单个角色上的纹理 */
    void ReplaceActorTextures(AActor* Actor);

    /** 启用屏幕空间后处理 */
    void EnableScreenSpacePerturbation();

    /** 禁用屏幕空间后处理 */
    void DisableScreenSpacePerturbation();

    /** 从资源文件加载 base texture 数据 */
    bool LoadBaseTextureData(
        UTexture2D* Texture,
        TArray<uint8>& OutData,
        int32& OutWidth,
        int32& OutHeight
    );
};
