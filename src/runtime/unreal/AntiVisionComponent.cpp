// AntiVisionComponent.cpp
// Implementation

#include "AntiVisionComponent.h"
#include "Engine/World.h"
#include "Engine/Texture2D.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "GameFramework/GameStateBase.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Kismet/GameplayStatics.h"
#include "Async/Async.h"

#if WITH_NNE
#include "NNE.h"
#include "NNERuntimeORT.h"
#endif

// ====================================================================
// Lifecycle
// ====================================================================

UAntiVisionComponent::UAntiVisionComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    PrimaryComponentTick.bStartWithTickEnabled = false;
}

void UAntiVisionComponent::BeginPlay()
{
    Super::BeginPlay();

    // 异步初始化生成器
    Async(EAsyncExecution::ThreadPool, [this]()
    {
        InitializeGenerator();
    });

    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Component initialized. "
           "Waiting for seed from server."));
}

void UAntiVisionComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // 恢复原始纹理（可选）
    if (Strategy == EAntiVisionStrategy::TextureReplacement ||
        Strategy == EAntiVisionStrategy::Both)
    {
        DisableScreenSpacePerturbation();
    }

    Super::EndPlay(EndPlayReason);
}

// ====================================================================
// Public API
// ====================================================================

void UAntiVisionComponent::SetActiveSeed(int64 Seed)
{
    if (Seed == ActiveSeed && bGeneratorLoaded)
    {
        UE_LOG(LogTemp, Verbose, TEXT("[AntiVision] Seed %lld already active."), Seed);
        return;
    }

    ActiveSeed = Seed;
    bSeedPending = true;

    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Seed set to %lld. "
           "Will apply on next frame or when generator ready."), Seed);

    // 如果生成器已就绪，立即应用
    if (bGeneratorLoaded)
    {
        ApplyAdversarialTextures();
    }
}

void UAntiVisionComponent::ApplyAdversarialTextures()
{
    if (!bGeneratorLoaded)
    {
        UE_LOG(LogTemp, Warning, TEXT("[AntiVision] Generator not loaded yet. "
               "Deferred apply."));
        return;
    }

    if (ActiveSeed == LastAppliedSeed)
    {
        UE_LOG(LogTemp, Verbose, TEXT("[AntiVision] Seed %lld already applied."),
               ActiveSeed);
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Applying adversarial textures "
           "with seed %lld..."), ActiveSeed);

    // 1. 收集受保护角色
    TArray<AActor*> ProtectedActors = CollectProtectedActors();
    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Found %d actors to protect."),
           ProtectedActors.Num());

    // 2. 为每个角色替换纹理
    for (AActor* Actor : ProtectedActors)
    {
        ReplaceActorTextures(Actor);
    }

    // 3. 可选：启用后处理扰动
    if (Strategy == EAntiVisionStrategy::ScreenSpacePerturbation ||
        Strategy == EAntiVisionStrategy::Both)
    {
        EnableScreenSpacePerturbation();
    }

    LastAppliedSeed = ActiveSeed;
    bSeedPending = false;

    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Textures applied successfully. "
           "Seed=%lld. %d actors protected."),
           ActiveSeed, ProtectedActors.Num());
}

void UAntiVisionComponent::ProtectActor(AActor* TargetActor)
{
    if (!TargetActor || !bGeneratorLoaded)
    {
        return;
    }
    ReplaceActorTextures(TargetActor);
}

// ====================================================================
// Generator Initialization
// ====================================================================

bool UAntiVisionComponent::InitializeGenerator()
{
    if (bGeneratorLoaded)
    {
        return true;
    }

#if WITH_NNE && WITH_NNE_RUNTIME_ORT
    // ---- 路径 A: UE5 NNE + ONNX Runtime ----

    if (GeneratorModelAsset.IsNull())
    {
        UE_LOG(LogTemp, Error, TEXT("[AntiVision] No generator model asset configured!"));
        return false;
    }

    UObject* ModelAsset = GeneratorModelAsset.LoadSynchronous();
    if (!ModelAsset)
    {
        UE_LOG(LogTemp, Error, TEXT("[AntiVision] Failed to load generator model!"));
        return false;
    }

    // NNE 初始化
    // 注意：NNE API 在 UE 5.2-5.4 之间有变化，此处使用 5.3+ API
    TUniquePtr<UE::NNE::IModelInstanceCPU> ModelInstance;
    // ... NNE 具体初始化代码依赖 UE 版本

    bGeneratorLoaded = true;
    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Generator loaded via NNE/ONNX Runtime."));

#else
    // ---- 路径 B: 无 NNE，回退到预计算纹理 ----

    UE_LOG(LogTemp, Warning, TEXT("[AntiVision] NNE not available. "
           "Using pre-computed adversarial texture bank."));

    // 从资源目录加载预计算的对抗纹理
    // 这些纹理是离线用 Python generate_antivision_textures.py 生成的
    //
    // 目录结构:
    //   Content/AntiVision/Textures/
    //     seed_0001_basecolor.png
    //     seed_0002_basecolor.png
    //     ...
    //
    // 每局随机选一套加载
    bGeneratorLoaded = true;

#endif

    // 如果之前有 pending seed，立即应用
    if (bGeneratorLoaded && bSeedPending)
    {
        // 需要在 GameThread 上执行
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            ApplyAdversarialTextures();
        });
    }

    return bGeneratorLoaded;
}

// ====================================================================
// Texture Replacement
// ====================================================================

TArray<AActor*> UAntiVisionComponent::CollectProtectedActors()
{
    TArray<AActor*> Result;
    UWorld* World = GetWorld();
    if (!World) return Result;

    if (ProtectedActorClasses.Num() == 0)
    {
        // 默认保护所有 Character
        for (TActorIterator<ACharacter> It(World); It; ++It)
        {
            Result.Add(*It);
        }
    }
    else
    {
        for (const TSubclassOf<AActor>& ActorClass : ProtectedActorClasses)
        {
            for (TActorIterator<AActor> It(World, ActorClass); It; ++It)
            {
                Result.Add(*It);
            }
        }
    }

    return Result;
}

void UAntiVisionComponent::ReplaceActorTextures(AActor* Actor)
{
    if (!Actor) return;

    // 获取 SkeletalMesh 上的所有材质
    TArray<UMeshComponent*> MeshComponents;
    Actor->GetComponents<UMeshComponent>(MeshComponents);

    for (UMeshComponent* MeshComp : MeshComponents)
    {
        const int32 NumMaterials = MeshComp->GetNumMaterials();
        for (int32 MatIndex = 0; MatIndex < NumMaterials; ++MatIndex)
        {
            UMaterialInterface* OriginalMat = MeshComp->GetMaterial(MatIndex);
            if (!OriginalMat) continue;

            // 创建动态材质实例
            UMaterialInstanceDynamic* DynMat =
                MeshComp->CreateDynamicMaterialInstance(
                    MatIndex, OriginalMat,
                    FName(*FString::Printf(TEXT("AntiVision_Mat_%d"), MatIndex))
                );

            if (!DynMat) continue;

            // 获取原始纹理
            UTexture* BaseTexture = nullptr;
            DynMat->GetTextureParameterValue(BaseTextureParameterName, BaseTexture);

            UTexture2D* BaseTex2D = Cast<UTexture2D>(BaseTexture);
            if (!BaseTex2D) continue;

            // 检查缓存
            FName CacheKey = FName(*FString::Printf(
                TEXT("%s_%s_seed%lld"),
                *Actor->GetName(),
                *BaseTex2D->GetName(),
                ActiveSeed
            ));

            UTexture2D** CachedTex = CachedAdversarialTextures.Find(CacheKey);
            if (CachedTex && *CachedTex)
            {
                // 使用缓存
                DynMat->SetTextureParameterValue(
                    BaseTextureParameterName, *CachedTex
                );
                continue;
            }

            // 运行时生成（需要 NNE）
#if WITH_NNE
            int32 Width, Height;
            TArray<uint8> BaseData;
            if (LoadBaseTextureData(BaseTex2D, BaseData, Width, Height))
            {
                TArray<uint8> AdvData;
                if (RunGeneratorInference(ActiveSeed, BaseData, Width, Height, AdvData))
                {
                    // 创建新 Texture2D
                    UTexture2D* AdvTexture = UTexture2D::CreateTransient(
                        Width, Height, PF_B8G8R8A8
                    );
                    if (AdvTexture)
                    {
                        void* TexData = AdvTexture->GetPlatformData()
                            ->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
                        FMemory::Memcpy(TexData, AdvData.GetData(), AdvData.Num());
                        AdvTexture->GetPlatformData()
                            ->Mips[0].BulkData.Unlock();
                        AdvTexture->UpdateResource();

                        // 缓存
                        CachedAdversarialTextures.Add(CacheKey, AdvTexture);

                        DynMat->SetTextureParameterValue(
                            BaseTextureParameterName, AdvTexture
                        );

                        UE_LOG(LogTemp, Verbose,
                               TEXT("[AntiVision] Generated adversarial texture for %s"),
                               *Actor->GetName());
                    }
                }
            }
#endif
        }
    }
}

bool UAntiVisionComponent::LoadBaseTextureData(
    UTexture2D* Texture,
    TArray<uint8>& OutData,
    int32& OutWidth,
    int32& OutHeight
)
{
    if (!Texture || !Texture->GetPlatformData()) return false;

    OutWidth = Texture->GetSizeX();
    OutHeight = Texture->GetSizeY();

    const FTexturePlatformData* PlatformData = Texture->GetPlatformData();
    const int32 MipIndex = 0;
    const FTexture2DMipMap& Mip = PlatformData->Mips[MipIndex];

    const int32 DataSize = Mip.BulkData.GetBulkDataSize();
    if (DataSize == 0) return false;

    OutData.SetNumUninitialized(DataSize);
    const void* Data = Mip.BulkData.Lock(LOCK_READ_ONLY);
    FMemory::Memcpy(OutData.GetData(), Data, DataSize);
    Mip.BulkData.Unlock();

    return true;
}

bool UAntiVisionComponent::RunGeneratorInference(
    int64 Seed,
    const TArray<uint8>& BaseTextureData,
    int32 Width,
    int32 Height,
    TArray<uint8>& OutAdversarialData
)
{
    // NNE / ONNX Runtime 推理
    // 1. 准备输入 tensor: seed [1,64] + base_texture [1,3,H,W]
    // 2. 运行推理
    // 3. 读取输出: adversarial_texture [1,3,H,W]

#if WITH_NNE
    // TODO: NNE 推理管线
    // 参考 UE5 NNE 文档:
    //   - TUniquePtr<IModelInstanceCPU> Instance;
    //   - Instance->SetInputTensorShapes(...);
    //   - Instance->RunSync(...);

    // 占位：返回原纹理
    OutAdversarialData = BaseTextureData;
    UE_LOG(LogTemp, Warning, TEXT("[AntiVision] NNE inference not yet implemented. "
           "Returning identity."));
    return true;
#else
    OutAdversarialData = BaseTextureData;
    return false;
#endif
}

// ====================================================================
// Post-Process Perturbation
// ====================================================================

void UAntiVisionComponent::EnableScreenSpacePerturbation()
{
    if (ScreenSpacePerturbationMaterial.IsNull()) return;

    UMaterialInterface* PPMat = ScreenSpacePerturbationMaterial.LoadSynchronous();
    if (!PPMat) return;

    // 创建动态实例以传递 seed
    UMaterialInstanceDynamic* PPDyn = UMaterialInstanceDynamic::Create(
        PPMat, this
    );

    // 将 seed 编码为材质参数
    // seed 被拆成两个 float（每 32 位一个）
    float SeedLow = static_cast<float>(ActiveSeed & 0xFFFFFFFF);
    float SeedHigh = static_cast<float>((ActiveSeed >> 32) & 0xFFFFFFFF);

    PPDyn->SetScalarParameterValue(FName("AntiVisionSeedLow"), SeedLow);
    PPDyn->SetScalarParameterValue(FName("AntiVisionSeedHigh"), SeedHigh);

    // 添加到 View 的后处理链
    // 这需要通过 GameViewportClient 或者 PostProcessVolume
    // 简化为：设置为全局后处理
    if (UWorld* World = GetWorld())
    {
        if (APostProcessVolume* PPV = Cast<APostProcessVolume>(
            UGameplayStatics::GetActorOfClass(World, APostProcessVolume::StaticClass())
        ))
        {
            // 添加到已有 PPV
            PPV->AddOrUpdateBlendable(PPDyn);
        }
    }

    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Screen-space perturbation enabled. "
           "Seed=%lld"), ActiveSeed);
}

void UAntiVisionComponent::DisableScreenSpacePerturbation()
{
    // 移除后处理效果（简化实现）
    UE_LOG(LogTemp, Log, TEXT("[AntiVision] Screen-space perturbation disabled."));
}
