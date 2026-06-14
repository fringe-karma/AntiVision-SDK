// Copyright AntiVision SDK. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class AntiVision : ModuleRules
{
    public AntiVision(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        bUseUnity = false;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "RenderCore",
            "RHI",
            "NNE",              // Unreal Neural Network Engine (UE 5.2+)
            "NNERuntimeORT",    // ONNX Runtime backend
            "Projects",         // For plugin management
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "Slate",
            "SlateCore",
            "UMG",
        });

        // Optional: DirectML backend for AMD/Intel GPU support
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PrivateDependencyModuleNames.Add("NNERuntimeDML");
        }

        OptimizeCode = CodeOptimization.InShippingBuildsOnly;
    }
}
