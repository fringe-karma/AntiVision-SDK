// AntiVisionManager.cs
// Unity 游戏引擎集成组件
//
// 使用方式:
//   1. 将 AntiVisionManager 挂到场景中的 GameObject 上
//   2. 配置 GeneratorModel (ONNX via Barracuda / Sentis)
//   3. 在 OnMatchStart 时调用 SetActiveSeed(seed)
//   4. 自动替换受保护角色的纹理
//
// Copyright AntiVision SDK. All Rights Reserved.

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

#if UNITY_SENTIS
using Unity.Sentis;
#endif

namespace AntiVision
{
    /// <summary>
    /// 对抗纹理管理组件 — Unity 端的主入口
    /// </summary>
    public class AntiVisionManager : MonoBehaviour
    {
        [Header("Generator Model")]
        [Tooltip("ONNX generator model asset (for Unity Sentis / Barracuda)")]
        public ModelAsset GeneratorModel;

        [Header("Strategy")]
        public AntiVisionStrategy Strategy = AntiVisionStrategy.TextureReplacement;

        [Header("Targets")]
        [Tooltip("Which GameObjects (tag-based) to protect. Default: 'Player'")]
        public string[] ProtectedTags = { "Player" };

        [Tooltip("Material property name for base color texture")]
        public string BaseTexturePropertyName = "_BaseMap";

        [Header("Post-Process")]
        [Tooltip("Post-process material for screen-space perturbation (optional)")]
        public Material ScreenSpacePerturbationMaterial;

        // ------------------------------------------------------------------
        // State
        // ------------------------------------------------------------------
        [HideInInspector] public long ActiveSeed = 0;
        [HideInInspector] public long LastAppliedSeed = -1;
        private bool _generatorReady = false;
        private bool _seedPending = false;

        // Texture cache: [textureName_seed] -> Texture2D
        private Dictionary<string, Texture2D> _adversarialTextureCache = new();
        private Dictionary<string, Texture2D> _originalTextureCache = new();

        // ------------------------------------------------------------------
        // Unity Lifecycle
        // ------------------------------------------------------------------
        void Start()
        {
            Debug.Log($"[AntiVision] Initialized. Waiting for seed...");
            InitializeGenerator();
        }

        void OnDestroy()
        {
            if (ScreenSpacePerturbationMaterial != null)
            {
                // Cleanup
            }
        }

        // ------------------------------------------------------------------
        // Public API
        // ------------------------------------------------------------------

        /// <summary>
        /// Set the active adversarial seed. Called when server sends match start.
        /// </summary>
        public void SetActiveSeed(long seed)
        {
            if (seed == ActiveSeed && _generatorReady) return;

            ActiveSeed = seed;
            _seedPending = true;

            Debug.Log($"[AntiVision] Seed set to {seed}. " +
                      "Applying on next frame...");

            if (_generatorReady)
            {
                ApplyAdversarialTextures();
            }
        }

        /// <summary>
        /// Apply adversarial textures to all protected objects immediately.
        /// </summary>
        public void ApplyAdversarialTextures()
        {
            if (!_generatorReady)
            {
                Debug.LogWarning("[AntiVision] Generator not ready. Deferred.");
                return;
            }

            if (ActiveSeed == LastAppliedSeed) return;

            Debug.Log($"[AntiVision] Applying adversarial textures " +
                      $"with seed {ActiveSeed}...");

            int count = 0;
            foreach (string tag in ProtectedTags)
            {
                GameObject[] targets = GameObject.FindGameObjectsWithTag(tag);
                foreach (GameObject go in targets)
                {
                    ProtectGameObject(go);
                    count++;
                }
            }

            // Enable screen-space perturbation if configured
            if (Strategy == AntiVisionStrategy.ScreenSpacePerturbation ||
                Strategy == AntiVisionStrategy.Both)
            {
                EnablePostProcessPerturbation();
            }

            LastAppliedSeed = ActiveSeed;
            _seedPending = false;

            Debug.Log($"[AntiVision] Applied to {count} objects. " +
                      $"Seed={ActiveSeed}");
        }

        /// <summary>
        /// Protect a single GameObject by replacing its textures.
        /// </summary>
        public void ProtectGameObject(GameObject go)
        {
            if (go == null) return;

            Renderer[] renderers = go.GetComponentsInChildren<Renderer>();
            foreach (Renderer renderer in renderers)
            {
                foreach (Material mat in renderer.materials)
                {
                    ReplaceMaterialTextures(mat, go.name);
                }
            }
        }

        // ------------------------------------------------------------------
        // Internal
        // ------------------------------------------------------------------

#if UNITY_SENTIS
        private IWorker _generatorWorker;

        private void InitializeGenerator()
        {
            if (GeneratorModel == null)
            {
                Debug.LogWarning("[AntiVision] No generator model. " +
                                 "Using pre-computed texture bank.");
                _generatorReady = true;
                return;
            }

            try
            {
                var runtimeModel = ModelLoader.Load(GeneratorModel);
                _generatorWorker = WorkerFactory.CreateWorker(
                    BackendType.GPUCompute, runtimeModel
                );
                _generatorReady = true;
                Debug.Log("[AntiVision] Generator loaded via Unity Sentis.");
            }
            catch (Exception e)
            {
                Debug.LogError($"[AntiVision] Failed to load generator: {e.Message}");
                _generatorReady = false;
            }
        }

        private Texture2D GenerateAdversarialTexture(Texture2D baseTexture, long seed)
        {
            if (_generatorWorker == null) return baseTexture;

            // Convert seed to tensor [1, 64]
            using var seedTensor = new TensorFloat(
                new TensorShape(1, 64),
                SeedToFloatArray(seed)
            );

            // Convert texture to tensor [1, 3, H, W]
            using var texTensor = TextureToTensor(baseTexture);

            // Run inference
            var inputs = new Dictionary<string, Tensor>
            {
                {"seed", seedTensor},
                {"base_texture", texTensor}
            };

            _generatorWorker.Execute(inputs);

            var output = _generatorWorker.PeekOutput("adversarial_texture")
                as TensorFloat;
            var advTex = TensorToTexture(output, baseTexture.width,
                                         baseTexture.height);

            inputs.Clear();
            output.Dispose();

            return advTex;
        }
#else
        private void InitializeGenerator()
        {
            // No Sentis/Barracuda available → use pre-computed textures
            Debug.Log("[AntiVision] Unity Sentis not available. " +
                      "Using pre-computed adversarial texture bank.");
            _generatorReady = true;
        }

        private Texture2D GenerateAdversarialTexture(
            Texture2D baseTexture, long seed)
        {
            // Fallback: return base texture (client needs pre-computed assets)
            Debug.LogWarning("[AntiVision] Generator not available. " +
                             "Returning base texture.");
            return baseTexture;
        }
#endif

        private void ReplaceMaterialTextures(Material mat, string objectName)
        {
            if (!mat.HasProperty(BaseTexturePropertyName)) return;

            Texture baseTex = mat.GetTexture(BaseTexturePropertyName);
            if (baseTex is not Texture2D baseTex2D) return;

            string cacheKey = $"{objectName}_{baseTex2D.name}_seed{ActiveSeed}";

            // Check cache
            if (_adversarialTextureCache.TryGetValue(
                cacheKey, out Texture2D cachedTex))
            {
                mat.SetTexture(BaseTexturePropertyName, cachedTex);
                return;
            }

            // Generate
            Texture2D advTex = GenerateAdversarialTexture(baseTex2D, ActiveSeed);
            if (advTex != null && advTex != baseTex2D)
            {
                _adversarialTextureCache[cacheKey] = advTex;
                mat.SetTexture(BaseTexturePropertyName, advTex);

                Debug.Log($"[AntiVision] Replaced texture on {objectName}/{mat.name}");
            }
        }

        private void EnablePostProcessPerturbation()
        {
            if (ScreenSpacePerturbationMaterial == null) return;

            // Set seed as material parameters (split 64-bit into two floats)
            float seedLow = (float)(ActiveSeed & 0xFFFFFFFF);
            float seedHigh = (float)((ActiveSeed >> 32) & 0xFFFFFFFF);
            ScreenSpacePerturbationMaterial.SetFloat(
                "_AntiVisionSeedLow", seedLow);
            ScreenSpacePerturbationMaterial.SetFloat(
                "_AntiVisionSeedHigh", seedHigh);

            // Add to camera's post-process stack (URP/HDRP specific)
            // Implementation depends on render pipeline...

            Debug.Log($"[AntiVision] Post-process perturbation enabled. " +
                      $"Seed={ActiveSeed}");
        }

        // ------------------------------------------------------------------
        // Utilities
        // ------------------------------------------------------------------

        private static float[] SeedToFloatArray(long seed)
        {
            float[] bits = new float[64];
            for (int i = 0; i < 64; i++)
                bits[i] = ((seed >> i) & 1);
            return bits;
        }

        private static TensorFloat TextureToTensor(Texture2D tex)
        {
            var pixels = tex.GetPixels();
            // [H, W, 3] → [1, 3, H, W]
            float[] data = new float[1 * 3 * tex.height * tex.width];
            for (int y = 0; y < tex.height; y++)
                for (int x = 0; x < tex.width; x++)
                {
                    int srcIdx = y * tex.width + x;
                    int dstIdxR = 0 * tex.height * tex.width + y * tex.width + x;
                    int dstIdxG = 1 * tex.height * tex.width + y * tex.width + x;
                    int dstIdxB = 2 * tex.height * tex.width + y * tex.width + x;
                    data[dstIdxR] = pixels[srcIdx].r;
                    data[dstIdxG] = pixels[srcIdx].g;
                    data[dstIdxB] = pixels[srcIdx].b;
                }
            return new TensorFloat(new TensorShape(1, 3, tex.height, tex.width), data);
        }

        private static Texture2D TensorToTexture(
            TensorFloat tensor, int w, int h)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            float[] data = tensor.DownloadToArray();
            Color[] pixels = new Color[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float r = data[0 * h * w + y * w + x];
                    float g = data[1 * h * w + y * w + x];
                    float b = data[2 * h * w + y * w + x];
                    pixels[y * w + x] = new Color(r, g, b, 1f);
                }
            tex.SetPixels(pixels);
            tex.Apply();
            return tex;
        }
    }

    /// <summary>
    /// 对抗纹理策略
    /// </summary>
    public enum AntiVisionStrategy
    {
        TextureReplacement,
        ScreenSpacePerturbation,
        Both,
    }

    /// <summary>
    /// Match 级别的 Seed 广播组件
    /// 挂在 NetworkManager 或 GameManager 上，
    /// 服务端下发 seed → 客户端 AntiVisionManager 接收
    /// </summary>
    public class AntiVisionNetworkBridge : MonoBehaviour
    {
        public AntiVisionManager AntiVisionManager;

        /// <summary>
        /// 由网络层调用（如 Mirror / Netcode for GameObjects / Photon）
        /// 服务端: NetworkServer.SendToAll(seedMsg)
        /// 客户端: 收到后调用此方法
        /// </summary>
        public void OnReceiveSeedFromServer(long seed, string matchId)
        {
            Debug.Log($"[AntiVision] Received seed {seed} for match {matchId}");
            if (AntiVisionManager != null)
            {
                AntiVisionManager.SetActiveSeed(seed);
            }
        }

        /// <summary>
        /// 服务端生成 seed 并广播
        /// </summary>
        public long GenerateAndBroadcastSeed()
        {
            // 在实际产品中，seed 应该从游戏服务器 API 获取
            // 这里用客户端本地随机数做演示
            byte[] bytes = new byte[8];
            new System.Random().NextBytes(bytes);
            long seed = BitConverter.ToInt64(bytes, 0);

            Debug.Log($"[AntiVision] Broadcasting seed: {seed}");
            // NetworkServer.SendToAll(new SeedMessage { Seed = seed });
            return seed;
        }
    }
}
