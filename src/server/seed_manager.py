"""
AntiVision 服务端 Seed 管理组件

轻量集成：在你的游戏服务器里加几行代码即可。

用法:
  from seed_manager import SeedManager

  mgr = SeedManager()

  # 对局开始时
  seed = mgr.assign_match_seed(match_id)

  # 在 match start 的协议包里下发
  payload = mgr.get_seed_payload(match_id)

  # 协议示例:
  # {
  #   "match_id": "m_abc123",
  #   "texture_seed": 0x8F3A2B1C...,
  #   "generator_version": "1.0.0",
  #   "hmac_seed": "a1b2c3...",     # 白盒保护版本
  #   "nonce": 0x1234...,            # 一次性随机数
  # }

支持模式:
  - 开源: 只在日志中记录 (默认)
  - 生产: HMAC 混淆 seed 传输 (WARN: 需要客户端 whitebox key)
  - DLC: 不下发 seed, 直接下发预生成的整包对抗纹理
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

# 尝试导入 Redis（可选持久化）
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger("antivision.seed")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SeedRecord:
    match_id: str
    seed: int
    generator_version: str
    nonce: int
    created_at: float = field(default_factory=time.time)
    applied_count: int = 0

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "seed": hex(self.seed),
            "generator_version": self.generator_version,
            "nonce": hex(self.nonce),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Seed Manager
# ---------------------------------------------------------------------------

class SeedManager:
    """
    游戏服务端 Seed 管理器。

    三种工作模式:

    MODE_PLAIN (开发)
      seed 明文下发。
      用法: SeedManager(mode="plain")

    MODE_HMAC (生产推荐)
      seed = HMAC-SHA256(shared_secret, client_gpu_uuid + nonce) 的前 64 位。
      攻击者即使截获网络包也无法还原 seed。
      客户端需要用相同的 shared_secret 和 GPU UUID 派生。
      用法: SeedManager(mode="hmac", shared_secret=YOUR_SECRET)

    MODE_DLC (最高安全)
      不下发 seed。服务器直接下发预先生成的加密纹理包。
      客户端只能加载给定的纹理，没有生成器、没有 seed。
      带宽成本高（每局 ~10-50 MB）。
      用法: SeedManager(mode="dlc", texture_cdn_url="https://...")
    """

    MODES = ("plain", "hmac", "dlc")

    def __init__(
        self,
        mode: str = "plain",
        shared_secret: Optional[str] = None,
        generator_version: str = "1.0.0",
        redis_url: Optional[str] = None,
        texture_cdn_url: Optional[str] = None,
        seed_rotation_hours: int = 72,  # 种子轮换（DLC 模式），每 72 小时换一批纹理
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Use one of {self.MODES}")

        self.mode = mode
        self.generator_version = generator_version
        self.seed_rotation_hours = seed_rotation_hours
        self.texture_cdn_url = texture_cdn_url

        if mode == "hmac":
            if not shared_secret:
                raise ValueError("HMAC mode requires shared_secret")
            self.shared_secret = shared_secret.encode() if isinstance(
                shared_secret, str
            ) else shared_secret
        else:
            self.shared_secret = None

        # 种子去重 / 审计日志（可选 Redis）
        self.redis = None
        if redis_url and HAS_REDIS:
            self.redis = redis.from_url(redis_url)

        # 内存缓存
        self._active_seeds: dict[str, SeedRecord] = {}

        logger.info(
            f"[SeedManager] Initialized: mode={mode}, "
            f"version={generator_version}"
        )

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def assign_match_seed(self, match_id: str) -> int:
        """
        为一局游戏分配唯一的对抗纹理种子。

        种子从 2^64 空间随机采样，保证不可预测。
        """
        seed = secrets.randbits(64)

        record = SeedRecord(
            match_id=match_id,
            seed=seed,
            generator_version=self.generator_version,
            nonce=secrets.randbits(64),
        )

        self._active_seeds[match_id] = record

        # 持久化（用于赛后审计和作弊复查）
        if self.redis:
            self.redis.setex(
                f"antivision:seed:{match_id}",
                3600 * 24,  # 24 小时 TTL
                json.dumps(record.to_dict()),
            )

        logger.debug(f"Assigned seed {hex(seed)} to match {match_id}")
        return seed

    def get_seed_payload(
        self, match_id: str
    ) -> dict:
        """
        生成下发给客户端的 payload。

        Returns:
            {
                "match_id": str,
                "texture_seed": int | str,    # 根据 mode 不同
                "nonce": int,
                "generator_version": str,
                "mode": str,
                # DLC mode only:
                "texture_pack_url": str,
                "texture_pack_hash": str,
            }
        """
        record = self._active_seeds.get(match_id)
        if not record:
            # 没有预分配，即时生成
            seed = self.assign_match_seed(match_id)
            record = self._active_seeds[match_id]

        record.applied_count += 1

        payload = {
            "match_id": match_id,
            "nonce": record.nonce,
            "generator_version": record.generator_version,
            "mode": self.mode,
        }

        if self.mode == "plain":
            payload["texture_seed"] = record.seed

        elif self.mode == "hmac":
            # 下发 HMAC(seed, nonce) — 客户端用 GPU UUID 派生真实 seed
            # 简化版：HMAC-SHA256(secret, seed || nonce) 截取前 64 位
            h = hmac.new(
                self.shared_secret,
                record.seed.to_bytes(8, 'big') +
                record.nonce.to_bytes(8, 'big'),
                hashlib.sha256,
            )
            payload["hmac_seed"] = h.hexdigest()
            # 不传 seed 明文

        elif self.mode == "dlc":
            # 不下发 seed。下发预生成的对抗纹理包 URL。
            texture_index = record.seed % 1000  # 1000 套预生成的纹理
            texture_pack_url = (
                f"{self.texture_cdn_url}/"
                f"antivision_textures_batch{texture_index:04d}.pak"
            )
            payload["texture_pack_url"] = texture_pack_url
            payload["texture_pack_hash"] = hashlib.sha256(
                f"antivision_batch{texture_index:04d}_v{self.generator_version}"
                .encode()
            ).hexdigest()

        return payload

    def revoke_seed(self, match_id: str) -> bool:
        """
        撤销已分配的种子（对局取消时调用）。
        确保该 seed 不再被重放。
        """
        if match_id in self._active_seeds:
            del self._active_seeds[match_id]
        if self.redis:
            self.redis.delete(f"antivision:seed:{match_id}")
        logger.info(f"Revoked seed for match {match_id}")
        return True

    def lookup_seed(self, match_id: str) -> Optional[int]:
        """
        查证某局使用的种子（赛后审计用）。
        应先验证调用者身份。
        """
        record = self._active_seeds.get(match_id)
        if record:
            return record.seed

        if self.redis:
            data = self.redis.get(f"antivision:seed:{match_id}")
            if data:
                record_data = json.loads(data)
                return int(record_data["seed"], 16)

        return None

    # ------------------------------------------------------------------
    # 审计与统计
    # ------------------------------------------------------------------

    def get_active_match_count(self) -> int:
        return len(self._active_seeds)

    def get_stats(self) -> dict:
        return {
            "active_matches": len(self._active_seeds),
            "mode": self.mode,
            "generator_version": self.generator_version,
        }

    def cleanup_expired(self, max_age_hours: int = 12):
        """清理超时的种子记录"""
        cutoff = time.time() - max_age_hours * 3600
        expired = [
            mid for mid, rec in self._active_seeds.items()
            if rec.created_at < cutoff
        ]
        for mid in expired:
            del self._active_seeds[mid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired seed records")


# ---------------------------------------------------------------------------
# 游戏服务器集成示例
# ---------------------------------------------------------------------------

# === 三角洲行动 / 任何 UE DS 集成示例 ===
# 在你的 GameMode.on_match_start() 或等价函数中：

def example_ue_dedicated_server_integration():
    """
    伪代码：展示在 UE Dedicated Server 中如何集成。

    ```cpp
    // C++ 端: 在 AGameMode::StartMatch() 中
    void AMyGameMode::StartMatch()
    {
        Super::StartMatch();

        // 请求服务端分配 seed
        FString MatchId = GetMatchId();
        int64 Seed = AntiVisionSeedManager::RequestSeed(MatchId);

        // 通过 RPC 下发给所有客户端
        for (APlayerController* PC : GetWorld()->GetPlayerControllerIterator())
        {
            ClientReceiveAntiVisionSeed(PC, Seed);
        }
    }

    // 客户端收到后
    void AMyPlayerController::ClientReceiveAntiVisionSeed_Implementation(int64 Seed)
    {
        UAntiVisionComponent* AntiVision =
            GetWorld()->GetGameState()->FindComponentByClass<
                UAntiVisionComponent
            >();
        if (AntiVision)
        {
            AntiVision->SetActiveSeed(Seed);
        }
    }
    ```
    """
    pass


# ---------------------------------------------------------------------------
# Web API (FastAPI)
# ---------------------------------------------------------------------------

def example_fastapi_integration():
    """
    如果你的游戏服务器是 web-based (如 Node/Python 后台)：

    ```python
    from fastapi import FastAPI, HTTPException
    from seed_manager import SeedManager

    app = FastAPI()
    mgr = SeedManager(mode="hmac", shared_secret="your-256-bit-secret")

    @app.post("/api/antivision/seed/{match_id}")
    async def get_seed(match_id: str, auth_token: str):
        # 验证 auth_token
        if not verify_auth(auth_token, match_id):
            raise HTTPException(403)

        seed = mgr.assign_match_seed(match_id)
        payload = mgr.get_seed_payload(match_id)
        return payload

    @app.post("/api/antivision/seed/{match_id}/revoke")
    async def revoke_seed(match_id: str, auth_token: str):
        if not verify_auth(auth_token, match_id):
            raise HTTPException(403)
        mgr.revoke_seed(match_id)
        return {"status": "revoked"}
    ```
    """
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AntiVision Seed Manager"
    )
    parser.add_argument("--mode", default="plain",
                        choices=SeedManager.MODES)
    parser.add_argument("--secret", default="",
                        help="Shared secret for HMAC mode")
    parser.add_argument("--generate", action="store_true",
                        help="Generate and print a seed")
    parser.add_argument("--match-id", default="test_match_001",
                        help="Match ID for seed generation")

    args = parser.parse_args()

    mgr = SeedManager(
        mode=args.mode,
        shared_secret=args.secret if args.secret else None,
    )

    if args.generate:
        seed = mgr.assign_match_seed(args.match_id)
        payload = mgr.get_seed_payload(args.match_id)
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nReal seed (server only): {hex(seed)}")

    print(f"Stats: {json.dumps(mgr.get_stats(), indent=2)}")


if __name__ == "__main__":
    main()
