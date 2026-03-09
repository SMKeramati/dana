"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


class DatabaseConfig:
    host: str = _get("POSTGRES_HOST", "localhost")
    port: int = _get_int("POSTGRES_PORT", 5432)
    db: str = _get("POSTGRES_DB", "dana")
    user: str = _get("POSTGRES_USER", "dana_admin")
    password: str = _get("POSTGRES_PASSWORD", "")

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisConfig:
    host: str = _get("REDIS_HOST", "localhost")
    port: int = _get_int("REDIS_PORT", 6379)
    password: str = _get("REDIS_PASSWORD", "")

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/0"
        return f"redis://{self.host}:{self.port}/0"


class RabbitMQConfig:
    host: str = _get("RABBITMQ_HOST", "localhost")
    port: int = _get_int("RABBITMQ_PORT", 5672)
    user: str = _get("RABBITMQ_USER", "guest")
    password: str = _get("RABBITMQ_PASSWORD", "guest")
    vhost: str = _get("RABBITMQ_VHOST", "dana")

    @property
    def url(self) -> str:
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"


class AuthConfig:
    secret_key: str = _get("JWT_SECRET_KEY", "change-me")
    algorithm: str = _get("JWT_ALGORITHM", "HS512")
    api_key_salt: str = _get("API_KEY_SALT", "change-me")
    encryption_key: str = _get("ENCRYPTION_KEY", "0" * 32)
    token_expiry_minutes: int = _get_int("TOKEN_EXPIRY_MINUTES", 60)


class ModelConfig:
    name: str = _get("MODEL_NAME", "qwen3-235b-moe")
    path: str = _get("MODEL_PATH", "/models/qwen3-235b-moe-q4")
    draft_path: str = _get("DRAFT_MODEL_PATH", "/models/qwen3-0.6b")
    gpu_devices: str = _get("GPU_DEVICES", "0,1")
    max_batch_size: int = _get_int("MAX_BATCH_SIZE", 16)
    max_context_length: int = _get_int("MAX_CONTEXT_LENGTH", 32768)
    speculative_lookahead: int = _get_int("SPECULATIVE_LOOKAHEAD", 8)


class RateLimitConfig:
    free_rpm: int = _get_int("FREE_TIER_RPM", 5)
    free_tpd: int = _get_int("FREE_TIER_TPD", 1000)
    pro_rpm: int = _get_int("PRO_TIER_RPM", 60)
    pro_tpd: int = _get_int("PRO_TIER_TPD", 100000)


class AppConfig:
    environment: str = _get("ENVIRONMENT", "development")
    debug: bool = _get("DEBUG", "false").lower() == "true"
    log_level: str = _get("LOG_LEVEL", "INFO")


# Singleton instances
db = DatabaseConfig()
redis = RedisConfig()
rabbitmq = RabbitMQConfig()
auth = AuthConfig()
model = ModelConfig()
rate_limit = RateLimitConfig()
app = AppConfig()
