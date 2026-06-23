"""
src/utils/config.py

Central configuration loader for the LLM orchestration platform.
"""

from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent


def load_yaml(filename: str) -> dict:
    path = ROOT_DIR / "configs" / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


_gateway_cfg = load_yaml("gateway_config.yaml")
_db_cfg      = load_yaml("db_config.yaml")


class RoutingConfig:
    simple_model:          str   = _gateway_cfg["routing"]["simple_model"]
    complex_model:         str   = _gateway_cfg["routing"]["complex_model"]
    complexity_threshold:  float = _gateway_cfg["routing"]["complexity_threshold"]


class CacheConfig:
    enabled:             bool  = _gateway_cfg["cache"]["enabled"]
    similarity_threshold: float = _gateway_cfg["cache"]["similarity_threshold"]
    ttl_seconds:          int   = _gateway_cfg["cache"]["ttl_seconds"]
    embedding_model:      str   = _gateway_cfg["cache"]["embedding_model"]


class RateLimitConfig:
    requests_per_minute: int = _gateway_cfg["rate_limit"]["requests_per_minute"]
    burst_size:           int = _gateway_cfg["rate_limit"]["burst_size"]
    algorithm:            str = _gateway_cfg["rate_limit"]["algorithm"]


class RetryConfig:
    max_retries:        int   = _gateway_cfg["retry"]["max_retries"]
    base_delay_seconds:  float = _gateway_cfg["retry"]["base_delay_seconds"]
    max_delay_seconds:   float = _gateway_cfg["retry"]["max_delay_seconds"]
    exponential_base:    float = _gateway_cfg["retry"]["exponential_base"]


class CostConfig:
    simple_model_input_cost:   float = _gateway_cfg["cost"]["simple_model_input_cost"]
    simple_model_output_cost:  float = _gateway_cfg["cost"]["simple_model_output_cost"]
    complex_model_input_cost:  float = _gateway_cfg["cost"]["complex_model_input_cost"]
    complex_model_output_cost: float = _gateway_cfg["cost"]["complex_model_output_cost"]


class RedisConfig:
    host:    str = _db_cfg["redis"]["host"]
    port:    int = _db_cfg["redis"]["port"]
    db:      int = _db_cfg["redis"]["db"]
    cache_db: int = _db_cfg["redis"]["cache_db"]


class PostgresConfig:
    host:     str = _db_cfg["postgres"]["host"]
    port:     int = _db_cfg["postgres"]["port"]
    database: str = _db_cfg["postgres"]["database"]
    user:     str = _db_cfg["postgres"]["user"]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    anthropic_api_key: str = Field(default="")
    postgres_password:  str = Field(default="")
    app_env:            str = Field(default="development")
    log_level:           str = Field(default="INFO")
    hf_token: str = Field(default="")
    hf_model_simple:  str = Field(default="mistralai/Mistral-7B-Instruct-v0.3")
    hf_model_complex: str = Field(default="mistralai/Mixtral-8x7B-Instruct-v0.1")


class Settings:
    routing:    RoutingConfig    = RoutingConfig()
    cache:      CacheConfig      = CacheConfig()
    rate_limit: RateLimitConfig  = RateLimitConfig()
    retry:      RetryConfig      = RetryConfig()
    cost:       CostConfig       = CostConfig()
    redis:      RedisConfig      = RedisConfig()
    postgres:   PostgresConfig   = PostgresConfig()
    env:        EnvSettings      = EnvSettings()
    root_dir:   Path             = ROOT_DIR


settings = Settings()