"""Configuration management — YAML + env var overrides.

Precedence (highest → lowest):
1. Environment variables (SERVICE_PORT, POSTGRES_HOST, etc.)
2. Environment-specific config file (config/{environment}.yaml)
3. Default config file (config/default.yaml)

Env vars are mapped: nested keys joined by __, e.g. POSTGRES_HOST → database.postgres.host
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_env_key_map: dict[str, list[str]] = {
    "SERVICE_PORT": ["service", "port"],
    "SERVICE_ENV": ["service", "environment"],
    "LOG_LEVEL": ["service", "log_level"],
    "POSTGRES_HOST": ["database", "postgres", "host"],
    "POSTGRES_PORT": ["database", "postgres", "port"],
    "POSTGRES_DB": ["database", "postgres", "name"],
    "POSTGRES_USER": ["database", "postgres", "user"],
    "POSTGRES_PASSWORD": ["database", "postgres", "password"],
    "INFLUXDB_URL": ["database", "timeseries", "influxdb", "url"],
    "INFLUXDB_TOKEN": ["database", "timeseries", "influxdb", "token"],
    "INFLUXDB_ORG": ["database", "timeseries", "influxdb", "org"],
    "INFLUXDB_BUCKET": ["database", "timeseries", "influxdb", "bucket"],
    "PROMETHEUS_URL": ["ingestion", "prometheus", "url"],
    "KAFKA_BOOTSTRAP_SERVERS": ["ingestion", "kafka", "bootstrap_servers"],
    "JWT_SECRET": ["security", "auth", "jwt", "secret"],
    "CORS_ORIGINS": ["security", "cors", "allowed_origins"],
    "SLACK_WEBHOOK_URL": ["alerting", "channels", "slack", "webhook_url"],
    "PAGERDUTY_ROUTING_KEY": ["alerting", "channels", "pagerduty", "routing_key"],
    "ALERT_WEBHOOK_URL": ["alerting", "channels", "webhook", "url"],
    "JAEGER_ENDPOINT": ["observability", "tracing", "jaeger", "endpoint"],
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _deep_set(cfg: dict, keys: list[str], value: Any) -> None:
    """Set a nested dict value from a list of keys."""
    for key in keys[:-1]:
        cfg = cfg.setdefault(key, {})
    cfg[keys[-1]] = value


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_config(environment: str | None = None) -> dict:
    """Load merged configuration from files + env vars."""
    env = environment or os.getenv("SERVICE_ENV", "development")

    default_cfg = _load_yaml(_CONFIG_DIR / "default.yaml")
    env_cfg = _load_yaml(_CONFIG_DIR / f"{env}.yaml")
    cfg = _deep_merge(default_cfg, env_cfg)

    # Apply env var overrides
    for env_key, path in _env_key_map.items():
        val = os.getenv(env_key)
        if val is not None:
            # Type coercion for list fields
            if env_key == "CORS_ORIGINS":
                val = [v.strip() for v in val.split(",")]
            elif env_key in ("POSTGRES_PORT", "SERVICE_PORT"):
                val = int(val)
            _deep_set(cfg, path, val)

    # Also support generic AD__ prefix for arbitrary overrides
    # e.g. AD__ML__MODELS__ISOLATION_FOREST__CONTAMINATION=0.02
    prefix = "AD__"
    for env_key, val in os.environ.items():
        if env_key.startswith(prefix):
            keys = env_key[len(prefix):].lower().split("__")
            _deep_set(cfg, keys, val)

    return cfg


# Singleton config instance
_config: dict | None = None


def get_config() -> dict:
    """Get or initialize the global config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset config singleton (for testing)."""
    global _config
    _config = None
