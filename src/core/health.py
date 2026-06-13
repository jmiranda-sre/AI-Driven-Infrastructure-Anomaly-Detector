"""Health check system with dependency verification."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.core.logging import get_logger

logger = get_logger("health")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    detail: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"status": self.status.value, "latency_ms": self.latency_ms}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class HealthReport:
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: str = "unknown"
    uptime_seconds: int = 0
    checks: dict[str, HealthCheckResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        checks = {k: v.to_dict() for k, v in self.checks.items()}
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": checks,
        }


# Registry of health checks
_check_registry: dict[str, Callable[[], Coroutine[Any, Any, HealthCheckResult]]] = {}
_START_TIME = time.time()


def register_health_check(
    name: str,
    fn: Callable[[], Coroutine[Any, Any, HealthCheckResult]]
) -> None:
    """Register an async health check function."""
    _check_registry[name] = fn


async def run_health_checks(version: str = "unknown") -> HealthReport:
    """Execute all registered health checks and aggregate results."""
    report = HealthReport(version=version, uptime_seconds=int(time.time() - _START_TIME))
    overall = HealthStatus.HEALTHY

    for name, check_fn in _check_registry.items():
        try:
            start = time.monotonic()
            result = await asyncio.wait_for(check_fn(), timeout=2.0)
            result.latency_ms = round((time.monotonic() - start) * 1000, 1)
            report.checks[name] = result
            if result.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                overall = HealthStatus.DEGRADED
        except TimeoutError:
            report.checks[name] = HealthCheckResult(
                name=name, status=HealthStatus.UNHEALTHY, detail="timeout"
            )
            overall = HealthStatus.UNHEALTHY
        except Exception as e:
            logger.error("health_check_failed", check=name, error=str(e))
            report.checks[name] = HealthCheckResult(
                name=name, status=HealthStatus.UNHEALTHY, detail=str(e)
            )
            overall = HealthStatus.UNHEALTHY

    report.status = overall
    return report


# ── Built-in checks ──────────────────────────────────────────────

async def _check_database() -> HealthCheckResult:
    """Check PostgreSQL connectivity."""
    try:
        from src.core.database import get_db_pool
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return HealthCheckResult(name="database", status=HealthStatus.HEALTHY)
    except ImportError:
        return HealthCheckResult(name="database", status=HealthStatus.DEGRADED, detail="not configured")
    except Exception as e:
        return HealthCheckResult(name="database", status=HealthStatus.UNHEALTHY, detail=str(e))


async def _check_influxdb() -> HealthCheckResult:
    """Check InfluxDB connectivity."""
    try:
        from src.core.database import get_influxdb_client
        client = get_influxdb_client()
        if client:
            healthy = client.health()
            if healthy and healthy.status == "pass":
                return HealthCheckResult(name="influxdb", status=HealthStatus.HEALTHY)
        return HealthCheckResult(name="influxdb", status=HealthStatus.DEGRADED, detail="not reachable")
    except ImportError:
        return HealthCheckResult(name="influxdb", status=HealthStatus.DEGRADED, detail="not configured")
    except Exception as e:
        return HealthCheckResult(name="influxdb", status=HealthStatus.DEGRADED, detail=str(e))


async def _check_ml_models() -> HealthCheckResult:
    """Check if ML models are loaded."""
    try:
        from src.ml_processing.model_manager import ModelManager
        mm = ModelManager.get_instance()
        loaded = mm.list_loaded_models()
        if loaded:
            return HealthCheckResult(
                name="ml_models", status=HealthStatus.HEALTHY,
                detail=f"{len(loaded)} models loaded"
            )
        return HealthCheckResult(name="ml_models", status=HealthStatus.DEGRADED, detail="no models loaded")
    except Exception as e:
        return HealthCheckResult(name="ml_models", status=HealthStatus.DEGRADED, detail=str(e))


def register_default_checks() -> None:
    """Register default health checks for all subsystems."""
    register_health_check("database", _check_database)
    register_health_check("influxdb", _check_influxdb)
    register_health_check("ml_models", _check_ml_models)
