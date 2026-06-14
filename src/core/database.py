"""Database connection management — PostgreSQL (asyncpg) + InfluxDB.

Both asyncpg and influxdb-client are optional (part of [full] extras).
Import guards allow the app to start without them for development/test.
"""

from __future__ import annotations

from src.core.config import get_config
from src.core.errors import DatabaseError
from src.core.logging import get_logger

logger = get_logger("database")

# Lazy imports — only import when actually needed
asyncpg = None
InfluxDBClient = None
SYNCHRONOUS = None

def _lazy_import_asyncpg():
    global asyncpg
    if asyncpg is None:
        try:
            import asyncpg as _asyncpg
            asyncpg = _asyncpg
        except ImportError as e:
            raise ImportError("asyncpg is not installed — install with: pip install anomaly-detector[full]") from e
    return asyncpg


def _lazy_import_influxdb():
    global InfluxDBClient, SYNCHRONOUS
    if InfluxDBClient is None:
        try:
            from influxdb_client import InfluxDBClient as InfluxDBClientType
            from influxdb_client.client.write_api import SYNCHRONOUS as SYNC_WRITE
            InfluxDBClient = InfluxDBClientType
            SYNCHRONOUS = SYNC_WRITE
        except ImportError:
            return False
    return True


_pg_pool = None
_influx_client = None


async def get_db_pool():
    """Get or create the PostgreSQL connection pool."""
    global _pg_pool
    apg = _lazy_import_asyncpg()
    if _pg_pool is None or _pg_pool._closed:
        cfg = get_config()["database"]["postgres"]
        logger.info(
            "database.connecting",
            host=cfg["host"], port=cfg["port"], database=cfg["name"],
        )
        try:
            _pg_pool = await apg.create_pool(
                host=cfg["host"],
                port=cfg["port"],
                database=cfg["name"],
                user=cfg["user"],
                password=cfg["password"],
                min_size=2,
                max_size=cfg["pool_size"],
                max_inactive_connection_lifetime=300,
                command_timeout=cfg.get("statement_timeout", 5000) / 1000,
            )
        except Exception as e:
            logger.error("database.connection_failed", error=str(e))
            raise DatabaseError(f"Failed to connect to PostgreSQL: {e}") from e
    return _pg_pool


def get_influxdb_client():
    """Get or create the InfluxDB client."""
    global _influx_client
    if not _lazy_import_influxdb():
        logger.warning("influxdb.not_available", detail="influxdb-client not installed")
        return None
    if _influx_client is None:
        try:
            cfg = get_config()["database"]["timeseries"]["influxdb"]
            _influx_client = InfluxDBClient(
                url=cfg["url"],
                token=cfg["token"],
                org=cfg["org"],
                timeout=cfg.get("timeout", 10) * 1000,
            )
        except Exception as e:
            logger.warning("influxdb.connection_failed", error=str(e))
            return None
    return _influx_client


def get_influxdb_write_api():
    """Get InfluxDB synchronous write API."""
    client = get_influxdb_client()
    if client and SYNCHRONOUS is not None:
        return client.write_api(write_options=SYNCHRONOUS)
    return None


def get_influxdb_query_api():
    """Get InfluxDB query API."""
    client = get_influxdb_client()
    if client:
        return client.query_api()
    return None


async def close_db() -> None:
    """Gracefully close all database connections."""
    global _pg_pool, _influx_client
    if _pg_pool and not _pg_pool._closed:
        await _pg_pool.close()
        logger.info("database.pool_closed")
    if _influx_client:
        _influx_client.close()
        _influx_client = None
        logger.info("influxdb.client_closed")


# ── Schema DDL ───────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    version         VARCHAR(50) NOT NULL,
    algorithm       VARCHAR(50) NOT NULL,
    path            TEXT NOT NULL,
    metrics         JSONB DEFAULT '{}',
    config          JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    metric_pattern  VARCHAR(200) NOT NULL,
    condition_type  VARCHAR(20) NOT NULL,
    threshold       FLOAT NOT NULL,
    severity        VARCHAR(20) NOT NULL DEFAULT 'warning',
    cooldown_min    INT DEFAULT 30,
    enabled         BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    rule_id         INT REFERENCES alert_rules(id),
    server_id       VARCHAR(100) NOT NULL,
    metric_name     VARCHAR(100) NOT NULL,
    anomaly_score   FLOAT NOT NULL,
    severity        VARCHAR(20) NOT NULL,
    message         TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    acknowledged    BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alerts_server_time ON alerts(server_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active) WHERE is_active = TRUE;
"""


async def init_schema(pool=None) -> None:
    """Initialize database schema (idempotent). Safe no-op if asyncpg unavailable."""
    try:
        if pool is None:
            pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("database.schema_initialized")
    except ImportError:
        logger.warning("database.schema_skipped", detail="asyncpg not available")
    except Exception as e:
        logger.warning("database.schema_init_failed", error=str(e))
