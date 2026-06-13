"""Prometheus metric ingestion via HTTP API with circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from src.core.circuit_breaker import get_breaker
from src.core.config import get_config
from src.core.errors import PrometheusQueryError, TimeoutError
from src.core.logging import get_logger
from src.ingestion.models import MetricBatch, MetricPoint

logger = get_logger("ingestion.prometheus")


class PrometheusClient:
    """Async Prometheus HTTP API client with circuit breaker protection."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ingestion"]["prometheus"]
        self.base_url = cfg["url"].rstrip("/")
        self.timeout = cfg.get("query_timeout", 10)
        self.breaker = get_breaker("prometheus")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def query(self, promql: str, time: datetime | None = None) -> dict:
        """Execute an instant query against Prometheus.

        Args:
            promql: PromQL query string
            time: Evaluation timestamp (defaults to now)

        Returns:
            Prometheus API response dict
        """
        params: dict[str, Any] = {"query": promql}
        if time:
            params["time"] = time.timestamp()

        async def _do_query():
            client = await self._get_client()
            resp = await client.get("/api/v1/query", params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise PrometheusQueryError(
                    f"Prometheus query failed: {data.get('error', 'unknown')}",
                    context={"query": promql, "response": data},
                )
            return data

        try:
            return await self.breaker.call(_do_query)
        except httpx.TimeoutException as e:
            raise TimeoutError("prometheus_query", self.timeout) from e
        except httpx.HTTPStatusError as e:
            raise PrometheusQueryError(
                f"Prometheus HTTP {e.response.status_code}: {e.response.text[:200]}",
            ) from e

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "60s",
    ) -> dict:
        """Execute a range query against Prometheus.

        Args:
            promql: PromQL query string
            start: Range start timestamp
            end: Range end timestamp
            step: Query resolution step width
        """
        params = {
            "query": promql,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }

        async def _do_query():
            client = await self._get_client()
            resp = await client.get("/api/v1/query_range", params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise PrometheusQueryError(
                    f"Prometheus range query failed: {data.get('error', 'unknown')}",
                )
            return data

        try:
            return await self.breaker.call(_do_query)
        except httpx.TimeoutException as e:
            raise TimeoutError("prometheus_range_query", self.timeout) from e

    async def scrape_metrics(self, metric_configs: list[dict]) -> MetricBatch:
        """Scrape all configured metrics and return a MetricBatch.

        Args:
            metric_configs: List of dicts with 'name' and 'query' keys
        """
        batch = MetricBatch(source="prometheus")
        now = datetime.now(UTC)

        for mc in metric_configs:
            try:
                result = await self.query(mc["query"], time=now)
                for series in result.get("data", {}).get("result", []):
                    labels = series.get("metric", {})
                    server_id = labels.get("instance", labels.get("job", "unknown"))
                    value_str = series.get("value", [None, "0"])[1]
                    try:
                        value = float(value_str)
                    except (ValueError, TypeError):
                        logger.warn("prometheus.invalid_value", metric=mc["name"], value=value_str)
                        continue

                    point = MetricPoint(
                        name=mc["name"],
                        value=value,
                        timestamp=now,
                        server_id=server_id,
                        labels=labels,
                    )
                    batch.add(point)

            except Exception as e:
                logger.error(
                    "prometheus.scrape_failed",
                    metric=mc["name"],
                    error=str(e),
                    query=mc["query"],
                )

        logger.info(
            "prometheus.scrape_complete",
            points=len(batch),
            metrics_queried=len(metric_configs),
        )
        return batch

    async def backfill(
        self, metric_configs: list[dict], days: int = 7, step: str = "300s",
    ) -> list[MetricBatch]:
        """Backfill historical data for model training.

        Returns list of MetricBatch objects, one per metric config.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        batches = []

        for mc in metric_configs:
            try:
                result = await self.query_range(mc["query"], start, end, step)
                batch = MetricBatch(source="prometheus_backfill")

                for series in result.get("data", {}).get("result", []):
                    labels = series.get("metric", {})
                    server_id = labels.get("instance", "unknown")

                    for ts_val_pair in series.get("values", []):
                        ts = datetime.fromtimestamp(ts_val_pair[0], tz=UTC)
                        try:
                            value = float(ts_val_pair[1])
                        except (ValueError, TypeError):
                            continue
                        batch.add(MetricPoint(
                            name=mc["name"], value=value,
                            timestamp=ts, server_id=server_id, labels=labels,
                        ))

                batches.append(batch)
                logger.info(
                    "prometheus.backfill_complete",
                    metric=mc["name"],
                    points=len(batch),
                    days=days,
                )
            except Exception as e:
                logger.error("prometheus.backfill_failed", metric=mc["name"], error=str(e))

        return batches

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
