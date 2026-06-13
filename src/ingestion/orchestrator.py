"""Orchestrator — coordinates all ingestion sources and feeds the processing pipeline."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from src.core.config import get_config
from src.core.logging import get_logger
from src.ingestion.feature_engineering import FeatureEngineer
from src.ingestion.kafka_consumer import KafkaConsumer
from src.ingestion.models import MetricBatch, MetricPoint
from src.ingestion.prometheus_client import PrometheusClient

logger = get_logger("ingestion.orchestrator")


class IngestionOrchestrator:
    """Coordinates metric collection from all configured sources.

    Manages scraping schedules, Kafka consumption, feature engineering,
    and dispatches processed data to ML pipeline.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or get_config()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self.feature_engineer = FeatureEngineer(self._config)
        self._prom_client: PrometheusClient | None = None
        self._kafka_consumer: KafkaConsumer | None = None

        # Callback for processed data — set by ML pipeline
        self._on_data_ready: Callable[
            [str, dict[str, float], datetime], Coroutine[Any, Any, None]
        ] | None = None

        # In-memory metric windows per server
        self._latest_metrics: dict[str, dict[str, float]] = {}

    def register_handler(
        self,
        handler: Callable[[str, dict[str, float], datetime], Coroutine[Any, Any, None]],
    ) -> None:
        """Register async callback for when processed features are ready.

        Handler signature: (server_id, features_dict, timestamp) -> None
        """
        self._on_data_ready = handler

    async def _init_sources(self) -> None:
        """Initialize all enabled ingestion sources."""
        cfg = self._config["ingestion"]

        if cfg["prometheus"]["enabled"]:
            self._prom_client = PrometheusClient(self._config)
            logger.info("orchestrator.prometheus_enabled")

        if cfg["kafka"]["enabled"]:
            self._kafka_consumer = KafkaConsumer(self._config)
            await self._kafka_consumer.start()
            logger.info("orchestrator.kafka_enabled")

    async def start(self) -> None:
        """Start all ingestion loops."""
        await self._init_sources()
        self._running = True

        if self._prom_client:
            task = asyncio.create_task(self._prometheus_loop())
            self._tasks.append(task)

        if self._kafka_consumer:
            task = asyncio.create_task(self._kafka_loop())
            self._tasks.append(task)

        logger.info("orchestrator.started")

    async def stop(self) -> None:
        """Gracefully stop all ingestion."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        if self._prom_client:
            await self._prom_client.close()
        if self._kafka_consumer:
            await self._kafka_consumer.stop()

        logger.info("orchestrator.stopped")

    async def _prometheus_loop(self) -> None:
        """Periodic Prometheus scraping loop."""
        interval = self._config["ingestion"]["prometheus"].get("scrape_interval", 15)
        metrics = self._config["ingestion"]["prometheus"].get("metrics", [])

        while self._running:
            try:
                batch = await self._prom_client.scrape_metrics(metrics)
                await self._process_batch(batch)
            except Exception as e:
                logger.error("orchestrator.prometheus_loop_error", error=str(e))

            await asyncio.sleep(interval)

    async def _kafka_loop(self) -> None:
        """Kafka continuous consumption loop."""
        try:
            await self._kafka_consumer.consume_batch(self._process_batch)
        except Exception as e:
            logger.error("orchestrator.kafka_loop_error", error=str(e))

    async def _process_batch(self, batch: MetricBatch) -> None:
        """Process an incoming metric batch:
        1. Push to feature engineering windows
        2. Compute features per server
        3. Dispatch to registered handler
        """
        self.feature_engineer.push_batch(batch)

        # Group by server for per-server feature computation
        servers_in_batch: dict[str, list[MetricPoint]] = {}
        for point in batch:
            servers_in_batch.setdefault(point.server_id, []).append(point)

        for server_id, points in servers_in_batch.items():
            # Update latest metrics
            for p in points:
                self._latest_metrics.setdefault(server_id, {})[p.name] = p.value

            # Compute features
            features = self.feature_engineer.compute_all_features(server_id)
            if features and self._on_data_ready:
                timestamp = points[0].timestamp if points else datetime.now(UTC)
                try:
                    await self._on_data_ready(server_id, features, timestamp)
                except Exception as e:
                    logger.error(
                        "orchestrator.handler_error",
                        server_id=server_id, error=str(e),
                    )

    def get_latest_metrics(self, server_id: str | None = None) -> dict:
        """Get latest raw metrics, optionally filtered by server."""
        if server_id:
            return {server_id: self._latest_metrics.get(server_id, {})}
        return dict(self._latest_metrics)

    async def backfill(self, days: int = 7) -> list[MetricBatch]:
        """Backfill historical data from Prometheus for training."""
        if not self._prom_client:
            logger.warn("orchestrator.backfill_no_prometheus")
            return []

        metrics = self._config["ingestion"]["prometheus"].get("metrics", [])
        batches = await self._prom_client.backfill(metrics, days=days)

        for batch in batches:
            self.feature_engineer.push_batch(batch)

        total_pts = sum(len(b) for b in batches)
        logger.info("orchestrator.backfill_complete", total_points=total_pts, days=days)
        return batches
