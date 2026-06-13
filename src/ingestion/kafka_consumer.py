"""Kafka consumer for metric ingestion with async deserialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from src.core.circuit_breaker import get_breaker
from src.core.config import get_config
from src.core.errors import KafkaIngestionError
from src.core.logging import get_logger
from src.ingestion.models import MetricBatch, MetricPoint, MetricType

logger = get_logger("ingestion.kafka")


class KafkaConsumer:
    """Async Kafka consumer for infrastructure metrics.

    Uses aiokafka for non-blocking consumption with circuit breaker
    protection on connection.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ingestion"]["kafka"]
        self.bootstrap_servers = cfg["bootstrap_servers"]
        self.topic = cfg["topic"]
        self.group_id = cfg["consumer_group"]
        self.auto_offset_reset = cfg.get("auto_offset_reset", "latest")
        self.max_poll_records = cfg.get("max_poll_records", 500)
        self.session_timeout = cfg.get("session_timeout_ms", 30000)
        self.breaker = get_breaker("kafka")
        self._consumer = None
        self._running = False

    async def start(self) -> None:
        """Initialize and start the Kafka consumer."""
        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                max_poll_records=self.max_poll_records,
                session_timeout_ms=self.session_timeout,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                enable_auto_commit=False,
            )
            await self._consumer.start()
            self._running = True
            logger.info("kafka.consumer_started", topic=self.topic, group=self.group_id)
        except ImportError:
            logger.warn("kafka.aiokafka_not_installed")
            raise KafkaIngestionError("aiokafka package not installed — pip install aiokafka") from None
        except Exception as e:
            self.breaker.record_failure()
            raise KafkaIngestionError(f"Failed to start Kafka consumer: {e}") from e

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info("kafka.consumer_stopped")

    def parse_metric_message(self, msg_value: dict) -> MetricPoint | None:
        """Parse a Kafka message into a MetricPoint.

        Expected message format:
        {
            "name": "cpu_usage",
            "value": 75.3,
            "timestamp": "2026-06-13T10:00:00Z",
            "server_id": "server-01",
            "labels": {"instance": "10.0.0.1:9100"},
            "type": "gauge"
        }
        """
        try:
            ts = msg_value.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=UTC)
            else:
                ts = datetime.now(UTC)

            return MetricPoint(
                name=msg_value["name"],
                value=float(msg_value["value"]),
                timestamp=ts,
                server_id=msg_value.get("server_id", "unknown"),
                labels=msg_value.get("labels", {}),
                metric_type=MetricType(msg_value.get("type", "gauge")),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warn("kafka.parse_failed", error=str(e), raw=msg_value)
            return None

    async def consume_batch(
        self,
        handler: Callable[[MetricBatch], Coroutine[Any, Any, None]],
        timeout_ms: int = 1000,
    ) -> None:
        """Consume messages in batches and call handler for each batch.

        Args:
            handler: Async function to process each MetricBatch
            timeout_ms: Max time to wait for messages per poll
        """
        if not self._consumer:
            raise KafkaIngestionError("Consumer not started — call start() first")

        batch = MetricBatch(source="kafka")

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                point = self.parse_metric_message(msg.value)
                if point:
                    batch.add(point)

                # Flush batch when reaching max_poll_records or periodically
                if len(batch) >= self.max_poll_records:
                    await handler(batch)
                    await self._consumer.commit()
                    batch = MetricBatch(source="kafka")

        except Exception as e:
            logger.error("kafka.consume_error", error=str(e))
            self.breaker.record_failure()
            raise KafkaIngestionError(f"Consume error: {e}") from e

        # Flush remaining
        if batch.points:
            await handler(batch)
