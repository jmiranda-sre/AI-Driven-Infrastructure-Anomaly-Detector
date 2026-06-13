"""Base metric models and schemas for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import numpy as np


class MetricType(str, Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    """Single metric measurement at a point in time."""
    name: str
    value: float
    timestamp: datetime
    server_id: str
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "server_id": self.server_id,
            "labels": self.labels,
            "metric_type": self.metric_type.value,
        }


@dataclass
class MetricBatch:
    """Batch of metric points, typically from one scrape cycle."""
    points: list[MetricPoint] = field(default_factory=list)
    source: str = "unknown"
    scraped_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add(self, point: MetricPoint) -> None:
        self.points.append(point)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def to_numpy_array(self) -> np.ndarray:
        """Convert values to numpy array for ML processing."""
        return np.array([p.value for p in self.points], dtype=np.float64)


@dataclass
class MetricWindow:
    """Sliding window of historical metric values for a single server+metric."""
    server_id: str
    metric_name: str
    values: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    max_size: int = 1000  # keep last N values

    def push(self, value: float, timestamp: datetime) -> None:
        self.values.append(value)
        self.timestamps.append(timestamp)
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size:]
            self.timestamps = self.timestamps[-self.max_size:]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float64)

    @property
    def is_ready(self) -> bool:
        """Check if window has enough data for ML inference."""
        return len(self.values) >= 10

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ServerMetrics:
    """All current metrics for a single server."""
    server_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)

    def to_feature_vector(self, metric_names: list[str] | None = None) -> np.ndarray:
        """Convert to feature vector for ML model input."""
        names = metric_names or sorted(self.metrics.keys())
        return np.array([self.metrics.get(n, 0.0) for n in names], dtype=np.float64)
