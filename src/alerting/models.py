"""Alert models — structured alert format with severity, context, and actionable info."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Structured anomaly alert — the primary output of the system.

    Example JSON output:
    {
      "alert_id": "alt_abc123",
      "server_id": "prod-web-01",
      "metric_name": "cpu_usage",
      "anomaly_score": 0.87,
      "severity": "critical",
      "is_predicted": false,
      "prediction_horizon": null,
      "message": "CPU usage at 97.3% is significantly above predicted normal range of 45-65%",
      "suggested_action": "Scale horizontally or investigate runaway process",
      "model_details": {
        "ensemble": {"score": 0.87, "is_anomaly": true, "method": "weighted"},
        "isolation_forest": {"score": 0.91, "is_anomaly": true},
        "lstm_autoencoder": {"score": 0.85, "is_anomaly": true, "reconstruction_error": 0.042}
      },
      "dashboard_url": "http://grafana:3000/d/server-detail?var-server=prod-web-01",
      "timestamp": "2026-06-13T10:30:00Z",
      "correlation_id": "req_abc123"
    }
    """
    alert_id: str = field(default_factory=lambda: f"alt_{uuid4().hex[:12]}")
    server_id: str = ""
    metric_name: str = ""
    anomaly_score: float = 0.0
    severity: Severity = Severity.INFO
    is_predicted: bool = False
    prediction_horizon: dict[str, Any] | None = None
    message: str = ""
    suggested_action: str = ""
    model_details: dict[str, Any] = field(default_factory=dict)
    dashboard_url: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str = ""
    acknowledged: bool = False
    acknowledged_by: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "server_id": self.server_id,
            "metric_name": self.metric_name,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity.value,
            "is_predicted": self.is_predicted,
            "prediction_horizon": self.prediction_horizon,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "model_details": self.model_details,
            "dashboard_url": self.dashboard_url,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AlertRule:
    """Configurable rule for alert generation."""
    id: int | None = None
    name: str = ""
    metric_pattern: str = "*"     # glob-style match
    condition_type: str = "anomaly_score"  # anomaly_score | predicted_anomaly
    threshold: float = 0.7
    severity: Severity = Severity.WARNING
    cooldown_min: int = 30
    enabled: bool = True

    def matches(self, metric_name: str) -> bool:
        """Check if this rule matches the given metric name."""
        import fnmatch
        return fnmatch.fnmatch(metric_name, self.metric_pattern)


@dataclass
class SuppressionState:
    """Track alert suppression to prevent alert storms."""
    last_alert_time: dict[str, datetime] = field(default_factory=dict)  # key -> timestamp
    alert_counts: dict[str, int] = field(default_factory=dict)  # key -> hourly count
    hourly_reset: datetime = field(default_factory=lambda: datetime.now(UTC))
