"""Alert service — facade for the alerting subsystem.

Orchestrates evaluation and dispatch of anomaly alerts.
Connects ML pipeline output to alert channels.
"""

from __future__ import annotations

from datetime import datetime

from src.alerting.dispatcher import AlertDispatcher
from src.alerting.evaluator import AlertEvaluator
from src.alerting.models import Alert
from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("alerting.service")


class AlertService:
    """High-level alert service — bridges ML pipeline and notification channels."""

    _instance: AlertService | None = None

    def __init__(self, config: dict | None = None):
        self._config = config or get_config()
        self.evaluator = AlertEvaluator(self._config)
        self.dispatcher = AlertDispatcher(self._config)
        self._alert_history: list[Alert] = []
        self._max_history = 1000

    @classmethod
    def get_instance(cls) -> AlertService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    async def process_detection(
        self,
        server_id: str,
        result: DetectionResult,
        timestamp: datetime,
        metric_name: str = "unknown",
        correlation_id: str = "",
    ) -> Alert | None:
        """Main entry point: evaluate detection result and dispatch alerts.

        Called by ML pipeline when anomaly is detected.
        """
        # Evaluate → generates Alert if conditions are met
        alert = self.evaluator.evaluate(
            server_id=server_id,
            result=result,
            timestamp=timestamp,
            metric_name=metric_name,
            correlation_id=correlation_id,
        )

        if alert is None:
            return None

        # Store in history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        # Dispatch to all channels
        dispatch_results = await self.dispatcher.dispatch(alert)

        # Log alert outcome
        logger.info(
            "alert_service.alert_dispatched",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            channels=dispatch_results,
        )

        return alert

    def get_recent_alerts(
        self,
        limit: int = 50,
        severity: str | None = None,
        server_id: str | None = None,
    ) -> list[dict]:
        """Get recent alerts with optional filtering."""
        alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        if server_id:
            alerts = [a for a in alerts if a.server_id == server_id]

        return [a.to_dict() for a in alerts[-limit:]]

    def get_alert_by_id(self, alert_id: str) -> dict | None:
        """Get a specific alert by ID."""
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                return alert.to_dict()
        return None

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                # In production, persist to DB
                logger.info("alert.acknowledged", alert_id=alert_id)
                return True
        return False

    def get_stats(self) -> dict:
        """Get alert statistics."""
        from collections import Counter
        severity_counts = Counter(a.severity.value for a in self._alert_history)
        server_counts = Counter(a.server_id for a in self._alert_history)

        return {
            "total_alerts": len(self._alert_history),
            "by_severity": dict(severity_counts),
            "by_server": dict(server_counts),
            "channels": self.dispatcher.channel_names,
        }
