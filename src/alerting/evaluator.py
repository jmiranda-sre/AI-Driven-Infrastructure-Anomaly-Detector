"""Alert evaluation — transforms detection results into actionable alerts.

Applies rules, determines severity, generates human-readable messages,
and handles deduplication/suppression.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.alerting.models import Alert, AlertRule, Severity, SuppressionState
from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("alerting.evaluator")


# Suggested actions by metric type
_METRIC_ACTIONS: dict[str, str] = {
    "cpu_usage": "Investigate high CPU usage — check for runaway processes, consider scaling",
    "memory_usage": "Check for memory leaks, consider adding resources or restarting service",
    "disk_io_util": "Investigate high disk I/O — check for heavy writes, disk contention",
    "network_in": "Investigate network ingress spike — check for DDoS or traffic surge",
    "network_out": "Investigate network egress spike — check for data exfiltration or surge",
    "iowait": "Investigate I/O wait — check disk health and optimize I/O patterns",
    "error_rate": "Investigate increased error rate — check application logs for root cause",
    "load_avg_5m": "Investigate high load average — check running processes and resource allocation",
}


class AlertEvaluator:
    """Evaluates detection results and generates structured alerts."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["alerting"]
        # Severity levels with sensible fallback (config key was missing → empty list)
        self.severity_levels = cfg.get("severity_levels", [
            {"name": "info", "min_score": 0.5},
            {"name": "warning", "min_score": 0.7},
            {"name": "critical", "min_score": 0.85},
        ])
        self.suppression_state = SuppressionState()
        self._cooldown_minutes = cfg.get("suppression", {}).get("cooldown_minutes", 30)
        self._max_alerts_per_hour = cfg.get("suppression", {}).get("max_alerts_per_hour", 50)
        self._dedup_window_min = cfg.get("suppression", {}).get("deduplicate_window_minutes", 5)
        self._prediction_confidence = cfg.get("prediction", {}).get("confidence_threshold", 0.6)
        self._rules: list[AlertRule] = []
        # Grafana dashboard URL from config (observability.tracing.jaeger.endpoint region)
        observability_cfg = (config or get_config()).get("observability", {})
        self._grafana_url = observability_cfg.get("grafana_url", "")

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def determine_severity(self, score: float) -> Severity:
        """Map anomaly score to severity level based on config thresholds."""
        for level_cfg in reversed(self.severity_levels):
            if score >= level_cfg.get("min_score", 0.5):
                return Severity(level_cfg["name"])
        return Severity.INFO

    def should_suppress(
        self, server_id: str, metric_name: str, severity: Severity
    ) -> bool:
        """Check if alert should be suppressed (cooldown or rate limiting)."""
        now = datetime.now(UTC)
        key = f"{server_id}::{metric_name}::{severity.value}"

        # Check cooldown
        last_time = self.suppression_state.last_alert_time.get(key)
        if last_time and (now - last_time) < timedelta(minutes=self._cooldown_minutes):
            return True

        # Check hourly rate limit
        # Reset counter every hour
        if (now - self.suppression_state.hourly_reset) > timedelta(hours=1):
            self.suppression_state.alert_counts.clear()
            self.suppression_state.hourly_reset = now

        count = self.suppression_state.alert_counts.get(key, 0)
        if count >= self._max_alerts_per_hour:
            logger.warning("alert.rate_limited", key=key, count=count)
            return True

        return False

    def _record_alert(self, server_id: str, metric_name: str, severity: Severity) -> None:
        """Record that an alert was sent (for suppression tracking)."""
        now = datetime.now(UTC)
        key = f"{server_id}::{metric_name}::{severity.value}"
        self.suppression_state.last_alert_time[key] = now
        self.suppression_state.alert_counts[key] = self.suppression_state.alert_counts.get(key, 0) + 1

    def generate_message(
        self,
        server_id: str,
        metric_name: str,
        score: float,
        details: dict,
        is_predicted: bool,
    ) -> str:
        """Generate human-readable alert message."""
        severity = self.determine_severity(score).value.upper()

        if is_predicted:
            return (
                f"[PREDICTED] {metric_name} on {server_id} is forecast to reach "
                f"anomalous levels within the prediction horizon "
                f"(score: {score:.2f})"
            )

        # Extract current value from model details if available
        current_val = details.get("actual", "unknown")
        if isinstance(current_val, (int, float)):
            current_val = f"{current_val:.1f}%"

        forecast_val = details.get("forecast", "unknown")
        if isinstance(forecast_val, (int, float)):
            forecast_val = f"{forecast_val:.1f}%"

        if forecast_val != "unknown":
            return (
                f"{severity}: {metric_name} on {server_id} at {current_val} "
                f"deviates from predicted normal of {forecast_val} "
                f"(anomaly score: {score:.2f})"
            )

        return (
            f"{severity}: {metric_name} on {server_id} shows anomalous behavior "
            f"(score: {score:.2f}, threshold exceeded)"
        )

    def evaluate(
        self,
        server_id: str,
        result: DetectionResult,
        timestamp: datetime,
        metric_name: str = "unknown",
        correlation_id: str = "",
    ) -> Alert | None:
        """Evaluate a detection result and produce an alert if warranted.

        Args:
            server_id: Server that triggered the detection
            result: Detection result from the ensemble
            timestamp: Time of detection
            metric_name: Primary metric that triggered the alert
            correlation_id: Request correlation ID

        Returns:
            Alert object or None if suppressed or below threshold
        """
        severity = self.determine_severity(result.anomaly_score)

        # Skip info-level alerts unless is_anomaly is True
        if severity == Severity.INFO and not result.is_anomaly:
            return None

        # Check suppression
        if self.should_suppress(server_id, metric_name, severity):
            logger.debug("alert.suppressed", server_id=server_id, metric=metric_name)
            return None

        # Determine if this is a predicted alert
        is_predicted = bool(result.prediction_horizon)

        # Build alert
        message = self.generate_message(
            server_id, metric_name, result.anomaly_score,
            result.details, is_predicted,
        )

        suggested_action = _METRIC_ACTIONS.get(metric_name, "Investigate the anomalous metric behavior")

        alert = Alert(
            server_id=server_id,
            metric_name=metric_name,
            anomaly_score=result.anomaly_score,
            severity=severity,
            is_predicted=is_predicted,
            prediction_horizon=result.prediction_horizon if is_predicted else None,
            message=message,
            suggested_action=suggested_action,
            model_details=result.details,
            dashboard_url=f"{self._grafana_url}/d/server-detail?var-server={server_id}" if self._grafana_url else "",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )

        self._record_alert(server_id, metric_name, severity)
        logger.info(
            "alert.generated",
            alert_id=alert.alert_id,
            server_id=server_id,
            severity=severity.value,
            score=round(result.anomaly_score, 4),
            is_predicted=is_predicted,
        )

        return alert
