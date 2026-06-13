"""Unit tests for alerting module."""

from datetime import UTC, datetime

from src.alerting.evaluator import AlertEvaluator
from src.alerting.models import Alert, AlertRule, Severity
from src.ml_processing.model_manager import DetectionResult


class TestAlertModel:
    def test_creation(self):
        alert = Alert(
            server_id="prod-web-01",
            metric_name="cpu_usage",
            anomaly_score=0.87,
            severity=Severity.CRITICAL,
            message="CPU anomaly detected",
        )
        assert alert.alert_id.startswith("alt_")
        assert alert.severity == Severity.CRITICAL

    def test_to_dict(self):
        alert = Alert(
            server_id="s1",
            metric_name="mem",
            anomaly_score=0.6,
            severity=Severity.WARNING,
        )
        d = alert.to_dict()
        assert d["server_id"] == "s1"
        assert d["metric_name"] == "mem"
        assert d["severity"] == "warning"

    def test_to_json(self):
        alert = Alert(server_id="s1", metric_name="cpu", anomaly_score=0.5)
        j = alert.to_json()
        assert "alert_id" in j
        assert "cpu" in j


class TestAlertRule:
    def test_matches_exact(self):
        rule = AlertRule(metric_pattern="cpu_usage")
        assert rule.matches("cpu_usage")

    def test_matches_wildcard(self):
        rule = AlertRule(metric_pattern="cpu_*")
        assert rule.matches("cpu_usage")
        assert rule.matches("cpu_idle")
        assert not rule.matches("memory_usage")


class TestAlertEvaluator:
    def _make_evaluator(self):
        evaluator = AlertEvaluator.__new__(AlertEvaluator)
        evaluator.severity_levels = [
            {"name": "info", "min_score": 0.5, "color": "#36a3eb"},
            {"name": "warning", "min_score": 0.7, "color": "#f5a623"},
            {"name": "critical", "min_score": 0.85, "color": "#e74c3c"},
        ]
        evaluator.suppression_state = type("SuppressionState", (), {
            "last_alert_time": {},
            "alert_counts": {},
            "hourly_reset": datetime.now(UTC),
        })()
        evaluator._cooldown_minutes = 30
        evaluator._max_alerts_per_hour = 50
        evaluator._dedup_window_min = 5
        evaluator._prediction_confidence = 0.6
        evaluator._rules = []
        evaluator._grafana_url = ""
        return evaluator

    def test_determine_severity_info(self):
        ev = self._make_evaluator()
        assert ev.determine_severity(0.55) == Severity.INFO

    def test_determine_severity_warning(self):
        ev = self._make_evaluator()
        assert ev.determine_severity(0.75) == Severity.WARNING

    def test_determine_severity_critical(self):
        ev = self._make_evaluator()
        assert ev.determine_severity(0.9) == Severity.CRITICAL

    def test_suppression_cooldown(self):
        ev = self._make_evaluator()
        # First alert should not be suppressed
        assert not ev.should_suppress("s1", "cpu", Severity.WARNING)
        # Record alert
        ev._record_alert("s1", "cpu", Severity.WARNING)
        # Same key should be suppressed within cooldown
        assert ev.should_suppress("s1", "cpu", Severity.WARNING)

    def test_evaluate_creates_alert(self):
        ev = self._make_evaluator()
        result = DetectionResult(
            model_name="ensemble",
            anomaly_score=0.85,
            is_anomaly=True,
            details={"forecast": 45.0, "actual": 92.0},
        )
        alert = ev.evaluate("server-01", result, datetime.now(UTC), metric_name="cpu_usage")
        assert alert is not None
        assert alert.severity == Severity.CRITICAL
        assert alert.server_id == "server-01"

    def test_evaluate_skips_normal(self):
        ev = self._make_evaluator()
        result = DetectionResult(
            model_name="ensemble",
            anomaly_score=0.3,
            is_anomaly=False,
        )
        alert = ev.evaluate("server-01", result, datetime.now(UTC))
        assert alert is None
