"""Ensemble detector — combines multiple anomaly models for robust detection.

Supports three ensemble methods:
- weighted: Weighted average of anomaly scores
- majority: Anomaly if majority of models agree
- any: Anomaly if ANY model flags it (high recall, low precision)

Also integrates the forecaster for predictive alerting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("ml.ensemble")


class EnsembleDetector:
    """Ensemble anomaly detector combining multiple model outputs."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ml"]["inference"]
        self.method = cfg.get("ensemble_method", "weighted")
        self.weights = cfg.get("weights", {
            "isolation_forest": 0.4,
            "lstm_autoencoder": 0.4,
            "holt_winters": 0.2,
        })
        self._models: dict[str, Any] = {}
        self._max_latency_ms = cfg.get("max_latency_ms", 100)

    def register_model(self, name: str, model: Any, weight: float | None = None) -> None:
        """Register a detector model in the ensemble."""
        self._models[name] = model
        if weight is not None:
            self.weights[name] = weight
        logger.info("ensemble.model_registered", name=name, weight=self.weights.get(name, 0))

    @property
    def active_models(self) -> list[str]:
        return [name for name, model in self._models.items() if getattr(model, "is_trained", False)]

    def detect(self, features: np.ndarray, server_id: str = "") -> DetectionResult:
        """Run ensemble detection across all registered models.

        Args:
            features: Feature vector (1D or 2D)
            server_id: Server identifier for logging

        Returns:
            Aggregated DetectionResult
        """
        results: list[DetectionResult] = []
        active = self.active_models

        if not active:
            return DetectionResult(
                model_name="ensemble", anomaly_score=0.0, is_anomaly=False,
                details={"error": "no trained models in ensemble"},
            )

        for name in active:
            model = self._models[name]
            try:
                result = model.detect(features)
                results.append(result)
            except Exception as e:
                logger.error("ensemble.model_error", model=name, error=str(e))

        if not results:
            return DetectionResult(
                model_name="ensemble", anomaly_score=0.0, is_anomaly=False,
                details={"error": "all models failed"},
            )

        # Aggregate based on method
        if self.method == "weighted":
            return self._weighted_ensemble(results)
        elif self.method == "majority":
            return self._majority_ensemble(results)
        elif self.method == "any":
            return self._any_ensemble(results)
        else:
            return self._weighted_ensemble(results)

    def _weighted_ensemble(self, results: list[DetectionResult]) -> DetectionResult:
        """Weighted average of anomaly scores."""
        total_weight = 0.0
        weighted_score = 0.0
        model_details = {}

        for r in results:
            w = self.weights.get(r.model_name, 1.0 / len(results))
            weighted_score += r.anomaly_score * w
            total_weight += w
            model_details[r.model_name] = {
                "score": round(r.anomaly_score, 4),
                "is_anomaly": r.is_anomaly,
                "weight": w,
                "latency_ms": round(r.latency_ms, 2),
            }

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        is_anomaly = final_score > 0.7  # ensemble threshold

        return DetectionResult(
            model_name="ensemble",
            anomaly_score=final_score,
            is_anomaly=is_anomaly,
            details={
                "method": "weighted",
                "models": model_details,
                "total_weight": round(total_weight, 4),
            },
            latency_ms=sum(r.latency_ms for r in results),
        )

    def _majority_ensemble(self, results: list[DetectionResult]) -> DetectionResult:
        """Anomaly if majority of models agree."""
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        is_anomaly = anomaly_count > len(results) / 2
        avg_score = sum(r.anomaly_score for r in results) / len(results)

        return DetectionResult(
            model_name="ensemble",
            anomaly_score=avg_score,
            is_anomaly=is_anomaly,
            details={
                "method": "majority",
                "anomaly_count": anomaly_count,
                "total_models": len(results),
            },
            latency_ms=sum(r.latency_ms for r in results),
        )

    def _any_ensemble(self, results: list[DetectionResult]) -> DetectionResult:
        """Anomaly if ANY model flags it (high recall)."""
        is_anomaly = any(r.is_anomaly for r in results)
        max_score = max(r.anomaly_score for r in results)

        return DetectionResult(
            model_name="ensemble",
            anomaly_score=max_score,
            is_anomaly=is_anomaly,
            details={
                "method": "any",
                "flagging_models": [r.model_name for r in results if r.is_anomaly],
            },
            latency_ms=sum(r.latency_ms for r in results),
        )

    def predict_with_forecast(
        self, features: np.ndarray, server_id: str, horizon_steps: int = 24,
    ) -> DetectionResult:
        """Run detection and append prediction horizon from forecaster.

        If a forecaster is available (LSTM or Holt-Winters), adds
        predicted future values and probability of future anomaly.
        """
        result = self.detect(features, server_id)

        # Try LSTM forecaster first
        lstm_model = self._models.get("lstm_autoencoder")
        if lstm_model and hasattr(lstm_model, "predict"):
            try:
                forecast = lstm_model.predict(features, horizon=horizon_steps)
                if forecast is not None:
                    result.prediction_horizon = {
                        "source": "lstm",
                        "values": [round(float(v), 4) for v in forecast],
                        "steps": horizon_steps,
                    }
                    return result
            except Exception:
                pass

        # Fallback to Holt-Winters forecast
        hw_model = self._models.get("holt_winters")
        if hw_model and hasattr(hw_model, "forecast"):
            try:
                forecast = hw_model.forecast(steps=horizon_steps)
                if forecast is not None:
                    result.prediction_horizon = {
                        "source": "holt_winters",
                        "values": [round(float(v), 4) for v in forecast],
                        "steps": horizon_steps,
                    }
            except Exception:
                pass

        return result
