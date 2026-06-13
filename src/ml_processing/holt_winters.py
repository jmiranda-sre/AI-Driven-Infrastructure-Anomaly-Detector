"""Holt-Winters statistical anomaly detector — baseline model.

Triple exponential smoothing with additive seasonality. Serves as
a fast, interpretable baseline for anomaly detection.

Trade-offs:
+ Very fast (no ML framework required)
+ Highly interpretable (trend + seasonal decomposition)
+ Good for metrics with clear seasonality
+ Low memory footprint
- Cannot capture complex non-linear patterns
- Single univariate series only (no cross-metric correlation)
- Requires stable seasonal patterns
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("ml.holt_winters")


class HoltWintersDetector:
    """Holt-Winters Triple Exponential Smoothing anomaly detector."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ml"]["models"].get("holt_winters", {})
        self.model_name = "holt_winters"
        self.seasonal_periods = cfg.get("seasonal_periods", 288)
        self.trend = cfg.get("trend", "add")
        self.seasonal = cfg.get("seasonal", "add")
        self._model: ExponentialSmoothing | None = None
        self._fitted = None
        self._residual_std: float = 0.0
        self._trained = False
        self._data: np.ndarray | None = None

    @property
    def is_trained(self) -> bool:
        return self._trained and self._fitted is not None

    def train(self, data: np.ndarray, **kwargs) -> dict:
        """Fit Holt-Winters model on historical time series.

        Args:
            data: 1D array of time series values
        """
        start = time.time()
        if data.ndim > 1:
            data = data.flatten()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Need at least 2 full seasonal cycles
        min_data = self.seasonal_periods * 2
        if len(data) < min_data:
            logger.warn(
                "holt_winters.insufficient_data",
                n=len(data), min_required=min_data,
            )
            # Adjust seasonal periods if data too short
            self.seasonal_periods = max(2, len(data) // 2)

        try:
            self._model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            )
            self._fitted = self._model.fit()
            self._trained = True
            self._data = data

            # Compute residual statistics
            residuals = data[self.seasonal_periods:] - self._fitted.fittedvalues[self.seasonal_periods:]
            self._residual_std = float(np.std(residuals))

            training_time = time.time() - start
            metrics = {
                "training_time_s": round(training_time, 3),
                "n_samples": len(data),
                "seasonal_periods": self.seasonal_periods,
                "residual_std": round(self._residual_std, 6),
                "sse": round(float(np.sum(residuals**2)), 3),
            }
            logger.info("holt_winters.trained", **metrics)
            return metrics
        except Exception as e:
            logger.error("holt_winters.train_failed", error=str(e))
            return {"error": str(e)}

    def detect(self, features: np.ndarray) -> DetectionResult:
        """Detect anomaly based on forecast residual.

        Compares the last observed value with the model's forecast for it.
        A large residual (in standard deviations) indicates an anomaly.
        """
        if not self.is_trained:
            return DetectionResult(
                model_name=self.model_name, anomaly_score=0.0, is_anomaly=False,
                details={"error": "model not trained"},
            )

        start = time.monotonic()
        if features.ndim > 1:
            features = features.flatten()

        value = float(features[-1]) if len(features) > 0 else 0.0

        try:
            # Forecast one step ahead from the training data
            forecast = float(self._fitted.forecast(1)[0])
            residual = abs(value - forecast)

            # Anomaly score based on z-score of residual
            if self._residual_std > 0:
                z_score = residual / self._residual_std
                # Convert z-score to 0-1 anomaly score using sigmoid-like mapping
                anomaly_score = min(1.0, z_score / 4.0)  # 4-sigma → score 1.0
            else:
                anomaly_score = 0.0

            is_anomaly = anomaly_score > 0.7  # ~3-sigma threshold
        except Exception as e:
            logger.error("holt_winters.detect_error", error=str(e))
            forecast = value
            anomaly_score = 0.0
            is_anomaly = False
            residual = 0.0
            z_score = 0.0

        latency = (time.monotonic() - start) * 1000

        return DetectionResult(
            model_name=self.model_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            details={
                "forecast": round(forecast, 4),
                "actual": round(value, 4),
                "residual": round(residual, 4),
                "z_score": round(z_score, 2) if self._residual_std > 0 else None,
                "residual_std": round(self._residual_std, 6),
            },
            latency_ms=latency,
        )

    def forecast(self, steps: int = 24) -> np.ndarray | None:
        """Forecast future values.

        Args:
            steps: Number of steps to forecast

        Returns:
            Array of forecasted values, or None
        """
        if not self.is_trained:
            return None
        try:
            result = self._fitted.forecast(steps)
            if hasattr(result, "values"):
                return result.values
            return np.asarray(result)
        except Exception as e:
            logger.error("holt_winters.forecast_error", error=str(e))
            return None

    def save(self, path: Path) -> None:
        import joblib
        if self._fitted is not None:
            save_dict = {
                "fitted": self._fitted,
                "model": self._model,
                "residual_std": self._residual_std,
                "seasonal_periods": self.seasonal_periods,
                "trend": self.trend,
                "seasonal": self.seasonal,
            }
            joblib.dump(save_dict, path)
            logger.info("holt_winters.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> HoltWintersDetector:
        import joblib
        model = cls.__new__(cls)
        save_dict = joblib.load(path)
        model._fitted = save_dict["fitted"]
        model._model = save_dict["model"]
        model._residual_std = save_dict["residual_std"]
        model.seasonal_periods = save_dict["seasonal_periods"]
        model.trend = save_dict["trend"]
        model.seasonal = save_dict["seasonal"]
        model._trained = True
        model.model_name = "holt_winters"
        return model
