"""Isolation Forest anomaly detector — unsupervised, efficient, low-latency.

Isolation Forest isolates anomalies by randomly selecting features and split
values. Anomalies require fewer splits to isolate, resulting in shorter path
lengths — which translate to higher anomaly scores.

Trade-offs:
+ No labeled data required (unsupervised)
+ O(n·log(n)) training, O(n) inference
+ Handles high-dimensional data
+ Robust to distribution shape
- Lower precision on contextual/sequential anomalies
- Cannot predict future values (no forecasting)
- Sensitive to contamination parameter
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("ml.isolation_forest")


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ml"]["models"]["isolation_forest"]
        self.model_name = "isolation_forest"
        self.contamination = cfg.get("contamination", 0.05)
        self.n_estimators = cfg.get("n_estimators", 100)
        self.max_samples = cfg.get("max_samples", "auto")
        self.random_state = cfg.get("random_state", 42)
        self._model: IsolationForest | None = None
        self._trained = False
        self._feature_count: int = 0

    @property
    def is_trained(self) -> bool:
        return self._trained and self._model is not None

    def train(self, data: np.ndarray, **kwargs) -> dict:
        """Train the Isolation Forest on feature data.

        Args:
            data: 2D array of shape (n_samples, n_features)

        Returns:
            Training metrics dict
        """
        start = time.time()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Replace NaN/Inf with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        self._feature_count = data.shape[1]
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=kwargs.get("warm_start", False),
        )
        self._model.fit(data)
        self._trained = True

        # Compute training metrics
        scores = self._model.decision_function(data)
        training_time = time.time() - start

        metrics = {
            "training_time_s": round(training_time, 3),
            "n_samples": data.shape[0],
            "n_features": data.shape[1],
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "anomaly_rate": float(np.mean(self._model.predict(data) == -1)),
        }
        logger.info("isolation_forest.trained", **metrics)
        return metrics

    def detect(self, features: np.ndarray) -> DetectionResult:
        """Detect anomalies in a feature vector.

        Args:
            features: 1D or 2D feature array

        Returns:
            DetectionResult with anomaly score and flag
        """
        if not self.is_trained:
            return DetectionResult(
                model_name=self.model_name,
                anomaly_score=0.0,
                is_anomaly=False,
                details={"error": "model not trained"},
            )

        start = time.monotonic()
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # decision_function: higher = more normal, lower = more anomalous
        raw_score = self._model.decision_function(features)[0]
        predict = self._model.predict(features)[0]

        # Normalize to [0, 1] where 1.0 = most anomalous
        # decision_function typically ranges from -0.5 to 0.5
        anomaly_score = max(0.0, min(1.0, 0.5 - raw_score))

        latency = (time.monotonic() - start) * 1000

        return DetectionResult(
            model_name=self.model_name,
            anomaly_score=anomaly_score,
            is_anomaly=predict == -1,
            details={
                "raw_score": float(raw_score),
                "contamination": self.contamination,
            },
            latency_ms=latency,
        )

    def batch_detect(self, features: np.ndarray) -> list[DetectionResult]:
        """Detect anomalies for a batch of feature vectors."""
        if not self.is_trained:
            return []

        start = time.monotonic()
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        raw_scores = self._model.decision_function(features)
        predicts = self._model.predict(features)
        latency = (time.monotonic() - start) * 1000

        results = []
        for i in range(len(raw_scores)):
            anomaly_score = max(0.0, min(1.0, 0.5 - raw_scores[i]))
            results.append(DetectionResult(
                model_name=self.model_name,
                anomaly_score=anomaly_score,
                is_anomaly=predicts[i] == -1,
                details={"raw_score": float(raw_scores[i])},
                latency_ms=latency / len(raw_scores),
            ))
        return results

    def save(self, path: Path) -> None:
        if self._model is not None:
            joblib.dump(self._model, path)
            logger.info("isolation_forest.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        model = cls.__new__(cls)
        model._model = joblib.load(path)
        model._trained = True
        model.model_name = "isolation_forest"
        model._feature_count = model._model.n_features_in_ if hasattr(model._model, 'n_features_in_') else 0
        model.contamination = model._model.contamination if hasattr(model._model, 'contamination') else 0.05
        model.n_estimators = model._model.n_estimators if hasattr(model._model, 'n_estimators') else 100
        model.max_samples = model._model.max_samples if hasattr(model._model, 'max_samples') else "auto"
        model.random_state = model._model.random_state if hasattr(model._model, 'random_state') else 42
        return model
