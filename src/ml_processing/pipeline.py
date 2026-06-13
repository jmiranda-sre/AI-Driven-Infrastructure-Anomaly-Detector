"""ML Pipeline — orchestrates training, scheduling, and the full inference cycle.

Coordinates: data ingestion → feature engineering → model training/inference → drift detection
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.drift_detector import DriftDetector
from src.ml_processing.ensemble import EnsembleDetector
from src.ml_processing.holt_winters import HoltWintersDetector
from src.ml_processing.isolation_forest import IsolationForestDetector
from src.ml_processing.lstm_autoencoder import LSTMAnomalyDetector
from src.ml_processing.model_manager import ModelManager

logger = get_logger("ml.pipeline")


class MLPipeline:
    """Full ML pipeline: training, inference, drift detection, and retraining.

    Singleton pipeline that receives processed features from ingestion,
    runs ensemble detection, and emits anomaly results to alerting.
    """

    _instance: MLPipeline | None = None

    def __init__(self, config: dict | None = None):
        self._config = config or get_config()
        self.ensemble = EnsembleDetector(self._config)
        self.drift_detector = DriftDetector(self._config)
        self.model_manager = ModelManager(
            self._config["ml"]["training"].get("model_dir", "./models/trained")
        )
        self._alert_callback = None
        self._training_task: asyncio.Task | None = None
        self._data_buffer: dict[str, list[tuple[dict[str, float], datetime]]] = {}

        # Initialize models based on config
        self._init_models()

    @classmethod
    def get_instance(cls) -> MLPipeline:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def _init_models(self) -> None:
        """Initialize and register all enabled models."""
        models_cfg = self._config["ml"]["models"]

        if models_cfg.get("isolation_forest", {}).get("enabled", True):
            if_detector = IsolationForestDetector(self._config)
            self.ensemble.register_model("isolation_forest", if_detector)
            self.model_manager.register_model("isolation_forest", if_detector, algorithm="isolation_forest")

        if models_cfg.get("lstm_autoencoder", {}).get("enabled", False):
            try:
                lstm_detector = LSTMAnomalyDetector(self._config)
                self.ensemble.register_model("lstm_autoencoder", lstm_detector)
                self.model_manager.register_model("lstm_autoencoder", lstm_detector, algorithm="lstm_autoencoder")
            except ImportError:
                logger.warn("pipeline.lstm_unavailable")

        if models_cfg.get("holt_winters", {}).get("enabled", True):
            hw_detector = HoltWintersDetector(self._config)
            self.ensemble.register_model("holt_winters", hw_detector)
            self.model_manager.register_model("holt_winters", hw_detector, algorithm="holt_winters")

    def register_alert_callback(self, callback) -> None:
        """Register callback for anomaly detection results → alerting module."""
        self._alert_callback = callback

    async def process_features(
        self, server_id: str, features: dict[str, float], timestamp: datetime
    ) -> None:
        """Entry point: receive features from ingestion, run detection.

        This is the main callback registered with IngestionOrchestrator.
        """
        # Buffer data for training
        self._data_buffer.setdefault(server_id, []).append((features, timestamp))
        max_buffer = self._config["ml"]["training"].get("min_samples", 1000)
        if len(self._data_buffer[server_id]) > max_buffer * 2:
            self._data_buffer[server_id] = self._data_buffer[server_id][-max_buffer:]

        # Convert features to numpy vector
        if not features:
            return

        feature_vector = np.array(
            [features[k] for k in sorted(features.keys())], dtype=np.float64
        )

        # Check if models are trained
        active = self.ensemble.active_models
        if not active:
            # Auto-train if we have enough data
            min_samples = self._config["ml"]["training"].get("min_samples", 1000)
            total_samples = sum(len(v) for v in self._data_buffer.values())
            if total_samples >= min_samples:
                await self.train_all()
            return

        # Run ensemble detection with forecast
        try:
            horizon_steps = self._config["ml"]["inference"].get("prediction_steps", 24)
            result = self.ensemble.predict_with_forecast(
                feature_vector, server_id, horizon_steps=horizon_steps
            )

            # Record inference metrics
            for name in self.ensemble.active_models:
                self.model_manager.record_inference(name, result.latency_ms)

            # If anomaly detected, dispatch to alerting
            if (result.is_anomaly or result.anomaly_score > 0.5) and self._alert_callback:
                await self._alert_callback(server_id, result, timestamp)

            logger.debug(
                "pipeline.inference",
                server_id=server_id,
                score=round(result.anomaly_score, 4),
                is_anomaly=result.is_anomaly,
            )

        except Exception as e:
            logger.error("pipeline.inference_error", server_id=server_id, error=str(e))

    async def train_all(self) -> dict[str, dict]:
        """Train all enabled models using buffered data.

        Returns:
            Dict of model_name → training_metrics
        """
        results = {}
        all_features = []

        for _server_id, buffer in self._data_buffer.items():
            for features, _ in buffer:
                vec = [features[k] for k in sorted(features.keys())]
                all_features.append(vec)

        if len(all_features) < 10:
            logger.warn("pipeline.insufficient_training_data", n_samples=len(all_features))
            return results

        data = np.array(all_features, dtype=np.float64)
        # Replace NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Train each model
        for name, model in self.ensemble._models.items():
            try:
                if hasattr(model, "train"):
                    logger.info("pipeline.training", model=name, n_samples=len(data))
                    metrics = model.train(data)
                    results[name] = metrics

                    # Save trained model
                    self.model_manager.save_model(name, model, version="1.0.0")

                    # Set drift reference
                    self.drift_detector.set_reference(name, data)

            except Exception as e:
                logger.error("pipeline.training_error", model=name, error=str(e))
                results[name] = {"error": str(e)}

        logger.info("pipeline.training_complete", models=list(results.keys()))
        return results

    async def train_univariate(self, server_id: str, metric_name: str, values: np.ndarray) -> dict:
        """Train univariate models (Holt-Winters) on a single metric series."""
        hw = self.ensemble._models.get("holt_winters")
        if hw and hasattr(hw, "train"):
            return hw.train(values)
        return {}

    def check_drift(self, current_data: dict[str, np.ndarray]) -> list[dict]:
        """Check all models for concept drift."""
        return self.drift_detector.check_all_drifts(current_data)

    async def start_training_scheduler(self) -> None:
        """Start periodic model retraining (background task)."""
        schedule = self._config["ml"]["training"].get("schedule", "0 */6 * * *")
        logger.info("pipeline.scheduler_started", schedule=schedule)

        # Simplified: retrain every N seconds based on cron
        # In production, use APScheduler
        interval = 6 * 3600  # 6 hours default

        while True:
            await asyncio.sleep(interval)
            try:
                results = await self.train_all()
                logger.info("pipeline.scheduled_retrain", results=results)
            except Exception as e:
                logger.error("pipeline.retrain_error", error=str(e))

    def get_status(self) -> dict:
        """Get pipeline status summary."""
        return {
            "active_models": self.ensemble.active_models,
            "buffered_servers": list(self._data_buffer.keys()),
            "total_buffered": sum(len(v) for v in self._data_buffer.values()),
            "model_info": self.model_manager.get_all_info(),
        }
