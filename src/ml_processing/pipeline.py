"""ML Pipeline — orchestrates training, scheduling, and the full inference cycle.

Coordinates: data ingestion → feature engineering → model training/inference → drift detection
"""

from __future__ import annotations

import asyncio
from collections import deque
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

# Max total entries across all server buffers
_MAX_TOTAL_BUFFER = 100_000


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
        self._scheduler_task: asyncio.Task | None = None
        # Data buffer: server_id -> deque of (features_dict, timestamp)
        self._data_buffer: dict[str, deque[tuple[dict[str, float], datetime]]] = {}
        # Lock to prevent concurrent train/detect race conditions
        self._lock = asyncio.Lock()

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
                logger.warning("pipeline.lstm_unavailable")

        if models_cfg.get("holt_winters", {}).get("enabled", True):
            hw_detector = HoltWintersDetector(self._config)
            self.ensemble.register_model("holt_winters", hw_detector)
            self.model_manager.register_model("holt_winters", hw_detector, algorithm="holt_winters")

    def register_alert_callback(self, callback) -> None:
        """Register callback for anomaly detection results → alerting module."""
        self._alert_callback = callback

    def _trim_buffer(self, server_id: str) -> None:
        """Trim a server buffer to respect per-server and total caps."""
        max_per_server = self._config["ml"]["training"].get("min_samples", 1000) * 2
        buf = self._data_buffer.get(server_id)
        if buf and len(buf) > max_per_server:
            # Trim from front (oldest data first)
            while len(buf) > max_per_server:
                buf.popleft()

    def _trim_total_buffer(self) -> None:
        """Enforce total buffer cap across all servers (FIFO eviction)."""
        total = sum(len(v) for v in self._data_buffer.values())
        if total <= _MAX_TOTAL_BUFFER:
            return
        # Evict oldest entries from the server with the largest buffer
        while total > _MAX_TOTAL_BUFFER and self._data_buffer:
            largest_sid = max(self._data_buffer, key=lambda s: len(self._data_buffer[s]))
            buf = self._data_buffer[largest_sid]
            if buf:
                buf.popleft()
                total -= 1
            else:
                del self._data_buffer[largest_sid]

    def _features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert a features dict to a deterministic numpy vector."""
        return np.array(
            [features[k] for k in sorted(features.keys())], dtype=np.float64
        )

    async def process_features(
        self, server_id: str, features: dict[str, float], timestamp: datetime
    ) -> None:
        """Entry point: receive features from ingestion, run detection.

        This is the main callback registered with IngestionOrchestrator.
        """
        # Buffer data for training
        if server_id not in self._data_buffer:
            self._data_buffer[server_id] = deque(maxlen=self._config["ml"]["training"].get("min_samples", 1000) * 2)
        self._data_buffer[server_id].append((features, timestamp))
        self._trim_total_buffer()

        if not features:
            return

        feature_vector = self._features_to_vector(features)

        # Check if models are trained
        async with self._lock:
            active = self.ensemble.active_models
            if not active:
                min_samples = self._config["ml"]["training"].get("min_samples", 1000)
                total_samples = sum(len(v) for v in self._data_buffer.values())
                if total_samples >= min_samples:
                    await self.train_all()
                return

        # Run ensemble detection with forecast (no lock — reads are safe)
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

    def train_all_sync(self) -> dict[str, dict]:
        """Synchronous training — call via asyncio.to_thread to avoid blocking event loop."""
        results = {}
        all_features = []

        for _server_id, buffer in self._data_buffer.items():
            for features, _ in buffer:
                vec = [features[k] for k in sorted(features.keys())]
                all_features.append(vec)

        if len(all_features) < 10:
            logger.warning("pipeline.insufficient_training_data", n_samples=len(all_features))
            return results

        data = np.array(all_features, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

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

    async def train_all(self) -> dict[str, dict]:
        """Train all enabled models using buffered data.

        Runs sync sklearn .fit() in thread pool to avoid blocking the event loop.
        """
        async with self._lock:
            return await asyncio.to_thread(self.train_all_sync)

    async def train_univariate(self, server_id: str, metric_name: str, values: np.ndarray) -> dict:
        """Train univariate models (Holt-Winters) on a single metric series."""
        hw = self.ensemble._models.get("holt_winters")
        if hw and hasattr(hw, "train"):
            return hw.train(values)
        return {}

    def check_drift(self, current_data: dict[str, np.ndarray]) -> list[dict]:
        """Check all models for concept drift."""
        return self.drift_detector.check_all_drifts(current_data)

    def check_drift_from_buffer(self) -> list[dict]:
        """Check drift using data from the pipeline's internal buffer.

        Constructs current_data arrays from the buffer and runs drift
        detection against stored reference distributions.
        """
        if not self._data_buffer:
            return []

        # Build per-model feature arrays from buffer
        feature_keys = None

        for _server_id, buffer in self._data_buffer.items():
            if not buffer:
                continue
            features, _ = buffer[-1]
            if feature_keys is None:
                feature_keys = sorted(features.keys())

        if feature_keys is None:
            return []

        # Collect all feature vectors into an array for drift detection
        all_vectors = []
        for _server_id, buffer in self._data_buffer.items():
            for features, _ in buffer:
                try:
                    vec = [features.get(k, 0.0) for k in feature_keys]
                    all_vectors.append(vec)
                except Exception:
                    continue

        if len(all_vectors) < 10:
            return []

        data = np.array(all_vectors, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Run drift check for each registered model reference
        return self.drift_detector.check_all_drifts(
            dict.fromkeys(self.drift_detector.registered_features, data)
        )

    def start_training_scheduler_task(self) -> None:
        """Fire-and-forget: start the periodic retraining scheduler task."""
        if self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.ensure_future(self.start_training_scheduler())

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
