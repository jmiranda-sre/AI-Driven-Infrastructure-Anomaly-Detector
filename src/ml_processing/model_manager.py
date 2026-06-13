"""Model management — loading, versioning, and lifecycle for ML models.

Singleton ModelManager tracks loaded models, versions, and provides
a unified interface for inference across all model types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from src.core.config import get_config
from src.core.errors import ModelLoadError, ModelNotFoundError
from src.core.logging import get_logger

logger = get_logger("ml.model_manager")


class AnomalyModel(Protocol):
    """Protocol for anomaly detection models."""

    @property
    def model_name(self) -> str: ...

    def detect(self, features: np.ndarray) -> DetectionResult: ...

    def train(self, data: np.ndarray, **kwargs) -> dict: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> AnomalyModel: ...


@dataclass
class DetectionResult:
    """Result from an anomaly detection model."""
    model_name: str
    anomaly_score: float          # 0.0 (normal) to 1.0 (anomalous)
    is_anomaly: bool
    details: dict[str, Any] = field(default_factory=dict)
    prediction_horizon: dict[str, float] = field(default_factory=dict)  # future predictions
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    name: str
    version: str
    algorithm: str
    path: Path
    loaded_at: datetime
    last_inference_at: datetime | None = None
    inference_count: int = 0
    avg_latency_ms: float = 0.0
    training_metrics: dict = field(default_factory=dict)


class ModelManager:
    """Singleton model manager — loads, caches, and routes inference."""

    _instance: ModelManager | None = None

    def __init__(self, model_dir: str | None = None):
        cfg = get_config()
        self._model_dir = Path(model_dir or cfg["ml"]["training"]["model_dir"])
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}  # name -> model instance
        self._info: dict[str, ModelInfo] = {}
        self._active_versions: dict[str, str] = {}  # name -> version

    @classmethod
    def get_instance(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def register_model(self, name: str, model: Any, version: str = "1.0.0", algorithm: str = "unknown") -> None:
        """Register an already-instantiated model."""
        self._models[name] = model
        self._info[name] = ModelInfo(
            name=name, version=version, algorithm=algorithm,
            path=self._model_dir / name, loaded_at=datetime.now(UTC),
        )
        self._active_versions[name] = version
        logger.info("model.registered", name=name, version=version, algorithm=algorithm)

    def load_model(self, name: str, version: str | None = None) -> Any:
        """Load a model from disk by name and optional version."""
        version = version or self._active_versions.get(name, "latest")

        if name in self._models:
            return self._models[name]

        # Search for model file
        model_path = self._model_dir / f"{name}_{version}.joblib"
        if not model_path.exists():
            # Try latest available
            candidates = sorted(self._model_dir.glob(f"{name}_*.joblib"), reverse=True)
            if candidates:
                model_path = candidates[0]
            else:
                raise ModelNotFoundError(name)

        try:
            model = joblib.load(model_path)
            self._models[name] = model
            extracted_version = model_path.stem.split("_")[-1]
            self._info[name] = ModelInfo(
                name=name, version=extracted_version, algorithm=type(model).__name__,
                path=model_path, loaded_at=datetime.now(UTC),
            )
            self._active_versions[name] = extracted_version
            logger.info("model.loaded", name=name, path=str(model_path))
            return model
        except Exception as e:
            raise ModelLoadError(name, f"Failed to load model from {model_path}: {e}") from e

    def get_model(self, name: str) -> Any:
        """Get a loaded model by name (must be already loaded or registered)."""
        if name not in self._models:
            return self.load_model(name)
        return self._models[name]

    def save_model(self, name: str, model: Any, version: str = "1.0.0") -> Path:
        """Save a model to disk with versioning."""
        path = self._model_dir / f"{name}_{version}.joblib"
        joblib.dump(model, path)
        logger.info("model.saved", name=name, version=version, path=str(path))
        return path

    def list_loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def list_available_models(self) -> list[str]:
        """List all model files on disk."""
        return [p.stem for p in self._model_dir.glob("*.joblib")]

    def get_model_info(self, name: str) -> ModelInfo | None:
        return self._info.get(name)

    def record_inference(self, name: str, latency_ms: float) -> None:
        """Record inference metrics for a model."""
        info = self._info.get(name)
        if info:
            info.inference_count += 1
            info.last_inference_at = datetime.now(UTC)
            # Exponential moving average
            alpha = 0.1
            info.avg_latency_ms = info.avg_latency_ms * (1 - alpha) + latency_ms * alpha

    def unload_model(self, name: str) -> None:
        """Unload a model from memory."""
        if name in self._models:
            del self._models[name]
            logger.info("model.unloaded", name=name)

    def get_all_info(self) -> list[dict]:
        """Get info about all loaded models."""
        return [
            {
                "name": info.name,
                "version": info.version,
                "algorithm": info.algorithm,
                "loaded_at": info.loaded_at.isoformat(),
                "inference_count": info.inference_count,
                "avg_latency_ms": round(info.avg_latency_ms, 2),
            }
            for info in self._info.values()
        ]
