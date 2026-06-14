"""Feature engineering pipeline — transform raw metrics into ML features.

Computes rolling statistics (mean, std, min, max, skew, kurtosis) over
configurable time windows. Normalizes features and handles missing data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from src.core.config import get_config
from src.core.logging import get_logger
from src.ingestion.models import MetricBatch, MetricPoint, MetricWindow

logger = get_logger("ingestion.feature_engineering")


class FeatureEngineer:
    """Transform raw metric points into ML-ready feature vectors.

    Maintains sliding windows per (server_id, metric_name) pair and
    computes rolling statistics when requested.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ml"]["feature_engineering"]
        self.window_sizes = cfg.get("window_sizes", [5, 15, 60])
        self.feature_names = cfg.get("features", ["mean", "std", "min", "max", "skew", "kurtosis"])
        # Internal state: windows keyed by (server_id, metric_name)
        self._windows: dict[str, MetricWindow] = {}
        self._max_window = max(self.window_sizes) * 12  # 12 pts/min at 5s intervals
        # Cap on total number of windows to prevent unbounded memory
        self._max_windows = 10_000

    def _window_key(self, server_id: str, metric_name: str) -> str:
        return f"{server_id}::{metric_name}"

    def push(self, point: MetricPoint) -> None:
        """Add a metric point to the appropriate sliding window."""
        key = self._window_key(point.server_id, point.name)
        if key not in self._windows:
            # Evict oldest window if at capacity
            if len(self._windows) >= self._max_windows:
                oldest_key = next(iter(self._windows))
                del self._windows[oldest_key]
            self._windows[key] = MetricWindow(
                server_id=point.server_id,
                metric_name=point.name,
                max_size=self._max_window,
            )
        self._windows[key].push(point.value, point.timestamp)

    def push_batch(self, batch: MetricBatch) -> None:
        """Push all points from a MetricBatch."""
        for point in batch:
            self.push(point)

    def compute_features(
        self, server_id: str, metric_name: str, window_size: int | None = None
    ) -> dict[str, float] | None:
        """Compute rolling feature vector for a server+metric pair.

        Args:
            server_id: Target server
            metric_name: Target metric
            window_size: Window in minutes (uses last N points ~ window*12)

        Returns:
            Dict of feature_name -> value, or None if insufficient data
        """
        key = self._window_key(server_id, metric_name)
        window = self._windows.get(key)

        if window is None or len(window) < 5:
            return None

        ws = window_size or self.window_sizes[0]
        # Approximate number of points for window (12 pts/min at 5s scrape)
        n_points = min(ws * 12, len(window))
        values = window.values[-n_points:]
        arr = np.array(values, dtype=np.float64)

        if len(arr) < 2:
            return None

        features: dict[str, float] = {}
        for fname in self.feature_names:
            try:
                if fname == "mean":
                    features[f"{metric_name}_mean_{ws}m"] = float(np.mean(arr))
                elif fname == "std":
                    features[f"{metric_name}_std_{ws}m"] = float(np.std(arr))
                elif fname == "min":
                    features[f"{metric_name}_min_{ws}m"] = float(np.min(arr))
                elif fname == "max":
                    features[f"{metric_name}_max_{ws}m"] = float(np.max(arr))
                elif fname == "skew":
                    features[f"{metric_name}_skew_{ws}m"] = float(skew(arr))
                elif fname == "kurtosis":
                    features[f"{metric_name}_kurtosis_{ws}m"] = float(kurtosis(arr))
            except Exception:
                continue

        # Add rate of change feature
        if len(arr) >= 2:
            features[f"{metric_name}_rate_{ws}m"] = float(arr[-1] - arr[0])

        return features

    def compute_all_features(
        self, server_id: str, metric_names: list[str] | None = None
    ) -> dict[str, float]:
        """Compute features across all windows for a server.

        Returns:
            Flat dict of all features for ML model input
        """
        prefix = f"{server_id}::"
        all_features: dict[str, float] = {}

        for key, window in self._windows.items():
            if not key.startswith(prefix):
                continue
            if metric_names and window.metric_name not in metric_names:
                continue

            for ws in self.window_sizes:
                feats = self.compute_features(server_id, window.metric_name, ws)
                if feats:
                    all_features.update(feats)

        return all_features

    def get_feature_vector(
        self, server_id: str, metric_names: list[str] | None = None
    ) -> np.ndarray | None:
        """Get a numpy feature vector for ML model input."""
        features = self.compute_all_features(server_id, metric_names)
        if not features:
            return None
        # Sort by key for deterministic order
        return np.array([features[k] for k in sorted(features.keys())], dtype=np.float64)

    def get_feature_names(self, server_id: str) -> list[str]:
        """Get ordered feature names (matches get_feature_vector output)."""
        features = self.compute_all_features(server_id)
        return sorted(features.keys()) if features else []

    def normalize(
        self, vector: np.ndarray, method: str = "zscore"
    ) -> np.ndarray:
        """Normalize a feature vector.

        Args:
            vector: Feature vector
            method: 'zscore' or 'minmax'
        """
        if method == "zscore":
            std = np.std(vector)
            if std == 0:
                return vector
            return (vector - np.mean(vector)) / std
        elif method == "minmax":
            rng = np.ptp(vector)
            if rng == 0:
                return vector
            return (vector - np.min(vector)) / rng
        return vector

    @property
    def window_count(self) -> int:
        return len(self._windows)

    def get_servers(self) -> list[str]:
        """List all server IDs with data."""
        servers: set[str] = set()
        for key in self._windows:
            server_id = key.split("::")[0]
            servers.add(server_id)
        return sorted(servers)
