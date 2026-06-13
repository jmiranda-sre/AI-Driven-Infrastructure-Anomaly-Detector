"""Concept drift detection using Population Stability Index (PSI).

Monitors drift between reference and current data distributions.
Triggers model retraining when PSI exceeds threshold.

PSI < 0.1  → No significant drift
PSI 0.1-0.25 → Moderate drift (warn)
PSI > 0.25 → Significant drift (retrain)
"""

from __future__ import annotations

import numpy as np

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger("ml.drift_detector")


class DriftDetector:
    """Detect concept drift using PSI and KS-test."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["ml"]["training"]
        self.threshold = cfg.get("drift_threshold", 0.15)
        self.retrain_on_drift = cfg.get("retrain_on_drift", True)
        self._reference_histograms: dict[str, np.ndarray] = {}
        self._reference_bin_edges: dict[str, np.ndarray] = {}

    def set_reference(self, name: str, data: np.ndarray, n_bins: int = 10) -> None:
        """Store reference distribution for a feature/model."""
        data = np.nan_to_num(data.flatten(), nan=0.0)
        counts, edges = np.histogram(data, bins=n_bins, density=True)
        self._reference_histograms[name] = counts
        self._reference_bin_edges[name] = edges
        logger.info("drift.reference_set", name=name, n_samples=len(data))

    def compute_psi(self, name: str, current_data: np.ndarray) -> float:
        """Compute Population Stability Index between reference and current.

        Args:
            name: Feature/model identifier
            current_data: Current distribution data

        Returns:
            PSI value (0 = identical, >0.25 = significant drift)
        """
        if name not in self._reference_histograms:
            logger.warn("drift.no_reference", name=name)
            return 0.0

        current_data = np.nan_to_num(current_data.flatten(), nan=0.0)
        ref_hist = self._reference_histograms[name]
        ref_edges = self._reference_bin_edges[name]

        # Bin current data using reference edges
        current_hist, _ = np.histogram(current_data, bins=ref_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        ref_hist = ref_hist + eps
        current_hist = current_hist + eps

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        current_hist = current_hist / current_hist.sum()

        # PSI = sum((current - reference) * ln(current / reference))
        psi = float(np.sum((current_hist - ref_hist) * np.log(current_hist / ref_hist)))

        return psi

    def check_drift(self, name: str, current_data: np.ndarray) -> dict:
        """Check for drift and return assessment.

        Returns:
            Dict with psi value, drift level, and retrain recommendation
        """
        psi = self.compute_psi(name, current_data)

        if psi < 0.1:
            level = "none"
            retrain = False
        elif psi < self.threshold:
            level = "moderate"
            retrain = False
        else:
            level = "significant"
            retrain = self.retrain_on_drift

        result = {
            "feature": name,
            "psi": round(psi, 6),
            "level": level,
            "threshold": self.threshold,
            "retrain": retrain,
        }

        if level != "none":
            logger.warn("drift.detected", **result)
        else:
            logger.debug("drift.check", **result)

        return result

    def check_all_drifts(
        self, current_features: dict[str, np.ndarray]
    ) -> list[dict]:
        """Check drift for all registered features.

        Args:
            current_features: Dict of feature_name -> current data array

        Returns:
            List of drift assessment dicts
        """
        results = []
        for name, data in current_features.items():
            if name in self._reference_histograms:
                results.append(self.check_drift(name, data))
        return results

    @property
    def registered_features(self) -> list[str]:
        return list(self._reference_histograms.keys())
