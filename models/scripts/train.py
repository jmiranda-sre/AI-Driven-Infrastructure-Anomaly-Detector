"""Model training script — train all models on historical data.

Usage:
    python -m models.scripts.train --source prometheus --days 30
    python -m models.scripts.train --source csv --path data/sample/metrics.csv
    python -m models.scripts.train --source synthetic --samples 5000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger
from src.ml_processing.isolation_forest import IsolationForestDetector
from src.ml_processing.holt_winters import HoltWintersDetector
from src.ml_processing.model_manager import ModelManager

logger = get_logger("train")


def generate_synthetic_data(n_samples: int = 5000, n_features: int = 8, anomaly_rate: float = 0.05) -> np.ndarray:
    """Generate synthetic server metrics for training.

    Simulates realistic metric patterns with:
    - Daily seasonality (CPU, memory patterns)
    - correlated features
    - Injected anomalies
    """
    np.random.seed(42)
    t = np.linspace(0, n_samples / 288, n_samples)  # 288 = 24h in 5min intervals

    # Base patterns with seasonality
    cpu = 40 + 20 * np.sin(2 * np.pi * t) + np.random.normal(0, 3, n_samples)
    memory = 60 + 10 * np.sin(2 * np.pi * t + 1) + np.random.normal(0, 2, n_samples)
    disk_io = 20 + 15 * np.sin(2 * np.pi * t * 2) + np.random.normal(0, 4, n_samples)
    net_in = 100 + 50 * np.sin(2 * np.pi * t + 2) + np.random.normal(0, 10, n_samples)
    net_out = 80 + 30 * np.sin(2 * np.pi * t + 3) + np.random.normal(0, 8, n_samples)
    load = cpu / 30 + np.random.normal(0, 0.5, n_samples)
    iowait = 5 + 3 * np.abs(np.sin(2 * np.pi * t)) + np.random.normal(0, 1, n_samples)
    error_rate = np.random.exponential(0.5, n_samples)

    # Clip to realistic ranges
    cpu = np.clip(cpu, 0, 100)
    memory = np.clip(memory, 0, 100)
    disk_io = np.clip(disk_io, 0, 100)
    net_in = np.clip(net_in, 0, 1000)
    net_out = np.clip(net_out, 0, 500)
    load = np.clip(load, 0, 10)
    iowait = np.clip(iowait, 0, 50)
    error_rate = np.clip(error_rate, 0, 20)

    data = np.column_stack([cpu, memory, disk_io, net_in, net_out, load, iowait, error_rate])

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["spike", "drop", "gradual"])
        if anomaly_type == "spike":
            data[idx, :3] *= np.random.uniform(1.5, 3.0, 3)
        elif anomaly_type == "drop":
            data[idx, :2] *= np.random.uniform(0.1, 0.4, 2)
        else:
            end = min(idx + 10, n_samples)
            ramp = np.linspace(1.0, 2.0, end - idx)
            data[idx:end, 0] *= ramp

    return np.clip(data, 0, None)


def load_csv_data(path: str) -> np.ndarray:
    """Load metrics data from CSV file."""
    import pandas as pd
    df = pd.read_csv(path)
    # Assume numeric columns are features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].values
    # Fill NaN with column means
    col_means = np.nanmean(data, axis=0)
    for i in range(data.shape[1]):
        mask = np.isnan(data[:, i])
        data[mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0
    return data


def train_models(data: np.ndarray, model_dir: str, config: dict | None = None) -> dict:
    """Train all models and save to model_dir."""
    cfg = config or load_config()
    mm = ModelManager(model_dir)

    results = {}

    # 1. Isolation Forest
    logger.info("training.isolation_forest", n_samples=data.shape[0])
    if_detector = IsolationForestDetector(cfg)
    if_metrics = if_detector.train(data)
    path = mm.save_model("isolation_forest", if_detector, version="1.0.0")
    results["isolation_forest"] = if_metrics

    # 2. Holt-Winters (per-feature univariate)
    hw_results = {}
    for i in range(min(data.shape[1], 8)):
        hw = HoltWintersDetector(cfg)
        try:
            hw_metrics = hw.train(data[:, i])
            mm.save_model(f"holt_winters_feature_{i}", hw, version="1.0.0")
            hw_results[f"feature_{i}"] = hw_metrics
        except Exception as e:
            hw_results[f"feature_{i}"] = {"error": str(e)}
    results["holt_winters"] = hw_results

    # 3. LSTM Autoencoder (if PyTorch available)
    try:
        from src.ml_processing.lstm_autoencoder import LSTMAnomalyDetector
        lstm = LSTMAnomalyDetector(cfg)
        lstm_metrics = lstm.train(data)
        mm.save_model("lstm_autoencoder", lstm, version="1.0.0")
        results["lstm_autoencoder"] = lstm_metrics
    except ImportError:
        results["lstm_autoencoder"] = {"status": "skipped", "reason": "PyTorch not installed"}

    return results


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--source", choices=["prometheus", "csv", "synthetic"], default="synthetic")
    parser.add_argument("--days", type=int, default=7, help="Days of history for Prometheus backfill")
    parser.add_argument("--samples", type=int, default=5000, help="Synthetic samples to generate")
    parser.add_argument("--path", type=str, help="CSV file path for csv source")
    parser.add_argument("--model-dir", type=str, default="./models/trained")
    parser.add_argument("--output", type=str, help="Save training results to JSON file")
    args = parser.parse_args()

    configure_logging(log_level="info", json_format=False)

    # Load data
    if args.source == "synthetic":
        logger.info("train.using_synthetic_data", samples=args.samples)
        data = generate_synthetic_data(n_samples=args.samples)
    elif args.source == "csv":
        if not args.path:
            logger.error("train.csv_requires_path")
            sys.exit(1)
        data = load_csv_data(args.path)
    elif args.source == "prometheus":
        logger.info("train.using_prometheus_data", days=args.days)
        async def _backfill():
            from src.ingestion.orchestrator import IngestionOrchestrator
            orch = IngestionOrchestrator()
            batches = await orch.backfill(days=args.days)
            await orch.stop()
            if not batches:
                logger.error("train.no_data_from_prometheus")
                sys.exit(1)
            # Flatten into numpy array
            all_values = []
            for batch in batches:
                for point in batch:
                    all_values.append([point.value])
            return np.array(all_values, dtype=np.float64)

        data = asyncio.run(_backfill())
    else:
        logger.error("train.unknown_source", source=args.source)
        sys.exit(1)

    logger.info("train.data_loaded", shape=data.shape)

    # Train
    results = train_models(data, args.model_dir)

    logger.info("train.complete", models=list(results.keys()))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("train.results_saved", path=args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if not isinstance(v, dict):
                    print(f"  {k}: {v}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
