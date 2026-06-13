"""Performance tests — latency and throughput benchmarks."""

import time
from datetime import UTC

import numpy as np


class TestIsolationForestPerformance:
    """Benchmark Isolation Forest inference latency."""

    def test_single_inference_latency(self):
        from src.ml_processing.isolation_forest import IsolationForestDetector

        # Setup
        np.random.seed(42)
        data = np.random.normal(50, 10, (2000, 8))

        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 100
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(data)

        # Benchmark
        sample = data[0]
        latencies = []
        for _ in range(100):
            start = time.monotonic()
            detector.detect(sample)
            latencies.append((time.monotonic() - start) * 1000)

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        print(f"Isolation Forest — Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms")
        assert avg_latency < 50, f"Average latency {avg_latency}ms exceeds 50ms threshold"

    def test_batch_inference_throughput(self):
        from src.ml_processing.isolation_forest import IsolationForestDetector

        np.random.seed(42)
        data = np.random.normal(50, 10, (2000, 8))

        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 100
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(data)

        batch = data[:1000]
        start = time.monotonic()
        results = detector.batch_detect(batch)
        total_ms = (time.monotonic() - start) * 1000

        per_sample_ms = total_ms / len(batch)
        print(f"Batch 1000 — Total: {total_ms:.1f}ms, Per-sample: {per_sample_ms:.3f}ms")
        assert per_sample_ms < 1.0, f"Per-sample latency {per_sample_ms}ms exceeds 1ms threshold"


class TestFeatureEngineeringPerformance:
    """Benchmark feature computation latency."""

    def test_compute_features_latency(self):
        from datetime import datetime

        from src.ingestion.feature_engineering import FeatureEngineer
        from src.ingestion.models import MetricPoint

        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe.window_sizes = [5, 15, 60]
        fe.feature_names = ["mean", "std", "min", "max", "skew", "kurtosis"]
        fe._windows = {}
        fe._max_window = 10000

        ts = datetime.now(UTC)
        # Pre-populate
        for i in range(1000):
            fe.push(MetricPoint(name="cpu", value=50 + np.random.normal(0, 5), timestamp=ts, server_id="s1"))
            fe.push(MetricPoint(name="mem", value=60 + np.random.normal(0, 3), timestamp=ts, server_id="s1"))

        # Benchmark compute
        latencies = []
        for _ in range(100):
            start = time.monotonic()
            fe.compute_all_features("s1")
            latencies.append((time.monotonic() - start) * 1000)

        avg = np.mean(latencies)
        print(f"Feature computation — Avg: {avg:.3f}ms")
        assert avg < 10.0, f"Feature computation avg {avg}ms exceeds 10ms threshold"
