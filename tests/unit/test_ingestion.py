"""Unit tests for ingestion module."""

from datetime import UTC, datetime

import numpy as np

from src.ingestion.feature_engineering import FeatureEngineer
from src.ingestion.models import MetricBatch, MetricPoint, MetricWindow, ServerMetrics


class TestMetricPoint:
    def test_creation(self):
        mp = MetricPoint(name="cpu", value=75.0, timestamp=datetime.now(UTC), server_id="s1")
        assert mp.name == "cpu"
        assert mp.value == 75.0
        assert mp.server_id == "s1"

    def test_to_dict(self):
        mp = MetricPoint(name="cpu", value=50.0, timestamp=datetime.now(UTC), server_id="s1")
        d = mp.to_dict()
        assert d["name"] == "cpu"
        assert d["value"] == 50.0


class TestMetricBatch:
    def test_add_and_len(self):
        batch = MetricBatch(source="test")
        batch.add(MetricPoint(name="cpu", value=10.0, timestamp=datetime.now(UTC), server_id="s1"))
        assert len(batch) == 1

    def test_to_numpy(self):
        batch = MetricBatch(source="test")
        for i in range(5):
            batch.add(MetricPoint(name="cpu", value=float(i), timestamp=datetime.now(UTC), server_id="s1"))
        arr = batch.to_numpy_array()
        assert arr.shape == (5,)
        assert arr[2] == 2.0


class TestMetricWindow:
    def test_push_and_max_size(self):
        w = MetricWindow(server_id="s1", metric_name="cpu", max_size=5)
        for i in range(10):
            w.push(float(i), datetime.now(UTC))
        assert len(w) == 5
        assert w.values[-1] == 9.0

    def test_is_ready(self):
        w = MetricWindow(server_id="s1", metric_name="cpu")
        assert not w.is_ready
        for i in range(15):
            w.push(float(i), datetime.now(UTC))
        assert w.is_ready


class TestServerMetrics:
    def test_to_feature_vector(self):
        sm = ServerMetrics(server_id="s1", metrics={"cpu": 80.0, "mem": 60.0})
        vec = sm.to_feature_vector(["cpu", "mem"])
        assert vec.shape == (2,)
        assert vec[0] == 80.0


class TestFeatureEngineer:
    def test_push_and_compute(self):
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe.window_sizes = [5]
        fe.feature_names = ["mean", "std", "min", "max"]
        fe._windows = {}
        fe._max_window = 1000
        fe._max_windows = 10_000

        # Push 20 points
        ts = datetime.now(UTC)
        for i in range(20):
            fe.push(MetricPoint(name="cpu", value=50.0 + i, timestamp=ts, server_id="s1"))

        features = fe.compute_features("s1", "cpu", window_size=5)
        assert features is not None
        assert "cpu_mean_5m" in features
        assert "cpu_std_5m" in features
        assert "cpu_min_5m" in features
        assert "cpu_max_5m" in features

    def test_compute_all_features(self):
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe.window_sizes = [5]
        fe.feature_names = ["mean", "std"]
        fe._windows = {}
        fe._max_window = 1000
        fe._max_windows = 10_000

        ts = datetime.now(UTC)
        for _i in range(20):
            fe.push(MetricPoint(name="cpu", value=50.0, timestamp=ts, server_id="s1"))
            fe.push(MetricPoint(name="mem", value=60.0, timestamp=ts, server_id="s1"))

        all_features = fe.compute_all_features("s1")
        assert "cpu_mean_5m" in all_features
        assert "mem_mean_5m" in all_features

    def test_get_feature_vector(self):
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe.window_sizes = [5]
        fe.feature_names = ["mean", "std"]
        fe._windows = {}
        fe._max_window = 1000
        fe._max_windows = 10_000

        ts = datetime.now(UTC)
        for i in range(20):
            fe.push(MetricPoint(name="cpu", value=50.0 + i * 0.1, timestamp=ts, server_id="s1"))

        vec = fe.get_feature_vector("s1")
        assert vec is not None
        assert vec.shape[0] > 0

    def test_normalize_zscore(self):
        fe = FeatureEngineer.__new__(FeatureEngineer)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fe.normalize(vec, method="zscore")
        assert abs(np.mean(result)) < 0.01

    def test_normalize_minmax(self):
        fe = FeatureEngineer.__new__(FeatureEngineer)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fe.normalize(vec, method="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0
