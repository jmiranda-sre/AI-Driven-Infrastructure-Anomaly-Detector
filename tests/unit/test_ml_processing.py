"""Unit tests for ML processing module."""

import numpy as np

from src.ml_processing.drift_detector import DriftDetector
from src.ml_processing.ensemble import EnsembleDetector
from src.ml_processing.isolation_forest import IsolationForestDetector
from src.ml_processing.model_manager import DetectionResult, ModelManager


class TestIsolationForestDetector:
    def test_train_and_detect(self, sample_metric_data):
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        metrics = detector.train(sample_metric_data)
        assert detector.is_trained
        assert "training_time_s" in metrics
        assert metrics["n_features"] == sample_metric_data.shape[1]

    def test_detect_returns_result(self, sample_metric_data):
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(sample_metric_data)
        result = detector.detect(sample_metric_data[0])

        assert isinstance(result, DetectionResult)
        assert result.model_name == "isolation_forest"
        assert 0.0 <= result.anomaly_score <= 1.0
        assert result.latency_ms > 0

    def test_detect_anomalous_input(self, sample_metric_data):
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(sample_metric_data)

        # An extreme value should be flagged
        extreme = np.array([[99.0, 99.0]])
        result = detector.detect(extreme[0])
        assert result.anomaly_score > 0.3

    def test_batch_detect(self, sample_metric_data):
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(sample_metric_data)
        results = detector.batch_detect(sample_metric_data[:10])
        assert len(results) == 10

    def test_save_and_load(self, sample_metric_data, tmp_path):
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(sample_metric_data)
        path = tmp_path / "if_test.joblib"
        detector.save(path)
        assert path.exists()

        loaded = IsolationForestDetector.load(path)
        assert loaded.is_trained
        result = loaded.detect(sample_metric_data[0])
        assert 0.0 <= result.anomaly_score <= 1.0


class TestHoltWintersDetector:
    def test_train_univariate(self, sample_univariate):
        from src.ml_processing.holt_winters import HoltWintersDetector
        detector = HoltWintersDetector.__new__(HoltWintersDetector)
        detector.model_name = "holt_winters"
        detector.seasonal_periods = 12  # Short for testing
        detector.trend = "add"
        detector.seasonal = "add"
        detector._model = None
        detector._fitted = None
        detector._residual_std = 0.0
        detector._trained = False
        detector._data = None

        metrics = detector.train(sample_univariate)
        assert detector.is_trained
        assert "residual_std" in metrics

    def test_detect(self, sample_univariate):
        from src.ml_processing.holt_winters import HoltWintersDetector
        detector = HoltWintersDetector.__new__(HoltWintersDetector)
        detector.model_name = "holt_winters"
        detector.seasonal_periods = 12
        detector.trend = "add"
        detector.seasonal = "add"
        detector._model = None
        detector._fitted = None
        detector._residual_std = 0.0
        detector._trained = False
        detector._data = None

        detector.train(sample_univariate)
        result = detector.detect(sample_univariate[-5:])
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_forecast(self, sample_univariate):
        from src.ml_processing.holt_winters import HoltWintersDetector
        detector = HoltWintersDetector.__new__(HoltWintersDetector)
        detector.model_name = "holt_winters"
        detector.seasonal_periods = 12
        detector.trend = "add"
        detector.seasonal = "add"
        detector._model = None
        detector._fitted = None
        detector._residual_std = 0.0
        detector._trained = False
        detector._data = None

        detector.train(sample_univariate)
        forecast = detector.forecast(steps=10)
        assert forecast is not None
        assert len(forecast) == 10


class TestEnsembleDetector:
    def test_weighted_ensemble(self, sample_metric_data):
        ensemble = EnsembleDetector.__new__(EnsembleDetector)
        ensemble.method = "weighted"
        ensemble.weights = {"isolation_forest": 0.5, "holt_winters": 0.5}
        ensemble._models = {}
        ensemble._max_latency_ms = 100

        if_det = IsolationForestDetector.__new__(IsolationForestDetector)
        if_det.model_name = "isolation_forest"
        if_det.contamination = 0.05
        if_det.n_estimators = 10
        if_det.max_samples = "auto"
        if_det.random_state = 42
        if_det._model = None
        if_det._trained = False
        if_det._feature_count = 0
        if_det.train(sample_metric_data)
        ensemble.register_model("isolation_forest", if_det, weight=0.5)

        result = ensemble.detect(sample_metric_data[0])
        assert result.model_name == "ensemble"
        assert 0.0 <= result.anomaly_score <= 1.0


class TestDriftDetector:
    def test_set_reference_and_check(self):
        detector = DriftDetector.__new__(DriftDetector)
        detector.threshold = 0.15
        detector.retrain_on_drift = True
        detector._reference_histograms = {}
        detector._reference_bin_edges = {}

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        detector.set_reference("test_feature", reference)

        # Same distribution → low PSI
        current_similar = np.random.normal(50, 10, 500)
        result = detector.check_drift("test_feature", current_similar)
        assert result["psi"] < 0.15
        assert result["level"] in ("none", "moderate")

    def test_drift_detected(self):
        detector = DriftDetector.__new__(DriftDetector)
        detector.threshold = 0.15
        detector.retrain_on_drift = True
        detector._reference_histograms = {}
        detector._reference_bin_edges = {}

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        detector.set_reference("test_feature", reference)

        # Very different distribution → high PSI
        current_shifted = np.random.normal(80, 15, 500)
        result = detector.check_drift("test_feature", current_shifted)
        assert result["psi"] > 0.1


class TestModelManager:
    def test_save_and_load(self, sample_metric_data, tmp_path):
        mm = ModelManager(str(tmp_path))

        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        detector.train(sample_metric_data)
        mm.save_model("test_if", detector, version="0.1.0")

        loaded = mm.load_model("test_if")
        assert loaded is not None

    def test_record_inference(self, sample_metric_data, tmp_path):
        mm = ModelManager(str(tmp_path))
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0
        detector.train(sample_metric_data)

        mm.register_model("if", detector)
        mm.record_inference("if", latency_ms=5.0)
        info = mm.get_model_info("if")
        assert info is not None
        assert info.inference_count == 1
