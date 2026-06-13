"""Integration tests — full pipeline and API testing."""

from datetime import UTC, datetime

import numpy as np
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoints."""

    def test_health_returns_status(self):
        from src.api.app import create_app
        app = create_app()
        # Override lifespan for testing
        from fastapi import FastAPI

        test_app = FastAPI()
        from src.api.routes import health_router
        test_app.include_router(health_router)
        client = TestClient(test_app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_liveness(self):
        from fastapi import FastAPI

        from src.api.routes import health_router
        test_app = FastAPI()
        test_app.include_router(health_router)
        client = TestClient(test_app)
        resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"


class TestMetricPipeline:
    """Test the full metric → feature → detection → alert pipeline."""

    def test_feature_engineering_pipeline(self):
        from src.ingestion.feature_engineering import FeatureEngineer
        from src.ingestion.models import MetricPoint

        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe.window_sizes = [5]
        fe.feature_names = ["mean", "std", "min", "max"]
        fe._windows = {}
        fe._max_window = 1000

        ts = datetime.now(UTC)
        # Push normal data
        for i in range(50):
            fe.push(MetricPoint(name="cpu", value=45.0 + np.random.normal(0, 3), timestamp=ts, server_id="s1"))

        features = fe.compute_features("s1", "cpu")
        assert features is not None
        assert "cpu_mean_5m" in features

    def test_isolation_forest_end_to_end(self):
        from src.ml_processing.isolation_forest import IsolationForestDetector

        # Generate data
        np.random.seed(42)
        data = np.random.normal(50, 10, (500, 4))

        # Train
        detector = IsolationForestDetector.__new__(IsolationForestDetector)
        detector.model_name = "isolation_forest"
        detector.contamination = 0.05
        detector.n_estimators = 10
        detector.max_samples = "auto"
        detector.random_state = 42
        detector._model = None
        detector._trained = False
        detector._feature_count = 0

        metrics = detector.train(data)
        assert detector.is_trained

        # Detect normal
        normal_point = np.array([50.0, 50.0, 50.0, 50.0])
        result = detector.detect(normal_point)
        assert result.anomaly_score < 0.8

        # Detect anomaly
        extreme_point = np.array([95.0, 95.0, 95.0, 95.0])
        result = detector.detect(extreme_point)
        assert result.anomaly_score > result.anomaly_score or result.is_anomaly
