"""Test configuration and shared fixtures."""

# Ensure src is importable
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config, reset_config
from src.core.logging import configure_logging


@pytest.fixture(autouse=True)
def setup_test_config():
    """Ensure test configuration is loaded for every test."""
    reset_config()
    import os
    os.environ["SERVICE_ENV"] = "test"
    configure_logging(log_level="debug", json_format=False)
    yield
    reset_config()


@pytest.fixture
def sample_config():
    """Return test configuration dict."""
    return load_config()


@pytest.fixture
def sample_metric_data():
    """Generate synthetic metric data for testing."""
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, n / 288, n)

    cpu = 45 + 20 * np.sin(2 * np.pi * t) + np.random.normal(0, 5, n)
    memory = 60 + 10 * np.sin(2 * np.pi * t + 1) + np.random.normal(0, 3, n)
    data = np.column_stack([np.clip(cpu, 0, 100), np.clip(memory, 0, 100)])
    return data


@pytest.fixture
def sample_univariate():
    """Synthetic univariate time series."""
    np.random.seed(42)
    n = 600
    t = np.linspace(0, n / 288, n)
    return 50 + 20 * np.sin(2 * np.pi * t) + np.random.normal(0, 3, n)
