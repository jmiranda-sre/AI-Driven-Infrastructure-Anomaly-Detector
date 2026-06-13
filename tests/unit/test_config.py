"""Unit tests for core.config module."""

import os
from unittest.mock import patch

from src.core.config import _deep_merge, _deep_set, load_config, reset_config


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_deep_nested(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}


class TestDeepSet:
    def test_set_nested(self):
        cfg = {"a": {"b": {}}}
        _deep_set(cfg, ["a", "b", "c"], 42)
        assert cfg["a"]["b"]["c"] == 42

    def test_set_new_path(self):
        cfg = {}
        _deep_set(cfg, ["x", "y", "z"], "val")
        assert cfg["x"]["y"]["z"] == "val"


class TestLoadConfig:
    def test_loads_default(self):
        cfg = load_config()
        assert "service" in cfg
        assert "ml" in cfg
        assert "alerting" in cfg

    def test_service_defaults(self):
        with patch.dict(os.environ, {"SERVICE_ENV": "development"}):
            reset_config()
            cfg = load_config()
            assert cfg["service"]["name"] == "anomaly-detector"
            assert cfg["service"]["port"] == 8000

    def test_env_override(self):
        with patch.dict(os.environ, {"SERVICE_PORT": "9000"}):
            reset_config()
            cfg = load_config()
            assert cfg["service"]["port"] == 9000

    def test_cors_origins_from_env(self):
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://a.com,http://b.com"}):
            reset_config()
            cfg = load_config()
            assert cfg["security"]["cors"]["allowed_origins"] == ["http://a.com", "http://b.com"]

    def test_ad_prefix_override(self):
        with patch.dict(os.environ, {"AD__ML__MODELS__ISOLATION_FOREST__CONTAMINATION": "0.02"}):
            reset_config()
            cfg = load_config()
            assert cfg["ml"]["models"]["isolation_forest"]["contamination"] == "0.02"

    def test_test_environment(self):
        with patch.dict(os.environ, {"SERVICE_ENV": "test"}):
            reset_config()
            cfg = load_config()
            assert cfg["service"]["environment"] == "test"
