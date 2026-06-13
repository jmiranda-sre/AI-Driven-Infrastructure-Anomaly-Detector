"""API response schemas — standardized envelope for all API responses."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standard success envelope."""
    data: T


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated collection response."""
    data: list[T]
    pagination: dict = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    """Single field validation error."""
    field: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error envelope — DOMAIN_TYPE_DETAIL codes."""
    error: dict  # code, message, details, request_id, documentation_url


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: int
    checks: dict[str, Any]


# ── Domain Schemas ──────────────────────────────────────────────


class MetricPointSchema(BaseModel):
    name: str
    value: float
    timestamp: str
    server_id: str
    labels: dict[str, str] = {}


class AlertSchema(BaseModel):
    alert_id: str
    server_id: str
    metric_name: str
    anomaly_score: float
    severity: str
    is_predicted: bool = False
    message: str = ""
    suggested_action: str = ""
    timestamp: str
    correlation_id: str = ""


class ModelInfoSchema(BaseModel):
    name: str
    version: str
    algorithm: str
    loaded_at: str
    inference_count: int = 0
    avg_latency_ms: float = 0.0


class ServerStatusSchema(BaseModel):
    server_id: str
    metrics: dict[str, float] = {}
    active_models: list[str] = []
    has_anomaly: bool = False


class TrainRequest(BaseModel):
    server_id: str | None = None
    force: bool = False


class TrainResponse(BaseModel):
    models_trained: list[str]
    metrics: dict[str, Any] = {}
    duration_seconds: float = 0.0


class PredictionRequest(BaseModel):
    server_id: str
    metric_name: str | None = None
    horizon_steps: int = 24


class PredictionResponse(BaseModel):
    server_id: str
    prediction_horizon: dict[str, Any] = {}
    anomaly_score: float = 0.0


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
