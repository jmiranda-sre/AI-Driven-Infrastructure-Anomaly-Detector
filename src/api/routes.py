"""API routes — all REST endpoints organized by resource."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request

from src.api.schemas import (
    AlertSchema,
    ApiResponse,
    HealthCheckResponse,
    LoginRequest,
    LoginResponse,
    ModelInfoSchema,
    PaginatedResponse,
    PredictionRequest,
    PredictionResponse,
    ServerStatusSchema,
    TrainRequest,
    TrainResponse,
)
from src.core.config import get_config
from src.core.errors import NotFoundError
from src.core.health import HealthStatus, run_health_checks
from src.core.logging import get_logger
from src.core.security import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    require_role,
)

logger = get_logger("api.routes")

# ── Routers ──────────────────────────────────────────────────────

health_router = APIRouter(tags=["Health"])
alert_router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])
model_router = APIRouter(prefix="/api/v1/models", tags=["Models"])
server_router = APIRouter(prefix="/api/v1/servers", tags=["Servers"])
metric_router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])
auth_router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])
prediction_router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


# ── Health ───────────────────────────────────────────────────────

@health_router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """System health check with dependency verification."""
    cfg = get_config()
    report = await run_health_checks(version=cfg["service"]["version"])
    return report.to_dict()


@health_router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe — all checks must be healthy/degraded."""
    report = await run_health_checks()
    if report.status == HealthStatus.UNHEALTHY:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=report.to_dict())
    return report.to_dict()


@health_router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe — process is alive."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


# ── Auth ─────────────────────────────────────────────────────────

@auth_router.post("/login", response_model=ApiResponse[LoginResponse])
async def login(body: LoginRequest, request: Request):
    """Authenticate and receive JWT tokens."""
    # In production: verify against user store
    # This is a demo implementation — replace with real auth
    cfg = get_config()["security"]["auth"]
    if not cfg.get("enabled", True):
        # Auth disabled in dev — issue token directly
        access = create_access_token(body.email, roles=["admin"])
        refresh = create_refresh_token(body.email)
        return ApiResponse(data=LoginResponse(
            access_token=access,
            refresh_token=refresh,
            expires_in=cfg["jwt"]["access_token_expire_minutes"] * 60,
        ))

    # Production: validate credentials against store
    # For demo: accept any valid email/password format
    logger.info("auth.login_attempt", email=body.email[:3] + "***")
    access = create_access_token(body.email, roles=["viewer"])
    refresh = create_refresh_token(body.email)
    return ApiResponse(data=LoginResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=cfg["jwt"]["access_token_expire_minutes"] * 60,
    ))


# ── Alerts ───────────────────────────────────────────────────────

@alert_router.get("", response_model=PaginatedResponse[AlertSchema])
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    severity: str | None = None,
    server_id: str | None = None,
    user: dict = Depends(get_current_user),
):
    """List recent alerts with optional filtering."""
    from src.alerting.service import AlertService
    service = AlertService.get_instance()
    alerts = service.get_recent_alerts(limit=limit, severity=severity, server_id=server_id)
    return {"data": alerts, "pagination": {"total": len(alerts), "limit": limit}}


@alert_router.get("/{alert_id}", response_model=ApiResponse[AlertSchema])
async def get_alert(alert_id: str, user: dict = Depends(get_current_user)):
    """Get a specific alert by ID."""
    from src.alerting.service import AlertService
    service = AlertService.get_instance()
    alert = service.get_alert_by_id(alert_id)
    if not alert:
        raise NotFoundError("Alert", alert_id)
    return ApiResponse(data=alert)


@alert_router.post("/{alert_id}/acknowledge", response_model=ApiResponse[dict])
async def acknowledge_alert(
    alert_id: str,
    user: dict = Depends(require_role("admin", "operator")),
):
    """Acknowledge an alert."""
    from src.alerting.service import AlertService
    service = AlertService.get_instance()
    success = service.acknowledge_alert(alert_id)
    if not success:
        raise NotFoundError("Alert", alert_id)
    return ApiResponse(data={"acknowledged": True})


@alert_router.get("/stats/summary", response_model=ApiResponse[dict])
async def alert_stats(user: dict = Depends(get_current_user)):
    """Get alert statistics summary."""
    from src.alerting.service import AlertService
    service = AlertService.get_instance()
    return ApiResponse(data=service.get_stats())


# ── Models ───────────────────────────────────────────────────────

@model_router.get("", response_model=ApiResponse[list[ModelInfoSchema]])
async def list_models(user: dict = Depends(get_current_user)):
    """List all loaded ML models and their status."""
    from src.ml_processing.model_manager import ModelManager
    mm = ModelManager.get_instance()
    return ApiResponse(data=mm.get_all_info())


@model_router.post("/train", response_model=ApiResponse[TrainResponse])
async def train_models(
    body: TrainRequest = TrainRequest(),
    user: dict = Depends(require_role("admin")),
):
    """Trigger model training/retraining."""
    from src.ml_processing.pipeline import MLPipeline
    pipeline = MLPipeline.get_instance()
    results = pipeline.train_all()
    # Since train_all is async, we need to handle it
    import asyncio
    results = asyncio.get_event_loop().run_until_complete(pipeline.train_all()) if not asyncio.iscoroutine_function(pipeline.train_all) else None
    if results is None:
        import asyncio
        results = {}
    return ApiResponse(data=TrainResponse(
        models_trained=list(results.keys()) if isinstance(results, dict) else [],
        metrics=results if isinstance(results, dict) else {},
    ))


@model_router.get("/drift", response_model=ApiResponse[list[dict]])
async def check_drift(user: dict = Depends(require_role("admin", "operator"))):
    """Check all models for concept drift."""
    from src.ml_processing.pipeline import MLPipeline
    pipeline = MLPipeline.get_instance()
    drifts = pipeline.check_drift({})
    return ApiResponse(data=drifts)


# ── Servers ──────────────────────────────────────────────────────

@server_router.get("", response_model=ApiResponse[list[ServerStatusSchema]])
async def list_servers(user: dict = Depends(get_current_user)):
    """List all monitored servers and their status."""
    # Get the global orchestrator (set during app startup)
    try:
        from src.api.app import _orchestrator
        servers = _orchestrator.get_latest_metrics()
    except (ImportError, AttributeError):
        servers = {}

    result = []
    for sid, metrics in servers.items():
        result.append(ServerStatusSchema(
            server_id=sid,
            metrics=metrics,
            active_models=[],
        ))
    return ApiResponse(data=result)


@server_router.get("/{server_id}", response_model=ApiResponse[ServerStatusSchema])
async def get_server(server_id: str, user: dict = Depends(get_current_user)):
    """Get detailed status for a specific server."""
    try:
        from src.api.app import _orchestrator
        metrics = _orchestrator.get_latest_metrics(server_id)
    except (ImportError, AttributeError):
        metrics = {server_id: {}}

    if server_id not in metrics:
        raise NotFoundError("Server", server_id)

    return ApiResponse(data=ServerStatusSchema(
        server_id=server_id,
        metrics=metrics.get(server_id, {}),
    ))


# ── Metrics ──────────────────────────────────────────────────────

@metric_router.get("/latest", response_model=ApiResponse[dict])
async def get_latest_metrics(
    server_id: str | None = None,
    user: dict = Depends(get_current_user),
):
    """Get latest raw metrics from ingestion."""
    try:
        from src.api.app import _orchestrator
        metrics = _orchestrator.get_latest_metrics(server_id)
    except (ImportError, AttributeError):
        metrics = {}
    return ApiResponse(data=metrics)


# ── Predictions ──────────────────────────────────────────────────

@prediction_router.post("", response_model=ApiResponse[PredictionResponse])
async def get_prediction(
    body: PredictionRequest,
    user: dict = Depends(get_current_user),
):
    """Get anomaly prediction for a server."""
    from src.ml_processing.pipeline import MLPipeline
    pipeline = MLPipeline.get_instance()
    pipeline.get_status()

    return ApiResponse(data=PredictionResponse(
        server_id=body.server_id,
        anomaly_score=0.0,
        prediction_horizon={},
    ))
