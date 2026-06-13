"""FastAPI application factory — configures middleware, routes, and lifecycle."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY as prom_registry
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

from src.alerting.service import AlertService
from src.api.error_handler import app_error_handler, unhandled_error_handler
from src.api.routes import (
    alert_router,
    auth_router,
    health_router,
    metric_router,
    model_router,
    prediction_router,
    server_router,
)
from src.core.config import get_config
from src.core.errors import AppError
from src.core.health import register_default_checks
from src.core.logging import bind_correlation, clear_correlation, configure_logging, get_logger
from src.ingestion.orchestrator import IngestionOrchestrator
from src.ml_processing.pipeline import MLPipeline

logger = get_logger("api.app")

# Global orchestrator reference (set during startup)
_orchestrator: IngestionOrchestrator | None = None

# ── Prometheus Metrics ────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "endpoint"],
)
ANOMALY_DETECTIONS = Counter(
    "anomaly_detections_total", "Total anomaly detections",
    ["server_id", "severity"],
)
ML_INFERENCE_LATENCY = Histogram(
    "ml_inference_duration_seconds", "ML model inference latency",
    ["model_name"],
)


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    cfg = get_config()

    # ── Startup ──────────────────────────────────────────────────
    configure_logging(
        log_level=cfg["service"]["log_level"],
        json_format=cfg["observability"]["logging"]["format"] == "json",
    )
    logger.info("app.starting", version=cfg["service"]["version"])

    register_default_checks()

    # Initialize orchestrator
    global _orchestrator
    _orchestrator = IngestionOrchestrator(cfg)

    # Initialize ML pipeline
    pipeline = MLPipeline.get_instance()
    pipeline.register_alert_callback(_handle_anomaly)

    # Initialize alert service
    AlertService.get_instance()

    # Connect ingestion → ML pipeline
    _orchestrator.register_handler(pipeline.process_features)

    # Start ingestion
    await _orchestrator.start()

    logger.info("app.started")
    yield

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("app.shutting_down")
    await _orchestrator.stop()
    from src.core.database import close_db
    await close_db()
    logger.info("app.stopped")


async def _handle_anomaly(server_id: str, result, timestamp):
    """Bridge: ML pipeline → Alert service."""
    service = AlertService.get_instance()
    # Extract metric name from result details
    metric_name = result.details.get("metric_name", "unknown")
    await service.process_detection(
        server_id=server_id,
        result=result,
        timestamp=timestamp,
        metric_name=metric_name,
    )
    # Update Prometheus metric
    if result.is_anomaly:
        severity = "critical" if result.anomaly_score > 0.85 else "warning"
        ANOMALY_DETECTIONS.labels(server_id=server_id, severity=severity).inc()


# ── App Factory ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = get_config()

    app = FastAPI(
        title="AI-Driven Infrastructure Anomaly Detector",
        version=cfg["service"]["version"],
        description="Predictive anomaly detection for server infrastructure using ML",
        lifespan=lifespan,
        docs_url="/docs" if cfg["service"]["environment"] != "production" else None,
        redoc_url="/redoc" if cfg["service"]["environment"] != "production" else None,
    )

    # ── CORS ─────────────────────────────────────────────────────
    cors_cfg = cfg["security"]["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg["allowed_origins"],
        allow_methods=cors_cfg["allowed_methods"],
        allow_headers=cors_cfg["allowed_headers"],
        max_age=cors_cfg.get("max_age", 600),
    )

    # ── Middleware: Request ID + Correlation + Timing ─────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4())[:8])
        request.state.request_id = request_id
        bind_correlation(request_id)

        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500,
                content={"error": {"code": "SYSTEM_INTERNAL_ERROR", "message": "Internal error", "request_id": request_id}},
            )
        finally:
            clear_correlation()

        duration = time.monotonic() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        # Record Prometheus metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    # ── Security Headers ─────────────────────────────────────────
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        return response

    # ── Rate Limiting ────────────────────────────────────────────
    if cfg["security"]["rate_limiting"]["enabled"]:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[
                f"{cfg['security']['rate_limiting']['default_limit']}/{cfg['security']['rate_limiting']['default_window']}second"
            ],
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Exception Handlers ───────────────────────────────────────
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    # ── Routes ───────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(alert_router)
    app.include_router(model_router)
    app.include_router(server_router)
    app.include_router(metric_router)
    app.include_router(prediction_router)

    # ── Prometheus Metrics Endpoint ───────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus scrape endpoint."""
        return PlainTextResponse(
            generate_latest(prom_registry).decode("utf-8"),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


async def _validation_error_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors to standard ErrorResponse."""
    from src.api.error_handler import _build_error_response

    details = [
        {"field": str(e.get("loc", ["?"])[-1]), "message": e.get("msg", "Invalid value")}
        for e in exc.errors()
    ]

    return _build_error_response(
        code="VALIDATION_ERROR",
        message="Invalid request body",
        status_code=422,
        request=request,
        details=details,
    )


# ── Entry Point ──────────────────────────────────────────────────

app = create_app()
