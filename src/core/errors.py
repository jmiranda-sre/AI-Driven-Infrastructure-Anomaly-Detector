"""Hierarchical error classes with retryable classification.

Every error in the system inherits from AppError, providing:
- Structured error code (DOMAIN_TYPE_DETAIL)
- HTTP status mapping
- Retryable flag for circuit breaker / retry logic
- Context dict for structured logging
"""

from __future__ import annotations


class AppError(Exception):
    """Base application error — all errors inherit from this."""

    status_code: int = 500
    retryable: bool = False
    log_level: str = "error"
    code: str = "SYSTEM_INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        code: str | None = None,
        context: dict | None = None,
    ):
        self.message = message
        self.code = code or self.code
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context,
        }


# ── Validation Errors ──────────────────────────────────────────

class ValidationError(AppError):
    status_code = 400
    retryable = False
    log_level = "warn"
    code = "VALIDATION_ERROR"

    def __init__(self, message: str, fields: dict[str, str] | None = None, **kw):
        ctx = {**(kw.get("context") or {}), "fields": fields or {}}
        super().__init__(message, context=ctx)


# ── Auth Errors ─────────────────────────────────────────────────

class AuthError(AppError):
    status_code = 401
    retryable = False
    log_level = "warn"
    code = "AUTH_INVALID_CREDENTIALS"


class TokenExpiredError(AppError):
    status_code = 401
    retryable = True
    log_level = "warn"
    code = "AUTH_TOKEN_EXPIRED"


class ForbiddenError(AppError):
    status_code = 403
    retryable = False
    log_level = "warn"
    code = "AUTH_FORBIDDEN"


# ── Resource Errors ─────────────────────────────────────────────

class NotFoundError(AppError):
    status_code = 404
    retryable = False
    log_level = "debug"
    code = "RESOURCE_NOT_FOUND"

    def __init__(self, resource: str, resource_id: str, **kw):
        super().__init__(
            f"{resource} with id '{resource_id}' not found",
            context={"resource": resource, "resource_id": resource_id},
        )


class ConflictError(AppError):
    status_code = 409
    retryable = False
    log_level = "warn"
    code = "RESOURCE_CONFLICT"


# ── Rate Limit ──────────────────────────────────────────────────

class RateLimitError(AppError):
    status_code = 429
    retryable = True
    log_level = "warn"
    code = "AUTH_RATE_LIMITED"

    def __init__(self, retry_after: int = 60, **kw):
        super().__init__(
            f"Too many requests. Retry after {retry_after}s.",
            context={"retry_after": retry_after},
        )


# ── External Service Errors ─────────────────────────────────────

class ExternalServiceError(AppError):
    status_code = 502
    retryable = True
    log_level = "error"
    code = "SYSTEM_EXTERNAL_ERROR"

    def __init__(self, service: str, message: str, **kw):
        super().__init__(message, context={"service": service, **(kw.get("context") or {})})


class PrometheusQueryError(ExternalServiceError):
    code = "INGESTION_PROMETHEUS_ERROR"

    def __init__(self, message: str, **kw):
        super().__init__("prometheus", message, **kw)


class KafkaIngestionError(ExternalServiceError):
    code = "INGESTION_KAFKA_ERROR"

    def __init__(self, message: str, **kw):
        super().__init__("kafka", message, **kw)


# ── ML Errors ────────────────────────────────────────────────────

class ModelNotFoundError(NotFoundError):
    code = "ML_MODEL_NOT_FOUND"

    def __init__(self, model_name: str, **kw):
        super().__init__("Model", model_name)


class ModelLoadError(AppError):
    status_code = 500
    retryable = True
    log_level = "error"
    code = "ML_MODEL_LOAD_ERROR"

    def __init__(self, model_name: str, message: str, **kw):
        super().__init__(message, context={"model_name": model_name})


class PredictionError(AppError):
    status_code = 500
    retryable = True
    log_level = "error"
    code = "ML_PREDICTION_ERROR"


# ── Timeout Errors ───────────────────────────────────────────────

class TimeoutError(AppError):
    status_code = 504
    retryable = True
    log_level = "error"
    code = "SYSTEM_TIMEOUT"

    def __init__(self, operation: str, timeout_s: float, **kw):
        super().__init__(
            f"{operation} timed out after {timeout_s}s",
            context={"operation": operation, "timeout_s": timeout_s},
        )


class DatabaseError(AppError):
    status_code = 500
    retryable = True
    log_level = "error"
    code = "SYSTEM_DATABASE_ERROR"
