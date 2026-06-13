"""API error handlers — convert AppError hierarchy to standard ErrorResponse."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from src.core.errors import AppError
from src.core.logging import get_logger

logger = get_logger("api.error_handler")


def _build_error_response(
    code: str,
    message: str,
    status_code: int,
    request: Request,
    details: list | None = None,
    retryable: bool = False,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4())[:8])

    body = {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
            "request_id": request_id,
            "retryable": retryable,
        }
    }

    response = JSONResponse(status_code=status_code, content=body)

    # Don't leak retry info in production
    if isinstance(None, type(retryable)) or retryable:
        pass  # Could add Retry-After header for rate limits

    return response


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle all AppError subtypes."""
    logger.warn(
        "api.app_error",
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        context=exc.context,
    )

    details = []
    if "fields" in exc.context:
        details = [{"field": k, "message": v} for k, v in exc.context["fields"].items()]

    return _build_error_response(
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        request=request,
        details=details,
        retryable=exc.retryable,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors — log full detail, return generic message."""
    request_id = getattr(request.state, "request_id", str(uuid4())[:8])

    logger.error(
        "api.unhandled_error",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
        path=str(request.url),
    )

    # Never expose internal error details in production
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "SYSTEM_INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            }
        },
    )
