"""API error handlers — convert AppError hierarchy to standard ErrorResponse."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from src.core.errors import AppError, RateLimitError
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

    # Add Retry-After header for rate limit responses
    if status_code == 429:
        retry_after = 60
        if retryable and details:
            for d in (details if isinstance(details, list) else []):
                if isinstance(d, dict) and "retry_after" in d:
                    retry_after = d["retry_after"]
                    break
        response.headers["Retry-After"] = str(retry_after)

    return response


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle all AppError subtypes."""
    logger.warning(
        "api.app_error",
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        context=exc.context,
    )

    details = []
    if "fields" in exc.context:
        details = [{"field": k, "message": v} for k, v in exc.context["fields"].items()]

    # Include retry_after in details for rate limit errors
    if isinstance(exc, RateLimitError) and "retry_after" in exc.context:
        details.append(exc.context)

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
