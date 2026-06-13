"""JWT authentication and authorization middleware."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import get_config
from src.core.errors import AuthError, ForbiddenError, TokenExpiredError
from src.core.logging import get_logger

logger = get_logger("auth")

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_jwt_config() -> dict:
    cfg = get_config()["security"]["auth"]["jwt"]
    if not cfg.get("secret"):
        raise AuthError("JWT secret not configured — set JWT_SECRET env var")
    return cfg


def create_access_token(
    subject: str,
    roles: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT access token."""
    cfg = _get_jwt_config()
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iss": cfg["issuer"],
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=cfg["access_token_expire_minutes"]),
        "jti": secrets.token_urlsafe(16),
        "roles": roles or ["viewer"],
        **(extra_claims or {}),
    }
    return jwt.encode(payload, cfg["secret"], algorithm=cfg["algorithm"])


def create_refresh_token(subject: str) -> str:
    """Create a signed JWT refresh token (longer TTL)."""
    cfg = _get_jwt_config()
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iss": cfg["issuer"],
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=cfg["refresh_token_expire_days"]),
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, cfg["secret"], algorithm=cfg["algorithm"])


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token. Raises on invalid/expired."""
    cfg = _get_jwt_config()
    try:
        payload = jwt.decode(
            token,
            cfg["secret"],
            algorithms=[cfg["algorithm"]],
            issuer=cfg["issuer"],
            options={"require": ["sub", "exp", "iss"]},
        )
        return payload
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError("Access token expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}") from e


def _token_revocation_key(jti: str) -> str:
    """Generate a revocation check key (for future Redis integration)."""
    return f"token_revoked:{jti}"


# ── FastAPI Dependencies ──────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI dependency: validate Bearer token and return user claims."""
    cfg = get_config()["security"]["auth"]

    if not cfg.get("enabled", True):
        return {"sub": "anonymous", "roles": ["admin"]}

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    claims = decode_token(credentials.credentials)

    # Check if refresh token is used for access (not allowed)
    if claims.get("type") == "refresh":
        raise AuthError("Refresh token cannot be used for API access")

    return claims


def require_role(*required_roles: str):
    """FastAPI dependency factory: require one of the given roles."""
    async def _check(claims: dict = Depends(get_current_user)) -> dict:
        user_roles = set(claims.get("roles", []))
        if not user_roles.intersection(required_roles):
            raise ForbiddenError(
                f"Required role: one of {required_roles}",
                context={"user_roles": list(user_roles), "required": list(required_roles)},
            )
        return claims
    return _check


def hash_api_key(key: str) -> str:
    """Hash an API key for storage (SHA-256)."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash."""
    return secrets.compare_digest(hash_api_key(key), stored_hash)
