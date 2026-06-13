"""Circuit breaker implementation for external service calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from src.core.logging import get_logger

logger = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Tripped — requests are rejected
    HALF_OPEN = "half_open" # Testing — limited requests allowed


class CircuitBreaker:
    """Thread-safe circuit breaker with configurable thresholds."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("circuit_breaker.closed", breaker=self.name)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warn("circuit_breaker.reopened", breaker=self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warn(
                "circuit_breaker.opened",
                breaker=self.name,
                failure_count=self._failure_count,
            )

    def allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max:
                self._half_open_calls += 1
                return True
            return False
        return False  # OPEN

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute fn through the circuit breaker."""
        if not self.allow_request():
            from src.core.errors import ExternalServiceError
            raise ExternalServiceError(
                self.name,
                f"Circuit breaker '{self.name}' is open — service unavailable",
            )
        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


# ── Pre-configured breakers for external services ─────────────────

breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker by service name."""
    if name not in breakers:
        breakers[name] = CircuitBreaker(name)
    return breakers[name]
