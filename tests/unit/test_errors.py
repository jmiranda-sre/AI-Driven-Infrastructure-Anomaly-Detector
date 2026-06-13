"""Unit tests for core.errors module."""

from src.core.errors import (
    AppError,
    AuthError,
    ConflictError,
    ForbiddenError,
    KafkaIngestionError,
    ModelLoadError,
    ModelNotFoundError,
    NotFoundError,
    PrometheusQueryError,
    RateLimitError,
    TimeoutError,
    TokenExpiredError,
    ValidationError,
)


class TestAppError:
    def test_base_error_defaults(self):
        err = AppError("something broke")
        assert err.status_code == 500
        assert err.retryable is False
        assert err.log_level == "error"
        assert err.code == "SYSTEM_INTERNAL_ERROR"
        assert err.message == "something broke"

    def test_to_dict(self):
        err = AppError("test", code="TEST_CODE", context={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "TEST_CODE"
        assert d["message"] == "test"
        assert d["retryable"] is False
        assert d["context"] == {"key": "val"}


class TestValidationError:
    def test_fields_context(self):
        err = ValidationError("bad input", fields={"email": "invalid format"})
        assert err.status_code == 400
        assert err.context["fields"]["email"] == "invalid format"

    def test_not_retryable(self):
        err = ValidationError("bad")
        assert err.retryable is False


class TestAuthErrors:
    def test_auth_error(self):
        err = AuthError("bad credentials")
        assert err.status_code == 401
        assert err.code == "AUTH_INVALID_CREDENTIALS"

    def test_token_expired(self):
        err = TokenExpiredError("expired")
        assert err.status_code == 401
        assert err.retryable is True
        assert err.code == "AUTH_TOKEN_EXPIRED"

    def test_forbidden(self):
        err = ForbiddenError("no access")
        assert err.status_code == 403


class TestResourceErrors:
    def test_not_found(self):
        err = NotFoundError("User", "usr_123")
        assert err.status_code == 404
        assert "usr_123" in err.message
        assert err.context["resource"] == "User"

    def test_conflict(self):
        err = ConflictError("already exists")
        assert err.status_code == 409


class TestExternalServiceErrors:
    def test_prometheus_query_error(self):
        err = PrometheusQueryError("query failed")
        assert err.status_code == 502
        assert err.retryable is True
        assert err.context["service"] == "prometheus"

    def test_kafka_error(self):
        err = KafkaIngestionError("connection lost")
        assert err.context["service"] == "kafka"


class TestRateLimitError:
    def test_retry_after(self):
        err = RateLimitError(retry_after=60)
        assert err.status_code == 429
        assert err.retryable is True
        assert "60" in err.message


class TestTimeoutError:
    def test_timeout_details(self):
        err = TimeoutError("db_query", 5.0)
        assert err.status_code == 504
        assert err.context["operation"] == "db_query"
        assert err.context["timeout_s"] == 5.0


class TestModelError:
    def test_model_not_found(self):
        err = ModelNotFoundError("isolation_forest")
        assert err.status_code == 404
        assert err.code == "ML_MODEL_NOT_FOUND"

    def test_model_load_error(self):
        err = ModelLoadError("lstm", "corrupt file")
        assert err.retryable is True
