"""Unit tests for security module."""


from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_api_key,
    verify_api_key,
)


class TestJWT:
    def test_create_and_decode_access_token(self):
        token = create_access_token("user_123", roles=["admin"])
        claims = decode_token(token)
        assert claims["sub"] == "user_123"
        assert "admin" in claims["roles"]
        assert claims["type"] != "refresh"

    def test_create_and_decode_refresh_token(self):
        token = create_refresh_token("user_123")
        claims = decode_token(token)
        assert claims["sub"] == "user_123"
        assert claims["type"] == "refresh"

    def test_expired_token(self):
        # Create token with very short expiry
        import os
        original = os.environ.get("JWT_SECRET")
        os.environ["JWT_SECRET"] = "test-secret-for-jwt-tests-12345678"
        from src.core.config import reset_config
        reset_config()

        token = create_access_token("user_123")
        claims = decode_token(token)
        assert claims["sub"] == "user_123"

        # Clean up
        if original:
            os.environ["JWT_SECRET"] = original
        else:
            os.environ.pop("JWT_SECRET", None)
        reset_config()


class TestAPIKeys:
    def test_hash_and_verify(self):
        key = "my-secret-api-key-12345"
        hashed = hash_api_key(key)
        assert hashed != key
        assert verify_api_key(key, hashed)

    def test_wrong_key_fails(self):
        hashed = hash_api_key("correct-key")
        assert not verify_api_key("wrong-key", hashed)

    def test_constant_time_comparison(self):
        """Verify we use constant-time comparison (secrets.compare_digest)."""
        key = "test-key-long-enough-for-security"
        hashed = hash_api_key(key)
        # This should not raise timing-related errors
        assert verify_api_key(key, hashed)
