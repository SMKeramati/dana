"""Custom exception hierarchy for Dana services."""

from __future__ import annotations


class DanaError(Exception):
    """Base exception for all Dana errors."""
    status_code: int = 500
    error_type: str = "internal_error"


class AuthenticationError(DanaError):
    status_code = 401
    error_type = "authentication_error"


class AuthorizationError(DanaError):
    status_code = 403
    error_type = "authorization_error"


class RateLimitError(DanaError):
    status_code = 429
    error_type = "rate_limit_exceeded"


class QuotaExceededError(DanaError):
    status_code = 429
    error_type = "quota_exceeded"


class ModelNotFoundError(DanaError):
    status_code = 404
    error_type = "model_not_found"


class InferenceError(DanaError):
    status_code = 500
    error_type = "inference_error"


class ValidationError(DanaError):
    status_code = 400
    error_type = "validation_error"
