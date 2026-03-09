"""API key generator instance for the auth service."""

from dana_common import config
from dana_common.auth import APIKeyGenerator

api_key_generator = APIKeyGenerator(config.auth.api_key_salt)
