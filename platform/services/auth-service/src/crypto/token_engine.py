"""Token engine instance for the auth service."""

from dana_common import config
from dana_common.auth import TokenEngine

token_engine = TokenEngine(config.auth.secret_key, config.auth.algorithm)
