"""
Unified exceptions and configuration helpers for Krusch Agentic Proxy (ARCH-01 / BUG-01).
"""

import logging
import os

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError, ValueError):
    """Unified configuration error for the project."""

    pass


class RouteConfigurationError(ConfigurationError):
    """Raised when a waterfall route is missing required fields such as api_url."""

    pass


def resolve_api_key(raw_key: str, *, strict: bool = False) -> str:
    """Resolve an API key, supporting ENV: prefix for environment variable lookup."""
    if not raw_key:
        if strict:
            logger.error("API key is empty")
            raise ConfigurationError("API key is empty")
        return ""
    if raw_key.startswith("ENV:"):
        env_var = raw_key[4:].strip()
        resolved = os.environ.get(env_var, "")
        if not resolved:
            logger.error(f"Environment variable '{env_var}' for API key is not set or empty")
            if strict:
                raise ConfigurationError(
                    f"Environment variable '{env_var}' for API key is not set or empty"
                )
        return resolved
    return raw_key
