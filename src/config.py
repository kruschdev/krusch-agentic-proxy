"""
Unified configuration loader and schema validation for Krusch Agentic Proxy (ARCH-01, SEC-01, BUG-01).
"""

import json
import logging
import os
import sys
from typing import Any, Dict

from pydantic import ValidationError
from src.models import ProxyConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("KRUSCH_LLM_MODEL", "qwen2.5-coder:7b")
DEFAULT_OLLAMA_API_URL = os.getenv(
    "KRUSCH_LLM_API_URL",
    "http://127.0.0.1:11434/v1/chat/completions",
)
# Controls whether _safe_serialize_tool_calls raises ToolCallSerializationError or returns sentinel JSON (BUG-01).
RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR: bool = os.getenv(
    "KRUSCH_RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR", "false"
).strip().lower() in ("1", "true", "yes")

__all__ = (
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_API_URL",
    "RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR",
    "load_config",
)


def load_config(strict: bool = False) -> Dict[str, Any]:
    """Load and validate proxy configuration from disk with schema-validated fallback."""
    config_path = os.environ.get(
        "KRUSCH_PROXY_CONFIG",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
    )
    is_strict = (
        strict
        or os.environ.get("KRUSCH_STRICT_CONFIG", "0") == "1"
        or (
            "--strict-config" in sys.argv
            and "--allow-default" not in sys.argv
        )
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict):
            raise ValidationError.from_exception_data("ProxyConfig", [])
        config = ProxyConfig(**raw_data)
        return config.model_dump(exclude_none=True)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValidationError) as exc:
        logger.error("Failed to load or validate configuration from %s: %s", config_path, exc, exc_info=True)
        if is_strict:
            raise
        fallback = {
            "llm": {
                "model": os.environ.get("KRUSCH_LLM_MODEL", DEFAULT_MODEL),
                "api_url": os.environ.get("KRUSCH_LLM_API_URL", DEFAULT_OLLAMA_API_URL),
            }
        }
        try:
            validated_fallback = ProxyConfig(**fallback)
            return validated_fallback.model_dump(exclude_none=True)
        except ValidationError as fallback_exc:
            logger.error("Fallback configuration failed schema validation: %s", fallback_exc, exc_info=True)
            raise
