"""
Unit tests for async singleton initialization, defensive tool_calls serialization,
and Pydantic request validation in krusch-agentic-proxy (BUG-01, BUG-02, SEC-01, TEST-01).
"""

import asyncio
from unittest.mock import patch
from pydantic import ValidationError
import pytest

import src.api_gateway as gateway
from src.schemas import ChatCompletionRequest


def test_ensure_initialized_async_runs_once_under_concurrency(monkeypatch):
    """Concurrent coroutines calling _ensure_initialized_async initialize singletons only once (BUG-01, TEST-01)."""
    monkeypatch.setattr(gateway, "_config", None)
    monkeypatch.setattr(gateway, "_engine", None)
    monkeypatch.setattr(gateway, "_router", None)

    call_count = 0
    original_init = gateway._initialize_once

    def counting_init():
        nonlocal call_count
        call_count += 1
        original_init()

    with patch.object(gateway, "_initialize_once", side_effect=counting_init):
        async def _run():
            await asyncio.gather(
                gateway._ensure_initialized_async(),
                gateway._ensure_initialized_async(),
                gateway._ensure_initialized_async(),
            )

        asyncio.run(_run())

    assert call_count == 1
    assert gateway._config is not None
    assert gateway._engine is not None
    assert gateway._router is not None


def test_ensure_initialized_sync_and_async_share_unified_lock(monkeypatch):
    """Spawning a sync thread alongside async tasks executes _initialize_once only once via _init_global_lock (BUG-01)."""
    import threading

    monkeypatch.setattr(gateway, "_config", None)
    monkeypatch.setattr(gateway, "_engine", None)
    monkeypatch.setattr(gateway, "_router", None)

    call_count = 0
    original_init = gateway._initialize_once

    def counting_init():
        nonlocal call_count
        call_count += 1
        original_init()

    with patch.object(gateway, "_initialize_once", side_effect=counting_init):
        t = threading.Thread(target=gateway._ensure_initialized)

        async def _run():
            t.start()
            await asyncio.gather(
                gateway._ensure_initialized_async(),
                gateway._ensure_initialized_async(),
            )
            t.join()

        asyncio.run(_run())

    assert call_count == 1


def test_build_response_handles_non_serializable_tool_calls_gracefully(monkeypatch):
    """_build_response surfaces tool_calls_error indicator when tool_calls contain non-JSON-serializable objects and normalizes empty/None tool_calls (BUG-01, BUG-02)."""
    import src.config as config_mod

    assert "RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR" in config_mod.__all__
    assert gateway.RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR is config_mod.RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR
    assert gateway._safe_serialize_tool_calls(None) is None
    assert gateway._safe_serialize_tool_calls([]) is None

    empty_resp = gateway._build_response(
        model="qwen2.5-coder:7b",
        content=None,
        finish_reason="stop",
        tool_calls=[],
        prompt_text="hello",
    )
    assert empty_resp["choices"][0]["message"]["content"] == ""
    assert "tool_calls" not in empty_resp["choices"][0]["message"]

    class UnserializableObj:
        pass

    bad_tool_calls = [{"id": "call_1", "function": {"name": "test", "raw": UnserializableObj()}}]
    resp = gateway._build_response(
        model="qwen2.5-coder:7b",
        content=None,
        finish_reason="tool_calls",
        tool_calls=bad_tool_calls,
        prompt_text="hello world",
    )
    assert resp["object"] == "chat.completion"
    assert resp["usage"]["completion_tokens"] == 0
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp.get("tool_calls_error") == "tool_calls serialization failed"
    assert gateway._safe_serialize_tool_calls(bad_tool_calls) == '{"error": "tool_calls serialization failed"}'

    monkeypatch.setattr(gateway, "RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR", True)
    with pytest.raises(gateway.ToolCallSerializationError, match="tool_calls serialization failed"):
        gateway._safe_serialize_tool_calls(bad_tool_calls)


def test_chat_completion_request_schema_rejects_malformed_payload():
    """ChatCompletionRequest in src/schemas.py rejects empty messages or invalid temperature (SEC-01)."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            temperature=5.0,
        )


def test_unified_configuration_error_and_decoupled_client_imports():
    """src.exceptions.ConfigurationError is shared across src.client and src.router, and src.client has no upward top-level import of WaterfallRouter (ARCH-01, BUG-01)."""
    import src.client as client_mod
    import src.exceptions as exc_mod
    import src.router as router_mod

    assert client_mod.ConfigurationError is exc_mod.ConfigurationError
    assert router_mod.ConfigurationError is exc_mod.ConfigurationError
    assert "WaterfallRouter" not in client_mod.__dict__
