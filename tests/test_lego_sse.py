"""
Tests for INV-LEGO-27 migration in krusch-agentic-proxy:
- src/api_gateway.py (_build_sse_response -> sse_event_response passthrough)
- src/router.py (WaterfallRouter._try_stream -> sse_event_response passthrough)
- Standalone CI ImportError fallback when sibling monorepo lib-py is absent
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import importlib.util
import json
import os
import sys
from typing import Any, List
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

import src.api_gateway as api_gateway
import src.router as router_mod


async def _collect_stream_chunks(response: Any) -> List[bytes]:
    chunks: List[bytes] = []
    async with aclosing(response.body_iterator) as body_iter:
        async for chunk in body_iter:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
            else:
                chunks.append(str(chunk).encode("utf-8"))
    return chunks


def test_build_sse_response_headers_and_openai_wire_format():
    resp = api_gateway._build_sse_response(
        model="qwen2.5-coder:7b",
        content="Hello from Lego SSE",
        finish_reason="stop",
        prompt_text="Greet the user",
    )

    assert resp.media_type == "text/event-stream"
    assert resp.headers["X-Accel-Buffering"] == "no"
    assert resp.headers["Cache-Control"] == "no-cache, no-transform"
    assert resp.headers["Connection"] == "keep-alive"

    raw_text = b"".join(asyncio.run(_collect_stream_chunks(resp))).decode("utf-8")
    assert "event: done" not in raw_text
    assert "event:" not in raw_text
    assert ": ping" not in raw_text
    assert raw_text.endswith("data: [DONE]\n\n")

    frames = [frame for frame in raw_text.split("\n\n") if frame]
    assert len(frames) == 3
    assert frames[0].startswith("data: ")
    assert frames[1].startswith("data: ")
    assert frames[2] == "data: [DONE]"

    first_payload = json.loads(frames[0][len("data: ") :])
    final_payload = json.loads(frames[1][len("data: ") :])

    assert first_payload["object"] == "chat.completion.chunk"
    assert first_payload["model"] == "qwen2.5-coder:7b"
    assert first_payload["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "Hello from Lego SSE"},
            "finish_reason": None,
        }
    ]

    assert final_payload["object"] == "chat.completion.chunk"
    assert final_payload["id"] == first_payload["id"]
    assert final_payload["choices"] == [
        {"index": 0, "delta": {}, "finish_reason": "stop"}
    ]
    assert final_payload["usage"]["total_tokens"] >= 2


def test_chat_completions_stream_via_testclient(monkeypatch: pytest.MonkeyPatch):
    api_gateway._ensure_initialized()
    monkeypatch.setattr(
        api_gateway._engine,
        "generate",
        AsyncMock(return_value=("", "Streamed fast-path reply")),
    )

    client = TestClient(api_gateway.app)
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "Ping stream"}],
            "stream": True,
        },
    )

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["x-accel-buffering"] == "no"
    assert res.headers["cache-control"] == "no-cache, no-transform"
    assert "event: done" not in res.text
    assert res.text.endswith("data: [DONE]\n\n")

    frames = [frame for frame in res.text.split("\n\n") if frame]
    assert len(frames) == 3
    first_chunk = json.loads(frames[0][len("data: ") :])
    assert first_chunk["choices"][0]["delta"]["content"] == "Streamed fast-path reply"
    assert frames[2] == "data: [DONE]"


def test_waterfall_router_try_stream_passthrough_and_aclose_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    raw_upstream_frames = [
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class FakeUpstreamResponse:
        def __init__(self, chunks: List[bytes], status_code: int = 200):
            self.status_code = status_code
            self._chunks = chunks
            self.closed = False

        async def aiter_bytes(self):
            for item in self._chunks:
                yield item

        async def aread(self) -> bytes:
            return b"upstream error"

        async def aclose(self) -> None:
            self.closed = True

    created_clients: List[Any] = []
    created_responses: List[FakeUpstreamResponse] = []

    class FakeAsyncClient:
        def __init__(self, timeout: float = 120.0):
            self.timeout = timeout
            self.closed = False
            self.built_requests: List[Any] = []
            created_clients.append(self)

        def build_request(self, method: str, url: str, json: Any = None, headers: Any = None):
            req = {"method": method, "url": url, "json": json, "headers": headers}
            self.built_requests.append(req)
            return req

        async def send(self, req: Any, stream: bool = False) -> FakeUpstreamResponse:
            assert stream is True
            resp = FakeUpstreamResponse(raw_upstream_frames, status_code=200)
            created_responses.append(resp)
            return resp

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(router_mod.httpx, "AsyncClient", FakeAsyncClient)

    wf = router_mod.WaterfallRouter(
        routes=[
            {
                "name": "primary-stream",
                "api_url": "https://api.example.test/v1/chat/completions",
                "api_key": "REDACTED_TEST_TOKEN_12345",
                "model": "gpt-test",
            }
        ]
    )

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def _exercise_full_and_early_abort():
        resp = await wf.route_proxy(
            {"messages": [{"role": "user", "content": "stream test"}]},
            is_stream=True,
        )
        assert resp is not None
        assert resp.media_type == "text/event-stream"
        assert resp.headers["X-Accel-Buffering"] == "no"
        assert resp.headers["Cache-Control"] == "no-cache, no-transform"

        collected = await _collect_stream_chunks(resp)
        assert b"".join(collected) == b"".join(raw_upstream_frames)
        assert b"event: done" not in b"".join(collected)
        assert created_responses[0].closed is True
        assert created_clients[0].closed is True

        # Verify deterministic cleanup on early consumer disconnect after 1st chunk
        resp_early = await wf._try_stream(
            "https://api.example.test/v1/chat/completions",
            {"messages": [{"role": "user", "content": "abort early"}]},
            {"Authorization": "Bearer REDACTED_TEST_TOKEN_12345"},
        )
        assert resp_early is not None
        async with aclosing(resp_early.body_iterator) as body_iter:
            first = await anext(body_iter)
            assert first == raw_upstream_frames[0]
        assert created_responses[1].closed is True
        assert created_clients[1].closed is True

        # Verify deterministic cleanup on pre-iteration disconnect (AG_CREATED regression guard)
        resp_pre_disconnect = await wf.route_proxy(
            {"messages": [{"role": "user", "content": "disconnect before first chunk"}]},
            is_stream=True,
            request=DisconnectedRequest(),
        )
        assert resp_pre_disconnect is not None
        pre_chunks = await _collect_stream_chunks(resp_pre_disconnect)
        assert pre_chunks == []
        assert created_responses[2].closed is True
        assert created_clients[2].closed is True

    asyncio.run(_exercise_full_and_early_abort())


def _load_isolated_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_ci_importerror_fallback(monkeypatch: pytest.MonkeyPatch):
    real_isdir = os.path.isdir

    def _hide_lib_py(path: str) -> bool:
        if os.path.basename(os.path.normpath(str(path))) == "lib-py":
            return False
        return real_isdir(path)

    cleaned_sys_path = [
        entry
        for entry in sys.path
        if os.path.basename(os.path.normpath(entry or ".")) != "lib-py"
    ]
    monkeypatch.setattr(sys, "path", cleaned_sys_path)
    monkeypatch.setattr(os.path, "isdir", _hide_lib_py)
    monkeypatch.setitem(sys.modules, "krusch_toolkit", None)
    monkeypatch.setitem(sys.modules, "krusch_toolkit.fastapi_app", None)

    gateway_path = os.path.abspath(api_gateway.__file__)
    router_path = os.path.abspath(router_mod.__file__)

    standalone_gateway = _load_isolated_module("_standalone_api_gateway", gateway_path)
    standalone_router = _load_isolated_module("_standalone_router", router_path)

    assert standalone_gateway.sse_event_response.__module__ == "_standalone_api_gateway"
    assert standalone_router.sse_event_response.__module__ == "_standalone_router"

    # 1. Verify standalone _build_sse_response preserves exact headers & OpenAI SSE frames
    gw_resp = standalone_gateway._build_sse_response(
        model="qwen2.5-coder:7b",
        content="Standalone CI fallback OK",
        finish_reason="stop",
        prompt_text="ci check",
    )
    assert gw_resp.media_type == "text/event-stream"
    assert gw_resp.headers["X-Accel-Buffering"] == "no"
    assert gw_resp.headers["Cache-Control"] == "no-cache, no-transform"
    assert gw_resp.headers["Connection"] == "keep-alive"

    gw_body = b"".join(asyncio.run(_collect_stream_chunks(gw_resp))).decode("utf-8")
    assert "event: done" not in gw_body
    assert ": ping" not in gw_body
    gw_frames = [f for f in gw_body.split("\n\n") if f]
    assert len(gw_frames) == 3
    assert gw_frames[2] == "data: [DONE]"
    assert (
        json.loads(gw_frames[0][len("data: ") :])["choices"][0]["delta"]["content"]
        == "Standalone CI fallback OK"
    )

    # 2. Verify standalone heartbeat_comment default & generator aclosing
    closed_flag = {"closed": False}

    async def _gen_with_cleanup():
        try:
            yield "data: hello\n\n"
            yield "data: world\n\n"
        finally:
            closed_flag["closed"] = True

    hb_resp = standalone_router.sse_event_response(_gen_with_cleanup(), passthrough=True)
    hb_text = b"".join(asyncio.run(_collect_stream_chunks(hb_resp))).decode("utf-8")
    assert hb_text.startswith(": ping\n\n")
    assert "data: hello\n\ndata: world\n\n" in hb_text
    assert closed_flag["closed"] is True

    # 3. Verify standalone fallback closes source on pre-iteration disconnect
    pre_closed = {"closed": False}

    class _MockCloseableStream:
        async def aclose(self) -> None:
            pre_closed["closed"] = True

        async def __aiter__(self):
            yield b"data: never\n\n"

    class _DisconnectedReq:
        async def is_disconnected(self) -> bool:
            return True

    disc_resp = standalone_router.sse_event_response(
        _MockCloseableStream(),
        request=_DisconnectedReq(),
        passthrough=True,
    )
    assert asyncio.run(_collect_stream_chunks(disc_resp)) == []
    assert pre_closed["closed"] is True
