"""
Krusch Agentic Proxy — OpenAI-Compatible API Gateway.

Serves as a FastAPI REST endpoint on port 5440 (or PORT env var), routing requests
through the Dual-Engine pipeline, waterfall cloud routing, or fast-path NLP.
"""

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Ensure project root is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import close_http_client, get_http_client
from src.core import (
    KruschEngine,
    compact_messages_with_raw_rag,
    compact_tool_schemas,
    format_tools_for_prompt,
    parse_tool_calls,
    select_relevant_tools,
)
from pydantic import ValidationError
from src.schemas import ChatCompletionRequest, ProxyConfig
from src.router import WaterfallRouter
from src.tools import INTERNAL_TOOLS, execute_internal_tool

logger = logging.getLogger(__name__)

import threading
from src.config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_API_URL,
    RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR,
    load_config,
)

# Max wall-clock time for the autonomous agent loop (seconds)
AGENT_TIMEOUT = float(os.environ.get("KRUSCH_AGENT_TIMEOUT", "120"))


# --- Application Lifecycle ---
_engine: Optional[KruschEngine] = None
_router: Optional[WaterfallRouter] = None
_config: Optional[Dict[str, Any]] = None
# Unified re-entrant lock shared across synchronous and asynchronous initialization paths (BUG-01).
_init_global_lock = threading.RLock()


class ToolCallSerializationError(Exception):
    """Raised when tool_calls cannot be serialized and RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR is enabled."""


_TOOL_CALL_ERROR_SENTINEL = json.dumps({"error": "tool_calls serialization failed"})


def _initialize_once() -> None:
    """Internal once-only initialization helper guarded by _init_global_lock."""
    global _engine, _router, _config
    with _init_global_lock:
        if _config is None:
            _config = load_config()
        if _engine is None:
            _engine = KruschEngine(_config)
        if _router is None:
            _router = WaterfallRouter(
                _config.get("waterfall_routes", []),
                client=get_http_client(120.0),
            )


def _ensure_initialized() -> None:
    """Lazy-initialize engine and router synchronously with unified _init_global_lock guard (BUG-01)."""
    if _config is not None and _engine is not None and _router is not None:
        return
    with _init_global_lock:
        if _config is not None and _engine is not None and _router is not None:
            return
        _initialize_once()


async def _ensure_initialized_async() -> None:
    """Concurrency-safe async singleton initialization delegating to unified _init_global_lock (BUG-01)."""
    if _config is not None and _engine is not None and _router is not None:
        return
    await asyncio.to_thread(_ensure_initialized)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine and router on startup, clean up HTTP pool on shutdown."""
    await _ensure_initialized_async()
    logger.info("[Gateway] Engine and WaterfallRouter initialized.")
    yield
    await close_http_client()
    logger.info("[Gateway] Shutting down and closed HTTP connection pool.")


app = FastAPI(title="Krusch Agentic Proxy (OpenAI API)", version="0.4.0", lifespan=lifespan)


@app.middleware("http")
async def verify_api_key_middleware(request: Request, call_next):
    """
    Optional Bearer token verification middleware when KRUSCH_PROXY_API_KEY is set (PROXY-02).
    Exempts /health for liveness probes.
    """
    required_key = os.environ.get("KRUSCH_PROXY_API_KEY", "").strip()
    if required_key and request.url.path != "/health":
        auth_header = request.headers.get("Authorization", "")
        bearer_token = (
            auth_header[len("Bearer ") :].strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        x_api_key = request.headers.get("X-API-Key", "").strip()
        provided_token = bearer_token or x_api_key
        if not provided_token or not hmac.compare_digest(provided_token, required_key):
            return JSONResponse(
                {"error": "Unauthorized: invalid or missing Bearer token."},
                status_code=401,
            )
    return await call_next(request)


def _estimate_tokens(text: Optional[str]) -> int:
    """Fast deterministic token approximation (~4 chars/token) for OpenAI usage fields."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _safe_serialize_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Safely serialize tool_calls to JSON string, returning None when falsy or structured error sentinel on failure (BUG-02)."""
    if not tool_calls:
        return None
    try:
        return json.dumps(tool_calls)
    except (TypeError, ValueError) as exc:
        if RAISE_ON_TOOL_CALL_SERIALIZATION_ERROR:
            raise ToolCallSerializationError(f"tool_calls serialization failed: {exc}") from exc
        logger.warning(
            "[Gateway] Failed to JSON-serialize tool_calls (%s); returning error payload.",
            exc,
        )
        return _TOOL_CALL_ERROR_SENTINEL


def _build_response(
    model: str,
    content: Optional[str],
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_text: str = "",
) -> Dict[str, Any]:
    """Build a standard OpenAI-compatible response dict including usage statistics."""
    normalized_content: Optional[str] = content if content is not None else ("" if not tool_calls else None)
    msg: Dict[str, Any] = {"role": "assistant", "content": normalized_content}
    if tool_calls:
        msg["content"] = None
        msg["tool_calls"] = tool_calls

    prompt_tokens = _estimate_tokens(prompt_text)
    serialized_tools = _safe_serialize_tool_calls(tool_calls) if content is None else None
    has_tool_error = serialized_tools == _TOOL_CALL_ERROR_SENTINEL
    completion_basis = (
        content
        if content is not None
        else ("" if (has_tool_error or not serialized_tools) else serialized_tools)
    )
    completion_tokens = _estimate_tokens(completion_basis)

    resp: Dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if has_tool_error:
        resp["tool_calls_error"] = "tool_calls serialization failed"
    return resp


def _build_sse_response(
    model: str,
    content: Optional[str],
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_text: str = "",
) -> StreamingResponse:
    """Build an OpenAI-compatible Server-Sent Events (SSE) chat.completion.chunk stream."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_basis = content if content is not None else _safe_serialize_tool_calls(tool_calls)
    completion_tokens = _estimate_tokens(completion_basis)

    delta: Dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        delta["content"] = None
        delta["tool_calls"] = [
            {"index": idx, **tc} for idx, tc in enumerate(tool_calls)
        ]
    else:
        delta["content"] = content or ""

    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    async def _event_generator():
        yield f"data: {json.dumps(first_chunk)}\n\n"
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _format_completion_response(
    model: str,
    content: Optional[str],
    stream: bool = False,
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_text: str = "",
):
    """Return either an SSE StreamingResponse or a standard JSONResponse."""
    if stream:
        return _build_sse_response(
            model=model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            prompt_text=prompt_text,
        )
    return JSONResponse(
        _build_response(
            model=model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            prompt_text=prompt_text,
        )
    )


def _get_ollama_host(for_embeddings: bool = False) -> str:
    """Resolve the base Ollama host URL (without /v1/...)."""
    _ensure_initialized()
    if for_embeddings:
        embed_host = os.environ.get("OLLAMA_EMBED_HOST") or os.environ.get("KRUSCH_EMBED_URL")
        if embed_host:
            return embed_host.rstrip("/").split("/v1/")[0]
    target_url = (_config or {}).get("llm", {}).get("api_url", DEFAULT_OLLAMA_API_URL)
    return target_url.split("/v1/")[0].rstrip("/")


# --- Endpoints ---


@app.get("/health")
async def health():
    """Health check endpoint for monitoring and benchmark scripts."""
    return {"status": "ok", "engine": "krusch-agentic-proxy", "version": "0.4.0"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _ensure_initialized()
    assert _engine is not None and _router is not None and _config is not None

    try:
        raw_data = await request.json()
        validated = ChatCompletionRequest(**raw_data)
    except Exception as e:
        return JSONResponse({"error": f"Invalid request: {str(e)}"}, status_code=400)

    data = raw_data
    messages = [msg.model_dump(exclude_none=True) for msg in validated.messages]
    model = validated.model or _config.get("llm", {}).get("model", "qwen2.5-coder:7b")

    # --- WATERFALL AUTO-CLOUD ROUTING ---
    if model == "auto-cloud" or os.environ.get("USE_AUTO_CLOUD") == "1":
        logger.info("[Gateway] 'auto-cloud' detected. Intercepting for Waterfall Auto-Routing.")
        is_stream = validated.stream or False
        return await _router.route_proxy(data, is_stream=is_stream)

    # Extract the last user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    if not user_prompt:
        user_prompt = "No user prompt provided."

    logger.info(f"[Gateway] Request for model {model}. Prompt length: {len(user_prompt)}")

    tool_choice_none = (
        isinstance(validated.tool_choice, str)
        and validated.tool_choice.strip().lower() == "none"
    )
    provided_tools = [] if tool_choice_none else (validated.tools or [])
    is_stream = bool(validated.stream)
    is_passive_mode = os.environ.get("PASSIVE_MODE") == "1"
    # PROXY-02: Never execute autonomous host commands when PASSIVE_MODE=1
    force_autonomous = False if is_passive_mode else bool(validated.force_autonomous)
    force_dual_engine = os.environ.get("FORCE_DUAL_ENGINE") == "1"

    has_inline_tools = any(
        marker in user_prompt.lower()
        for marker in (
            "available tools:",
            "function signature:",
            '"parameters":',
            '"type": "function"',
            '"type":"function"',
        )
    )
    is_tool_request = (
        False
        if tool_choice_none
        else (bool(provided_tools) or (is_passive_mode and has_inline_tools))
    )

    # --- SMART ROUTING GATE ---
    model_lower = model.lower()
    is_small_model = any(tag in model_lower for tag in ["0.5b", "1.5b", "2b", "3b"])

    if is_small_model and (is_tool_request or force_dual_engine) and not is_passive_mode:
        logger.warning(
            f"[ROUTING GATE]: {model} detected as Small Model (<= 3B). "
            f"Dual-Engine will likely degrade performance due to Cognitive Penalty."
        )

    # --- SOURCE-PRESERVING MULTI-TURN RAW-RAG COMPACTION (arXiv:2610.30755) ---
    compaction = compact_messages_with_raw_rag(messages, query=str(user_prompt or ""))
    effective_messages = compaction["messages"]
    if compaction["compacted"]:
        logger.info(
            "[Gateway] Raw-RAG compacted multi-turn history: %d -> %d tokens (%d recalled windows, policy=%s).",
            compaction["original_tokens"],
            compaction["compacted_tokens"],
            compaction["recalled_windows"],
            compaction["budget_policy"],
        )

    # --- FAST-PATH: Standard NLP ---
    if not is_tool_request and not force_autonomous and not force_dual_engine:
        logger.info("[Gateway] Fast-path routing: Standard NLP request.")

        chat_context = ""
        for msg in effective_messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"

        try:
            _, response_text = await _engine.generate(
                prompt=chat_context,
                is_code_exec=False,
                is_tool_call=False,
                target_model=model,
                max_tokens=validated.max_tokens or 2048,
                temperature=validated.temperature if validated.temperature is not None else 0.6,
            )
        except RuntimeError as e:
            logger.error(f"[Gateway] Upstream LLM failure in standard NLP: {e}")
            return JSONResponse({"error": f"Upstream LLM failure: {str(e)}"}, status_code=502)

        return _format_completion_response(
            model=model,
            content=response_text,
            stream=is_stream,
            prompt_text=chat_context,
        )

    # --- FORCED DUAL-ENGINE NLP (Benchmarking) ---
    if force_dual_engine and not is_tool_request and not force_autonomous:
        logger.info("[Gateway] Forced Dual-Engine NLP routing active.")
        chat_context = ""
        for msg in effective_messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"

        try:
            blueprint, response_text = await _engine.generate(
                prompt=chat_context,
                is_code_exec=False,
                is_tool_call=False,
                target_model=model,
                max_tokens=validated.max_tokens or 2048,
                temperature=0.1,
            )
        except RuntimeError as e:
            logger.error(f"[Gateway] Upstream LLM failure in Dual-Engine NLP: {e}")
            return JSONResponse({"error": f"Upstream LLM failure: {str(e)}"}, status_code=502)

        combined = response_text
        if blueprint:
            combined = f"<holodata>\n{blueprint}\n</holodata>\n\n{response_text}"

        return _format_completion_response(
            model=model,
            content=combined,
            stream=is_stream,
            prompt_text=chat_context,
        )

    # --- TOOL-CALLING MODE ---
    if is_tool_request and not force_autonomous:
        logger.info("[Gateway] Tool-calling Mode Active.")
        chat_history = ""
        user_messages = [
            msg
            for msg in messages
            if msg.get("role", "user").upper() == "USER" and msg.get("content")
        ]

        # PROXY-03: Only exclude effective_messages[-1] if it is the trailing USER message already
        # extracted as latest_objective; preserve trailing TOOL or ASSISTANT messages.
        if effective_messages and effective_messages[-1].get("role", "").upper() == "USER":
            history_msgs = effective_messages[:-1]
        else:
            history_msgs = list(effective_messages)

        for msg in history_msgs:
            role = msg.get("role", "user").upper()
            content = msg.get("content")
            tool_calls_data = msg.get("tool_calls")

            if role == "SYSTEM" and content:
                chat_history += f"SYSTEM: {content}\n"
            elif role == "DEVELOPER" and content:
                chat_history += f"DEVELOPER: {content}\n"
            elif role == "USER" and content:
                chat_history += f"USER: {content}\n"
            elif role == "ASSISTANT":
                if content:
                    chat_history += f"ASSISTANT: {content}\n"
                if tool_calls_data:
                    for tc in tool_calls_data:
                        fn = tc.get("function", {})
                        chat_history += (
                            f"ASSISTANT CALLED TOOL: {fn.get('name')}\n"
                            f"WITH ARGS: {fn.get('arguments')}\n"
                        )
            elif role == "TOOL":
                tool_label = msg.get("name") or msg.get("tool_call_id") or "unknown"
                chat_history += f"TOOL RESULT ({tool_label}): {content}\n"

        latest_objective = user_messages[-1].get("content") if user_messages else user_prompt

        prompt_str = ""
        if chat_history:
            prompt_str += f"CONVERSATION HISTORY:\n{chat_history}\n"

        if provided_tools:
            # PROXY-04: 3-Stage Smart Tool Token Limiter (relevance + diversity + compact JSON)
            compacted_tools_json = format_tools_for_prompt(
                provided_tools,
                objective=str(latest_objective or ""),
                messages=messages,
                max_tools=8,
            )
            prompt_str += f"AVAILABLE TOOLS:\n{compacted_tools_json}\n\n"

        prompt_str += f"CURRENT OBJECTIVE:\n{latest_objective}\n\n"
        prompt_str += (
            "Based on the AVAILABLE TOOLS, select the correct tool(s) to execute the "
            "CURRENT OBJECTIVE. Follow the CRITICAL RULES from the system prompt exactly."
        )

        try:
            _, response_text = await _engine.generate(
                prompt=prompt_str,
                is_code_exec=False,
                is_tool_call=True,
                exact_signature="",
                target_model=model,
                max_tokens=validated.max_tokens or 2048,
                temperature=0.1,
            )
        except RuntimeError as e:
            logger.error(f"[Gateway] Upstream LLM failure in Tool-Calling Mode: {e}")
            return JSONResponse({"error": f"Upstream LLM failure: {str(e)}"}, status_code=502)

        if not provided_tools:
            return _format_completion_response(
                model=model,
                content=response_text,
                stream=is_stream,
                prompt_text=prompt_str,
            )

        normalized_calls, fallback_msg = parse_tool_calls(response_text)
        if not normalized_calls and fallback_msg is None:
            logger.warning(f"[Gateway] Failed to parse tool-call JSON: {response_text[:200]}")

        openai_tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for tc in normalized_calls
        ]

        if openai_tool_calls:
            return _format_completion_response(
                model=model,
                content=None,
                stream=is_stream,
                finish_reason="tool_calls",
                tool_calls=openai_tool_calls,
                prompt_text=prompt_str,
            )

        content_to_return = fallback_msg if fallback_msg is not None else response_text
        return _format_completion_response(
            model=model,
            content=content_to_return,
            stream=is_stream,
            prompt_text=prompt_str,
        )

    # --- AUTONOMOUS AGENT LOOP ---
    async def _autonomous_loop() -> str:
        compact_internal_tools = json.dumps(INTERNAL_TOOLS, separators=(",", ":"))
        chat_history = (
            f"You are the Krusch Autonomous Proxy. Achieve the objective.\n"
            f"CRITICAL DIRECTIVE: Check RETRIEVED HOMELAB CONTEXT (<homelab_context>) first! "
            f"If the answer to the OBJECTIVE is already present in <homelab_context> or after tool execution, "
            f"DO NOT call `run_bash_command` or `read_file` — you MUST immediately call ONLY the `final_answer` tool "
            f"with the complete explanation.\n"
            f"AVAILABLE TOOLS:\n{compact_internal_tools}\n\n"
            f"OBJECTIVE: {user_prompt}\n"
        )
        max_loops = 5
        final_answer = "Error: Agent exceeded max iterations."
        seen_calls: Set[Tuple[str, str]] = set()
        last_tool_result = ""

        for iteration in range(max_loops):
            logger.info(f"  -> Iteration {iteration + 1}...")
            _, response_text = await _engine.generate(
                prompt=chat_history,
                is_code_exec=False,
                is_tool_call=True,
                exact_signature="",
                target_model=model,
                max_tokens=2048,
                temperature=0.1,
            )

            tool_calls, fallback_msg = parse_tool_calls(response_text)

            if tool_calls:
                for tool in tool_calls:
                    tool_name = tool["name"]
                    tool_args = tool["arguments"]

                    if tool_name == "final_answer":
                        return str(tool_args.get("answer", "Objective complete."))

                    call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                    if call_sig in seen_calls:
                        logger.warning(
                            f"[Gateway] Duplicate tool call detected ({tool_name}); breaking loop."
                        )
                        return (
                            f"Error: Duplicate tool call detected for '{tool_name}'. "
                            f"Last result: {last_tool_result}"
                        )
                    seen_calls.add(call_sig)

                    tool_result = await asyncio.to_thread(
                        execute_internal_tool, tool_name, tool_args
                    )
                    last_tool_result = tool_result
                    chat_history += (
                        f"\nASSISTANT CALLED TOOL: {tool_name}\n"
                        f"WITH ARGS: {json.dumps(tool_args)}\n"
                        f"RESULT:\n{tool_result}\n"
                    )
            elif fallback_msg is not None:
                return fallback_msg
            else:
                stripped_resp = response_text.strip() if response_text else ""
                if (
                    stripped_resp
                    and not stripped_resp.startswith(("{", "["))
                    and "final_answer" not in stripped_resp
                ):
                    return response_text
                chat_history += "\nSYSTEM: You must output a valid JSON tool call.\n"

        return final_answer

    try:
        final_answer = await asyncio.wait_for(_autonomous_loop(), timeout=AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        final_answer = f"Error: Autonomous agent timed out after {AGENT_TIMEOUT}s."
    except RuntimeError as e:
        logger.error(f"[Gateway] Upstream LLM failure in autonomous loop: {e}")
        return JSONResponse({"error": f"Upstream LLM failure: {str(e)}"}, status_code=502)

    return _format_completion_response(
        model=model,
        content=final_answer,
        stream=is_stream,
        prompt_text=user_prompt,
    )


@app.get("/api/tags")
async def api_tags():
    """Forward tags request to healthy local GPU-enabled Ollama endpoint."""
    client = get_http_client(30.0)
    try:
        host_part = _get_ollama_host()
        resp = await client.get(f"{host_part}/api/tags")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


@app.get("/v1/models")
async def v1_models():
    """Forward models request to healthy local GPU-enabled Ollama endpoint."""
    client = get_http_client(30.0)
    try:
        host_part = _get_ollama_host()
        resp = await client.get(f"{host_part}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


@app.post("/api/embed")
async def api_embed(request: Request):
    """Forward modern batched /api/embed request to fleet embedding host (krusch)."""
    body = await request.json()
    client = get_http_client(60.0)
    try:
        host_part = _get_ollama_host(for_embeddings=True)
        resp = await client.post(f"{host_part}/api/embed", json=body, timeout=60.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


@app.post("/api/embeddings")
async def api_embeddings(request: Request):
    """Forward embeddings request to fleet embedding host (krusch bge-large)."""
    body = await request.json()
    client = get_http_client(60.0)
    try:
        host_part = _get_ollama_host(for_embeddings=True)
        resp = await client.post(f"{host_part}/api/embeddings", json=body, timeout=60.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    """Forward v1/embeddings request to fleet embedding host (krusch bge-large)."""
    body = await request.json()
    client = get_http_client(60.0)
    try:
        host_part = _get_ollama_host(for_embeddings=True)
        resp = await client.post(f"{host_part}/v1/embeddings", json=body, timeout=60.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


@app.post("/api/generate")
async def api_generate(request: Request):
    """Forward generate request to healthy local GPU-enabled Ollama endpoint."""
    body = await request.json()
    client = get_http_client(120.0)
    try:
        host_part = _get_ollama_host()
        resp = await client.post(f"{host_part}/api/generate", json=body, timeout=120.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to forward to Ollama: {str(e)}"}, status_code=502
        )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5440))
    print(f"[*] Starting Krusch Agentic Universal API Gateway on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")
