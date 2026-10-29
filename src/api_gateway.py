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
from src.budget_assembler import DynamicContextBudgetAssembler
from src.cost_killswitch import CostKillSwitch
from src.orchestrator import AgentTaskSpec, AsyncAgentOrchestrator
from src.pydantic_tool_router import StrictPydanticToolRouter
from src.schemas import ChatCompletionRequest, ProxyConfig
from src.router import WaterfallRouter
from src.shadow_comparator import ShadowTrajectoryComparator
from src.tools import INTERNAL_TOOLS, execute_internal_tool
from src.typed_swarm import TypedMultiAgentSwarm

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
_cost_killswitch = CostKillSwitch(
    hard_limit_usd=float(os.environ.get("KRUSCH_COST_HARD_LIMIT_USD", "0.50")),
    max_loop_depth=int(os.environ.get("KRUSCH_MAX_LOOP_DEPTH", "12")),
    fallback_model=os.environ.get("KRUSCH_FALLBACK_MODEL", "qwen2.5-coder:7b"),
)
_shadow_comparator = ShadowTrajectoryComparator(
    db_path=os.environ.get("KRUSCH_SHADOW_DB", ":memory:"),
    sample_rate=float(os.environ.get("KRUSCH_SHADOW_SAMPLE_RATE", "0.05")),
)
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
    Enforces bounded request payload size (413 Payload Too Large) and optional
    Bearer token verification when KRUSCH_PROXY_API_KEY is set (PROXY-02).
    Exempts /health from auth for liveness probes.
    """
    if request.method in ("POST", "PUT", "PATCH"):
        max_bytes = max(1024, int(os.environ.get("KRUSCH_MAX_REQUEST_BYTES", str(2 * 1024 * 1024))))
        cl_header = request.headers.get("content-length")
        if cl_header is not None:
            try:
                if int(cl_header) > max_bytes:
                    return JSONResponse(
                        {"error": f"Payload Too Large: exceeds {max_bytes} byte limit."},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
        raw_body = await request.body()
        if len(raw_body) > max_bytes:
            return JSONResponse(
                {"error": f"Payload Too Large: exceeds {max_bytes} byte limit."},
                status_code=413,
            )

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

    # --- OPTIONAL SESSION COST KILL-SWITCH & LOCAL FALLBACK (#12) ---
    session_id = (
        request.headers.get("X-Session-Id", "").strip()
        or str(raw_data.get("session_id") or "").strip()
    )
    if session_id:
        ks_decision = _cost_killswitch.evaluate_pre_call(
            session_id,
            model=model,
            estimated_prompt_tokens=_estimate_tokens(json.dumps(messages)),
            max_completion_tokens=validated.max_tokens or 2048,
        )
        if not ks_decision.allowed:
            status_code = 429 if ks_decision.action == "HALTED_MAX_DEPTH" else 402
            return JSONResponse(
                {
                    "error": ks_decision.reason,
                    "killswitch": ks_decision.model_dump(),
                },
                status_code=status_code,
            )
        if ks_decision.action == "FALLBACK_LOCAL":
            logger.warning(
                "[Gateway] CostKillSwitch downgraded session %s: %s -> %s (%s)",
                session_id,
                model,
                ks_decision.effective_model,
                ks_decision.reason,
            )
            model = ks_decision.effective_model

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

    def _record_session_spend(prompt_str_used: str, completion_str_used: Optional[str]) -> None:
        if session_id:
            _cost_killswitch.record_usage(
                session_id,
                model=model,
                prompt_tokens=_estimate_tokens(prompt_str_used),
                completion_tokens=_estimate_tokens(completion_str_used or ""),
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

        _record_session_spend(chat_context, response_text)
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

        _record_session_spend(chat_context, combined)
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

        t_primary0 = time.perf_counter()
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
        primary_latency_ms = max(0.001, (time.perf_counter() - t_primary0) * 1000.0)
        _record_session_spend(prompt_str, response_text)

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
            force_shadow = request.headers.get("X-Krusch-Shadow", "").strip().lower() == "force"
            if force_shadow or os.environ.get("KRUSCH_SHADOW_ENABLED") == "1":
                shadow_model = request.headers.get("X-Krusch-Shadow-Model") or os.environ.get(
                    "KRUSCH_SHADOW_MODEL", "gemma4:e4b"
                )
                req_id = request.headers.get("X-Request-Id") or f"req-{uuid.uuid4().hex[:12]}"

                async def _run_shadow() -> Dict[str, Any]:
                    t0 = time.perf_counter()
                    _, s_text = await _engine.generate(
                        prompt=prompt_str,
                        is_code_exec=False,
                        is_tool_call=True,
                        exact_signature="",
                        target_model=shadow_model,
                        max_tokens=validated.max_tokens or 2048,
                        temperature=0.1,
                    )
                    s_calls, _ = parse_tool_calls(s_text)
                    return {
                        "tool_calls": s_calls,
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "cost_usd": _cost_killswitch.estimate_cost_usd(
                            shadow_model,
                            _estimate_tokens(prompt_str),
                            _estimate_tokens(s_text),
                        ),
                    }

                _shadow_comparator.schedule_background_shadow(
                    request_id=req_id,
                    primary_model=model,
                    shadow_model=shadow_model,
                    primary_calls=openai_tool_calls,
                    primary_latency_ms=primary_latency_ms,
                    primary_cost_usd=_cost_killswitch.estimate_cost_usd(
                        model,
                        _estimate_tokens(prompt_str),
                        _estimate_tokens(response_text),
                    ),
                    shadow_runner=_run_shadow,
                    force=force_shadow,
                )

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
            _record_session_spend(chat_history, response_text)

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


# --- Agentic Production Systems Endpoints (#1, #2, #3, #6, #12, #14) ---


@app.post("/v1/orchestrate/batch")
async def v1_orchestrate_batch(request: Request):
    """Execute 100+ concurrent async agent tasks with non-blocking tools and per-host circuit breakers (#1)."""
    try:
        body = await request.json()
        raw_tasks = body.get("tasks", [])
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return JSONResponse({"error": "'tasks' must be a non-empty list."}, status_code=400)
        max_batch_tasks = max(1, int(os.environ.get("KRUSCH_MAX_BATCH_TASKS", "1000")))
        if len(raw_tasks) > max_batch_tasks:
            return JSONResponse(
                {
                    "error": (
                        f"'tasks' count ({len(raw_tasks)}) exceeds maximum batch limit "
                        f"({max_batch_tasks})."
                    )
                },
                status_code=400,
            )
        max_concurrency = max(1, min(512, int(body.get("max_concurrency", 128))))
        specs = [AgentTaskSpec.model_validate(item) for item in raw_tasks]
    except Exception as exc:
        return JSONResponse({"error": f"Invalid batch payload: {exc}"}, status_code=400)

    orchestrator = AsyncAgentOrchestrator(max_concurrency=max_concurrency)
    report = await orchestrator.run_batch(specs, batch_id=body.get("batch_id"))
    return JSONResponse(report.model_dump())


@app.post("/v1/context/budget")
async def v1_context_budget(request: Request):
    """Pre-flight strict 3-bucket + history token budget allocator (#2)."""
    try:
        body = await request.json()
        assembler = DynamicContextBudgetAssembler(
            max_context_tokens=int(body.get("max_context_tokens", 16384)),
            output_reserve_tokens=int(body.get("output_reserve_tokens", 2048)),
            system_cap_ratio=float(body.get("system_cap_ratio", 0.25)),
            tools_cap_ratio=float(body.get("tools_cap_ratio", 0.20)),
            docs_cap_ratio=float(body.get("docs_cap_ratio", 0.35)),
        )
        package = assembler.assemble(
            system_prompt=str(body.get("system_prompt", "")),
            query=str(body.get("query", "")),
            docs=body.get("docs") or [],
            tools=body.get("tools") or [],
            messages=body.get("messages") or [],
        )
        return JSONResponse(package.model_dump())
    except Exception as exc:
        return JSONResponse({"error": f"Invalid budget request: {exc}"}, status_code=400)


@app.post("/v1/tools/execute-strict")
async def v1_tools_execute_strict(request: Request):
    """Strict Pydantic v2 JSON-Schema tool router with automatic ValidationError trace retry (#3)."""
    _ensure_initialized()
    assert _engine is not None
    try:
        body = await request.json()
        prompt = str(body.get("prompt", "")).strip()
        tool_name = str(body.get("tool_name", "")).strip()
        parameters_schema = body.get("parameters_schema")
        max_retries = max(0, min(5, int(body.get("max_retries", 3))))
        if not prompt or not tool_name or not isinstance(parameters_schema, dict):
            return JSONResponse(
                {"error": "'prompt', 'tool_name', and dict 'parameters_schema' are required."},
                status_code=400,
            )
    except Exception as exc:
        return JSONResponse({"error": f"Invalid tool request: {exc}"}, status_code=400)

    router = StrictPydanticToolRouter(max_retries=max_retries)
    router.register_tool(
        name=tool_name,
        schema=parameters_schema,
        handler=lambda **kwargs: {"executed": tool_name, "validated": kwargs},
        description=str(body.get("description", "")),
    )

    scripted_outputs = body.get("scripted_llm_outputs")
    if isinstance(scripted_outputs, list) and scripted_outputs:
        queue = list(scripted_outputs)

        async def _scripted_llm(_repair_prompt: str) -> str:
            return str(queue.pop(0) if len(queue) > 1 else queue[0])

        llm_fn = _scripted_llm
    else:
        target_model = str(body.get("model") or DEFAULT_MODEL)

        async def _live_llm(repair_prompt: str) -> str:
            _, text = await _engine.generate(
                prompt=repair_prompt,
                is_code_exec=False,
                is_tool_call=True,
                target_model=target_model,
                max_tokens=1024,
                temperature=0.0,
            )
            return text

        llm_fn = _live_llm

    result = await router.execute_with_retry(
        prompt=prompt,
        llm_generate_fn=llm_fn,
        expected_tool=tool_name,
    )
    status_code = 200 if result.ok else 422
    return JSONResponse(result.model_dump(), status_code=status_code)


@app.post("/v1/swarm/run")
async def v1_swarm_run(request: Request):
    """Run strictly typed Researcher -> Coder -> Reviewer multi-agent state machine (#6)."""
    try:
        body = await request.json()
        objective = str(body.get("objective", "")).strip()
        if not objective:
            return JSONResponse({"error": "'objective' must be non-empty."}, status_code=400)
        max_revisions = max(0, min(5, int(body.get("max_revisions", 2))))
    except Exception as exc:
        return JSONResponse({"error": f"Invalid swarm request: {exc}"}, status_code=400)

    swarm = TypedMultiAgentSwarm(max_revisions=max_revisions)
    state = await swarm.run(objective=objective, task_id=body.get("task_id"))
    return JSONResponse(state.model_dump())


@app.post("/v1/cost/evaluate")
async def v1_cost_evaluate(request: Request):
    """Evaluate pre-turn loop depth and $0.50 projected spend kill-switch / local fallback (#12)."""
    try:
        body = await request.json()
        session_id = str(body.get("session_id", "default")).strip() or "default"
        if body.get("reset"):
            _cost_killswitch.reset_session(session_id)
        actual = body.get("record_usage")
        if isinstance(actual, dict):
            _cost_killswitch.record_usage(
                session_id,
                model=str(actual.get("model", "gemini-4-argon")),
                prompt_tokens=int(actual.get("prompt_tokens", 0)),
                completion_tokens=int(actual.get("completion_tokens", 0)),
            )
        decision = _cost_killswitch.evaluate_pre_call(
            session_id,
            model=str(body.get("model", "gemini-4-argon")),
            estimated_prompt_tokens=int(body.get("estimated_prompt_tokens", 1000)),
            max_completion_tokens=int(body.get("max_completion_tokens", 2048)),
        )
        return JSONResponse(
            {
                "decision": decision.model_dump(),
                "ledger": _cost_killswitch.get_session_ledger(session_id),
            },
            status_code=200 if decision.allowed else 402,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Invalid cost evaluation request: {exc}"}, status_code=400)


@app.get("/v1/shadow/comparisons")
async def v1_shadow_comparisons(request: Request):
    """Inspect recent silent 5% shadow trajectory comparisons and aggregate metrics (#14)."""
    if request.query_params.get("flush") == "1":
        await _shadow_comparator.flush_pending()
    limit = max(1, min(200, int(request.query_params.get("limit", "50"))))
    return JSONResponse(
        {
            "ok": True,
            "summary": _shadow_comparator.summary_metrics(),
            "comparisons": _shadow_comparator.list_recent(limit=limit),
        }
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5440))
    print(f"[*] Starting Krusch Agentic Universal API Gateway on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")
