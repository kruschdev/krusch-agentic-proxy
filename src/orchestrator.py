"""
Krusch Agentic Proxy — Production Async Agent Orchestrator (INV-10).

Executes 100+ concurrent agent tasks on a single asyncio event loop with:
  - Bounded concurrency via asyncio.Semaphore(max_concurrency)
  - Non-blocking tool execution (direct await for coroutines, asyncio.to_thread for sync callables)
  - Per-attempt asyncio.wait_for timeouts and exponential-jitter transient retry
  - Per-host circuit breakers (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
  - Strict fault isolation so one failing task never aborts sibling tasks in run_batch
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_ORCHESTRATOR_MODEL: str = os.getenv("KRUSCH_LLM_MODEL", "qwen2.5-coder:7b")
DEFAULT_LOCAL_API_URL: str = os.getenv(
    "KRUSCH_LLM_API_URL", "http://127.0.0.1:11434/v1/chat/completions"
)

__all__ = (
    "AgentTaskOutcome",
    "AgentTaskSpec",
    "AsyncAgentOrchestrator",
    "BatchOrchestrationReport",
    "CircuitState",
    "HostCircuitBreaker",
)


class CircuitState(str, Enum):
    """Tri-state circuit breaker lifecycle for upstream model/tool hosts."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class _HostCircuitRecord:
    """Mutable per-host state protected by HostCircuitBreaker's lock."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    half_open_probe_active: bool = False


class HostCircuitBreaker:
    """
    Per-host consecutive failure tracker implementing:
      CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 15.0,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout_s = float(recovery_timeout_s)
        self._clock_fn = clock_fn
        self._records: dict[str, _HostCircuitRecord] = {}
        self._lock = threading.RLock()

    def _get_or_create(self, host_key: str) -> _HostCircuitRecord:
        key = host_key or "local"
        rec = self._records.get(key)
        if rec is None:
            rec = _HostCircuitRecord()
            self._records[key] = rec
        return rec

    def _refresh_state_locked(self, rec: _HostCircuitRecord) -> CircuitState:
        if rec.state == CircuitState.OPEN:
            elapsed = self._clock_fn() - rec.opened_at
            if elapsed >= self.recovery_timeout_s:
                rec.state = CircuitState.HALF_OPEN
                rec.half_open_probe_active = False
        return rec.state

    def get_state(self, host_key: str = "local") -> CircuitState:
        """Return current CircuitState for host_key, advancing OPEN -> HALF_OPEN after cooldown."""
        with self._lock:
            rec = self._get_or_create(host_key)
            return self._refresh_state_locked(rec)

    def consecutive_failures(self, host_key: str = "local") -> int:
        """Return current consecutive failure count for host_key."""
        with self._lock:
            return self._get_or_create(host_key).consecutive_failures

    def allow_request(self, host_key: str = "local") -> bool:
        """
        Check whether a request to host_key may proceed.
        Allows traffic when CLOSED, blocks when OPEN, and admits a recovery probe when HALF_OPEN.
        """
        with self._lock:
            rec = self._get_or_create(host_key)
            state = self._refresh_state_locked(rec)
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.OPEN:
                return False
            # HALF_OPEN: admit single probe at a time to test upstream recovery
            if not rec.half_open_probe_active:
                rec.half_open_probe_active = True
                return True
            return False

    can_execute = allow_request

    def record_success(self, host_key: str = "local") -> CircuitState:
        """Record a successful call and transition HALF_OPEN/CLOSED -> CLOSED."""
        with self._lock:
            rec = self._get_or_create(host_key)
            rec.consecutive_failures = 0
            rec.opened_at = 0.0
            rec.half_open_probe_active = False
            rec.state = CircuitState.CLOSED
            return rec.state

    def record_failure(self, host_key: str = "local") -> CircuitState:
        """
        Record a failed call.
        Trips CLOSED -> OPEN at failure_threshold, or immediately re-opens from HALF_OPEN.
        """
        with self._lock:
            rec = self._get_or_create(host_key)
            self._refresh_state_locked(rec)
            rec.consecutive_failures += 1
            if (
                rec.state == CircuitState.HALF_OPEN
                or rec.consecutive_failures >= self.failure_threshold
            ):
                rec.state = CircuitState.OPEN
                rec.opened_at = self._clock_fn()
                rec.half_open_probe_active = False
            return rec.state

    def reset(self, host_key: str | None = None) -> None:
        """Reset circuit state for a specific host_key, or all hosts when host_key is None."""
        with self._lock:
            if host_key is None:
                self._records.clear()
            else:
                self._records.pop(host_key or "local", None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a serializable diagnostic snapshot of all tracked host circuits."""
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for key, rec in self._records.items():
                state = self._refresh_state_locked(rec)
                out[key] = {
                    "state": state.value,
                    "consecutive_failures": rec.consecutive_failures,
                    "opened_at": rec.opened_at,
                }
            return out


class AgentTaskSpec(BaseModel):
    """Specification for a single asynchronous agent execution task."""

    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(..., min_length=1, description="Unique task identifier")
    prompt: str = Field(..., description="Task prompt or user objective")
    model: str = Field(default=DEFAULT_ORCHESTRATOR_MODEL, description="Target model name")
    tools: list[dict[str, Any]] = Field(default_factory=list, description="Optional tool schemas")
    timeout_s: float = Field(default=30.0, gt=0, description="Per-attempt timeout in seconds")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max transient retries")
    host_key: str = Field(default="local", min_length=1, description="Circuit breaker host bucket")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary task metadata")


class AgentTaskOutcome(BaseModel):
    """Normalized execution result for a single AgentTaskSpec."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    ok: bool
    status: Literal["SUCCEEDED", "FAILED", "TIMED_OUT", "CIRCUIT_OPEN"]
    output: Any | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    attempts: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    error: str | None = None


class BatchOrchestrationReport(BaseModel):
    """Aggregated telemetry and ordered per-task outcomes for a batch execution."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    total_tasks: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    max_concurrency: int = Field(ge=1)
    wall_clock_ms: float = Field(ge=0.0)
    throughput_tasks_per_sec: float = Field(ge=0.0)
    results: list[AgentTaskOutcome] = Field(default_factory=list)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    """Return True if fn or its __call__ method is a native coroutine function."""
    if inspect.iscoroutinefunction(fn):
        return True
    call_attr = getattr(fn, "__call__", None)  # noqa: B004
    return bool(call_attr and inspect.iscoroutinefunction(call_attr))


def _normalize_tool_call_item(raw: Any) -> tuple[str, dict[str, Any]] | None:
    """Normalize a tool call dict into (tool_name, arguments_dict)."""
    if not isinstance(raw, dict):
        return None
    target = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = target.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    raw_args = target.get("arguments", target.get("parameters", {}))
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            args = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            args = {"raw": raw_args}
    elif isinstance(raw_args, dict):
        args = dict(raw_args)
    else:
        args = {}
    return name.strip(), args


class AsyncAgentOrchestrator:
    """
    Production-grade asyncio orchestrator capable of executing 100+ concurrent agent tasks
    with semaphore backpressure, non-blocking sync/async tool fan-out, exponential-jitter
    retries, and per-host circuit breakers.
    """

    def __init__(
        self,
        max_concurrency: int = 128,
        llm_fn: Callable[..., Any] | None = None,
        tool_fn: Callable[..., Any] | None = None,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 15.0,
        base_backoff_s: float = 0.02,
        *,
        use_live_engine: bool = False,
        circuit_breaker: HostCircuitBreaker | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if base_backoff_s < 0:
            raise ValueError("base_backoff_s must be >= 0")

        self.max_concurrency = int(max_concurrency)
        self.llm_fn = llm_fn
        self.tool_fn = tool_fn
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout_s = float(recovery_timeout_s)
        self.base_backoff_s = float(base_backoff_s)
        self.use_live_engine = bool(use_live_engine)
        self.circuit_breaker = circuit_breaker or HostCircuitBreaker(
            failure_threshold=self.failure_threshold,
            recovery_timeout_s=self.recovery_timeout_s,
        )

        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._live_engine: Any | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return an asyncio.Semaphore bound safely to the active running event loop."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Expose current concurrency semaphore."""
        if self._semaphore is not None:
            try:
                loop = asyncio.get_running_loop()
                if self._semaphore_loop is loop:
                    return self._semaphore
            except RuntimeError:
                return self._semaphore
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        try:
            self._semaphore_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._semaphore_loop = None
        return self._semaphore

    async def _default_llm_fn(self, spec: AgentTaskSpec) -> str | dict[str, Any]:
        """
        Deterministic fast async echo/planner for offline testing, or live KruschEngine bridge
        when use_live_engine is enabled.
        """
        use_live = self.use_live_engine or bool(spec.metadata.get("use_live_engine"))
        if use_live:
            if self._live_engine is None:
                from src.config import load_config
                from src.core import KruschEngine

                self._live_engine = KruschEngine(load_config())
            _, response = await self._live_engine.generate(
                spec.prompt,
                is_tool_call=bool(spec.tools),
                target_model=spec.model,
            )
            if spec.tools:
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, list):
                        return {"content": None, "tool_calls": parsed}
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        return parsed
                    if isinstance(parsed, dict) and "name" in parsed:
                        return {"content": None, "tool_calls": [parsed]}
                except json.JSONDecodeError:
                    pass
            return response

        # Yield once to the event loop so 100+ tasks genuinely interleave
        await asyncio.sleep(0)

        if spec.tools:
            planned_calls: list[dict[str, Any]] = []
            override_args = spec.metadata.get("tool_arguments", {})
            for idx, raw_tool in enumerate(spec.tools):
                fn_block = (
                    raw_tool.get("function")
                    if isinstance(raw_tool, dict) and isinstance(raw_tool.get("function"), dict)
                    else raw_tool
                )
                if not isinstance(fn_block, dict):
                    continue
                t_name = str(fn_block.get("name") or f"tool_{idx}").strip()
                args = (
                    dict(override_args)
                    if isinstance(override_args, dict)
                    else {}
                )
                planned_calls.append({"name": t_name, "arguments": args})
                if not spec.metadata.get("plan_all_tools"):
                    break
            return {
                "content": f"[offline-planner:{spec.model}] {spec.prompt}",
                "tool_calls": planned_calls,
            }

        return f"[offline-echo:{spec.model}] {spec.prompt}"

    @staticmethod
    def _default_tool_fn(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback tool executor when tool_fn is omitted."""
        return {"ok": True, "tool": name, "arguments": arguments}

    async def _invoke_llm(self, spec: AgentTaskSpec) -> Any:
        """Invoke custom or default llm_fn without blocking the event loop."""
        if self.llm_fn is None:
            return await self._default_llm_fn(spec)

        fn = self.llm_fn
        call_args: tuple[Any, ...] = (spec,)
        call_kwargs: dict[str, Any] = {}

        try:
            sig = inspect.signature(fn)
            pos_params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            kw_only = [p for p in sig.parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY]
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            if len(pos_params) == 0 and not kw_only and not has_var_pos and not has_var_kw:
                call_args = ()
            elif len(pos_params) == 1 and not kw_only and not has_var_kw:
                p0 = pos_params[0]
                if p0.annotation in (str, "str") or p0.name in ("prompt", "text", "query"):
                    call_args = (spec.prompt,)
                else:
                    call_args = (spec,)
            elif len(pos_params) >= 2 or kw_only:
                first_is_spec = bool(pos_params) and (
                    pos_params[0].annotation in (AgentTaskSpec, "AgentTaskSpec")
                    or pos_params[0].name in ("spec", "task", "task_spec")
                )
                available = {
                    "spec": spec,
                    "task": spec,
                    "prompt": spec.prompt,
                    "model": spec.model,
                    "tools": spec.tools,
                    "task_id": spec.task_id,
                    "host_key": spec.host_key,
                    "metadata": spec.metadata,
                }
                if first_is_spec:
                    call_args = (spec,)
                    for p in pos_params[1:] + kw_only:
                        if p.name in available:
                            call_kwargs[p.name] = available[p.name]
                else:
                    built_pos: list[Any] = []
                    for p in pos_params:
                        if p.name in available:
                            built_pos.append(available[p.name])
                        elif p.default is inspect.Parameter.empty:
                            built_pos.append(spec)
                    call_args = tuple(built_pos)
                    for p in kw_only:
                        if p.name in available:
                            call_kwargs[p.name] = available[p.name]
        except (TypeError, ValueError):
            call_args = (spec,)
            call_kwargs = {}

        if _is_async_callable(fn):
            return await fn(*call_args, **call_kwargs)

        res = await asyncio.to_thread(fn, *call_args, **call_kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    async def _invoke_single_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call non-blockingly (await coroutine or offload sync to_thread)."""
        fn = self.tool_fn or self._default_tool_fn
        call_args: tuple[Any, ...] = (tool_name, arguments)
        call_kwargs: dict[str, Any] = {}

        try:
            sig = inspect.signature(fn)
            pos_params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            if len(pos_params) == 1 and not has_var_pos and not has_var_kw:
                call_args = ({"name": tool_name, "arguments": arguments},)
            elif len(pos_params) == 1 and not has_var_pos and has_var_kw:
                call_args = (tool_name,)
                call_kwargs = dict(arguments)
        except (TypeError, ValueError):
            call_args = (tool_name, arguments)

        if _is_async_callable(fn):
            result = await fn(*call_args, **call_kwargs)
        else:
            result = await asyncio.to_thread(fn, *call_args, **call_kwargs)
            if inspect.isawaitable(result):
                result = await result

        return {
            "name": tool_name,
            "arguments": arguments,
            "ok": True,
            "result": result,
        }

    async def _execute_attempt(self, spec: AgentTaskSpec) -> tuple[Any, list[dict[str, Any]]]:
        """Run one LLM generation + concurrent tool fan-out pass."""
        llm_resp = await self._invoke_llm(spec)

        tool_calls_raw: list[Any] = []
        output_val: Any = llm_resp

        if isinstance(llm_resp, dict):
            if isinstance(llm_resp.get("tool_calls"), list):
                tool_calls_raw = llm_resp["tool_calls"]
                output_val = llm_resp.get("content", llm_resp)
            elif "name" in llm_resp and ("arguments" in llm_resp or "parameters" in llm_resp):
                tool_calls_raw = [llm_resp]
                output_val = llm_resp.get("content")
        elif isinstance(llm_resp, str) and spec.tools:
            stripped = llm_resp.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
                        tool_calls_raw = parsed["tool_calls"]
                        output_val = parsed.get("content", llm_resp)
                except json.JSONDecodeError:
                    pass

        normalized_calls: list[tuple[str, dict[str, Any]]] = []
        for raw_tc in tool_calls_raw:
            norm = _normalize_tool_call_item(raw_tc)
            if norm is not None:
                normalized_calls.append(norm)

        tool_results: list[dict[str, Any]] = []
        if normalized_calls:
            gathered = await asyncio.gather(
                *(self._invoke_single_tool(name, args) for name, args in normalized_calls)
            )
            tool_results = list(gathered)

        return output_val, tool_results

    async def run_task(self, spec: AgentTaskSpec | dict[str, Any]) -> AgentTaskOutcome:
        """
        Execute a single AgentTaskSpec under semaphore concurrency control, timeout protection,
        transient exponential-jitter retry, and host circuit-breaker gating.
        """
        t_start = time.perf_counter()
        if not isinstance(spec, AgentTaskSpec):
            try:
                spec = AgentTaskSpec.model_validate(spec)
            except ValidationError as val_exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                fallback_id = (
                    str(spec.get("task_id", "invalid_task"))
                    if isinstance(spec, dict)
                    else "invalid_task"
                )
                return AgentTaskOutcome(
                    task_id=fallback_id or "invalid_task",
                    ok=False,
                    status="FAILED",
                    output=None,
                    tool_results=[],
                    attempts=0,
                    latency_ms=round(elapsed_ms, 4),
                    error=f"Invalid AgentTaskSpec: {val_exc}",
                )

        sem = self._get_semaphore()
        async with sem:
            if not self.circuit_breaker.allow_request(spec.host_key):
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return AgentTaskOutcome(
                    task_id=spec.task_id,
                    ok=False,
                    status="CIRCUIT_OPEN",
                    output=None,
                    tool_results=[],
                    attempts=0,
                    latency_ms=round(elapsed_ms, 4),
                    error=f"Circuit breaker OPEN for host '{spec.host_key}'",
                )

            max_attempts = 1 + spec.max_retries
            last_status: Literal["FAILED", "TIMED_OUT", "CIRCUIT_OPEN"] = "FAILED"
            last_error: str = "Unknown execution error"
            attempts_used = 0

            for attempt in range(1, max_attempts + 1):
                if attempt > 1 and not self.circuit_breaker.allow_request(spec.host_key):
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    return AgentTaskOutcome(
                        task_id=spec.task_id,
                        ok=False,
                        status="CIRCUIT_OPEN",
                        output=None,
                        tool_results=[],
                        attempts=attempts_used,
                        latency_ms=round(elapsed_ms, 4),
                        error=f"Circuit breaker tripped OPEN for host '{spec.host_key}'",
                    )

                attempts_used = attempt
                try:
                    output_val, tool_results = await asyncio.wait_for(
                        self._execute_attempt(spec),
                        timeout=spec.timeout_s,
                    )
                    self.circuit_breaker.record_success(spec.host_key)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    return AgentTaskOutcome(
                        task_id=spec.task_id,
                        ok=True,
                        status="SUCCEEDED",
                        output=output_val,
                        tool_results=tool_results,
                        attempts=attempts_used,
                        latency_ms=round(elapsed_ms, 4),
                        error=None,
                    )
                except asyncio.TimeoutError:
                    last_status = "TIMED_OUT"
                    last_error = f"Task '{spec.task_id}' timed out after {spec.timeout_s}s"
                    new_state = self.circuit_breaker.record_failure(spec.host_key)
                    logger.debug(
                        "Task %s attempt %d/%d timed out (host=%s, circuit=%s)",
                        spec.task_id,
                        attempt,
                        max_attempts,
                        spec.host_key,
                        new_state.value,
                    )
                    if attempt < max_attempts and new_state != CircuitState.OPEN:
                        backoff = (
                            self.base_backoff_s
                            * (2 ** (attempt - 1))
                            * random.uniform(0.5, 1.5)
                        )
                        if backoff > 0:
                            await asyncio.sleep(backoff)
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 — fault-isolation boundary per task
                    last_status = "FAILED"
                    last_error = f"{type(exc).__name__}: {exc}"
                    new_state = self.circuit_breaker.record_failure(spec.host_key)
                    logger.debug(
                        "Task %s attempt %d/%d failed with %s (host=%s, circuit=%s)",
                        spec.task_id,
                        attempt,
                        max_attempts,
                        last_error,
                        spec.host_key,
                        new_state.value,
                    )
                    if attempt < max_attempts and new_state != CircuitState.OPEN:
                        backoff = (
                            self.base_backoff_s
                            * (2 ** (attempt - 1))
                            * random.uniform(0.5, 1.5)
                        )
                        if backoff > 0:
                            await asyncio.sleep(backoff)
                        continue
                    break

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return AgentTaskOutcome(
                task_id=spec.task_id,
                ok=False,
                status=last_status,
                output=None,
                tool_results=[],
                attempts=attempts_used,
                latency_ms=round(elapsed_ms, 4),
                error=last_error,
            )

    async def run_batch(
        self,
        tasks: Sequence[AgentTaskSpec | dict[str, Any]],
        *,
        batch_id: str | None = None,
    ) -> BatchOrchestrationReport:
        """
        Execute a batch of AgentTaskSpec items concurrently on a single event loop.
        Guarantees fault isolation: an exception in any task never aborts sibling tasks.
        """
        resolved_batch_id = batch_id or f"batch_{uuid.uuid4().hex[:12]}"
        t_batch_start = time.perf_counter()

        if not tasks:
            return BatchOrchestrationReport(
                batch_id=resolved_batch_id,
                total_tasks=0,
                succeeded=0,
                failed=0,
                max_concurrency=self.max_concurrency,
                wall_clock_ms=0.0,
                throughput_tasks_per_sec=0.0,
                results=[],
            )

        raw_outcomes = await asyncio.gather(
            *(self.run_task(task) for task in tasks),
            return_exceptions=True,
        )

        wall_clock_ms = max(0.001, (time.perf_counter() - t_batch_start) * 1000.0)
        outcomes: list[AgentTaskOutcome] = []
        for idx, item in enumerate(raw_outcomes):
            if isinstance(item, AgentTaskOutcome):
                outcomes.append(item)
            elif isinstance(item, BaseException):
                if isinstance(item, (KeyboardInterrupt, SystemExit)):
                    raise item
                src = tasks[idx]
                fallback_id = (
                    src.task_id
                    if isinstance(src, AgentTaskSpec)
                    else str(src.get("task_id", f"task_{idx}"))
                    if isinstance(src, dict)
                    else f"task_{idx}"
                )
                outcomes.append(
                    AgentTaskOutcome(
                        task_id=fallback_id,
                        ok=False,
                        status="FAILED",
                        output=None,
                        tool_results=[],
                        attempts=1,
                        latency_ms=round(wall_clock_ms, 4),
                        error=f"{type(item).__name__}: {item}",
                    )
                )

        total_tasks = len(outcomes)
        succeeded = sum(1 for r in outcomes if r.ok)
        failed = total_tasks - succeeded
        throughput = round(total_tasks / (wall_clock_ms / 1000.0), 4)

        return BatchOrchestrationReport(
            batch_id=resolved_batch_id,
            total_tasks=total_tasks,
            succeeded=succeeded,
            failed=failed,
            max_concurrency=self.max_concurrency,
            wall_clock_ms=round(wall_clock_ms, 4),
            throughput_tasks_per_sec=throughput,
            results=outcomes,
        )
