"""
Loop-Depth & Projected-Spend Cost Kill-Switch Middleware (INV-14, #12).

Enforces a hard $0.50 paid-spend circuit breaker and loop-depth ceiling across
agentic sessions, automatically downgrading paid frontier calls to $0.00 local
Ollama models when projected spend crosses the soft/hard budget threshold.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MODEL_RATES_PER_1M: Dict[str, Dict[str, float]] = {
    # Paid frontier models (USD per 1M tokens)
    "gemini-4-argon": {"prompt": 1.80, "completion": 10.80},
    "anthropic/claude-3-7-sonnet": {"prompt": 3.00, "completion": 15.00},
    "openai/gpt-5": {"prompt": 10.00, "completion": 30.00},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "deepseek/deepseek-r1": {"prompt": 0.55, "completion": 2.19},
    # $0.00 Sovereign local Ollama models
    "qwen2.5-coder:7b": {"prompt": 0.0, "completion": 0.0},
    "qwen3-coder-30b-moe:latest": {"prompt": 0.0, "completion": 0.0},
    "gemma4:e4b": {"prompt": 0.0, "completion": 0.0},
    "hermes3:latest": {"prompt": 0.0, "completion": 0.0},
    "gpt-oss:120b": {"prompt": 0.0, "completion": 0.0},
}

_LOCAL_ZERO_COST_PREFIXES: Tuple[str, ...] = (
    "ollama/",
    "local/",
    "qwen2.5-coder",
    "qwen3-coder",
    "gemma4",
    "hermes3",
    "gpt-oss",
    "snowflake-arctic-embed2",
)

_DEFAULT_UNKNOWN_PAID_RATE: Dict[str, float] = {"prompt": 2.50, "completion": 10.00}

KillSwitchAction = Literal[
    "PROCEED",
    "FALLBACK_LOCAL",
    "HALTED_BUDGET_EXCEEDED",
    "HALTED_MAX_DEPTH",
]


class KillSwitchDecision(BaseModel):
    """Structured pre-call authorization and routing verdict."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    action: KillSwitchAction
    effective_model: str
    original_model: str
    loop_depth: int
    spent_usd: float
    projected_turn_usd: float
    projected_total_usd: float
    hard_limit_usd: float
    reason: str


@dataclass
class _SessionSpendState:
    loop_depth: int = 0
    spent_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_recorded: int = 0
    fallback_count: int = 0
    last_action: Optional[KillSwitchAction] = None


class CostKillSwitch:
    """
    Thread-safe per-session loop-depth and USD budget circuit breaker.

    Halts runaway recursion beyond ``max_loop_depth``, halts sessions once actual
    spend reaches ``hard_limit_usd``, and routes paid calls to ``fallback_model``
    ($0.00 local Ollama) before breaching the budget ceiling.
    """

    def __init__(
        self,
        hard_limit_usd: float = 0.50,
        soft_fallback_ratio: float = 0.80,
        max_loop_depth: int = 12,
        fallback_model: str = "qwen2.5-coder:7b",
    ) -> None:
        if hard_limit_usd <= 0.0:
            raise ValueError("hard_limit_usd must be > 0.0.")
        if not (0.0 < soft_fallback_ratio <= 1.0):
            raise ValueError("soft_fallback_ratio must be in (0.0, 1.0].")
        if max_loop_depth < 1:
            raise ValueError("max_loop_depth must be >= 1.")

        self.hard_limit_usd: float = float(hard_limit_usd)
        self.soft_fallback_ratio: float = float(soft_fallback_ratio)
        self.max_loop_depth: int = int(max_loop_depth)
        self.fallback_model: str = str(fallback_model)
        self._lock = threading.RLock()
        self._sessions: Dict[str, _SessionSpendState] = {}

    @staticmethod
    def _resolve_model_rate(model: str) -> Dict[str, float]:
        normalized = str(model or "").strip()
        if normalized in MODEL_RATES_PER_1M:
            return MODEL_RATES_PER_1M[normalized]
        lowered = normalized.lower()
        if lowered in MODEL_RATES_PER_1M:
            return MODEL_RATES_PER_1M[lowered]
        if any(lowered.startswith(prefix) for prefix in _LOCAL_ZERO_COST_PREFIXES):
            return {"prompt": 0.0, "completion": 0.0}
        return _DEFAULT_UNKNOWN_PAID_RATE

    def _is_paid_model(self, model: str) -> bool:
        rate = self._resolve_model_rate(model)
        return (rate.get("prompt", 0.0) > 0.0) or (rate.get("completion", 0.0) > 0.0)

    def estimate_cost_usd(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Compute estimated USD spend for ``(model, prompt_tokens, completion_tokens)``."""
        p_tokens = max(0, int(prompt_tokens))
        c_tokens = max(0, int(completion_tokens))
        rate = self._resolve_model_rate(model)
        raw_usd = (
            p_tokens * float(rate["prompt"]) + c_tokens * float(rate["completion"])
        ) / 1_000_000.0
        return round(raw_usd, 10)

    def _get_or_create_session(self, session_id: str) -> _SessionSpendState:
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionSpendState()
            self._sessions[session_id] = state
        return state

    def evaluate_pre_call(
        self,
        session_id: str,
        *,
        model: str,
        estimated_prompt_tokens: int,
        max_completion_tokens: int = 2048,
    ) -> KillSwitchDecision:
        """
        Evaluate loop depth and projected spend prior to dispatching an LLM call.

        Order of enforcement:
        1. Increment ``loop_depth``; halt if ``loop_depth > max_loop_depth``.
        2. Halt if ``spent_usd >= hard_limit_usd``.
        3. Downgrade paid model to ``fallback_model`` if projected total reaches
           ``hard_limit_usd * soft_fallback_ratio`` or exceeds ``hard_limit_usd``.
        4. Proceed on requested ``model``.
        """
        projected_turn_usd = self.estimate_cost_usd(
            model,
            estimated_prompt_tokens,
            max_completion_tokens,
        )
        soft_ceiling_usd = round(self.hard_limit_usd * self.soft_fallback_ratio, 10)
        has_paid_cost = projected_turn_usd > 0.0 or self._is_paid_model(model)

        with self._lock:
            state = self._get_or_create_session(session_id)
            state.loop_depth += 1
            loop_depth = state.loop_depth
            spent_usd = state.spent_usd
            projected_total_usd = round(spent_usd + projected_turn_usd, 10)

            if loop_depth > self.max_loop_depth:
                state.last_action = "HALTED_MAX_DEPTH"
                return KillSwitchDecision(
                    allowed=False,
                    action="HALTED_MAX_DEPTH",
                    effective_model=model,
                    original_model=model,
                    loop_depth=loop_depth,
                    spent_usd=spent_usd,
                    projected_turn_usd=projected_turn_usd,
                    projected_total_usd=projected_total_usd,
                    hard_limit_usd=self.hard_limit_usd,
                    reason=(
                        f"Session '{session_id}' exceeded max_loop_depth "
                        f"({loop_depth} > {self.max_loop_depth})."
                    ),
                )

            if spent_usd >= self.hard_limit_usd:
                state.last_action = "HALTED_BUDGET_EXCEEDED"
                return KillSwitchDecision(
                    allowed=False,
                    action="HALTED_BUDGET_EXCEEDED",
                    effective_model=model,
                    original_model=model,
                    loop_depth=loop_depth,
                    spent_usd=spent_usd,
                    projected_turn_usd=projected_turn_usd,
                    projected_total_usd=projected_total_usd,
                    hard_limit_usd=self.hard_limit_usd,
                    reason=(
                        f"Session '{session_id}' reached hard budget ceiling "
                        f"(${spent_usd:.4f} >= ${self.hard_limit_usd:.2f})."
                    ),
                )

            if has_paid_cost and (
                projected_total_usd > self.hard_limit_usd
                or projected_total_usd >= soft_ceiling_usd
            ):
                state.fallback_count += 1
                state.last_action = "FALLBACK_LOCAL"
                return KillSwitchDecision(
                    allowed=True,
                    action="FALLBACK_LOCAL",
                    effective_model=self.fallback_model,
                    original_model=model,
                    loop_depth=loop_depth,
                    spent_usd=spent_usd,
                    projected_turn_usd=projected_turn_usd,
                    projected_total_usd=projected_total_usd,
                    hard_limit_usd=self.hard_limit_usd,
                    reason=(
                        f"Projected spend ${projected_total_usd:.4f} breaches threshold "
                        f"(${soft_ceiling_usd:.2f} soft / ${self.hard_limit_usd:.2f} hard); "
                        f"downgrading '{model}' -> '{self.fallback_model}' ($0.00 local)."
                    ),
                )

            state.last_action = "PROCEED"
            return KillSwitchDecision(
                allowed=True,
                action="PROCEED",
                effective_model=model,
                original_model=model,
                loop_depth=loop_depth,
                spent_usd=spent_usd,
                projected_turn_usd=projected_turn_usd,
                projected_total_usd=projected_total_usd,
                hard_limit_usd=self.hard_limit_usd,
                reason=f"Within budget (${projected_total_usd:.4f} / ${self.hard_limit_usd:.2f}).",
            )

    def record_usage(
        self,
        session_id: str,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Record actual token consumption for ``session_id`` and return the incurred turn USD."""
        p_tokens = max(0, int(prompt_tokens))
        c_tokens = max(0, int(completion_tokens))
        incurred_usd = self.estimate_cost_usd(model, p_tokens, c_tokens)

        with self._lock:
            state = self._get_or_create_session(session_id)
            state.spent_usd = round(state.spent_usd + incurred_usd, 10)
            state.prompt_tokens += p_tokens
            state.completion_tokens += c_tokens
            state.calls_recorded += 1
            return incurred_usd

    def get_session_ledger(self, session_id: str) -> Dict[str, Any]:
        """Return a snapshot of cumulative spend, token totals, and depth for ``session_id``."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return {
                    "session_id": session_id,
                    "spent_usd": 0.0,
                    "remaining_usd": self.hard_limit_usd,
                    "hard_limit_usd": self.hard_limit_usd,
                    "loop_depth": 0,
                    "max_loop_depth": self.max_loop_depth,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_recorded": 0,
                    "fallback_count": 0,
                    "last_action": None,
                }
            remaining = max(0.0, round(self.hard_limit_usd - state.spent_usd, 10))
            return {
                "session_id": session_id,
                "spent_usd": state.spent_usd,
                "remaining_usd": remaining,
                "hard_limit_usd": self.hard_limit_usd,
                "loop_depth": state.loop_depth,
                "max_loop_depth": self.max_loop_depth,
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
                "calls_recorded": state.calls_recorded,
                "fallback_count": state.fallback_count,
                "last_action": state.last_action,
            }

    def reset_session(self, session_id: str) -> None:
        """Clear accumulated spend and loop depth for ``session_id``."""
        with self._lock:
            self._sessions.pop(session_id, None)


__all__ = (
    "CostKillSwitch",
    "KillSwitchAction",
    "KillSwitchDecision",
    "MODEL_RATES_PER_1M",
)
