"""
Krusch Agentic Proxy — Dynamic Context Budget Assembler (INV-11 / INV-BUDGET-01).

Strictly allocates `max_context_tokens - output_reserve_tokens` across:
  1. System prompt (head + tail truncation only when exceeding system_cap)
  2. Tool schemas (pinned first, then query lexical overlap / explicit relevance, compact JSON)
  3. Retrieved documents (pinned first, then descending salience score)
  4. Conversation history (newest-to-oldest, always preserving latest user objective,
     restored in chronological order)

Guarantees `ledger.total_allocated_tokens <= ledger.input_ceiling_tokens` (INV-BUDGET-01).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "to",
    "in",
    "on",
    "at",
    "of",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "with",
    "by",
    "from",
    "as",
    "it",
    "this",
    "that",
    "please",
    "can",
    "you",
    "me",
    "my",
}

__all__ = (
    "AssembledContextPackage",
    "BudgetDocItem",
    "BudgetToolItem",
    "ContextBudgetLedger",
    "DynamicContextBudgetAssembler",
    "estimate_tokens",
)


def estimate_tokens(text: str | None) -> int:
    """
    Deterministic fast token estimator.
    Returns 0 for None or empty strings, and `max(1, (len(text) + 3) // 4)` otherwise.
    Monotonic under character slicing: any string of length <= 4 * K has <= K tokens.
    """
    if not text:
        return 0
    s = str(text)
    if not s:
        return 0
    return max(1, (len(s) + 3) // 4)


def _truncate_head_tail(
    text: str,
    max_tokens: int,
    marker: str = "\n...[TRUNCATED]...\n",
) -> tuple[str, bool]:
    """
    Truncate `text` to strictly `<= max_tokens` while preserving head and tail context.
    Returns `(result_text, was_truncated)`.
    """
    if not text:
        return "", False
    if max_tokens <= 0:
        return "", True
    if estimate_tokens(text) <= max_tokens:
        return text, False

    max_chars = max_tokens * 4
    if max_chars <= len(marker) + 4:
        return text[:max_chars], True

    avail = max_chars - len(marker)
    head_chars = (avail + 1) // 2
    tail_chars = avail - head_chars
    truncated = (
        f"{text[:head_chars]}{marker}{text[-tail_chars:] if tail_chars > 0 else ''}"
    )
    return truncated[:max_chars], True


def _tokenize_lexical(text: str) -> set[str]:
    """Extract lowercase alphanumeric intent tokens (splitting camelCase) without stopwords."""
    if not text:
        return set()
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return {tok for tok in _WORD_RE.findall(expanded.lower()) if tok not in _STOPWORDS}


def _lexical_overlap_score(query: str, target_text: str) -> float:
    """Deterministic lexical overlap score in [0.0, 1.0] between query and tool metadata."""
    q_tokens = _tokenize_lexical(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize_lexical(target_text)
    if not t_tokens:
        return 0.0
    hits = len(q_tokens & t_tokens)
    # Include substring stem bonus for identifiers like `search_docs` vs `search`
    if hits < len(q_tokens):
        t_lower = target_text.lower()
        partial_hits = sum(1 for qt in (q_tokens - t_tokens) if len(qt) >= 3 and qt in t_lower)
        return min(1.0, (hits + 0.5 * partial_hits) / len(q_tokens))
    return min(1.0, hits / len(q_tokens))


def _compact_schema_dict(schema_def: dict[str, Any], max_desc_chars: int = 160) -> dict[str, Any]:
    """Strip redundant JSON Schema metadata and bound overly long descriptions."""
    if not isinstance(schema_def, dict):
        return {}
    is_wrapped = "function" in schema_def and isinstance(schema_def.get("function"), dict)
    fn = dict(schema_def["function"]) if is_wrapped else dict(schema_def)

    for noisy_key in ("$schema", "title"):
        fn.pop(noisy_key, None)

    desc = fn.get("description")
    if isinstance(desc, str) and len(desc) > max_desc_chars:
        fn["description"] = desc[: max_desc_chars - 3].rstrip() + "..."

    params = fn.get("parameters")
    if isinstance(params, dict):
        clean_params: dict[str, Any] = {
            k: v for k, v in params.items() if k not in ("$schema", "title")
        }
        props = clean_params.get("properties")
        if isinstance(props, dict):
            clean_props: dict[str, Any] = {}
            for p_name, p_val in props.items():
                if isinstance(p_val, dict):
                    p_copy = {
                        k: v for k, v in p_val.items() if k not in ("$schema", "title")
                    }
                    p_desc = p_copy.get("description")
                    if isinstance(p_desc, str) and len(p_desc) > max_desc_chars:
                        p_copy["description"] = p_desc[: max_desc_chars - 3].rstrip() + "..."
                    clean_props[p_name] = p_copy
                else:
                    clean_props[p_name] = p_val
            clean_params["properties"] = clean_props
        fn["parameters"] = clean_params

    if is_wrapped:
        return {"type": schema_def.get("type", "function"), "function": fn}
    return fn


def _serialize_message_for_budget(msg: dict[str, Any]) -> str:
    """Deterministic canonical string used to measure a conversation message's token cost."""
    role = str(msg.get("role") or "user")
    content = msg.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        content_str = "\n".join(p for p in parts if p)
    elif content is None:
        content_str = ""
    else:
        content_str = str(content)

    extras = ""
    if msg.get("tool_calls"):
        extras = " " + json.dumps(msg["tool_calls"], separators=(",", ":"), sort_keys=True)
    return f"{role}: {content_str}{extras}"


def _truncate_message_to_budget(msg: dict[str, Any], max_tokens: int) -> tuple[dict[str, Any], int, bool]:
    """
    Return `(copied_msg, msg_tokens, was_truncated)` guaranteed to fit within `max_tokens`.
    """
    msg_copy = dict(msg)
    current_serialized = _serialize_message_for_budget(msg_copy)
    current_toks = estimate_tokens(current_serialized)
    if current_toks <= max_tokens:
        return msg_copy, current_toks, False

    if max_tokens <= 0:
        msg_copy["content"] = ""
        msg_copy.pop("tool_calls", None)
        return msg_copy, 0, True

    role = str(msg_copy.get("role") or "user")
    # Remove bulky tool_calls first if necessary to fit tight budget
    if "tool_calls" in msg_copy:
        without_tc = dict(msg_copy)
        without_tc.pop("tool_calls", None)
        if estimate_tokens(_serialize_message_for_budget(without_tc)) <= max_tokens:
            serialized = _serialize_message_for_budget(without_tc)
            return without_tc, estimate_tokens(serialized), True
        msg_copy = without_tc

    raw_content = str(msg_copy.get("content") or "")
    prefix_len = len(f"{role}: ")
    max_total_chars = max_tokens * 4
    avail_content_chars = max(0, max_total_chars - prefix_len)

    if avail_content_chars == 0:
        # Even `role: ` touches the ceiling; trim role + empty content so estimate <= max_tokens
        trimmed_role = role[: max(1, max_total_chars - 2)]
        msg_copy["role"] = trimmed_role
        msg_copy["content"] = ""
    else:
        marker = " ...[TRUNCATED]... "
        if avail_content_chars <= len(marker) + 4:
            msg_copy["content"] = raw_content[:avail_content_chars]
        else:
            body_avail = avail_content_chars - len(marker)
            head_c = (body_avail + 1) // 2
            tail_c = body_avail - head_c
            msg_copy["content"] = (
                f"{raw_content[:head_c]}{marker}{raw_content[-tail_c:] if tail_c > 0 else ''}"
            )[:avail_content_chars]

    final_toks = min(max_tokens, estimate_tokens(_serialize_message_for_budget(msg_copy)))
    return msg_copy, final_toks, True


class BudgetDocItem(BaseModel):
    """Candidate retrieved document with salience score and pin flag."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    content: str
    score: float = 0.5
    pinned: bool = False


class BudgetToolItem(BaseModel):
    """Candidate tool schema definition with optional explicit relevance and pin flag."""

    model_config = ConfigDict(extra="ignore")

    name: str
    schema_def: dict[str, Any]
    pinned: bool = False
    relevance: float | None = None


class ContextBudgetLedger(BaseModel):
    """Exact pre-flight token accounting ledger (INV-BUDGET-01)."""

    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int
    output_reserve_tokens: int
    input_ceiling_tokens: int
    system_tokens: int
    tools_tokens: int
    docs_tokens: int
    history_tokens: int
    total_allocated_tokens: int
    remaining_tokens: int
    utilization_ratio: float
    evicted_doc_ids: list[str] = Field(default_factory=list)
    evicted_tool_names: list[str] = Field(default_factory=list)
    dropped_history_turns: int = 0
    overflow_prevented: bool = False


class AssembledContextPackage(BaseModel):
    """Final budget-verified context payload ready for LLM dispatch."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    selected_tools: list[dict[str, Any]]
    selected_docs: list[BudgetDocItem]
    selected_messages: list[dict[str, Any]]
    ledger: ContextBudgetLedger


class DynamicContextBudgetAssembler:
    """
    Deterministic pre-flight token budget allocator across system prompts, tool schemas,
    retrieved RAG documents, and multi-turn conversation history.
    """

    def __init__(
        self,
        max_context_tokens: int = 16384,
        output_reserve_tokens: int = 2048,
        system_cap_ratio: float = 0.25,
        tools_cap_ratio: float = 0.20,
        docs_cap_ratio: float = 0.35,
    ) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        if output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must be >= 0")
        if max_context_tokens <= output_reserve_tokens:
            raise ValueError(
                f"max_context_tokens ({max_context_tokens}) must be strictly greater than "
                f"output_reserve_tokens ({output_reserve_tokens})"
            )
        for ratio_name, ratio_val in (
            ("system_cap_ratio", system_cap_ratio),
            ("tools_cap_ratio", tools_cap_ratio),
            ("docs_cap_ratio", docs_cap_ratio),
        ):
            if not (0.0 <= ratio_val <= 1.0):
                raise ValueError(f"{ratio_name} must be in [0.0, 1.0], got {ratio_val}")

        self.max_context_tokens = int(max_context_tokens)
        self.output_reserve_tokens = int(output_reserve_tokens)
        self.input_ceiling_tokens = self.max_context_tokens - self.output_reserve_tokens
        self.system_cap_ratio = float(system_cap_ratio)
        self.tools_cap_ratio = float(tools_cap_ratio)
        self.docs_cap_ratio = float(docs_cap_ratio)

    @staticmethod
    def _normalize_doc(item: BudgetDocItem | dict[str, Any], idx: int) -> BudgetDocItem:
        if isinstance(item, BudgetDocItem):
            return item
        if isinstance(item, dict):
            if "doc_id" in item and "content" in item:
                return BudgetDocItem.model_validate(item)
            return BudgetDocItem(
                doc_id=str(item.get("doc_id") or item.get("id") or f"doc_{idx}"),
                content=str(item.get("content") or item.get("text") or ""),
                score=float(item.get("score", 0.5)),
                pinned=bool(item.get("pinned", False)),
            )
        raise TypeError(f"Unsupported doc item type: {type(item).__name__}")

    @staticmethod
    def _normalize_tool(item: BudgetToolItem | dict[str, Any], idx: int) -> BudgetToolItem:
        if isinstance(item, BudgetToolItem):
            return item
        if isinstance(item, dict):
            if "name" in item and "schema_def" in item and isinstance(item["schema_def"], dict):
                return BudgetToolItem.model_validate(item)
            fn_block = item.get("function") if isinstance(item.get("function"), dict) else item
            derived_name = str(
                item.get("name")
                or (fn_block.get("name") if isinstance(fn_block, dict) else None)
                or f"tool_{idx}"
            )
            schema_copy = {
                k: v for k, v in item.items() if k not in ("pinned", "relevance")
            }
            return BudgetToolItem(
                name=derived_name,
                schema_def=schema_copy,
                pinned=bool(item.get("pinned", False)),
                relevance=float(item["relevance"]) if item.get("relevance") is not None else None,
            )
        raise TypeError(f"Unsupported tool item type: {type(item).__name__}")

    def assemble(
        self,
        *,
        system_prompt: str = "",
        query: str = "",
        docs: Sequence[BudgetDocItem | dict[str, Any]] = (),
        tools: Sequence[BudgetToolItem | dict[str, Any]] = (),
        messages: Sequence[dict[str, Any]] = (),
    ) -> AssembledContextPackage:
        """
        Pack system_prompt, tools, docs, and messages strictly within `input_ceiling_tokens`.
        """
        ceiling = self.input_ceiling_tokens
        overflow_prevented = False

        # Derive fallback lexical query from latest user message if explicit query omitted
        effective_query = query or ""
        if not effective_query and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    effective_query = str(msg["content"])
                    break

        # Reserve floor for latest user turn when messages exist so system/tools/docs never starve it
        unassigned_ratio = max(
            0.0, 1.0 - (self.system_cap_ratio + self.tools_cap_ratio + self.docs_cap_ratio)
        )
        history_reserve = (
            min(ceiling, max(1, int(ceiling * max(unassigned_ratio, 0.10))))
            if messages and ceiling >= 2
            else 0
        )
        non_history_ceiling = max(0, ceiling - history_reserve)

        # 1. System Prompt Allocation (truncate head + tail only when exceeding system_cap)
        base_system_cap = min(non_history_ceiling if messages else ceiling, int(ceiling * self.system_cap_ratio))
        if system_prompt and base_system_cap == 0 and (non_history_ceiling > 0 or not messages):
            base_system_cap = 1
        selected_system, sys_truncated = _truncate_head_tail(system_prompt or "", base_system_cap)
        if sys_truncated:
            overflow_prevented = True
        system_tokens = estimate_tokens(selected_system)
        system_slack = max(0, base_system_cap - system_tokens)

        # 2. Tool Schema Ranking & Packing (pinned first, then explicit relevance or lexical overlap)
        base_tools_cap = int(ceiling * self.tools_cap_ratio)
        effective_tools_cap = min(
            max(0, (non_history_ceiling if messages else ceiling) - system_tokens),
            base_tools_cap + system_slack,
        )

        normalized_tools = [self._normalize_tool(t, i) for i, t in enumerate(tools)]
        scored_tools: list[tuple[int, float, int, BudgetToolItem, dict[str, Any], int]] = []
        for idx, t_item in enumerate(normalized_tools):
            compacted = _compact_schema_dict(t_item.schema_def)
            serialized = json.dumps(compacted, separators=(",", ":"), sort_keys=True)
            t_toks = estimate_tokens(serialized)
            rel_score = (
                float(t_item.relevance)
                if t_item.relevance is not None
                else _lexical_overlap_score(effective_query, f"{t_item.name} {serialized}")
            )
            pin_rank = 0 if t_item.pinned else 1
            scored_tools.append((pin_rank, -rel_score, idx, t_item, compacted, t_toks))

        scored_tools.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

        selected_tools: list[dict[str, Any]] = []
        evicted_tool_names: list[str] = []
        tools_tokens = 0
        for _, _, _, t_item, compacted, t_toks in scored_tools:
            if (
                tools_tokens + t_toks <= effective_tools_cap
                and system_tokens + tools_tokens + t_toks <= ceiling
            ):
                selected_tools.append(compacted)
                tools_tokens += t_toks
            else:
                evicted_tool_names.append(t_item.name)
                overflow_prevented = True

        # 3. Retrieved Docs Ranking & Packing (pinned first, then descending score)
        base_docs_cap = int(ceiling * self.docs_cap_ratio)
        prior_slack = max(0, (base_system_cap + base_tools_cap) - (system_tokens + tools_tokens))
        effective_docs_cap = min(
            max(0, (non_history_ceiling if messages else ceiling) - system_tokens - tools_tokens),
            base_docs_cap + prior_slack,
        )

        normalized_docs = [self._normalize_doc(d, i) for i, d in enumerate(docs)]
        ranked_docs = sorted(
            enumerate(normalized_docs),
            key=lambda pair: (0 if pair[1].pinned else 1, -float(pair[1].score), pair[0]),
        )

        selected_docs: list[BudgetDocItem] = []
        evicted_doc_ids: list[str] = []
        docs_tokens = 0
        for _, d_item in ranked_docs:
            d_toks = estimate_tokens(d_item.content)
            if (
                docs_tokens + d_toks <= effective_docs_cap
                and system_tokens + tools_tokens + docs_tokens + d_toks <= ceiling
            ):
                selected_docs.append(d_item)
                docs_tokens += d_toks
            else:
                evicted_doc_ids.append(d_item.doc_id)
                overflow_prevented = True

        # 4. Conversation History Packing (newest-to-oldest, preserving latest user objective)
        remaining_for_history = max(0, ceiling - (system_tokens + tools_tokens + docs_tokens))
        selected_indexed_messages: list[tuple[int, dict[str, Any]]] = []
        history_tokens = 0
        dropped_history_turns = 0

        if messages:
            msg_list = [dict(m) for m in messages if isinstance(m, dict)]
            if msg_list:
                # Locate latest user objective (or newest message if no 'user' role present)
                latest_user_idx = len(msg_list) - 1
                for i in range(len(msg_list) - 1, -1, -1):
                    if str(msg_list[i].get("role", "")).lower() == "user":
                        latest_user_idx = i
                        break

                anchor_msg, anchor_toks, anchor_truncated = _truncate_message_to_budget(
                    msg_list[latest_user_idx], remaining_for_history
                )
                if anchor_truncated:
                    overflow_prevented = True
                selected_indexed_messages.append((latest_user_idx, anchor_msg))
                history_tokens += anchor_toks

                # Pack remaining turns from newest to oldest within remaining ceiling
                for idx in range(len(msg_list) - 1, -1, -1):
                    if idx == latest_user_idx:
                        continue
                    candidate = msg_list[idx]
                    cand_toks = estimate_tokens(_serialize_message_for_budget(candidate))
                    if history_tokens + cand_toks <= remaining_for_history:
                        selected_indexed_messages.append((idx, candidate))
                        history_tokens += cand_toks
                    else:
                        dropped_history_turns += 1
                        overflow_prevented = True

                # Restore exact chronological order
                selected_indexed_messages.sort(key=lambda pair: pair[0])

        selected_messages = [m for _, m in selected_indexed_messages]

        # 5. Final Invariant Gate (INV-BUDGET-01)
        total_allocated = system_tokens + tools_tokens + docs_tokens + history_tokens
        if total_allocated > ceiling:  # Defensive hard assertion (unreachable by construction)
            raise AssertionError(
                f"INV-BUDGET-01 violated: allocated {total_allocated} > ceiling {ceiling}"
            )

        remaining_tokens = ceiling - total_allocated
        utilization_ratio = round(total_allocated / ceiling, 6) if ceiling > 0 else 0.0

        ledger = ContextBudgetLedger(
            max_context_tokens=self.max_context_tokens,
            output_reserve_tokens=self.output_reserve_tokens,
            input_ceiling_tokens=ceiling,
            system_tokens=system_tokens,
            tools_tokens=tools_tokens,
            docs_tokens=docs_tokens,
            history_tokens=history_tokens,
            total_allocated_tokens=total_allocated,
            remaining_tokens=remaining_tokens,
            utilization_ratio=utilization_ratio,
            evicted_doc_ids=evicted_doc_ids,
            evicted_tool_names=evicted_tool_names,
            dropped_history_turns=dropped_history_turns,
            overflow_prevented=overflow_prevented,
        )

        return AssembledContextPackage(
            system_prompt=selected_system,
            selected_tools=selected_tools,
            selected_docs=selected_docs,
            selected_messages=selected_messages,
            ledger=ledger,
        )
