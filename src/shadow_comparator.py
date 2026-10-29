"""
Silent 5% Shadow Trajectory Comparator (INV-14, #14).

Mirrors a configurable fraction (default 5%) of live agentic traffic to a candidate
shadow model/topology in a non-blocking background asyncio task, computing exact
Levenshtein edit distance on tool-call sequences, argument key-set Jaccard
similarity, DAG equivalence, and latency/cost deltas persisted to SQLite WAL.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ShadowComparisonRecord(BaseModel):
    """Persisted trajectory comparison telemetry between primary and shadow runs."""

    model_config = ConfigDict(extra="forbid")

    comparison_id: str
    request_id: str
    primary_model: str
    shadow_model: str
    primary_tools: List[str]
    shadow_tools: List[str]
    sequence_similarity: float
    levenshtein_distance: int
    arg_key_jaccard: float
    exact_dag_match: bool
    primary_latency_ms: float
    shadow_latency_ms: float
    latency_delta_ms: float
    primary_cost_usd: float
    shadow_cost_usd: float
    cost_delta_usd: float
    shadow_error: Optional[str] = None
    created_at: float


def _extract_normalized_call(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Extract ``(tool_name, args_dict)`` from OpenAI or normalized tool call dicts."""
    if not isinstance(call, dict):
        return ("", {})

    fn_block = call.get("function")
    if isinstance(fn_block, dict):
        name = str(fn_block.get("name") or "").strip()
        raw_args = fn_block.get("arguments", {})
    else:
        name = str(call.get("name") or "").strip()
        raw_args = call.get("arguments", call.get("args", {}))

    if isinstance(raw_args, dict):
        args_dict = raw_args
    elif isinstance(raw_args, str):
        stripped = raw_args.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
                args_dict = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                args_dict = {}
        else:
            args_dict = {}
    else:
        args_dict = {}

    return (name, args_dict)


def _levenshtein_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Exact dynamic-programming Levenshtein edit distance over tool-name sequences."""
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev_row = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr_row = [i] + [0] * len_b
        item_a = seq_a[i - 1]
        for j in range(1, len_b + 1):
            sub_cost = 0 if item_a == seq_b[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,          # deletion
                curr_row[j - 1] + 1,      # insertion
                prev_row[j - 1] + sub_cost,  # substitution / match
            )
        prev_row = curr_row
    return prev_row[len_b]


def compute_tool_sequence_diff(
    primary_calls: Sequence[Dict[str, Any]],
    shadow_calls: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare primary and shadow tool-call trajectories.

    Returns exact Levenshtein distance, normalized sequence similarity, argument
    key-set Jaccard similarity across calls, and exact DAG match status.
    """
    p_norm = [_extract_normalized_call(call) for call in (primary_calls or ())]
    s_norm = [_extract_normalized_call(call) for call in (shadow_calls or ())]

    p_tools = [name for name, _ in p_norm]
    s_tools = [name for name, _ in s_norm]

    dist = _levenshtein_distance(p_tools, s_tools)
    denom = max(len(p_tools), len(s_tools), 1)
    sequence_similarity = 1.0 - (dist / float(denom))

    p_arg_keys: Set[str] = {str(k) for _, args in p_norm for k in args.keys()}
    s_arg_keys: Set[str] = {str(k) for _, args in s_norm for k in args.keys()}
    union_keys = p_arg_keys | s_arg_keys
    if not union_keys:
        arg_key_jaccard = 1.0
    else:
        arg_key_jaccard = len(p_arg_keys & s_arg_keys) / float(len(union_keys))

    exact_dag_match = (
        len(p_norm) == len(s_norm)
        and dist == 0
        and all(
            p_name == s_name and p_args == s_args
            for (p_name, p_args), (s_name, s_args) in zip(p_norm, s_norm)
        )
    )

    return {
        "primary_tools": p_tools,
        "shadow_tools": s_tools,
        "levenshtein_distance": dist,
        "sequence_similarity": sequence_similarity,
        "arg_key_jaccard": arg_key_jaccard,
        "exact_dag_match": exact_dag_match,
    }


class ShadowTrajectoryComparator:
    """
    Asynchronous 5% traffic mirror & SQLite trajectory comparator.

    Executes candidate shadow runs completely off the critical user-response path via
    ``schedule_background_shadow`` and persists structured diffs into ``shadow_comparisons``.
    """

    def __init__(self, db_path: str = ":memory:", sample_rate: float = 0.05) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be between 0.0 and 1.0 inclusive.")

        self.db_path: str = db_path
        self.sample_rate: float = float(sample_rate)
        self._lock = threading.RLock()
        self._pending_tasks: Set[asyncio.Task[Any]] = set()
        self._conn = self._init_sqlite(db_path)

    def _init_sqlite(self, db_path: str) -> sqlite3.Connection:
        is_memory = db_path == ":memory:" or db_path.startswith("file::memory:")
        if not is_memory:
            parent_dir = os.path.dirname(os.path.abspath(db_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with self._lock:
            if not is_memory:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    primary_model TEXT NOT NULL,
                    shadow_model TEXT NOT NULL,
                    primary_tools TEXT NOT NULL,
                    shadow_tools TEXT NOT NULL,
                    sequence_similarity REAL NOT NULL,
                    levenshtein_distance INTEGER NOT NULL,
                    arg_key_jaccard REAL NOT NULL,
                    exact_dag_match INTEGER NOT NULL,
                    primary_latency_ms REAL NOT NULL,
                    shadow_latency_ms REAL NOT NULL,
                    latency_delta_ms REAL NOT NULL,
                    primary_cost_usd REAL NOT NULL,
                    shadow_cost_usd REAL NOT NULL,
                    cost_delta_usd REAL NOT NULL,
                    shadow_error TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_comparisons_created_at
                ON shadow_comparisons (created_at DESC);
                """
            )
            conn.commit()
        return conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def should_sample(self, request_id: str, force: bool = False) -> bool:
        """Deterministic SHA-256 hash bucketing for reproducible traffic sampling."""
        if force:
            return True
        if self.sample_rate <= 0.0:
            return False
        if self.sample_rate >= 1.0:
            return True
        digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return bucket < self.sample_rate

    def _insert_record(self, record: ShadowComparisonRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO shadow_comparisons (
                    comparison_id,
                    request_id,
                    primary_model,
                    shadow_model,
                    primary_tools,
                    shadow_tools,
                    sequence_similarity,
                    levenshtein_distance,
                    arg_key_jaccard,
                    exact_dag_match,
                    primary_latency_ms,
                    shadow_latency_ms,
                    latency_delta_ms,
                    primary_cost_usd,
                    shadow_cost_usd,
                    cost_delta_usd,
                    shadow_error,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.comparison_id,
                    record.request_id,
                    record.primary_model,
                    record.shadow_model,
                    json.dumps(record.primary_tools),
                    json.dumps(record.shadow_tools),
                    record.sequence_similarity,
                    record.levenshtein_distance,
                    record.arg_key_jaccard,
                    1 if record.exact_dag_match else 0,
                    record.primary_latency_ms,
                    record.shadow_latency_ms,
                    record.latency_delta_ms,
                    record.primary_cost_usd,
                    record.shadow_cost_usd,
                    record.cost_delta_usd,
                    record.shadow_error,
                    record.created_at,
                ),
            )
            self._conn.commit()

    async def run_shadow_comparison(
        self,
        *,
        request_id: str,
        primary_model: str,
        shadow_model: str,
        primary_calls: Sequence[Dict[str, Any]],
        primary_latency_ms: float,
        primary_cost_usd: float,
        shadow_runner: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> ShadowComparisonRecord:
        """Execute candidate ``shadow_runner``, compute trajectory diff, and persist to SQLite."""
        started = time.perf_counter()
        shadow_calls: Sequence[Dict[str, Any]] = ()
        shadow_cost_usd = 0.0
        shadow_error: Optional[str] = None

        try:
            maybe_coro = shadow_runner()
            payload = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if isinstance(payload, dict):
                raw_calls = payload.get("tool_calls", payload.get("calls", []))
                shadow_calls = raw_calls if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)) else ()
                shadow_latency_ms = float(payload.get("latency_ms", elapsed_ms))
                shadow_cost_usd = float(payload.get("cost_usd", 0.0))
                if payload.get("error") is not None:
                    shadow_error = str(payload["error"])
            else:
                shadow_latency_ms = elapsed_ms
                shadow_error = f"Invalid shadow_runner return type: {type(payload).__name__}"
        except Exception as exc:  # noqa: BLE001 - isolate shadow failures from user traffic
            shadow_latency_ms = (time.perf_counter() - started) * 1000.0
            shadow_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Shadow comparison execution failed for request_id=%s: %s",
                request_id,
                shadow_error,
            )

        diff = compute_tool_sequence_diff(primary_calls, shadow_calls)
        p_lat = float(primary_latency_ms)
        p_cost = float(primary_cost_usd)

        record = ShadowComparisonRecord(
            comparison_id=f"shdw-{uuid.uuid4().hex[:16]}",
            request_id=str(request_id),
            primary_model=str(primary_model),
            shadow_model=str(shadow_model),
            primary_tools=diff["primary_tools"],
            shadow_tools=diff["shadow_tools"],
            sequence_similarity=diff["sequence_similarity"],
            levenshtein_distance=diff["levenshtein_distance"],
            arg_key_jaccard=diff["arg_key_jaccard"],
            exact_dag_match=diff["exact_dag_match"],
            primary_latency_ms=p_lat,
            shadow_latency_ms=shadow_latency_ms,
            latency_delta_ms=shadow_latency_ms - p_lat,
            primary_cost_usd=p_cost,
            shadow_cost_usd=shadow_cost_usd,
            cost_delta_usd=shadow_cost_usd - p_cost,
            shadow_error=shadow_error,
            created_at=time.time(),
        )

        await asyncio.to_thread(self._insert_record, record)
        return record

    def schedule_background_shadow(self, **kwargs: Any) -> Optional[asyncio.Task[Any]]:
        """
        Fire-and-forget shadow mirror when ``should_sample`` passes.

        Never blocks or raises into the foreground user request path.
        """
        force = bool(kwargs.pop("force", False))
        request_id = str(kwargs.get("request_id", ""))
        if not self.should_sample(request_id, force=force):
            return None

        async def _guarded_runner() -> Optional[ShadowComparisonRecord]:
            try:
                return await self.run_shadow_comparison(**kwargs)
            except asyncio.CancelledError:
                logger.debug("Shadow comparison task cancelled for request_id=%s", request_id)
                return None
            except Exception as exc:  # noqa: BLE001 - never surface background shadow faults
                logger.warning(
                    "Unhandled background shadow error for request_id=%s: %s",
                    request_id,
                    exc,
                )
                return None

        task: asyncio.Task[Any] = asyncio.create_task(_guarded_runner())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def flush_pending(self) -> None:
        """Await all currently scheduled background shadow tasks."""
        while self._pending_tasks:
            snapshot = list(self._pending_tasks)
            await asyncio.gather(*snapshot, return_exceptions=True)

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent persisted shadow comparison rows."""
        capped_limit = max(1, int(limit))
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    comparison_id,
                    request_id,
                    primary_model,
                    shadow_model,
                    primary_tools,
                    shadow_tools,
                    sequence_similarity,
                    levenshtein_distance,
                    arg_key_jaccard,
                    exact_dag_match,
                    primary_latency_ms,
                    shadow_latency_ms,
                    latency_delta_ms,
                    primary_cost_usd,
                    shadow_cost_usd,
                    cost_delta_usd,
                    shadow_error,
                    created_at
                FROM shadow_comparisons
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?;
                """,
                (capped_limit,),
            )
            rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "comparison_id": row["comparison_id"],
                    "request_id": row["request_id"],
                    "primary_model": row["primary_model"],
                    "shadow_model": row["shadow_model"],
                    "primary_tools": json.loads(row["primary_tools"]),
                    "shadow_tools": json.loads(row["shadow_tools"]),
                    "sequence_similarity": float(row["sequence_similarity"]),
                    "levenshtein_distance": int(row["levenshtein_distance"]),
                    "arg_key_jaccard": float(row["arg_key_jaccard"]),
                    "exact_dag_match": bool(row["exact_dag_match"]),
                    "primary_latency_ms": float(row["primary_latency_ms"]),
                    "shadow_latency_ms": float(row["shadow_latency_ms"]),
                    "latency_delta_ms": float(row["latency_delta_ms"]),
                    "primary_cost_usd": float(row["primary_cost_usd"]),
                    "shadow_cost_usd": float(row["shadow_cost_usd"]),
                    "cost_delta_usd": float(row["cost_delta_usd"]),
                    "shadow_error": row["shadow_error"],
                    "created_at": float(row["created_at"]),
                }
            )
        return results

    def summary_metrics(self) -> Dict[str, Any]:
        """Return aggregate parity, latency, cost, and error metrics across shadow runs."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total_comparisons,
                    AVG(CASE WHEN exact_dag_match = 1 THEN 1.0 ELSE 0.0 END) AS exact_dag_match_rate,
                    AVG(sequence_similarity) AS avg_sequence_similarity,
                    AVG(arg_key_jaccard) AS avg_arg_key_jaccard,
                    AVG(latency_delta_ms) AS avg_latency_delta_ms,
                    AVG(cost_delta_usd) AS avg_cost_delta_usd,
                    SUM(CASE WHEN shadow_error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
                FROM shadow_comparisons;
                """
            ).fetchone()

        total = int(row["total_comparisons"] or 0) if row is not None else 0
        errors = int(row["error_count"] or 0) if row is not None else 0
        if total == 0:
            return {
                "total_comparisons": 0,
                "sample_rate": self.sample_rate,
                "exact_dag_match_rate": 0.0,
                "avg_sequence_similarity": 0.0,
                "avg_arg_key_jaccard": 0.0,
                "avg_latency_delta_ms": 0.0,
                "avg_cost_delta_usd": 0.0,
                "error_count": 0,
                "error_rate": 0.0,
                "pending_tasks": len(self._pending_tasks),
            }

        return {
            "total_comparisons": total,
            "sample_rate": self.sample_rate,
            "exact_dag_match_rate": float(row["exact_dag_match_rate"] or 0.0),
            "avg_sequence_similarity": float(row["avg_sequence_similarity"] or 0.0),
            "avg_arg_key_jaccard": float(row["avg_arg_key_jaccard"] or 0.0),
            "avg_latency_delta_ms": float(row["avg_latency_delta_ms"] or 0.0),
            "avg_cost_delta_usd": float(row["avg_cost_delta_usd"] or 0.0),
            "error_count": errors,
            "error_rate": errors / float(total),
            "pending_tasks": len(self._pending_tasks),
        }


__all__ = (
    "ShadowComparisonRecord",
    "ShadowTrajectoryComparator",
    "compute_tool_sequence_diff",
)
