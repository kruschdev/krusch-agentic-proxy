"""
Comprehensive pytest suite for the 6 new Production Agentic Systems in krusch-agentic-proxy:
  - #1 Async Agent Orchestrator (120+ concurrent asyncio tasks, sync tool offloading, circuit breaker)
  - #2 Dynamic Context Budget Assembler (strict 3-bucket + history token allocator, overflow prevention)
  - #3 Strict Pydantic Tool Router (JSON-Schema -> Pydantic v2 compilation + error-trace retry)
  - #6 Typed Multi-Agent Swarm (frozen Pydantic v2 ResearchBrief -> CodeArtifact -> ReviewVerdict FSM)
  - #12 Cost Kill-Switch Middleware ($0.50 hard cap, local Ollama fallback, loop-depth breaker)
  - #14 Shadow Trajectory Comparator (non-blocking 5% background mirror + SQLite WAL Levenshtein/Jaccard diff)
"""

import asyncio
import json
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from src.api_gateway import _cost_killswitch, _shadow_comparator, app
from src.budget_assembler import DynamicContextBudgetAssembler
from src.cost_killswitch import CostKillSwitch
from src.orchestrator import AgentTaskSpec, AsyncAgentOrchestrator, CircuitState
from src.pydantic_tool_router import StrictPydanticToolRouter
from src.shadow_comparator import ShadowTrajectoryComparator, compute_tool_sequence_diff
from src.typed_swarm import (
    CodeArtifact,
    EvidenceFinding,
    ResearchBrief,
    ReviewIssue,
    ReviewVerdict,
    SwarmPhase,
    TypedMultiAgentSwarm,
    assert_typed_handoff,
)


# ---------------------------------------------------------------------------
# #1 Async Agent Orchestrator (120+ concurrent tasks & circuit breaker)
# ---------------------------------------------------------------------------


def test_async_orchestrator_120_concurrent_tasks_and_fault_isolation():
    async def _run():
        active = 0
        peak_concurrency = 0

        async def fake_llm(spec: AgentTaskSpec) -> Dict[str, Any]:
            nonlocal active, peak_concurrency
            active += 1
            peak_concurrency = max(peak_concurrency, active)
            await asyncio.sleep(0.01)
            active -= 1
            if spec.task_id == "task-poison":
                raise RuntimeError("transient upstream reset")
            return {
                "content": f"done:{spec.task_id}",
                "tool_calls": [{"name": "lookup_metric", "arguments": {"id": spec.task_id}}],
            }

        def sync_blocking_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            # Sync tool offloaded via asyncio.to_thread without blocking event loop
            time.sleep(0.002)
            return {"tool": name, "echo": arguments["id"]}

        orch = AsyncAgentOrchestrator(
            max_concurrency=64,
            llm_fn=fake_llm,
            tool_fn=sync_blocking_tool,
            base_backoff_s=0.002,
        )
        specs = [
            AgentTaskSpec(task_id=f"task-{i:03d}", prompt=f"Analyze shard {i}", max_retries=1)
            for i in range(120)
        ]
        specs.append(AgentTaskSpec(task_id="task-poison", prompt="Fail cleanly", max_retries=1))

        report = await orch.run_batch(specs)
        assert report.total_tasks == 121
        assert report.succeeded == 120
        assert report.failed == 1
        assert peak_concurrency > 10

    asyncio.run(_run())


def test_async_orchestrator_host_circuit_breaker():
    async def _run():
        async def dead_host_llm(_spec: AgentTaskSpec) -> str:
            raise ConnectionError("upstream refused")

        orch = AsyncAgentOrchestrator(
            max_concurrency=8,
            llm_fn=dead_host_llm,
            failure_threshold=3,
            recovery_timeout_s=60.0,
            base_backoff_s=0.001,
        )
        for idx in range(3):
            out = await orch.run_task(
                AgentTaskSpec(task_id=f"fail-{idx}", prompt="x", host_key="node-alpha", max_retries=0)
            )
            assert out.status == "FAILED"

        # 4th call trips fast-fail CIRCUIT_OPEN immediately
        short_circuited = await orch.run_task(
            AgentTaskSpec(task_id="fail-open", prompt="x", host_key="node-alpha", max_retries=0)
        )
        assert short_circuited.ok is False
        assert short_circuited.status == "CIRCUIT_OPEN"
        assert orch.circuit_breaker.get_state("node-alpha") == CircuitState.OPEN

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# #2 Dynamic Context Budget Assembler
# ---------------------------------------------------------------------------


def test_dynamic_context_budget_assembler_enforces_hard_ceiling():
    assembler = DynamicContextBudgetAssembler(
        max_context_tokens=2048,
        output_reserve_tokens=512,
        system_cap_ratio=0.25,
        tools_cap_ratio=0.20,
        docs_cap_ratio=0.35,
    )
    docs = [
        {"doc_id": f"doc-{i}", "content": "Relevant architecture clause. " * 60, "score": 1.0 - i * 0.08}
        for i in range(10)
    ]
    tools = [
        {
            "name": f"tool_{i}",
            "pinned": i == 0,
            "schema_def": {"type": "object", "properties": {"q": {"type": "string", "description": "x" * 300}}},
        }
        for i in range(8)
    ]
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}: " + ("history " * 50)}
        for i in range(10)
    ]

    pkg = assembler.assemble(
        system_prompt="System guardrail policy. " * 80,
        query="Use tool_0 to inspect architecture clause",
        docs=docs,
        tools=tools,
        messages=messages,
    )
    assert pkg.ledger.input_ceiling_tokens == 1536
    assert pkg.ledger.total_allocated_tokens <= pkg.ledger.input_ceiling_tokens
    assert pkg.ledger.overflow_prevented is True
    assert len(pkg.ledger.evicted_doc_ids) > 0
    assert pkg.selected_messages[-1]["content"].startswith("Turn 9:")


# ---------------------------------------------------------------------------
# #3 Strict Pydantic Tool Router
# ---------------------------------------------------------------------------


def test_strict_pydantic_tool_router_retries_with_validation_trace():
    router = StrictPydanticToolRouter(max_retries=3)
    router.register_tool(
        name="provision_sandbox",
        schema={
            "type": "object",
            "properties": {
                "memory_mb": {"type": "integer", "minimum": 128, "maximum": 8192},
                "network_mode": {"type": "string", "enum": ["none", "loopback"]},
            },
            "required": ["memory_mb", "network_mode"],
        },
        handler=lambda memory_mb, network_mode: {"provisioned": True, "memory_mb": memory_mb, "mode": network_mode},
    )

    prompts_seen = []
    responses = [
        # Attempt 1: hallucinated extra parameter + out-of-range integer + invalid enum
        '{"name": "provision_sandbox", "arguments": {"memory_mb": 64, "network_mode": "public_wan", "sudo": true}}',
        # Attempt 2: repaired after reading Pydantic v2 ValidationError trace
        '<think>Fixing schema violations.</think>\n```json\n{"name": "provision_sandbox", "arguments": {"memory_mb": 1024, "network_mode": "none"}}\n```',
    ]

    async def fake_llm(prompt: str) -> str:
        prompts_seen.append(prompt)
        return responses[len(prompts_seen) - 1]

    res = asyncio.run(
        router.execute_with_retry(
            prompt="Provision an isolated 1GB sandbox",
            llm_generate_fn=fake_llm,
            expected_tool="provision_sandbox",
        )
    )
    assert res.ok is True
    assert res.attempts == 2
    assert res.validated_args == {"memory_mb": 1024, "network_mode": "none"}
    assert res.output == {"provisioned": True, "memory_mb": 1024, "mode": "none"}
    assert len(prompts_seen) == 2
    assert "extra_forbidden" in prompts_seen[1] or "memory_mb" in prompts_seen[1]


# ---------------------------------------------------------------------------
# #6 Typed Multi-Agent Swarm
# ---------------------------------------------------------------------------


def test_typed_multi_agent_swarm_forbids_raw_text_and_converges_revision_cycle():
    # Raw string handoff is deterministically rejected
    with pytest.raises(TypeError, match="Raw text handoffs are forbidden"):
        assert_typed_handoff("Here is my raw text research summary", ResearchBrief)

    def researcher(state):
        return ResearchBrief(
            objective=state.objective,
            constraints=["Zero network egress"],
            findings=[EvidenceFinding(source="spec.md", summary="Enforce bwrap --unshare-net", confidence=0.98)],
            target_files=["src/sandbox.py"],
            acceptance_criteria=["Unit test covers --unshare-net"],
        )

    def coder(state):
        rev = state.revision_count + 1
        impl = "def run(): return 'unverified'" if rev == 1 else "def run(): return 'bwrap --unshare-net'"
        return CodeArtifact(
            revision=rev,
            file_path="src/sandbox.py",
            language="python",
            implementation=impl,
            unit_test_code="def test_run(): assert '--unshare-net' in run()",
            addressed_findings=["Enforce bwrap --unshare-net"],
        )

    def reviewer(state):
        assert state.artifact is not None
        if "--unshare-net" not in state.artifact.implementation:
            return ReviewVerdict(
                approved=False,
                revision_evaluated=state.artifact.revision,
                issues=[
                    ReviewIssue(
                        severity="P0",
                        location="src/sandbox.py:1",
                        description="Missing --unshare-net flag",
                        remediation="Include bwrap --unshare-net",
                    )
                ],
                rationale="Sandbox lacks network namespace isolation.",
            )
        return ReviewVerdict(
            approved=True,
            revision_evaluated=state.artifact.revision,
            issues=[],
            rationale="Verified zero-egress flag.",
        )

    swarm = TypedMultiAgentSwarm(
        researcher_fn=researcher,
        coder_fn=coder,
        reviewer_fn=reviewer,
        max_revisions=2,
    )
    final_state = asyncio.run(swarm.run("Harden sandbox execution"))
    assert final_state.phase == SwarmPhase.APPROVED
    assert final_state.revision_count == 1
    assert final_state.artifact.revision == 2
    phases = [(t.from_phase, t.to_phase) for t in final_state.transitions]
    assert phases == [
        (SwarmPhase.RESEARCH, SwarmPhase.CODE),
        (SwarmPhase.CODE, SwarmPhase.REVIEW),
        (SwarmPhase.REVIEW, SwarmPhase.CODE),
        (SwarmPhase.CODE, SwarmPhase.REVIEW),
        (SwarmPhase.REVIEW, SwarmPhase.APPROVED),
    ]


# ---------------------------------------------------------------------------
# #12 Cost Kill-Switch & #14 Shadow Comparator + FastAPI Choke-Point Integration
# ---------------------------------------------------------------------------


def test_cost_killswitch_and_shadow_comparator_and_gateway_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")

    # 1. Direct CostKillSwitch unit verification ($0.50 cap + local Ollama downgrade)
    ks = CostKillSwitch(hard_limit_usd=0.50, soft_fallback_ratio=0.80, max_loop_depth=4)
    ks.record_usage("sess-1", model="openai/gpt-5", prompt_tokens=15_000, completion_tokens=9_000)  # $0.42
    d_fallback = ks.evaluate_pre_call(
        "sess-1", model="openai/gpt-5", estimated_prompt_tokens=4_000, max_completion_tokens=2_000
    )
    assert d_fallback.allowed is True
    assert d_fallback.action == "FALLBACK_LOCAL"
    assert d_fallback.effective_model == "qwen2.5-coder:7b"

    ks.record_usage("sess-1", model="openai/gpt-5", prompt_tokens=5_000, completion_tokens=2_000)  # +$0.11 -> $0.53
    d_halt = ks.evaluate_pre_call(
        "sess-1", model="gemini-4-argon", estimated_prompt_tokens=1_000, max_completion_tokens=1_000
    )
    assert d_halt.allowed is False
    assert d_halt.action == "HALTED_BUDGET_EXCEEDED"

    # 2. Direct ShadowTrajectoryComparator non-blocking verification on SQLite WAL
    shadow_db = str(tmp_path / "shadow.db")
    comp = ShadowTrajectoryComparator(db_path=shadow_db, sample_rate=0.05)

    async def _verify_shadow_non_blocking():
        async def slow_candidate():
            await asyncio.sleep(0.08)
            return {
                "tool_calls": [
                    {"name": "read_file", "arguments": {"path": "a.py"}},
                    {"name": "run_tests", "arguments": {"cmd": "pytest"}},
                ],
                "latency_ms": 80.0,
                "cost_usd": 0.0,
            }

        t0 = time.perf_counter()
        task = comp.schedule_background_shadow(
            request_id="req-mirror-01",
            primary_model="gemini-4-argon",
            shadow_model="qwen2.5-coder:7b",
            primary_calls=[
                {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"function": {"name": "run_tests", "arguments": '{"cmd": "pytest"}'}},
            ],
            primary_latency_ms=420.0,
            primary_cost_usd=0.012,
            shadow_runner=slow_candidate,
            force=True,
        )
        dispatch_ms = (time.perf_counter() - t0) * 1000.0
        assert dispatch_ms < 20.0  # Main request thread returns immediately without awaiting 80ms shadow
        assert task is not None
        await comp.flush_pending()
        rows = comp.list_recent(limit=5)
        assert len(rows) == 1
        assert rows[0]["exact_dag_match"] is True
        assert rows[0]["levenshtein_distance"] == 0
        comp.close()

    asyncio.run(_verify_shadow_non_blocking())

    diff = compute_tool_sequence_diff(
        [{"name": "read_file", "arguments": {"path": "x"}}],
        [{"name": "read_file", "arguments": {"path": "x"}}, {"name": "apply_diff", "arguments": {"path": "x"}}],
    )
    assert diff["levenshtein_distance"] == 1
    assert diff["exact_dag_match"] is False

    # 3. FastAPI Gateway Choke-Point Integration Tests
    client = TestClient(app)

    r_batch = client.post(
        "/v1/orchestrate/batch",
        json={
            "max_concurrency": 32,
            "tasks": [{"task_id": f"gw-{i}", "prompt": f"Ping {i}"} for i in range(25)],
        },
    )
    assert r_batch.status_code == 200
    assert r_batch.json()["succeeded"] == 25

    r_budget = client.post(
        "/v1/context/budget",
        json={
            "max_context_tokens": 2048,
            "output_reserve_tokens": 512,
            "system_prompt": "Follow invariants.",
            "query": "search",
            "docs": [{"doc_id": "d1", "content": "alpha " * 100, "score": 0.9}],
        },
    )
    assert r_budget.status_code == 200
    assert r_budget.json()["ledger"]["total_allocated_tokens"] <= 1536

    r_strict = client.post(
        "/v1/tools/execute-strict",
        json={
            "prompt": "Deploy replica count",
            "tool_name": "scale_service",
            "parameters_schema": {
                "type": "object",
                "properties": {"replicas": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["replicas"],
            },
            "scripted_llm_outputs": [
                '{"replicas": 99}',
                '{"replicas": 3}',
            ],
        },
    )
    assert r_strict.status_code == 200
    assert r_strict.json()["ok"] is True
    assert r_strict.json()["attempts"] == 2
    assert r_strict.json()["validated_args"] == {"replicas": 3}

    r_swarm = client.post(
        "/v1/swarm/run",
        json={"objective": "Implement strict token budget middleware", "max_revisions": 2},
    )
    assert r_swarm.status_code == 200
    assert r_swarm.json()["phase"] == "APPROVED"

    _cost_killswitch.reset_session("gw-cost-sess")
    r_cost_ok = client.post(
        "/v1/cost/evaluate",
        json={
            "session_id": "gw-cost-sess",
            "record_usage": {"model": "openai/gpt-5", "prompt_tokens": 20_000, "completion_tokens": 12_000},
            "model": "gemini-4-argon",
            "estimated_prompt_tokens": 1000,
        },
    )
    assert r_cost_ok.status_code == 402
    assert r_cost_ok.json()["decision"]["action"] == "HALTED_BUDGET_EXCEEDED"

    # Also verify /v1/chat/completions halts with 402 when X-Session-Id is over $0.50
    r_chat_halted = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "gw-cost-sess"},
        json={"model": "gemini-4-argon", "messages": [{"role": "user", "content": "Continue loop"}]},
    )
    assert r_chat_halted.status_code == 402
    _cost_killswitch.reset_session("gw-cost-sess")

    # Verify /v1/chat/completions X-Krusch-Shadow: force schedules background shadow comparison
    # AND live X-Session-Id traffic records cumulative spend post-generation
    _cost_killswitch.reset_session("gw-live-spend-sess")
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"name": "get_status", "arguments": {"service": "db"}}'
        r_shadow_chat = client.post(
            "/v1/chat/completions",
            headers={
                "X-Krusch-Shadow": "force",
                "X-Request-Id": "gw-shadow-req-1",
                "X-Session-Id": "gw-live-spend-sess",
            },
            json={
                "model": "openai/gpt-5",
                "messages": [{"role": "user", "content": "Check db " * 100}],
                "tools": [{"type": "function", "function": {"name": "get_status", "parameters": {"type": "object"}}}],
            },
        )
        assert r_shadow_chat.status_code == 200
        live_ledger = _cost_killswitch.get_session_ledger("gw-live-spend-sess")
        assert live_ledger["calls_recorded"] == 1
        assert live_ledger["spent_usd"] > 0.0

        r_shadow_list = client.get("/v1/shadow/comparisons?flush=1")
        assert r_shadow_list.status_code == 200
        assert r_shadow_list.json()["summary"]["total_comparisons"] >= 1
        latest_cmp = r_shadow_list.json()["comparisons"][0]
        assert latest_cmp["primary_latency_ms"] != 1.0
        assert latest_cmp["primary_latency_ms"] > 0.0
    _cost_killswitch.reset_session("gw-live-spend-sess")

    # Verify batch task cap (400) and bounded HTTP payload limit (413)
    monkeypatch.setenv("KRUSCH_MAX_BATCH_TASKS", "10")
    r_batch_over = client.post(
        "/v1/orchestrate/batch",
        json={"tasks": [{"task_id": f"t-{i}", "prompt": "hi"} for i in range(11)]},
    )
    assert r_batch_over.status_code == 400
    monkeypatch.delenv("KRUSCH_MAX_BATCH_TASKS", raising=False)

    monkeypatch.setenv("KRUSCH_MAX_REQUEST_BYTES", "1024")
    r_too_large = client.post(
        "/v1/context/budget",
        json={"system_prompt": "x" * 2048},
    )
    assert r_too_large.status_code == 413
    monkeypatch.delenv("KRUSCH_MAX_REQUEST_BYTES", raising=False)
