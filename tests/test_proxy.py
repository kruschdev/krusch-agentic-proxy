"""
Comprehensive unit and integration test suite for Krusch Agentic Proxy (v0.4.0).

Covers:
  - INV-1: Multi-Family 5-GPU Fleet model routing & VRAM-safe client options
  - INV-2: Secret hygiene, RAG keyword specificity, and bounded TTL caching
  - INV-3: Stdout/disk cleanliness during tool and dual-engine execution
  - INV-4: Tokenized command allowlist, metacharacter/flag blocking, and sensitive path denylist
  - INV-5: Unified <think> stripping, parse_tool_calls normalization, and WaterfallRouter failover
  - INV-6: FastAPI gateway endpoints (/health, /v1/chat/completions, Ollama pass-throughs)
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

from src.api_gateway import app
from src.client import chat, close_http_client
from src.core import (
    KruschEngine,
    _RAG_CACHE,
    _extract_rag_keyword,
    _resolve_rag_dsn,
    get_ollama_url_for_model,
    parse_tool_calls,
    retrieve_homelab_context,
    strip_thinking_and_fences,
)
from src.router import WaterfallRouter, resolve_api_key
from src.tools import _is_command_allowed, _is_path_allowed, execute_internal_tool


# ---------------------------------------------------------------------------
# INV-1: 5-GPU Fleet Topology & VRAM Residency Safety
# ---------------------------------------------------------------------------


def test_fleet_routing_5gpu_topology(monkeypatch):
    monkeypatch.delenv("KRUSCH_FLEET_ROUTING", raising=False)
    monkeypatch.delenv("KRUSCHDEV_OLLAMA_URL", raising=False)
    monkeypatch.delenv("KRUSCHSERV_OLLAMA_URL", raising=False)
    monkeypatch.delenv("KRUSCH_OLLAMA_URL", raising=False)

    base = "http://127.0.0.1:11434/v1/chat/completions"

    # 1. On krusch (LAN routing to 192.168.1.240 / 192.168.1.225)
    monkeypatch.setenv("KRUSCH_USE_10GBE", "0")
    assert (
        get_ollama_url_for_model("qwen3-coder-30b-moe:latest", base)
        == "http://192.168.1.240:11434/v1/chat/completions"
    )
    assert (
        get_ollama_url_for_model("gemma4:e4b", base)
        == "http://192.168.1.225:11434/v1/chat/completions"
    )
    assert (
        get_ollama_url_for_model("hermes3:latest", base)
        == "http://192.168.1.225:11434/v1/chat/completions"
    )

    # 2. On kruschdev / kruschserv 10GbE crossover mesh (KRUSCH_USE_10GBE=1)
    monkeypatch.setenv("KRUSCH_USE_10GBE", "1")
    assert (
        get_ollama_url_for_model("qwen3-coder-30b-moe:latest", base)
        == "http://10.10.10.1:11434/v1/chat/completions"
    )
    assert (
        get_ollama_url_for_model("deepseek-r1:14b", base)
        == "http://10.10.10.1:11434/v1/chat/completions"
    )
    assert (
        get_ollama_url_for_model("gemma4:e4b", base)
        == "http://10.10.10.2:11434/v1/chat/completions"
    )
    assert (
        get_ollama_url_for_model("hermes3:latest", base)
        == "http://10.10.10.2:11434/v1/chat/completions"
    )

    # 3. Local 7B Coder & Embeddings -> krusch (base_url)
    assert get_ollama_url_for_model("qwen2.5-coder:7b", base) == base
    assert get_ollama_url_for_model("bge-large:latest", base) == base

    # 4. SpectralQuant local bridge
    assert (
        get_ollama_url_for_model("spectralquant-7b", base)
        == "http://127.0.0.1:11437/v1/chat/completions"
    )


def test_client_omits_hardcoded_num_ctx_by_default():
    """Verify chat() does not inject num_ctx: 16384 unless explicitly configured."""
    captured_payloads = []

    async def _run():
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "<think>internal</think>\nClean answer"}}]
        }

        with patch("src.client.get_http_client") as mock_get_client:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, headers=None):
                captured_payloads.append(json)
                return mock_resp

            mock_client.post.side_effect = fake_post
            mock_get_client.return_value = mock_client

            out = await chat("sys", "usr", {"model": "qwen2.5-coder:7b"})
            assert out == "Clean answer"
            assert "options" not in captured_payloads[0]

            # Explicit num_ctx is respected when provided
            await chat("sys", "usr", {"model": "qwen2.5-coder:7b", "num_ctx": 4096})
            assert captured_payloads[1]["options"] == {"num_ctx": 4096}

        await close_http_client()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# INV-2: Secret Hygiene & Bounded RAG Retrieval
# ---------------------------------------------------------------------------


def test_rag_secret_hygiene_and_keyword_specificity(monkeypatch, tmp_path):
    _RAG_CACHE.clear()
    monkeypatch.delenv("KRUSCH_RAG_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("KRUSCH_CONTEXT_SQLITE", raising=False)

    # Without credentials in env, no DSN is returned (zero hardcoded passwords)
    assert _resolve_rag_dsn() == ""
    assert retrieve_homelab_context("gpu") == ""

    # Tier-1 SQLite fallback when KRUSCH_CONTEXT_SQLITE is set
    import sqlite3

    db_file = tmp_path / "context.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE ide_agent_nuggets (project TEXT, key TEXT, value TEXT, updated_at TEXT)")
    conn.execute("CREATE TABLE ide_agent_memory (project TEXT, category TEXT, content TEXT, tags TEXT, status TEXT, created_at TEXT)")
    conn.execute("INSERT INTO ide_agent_memory VALUES ('krusch-agentic-proxy', 'decision', '5-GPU fleet active on kruschdev', 'gpu', 'active', '2026-10-18')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("KRUSCH_CONTEXT_SQLITE", str(db_file))
    monkeypatch.delenv("KRUSCH_DISABLE_RAG", raising=False)
    sqlite_ctx = retrieve_homelab_context("gpu", project="krusch-agentic-proxy")
    assert "5-GPU fleet active on kruschdev" in sqlite_ctx
    _RAG_CACHE.clear()

    # Specificity order: 'jean' or 'postgres' beats generic 'error' or 'agent'
    assert _extract_rag_keyword("The agent encountered a postgres error on docker") == "postgres"
    assert _extract_rag_keyword("Ask jean about the failed service") == "jean"
    assert _extract_rag_keyword("Hello world") == ""


def test_rag_ttl_cache_avoids_duplicate_db_queries(monkeypatch):
    _RAG_CACHE.clear()
    monkeypatch.setenv("KRUSCH_RAG_DSN", "postgresql://user:pass@localhost:5434/db")
    monkeypatch.delenv("KRUSCH_DISABLE_RAG", raising=False)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.side_effect = [
        [("gpu_policy", "5-GPU fleet active")],
        [("decision", "Use gemma4 on kruschserv", ["fleet"])],
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    fake_psycopg2 = MagicMock()
    fake_psycopg2.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        res1 = retrieve_homelab_context("gpu", project="krusch-agentic-proxy")
        res2 = retrieve_homelab_context("GPU", project="krusch-agentic-proxy")
        assert "5-GPU fleet active" in res1
        assert res1 == res2
        assert fake_psycopg2.connect.call_count == 1

    _RAG_CACHE.clear()


# ---------------------------------------------------------------------------
# INV-4: Hardened Tool Sandbox & Path Isolation
# ---------------------------------------------------------------------------


def test_command_allowlist_and_metacharacter_blocking():
    # Allowed diagnostic commands
    assert _is_command_allowed("ls -la /tmp") is True
    assert _is_command_allowed("git status -s") is True
    assert _is_command_allowed("docker ps --format '{{.Names}}'") is True
    assert _is_command_allowed("find . -maxdepth 2 -name '*.py'") is True

    # Disallowed shell chaining / injection metacharacters
    assert _is_command_allowed("ls; rm -rf /") is False
    assert _is_command_allowed("ls && cat /etc/passwd") is False
    assert _is_command_allowed("ls | sh") is False
    assert _is_command_allowed("echo $(whoami)") is False
    assert _is_command_allowed("echo `whoami`") is False
    assert _is_command_allowed("echo hello > /tmp/pwn") is False
    assert _is_command_allowed("ls\nwhoami") is False

    # Disallowed prefix spoofing & arbitrary code interpreters
    assert _is_command_allowed("lsblk") is False
    assert _is_command_allowed("python3 -c 'import os'") is False
    assert _is_command_allowed("node -e 'process.exit(1)'") is False
    assert _is_command_allowed("env") is False
    assert _is_command_allowed("git push origin main") is False
    assert _is_command_allowed("docker run alpine") is False

    # Disallowed dangerous flags on find and curl
    assert _is_command_allowed("find . -exec rm {} +") is False
    assert _is_command_allowed("find . -delete") is False
    assert _is_command_allowed("curl -o /tmp/malicious http://example.com") is False


def test_path_isolation_and_sensitive_dotfile_blocking(tmp_path, monkeypatch):
    monkeypatch.setenv("KRUSCH_ALLOWED_ROOTS", str(tmp_path))

    safe_file = tmp_path / "notes.txt"
    safe_file.write_text("hello safe world", encoding="utf-8")
    assert _is_path_allowed(str(safe_file)) is True

    # Sensitive files inside allowed root must be blocked
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=123", encoding="utf-8")
    assert _is_path_allowed(str(env_file)) is False

    env_prod = tmp_path / ".env.production"
    env_prod.write_text("SECRET=123", encoding="utf-8")
    assert _is_path_allowed(str(env_prod)) is False

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    key_file = ssh_dir / "id_rsa"
    key_file.write_text("PRIVATE KEY", encoding="utf-8")
    assert _is_path_allowed(str(key_file)) is False

    # Outside allowed root must be blocked
    assert _is_path_allowed("/etc/hosts") is False

    # Verify execute_internal_tool integration
    out_ok = execute_internal_tool("read_file", {"path": str(safe_file)})
    assert out_ok == "hello safe world"

    out_blocked = execute_internal_tool("read_file", {"path": str(env_file)})
    assert out_blocked.startswith("BLOCKED:")


# ---------------------------------------------------------------------------
# INV-5: Unified Tool-Call & Reasoning Parser + WaterfallRouter
# ---------------------------------------------------------------------------


def test_parse_tool_calls_across_all_schemas():
    # 1. Single tool call with <think> block and markdown fence
    raw1 = """<think>I should run git status.</think>
```json
{"name": "run_bash_command", "arguments": {"command": "git status"}}
```"""
    calls1, msg1 = parse_tool_calls(raw1)
    assert msg1 is None
    assert calls1 == [{"name": "run_bash_command", "arguments": {"command": "git status"}}]

    # 2. Multi-tool wrapper {"tool_calls": [...]}
    raw2 = json.dumps(
        {
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": "/home/krusch/README.md"}},
                {"name": "final_answer", "arguments": {"answer": "Done"}},
            ]
        }
    )
    calls2, msg2 = parse_tool_calls(raw2)
    assert msg2 is None
    assert len(calls2) == 2
    assert calls2[0]["name"] == "read_file"
    assert calls2[1]["arguments"] == {"answer": "Done"}

    # 3. OpenAI nested function format with stringified arguments
    raw3 = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "run_bash_command",
                    "arguments": '{"command": "pwd"}',
                },
            }
        ]
    )
    calls3, _ = parse_tool_calls(raw3)
    assert calls3 == [{"name": "run_bash_command", "arguments": {"command": "pwd"}}]

    # 4. Conversational / refusal message fallback
    raw4 = '{"message": "I cannot fulfill that request without a target path."}'
    calls4, msg4 = parse_tool_calls(raw4)
    assert calls4 == []
    assert msg4 == "I cannot fulfill that request without a target path."


def test_waterfall_router_failover_and_env_key(monkeypatch):
    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-secret-999")
    assert resolve_api_key("ENV:TEST_CLOUD_KEY") == "sk-secret-999"

    routes = [
        {"name": "Primary Broken", "api_url": "http://primary.invalid/v1/chat/completions", "model": "m1"},
        {"name": "Secondary Working", "api_url": "http://secondary.local/v1/chat/completions", "model": "m2", "api_key": "ENV:TEST_CLOUD_KEY"},
    ]

    async def _run():
        mock_client = AsyncMock()
        fail_resp = MagicMock(status_code=503, text="Service Unavailable")
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"choices": [{"message": {"content": "Fallback succeeded"}}]}
        mock_client.post.side_effect = [fail_resp, ok_resp]

        router = WaterfallRouter(routes, client=mock_client)
        result = await router.route_chat("sys", "user")
        assert result == "Fallback succeeded"
        assert mock_client.post.call_count == 2

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# INV-3 & INV-6: KruschEngine & FastAPI Gateway Verification
# ---------------------------------------------------------------------------


def test_krusch_engine_dual_engine_and_no_disk_pollution(monkeypatch, tmp_path):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    engine = KruschEngine({"llm": {"model": "qwen2.5-coder:7b"}})

    async def _run():
        with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
            # 1. Tool-call fast path
            mock_chat.return_value = '{"name": "run_bash_command", "arguments": {"command": "pwd"}}'
            bp, resp = await engine.generate("Check pwd", is_tool_call=True)
            assert bp == "FAST_PATH_TOOL"
            assert json.loads(resp)["name"] == "run_bash_command"
            assert not os.path.exists("debug_tool_prompt.txt")
            assert not os.path.exists("debug_tool_response.txt")

            # 2. Code execution Dual-Engine (Layer 1 Thinker + Layer 2 Implementer)
            mock_chat.side_effect = [
                '<holodata>{"cognitive_scratchpad": "add a and b", "task_name": "add", "strict_problem_constraints": "none", "components": []}</holodata>',
                "```python\ndef add(a, b):\n    return a + b\n```",
            ]
            bp2, code2 = await engine.generate(
                "Write add(a, b)", is_code_exec=True, exact_signature="def add(a, b):"
            )
            assert "cognitive_scratchpad" in bp2
            assert code2 == "def add(a, b):\n    return a + b"

    asyncio.run(_run())


def test_api_gateway_endpoints(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    # 1. Health check
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["version"] == "0.4.0"

    # 2. Fast-path NLP completion
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "The 5-GPU fleet is operational."
        r_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Status report"}],
            },
        )
        assert r_chat.status_code == 200
        body = r_chat.json()
        assert body["choices"][0]["message"]["content"] == "The 5-GPU fleet is operational."
        assert body["choices"][0]["finish_reason"] == "stop"

    # 3. Tool-calling completion with multi-tool wrapper
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = json.dumps(
            {
                "tool_calls": [
                    {"name": "get_node_status", "arguments": {"node": "kruschserv"}}
                ]
            }
        )
        r_tool = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Check kruschserv"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_node_status", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        assert r_tool.status_code == 200
        t_body = r_tool.json()
        assert t_body["choices"][0]["finish_reason"] == "tool_calls"
        tc = t_body["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_node_status"
        assert json.loads(tc["function"]["arguments"]) == {"node": "kruschserv"}


def test_unified_execution_mode_and_mcp_server_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    monkeypatch.setenv("KRUSCH_ALLOWED_ROOTS", str(tmp_path))

    sample = tmp_path / "info.txt"
    sample.write_text("fleet=5gpu", encoding="utf-8")

    # 1. Unified execution mode on KruschEngine
    engine = KruschEngine({"llm": {"model": "qwen2.5-coder:7b", "unified_execution": True}})

    async def _run_unified():
        with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = (
                '<holodata>{"cognitive_scratchpad": "plan"}</holodata>\n'
                "```python\ndef solve():\n    return 42\n```"
            )
            bp, ans = await engine.generate("Solve problem", is_code_exec=True)
            assert bp == '{"cognitive_scratchpad": "plan"}'
            assert ans == "def solve():\n    return 42"

    asyncio.run(_run_unified())

    # 2. MCP Server krusch_execute_task multi-step loop
    from src.mcp_server import krusch_execute_task

    async def _run_mcp():
        with patch("src.mcp_server.krusch_engine.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [
                (
                    "FAST_PATH_TOOL",
                    json.dumps(
                        {
                            "tool_calls": [
                                {"name": "read_file", "arguments": {"path": str(sample)}}
                            ]
                        }
                    ),
                ),
                (
                    "FAST_PATH_TOOL",
                    json.dumps(
                        {
                            "name": "final_answer",
                            "arguments": {"answer": "Read fleet=5gpu successfully"},
                        }
                    ),
                ),
            ]
            result = await krusch_execute_task("Read info.txt and report")
            assert result == "Read fleet=5gpu successfully"
            assert mock_gen.call_count == 2

    asyncio.run(_run_mcp())


def test_api_gateway_autonomous_loop_and_ollama_passthroughs(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    # 1. Autonomous loop via force_autonomous=True
    with patch("src.api_gateway._engine.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = [
            (
                "FAST_PATH_TOOL",
                '{"name": "run_bash_command", "arguments": {"command": "echo hello"}}',
            ),
            (
                "FAST_PATH_TOOL",
                '{"name": "final_answer", "arguments": {"answer": "Echoed hello"}}',
            ),
        ]
        r_auto = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Run echo hello"}],
                "force_autonomous": True,
            },
        )
        assert r_auto.status_code == 200
        assert r_auto.json()["choices"][0]["message"]["content"] == "Echoed hello"

    # 2. Ollama pass-through endpoints (/api/tags and /v1/embeddings)
    mock_http = AsyncMock()
    tags_resp = MagicMock(status_code=200)
    tags_resp.json.return_value = {"models": [{"name": "qwen2.5-coder:7b"}]}
    mock_http.get.return_value = tags_resp

    embed_resp = MagicMock(status_code=200)
    embed_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_http.post.return_value = embed_resp

    with patch("src.api_gateway.get_http_client", return_value=mock_http):
        r_tags = client.get("/api/tags")
        assert r_tags.status_code == 200
        assert r_tags.json()["models"][0]["name"] == "qwen2.5-coder:7b"

        r_emb = client.post("/v1/embeddings", json={"model": "bge-large:latest", "input": "test"})
        assert r_emb.status_code == 200
        assert r_emb.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# INV-7: OpenAI SSE Streaming (stream=True) & Usage Token Compliance
# ---------------------------------------------------------------------------


def test_sse_streaming_and_usage_metrics(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    # 1. Non-streaming response includes usage dict
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Streaming and usage verified."
        r_sync = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Verify usage dict"}],
                "stream": False,
            },
        )
        assert r_sync.status_code == 200
        sync_body = r_sync.json()
        assert "usage" in sync_body
        assert sync_body["usage"]["prompt_tokens"] >= 1
        assert sync_body["usage"]["completion_tokens"] >= 1
        assert (
            sync_body["usage"]["total_tokens"]
            == sync_body["usage"]["prompt_tokens"] + sync_body["usage"]["completion_tokens"]
        )

    # 2. SSE streaming response (stream=True) emits chat.completion.chunk events + [DONE]
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "SSE chunk content"
        r_sse = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Stream to me"}],
                "stream": True,
            },
        )
        assert r_sse.status_code == 200
        assert "text/event-stream" in r_sse.headers.get("content-type", "")
        lines = [line for line in r_sse.text.splitlines() if line.startswith("data: ")]
        assert len(lines) == 3
        assert lines[-1] == "data: [DONE]"

        first_chunk = json.loads(lines[0][len("data: ") :])
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["content"] == "SSE chunk content"
        assert first_chunk["choices"][0]["finish_reason"] is None

        second_chunk = json.loads(lines[1][len("data: ") :])
        assert second_chunk["choices"][0]["finish_reason"] == "stop"
        assert second_chunk["usage"]["total_tokens"] >= 2

    # 3. SSE streaming response for tool_calls
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"name": "get_weather", "arguments": {"city": "LA"}}'
        r_sse_tool = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Weather in LA"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        assert r_sse_tool.status_code == 200
        t_lines = [line for line in r_sse_tool.text.splitlines() if line.startswith("data: ")]
        assert t_lines[-1] == "data: [DONE]"
        t_first = json.loads(t_lines[0][len("data: ") :])
        assert t_first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        t_second = json.loads(t_lines[1][len("data: ") :])
        assert t_second["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# INV-8: Weklund Multi-Tool Wrapper Parsing & Output Canonicalization
# ---------------------------------------------------------------------------


def test_weklund_multi_tool_wrapper_and_canonicalization(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    from benchmarks.mtb.quality_benchmarks.tool_call_parser import (
        parse_tool_calls as mtb_parse_tool_calls,
    )

    # 1. Weklund MTB parser unwraps {"tool_calls": [...]} directly
    wrapper_json = json.dumps(
        {
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Seattle"}},
                {"name": "get_time", "arguments": {"timezone": "PST"}},
            ]
        }
    )
    parsed_mtb = mtb_parse_tool_calls(wrapper_json)
    assert parsed_mtb is not None
    assert len(parsed_mtb) == 2
    assert parsed_mtb[0].name == "get_weather"
    assert parsed_mtb[1].name == "get_time"

    # 2. KruschEngine.generate(is_tool_call=True) canonicalizes multi-tool wrapper into JSON array
    engine = KruschEngine({"llm": {"model": "qwen2.5-coder:7b"}})

    async def _run():
        with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = wrapper_json
            bp, out_text = await engine.generate("Weather and time", is_tool_call=True)
            assert bp == "FAST_PATH_TOOL"
            decoded = json.loads(out_text)
            assert isinstance(decoded, list)
            assert len(decoded) == 2
            assert decoded[0]["name"] == "get_weather"
            assert decoded[1]["name"] == "get_time"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# INV-9: Dual-Port Lifecycle Scripts & Tiered Benchmark Runner Parity
# ---------------------------------------------------------------------------


def test_lifecycle_scripts_and_humaneval_imports():
    import importlib
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Verify bash -n syntax and dual-port PID parity across lifecycle scripts
    for script in ("start_proxies.sh", "stop_proxies.sh", "benchmark.sh"):
        script_path = os.path.join(repo_root, script)
        res = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Syntax error in {script}: {res.stderr}"

    start_txt = open(os.path.join(repo_root, "start_proxies.sh"), encoding="utf-8").read()
    stop_txt = open(os.path.join(repo_root, "stop_proxies.sh"), encoding="utf-8").read()
    bench_txt = open(os.path.join(repo_root, "benchmark.sh"), encoding="utf-8").read()

    assert "gateway_5440.pid" in start_txt and "gateway_5443.pid" in start_txt
    assert 'stop_proxy "gateway_5440"' in stop_txt and 'stop_proxy "gateway_5443"' in stop_txt
    assert "krusch-brain" not in bench_txt and "qwen2.5-coder:7b" in bench_txt

    # 2. Verify all benchmark runner modules import without ModuleNotFoundError
    for mod_name in (
        "benchmarks.weklund_tools",
        "benchmarks.run_weklund_all_models",
        "benchmarks.run_distributed_humaneval",
        "benchmarks.run_distributed_ifeval",
        "benchmarks.run_humaneval",
        "benchmarks.run_humaneval_gemma4_tier",
        "benchmarks.run_humaneval_9b_tier",
        "benchmarks.run_humaneval_14b_tier",
    ):
        mod = importlib.import_module(mod_name)
        assert mod is not None


