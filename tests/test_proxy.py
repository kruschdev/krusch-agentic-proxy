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
from src.router import (
    ConfigurationError,
    RouteConfigurationError,
    WaterfallRouter,
    resolve_api_key,
    validate_api_key,
)
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

    # Sensitive files inside allowed root must be blocked (generate dynamic test token to avoid hardcoded secret literals)
    import secrets
    test_secret = secrets.token_hex(16)
    env_file = tmp_path / ".env"
    env_file.write_text(f"SECRET={test_secret}", encoding="utf-8")
    assert _is_path_allowed(str(env_file)) is False

    env_prod = tmp_path / ".env.production"
    env_prod.write_text(f"SECRET={test_secret}", encoding="utf-8")
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


def test_waterfall_router_configuration_errors(monkeypatch):
    monkeypatch.delenv("MISSING_CLOUD_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        validate_api_key("ENV:MISSING_CLOUD_KEY")
    with pytest.raises(ConfigurationError):
        validate_api_key("")

    async def _run():
        empty_router = WaterfallRouter([])
        with pytest.raises(ConfigurationError):
            await empty_router.route_chat("sys", "user")

        missing_url_router = WaterfallRouter([{"name": "BrokenRoute"}], strict_routes=True)
        with pytest.raises(RouteConfigurationError):
            await missing_url_router.route_chat("sys", "user")

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


# ---------------------------------------------------------------------------
# PROXY-01 through PROXY-11: Security, Reliability & Protocol Audit Remediations
# ---------------------------------------------------------------------------


def test_proxy_01_bash_command_path_enforcement_and_curl_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("KRUSCH_ALLOWED_ROOTS", str(tmp_path))

    safe_file = tmp_path / "safe.log"
    safe_file.write_text("ok", encoding="utf-8")

    # Allowed paths inside KRUSCH_ALLOWED_ROOTS
    assert _is_command_allowed(f"cat {safe_file}") is True
    assert _is_command_allowed(f"head -n 5 {safe_file}") is True
    assert _is_command_allowed(f"grep ok {safe_file}") is True
    assert _is_command_allowed(f"find {tmp_path} -maxdepth 1 -name '*.log'") is True

    # Blocked sensitive files and paths outside allowed roots
    assert _is_command_allowed("cat .env") is False
    assert _is_command_allowed(f"cat {tmp_path}/.env") is False
    assert _is_command_allowed(f"cat {tmp_path}/.env.local") is False
    assert _is_command_allowed("cat ~/.ssh/id_rsa") is False
    assert _is_command_allowed("cat /etc/passwd") is False
    assert _is_command_allowed("head -n 10 /etc/shadow") is False
    assert _is_command_allowed("grep root /etc/passwd") is False
    assert _is_command_allowed("find /etc -name '*.conf'") is False
    assert _is_command_allowed("ls -la /var/log") is False

    # curl removed from ALLOWED_SINGLE_COMMANDS
    assert _is_command_allowed("curl http://example.com") is False
    assert _is_command_allowed("curl file:///etc/passwd") is False
    assert _is_command_allowed("curl http://127.0.0.1:5440/health") is False


def test_proxy_02_passive_mode_blocks_autonomous_and_bearer_auth(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    # 1. When PASSIVE_MODE=1, force_autonomous=True is ignored and host tools never run
    monkeypatch.setenv("PASSIVE_MODE", "1")
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat, patch(
        "src.api_gateway.execute_internal_tool"
    ) as mock_exec:
        mock_chat.return_value = "Passive NLP response only."
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Run uname -a"}],
                "force_autonomous": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "Passive NLP response only."
        assert mock_exec.call_count == 0
    monkeypatch.delenv("PASSIVE_MODE", raising=False)

    # 2. Optional KRUSCH_PROXY_API_KEY Bearer token middleware
    monkeypatch.setenv("KRUSCH_PROXY_API_KEY", "krusch-test-secret-token")
    # /health remains public
    assert client.get("/health").status_code == 200
    # Unauthenticated request to /v1/chat/completions is rejected with 401
    r_unauth = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder:7b", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r_unauth.status_code == 401

    # Authenticated request with valid Bearer token succeeds
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Authenticated response"
        r_auth = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer krusch-test-secret-token"},
            json={"model": "qwen2.5-coder:7b", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r_auth.status_code == 200
        assert r_auth.json()["choices"][0]["message"]["content"] == "Authenticated response"
    monkeypatch.delenv("KRUSCH_PROXY_API_KEY", raising=False)


def test_proxy_03_and_10_multiturn_tool_history_multipart_content_and_tool_choice_none(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    captured_prompts = []

    async def fake_generate(prompt, **kwargs):
        captured_prompts.append((prompt, kwargs))
        return "FAST_PATH_TOOL", '{"message": "The weather in Seattle is 72F and sunny."}'

    # 1. Multi-turn tool callback preserves SYSTEM, DEVELOPER, and trailing TOOL message
    with patch("src.api_gateway._engine.generate", side_effect=fake_generate):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [
                    {"role": "system", "content": "Strict system rule."},
                    {"role": "developer", "content": "Developer formatting rule."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the weather"},
                            {"type": "text", "text": "in Seattle?"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Seattle"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_weather",
                        "tool_call_id": "call_1",
                        "content": "72F and sunny",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        assert r.status_code == 200
        sent_prompt, sent_kwargs = captured_prompts[0]
        assert sent_kwargs["is_tool_call"] is True
        assert "SYSTEM: Strict system rule." in sent_prompt
        assert "DEVELOPER: Developer formatting rule." in sent_prompt
        assert "TOOL RESULT (get_weather): 72F and sunny" in sent_prompt
        assert "What is the weather\nin Seattle?" in sent_prompt

    # 2. tool_choice == "none" bypasses Tool-Calling Mode
    with patch("src.core.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Direct answer without tools."
        r_none = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Just chat with me"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": "none",
            },
        )
        assert r_none.status_code == 200
        assert r_none.json()["choices"][0]["message"]["content"] == "Direct answer without tools."
        assert r_none.json()["choices"][0]["finish_reason"] == "stop"


def test_proxy_04_smart_tool_token_limiter():
    from src.core import format_tools_for_prompt, select_relevant_tools

    many_tools = [
        {
            "type": "function",
            "function": {
                "name": f"unrelated_tool_{i}",
                "description": "A" * 250,
                "parameters": {
                    "type": "object",
                    "title": "UnneededTitle",
                    "additionalProperties": False,
                    "properties": {"x": {"type": "string", "description": "B" * 200}},
                },
            },
        }
        for i in range(12)
    ]
    many_tools.append(
        {
            "type": "function",
            "function": {
                "name": "query_postgres_database",
                "description": "Execute a read-only SQL query against the Postgres database",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            },
        }
    )

    selected = select_relevant_tools(
        many_tools,
        objective="Run a SQL query on the postgres database",
        messages=[{"role": "tool", "name": "unrelated_tool_3", "content": "prev"}],
        max_tools=8,
    )
    assert len(selected) == 8
    selected_names = {t["function"]["name"] for t in selected}
    assert "query_postgres_database" in selected_names
    assert "unrelated_tool_3" in selected_names

    compact_str = format_tools_for_prompt(
        many_tools,
        objective="Run a SQL query on the postgres database",
        max_tools=8,
    )
    # Compact JSON has no newline indentation and strips redundant title/additionalProperties
    assert "\n" not in compact_str
    assert "UnneededTitle" not in compact_str
    assert "query_postgres_database" in compact_str


def test_proxy_05_duplicate_call_guard_output_cap_and_malformed_json(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    client = TestClient(app)

    # 1. Duplicate tool call guard in API gateway _autonomous_loop()
    with patch("src.api_gateway._engine.generate", new_callable=AsyncMock) as mock_gen, patch(
        "src.api_gateway.execute_internal_tool", return_value="STDOUT:\nhi\nSTDERR:\n"
    ) as mock_exec:
        mock_gen.side_effect = [
            (
                "FAST_PATH_TOOL",
                '{"name": "run_bash_command", "arguments": {"command": "echo hi"}}',
            ),
            (
                "FAST_PATH_TOOL",
                '{"name": "run_bash_command", "arguments": {"command": "echo hi"}}',
            ),
        ]
        r_dup = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Loop test"}],
                "force_autonomous": True,
            },
        )
        assert r_dup.status_code == 200
        assert "Duplicate tool call detected" in r_dup.json()["choices"][0]["message"]["content"]
        assert mock_exec.call_count == 1

    # 2. Malformed JSON starting with '{' in _autonomous_loop() is NOT returned as final answer
    with patch("src.api_gateway._engine.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = [
            ("FAST_PATH_TOOL", '{"name": "run_bash_command", "arguments": {"command": "pwd"'),
            ("FAST_PATH_TOOL", '{"name": "final_answer", "arguments": {"answer": "Recovered"}}'),
        ]
        r_mal = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [{"role": "user", "content": "Recover from broken JSON"}],
                "force_autonomous": True,
            },
        )
        assert r_mal.status_code == 200
        assert r_mal.json()["choices"][0]["message"]["content"] == "Recovered"

    # 3. run_bash_command output capped at 16,000 chars
    from src.tools import MAX_BASH_OUTPUT_CHARS

    fake_proc = MagicMock(stdout="X" * 25_000, stderr="")
    with patch("src.tools.subprocess.run", return_value=fake_proc):
        capped_out = execute_internal_tool("run_bash_command", {"command": "pwd"})
        assert len(capped_out) == MAX_BASH_OUTPUT_CHARS
        assert "TRUNCATED" in capped_out


def test_proxy_06_07_08_09_11_locks_failover_502_truncated_think_and_max_loops(monkeypatch):
    monkeypatch.setenv("KRUSCH_DISABLE_RAG", "1")
    from src.core import _RAG_CACHE_LOCK
    from src.mcp_server import krusch_execute_task

    # 1. _RAG_CACHE_LOCK is a threading.Lock
    assert hasattr(_RAG_CACHE_LOCK, "acquire") and hasattr(_RAG_CACHE_LOCK, "release")

    # 2. Truncated <think> block without closing </think> is stripped
    assert (
        strip_thinking_and_fences("<think>done</think>Keep this<think>unclosed reasoning...")
        == "Keep this"
    )

    # 3. Fleet node failover in src/client.py:chat() from remote node to local fallback
    async def _run_failover():
        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": "<think>unclosed..."}}]
        }
        ok_resp2 = MagicMock()
        ok_resp2.raise_for_status = MagicMock()
        ok_resp2.json.return_value = {
            "choices": [{"message": {"content": "Local fallback succeeded"}}]
        }

        with patch("src.client.get_http_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post.side_effect = [
                httpx.ConnectError("kruschdev unreachable"),
                ok_resp2,
            ]
            mock_get_client.return_value = mock_http

            res = await chat(
                "sys",
                "user",
                {
                    "api_url": "http://192.168.1.240:11434/v1/chat/completions",
                    "model": "qwen3-coder-30b-moe:latest",
                },
            )
            assert res == "Local fallback succeeded"
            assert mock_http.post.call_count == 2

    asyncio.run(_run_failover())

    # 4. RuntimeError in api_gateway returns HTTP 502 instead of 500
    client = TestClient(app)
    with patch(
        "src.api_gateway._engine.generate",
        new_callable=AsyncMock,
        side_effect=RuntimeError("All upstream nodes offline"),
    ):
        r_502 = client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5-coder:7b", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r_502.status_code == 502
        assert "Upstream LLM failure" in r_502.json()["error"]

    # 5. krusch_execute_task accepts max_loops and breaks on duplicate tool calls
    async def _run_mcp_guard():
        with patch("src.mcp_server.krusch_engine.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ("FAST_PATH_TOOL", "not-json")
            out = await krusch_execute_task("Test custom max_loops", max_loops=2)
            assert "exceeded maximum iterations" in out
            assert mock_gen.call_count == 2

    asyncio.run(_run_mcp_guard())


def test_bug_01_and_bug_02_config_error_handling_and_singleton_lock(monkeypatch, tmp_path):
    import src.api_gateway as gw

    # BUG-01: Missing config logs error and falls back to DEFAULT_MODEL (or raises in strict mode)
    missing_cfg = tmp_path / "nonexistent_config.json"
    monkeypatch.setenv("KRUSCH_PROXY_CONFIG", str(missing_cfg))
    cfg = gw.load_config(strict=False)
    assert cfg["llm"]["model"] == gw.DEFAULT_MODEL

    with pytest.raises(FileNotFoundError):
        gw.load_config(strict=True)

    bad_cfg = tmp_path / "corrupt.json"
    bad_cfg.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setenv("KRUSCH_PROXY_CONFIG", str(bad_cfg))
    with pytest.raises(json.JSONDecodeError):
        gw.load_config(strict=True)

    # BUG-01 (Lap 3): Valid JSON missing required 'llm' block fails ProxyConfig validation
    from pydantic import ValidationError

    invalid_schema_cfg = tmp_path / "invalid_schema.json"
    invalid_schema_cfg.write_text('{"waterfall_routes": []}', encoding="utf-8")
    monkeypatch.setenv("KRUSCH_PROXY_CONFIG", str(invalid_schema_cfg))
    fallback_cfg = gw.load_config(strict=False)
    assert fallback_cfg["llm"]["model"] == gw.DEFAULT_MODEL
    with pytest.raises(ValidationError):
        gw.load_config(strict=True)

    # BUG-02: Concurrent _ensure_initialized_async calls safely initialize singletons once
    async def _run_concurrent_init():
        await asyncio.gather(
            gw._ensure_initialized_async(),
            gw._ensure_initialized_async(),
            gw._ensure_initialized_async(),
        )
        assert gw._engine is not None
        assert gw._router is not None

    asyncio.run(_run_concurrent_init())
