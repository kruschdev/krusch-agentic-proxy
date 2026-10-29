# Specification: Krusch Agentic Proxy v0.4.0 Hardening & Fleet Optimization

- **Project**: `krusch-agentic-proxy`
- **Version**: `0.4.0`
- **Date**: `2026-10-18`
- **Status**: `FROZEN`
- **Protocol**: L0 Specification-First Refactor (`/spec-first-refactor`, Dual-Zero-Finding Verification Gate)

---

## 1. Core Invariants

1. **INV-1 (Multi-Family 5-GPU Fleet Topology & VRAM Residency Safety)**:
   - `get_ollama_url_for_model(model: str, base_url: str) -> str` MUST route models to the authoritative 2026-10-18 5-GPU Fleet endpoints (configurable via `KRUSCHDEV_OLLAMA_URL`, `KRUSCHSERV_OLLAMA_URL`, `KRUSCH_OLLAMA_URL`, and `SPECTRALQUANT_URL`):
     - Heavy MoE / Reasoners (`qwen3-coder-30b-moe`, `30b`, `31b`, `32b`, `deepseek-r1:14b`, `r1:14b`) -> `kruschdev` (`http://10.10.10.1:11434/v1/chat/completions`).
     - Non-Qwen Executor / Generalist Pool (`gemma4`, `hermes3`) -> `kruschserv` (`http://10.10.10.2:11434/v1/chat/completions`).
     - Local 7B Coder & Embeddings (`qwen2.5-coder:7b`, `bge-large`) -> `base_url` (`http://127.0.0.1:11434/v1/chat/completions` / `KRUSCH_OLLAMA_URL`).
     - SpectralQuant (`spectralquant`) -> `http://127.0.0.1:11437/v1/chat/completions`.
   - `src/client.py` MUST NOT inject a hardcoded `"options": {"num_ctx": 16384}` into requests unless `num_ctx` or `options` is explicitly configured, preventing VRAM eviction of resident models pinned with `keep_alive: -1`.
   - Default model across all configs and schemas MUST be `qwen2.5-coder:7b`.

2. **INV-2 (Secret Hygiene & Non-Blocking Bounded RAG Retrieval)**:
   - `retrieve_homelab_context(query: str, project: str = "krusch-agentic-proxy") -> str` MUST NOT contain hardcoded database passwords. It MUST read connection settings from `KRUSCH_RAG_DSN` or `DATABASE_URL` (or `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`), returning `""` immediately if `KRUSCH_DISABLE_RAG == "1"` or no DSN/password is configured.
   - RAG results MUST be cached in a bounded in-memory TTL cache (`_RAG_CACHE`, max 128 entries, 60s TTL) and executed via `await asyncio.to_thread(...)` inside `KruschEngine.generate()` so synchronous DB queries never block the event loop.

3. **INV-3 (No Disk Pollution or Stdout Corruption)**:
   - `KruschEngine.generate()` MUST NOT write `debug_tool_prompt.txt` or `debug_tool_response.txt` to disk during normal operation.
   - `KruschEngine.generate()` MUST NOT use `print()` to `stdout` (which corrupts MCP `stdio` JSON-RPC framing); all diagnostics MUST use `logger.debug()`.

4. **INV-4 (Hardened Tool Sandbox & Path Isolation)**:
   - `_is_command_allowed(command: str) -> bool` MUST:
     - Reject empty strings, control characters (`\n`, `\r`, `\x00`), and shell chaining/redirection metacharacters (`;`, `&&`, `||`, `|`, `` ` ``, `$(`, `>`, `<`, `&`).
     - Tokenize via `shlex.split(command)` and reject malformed shell syntax.
     - Match exact binary names (`ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `echo`, `pwd`, `whoami`, `date`, `uname`, `curl`) or exact two-token subcommands (`git status`, `git log`, `git diff`, `git show`, `git branch`, `docker ps`, `docker logs`, `docker inspect`).
     - Reject arbitrary code execution interpreters (`python`, `python3`, `node`, `pip`, `env`) and dangerous `find` flags (`-exec`, `-execdir`, `-ok`, `-delete`, `-fprint`).
   - `_is_path_allowed(path: str) -> bool` MUST resolve symlinks via `os.path.realpath`, verify containment within `_get_allowed_roots()`, and block sensitive paths/filenames (`.ssh`, `.gnupg`, `.aws`, `.kube`, `.docker`, `.env`, `.git/config`, `id_rsa`, `id_ed25519`, `shadow`, `passwd`).
   - `execute_internal_tool` calls inside async loops (`src/api_gateway.py`, `src/mcp_server.py`) MUST run via `await asyncio.to_thread(execute_internal_tool, ...)`.

5. **INV-5 (Unified Reasoning/Tool Parser & Shared HTTP Connection Pool)**:
   - `strip_thinking_and_fences(text: str) -> str` and `parse_tool_calls(response_text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]` in `src/core.py` MUST strip `<think>...</think>` tags and markdown fences, normalize tool calls from list/dict/`tool_calls`/`calls`/OpenAI `function` formats into `[{"name": str, "arguments": dict}]`, and extract conversational `"message"` fallbacks.
   - `src/client.py`, `src/router.py`, and `src/api_gateway.py` MUST share pooled `httpx.AsyncClient` instances closed cleanly during application shutdown, and Ollama pass-through routes in `src/api_gateway.py` MUST default to port `11434` (fixing the `11435` typo).

6. **INV-6 (Verification & Documentation Parity)**:
   - All unit and integration tests in `tests/test_proxy.py` MUST pass (`pytest`).
   - `AGENTS.md`, `GEMINI_INFLIGHT.md`, `README.md`, `config.example.json`, and `config.json` MUST reflect `v0.4.0` and the 2026-10-18 fleet topology.

---

## 2. Target Files

1. `src/core.py` — Fleet model router, `<think>`/fence stripper, `parse_tool_calls`, non-blocking cached RAG retriever, clean logging.
2. `src/client.py` — Shared `httpx.AsyncClient` pool, `<think>` tag stripping, conditional `options`/`num_ctx`.
3. `src/router.py` — Shared `httpx.AsyncClient` support, consistent timeout & header handling.
4. `src/tools.py` — Tokenized command allowlist, metacharacter & flag blocking, sensitive path denylist.
5. `src/api_gateway.py` — Unified `parse_tool_calls`, non-blocking `asyncio.to_thread(execute_internal_tool)`, shared HTTP client lifecycle, fixed `11434` Ollama forwarding endpoints.
6. `src/mcp_server.py` — Unified `parse_tool_calls` (supporting multi-tool and wrapped `tool_calls` responses), non-blocking tool execution, stderr-safe logging.
7. `src/models.py` — Updated model defaults and optional OpenAI compatibility fields (`tool_choice`, `response_format`).
8. `config.example.json` & `config.json` — Updated to `qwen2.5-coder:7b` and 5-GPU fleet comments.
9. `pyproject.toml`, `.gitignore`, `AGENTS.md`, `GEMINI_INFLIGHT.md`, `README.md` — Version `0.4.0`, pytest config, governance rules.
10. `tests/test_proxy.py` — Comprehensive unit & API gateway test suite.

---

## 3. Phase 2 Enhancements (v0.4.0 Extended Hardening)

7. **INV-7 (OpenAI SSE Streaming & Token Usage Compliance)**:
   - `_build_response` in `src/api_gateway.py` MUST populate `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) using deterministic token estimation so OpenAI SDKs, LangChain, and evaluation harnesses never fail on missing `usage` fields.
   - When `validated.stream == True` on `/v1/chat/completions`, `src/api_gateway.py` MUST return an OpenAI-compliant `StreamingResponse` (`media_type="text/event-stream"`) emitting `chat.completion.chunk` payloads (`delta.content` or `delta.tool_calls`) terminated by `data: [DONE]\n\n`.

8. **INV-8 (Weklund Multi-Tool Benchmark & 5-GPU Distributed Harness Parity)**:
   - `KruschEngine.generate(is_tool_call=True)` MUST canonicalize wrapped `{"tool_calls": [...]}` responses into a standard JSON array `[{"name": ..., "arguments": {...}}]` when tool calls are present, and unwrap `{"message": "..."}` into plain text when no tool matches, ensuring 100% compatibility with both OpenAI tool callers and Weklund's `TS_NONE_SELECTION` / Multi-Tool evaluators.
   - `benchmarks/mtb/quality_benchmarks/tool_call_parser.py` (`_extract_tool_calls_from_parsed`) MUST also unwrap `{"tool_calls": [...]}` and `{"calls": [...]}` dictionaries.
   - `benchmarks/weklund_tools.py` and `benchmarks/run_weklund_all_models.py` MUST configure `sys.path` with both `PROXY_DIR` and `SCRIPT_DIR` (`benchmarks/`) so they execute cleanly from the repository root (`./benchmark.sh`).
   - `benchmarks/run_distributed_humaneval.py`, `benchmarks/run_distributed_ifeval.py`, and `docs/openclaw_integration.md` MUST reflect the 2026-10-18 5-GPU Fleet Topology (`krusch`, `kruschserv`, `kruschdev`) and `/home/krusch/homelab/projects/krusch-agentic-proxy`.

9. **INV-9 (Dual-Port Lifecycle Scripts & Tiered Benchmark Runner Parity)**:
   - `start_proxies.sh` and `stop_proxies.sh` MUST manage both the Active Universal Gateway (`:5440`, `gateway_5440.pid`) and the Passive OpenClaw Interceptor (`:5443`, `PASSIVE_MODE=1`, `gateway_5443.pid` / `nvidia.pid`).
   - `benchmark.sh` MUST default to `MODEL="${MODEL:-qwen2.5-coder:7b}"` rather than the obsolete `krusch-brain` model identifier, and `benchmarks/run_humaneval*.py` (`run_humaneval.py`, `run_humaneval_gemma4_tier.py`, `run_humaneval_9b_tier.py`, `run_humaneval_14b_tier.py`) MUST include `SCRIPT_DIR` on `sys.path` and route to the 2026-10-18 5-GPU Fleet endpoints.


---

## 4. Phase 3 Enhancements (v0.5.0 — Production Agentic Systems Parity, FROZEN 2026-10-28)

10. **`INV-10` (`src/orchestrator.py` — 100+ Concurrent Async Agent Orchestrator)**:
    - `AsyncAgentOrchestrator` MUST execute `100+` concurrent agent tasks on a single `asyncio` event loop using `asyncio.Semaphore(max_concurrency)`, per-task `asyncio.wait_for` timeouts, exponential-jitter retry, non-blocking tool execution (`asyncio.iscoroutinefunction` or `asyncio.to_thread`), and per-host circuit breakers (`CLOSED` / `OPEN` / `HALF_OPEN`).
11. **`INV-11` (`src/budget_assembler.py` — Pre-Flight Dynamic Context Budget Assembler)**:
    - `DynamicContextBudgetAssembler` MUST strictly allocate `max_context_tokens - output_reserve_tokens` across `system_prompt`, `tool_schemas`, `retrieved_docs`, and `conversation_history`, deterministically evicting lowest-salience docs and lowest-relevance non-pinned tools on overflow and emitting a `ContextBudgetLedger`.
12. **`INV-12` (`src/pydantic_tool_router.py` — Strict Pydantic v2 Tool Router & Validation Retry)**:
    - `StrictPydanticToolRouter` MUST validate every tool argument payload against a strict Pydantic v2 `BaseModel` (`extra="forbid"`), formatting `ValidationError` / `JSONDecodeError` traces into structured repair prompts and retrying up to `max_retries=3`.
13. **`INV-13` (`src/typed_swarm.py` — Strictly Typed Multi-Agent Swarm State Machine)**:
    - `TypedMultiAgentSwarm` (`RESEARCH` -> `CODE` -> `REVIEW` -> `APPROVED` | `REJECTED` with bounded `REVIEW -> CODE` revisions) MUST pass frozen Pydantic v2 models (`ResearchBrief`, `CodeArtifact`, `ReviewVerdict`, `SwarmState`) and raise `TypeError` if raw `str` handoffs are attempted.
14. **`INV-14` (`src/shadow_comparator.py` & `src/cost_killswitch.py` — 5% Shadow Comparator & `$0.50` Cost Kill-Switch)**:
    - `ShadowTrajectoryComparator` MUST mirror `sample_rate` (default `0.05`) requests via fire-and-forget `asyncio.create_task` (`0 ms` user blocking), persisting tool-call Levenshtein/Jaccard diff & cost/latency deltas to SQLite WAL (`shadow_comparisons`).
    - `CostKillSwitch` MUST enforce `hard_limit_usd = 0.50`, automatically downgrading paid frontier calls to `$0.00` local Ollama (`qwen2.5-coder:7b` / `qwen3-coder-30b-moe:latest`) when projected spend breaches the ceiling and halting (`HALTED_BUDGET_EXCEEDED`) if paid spend reaches `$0.50` or `loop_depth > max_loop_depth`.
