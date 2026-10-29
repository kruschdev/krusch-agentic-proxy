# Krusch Agentic MCP

![Krusch Agentic Proxy Banner](docs/assets/banner.png)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastMCP](https://img.shields.io/badge/FastMCP-005571?style=flat)
![Version: 0.4.0](https://img.shields.io/badge/Version-0.4.0-green.svg)
[![CI](https://github.com/kruschdev/krusch-agentic-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/kruschdev/krusch-agentic-proxy/actions/workflows/ci.yml)
![Pytest: 35 passed](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)
[![HOL Guard Catalog](https://img.shields.io/badge/HOL_Guard-Catalog_Listed-0B0F17?logo=shield&logoColor=00F5D4)](https://hol.org/guard/extension-studio?extension=mcp.krusch-agentic-proxy)

**Krusch Agentic MCP (v0.4.0)** is a standalone Model Context Protocol (MCP) server and OpenAI-compatible API gateway that transforms local-first LLMs (like `qwen2.5-coder:7b`, `gemma4:e4b`, and `qwen3-coder-30b-moe`) into highly reliable, structured execution nodes for autonomous agents. Equipped with non-blocking cached PostgreSQL RAG memory injection, connection-pooled HTTP transport, hardened tool sandboxing, and Multi-Family 5-GPU Fleet hardware routing.

If you are running agent frameworks like **OpenClaw**, **Hermes**, or using IDEs like **Cursor**, you know the struggle: 7B models frequently hallucinate JSON tools, forget code architectures, or break under complex constraints. 

Krusch Agentic MCP solves this by exposing a single MCP Tool (`krusch_execute_task`) and an OpenAI-compatible `/v1/chat/completions` gateway that internally route tasks through a rigid **Dual-Engine & Fast-Path Topology**.

## 🤔 Why Use Krusch Agentic MCP?

1. **Zero-Hallucination Tool Calling**: Forces models into strict JSON schema conformance and `<holodata>` cognitive blueprints before writing code, with unified `<think>...</think>` reasoning stripping and multi-format tool normalization (`parse_tool_calls`).
2. **Unified API Gateway**: Acts as an OpenAI-compatible proxy (`localhost:5440` / `:5443`) and an MCP server (`stdio`) simultaneously, allowing any legacy or modern client (OpenClaw, Cursor, Chatbots) to benefit from the pipeline.
3. **VRAM Residency & Connection Pooling**: Preserves `keep_alive: -1` GPU residency by avoiding forced `num_ctx` overrides and shares persistent `httpx.AsyncClient` connection pools across requests.
4. **Hardened Execution Sandbox**: Tokenizes shell commands via `shlex`, blocks shell chaining/redirection metacharacters (`;`, `&&`, `|`, `` ` ``, `$()`), restricts dangerous flags (`find -exec`, `curl -o`), and isolates sensitive dotfiles (`.ssh`, `.env`, `.aws`, `.gnupg`).

### 🌊 Multi-Provider Waterfall Auto-Routing (v0.2.0+)

Krusch Agentic MCP features a resilient **Waterfall Auto-Routing** proxy. You can configure multiple LLM providers to ensure task continuity if your primary model fails.

- **Primary Local/VPS Models:** Route tasks to a private node or VPS-hosted model (like `gemma4:e4b` or `hermes3`) via secure API keys.
- **Cost-Effective Fallbacks:** If the primary node is unreachable or encounters an error, the proxy automatically falls back to secondary routes like OpenRouter or Gemini Flash.
- **Environment Variable Secrets:** Safely pass API keys via `ENV:MY_API_KEY` configurations instead of hardcoding them.

### 🧠 Non-Blocking Episodic RAG & 5-GPU Fleet Routing (v0.4.0)

Version 0.4.0 synchronizes the proxy with the **Multi-Family 5-GPU Local LLM Runtime Policy** and hardens memory retrieval:

- **Non-Blocking Cached PostgreSQL RAG Hook:** Intercepts domain keywords (prioritizing specific subsystems like `jean`, `dbos`, `vulkan`, `systemd`, `postgres`) and retrieves episodic lessons and nuggets via `asyncio.to_thread` with a bounded 60s TTL LRU cache (`_RAG_CACHE`) and zero hardcoded credentials (`KRUSCH_RAG_DSN` / `DATABASE_URL`).
- **Multi-Family GPU Fleet Routing:**
  - **Heavy Reasoning Pool (`KRUSCHDEV_OLLAMA_URL`, Dual GPU Node):** Routes heavy MoE and reasoning models (`qwen3-coder-30b-moe:latest`, `deepseek-r1:14b`).
  - **Non-Qwen Diversity Pool (`KRUSCHSERV_OLLAMA_URL`, Secondary GPU Node):** Routes the non-Qwen diversity pool (`gemma4:e4b`, `hermes3:latest`).
  - **Primary Local Pool (`KRUSCH_OLLAMA_URL`, default `127.0.0.1:11434`):** Routes `qwen2.5-coder:7b` for fast local coding/tool routing and `snowflake-arctic-embed2:latest` (`8K` ctx, `1024d`) for embeddings.


### 🔄 Dual Integration Modes (MCP vs API Proxy)

Is it an MCP or a Proxy? **It is both.** 

Because different frameworks have different needs, this project can be integrated in two ways:
1. **As an MCP Server (Native)**: Runs via `stdio` and provides the `krusch_execute_task` tool to MCP clients (like OpenClaw, Cursor, or Claude Desktop).
2. **As an OpenAI API Proxy (REST)**: Runs a local FastAPI server on `Port 5440`. If you have a dumb chatbot or a legacy framework that doesn't support MCP, you can simply change its API URL to point to `http://localhost:5440/v1` and the proxy will transparently route complex tool calls through the Dual-Engine pipeline.

---

## 🧠 The Architecture

Most open-source AI frameworks try to force a single 7B or 8B model to do everything simultaneously: read a massive prompt, reason step-by-step, format complex nested JSON tool calls, and execute code. **They inevitably hallucinate, drop tags, or fail.**

Krusch Agentic MCP decouples the cognitive load into two distinct, specialized phases using the *exact same model*:

1. **The Thinker (Layer 1)**: Acts as the Executive Intelligence. It reads the complex objective and generates a strict, validated **Cognitive Blueprint** wrapped in a JSON `<holodata>` array.
2. **The Implementer (Layer 2)**: Operating under strict sandbox rules, it blindly translates the Thinker's blueprint into the final JSON array, tool call, or code block without deviating or hallucinating.

**The Hardware Hack:** By orchestrating identical parallel phases, Krusch Agentic MCP ensures the target model's weights remain securely pinned in your GPU's VRAM (e.g., RTX 3060). This eliminates the massive latency penalty of swapping different models out of memory.

### 📊 Unassisted vs Dual-Engine Performance

When an unassisted model attempts to generate code or execute tools, it hallucinates parameters, drops JSON brackets, and skips syntax. 

The Krusch Agentic Proxy acts as a structural safety net. By forcing the model through the Dual-Engine (Thinker + Implementer) pipeline, we see massive gains in reliability for capable edge models (7B-14B) without needing a massive LLM.

#### 1. Weklund Tool-Calling Accuracy (`v0.4.0` Verified across 5-GPU Fleet: 2026-10-18)
*Demonstrating the elimination of hallucinated parameters and complete multi-tool / conditional planning via `v0.4.0` JSON canonicalization (`temperature=0.0`) across all 3 multi-family GPU compute pools.*

| Model & GPU Pool | Naked (Baseline) | Krusch Agentic Proxy (`v0.4.0`) | Avg Latency / Task |
| :--- | :--- | :--- | :--- |
| **Qwen3 Coder 30B MoE** (Heavy Pool — Dual RTX 2080 Ti NVLink) | 82.5% | **100.0% (40/40)** | **~0.5s** |
| **Qwen 2.5 Coder 7B** (Primary Local — Single RTX 2080 Ti) | 69.1% | **100.0% (40/40)** | **~1.4s** |
| **Google Gemma 4 E4B** (Diversity Pool — Dual RTX 3060) | 72.5% | **100.0% (40/40)** | **~1.2s** |
| **Nous Hermes 3** (Diversity Pool — Dual RTX 3060) | 70.0% | **100.0% (40/40)** | **~1.9s** |
| **DeepSeek-R1-14B** (Heavy Pool — Dual RTX 2080 Ti) | 76.2% | **97.5%** | **~2.1s** |
| **Yi-Coder 9B** | 63.8% | **81.4%** | **~1.8s** |

*(Note: Fleet-wide Weklund Tool-Calling score across all 4 resident 5-GPU models is **160/160 = 100.0%**. IFEval Instruction Following on Qwen 7B also improved from 45.2% to 58.1%.)*

#### 2. HumanEval (Code Gen) Across Weight Classes
*A distributed benchmark run highlighting the cognitive load effects of the Dual-Engine constraint across different parameter sizes.*

| Model | Naked (Baseline) | Krusch Dual-Engine | Delta |
| :--- | :---: | :---: | :---: |
| **DeepSeek-R1-14B** | 70.7% | **90.2%** | 📈 **+19.5%** |
| **Qwen 2.5 Coder 7B** | 73.8% | **86.6%** | 📈 **+12.8%** |
| **Yi-Coder 9B** | 69.5% | **73.8%** | 📈 **+4.3%** |
| **Qwen 2.5 Coder 3B** | **79.3%** | 75.6% | 📉 *-3.7%* |
| **Llama 3.1 8B** | **61.6%** | 53.7% | 📉 *-7.9%* |
| **Llama 3.2 3B** | **50.6%** | 47.0% | 📉 *-3.6%* |
| **Qwen 2.5 Coder 0.5B** | **47.0%** | 26.2% | 📉 *-20.8%* |

> [!WARNING]
> **The Small-Model Cognitive Penalty:** The Dual-Engine Proxy is a massive multiplier for capable 7B-30B models. However, **it penalizes small models (≤ 3B).** The cognitive load of formatting the massive `<holodata>` JSON "Thinker" blueprint distracts smaller models. **Do not route 3B or sub-3B models through the proxy.**

---

## ⚙️ Production Agentic Systems & `/v1/*` Gateway Modules

In addition to OpenAI-compatible `/v1/chat/completions`, `krusch-agentic-proxy` bundles six deterministic production agentic subsystems (see [**Paper #23 — The 15 Production Agentic Systems**](https://krusch.dev/articles/fifteen-production-agent-systems) for full state-machine teardowns and invariants):

| Subsystem | Module | Core Guarantee & FastAPI Endpoint |
| :--- | :--- | :--- |
| **1. Async Agent Orchestrator** | `src/orchestrator.py` (`AsyncAgentOrchestrator`, `HostCircuitBreaker`) | Executes `100+` concurrent agent tasks via `asyncio.Semaphore(128)`, offloads synchronous tools via `asyncio.to_thread` so the event loop never stalls, and isolates faulty upstreams with per-host `CLOSED → OPEN → HALF_OPEN` circuit breakers (`POST /v1/orchestrate/batch`). |
| **2. Dynamic Context Budget Assembler** | `src/budget_assembler.py` (`DynamicContextBudgetAssembler`) | Strict pre-flight token allocator across system prompts, ranked tool schemas, retrieved documents, and chronological message history guaranteeing `total_allocated_tokens <= input_ceiling_tokens` (`POST /v1/context/budget`). |
| **3. Strict Pydantic v2 Tool Router** | `src/pydantic_tool_router.py` (`StrictPydanticToolRouter`) | Compiles OpenAI/MCP JSON Schemas into `ConfigDict(extra="forbid")` Pydantic v2 models (`json_schema_to_pydantic_model`) and automatically feeds structured validation traces back to the model for up to `3` self-repair retries (`POST /v1/tools/execute-strict`). |
| **4. Typed Multi-Agent Swarm** | `src/typed_swarm.py` (`TypedMultiAgentSwarm`) | Deterministic `RESEARCH → CODE → REVIEW → APPROVED \| REJECTED` state machine passing frozen Pydantic v2 handoff objects (`ResearchBrief`, `CodeArtifact`, `ReviewVerdict`)—raising `TypeError` on any raw text handoff (`POST /v1/swarm/run`). |
| **5. `$0.50` Cost Kill-Switch Middleware** | `src/cost_killswitch.py` (`CostKillSwitch`) | Tracks per-session `loop_depth` and projected token spend; downgrades paid frontier calls to `$0.00` sovereign local `qwen2.5-coder:7b` before breaching `$0.50` and hard-halts paid spend at `$0.50` (`POST /v1/cost/evaluate`). |
| **6. 5% Shadow Trajectory Comparator** | `src/shadow_comparator.py` (`ShadowTrajectoryComparator`) | Non-blocking `asyncio.create_task` `5%` traffic mirror computing dynamic-programming Levenshtein tool-sequence edit distance, argument-key Jaccard similarity, latency delta, and cost delta in SQLite WAL (`GET /v1/shadow/comparisons`). |

---

## 🤝 The Krusch 3-Tier & DBOS Agentic Ecosystem

This project operates within the **Krusch Agentic Ecosystem**, working alongside:

- **[`krusch-stack`](https://github.com/kruschdev/krusch-stack)**: Unified 7-pillar 39-tool MCP, CLI (`npx krusch-stack`), and zero-dep resilience gateway (`:3780`).
- **[`krusch-context-mcp`](https://github.com/kruschdev/krusch-context-mcp)**: Tier 1 Working Memory, Invariant Steering & Decision Lineage (`.agent/context.db`).
- **[`krusch-git`](https://github.com/kruschdev/krusch-git)**: Tier 2A Git DAG, AST Symbol Graphs & Hybrid RRF Semantic Search (`:4890`).
- **[`krusch-fs-mcp`](https://github.com/kruschdev/krusch-fs-mcp)**: Tier 2B Multi-Node OS Filesystem & Page-Anchored Memory.
- **[`krusch-coding-harness`](https://github.com/kruschdev/krusch-coding-harness)**: Tier 3 Staged Execution, PostgreSQL 2PC Apply Journal, 100-Trajectory Golden DAG CI Gate & `:3778` Control Bridge.
- **[`Jean SRE`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)**: Autonomous 24/7 Fleet SRE Daemon (`:3005`).

---

## 🚀 Quick Start Guide

### 1. Ensure Ollama is Running on Your Compute Node(s)
Ensure local Ollama is serving on `http://127.0.0.1:11434/v1/chat/completions` (or configure `KRUSCH_OLLAMA_URL`, `KRUSCHSERV_OLLAMA_URL`, and `KRUSCHDEV_OLLAMA_URL` for multi-node GPU routing).

### 2. Install & Run Tests
Start both the Active Gateway (`:5440`) and Passive OpenClaw Interceptor (`:5443`):

```bash
git clone https://github.com/kruschdev/krusch-agentic-proxy.git
cd krusch-agentic-proxy

# Install via pip (including pytest dev extras)
pip install -e ".[dev]"

# Verify all 35 unit & production agentic system tests
pytest -q
```

### 3. Configure
Copy the config template:
```bash
cp config.example.json config.json
```
Edit `config.json` to point to your local LLM API endpoint.

### 4. Integration

**For OpenClaw, Claude Desktop, or Cursor**, add the Krusch Agentic MCP to your `mcp_client_config.json`:

```json
{
  "mcpServers": {
    "krusch_engine": {
      "command": "krusch-agentic-mcp",
      "args": []
    }
  }
}
```

**For Hermes Agent**, add the following to your `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  krusch_engine:
    command: "krusch-agentic-mcp"
    args: []
```

## 🛠️ Exposed Tools

When connected, the server exposes the following tool to your parent agent:

### `krusch_execute_task(objective: str)`
Delegates a complex, multi-step objective or tool-calling task to the Krusch Dual-Engine reasoning model. The sub-agent will autonomously execute shell commands and read files to satisfy the objective before returning a final answer.

## 📈 Benchmarking & Architecture Reference

- [Benchmarks & Evaluation](BENCHMARKS.md) — Weklund Tool-Calling & HumanEval reproduction methodology.
- [Paper #23: The 15 Production Agentic Systems](https://krusch.dev/articles/fifteen-production-agent-systems) — Reference architecture, state-machine invariants, and benchmark proofs across the sovereign stack.
