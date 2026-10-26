# Krusch Agentic MCP

![Krusch Agentic Proxy Banner](docs/assets/banner.png)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastMCP](https://img.shields.io/badge/FastMCP-005571?style=flat)
![Version: 0.4.0](https://img.shields.io/badge/Version-0.4.0-green.svg)
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
*Demonstrating the elimination of hallucinated parameters and complete multi-tool / conditional planning via `v0.4.0` JSON canonicalization (`temperature=0.0`) across all 3 compute nodes (`krusch`, `kruschdev`, `kruschserv`).*

| Model & Fleet Node | Naked (Baseline) | Krusch Agentic Proxy (`v0.4.0`) | Avg Latency / Task |
| :--- | :--- | :--- | :--- |
| **Qwen3 Coder 30B MoE** (`kruschdev` Dual RTX 2080 Ti NVLink) | 82.5% | **100.0% (40/40)** | ⚡ **~0.5s** |
| **Qwen 2.5 Coder 7B** (`krusch` Single RTX 2080 Ti) | 69.1% | **100.0% (40/40)** | ⚡ **~1.4s** |
| **Google Gemma 4 E4B** (`kruschserv` Dual RTX 3060) | 72.5% | **100.0% (40/40)** | ⚡ **~1.2s** |
| **Nous Hermes 3** (`kruschserv` Dual RTX 3060) | 70.0% | **100.0% (40/40)** | ⚡ **~1.9s** |
| **DeepSeek-R1-14B** (`kruschdev` Dual RTX 2080 Ti) | 76.2% | **97.5%** | **~2.1s** |
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

## 🤝 The Krusch 3-Tier & DBOS Agentic Ecosystem

This project operates within the **Krusch Homelab Agentic Ecosystem**, working alongside:

- **`krusch-context-mcp`**: Tier 1 Working Memory, Invariant Steering & Decision Lineage (`.agent/context.db`).
- **`krusch-git`**: Tier 2A Git DAG, AST Symbol Graphs & Hybrid RRF Semantic Search (`:4890`).
- **`krusch-fs`**: Tier 2B Multi-Node OS Filesystem & Page-Anchored Memory across all 8 fleet nodes.
- **`krusch-harness`**: Tier 3 Staged Execution, PostgreSQL 2PC Apply Journal & `:3778` Control Bridge.
- **`jean-sre`**: Autonomous 24/7 Fleet SRE Daemon (`kruschdev:3005`).

---

## 🚀 Quick Start Guide

### 1. Ensure Ollama is Running on Your Fleet Node(s)
Ensure local Ollama is serving on `http://127.0.0.1:11434/v1/chat/completions` (or configure `KRUSCH_OLLAMA_URL`, `KRUSCHSERV_OLLAMA_URL`, and `KRUSCHDEV_OLLAMA_URL` for multi-node GPU routing).

### 2. Install or Launch the Proxy Services
Start both the Active Gateway (`:5440`) and Passive OpenClaw Interceptor (`:5443`):

```bash
git clone https://github.com/kruschdev/krusch_agentic_proxy.git
cd krusch_agentic_proxy

# Install via pip
pip install -e .
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

## 📈 Benchmarking

To verify the structural accuracy and view the complete methodology for reproducing these results on your own hardware using the EleutherAI LM Evaluation Harness, please see the [Benchmarks & Evaluation](BENCHMARKS.md) documentation.
