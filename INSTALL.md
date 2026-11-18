# Install `krusch-agentic-proxy`

These are instructions for a coding agent (Claude Code, Codex, Cursor, Windsurf, or Antigravity) that a person asked to install and configure `krusch-agentic-proxy` (`v0.4.0`) on their machine.

`krusch-agentic-proxy` is a Python `>= 3.10` Thinker/Implementer Dual-Engine execution proxy and MCP server for local LLMs (`7B` to `30B+` models). It exposes both a native `stdio` MCP server (`krusch-agentic-mcp`, providing `krusch_execute_task`) and an OpenAI-compatible FastAPI SSE gateway (`/v1/chat/completions` plus six `/v1/*` production agentic endpoints on port `5440`, and an optional passive interceptor on port `5443`).

## 1. Check Prerequisites First (All at Once)

Run a single read-only check before making any changes:

```bash
id -u
python3 --version 2>/dev/null || echo "MISSING: python3"
python3 -m venv --help >/dev/null 2>&1 || echo "MISSING: python3-venv"
git --version 2>/dev/null || echo "MISSING: git"
ollama --version 2>/dev/null || echo "OPTIONAL_MISSING: ollama"
curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null || echo "OLLAMA_NOT_RUNNING_OR_UNREACHABLE"
ss -ltnp 2>/dev/null | grep -E ':(5440|5443)\b' || lsof -iTCP:5440 -iTCP:5443 -sTCP:LISTEN 2>/dev/null || true
```

Evaluate the results:
1. **User check**: If `id -u` prints `0` (`root`), stop and ask the person to run you under a standard non-root user account.
2. **Python `>= 3.10` with `venv` (Required)**: `pyproject.toml` requires Python `3.10+` (`3.11` or `3.12` recommended). If `python3` is missing, older than `3.10`, or lacks `venv`, ask the person before installing Python `3.11+` (`uv python install 3.12`, `pyenv`, `brew`, or OS packages).
3. **Local LLM Backend (Ollama or OpenAI-Compatible Endpoint)**:
   - Unit tests (`pytest`) run completely offline without Ollama.
   - Live inference uses `http://127.0.0.1:11434/v1/chat/completions` (`qwen2.5-coder:7b` by default, or any `7B` to `30B+` model such as `gemma4:e4b`, `hermes3`, or `qwen3-coder-30b-moe`). **Important**: Do not configure `3B` or smaller models (`<= 3B`), because `<holodata>` blueprint formatting degrades small-model accuracy. If Ollama is running without a `7B+` coding model, ask the person before running `ollama pull qwen2.5-coder:7b`.
4. **PostgreSQL / SQLite Episodic RAG (Strictly Optional)**: Leave `KRUSCH_DISABLE_RAG=1` on standalone machines unless the person explicitly supplies `KRUSCH_CONTEXT_SQLITE` or `KRUSCH_RAG_DSN`.

## 2. Choose the Right Setup Mode

- **Mode A: Native `stdio` MCP Server (Recommended Default for Coding Agents & IDEs)**: Installs `krusch-agentic-mcp` inside `.venv`, configures single-node local LLM routing, verifies all `39` automated `pytest` tests, and registers `krusch_engine` in the person's MCP client.
- **Mode B: OpenAI-Compatible FastAPI SSE Gateway (`127.0.0.1:5440`, plus optional `:5443` Passive Mode)**: Starts `src/api_gateway.py` on `127.0.0.1:5440` for clients that speak `/v1/chat/completions` or call `/v1/orchestrate/batch`, `/v1/context/budget`, `/v1/tools/execute-strict`, `/v1/swarm/run`, `/v1/cost/evaluate`, and `/v1/shadow/comparisons`.

## 3. Install in an Isolated Virtual Environment & Configure

Check `pwd` first so you clone into an appropriate workspace directory (for example, `$HOME/src/krusch-agentic-proxy` if `pwd` is `$HOME`):

```bash
git clone https://github.com/kruschdev/krusch-agentic-proxy.git
cd krusch-agentic-proxy
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
[ -f config.json ] || cp config.example.json config.json
```

If the person already has a `7B+` model loaded in local Ollama (check the `/api/tags` output from Step 1), update `llm.model` in `config.json` to match their installed `7B+` model (default `qwen2.5-coder:7b`) and keep `llm.api_url` pointed at `http://127.0.0.1:11434/v1/chat/completions` (unless they specified another endpoint).

Run the automated `pytest` verification suite (`39` tests covering Dual-Engine `<holodata>` parsing, `shlex` sandbox guards, async batch circuit breakers, context budget allocation, strict Pydantic v2 tool self-repair, typed multi-agent swarm handoffs, `$0.50` cost kill-switch, and `5%` shadow trajectory comparison):

```bash
.venv/bin/pytest tests/ -v
```

## 4. Connect MCP Clients or Launch the HTTP Gateway

### Mode A: Register `krusch-agentic-mcp` (`stdio` MCP Server)

Capture the absolute path `"$(pwd)/.venv/bin/krusch-agentic-mcp"` and `"$(pwd)/config.json"`, then merge `krusch_engine` into the person's active MCP client configuration without removing existing servers. Setting `KRUSCH_FLEET_ROUTING=0`, `KRUSCH_FLEET_FAILOVER=0`, and `KRUSCH_DISABLE_RAG=1` keeps all requests strictly on the local machine (`127.0.0.1:11434`):

- **Claude Code CLI**:
  ```bash
  claude mcp add krusch_engine \
    -e KRUSCH_PROXY_CONFIG="$(pwd)/config.json" \
    -e KRUSCH_FLEET_ROUTING=0 \
    -e KRUSCH_FLEET_FAILOVER=0 \
    -e KRUSCH_DISABLE_RAG=1 \
    -- "$(pwd)/.venv/bin/krusch-agentic-mcp"
  ```

- **Cursor (`~/.cursor/mcp.json`), Windsurf (`~/.codeium/windsurf/mcp_config.json`), or Antigravity (`~/.gemini/antigravity/mcp_config.json`)**:
  *(Substitute `/absolute/path/to/krusch-agentic-proxy` with the real output of `pwd`.)*
  ```json
  {
    "mcpServers": {
      "krusch_engine": {
        "command": "/absolute/path/to/krusch-agentic-proxy/.venv/bin/krusch-agentic-mcp",
        "args": [],
        "env": {
          "KRUSCH_PROXY_CONFIG": "/absolute/path/to/krusch-agentic-proxy/config.json",
          "KRUSCH_FLEET_ROUTING": "0",
          "KRUSCH_FLEET_FAILOVER": "0",
          "KRUSCH_DISABLE_RAG": "1"
        }
      }
    }
  }
  ```

- **Codex (`~/.codex/config.toml`)**:
  ```toml
  [mcp_servers.krusch_engine]
  command = "/absolute/path/to/krusch-agentic-proxy/.venv/bin/krusch-agentic-mcp"
  args = []

  [mcp_servers.krusch_engine.env]
  KRUSCH_PROXY_CONFIG = "/absolute/path/to/krusch-agentic-proxy/config.json"
  KRUSCH_FLEET_ROUTING = "0"
  KRUSCH_FLEET_FAILOVER = "0"
  KRUSCH_DISABLE_RAG = "1"
  ```

- **Hermes Agent (`~/.hermes/config.yaml`)**:
  ```yaml
  mcp_servers:
    krusch_engine:
      command: "/absolute/path/to/krusch-agentic-proxy/.venv/bin/krusch-agentic-mcp"
      args: []
      env:
        KRUSCH_PROXY_CONFIG: "/absolute/path/to/krusch-agentic-proxy/config.json"
        KRUSCH_FLEET_ROUTING: "0"
        KRUSCH_FLEET_FAILOVER: "0"
        KRUSCH_DISABLE_RAG: "1"
  ```

### Mode B: Start the FastAPI SSE Gateway (`127.0.0.1:5440` and Optional `:5443`)

Only start HTTP listeners if the person requested the OpenAI-compatible REST proxy and port `5440` (or `5443`) is free. Always pass `HOST=127.0.0.1` so the gateway binds strictly to loopback rather than `0.0.0.0`:

```bash
# Active Dual-Engine Gateway on 127.0.0.1:5440
HOST=127.0.0.1 PORT=5440 KRUSCH_FLEET_ROUTING=0 KRUSCH_FLEET_FAILOVER=0 KRUSCH_DISABLE_RAG=1 \
  .venv/bin/python src/api_gateway.py
```

Verify liveness from a second shell:

```bash
curl -fsS http://127.0.0.1:5440/health
```

If the person also wants the Passive Interceptor (`PASSIVE_MODE=1`) on `127.0.0.1:5443` for legacy OpenAI-compatible bots:

```bash
HOST=127.0.0.1 PORT=5443 PASSIVE_MODE=1 KRUSCH_FLEET_ROUTING=0 KRUSCH_FLEET_FAILOVER=0 KRUSCH_DISABLE_RAG=1 \
  .venv/bin/python src/api_gateway.py
```

*(Optional Bearer Auth: Set `KRUSCH_PROXY_API_KEY` to a caller-supplied token to enforce constant-time `Authorization: Bearer <token>` or `X-API-Key` checks on all routes except `/health`.)*

## Never

- **Never run global `pip install` or `sudo pip`**: always install inside `.venv` (`python3 -m venv .venv && .venv/bin/pip install ...`).
- **Never run `krusch-agentic-mcp`, `api_gateway.py`, or `pytest` as `root` or with `sudo`**.
- **Never overwrite an existing `config.json` or wipe existing servers in an IDE MCP config file**; read, parse, and merge only the `krusch_engine` entry.
- **Never route `<= 3B` models** (`0.5B`, `1.5B`, `3B`) through the Dual-Engine proxy, and never run `ollama pull` without asking the person first.
- **Never hardcode plaintext cloud API keys into `config.json`**: always reference environment variables via `"api_key": "ENV:OPENROUTER_API_KEY"`.
- **Never omit `HOST=127.0.0.1`** when starting `src/api_gateway.py` unless the person explicitly asks for LAN binding and configures `KRUSCH_PROXY_API_KEY`.

## Tell the Person After

Share a concise status summary:
1. **Virtual environment & config**: Report the `.venv` path, `config.json` location, and configured `7B+` local model (`llm.model`).
2. **Verification status**: Confirm `.venv/bin/pytest tests/ -v` (`39/39` tests) passed cleanly (and `curl http://127.0.0.1:5440/health` if Gateway Mode was started).
3. **Configured clients / ports**: List updated MCP config files (or active `127.0.0.1:5440` / `127.0.0.1:5443` endpoints) and remind the person to reload MCP servers in their IDE.
4. **First-turn test prompt**: Suggest testing in their next session with: `"Use krusch_execute_task to inspect the current directory layout and summarize pyproject.toml."`
