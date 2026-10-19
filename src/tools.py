"""
Krusch Agentic Proxy — Internal Tool Definitions & Sandboxed Executor.

Centralizes all internal tools available to the autonomous agent loop.
Imported by both mcp_server.py and api_gateway.py to avoid circular imports
and ensure consistent tool behavior.

Security: All shell/file operations are sandboxed behind tokenized command
allowlists, shell metacharacter guards, and sensitive path denylists (INV-4).
"""

import logging
import os
import re
import shlex
import subprocess
from typing import Any, Dict, List, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

# --- SECURITY: Tokenized Command Allowlist ---
# Single-binary read-only/diagnostic commands permitted in the sandbox.
ALLOWED_SINGLE_COMMANDS: Set[str] = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "echo",
    "pwd",
    "whoami",
    "date",
    "uname",
    "curl",
}

# Two-token subcommands permitted for git and docker inspection.
ALLOWED_SUBCOMMANDS: Set[Tuple[str, str]] = {
    ("git", "status"),
    ("git", "log"),
    ("git", "diff"),
    ("git", "show"),
    ("git", "branch"),
    ("docker", "ps"),
    ("docker", "logs"),
    ("docker", "inspect"),
}

# Human-readable allowlist summary for error messages and backwards compatibility
ALLOWED_COMMAND_PREFIXES: List[str] = sorted(
    list(ALLOWED_SINGLE_COMMANDS) + [f"{a} {b}" for a, b in ALLOWED_SUBCOMMANDS]
)

# Shell chaining, redirection, subshell, and control characters are strictly forbidden
_DISALLOWED_METACHAR_RE = re.compile(r"[;&|`$><\n\r\x00]")

# Dangerous flags on otherwise allowed binaries (e.g. find -exec / -delete, curl -o / -K)
_DANGEROUS_FIND_FLAGS: Set[str] = {
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-delete",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-fls",
}
_DANGEROUS_CURL_FLAGS: Set[str] = {
    "-o",
    "--output",
    "-O",
    "--remote-name",
    "-K",
    "--config",
    "-T",
    "--upload-file",
}

# --- SECURITY: Filesystem Read Restrictions ---
DEFAULT_ALLOWED_ROOTS = [
    os.path.expanduser("~"),
]

# Sensitive directory/file names blocked even inside allowed roots
_SENSITIVE_PATH_COMPONENTS: Set[str] = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".kube",
    ".docker",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "authorized_keys",
    "shadow",
    "gshadow",
    "passwd",
}


def _get_allowed_roots() -> List[str]:
    """Return the list of allowed filesystem roots, configurable via env var."""
    env_roots = os.environ.get("KRUSCH_ALLOWED_ROOTS")
    if env_roots:
        return [os.path.expanduser(r.strip()) for r in env_roots.split(":") if r.strip()]
    return DEFAULT_ALLOWED_ROOTS


def _is_command_allowed(command: str) -> bool:
    """
    Validate that a command contains no shell chaining/injection metacharacters
    and matches an exact tokenized binary or two-token subcommand allowlist.
    """
    if not isinstance(command, str):
        return False
    cmd_stripped = command.strip()
    if not cmd_stripped:
        return False

    if _DISALLOWED_METACHAR_RE.search(cmd_stripped):
        return False

    try:
        tokens = shlex.split(cmd_stripped)
    except ValueError:
        return False

    if not tokens:
        return False

    binary = tokens[0]
    if "/" in binary or "\\" in binary:
        return False

    if binary in ALLOWED_SINGLE_COMMANDS:
        if binary == "find":
            if any(tok in _DANGEROUS_FIND_FLAGS for tok in tokens[1:]):
                return False
        elif binary == "curl":
            if any(tok in _DANGEROUS_CURL_FLAGS for tok in tokens[1:]):
                return False
        return True

    if len(tokens) >= 2 and (binary, tokens[1]) in ALLOWED_SUBCOMMANDS:
        return True

    return False


def _is_path_allowed(path: str) -> bool:
    """
    Check if a file path is within the allowed filesystem roots AND does not
    target sensitive credential files, secret dotfiles, or key stores.
    """
    if not isinstance(path, str) or not path.strip():
        return False

    resolved = os.path.realpath(os.path.expanduser(path.strip()))
    parts = [p for p in resolved.split(os.sep) if p]
    basename = os.path.basename(resolved)

    # Block sensitive directories/files and .env secrets
    for part in parts:
        if part in _SENSITIVE_PATH_COMPONENTS:
            return False
    if basename == ".env" or basename.startswith(".env."):
        return False
    if len(parts) >= 2 and parts[-2] == ".git" and parts[-1] == "config":
        return False

    for root in _get_allowed_roots():
        resolved_root = os.path.realpath(root)
        if resolved.startswith(resolved_root + os.sep) or resolved == resolved_root:
            return True
    return False


# --- Tool Schema Definitions (OpenAI function calling format) ---
INTERNAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Executes an allowlisted diagnostic shell command on the host and returns stdout/stderr. Commands are restricted to a strict safety allowlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run (must match the tokenized allowlist, no shell chaining)",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file on the local filesystem. Paths are restricted to allowed roots and exclude sensitive dotfiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file (must be within allowed roots)",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this tool to return the final answer to the user once the objective is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final detailed answer to the user's objective",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


def execute_internal_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    Execute an internal tool by name with the given parameters.
    All operations are sandboxed behind security checks.
    """
    parameters = parameters if isinstance(parameters, dict) else {}

    if tool_name == "run_bash_command":
        cmd = parameters.get("command", "")

        if not _is_command_allowed(cmd):
            logger.warning(f"[Security] Blocked disallowed command: {cmd}")
            return (
                f"BLOCKED: Command not permitted by sandbox allowlist or contains disallowed "
                f"shell metacharacters. Allowed commands: {', '.join(ALLOWED_COMMAND_PREFIXES)}"
            )

        # TAC Bridge mode (remote execution via HTTP)
        if os.getenv("TAC_BRIDGE_MODE") == "1":
            try:
                response = httpx.post(
                    "http://127.0.0.1:11441/tac/run",
                    json={"command": cmd},
                    timeout=120,
                )
                data = response.json()
                if "error" in data and not data.get("output"):
                    return f"Execution Failed:\n{data['error']}"
                return f"STDOUT:\n{data.get('output', '')}\nEXIT_CODE: {data.get('exit_code', '')}"
            except Exception as e:
                return f"TAC Bridge Execution Failed: {str(e)}"

        # Local sandboxed execution
        try:
            result = subprocess.run(
                shlex.split(cmd),
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.expanduser("~"),
            )
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e:
            return f"Execution Failed: {str(e)}"

    elif tool_name == "read_file":
        path = parameters.get("path", "")

        if not _is_path_allowed(path):
            logger.warning(f"[Security] Blocked read of restricted path: {path}")
            return (
                f"BLOCKED: Path is outside allowed roots or targets a protected credential/secret file. "
                f"Allowed roots: {', '.join(_get_allowed_roots())}"
            )

        try:
            resolved = os.path.realpath(os.path.expanduser(path.strip()))
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(50_001)
            if len(content) > 50_000:
                total_size = os.path.getsize(resolved)
                return (
                    content[:50_000]
                    + f"\n\n[TRUNCATED — file is {total_size} bytes, showing first 50,000 chars]"
                )
            return content
        except Exception as e:
            return f"Failed to read file: {str(e)}"

    elif tool_name == "final_answer":
        return str(parameters.get("answer", "Objective complete."))

    logger.warning(f"[Tools] Unknown tool called: {tool_name}")
    return f"Unknown tool: {tool_name}"
