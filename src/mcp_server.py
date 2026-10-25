"""
Krusch Agentic Proxy — MCP Server (stdio transport).

Exposes the `krusch_execute_task` tool via the Model Context Protocol.
Designed for integration with MCP clients like Cursor, OpenClaw, and Claude Desktop.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Set, Tuple

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:

    class FastMCP:  # type: ignore[no-redef]
        """Lightweight fallback decorator shim when the 'mcp' package is not installed."""

        def __init__(self, name: str, **kwargs: Any):
            self.name = name

        def tool(self):
            def _decorator(fn):
                return fn

            return _decorator

        def run(self) -> None:
            raise RuntimeError("The 'mcp' package is required to run the stdio MCP server.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT_MODEL, DEFAULT_OLLAMA_API_URL, load_config
from src.core import KruschEngine, parse_tool_calls
from src.tools import INTERNAL_TOOLS, execute_internal_tool

logger = logging.getLogger(__name__)

# Initialize the MCP Server
mcp = FastMCP("KruschAgenticMCP", dependencies=["httpx", "pydantic"])

config = load_config()
krusch_engine = KruschEngine(config)

# Max wall-clock time for the autonomous loop (seconds)
AGENT_TIMEOUT = float(os.environ.get("KRUSCH_AGENT_TIMEOUT", "120"))


@mcp.tool()
async def krusch_execute_task(objective: str, max_loops: int = 10) -> str:
    """
    Delegates a complex, multi-step objective to the Krusch Dual-Engine reasoning model.
    The sub-agent will autonomously execute allowlisted shell commands and read files
    to satisfy the objective before returning a final answer.
    """
    compact_tools = json.dumps(INTERNAL_TOOLS, separators=(",", ":"))
    system_prompt = f"""You are the Krusch Autonomous Sub-Agent. 
You have been delegated a complex objective by a frontend agent (like OpenClaw or Hermes).
Your goal is to use your available tools to achieve the objective, and then call 'final_answer' with the result.

OBJECTIVE: {objective}

AVAILABLE TOOLS:
{compact_tools}
"""

    chat_history = system_prompt
    effective_max_loops = max(1, min(int(max_loops), 25))

    async def _run_loop() -> str:
        nonlocal chat_history
        target_model = config.get("llm", {}).get("model", "qwen2.5-coder:7b")
        seen_calls: Set[Tuple[str, str]] = set()
        last_tool_result = ""

        for iteration in range(effective_max_loops):
            logger.info(f"--- Krusch Iteration {iteration + 1} ---")

            _, response_text = await krusch_engine.generate(
                prompt=chat_history,
                is_code_exec=False,
                is_tool_call=True,
                exact_signature="",
                target_model=target_model,
                max_tokens=2048,
                temperature=0.1,
            )

            tool_calls, fallback_msg = parse_tool_calls(response_text)

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]

                    logger.info(f"[*] Calling Tool: {tool_name}")

                    if tool_name == "final_answer":
                        return str(tool_args.get("answer", "Objective complete."))

                    call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                    if call_sig in seen_calls:
                        logger.warning(
                            f"[MCP] Duplicate tool call detected ({tool_name}); breaking loop."
                        )
                        return (
                            f"Error: Duplicate tool call detected for '{tool_name}'. "
                            f"Last result: {last_tool_result}"
                        )
                    seen_calls.add(call_sig)

                    tool_result = await asyncio.to_thread(
                        execute_internal_tool, tool_name, tool_args
                    )
                    last_tool_result = tool_result
                    logger.info(f"[*] Tool Result: {tool_result[:100]}...")

                    chat_history += (
                        f"\n\nASSISTANT CALLED TOOL: {tool_name}\n"
                        f"WITH ARGS: {json.dumps(tool_args)}\n"
                        f"TOOL RESULT:\n{tool_result}\n"
                    )
            elif fallback_msg is not None:
                return fallback_msg
            else:
                logger.warning(f"[MCP] Failed to parse tool call JSON: {response_text[:200]}")
                chat_history += (
                    f"\n\nASSISTANT RESPONSE: {response_text}\n"
                    f"SYSTEM: You must call a tool. Use 'final_answer' to finish."
                )

        return "Error: Krusch Agent exceeded maximum iterations without calling final_answer."

    try:
        return await asyncio.wait_for(_run_loop(), timeout=AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        return f"Error: Krusch Agent timed out after {AGENT_TIMEOUT}s."


def main():
    mcp.run()


if __name__ == "__main__":
    main()
