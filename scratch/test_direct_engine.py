import asyncio
import json
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KRUSCH_CONTEXT_SQLITE", "/home/krusch/homelab/.agent/context.db")
from src.core import KruschEngine

async def run_direct():
    config = {
        "llm": {
            "api_url": "http://127.0.0.1:11434/v1/chat/completions",
            "model": "qwen2.5-coder:7b",
            "temperature": 0.1
        }
    }
    
    engine = KruschEngine(config)
    
    prompt = "Explain what GPU we are currently using for our homelab fleet."
    print("--- Running KruschEngine.generate directly ---")
    print(f"Prompt: {prompt}")
    
    try:
        blueprint, response = await engine.generate(
            prompt=prompt,
            is_code_exec=False,
            is_tool_call=False,
            target_model="qwen2.5-coder:7b"
        )
        print("\n--- BLUEPRINT ---")
        print(blueprint)
        print("\n--- RESPONSE ---")
        print(response)
    except Exception as e:
        print(f"Engine failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(run_direct())
