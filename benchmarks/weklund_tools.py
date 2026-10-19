import os
import sys
import json
import asyncio
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.dirname(SCRIPT_DIR)

# Add proxy root and benchmarks dir to path
sys.path.insert(0, PROXY_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.core import KruschEngine
from mtb.quality_benchmarks.tool_calling_problems import TOOL_CALLING_NEW_PROBLEMS
from mtb.quality_benchmarks.run_quality_benchmark import _is_tool_calling, _evaluate_tool_calling

async def run_tools_benchmark(cli_model: str = None):
    # Load config
    config_path = os.path.join(PROXY_DIR, 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception:
        print("[!] No config found, using defaults.")
        config = {}

    engine = KruschEngine(config)
    problems = TOOL_CALLING_NEW_PROBLEMS
    target_model = (
        cli_model
        or os.environ.get("MODEL")
        or config.get("llm", {}).get("model", "qwen2.5-coder:7b")
    )
    
    print(f"=== Starting Weklund Tool-Calling Benchmark ===")
    print(f"[*] Target Model: {target_model}")
    print(f"[*] Task Count: {len(problems)}\\n")

    results = []
    
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] Evaluating: {problem.name}")
        prompt = problem.prompt
        is_tool_call = _is_tool_calling(problem)
        exact_signature = getattr(problem, 'function_signature', '')
        
        start_t = time.time()
        try:
            blueprint, response_text = await engine.generate(
                prompt=prompt,
                is_code_exec=False,
                is_tool_call=is_tool_call,
                exact_signature=exact_signature,
                target_model=target_model,
                max_tokens=1024,
                temperature=0.1
            )
            
            passed, parsed_json = _evaluate_tool_calling(problem, response_text)
            details = parsed_json
        except Exception as e:
            passed = False
            details = f"Eval Error: {str(e)}"
            
        elapsed = time.time() - start_t
        status_str = "✅ PASS" if passed else "❌ FAIL"
        
        clean_details = str(details).replace('\\n', ' ')[:100].strip() if details else ""
        print(f" -> {status_str} ({elapsed:.1f}s) | Details: {clean_details}")
        
        if not passed:
            # If failed, print a snippet of the exact response for debugging
            try:
                print(f"   [Raw Model Output]: {response_text[:300].replace(chr(10), ' ')}...")
            except:
                pass
        print("")
        
        results.append(passed)

    passes = sum(1 for r in results if r)
    accuracy = (passes / len(problems)) * 100 if problems else 0
    
    print("=== FINAL TOOL-CALLING BENCHMARK ===")
    print(f"Score: {passes}/{len(problems)} ({accuracy:.1f}%)")
    print("====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weklund 40-problem Tool-Calling Benchmark")
    parser.add_argument("--model", type=str, default=None, help="Target model to benchmark")
    args = parser.parse_args()
    asyncio.run(run_tools_benchmark(args.model))
