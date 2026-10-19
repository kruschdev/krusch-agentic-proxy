import os
import sys
import json
import asyncio
import time
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROXY_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.core import KruschEngine, get_ollama_url_for_model
from src.client import chat
from mtb.quality_benchmarks.tool_calling_problems import TOOL_CALLING_NEW_PROBLEMS
from mtb.quality_benchmarks.run_quality_benchmark import _is_tool_calling, _evaluate_tool_calling

MODELS = [
    "qwen2.5-coder:7b",
    "gemma4:e4b",
    "hermes3:latest",
    "qwen3-coder-30b-moe:latest",
]

API_URL = "http://127.0.0.1:11434/v1/chat/completions"


async def evaluate_naked_model(model: str, problem) -> tuple[bool, str]:
    ai_config = {
        "api_url": get_ollama_url_for_model(model, API_URL),
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    
    # Minimal system prompt asking it to output JSON tools
    system_prompt = """You are a helpful AI assistant. If valid tools match the request, respond ONLY with a raw JSON array representing the Tool Calls. DO NOT wrap the output in markdown code blocks.
The JSON array MUST conform exactly to this flat schema: [{"name": "<function_name>", "arguments": {"<param1>": "<val1>"}}]"""

    try:
        response_text = await chat(system_prompt, problem.prompt, ai_config)
        # clean up response text in case it uses markdown code blocks
        cleaned = response_text.strip()
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", cleaned, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        elif cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()
            
        passed, parsed_json = _evaluate_tool_calling(problem, response_text)
        details = parsed_json
    except Exception as e:
        passed = False
        details = f"Eval Error: {str(e)}"
        
    return passed, details


async def evaluate_dual_engine(model: str, engine: KruschEngine, problem) -> tuple[bool, str]:
    is_tool_call = _is_tool_calling(problem)
    exact_signature = getattr(problem, 'function_signature', '')
    try:
        blueprint, response_text = await engine.generate(
            prompt=problem.prompt,
            is_code_exec=False,
            is_tool_call=is_tool_call,
            exact_signature=exact_signature,
            target_model=model,
            max_tokens=1024,
            temperature=0.1
        )
        passed, parsed_json = _evaluate_tool_calling(problem, response_text)
        details = parsed_json
    except Exception as e:
        passed = False
        details = f"Eval Error: {str(e)}"
    return passed, details

async def run_all_benchmarks():
    problems = TOOL_CALLING_NEW_PROBLEMS
    print(f"=== Starting Weklund Tool-Calling Benchmark on ALL MODELS ===")
    print(f"[*] Task Count: {len(problems)}\\n")

    results_summary = {}

    config = {"llm": {"api_url": API_URL}}
    engine = KruschEngine(config)

    for model in MODELS:
        print(f"\n==============================================")
        print(f"🏃 RUNNING BENCHMARKS FOR MODEL: {model}")
        print(f"==============================================\n")
        
        results_summary[model] = {"naked": 0, "dual": 0}

        # 1. NAKED MODEL
        print(f"--- 1. NAKED / INDIVIDUAL MODEL ---")
        passes_naked = 0
        for i, problem in enumerate(problems):
            start_t = time.time()
            passed, details = await evaluate_naked_model(model, problem)
            elapsed = time.time() - start_t
            if passed: passes_naked += 1
            status_str = "✅ PASS" if passed else "❌ FAIL"
            clean_details = str(details).replace('\\n', ' ')[:100].strip() if details else ""
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem.name} | Details: {clean_details}")
        
        results_summary[model]["naked"] = passes_naked
        print(f"👉 Naked Score: {passes_naked}/{len(problems)} ({(passes_naked/len(problems))*100:.1f}%)\n")

        # 2. DUAL ENGINE
        print(f"--- 2. KRUSCH AGENTIC DUAL ENGINE ---")
        passes_dual = 0
        for i, problem in enumerate(problems):
            start_t = time.time()
            passed, details = await evaluate_dual_engine(model, engine, problem)
            elapsed = time.time() - start_t
            if passed: passes_dual += 1
            status_str = "✅ PASS" if passed else "❌ FAIL"
            clean_details = str(details).replace('\\n', ' ')[:100].strip() if details else ""
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem.name} | Details: {clean_details}")
            
        results_summary[model]["dual"] = passes_dual
        print(f"👉 Dual Engine Score: {passes_dual}/{len(problems)} ({(passes_dual/len(problems))*100:.1f}%)\n")


    print("\n\n=================================================")
    print("📊 FINAL BENCHMARK SUMMARY")
    print("=================================================")
    print(f"{'Model':<40} | {'Naked':<10} | {'Dual Engine':<10}")
    print("-" * 65)
    for model in MODELS:
        naked_pct = (results_summary[model]['naked'] / len(problems)) * 100
        dual_pct = (results_summary[model]['dual'] / len(problems)) * 100
        print(f"{model:<40} | {naked_pct:>5.1f}%     | {dual_pct:>5.1f}%")
    print("=================================================")

if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
