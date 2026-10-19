import os
import sys
import json
import asyncio
import time
import re
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, os.path.join(PROXY_DIR, 'src'))
sys.path.insert(0, PROXY_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.core import KruschEngine, get_ollama_url_for_model
from src.client import chat
from mtb.quality_benchmarks.sandbox import execute_code, SandboxResult

MODELS = [
    "qwen2.5-coder:7b"
]

API_URL = "http://127.0.0.1:11434/v1/chat/completions"

def _build_test_code(problem: dict, code: str) -> str:
    """Build executable code combining prompt + extracted model code + tests."""
    lines = [problem["prompt"], code, problem["test"], f"check({problem['entry_point']})", "print('ALL TESTS PASSED')"]
    return "\n".join(lines)

async def evaluate_naked_model(model: str, problem: dict) -> tuple[bool, str]:
    ai_config = {
        'provider': 'ollama',
        'api_url': get_ollama_url_for_model(model, API_URL),
        'model': model,
        'temperature': 0.1,
        'max_tokens': 1024
    }
    
    system_prompt = """You are an expert Python developer. Complete the following Python function.
CRITICAL RULE: Respond ONLY with the Python code completing the function. DO NOT wrap the output in markdown code blocks. DO NOT output conversational text, explanations, or print statements. Your output will be piped directly into a strict Python sandbox."""
    
    prompt = problem["prompt"]

    try:
        response_text = await chat(system_prompt, prompt, ai_config)
        cleaned = response_text.strip()
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", cleaned, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        elif cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()
            
        test_code = _build_test_code(problem, response_text)
        result = execute_code(test_code, timeout=10)
        passed = result.success
        details = "Passed" if passed else f"Failed: {result.stderr or result.stdout}"
    except Exception as e:
        passed = False
        details = f"Eval Error: {str(e)}"
        
    return passed, details


async def evaluate_dual_engine(model: str, engine: KruschEngine, problem: dict) -> tuple[bool, str]:
    try:
        exact_signature = f"def {problem['entry_point']}"
        blueprint, response_text = await engine.generate(
            prompt=problem["prompt"],
            is_code_exec=True,
            is_tool_call=False,
            exact_signature=exact_signature,
            target_model=model,
            max_tokens=1024,
            temperature=0.1
        )
        
        test_code = _build_test_code(problem, response_text)
        result = execute_code(test_code, timeout=10)
        passed = result.success
        details = "Passed" if passed else f"Failed: {result.stderr or result.stdout}"
    except Exception as e:
        passed = False
        details = f"Eval Error: {str(e)}"
    return passed, details


async def run_all_benchmarks():
    dataset_path = os.path.join(SCRIPT_DIR, "datasets", "HumanEval.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please download it first.")
        return

    problems = []
    with open(dataset_path, "r") as f:
        for line in f:
            problems.append(json.loads(line))
            
    # For quick testing, you can limit: problems = problems[:5]

    print(f"=== Starting HumanEval Code Generation Benchmark ===")
    print(f"[*] Task Count: {len(problems)}\\n")

    results_summary = {}
    config = {"llm": {"api_url": API_URL, "model": MODELS[0]}}
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
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem['task_id']} | {clean_details}")
        
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
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem['task_id']} | {clean_details}")
            
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
