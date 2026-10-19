import os
import sys
import json
import asyncio
import time
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, os.path.join(PROXY_DIR, 'src'))
sys.path.insert(0, PROXY_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.core import KruschEngine, get_ollama_url_for_model
from src.client import chat
from mtb.quality_benchmarks.sandbox import execute_code, SandboxResult

# Running on kruschserv (Dual RTX 3060 24GB - Non-Qwen Diversity Pool)
NODE_CONFIGS = {
    "kruschserv_Dual_RTX_3060": {
        "url": get_ollama_url_for_model("gemma4:e4b"),
        "models": ["gemma4:e4b", "hermes3:latest"]
    }
}

def _build_test_code(problem: dict, code: str) -> str:
    """Build executable code combining prompt + extracted model code + tests."""
    lines = [problem["prompt"], code, problem["test"], f"check({problem['entry_point']})", "print('ALL TESTS PASSED')"]
    return "\n".join(lines)

async def evaluate_naked_model(model: str, url: str, problem: dict) -> tuple[bool, str]:
    ai_config = {
        'provider': 'ollama',
        'api_url': url,
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


async def evaluate_dual_engine(model: str, url: str, problem: dict) -> tuple[bool, str]:
    config = {"llm": {"api_url": url, "model": model}}
    engine = KruschEngine(config)
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


async def run_node_benchmarks(node_name: str, config: dict, problems: list, results_summary: dict):
    url = config["url"]
    models = config["models"]
    
    for model in models:
        print(f"\n==============================================")
        print(f"🏃 [{node_name}] RUNNING BENCHMARKS FOR: {model}")
        print(f"==============================================\n")
        
        results_summary[model] = {"naked": 0, "dual": 0}

        # 1. NAKED MODEL
        print(f"--- 1. NAKED / INDIVIDUAL MODEL ({model}) ---")
        passes_naked = 0
        for i, problem in enumerate(problems):
            start_t = time.time()
            passed, details = await evaluate_naked_model(model, url, problem)
            elapsed = time.time() - start_t
            if passed: passes_naked += 1
            status_str = "✅ PASS" if passed else "❌ FAIL"
            clean_details = str(details).replace('\\n', ' ')[:100].strip() if details else ""
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem['task_id']} | {clean_details}")
        
        results_summary[model]["naked"] = passes_naked
        print(f"👉 Naked Score: {passes_naked}/{len(problems)} ({(passes_naked/len(problems))*100:.1f}%)\n")

        # 2. DUAL ENGINE
        print(f"--- 2. KRUSCH AGENTIC DUAL ENGINE ({model}) ---")
        passes_dual = 0
        for i, problem in enumerate(problems):
            start_t = time.time()
            passed, details = await evaluate_dual_engine(model, url, problem)
            elapsed = time.time() - start_t
            if passed: passes_dual += 1
            status_str = "✅ PASS" if passed else "❌ FAIL"
            clean_details = str(details).replace('\\n', ' ')[:100].strip() if details else ""
            print(f"[{i+1}/{len(problems)}] {status_str} ({elapsed:.1f}s) | {problem['task_id']} | {clean_details}")
            
        results_summary[model]["dual"] = passes_dual
        print(f"👉 Dual Engine Score: {passes_dual}/{len(problems)} ({(passes_dual/len(problems))*100:.1f}%)\n")


async def run_all_benchmarks():
    dataset_path = os.path.join(SCRIPT_DIR, "datasets", "HumanEval.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please download it first.")
        return

    problems = []
    with open(dataset_path, "r") as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"=== Starting Gemma4/Mistral/Granite Code Gen Benchmark ===")
    print(f"[*] Task Count: {len(problems)}\n")

    results_summary = {}

    # Run concurrently across the configured nodes
    tasks = []
    for node_name, config in NODE_CONFIGS.items():
        tasks.append(run_node_benchmarks(node_name, config, problems, results_summary))
        
    await asyncio.gather(*tasks)

    print("\n\n=================================================")
    print("📊 FINAL GEMMA4/MISTRAL/GRANITE BENCHMARK SUMMARY")
    print("=================================================")
    print(f"{'Model':<40} | {'Naked':<10} | {'Dual Engine':<10}")
    print("-" * 65)
    for node, config in NODE_CONFIGS.items():
        for model in config["models"]:
            naked_pct = (results_summary[model]['naked'] / len(problems)) * 100
            dual_pct = (results_summary[model]['dual'] / len(problems)) * 100
            print(f"{model:<40} | {naked_pct:>5.1f}%     | {dual_pct:>5.1f}%")
    print("=================================================")

if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
