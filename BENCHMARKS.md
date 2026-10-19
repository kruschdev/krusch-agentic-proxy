# Krusch Agentic Proxy: Benchmarks & Evaluation

The Krusch Agentic Proxy was explicitly designed to combat the reasoning and formatting degradation that occurs when consumer GPUs attempt complex tool-calling and code execution. 

We ran a massive 8-hour empirical evaluation against the top open-source reasoning benchmarks using a 7B parameter model. The data proves that forcing a small model into a `<holodata>` Thinker/Implementer loop dramatically increases its structural reliability.

## 📊 The Structural Benchmark Data

When an unassisted model attempts to generate code or execute tools, it hallucinates parameters, drops JSON brackets, and skips syntax. The Krusch Agentic Proxy acts as a structural safety net. By forcing the model through the Dual-Engine (Thinker + Implementer) pipeline, we see massive gains in reliability for capable edge models (7B-14B).

### 1. Weklund Tool-Calling Accuracy (`v0.4.0` Verified across 5-GPU Fleet: 2026-10-18)
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

### 2. HumanEval (Code Gen) Across Weight Classes
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
> **The Small-Model Cognitive Penalty:** The Dual-Engine Proxy is a massive multiplier for capable 7B-30B models. However, **it penalizes small models (≤ 3B).** The cognitive load of formatting the `<holodata>` JSON "Thinker" blueprint distracts smaller models. **Do not route 3B or sub-3B models through the Dual-Engine mode.**

### Why it Crushes Weklund & IFEval
When a 7B model is asked to generate code or execute tools natively, it hallucinates parameters, drops JSON brackets, or skips syntax. The Krusch Agentic Proxy forces the model into a deterministic schema with canonicalized multi-tool arrays (`[{"name": ..., "arguments": {...}}]`) and `<holodata>` cognitive blueprints.

By separating the cognitive load into a Thinker (planning) and an Implementer (execution), the proxy achieves significantly higher zero-shot compilation rates, **40/40 (100.0%)** Weklund tool-calling accuracy (~1.5s/task on a single RTX 2080 Ti), and strict adherence to formatting constraints.

---

## 🚫 Anti-Patterns: Why Pure Conversational Math Uses Fast-Path

Standard generalized arithmetic benchmarks like **GSM8K (Math)** and **BBH (Logic)** should use the proxy's **Fast-Path NLP** mode (`FORCE_DUAL_ENGINE=0`, the default when no `tools` are passed) or query Ollama directly (`:11434`), whereas tool-calling and multi-file code execution use the structured Tool Router and Dual-Engine pipeline (`:5440` / `:5443`).

---

## ⚙️ Running the Suite

If you wish to reproduce these evaluations on your own hardware, the repository includes `benchmark.sh` and `benchmarks/weklund_tools.py`.

To run the 40-task Weklund Tool-Calling evaluation directly:

```bash
/home/krusch/homelab/projects/krusch-law/.venv/bin/python benchmarks/weklund_tools.py
```

To run the full LM Evaluation Harness + Weklund suite against the running gateway (`:5440`):

```bash
./benchmark.sh
```

### What `benchmark.sh` does:
1. **Isolated Environment**: Automatically creates a dedicated `bench_venv` to prevent polluting your API's primary Python environment.
2. **Harness Installation**: Installs `lm-eval[api]` with required dependencies.
3. **Execution**: Executes a test pass against the core structural benchmarks (`ifeval`, `humaneval`) using `MODEL="${MODEL:-qwen2.5-coder:7b}"` against `http://localhost:5440/v1/chat/completions`, followed by the 40-task Weklund Tool-Calling suite.
