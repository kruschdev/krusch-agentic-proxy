#!/bin/bash
set -e

MODEL="${MODEL:-qwen2.5-coder:7b}"
PROXY_URL="${PROXY_URL:-http://localhost:5440}"

echo "============================================================"
echo " Krusch Agentic Proxy - Official LM Evaluation Harness Benchmark Tool"
echo "============================================================"
echo ""
echo "Target Proxy : $PROXY_URL"
echo "Target Model : $MODEL"
echo ""

# Ensure API is running
if ! curl -s "$PROXY_URL/health" > /dev/null; then
    echo "[!] Error: The Krusch Agentic Proxy API is not running on $PROXY_URL."
    echo "    Please start the API first: ./start_proxies.sh"
    exit 1
fi

echo "[*] Krusch Agentic Proxy API detected at $PROXY_URL."

# Create isolated venv for benchmarking if needed
if [ ! -d "bench_venv" ]; then
    echo "[*] Creating virtual environment for benchmarks..."
    python3 -m venv bench_venv
fi

source bench_venv/bin/activate

echo "[*] Installing lm-eval and dependencies (this may take a moment)..."
pip install --quiet "lm-eval[api]"

echo ""
echo "[*] Running Structural Benchmarks (Limit: 5 questions per task)..."
echo "    (Tasks: ifeval, humaneval)"
echo "------------------------------------------------------------"

lm_eval \
    --model local-chat-completions \
    --tasks ifeval,humaneval \
    --model_args "model=$MODEL,base_url=$PROXY_URL/v1/chat/completions" \
    --apply_chat_template \
    --num_fewshot 5 \
    --limit 5 \
    --batch_size 1

echo "------------------------------------------------------------"
echo "[*] Running Weklund Tool-Calling Benchmark..."
echo "------------------------------------------------------------"
export PASSIVE_MODE=1
python benchmarks/weklund_tools.py

echo "------------------------------------------------------------"
echo "[*] Benchmark complete!"
echo "To run a full, unabridged benchmark, run:"
echo "source bench_venv/bin/activate && lm_eval --model local-chat-completions --tasks ifeval,humaneval --model_args model=$MODEL,base_url=$PROXY_URL/v1/chat/completions"
