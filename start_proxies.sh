#!/bin/bash
# start_proxies.sh
# Boots the Active Universal Gateway (:5440) and Passive OpenClaw Interceptor (:5443)
# of the Krusch Agentic Proxy (v0.4.0).
# PID files are written to .pids/ for clean shutdown via stop_proxies.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

ACTIVE_PORT="${ACTIVE_PORT:-5440}"
PASSIVE_PORT="${PASSIVE_PORT:-5443}"
export KRUSCH_CONTEXT_SQLITE="${KRUSCH_CONTEXT_SQLITE:-/home/krusch/homelab/.agent/context.db}"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "/home/krusch/homelab/projects/krusch-law/.venv/bin/python" ]; then
    PYTHON_CMD="/home/krusch/homelab/projects/krusch-law/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

echo "[*] Starting Krusch Agentic Proxy Active Gateway (Port $ACTIVE_PORT) using: $PYTHON_CMD"
PORT="$ACTIVE_PORT" setsid nohup $PYTHON_CMD "$SCRIPT_DIR/src/api_gateway.py" > "$SCRIPT_DIR/gateway_5440.log" 2>&1 < /dev/null &
echo $! > "$PID_DIR/gateway_5440.pid"
echo "    Active Gateway PID: $(cat "$PID_DIR/gateway_5440.pid") (Log: gateway_5440.log)"

echo "[*] Starting Krusch Agentic Proxy Passive Interceptor (Port $PASSIVE_PORT, PASSIVE_MODE=1)"
PORT="$PASSIVE_PORT" PASSIVE_MODE=1 setsid nohup $PYTHON_CMD "$SCRIPT_DIR/src/api_gateway.py" > "$SCRIPT_DIR/nvidia_proxy.log" 2>&1 < /dev/null &
PASSIVE_PID=$!
echo "$PASSIVE_PID" > "$PID_DIR/gateway_5443.pid"
echo "$PASSIVE_PID" > "$PID_DIR/nvidia.pid"
echo "    Passive Interceptor PID: $PASSIVE_PID (Log: nvidia_proxy.log)"

echo "[*] Both proxy instances launched in the background. Use './stop_proxies.sh' to stop."
