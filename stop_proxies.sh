#!/bin/bash
# stop_proxies.sh
# Gracefully stops all Krusch Agentic Proxy instances (:5440 Active & :5443 Passive) using saved PID files.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

stop_proxy() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"

    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[*] Stopping $name proxy (PID: $pid)..."
            kill -TERM "$pid"
            for _ in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "    $name stopped."
                    rm -f "$pidfile"
                    return 0
                fi
                sleep 0.5
            done
            echo "    $name didn't stop gracefully, sending SIGKILL..."
            kill -9 "$pid" 2>/dev/null
            rm -f "$pidfile"
        else
            echo "[*] $name proxy (PID: $pid) is not running."
            rm -f "$pidfile"
        fi
    fi
}

stop_proxy "gateway_5440"
stop_proxy "gateway_5443"
stop_proxy "nvidia"

echo "[*] Krusch Agentic Proxy instances stopped."
