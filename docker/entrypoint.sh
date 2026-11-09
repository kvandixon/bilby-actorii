#!/usr/bin/env bash
set -e

# Seed initial sandbox RGB lighting config
/usr/local/bin/valkyrie-rgb-ctl status >/dev/null 2>&1 || true

# Start container-side Unix-socket -> localhost TCP bridge for LLM (:8080, :8082, :11434) & Neo4j (:7474)
if [ -d "/llm-bridge" ]; then
    python3 /app/docker/llm_socket_bridge.py --mode container --sock-dir /llm-bridge &
    sleep 0.3
fi

exec "$@"
