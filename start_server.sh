#!/usr/bin/env bash
# Start the ActorII Docker Sandbox & systemd LLM Unix-Socket Bridge
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Stop actor-surface.service while rebuilding/recreating the container
systemctl --user stop actor-surface.service 2>/dev/null || true
pkill -f "$SCRIPT_DIR/websocket_server.py" 2>/dev/null || true
pkill -f "llm_socket_bridge.py --mode host" 2>/dev/null || true

# 2. Ensure socket directory exists
mkdir -p /tmp/actorii-llm-bridge
chmod 700 /tmp/actorii-llm-bridge

# 3. Build and launch the isolated actorii-sandbox Docker container
docker-compose up -d --build

# 4. Reload and start actor-surface.service (which supervises the host-side LLM socket bridge)
systemctl --user daemon-reload
systemctl --user start actor-surface.service

echo "ActorII Sandbox Container running:"
echo "  HTML5 Studio : http://127.0.0.1:8090"
echo "  WebSocket    : ws://127.0.0.1:8765"
echo "  TUI          : actor-tui"
