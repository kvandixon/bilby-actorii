#!/usr/bin/env bash
# Start the ActorII Docker Sandbox & systemd LLM Unix-Socket Bridge
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Stop actor-surface.service and remove legacy root container if present
systemctl --user stop actor-surface.service 2>/dev/null || true
pkill -f "$SCRIPT_DIR/websocket_server.py" 2>/dev/null || true
pkill -f "llm_socket_bridge.py --mode host" 2>/dev/null || true
docker rm -f actorii-sandbox 2>/dev/null || true

# 2. Ensure socket & container workspace directories exist for non-root bilby (UID 10001)
mkdir -p /tmp/actorii-llm-bridge "$SCRIPT_DIR/workspace"
chmod 0777 /tmp/actorii-llm-bridge "$SCRIPT_DIR/workspace"

# 3. Build and launch isolated bilby-actorii container
#    If Neo4j is already listening on host 127.0.0.1:7474 (bridged via /tmp/actorii-llm-bridge),
#    skip starting a duplicate neo4j container that would collide on 7474/7687.
if ss -tln 2>/dev/null | grep -q ':7474 '; then
    export ACTORII_NEO4J_HTTP_URL="${ACTORII_NEO4J_HTTP_URL:-http://127.0.0.1:7474/db/neo4j/tx/commit}"
    docker-compose up -d --build --no-deps bilby
else
    docker-compose up -d --build
fi

# 4. Sync optional host ~/.actorii/persona.md into container ~/.actorii/persona.md
if [ -f "$HOME/.actorii/persona.md" ]; then
    docker cp "$HOME/.actorii/persona.md" bilby-actorii:/home/bilby/.actorii/persona.md 2>/dev/null || true
fi

# 5. Reload and start actor-surface.service (supervises host-side LLM & Neo4j Unix socket bridge)
systemctl --user daemon-reload
systemctl --user start actor-surface.service

echo "ActorII v4.2.0 Sandbox Container (bilby-actorii) running:"
echo "  HTML5 Studio : http://127.0.0.1:8090"
echo "  WebSocket    : ws://127.0.0.1:8765"
echo "  TUI          : actor-tui"
