#!/usr/bin/env bash
# Start the ActorII Docker Sandbox, LLM Unix-Socket Bridge, & Bilby WLAN/Tailscale Reverse Proxy
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Stop services and remove legacy container if present
systemctl --user stop bilby-reverse-proxy.service 2>/dev/null || true
systemctl --user stop actor-surface.service 2>/dev/null || true
pkill -f "$SCRIPT_DIR/websocket_server.py" 2>/dev/null || true
pkill -f "llm_socket_bridge.py --mode host" 2>/dev/null || true
docker rm -f actorii-sandbox 2>/dev/null || true

# 2. Ensure socket, proxy state, & container workspace directories exist
mkdir -p /tmp/actorii-llm-bridge "$SCRIPT_DIR/workspace" "$HOME/.actorii"
chmod 0777 /tmp/actorii-llm-bridge "$SCRIPT_DIR/workspace" "$HOME/.actorii"

# 3. Build and launch isolated bilby-actorii container (bound strictly to 127.0.0.1:18090 / 127.0.0.1:18765)
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

# 5. Reload and start host bridge + Bilby Reverse Proxy (:8090 / :8765)
systemctl --user daemon-reload
systemctl --user enable --now actor-surface.service >/dev/null 2>&1 || systemctl --user start actor-surface.service
if systemctl --user cat bilby-reverse-proxy.service >/dev/null 2>&1; then
    systemctl --user enable --now bilby-reverse-proxy.service >/dev/null 2>&1 || systemctl --user start bilby-reverse-proxy.service
fi

PYTHONPATH="$SCRIPT_DIR/src" python3 -c 'from actorii.harness.twins import ReverseProxyTwin; print(ReverseProxyTwin().status())'
