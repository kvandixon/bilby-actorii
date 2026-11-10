FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ACTORII_IN_DOCKER=1 \
    ACTORII_HOST_LOCKDOWN=1 \
    ACTOR_SURFACE_HOST=0.0.0.0 \
    ACTOR_SURFACE_PORT=8765

WORKDIR /app

# Install git/file utilities + Tier-1 read-only research dependencies (PDF & Web)
RUN apt-get update && apt-get install -y --no-install-recommends git file ca-certificates && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir \
        websockets \
        pytest \
        mpmath \
        pandas \
        pyarrow \
        pypdf \
        pyyaml \
        scipy \
        cryptography \
        trafilatura \
        duckduckgo-search

COPY pyproject.toml README.md websocket_server.py browser_client.html index.html thermometer_surface.html actor_tui.py actor_telegram.py /app/
COPY benchmarks/scoreboard.json benchmarks/scoreboard.html benchmarks/scoreboard.py /app/benchmarks/
COPY src /app/src
COPY examples /app/examples
COPY docker /app/docker

# Create unprivileged bilby user (UID 10001) and writable /workspace + /data directories
RUN groupadd -g 10001 bilby && \
    useradd -u 10001 -g bilby -m -d /home/bilby -s /bin/bash bilby && \
    chmod +x /app/docker/valkyrie-rgb-ctl /app/docker/llm_socket_bridge.py /app/docker/entrypoint.sh /app/actor_tui.py /app/actor_telegram.py && \
    cp /app/docker/valkyrie-rgb-ctl /usr/local/bin/valkyrie-rgb-ctl && \
    ln -sf /app/actor_tui.py /usr/local/bin/actor-tui && \
    ln -sf /app/actor_tui.py /usr/local/bin/bilby && \
    mkdir -p /etc/valkyrie-rgb /data /workspace /home/bilby/.actorii /app/examples/fileout && \
    chown -R bilby:bilby /etc/valkyrie-rgb /data /workspace /home/bilby /app/examples/fileout && \
    chmod -R a-w /app/src /app/docker && \
    /usr/local/bin/valkyrie-rgb-ctl status

USER bilby
EXPOSE 8090 8765

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python3", "/app/websocket_server.py"]
