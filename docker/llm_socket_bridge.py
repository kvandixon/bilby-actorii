#!/usr/bin/env python3
"""Minimal Unix-domain-socket <-> TCP loopback bridge for `actorii-sandbox`.

Allows an isolated Docker container (without `--network=host`) to reach ONLY
explicitly bridged localhost ports (`127.0.0.1:8080` LLM and `127.0.0.1:7474` Neo4j)
via a shared socket directory (`/tmp/actorii-llm-bridge`), without exposing
any other host loopback services or requiring firewalld rules on `docker0`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Tuple


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_unix_to_tcp(
    client_r: asyncio.StreamReader,
    client_w: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    try:
        upstream_r, upstream_w = await asyncio.open_connection(host, port)
    except Exception:
        client_w.close()
        return
    await asyncio.gather(_pipe(client_r, upstream_w), _pipe(upstream_r, client_w))


async def _handle_tcp_to_unix(
    client_r: asyncio.StreamReader,
    client_w: asyncio.StreamWriter,
    sock_path: str,
) -> None:
    try:
        upstream_r, upstream_w = await asyncio.open_unix_connection(sock_path)
    except Exception:
        client_w.close()
        return
    await asyncio.gather(_pipe(client_r, upstream_w), _pipe(upstream_r, client_w))


async def run_host_side(sock_dir: Path, ports: List[int]) -> None:
    sock_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(sock_dir, 0o777)
    servers = []
    for port in ports:
        sock_file = sock_dir / f"port_{port}.sock"
        if sock_file.exists():
            sock_file.unlink()
        srv = await asyncio.start_unix_server(
            lambda r, w, p=port: _handle_unix_to_tcp(r, w, "127.0.0.1", p),
            path=str(sock_file),
        )
        os.chmod(sock_file, 0o666)
        servers.append(srv)
        print(f"[host-bridge] {sock_file} -> 127.0.0.1:{port}", flush=True)
    await asyncio.Future()


async def run_container_side(sock_dir: Path, ports: List[int]) -> None:
    servers = []
    for port in ports:
        sock_file = sock_dir / f"port_{port}.sock"
        srv = await asyncio.start_server(
            lambda r, w, s=str(sock_file): _handle_tcp_to_unix(r, w, s),
            host="127.0.0.1",
            port=port,
        )
        servers.append(srv)
        print(f"[container-bridge] 127.0.0.1:{port} -> {sock_file}", flush=True)
    await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("host", "container"), required=True)
    parser.add_argument("--sock-dir", default="/tmp/actorii-llm-bridge")
    parser.add_argument("--ports", nargs="+", type=int, default=[8080, 8082, 8084, 11434, 7474])
    args = parser.parse_args()

    sdir = Path(args.sock_dir)
    if args.mode == "host":
        asyncio.run(run_host_side(sdir, args.ports))
    else:
        asyncio.run(run_container_side(sdir, args.ports))


if __name__ == "__main__":
    main()
