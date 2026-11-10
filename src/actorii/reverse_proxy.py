"""Bilby-Managed HTTP & WebSocket Reverse Proxy (`actorii.reverse_proxy`).

Exposes the localhost-bound `bilby-actorii` container (`127.0.0.1:18090` / `127.0.0.1:18765`)
to authorized WLAN (`192.168.x.x`) and Tailscale (`100.64.0.0/10`) devices on `:8090` and `:8765`
while enforcing hot-reloaded access policies controlled live by Bilby (`Actor[#Proxy]` & `/proxy`
in Telegram and TUI):

  - Modes: `lan` (WLAN + Tailscale + loopback), `tailscale_only`, `local_only`, `paused`
  - CIDR allowlist & instant per-IP blocklist
  - Optional 1-tap Telegram Magic-Link / session-cookie gate (`require_auth`)
  - Structured JSONL connection & security audit log (`proxy_access.jsonl`)
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import urllib.parse
from typing import Any

DEFAULT_ALLOWED_CIDRS = [
    "127.0.0.0/8",
    "::1/128",
    "192.168.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "100.64.0.0/10",
]

VALID_MODES = ("lan", "tailscale_only", "local_only", "paused")
COOKIE_NAME = "bilby_proxy_session"


def resolve_proxy_state_dir() -> Path:
    override = os.environ.get("ACTORII_PROXY_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    docker_shared = Path("/proxy-state")
    if docker_shared.is_dir():
        return docker_shared
    return Path.home() / ".actorii"


def get_proxy_config_path() -> Path:
    override = os.environ.get("ACTORII_PROXY_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return resolve_proxy_state_dir() / "proxy_config.json"


def get_proxy_log_path() -> Path:
    return get_proxy_config_path().parent / "proxy_access.jsonl"


def default_proxy_config() -> dict[str, Any]:
    return {
        "mode": "lan",
        "listen_host": "0.0.0.0",
        "http_port": 8090,
        "ws_port": 8765,
        "upstream_host": "127.0.0.1",
        "upstream_http_port": 18090,
        "upstream_ws_port": 18765,
        "require_auth": False,
        "access_token": secrets.token_urlsafe(18),
        "allowed_cidrs": list(DEFAULT_ALLOWED_CIDRS),
        "blocked_ips": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_proxy_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or get_proxy_config_path()
    base = default_proxy_config()
    if cfg_path.is_file():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                base.update(loaded)
        except Exception:
            pass
    else:
        save_proxy_config(base, cfg_path)
    if not base.get("access_token"):
        base["access_token"] = secrets.token_urlsafe(18)
    if base.get("mode") not in VALID_MODES:
        base["mode"] = "lan"
    return base


def save_proxy_config(cfg: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or get_proxy_config_path()
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, cfg_path)
    return cfg


def append_proxy_log(entry: dict[str, Any], log_path: Path | None = None) -> None:
    lp = log_path or get_proxy_log_path()
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with lp.open("a", encoding="utf-8") as fh:
            fh.write(line)
        # Keep log bounded (~500 KB max)
        if lp.stat().st_size > 512_000:
            lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
            lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def read_recent_proxy_logs(limit: int = 10, log_path: Path | None = None) -> list[dict[str, Any]]:
    lp = log_path or get_proxy_log_path()
    if not lp.is_file():
        return []
    try:
        raw_lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for ln in raw_lines[-max(1, min(100, int(limit))):]:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []


def discover_host_network_ips() -> dict[str, str]:
    """Discover primary WLAN/LAN IPv4 and Tailscale IPv4 addresses on the host."""
    cfg = load_proxy_config()
    cached = cfg.get("host_ips")
    ips: dict[str, str] = {}

    # Probe routing table via UDP socket (no packets sent) for primary LAN IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.68.1", 80))
            cand = s.getsockname()[0]
            if cand and not cand.startswith("127.") and not cand.startswith("172."):
                ips["lan"] = cand
    except Exception:
        pass

    # Scan `/proc/net/fib_trie` or `socket.getaddrinfo` / `ip -4 addr` if available
    try:
        import subprocess

        proc = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.split()
                if len(parts) < 4:
                    continue
                iface = parts[1]
                cidr = parts[3]
                ip_str = cidr.split("/")[0]
                if iface.startswith(("wl", "en", "eth")) and not ip_str.startswith("127."):
                    ips["lan"] = ip_str
                elif iface.startswith("tailscale") or ip_str.startswith("100."):
                    ips["tailscale"] = ip_str
    except Exception:
        pass

    if not ips and isinstance(cached, dict):
        return {str(k): str(v) for k, v in cached.items() if v}
    return ips


def _normalize_ip(addr: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    raw = (addr or "").strip()
    if not raw:
        return None
    try:
        ip_obj = ipaddress.ip_address(raw)
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            return ip_obj.ipv4_mapped
        return ip_obj
    except ValueError:
        return None


def _session_cookie_digest(access_token: str) -> str:
    return hmac.new(
        access_token.encode("utf-8"),
        b"bilby-proxy-session-v1",
        hashlib.sha256,
    ).hexdigest()


def evaluate_client_access(
    client_ip: str,
    cfg: dict[str, Any],
    *,
    path_and_query: str = "/",
    headers: dict[str, str] | None = None,
) -> tuple[bool, int, str, str | None]:
    """Evaluate whether a client request is permitted.

    Returns `(allowed, status_code_if_denied, reason, set_cookie_token_if_upgraded)`.
    """
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    ip_obj = _normalize_ip(client_ip)
    if ip_obj is None:
        return False, 403, "Invalid client IP", None

    # 1. Loopback is always permitted so local CLI/TUI/health checks never lock out
    if ip_obj.is_loopback:
        return True, 200, "loopback", None

    # 2. Explicit IP blocklist
    for blocked in cfg.get("blocked_ips") or []:
        b_str = str(blocked).strip()
        if not b_str:
            continue
        try:
            if "/" in b_str:
                if ip_obj in ipaddress.ip_network(b_str, strict=False):
                    return False, 403, f"Blocked by IP rule ({b_str})", None
            elif ip_obj == _normalize_ip(b_str):
                return False, 403, f"Blocked IP ({b_str})", None
        except ValueError:
            continue

    # 3. Proxy operational mode
    mode = str(cfg.get("mode", "lan")).lower()
    if mode == "paused":
        return False, 503, "Proxy paused by Bilby", None
    if mode == "local_only":
        return False, 403, "Proxy locked to localhost only", None
    if mode == "tailscale_only":
        ts_net = ipaddress.ip_network("100.64.0.0/10")
        if not (isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in ts_net):
            return False, 403, "Proxy restricted to Tailscale VPN (100.64.0.0/10)", None
    else:
        # `lan` mode: check allowed_cidrs
        matched = False
        for cidr in cfg.get("allowed_cidrs") or DEFAULT_ALLOWED_CIDRS:
            try:
                net = ipaddress.ip_network(str(cidr).strip(), strict=False)
                if ip_obj.version == net.version and ip_obj in net:
                    matched = True
                    break
            except ValueError:
                continue
        if not matched:
            return False, 403, f"Client IP {ip_obj} outside allowed CIDRs", None

    # 4. Optional token / magic-link session cookie gate
    if bool(cfg.get("require_auth", False)):
        token = str(cfg.get("access_token") or "").strip()
        if token:
            expected_cookie = _session_cookie_digest(token)
            # Check Bearer header
            auth_hdr = hdrs.get("authorization", "")
            if auth_hdr.lower().startswith("bearer "):
                cand = auth_hdr[7:].strip()
                if hmac.compare_digest(cand, token):
                    return True, 200, "bearer-token", None

            # Check Cookie header
            cookie_hdr = hdrs.get("cookie", "")
            for part in cookie_hdr.split(";"):
                if "=" in part:
                    ck, cv = part.strip().split("=", 1)
                    if ck == COOKIE_NAME and hmac.compare_digest(cv.strip(), expected_cookie):
                        return True, 200, "session-cookie", None

            # Check `?pass=<token>` or `?token=<token>` magic link query parameter
            parsed = urllib.parse.urlsplit(path_and_query or "/")
            qs = urllib.parse.parse_qs(parsed.query)
            q_tok = (qs.get("pass") or qs.get("token") or [""])[0].strip()
            if q_tok and hmac.compare_digest(q_tok, token):
                return True, 200, "magic-link", expected_cookie

            return False, 401, "Authentication required (request magic link via `/proxy links` in Telegram)", None

    return True, 200, f"allowed-{mode}", None


def _build_error_http_response(status: int, reason_phrase: str, message: str, *, show_login: bool = False) -> bytes:
    if show_login:
        body_str = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bilby Proxy Gate</title>
<style>
body{{background:#0b0f19;color:#e5e7eb;font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:16px}}
.box{{background:#111827;border:1px solid #1f2d47;border-radius:12px;padding:24px;max-width:400px;width:100%}}
h2{{margin:0 0 8px;color:#38bdf8;font-size:1.2rem}}
p{{font-size:0.86rem;color:#9ca3af;margin:0 0 16px}}
input{{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #1f2d47;background:#070a12;color:#fff;margin-bottom:12px}}
button{{width:100%;padding:10px;border-radius:8px;border:none;background:#0284c7;color:#fff;font-weight:700;cursor:pointer}}
</style></head>
<body><div class="box">
<h2>Bilby Reverse Proxy Gate</h2>
<p>{message}</p>
<form method="GET" action="/">
<input type="password" name="pass" placeholder="Enter proxy passcode..." autofocus>
<button type="submit">Unlock Session</button>
</form>
</div></body></html>"""
        ctype = "text/html; charset=utf-8"
    else:
        body_str = json.dumps({"status": "error", "code": status, "error": message})
        ctype = "application/json; charset=utf-8"

    body = body_str.encode("utf-8")
    headers = [
        f"HTTP/1.1 {status} {reason_phrase}",
        f"Content-Type: {ctype}",
        f"Content-Length: {len(body)}",
        "Cache-Control: no-store",
        "Connection: close",
        "",
        "",
    ]
    return "\r\n".join(headers).encode("latin-1") + body


async def _pipe_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


class BilbyReverseProxyServer:
    """Async Layer-7 HTTP + WebSocket reverse proxy with hot-reloaded Bilby policy."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or get_proxy_config_path()
        self.active_connections = 0
        self._servers: list[asyncio.AbstractServer] = []

    async def handle_client(
        self,
        client_r: asyncio.StreamReader,
        client_w: asyncio.StreamWriter,
        *,
        upstream_port: int,
        port_label: str,
    ) -> None:
        peer = client_w.get_extra_info("peername")
        client_ip = str(peer[0]) if isinstance(peer, tuple) and peer else "unknown"
        cfg = load_proxy_config(self.config_path)
        upstream_host = str(cfg.get("upstream_host", "127.0.0.1"))

        try:
            raw_head = await asyncio.wait_for(client_r.readuntil(b"\r\n\r\n"), timeout=10.0)
        except Exception:
            client_w.close()
            return

        head_text = raw_head.decode("latin-1", errors="replace")
        lines = head_text.split("\r\n")
        req_line = lines[0] if lines else ""
        parts = req_line.split(" ")
        method = parts[0] if len(parts) >= 1 else "GET"
        target_path = parts[1] if len(parts) >= 2 else "/"

        headers: dict[str, str] = {}
        header_lines_out: list[str] = []
        for ln in lines[1:]:
            if not ln:
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                kl = k.strip().lower()
                headers[kl] = v.strip()
                if kl in ("x-forwarded-for", "x-real-ip"):
                    continue
            header_lines_out.append(ln)

        allowed, deny_code, reason, set_cookie_digest = evaluate_client_access(
            client_ip,
            cfg,
            path_and_query=target_path,
            headers=headers,
        )

        if not _normalize_ip(client_ip) or not _normalize_ip(client_ip).is_loopback:
            append_proxy_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "ip": client_ip,
                    "port": port_label,
                    "method": method,
                    "path": target_path.split("?", 1)[0],
                    "allowed": allowed,
                    "status": 200 if allowed else deny_code,
                    "reason": reason,
                },
                log_path=self.config_path.parent / "proxy_access.jsonl",
            )

        if not allowed:
            phrase = {401: "Unauthorized", 403: "Forbidden", 503: "Service Unavailable"}.get(deny_code, "Forbidden")
            is_ws = headers.get("upgrade", "").lower() == "websocket"
            resp = _build_error_http_response(
                deny_code,
                phrase,
                reason,
                show_login=(deny_code == 401 and method == "GET" and not is_ws),
            )
            client_w.write(resp)
            await client_w.drain()
            client_w.close()
            return

        # If authenticated via `?pass=<token>` on a browser GET, set session cookie and redirect to clean URL
        if set_cookie_digest and method == "GET" and headers.get("upgrade", "").lower() != "websocket":
            parsed = urllib.parse.urlsplit(target_path)
            qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            qs.pop("pass", None)
            clean_query = urllib.parse.urlencode(qs, doseq=True)
            clean_target = urllib.parse.urlunsplit(("", "", parsed.path or "/", clean_query, ""))
            redir = "\r\n".join([
                "HTTP/1.1 302 Found",
                f"Location: {clean_target}",
                f"Set-Cookie: {COOKIE_NAME}={set_cookie_digest}; HttpOnly; Path=/; Max-Age=604800; SameSite=Lax",
                "Content-Length: 0",
                "Cache-Control: no-store",
                "Connection: close",
                "",
                "",
            ]).encode("latin-1")
            client_w.write(redir)
            await client_w.drain()
            client_w.close()
            return

        try:
            up_r, up_w = await asyncio.wait_for(
                asyncio.open_connection(upstream_host, upstream_port),
                timeout=5.0,
            )
        except Exception as exc:
            err = _build_error_http_response(502, "Bad Gateway", f"Upstream Studio unavailable: {exc}")
            client_w.write(err)
            await client_w.drain()
            client_w.close()
            return

        header_lines_out.append(f"X-Forwarded-For: {client_ip}")
        header_lines_out.append(f"X-Real-IP: {client_ip}")
        rebuilt_head = "\r\n".join([req_line, *header_lines_out, "", ""]).encode("latin-1")
        up_w.write(rebuilt_head)
        await up_w.drain()

        self.active_connections += 1
        try:
            await asyncio.gather(_pipe_stream(client_r, up_w), _pipe_stream(up_r, client_w))
        finally:
            self.active_connections = max(0, self.active_connections - 1)

    async def start(self) -> None:
        cfg = load_proxy_config(self.config_path)
        host_ips = discover_host_network_ips()
        if host_ips:
            cfg["host_ips"] = host_ips
            save_proxy_config(cfg, self.config_path)

        listen_host = str(cfg.get("listen_host", "0.0.0.0"))
        http_port = int(cfg.get("http_port", 8090))
        ws_port = int(cfg.get("ws_port", 8765))
        up_http = int(cfg.get("upstream_http_port", 18090))
        up_ws = int(cfg.get("upstream_ws_port", 18765))

        srv_http = await asyncio.start_server(
            lambda r, w: self.handle_client(r, w, upstream_port=up_http, port_label=str(http_port)),
            host=listen_host,
            port=http_port,
        )
        self._servers.append(srv_http)
        print(f"[bilby-proxy] Listening on {listen_host}:{http_port} -> 127.0.0.1:{up_http} (mode={cfg['mode']})", flush=True)

        if ws_port != http_port and ws_port > 0:
            srv_ws = await asyncio.start_server(
                lambda r, w: self.handle_client(r, w, upstream_port=up_ws, port_label=str(ws_port)),
                host=listen_host,
                port=ws_port,
            )
            self._servers.append(srv_ws)
            print(f"[bilby-proxy] Listening on {listen_host}:{ws_port} -> 127.0.0.1:{up_ws}", flush=True)

    async def close(self) -> None:
        for s in self._servers:
            s.close()
            await s.wait_closed()
        self._servers.clear()


async def _amain(config_path: Path | None) -> None:
    proxy = BilbyReverseProxyServer(config_path=config_path)
    await proxy.start()
    await asyncio.Future()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bilby-managed HTTP & WebSocket reverse proxy.")
    parser.add_argument("--config", type=Path, default=None, help="Path to proxy_config.json")
    args = parser.parse_args()
    asyncio.run(_amain(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
