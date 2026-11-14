"""Tests for Bilby's HTTP/WebSocket Reverse Proxy (`actorii.reverse_proxy`), `Actor[#Proxy]`, and Telegram `/proxy`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from actorii.harness import bootstrap_agent_harness
from actorii.reverse_proxy import (
    COOKIE_NAME,
    BilbyReverseProxyServer,
    default_proxy_config,
    evaluate_client_access,
    load_proxy_config,
    save_proxy_config,
)
from actorii.telegram_gateway import BilbyTelegramGateway


def test_proxy_access_modes_cidrs_and_magic_link_cookie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "proxy_config.json"
    monkeypatch.setenv("ACTORII_PROXY_CONFIG", str(cfg_path))
    cfg = default_proxy_config()
    cfg["access_token"] = "magic-secret-token"
    save_proxy_config(cfg, cfg_path)

    # 1. Default `lan` mode: WLAN (192.168.68.90) & Tailscale (100.124.50.33) allowed; public IP blocked
    ok_wlan, code_wlan, _, _ = evaluate_client_access("192.168.68.90", cfg)
    assert ok_wlan is True and code_wlan == 200
    ok_ts, code_ts, _, _ = evaluate_client_access("100.124.50.33", cfg)
    assert ok_ts is True and code_ts == 200
    ok_pub, code_pub, _, _ = evaluate_client_access("8.8.8.8", cfg)
    assert ok_pub is False and code_pub == 403

    # 2. `tailscale_only` mode: blocks WLAN, allows Tailscale & loopback
    cfg["mode"] = "tailscale_only"
    assert evaluate_client_access("192.168.68.90", cfg)[0] is False
    assert evaluate_client_access("100.124.50.33", cfg)[0] is True
    assert evaluate_client_access("127.0.0.1", cfg)[0] is True

    # 3. `local_only` and `paused`: loopback still works; external gets 403 / 503
    cfg["mode"] = "local_only"
    assert evaluate_client_access("100.124.50.33", cfg)[:2] == (False, 403)
    assert evaluate_client_access("127.0.0.1", cfg)[0] is True

    cfg["mode"] = "paused"
    assert evaluate_client_access("192.168.68.90", cfg)[:2] == (False, 503)
    assert evaluate_client_access("127.0.0.1", cfg)[0] is True

    # 4. `require_auth=True` magic-link -> session cookie flow
    cfg["mode"] = "lan"
    cfg["require_auth"] = True
    assert evaluate_client_access("192.168.68.90", cfg, path_and_query="/benchmarks")[:2] == (False, 401)

    ok_magic, _, reason, cookie_digest = evaluate_client_access(
        "192.168.68.90", cfg, path_and_query="/benchmarks?pass=magic-secret-token"
    )
    assert ok_magic is True and reason == "magic-link" and cookie_digest

    ok_cookie, _, c_reason, _ = evaluate_client_access(
        "192.168.68.90",
        cfg,
        path_and_query="/api/benchmarks",
        headers={"Cookie": f"{COOKIE_NAME}={cookie_digest}"},
    )
    assert ok_cookie is True and c_reason == "session-cookie"


def test_proxy_twin_and_telegram_slash_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "proxy_config.json"
    monkeypatch.setenv("ACTORII_PROXY_CONFIG", str(cfg_path))
    save_proxy_config(default_proxy_config(), cfg_path)

    vm, _, _ = bootstrap_agent_harness()
    assert "ReverseProxyTwin" in str(vm.eval_chunk("summary(Actor[#Proxy])!!"))

    vm.eval_chunk('setMode(Actor[#Proxy], "tailscale_only")!!')
    assert load_proxy_config(cfg_path)["mode"] == "tailscale_only"

    vm.eval_chunk('blockIp(Actor[#Proxy], "192.168.68.200")!!')
    assert "192.168.68.200" in load_proxy_config(cfg_path)["blocked_ips"]

    gw = BilbyTelegramGateway(
        token="dummy:tok",
        allowed_users=[12345],
        state_path=tmp_path / "tg_state.json",
        api_caller=lambda *a, **kw: {"ok": True, "result": {}},
    )
    reply = gw.handle_text_command_or_goal("/proxy auth on")
    assert "ENABLED" in reply
    assert load_proxy_config(cfg_path)["require_auth"] is True

    reply_lan = gw.handle_text_command_or_goal("/proxy lan")
    assert "`lan`" in reply_lan
    assert load_proxy_config(cfg_path)["mode"] == "lan"


def test_live_reverse_proxy_forwarding_and_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "proxy_config.json"
    monkeypatch.setenv("ACTORII_PROXY_CONFIG", str(cfg_path))

    async def _run() -> None:
        # Spin up tiny dummy upstream HTTP server
        async def _upstream(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            head = await r.readuntil(b"\r\n\r\n")
            text = head.decode("latin-1")
            has_xff = "X-Forwarded-For:" in text
            body = f"OK:{has_xff}".encode("utf-8")
            w.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("latin-1") + body
            )
            await w.drain()
            w.close()

        up_srv = await asyncio.start_server(_upstream, "127.0.0.1", 0)
        up_port = up_srv.sockets[0].getsockname()[1]

        cfg = default_proxy_config()
        cfg["listen_host"] = "127.0.0.1"
        cfg["http_port"] = 0
        cfg["ws_port"] = 0
        cfg["upstream_http_port"] = up_port
        save_proxy_config(cfg, cfg_path)

        proxy = BilbyReverseProxyServer(config_path=cfg_path)
        px_srv = await asyncio.start_server(
            lambda r, w: proxy.handle_client(r, w, upstream_port=up_port, port_label="test"),
            "127.0.0.1",
            0,
        )
        px_port = px_srv.sockets[0].getsockname()[1]

        try:
            cr, cw = await asyncio.open_connection("127.0.0.1", px_port)
            cw.write(b"GET /benchmarks HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            await cw.drain()
            resp = await cr.read()
            cw.close()
            assert b"200 OK" in resp and b"OK:True" in resp
        finally:
            px_srv.close()
            await px_srv.wait_closed()
            up_srv.close()
            await up_srv.wait_closed()

    asyncio.run(_run())
