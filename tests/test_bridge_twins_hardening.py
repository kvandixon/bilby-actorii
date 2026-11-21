"""Focused regression & hardening tests for ActorII bridge, host_bridge, twins, and `.act` modules."""

from __future__ import annotations

import http.server
import io
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from actorii.bridge import (
    ActorBridgeObject,
    actor_bridge,
    actor_command,
    actor_query,
    expose_python_object,
)
from actorii.harness.host_bridge import (
    HostWorkspaceBridge,
    VMReflectionBridge,
    validate_safe_shell_command,
)
from actorii.harness.twins import (
    JetsonEdgeTwin,
    NetTransportBridge,
    NetworkTopologyBridge,
    PantryGraphTwin,
    ReverseEngineeringTwin,
    ValkyrieSystemTwin,
    _SafeRedirectHandler,
    _validate_probe_target_url,
    _validate_public_http_url,
)
from actorii.vm import ActorVM


# ---------------------------------------------------------------------------
# 1. actorii.bridge hardening (aliases, bounded audit_log, lazy sync)
# ---------------------------------------------------------------------------


def test_bridge_aliased_query_and_command_dispatch_and_bounded_audit():
    prop_calls = 0

    @actor_bridge(name="AliasDemo")
    class AliasDemo:
        def __init__(self) -> None:
            self.counter = 0

        @actor_query(name="aliasedRead")
        def impl_read(self, x: int) -> int:
            return x * 2

        @actor_command(name="aliasedInc")
        def impl_inc(self, delta: int = 1) -> int:
            self.counter += delta
            return self.counter

        @actor_query(name="expensiveProp")
        @property
        def prop(self) -> int:
            nonlocal prop_calls
            prop_calls += 1
            return self.counter

    vm = ActorVM()
    demo = AliasDemo()
    bridge = expose_python_object(vm, demo, global_key="AliasDemo")

    # __init__ synced properties once
    assert prop_calls == 1

    # Aliased query works & does NOT re-evaluate expensive property
    assert bridge.invoke_bridge_member("aliasedRead", 21) == 42
    assert prop_calls == 1

    # Aliased command works & DOES sync properties after Tier-2 mutation
    assert bridge.invoke_bridge_member("aliasedInc", 5) == 5
    assert prop_calls == 2
    assert bridge._fields["counter"] == 5

    # Audit log is bounded to 500 entries and exposes duration_ms/ok via recentAuditLog
    for i in range(520):
        bridge.invoke_bridge_member("aliasedRead", i)
    assert len(bridge.audit_log) == 500
    recent = bridge.recentAuditLog(3)
    assert len(recent) == 3
    assert recent[-1]["selector"] == "aliasedRead"
    assert recent[-1]["ok"] is True
    assert recent[-1]["duration_ms"] >= 0.0


# ---------------------------------------------------------------------------
# 2. host_bridge hardening
# ---------------------------------------------------------------------------


def test_validate_safe_shell_command_blocks_whitespace_and_flag_variants():
    for bad in (
        "rm  -rf  /",
        "rm -fr /",
        "rm -r -f ~",
        "kill -KILL 1234",
        "kill -s 9 1234",
        "systemctl\tstop foo",
        "sudo\tid",
    ):
        with pytest.raises(PermissionError):
            validate_safe_shell_command(bad)


def test_host_bridge_search_file_flag_injection_and_extensions(tmp_path: Path):
    vm = ActorVM()
    hb = HostWorkspaceBridge(vm, allowed_roots=[tmp_path], max_output_chars=1200)
    (tmp_path / "sample.act").write_text("Def hello(self) { ^'-v flag test'; }!!\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("marker -v line\n", encoding="utf-8")

    # Pattern starting with '-' must not be treated as a grep CLI flag
    out = hb.searchFile(str(tmp_path), "-v")
    assert "sample.act" in out
    assert "notes.md" in out


def test_host_bridge_read_symbol_dotted_and_truncated(tmp_path: Path):
    vm = ActorVM()
    hb = HostWorkspaceBridge(vm, allowed_roots=[tmp_path], max_output_chars=200)
    src = (
        "class Alpha:\n"
        "    def run(self):\n"
        "        return 'alpha'\n\n"
        "class Beta:\n"
        "    def run(self):\n"
        "        return 'beta'\n"
    )
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")

    out = hb.readSymbol(str(f), "Beta.run")
    assert "return 'beta'" in out
    assert "return 'alpha'" not in out


def test_host_bridge_preview_edit_preserves_outer_shadow_vfs(tmp_path: Path):
    vm = ActorVM()
    hb = HostWorkspaceBridge(vm, allowed_roots=[tmp_path])
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello world\n", encoding="utf-8")
    f2.write_text("foo bar\n", encoding="utf-8")

    vm._in_shadow_run = True
    try:
        hb.writeFile(str(f1), "staged-a\n")
        assert hb.vfsOverlayCount() == 1
        _ = hb.previewEdit(str(f2), "foo", "baz")
        # Outer shadow VFS entry for f1 must survive previewEdit!
        assert hb.vfsOverlayCount() == 1
        assert "staged-a" in hb.readFile(str(f1), 1, 10)
    finally:
        vm._in_shadow_run = False
        hb.clear_shadow_vfs()


def test_vm_reflection_source_of_blocks_unexposed_internal_methods():
    vm = ActorVM()
    twin = ValkyrieSystemTwin()
    expose_python_object(vm, twin, global_key="Valkyrie")
    refl = VMReflectionBridge(vm)

    # Exposed method is visible
    exposed = refl.sourceOf("Valkyrie", "gpuStatus")
    assert "Bridge Actor[#Valkyrie]>>gpuStatus" in exposed

    # Unexposed internal helper must NOT be leaked
    leaked = refl.sourceOf("Valkyrie", "_query_nvidia_smi")
    assert "not found" in leaked


# ---------------------------------------------------------------------------
# 3. twins.py SSRF, UnboundLocalError, WebSocket EOF, Jetson, Topology fixes
# ---------------------------------------------------------------------------


def test_ssrf_guards_block_ipv4_mapped_ipv6_and_redirects():
    with pytest.raises(PermissionError):
        _validate_probe_target_url("http://[::ffff:127.0.0.1]:8090/", allowed_schemes=("http", "https"))
    with pytest.raises(PermissionError):
        _validate_public_http_url("http://[::ffff:127.0.0.1]:8090/")
    with pytest.raises(PermissionError):
        _validate_public_http_url("http://100.64.0.1/")

    handler = _SafeRedirectHandler(lambda u: _validate_public_http_url(u, label="redir"))
    dummy_req = urllib.request.Request("https://example.com/start")
    with pytest.raises(PermissionError):
        handler.redirect_request(dummy_req, None, 302, "Found", {}, "http://127.0.0.1:7474/db/neo4j")


def test_net_transport_empty_results_no_unbound_local_error():
    net = NetTransportBridge()

    class _FakeResp:
        def __init__(self, payload: dict) -> None:
            self._data = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with mock.patch("actorii.harness.twins._validate_public_http_url", side_effect=lambda u, **kw: u):
        with mock.patch("actorii.harness.twins._safe_urlopen", return_value=_FakeResp({"data": {}})):
            r1 = net.firecrawlPostScrape("https://api.firecrawl.dev", "https://example.com")
            assert r1.at("ok") is False
            assert r1.at("error")

        with mock.patch("actorii.harness.twins._safe_urlopen", return_value=_FakeResp({"content": ""})):
            with mock.patch("urllib.request.urlopen", return_value=_FakeResp({"result": {"content": []}})):
                r2 = net.keylessRingExtract("https://example.com")
                assert r2.at("ok") is False
                assert r2.at("error")


def test_probe_websocket_eof_does_not_infinite_loop():
    """Server accepts WS upgrade, sends 2-byte header with plen=126, then closes immediately."""
    import base64
    import hashlib

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    def _serve():
        conn, _ = srv.accept()
        try:
            data = conn.recv(4096).decode("iso-8859-1", errors="replace")
            key = ""
            for ln in data.splitlines():
                if ln.lower().startswith("sec-websocket-key:"):
                    key = ln.split(":", 1)[1].strip()
            acc = base64.b64encode(
                hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
            ).decode("ascii")
            resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {acc}\r\n\r\n"
            ).encode("ascii")
            # Send truncated 126-length frame header and immediately close socket
            conn.sendall(resp + b"\x81\x7e")
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    net = NetTransportBridge()
    with mock.patch.dict(os.environ, {"ACTORII_ALLOW_PRIVATE_URLS": "1"}):
        t0 = time.monotonic()
        res = net.probeWebSocket(f"ws://127.0.0.1:{port}/ws", "", 3)
        elapsed = time.monotonic() - t0
    t.join(timeout=2.0)
    assert elapsed < 2.5
    assert res.at("ok") is True


def test_reverse_engineering_and_valkyrie_enforce_forbidden_read_prefixes():
    re_twin = ReverseEngineeringTwin()
    assert "BLOCKED:" in re_twin.fileIdentity("/etc/shadow")
    assert "BLOCKED:" in re_twin.staticAudit("/etc/shadow")
    assert "BLOCKED:" in re_twin.detonateSandbox("/etc/shadow")

    valk = ValkyrieSystemTwin()
    with mock.patch.object(valk, "isSandboxed", return_value=True):
        assert "BLOCKED:" in valk.readTextFile("/proc/self/environ")


def test_pantry_graph_twin_surfaces_neo4j_errors():
    pantry = PantryGraphTwin()

    class _ErrResp:
        def read(self) -> bytes:
            return json.dumps({"results": [], "errors": [{"code": "Neo.SyntaxError", "message": "bad cypher"}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with mock.patch("urllib.request.urlopen", return_value=_ErrResp()):
        with pytest.raises(RuntimeError, match="bad cypher"):
            pantry._run_cypher("INVALID CYPHER")


def test_jetson_guardrails_and_deploy_service_quoting(tmp_path: Path):
    jt = JetsonEdgeTwin(state_dir=tmp_path)
    # apt-get dist-upgrade is blocked
    assert jt._validate_jetson_command("sudo apt-get dist-upgrade") is not None
    # word boundary prevents false positives on "asphalt" / "halted" in commands
    assert jt._validate_jetson_command("echo asphalt halted") is None
    # real "halt" command is still blocked
    assert jt._validate_jetson_command("sudo halt") is not None
    # source files containing "shutdown_handler" are allowed
    assert jt._validate_jetson_command("def shutdown_handler(): pass", is_source_file=True) is None
    # deployService blocks shell metacharacters in work_dir
    res = jt.deployService("demo", "$HOME/dir; echo pwned", "python3 -m http.server 8099")
    assert res.at("ok") is False
    assert "metacharacters" in str(res.at("stderr"))


def test_network_topology_expose_lan_port_blocks_link_local_and_ipv6(tmp_path: Path):
    topo = NetworkTopologyBridge(state_dir=tmp_path)
    r_ll = topo.exposeLanPort("169.254.169.254", 80, 8095)
    assert r_ll.at("ok") is False
    assert "RELAY_SSRF_GUARD" in str(r_ll.at("error"))

    r_v6 = topo.exposeLanPort("fd00::1", 80, 8095)
    assert r_v6.at("ok") is False
    assert "RELAY_SSRF_GUARD" in str(r_v6.at("error"))


# ---------------------------------------------------------------------------
# 4. .act regression checks (oop_foundation, web_protocols)
# ---------------------------------------------------------------------------


def test_oop_foundation_max_column_handles_nil_and_count_where_above():
    vm = ActorVM()
    act_path = Path(__file__).resolve().parents[1] / "src/actorii/harness/oop_foundation.act"
    vm.eval_chunk(act_path.read_text(encoding="utf-8"))
    res = vm.eval_chunk(
        """
        gs := initService(new(GraphService), "Test");
        rows := new(OrderedCollection);
        d1 := new(Dictionary, 4); atPut(d1, #score, nil); add(rows, d1);
        d2 := new(Dictionary, 4); atPut(d2, #score, 10); add(rows, d2);
        d3 := new(Dictionary, 4); atPut(d3, #score, 25); add(rows, d3);
        (maxColumn(gs, rows, #score) * 10) + countWhereAbove(gs, rows, #score, 12)!!
        """
    )
    assert res == 251


def test_web_protocols_mjpeg_python_blueprint_imports_time():
    vm = ActorVM()
    base = Path(__file__).resolve().parents[1] / "src/actorii/harness"
    vm.eval_chunk((base / "oop_foundation.act").read_text(encoding="utf-8"))
    vm.eval_chunk((base / "web_protocols.act").read_text(encoding="utf-8"))
    bp = vm.eval_chunk(
        """
        suite := initSuite(new(ProtocolSuite), nil);
        blueprint(suite, #mjpeg, #python)!!
        """
    )
    assert "import subprocess, threading, time" in str(bp)
