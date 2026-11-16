"""Real-socket conformance & Actor 4.1 OOP tests for `Actor[#Protocols]`, `Actor[#Net]` wire probes, and `Actor[#Jetson]` services."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from actorii.harness import bootstrap_agent_harness
from actorii.tui import ActorTUISession


def _make_minimal_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Build a syntactically valid baseline JFIF (`SOI` + `APP0` + `SOF0` + `EOI`) with exact dimensions."""
    soi = b"\xff\xd8"
    app0_payload = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app0 = b"\xff\xe0" + struct.pack(">H", len(app0_payload) + 2) + app0_payload
    sof0_payload = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    sof0 = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload
    eoi = b"\xff\xd9"
    return soi + app0 + sof0 + eoi


def test_protocol_suite_actor_hierarchy_rtsp_send_receive_and_subclassing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `Actor[#Protocols]` compiles cleanly, exposes RTSP send/receive & MJPEG blueprints, and supports `inherit(ProtocolSpec, ...)`."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)

    listing = str(vm.eval_chunk("listProtocols(Actor[#Protocols])!!"))
    assert "#mjpeg" in listing
    assert "#rtsp" in listing
    assert "#sse" in listing
    assert "#ws" in listing
    assert "#hls" in listing

    rtsp_spec = str(vm.eval_chunk("spec(Actor[#Protocols], #rtsp)!!"))
    assert "omxh264enc" in rtsp_spec
    assert "nvv4l2h264enc" in rtsp_spec
    assert "RTSP/1.0" in rtsp_spec

    rtsp_send = str(vm.eval_chunk("blueprint(Actor[#Protocols], #rtsp, #send)!!"))
    assert "RTSP SEND" in rtsp_send
    assert "omxh264enc" in rtsp_send
    assert "rtph264pay" in rtsp_send

    rtsp_recv = str(vm.eval_chunk("blueprint(Actor[#Protocols], #rtsp, #receive)!!"))
    assert "RTSP RECEIVE" in rtsp_recv
    assert "-rtsp_transport tcp" in rtsp_recv
    assert "probeRtsp" in rtsp_recv

    mjpeg_py = str(vm.eval_chunk("blueprint(Actor[#Protocols], #mjpeg, #python)!!"))
    assert "multipart/x-mixed-replace; boundary=frame" in mjpeg_py
    assert "ThreadingHTTPServer" in mjpeg_py

    # Grammar graph surfaces `web_and_streaming_protocols` on streaming/RTSP goals
    g_ctx = sandbox.grammar_graph.queryConstructsForTask("Build a live RTSP and browser MJPEG video stream")
    assert "Actor[#Protocols]" in g_ctx
    assert "probeRtsp" in g_ctx

    # Bilby can subclass `ProtocolSpec` cleanly in Ring-2 user code (`inherit(ProtocolSpec, ...)`) while `Actor[#Protocols]` singleton stays locked
    sub_code = (
        "inherit(ProtocolSpec, #WebRtcWhipSpec, #(#iceServers), 1, nil)!!\n"
        "now(WebRtcWhipSpec)!!\n"
        "Def initWhip(self) {\n"
        '    initProtocol(self, #whip, "WebRTC WHIP Ingest", "UDP/DTLS/SRTP", "application/sdp", #http);\n'
        '    self.iceServers := "stun:stun.l.google.com:19302";\n'
        "    ^self;\n"
        "}!!"
    )
    lint_res = sandbox.lintChunk(sub_code)
    assert lint_res.startswith("OK:"), lint_res
    vm.eval_chunk(sub_code)
    whip_desc = str(vm.eval_chunk("describe(initWhip(new(WebRtcWhipSpec)))!!"))
    assert "WebRtcWhipSpec" in whip_desc
    assert "whip" in whip_desc

    # Attempting to overwrite `Actor[#Protocols]` singleton is blocked by guardrail
    hijack = sandbox.executeStepAction(harness, "Actor[#Protocols] := 99!!", 15000)
    assert "ERROR:" in str(hijack)


def test_ssrf_probe_guard_blocks_loopback_and_local_stack_ports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_validate_probe_target_url` blocks loopback, `localhost`, and protected stack ports when `ACTORII_ALLOW_PRIVATE_URLS` is off."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ACTORII_ALLOW_PRIVATE_URLS", raising=False)
    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

    r1 = vm.eval_chunk('probeHttp(Actor[#Net], "http://127.0.0.1:8080/v1/models", 3)!!')
    assert r1.at("ok") is False
    assert "BLOCKED" in str(r1.at("diagnosis"))

    r2 = vm.eval_chunk('probeHttp(Actor[#Net], "http://localhost:9999/", 3)!!')
    assert r2.at("ok") is False
    assert "BLOCKED" in str(r2.at("diagnosis"))

    r3 = vm.eval_chunk('probeRtsp(Actor[#Net], "rtsp://127.0.0.1:8554/cam", "", 3)!!')
    assert r3.at("ok") is False
    assert "BLOCKED" in str(r3.at("diagnosis"))


def test_raw_binary_tcp_bug_caught_and_valid_mjpeg_sse_hls_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `probeHttp`/`probeMjpeg` catch raw `[uint32 len][JPEG]` TCP sockets AND pass compliant MJPEG, SSE, and HLS servers."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ACTORII_MEDIA_STAGE_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("ACTORII_ALLOW_PRIVATE_URLS", "1")
    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

    sample_jpg = _make_minimal_jpeg(640, 480)

    # 1. Reproduce today's exact `julia_stream` raw TCP socket (`[4-byte little-endian len][JFIF]`, NO HTTP status line)
    raw_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw_srv.bind(("127.0.0.1", 0))
    raw_port = raw_srv.getsockname()[1]
    raw_srv.listen(4)
    stop_raw = threading.Event()

    def _serve_raw_tcp() -> None:
        raw_srv.settimeout(0.3)
        while not stop_raw.is_set():
            try:
                conn, _ = raw_srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.recv(256)
                conn.sendall(struct.pack("<I", len(sample_jpg)) + sample_jpg)
            except Exception:
                pass
            finally:
                conn.close()

    t_raw = threading.Thread(target=_serve_raw_tcp, daemon=True)
    t_raw.start()

    # 2. Spin up a compliant HTTP/1.1 server serving `/`, `/stream` (MJPEG), `/events` (SSE), and `/index.m3u8` + `/seg0.ts` (HLS)
    ts_packet = (b"\x47" + b"\x00" * 187) * 4

    class CompliantMediaHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for _ in range(3):
                        part = (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(sample_jpg)).encode("ascii")
                            + b"\r\n\r\n"
                            + sample_jpg
                            + b"\r\n"
                        )
                        self.wfile.write(part)
                        self.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b'event: tick\ndata: {"fps": 29.97}\n\ndata: {"fps": 30.0}\n\n')
                self.wfile.flush()
            elif self.path == "/index.m3u8":
                body = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nseg0.ts\n"
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.apple.mpegurl")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/seg0.ts":
                self.send_response(200)
                self.send_header("Content-Type", "video/mp2t")
                self.send_header("Content-Length", str(len(ts_packet)))
                self.end_headers()
                self.wfile.write(ts_packet)
            else:
                html = b"<html><body><img src='/stream'/></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

    http_srv = ThreadingHTTPServer(("127.0.0.1", 0), CompliantMediaHandler)
    http_port = http_srv.server_address[1]
    t_http = threading.Thread(target=http_srv.serve_forever, daemon=True)
    t_http.start()

    try:
        # A. Raw TCP socket is caught & diagnosed clearly
        bad_http = vm.eval_chunk(f'probeHttp(Actor[#Net], "http://127.0.0.1:{raw_port}/", 4)!!')
        assert bad_http.at("ok") is False
        assert "InvalidProtocolFraming" in str(bad_http.at("diagnosis"))
        assert "length-prefixed binary JPEG" in str(bad_http.at("diagnosis"))

        bad_mjpeg = vm.eval_chunk(f'verify(Actor[#Protocols], "http://127.0.0.1:{raw_port}/stream", #mjpeg)!!')
        assert bad_mjpeg.at("ok") is False
        assert "InvalidProtocolFraming" in str(bad_mjpeg.at("diagnosis"))

        # B. Compliant HTTP + MJPEG passes end-to-end and stages sample frame
        good_http = vm.eval_chunk(f'verify(Actor[#Protocols], "http://127.0.0.1:{http_port}/", #http)!!')
        assert good_http.at("ok") is True
        assert good_http.at("statusCode") == 200

        good_mjpeg = vm.eval_chunk(f'verify(Actor[#Protocols], "http://127.0.0.1:{http_port}/stream", #mjpeg)!!')
        assert good_mjpeg.at("ok") is True
        assert good_mjpeg.at("framesVerified") == 2
        assert good_mjpeg.at("width") == 640
        assert good_mjpeg.at("height") == 480
        assert good_mjpeg.at("survivedDisconnect") is True
        saved_jpg = Path(str(good_mjpeg.at("samplePath")))
        assert saved_jpg.is_file() and saved_jpg.read_bytes() == sample_jpg

        # C. SSE & HLS pass conformance
        good_sse = vm.eval_chunk(f'verify(Actor[#Protocols], "http://127.0.0.1:{http_port}/events", #sse)!!')
        assert good_sse.at("ok") is True
        assert good_sse.at("eventsRead") == 2

        good_hls = vm.eval_chunk(f'verify(Actor[#Protocols], "http://127.0.0.1:{http_port}/index.m3u8", #hls)!!')
        assert good_hls.at("ok") is True
        assert good_hls.at("container") == "mpeg-ts"
    finally:
        stop_raw.set()
        raw_srv.close()
        http_srv.shutdown()
        http_srv.server_close()


def test_rtsp_send_receive_probe_and_websocket_rfc6455_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `probeRtsp` (`OPTIONS`+`DESCRIBE` SDP + TX1 `level_idc=0` detection) and `probeWebSocket` (RFC 6455)."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ACTORII_ALLOW_PRIVATE_URLS", "1")
    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

    # 1. Ephemeral RFC 2326 RTSP server supporting `/cam` (valid baseline SPS), `/bad_tx1` (`level_idc=0` bug), and `/publish` (`404` empty send mount)
    rtsp_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rtsp_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rtsp_srv.bind(("127.0.0.1", 0))
    rtsp_port = rtsp_srv.getsockname()[1]
    rtsp_srv.listen(4)
    stop_rtsp = threading.Event()

    def _serve_rtsp() -> None:
        rtsp_srv.settimeout(0.3)
        while not stop_rtsp.is_set():
            try:
                conn, _ = rtsp_srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(2.0)
                for _ in range(3):
                    data = conn.recv(4096)
                    if not data:
                        break
                    req_txt = data.decode("iso-8859-1", errors="replace")
                    first_ln = req_txt.splitlines()[0] if req_txt else ""
                    if first_ln.startswith("OPTIONS "):
                        resp = (
                            "RTSP/1.0 200 OK\r\n"
                            "CSeq: 1\r\n"
                            "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, ANNOUNCE, RECORD\r\n\r\n"
                        ).encode("ascii")
                        conn.sendall(resp)
                    elif first_ln.startswith("DESCRIBE "):
                        if "/publish" in first_ln:
                            conn.sendall(b"RTSP/1.0 404 Stream Not Found\r\nCSeq: 2\r\nContent-Length: 0\r\n\r\n")
                        else:
                            plid = "42e000" if "/bad_tx1" in first_ln else "42e01f"
                            sdp = (
                                "v=0\r\n"
                                "o=- 1 1 IN IP4 127.0.0.1\r\n"
                                "s=JetsonRTSP\r\n"
                                "m=video 0 RTP/AVP 96\r\n"
                                "a=rtpmap:96 H264/90000\r\n"
                                f"a=fmtp:96 packetization-mode=1;profile-level-id={plid}\r\n"
                                "a=control:stream=0\r\n"
                            ).encode("ascii")
                            hdr = (
                                "RTSP/1.0 200 OK\r\n"
                                "CSeq: 2\r\n"
                                "Content-Type: application/sdp\r\n"
                                f"Content-Length: {len(sdp)}\r\n\r\n"
                            ).encode("ascii")
                            conn.sendall(hdr + sdp)
            except Exception:
                pass
            finally:
                conn.close()

    t_rtsp = threading.Thread(target=_serve_rtsp, daemon=True)
    t_rtsp.start()

    # 2. Ephemeral RFC 6455 WebSocket server
    ws_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ws_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ws_srv.bind(("127.0.0.1", 0))
    ws_port = ws_srv.getsockname()[1]
    ws_srv.listen(4)
    stop_ws = threading.Event()

    def _serve_ws() -> None:
        ws_srv.settimeout(0.3)
        while not stop_ws.is_set():
            try:
                conn, _ = ws_srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(2.0)
                req = conn.recv(4096).decode("iso-8859-1", errors="replace")
                ws_key = ""
                for ln in req.split("\r\n"):
                    if ln.lower().startswith("sec-websocket-key:"):
                        ws_key = ln.split(":", 1)[1].strip()
                accept = base64.b64encode(
                    hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
                ).decode("ascii")
                conn.sendall(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("ascii")
                )
                frame_in = conn.recv(256)
                if len(frame_in) >= 6 and (frame_in[1] & 0x80):
                    plen = frame_in[1] & 0x7F
                    mask = frame_in[2:6]
                    msg = bytes(frame_in[6 + i] ^ mask[i % 4] for i in range(plen))
                    reply = b"pong:" + msg
                    conn.sendall(bytes([0x81, len(reply)]) + reply)
            except Exception:
                pass
            finally:
                conn.close()

    t_ws = threading.Thread(target=_serve_ws, daemon=True)
    t_ws.start()

    try:
        good_rtsp = vm.eval_chunk(f'probeRtsp(Actor[#Net], "rtsp://127.0.0.1:{rtsp_port}/cam", "", 5)!!')
        assert good_rtsp.at("ok") is True
        assert good_rtsp.at("canReceive") is True
        assert good_rtsp.at("canSendPublish") is True
        assert "H264/90000" in str(good_rtsp.at("codecs"))

        bad_tx1 = vm.eval_chunk(f'probeRtsp(Actor[#Net], "rtsp://127.0.0.1:{rtsp_port}/bad_tx1", "", 5)!!')
        assert bad_tx1.at("ok") is False
        assert "omxh264enc" in str(bad_tx1.at("diagnosis"))

        empty_mount = vm.eval_chunk(f'probeRtsp(Actor[#Net], "rtsp://127.0.0.1:{rtsp_port}/publish", "", 5)!!')
        assert empty_mount.at("ok") is False
        assert empty_mount.at("optionsStatus") == 200
        assert empty_mount.at("statusCode") == 404
        assert empty_mount.at("canSendPublish") is True

        ws_res = vm.eval_chunk(f'verify(Actor[#Protocols], "ws://127.0.0.1:{ws_port}/ws", #ws)!!')
        assert ws_res.at("ok") is True
        assert ws_res.at("receivedMessage") == "pong:ping"
    finally:
        stop_rtsp.set()
        rtsp_srv.close()
        stop_ws.set()
        ws_srv.close()


def test_jetson_deploy_service_lifecycle_and_tui_protocols_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `JetsonEdgeTwin` `deployService`/`serviceStatus`/`stopService` guardrails & command generation, plus TUI `/protocols`."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    session = ActorTUISession(use_color=False)

    # TUI `/protocols` command
    before = len(session.transcript)
    session.handle_input("/protocols rtsp send")
    out_txt = "\n".join(session.transcript[before:])
    assert "RTSP SEND" in out_txt
    assert "omxh264enc" in out_txt

    jetson = session.vm.global_vars["Actor"]["Jetson"].target

    # Protected system daemon names (`sshd`, `jetson-rtsp`, `gdm3`) are hard-blocked
    blocked_svc = jetson.deployService("jetson-rtsp", "$HOME/services/x", "python3 app.py", 8554)
    assert blocked_svc.at("ok") is False
    assert "reserved" in str(blocked_svc.at("stderr"))

    # Guardrail blocks destructive OS/bootloader commands inside `deployService` without ever invoking SSH
    blocked_cmd = jetson.deployService("my_stream", "$HOME/services/my_stream", "sudo apt upgrade -y", 8090)
    assert blocked_cmd.at("ok") is False
    assert "JETSON_GUARDRAIL_BLOCKED" in str(blocked_cmd.at("stderr"))

    captured: List[str] = []

    def fake_ssh(remote_cmd: str, timeout_sec: float = 15.0) -> dict:
        captured.append(remote_cmd)
        return {
            "ok": True,
            "host": "192.168.68.55",
            "exitCode": 0,
            "stdout": "PID=24680 ALIVE=1 LISTEN=1\nServing HTTP MJPEG on 0.0.0.0:8090\nSTATUS=RUNNING PID=24680",
            "stderr": "",
        }

    monkeypatch.setattr(jetson, "_ssh_exec_raw", fake_ssh)
    dep = jetson.deployService("julia_web", "$HOME/services/julia_web", "python3 mjpeg_server.py", 8090, 20)
    assert dep.at("ok") is True
    assert dep.at("pid") == 24680
    assert dep.at("portListening") is True
    assert "setsid nohup" in captured[-1]

    st = jetson.serviceStatus("julia_web", 20)
    assert st.at("ok") is True
    assert st.at("running") is True

    stopped = jetson.stopService("julia_web")
    assert stopped.at("ok") is True
