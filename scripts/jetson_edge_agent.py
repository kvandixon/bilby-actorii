#!/usr/bin/env python3
"""Jetson TX1 Lightweight ActorII Edge Agent (`jetson-actor-agent.service`, port 8095).

Runs as unprivileged `bilby-edge` on the NVIDIA Jetson TX1 (4 GB RAM) with:
- Local 1.5B LLM (`qwen2.5-1.5b-instruct` via `llama-server` on `127.0.0.1:8088`, CPU NEON)
- Metacircular `ActorVM` + `AgentHarness` (`enable_memory_graph=False`, `ACTORII_DISABLE_DEMO_TWINS=1`)
- Strictly sandboxed to `bilby-edge`'s unprivileged OS permissions + cgroup cap (`MemoryMax=1800M`)

Endpoints:
  GET  /health  -> Jetson hardware/RAM, local LLM (:8088), RTSP camera, and ActorVM status
  POST /ask     -> {"goal": "...", "max_steps": 4, "mode": "harness"|"direct"}
  POST /eval    -> {"chunk": "6 * 7!!"} (guarded ActorVM eval)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("ACTORII_DISABLE_DEMO_TWINS", "1")
os.environ.setdefault("ACTORII_ENABLE_THINKING", "0")
os.environ.setdefault("ACTORII_THINKING_MODE", "off")
os.environ.setdefault("ACTORII_HOST_LOCKDOWN", "0")  # OS-level isolation enforced by unprivileged `bilby-edge` user
os.environ.setdefault("ACTORII_LLM_ENDPOINT", "http://127.0.0.1:8088/v1/chat/completions")
os.environ.setdefault("ACTORII_LLM_MODEL", "qwen2.5-1.5b-instruct")

ACTORII_SRC = Path("/opt/jetson-actor/actorii/src")
if str(ACTORII_SRC) not in sys.path:
    sys.path.insert(0, str(ACTORII_SRC))

from actorii.harness import bootstrap_agent_harness  # noqa: E402

LLM_URL = os.environ["ACTORII_LLM_ENDPOINT"]
LLM_HEALTH_URL = LLM_URL.rsplit("/v1/", 1)[0] + "/health"
LLM_MODEL = os.environ["ACTORII_LLM_MODEL"]
BIND_HOST = os.environ.get("JETSON_ACTOR_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("JETSON_ACTOR_PORT", "8095") or 8095)

_VM_LOCK = threading.Lock()
_BOOT_TS = time.time()

_VM, _HARNESS, _SANDBOX = bootstrap_agent_harness(
    name="JetsonBilby",
    model=LLM_MODEL,
    endpoint=LLM_URL,
    max_steps=4,
    max_failures=3,
    gas_limit=50000,
    enable_guardrails=True,
    enable_memory_graph=False,
    enable_engrams=False,
)


def _probe_http(url: str, timeout: float = 2.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return f"online (HTTP {resp.status})"
    except urllib.error.HTTPError as exc:
        return f"loading/http_{exc.code}"
    except Exception as exc:
        return f"offline ({type(exc).__name__})"


def _probe_tcp(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _mem_info_mb() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:", "SwapFree:"):
                out[parts[0].rstrip(":")] = int(parts[1]) // 1024
    except Exception:
        pass
    return out


def _direct_llm_ask(goal: str, max_tokens: int = 400) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mem = _mem_info_mb()
    sys_prompt = (
        "You are JetsonBilby, a lightweight local edge AI running on an NVIDIA Jetson TX1 "
        "(192.168.68.57, user bilby-edge, 4 GB RAM, CSI camera RTSP :8554). "
        f"Live telemetry: MemAvailable={mem.get('MemAvailable', '?')}MB / {mem.get('MemTotal', '?')}MB, "
        f"RTSP_8554={'up' if _probe_tcp('127.0.0.1', 8554) else 'down'}. "
        "Be concise, accurate, and practical."
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": goal},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90.0) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    msg = ((body.get("choices") or [{}])[0].get("message") or {})
    usage = body.get("usage") or {}
    c_tok = int(usage.get("completion_tokens") or 0)
    return {
        "ok": True,
        "mode": "direct",
        "model": LLM_MODEL,
        "reply": str(msg.get("content") or "").strip(),
        "completionTokens": c_tok,
        "tokensPerSec": round(c_tok / max(0.001, elapsed_ms / 1000.0), 1) if c_tok else 0.0,
        "elapsedMs": elapsed_ms,
    }


def _harness_ask(goal: str, max_steps: int = 4) -> Dict[str, Any]:
    t0 = time.perf_counter()
    steps = max(1, min(6, int(max_steps or 4)))
    with _VM_LOCK:
        res = _VM.eval_chunk(f"runGoal(Actor[#Harness], {json.dumps(goal)}, {steps})!!")
        status = str(_HARNESS._fields.get("status", "unknown"))
        step_count = int(_HARNESS._fields.get("stepCount", 0) or 0)
    return {
        "ok": True,
        "mode": "harness",
        "model": LLM_MODEL,
        "status": status,
        "steps": step_count,
        "reply": str(res or "").strip(),
        "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1),
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "agent": "JetsonBilby",
                    "user": os.environ.get("USER", "bilby-edge"),
                    "hostname": socket.gethostname(),
                    "uptimeSec": round(time.time() - _BOOT_TS, 1),
                    "llmModel": LLM_MODEL,
                    "llmStatus": _probe_http(LLM_HEALTH_URL),
                    "rtsp8554": "up" if _probe_tcp("127.0.0.1", 8554) else "down",
                    "memMb": _mem_info_mb(),
                    "actorClasses": len(getattr(_VM.hierarchy, "classes", {}) or {}),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 262144:
            self._send_json(413, {"ok": False, "error": "payload_too_large"})
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8", errors="replace") or "{}")
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"invalid_json: {exc}"})
            return

        if path == "/ask":
            goal = str(data.get("goal") or "").strip()
            if not goal:
                self._send_json(400, {"ok": False, "error": "missing_goal"})
                return
            mode = str(data.get("mode") or "harness").strip().lower()
            try:
                if mode == "direct":
                    out = _direct_llm_ask(goal, max_tokens=int(data.get("max_tokens", 400) or 400))
                else:
                    out = _harness_ask(goal, max_steps=int(data.get("max_steps", 4) or 4))
                self._send_json(200, out)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        if path == "/eval":
            chunk = str(data.get("chunk") or "").strip()
            if not chunk:
                self._send_json(400, {"ok": False, "error": "missing_chunk"})
                return
            if not chunk.endswith("!!"):
                chunk = chunk.rstrip("!") + "!!"
            try:
                with _VM_LOCK:
                    lint = _SANDBOX.lintChunk(chunk)
                    if str(lint).startswith("ERROR:"):
                        self._send_json(403, {"ok": False, "error": str(lint)})
                        return
                    res = _VM.eval_transactional(chunk, max_opcodes=25000, shadow_only=False)
                self._send_json(200, {"ok": True, "result": str(res)})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        self._send_json(404, {"ok": False, "error": "not_found"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer((BIND_HOST, BIND_PORT), _Handler)
    print(f"[jetson-actor-agent] Listening on http://{BIND_HOST}:{BIND_PORT} (model={LLM_MODEL})", flush=True)
    srv.serve_forever()
