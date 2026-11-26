#!/usr/bin/env python3
"""Official AgentBeats (`agentbeats.dev` Berkeley RDI) Purple-Agent A2A Server for Bilby (`ActorII`).

Exposes an A2A-compatible HTTP + JSON-RPC 2.0 server with zero third-party web-framework
dependencies (pure Python stdlib `http.server.ThreadingHTTPServer`):

Endpoints:
- `GET  /health`                    -> Health & readiness probe (zero LLM calls)
- `GET  /.well-known/agent.json`    -> A2A Agent Card (legacy/standard discovery path)
- `GET  /.well-known/agent-card.json` -> A2A Agent Card (v0.3 discovery alias)
- `POST /`                          -> A2A JSON-RPC 2.0 (`message/send`, `tasks/send`, `tasks/get`, `tasks/cancel`)
- `POST /tasks`                     -> Direct REST convenience endpoint for task execution

CLI contract (AgentBeats Docker entrypoint compatible):
  python3 agentbeats_purple_agent.py --host 0.0.0.0 --port 9009 --card-url http://localhost:9009
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

AGENT_NAME = "Bilby (ActorII)"
AGENT_VERSION = "4.4.0"
A2A_PROTOCOL_VERSION = "0.2.5"
MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def build_agent_card(
    card_url: str,
    *,
    model: Optional[str] = None,
    version: str = AGENT_VERSION,
) -> Dict[str, Any]:
    """Construct the A2A Agent Card advertised at `/.well-known/agent.json`."""
    base_url = card_url.rstrip("/")
    active_model = model or os.environ.get("ACTORII_LLM_MODEL", "qwen3.8:latest")
    return {
        "name": AGENT_NAME,
        "description": (
            "Bilby — metacircular ActorII cognitive agent (compiled Actor 4.1 VM + "
            "self-correcting AgentHarness) packaged as an AgentBeats purple agent."
        ),
        "url": base_url,
        "version": version,
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "provider": {
            "organization": "ActorII / Bilby",
            "url": base_url,
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "general-task-execution",
                "name": "General Autonomous Task Execution",
                "description": (
                    "Executes coding, terminal, debugging, and structured reasoning tasks "
                    f"via ActorII AgentHarness (model={active_model})."
                ),
                "tags": ["actorii", "bilby", "coding", "terminal", "reasoning", "purple-agent"],
                "examples": [
                    "Debug the failing pytest suite in /workspace and report the root cause.",
                    "Write and verify a Python script to parse the CSV logs in /workspace/data.",
                ],
            }
        ],
    }


def extract_prompt_from_a2a_params(params: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract `(prompt_text, task_id, context_id)` from A2A `message/send` or `tasks/send` params."""
    task_id = params.get("id") or params.get("taskId")
    context_id = params.get("contextId") or params.get("sessionId")

    message = params.get("message")
    text_parts: List[str] = []
    if isinstance(message, dict):
        if not task_id:
            task_id = message.get("taskId")
        if not context_id:
            context_id = message.get("contextId")
        parts = message.get("parts") or []
        for part in parts:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("kind") == "data" and "data" in part:
                    text_parts.append(json.dumps(part["data"]))
            elif isinstance(part, str):
                text_parts.append(part)
    elif isinstance(params.get("prompt"), str):
        text_parts.append(params["prompt"])
    elif isinstance(params.get("goal"), str):
        text_parts.append(params["goal"])
    elif isinstance(params.get("instruction"), str):
        text_parts.append(params["instruction"])

    prompt = "\n".join(p.strip() for p in text_parts if p and p.strip()).strip()
    return prompt, (str(task_id) if task_id else None), (str(context_id) if context_id else None)


class BilbyPurpleRuntime:
    """Lazy, thread-safe ActorII execution runtime backing the A2A purple agent."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_steps: int = 20,
        gas_limit: int = 100000,
        enable_memory_graph: bool = False,
        enable_engrams: bool = False,
        runner: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    ) -> None:
        self.model = model or os.environ.get("ACTORII_LLM_MODEL", "qwen3.8:latest").strip() or "qwen3.8:latest"
        self.endpoint = endpoint or os.environ.get("ACTORII_LLM_ENDPOINT")
        self.max_steps = int(max_steps)
        self.gas_limit = int(gas_limit)
        self.enable_memory_graph = bool(enable_memory_graph)
        self.enable_engrams = bool(enable_engrams)
        self._custom_runner = runner

        self._lock = threading.Lock()
        self._vm: Any = None
        self._harness: Any = None
        self._sandbox: Any = None
        self._boot_ts = time.time()
        self.tasks: Dict[str, Dict[str, Any]] = {}

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._boot_ts, 2)

    @property
    def is_bootstrapped(self) -> bool:
        return self._vm is not None and self._harness is not None

    def _ensure_bootstrapped_locked(self) -> Tuple[Any, Any, Any]:
        if self._vm is None or self._harness is None:
            os.environ.setdefault("ACTORII_HOST_LOCKDOWN", "0")
            os.environ.setdefault("ACTORII_BENCH_STRICT", "1")
            os.environ.setdefault("ACTORII_DISABLE_DEMO_TWINS", "1")
            os.environ["ACTORII_LLM_MODEL"] = self.model
            if self.endpoint:
                os.environ["ACTORII_LLM_ENDPOINT"] = self.endpoint

            from actorii.harness import bootstrap_agent_harness

            self._vm, self._harness, self._sandbox = bootstrap_agent_harness(
                name="Bilby",
                model=self.model,
                endpoint=self.endpoint,
                max_steps=self.max_steps,
                gas_limit=self.gas_limit,
                enable_guardrails=True,
                enable_memory_graph=self.enable_memory_graph,
                enable_engrams=self.enable_engrams,
                memory_namespace=f"agentbeats_{uuid.uuid4().hex[:8]}",
            )
        return self._vm, self._harness, self._sandbox

    def run_task(
        self,
        prompt: str,
        *,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a task through ActorII (or injected runner) and return an A2A Task object."""
        tid = task_id or f"task-{uuid.uuid4().hex[:12]}"
        cid = context_id or f"ctx-{uuid.uuid4().hex[:12]}"
        steps_budget = int(max_steps or self.max_steps)
        t0 = time.perf_counter()

        with self._lock:
            try:
                if self._custom_runner is not None:
                    exec_out = self._custom_runner(prompt, steps_budget)
                else:
                    vm, harness, _sandbox = self._ensure_bootstrapped_locked()
                    raw = vm.eval_chunk(
                        f"runGoal(Actor[#Harness], {json.dumps(prompt)}, {steps_budget})!!"
                    )
                    fields = getattr(harness, "_fields", {}) or {}
                    exec_out = {
                        "reply": str(raw or "").strip(),
                        "status": str(fields.get("status", "completed") or "completed"),
                        "steps": int(fields.get("stepCount", 0) or 0),
                    }
                state = "completed"
                reply_text = str(exec_out.get("reply") or "").strip()
                meta = {
                    "model": self.model,
                    "harnessStatus": exec_out.get("status", "completed"),
                    "steps": int(exec_out.get("steps", 0) or 0),
                    "elapsedSeconds": round(time.perf_counter() - t0, 3),
                }
            except Exception as exc:
                state = "failed"
                reply_text = f"ERROR ({type(exc).__name__}): {exc}"
                meta = {
                    "model": self.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsedSeconds": round(time.perf_counter() - t0, 3),
                }

            agent_message = {
                "kind": "message",
                "messageId": f"msg-{uuid.uuid4().hex[:12]}",
                "role": "agent",
                "parts": [{"kind": "text", "text": reply_text}],
            }
            task_record: Dict[str, Any] = {
                "kind": "task",
                "id": tid,
                "contextId": cid,
                "status": {
                    "state": state,
                    "timestamp": _utc_now_iso(),
                    "message": agent_message,
                },
                "artifacts": [
                    {
                        "artifactId": f"art-{uuid.uuid4().hex[:10]}",
                        "name": "response",
                        "parts": [{"kind": "text", "text": reply_text}],
                    }
                ],
                "metadata": meta,
            }
            self.tasks[tid] = task_record
            return task_record

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            task["status"]["state"] = "canceled"
            task["status"]["timestamp"] = _utc_now_iso()
            return task


def make_a2a_handler(
    runtime: BilbyPurpleRuntime,
    card_url: str,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to `runtime` and `card_url`."""

    class _A2AHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            pass

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _jsonrpc_error(
            self,
            req_id: Any,
            code: int,
            message: str,
            http_status: int = 200,
        ) -> None:
            self._send_json(
                http_status,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": code, "message": message},
                },
            )

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "status": "healthy",
                        "agent": AGENT_NAME,
                        "version": AGENT_VERSION,
                        "protocolVersion": A2A_PROTOCOL_VERSION,
                        "model": runtime.model,
                        "bootstrapped": runtime.is_bootstrapped,
                        "uptimeSeconds": runtime.uptime_seconds,
                    },
                )
                return

            if path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
                self._send_json(200, build_agent_card(card_url, model=runtime.model))
                return

            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"ok": False, "error": "payload_too_large"})
                return

            try:
                body = json.loads(self.rfile.read(length).decode("utf-8", errors="replace") or "{}")
            except Exception as exc:
                self._jsonrpc_error(None, -32700, f"Parse error: {exc}", http_status=400)
                return

            if not isinstance(body, dict):
                self._jsonrpc_error(None, -32600, "Invalid Request: expected JSON object", http_status=400)
                return

            # Convenience REST endpoint: POST /tasks
            if path in ("/tasks", "/run"):
                prompt, tid, cid = extract_prompt_from_a2a_params(body)
                if not prompt:
                    self._send_json(400, {"ok": False, "error": "missing_prompt"})
                    return
                max_steps = body.get("max_steps")
                task = runtime.run_task(prompt, task_id=tid, context_id=cid, max_steps=max_steps)
                self._send_json(200, task)
                return

            # Standard A2A JSON-RPC 2.0 endpoint: POST /
            if path in ("/", "/a2a", "/rpc"):
                req_id = body.get("id")
                method = str(body.get("method") or "").strip()
                params = body.get("params") if isinstance(body.get("params"), dict) else {}

                if method in ("message/send", "tasks/send"):
                    prompt, tid, cid = extract_prompt_from_a2a_params(params)
                    if not prompt:
                        self._jsonrpc_error(req_id, -32602, "Invalid params: empty message/prompt")
                        return
                    task = runtime.run_task(prompt, task_id=tid, context_id=cid)
                    self._send_json(200, {"jsonrpc": "2.0", "id": req_id, "result": task})
                    return

                if method == "tasks/get":
                    tid = str(params.get("id") or params.get("taskId") or "").strip()
                    task = runtime.get_task(tid) if tid else None
                    if task is None:
                        self._jsonrpc_error(req_id, -32001, f"Task not found: {tid}")
                        return
                    self._send_json(200, {"jsonrpc": "2.0", "id": req_id, "result": task})
                    return

                if method == "tasks/cancel":
                    tid = str(params.get("id") or params.get("taskId") or "").strip()
                    task = runtime.cancel_task(tid) if tid else None
                    if task is None:
                        self._jsonrpc_error(req_id, -32001, f"Task not found: {tid}")
                        return
                    self._send_json(200, {"jsonrpc": "2.0", "id": req_id, "result": task})
                    return

                self._jsonrpc_error(req_id, -32601, f"Method not found: {method}")
                return

            self._send_json(404, {"ok": False, "error": "not_found"})

    return _A2AHandler


def create_server(
    *,
    host: str = "0.0.0.0",
    port: int = 9009,
    card_url: Optional[str] = None,
    runtime: Optional[BilbyPurpleRuntime] = None,
) -> ThreadingHTTPServer:
    """Instantiate a configured A2A ThreadingHTTPServer without starting `serve_forever()`."""
    rt = runtime or BilbyPurpleRuntime()
    effective_card_url = card_url or f"http://{host}:{port}"
    handler_cls = make_a2a_handler(rt, effective_card_url)
    return ThreadingHTTPServer((host, int(port)), handler_cls)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bilby (ActorII) AgentBeats A2A Purple-Agent Server"
    )
    parser.add_argument("--host", type=str, default=os.environ.get("AGENTBEATS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTBEATS_PORT", "9009")))
    parser.add_argument(
        "--card-url",
        type=str,
        default=os.environ.get("AGENTBEATS_CARD_URL"),
        help="Public base URL advertised in /.well-known/agent.json",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--endpoint", type=str, default=None, help="OpenAI-compatible LLM endpoint")
    parser.add_argument("--max-steps", type=int, default=20, help="Max ActorII harness steps")
    parser.add_argument("--gas-limit", type=int, default=100000, help="ActorVM opcode gas limit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    runtime = BilbyPurpleRuntime(
        model=args.model,
        endpoint=args.endpoint,
        max_steps=args.max_steps,
        gas_limit=args.gas_limit,
    )
    card_url = args.card_url or f"http://{args.host}:{args.port}"
    server = create_server(host=args.host, port=args.port, card_url=card_url, runtime=runtime)
    print(
        f"[agentbeats-purple] Bilby (ActorII) listening on http://{args.host}:{args.port} "
        f"(card_url={card_url}, model={runtime.model})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
