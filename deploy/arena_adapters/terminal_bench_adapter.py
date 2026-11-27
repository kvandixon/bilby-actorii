#!/usr/bin/env python3
"""Official Terminal-Bench (`tbench.ai` / Harbor) Adapter for Bilby (`ActorII`).

Supports three invocation surfaces with zero module-import side effects:
1. Harbor `BaseAgent` (`harbor run --agent-import-path deploy.arena_adapters.terminal_bench_adapter:BilbyTerminalBenchAgent`)
   - Implements `name()`, `version()`, `async setup(environment)`, and `async run(instruction, environment, context)`.
2. Classic Terminal-Bench `BaseAgent` (`tb run --agent-import-path ...`)
   - Implements `perform_task(instruction, session=None, logging_dir=None)`.
3. Standalone container CLI:
   - `python3 terminal_bench_adapter.py --instruction "..." --logging-dir /tmp/tb_logs`

Runtime guardrails inside isolated benchmark containers:
- Sets `ACTORII_HOST_LOCKDOWN=0` (container provides OS isolation)
- Sets `ACTORII_BENCH_STRICT=1` (strict benchmark evaluation mode, no demo twins)
- Sets `ACTORII_DISABLE_DEMO_TWINS=1`
- Writes ATIF-compatible `trajectory.json` (Agent Trajectory Interchange Format).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure `actorii` is importable whether installed as a package or run from repo checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Optional upstream Harbor BaseAgent inheritance when Harbor is installed.
try:  # pragma: no cover - optional runtime dependency
    from harbor.agents.base import AgentCapabilities as _HarborAgentCapabilities
    from harbor.agents.base import BaseAgent as _HarborBaseAgent

    _HAS_HARBOR = True
except Exception:  # pragma: no cover - offline / standalone environments
    _HAS_HARBOR = False

    class _HarborBaseAgent:  # type: ignore[no-redef]
        """Lightweight fallback base class when `harbor` is not installed."""

        def __init__(self, logs_dir: Optional[Path] = None, model_name: Optional[str] = None, **kwargs: Any) -> None:
            self.logs_dir = Path(logs_dir) if logs_dir is not None else None
            self.model_name = model_name
            self.extra_kwargs = kwargs

    class _HarborAgentCapabilities:  # type: ignore[no-redef]
        def __init__(self, atif: bool = True, **kwargs: Any) -> None:
            self.atif = atif
            self.extra = kwargs


ADAPTER_NAME = "bilby-actorii"
ADAPTER_VERSION = "4.4.0"
ATIF_SCHEMA_VERSION = "ATIF-v1.2"


def configure_benchmark_environment(
    *,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    host_lockdown: str = "0",
    bench_strict: str = "1",
    thinking_mode: Optional[str] = None,
) -> Dict[str, str]:
    """Apply required ActorII benchmark container environment flags."""
    tmode = (thinking_mode or os.environ.get("ACTORII_THINKING_MODE") or "auto").strip().lower()
    applied: Dict[str, str] = {
        "ACTORII_HOST_LOCKDOWN": str(host_lockdown),
        "ACTORII_BENCH_STRICT": str(bench_strict),
        "ACTORII_DISABLE_DEMO_TWINS": "1",
        "ACTORII_THINKING_MODE": tmode,
        "ACTORII_ENABLE_THINKING": "0" if tmode == "off" else "1",
    }
    if model:
        applied["ACTORII_LLM_MODEL"] = model
    if endpoint:
        applied["ACTORII_LLM_ENDPOINT"] = endpoint
    for key, val in applied.items():
        os.environ[key] = val
    return applied


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TrajectoryStep:
    """Single ATIF-compatible trajectory step."""

    step_id: int
    timestamp: str
    source: str  # "user" | "agent" | "system"
    message: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observation: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    """Records agent steps and serializes an ATIF-compatible `trajectory.json`."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        agent_name: str = ADAPTER_NAME,
        agent_version: str = ADAPTER_VERSION,
        model_name: str = "unknown",
    ) -> None:
        self.session_id = session_id or f"bilby-tb-{uuid.uuid4().hex[:12]}"
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.model_name = model_name
        self.started_at = _utc_now_iso()
        self.steps: List[TrajectoryStep] = []
        self.final_metrics: Dict[str, Any] = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_steps": 0,
            "elapsed_seconds": 0.0,
            "status": "initialized",
        }

    def add_step(
        self,
        source: str,
        message: str,
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        observation: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            step_id=len(self.steps) + 1,
            timestamp=_utc_now_iso(),
            source=source,
            message=message,
            tool_calls=list(tool_calls or []),
            observation=observation,
            metrics=dict(metrics or {}),
        )
        self.steps.append(step)
        return step

    def set_final_metrics(
        self,
        *,
        status: str,
        elapsed_seconds: float,
        total_prompt_tokens: int = 0,
        total_completion_tokens: int = 0,
        harness_steps: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.final_metrics = {
            "status": status,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_completion_tokens": int(total_completion_tokens),
            "total_tokens": int(total_prompt_tokens) + int(total_completion_tokens),
            "harness_steps": int(harness_steps),
            "total_steps": len(self.steps),
            "completed_at": _utc_now_iso(),
        }
        if extra:
            self.final_metrics.update(extra)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ATIF_SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "agent": {
                "name": self.agent_name,
                "version": self.agent_version,
                "model_name": self.model_name,
            },
            "steps": [asdict(step) for step in self.steps],
            "final_metrics": dict(self.final_metrics),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def _default_harness_factory(
    *,
    name: str,
    model: str,
    endpoint: Optional[str],
    max_steps: int,
    max_failures: int = 10,
    gas_limit: int,
    enable_memory_graph: bool,
    enable_engrams: bool,
    memory_namespace: Optional[str],
) -> Tuple[Any, Any, Any]:
    """Lazily import and bootstrap ActorII's AgentHarness."""
    from actorii.harness import bootstrap_agent_harness

    return bootstrap_agent_harness(
        name=name,
        model=model,
        endpoint=endpoint,
        max_steps=max_steps,
        max_failures=max_failures,
        gas_limit=gas_limit,
        enable_guardrails=True,
        enable_memory_graph=enable_memory_graph,
        enable_engrams=enable_engrams,
        memory_namespace=memory_namespace,
    )


def _build_harbor_host_bridge(
    environment: Any,
    loop: asyncio.AbstractEventLoop,
    cancel_event: Optional[threading.Event] = None,
) -> Any:
    """Wrap Harbor's async Docker `environment.exec(...)` as an ActorII `Actor[#Host]` bridge."""
    import base64
    import shlex
    from actorii.bridge import actor_bridge, actor_command, actor_query

    @actor_bridge
    class HarborContainerHostBridge:
        def __init__(
            self,
            env: Any,
            ev_loop: asyncio.AbstractEventLoop,
            cancel_ev: Optional[threading.Event] = None,
        ) -> None:
            self._env = env
            self._loop = ev_loop
            self._cancel = cancel_ev or threading.Event()

        def _exec_raw(self, command: str, timeout_sec: int = 60) -> Tuple[int, str, str]:
            if self._cancel.is_set():
                raise RuntimeError("Harbor task cancelled/timed out — halting container exec")
            capped = min(240, max(1, int(timeout_sec or 60)))
            fut = asyncio.run_coroutine_threadsafe(
                self._env.exec(command, timeout_sec=capped),
                self._loop,
            )
            res = fut.result(timeout=capped + 15)
            rc = int(getattr(res, "return_code", 0) or 0)
            out = str(getattr(res, "stdout", "") or "")
            err = str(getattr(res, "stderr", "") or "")
            return rc, out, err

        @actor_command
        def execShell(self, command: str, timeout: int = 60) -> str:
            rc, out, err = self._exec_raw(str(command), timeout_sec=int(timeout or 60))
            parts = []
            if out.strip():
                parts.append(out.rstrip())
            if err.strip():
                parts.append(f"[stderr]\n{err.rstrip()}")
            body = "\n".join(parts) if parts else "(no output)"
            if len(body) > 6000:
                body = body[:5900] + f"\n... [truncated {len(body) - 5900} chars]"
            return f"[exit={rc}]\n{body}"

        @actor_query
        def readFile(self, path: str, startLine: int = 1, endLine: int = 200) -> str:
            s = max(1, int(startLine or 1))
            e = max(s, int(endLine or 200))
            cmd = f"sed -n '{s},{e}p' {shlex.quote(str(path))}"
            rc, out, err = self._exec_raw(cmd, timeout_sec=15)
            return out if rc == 0 else f"ERROR (rc={rc}): {err or out}"

        @actor_command
        def writeFile(self, path: str, content: str) -> str:
            b64 = base64.b64encode(str(content).encode("utf-8")).decode("ascii")
            qpath = shlex.quote(str(path))
            cmd = f"mkdir -p $(dirname {qpath}) && printf %s {shlex.quote(b64)} | base64 -d > {qpath}"
            rc, out, err = self._exec_raw(cmd, timeout_sec=15)
            return f"Wrote {len(str(content))} bytes to {path}" if rc == 0 else f"ERROR (rc={rc}): {err or out}"

        @actor_command
        def editFile(self, path: str, target: str, replacement: str) -> str:
            """Host-side exact substring replacement via base64 (zero dependency on python3 inside the container)."""
            qpath = shlex.quote(str(path))
            rc, out, err = self._exec_raw(f"base64 {qpath} | tr -d '\\r\\n'", timeout_sec=15)
            if rc != 0:
                return f"ERROR (rc={rc}): {err or out}"
            try:
                content = base64.b64decode(out.strip().encode("ascii")).decode("utf-8", errors="replace")
            except Exception as exc:
                return f"ERROR: failed to decode {path}: {exc}"
            t_str = str(target)
            r_str = str(replacement)
            if t_str not in content:
                return f"ERROR: target not found in {path} (read the file first to copy the exact lines)"
            updated = content.replace(t_str, r_str, 1)
            b64_new = base64.b64encode(updated.encode("utf-8")).decode("ascii")
            w_cmd = f"printf %s {shlex.quote(b64_new)} | base64 -d > {qpath}"
            rc2, out2, err2 = self._exec_raw(w_cmd, timeout_sec=15)
            return f"Edited {path}" if rc2 == 0 else f"ERROR (rc={rc2}): {err2 or out2}"

        @actor_query
        def listFiles(self, path: str = ".") -> str:
            rc, out, err = self._exec_raw(f"ls -la {shlex.quote(str(path))}", timeout_sec=15)
            return out if rc == 0 else f"ERROR (rc={rc}): {err or out}"

        @actor_query
        def searchFile(self, pattern: str, path: str = ".") -> str:
            rc, out, err = self._exec_raw(
                f"grep -rn {shlex.quote(str(pattern))} {shlex.quote(str(path))} | head -n 60",
                timeout_sec=15,
            )
            return out if out.strip() else f"(no matches, rc={rc})"

    return HarborContainerHostBridge(environment, loop, cancel_event)


def _extract_usage_tokens(harness: Any) -> Tuple[int, int]:
    """Best-effort token extraction from ActorII AgentHarness fields."""
    fields = getattr(harness, "_fields", {}) or {}
    prompt_tok = int(fields.get("promptTokens", 0) or getattr(harness, "total_prompt_tokens", 0) or 0)
    comp_tok = int(fields.get("completionTokens", 0) or getattr(harness, "total_completion_tokens", 0) or 0)
    return prompt_tok, comp_tok


class BilbyTerminalBenchAgent(_HarborBaseAgent):
    """Bilby (`ActorII`) custom agent adapter for Terminal-Bench & Harbor."""

    capabilities = _HarborAgentCapabilities(atif=True)

    def __init__(
        self,
        logs_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        *,
        endpoint: Optional[str] = None,
        max_steps: int = 60,
        max_failures: int = 12,
        gas_limit: int = 100000,
        enable_memory_graph: bool = False,
        enable_engrams: bool = False,
        memory_namespace: Optional[str] = None,
        harness_factory: Optional[Callable[..., Tuple[Any, Any, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.logs_dir = Path(logs_dir) if logs_dir is not None else None
        self.model_name = (
            model_name
            or os.environ.get("ACTORII_LLM_MODEL", "qwen3.8:latest").strip()
            or "qwen3.8:latest"
        )
        self.endpoint = endpoint or os.environ.get("ACTORII_LLM_ENDPOINT")
        self.max_steps = int(max_steps)
        self.max_failures = int(max_failures)
        self.gas_limit = int(gas_limit)
        self.enable_memory_graph = bool(enable_memory_graph)
        self.enable_engrams = bool(enable_engrams)
        self.memory_namespace = memory_namespace or f"tb_{uuid.uuid4().hex[:8]}"
        self._harness_factory = harness_factory or _default_harness_factory
        self._vm: Any = None
        self._harness: Any = None
        self._sandbox: Any = None
        self._cancel_event: threading.Event = threading.Event()

    @staticmethod
    def name() -> str:
        return ADAPTER_NAME

    def version(self) -> Optional[str]:
        return ADAPTER_VERSION

    def _ensure_bootstrapped(self) -> Tuple[Any, Any, Any]:
        if self._vm is None or self._harness is None:
            configure_benchmark_environment(
                model=self.model_name,
                endpoint=self.endpoint,
                host_lockdown="0",
                bench_strict="1",
            )
            self._vm, self._harness, self._sandbox = self._harness_factory(
                name="Bilby",
                model=self.model_name,
                endpoint=self.endpoint,
                max_steps=self.max_steps,
                max_failures=self.max_failures,
                gas_limit=self.gas_limit,
                enable_memory_graph=self.enable_memory_graph,
                enable_engrams=self.enable_engrams,
                memory_namespace=self.memory_namespace,
            )
        return self._vm, self._harness, self._sandbox

    def _execute_task_sync(
        self,
        instruction: str,
        *,
        logging_dir: Optional[Path] = None,
        context: Optional[Any] = None,
        environment: Optional[Any] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger = TrajectoryLogger(
            agent_name=self.name(),
            agent_version=self.version() or ADAPTER_VERSION,
            model_name=self.model_name,
        )
        logger.add_step("user", instruction)

        status = "error"
        reply = ""
        harness_steps = 0
        prompt_tokens = 0
        completion_tokens = 0
        failure_mode: Optional[str] = None

        try:
            vm, harness, _sandbox = self._ensure_bootstrapped()
            if environment is not None and event_loop is not None and hasattr(environment, "exec"):
                from actorii.core import Symbol
                prev_lock = getattr(vm, "lock_system_singletons", True)
                vm.lock_system_singletons = False
                try:
                    bridge = _build_harbor_host_bridge(
                        environment, event_loop, cancel_event=self._cancel_event
                    )
                    # Register via vm.expose_object so Actor[#Host] is a real ActorBridgeObject
                    # with `_bridge_specs` and survives `eval_transactional` rollbacks!
                    vm.expose_object(bridge, global_key="#Host")
                    actor_pool = vm.global_vars.get("Actor")
                    if isinstance(actor_pool, dict):
                        # Isolate container benchmark runs from host-only twins (both string & Symbol keys)
                        for host_only_key in (
                            "Valkyrie", "System", "Jetson", "Pantry", "Hue",
                            "Telegram", "Proxy", "Ops", "Thermometer", "ThermoUI",
                            "Network", "NetTopo", "ProductLearner", "Web",
                        ):
                            actor_pool.pop(host_only_key, None)
                            actor_pool.pop(Symbol(host_only_key), None)
                finally:
                    vm.lock_system_singletons = prev_lock
                goal_text = (
                    "TERMINAL-BENCH CONTAINER TASK (live Docker environment bound to `Actor[#Host]`):\n"
                    "- On Step 1, immediately emit an Actor 4.1 command ending in `!!` "
                    "(e.g. `execShell(Actor[#Host], \"ls -la /app\", 15)!!`) to inspect the container.\n"
                    "- Prefer `writeFile(Actor[#Host], path, \"\"\"...\"\"\")!!` and `editFile(Actor[#Host], path, \"\"\"old\"\"\", \"\"\"new\"\"\")!!` "
                    "over brittle nested shell quoting (`perl -0pi`, `sed -i`, multiline `python3 -c`). Both work even without `python3` in the container.\n"
                    "- When compiling inside Docker, use modest parallelism (`make -j4`, never `-j28`) to avoid container OOM/swap hangs; "
                    "when a build or test fails, inspect the FIRST error (`head -40`), not just `tail -20`.\n"
                    "- Before replying `COMPLETE: <summary>`, verify EVERY requirement in the instruction — including edge cases, "
                    "cancellation/error paths, exact output file formats, and exhaustive candidate checks.\n\n"
                    f"TASK INSTRUCTION:\n{instruction}"
                )
            else:
                goal_text = instruction

            raw_reply = vm.eval_chunk(
                f"runGoal(Actor[#Harness], {json.dumps(goal_text)}, {self.max_steps})!!"
            )
            reply = str(raw_reply or "").strip()
            fields = getattr(harness, "_fields", {}) or {}
            status = str(fields.get("status", "completed") or "completed")
            harness_steps = int(fields.get("stepCount", 0) or 0)
            prompt_tokens, completion_tokens = _extract_usage_tokens(harness)

            # Record full per-step ATIF trajectory for post-run forensic review.
            raw_step_log = fields.get("stepLog")
            step_entries = list(raw_step_log) if raw_step_log is not None else []
            for se in step_entries:
                if not isinstance(se, dict):
                    continue
                act_s = str(se.get("action") or "")
                res_s = str(se.get("error") if se.get("status") == "error" else se.get("result") or "")
                logger.add_step(
                    "agent",
                    act_s[:2000],
                    observation={"results": [{"content": res_s[:2000], "status": se.get("status")}]},
                    metrics={
                        "step": int(se.get("step") or 0),
                        "gas_used": int(se.get("gas_used") or 0),
                    },
                )

            logger.add_step(
                "agent",
                reply,
                metrics={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "harness_steps": harness_steps,
                    "status": status,
                },
            )
        except Exception as exc:
            failure_mode = f"{type(exc).__name__}: {exc}"
            status = "failed"
            logger.add_step(
                "system",
                f"Execution error: {failure_mode}",
                observation={"results": [{"content": failure_mode}]},
            )

        elapsed = time.perf_counter() - t0
        logger.set_final_metrics(
            status=status,
            elapsed_seconds=elapsed,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            harness_steps=harness_steps,
            extra={"failure_mode": failure_mode} if failure_mode else None,
        )

        # Populate Harbor AgentContext metrics when provided.
        if context is not None:
            for attr, val in (
                ("n_input_tokens", prompt_tokens),
                ("n_output_tokens", completion_tokens),
            ):
                if hasattr(context, attr):
                    setattr(context, attr, val)

        out_dir = logging_dir or self.logs_dir
        trajectory_path: Optional[Path] = None
        if out_dir is not None:
            trajectory_path = logger.write(Path(out_dir) / "trajectory.json")

        return {
            "agent": self.name(),
            "version": self.version(),
            "model": self.model_name,
            "status": status,
            "reply": reply,
            "steps": harness_steps,
            "total_input_tokens": prompt_tokens,
            "total_output_tokens": completion_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "failure_mode": failure_mode,
            "trajectory_path": str(trajectory_path) if trajectory_path else None,
            "trajectory": logger.to_dict(),
        }

    # --- Harbor BaseAgent async contract ---
    async def setup(self, environment: Any) -> None:
        """Prepare environment variables and ActorII runtime before task execution."""
        configure_benchmark_environment(
            model=self.model_name,
            endpoint=self.endpoint,
            host_lockdown="0",
            bench_strict="1",
        )
        # Keep VM bootstrap lazy until `run()` so setup remains fast and side-effect free.
        self._environment = environment

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> Dict[str, Any]:
        """Execute a Terminal-Bench task under Harbor."""
        self._environment = environment
        self._cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.to_thread(
                self._execute_task_sync,
                instruction,
                logging_dir=self.logs_dir,
                context=context,
                environment=environment,
                event_loop=loop,
            )
        except BaseException:
            # Immediately halt the sync runGoal worker thread so timed-out trials never leak zombie GPU/Docker work.
            self._cancel_event.set()
            if self._harness is not None and hasattr(self._harness, "_fields"):
                try:
                    from actorii.core import Symbol
                    self._harness._fields["status"] = Symbol("halted")
                except Exception:
                    pass
            raise

    # --- Classic Terminal-Bench BaseAgent sync contract ---
    def perform_task(
        self,
        instruction: str,
        session: Any = None,
        logging_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Execute a Terminal-Bench task under classic `tb run` interface."""
        self._session = session
        target_dir = Path(logging_dir) if logging_dir is not None else self.logs_dir
        return self._execute_task_sync(instruction, logging_dir=target_dir)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bilby (ActorII) Terminal-Bench / Harbor adapter CLI"
    )
    parser.add_argument("--instruction", type=str, help="Task instruction text")
    parser.add_argument("--instruction-file", type=Path, help="Path to instruction.md")
    parser.add_argument("--logging-dir", type=Path, default=None, help="Directory for trajectory.json")
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier")
    parser.add_argument("--endpoint", type=str, default=None, help="OpenAI-compatible LLM endpoint")
    parser.add_argument("--max-steps", type=int, default=20, help="Max ActorII harness steps")
    parser.add_argument("--gas-limit", type=int, default=100000, help="ActorVM opcode gas limit")
    parser.add_argument(
        "--enable-memory-graph",
        action="store_true",
        help="Enable Neo4j memory graph (disabled by default for isolated benchmark runs)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    instruction = args.instruction
    if not instruction and args.instruction_file:
        instruction = args.instruction_file.read_text(encoding="utf-8")
    if not instruction:
        print("ERROR: provide --instruction or --instruction-file", file=sys.stderr)
        return 2

    agent = BilbyTerminalBenchAgent(
        logs_dir=args.logging_dir,
        model_name=args.model,
        endpoint=args.endpoint,
        max_steps=args.max_steps,
        gas_limit=args.gas_limit,
        enable_memory_graph=args.enable_memory_graph,
    )
    result = agent.perform_task(instruction, logging_dir=args.logging_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}, indent=2))
    return 0 if result.get("failure_mode") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
