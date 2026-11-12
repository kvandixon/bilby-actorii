"""Hermes / Antigravity-Style Terminal User Interface for ActorII (`actorii.tui`).

Provides a dark-themed, keyboard-driven TUI for interacting with the live
1993 Whitewater Actor 4.1 object pool and the Actor-native `AgentHarness`:
- Natural language conversation & autonomous `/goal` execution (streaming each
  ReAct step, static lint tier, opcode gas count, and state mutation live).
- Direct 1993 Actor 4.1 `!!` REPL evaluation against the shared object pool.
- Speculative copy-on-write dry runs (`/shadow`), static bytecode linting (`/lint`),
  JIT method synthesis (`/synth`), live method hot-patching (`/patch`), whole-world
  image checkpointing (`/save`, `/load`), and `.act` File-Out (`/fileout`).
- Full-screen curses split-pane dock mode (`--dock` or `/dock`) with a live
  Object Pool & Scratchpad sidebar alongside the interactive console mode.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import readline
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actorii.classes.agent import Agent
from actorii.classes.collections import ActorOrderedCollection
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorClass, ActorObject, Symbol
from actorii.guardrails.linter import PROTECTED_SYSTEM_CLASSES
from actorii.harness import BENCHMARK_TASKS, VMSandboxBridge, bootstrap_agent_harness
from actorii.vm import ActorVM, CompiledMethod

DEFAULT_STATE_DIR = Path.home() / ".actorii"
DEFAULT_SESSION_IMG = DEFAULT_STATE_DIR / "last_session.img"
DEFAULT_AUDIT_JSONL = DEFAULT_STATE_DIR / "session_audit.jsonl"

# =============================================================================
# Dark-Theme ANSI Palette (Antigravity / Hermes Style)
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[38;5;81m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;221m"
RED = "\033[38;5;203m"
MAGENTA = "\033[38;5;177m"
BLUE = "\033[38;5;75m"
GRAY = "\033[38;5;245m"
BG_DARK = "\033[48;5;235m"

SLASH_COMMANDS: Dict[str, str] = {
    "/chat": "/chat <message>            Conversational & action turn with multi-turn episodic memory",
    "/goal": "/goal <objective>          Run autonomous multi-step AgentHarness loop in Actor bytecode",
    "/history": "/history                   Display multi-turn conversation history (`chatHistory(Actor[#Harness])!!`)",
    "/clear": "/clear                     Clear multi-turn conversation history & reset circuit breaker",
    "/bench": "/bench [list|run|humaneval|gsm8k] Run 5-task in-image bench or official OpenAI HumanEval / GSM8K",
    "/grammar": "/grammar [topic]           Query Neo4j 1993 Actor Grammar & Error-Remediation Graph (`Actor[#Grammar]`)",
    "/engrams": "/engrams                   List saved Neo4j `:ActorEngram` computational memory-skills (`Actor[#Memory]`)",
    "/schema": "/schema                    Inspect active Neo4j domain node labels & relationships (`graphSchema(Actor[#Memory])`)",
    "/hydrate": "/hydrate <EngramName>      Hydrate a saved `:ActorEngram` from Neo4j into `Actor[#EngramName]`",
    "/twins": "/twins                     Inspect live Digital Twins (`#Valkyrie`, `#Proxy`, `#Hue`, `#Pantry`, `#Thermometer`)",
    "/proxy": "/proxy [status|lan|tailscale|local|pause|auth on|auth off|rotate|logs] Manage WLAN/Tailscale reverse proxy",
    "/connect": "/connect [url|off]         Attach TUI to shared `:8090` Web Studio VM (`http://127.0.0.1:8090`)",
    "/step": "/step [objective]          Execute a single ReAct tick (`step(Actor[#Harness])!!`)",
    "/shadow": "/shadow <code!!>           Speculative copy-on-write dry-run (`shadow_only=True`, 0 side effects)",
    "/lint": "/lint <code!!>             Static bytecode analysis (Risk Tier 0/1/2 & ivar schema check)",
    "/synth": "/synth <Cls> <sel> <arity> <spec>  JIT-synthesize, lint, judge & attach a bytecode method",
    "/patch": "/patch <ClassName> <Def..> Hot-patch a method on any live class (including AgentHarness)",
    "/harness": "/harness [rollback]        Inspect live compiled AgentHarness.act bytecode or rollback version",
    "/evolve": "/evolve <Def..> [| <val>]  Safely evolve AgentHarness in a shadow transaction + Neo4j Engram",
    "/pool": "/pool                      Inspect all live objects and classes in `Actor[...]`",
    "/inspect": "/inspect <target>          Deep-inspect an object (`#Harness`, `Actor[#Sam]`) or class",
    "/classes": "/classes                   List class hierarchy & custom/synthesized classes",
    "/scratch": "/scratch                   Show `Actor[#Harness]` scratchpad & semantic memory",
    "/save": "/save [path.img]           Save whole-world `ACTOR.IMG` snapshot",
    "/load": "/load [path.img]           Restore whole-world `ACTOR.IMG` snapshot",
    "/resume": "/resume                    Restore last auto-saved session (`~/.actorii/last_session.img`)",
    "/fileout": "/fileout <ClassName>       Export live class to standalone Git-diffable `.act` file",
    "/model": "/model <name> [endpoint]   Switch active LLM model/endpoint on `Actor[#Harness]`",
    "/config": "/config [k=v]              Inspect or update runtime LLM, Memory & Containment configuration",
    "/capabilities": "/capabilities              Show 4-Tier Capability vs. Containment Matrix (`capabilities(Actor[#Sandbox])!!`)",
    "/prompt": "/prompt [set <text>]       Inspect or update the active `Actor[#Harness]` system prompt",
    "/reset": "/reset                     Reset `AgentHarness` 3-strike circuit breaker",
    "/status": "/status                    Display VM, Guardrail, Bridge Audit & Harness telemetry",
    "/dock": "/dock                      Launch full-screen split-pane Curses TUI dock",
    "/help": "/help                      Show command reference",
    "/quit": "/quit                      Auto-save session image & exit ActorII TUI",
}


def _looks_like_actor_code(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.endswith("!!"):
        return True
    prefixes = (
        "Def ",
        "Action ",
        "inherit(",
        "now(",
        "Actor[",
        "new(",
        "sieve(",
        "step(",
        "runGoal(",
        "dryRun(",
        "lint(",
        "storeScratch(",
        "readScratch(",
        "chat(",
        "chatHistory(",
        "clearChat(",
    )
    return s.startswith(prefixes)


class ActorTUISession:
    """Manages the live `ActorVM`, `AgentHarness`, and command dispatch for the TUI."""

    def __init__(
        self,
        vm: Optional[ActorVM] = None,
        *,
        agent_name: str = "Bilby",
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_steps: int = 0,
        max_failures: int = 0,
        gas_limit: int = 100000,
        use_color: bool = True,
        remote_studio_url: Optional[str] = None,
    ) -> None:
        self.use_color = use_color and sys.stdout.isatty()
        self.remote_studio_url = remote_studio_url.rstrip("/") if remote_studio_url else None
        resolved_model = model or os.environ.get("ACTORII_LLM_MODEL", "qwen3.8:latest").strip() or "qwen3.8:latest"
        self.vm, self.harness, self.sandbox = bootstrap_agent_harness(
            vm,
            name=agent_name,
            model=resolved_model,
            endpoint=endpoint,
            max_steps=max_steps,
            max_failures=max_failures,
            gas_limit=gas_limit,
            enable_guardrails=True,
        )
        # Load Sam benchmark object so historical benchmarks work out-of-the-box too
        self._seed_convenience_objects()
        self.transcript: List[str] = []
        self.sandbox.step_callbacks.append(self._on_harness_step)
        DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _record_audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
            rec = {"timestamp": time.time(), "event": event_type, **payload}
            with DEFAULT_AUDIT_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def auto_save_session(self, path: Optional[Path] = None) -> Optional[str]:
        target = path or DEFAULT_SESSION_IMG
        try:
            DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
            out = self.sandbox.checkpointImage(str(target))
            return out
        except Exception:
            return None

    def _seed_convenience_objects(self) -> None:
        clean_sieve = Path(__file__).resolve().parents[2] / "examples" / "cleanroom" / "sieve.act"
        hist_sieve = Path(__file__).resolve().parents[2] / "historical_actor_41_source" / "actact" / "SIEVE.ACT"
        target_path = clean_sieve if clean_sieve.is_file() else hist_sieve
        if target_path.is_file():
            try:
                self.vm.eval_chunk(target_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _c(self, color: str, text: str) -> str:
        return f"{color}{text}{RESET}" if self.use_color else text

    def _emit(self, line: str = "") -> None:
        self.transcript.append(line)
        print(line)

    def _on_harness_step(self, entry: Dict[str, Any]) -> None:
        self._record_audit_event("harness_step", entry)
        step = entry.get("step", "?")
        status = entry.get("status", "ok")
        tier = entry.get("tier", "TIER_0")
        gas = entry.get("gas_used", 0)
        action = entry.get("action", "")

        tier_color = GREEN if "TIER_0" in tier else (YELLOW if "TIER_1" in tier else MAGENTA)
        badge = self._c(tier_color, f"[{tier}]")
        gas_badge = self._c(CYAN, f"[gas={gas} ops]")

        if status == "complete":
            self._emit(
                f"  {self._c(GREEN + BOLD, f'✔ [Step {step} COMPLETE]')} {gas_badge} "
                f"{self._c(BOLD, str(entry.get('result', '')))}"
            )
        elif status == "error":
            self._emit(
                f"  {self._c(RED + BOLD, f'✖ [Step {step} ROLLED BACK]')} {badge} {gas_badge}\n"
                f"    {self._c(GRAY, 'Action:')} {action}\n"
                f"    {self._c(RED, 'Error :')} {entry.get('error')} (strikes={entry.get('failures')})"
            )
        else:
            auto_eng = entry.get("auto_engrams") or []
            eng_note = (
                f"\n    {self._c(MAGENTA + BOLD, 'Auto-Crystallized Engram(s) in Neo4j:')} {auto_eng}"
                if auto_eng
                else ""
            )
            self._emit(
                f"  {self._c(BLUE + BOLD, f'⚡ [Step {step}]')} {badge} {gas_badge}\n"
                f"    {self._c(GRAY, 'Action:')} {action}\n"
                f"    {self._c(GREEN, 'Result:')} {entry.get('result')!r}{eng_note}"
            )
        self.flush_usage_file()

    def flush_usage_file(self) -> None:
        u_path = getattr(self, "usage_file", None)
        if not u_path:
            return
        try:
            from actorii.ai import ValkyrieLLMClient

            raw_steps = list(self.harness._fields.get("stepLog") or [])
            slim_steps = [
                {
                    "step": s.get("step"),
                    "status": s.get("status"),
                    "action": str(s.get("action") or "")[:300],
                    "result": str(s.get("result") or s.get("error") or "")[:300],
                }
                for s in raw_steps
                if isinstance(s, dict)
            ]
            usage_payload = {
                "model": self.harness.model,
                "steps": int(self.harness._fields.get("stepCount", 0) or 0),
                "status": str(self.harness._fields.get("status", "")),
                "last_result": str(self.harness._fields.get("lastResult", "")),
                "prompt_tokens": int(ValkyrieLLMClient.GLOBAL_PROMPT_TOKENS),
                "completion_tokens": int(ValkyrieLLMClient.GLOBAL_COMPLETION_TOKENS),
                "total_tokens": int(ValkyrieLLMClient.GLOBAL_TOTAL_TOKENS),
                "api_calls": int(ValkyrieLLMClient.GLOBAL_API_CALLS),
                "step_log": slim_steps,
            }
            Path(u_path).write_text(json.dumps(usage_payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    # =========================================================================
    # Status & Inspection Helpers
    # =========================================================================

    def status_ribbon_text(self) -> str:
        h = self.harness
        status = h._fields.get("status", Symbol("idle"))
        raw_ms = h._fields.get("maxSteps", 0)
        raw_mf = h._fields.get("maxFailures", 0)
        ms_lbl = "∞" if (raw_ms is not None and int(raw_ms) <= 0) else str(raw_ms)
        mf_lbl = "∞" if (raw_mf is not None and int(raw_mf) <= 0) else str(raw_mf)
        steps = f"{h._fields.get('stepCount', 0)}/{ms_lbl}"
        failures = f"{h._fields.get('failures', 0)}/{mf_lbl}"
        total_gas = h._fields.get("totalGasUsed", 0)
        pool_count = len(self.get_pool_objects())
        turn_count = len(self.get_chat_history(local_only=True))
        return (
            f"ActorII v4.2.0 │ {h.name} ({h.model}) │ status={status} │ "
            f"turns={turn_count} │ steps={steps} │ strikes={failures} │ gas={total_gas} │ pool={pool_count} objs"
        )

    def _remote_studio_request(
        self,
        path_and_query: str,
        *,
        method: str = "GET",
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        import urllib.request

        hdrs: Dict[str, str] = {"X-Actor-Request": "1"}
        tok = os.environ.get("ACTORII_STUDIO_TOKEN", "").strip()
        if tok:
            hdrs["Authorization"] = f"Bearer {tok}"
        req = urllib.request.Request(
            f"{self.remote_studio_url}{path_and_query}",
            data=b"" if method.upper() == "POST" else None,
            headers=hdrs,
            method=method.upper(),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def get_chat_history(self, local_only: bool = False) -> List[Dict[str, Any]]:
        if self.remote_studio_url and not local_only:
            try:
                data = self._remote_studio_request("/api/chat/history", method="GET", timeout=5.0)
                if isinstance(data.get("chat_history"), list):
                    return data["chat_history"]
            except Exception:
                pass
        return self.sandbox.getChatHistory(self.harness)

    def get_pool_objects(self) -> Dict[str, Any]:
        actor_dict = self.vm.global_vars.get("Actor", {})
        out: Dict[str, Any] = {}
        if isinstance(actor_dict, dict):
            for k, v in actor_dict.items():
                norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                if not isinstance(v, (ActorClass, type)):
                    out[norm_k] = v
        return out

    def get_user_classes(self) -> Dict[str, ActorClass]:
        out: Dict[str, ActorClass] = {}
        for cname, cls in sorted(self.vm.hierarchy._classes.items()):
            if cname not in PROTECTED_SYSTEM_CLASSES:
                out[cname] = cls
        return out

    def get_scratchpad_dict(self) -> Dict[str, Any]:
        scratch = self.harness._fields.get("scratchpad")
        out: Dict[str, Any] = {}
        if isinstance(scratch, Dictionary):
            for k, v in scratch._entries.items():
                norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                out[norm_k] = v
        return out

    # =========================================================================
    # Command Execution
    # =========================================================================

    def handle_input(self, raw_line: str) -> bool:
        """Process one user input line. Returns False if the session should exit."""
        line = (raw_line or "").strip()
        if not line:
            return True

        self._record_audit_event("user_input", {"input": line})

        if line.lower() in ("exit", "quit"):
            return self._handle_slash_command("/quit")

        if line.startswith("/"):
            return self._handle_slash_command(line)

        if _looks_like_actor_code(line):
            self._handle_actor_eval(line)
            return True

        # Unified Agentic Turn: every plain-English message runs through `AgentHarness`
        # so the agent executes real 1993 Actor bytecode when asked to act, or replies via COMPLETE:.
        self._run_harness_goal(line)
        return True

    def _run_harness_goal(self, objective: str) -> Any:
        raw_ms = self.harness._fields.get("maxSteps")
        max_steps = 0 if raw_ms is None else int(raw_ms)
        ms_disp = "unlimited" if max_steps <= 0 else str(max_steps)
        if self.remote_studio_url:
            import urllib.parse

            self._emit(
                f"\n{self._c(MAGENTA + BOLD, f'▶ Remote Studio AgentHarness ({self.remote_studio_url})')}\n"
                f"  {self._c(GRAY, 'Prompt:')} {objective} (maxSteps={ms_disp})"
            )
            q_enc = urllib.parse.quote(objective)
            try:
                data = self._remote_studio_request(
                    f"/api/goal?q={q_enc}&max_steps={max_steps}",
                    method="POST",
                    timeout=240.0,
                )
                for entry in data.get("step_log", []):
                    if isinstance(entry, dict):
                        self._on_harness_step(entry)
                final_res = data.get("final_summary") or data.get("result")
                self._emit(f"  {self._c(GREEN + BOLD, f'💬 {self.harness.name}:')} {final_res}")
                self._emit(
                    f"{self._c(CYAN + BOLD, '■ Remote Harness Finished:')} status={data.get('harness_status')} │ "
                    f"steps={data.get('steps_count', data.get('steps'))} │ totalGas={data.get('gas_used')} │ result={final_res!r}\n"
                )
                return final_res
            except Exception as exc:
                self._emit(self._c(RED + BOLD, f"✖ Remote Studio `/api/goal` failed ({exc}); falling back to local VM."))

        self._emit(
            f"\n{self._c(MAGENTA + BOLD, '▶ AgentHarness (`runGoal`)')}\n"
            f"  {self._c(GRAY, 'Prompt:')} {objective} (maxSteps={ms_disp}, gasLimit={self.harness._fields.get('gasLimit')})"
        )
        escaped = objective.replace("\\", "\\\\").replace('"', '\\"')
        try:
            final_res = self.vm.eval_chunk(f'runGoal(Actor[#Harness], "{escaped}", {max_steps})!!')
        except KeyboardInterrupt:
            self.harness._fields["status"] = Symbol("halted")
            self._emit(self._c(YELLOW + BOLD, "\n⏹ Goal interrupted by user (Ctrl+C)."))
            final_res = self.harness._fields.get("lastResult")
        status = self.harness._fields.get("status")
        total_gas = self.harness._fields.get("totalGasUsed", 0)
        turns_cnt = len(self.get_chat_history(local_only=True))
        self._emit(f"  {self._c(GREEN + BOLD, f'💬 {self.harness.name}:')} {final_res}")
        self._emit(
            f"{self._c(CYAN + BOLD, f'■ Harness Finished:')} status=#{status} │ "
            f"steps={self.harness._fields.get('stepCount')} │ turns={turns_cnt} │ totalGas={total_gas} │ result={final_res!r}\n"
        )
        self.auto_save_session()
        return final_res

    def _handle_actor_eval(self, source: str) -> Any:
        chunk = source.strip()
        if not chunk.endswith("!!"):
            chunk = chunk.rstrip("!") + "!!"
        if self.remote_studio_url:
            import urllib.parse

            c_enc = urllib.parse.quote(chunk)
            try:
                data = self._remote_studio_request(
                    f"/api/eval?code={c_enc}",
                    method="POST",
                    timeout=30.0,
                )
                if data.get("success"):
                    res = data.get("result")
                    self._emit(
                        f"{self._c(CYAN, f'[REMOTE {self.remote_studio_url}]')} "
                        f"{self._c(GREEN + BOLD, '⇒')} {res!r}"
                    )
                    return res
                else:
                    self._emit(f"{self._c(RED + BOLD, '✖ [REMOTE ERROR]')} {data.get('error')}")
                    return None
            except Exception as exc:
                self._emit(self._c(RED, f"Remote eval error ({exc}); evaluating locally:"))

        lint_info = self.sandbox.lintChunk(chunk)
        tier_badge = lint_info.split(":", 1)[1] if lint_info.startswith("OK:") else lint_info
        try:
            res = self.vm.eval_transactional(chunk, max_opcodes=self.harness._fields.get("gasLimit", 25000))
            gas = getattr(self.vm, "last_eval_opcodes", 0)
            self._emit(
                f"{self._c(CYAN, f'[{tier_badge} | gas={gas} ops]')} "
                f"{self._c(GREEN + BOLD, '⇒')} {res!r}"
            )
            self._record_audit_event("actor_eval", {"source": chunk, "tier": tier_badge, "gas_used": gas, "result": res})
            return res
        except Exception as exc:
            gas = getattr(self.vm, "last_eval_opcodes", 0)
            self._emit(
                f"{self._c(RED + BOLD, f'✖ [ROLLED BACK | gas={gas} ops]')} "
                f"{type(exc).__name__}: {exc}"
            )
            self._record_audit_event("actor_eval_error", {"source": chunk, "gas_used": gas, "error": str(exc)})
            return None

    def _handle_slash_command(self, line: str) -> bool:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            saved_path = self.auto_save_session()
            if saved_path:
                self._emit(self._c(GRAY, f"Auto-saved session image to {saved_path}. Exiting ActorII TUI."))
            else:
                self._emit(self._c(GRAY, "Exiting ActorII TUI."))
            return False

        if cmd == "/help":
            self._emit(self._c(CYAN + BOLD, "\nActorII Harness & TUI Commands:"))
            for desc in SLASH_COMMANDS.values():
                self._emit(f"  {desc}")
            self._emit(
                f"\n  {self._c(GRAY, 'Tip: Any input ending in `!!` runs directly as 1993 Actor bytecode.')}"
            )
            self._emit(
                f"  {self._c(GRAY, 'Tip: Plain text chats with Actor[#Harness] with live pool visibility.')}\n"
            )
            return True

        if cmd == "/status":
            turns = self.get_chat_history()
            self._emit(f"\n{self._c(CYAN + BOLD, '=== ActorII VM & AgentHarness Telemetry ===')}")
            self._emit(f"  {self.status_ribbon_text()}")
            self._emit(f"  Endpoint        : {self.harness.endpoint}")
            self._emit(f"  Chat Turns      : {len(turns)} multi-turn exchanges")
            self._emit(f"  Current Goal    : {self.harness._fields.get('goal')!r}")
            self._emit(f"  Last Action     : {self.harness._fields.get('lastAction')!r}")
            self._emit(f"  Last Result     : {self.harness._fields.get('lastResult')!r}")
            self._emit(f"  Last Error      : {self.harness._fields.get('lastError')!r}")
            self._emit(f"  Guardrail Audits: {len(self.vm.guardrail_audit_log)} methods verified")
            self._emit(f"  Sandbox Audits  : {len(self.vm.global_vars['Actor']['Sandbox'].audit_log)} calls\n")
            return True

        if cmd == "/pool":
            objs = self.get_pool_objects()
            self._emit(f"\n{self._c(CYAN + BOLD, f'=== Live Object Pool (`Actor[...]`: {len(objs)} objects) ===')}")
            for k, v in sorted(objs.items()):
                cls_name = getattr(v, "class_name", type(v).__name__)
                fields_preview = ""
                if isinstance(v, ActorObject) and getattr(v, "_fields", None):
                    fkeys = sorted(str(fk) for fk in v._fields.keys())
                    fields_preview = f" fields={fkeys[:6]}"
                self._emit(
                    f"  {self._c(YELLOW, f'Actor[#{k}]'):<28} : {self._c(GREEN, cls_name):<18} "
                    f"{repr(v)[:55]}{fields_preview}"
                )
            self._emit("")
            return True

        if cmd == "/classes":
            uclasses = self.get_user_classes()
            self._emit(f"\n{self._c(CYAN + BOLD, '=== User, Bridge & Agent Classes in Image ===')}")
            for cname, cls in uclasses.items():
                super_c = cls.bases[0] if getattr(cls, "bases", None) else "Object"
                methods = sorted(cls._methods.keys())
                ivars = getattr(cls, "ivars", [])
                self._emit(
                    f"  {self._c(GREEN + BOLD, cname)} < {super_c} "
                    f"│ ivars={ivars} │ methods={methods}"
                )
            self._emit("")
            return True

        if cmd == "/scratch":
            scratch = self.get_scratchpad_dict()
            self._emit(f"\n{self._c(CYAN + BOLD, '=== AgentHarness Scratchpad & Memory ===')}")
            if not scratch:
                self._emit("  (empty)")
            else:
                for k, v in sorted(scratch.items()):
                    self._emit(f"  {self._c(YELLOW, f'#{k}'):<22} = {v!r}")
            self._emit("")
            return True

        if cmd == "/inspect":
            target_name = arg or "#Harness"
            clean = target_name.lstrip("#")
            if clean.startswith("Actor[") and clean.endswith("]"):
                clean = clean[6:-1].lstrip("#").strip("\"'")
            actor_dict = self.vm.global_vars.get("Actor", {})
            target = actor_dict.get(clean) or self.vm.hierarchy.get(clean) or self.vm.global_vars.get(clean)
            if target is None:
                self._emit(self._c(RED, f"Object or class '{target_name}' not found in pool."))
                return True
            info = self.harness.inspectObject(target)
            self._emit(f"\n{self._c(CYAN + BOLD, f'=== Inspector: {target_name} ===')}\n{info}\n")
            return True

        if cmd == "/lint":
            if not arg:
                self._emit(self._c(RED, "Usage: /lint <actor_code!!>"))
                return True
            escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
            res = self.vm.eval_chunk(f'lint(Actor[#Harness], "{escaped}")!!')
            color = GREEN if str(res).startswith("OK:") else RED
            self._emit(f"{self._c(color + BOLD, f'[Lint Report] {res}')}")
            return True

        if cmd == "/shadow":
            if not arg:
                self._emit(self._c(RED, "Usage: /shadow <actor_code!!>"))
                return True
            escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
            res = self.vm.eval_chunk(f'dryRun(Actor[#Harness], "{escaped}")!!')
            color = CYAN if str(res).startswith("SHADOW_OK") else RED
            self._emit(f"{self._c(color + BOLD, f'[Speculative Dry-Run] {res}')}")
            return True

        if cmd == "/goal":
            if not arg:
                self._emit(self._c(RED, "Usage: /goal <objective>"))
                return True
            self._run_harness_goal(arg)
            return True

        if cmd == "/chat":
            if not arg:
                return self._handle_slash_command("/history")
            self._run_harness_goal(arg)
            return True

        if cmd in ("/history", "/turns"):
            turns = self.get_chat_history()
            self._emit(f"\n{self._c(CYAN + BOLD, f'=== Multi-Turn Conversation History ({len(turns)} turns) ===')}")
            if not turns:
                self._emit("  (no conversation turns recorded yet — type any message or `/chat <msg>`)")
            else:
                for t in turns:
                    if isinstance(t, dict):
                        tnum = t.get("turn", "?")
                        usr = t.get("user", "")
                        rep = t.get("reply", "")
                        acts = t.get("actions") or []
                        gas = t.get("gas_used", 0)
                        steps = t.get("steps", 1)
                        self._emit(
                            f"  {self._c(YELLOW + BOLD, f'[Turn {tnum}]')} "
                            f"{self._c(GRAY, f'(steps={steps}, gas={gas} ops)')}"
                        )
                        self._emit(f"    {self._c(CYAN, 'User         :')} {usr}")
                        if acts:
                            self._emit(f"    {self._c(MAGENTA, 'Bytecode     :')} {' ; '.join(acts)}")
                        self._emit(f"    {self._c(GREEN + BOLD, 'Bilby        :')} {rep}")
            self._emit("")
            return True

        if cmd == "/clear":
            if self.remote_studio_url:
                try:
                    self._remote_studio_request("/api/chat/clear", method="POST", timeout=5.0)
                except Exception:
                    pass
            self.vm.eval_chunk("clearChat(Actor[#Harness])!!")
            self.vm.eval_chunk("resetBreaker(Actor[#Harness])!!")
            self._emit(
                self._c(
                    GREEN + BOLD,
                    "✔ Cleared multi-turn conversation history & reset circuit breaker (VM object pool preserved).",
                )
            )
            return True

        if cmd == "/grammar":
            stats = self.vm.eval_chunk("graphStats(Actor[#Grammar])!!")
            self._emit(f"\n{self._c(CYAN + BOLD, str(stats))}")
            if arg:
                escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
                out = self.vm.eval_chunk(f'lookupSyntax(Actor[#Grammar], "{escaped}")!!')
                self._emit(f"{out}\n")
            else:
                out = self.vm.eval_chunk('queryConstructsForTask(Actor[#Grammar], "general", "", "")!!')
                self._emit(f"{out}\n")
            return True

        if cmd == "/engrams":
            out = self.vm.eval_chunk("listEngrams(Actor[#Memory])!!")
            self._emit(f"\n{self._c(MAGENTA + BOLD, str(out))}\n")
            return True

        if cmd == "/harness":
            if arg.strip().lower() == "rollback":
                out = self.vm.eval_chunk("rollbackHarness(Actor[#Harness])!!")
                self._emit(f"\n{self._c(GREEN + BOLD, str(out))}\n")
                return True
            from actorii.guardrails.fileout import ActorFileOutManager

            mgr = ActorFileOutManager("/tmp/actorii_tui_inspect")
            src = mgr.format_class_source(self.vm, "AgentHarness")
            self._emit(f"\n{self._c(CYAN + BOLD, '=== Live Compiled AgentHarness Bytecode Source ===')}\n{src}\n")
            return True

        if cmd == "/evolve":
            if not arg:
                self._emit(self._c(RED, 'Usage: /evolve Def policyNote(self) { ^"Custom rule"; }!! [| <validation!!>]'))
                return True
            if " | " in arg and not arg.strip().startswith("Def"):
                patch_part, val_part = arg.split(" | ", 1)
            else:
                patch_part, val_part = arg, ""
            p_esc = patch_part.strip().replace("\\", "\\\\").replace('"', '\\"')
            v_esc = val_part.strip().replace("\\", "\\\\").replace('"', '\\"')
            out = self.vm.eval_chunk(f'evolve(Actor[#Harness], "{p_esc}", "{v_esc}")!!')
            color = GREEN if str(out).startswith("EVOLVE_COMMITTED") else RED
            self._emit(f"\n{self._c(color + BOLD, str(out))}\n")
            return True

        if cmd == "/schema":
            out = self.vm.eval_chunk("graphSchema(Actor[#Memory])!!")
            self._emit(f"\n{self._c(CYAN + BOLD, str(out))}\n")
            return True

        if cmd == "/hydrate":
            if not arg:
                self._emit(self._c(RED, "Usage: /hydrate <EngramName>"))
                return True
            ename = arg.strip().lstrip("#")
            try:
                inst = self.vm.eval_chunk(f'loadEngram(Actor[#Memory], "{ename}")!!')
                self._emit(
                    f"{self._c(GREEN + BOLD, f'✔ Hydrated `:ActorEngram` into Actor[#{ename}]: {inst!r}')}\n"
                )
            except Exception as exc:
                self._emit(self._c(RED + BOLD, f"✖ Failed to hydrate '{ename}': {exc}\n"))
            return True

        if cmd == "/proxy":
            sub_raw = (arg or "status").strip()
            parts = sub_raw.split(maxsplit=1)
            action = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            from actorii.harness.twins import ReverseProxyTwin

            ptwin = ReverseProxyTwin()
            if action in ("status", "urls", "links", "info"):
                out = ptwin.status()
            elif action in ("lan", "wlan", "open", "on"):
                out = ptwin.setMode("lan")
            elif action in ("tailscale", "tailscale_only", "vpn"):
                out = ptwin.setMode("tailscale_only")
            elif action in ("local", "local_only", "localhost"):
                out = ptwin.setMode("local_only")
            elif action in ("pause", "paused", "off"):
                out = ptwin.setMode("paused")
            elif action == "auth":
                out = ptwin.setAuth(rest or "on")
            elif action == "rotate":
                out = ptwin.rotateToken()
            elif action == "allow" and rest:
                out = ptwin.allowCidr(rest)
            elif action in ("block", "deny") and rest:
                out = ptwin.blockIp(rest)
            elif action == "unblock" and rest:
                out = ptwin.unblockIp(rest)
            elif action == "logs":
                lim = int(rest) if rest.isdigit() else 10
                out = ptwin.recentLogs(lim)
            else:
                out = (
                    "Usage: /proxy [status | links | lan | tailscale | local | pause | "
                    "auth on | auth off | rotate | allow <cidr> | block <ip> | unblock <ip> | logs [n]]"
                )
            self._emit(f"\n{out}\n")
            return True

        if cmd == "/twins":
            self._emit(f"\n{self._c(CYAN + BOLD, '=== ActorII Live Digital Twins (`Actor[...]`) ===')}")
            valk = self.vm.eval_chunk("summary(Actor[#Valkyrie])!!")
            jetson_twin = self.vm.eval_chunk("summary(Actor[#Jetson])!!")
            proxy_twin = self.vm.eval_chunk("summary(Actor[#Proxy])!!")
            re_twin = self.vm.eval_chunk("summary(Actor[#ReverseEngineer])!!")
            tg_twin = self.vm.eval_chunk("summary(Actor[#Telegram])!!")
            web_twin = self.vm.eval_chunk("summary(Actor[#Web])!!")
            pdf_twin = self.vm.eval_chunk("summary(Actor[#PDF])!!")
            learner_twin = self.vm.eval_chunk("summary(Actor[#ProductLearner])!!")
            hue = self.vm.eval_chunk("listLights(Actor[#Hue])!!")
            pantry_cnt = self.vm.eval_chunk("recipeCount(Actor[#Pantry])!!")
            pantry_list = self.vm.eval_chunk("listRecipes(Actor[#Pantry], 5)!!")
            self._emit(f"  • {self._c(GREEN + BOLD, 'Actor[#Valkyrie]')}        : {valk}")
            self._emit(f"  • {self._c(GREEN + BOLD, 'Actor[#Jetson]')}          : {jetson_twin}")
            self._emit(f"  • {self._c(MAGENTA + BOLD, 'Actor[#ProductLearner]')}  : {learner_twin}")
            self._emit(f"  • {self._c(BLUE + BOLD, 'Actor[#Proxy]')}           : {proxy_twin}")
            self._emit(f"  • {self._c(CYAN + BOLD, 'Actor[#ReverseEngineer]')} : {re_twin}")
            self._emit(f"  • {self._c(BLUE + BOLD, 'Actor[#Telegram]')}        : {tg_twin}")
            self._emit(f"  • {self._c(GREEN + BOLD, 'Actor[#Web]')}             : {web_twin}")
            self._emit(f"  • {self._c(CYAN + BOLD, 'Actor[#PDF]')}             : {pdf_twin}")
            self._emit(f"  • {self._c(YELLOW + BOLD, 'Actor[#Hue]')}             : {hue}")
            self._emit(f"  • {self._c(MAGENTA + BOLD, 'Actor[#Pantry]')}          : {pantry_cnt} recipes ({pantry_list})")
            try:
                th_sum = self.vm.eval_chunk("summary(Actor[#Thermometer])!!")
                if th_sum:
                    self._emit(f"  • {self._c(CYAN + BOLD, 'Actor[#Thermometer]')}     : {th_sum}\n")
                else:
                    raise RuntimeError("No local #Thermometer")
            except Exception:
                self._emit(f"  • {self._c(GRAY, 'Actor[#Thermometer]: offline')}\n")
            return True

        if cmd == "/connect":
            target = arg.strip() if arg else "http://127.0.0.1:8090"
            if target.lower() in ("off", "none", "local", "disconnect"):
                self.remote_studio_url = None
                self._emit(self._c(YELLOW + BOLD, "✔ Disconnected from remote studio; running on local ActorVM."))
                return True
            if not target.startswith("http"):
                target = f"http://{target}"
            self.remote_studio_url = target.rstrip("/")
            self._emit(
                self._c(
                    GREEN + BOLD,
                    f"✔ Attached to shared ActorII Web Studio VM at {self.remote_studio_url}",
                )
            )
            return True

        if cmd == "/bench":
            sub = arg.split() if arg else ["list"]
            action = sub[0].lower()
            if action == "list":
                out = self.vm.eval_chunk("listTasks(Actor[#Bench])!!")
                self._emit(f"\n{self._c(CYAN + BOLD, str(out))}\n")
                self._emit(
                    self._c(
                        GRAY,
                        "Run `/bench setup <task_id>`, `/bench run <task_id>`, `/bench run all`, `/bench humaneval [n]`, or `/bench gsm8k [n]`.",
                    )
                )
                return True
            if action in ("humaneval", "gsm8k"):
                limit = int(sub[1]) if (len(sub) >= 2 and sub[1].isdigit()) else (5 if action == "humaneval" else 5)
                bench_dir = Path(__file__).resolve().parents[2] / "benchmarks"
                if str(bench_dir) not in sys.path:
                    sys.path.insert(0, str(bench_dir))
                from run_standard_benchmarks import run_benchmark_suite

                he_lim = limit if action == "humaneval" else 0
                gsm_lim = limit if action == "gsm8k" else 0
                self._emit(self._c(MAGENTA + BOLD, f"\n▶ Running Official OpenAI {action.upper()} (N={limit})..."))
                summary = run_benchmark_suite(humaneval_limit=he_lim, gsm8k_limit=gsm_lim, model=self.harness.model)
                if action == "humaneval":
                    he = summary["humaneval"]
                    self._emit(
                        f"{self._c(GREEN + BOLD, '✔ HumanEval Summary:')} "
                        f"1-Shot Actor={he['single_shot_actor_pass1']} │ "
                        f"ActorHarness={he['actor_harness_pass']} (+{he['self_corrected_count']} self-corrected) │ "
                        f"Python Pass@1={he['python_reference_pass1']}\n"
                    )
                else:
                    gsm = summary["gsm8k"]
                    self._emit(
                        f"{self._c(GREEN + BOLD, '✔ GSM8K Summary:')} "
                        f"1-Shot Actor={gsm['single_shot_actor_pass1']} │ "
                        f"ActorHarness={gsm['actor_harness_pass']}\n"
                    )
                return True
            if action == "setup" and len(sub) >= 2:
                tid = sub[1].lstrip("#")
                res = self.vm.eval_chunk(f'setupTask(Actor[#Bench], "{tid}")!!')
                self._emit(f"{self._c(GREEN + BOLD, f'✔ Benchmark task seeded in pool: {res}')}")
                return True
            if action == "report":
                rep = self.vm.eval_chunk("scoreReport(Actor[#Bench])!!")
                self._emit(f"\n{self._c(CYAN + BOLD, str(rep))}\n")
                return True
            if action == "run":
                target = sub[1].lstrip("#") if len(sub) >= 2 else "all"
                if target == "all":
                    self._emit(self._c(MAGENTA + BOLD, "\n▶ Running All 5 ActorII Agentic Benchmark Tasks..."))
                    rep = self.vm.eval_chunk("runAll(Actor[#Bench])!!")
                    self._emit(f"\n{self._c(GREEN + BOLD, str(rep))}\n")
                else:
                    self._emit(self._c(MAGENTA + BOLD, f"\n▶ Running Benchmark Task #{target}..."))
                    res = self.vm.eval_chunk(f'runTask(Actor[#Bench], "{target}")!!')
                    self._emit(f"\n{self._c(GREEN + BOLD, str(res))}\n")
                self.auto_save_session()
                return True
            # If `/bench <task_id>` was passed directly
            tid = action.lstrip("#")
            if tid in BENCHMARK_TASKS:
                self._emit(self._c(MAGENTA + BOLD, f"\n▶ Running Benchmark Task #{tid}..."))
                res = self.vm.eval_chunk(f'runTask(Actor[#Bench], "{tid}")!!')
                self._emit(f"\n{self._c(GREEN + BOLD, str(res))}\n")
                self.auto_save_session()
                return True
            self._emit(self._c(RED, f"Unknown /bench argument '{arg}'. Try `/bench list` or `/bench run all`."))
            return True

        if cmd == "/resume":
            target_path = Path(arg) if arg else DEFAULT_SESSION_IMG
            return self._handle_slash_command(f"/load {target_path}")

        if cmd == "/step":
            if arg:
                escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
                self.vm.eval_chunk(f'setGoal(Actor[#Harness], "{escaped}")!!')
            elif not self.harness._fields.get("goal"):
                self._emit(self._c(RED, "No active goal set. Use `/step <goal>` or `/goal <goal>`."))
                return True
            res = self.vm.eval_chunk("step(Actor[#Harness])!!")
            self._emit(f"{self._c(CYAN, f'[Step Tick Return]: {res!r}')}")
            return True

        if cmd == "/synth":
            # /synth <ClassName> <selector> [arity] <spec>
            sp = arg.split(None, 3)
            if len(sp) < 2:
                self._emit(self._c(RED, "Usage: /synth <ClassName> <selector> [arity] <spec_prompt>"))
                return True
            cls_name = sp[0].lstrip("#")
            selector = sp[1].lstrip("#")
            arity = 0
            spec = ""
            if len(sp) >= 3:
                if sp[2].isdigit():
                    arity = int(sp[2])
                    spec = sp[3] if len(sp) >= 4 else ""
                else:
                    spec = " ".join(sp[2:])
            try:
                src = self.harness.synthesizeMethod(cls_name, selector, spec_prompt=spec, arity=arity)
                self._emit(
                    f"{self._c(GREEN + BOLD, f'✔ Synthesized & Attached {cls_name}>>{selector}:')}\n"
                    f"  {self._c(CYAN, str(src))}"
                )
            except Exception as exc:
                self._emit(f"{self._c(RED + BOLD, f'✖ Synthesis Failed: {exc}')}")
            return True

        if cmd == "/patch":
            # /patch <ClassName> <Def method(self) { ... }!!>
            sp = arg.split(None, 1)
            if len(sp) < 2:
                self._emit(self._c(RED, "Usage: /patch <ClassName> <Def method(self) { ... }!!>"))
                return True
            cls_name, method_src = sp[0].lstrip("#"), sp[1].strip()
            if not method_src.endswith("!!"):
                method_src = method_src.rstrip("!") + "!!"
            chunk = f"now({cls_name})!!\n{method_src}"
            self._handle_actor_eval(chunk)
            return True

        if cmd == "/save":
            path = arg or "ACTOR_HARNESS.IMG"
            out = self.sandbox.checkpointImage(path)
            self._emit(f"{self._c(GREEN + BOLD, f'✔ Saved whole-world image snapshot to {out}')}")
            return True

        if cmd == "/load":
            path = arg or "ACTOR_HARNESS.IMG"
            try:
                self.vm.load_image(path)
                # Re-bind bridge objects and self.harness in restored image
                self.vm.expose_object(self.sandbox, global_key="#Sandbox")
                from actorii.harness.benchmark import ActorBenchSuite
                self.vm.expose_object(ActorBenchSuite(self.vm), global_key="#Bench")
                if "Harness" in self.vm.global_vars.get("Actor", {}):
                    self.harness = self.vm.global_vars["Actor"]["Harness"]
                    self.harness._fields["sandbox"] = self.vm.global_vars["Actor"]["Sandbox"]
                self._emit(f"{self._c(GREEN + BOLD, f'✔ Restored whole-world image from {path}')}")
            except Exception as exc:
                self._emit(f"{self._c(RED + BOLD, f'✖ Load failed: {exc}')}")
            return True

        if cmd == "/fileout":
            if not arg:
                self._emit(self._c(RED, "Usage: /fileout <ClassName>"))
                return True
            try:
                out = self.sandbox.fileOutClass(arg)
                self._emit(f"{self._c(GREEN + BOLD, f'✔ Exported {arg} to {out}')}")
            except Exception as exc:
                self._emit(f"{self._c(RED + BOLD, f'✖ File-Out failed: {exc}')}")
            return True

        if cmd == "/model":
            if not arg:
                self._emit(
                    f"Current model: {self.harness.model} @ {self.harness.endpoint}"
                )
                return True
            sp = arg.split()
            self.harness.model = sp[0]
            if len(sp) > 1:
                self.harness.endpoint = sp[1]
            self._emit(
                f"{self._c(GREEN, f'✔ Switched Actor[#Harness] to model={self.harness.model} @ {self.harness.endpoint}')}"
            )
            return True

        if cmd in ("/capabilities", "/caps", "/matrix"):
            rep = self.vm.eval_chunk("capabilities(Actor[#Sandbox])!!")
            self._emit(f"\n{self._c(CYAN + BOLD, str(rep))}\n")
            return True

        if cmd == "/config":
            if not arg:
                rep = self.vm.eval_chunk("capabilities(Actor[#Sandbox])!!")
                self._emit(f"\n{self._c(CYAN + BOLD, str(rep))}\n")
                self._emit(
                    self._c(
                        GRAY,
                        "Runtime updates: `/config model=<name>`, `/config endpoint=<url>`, `/config provider=<openai|ollama|llamacpp|openrouter|gemini>`, `/config namespace=<ns>`, `/config lockdown=<0|1>`, `/config max_steps=<n>`, `/config gas_limit=<n>`\n",
                    )
                )
                return True
            for token in arg.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                k_low = k.strip().lower()
                v_clean = v.strip()
                if k_low == "model":
                    self.harness.model = v_clean
                elif k_low == "endpoint":
                    self.harness.endpoint = v_clean
                elif k_low == "provider":
                    os.environ["ACTORII_LLM_PROVIDER"] = v_clean
                    from actorii.ai import resolve_default_endpoint_for_model
                    self.harness.endpoint = resolve_default_endpoint_for_model(self.harness.model)
                elif k_low == "api_key":
                    os.environ["ACTORII_LLM_API_KEY"] = v_clean
                elif k_low == "namespace":
                    self.vm.eval_chunk(f'setNamespace(Actor[#Memory], "{v_clean}")!!')
                elif k_low == "lockdown":
                    os.environ["ACTORII_HOST_LOCKDOWN"] = "1" if v_clean in ("1", "true", "on") else "0"
                elif k_low == "max_steps" and v_clean.lstrip("-").isdigit():
                    self.vm.eval_chunk(f"setMaxSteps(Actor[#Harness], {int(v_clean)})!!")
                elif k_low == "max_failures" and v_clean.lstrip("-").isdigit():
                    self.vm.eval_chunk(f"setMaxFailures(Actor[#Harness], {int(v_clean)})!!")
                elif k_low == "gas_limit" and v_clean.isdigit():
                    self.vm.eval_chunk(f"setGasLimit(Actor[#Harness], {int(v_clean)})!!")
            self._emit(self._c(GREEN + BOLD, f"✔ Updated runtime configuration ({arg})."))
            return True

        if cmd == "/prompt":
            if arg.lower().startswith("set "):
                new_p = arg[4:].strip()
                self.harness.system_prompt = new_p
                self._emit(self._c(GREEN + BOLD, "✔ Updated `Actor[#Harness]` system prompt."))
                return True
            self._emit(f"\n{self._c(CYAN + BOLD, '=== Active `Actor[#Harness]` System Prompt ===')}")
            self._emit(self.harness.system_prompt)
            self._emit(f"\n{self._c(CYAN + BOLD, '=== Live Pool Method Summary (`poolSummary`) ===')}")
            self._emit(f"  {self.sandbox.poolSummary()}\n")
            return True

        if cmd == "/reset":
            self.vm.eval_chunk("resetBreaker(Actor[#Harness])!!")
            self._emit(f"{self._c(GREEN, '✔ Circuit breaker reset (`status=#ready`, `failures=0`).')}")
            return True

        if cmd == "/dock":
            run_curses_dock(self)
            return True

        self._emit(self._c(RED, f"Unknown slash command '{cmd}'. Type /help for available commands."))
        return True


# =============================================================================
# Full-Screen Curses Split-Pane Dock Mode (`--dock` / `/dock`)
# =============================================================================


def run_curses_dock(session: ActorTUISession) -> None:
    """Launch the full-screen split-pane Curses TUI dock with live Object Pool sidebar."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        session._emit("Cannot launch full-screen curses dock in a non-TTY stream.")
        return

    def _curses_main(stdscr: Any) -> None:
        curses.curs_set(1)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)

        input_buf = ""
        scroll_offset = 0
        show_sidebar = True
        history: List[str] = []
        hist_idx = -1

        # Capture session output into clean plain-text lines for the left pane
        prev_use_color = session.use_color
        session.use_color = False
        if not session.transcript:
            session.transcript.append("Welcome to ActorII v4.1 Split-Pane TUI Dock.")
            session.transcript.append("Type `/goal <task>`, `3 + 4 * 5!!`, or `/help`. Press ESC or `/quit` to exit dock.")

        try:
            while True:
                stdscr.erase()
                max_y, max_x = stdscr.getmaxyx()
                if max_y < 8 or max_x < 40:
                    stdscr.addstr(0, 0, "Terminal window too small.")
                    stdscr.refresh()
                    ch = stdscr.getch()
                    if ch in (27, ord("q")):
                        break
                    continue

                # 1. Top Status Bar
                ribbon = session.status_ribbon_text()[: max_x - 1]
                stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
                stdscr.addstr(0, 0, ribbon.ljust(max_x - 1))
                stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

                sidebar_w = min(44, max_x // 3) if (show_sidebar and max_x >= 80) else 0
                main_w = max_x - sidebar_w - (1 if sidebar_w else 0)
                body_h = max_y - 3

                # 2. Left Main Transcript Pane
                wrapped_lines: List[str] = []
                for raw_ln in session.transcript[-200:]:
                    for sub_ln in str(raw_ln).splitlines() or [""]:
                        chunks = textwrap.wrap(sub_ln, width=max(20, main_w - 2)) or [""]
                        wrapped_lines.extend(chunks)

                visible_start = max(0, len(wrapped_lines) - body_h - scroll_offset)
                visible_slice = wrapped_lines[visible_start : visible_start + body_h]

                for idx, txt in enumerate(visible_slice):
                    y = 1 + idx
                    color = 0
                    if txt.strip().startswith("✔") or txt.strip().startswith("💬") or "⇒" in txt:
                        color = curses.color_pair(2) | (curses.A_BOLD if "💬" in txt else 0)
                    elif txt.strip().startswith("✖") or "ERROR" in txt:
                        color = curses.color_pair(4)
                    elif txt.strip().startswith("⚡") or txt.strip().startswith("▶"):
                        color = curses.color_pair(1) | curses.A_BOLD
                    elif txt.strip().startswith("❯"):
                        color = curses.color_pair(3) | curses.A_BOLD
                    stdscr.addstr(y, 1, txt[: main_w - 2], color)

                # 3. Right Sidebar Dock (Live Object Pool + Scratchpad + Classes + Chat Turns)
                if sidebar_w > 0:
                    sep_x = main_w
                    for y in range(1, max_y - 2):
                        stdscr.addch(y, sep_x, curses.ACS_VLINE, curses.color_pair(1))

                    sx = sep_x + 2
                    sw = sidebar_w - 3
                    sy = 1

                    def _put_side(text: str, attr: int = 0) -> None:
                        nonlocal sy
                        if sy < max_y - 2:
                            stdscr.addstr(sy, sx, text[:sw], attr)
                            sy += 1

                    _put_side("■ LIVE OBJECT POOL (`Actor[]`)", curses.color_pair(1) | curses.A_BOLD)
                    for k, v in sorted(session.get_pool_objects().items())[:7]:
                        cls_n = getattr(v, "class_name", type(v).__name__)
                        _put_side(f" #{k}: {cls_n}", curses.color_pair(2))

                    sy += 1
                    _put_side("■ HARNESS SCRATCHPAD", curses.color_pair(3) | curses.A_BOLD)
                    scratch = session.get_scratchpad_dict()
                    if not scratch:
                        _put_side(" (empty)")
                    else:
                        for k, v in list(scratch.items())[:6]:
                            _put_side(f" #{k} = {v!r}", curses.color_pair(3))

                    sy += 1
                    _put_side("■ USER & AGENT CLASSES", curses.color_pair(5) | curses.A_BOLD)
                    for cname, cls in list(session.get_user_classes().items())[:5]:
                        m_cnt = len(cls._methods)
                        _put_side(f" {cname} ({m_cnt} methods)")

                    sy += 1
                    turns = session.get_chat_history(local_only=True)
                    _put_side(f"■ RECENT CHAT TURNS ({len(turns)})", curses.color_pair(2) | curses.A_BOLD)
                    if not turns:
                        _put_side(" (no turns yet)")
                    else:
                        for t in turns[-4:]:
                            if isinstance(t, dict):
                                _put_side(
                                    f" T{t.get('turn')}: {str(t.get('user', ''))[:14]} -> {str(t.get('reply', ''))[:16]}",
                                    curses.color_pair(2),
                                )
                    sy += 1
                    _put_side("F2:Sidebar F3:Clear ESC:Exit", curses.color_pair(1))

                # 4. Bottom Separator & Input Prompt Bar
                stdscr.hline(max_y - 2, 0, curses.ACS_HLINE, max_x - 1)
                prompt_prefix = f"actorii [{session.harness._fields.get('status')}] ❯ "
                avail_in = max(10, max_x - len(prompt_prefix) - 2)
                tail_input = input_buf[-avail_in:]
                stdscr.addstr(max_y - 1, 0, prompt_prefix, curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(max_y - 1, len(prompt_prefix), tail_input)
                stdscr.move(max_y - 1, min(max_x - 1, len(prompt_prefix) + len(tail_input)))
                stdscr.refresh()

                ch = stdscr.getch()
                if ch == 27:  # ESC exits dock back to REPL/shell
                    break
                elif ch == curses.KEY_F2:
                    show_sidebar = not show_sidebar
                elif ch == curses.KEY_F3:
                    session.handle_input("/clear")
                elif ch == curses.KEY_PPAGE:
                    scroll_offset = min(max(0, len(wrapped_lines) - body_h), scroll_offset + 5)
                elif ch == curses.KEY_NPAGE:
                    scroll_offset = max(0, scroll_offset - 5)
                elif ch == curses.KEY_UP:
                    if history and hist_idx + 1 < len(history):
                        hist_idx += 1
                        input_buf = history[-(hist_idx + 1)]
                elif ch == curses.KEY_DOWN:
                    if hist_idx > 0:
                        hist_idx -= 1
                        input_buf = history[-(hist_idx + 1)]
                    else:
                        hist_idx = -1
                        input_buf = ""
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    input_buf = input_buf[:-1]
                elif ch in (10, 13, curses.KEY_ENTER):
                    cmd_str = input_buf.strip()
                    input_buf = ""
                    hist_idx = -1
                    scroll_offset = 0
                    if not cmd_str:
                        continue
                    history.append(cmd_str)
                    session.transcript.append(f"❯ {cmd_str}")
                    if cmd_str.lower() in ("/quit", "/exit", "/q", "/dock"):
                        break
                    # Temporarily redirect stdout to session.transcript
                    session.handle_input(cmd_str)
                elif 32 <= ch <= 126:
                    input_buf += chr(ch)
        finally:
            session.use_color = prev_use_color

    curses.wrapper(_curses_main)


# =============================================================================
# Readline Interactive Console Mode & CLI Entrypoint
# =============================================================================


def _configure_readline(session: ActorTUISession) -> None:
    hist_file = Path.home() / ".actorii_tui_history"
    try:
        if hist_file.is_file():
            readline.read_history_file(str(hist_file))
    except Exception:
        pass

    def _completer(text: str, state: int) -> Optional[str]:
        candidates: List[str] = []
        if text.startswith("/"):
            candidates = [c for c in SLASH_COMMANDS.keys() if c.startswith(text)]
        else:
            pool_keys = [f"Actor[#{k}]" for k in session.get_pool_objects().keys()]
            cls_names = list(session.get_user_classes().keys())
            builtins = ["step(Actor[#Harness])!!", "runGoal(Actor[#Harness], ", "sieve(Actor[#Sam], 100)!!"]
            all_tokens = pool_keys + cls_names + builtins + list(SLASH_COMMANDS.keys())
            candidates = [tok for tok in all_tokens if tok.startswith(text)]
        return candidates[state] if state < len(candidates) else None

    try:
        readline.set_completer_delims(" \t\n;")
        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ActorII v4.1 — Hermes/Antigravity-Style TUI & Metacircular Agent Harness"
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        help="Execute one or more commands/prompts non-interactively and exit.",
    )
    parser.add_argument(
        "--dock",
        action="store_true",
        help="Launch directly into the full-screen split-pane Curses TUI dock.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name for `Actor[#Harness]` (default: ACTORII_LLM_MODEL or qwen3.8:latest).",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible endpoint URL (default: http://127.0.0.1:8080/v1/chat/completions).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Default max steps per `/goal` execution (0 = unlimited, default: 0).",
    )
    parser.add_argument(
        "--gas-limit",
        type=int,
        default=100000,
        help="Per-step opcode gas budget (default: 100000).",
    )
    parser.add_argument(
        "--load-image",
        default=None,
        help="Optional `.img` snapshot path to load at startup.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from `~/.actorii/last_session.img` if present.",
    )
    parser.add_argument(
        "--connect",
        default=None,
        help="Attach to a running ActorII Web Studio server (e.g., http://127.0.0.1:8090).",
    )
    parser.add_argument(
        "--usage-file",
        default=None,
        help="Optional JSON file path to write token usage and step telemetry upon exit.",
    )
    args = parser.parse_args(argv)

    session = ActorTUISession(
        model=args.model,
        endpoint=args.endpoint,
        max_steps=args.max_steps,
        gas_limit=args.gas_limit,
        remote_studio_url=args.connect,
    )
    session.usage_file = args.usage_file

    if args.load_image:
        session.handle_input(f"/load {args.load_image}")
    elif args.resume and DEFAULT_SESSION_IMG.is_file():
        session.handle_input(f"/load {DEFAULT_SESSION_IMG}")

    if args.command:
        for cmd in args.command:
            keep_going = session.handle_input(cmd)
            if not keep_going:
                break
        session.auto_save_session()
        if args.usage_file:
            session.flush_usage_file()
        return 0

    if args.dock:
        run_curses_dock(session)
        return 0

    # Interactive Streaming Console Mode (Hermes / Antigravity CLI style)
    _configure_readline(session)
    print(session._c(BG_DARK + CYAN + BOLD, f" {session.status_ribbon_text()} "))
    print(
        session._c(
            GRAY,
            "Type `/goal <task>` for autonomous agent loops, `code!!` for Actor 4.1 eval, `/dock` for split-pane UI, or `/help`.",
        )
    )

    hist_file = Path.home() / ".actorii_tui_history"
    try:
        while True:
            status_sym = session.harness._fields.get("status", Symbol("idle"))
            prompt_str = (
                f"{session._c(CYAN + BOLD, 'actorii')} "
                f"{session._c(GRAY, f'[{session.harness.name}|{status_sym}]')} "
                f"{session._c(GREEN + BOLD, '❯ ')} "
            )
            try:
                user_line = input(prompt_str)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n" + session._c(GRAY, "(Interrupted — type /quit to exit)"))
                continue

            if not session.handle_input(user_line):
                break
    finally:
        session.auto_save_session()
        try:
            readline.write_history_file(str(hist_file))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
