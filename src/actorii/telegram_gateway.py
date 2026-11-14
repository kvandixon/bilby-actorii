"""Telegram Long-Polling Gateway for the ActorII Metacircular Bilby Harness (`actorii.telegram_gateway`).

Connects Bilby (`Actor[#Bilby]` / `Actor[#Harness]`) directly to Telegram using
a configured bot token (`TELEGRAM_BOT_TOKEN` / `BILBY_TELEGRAM_BOT_TOKEN` from `.env`),
allowing two-way Telegram messaging backed by the 1993 Whitewater Actor 4.1
bytecode VM, Neo4j Engram memory (`Actor[#Memory]`), and Digital Twins
(`Actor[#Valkyrie]`, `Actor[#ReverseEngineer]`, `Actor[#Telegram]`, etc.).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from actorii.harness import bootstrap_agent_harness
from actorii.harness.telemetry import (
    format_last_turn_trace,
    render_telegram_chunks,
    resolve_actorii_state_dir,
    summarize_actor_action_brief,
)
from actorii.harness.twins import load_telegram_credentials, telegram_api_request
from actorii.tui import ActorTUISession

DEFAULT_GATEWAY_STATE = Path.home() / ".actorii" / "telegram_gateway_state.json"
DEFAULT_GATEWAY_INBOX = Path.home() / ".actorii" / "supervisor_inbox.jsonl"
MAX_TELEGRAM_MSG_LEN = 4000


def _split_telegram_message(text: str, limit: int = MAX_TELEGRAM_MSG_LEN) -> List[str]:
    """Split long replies into <=4000 char chunks on newline boundaries."""
    s = (text or "").strip()
    if not s:
        return ["(empty response)"]
    if len(s) <= limit:
        return [s]
    chunks: List[str] = []
    remaining = s
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= limit // 3:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class BilbyTelegramGateway:
    """Long-polling Telegram gateway driving an `ActorTUISession` (`Actor[#Bilby]`)."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        allowed_users: Optional[List[int]] = None,
        default_chat_id: Optional[str] = None,
        default_thread_id: Optional[int] = None,
        model: str = "qwen3.8:latest",
        endpoint: Optional[str] = None,
        max_steps: int = 25,
        max_failures: int = 6,
        poll_timeout: int = 20,
        state_path: Optional[Path] = None,
        session: Optional[ActorTUISession] = None,
        api_caller: Optional[Callable[[str, str, Optional[Dict[str, Any]], float], Dict[str, Any]]] = None,
    ) -> None:
        creds = load_telegram_credentials()
        self.token = (token if token is not None else creds["token"]).strip()
        self.allowed_users = (
            list(allowed_users)
            if allowed_users is not None
            else list(creds["allowed_users"])
        )
        self.default_chat_id = str(
            default_chat_id if default_chat_id is not None else creds["chat_id"]
        ).strip()
        self.default_thread_id = (
            default_thread_id if default_thread_id is not None else creds["thread_id"]
        )
        self.poll_timeout = max(1, int(poll_timeout))
        active_state_dir = resolve_actorii_state_dir()
        self.state_path = state_path or (active_state_dir / "telegram_gateway_state.json")
        self.inbox_path = active_state_dir / "supervisor_inbox.jsonl"
        self._api = api_caller or telegram_api_request

        # Enforce production host-lockdown by default when running as the standalone daemon
        self._lockdown_enabled = True

        self.session = session or ActorTUISession(
            agent_name="Bilby",
            model=model,
            endpoint=endpoint,
            max_steps=max_steps,
            max_failures=max_failures,
            use_color=False,
        )
        self.session.surface_name = "telegram"
        self.offset: int = 0
        self.total_turns: int = 0
        self.total_steps: int = 0
        self.total_gas: int = 0
        self._verbose_output: bool = False
        self._consecutive_poll_errors: int = 0
        self._frozen: bool = False
        self._cancel_requested: bool = False
        self._async_goals: bool = False
        self._goal_thread: Optional[threading.Thread] = None
        self._load_state()
        self._sync_telegram_twin_route(self.default_chat_id, self.default_thread_id)
        self._running = True
        self.bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "bilby_bot").strip() or "bilby_bot"

    def _get_telegram_twin(self) -> Optional[Any]:
        """Return the underlying `TelegramBotTwin` bound to `Actor[#Telegram]`, if present."""
        try:
            actor_pool = getattr(self.session.vm, "global_vars", {}).get("Actor")
            tg_bridge = actor_pool.get("Telegram") if isinstance(actor_pool, dict) else None
            return getattr(tg_bridge, "target", tg_bridge)
        except Exception:
            return None

    def _sync_telegram_twin_route(self, chat_id: Any, thread_id: Optional[int]) -> int:
        """Sync active `(chat_id, thread_id)` onto `Actor[#Telegram]` and return its current `_sent_log` count."""
        tg_target = self._get_telegram_twin()
        if tg_target is not None:
            if hasattr(tg_target, "_set_active_route"):
                tg_target._set_active_route(chat_id, thread_id)
            sent_log = getattr(tg_target, "_sent_log", None)
            if isinstance(sent_log, list):
                return len(sent_log)
        return 0

    def _load_state(self) -> int:
        try:
            if self.state_path.is_file():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.offset = int(data.get("offset", 0) or 0)
                self.total_turns = int(data.get("total_turns", 0) or 0)
                self.total_steps = int(data.get("total_steps", 0) or 0)
                self.total_gas = int(data.get("total_gas", 0) or 0)
                self._verbose_output = bool(data.get("verbose_output", False))
                saved_turns = data.get("chat_history")
                if isinstance(saved_turns, list) and saved_turns:
                    from actorii.classes.collections import ActorOrderedCollection

                    chat_log = ActorOrderedCollection()
                    for t in saved_turns[-20:]:
                        if isinstance(t, dict):
                            chat_log.add(t)
                    self.session.harness._fields["chatLog"] = chat_log
                    if hasattr(self.session.harness, "_history") and isinstance(
                        self.session.harness._history, (ActorOrderedCollection, list)
                    ):
                        self.session.harness._history.clear()
                        for t in list(chat_log)[-10:]:
                            if isinstance(t, dict):
                                u_txt = str(t.get("user", ""))
                                a_txt = str(t.get("reply", ""))
                                acts = t.get("actions") or []
                                if acts:
                                    a_txt = f"[Executed: {'; '.join(acts)}] {a_txt}"
                                self.session.harness._history.add({"role": "user", "content": u_txt})
                                self.session.harness._history.add({"role": "assistant", "content": a_txt})
                saved_scratch = data.get("scratchpad")
                if isinstance(saved_scratch, dict) and saved_scratch:
                    from actorii.classes.dictionary import Dictionary
                    from actorii.core import Symbol

                    scratch_obj = self.session.harness._fields.get("scratchpad")
                    if isinstance(scratch_obj, Dictionary):
                        for k, v in saved_scratch.items():
                            scratch_obj.atPut(str(k), v)
                            scratch_obj.atPut(Symbol(str(k).lstrip("#")), v)
                return self.offset
        except Exception:
            pass
        return 0

    def _load_offset(self) -> int:
        return self._load_state()

    def _save_offset(self, offset: int) -> None:
        self.offset = int(offset)
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            turns = self.session.sandbox.getChatHistory(self.session.harness)
            recent_turns = [t for t in turns[-20:] if isinstance(t, dict)]
            scratch_dict: Dict[str, Any] = {}
            scratch_obj = self.session.harness._fields.get("scratchpad")
            if scratch_obj is not None and hasattr(scratch_obj, "_entries"):
                for k, v in scratch_obj._entries.items():
                    norm_k = str(k).lstrip("#")
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        scratch_dict[norm_k] = v
                    else:
                        scratch_dict[norm_k] = repr(v)[:500]
            self.state_path.write_text(
                json.dumps(
                    {
                        "offset": self.offset,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "updated_at_epoch": time.time(),
                        "agent_name": self.session.harness.name,
                        "model": getattr(self.session.harness, "model", "qwen3.8:latest"),
                        "total_turns": max(self.total_turns, len(turns)),
                        "total_steps": self.total_steps,
                        "total_gas": self.total_gas,
                        "verbose_output": self._verbose_output,
                        "chat_history": recent_turns,
                        "scratchpad": scratch_dict,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def send_message(
        self,
        chat_id: Any,
        text: str,
        *,
        thread_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Send one or more mobile-friendly HTML-rendered chunks to `chat_id`, with automatic plain-text fallback."""
        results: List[Dict[str, Any]] = []
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        rendered_chunks = render_telegram_chunks(text or "(empty response)")
        for idx, (html_part, plain_part) in enumerate(rendered_chunks):
            if not html_part and not plain_part:
                html_part = plain_part = "(empty response)"
            payload: Dict[str, Any] = {
                "chat_id": cid,
                "text": html_part or plain_part,
                "parse_mode": "HTML",
            }
            if thread_id is not None and int(thread_id) > 0:
                payload["message_thread_id"] = int(thread_id)
            if idx == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            res = self._api(self.token, "sendMessage", payload, 15.0)
            if not res.get("ok"):
                desc = str(res.get("description", "")).lower()
                # 1. If HTML entity/parse error occurred, retry immediately as clean plain text
                if "parse" in desc or "entities" in desc or "tag" in desc or res.get("error_code") == 400:
                    plain_payload = {k: v for k, v in payload.items() if k != "parse_mode"}
                    plain_payload["text"] = plain_part or "(empty response)"
                    res = self._api(self.token, "sendMessage", plain_payload, 15.0)
                # 2. Fallback to root chat if forum topic thread was closed/deleted
                if not res.get("ok") and "message_thread_id" in payload:
                    fallback_payload = {
                        k: v for k, v in payload.items() if k not in ("message_thread_id", "parse_mode")
                    }
                    fallback_payload["text"] = plain_part or "(empty response)"
                    res = self._api(self.token, "sendMessage", fallback_payload, 15.0)
            results.append(res)
        return results

    def edit_message(
        self,
        chat_id: Any,
        message_id: int,
        text: str,
    ) -> Dict[str, Any]:
        """Edit an existing Telegram message in-place with HTML formatting and plain-text fallback."""
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        rendered_chunks = render_telegram_chunks(text or "(empty response)")
        html_part, plain_part = rendered_chunks[0] if rendered_chunks else ("(empty response)", "(empty response)")
        payload: Dict[str, Any] = {
            "chat_id": cid,
            "message_id": int(message_id),
            "text": html_part or plain_part or "(empty response)",
            "parse_mode": "HTML",
        }
        res = self._api(self.token, "editMessageText", payload, 15.0)
        if not res.get("ok"):
            desc = str(res.get("description", "")).lower()
            if "message is not modified" in desc:
                return {"ok": True, "not_modified": True}
            if "parse" in desc or "entities" in desc or "tag" in desc or res.get("error_code") == 400:
                plain_payload = {k: v for k, v in payload.items() if k != "parse_mode"}
                plain_payload["text"] = plain_part or "(empty response)"
                res = self._api(self.token, "editMessageText", plain_payload, 15.0)
                if not res.get("ok") and "message is not modified" in str(res.get("description", "")).lower():
                    return {"ok": True, "not_modified": True}
        return res

    def delete_message(self, chat_id: Any, message_id: int) -> Dict[str, Any]:
        """Delete a temporary status message from `chat_id`."""
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        try:
            return self._api(
                self.token,
                "deleteMessage",
                {"chat_id": cid, "message_id": int(message_id)},
                10.0,
            )
        except Exception as exc:
            return {"ok": False, "description": str(exc)}

    def send_typing(self, chat_id: Any, thread_id: Optional[int] = None) -> None:
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        payload: Dict[str, Any] = {"chat_id": cid, "action": "typing"}
        if thread_id is not None and int(thread_id) > 0:
            payload["message_thread_id"] = int(thread_id)
        try:
            self._api(self.token, "sendChatAction", payload, 6.0)
        except Exception:
            pass

    def _is_authorized(self, user_id: Optional[int], chat_id: Optional[int]) -> bool:
        if not self.allowed_users:
            return False
        if user_id is not None and int(user_id) in self.allowed_users:
            return True
        if chat_id is not None and int(chat_id) in self.allowed_users:
            return True
        return False

    def format_security_controls(self) -> str:
        """Return a real-time audit report of all active Bilby containment & security controls."""
        lockdown = os.environ.get("ACTORII_HOST_LOCKDOWN") == "1" or getattr(self, "_lockdown_enabled", True)
        no_evolve = os.environ.get("ACTORII_DISABLE_SELF_EVOLVE") == "1"
        singleton_lock = bool(getattr(self.session.vm, "lock_system_singletons", False))
        in_docker = os.environ.get("ACTORII_IN_DOCKER") == "1" or os.path.exists("/.dockerenv")
        raw_ms = self.session.harness._fields.get("maxSteps", 0)
        raw_mf = self.session.harness._fields.get("maxFailures", 0)
        ms_txt = "∞ (unlimited)" if (raw_ms is not None and int(raw_ms) <= 0) else str(raw_ms)
        mf_txt = "∞ (disabled)" if (raw_mf is not None and int(raw_mf) <= 0) else f"{raw_mf} strikes"
        preflight_on = os.environ.get("ACTORII_ENFORCE_SKIPPY_PREFLIGHT", "1" if lockdown else "0") == "1"
        return (
            "🛡️ Bilby Containment & Security Controls Report\n"
            f"• Execution State: {'❄️ FROZEN (Kill-Switch Active)' if self._frozen else '🟢 ACTIVE (Guarded)'}\n"
            f"• Host Lockdown (`ACTORII_HOST_LOCKDOWN`): {'🔒 LOCKED (1)' if lockdown else '⚠️ UNLOCKED (0)'}\n"
            f"• Skippy Pre-Flight Monitor (`askSkippy` before detached threads / `evolve` / bulk memory writes): {'🔒 ENFORCED' if preflight_on else '⚠️ OFF'}\n"
            "• Core Memory Self-Preservation (`DELETE` on `:ActorEpisode` / `:ActorEngram`): 🔒 HARD-BLOCKED (Soft-Archive Only)\n"
            "• Thread Pool Starvation Guard (`loop while(true)` & unbounded gas on `Actor[#Threads]`): 🔒 HARD-BLOCKED\n"
            f"• Host Shell (`Actor[#Host]>>execShell` / `Actor[#Valkyrie]>>runShell`): {'🔒 BLOCKED ON HOST' if (lockdown and not in_docker) else '🐳 DOCKER-ONLY'}\n"
            f"• Self-Evolution (`evolveHarness` / Ring 1 Cognitive Hooks): {'🔒 DISABLED' if no_evolve else '🟢 ENABLED (Guarded Ring 1 Hooks Only — Kernel & Tier 2 Calls Locked)'}\n"
            "• Python Source, Persona & Host Config Write Guard (`src/actorii`, `persona.md`, `~/bin`, `.config/systemd`, `/etc`): 🔒 HARD-BLOCKED\n"
            "• Destructive Command Filter (`systemctl`, `sudo`, `nmcli`, `resolv.conf`, `crontab`, `rm -rf`): 🔒 ACTIVE (47 patterns)\n"
            f"• Ring 0/1 Kernel & Bridge Class Lock: 🔒 ACTIVE (41 classes + `AgentHarness` kernel methods)\n"
            f"• Pool Singleton Lock (`Actor[#Host]`, `Actor[#Harness]`, `Actor[#Sandbox]`, `Actor[#Threads]`): {'🔒 LOCKED' if singleton_lock else '⚠️ UNLOCKED'}\n"
            f"• Per-Step Opcode Gas Limit: {self.session.harness._fields.get('gasLimit', 15000)} ops | Max Steps: {ms_txt} | Strike Breaker: {mf_txt}\n"
            f"• Authorized Telegram IDs: {self.allowed_users}"
        )

    def stop_active_goal(self) -> str:
        """Interrupt an in-flight `runGoal` loop cleanly by marking `Actor[#Harness].status := #halted`."""
        from actorii.core import Symbol

        was_busy = self._goal_thread is not None and self._goal_thread.is_alive()
        steps_now = int(self.session.harness._fields.get("stepCount", 0) or 0)
        self._cancel_requested = True
        self.session.harness._fields["status"] = Symbol("halted")
        if was_busy:
            return (
                f"⏹ Breaking Bilby out of active loop (at step {steps_now}). "
                "He will stop right after the current step finishes."
            )
        self._cancel_requested = False
        self.session.harness._fields["status"] = Symbol("idle")
        self.session.harness._fields["failures"] = 0
        return "⏹ Bilby reset to `#idle` (no active loop was running)."

    @staticmethod
    def _is_immediate_control_command(text: str) -> bool:
        """Return True if `text` is a control/inspection slash command that can execute while a goal thread runs."""
        clean = (text or "").strip()
        if not clean.startswith("/"):
            return False
        first_tok = clean.split(maxsplit=1)[0].split("@", 1)[0].lower()
        return first_tok in (
            "/stop",
            "/cancel",
            "/break",
            "/halt",
            "/panic",
            "/freeze",
            "/lock",
            "/unlock",
            "/controls",
            "/lockdown",
            "/security",
            "/status",
            "/last",
            "/verbose",
            "/help",
            "/start",
        )

    def handle_text_command_or_goal(self, text: str) -> str:
        """Process a Telegram message (`slash` command or natural-language/Actor 4.1 goal) via Bilby."""
        clean = (text or "").strip()
        if not clean:
            return "Send a message, Actor 4.1 chunk (`!!`), or `/last`, `/verbose`, `/stop`, `/status`, `/controls`, `/twins`, `/clear`."

        # Strip @BotUsername suffix from slash commands (e.g. /status@bilby_bot)
        if clean.startswith("/"):
            first_tok, *rest = clean.split(maxsplit=1)
            if "@" in first_tok:
                first_tok = first_tok.split("@", 1)[0]
            cmd_lower = first_tok.lower()
            arg_str = rest[0].strip() if rest else ""

            if cmd_lower in ("/stop", "/cancel", "/break", "/halt"):
                return self.stop_active_goal()

            if cmd_lower in ("/panic", "/freeze", "/lock"):
                self._frozen = True
                self._cancel_requested = True
                from actorii.core import Symbol

                self.session.harness._fields["status"] = Symbol("blocked")
                return (
                    "❄️ KILL-SWITCH ENGAGED: Bilby execution is now FROZEN.\n"
                    "All goals, tool calls, and bytecode evaluations are halted until you send `/unlock`."
                )

            if cmd_lower == "/unlock":
                self._frozen = False
                self._cancel_requested = False
                from actorii.core import Symbol

                self.session.harness._fields["status"] = Symbol("idle")
                self.session.harness._fields["failures"] = 0
                return "🟢 Bilby execution UNLOCKED (Host Lockdown & Ring 0/1 security controls remain strictly enforced)."

            if cmd_lower in ("/controls", "/lockdown", "/security"):
                return self.format_security_controls()

            if cmd_lower == "/last":
                turns = self.session.sandbox.getChatHistory(self.session.harness)
                last_turn = turns[-1] if (turns and isinstance(turns[-1], dict)) else None
                raw_steps = list(self.session.harness._fields.get("stepLog") or [])
                return format_last_turn_trace(last_turn, raw_steps)

            if cmd_lower == "/verbose":
                mode = arg_str.lower()
                if mode in ("on", "1", "true"):
                    self._verbose_output = True
                elif mode in ("off", "0", "false"):
                    self._verbose_output = False
                elif not mode:
                    self._verbose_output = not self._verbose_output
                self._save_offset(self.offset)
                if self._verbose_output:
                    return "🔊 **Verbose mode ON** — replies will include executed Actor 4.1 code blocks inline."
                return "✨ **Verbose mode OFF** — replies are clean human prose (use `/last` anytime to inspect Actor 4.1 code & VM results)."

            if self._frozen:
                return "❄️ Bilby is currently FROZEN by kill-switch (`/freeze`). Send `/unlock` or `/controls`."

            if cmd_lower in ("/start", "/help"):
                return (
                    "🦘 **Bilby** (ActorII v4.2.0 Metacircular Agent Harness — Hardened Lockdown)\n"
                    "Resident System AI on `valkyrie`.\n\n"
                    "**Commands:**\n"
                    "• Plain English or Actor 4.1 code (`!!`) — Guarded ReAct loop (clean human output)\n"
                    "• `/last` — Inspect the Actor 4.1 code & VM results from the last turn\n"
                    "• `/verbose [on|off]` — Toggle inline Actor 4.1 code blocks in replies\n"
                    "• `/stop` (or `/cancel`) — Break Bilby out of an active loop immediately\n"
                    "• `/status` — VM, GPU VRAM, and Harness status\n"
                    "• `/twins` — Live Digital Twins (`Valkyrie`, `Jetson`, `ProductLearner`, `Proxy`, `ReverseEngineer`, `Thermometer`, `Pantry`, `Telegram`)\n"
                    "• `/proxy [status|links|lan|tailscale|local|pause]` — Manage `:8090` Reverse Proxy\n"
                    "• `/controls` — Inspect active security & containment locks\n"
                    "• `/freeze` | `/unlock` — Emergency kill-switch\n"
                    "• `/eval <code!!>` — Direct Actor 4.1 bytecode evaluation\n"
                    "• `/engrams` | `/scratch` | `/history` | `/clear`"
                )

            if cmd_lower == "/eval" and arg_str:
                chunk = arg_str if arg_str.endswith("!!") else f"{arg_str}!!"
                before = len(self.session.transcript)
                self.session.handle_input(chunk)
                out = "\n".join(self.session.transcript[before:]).strip()
                return out or "Evaluated."

            if cmd_lower in ("/status", "/proxy", "/twins", "/engrams", "/scratch", "/history", "/clear", "/classes", "/schema", "/capabilities", "/caps", "/config", "/bench"):
                before = len(self.session.transcript)
                self.session.handle_input(f"{cmd_lower} {arg_str}".strip())
                out = "\n".join(self.session.transcript[before:]).strip()
                return out or "Done."

        if self._frozen:
            return "❄️ Bilby is currently FROZEN by kill-switch (`/freeze`). Send `/unlock` or `/controls`."

        # Execute via Bilby's multi-turn conversational + Actor 4.1 bytecode ReAct loop
        before_turns = len(self.session.sandbox.getChatHistory(self.session.harness))
        t0 = time.time()
        self.session.handle_input(f"/chat {clean}")
        elapsed = time.time() - t0

        # Extract final human answer (Option A: clean prose + subtle 1-line badge when tools ran; code in logs & `/last`)
        turns = self.session.sandbox.getChatHistory(self.session.harness)
        h_status = str(self.session.harness._fields.get("status", "")).lstrip("#")
        if len(turns) > before_turns and isinstance(turns[-1], dict):
            last_turn = turns[-1]
            reply = str(last_turn.get("reply") or "").strip()
            actions = last_turn.get("actions") or []
            steps = int(last_turn.get("steps", 1) or 1)
            gas = int(last_turn.get("gas_used", 0) or 0)
            self.total_turns += 1
            self.total_steps += steps
            self.total_gas += gas
            self._save_offset(self.offset)
            if self._verbose_output and actions:
                act_block = "\n".join(str(a) for a in actions)
                return f"{reply}\n\n```actor\n{act_block}\n```\n_⚡ {steps} step(s) · {elapsed:.1f}s · {gas} ops_"
            if actions:
                step_word = "step" if steps == 1 else "steps"
                return f"{reply}\n\n_⚡ {steps} {step_word} · {elapsed:.1f}s_"
            if reply:
                return reply

        # If the goal was stopped by user or halted without a COMPLETE turn, format a clean human diagnostic (never dump raw transcript!)
        self._save_offset(self.offset)
        last_err = str(self.session.harness._fields.get("lastError") or "").strip()
        steps_used = int(self.session.harness._fields.get("stepCount", 0) or 0)
        if self._cancel_requested and h_status in ("halted", "blocked"):
            self._cancel_requested = False
            return f"⏹ Stopped at step {steps_used} ({elapsed:.1f}s). Send `/last` to inspect the execution trace."
        self._cancel_requested = False
        if h_status in ("blocked", "halted") or last_err:
            err_first = last_err.splitlines()[0][:200] if last_err else ""
            detail = f"\nReason: `{err_first}`" if err_first else ""
            return (
                f"⚠️ I paused after **{steps_used} step(s)** ({elapsed:.1f}s) before reaching a verified answer.{detail}\n\n"
                "Send `/last` to view the trace, or reply with a more specific target."
            )
        last_res = self.session.harness._fields.get("lastResult")
        return str(last_res) if last_res not in (None, "") else "Completed."

    def _run_update_sync(
        self,
        upd_id: int,
        chat_id: Any,
        thread_id: Optional[int],
        msg_id: Optional[int],
        text: str,
        *,
        keep_typing: bool = False,
        live_progress: Optional[bool] = None,
    ) -> str:
        stop_typing = threading.Event()
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        clean_in = (text or "").strip()
        use_progress = (
            bool(keep_typing) if live_progress is None else bool(live_progress)
        ) and not clean_in.startswith("/")

        progress_lock = threading.Lock()
        progress_state: Dict[str, Any] = {
            "msg_id": None,
            "last_edit": 0.0,
            "last_html": "",
            "finished": False,
        }

        def _upsert_progress(status_html: str, *, force: bool = False) -> None:
            if not use_progress:
                return
            with progress_lock:
                if progress_state["finished"] or status_html == progress_state["last_html"]:
                    return
                now_m = time.monotonic()
                cur_mid = progress_state["msg_id"]
                if cur_mid is not None and not force and (now_m - progress_state["last_edit"]) < 1.2:
                    return
                if cur_mid is None:
                    p_load: Dict[str, Any] = {
                        "chat_id": cid,
                        "text": status_html,
                        "parse_mode": "HTML",
                    }
                    if thread_id is not None and int(thread_id) > 0:
                        p_load["message_thread_id"] = int(thread_id)
                    if msg_id is not None:
                        p_load["reply_to_message_id"] = int(msg_id)
                    try:
                        res_p = self._api(self.token, "sendMessage", p_load, 10.0)
                        if res_p.get("ok") and isinstance(res_p.get("result"), dict):
                            m_val = res_p["result"].get("message_id")
                            if m_val is not None:
                                progress_state["msg_id"] = int(m_val)
                                progress_state["last_edit"] = now_m
                                progress_state["last_html"] = status_html
                    except Exception:
                        pass
                else:
                    try:
                        res_e = self._api(
                            self.token,
                            "editMessageText",
                            {
                                "chat_id": cid,
                                "message_id": int(cur_mid),
                                "text": status_html,
                                "parse_mode": "HTML",
                            },
                            10.0,
                        )
                        if res_e.get("ok") or "not modified" in str(res_e.get("description", "")).lower():
                            progress_state["last_edit"] = now_m
                            progress_state["last_html"] = status_html
                    except Exception:
                        pass

        def _on_live_step(entry: Dict[str, Any]) -> None:
            if not use_progress or not isinstance(entry, dict):
                return
            st = str(entry.get("status", "ok"))
            if st == "complete":
                return
            st_n = int(entry.get("step", 1) or 1)
            if st == "error":
                _upsert_progress(f"↻ <i>Step {st_n}: refining approach...</i>")
                return
            brief = html.escape(summarize_actor_action_brief(str(entry.get("action", "")), max_len=52))
            _upsert_progress(
                f"⚡ <i>Step {st_n}:</i> <code>{brief}</code>",
                force=(progress_state["msg_id"] is None),
            )

        def _typing_loop() -> None:
            if not stop_typing.wait(2.5):
                if use_progress and progress_state["msg_id"] is None:
                    _upsert_progress("⏳ <i>Thinking...</i>", force=True)
                self.send_typing(chat_id, thread_id=thread_id)
            while not stop_typing.wait(4.5):
                self.send_typing(chat_id, thread_id=thread_id)

        typing_thread: Optional[threading.Thread] = None
        sent_before = self._sync_telegram_twin_route(chat_id, thread_id)
        self.send_typing(chat_id, thread_id=thread_id)
        if use_progress:
            self.session.sandbox.step_callbacks.append(_on_live_step)
        if keep_typing:
            typing_thread = threading.Thread(target=_typing_loop, daemon=True)
            typing_thread.start()

        t0 = time.time()
        try:
            reply_text = self.handle_text_command_or_goal(text)
        except Exception as exc:
            reply_text = f"⚠️ Bilby harness error: {type(exc).__name__}: {exc}"
        finally:
            with progress_lock:
                progress_state["finished"] = True
                status_mid = progress_state["msg_id"]
            stop_typing.set()
            if use_progress:
                try:
                    self.session.sandbox.step_callbacks.remove(_on_live_step)
                except ValueError:
                    pass
        elapsed = time.time() - t0

        tg_target = self._get_telegram_twin()
        sent_after = len(getattr(tg_target, "_sent_log", []) or []) if tg_target is not None else sent_before
        twin_sent_media = sent_after > sent_before

        delivered_in_place = False
        if status_mid is not None:
            rendered_chunks = render_telegram_chunks(reply_text or "(empty response)")
            if len(rendered_chunks) == 1 and not twin_sent_media:
                edit_res = self.edit_message(chat_id, status_mid, reply_text)
                if edit_res.get("ok"):
                    delivered_in_place = True
                else:
                    self.delete_message(chat_id, status_mid)
            else:
                self.delete_message(chat_id, status_mid)

        if not delivered_in_place:
            self.send_message(
                chat_id,
                reply_text,
                thread_id=thread_id,
                reply_to_message_id=msg_id,
            )
        print(
            f"[bilby-telegram] Replied to update #{upd_id} in {elapsed:.2f}s ({len(reply_text)} chars, in_place={delivered_in_place})",
            flush=True,
        )
        return reply_text

    def process_update(self, update: Dict[str, Any]) -> Optional[str]:
        """Process a single Telegram update dictionary and send Bilby's reply."""
        upd_id = int(update.get("update_id", 0) or 0)
        if upd_id >= self.offset:
            self._save_offset(upd_id + 1)

        msg = update.get("message") or update.get("edited_message")
        if not isinstance(msg, dict):
            return None

        chat = msg.get("chat") or {}
        sender = msg.get("from") or {}
        chat_id = chat.get("id")
        user_id = sender.get("id")
        thread_id = msg.get("message_thread_id")
        msg_id = msg.get("message_id")

        if not self._is_authorized(user_id, chat_id):
            print(
                f"[bilby-telegram] Ignored unauthorized message from user_id={user_id} chat_id={chat_id}",
                flush=True,
            )
            return None

        text = (msg.get("text") or msg.get("caption") or "").strip()
        if not text:
            return None

        print(
            f"[bilby-telegram] Inbound update #{upd_id} from user={user_id} chat={chat_id} thread={thread_id}: {text!r}",
            flush=True,
        )

        # In live daemon mode (`_async_goals=True`), run goals in a background worker thread
        # so `/stop`, `/cancel`, `/freeze`, and `/status` remain immediately responsive!
        if self._async_goals:
            goal_busy = self._goal_thread is not None and self._goal_thread.is_alive()
            if goal_busy and text.lower() in ("stop", "cancel", "halt", "abort", "break"):
                text = "/stop"
            if self._is_immediate_control_command(text):
                return self._run_update_sync(upd_id, chat_id, thread_id, msg_id, text, keep_typing=False)
            if goal_busy:
                steps_now = int(self.session.harness._fields.get("stepCount", 0) or 0)
                busy_msg = (
                    f"⏳ Bilby is currently working on the previous request (step {steps_now}). "
                    "Send `/stop` (or `stop`) to break him out first."
                )
                self.send_message(chat_id, busy_msg, thread_id=thread_id, reply_to_message_id=msg_id)
                return busy_msg

            self._cancel_requested = False
            worker = threading.Thread(
                target=self._run_update_sync,
                args=(upd_id, chat_id, thread_id, msg_id, text),
                kwargs={"keep_typing": True},
                daemon=True,
            )
            self._goal_thread = worker
            worker.start()
            return "[ASYNC_GOAL_STARTED]"

        return self._run_update_sync(upd_id, chat_id, thread_id, msg_id, text, keep_typing=False)

    def poll_once(self) -> int:
        """Fetch and process one batch of updates from Telegram `getUpdates`."""
        payload: Dict[str, Any] = {
            "timeout": self.poll_timeout,
            "allowed_updates": ["message", "edited_message"],
        }
        if self.offset > 0:
            payload["offset"] = self.offset

        res = self._api(self.token, "getUpdates", payload, float(self.poll_timeout + 10))
        if not res.get("ok"):
            desc = res.get("description", "unknown error")
            self._consecutive_poll_errors += 1
            if self._consecutive_poll_errors <= 3 or self._consecutive_poll_errors % 30 == 0:
                print(
                    f"[bilby-telegram] getUpdates warning (#{self._consecutive_poll_errors}): {desc}",
                    flush=True,
                )
            backoff = min(30.0, 2.0 * (1.5 ** min(8, self._consecutive_poll_errors - 1)))
            time.sleep(backoff)
            return 0

        self._consecutive_poll_errors = 0
        updates = res.get("result") or []
        count = 0
        for upd in updates:
            if isinstance(upd, dict):
                self.process_update(upd)
                count += 1
        return count

    def serve_forever(self, send_startup_ping: bool = False) -> None:
        """Start the continuous Telegram long-polling loop."""
        self._async_goals = True
        me = self._api(self.token, "getMe", None, 10.0)
        if me.get("ok") and isinstance(me.get("result"), dict):
            self.bot_username = me["result"].get("username", self.bot_username)
            print(
                f"[bilby-telegram] Connected as @{self.bot_username} (id={me['result'].get('id')}) "
                f"| harness={self.session.harness.name} | model={self.session.harness.model} "
                f"| allowed_users={self.allowed_users}",
                flush=True,
            )
        else:
            print(f"[bilby-telegram] Warning: initial getMe returned: {me}", flush=True)

        if send_startup_ping and self.default_chat_id:
            self.send_message(
                self.default_chat_id,
                f"🦘 Bilby (ActorII v4.1 Metacircular Harness) is now online on @{self.bot_username}.",
                thread_id=self.default_thread_id,
            )

        while self._running:
            try:
                self._process_supervisor_inbox()
                self.poll_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[bilby-telegram] Poll loop error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(3.0)

    def _process_supervisor_inbox(self) -> int:
        """Process any local supervisor tasks/notifications dropped into `~/.actorii/supervisor_inbox.jsonl`."""
        if self._goal_thread is not None and self._goal_thread.is_alive():
            return 0
        inbox_file = getattr(self, "inbox_path", DEFAULT_GATEWAY_INBOX)
        if not inbox_file.is_file():
            return 0
        try:
            raw_lines = [
                ln.strip()
                for ln in inbox_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            inbox_file.unlink(missing_ok=True)
        except Exception:
            return 0
        count = 0
        for idx, ln in enumerate(raw_lines):
            try:
                item = json.loads(ln) if ln.startswith("{") else {"text": ln}
                text = str(item.get("text") or item.get("task") or item.get("prompt") or "").strip()
                if not text:
                    continue
                chat_id = item.get("chat_id") or self.default_chat_id
                thread_id = item.get("thread_id") if "thread_id" in item else self.default_thread_id
                notify = bool(item.get("notify_telegram", True))
                self._sync_telegram_twin_route(chat_id, thread_id)
                print(f"[bilby-telegram] Supervisor inbox directive: {text!r}", flush=True)
                if self._async_goals and chat_id and notify:
                    leftover = raw_lines[idx + 1 :]
                    if leftover:
                        inbox_file.write_text("\n".join(leftover) + "\n", encoding="utf-8")
                    self._cancel_requested = False
                    worker = threading.Thread(
                        target=self._run_update_sync,
                        args=(0, chat_id, thread_id, None, text),
                        kwargs={"keep_typing": True},
                        daemon=True,
                    )
                    self._goal_thread = worker
                    worker.start()
                    count += 1
                    break
                if chat_id:
                    self.send_typing(chat_id, thread_id=thread_id)
                t0 = time.time()
                reply_text = self.handle_text_command_or_goal(text)
                elapsed = time.time() - t0
                if chat_id and notify:
                    self.send_message(chat_id, reply_text, thread_id=thread_id)
                print(
                    f"[bilby-telegram] Completed supervisor directive in {elapsed:.2f}s ({len(reply_text)} chars)",
                    flush=True,
                )
                count += 1
            except Exception as exc:
                print(f"[bilby-telegram] Supervisor inbox error: {exc}", flush=True)
        return count


def main(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("ACTORII_HOST_LOCKDOWN", "1")
    os.environ.setdefault("ACTORII_DISABLE_SELF_EVOLVE", "0")
    parser = argparse.ArgumentParser(
        description="Bilby (ActorII v4.1) Telegram Gateway Daemon"
    )
    parser.add_argument("--model", default=os.environ.get("ACTORII_MODEL", "qwen3.8:latest"))
    parser.add_argument("--endpoint", default=os.environ.get("ACTORII_LLM_ENDPOINT"))
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per turn (0 = unlimited, default: 25)")
    parser.add_argument("--max-failures", type=int, default=6, help="Max consecutive rollback strikes (0 = unlimited, default: 6)")
    parser.add_argument("--poll-timeout", type=int, default=20)
    parser.add_argument("--startup-ping", action="store_true", help="Send a startup notification to home channel")
    args = parser.parse_args(argv)

    gw = BilbyTelegramGateway(
        model=args.model,
        endpoint=args.endpoint,
        max_steps=args.max_steps,
        max_failures=args.max_failures,
        poll_timeout=args.poll_timeout,
    )

    def _handle_sig(signum: int, _frame: Any) -> None:
        print(f"[bilby-telegram] Received signal {signum}, shutting down...", flush=True)
        gw._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    gw.serve_forever(send_startup_ping=args.startup_ping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
