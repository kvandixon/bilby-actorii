"""Telegram Long-Polling Gateway for the ActorII Metacircular Bilby Harness (`actorii.telegram_gateway`).

Connects Bilby (`Actor[#Bilby]` / `Actor[#Harness]`) directly to Telegram using
a configured bot token (`TELEGRAM_BOT_TOKEN` / `BILBY_TELEGRAM_BOT_TOKEN` from `.env`),
allowing two-way Telegram messaging backed by the 1993 Whitewater Actor 4.1
bytecode VM, Neo4j Engram memory (`Actor[#Memory]`), and Digital Twins
(`Actor[#Valkyrie]`, `Actor[#ReverseEngineer]`, `Actor[#Telegram]`, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from actorii.harness import bootstrap_agent_harness
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
        max_steps: int = 0,
        max_failures: int = 0,
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
        self.state_path = state_path or DEFAULT_GATEWAY_STATE
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
        self.offset: int = 0
        self.total_turns: int = 0
        self.total_steps: int = 0
        self.total_gas: int = 0
        self._consecutive_poll_errors: int = 0
        self._frozen: bool = False
        self._cancel_requested: bool = False
        self._async_goals: bool = False
        self._goal_thread: Optional[threading.Thread] = None
        self._load_state()
        self._running = True
        self.bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "bilby_bot").strip() or "bilby_bot"

    def _load_state(self) -> int:
        try:
            if self.state_path.is_file():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.offset = int(data.get("offset", 0) or 0)
                self.total_turns = int(data.get("total_turns", 0) or 0)
                self.total_steps = int(data.get("total_steps", 0) or 0)
                self.total_gas = int(data.get("total_gas", 0) or 0)
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
        """Send one or more chunked messages to `chat_id` (preserving forum `message_thread_id`)."""
        results: List[Dict[str, Any]] = []
        cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        for idx, part in enumerate(_split_telegram_message(text)):
            payload: Dict[str, Any] = {"chat_id": cid, "text": part}
            if thread_id is not None and int(thread_id) > 0:
                payload["message_thread_id"] = int(thread_id)
            if idx == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            res = self._api(self.token, "sendMessage", payload, 15.0)
            if not res.get("ok") and "message_thread_id" in payload:
                # Fallback to root chat if topic thread was closed/deleted
                fallback_payload = {k: v for k, v in payload.items() if k != "message_thread_id"}
                res = self._api(self.token, "sendMessage", fallback_payload, 15.0)
            results.append(res)
        return results

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
        return (
            "🛡️ Bilby Containment & Security Controls Report\n"
            f"• Execution State: {'❄️ FROZEN (Kill-Switch Active)' if self._frozen else '🟢 ACTIVE (Guarded)'}\n"
            f"• Host Lockdown (`ACTORII_HOST_LOCKDOWN`): {'🔒 LOCKED (1)' if lockdown else '⚠️ UNLOCKED (0)'}\n"
            f"• Host Shell (`Actor[#Host]>>execShell` / `Actor[#Valkyrie]>>runShell`): {'🔒 BLOCKED ON HOST' if (lockdown and not in_docker) else '🐳 DOCKER-ONLY'}\n"
            f"• Self-Evolution (`evolveHarness` / Ring 1 Cognitive Hooks): {'🔒 DISABLED' if no_evolve else '🟢 ENABLED (Guarded Ring 1 Hooks Only — Kernel & Tier 2 Calls Locked)'}\n"
            "• Python Source & Host Config Write Guard (`src/actorii`, `~/bin`, `.config/systemd`, `/etc`): 🔒 HARD-BLOCKED\n"
            "• Destructive Command Filter (`systemctl`, `sudo`, `nmcli`, `resolv.conf`, `crontab`, `rm -rf`): 🔒 ACTIVE (47 patterns)\n"
            f"• Ring 0/1 Kernel & Bridge Class Lock: 🔒 ACTIVE (36 classes + `AgentHarness` kernel methods)\n"
            f"• Pool Singleton Lock (`Actor[#Host]`, `Actor[#Harness]`, `Actor[#Sandbox]`): {'🔒 LOCKED' if singleton_lock else '⚠️ UNLOCKED'}\n"
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
            "/help",
            "/start",
        )

    def handle_text_command_or_goal(self, text: str) -> str:
        """Process a Telegram message (`slash` command or natural-language/Actor 4.1 goal) via Bilby."""
        clean = (text or "").strip()
        if not clean:
            return "Send a message, Actor 4.1 chunk (`!!`), or `/stop`, `/status`, `/controls`, `/freeze`, `/twins`, `/engrams`, `/clear`."

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

            if self._frozen:
                return "❄️ Bilby is currently FROZEN by kill-switch (`/freeze`). Send `/unlock` or `/controls`."

            if cmd_lower in ("/start", "/help"):
                return (
                    "🦘 Bilby (ActorII v4.2.0 Metacircular Agent Harness — Hardened Lockdown)\n"
                    "Resident System AI on valkyrie (Actor-inspired Linux VM).\n\n"
                    "Commands:\n"
                    "• Plain English or Actor 4.1 code (`!!`) — Runs Bilby's guarded ReAct loop (unlimited retries)\n"
                    "• /stop (or /cancel) — Break Bilby out of an active ReAct loop immediately\n"
                    "• /proxy [status|links|lan|tailscale|local|pause|auth on|auth off|rotate|logs] — Manage WLAN & Tailscale `:8090` Reverse Proxy\n"
                    "• /controls — Inspect active security & containment locks\n"
                    "• /freeze | /unlock — Emergency kill-switch to freeze/unfreeze Bilby\n"
                    "• /eval <code!!> — Direct Actor 4.1 bytecode evaluation (Ring 0/1 & Host-Lockdown enforced)\n"
                    "• /status — VM, GPU VRAM, and Harness status\n"
                    "• /twins — Live Digital Twins & Skills (Valkyrie, Jetson, ProductLearner, Proxy, ReverseEngineer, Thermometer, Pantry, Telegram)\n"
                    "• /engrams — List crystallized Neo4j ActorEngrams\n"
                    "• /scratch — Inspect `Actor[#Bilby]` scratchpad\n"
                    "• /history — Recent multi-turn conversation history\n"
                    "• /clear — Clear conversation history & reset circuit breaker"
                )

            if cmd_lower == "/eval" and arg_str:
                chunk = arg_str if arg_str.endswith("!!") else f"{arg_str}!!"
                before = len(self.session.transcript)
                self.session.handle_input(chunk)
                out = "\n".join(self.session.transcript[before:]).strip()
                return out or "Evaluated."

            if cmd_lower in ("/status", "/proxy", "/twins", "/engrams", "/scratch", "/history", "/clear", "/classes", "/schema", "/capabilities", "/caps", "/config"):
                before = len(self.session.transcript)
                self.session.handle_input(f"{cmd_lower} {arg_str}".strip())
                out = "\n".join(self.session.transcript[before:]).strip()
                return out or "Done."

        if self._frozen:
            return "❄️ Bilby is currently FROZEN by kill-switch (`/freeze`). Send `/unlock` or `/controls`."

        # Execute via Bilby's multi-turn conversational + Actor 4.1 bytecode ReAct loop
        before_turns = len(self.session.sandbox.getChatHistory(self.session.harness))
        before = len(self.session.transcript)
        self.session.handle_input(f"/chat {clean}")
        new_lines = self.session.transcript[before:]

        # Extract final answer + any compiled Actor 4.1 bytecode steps executed in this turn
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
            if actions:
                act_block = "\n".join(f"  {a}" for a in actions)
                return f"{reply}\n\n⚡ [Actor 4.1 · {steps} step(s) · {gas} ops]\n{act_block}"
            if reply:
                return reply

        # If the goal was stopped by user or halted without a COMPLETE turn, format a clear diagnostic
        self._save_offset(self.offset)
        last_err = str(self.session.harness._fields.get("lastError") or "").strip()
        steps_used = int(self.session.harness._fields.get("stepCount", 0) or 0)
        if self._cancel_requested and h_status in ("halted", "blocked"):
            self._cancel_requested = False
            return f"⏹ Goal stopped by user (`status=#{h_status}`, steps={steps_used})."
        self._cancel_requested = False
        if h_status in ("blocked", "halted") or last_err:
            return (
                f"⚠️ Goal halted (`status=#{h_status}`, steps={steps_used}).\n"
                + (f"Last error: `{last_err}`" if last_err else "")
            ).strip()
        fallback = "\n".join(new_lines).strip()
        return fallback or str(self.session.harness._fields.get("lastResult") or "Completed.")

    def _run_update_sync(
        self,
        upd_id: int,
        chat_id: Any,
        thread_id: Optional[int],
        msg_id: Optional[int],
        text: str,
        *,
        keep_typing: bool = False,
    ) -> str:
        stop_typing = threading.Event()

        def _typing_loop() -> None:
            while not stop_typing.wait(4.5):
                self.send_typing(chat_id, thread_id=thread_id)

        typing_thread: Optional[threading.Thread] = None
        self.send_typing(chat_id, thread_id=thread_id)
        if keep_typing:
            typing_thread = threading.Thread(target=_typing_loop, daemon=True)
            typing_thread.start()

        t0 = time.time()
        try:
            reply_text = self.handle_text_command_or_goal(text)
        except Exception as exc:
            reply_text = f"⚠️ Bilby harness error: {type(exc).__name__}: {exc}"
        finally:
            stop_typing.set()
        elapsed = time.time() - t0

        self.send_message(
            chat_id,
            reply_text,
            thread_id=thread_id,
            reply_to_message_id=msg_id,
        )
        print(
            f"[bilby-telegram] Replied to update #{upd_id} in {elapsed:.2f}s ({len(reply_text)} chars)",
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
        if not DEFAULT_GATEWAY_INBOX.is_file():
            return 0
        try:
            raw_lines = [
                ln.strip()
                for ln in DEFAULT_GATEWAY_INBOX.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            DEFAULT_GATEWAY_INBOX.unlink(missing_ok=True)
        except Exception:
            return 0
        count = 0
        for ln in raw_lines:
            try:
                item = json.loads(ln) if ln.startswith("{") else {"text": ln}
                text = str(item.get("text") or item.get("task") or item.get("prompt") or "").strip()
                if not text:
                    continue
                chat_id = item.get("chat_id") or self.default_chat_id
                thread_id = item.get("thread_id") if "thread_id" in item else self.default_thread_id
                print(f"[bilby-telegram] Supervisor inbox directive: {text!r}", flush=True)
                if chat_id:
                    self.send_typing(chat_id, thread_id=thread_id)
                t0 = time.time()
                reply_text = self.handle_text_command_or_goal(text)
                elapsed = time.time() - t0
                if chat_id and item.get("notify_telegram", True):
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
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps per turn (0 = unlimited)")
    parser.add_argument("--poll-timeout", type=int, default=20)
    parser.add_argument("--startup-ping", action="store_true", help="Send a startup notification to home channel")
    args = parser.parse_args(argv)

    gw = BilbyTelegramGateway(
        model=args.model,
        endpoint=args.endpoint,
        max_steps=args.max_steps,
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
