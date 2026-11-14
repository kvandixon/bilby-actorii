"""Dual-Channel Telemetry, Pytest State Isolation, and Mobile HTML Formatting for ActorII & Bilby.

Separates ActorII output into two clean channels:
1. **Channel 1 (Human Interactive Stream)**: Crisp, readable natural-language output for Telegram
   (with safe Markdown -> Telegram HTML conversion and mobile bullet-card rendering for tables),
   TUI (single-print replies + compact 1-line step status), and Browser Studio.
2. **Channel 2 (Structured Agent & Verbose Trace Logs)**:
   - `~/.actorii/logs/agent_trace.jsonl`: Structured JSONL telemetry of every turn, LLM call,
     token usage, KV latency, Actor 4.1 chunk, VM result, and error for agent/supervisor inspection.
   - `~/.actorii/logs/verbose_steps.log`: Human-tailable developer trace (`tail -f`).

Also enforces automatic state isolation under `pytest` (`PYTEST_CURRENT_TEST`) or custom
`ACTORII_STATE_DIR` so unit tests never pollute live `~/.actorii/` files.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_LOG_LOCK = threading.Lock()
_MAX_AGENT_TRACE_BYTES = 25 * 1024 * 1024  # 25 MB rotation cap
_MAX_VERBOSE_LOG_BYTES = 15 * 1024 * 1024  # 15 MB rotation cap


def is_test_environment() -> bool:
    """Return True when running inside pytest unless live state was explicitly forced."""
    if os.environ.get("ACTORII_FORCE_LIVE_STATE") == "1":
        return False
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


def resolve_actorii_state_dir() -> Path:
    """Return the active `.actorii` state directory, isolating pytest runs automatically."""
    explicit = os.environ.get("ACTORII_STATE_DIR", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if is_test_environment():
        uid = getattr(os, "getuid", lambda: 0)()
        p = Path(tempfile.gettempdir()) / f"actorii-pytest-{uid}-{os.getpid()}"
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path.home() / ".actorii"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rotate_if_needed(path: Path, max_bytes: int) -> None:
    try:
        if path.is_file() and path.stat().st_size > max_bytes:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
    except Exception:
        pass


class DualChannelLogger:
    """Thread-safe dual writer for `agent_trace.jsonl` (machine/agent) and `verbose_steps.log` (tail -f)."""

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self._explicit_dir = state_dir

    @property
    def logs_dir(self) -> Path:
        base = self._explicit_dir if self._explicit_dir is not None else resolve_actorii_state_dir()
        d = base / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def agent_trace_path(self) -> Path:
        return self.logs_dir / "agent_trace.jsonl"

    @property
    def verbose_log_path(self) -> Path:
        return self.logs_dir / "verbose_steps.log"

    def new_turn_id(self) -> str:
        return f"turn-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

    def write_event(self, event_type: str, payload: Dict[str, Any], verbose_block: Optional[str] = None) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        record = {"ts": now_iso, "event": event_type, **payload}
        try:
            with _LOG_LOCK:
                trace_p = self.agent_trace_path
                _rotate_if_needed(trace_p, _MAX_AGENT_TRACE_BYTES)
                with trace_p.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

                if verbose_block:
                    verb_p = self.verbose_log_path
                    _rotate_if_needed(verb_p, _MAX_VERBOSE_LOG_BYTES)
                    with verb_p.open("a", encoding="utf-8") as vfh:
                        vfh.write(f"[{now_iso}] {verbose_block.rstrip()}\n")
        except Exception:
            pass

    def log_turn_start(
        self,
        *,
        turn_id: str,
        surface: str,
        goal: str,
        model: str,
        max_steps: int,
    ) -> None:
        lockdown = os.environ.get("ACTORII_HOST_LOCKDOWN") == "1"
        block = (
            f"━━━ TURN START ({turn_id}) surface={surface} model={model} max_steps={max_steps or '∞'} lockdown={lockdown} ━━━\n"
            f"  Goal: {goal}"
        )
        self.write_event(
            "turn_start",
            {
                "turn_id": turn_id,
                "surface": surface,
                "goal": goal,
                "model": model,
                "max_steps": max_steps,
                "host_lockdown": lockdown,
            },
            verbose_block=block,
        )

    def log_llm_call(
        self,
        *,
        turn_id: str,
        step: int,
        endpoint: str,
        model: str,
        message_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        raw_reply: str,
        extracted_action: str,
    ) -> None:
        preview = (extracted_action or raw_reply or "").strip().replace("\n", " ↵ ")
        if len(preview) > 260:
            preview = preview[:257] + "..."
        block = (
            f"  [LLM step={step}] model={model} msgs={message_count} "
            f"prompt_tok={prompt_tokens} comp_tok={completion_tokens} latency={latency_ms:.0f}ms\n"
            f"    -> {preview}"
        )
        self.write_event(
            "llm_call",
            {
                "turn_id": turn_id,
                "step": step,
                "endpoint": endpoint,
                "model": model,
                "message_count": message_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency_ms, 1),
                "raw_reply": raw_reply[:6000],
                "extracted_action": extracted_action[:6000],
            },
            verbose_block=block,
        )

    def log_step_exec(self, *, turn_id: str, entry: Dict[str, Any]) -> None:
        step = entry.get("step", "?")
        status = entry.get("status", "ok")
        gas = entry.get("gas_used", 0)
        tier = entry.get("tier") or entry.get("risk_tier", "")
        act = str(entry.get("action") or "").strip()
        res = str(entry.get("result") if entry.get("result") is not None else "")
        err = str(entry.get("error") or "")
        lines = [f"  [VM step={step} status={status} tier={tier} gas={gas} ops]"]
        if act:
            lines.append(f"    Action: {act}")
        if err:
            lines.append(f"    Error : {err[:1500]}")
        elif res:
            lines.append(f"    Result: {res[:1500]}")
        self.write_event(
            "step_exec",
            {
                "turn_id": turn_id,
                "step": step,
                "status": status,
                "risk_tier": tier,
                "gas_used": gas,
                "failures": entry.get("failures", 0),
                "action": act[:8000],
                "result": res[:8000],
                "error": err[:4000],
            },
            verbose_block="\n".join(lines),
        )

    def log_turn_end(
        self,
        *,
        turn_id: str,
        surface: str,
        status: str,
        steps: int,
        gas_used: int,
        elapsed_s: float,
        reply: str,
        actions: List[str],
    ) -> None:
        reply_one = (reply or "").strip().replace("\n", " ↵ ")
        if len(reply_one) > 260:
            reply_one = reply_one[:257] + "..."
        block = (
            f"━━━ TURN END ({turn_id}) status={status} steps={steps} gas={gas_used} ops elapsed={elapsed_s:.2f}s ━━━\n"
            f"  Reply: {reply_one}"
        )
        self.write_event(
            "turn_end",
            {
                "turn_id": turn_id,
                "surface": surface,
                "status": status,
                "steps": steps,
                "gas_used": gas_used,
                "elapsed_s": round(elapsed_s, 3),
                "action_count": len(actions),
                "actions": [a[:4000] for a in actions[:40]],
                "reply": (reply or "")[:8000],
            },
            verbose_block=block,
        )


GLOBAL_TELEMETRY = DualChannelLogger()


def summarize_actor_action_brief(action_src: str, max_len: int = 72) -> str:
    """Return a clean 1-line human summary of an Actor 4.1 expression (e.g. `temperatureF(Actor[#Thermometer])`)."""
    s = (action_src or "").strip()
    if not s:
        return "complete"
    if s.upper().startswith("COMPLETE:"):
        return "finalizing reply"
    first_line = next((ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("/*")), s)
    if first_line.endswith("!!"):
        first_line = first_line[:-2].strip()
    first_line = re.sub(r"\s+", " ", first_line)
    if len(first_line) > max_len:
        first_line = first_line[: max_len - 3] + "..."
    return first_line


def format_last_turn_trace(
    last_turn: Optional[Dict[str, Any]],
    step_log: Optional[List[Dict[str, Any]]] = None,
    *,
    max_chars: int = 3600,
) -> str:
    """Format the most recent turn's Actor 4.1 execution trace on demand for `/last`."""
    if not last_turn and not step_log:
        return "No execution trace recorded yet in this session."
    lines: List[str] = []
    if last_turn:
        user_q = str(last_turn.get("user") or "").strip()
        steps = int(last_turn.get("steps", 1) or 1)
        gas = int(last_turn.get("gas_used", 0) or 0)
        lines.append(f"🔎 **Last Turn Execution Trace** ({steps} step(s) · {gas} ops)")
        if user_q:
            lines.append(f"**Prompt**: {user_q[:240]}")

    if step_log:
        lines.append("")
        for entry in step_log:
            if not isinstance(entry, dict):
                continue
            st_num = entry.get("step", "?")
            st_status = entry.get("status", "ok")
            st_gas = entry.get("gas_used", 0)
            act = str(entry.get("action") or "").strip()
            err = str(entry.get("error") or "").strip()
            res = str(entry.get("result") if entry.get("result") is not None else "").strip()
            badge = "✔" if st_status in ("ok", "complete") else "✖"
            lines.append(f"{badge} **Step {st_num}** (`{st_status}`, `{st_gas} ops`)")
            if act and not act.upper().startswith("COMPLETE:"):
                act_clip = act if len(act) <= 500 else (act[:497] + "...")
                lines.append(f"```actor\n{act_clip}\n```")
            if err:
                lines.append(f"Error: `{err[:320]}`")
            elif res and st_status != "complete":
                res_clip = res.replace("\n", " ")
                if len(res_clip) > 260:
                    res_clip = res_clip[:257] + "..."
                lines.append(f"Result: `{res_clip}`")
    elif last_turn and last_turn.get("actions"):
        lines.append("")
        for idx, act in enumerate(last_turn.get("actions") or [], 1):
            act_s = str(act).strip()
            if len(act_s) > 500:
                act_s = act_s[:497] + "..."
            lines.append(f"**Action {idx}**:\n```actor\n{act_s}\n```")

    lines.append(f"\nFull trace file: `{GLOBAL_TELEMETRY.agent_trace_path}`")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 24].rstrip() + "\n...(truncated)"
    return out


# ── Mobile-First Markdown -> Telegram HTML Converter ─────────────────────────

_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")


def _parse_pipe_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def convert_markdown_tables_for_mobile(md_text: str) -> str:
    """Convert wide Markdown pipe tables into compact mobile-friendly bullet cards outside code fences."""
    if "|" not in md_text:
        return md_text

    # Split out fenced code blocks so tables inside ```...``` are untouched
    parts = re.split(r"(```[\s\S]*?```)", md_text)
    out_parts: List[str] = []
    for idx, chunk in enumerate(parts):
        if idx % 2 == 1:
            out_parts.append(chunk)
            continue
        lines = chunk.splitlines()
        new_lines: List[str] = []
        i = 0
        while i < len(lines):
            if (
                i + 1 < len(lines)
                and "|" in lines[i]
                and _TABLE_SEP_RE.match(lines[i + 1])
            ):
                headers = _parse_pipe_row(lines[i])
                i += 2
                rows: List[List[str]] = []
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    if not _TABLE_SEP_RE.match(lines[i]):
                        rows.append(_parse_pipe_row(lines[i]))
                    i += 1
                if headers and rows:
                    for r in rows:
                        if not any(r):
                            continue
                        raw_lead = (r[0] if len(r) > 0 else "").strip()
                        lead = re.sub(r"^\*\*(.+?)\*\*$", r"\1", raw_lead)
                        if len(headers) == 2 and len(r) >= 2:
                            new_lines.append(f"• **{lead}**: {r[1]}")
                        else:
                            kv_pairs = []
                            for col_idx in range(1, min(len(headers), len(r))):
                                h_name = re.sub(r"^\*\*(.+?)\*\*$", r"\1", headers[col_idx].strip())
                                c_val = r[col_idx]
                                if c_val:
                                    kv_pairs.append(f"{h_name}: {c_val}" if h_name else c_val)
                            suffix = " · ".join(kv_pairs)
                            new_lines.append(f"• **{lead}**" + (f" — {suffix}" if suffix else ""))
                    continue
            new_lines.append(lines[i])
            i += 1
        joined = "\n".join(new_lines)
        if chunk.endswith("\n") and not joined.endswith("\n"):
            joined += "\n"
        out_parts.append(joined)
    return "".join(out_parts)


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard GitHub/LLM Markdown into safe, balanced Telegram `parse_mode='HTML'`."""
    if not text:
        return ""

    # 1. Convert wide pipe tables into mobile-readable bullet cards first
    working = convert_markdown_tables_for_mobile(text)

    placeholders: Dict[str, str] = {}
    counter = 0

    def _stash(html_fragment: str) -> str:
        nonlocal counter
        counter += 1
        token = f"\x00TGHTMLPH{counter}\x00"
        placeholders[token] = html_fragment
        return token

    # 2. Stash fenced code blocks (` ```lang\ncode\n``` `)
    def _repl_fence(m: re.Match) -> str:
        code_body = m.group(2) or ""
        if code_body.startswith("\n"):
            code_body = code_body[1:]
        code_body = code_body.rstrip("\n")
        return _stash(f"<pre><code>{html.escape(code_body)}</code></pre>")

    working = re.sub(r"```([A-Za-z0-9_+-]*)[ \t]*\n?([\s\S]*?)```", _repl_fence, working)

    # 3. Stash inline code spans (`` `code` ``)
    def _repl_inline_code(m: re.Match) -> str:
        return _stash(f"<code>{html.escape(m.group(1))}</code>")

    working = re.sub(r"`([^`\n]+)`", _repl_inline_code, working)

    # 4. Stash markdown links `[label](https://...)`
    def _repl_link(m: re.Match) -> str:
        label = html.escape(m.group(1).strip())
        url = html.escape(m.group(2).strip(), quote=True)
        return _stash(f'<a href="{url}">{label}</a>')

    working = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", _repl_link, working)

    # 5. Escape HTML entities in all remaining prose
    working = html.escape(working)

    # 6. Headings (`# Title` .. `###### Title`) -> bold line
    working = re.sub(r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*$", r"<b>\1</b>", working)

    # 7. Horizontal rules (`---` or `***`) -> subtle divider
    working = re.sub(r"(?m)^[ \t]*(?:-\s*){3,}$", "──────────", working)

    # 8. Unordered list bullets (`- item` or `* item`) -> clean `• item`
    working = re.sub(r"(?m)^([ \t]*)[-*][ \t]+", r"\1• ", working)

    # 9. Bold (`**text**` or `__text__`), Strikethrough (`~~text~~`), Italic (`*text*` or `_text_`)
    working = re.sub(r"\*\*([^*\n]+(?:\*(?!\*)[^*\n]+)*)\*\*", r"<b>\1</b>", working)
    working = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", working)
    working = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", working)
    working = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", working)
    working = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", working)

    # 10. Restore stashed `<pre>`, `<code>`, and `<a>` blocks
    for token, fragment in placeholders.items():
        working = working.replace(html.escape(token), fragment).replace(token, fragment)

    return working.strip()


def split_markdown_fence_aware(text: str, max_len: int = 3500) -> List[str]:
    """Split Markdown text into chunks <= `max_len` without slicing open ``` code fences in half."""
    raw = (text or "").strip()
    if not raw:
        return [""]
    if len(raw) <= max_len:
        return [raw]

    chunks: List[str] = []
    lines = raw.splitlines(keepends=True)
    buf: List[str] = []
    buf_len = 0
    in_fence = False
    fence_header = "```"

    for ln in lines:
        stripped = ln.lstrip()
        toggles_fence = stripped.startswith("```")
        if buf_len + len(ln) > max_len and buf:
            chunk_str = "".join(buf)
            if in_fence:
                chunk_str = chunk_str.rstrip("\n") + "\n```"
            chunks.append(chunk_str.strip())
            buf = [f"{fence_header}\n"] if in_fence else []
            buf_len = len(buf[0]) if buf else 0

        if len(ln) > max_len:
            # Hard slice ultra-long single line safely
            pos = 0
            while pos < len(ln):
                piece = ln[pos : pos + max_len - 20]
                chunks.append(piece.strip())
                pos += max_len - 20
            continue

        buf.append(ln)
        buf_len += len(ln)
        if toggles_fence:
            if not in_fence:
                in_fence = True
                fence_header = stripped.rstrip("\r\n")
            else:
                in_fence = False
                fence_header = "```"

    if buf:
        chunk_str = "".join(buf).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks or [raw[:max_len]]


def render_telegram_chunks(markdown_text: str, max_md_chunk: int = 3400) -> List[Tuple[str, str]]:
    """Return `[(html_chunk, plain_fallback_chunk), ...]` with balanced HTML tags per Telegram message."""
    md_chunks = split_markdown_fence_aware(markdown_text, max_len=max_md_chunk)
    out: List[Tuple[str, str]] = []
    for md_part in md_chunks:
        html_part = markdown_to_telegram_html(md_part)
        # Plain fallback converts tables cleanly too, without raw HTML entities
        plain_part = convert_markdown_tables_for_mobile(md_part)
        out.append((html_part, plain_part))
    return out
