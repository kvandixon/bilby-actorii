"""Local CPU-Only Coding Assistant Bridge (`Actor[#Coder]`, 0 MB GPU VRAM).

Gives Bilby an independent, zero-VRAM local second-opinion coding & debugging
partner so he can review code, diagnose subtle bugs, and draft patches locally
without relying on Cloud Skippy or evicting `qwen3.8:latest` from the RTX 5070 Ti.

Backends (strictly CPU RAM, 0 MB GPU VRAM):
  1. Primary  : `http://127.0.0.1:8083/v1/chat/completions`
                (`Nemotron-Cascade-2-30B-A3B.i1-IQ4_XS`, 3B active MoE via `bilby-coder-8083.service`)
  2. Fallback : `http://127.0.0.1:11434/v1/chat/completions`
                (`qwen2.5-coder-cpu:7b` via Ollama CPU)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from actorii.bridge import actor_bridge, actor_query
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol

DEFAULT_CODER_PRIMARY_URL = os.environ.get(
    "ACTORII_CODER_URL",
    "http://127.0.0.1:8083/v1/chat/completions",
)
DEFAULT_CODER_PRIMARY_MODEL = os.environ.get(
    "ACTORII_CODER_MODEL",
    "nemotron-cascade-2-30b-coder",
)
DEFAULT_CODER_FALLBACK_URL = os.environ.get(
    "ACTORII_CODER_FALLBACK_URL",
    "http://127.0.0.1:11434/v1/chat/completions",
)
DEFAULT_CODER_FALLBACK_MODEL = os.environ.get(
    "ACTORII_CODER_FALLBACK_MODEL",
    "qwen2.5-coder-cpu:7b",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def _to_actor_dict(d: Dict[str, Any]) -> Dictionary:
    out = Dictionary()
    for k, v in d.items():
        out.atPut(Symbol(str(k)), v)
    return out


def _strip_think(text: str) -> Tuple[str, str]:
    """Separate `<think>...</think>` reasoning from the final answer."""
    raw = str(text or "")
    m = _THINK_BLOCK_RE.search(raw)
    if not m:
        return raw.strip(), ""
    think_txt = m.group(0).strip()
    cleaned = _THINK_BLOCK_RE.sub("", raw).strip()
    return cleaned or raw.strip(), think_txt[:600]


@actor_bridge(name="LocalCoderBridge")
class LocalCoderBridge:
    """Zero-VRAM CPU coding assistant exposed to Bilby as `Actor[#Coder]`."""

    def __init__(
        self,
        primary_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.primary_url = str(primary_url or os.environ.get("ACTORII_CODER_URL") or DEFAULT_CODER_PRIMARY_URL)
        self.primary_model = str(primary_model or os.environ.get("ACTORII_CODER_MODEL") or DEFAULT_CODER_PRIMARY_MODEL)
        self.fallback_url = str(fallback_url or os.environ.get("ACTORII_CODER_FALLBACK_URL") or DEFAULT_CODER_FALLBACK_URL)
        self.fallback_model = str(fallback_model or os.environ.get("ACTORII_CODER_FALLBACK_MODEL") or DEFAULT_CODER_FALLBACK_MODEL)
        self.timeout_s = max(5.0, float(timeout_s or 120.0))

    def _post_chat(
        self,
        url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        deep_think: bool = False,
        max_tokens: int = 1600,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": 0.4 if deep_think else 0.15,
            "stream": False,
        }
        if ":8083" in url or "nemotron" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": bool(deep_think)}
            if not deep_think:
                payload["reasoning_effort"] = "low"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        choices = body.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        content = str(msg.get("content") or "").strip()
        reasoning = str(msg.get("reasoning_content") or "").strip()
        answer, inline_think = _strip_think(content)
        if not answer and reasoning:
            answer = reasoning
        usage = body.get("usage") or {}
        c_tok = int(usage.get("completion_tokens") or 0)
        tok_s = round(c_tok / max(0.001, elapsed_ms / 1000.0), 1) if c_tok else 0.0
        return {
            "ok": True,
            "reply": answer,
            "reasoningExcerpt": (reasoning or inline_think)[:400],
            "backend": url,
            "model": model,
            "deepThink": bool(deep_think),
            "promptTokens": int(usage.get("prompt_tokens") or 0),
            "completionTokens": c_tok,
            "tokensPerSec": tok_s,
            "elapsedMs": elapsed_ms,
            "error": "",
        }

    def _call_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        deep_think: bool = False,
        max_tokens: int = 1600,
    ) -> Dictionary:
        errors: List[str] = []
        attempts = [
            (self.primary_url, self.primary_model, deep_think),
            (self.fallback_url, self.fallback_model, False),
        ]
        for url, model, use_think in attempts:
            if not url:
                continue
            try:
                res = self._post_chat(
                    url,
                    model,
                    system_prompt,
                    user_prompt,
                    deep_think=use_think,
                    max_tokens=max_tokens,
                )
                return _to_actor_dict(res)
            except Exception as exc:
                errors.append(f"{model}@{url}: {exc}")
        return _to_actor_dict({
            "ok": False,
            "reply": "",
            "reasoningExcerpt": "",
            "backend": "offline",
            "model": "",
            "deepThink": bool(deep_think),
            "promptTokens": 0,
            "completionTokens": 0,
            "tokensPerSec": 0.0,
            "elapsedMs": 0.0,
            "error": "All local CPU coder backends unreachable: " + " | ".join(errors),
        })

    @actor_query
    def reviewCode(self, code: str, issue: str = "", deep_think: bool = False) -> Dictionary:
        """Review code for bugs, edge cases, syntax errors, and concrete fixes (0 MB VRAM)."""
        sys_p = (
            "You are a concise, rigorous senior software engineer acting as a local CPU second-opinion code reviewer "
            "for Bilby. Identify real bugs, null/edge-case crashes, syntax issues, and concrete line-level fixes. "
            "Be direct and technical; no fluff."
        )
        code_trim = str(code or "")[:12000]
        issue_s = str(issue or "").strip()
        user_p = (
            (f"Focus / Reported Issue: {issue_s}\n\n" if issue_s else "")
            + f"Code to review:\n```\n{code_trim}\n```\n\n"
            "List concrete bugs (with exact line/symbol) and exact minimal fixes."
        )
        return self._call_with_fallback(sys_p, user_p, deep_think=bool(deep_think), max_tokens=1600)

    @actor_query
    def diagnoseBug(self, code: str, error_or_symptom: str, deep_think: bool = True) -> Dictionary:
        """Deep root-cause debugging for a failing script or runtime symptom (uses Cascade RL thinking by default)."""
        sys_p = (
            "You are an expert debugging specialist (0-VRAM local CPU partner for Bilby). "
            "Trace the exact execution path that causes the bug, identify the root cause, and give the minimal surgical fix."
        )
        code_trim = str(code or "")[:12000]
        err_s = str(error_or_symptom or "")[:2000]
        user_p = (
            f"Error / Symptom:\n{err_s}\n\n"
            f"Relevant Code:\n```\n{code_trim}\n```\n\n"
            "1) Exact root cause, 2) Why it triggers, 3) Exact minimal fix."
        )
        return self._call_with_fallback(sys_p, user_p, deep_think=bool(deep_think), max_tokens=2048)

    @actor_query
    def suggestPatch(self, code: str, goal: str, deep_think: bool = False) -> Dictionary:
        """Draft clean, formatted code or a surgical patch for a concrete coding goal (0 MB VRAM)."""
        sys_p = (
            "You are a concise local CPU coding specialist for Bilby. Produce clean, well-formatted, multi-line code "
            "or a minimal surgical patch that directly satisfies the goal with zero syntax errors."
        )
        code_trim = str(code or "")[:12000]
        goal_s = str(goal or "")[:2000]
        user_p = (
            f"Goal:\n{goal_s}\n\n"
            + (f"Existing Code:\n```\n{code_trim}\n```\n\n" if code_trim.strip() else "")
            + "Return the clean, ready-to-write implementation or surgical diff."
        )
        return self._call_with_fallback(sys_p, user_p, deep_think=bool(deep_think), max_tokens=2400)

    @actor_query
    def askCoder(self, prompt: str, deep_think: bool = False) -> Dictionary:
        """General local CPU coding/architecture second opinion (offline alternative to Cloud Skippy)."""
        sys_p = (
            "You are Bilby's local CPU coding assistant (Nemotron-Cascade-2-30B-A3B, 0 MB GPU VRAM). "
            "Give direct, accurate, verifiable engineering advice and code."
        )
        return self._call_with_fallback(sys_p, str(prompt or "")[:12000], deep_think=bool(deep_think), max_tokens=1800)

    @actor_query
    def health(self) -> Dictionary:
        """Check live availability of the primary (:8083) and fallback (:11434) CPU coder backends."""
        def _probe(base_chat_url: str) -> str:
            try:
                root = base_chat_url.split("/v1/")[0]
                health_url = root + ("/health" if ":8083" in root else "/api/tags")
                with urllib.request.urlopen(health_url, timeout=2.0) as r:
                    return f"online (HTTP {r.status})"
            except Exception as exc:
                return f"offline ({exc.__class__.__name__})"

        return _to_actor_dict({
            "primaryModel": self.primary_model,
            "primaryUrl": self.primary_url,
            "primaryStatus": _probe(self.primary_url),
            "fallbackModel": self.fallback_model,
            "fallbackUrl": self.fallback_url,
            "fallbackStatus": _probe(self.fallback_url),
            "vramMb": 0,
        })

    @actor_query
    def describe(self) -> str:
        """Self-documentation for Bilby's lazy twin discovery."""
        return (
            "`Actor[#Coder]` (LocalCoderBridge — 0 MB GPU VRAM CPU Coding Assistant, "
            f"primary={self.primary_model} on :8083, fallback={self.fallback_model} on :11434):\n"
            "  - `reviewCode(Actor[#Coder], codeStr, issueStr, deepThinkBool)!!` -> second-opinion bug/syntax review\n"
            "  - `diagnoseBug(Actor[#Coder], codeStr, errorOrSymptom, deepThinkBool)!!` -> deep root-cause trace (Cascade RL thinking)\n"
            "  - `suggestPatch(Actor[#Coder], codeStr, goalStr, deepThinkBool)!!` -> clean multi-line code or surgical patch\n"
            "  - `askCoder(Actor[#Coder], promptStr, deepThinkBool)!!` -> general local coding consultation (no cloud needed)\n"
            "  - `health(Actor[#Coder])!!` -> check :8083 & :11434 backend status\n"
            "Use `Actor[#Coder]` FIRST for coding/debugging second opinions before escalating to `askSkippy`!"
        )
