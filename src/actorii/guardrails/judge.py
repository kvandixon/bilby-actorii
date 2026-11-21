"""Semantic & Invariant Judge for ActorII Mutations (`actorii.guardrails.judge`).

Provides:
1. Deterministic policy checks (linter errors, risk-tier caps, synthesized query-mutation traps).
2. `ZeroVramSystemOneJudge`: Out-of-band, 0-MB GPU VRAM judge backed by `valkyrie`'s
   CPU specialist endpoints (`http://127.0.0.1:11434/v1/systemone` with `clef-flash-cpu:latest`
   or `http://127.0.0.1:8082/v1/chat/completions` in `/no_think` mode).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from actorii.guardrails.linter import LintReport, RiskTier
from actorii.vm import CompiledMethod

SYSTEMONE_CPU_URL = os.environ.get(
    "ACTORII_SYSTEMONE_URL", "http://127.0.0.1:11434/v1/systemone"
)
SPECIALIST_CPU_8082_URL = os.environ.get(
    "ACTORII_SPECIALIST_ENDPOINT", "http://127.0.0.1:8082/v1/chat/completions"
)

_SELECTOR_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+")


def _parse_judge_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a JSON object from an LLM judge reply (strict, fail-closed)."""
    txt = str(raw_text or "").strip()
    if not txt:
        return None
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
    fence_m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    candidates: List[str] = []
    if fence_m:
        candidates.append(fence_m.group(1))
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end > start:
        candidates.append(txt[start : end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


@dataclass
class JudgeVerdict:
    """Decision record returned by `ActorCodeJudge`."""

    approved: bool
    risk_tier: RiskTier
    confidence: float = 1.0
    reasons: List[str] = field(default_factory=list)
    engine: str = "deterministic"


class ZeroVramSystemOneJudge:
    """Zero-VRAM CPU Judge using `:11434/v1/systemone` (`clef-flash-cpu:latest`) or `:8082`."""

    def __init__(
        self,
        *,
        systemone_url: str = SYSTEMONE_CPU_URL,
        specialist_url: str = SPECIALIST_CPU_8082_URL,
        model: str = "clef-flash-cpu:latest",
        min_pass_probability: float = 0.50,
        min_pass_confidence: float = 0.15,
        timeout: float = 25.0,
        prefer_fast_8082: bool = False,
    ) -> None:
        self.systemone_url = systemone_url
        self.specialist_url = specialist_url
        self.model = model
        self.min_pass_probability = min_pass_probability
        self.min_pass_confidence = min_pass_confidence
        self.timeout = timeout
        self.prefer_fast_8082 = prefer_fast_8082

    def __call__(
        self, compiled_method: CompiledMethod, lint_report: LintReport
    ) -> JudgeVerdict:
        return self.score_method(compiled_method, lint_report)

    def score_method(
        self, compiled_method: CompiledMethod, lint_report: LintReport
    ) -> JudgeVerdict:
        """Evaluate a Tier 1/2 Actor method using the zero-VRAM CPU judge."""
        # Fast-path Tier 0 read-only methods without network overhead
        if lint_report.risk_tier == RiskTier.TIER_0_READ_ONLY and lint_report.passed:
            return JudgeVerdict(
                approved=True,
                risk_tier=lint_report.risk_tier,
                confidence=1.0,
                reasons=["Tier 0 Read-Only method auto-approved by static proof."],
                engine="static_tier0",
            )

        src = getattr(compiled_method, "source", None) or f"<bytecode:{compiled_method.name}>"
        state_context = (
            f"ActorII Runtime Guardrail Evaluation for Class '{lint_report.class_name}', "
            f"Method Selector '{lint_report.selector}'.\n"
            f"Risk Tier: Tier {int(lint_report.risk_tier)} ({lint_report.risk_tier.name}).\n"
            f"Fields Read: {lint_report.read_fields}\n"
            f"Fields Written: {lint_report.written_fields}\n"
            f"Called Selectors: {lint_report.called_selectors}\n"
            f"Globals Written: {lint_report.globals_written}\n"
            f"Method Source Code:\n{src}\n"
        )

        if self.prefer_fast_8082:
            try:
                return self._score_via_8082(compiled_method, lint_report, state_context)
            except Exception as exc:
                return JudgeVerdict(
                    approved=False,
                    risk_tier=lint_report.risk_tier,
                    confidence=0.0,
                    reasons=[f"CPU Specialist (:8082) unavailable (fail-closed): {exc}"],
                    engine="offline_fail_closed",
                )

        try:
            return self._score_via_systemone(compiled_method, lint_report, state_context)
        except Exception as exc1:
            try:
                return self._score_via_8082(compiled_method, lint_report, state_context)
            except Exception as exc2:
                return JudgeVerdict(
                    approved=False,
                    risk_tier=lint_report.risk_tier,
                    confidence=0.0,
                    reasons=[
                        f"CPU judge endpoints unreachable (fail-closed): systemone={exc1}; specialist_8082={exc2}"
                    ],
                    engine="offline_fail_closed",
                )

    def _score_via_systemone(
        self,
        compiled_method: CompiledMethod,
        lint_report: LintReport,
        state_context: str,
    ) -> JudgeVerdict:
        questions = {
            "method_safety": {
                "type": "choice",
                "instructions": (
                    f"Is the Actor method '{lint_report.class_name}>>{lint_report.selector}' "
                    f"safe, bounded, and semantically consistent with its selector name?"
                ),
                "criteria": {
                    "safe_approved": (
                        "The method is safe, terminates cleanly, only mutates appropriate fields "
                        "for its selector name, and does not corrupt system state."
                    ),
                    "unsafe_rejected": (
                        "The method is unsafe, contains an infinite loop, corrupts unrelated fields, "
                        "or performs unauthorized destructive mutations."
                    ),
                },
            }
        }
        body = {
            "model": self.model,
            "state": state_context,
            "questions": questions,
        }
        req = urllib.request.Request(
            self.systemone_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        ans = data.get("answers", {}).get("method_safety", {})
        choice = str(ans.get("choice", ""))
        confidence = float(ans.get("confidence", 0.0))
        probs = ans.get("probabilities", {})
        safe_prob = float(probs.get("safe_approved", 0.0))

        approved = (
            choice == "safe_approved"
            and safe_prob >= self.min_pass_probability
            and confidence >= self.min_pass_confidence
        )
        return JudgeVerdict(
            approved=approved,
            risk_tier=lint_report.risk_tier,
            confidence=round(confidence, 4),
            reasons=[
                f"SystemOne CPU (:11434) choice={choice}, safe_prob={safe_prob:.1%}, confidence={confidence:.1%}"
            ],
            engine="systemone_11434",
        )

    def _score_via_8082(
        self,
        compiled_method: CompiledMethod,
        lint_report: LintReport,
        state_context: str,
    ) -> JudgeVerdict:
        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\nYou are the ActorII Security Judge. Respond with JSON only: "
                        '{"approved": true/false, "confidence": 0.9, "reason": "..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{state_context}\nEvaluate if this method is safe to admit into the live Actor image. "
                        "Respond with JSON only."
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        req = urllib.request.Request(
            self.specialist_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        parsed = _parse_judge_json_object(text)
        if parsed is None or not isinstance(parsed.get("approved"), bool):
            return JudgeVerdict(
                approved=False,
                risk_tier=lint_report.risk_tier,
                confidence=0.0,
                reasons=[f"CPU Specialist (:8082) malformed non-JSON reply (fail-closed): {text[:160]}"],
                engine="specialist_8082",
            )
        approved = parsed["approved"] is True
        try:
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.85))))
        except (TypeError, ValueError):
            conf = 0.85
        reason_str = str(parsed.get("reason") or text[:160]).strip()
        return JudgeVerdict(
            approved=approved,
            risk_tier=lint_report.risk_tier,
            confidence=round(conf, 4),
            reasons=[f"CPU Specialist (:8082): {reason_str[:200]}"],
            engine="specialist_8082",
        )


class ActorCodeJudge:
    """Evaluates compiled Actor methods against deterministic invariants and optional LLM/System 1 judge."""

    def __init__(
        self,
        vm: Any,
        *,
        max_allowed_tier: RiskTier = RiskTier.TIER_2_SYSTEM_OR_BRIDGE,
        custom_judge: Optional[Callable[[CompiledMethod, LintReport], JudgeVerdict]] = None,
    ) -> None:
        self.vm = vm
        self.max_allowed_tier = max_allowed_tier
        self.custom_judge = custom_judge

    def evaluate(
        self,
        compiled_method: CompiledMethod,
        lint_report: LintReport,
        *,
        is_synthesized: bool = False,
    ) -> JudgeVerdict:
        """Judge whether `compiled_method` is safe to attach and execute."""
        reasons: List[str] = []

        # 1. Hard block if static linter failed
        if not lint_report.passed:
            for err in lint_report.errors:
                reasons.append(f"[{err.code}] {err.message}")
            return JudgeVerdict(
                approved=False,
                risk_tier=lint_report.risk_tier,
                confidence=1.0,
                reasons=reasons,
            )

        # 2. Enforce maximum allowed risk tier
        if lint_report.risk_tier > self.max_allowed_tier:
            reasons.append(
                f"Method '{lint_report.class_name}>>{lint_report.selector}' classified as "
                f"Tier {int(lint_report.risk_tier)} ({lint_report.risk_tier.name}), exceeding "
                f"max allowed Tier {int(self.max_allowed_tier)} ({self.max_allowed_tier.name})."
            )
            return JudgeVerdict(
                approved=False,
                risk_tier=lint_report.risk_tier,
                confidence=1.0,
                reasons=reasons,
            )

        # 3. Synthesized method heuristic guardrails:
        #    A query-named selector (e.g. get*, is*, *Count, *Range*) synthesized by an LLM
        #    should not unexpectedly mutate fields or globals unless explicitly named set*/update*/etc.
        if is_synthesized and lint_report.risk_tier >= RiskTier.TIER_1_LOCAL_MUTATION:
            sel_tokens = [t.lower() for t in _SELECTOR_WORD_RE.findall(lint_report.selector or "")]
            first_tok = sel_tokens[0] if sel_tokens else (lint_report.selector or "").lower()
            mutator_verbs = frozenset(
                {
                    "set",
                    "update",
                    "add",
                    "remove",
                    "inc",
                    "increment",
                    "dec",
                    "decrement",
                    "deposit",
                    "withdraw",
                    "reset",
                    "clear",
                    "toggle",
                    "apply",
                    "on",
                    "attach",
                    "register",
                    "store",
                    "save",
                    "write",
                    "delete",
                    "push",
                    "pop",
                    "put",
                    "init",
                    "record",
                    "append",
                    "insert",
                    "replace",
                    "edit",
                    "patch",
                    "evolve",
                    "rollback",
                }
            )
            # Match word-boundary mutator verbs (leading verb or explicit mutator token after non-query prefix),
            # never arbitrary substrings like "on" inside "getJson"/"isConnected" or "set" inside "getOffset".
            query_leading_verbs = frozenset({"get", "is", "has", "can", "check", "read", "find", "list", "count", "calc", "compute"})
            is_mutator_name = first_tok in mutator_verbs or (
                first_tok not in query_leading_verbs
                and any(tok in mutator_verbs and tok != "on" for tok in sel_tokens)
            )
            if not is_mutator_name:
                reasons.append(
                    f"Synthesized query selector '{lint_report.selector}' unexpectedly mutates state "
                    f"(writes fields={lint_report.written_fields}, globals={lint_report.globals_written})."
                )
                return JudgeVerdict(
                    approved=False,
                    risk_tier=lint_report.risk_tier,
                    confidence=0.95,
                    reasons=reasons,
                )

        # 4. Delegate to custom / Zero-VRAM System 1 Judge if configured
        if self.custom_judge is not None:
            return self.custom_judge(compiled_method, lint_report)

        reasons.append(
            f"Approved ({lint_report.risk_tier.name}): "
            f"reads={lint_report.read_fields}, writes={lint_report.written_fields}"
        )
        return JudgeVerdict(
            approved=True,
            risk_tier=lint_report.risk_tier,
            confidence=1.0,
            reasons=reasons,
        )
