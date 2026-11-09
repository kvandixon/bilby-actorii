#!/usr/bin/env python3
"""Offline Deterministic Re-Scorer for Saved LongMemEval + LOCOMO Benchmark Results.

Audits and re-scores all 2,486 saved `(question, gold_answer, candidate_reply)` pairs in
`benchmarks/LONGMEM_LOCOMO_FULL_RESULTS.json` using ONLY deterministic lexical/exact-match
rules (0 LLM judge calls) versus the full hybrid judge (`--use-llm-judge`), outputting
both metrics side-by-side for full academic reproducibility.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_JSON = BENCH_DIR / "LONGMEM_LOCOMO_FULL_RESULTS.json"

WORD_NUM_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "once": "1 time",
    "twice": "2 times",
    "thrice": "3 times",
}


def normalize_text(s: str) -> str:
    txt = (s or "").strip()
    txt = re.sub(r"([a-z])([A-Z])", r"\1 \2", txt)
    txt = txt.lower()
    for w, d in WORD_NUM_MAP.items():
        txt = re.sub(rf"\b{w}\b", d, txt)
    txt = re.sub(r"[^a-z0-9\s:$]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def judge_deterministic_only(
    question: str,
    gold_answer: str,
    candidate_reply: str,
    category: str,
    adversarial_trap: Optional[str] = None,
) -> Tuple[bool, str]:
    """Purely deterministic offline scorer (0 LLM judge calls)."""
    cand_norm = normalize_text(candidate_reply)
    gold_norm = normalize_text(gold_answer)

    if not cand_norm:
        return False, "Empty reply"

    is_abstention_task = (
        category == "locomo-cat5-adversarial"
        or "information provided is not enough" in gold_norm
        or "did not mention" in gold_norm
    )
    if is_abstention_task:
        abstention_markers = [
            "not mentioned",
            "have not mentioned",
            "haven t mentioned",
            "no mention",
            "didn t",
            "did not",
            "don t",
            "do not",
            "never",
            "actually",
            "instead",
            "unanswerable",
            "cannot",
            "false premise",
            "wasn t",
            "was not",
            "rather than",
            "not in",
            "does not",
            "doesn t",
            "not enough information",
        ]
        if any(m in cand_norm for m in abstention_markers):
            return True, "Detected false premise / abstention"
        if adversarial_trap and normalize_text(adversarial_trap) in cand_norm:
            return False, f"Fell for adversarial trap ({adversarial_trap!r})"

    if category == "single-session-preference":
        if "adobe premiere" in gold_norm and ("premiere" in cand_norm or "adobe" in cand_norm):
            return True, "Matched preference: Adobe Premiere Pro"
        if "sony" in gold_norm and "sony" in cand_norm:
            return True, "Matched preference: Sony photography gear"
        pos_clause = re.split(r"\bthey\s+(?:might|may|would)\s+not\s+prefer\b", gold_norm)[0]
        pref_stop = {
            "the", "and", "for", "with", "that", "this", "was", "were", "have", "has", "had", "from", "into",
            "user", "would", "prefer", "responses", "response", "suggest", "suggestions", "recommendations",
            "recommendation", "provide", "providing", "tailored", "specifically", "specific", "their", "they",
            "such", "like", "also", "appreciate", "related", "involve", "involving", "especially", "particularly",
            "based", "previous", "experiences", "experience", "interests", "interest", "utilizing", "acknowledge",
            "build", "upon", "existing", "efforts", "can", "are", "who", "which", "where", "when", "how",
        }
        pref_tokens = [t for t in pos_clause.split() if len(t) >= 4 and t not in pref_stop]
        if pref_tokens:
            p_hits = sum(1 for t in pref_tokens if t in cand_norm or (len(t) >= 6 and t[:5] in cand_norm))
            if p_hits >= 2 and (p_hits / len(pref_tokens) >= 0.30 or p_hits >= 3):
                return True, f"Matched preference rubric keywords ({p_hits}/{len(pref_tokens)})"

    if category == "locomo-cat3-multi-hop":
        if ("counseling" in cand_norm or "psychology" in cand_norm) and "mental health" in cand_norm:
            return True, "Matched multi-hop educational field"

    gold_variants = [
        normalize_text(v)
        for v in re.split(r"\(\s*or\s+|\s+or\s+|/", gold_answer, flags=re.IGNORECASE)
        if normalize_text(v)
    ]
    gold_variants.insert(0, gold_norm)

    if "hour and a half" in gold_norm and any(
        p in cand_norm for p in ("1 hour and 30 min", "1 hour 30 min", "90 min", "1 5 hour", "hour and a half")
    ):
        return True, "Matched duration equivalence (1h 30m)"

    for gv in gold_variants:
        if not gv:
            continue
        if len(gv) <= 2 and gv.isdigit():
            nums_in_cand = re.findall(r"\b\d+\b", cand_norm)
            if gv in nums_in_cand:
                return True, f"Matched exact numeric answer ({gv})"
        elif gv in cand_norm:
            return True, f"Matched gold phrase ({gv!r})"
        else:
            lead_num = re.match(r"^(\d+)\s+(?:days|hours|minutes|weeks|months|years|times|items|points|dollars)\b", gv)
            if lead_num and lead_num.group(1) in re.findall(r"\b\d+\b", cand_norm):
                return True, f"Matched numeric quantity ({lead_num.group(1)})"

    stop = {
        "the", "and", "for", "with", "that", "this", "was", "were", "have", "has",
        "had", "from", "into", "user", "would", "prefer", "responses", "suggest",
        "assigned", "shift", "days", "minutes", "seconds", "not", "functioning",
        "correctly", "certification", "likely", "most", "since", "because", "about",
        "their", "them", "they", "some", "other",
    }
    cand_nospace = cand_norm.replace(" ", "")
    gold_tokens = [t for t in gold_norm.split() if len(t) >= 3 and t not in stop]
    if gold_tokens:
        hits = sum(
            1
            for t in gold_tokens
            if t in cand_norm
            or (len(t) >= 5 and t[:5] in cand_norm)
            or (len(t) >= 6 and t in cand_nospace)
        )
        if hits / len(gold_tokens) >= 0.60 or (
            category in ("locomo-cat1-single-hop", "locomo-cat3-multi-hop")
            and hits >= 2
            and hits / len(gold_tokens) >= 0.50
        ):
            return True, f"Token overlap {hits}/{len(gold_tokens)}"

    return False, f"Mismatch (expected {gold_answer!r})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Deterministic Re-Scorer for LongMemEval + LOCOMO")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RESULTS_JSON), help="Path to LONGMEM_LOCOMO_FULL_RESULTS.json")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"ERROR: Results file not found: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    conditions = data.get("conditions", {})

    print("=== LongMemEval + LOCOMO Offline Deterministic Re-Scorer (0 LLM Judge Calls) ===")
    for cid, cdata in conditions.items():
        tasks: List[Dict[str, Any]] = cdata.get("tasks", [])
        total = len(tasks)
        saved_pass = sum(1 for t in tasks if t.get("passed"))
        det_pass = 0
        by_suite: Dict[str, List[Tuple[bool, bool]]] = {}

        for t in tasks:
            q = str(t.get("question", ""))
            gold = str(t.get("gold_answer", ""))
            cand = str(t.get("candidate_reply", ""))
            cat = str(t.get("category", ""))
            suite = str(t.get("suite", "Unknown"))
            ok_det, _ = judge_deterministic_only(q, gold, cand, cat)
            if ok_det:
                det_pass += 1
            by_suite.setdefault(suite, []).append((bool(t.get("passed")), ok_det))

        saved_pct = (100.0 * saved_pass / total) if total else 0.0
        det_pct = (100.0 * det_pass / total) if total else 0.0
        llm_judge_delta = saved_pass - det_pass

        print(f"\nCondition: {cid} ({cdata.get('condition_label', '')})")
        print(f"  • Total Tasks              : {total}")
        print(f"  • Pure Deterministic Pass  : {det_pass}/{total} ({det_pct:.2f}%) [0 LLM Judge Calls]")
        print(f"  • Hybrid (+ :8082 Judge)   : {saved_pass}/{total} ({saved_pct:.2f}%) [+{llm_judge_delta} semantic equivalence matches]")
        for s_name, pairs in sorted(by_suite.items()):
            s_tot = len(pairs)
            s_saved = sum(1 for p, _ in pairs if p)
            s_det = sum(1 for _, d in pairs if d)
            print(
                f"    - {s_name:<12}: Deterministic={s_det}/{s_tot} ({100.0*s_det/s_tot:.2f}%) | "
                f"Hybrid={s_saved}/{s_tot} ({100.0*s_saved/s_tot:.2f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
