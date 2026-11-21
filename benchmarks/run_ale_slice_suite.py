#!/usr/bin/env python3
"""Agents' Last Exam (ALE v1.1, UC Berkeley RDI) Stratified 3-Domain Local Slice Runner.

Evaluates ActorII (or Pi / Hermes) inside isolated Docker sandboxes (`actorii-sandbox:latest`)
against 3 official, self-gradable ALE v1.1 tasks spanning 3 distinct professional domains
using the unmodified official Berkeley RDI deterministic Python graders:
  1. `ALE/k8s_payment_api_root_cause_analysis` (DevOps / SRE Incident Root-Cause Analysis)
  2. `ALE/cfr_game_theory_equilibrium`         (Algorithms / Game Theory — Matrix, Kuhn CFR, Leduc MCCFR)
  3. `ALE/power_10kv_feeder_reliability_001`   (Electrical Engineering — CIM/RDF XML + SVG Topology + Affine Indices)

Outputs:
  - `benchmarks/ALE_SLICE_RESULTS.json`
  - `benchmarks/ALE_SLICE_REPORT.md`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
ALE_REPO = ROOT / "benchmarks" / "data" / "agents-last-exam"
ALE_INPUTS = ROOT / "benchmarks" / "data" / "ale_inputs"

for p in (str(ROOT), str(ALE_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks.run_industry_harness_suite import (
    CONTAINER_IMAGES,
    TaskMetrics,
    dispatch_container,
)

ALE_SLICE_TASKS = [
    {
        "short_id": "k8s_payment_api_root_cause_analysis",
        "rel_path": "computing_math/k8s_payment_api_root_cause_analysis",
        "domain": "DevOps / SRE Incident RCA",
        "output_rel": "output/root_cause_analysis.json",
    },
    {
        "short_id": "cfr_game_theory_equilibrium",
        "rel_path": "computing_math/cfr_game_theory_equilibrium",
        "domain": "Algorithms / Game Theory (CFR + MCCFR)",
        "output_rel": "output/results.json",
    },
    {
        "short_id": "power_10kv_feeder_reliability_001",
        "rel_path": "engineering/power_10kv_feeder_reliability_001",
        "domain": "Electrical Engineering / CIM-RDF + SVG Reliability",
        "output_rel": "output/reliability_indices.json",
    },
]

RESULTS_JSON = ROOT / "benchmarks" / "ALE_SLICE_RESULTS.json"
REPORT_MD = ROOT / "benchmarks" / "ALE_SLICE_REPORT.md"


def _ensure_power_reference() -> Path:
    ref_path = ALE_INPUTS / "engineering/power_10kv_feeder_reliability_001/reference_reliability_indices.json"
    if ref_path.is_file():
        return ref_path
    in_dir = ALE_INPUTS / "engineering/power_10kv_feeder_reliability_001/input"
    ev_path = ref_path.parent / "reference_evidence.json"
    script = ALE_REPO / "tasks/engineering/power_10kv_feeder_reliability_001/scripts/build_reference_v2.py"
    subprocess.run(
        ["python3", str(script), "--input", str(in_dir), "--output", str(ref_path), "--evidence", str(ev_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return ref_path


def _grade_ale_task(short_id: str, workspace_dir: Path) -> Tuple[float, bool, Dict[str, Any]]:
    if short_id == "k8s_payment_api_root_cause_analysis":
        from tasks.computing_math.k8s_payment_api_root_cause_analysis.scripts.score_root_cause_analysis import (
            score_report,
        )

        out_file = workspace_dir / "output" / "root_cause_analysis.json"
        if not out_file.is_file():
            return 0.0, False, {"error": "Missing output/root_cause_analysis.json"}
        in_dir = workspace_dir / "input"
        res = score_report(
            out_file.read_bytes(),
            cluster_state=(in_dir / "cluster_state.txt").read_text(encoding="utf-8"),
            deployment_yaml=(in_dir / "deployment.yaml").read_text(encoding="utf-8"),
            failing_pod_log=(in_dir / "failing_pod.log").read_text(encoding="utf-8"),
        )
        score = float(getattr(res, "score", 0.0))
        details = getattr(res, "details", None) or getattr(res, "checks", None) or str(res)
        return score, score >= 0.999, {"score": score, "grader": details}

    if short_id == "cfr_game_theory_equilibrium":
        from tasks.computing_math.cfr_game_theory_equilibrium.scripts.score_outputs import (
            score_submission_texts,
        )

        out_file = workspace_dir / "output" / "results.json"
        if not out_file.is_file():
            return 0.0, False, {"error": "Missing output/results.json"}
        res = score_submission_texts(out_file.read_text(encoding="utf-8"), reference_text=None)
        score = float(getattr(res, "score", 0.0))
        details = getattr(res, "details", None) or getattr(res, "checks", None) or str(res)
        return score, score >= 0.999, {"score": score, "grader": details}

    if short_id == "power_10kv_feeder_reliability_001":
        from tasks.engineering.power_10kv_feeder_reliability_001.scripts.verify_reliability_indices import (
            score_submission,
        )

        out_file = workspace_dir / "output" / "reliability_indices.json"
        if not out_file.is_file():
            return 0.0, False, {"error": "Missing output/reliability_indices.json"}
        ref_path = _ensure_power_reference()
        agent_obj = json.loads(out_file.read_text(encoding="utf-8"))
        ref_obj = json.loads(ref_path.read_text(encoding="utf-8"))
        res = score_submission(agent_obj, ref_obj)
        score = float(res.get("score", 0.0))
        return score, score >= 0.999, res

    raise ValueError(f"Unknown ALE slice task: {short_id}")


def run_ale_task(
    task_spec: Dict[str, str],
    harness_name: str,
    model: str,
    *,
    max_steps: int = 32,
    timeout_s: int = 900,
) -> Tuple[TaskMetrics, Dict[str, Any]]:
    short_id = task_spec["short_id"]
    rel_path = task_spec["rel_path"]
    tid = f"ALE/{short_id}"

    card = json.loads((ALE_REPO / "tasks" / rel_path / "task_card.json").read_text(encoding="utf-8"))
    raw_prompt = card.get("taskPrompt", "")
    # Map ALE's `base/` paths to `/workspace/` inside the Docker sandbox
    mapped_prompt = (
        raw_prompt.replace("`base/", "`/workspace/")
        .replace("`base`", "`/workspace`")
        .replace("base/input/", "/workspace/input/")
        .replace("base/output/", "/workspace/output/")
        .replace("base/software/", "/workspace/software/")
    )
    prompt = (
        f"Complete the following Agents' Last Exam (ALE v1.1) task inside `/workspace`.\n"
        f"All staged input files are in `/workspace/input/`. Write the required deliverable to "
        f"`/workspace/{task_spec['output_rel']}` using Python (`python3` with numpy/scipy/pandas/pyyaml "
        f"is pre-installed). Verify the output file exists and matches the exact schema before completing.\n\n"
        f"{mapped_prompt}\n"
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_ale_{short_id}_{harness_name}_"))
    t0 = time.time()
    score = 0.0
    passed = False
    grader_info: Dict[str, Any] = {}
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0
    stdout_tail = ""

    try:
        src_input = ALE_INPUTS / rel_path / "input"
        shutil.copytree(src_input, tmp_dir / "input", dirs_exist_ok=True)
        (tmp_dir / "output").mkdir(parents=True, exist_ok=True)

        rc, out_txt, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=max_steps, timeout_s=timeout_s
        )
        stdout_tail = (out_txt or "")[-2500:]
        score, passed, grader_info = _grade_ale_task(short_id, tmp_dir)
    except Exception as exc:
        grader_info = {"error": f"Exception: {exc}"}
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = f"ALE score={score:.4f} ({'PASS' if passed else 'FAIL'})"
    if isinstance(grader_info, dict) and grader_info.get("error"):
        summary += f" | {str(grader_info['error'])[:120]}"

    metrics = TaskMetrics(
        task_id=tid,
        track="Agents' Last Exam v1.1 (UC Berkeley RDI)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=summary,
    )
    extra = {
        "short_id": short_id,
        "domain": task_spec["domain"],
        "score": round(score, 4),
        "grader_info": grader_info,
        "stdout_tail": stdout_tail,
    }
    return metrics, extra


def write_markdown_report(rows: List[Dict[str, Any]], model: str, out_md: Path) -> None:
    lines = [
        "# Agents' Last Exam (ALE v1.1) Stratified Slice Report",
        "",
        f"- **Model:** `{model}`",
        f"- **Tasks:** {len(rows)} official UC Berkeley RDI ALE v1.1 tasks (deterministic official Python graders)",
        "",
        "| Task | Domain | Harness | Score | Pass | Steps | Tokens | Latency | Details |",
        "|---|---|---|---:|:---:|---:|---:|---:|---|",
    ]
    total_score = 0.0
    total_pass = 0
    for r in rows:
        total_score += float(r.get("score", 0.0))
        total_pass += 1 if r.get("passed") else 0
        lines.append(
            f"| `{r['task_id']}` | {r['domain']} | `{r['harness']}` | **{r['score']:.4f}** | "
            f"{'PASS' if r['passed'] else 'FAIL'} | {r['steps']} | {r['total_tokens']:,} | "
            f"{r['latency_s']}s | {r['details']} |"
        )
    avg_score = total_score / max(1, len(rows))
    lines.extend([
        "",
        f"**Aggregate:** `{total_pass}/{len(rows)}` full pass | Mean ALE Score: **`{avg_score:.4f}`**",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stratified ALE v1.1 slice on ActorII.")
    parser.add_argument("--model", default="qwen3.8:latest", help="Model served on :8080")
    parser.add_argument("--harness", choices=["actorii", "pi", "hermes"], default="actorii")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tasks", nargs="*", default=[], help="Optional subset of short_ids")
    args = parser.parse_args()

    selected = (
        [t for t in ALE_SLICE_TASKS if t["short_id"] in args.tasks]
        if args.tasks
        else ALE_SLICE_TASKS
    )
    rows: List[Dict[str, Any]] = []
    print(
        f"\n================ AGENTS' LAST EXAM (ALE v1.1) SLICE ({len(selected)} TASKS | harness={args.harness} | model={args.model}) ================",
        flush=True,
    )
    for spec in selected:
        print(f"  -> Running ALE/{spec['short_id']} ({spec['domain']}) ...", flush=True)
        m, extra = run_ale_task(
            spec, args.harness, args.model, max_steps=args.max_steps, timeout_s=args.timeout
        )
        row = {**asdict(m), **extra}
        rows.append(row)
        print(
            f"     [{args.harness}] score={row['score']:.4f} ({'PASS' if row['passed'] else 'FAIL'}) | "
            f"{m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
            flush=True,
        )
        RESULTS_JSON.write_text(
            json.dumps({"model": args.model, "suite": "ALE v1.1 Stratified 3-Domain Slice", "results": rows}, indent=2, default=str),
            encoding="utf-8",
        )
        write_markdown_report(rows, args.model, REPORT_MD)

    print(f"\nSaved results to {RESULTS_JSON} and {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
