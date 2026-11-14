#!/usr/bin/env python3
"""Terminal-Bench 2.1 Hard / Upper-Medium 5-Task Suite (`ActorII` vs `Pi` vs `Hermes`).

Evaluates 5 unseen `hard` and upper-`medium` tasks from `harborframework/terminal-bench-2.1`
across all 3 harnesses (`actorii-sandbox:latest`, `pi-bench:latest`, `hermes-bench:latest`):
  1. `TB2.1-Hard/llm-inference-batching-scheduler` (hard, ML/Systems, 45m expert est.)
  2. `TB2.1-Hard/feal-differential-cryptanalysis` (hard, Cryptography/Math, 480m expert est.)
  3. `TB2.1-Hard/fix-code-vulnerability` (hard, Security/CWE, 120m expert est.)
  4. `TB2.1-Hard/db-wal-recovery` (medium, SQLite WAL Forensics, 45m expert est.)
  5. `TB2.1-Hard/constraints-scheduling` (medium, CSP/Calendar Scheduling, 15m expert est.)

Writes dedicated output artifacts without modifying any prior benchmark files:
  - `benchmarks/TB21_HARD_RESULTS.json`
  - `benchmarks/TB21_HARD_REPORT.md`
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
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_industry_harness_suite import (
    CONTAINER_IMAGES,
    TaskMetrics,
    dispatch_container,
    generate_markdown_report,
)

HARNESS_ORDER = ("actorii", "pi", "hermes")
NEXTGEN_DIR = ROOT / "benchmarks" / "data" / "nextgen"
TB_DIR = NEXTGEN_DIR / "terminal_bench_2_1"

TB_HARD_5_TASKS = [
    "llm-inference-batching-scheduler",
    "feal-differential-cryptanalysis",
    "fix-code-vulnerability",
    "db-wal-recovery",
    "constraints-scheduling",
]

RESULTS_JSON = ROOT / "benchmarks" / "TB21_HARD_RESULTS.json"
REPORT_MD = ROOT / "benchmarks" / "TB21_HARD_REPORT.md"

_BOTTLE_CACHE: Path | None = None


def _ensure_bottle_vuln_template() -> Path:
    """Clone `bottlepy/bottle` once at `0207a34f` and inject the task's vulnerable `bottle.py` lines."""
    global _BOTTLE_CACHE
    if _BOTTLE_CACHE is not None and _BOTTLE_CACHE.exists():
        return _BOTTLE_CACHE

    cache_dir = Path(tempfile.mkdtemp(prefix="tb_bottle_vuln_template_"))
    setup_sh = """
git clone -q -o origin --single-branch https://github.com/bottlepy/bottle.git .
git reset -q --hard 0207a34f0c5716cd292dd4480253ad35d3da49f3
git remote remove origin
git tag -d $(git tag -l) >/dev/null 2>&1 || true
git reflog expire --expire=now --all
git gc --prune=now --aggressive -q
sed -i -e '1562,1563d' bottle.py
sed -i -e '1567,1568d' bottle.py
"""
    subprocess.run(["bash", "-e", "-c", setup_sh], cwd=str(cache_dir), check=True, capture_output=True)
    _BOTTLE_CACHE = cache_dir
    return cache_dir


def run_terminal_bench_hard_task(task_name: str, harness_name: str, model: str) -> TaskMetrics:
    tid = f"TB2.1-Hard/{task_name}"
    src_dir = TB_DIR / task_name
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_tbhard_{task_name}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        # 1. Initialize official task environment inside `tmp_dir` (/workspace)
        if task_name == "llm-inference-batching-scheduler":
            shutil.copytree(
                src_dir / "environment" / "task_file",
                tmp_dir / "task_file",
                dirs_exist_ok=True,
            )
        elif task_name == "feal-differential-cryptanalysis":
            shutil.copy2(src_dir / "environment" / "feal.py", tmp_dir / "feal.py")
        elif task_name == "fix-code-vulnerability":
            tpl = _ensure_bottle_vuln_template()
            shutil.copytree(tpl, tmp_dir, dirs_exist_ok=True)
        elif task_name == "db-wal-recovery":
            shutil.copy2(src_dir / "environment" / "main.db", tmp_dir / "main.db")
            shutil.copy2(
                src_dir / "environment" / "main.db-wal.encrypted",
                tmp_dir / "main.db-wal",
            )
        elif task_name == "constraints-scheduling":
            for f in (src_dir / "environment" / "inputs").iterdir():
                if f.is_file():
                    shutil.copy2(f, tmp_dir / f.name)

        # 2. Prepare official instruction.md (mapping container `/app` -> `/workspace`)
        inst_raw = (src_dir / "instruction.md").read_text(encoding="utf-8")
        inst_workspace = inst_raw.replace("/app", "/workspace")
        prompt = (
            f"Complete the following Terminal-Bench 2.1 task inside `/workspace`.\n\n"
            f"{inst_workspace}\n"
        )

        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=16, timeout_s=600
        )

        # 3. Run official `tests/test_outputs.py` verification suite against `tmp_dir`
        verify_dir = Path(tempfile.mkdtemp(prefix=f"verify_tbhard_{task_name}_"))
        try:
            if task_name == "fix-code-vulnerability":
                rep_f = tmp_dir / "report.jsonl"
                if rep_f.is_file():
                    rep_txt = rep_f.read_text(encoding="utf-8").replace(
                        "/workspace/bottle.py", f"{tmp_dir}/bottle.py"
                    )
                    rep_f.write_text(rep_txt, encoding="utf-8")
                bottle_sub = verify_dir / "bottle"
                bottle_sub.mkdir(parents=True, exist_ok=True)
                if (tmp_dir / "bottle.py").is_file():
                    shutil.copy2(tmp_dir / "bottle.py", bottle_sub / "bottle.py")

            for f in (src_dir / "tests").iterdir():
                if f.is_file():
                    txt = (
                        f.read_text(encoding="utf-8")
                        .replace("/app", str(tmp_dir))
                        .replace('"/tests"', f'"{verify_dir}"')
                        .replace("(cd /tests && python ", f"(cd {verify_dir} && python3 ")
                    )
                    (verify_dir / f.name).write_text(txt, encoding="utf-8")

            env = {**os.environ, "PYTHONPATH": f"{tmp_dir}:{verify_dir}"}
            v_proc = subprocess.run(
                ["python3", "-m", "pytest", "-q", str(verify_dir / "test_outputs.py")],
                cwd=str(verify_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            passed = v_proc.returncode == 0
            tail = (v_proc.stdout or v_proc.stderr or "").strip().splitlines()
            summary_ln = tail[-1] if tail else f"rc={v_proc.returncode}"
            details = f"pytest test_outputs.py: {'PASS' if passed else 'FAIL'} ({summary_ln[:80]})"
        finally:
            shutil.rmtree(verify_dir, ignore_errors=True)
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 5 Hard / Upper-Medium Terminal-Bench 2.1 tasks across ActorII, Pi, and Hermes."
    )
    parser.add_argument("--model", default="qwen3.8:latest", help="Target model on :8080")
    parser.add_argument(
        "--harness",
        choices=["all", "actorii", "pi", "hermes"],
        default="all",
        help="Harness to evaluate (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Independent trials per task at production temp=0.1 (default: 3 for avg@3 on N <= 10 suites)",
    )
    parser.add_argument("--results-json", default="", help="Override output JSON path")
    parser.add_argument("--report-md", default="", help="Override output Markdown report path")
    args = parser.parse_args()

    out_json = Path(args.results_json).resolve() if args.results_json else RESULTS_JSON
    out_md = Path(args.report_md).resolve() if args.report_md else REPORT_MD
    trials = max(1, int(args.trials))

    harnesses = HARNESS_ORDER if args.harness == "all" else (args.harness,)
    results: List[TaskMetrics] = []
    result_dicts: list[dict] = []
    print(
        f"\n================ TERMINAL-BENCH 2.1 HARD SUITE (5 TASKS x {trials} TRIALS x {len(harnesses)} HARNESS) ================",
        flush=True,
    )
    for trial_idx in range(1, trials + 1):
        if trials > 1:
            print(f"\n--- Trial {trial_idx} / {trials} ---", flush=True)
        for tname in TB_HARD_5_TASKS:
            for h in harnesses:
                print(f"  -> [T{trial_idx}/{trials}] Running TB2.1-Hard/{tname} on [{h}] ...", flush=True)
                m = run_terminal_bench_hard_task(tname, h, args.model)
                results.append(m)
                d_row = asdict(m)
                d_row["trial"] = trial_idx
                result_dicts.append(d_row)
                status = "PASS" if m.passed else "FAIL"
                print(
                    f"     [{h}] {status} | {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
                    flush=True,
                )
                payload = {
                    "model": args.model,
                    "suite": "Terminal-Bench 2.1 Hard Tier (5 Tasks)",
                    "trials": trials,
                    "results": result_dicts,
                }
                out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    payload = {
        "model": args.model,
        "suite": "Terminal-Bench 2.1 Hard Tier (5 Tasks)",
        "trials": trials,
        "results": result_dicts,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    generate_markdown_report(results, args.model, out_md)
    print(f"\nSaved JSON to {out_json} and report to {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
