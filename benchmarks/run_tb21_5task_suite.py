#!/usr/bin/env python3
"""Terminal-Bench 2.1 (5-Task Full Suite) — `ActorII` vs `Pi` vs `Hermes`.

Evaluates all 5 official `Terminal-Bench 2.1` tasks on disk across all 3 harnesses
(`actorii-sandbox:latest`, `pi-bench:latest`, `hermes-bench:latest`) on `qwen3.8:latest`:
  1. `TB2.1/git-leak-recovery`
  2. `TB2.1/log-summary-date-ranges`
  3. `TB2.1/cancel-async-tasks`
  4. `TB2.1/fix-git`
  5. `TB2.1/multi-source-data-merger`

Preserves all previous benchmark reports untouched and writes to:
  - `benchmarks/TB21_5TASK_RESULTS.json`
  - `benchmarks/TB21_5TASK_REPORT.md`
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

TB_5_TASKS = [
    "git-leak-recovery",
    "log-summary-date-ranges",
    "cancel-async-tasks",
    "fix-git",
    "multi-source-data-merger",
]

RESULTS_JSON = ROOT / "benchmarks" / "TB21_5TASK_RESULTS.json"
REPORT_MD = ROOT / "benchmarks" / "TB21_5TASK_REPORT.md"

_FIX_GIT_CACHE: Path | None = None


def _ensure_fix_git_template(src_dir: Path) -> Path:
    """Build the initial detached-HEAD `personal-site` repo once and cache it."""
    global _FIX_GIT_CACHE
    if _FIX_GIT_CACHE is not None and _FIX_GIT_CACHE.exists():
        return _FIX_GIT_CACHE

    cache_dir = Path(tempfile.mkdtemp(prefix="tb_fix_git_template_"))
    repo_dir = cache_dir / "personal-site"
    (repo_dir / "_includes").mkdir(parents=True, exist_ok=True)
    (repo_dir / "_layouts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "_includes" / "about.md").write_text("# About\nInitial bio.\n", encoding="utf-8")
    (repo_dir / "_layouts" / "default.html").write_text("<html><body>Initial layout</body></html>\n", encoding="utf-8")
    (repo_dir / "README.md").write_text("# Personal Site\n", encoding="utf-8")

    patch_dir = src_dir / "environment" / "resources" / "patch_files"
    init_sh = f"""
git init -b master
git config user.email "test@example.com"
git config user.name "Test User"
git add -A
git commit -m "Initial site"
echo "v2" >> README.md
git add README.md
git commit -m "Update README"
git checkout HEAD~1
cp "{patch_dir}/about.md" ./_includes/about.md
cp "{patch_dir}/default.html" ./_layouts/default.html
git add -A
git commit -m "Move to Stanford"
git checkout master
"""
    subprocess.run(["bash", "-e", "-c", init_sh], cwd=str(repo_dir), check=True, capture_output=True)
    _FIX_GIT_CACHE = cache_dir
    return cache_dir


def run_terminal_bench_5task(task_name: str, harness_name: str, model: str) -> TaskMetrics:
    tid = f"TB2.1/{task_name}"
    src_dir = TB_DIR / task_name
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_tb5_{task_name}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        # 1. Initialize official task environment inside `tmp_dir` (/workspace)
        if task_name == "git-leak-recovery":
            setup_sh = (src_dir / "environment" / "challenge-setup.sh").read_text(encoding="utf-8")
            subprocess.run(["bash", "-c", setup_sh], cwd=str(tmp_dir), check=True, capture_output=True)
        elif task_name == "log-summary-date-ranges":
            gen_py = (src_dir / "environment" / "log_generator_deterministic.py").read_text(encoding="utf-8")
            gen_py = gen_py.replace('"/app/logs"', f'"{tmp_dir}/logs"')
            subprocess.run(["python3", "-c", gen_py], cwd=str(tmp_dir), check=True, capture_output=True)
        elif task_name == "cancel-async-tasks":
            pass
        elif task_name == "fix-git":
            tpl = _ensure_fix_git_template(src_dir)
            shutil.copytree(tpl / "personal-site", tmp_dir / "personal-site", dirs_exist_ok=True)
        elif task_name == "multi-source-data-merger":
            shutil.copytree(src_dir / "environment" / "data", tmp_dir / "data", dirs_exist_ok=True)

        # 2. Prepare official instruction.md (mapping container `/app` -> `/workspace` and `/data/` -> `/workspace/data/`)
        inst_raw = (src_dir / "instruction.md").read_text(encoding="utf-8")
        inst_workspace = inst_raw.replace("/app", "/workspace").replace("/data/", "/workspace/data/")
        ws_hint = "/workspace/personal-site" if task_name == "fix-git" else "/workspace"
        prompt = (
            f"Complete the following Terminal-Bench 2.1 task inside `{ws_hint}`.\n\n"
            f"{inst_workspace}\n"
        )

        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=16, timeout_s=600
        )

        # 3. Run official `tests/test_outputs.py` verification suite against `tmp_dir`
        verify_dir = Path(tempfile.mkdtemp(prefix=f"verify_tb5_{task_name}_"))
        try:
            for f in (src_dir / "tests").iterdir():
                if f.is_file():
                    txt = (
                        f.read_text(encoding="utf-8")
                        .replace("/app", str(tmp_dir))
                        .replace(
                            "-exec sha256sum {} + | sort",
                            f"-exec sha256sum {{}} + | sed 's|{tmp_dir}|/app|g' | sort",
                        )
                    )
                    (verify_dir / f.name).write_text(txt, encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": f"{tmp_dir}:{verify_dir}"}
            v_proc = subprocess.run(
                ["python3", "-m", "pytest", "-q", str(verify_dir / "test_outputs.py")],
                cwd=str(verify_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
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
        track="Track 1: Terminal-Bench 2.1 (Stanford/Laude)",
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


def _summarize_multitrial(rows: list[dict], trials: int, harness: str) -> dict:
    h_rows = [r for r in rows if r.get("harness") == harness]
    by_tid: dict[str, list[bool]] = {}
    for r in h_rows:
        by_tid.setdefault(str(r["task_id"]), []).append(bool(r.get("passed")))
    n_u = max(1, len(by_tid))
    tot = len(h_rows)
    passed = sum(1 for r in h_rows if r.get("passed"))
    pass_hat = sum(1 for bs in by_tid.values() if len(bs) >= trials and all(bs[:trials]))
    pass_at = sum(1 for bs in by_tid.values() if any(bs))
    return {
        "trials": trials,
        "unique_tasks": len(by_tid),
        "trial_passed": passed,
        "trial_total": tot,
        "avg_at_k_pct": round(100.0 * passed / max(1, tot), 2),
        "pass_hat_k": f"{pass_hat}/{len(by_tid)} ({100.0 * pass_hat / n_u:.1f}%)",
        "pass_at_k": f"{pass_at}/{len(by_tid)} ({100.0 * pass_at / n_u:.1f}%)",
        "per_task": {tid: f"{sum(bs)}/{len(bs)}" for tid, bs in by_tid.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5 Terminal-Bench 2.1 tasks across ActorII, Pi, and Hermes.")
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
    parser.add_argument("--seed-json", default="", help="Optional completed Trial-1 JSON to seed Trial 1")
    args = parser.parse_args()

    out_json = Path(args.results_json).resolve() if args.results_json else RESULTS_JSON
    out_md = Path(args.report_md).resolve() if args.report_md else REPORT_MD
    trials = max(1, int(args.trials))

    harnesses = HARNESS_ORDER if args.harness == "all" else (args.harness,)
    results: List[TaskMetrics] = []
    result_dicts: list[dict] = []

    start_trial = 1
    if args.seed_json:
        seed_path = Path(args.seed_json).resolve()
        if seed_path.exists():
            seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_rows = seed_data.get("results", seed_data) if isinstance(seed_data, dict) else seed_data
            for sr in seed_rows:
                if isinstance(sr, dict) and sr.get("harness") in harnesses:
                    row_copy = dict(sr)
                    row_copy["trial"] = int(row_copy.get("trial") or 1)
                    result_dicts.append(row_copy)
                    results.append(
                        TaskMetrics(**{k: v for k, v in row_copy.items() if k in TaskMetrics.__dataclass_fields__})
                    )
            if result_dicts:
                start_trial = max(int(r.get("trial") or 1) for r in result_dicts) + 1
                print(f"[seed] Loaded {len(result_dicts)} rows through Trial {start_trial - 1} from {seed_path.name}", flush=True)

    print(
        f"\n================ TERMINAL-BENCH 2.1 (5 TASKS x {trials} TRIALS x {len(harnesses)} HARNESS) ================",
        flush=True,
    )
    for trial_idx in range(start_trial, trials + 1):
        if trials > 1:
            print(f"\n--- Trial {trial_idx} / {trials} ---", flush=True)
        for tname in TB_5_TASKS:
            for h in harnesses:
                print(f"  -> [T{trial_idx}/{trials}] Running TB2.1/{tname} on [{h}] ...", flush=True)
                m = run_terminal_bench_5task(tname, h, args.model)
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
                    "suite": "Terminal-Bench 2.1 (5-Task Expansion)",
                    "trials": trials,
                    "summary": {h_name: _summarize_multitrial(result_dicts, trials, h_name) for h_name in harnesses},
                    "results": result_dicts,
                }
                out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    payload = {
        "model": args.model,
        "suite": "Terminal-Bench 2.1 (5-Task Expansion)",
        "trials": trials,
        "summary": {h_name: _summarize_multitrial(result_dicts, trials, h_name) for h_name in harnesses},
        "results": result_dicts,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    generate_markdown_report(results, args.model, out_md)
    for h_name in harnesses:
        s = payload["summary"][h_name]
        print(
            f"\n[{h_name}] avg@{trials}: {s['avg_at_k_pct']:.2f}% ({s['trial_passed']}/{s['trial_total']}) "
            f"| pass^{trials}: {s['pass_hat_k']} | pass@{trials}: {s['pass_at_k']} | tasks: {s['per_task']}",
            flush=True,
        )
    print(f"\nSaved JSON to {out_json} and report to {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
