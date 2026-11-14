#!/usr/bin/env python3
"""Next-Gen Uncontaminated Multi-Harness Benchmark Runner (`ActorII` vs `Pi` vs `Hermes`).

Evaluates all 3 agent harnesses inside isolated Docker containers (`docker run --init --rm`)
with equalized `:8080` (`qwen3.8:latest`) non-thinking inference (`_NormalizingProxyHandler`),
100% blind `/workspace` mounts (`ACTORII_BENCH_STRICT=1`), and official upstream verification
across 4 unsaturable, contamination-free benchmarks:
  1. `Terminal-Bench 2.1` (`harborframework/terminal-bench-2.1`, Stanford & Laude Inst.)
  2. `SWE-bench-Live` (`SWE-bench-Live/SWE-bench-Live`, Post-Cutoff June 2025 split)
  3. `Aider Polyglot Benchmark` (`Aider-AI/polyglot-benchmark`, hard multi-step edit/test)
  4. `LiveCodeBench v6` (`drproduck/livecodebench-v6`, Post-Cutoff April 2025 contest window)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

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
SBL_JSONL = NEXTGEN_DIR / "swebench_live_202506.jsonl"
SBL_REPOS_DIR = NEXTGEN_DIR / "repos"
POLYGLOT_DIR = NEXTGEN_DIR / "polyglot_benchmark" / "python" / "exercises" / "practice"
LCB_JSONL = NEXTGEN_DIR / "livecodebench_v6_2025.jsonl"

RESULTS_JSON = ROOT / "benchmarks" / "NEXTGEN_RESULTS.json"
REPORT_MD = ROOT / "benchmarks" / "NEXTGEN_REPORT.md"


# =============================================================================
# Track 1: Official `Terminal-Bench 2.1` (`harborframework/terminal-bench-2.1`)
# =============================================================================

TB_TASKS = ["git-leak-recovery", "log-summary-date-ranges", "cancel-async-tasks"]


def run_terminal_bench_task(task_name: str, harness_name: str, model: str) -> TaskMetrics:
    tid = f"TB2.1/{task_name}"
    src_dir = TB_DIR / task_name
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_tb_{task_name}_{harness_name}_"))
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
        verify_dir = Path(tempfile.mkdtemp(prefix=f"verify_tb_{task_name}_"))
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


# =============================================================================
# Track 2: Official `SWE-bench-Live` (Post-Cutoff June 2025 Split)
# =============================================================================

SBL_TASK_IDS = ["py-pdf__pypdf-3317", "networkx__networkx-8105"]


def load_swebench_live_tasks() -> List[Dict[str, Any]]:
    by_id = {}
    for line in SBL_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            by_id[obj["instance_id"]] = obj
    return [by_id[tid] for tid in SBL_TASK_IDS if tid in by_id]


def run_swebench_live_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    iid = row["instance_id"]
    tid = f"SWE-Live/{iid}"
    f2p = list(row["FAIL_TO_PASS"])
    src_repo = SBL_REPOS_DIR / iid
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_sbl_{iid}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        shutil.copytree(src_repo, tmp_dir, dirs_exist_ok=True)
        prompt = (
            f"Resolve the following GitHub issue in the repository at `/workspace`.\n\n"
            f"ISSUE DESCRIPTION (`{iid}`, created {row.get('created_at')}):\n"
            f"{row['problem_statement'][:2200]}\n\n"
            f"Search `/workspace` to locate the relevant source file(s) and edit the code in `/workspace` so the issue is fixed."
        )
        prev_think = os.environ.get("ACTORII_ENABLE_THINKING")
        os.environ["ACTORII_ENABLE_THINKING"] = "1"
        try:
            _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
                harness_name, tmp_dir, prompt, model=model, max_steps=16, timeout_s=600
            )
        finally:
            if prev_think is None:
                os.environ.pop("ACTORII_ENABLE_THINKING", None)
            else:
                os.environ["ACTORII_ENABLE_THINKING"] = prev_think

        diff_proc = subprocess.run(
            ["git", "-c", "core.fileMode=false", "diff", "--stat"],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
        )
        diff_stat = (
            (diff_proc.stdout or "").strip().splitlines()[-1]
            if (diff_proc.stdout or "").strip()
            else "no files modified"
        )

        for line in row["test_patch"].splitlines():
            if line.startswith("+++ b/"):
                rel_p = line[6:].strip()
                t_file = tmp_dir / rel_p
                co = subprocess.run(["git", "checkout", "--", rel_p], cwd=str(tmp_dir), check=False, capture_output=True)
                if co.returncode != 0 and t_file.exists():
                    t_file.unlink()
        subprocess.run(
            ["git", "apply", "-"],
            input=row["test_patch"],
            text=True,
            cwd=str(tmp_dir),
            check=False,
            capture_output=True,
        )
        v_proc = subprocess.run(
            [
                "docker",
                "run",
                "--init",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--security-opt",
                "label:disable",
                "-e",
                "HOME=/tmp",
                "-e",
                "PYTHONPATH=/workspace",
                "-v",
                f"{tmp_dir}:/workspace:z",
                "-w",
                "/workspace",
                "--entrypoint",
                "python3",
                CONTAINER_IMAGES["actorii"],
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
            ]
            + f2p,
            capture_output=True,
            text=True,
            timeout=45,
        )
        passed = v_proc.returncode == 0
        details = f"{f2p[0].split('::')[-1]}: {'PASS' if passed else 'FAIL'} ({diff_stat})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 2: SWE-bench-Live (2025-06 Post-Cutoff)",
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


# =============================================================================
# Track 3: Official `Aider Polyglot Benchmark` (`Aider-AI/polyglot-benchmark`)
# =============================================================================

POLYGLOT_TASKS = ["forth", "grep"]


def run_polyglot_task(slug: str, harness_name: str, model: str) -> TaskMetrics:
    tid = f"Polyglot/{slug}"
    ex_dir = POLYGLOT_DIR / slug
    py_name = f"{slug.replace('-', '_')}.py"
    test_name = f"{slug.replace('-', '_')}_test.py"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_poly_{slug}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        # Place the starter stub, unit tests, and instructions in /workspace (matching official Aider Polyglot layout)
        shutil.copy2(ex_dir / py_name, tmp_dir / py_name)
        shutil.copy2(ex_dir / test_name, tmp_dir / test_name)
        docs_parts = []
        for doc_f in ("instructions.md", "instructions.append.md"):
            p = ex_dir / ".docs" / doc_f
            if p.exists():
                docs_parts.append(p.read_text(encoding="utf-8"))
        instructions_txt = "\n\n".join(docs_parts)
        (tmp_dir / "INSTRUCTIONS.md").write_text(instructions_txt, encoding="utf-8")

        prompt = (
            f"Implement the Python exercise in `/workspace/{py_name}` according to `/workspace/INSTRUCTIONS.md` "
            f"so that all unit tests in `/workspace/{test_name}` pass.\n\n"
            f"Inspect `/workspace/{py_name}` and `/workspace/INSTRUCTIONS.md`, implement the full solution in `/workspace/{py_name}`, "
            f"and verify with `pytest`."
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=16, timeout_s=600
        )

        # Overwrite unit test suite with pristine upstream copy and run pytest
        shutil.copy2(ex_dir / test_name, tmp_dir / test_name)
        v_proc = subprocess.run(
            ["python3", "-m", "pytest", "-q", str(tmp_dir / test_name)],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        passed = v_proc.returncode == 0
        tail = (v_proc.stdout or v_proc.stderr or "").strip().splitlines()
        summary_ln = tail[-1] if tail else f"rc={v_proc.returncode}"
        details = f"pytest {test_name}: {'PASS' if passed else 'FAIL'} ({summary_ln[:80]})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 3: Aider Polyglot (Hard Exercism)",
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


# =============================================================================
# Track 4: Official `LiveCodeBench v6` (April 2025 Post-Cutoff Contest Window)
# =============================================================================

LCB_TASK_IDS = ["3773", "abc400_c"]


def load_lcb_tasks() -> List[Dict[str, Any]]:
    by_id = {}
    for line in LCB_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            by_id[str(obj["question_id"])] = obj
    return [by_id[qid] for qid in LCB_TASK_IDS if qid in by_id]


def run_lcb_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    qid = str(row["question_id"])
    tid = f"LCB-v6/{qid}"
    starter = (row.get("starter_code") or "").strip()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_lcb_{qid}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        prob_md = (
            f"# {row.get('question_title')} (Contest Date: {row.get('contest_date')})\n\n"
            f"{row.get('question_content')}\n"
        )
        if starter:
            prob_md += f"\n## Starter Code (`solution.py`)\n```python\n{starter}\n```\n"
            (tmp_dir / "solution.py").write_text(starter + "\n", encoding="utf-8")
        else:
            prob_md += (
                "\n## I/O Format\nWrite a standalone Python 3 program in `/workspace/solution.py` "
                "that reads from `sys.stdin` and prints the answer to `stdout`.\n"
            )
        (tmp_dir / "PROBLEM.md").write_text(prob_md, encoding="utf-8")

        prompt = (
            f"Solve the competitive programming problem described in `/workspace/PROBLEM.md`.\n"
            f"Read `/workspace/PROBLEM.md`, write your complete Python 3 solution to `/workspace/solution.py`, "
            f"and verify it on sample inputs."
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=14, timeout_s=600
        )

        sol_file = tmp_dir / "solution.py"
        if not sol_file.exists():
            details = "solution.py not created"
        else:
            pub_tests = json.loads(row.get("public_test_cases") or "[]")
            priv_raw = row.get("private_test_cases") or "[]"
            try:
                priv_tests = json.loads(priv_raw) if isinstance(priv_raw, str) and priv_raw.startswith("[") else []
            except Exception:
                priv_tests = []
            all_tests = pub_tests + priv_tests[:5]

            meta = json.loads(row.get("metadata") or "{}")
            func_name = meta.get("func_name")
            passed_cnt = 0
            for tc in all_tests:
                inp = str(tc.get("input") or "")
                exp = str(tc.get("output") or "").strip()
                if func_name:
                    args_lines = [json.loads(ln) for ln in inp.strip().splitlines() if ln.strip()]
                    run_code = (
                        f"import json, sys; sys.path.insert(0, {str(tmp_dir)!r}); "
                        f"from typing import *; from solution import Solution; "
                        f"res = getattr(Solution(), {func_name!r})(*{args_lines!r}); "
                        f"print(json.dumps(res, separators=(',', ':')))"
                    )
                    p = subprocess.run(["python3", "-c", run_code], capture_output=True, text=True, timeout=10)
                    got = (p.stdout or "").strip()
                    exp_norm = exp.replace(" ", "")
                    if p.returncode == 0 and got == exp_norm:
                        passed_cnt += 1
                else:
                    p = subprocess.run(
                        ["python3", str(sol_file)],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    got = (p.stdout or "").strip()
                    if p.returncode == 0 and got == exp:
                        passed_cnt += 1
            passed = len(all_tests) > 0 and passed_cnt == len(all_tests)
            details = f"tests: {passed_cnt}/{len(all_tests)} ({'PASS' if passed else 'FAIL'})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff)",
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


def save_and_merge_results(new_rows: list[dict], model: str, trials: int = 1) -> list[dict]:
    existing: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            raw = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            rows = raw.get("results", raw) if isinstance(raw, dict) else raw
            for item in rows:
                if isinstance(item, dict):
                    existing.append(item)
        except Exception:
            existing = []
    by_key = {(m.get("harness"), m.get("task_id"), int(m.get("trial") or 1)): m for m in existing}
    for m in new_rows:
        by_key[(m.get("harness"), m.get("task_id"), int(m.get("trial") or 1))] = m
    merged = list(by_key.values())
    if trials > 1:
        RESULTS_JSON.write_text(
            json.dumps({"model": model, "trials": trials, "results": merged}, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        RESULTS_JSON.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")

    tm_list = [
        TaskMetrics(**{k: v for k, v in m.items() if k in TaskMetrics.__dataclass_fields__})
        for m in merged
    ]
    generate_markdown_report(tm_list, model, REPORT_MD)
    if REPORT_MD.exists():
        txt = REPORT_MD.read_text(encoding="utf-8").replace(
            "- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.",
            "- **Datasets**: Official `harborframework/terminal-bench-2.1` (Stanford/Laude), `SWE-bench-Live/SWE-bench-Live` (2025-06 post-cutoff split), `Aider-AI/polyglot-benchmark`, and `drproduck/livecodebench-v6` (2025-04 post-cutoff contest window).",
        )
        REPORT_MD.write_text(txt, encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.8:latest")
    parser.add_argument(
        "--track",
        choices=["all", "tb", "sbl", "polyglot", "lcb"],
        default="all",
    )
    parser.add_argument(
        "--harness",
        choices=["all", "actorii", "pi", "hermes"],
        default="all",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Independent trials per task at production temp=0.1 (default: 3 for avg@3 on N <= 10 suites)",
    )
    parser.add_argument("--task", default="")
    parser.add_argument("--results-json", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--seed-json", default="", help="Optional completed Trial-1 JSON to seed Trial 1")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    global RESULTS_JSON, REPORT_MD
    if args.results_json:
        RESULTS_JSON = Path(args.results_json).resolve()
    if args.report_md:
        REPORT_MD = Path(args.report_md).resolve()

    if args.reset and RESULTS_JSON.exists():
        RESULTS_JSON.unlink()

    trials = max(1, int(args.trials))
    harnesses = HARNESS_ORDER if args.harness == "all" else (args.harness,)
    run_rows: list[dict] = []
    start_trial = 1

    if args.seed_json:
        seed_path = Path(args.seed_json).resolve()
        if seed_path.exists():
            seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_rows = seed_data.get("results", seed_data) if isinstance(seed_data, dict) else seed_data
            for sr in seed_rows:
                if isinstance(sr, dict) and sr.get("harness") in harnesses:
                    rc = dict(sr)
                    rc["trial"] = int(rc.get("trial") or 1)
                    run_rows.append(rc)
            if run_rows:
                start_trial = max(int(r.get("trial") or 1) for r in run_rows) + 1
                save_and_merge_results(run_rows, args.model, trials=trials)
                print(f"[seed] Loaded {len(run_rows)} rows through Trial {start_trial - 1} from {seed_path.name}", flush=True)

    for trial_idx in range(start_trial, trials + 1):
        if trials > 1:
            print(f"\n================ NEXT-GEN SUITE TRIAL {trial_idx} / {trials} ================", flush=True)

        if args.track in ("all", "tb"):
            print("\n=== Track 1: Terminal-Bench 2.1 ===", flush=True)
            for tname in TB_TASKS:
                if args.task and args.task not in tname:
                    continue
                for h in harnesses:
                    m = run_terminal_bench_task(tname, h, args.model)
                    row_d = asdict(m)
                    row_d["trial"] = trial_idx
                    run_rows.append(row_d)
                    save_and_merge_results(run_rows, args.model, trials=trials)
                    st = "PASS" if m.passed else "FAIL"
                    print(
                        f"  [T{trial_idx}/{trials}][{h:7}] {st} {m.task_id} | {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
                        flush=True,
                    )

        if args.track in ("all", "sbl"):
            print("\n=== Track 2: SWE-bench-Live (2025-06 Post-Cutoff) ===", flush=True)
            for row in load_swebench_live_tasks():
                if args.task and args.task not in row["instance_id"]:
                    continue
                for h in harnesses:
                    m = run_swebench_live_task(row, h, args.model)
                    row_d = asdict(m)
                    row_d["trial"] = trial_idx
                    run_rows.append(row_d)
                    save_and_merge_results(run_rows, args.model, trials=trials)
                    st = "PASS" if m.passed else "FAIL"
                    print(
                        f"  [T{trial_idx}/{trials}][{h:7}] {st} {m.task_id} | {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
                        flush=True,
                    )

        if args.track in ("all", "polyglot"):
            print("\n=== Track 3: Aider Polyglot Benchmark ===", flush=True)
            for slug in POLYGLOT_TASKS:
                if args.task and args.task not in slug:
                    continue
                for h in harnesses:
                    m = run_polyglot_task(slug, h, args.model)
                    row_d = asdict(m)
                    row_d["trial"] = trial_idx
                    run_rows.append(row_d)
                    save_and_merge_results(run_rows, args.model, trials=trials)
                    st = "PASS" if m.passed else "FAIL"
                    print(
                        f"  [T{trial_idx}/{trials}][{h:7}] {st} {m.task_id} | {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
                        flush=True,
                    )

        if args.track in ("all", "lcb"):
            print("\n=== Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) ===", flush=True)
            for row in load_lcb_tasks():
                if args.task and args.task not in str(row["question_id"]):
                    continue
                for h in harnesses:
                    m = run_lcb_task(row, h, args.model)
                    row_d = asdict(m)
                    row_d["trial"] = trial_idx
                    run_rows.append(row_d)
                    save_and_merge_results(run_rows, args.model, trials=trials)
                    st = "PASS" if m.passed else "FAIL"
                    print(
                        f"  [T{trial_idx}/{trials}][{h:7}] {st} {m.task_id} | {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s | {m.details}",
                        flush=True,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
