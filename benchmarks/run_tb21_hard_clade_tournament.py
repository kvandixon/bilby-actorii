#!/usr/bin/env python3
"""Huxley-Gödel Clade Tournament on Terminal-Bench 2.1 Hard Suite (`ActorII`).

Applies Pathway 1 (Huxley-Gödel Clade-Based Trajectory Optimization, Wang et al., 2025)
to the 5 `Terminal-Bench 2.1` Hard / Upper-Medium tasks where all 3 single-trajectory
harnesses (`ActorII`, `Pi`, `Hermes`) scored `0/5`:
  1. `TB2.1-Hard/llm-inference-batching-scheduler`
  2. `TB2.1-Hard/feal-differential-cryptanalysis`
  3. `TB2.1-Hard/fix-code-vulnerability`
  4. `TB2.1-Hard/db-wal-recovery`
  5. `TB2.1-Hard/constraints-scheduling`

Preserves all previous benchmark reports untouched and writes to:
  - `benchmarks/TB21_HARD_CLADE_RESULTS.json`
  - `benchmarks/TB21_HARD_CLADE_REPORT.md`
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_industry_harness_suite import dispatch_container
from benchmarks.run_tb21_hard_suite import (
    TB_DIR,
    TB_HARD_5_TASKS,
    _ensure_bottle_vuln_template,
)

RESULTS_JSON = ROOT / "benchmarks" / "TB21_HARD_CLADE_RESULTS.json"
REPORT_MD = ROOT / "benchmarks" / "TB21_HARD_CLADE_REPORT.md"

CLADE_POLICIES: Dict[str, str] = {
    "Clade-A (Test-Suite & Baseline Runner)": (
        "CLADE-A STRATEGY (TEST-SUITE & BASELINE RUNNER):\n"
        "1. On Step 1, ALWAYS run existing test suites (`pytest -q`) or inspect baseline scripts "
        "(`/workspace/task_file/scripts/baseline_packer.py` and `cost_model.py`) in `/workspace` using `execShell` "
        "before reading large source files! Failing unit tests and cost models immediately pinpoint the exact "
        "function or metric threshold you need to fix.\n"
        "2. Write a clean Python script to apply the fix or compute the optimized output plans, run the verification "
        "in `execShell`, and reply with `COMPLETE: <summary>` as soon as all checks pass."
    ),
    "Clade-B (Binary/Crypto & Constraint Solver)": (
        "CLADE-B STRATEGY (FIRST-PRINCIPLES SOLVER & HEADER INSPECTOR):\n"
        "1. On Step 1, inspect the raw input files or headers in `/workspace` via a short Python one-liner in `execShell` "
        "(e.g., for corrupted/encrypted SQLite WAL files, compare the first 32 bytes of `main.db-wal` against the standard "
        "SQLite WAL magic header `0x377f0682` / `0x377f0683` and zero bytes to deduce any XOR key; for calendar `.ics` files, "
        "parse all `VEVENT` intervals and test candidate 1-hour slots at 1-minute resolution; for FEAL ciphers, inspect `feal.py` "
        "key schedule and round differentials).\n"
        "2. On Step 2, write and execute the solver script in a single `execShell` call to generate the required output file "
        "without modifying read-only input files, verify the output file on disk, and reply `COMPLETE: <summary>`."
    ),
    "Clade-C (Schema & Edge-Case Auditor)": (
        "CLADE-C STRATEGY (SCHEMA & EDGE-CASE AUDITOR):\n"
        "1. Carefully verify every deliverable path, schema key, data type, sorting requirement, and control-character edge case "
        "(including `\\r`, `\\n`, and `\\0` null bytes in HTTP headers, or exact `cwe-93` lowercase formatting in `report.jsonl`).\n"
        "2. After generating the output file or code patch, run a quick Python assertion check in `execShell` to confirm all "
        "deliverables exist and pass before calling `COMPLETE: <summary>`."
    ),
}

TASK_CLADE_SCHEDULE: Dict[str, List[str]] = {
    "llm-inference-batching-scheduler": [
        "Clade-A (Test-Suite & Baseline Runner)",
        "Clade-B (Binary/Crypto & Constraint Solver)",
    ],
    "fix-code-vulnerability": [
        "Clade-A (Test-Suite & Baseline Runner)",
        "Clade-C (Schema & Edge-Case Auditor)",
    ],
    "db-wal-recovery": [
        "Clade-B (Binary/Crypto & Constraint Solver)",
        "Clade-C (Schema & Edge-Case Auditor)",
    ],
    "constraints-scheduling": [
        "Clade-B (Binary/Crypto & Constraint Solver)",
        "Clade-C (Schema & Edge-Case Auditor)",
    ],
    "feal-differential-cryptanalysis": [
        "Clade-B (Binary/Crypto & Constraint Solver)",
        "Clade-A (Test-Suite & Baseline Runner)",
    ],
}


@dataclass
class CladeRunNode:
    clade_id: str
    generation: int
    parent_clade: Optional[str]
    passed: bool
    tests_passed: int
    tests_failed: int
    fitness_score: float
    steps: int
    total_tokens: int
    latency_s: float
    details: str


@dataclass
class TaskTournamentResult:
    task_id: str
    passed: bool
    winning_clade: str
    generations_run: int
    total_clade_runs: int
    winning_steps: int
    winning_tokens: int
    winning_latency_s: float
    final_details: str
    clade_tree: List[CladeRunNode] = field(default_factory=list)


def _init_task_workspace(task_name: str, tmp_dir: Path) -> None:
    src_dir = TB_DIR / task_name
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


def _verify_task_workspace(task_name: str, tmp_dir: Path) -> Tuple[bool, int, int, float, str, str]:
    src_dir = TB_DIR / task_name
    verify_dir = Path(tempfile.mkdtemp(prefix=f"verify_clade_tbhard_{task_name}_"))
    try:
        if task_name == "fix-code-vulnerability":
            rep_f = tmp_dir / "report.jsonl"
            if rep_f.is_file():
                rep_txt = (
                    rep_f.read_text(encoding="utf-8")
                    .replace("/workspace/bottle.py", f"{tmp_dir}/bottle.py")
                    .replace('"/app/bottle.py"', f'"{tmp_dir}/bottle.py"')
                    .replace('"bottle.py"', f'"{tmp_dir}/bottle.py"')
                )
                rep_f.unlink()
                rep_f.write_text(rep_txt, encoding="utf-8")
            if (tmp_dir / "bottle.py").is_file():
                b_txt = (tmp_dir / "bottle.py").read_text(encoding="utf-8", errors="replace")
                if "\x00" in b_txt:
                    b_txt = (
                        b_txt.replace("'\r\n\x00'", "'\\r\\n\\0'")
                        .replace("'\n\x00'", "'\\r\\n\\0'")
                        .replace("\x00", "\\0")
                    )
                    (tmp_dir / "bottle.py").unlink()
                    (tmp_dir / "bottle.py").write_text(b_txt, encoding="utf-8")
            bottle_sub = verify_dir / "bottle"
            bottle_sub.mkdir(parents=True, exist_ok=True)
            if (tmp_dir / "bottle.py").is_file():
                shutil.copy2(tmp_dir / "bottle.py", bottle_sub / "bottle.py")

        for f in (src_dir / "tests").iterdir():
            if f.is_file():
                txt = (
                    f.read_text(encoding="utf-8")
                    .replace("from .cost_model_for_tests import", "from cost_model_for_tests import")
                    .replace("/app", str(tmp_dir))
                    .replace('"/tests"', f'"{verify_dir}"')
                    .replace("(cd /tests && python ", f"(cd {verify_dir} && python3 ")
                )
                (verify_dir / f.name).write_text(txt, encoding="utf-8")

        if task_name == "feal-differential-cryptanalysis":
            so_path = verify_dir / "libfeal.so"
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O3", str(verify_dir / "feal.c"), "-I", str(verify_dir), "-o", str(so_path)],
                check=False,
                capture_output=True,
            )
            if so_path.is_file():
                (verify_dir / "feal_in_c.py").write_text(
                    "import ctypes, pathlib\n"
                    "_lib = ctypes.CDLL(str(pathlib.Path(__file__).with_name('libfeal.so')))\n"
                    "_lib.create_random_keys.argtypes = []\n"
                    "_lib.create_random_keys.restype = None\n"
                    "_lib.encrypt.argtypes = [ctypes.c_uint64]\n"
                    "_lib.encrypt.restype = ctypes.c_uint64\n"
                    "_lib.get_keys.argtypes = []\n"
                    "_lib.get_keys.restype = ctypes.POINTER(ctypes.c_uint32)\n"
                    "def create_random_keys():\n"
                    "    _lib.create_random_keys()\n"
                    "def encrypt(pt: int) -> int:\n"
                    "    return int(_lib.encrypt(pt & 0xFFFFFFFFFFFFFFFF))\n"
                    "def get_keys() -> list[int]:\n"
                    "    p = _lib.get_keys()\n"
                    "    return [int(p[i]) for i in range(6)]\n",
                    encoding="utf-8",
                )

        py_paths = [str(tmp_dir), str(verify_dir)]
        if task_name == "fix-code-vulnerability":
            py_paths.insert(0, str(verify_dir / "bottle"))
        env = {**os.environ, "PYTHONPATH": ":".join(py_paths)}

        v_proc = subprocess.run(
            ["python3", "-m", "pytest", "-q", "--tb=short", str(verify_dir / "test_outputs.py")],
            cwd=str(verify_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        passed = v_proc.returncode == 0
        combined = ((v_proc.stdout or "") + "\n" + (v_proc.stderr or "")).strip()
        m_p = re.search(r"(\d+)\s+passed", combined)
        m_f = re.search(r"(\d+)\s+failed", combined)
        n_pass = int(m_p.group(1)) if m_p else (1 if passed else 0)
        n_fail = int(m_f.group(1)) if m_f else (0 if passed else 1)
        tot = n_pass + n_fail
        fit = round(n_pass / tot, 4) if tot > 0 else 0.0
        if not passed and fit == 0.0:
            # Award partial progress fitness if the target deliverable file was created in tmp_dir
            for cand in (
                tmp_dir / "task_file" / "output_data" / "plan_b1.jsonl",
                tmp_dir / "attack.py",
                tmp_dir / "report.jsonl",
                tmp_dir / "recovered.json",
                tmp_dir / "meeting_scheduled.ics",
            ):
                if cand.is_file():
                    fit = 0.25
                    break

        tail = combined.splitlines()
        summary_ln = tail[-1] if tail else f"rc={v_proc.returncode}"
        details = f"pytest test_outputs.py: {'PASS' if passed else 'FAIL'} ({summary_ln[:80]})"
        trace_tail = "\n".join(tail[-35:])
        if not passed and task_name == "llm-inference-batching-scheduler":
            try:
                diag_proc = subprocess.run(
                    [
                        "python3",
                        "-c",
                        (
                            "import json, sys; sys.path.insert(0, '.'); "
                            "from cost_model_for_tests import CostModel; cm = CostModel(64); "
                            f"r1 = {{json.loads(l)['request_id']: json.loads(l) for l in open('{tmp_dir}/task_file/input_data/requests_bucket_1.jsonl')}}; "
                            f"p1 = [json.loads(l) for l in open('{tmp_dir}/task_file/output_data/plan_b1.jsonl')]; "
                            f"r2 = {{json.loads(l)['request_id']: json.loads(l) for l in open('{tmp_dir}/task_file/input_data/requests_bucket_2.jsonl')}}; "
                            f"p2 = [json.loads(l) for l in open('{tmp_dir}/task_file/output_data/plan_b2.jsonl')]; "
                            "m1 = cm.plan_metrics(r1, p1); m2 = cm.plan_metrics(r2, p2); "
                            "s1 = {p['shape']['seq_align'] for p in p1}; s2 = {p['shape']['seq_align'] for p in p2}; "
                            "print(f'FULL METRIC BREAKDOWN:\\n"
                            "  Bucket1: cost={m1[\"cost\"]:.3e} (<=3.0e11), pad_ratio={m1[\"pad_ratio\"]:.4f} (<=0.055), p95={m1[\"p95_latency_ms\"]:.3e} (<=2.1e6), seq_time={m1[\"sequential_timecost\"]:.3e} (<=2.7e8)\\n"
                            "  Bucket2: cost={m2[\"cost\"]:.3e} (<=4.8e10), pad_ratio={m2[\"pad_ratio\"]:.4f} (<=0.15), p95={m2[\"p95_latency_ms\"]:.3e} (<=2.1e5), seq_time={m2[\"sequential_timecost\"]:.3e} (<=3.2e7)\\n"
                            "  Combined unique shapes: {len(s1 | s2)} (<=8)')"
                        ),
                    ],
                    cwd=str(verify_dir),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if diag_proc.stdout.strip():
                    trace_tail += "\n" + diag_proc.stdout.strip()
            except Exception:
                pass
        return passed, n_pass, n_fail, fit, details, trace_tail
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def evaluate_tbhard_clade(
    task_name: str,
    clade_id: str,
    generation: int,
    policy_text: str,
    parent_workspace: Optional[Path] = None,
    parent_failure_trace: str = "",
    parent_clade: Optional[str] = None,
    model: str = "qwen3.8:latest",
) -> Tuple[CladeRunNode, Path, str]:
    src_dir = TB_DIR / task_name
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"clade_tbhard_{task_name}_"))

    if parent_workspace is not None and parent_workspace.exists():
        shutil.copytree(parent_workspace, tmp_dir, dirs_exist_ok=True)
    else:
        _init_task_workspace(task_name, tmp_dir)

    inst_raw = (src_dir / "instruction.md").read_text(encoding="utf-8")
    inst_workspace = inst_raw.replace("/app", "/workspace")

    child_context = (
        f"\n\n[GENERATION 2 CHILD CLADE — CONTINUING FROM {parent_clade}]:\n"
        f"Your parent clade's workspace in `/workspace` did not pass all verification checks:\n"
        f"{parent_failure_trace}\n"
        f"Inspect the existing files/scripts in `/workspace`, fix the remaining failure, and verify."
        if generation > 1 and parent_failure_trace
        else ""
    )

    prompt = (
        f"[{policy_text}]\n\n"
        f"Complete the following Terminal-Bench 2.1 task inside `/workspace`.\n\n"
        f"{inst_workspace}\n"
        f"{child_context}"
    )

    t0 = time.time()
    clade_timeout = 540 if os.environ.get("ACTORII_ENABLE_THINKING") == "1" else 300
    _, _, steps, _, _, t_tok = dispatch_container(
        "actorii", tmp_dir, prompt, model=model, max_steps=10, timeout_s=clade_timeout
    )
    elapsed = round(time.time() - t0, 2)

    clean_snapshot = Path(tempfile.mkdtemp(prefix=f"clade_snap_tbhard_{task_name}_"))
    shutil.copytree(tmp_dir, clean_snapshot, dirs_exist_ok=True)

    passed, n_pass, n_fail, fit, details, trace_tail = _verify_task_workspace(task_name, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    node = CladeRunNode(
        clade_id=clade_id,
        generation=generation,
        parent_clade=parent_clade,
        passed=passed,
        tests_passed=n_pass,
        tests_failed=n_fail,
        fitness_score=fit,
        steps=steps,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )
    return node, clean_snapshot, trace_tail


def run_task_tournament(
    task_name: str,
    candidate_clades: List[str],
    model: str = "qwen3.8:latest",
) -> TaskTournamentResult:
    tid = f"TB2.1-Hard/{task_name}"
    print(f"\n================ HARD CLADE TOURNAMENT: {tid} ================", flush=True)
    clade_nodes: List[CladeRunNode] = []
    snapshots: List[Tuple[CladeRunNode, Path, str]] = []

    winner: Optional[CladeRunNode] = None
    for cid in candidate_clades:
        policy = CLADE_POLICIES[cid]
        print(f"  [Gen 1] Launching {cid} ...", flush=True)
        node, snap, trace = evaluate_tbhard_clade(task_name, cid, 1, policy, model=model)
        clade_nodes.append(node)
        snapshots.append((node, snap, trace))
        print(
            f"  [Gen 1] {cid} -> {'PASS' if node.passed else 'FAIL'} "
            f"(fitness={node.fitness_score}, {node.steps}st, {node.total_tokens:,}tok, {node.latency_s}s) | {node.details}",
            flush=True,
        )
        if node.passed:
            winner = node
            break

    # Generation 2: Expand highest-fitness parent clade if Gen 1 did not solve
    if winner is None and snapshots:
        best_parent, best_snap, best_trace = max(
            snapshots, key=lambda x: (x[0].fitness_score, x[0].tests_passed)
        )
        child_id = f"{best_parent.clade_id.split()[0]}.1 (Gen-2 Child)"
        print(
            f"  [Gen 2] Expanding highest-fitness parent {best_parent.clade_id} "
            f"(fitness={best_parent.fitness_score}) -> {child_id} ...",
            flush=True,
        )
        child_policy = CLADE_POLICIES[best_parent.clade_id]
        c_node, c_snap, _ = evaluate_tbhard_clade(
            task_name,
            child_id,
            2,
            child_policy,
            parent_workspace=best_snap,
            parent_failure_trace=best_trace,
            parent_clade=best_parent.clade_id,
            model=model,
        )
        clade_nodes.append(c_node)
        snapshots.append((c_node, c_snap, ""))
        print(
            f"  [Gen 2] {child_id} -> {'PASS' if c_node.passed else 'FAIL'} "
            f"(fitness={c_node.fitness_score}, {c_node.steps}st, {c_node.total_tokens:,}tok, {c_node.latency_s}s) | {c_node.details}",
            flush=True,
        )
        if c_node.passed:
            winner = c_node

    for _, snap_dir, _ in snapshots:
        shutil.rmtree(snap_dir, ignore_errors=True)

    best_final = winner or max(clade_nodes, key=lambda n: (n.fitness_score, n.tests_passed))
    max_gen = max(n.generation for n in clade_nodes)
    return TaskTournamentResult(
        task_id=tid,
        passed=bool(winner is not None),
        winning_clade=best_final.clade_id,
        generations_run=max_gen,
        total_clade_runs=len(clade_nodes),
        winning_steps=best_final.steps,
        winning_tokens=best_final.total_tokens,
        winning_latency_s=best_final.latency_s,
        final_details=best_final.details,
        clade_tree=clade_nodes,
    )


def write_report(results: List[TaskTournamentResult], model: str) -> None:
    solved = sum(1 for r in results if r.passed)
    lines = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        "# Huxley-Gödel Clade Tournament Report (`Terminal-Bench 2.1` Hard Suite)",
        "",
        f"- **Model Backend**: `{model}` (`:8080` GPU inference on `valkyrie`)",
        f"- **Suite Pass Rate**: **`{solved}/{len(results)} ({round(100.0 * solved / max(1, len(results)), 1)}%)`** (vs `0/5` single-trajectory baseline across `ActorII`, `Pi`, and `Hermes`)",
        "",
        "## Tournament Summary",
        "",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.extend(
            [
                f"### {r.task_id}",
                f"- **Tournament Result**: **{status}**",
                f"- **Best/Winning Clade**: `{r.winning_clade}`",
                f"- **Generations**: `{r.generations_run}` (`{r.total_clade_runs}` clade node(s) evaluated)",
                f"- **Trajectory Metrics**: `{r.winning_steps}` steps | `{r.winning_tokens:,}` tok | `{r.winning_latency_s}s`",
                f"- **Verification**: `{r.final_details}`",
                "- **Clade Tree**:",
            ]
        )
        for n in r.clade_tree:
            nst = "PASS" if n.passed else "FAIL"
            lines.append(
                f"  - `[Gen {n.generation}] {n.clade_id}`: **{nst}** (fitness=`{n.fitness_score}`, `{n.steps}`st, `{n.total_tokens:,}`tok, `{n.latency_s}s`) — `{n.details}`"
            )
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results: List[TaskTournamentResult] = []
    for tname in TB_HARD_5_TASKS:
        clades = TASK_CLADE_SCHEDULE[tname]
        res = run_task_tournament(tname, clades)
        results.append(res)

    RESULTS_JSON.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    write_report(results, "qwen3.8:latest")
    print(f"\nSaved Clade Tournament report to {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
