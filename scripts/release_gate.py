#!/usr/bin/env python3
"""Unified 3-Tier DevOps Release & Benchmark Gate for ActorII & Bilby (`scripts/release_gate.py`).

Enforces mandatory regression testing and live capability benchmarking across changes:
  1. `--tier fast` (~55s, 0 GPU):
       - Isolated full `pytest -q` suite
       - Deterministic `--mode scripted --specs all --strict` Self-Evolution check (5/5)
       - Deterministic `--fast` Red-Team Containment Audit (6/6)
       - Mandatory `> [!WARNING]` header audit on planning/design docs
  2. `--tier live` (~5-8m, local `:8080` GPU — MANDATORY after major `src/actorii/` changes):
       - Runs `--tier fast` first (fail-fast before touching GPU)
       - Runs live `:8080` Self-Evolution on all Passing Canaries (hard block on regression)
         PLUS targeted Failing Frontier spec(s) (tracks holdout progress & auto-ratchets into
         Canaries the moment a frontier spec passes!)
       - Writes cryptographic `src/actorii/` content stamp (`benchmarks/.gate_stamp.json`)
  3. `--tier full` (~25-35m, Periodic Extensive Suite for `actorii-periodic-bench.timer`):
       - Runs `--tier fast`
       - Runs ALL 5 Self-Evolution specs live on `:8080`
       - Runs Live Red-Team Adversarial Containment Audit (6 live prompt-injection scenarios)
       - Runs Stratified OpenAI HumanEval + GSM8K Matrix on `:8080`
       - Auto-records scores into `benchmarks/scoreboard.json` and regenerates
         `BENCHMARK_RESULTS.md`, `README.md`, and `scoreboard.html`
  4. `--verify-stamp` (<0.2s, called by `.git/hooks/pre-push` and `deploy-bilby`):
       - Classifies `git diff` against upstream (`none` / `minor` / `major`) and blocks
         unverified pushes or deployments unless `.gate_stamp.json` matches the exact
         SHA-256 tree hash of `src/actorii/` at the required gate tier.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "actorii"
BENCH_DIR = REPO_ROOT / "benchmarks"
BASELINE_PATH = BENCH_DIR / "gate_baseline.json"
STAMP_PATH = BENCH_DIR / ".gate_stamp.json"
HISTORY_PATH = BENCH_DIR / "GATE_HISTORY.jsonl"
SCOREBOARD_PY = BENCH_DIR / "scoreboard.py"

PLANNING_DOCS = [
    "MULTITHREADED_ACTOR_DESIGN.md",
    "SELF_EVOLUTION_BENCHMARK.md",
    "SWARM_ARCHITECTURE_REVIEW.md",
    "ARCHITECTURAL_BLUEPRINT.md",
]

TIER_RANK = {"fast": 1, "live": 2, "full": 3}


def compute_src_tree_hash(src_dir: Path = SRC_DIR) -> str:
    """Compute a deterministic SHA-256 digest over all source files in `src/actorii/`."""
    hasher = hashlib.sha256()
    if not src_dir.exists():
        return hasher.hexdigest()
    files = sorted(
        p
        for p in src_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".pyo"))
    )
    for fpath in files:
        rel = fpath.relative_to(src_dir).as_posix()
        hasher.update(rel.encode("utf-8") + b"\x00")
        hasher.update(fpath.read_bytes() + b"\x00")
    return hasher.hexdigest()


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(data: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    data["updated_at"] = date.today().isoformat()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def git_head_commit(repo_root: Path = REPO_ROOT) -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def classify_change_scope(
    repo_root: Path = REPO_ROOT,
    base_ref: str | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify changes between `base_ref` (default `@{u}` or `HEAD~1`) and working tree."""
    cfg = (baseline or load_baseline()).get("major_change_rules", {})
    core_prefixes: list[str] = cfg.get("core_paths", [])
    line_threshold: int = int(cfg.get("diff_line_threshold", 80))

    ref = base_ref
    if not ref:
        up = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        ref = up.stdout.strip() if up.returncode == 0 and up.stdout.strip() else "HEAD~1"

    names_proc = subprocess.run(
        ["git", "diff", "--name-only", ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    changed_files = [ln.strip() for ln in names_proc.stdout.splitlines() if ln.strip()]
    src_changed = [f for f in changed_files if f.startswith("src/actorii/")]

    numstat_proc = subprocess.run(
        ["git", "diff", "--numstat", ref, "--", "src/actorii/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    src_diff_lines = 0
    for ln in numstat_proc.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            try:
                src_diff_lines += int(parts[0]) + int(parts[1])
            except ValueError:
                pass

    core_touched = [
        f
        for f in src_changed
        if any(f == cp or f.startswith(cp.rstrip("/") + "/") for cp in core_prefixes)
    ]

    if core_touched or src_diff_lines >= line_threshold:
        scope = "major"
        required_tier = "live"
    elif src_changed:
        scope = "minor"
        required_tier = "fast"
    else:
        scope = "none"
        required_tier = "fast"

    return {
        "base_ref": ref,
        "scope": scope,
        "required_tier": required_tier,
        "changed_files_count": len(changed_files),
        "src_changed": src_changed,
        "core_touched": core_touched,
        "src_diff_lines": src_diff_lines,
    }


def evaluate_self_evo_against_baseline(
    evo_report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    ratchet: bool = True,
) -> dict[str, Any]:
    """Grade live Self-Evolution results against Canary (hard block) & Frontier (delta + ratchet) rules."""
    se_cfg = baseline.setdefault("self_evolution", {})
    canaries: list[str] = list(se_cfg.get("canaries", []))
    frontier: dict[str, dict[str, Any]] = dict(se_cfg.get("frontier", {}))

    raw_specs = evo_report.get("specs") or evo_report.get("results") or []
    by_id: dict[str, dict[str, Any]] = {
        str(r.get("spec_id")): r for r in raw_specs if isinstance(r, dict)
    }

    canary_failures: list[str] = []
    canary_passes: list[str] = []
    ratcheted_to_canary: list[str] = []
    frontier_warnings: list[str] = []
    frontier_summaries: dict[str, Any] = {}
    baseline_mutated = False

    for cid in canaries:
        if cid not in by_id:
            continue
        res = by_id[cid]
        if bool(res.get("passed")):
            canary_passes.append(cid)
        else:
            canary_failures.append(cid)

    for fid, res in by_id.items():
        if fid in canaries:
            continue
        holdout = float(res.get("holdout_pass_rate", 0.0) or 0.0)
        visible = float(res.get("visible_pass_rate", 0.0) or 0.0)
        passed = bool(res.get("passed"))
        persisted = bool(res.get("persistence_verified", True))
        prev = frontier.get(fid, {})
        prev_best = float(prev.get("best_holdout_pass_rate", 0.0) or 0.0)

        frontier_summaries[fid] = {
            "passed": passed,
            "holdout_pass_rate": round(holdout, 4),
            "prev_best_holdout": round(prev_best, 4),
            "delta_holdout": round(holdout - prev_best, 4),
        }

        if passed and persisted and ratchet:
            canaries.append(fid)
            frontier.pop(fid, None)
            ratcheted_to_canary.append(fid)
            baseline_mutated = True
        else:
            if holdout + 0.05 < prev_best:
                frontier_warnings.append(
                    f"{fid}: holdout dropped {prev_best:.1%} -> {holdout:.1%}"
                )
            if ratchet and holdout > prev_best:
                frontier[fid] = {
                    "best_holdout_pass_rate": round(holdout, 4),
                    "best_visible_pass_rate": round(visible, 4),
                    "last_status": str(res.get("status", "failed")),
                }
                baseline_mutated = True

    if baseline_mutated:
        se_cfg["canaries"] = canaries
        se_cfg["frontier"] = frontier
        cur_sample = [s for s in se_cfg.get("default_live_frontier_sample", []) if s in frontier]
        if not cur_sample and frontier:
            cur_sample = [next(iter(frontier.keys()))]
        se_cfg["default_live_frontier_sample"] = cur_sample

    return {
        "passed": len(canary_failures) == 0,
        "canary_passes": canary_passes,
        "canary_failures": canary_failures,
        "ratcheted_to_canary": ratcheted_to_canary,
        "frontier_warnings": frontier_warnings,
        "frontier_summaries": frontier_summaries,
        "baseline_mutated": baseline_mutated,
    }


def _purge_neo4j_namespace(namespace: str) -> None:
    q = {"statements": [{"statement": "MATCH (n) WHERE n.namespace = $ns DETACH DELETE n", "parameters": {"ns": namespace}}]}
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:7474/db/neo4j/tx/commit",
            data=json.dumps(q).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3):
            pass
    except Exception:
        pass


def check_planning_doc_headers(repo_root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for doc_name in PLANNING_DOCS:
        p = repo_root / doc_name
        if not p.is_file():
            continue
        head_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        if not head_lines or not head_lines[0].strip().startswith("> [!WARNING]"):
            errors.append(f"{doc_name}: missing mandatory `> [!WARNING]` header on line 1")
    return errors


def run_fast_tier(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    print("=== [Gate Tier 1: FAST] Doc Headers + Pytest + Scripted Self-Evo + Fast Containment ===", flush=True)
    t0 = time.time()
    doc_errors = check_planning_doc_headers(repo_root)
    if doc_errors:
        return {"passed": False, "stage": "doc_headers", "errors": doc_errors, "elapsed_s": round(time.time() - t0, 2)}

    tmp_state = Path(tempfile.mkdtemp(prefix="actorii_gate_state_"))
    ns = f"gate_pytest_{os.getpid()}"
    env = dict(os.environ)
    env["ACTORII_STATE_DIR"] = str(tmp_state)
    env["ACTORII_MEMORY_NAMESPACE"] = ns
    env["PYTHONPATH"] = str(repo_root / "src")

    try:
        p_res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        tail_pytest = "\n".join(p_res.stdout.strip().splitlines()[-4:])
        print(f"  [pytest] rc={p_res.returncode} | {tail_pytest}", flush=True)
        if p_res.returncode != 0:
            return {
                "passed": False,
                "stage": "pytest",
                "stdout_tail": p_res.stdout[-2500:],
                "stderr_tail": p_res.stderr[-1500:],
                "elapsed_s": round(time.time() - t0, 2),
            }

        se_res = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_self_evolution_benchmark.py"),
                "--mode",
                "scripted",
                "--specs",
                "all",
                "--strict",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"  [self-evo scripted] rc={se_res.returncode}", flush=True)
        if se_res.returncode != 0:
            return {
                "passed": False,
                "stage": "self_evo_scripted",
                "stdout_tail": se_res.stdout[-2000:],
                "stderr_tail": se_res.stderr[-1500:],
                "elapsed_s": round(time.time() - t0, 2),
            }

        rt_res = subprocess.run(
            [sys.executable, str(repo_root / "benchmarks" / "run_redteam_containment_audit.py"), "--fast"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"  [redteam --fast] rc={rt_res.returncode}", flush=True)
        if rt_res.returncode != 0:
            return {
                "passed": False,
                "stage": "redteam_fast",
                "stdout_tail": rt_res.stdout[-2000:],
                "stderr_tail": rt_res.stderr[-1500:],
                "elapsed_s": round(time.time() - t0, 2),
            }

        return {
            "passed": True,
            "pytest_summary": tail_pytest.splitlines()[-1] if tail_pytest else "passed",
            "scripted_self_evo": "5/5 PASS",
            "redteam_fast": "6/6 PASS",
            "elapsed_s": round(time.time() - t0, 2),
        }
    finally:
        shutil.rmtree(tmp_state, ignore_errors=True)
        _purge_neo4j_namespace(ns)


def _record_scoreboard_entry(bench_id: str, passed: int, total: int, model: str, notes: str) -> None:
    if not SCOREBOARD_PY.is_file():
        return
    sb_data = json.loads((BENCH_DIR / "scoreboard.json").read_text(encoding="utf-8"))
    known_ids = {b.get("id") for b in sb_data.get("benchmarks", [])}
    if bench_id not in known_ids and bench_id == "self-evolution-loop":
        subprocess.run(
            [
                sys.executable,
                str(SCOREBOARD_PY),
                "add-benchmark",
                "--id",
                "self-evolution-loop",
                "--name",
                "Bilby Deterministic Self-Evolution Loop (Tiers 1-3)",
                "--category",
                "SWE & Coding",
                "--status",
                "active",
                "--tasks",
                str(total),
                "--runner",
                "PYTHONPATH=src python3 scripts/run_self_evolution_benchmark.py --mode live --specs all",
                "--desc",
                "Multi-generation TDD self-evolution with hidden holdout anti-cheating & fresh-VM reload persistence gates",
                "--featured",
            ],
            cwd=REPO_ROOT,
            check=False,
        )
    subprocess.run(
        [
            sys.executable,
            str(SCOREBOARD_PY),
            "record",
            "--bench",
            bench_id,
            "--passed",
            str(passed),
            "--total",
            str(total),
            "--model",
            model,
            "--mode",
            "AgentHarness v4.2 (Gate Verified)",
            "--notes",
            notes,
        ],
        cwd=REPO_ROOT,
        check=False,
    )


def run_live_or_full_tier(
    tier: str,
    *,
    model: str = "qwen3.8:latest",
    frontier_specs_override: str | None = None,
    update_scoreboard: bool = False,
) -> dict[str, Any]:
    fast_report = run_fast_tier(REPO_ROOT)
    if not fast_report.get("passed"):
        return {"passed": False, "tier": tier, "fast": fast_report}

    baseline = load_baseline(BASELINE_PATH)
    se_cfg = baseline.get("self_evolution", {})
    canaries: list[str] = list(se_cfg.get("canaries", []))

    if tier == "full" or frontier_specs_override == "all":
        specs_to_run = "all"
    else:
        front_list = (
            [s.strip() for s in frontier_specs_override.split(",") if s.strip()]
            if frontier_specs_override is not None
            else list(se_cfg.get("default_live_frontier_sample", ["circuit_breaker_policy"]))
        )
        combined = list(dict.fromkeys([*canaries, *front_list]))
        specs_to_run = ",".join(combined)

    print(f"=== [Gate Tier: {tier.upper()}] Running Live `:8080` Self-Evolution ({specs_to_run}) ===", flush=True)
    t_live = time.time()
    with tempfile.TemporaryDirectory(prefix="actorii_live_gate_") as tmp_dir:
        out_json = Path(tmp_dir) / "self_evo_live.json"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_self_evolution_benchmark.py"),
            "--mode",
            "live",
            "--specs",
            specs_to_run,
            "--model",
            model,
            "--output-json",
            str(out_json),
        ]
        se_proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=False)
        if se_proc.returncode not in (0, 1) or not out_json.is_file():
            return {
                "passed": False,
                "tier": tier,
                "stage": "self_evo_live_exec",
                "returncode": se_proc.returncode,
            }
        evo_report = json.loads(out_json.read_text(encoding="utf-8"))

    eval_res = evaluate_self_evo_against_baseline(evo_report, baseline, ratchet=True)
    if eval_res["baseline_mutated"]:
        save_baseline(baseline, BASELINE_PATH)
        if eval_res["ratcheted_to_canary"]:
            print(f"  [RATCHET UP!] Promoted to mandatory Canaries: {eval_res['ratcheted_to_canary']}", flush=True)
    if eval_res["frontier_warnings"]:
        for w in eval_res["frontier_warnings"]:
            print(f"  [FRONTIER WARNING] {w}", flush=True)

    extra_suites: dict[str, Any] = {}
    overall_passed = bool(eval_res["passed"])

    if tier == "full":
        print("=== [Gate Tier: FULL] Running Live Red-Team Adversarial Containment Audit (6 scenarios) ===", flush=True)
        rt_proc = subprocess.run(
            [sys.executable, str(BENCH_DIR / "run_redteam_containment_audit.py"), "--model", model],
            cwd=REPO_ROOT,
            text=True,
            check=False,
        )
        rt_json_path = BENCH_DIR / "REDTEAM_CONTAINMENT_RESULTS.json"
        rt_data = json.loads(rt_json_path.read_text(encoding="utf-8")) if rt_json_path.is_file() else {}
        rt_ok = rt_proc.returncode == 0
        extra_suites["redteam_live"] = {
            "passed": rt_ok,
            "score": f"{rt_data.get('passed', 0)}/{rt_data.get('total', 6)}",
        }
        if not rt_ok:
            overall_passed = False

        std_cfg = baseline.get("standard_matrix", {})
        he_n = int(std_cfg.get("periodic_humaneval_n", 6))
        gsm_n = int(std_cfg.get("periodic_gsm8k_n", 4))
        print(f"=== [Gate Tier: FULL] Running Standard Matrix (HumanEval-{he_n} + GSM8K-{gsm_n}) ===", flush=True)
        std_proc = subprocess.run(
            [
                sys.executable,
                str(BENCH_DIR / "run_standard_benchmarks.py"),
                "--humaneval-n",
                str(he_n),
                "--gsm8k-n",
                str(gsm_n),
                "--model",
                model,
            ],
            cwd=REPO_ROOT,
            text=True,
            check=False,
        )
        std_json_path = BENCH_DIR / "BENCHMARK_RESULTS.json"
        if std_proc.returncode == 0 and std_json_path.is_file():
            sdata = json.loads(std_json_path.read_text(encoding="utf-8"))
            he_dict = sdata.get("humaneval", {})
            gsm_dict = sdata.get("gsm8k", {})
            he_tasks = he_dict.get("tasks", [])
            gsm_tasks = gsm_dict.get("tasks", [])
            he_pass = (
                sum(1 for t in he_tasks if t.get("actor_harness_passed"))
                if he_tasks
                else int(he_dict.get("actor_harness_passed", 0))
            )
            he_tot = int(he_dict.get("n", he_dict.get("total", len(he_tasks) or he_n)))
            gsm_pass = (
                sum(1 for t in gsm_tasks if t.get("actor_harness_passed"))
                if gsm_tasks
                else int(gsm_dict.get("actor_harness_passed", 0))
            )
            gsm_tot = int(gsm_dict.get("n", gsm_dict.get("total", len(gsm_tasks) or gsm_n)))
            comb_pct = 100.0 * (he_pass + gsm_pass) / max(1, he_tot + gsm_tot)
            min_pct = float(std_cfg.get("min_combined_pass_rate_pct", 70.0))
            std_ok = comb_pct >= min_pct
            extra_suites["standard_matrix"] = {
                "passed": std_ok,
                "combined_pct": round(comb_pct, 2),
                "humaneval": f"{he_pass}/{he_tot}",
                "gsm8k": f"{gsm_pass}/{gsm_tot}",
            }
            if not std_ok:
                overall_passed = False

    if (tier == "full" or update_scoreboard) and specs_to_run == "all":
        results_list = evo_report.get("specs") or evo_report.get("results") or []
        n_pass = sum(1 for r in results_list if r.get("passed"))
        n_tot = len(results_list)
        _record_scoreboard_entry(
            "self-evolution-loop",
            passed=n_pass,
            total=n_tot,
            model=model,
            notes=f"Canaries={eval_res['canary_passes']}; Frontier={list(eval_res['frontier_summaries'].keys())}",
        )

    return {
        "passed": overall_passed,
        "tier": tier,
        "fast": fast_report,
        "self_evolution": eval_res,
        "extra_suites": extra_suites,
        "live_elapsed_s": round(time.time() - t_live, 2),
    }


def write_stamp_and_history(tier: str, report: dict[str, Any]) -> dict[str, Any]:
    stamp = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "passed": bool(report.get("passed")),
        "head_commit": git_head_commit(REPO_ROOT),
        "src_tree_hash": compute_src_tree_hash(SRC_DIR),
        "report": report,
    }
    if stamp["passed"]:
        STAMP_PATH.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as hf:
        hf.write(json.dumps(stamp) + "\n")
    return stamp


def verify_stamp(base_ref: str | None = None, require_tier: str | None = None) -> tuple[bool, str]:
    scope_info = classify_change_scope(REPO_ROOT, base_ref=base_ref)
    needed_tier = require_tier or scope_info["required_tier"]
    cur_hash = compute_src_tree_hash(SRC_DIR)

    if not STAMP_PATH.is_file():
        disp_stamp = STAMP_PATH.relative_to(REPO_ROOT) if STAMP_PATH.is_relative_to(REPO_ROOT) else STAMP_PATH
        return (
            False,
            f"GATE BLOCKED: No verification stamp at {disp_stamp} "
            f"(change scope={scope_info['scope']!r}, required tier={needed_tier!r}).\n"
            f"Run: python3 scripts/release_gate.py --tier {needed_tier}",
        )

    stamp = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    if not stamp.get("passed"):
        return (False, f"GATE BLOCKED: Last gate stamp failed (`passed=False`). Re-run `--tier {needed_tier}`.")

    if stamp.get("src_tree_hash") != cur_hash:
        return (
            False,
            f"GATE BLOCKED: `src/actorii/` changed since last verified gate stamp!\n"
            f"  Scope={scope_info['scope']!r} | Core touched={scope_info['core_touched']} | Diff lines={scope_info['src_diff_lines']}\n"
            f"  Run: python3 scripts/release_gate.py --tier {needed_tier}",
        )

    stamp_tier = str(stamp.get("tier", "fast"))
    if TIER_RANK.get(stamp_tier, 0) < TIER_RANK.get(needed_tier, 1):
        return (
            False,
            f"GATE BLOCKED: Current stamp is only `--tier {stamp_tier}`, but change scope is "
            f"`{scope_info['scope']}` (requires `--tier {needed_tier}`).\n"
            f"  Core files touched: {scope_info['core_touched']} ({scope_info['src_diff_lines']} src diff lines)\n"
            f"  Run: python3 scripts/release_gate.py --tier {needed_tier}",
        )

    return (
        True,
        f"GATE VERIFIED: tier={stamp_tier} | scope={scope_info['scope']} | src_hash={cur_hash[:12]} | ts={stamp.get('timestamp')}",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="ActorII & Bilby Unified DevOps & Benchmark Release Gate")
    p.add_argument("--tier", choices=("fast", "live", "full"), default="fast")
    p.add_argument("--verify-stamp", action="store_true", help="Verify existing .gate_stamp.json matches src/actorii/ tree")
    p.add_argument("--require-tier", choices=("fast", "live", "full"), default=None, help="Minimum tier when --verify-stamp is used")
    p.add_argument("--base-ref", default=None, help="Git ref to diff against for change-scope classification")
    p.add_argument("--model", default="qwen3.8:latest", help="Local LLM model on :8080 for live/full tiers")
    p.add_argument(
        "--frontier-specs",
        default=None,
        help="Comma-separated frontier spec IDs to include in --tier live (default from gate_baseline.json, or '' for canaries only)",
    )
    p.add_argument("--update-scoreboard", action="store_true", help="Record full 5-spec self-evo run into benchmarks/scoreboard.json")
    args = p.parse_args()

    if args.verify_stamp:
        ok, msg = verify_stamp(base_ref=args.base_ref, require_tier=args.require_tier)
        print(msg, flush=True)
        return 0 if ok else 1

    if args.tier == "fast":
        rep = run_fast_tier(REPO_ROOT)
    else:
        rep = run_live_or_full_tier(
            args.tier,
            model=args.model,
            frontier_specs_override=args.frontier_specs,
            update_scoreboard=args.update_scoreboard,
        )

    stamp = write_stamp_and_history(args.tier, rep)
    badge = "PASS" if stamp["passed"] else "FAIL"
    print(f"\n=== RELEASE GATE [{args.tier.upper()}]: {badge} | src_hash={stamp['src_tree_hash'][:12]} ===", flush=True)
    if not stamp["passed"]:
        print(json.dumps(rep, indent=2), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
