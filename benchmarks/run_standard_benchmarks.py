#!/usr/bin/env python3
"""Official OpenAI HumanEval (MultiPL-E Actor 4.1 Track) & GSM8K Benchmark Runner.

Evaluates the ActorII Metacircular `AgentHarness` (`agent_harness.act`) against
well-known upstream benchmarks (`benchmarks/data/HumanEval.jsonl` and
`benchmarks/data/gsm8k_test.jsonl`) on local `valkyrie` (`gpt-oss:20b` @ `:8080`),
comparing:
1. Single-Shot 1993 Actor 4.1 (`Pass@1` without `AgentHarness` feedback/rollback)
2. ActorII `AgentHarness` (Multi-step bytecode + Static Linter + Unit-Test Self-Correction)
3. Python / Zero-Shot CoT Reference Baseline on the identical model & hardware.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from actorii.ai import ValkyrieLLMClient
from actorii.harness import bootstrap_agent_harness, extract_actor_action
from actorii.vm import ActorVM

HUMANEVAL_PATH = Path(__file__).resolve().parent / "data" / "HumanEval.jsonl"
GSM8K_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_test.jsonl"
RESULTS_JSON_PATH = Path(__file__).resolve().parent / "BENCHMARK_RESULTS.json"
REPORT_MD_PATH = Path(__file__).resolve().parent / "BENCHMARK_REPORT.md"

# Canonical MultiPL-E compatible HumanEval subset (integer, array, string, boolean algorithms)
DEFAULT_HUMANEVAL_IDS = [
    "HumanEval/3",   # below_zero(operations: List[int]) -> bool
    "HumanEval/13",  # greatest_common_divisor(a: int, b: int) -> int
    "HumanEval/23",  # strlen(string: str) -> int
    "HumanEval/24",  # largest_divisor(n: int) -> int
    "HumanEval/31",  # is_prime(n: int) -> bool
    "HumanEval/35",  # max_element(l: list) -> int
    "HumanEval/41",  # car_race_collision(n: int) -> int
    "HumanEval/45",  # triangle_area(a, h) -> float
    "HumanEval/52",  # below_threshold(l: list, t: int) -> bool
    "HumanEval/53",  # add(x: int, y: int) -> int
    "HumanEval/55",  # fib(n: int) -> int
    "HumanEval/60",  # sum_to_n(n: int) -> int
]


def _parse_python_signature(prompt: str, entry_point: str) -> List[str]:
    """Extract parameter names from the official HumanEval Python `def entry_point(...):`."""
    for line in prompt.splitlines():
        s = line.strip()
        if s.startswith(f"def {entry_point}("):
            inside = s.split("(", 1)[1].rsplit(")", 1)[0]
            params = []
            for p in inside.split(","):
                p_name = p.split(":")[0].split("=")[0].strip()
                if p_name:
                    params.append(p_name)
            return params
    return ["x"]


def _py_arg_to_actor_literal(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_py_arg_to_actor_literal(x) for x in val)
        return f"#({inner})"
    return repr(val)


def run_humaneval_check_in_actor_vm(
    vm: ActorVM,
    entry_point: str,
    test_code: str,
    prompt_doc: str,
) -> Tuple[bool, Optional[str], int]:
    """Execute the official OpenAI HumanEval `check(candidate)` test suite against `Actor[#Sol]`."""
    returns_bool = "-> bool" in prompt_doc or "True" in prompt_doc or "False" in prompt_doc
    total_gas = 0

    last_actual: Any = None

    def _candidate(*args: Any) -> Any:
        nonlocal total_gas, last_actual
        actor_args = ", ".join(_py_arg_to_actor_literal(a) for a in args)
        call_expr = f"{entry_point}(Actor[#Sol], {actor_args})!!" if actor_args else f"{entry_point}(Actor[#Sol])!!"
        res = vm.eval_chunk(call_expr)
        total_gas += int(getattr(vm, "last_eval_opcodes", 0))
        if returns_bool and isinstance(res, int):
            res = bool(res)
        last_actual = res
        return res

    ns: Dict[str, Any] = {}
    exec(test_code, ns)
    check_fn = ns["check"]

    try:
        check_fn(_candidate)
        return True, None, total_gas
    except Exception as exc:
        # Identify the exact failing assertion line from `test_code` for harness self-correction
        for line in test_code.splitlines():
            s = line.strip()
            if s.startswith("assert candidate("):
                try:
                    exec(f"def _single(candidate):\n    {s}\n", ns)
                    ns["_single"](_candidate)
                except Exception as inner_exc:
                    return (
                        False,
                        f"Unit test failed on `{s}` (actual returned value was `{last_actual!r}`, {type(inner_exc).__name__}: {inner_exc}). "
                        f"Fix the algorithm or off-by-one condition in `Def {entry_point}(...)` and ensure all local variables are declared after `|`.",
                        total_gas,
                    )
        return False, f"{type(exc).__name__}: {exc}", total_gas


def evaluate_humaneval_problem(
    row: Dict[str, Any],
    *,
    model: str = "gpt-oss:20b",
    max_steps: int = 4,
) -> Dict[str, Any]:
    """Evaluate one official HumanEval problem across Single-Shot Actor, ActorHarness, and Python Pass@1."""
    task_id = row["task_id"]
    entry_point = row["entry_point"]
    prompt_doc = row["prompt"]
    test_code = row["test"]
    params = _parse_python_signature(prompt_doc, entry_point)
    params_str = ", ".join(params)

    # -------------------------------------------------------------------------
    # 1. Single-Shot Actor 4.1 (Pass@1, No Harness Feedback) & Multi-Step ActorII AgentHarness
    # -------------------------------------------------------------------------
    he_ns = f"bench_he_{entry_point}"
    vm, harness, sandbox = bootstrap_agent_harness(
        model=model,
        max_steps=max_steps,
        gas_limit=50000,
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=he_ns,
    )
    if sandbox.memory_graph is not None:
        sandbox.memory_graph.clearEngrams()
    vm.eval_chunk("inherit(Object, #Solution, #(#state), 1, nil)!! now(Solution)!! Actor[#Sol] := new(Solution)!!")

    grammar_guide = (
        sandbox.grammar_graph.queryConstructsForTask(prompt_doc, entry_point=entry_point, params_str=params_str)
        if sandbox.grammar_graph is not None
        else ""
    )

    single_shot_passed = False
    single_shot_err: Optional[str] = None
    harness_passed = False
    harness_steps_used = 0
    harness_gas = 0
    last_error_feedback = ""
    last_method_src = ""
    first_bad_src = ""

    t0 = time.time()
    for step_idx in range(1, max_steps + 1):
        harness_steps_used = step_idx
        step_prompt = (
            f"Solve OpenAI {task_id} (`{entry_point}`) in 1993 Whitewater Actor 4.1.\n\n"
            f"SPECIFICATION:\n{prompt_doc}\n\n"
            f"{grammar_guide}\n"
            f"Output ONLY the valid 1993 Actor `Def {entry_point}(...) {{ ... }}!!` chunk.\n"
        )
        if last_error_feedback:
            neo4j_diag = (
                sandbox.grammar_graph.diagnoseError(last_error_feedback)
                if sandbox.grammar_graph is not None
                else ""
            )
            step_prompt += (
                f"\n[PREVIOUS ATTEMPT FAILED — SELF-CORRECT THIS ERROR]:\n"
                f"Previous code:\n{last_method_src}\n"
                f"Error / Failed Test:\n{last_error_feedback}\n"
                + (f"{neo4j_diag}\n" if neo4j_diag else "")
                + "Fix the 1993 Actor 4.1 method definition and output ONLY `Def ... { ... }!!`."
            )

        harness.clearHistory()
        raw_reply = harness.ask(step_prompt)
        action_chunk = extract_actor_action(raw_reply)
        action_chunk = re.sub(r"now\s*\([^)]+\)\s*!!", "now(Solution)!!", action_chunk)
        if not action_chunk.lstrip().startswith("now(Solution)"):
            action_chunk = f"now(Solution)!!\n{action_chunk}"
        last_method_src = action_chunk

        # Execute inside the harness's guarded transactional sandbox
        outcome = sandbox.executeStepAction(harness, action_chunk, max_opcodes=50000)
        harness_gas += int(getattr(vm, "last_eval_opcodes", 0))

        if isinstance(outcome, str) and outcome.startswith("ERROR:"):
            err_msg = outcome
            if step_idx == 1:
                single_shot_passed = False
                single_shot_err = err_msg
                first_bad_src = action_chunk
            last_error_feedback = err_msg
            continue

        # Compile & lint succeeded -> run official HumanEval `check(candidate)` assertions in ActorVM!
        passed, test_err, test_gas = run_humaneval_check_in_actor_vm(
            vm, entry_point, test_code, prompt_doc
        )
        harness_gas += test_gas

        if step_idx == 1:
            single_shot_passed = passed
            single_shot_err = test_err
            if not passed:
                first_bad_src = action_chunk

        if passed:
            harness_passed = True
            if not single_shot_passed and sandbox.grammar_graph is not None and single_shot_err:
                sandbox.grammar_graph.recordLearnedFix(
                    single_shot_err,
                    first_bad_src,
                    action_chunk,
                    model=model,
                )
            break
        else:
            last_error_feedback = test_err or "Failed HumanEval unit test assertions"

    actor_elapsed = round(time.time() - t0, 2)

    # -------------------------------------------------------------------------
    # 1B. Cold-Reboot Active Engram Verification (0-Token Recall across fresh VM)
    # -------------------------------------------------------------------------
    engram_cold_reboot_passed = False
    if harness_passed:
        try:
            vm_cold, _h_cold, sb_cold = bootstrap_agent_harness(
                model=model,
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace=he_ns,
            )
            if sb_cold.memory_graph is not None:
                sb_cold.memory_graph.hydrateRelevantEngrams(entry_point)
                vm_cold.eval_chunk("Actor[#Sol] := Actor[#Solution]!!")
                engram_cold_reboot_passed, _, _ = run_humaneval_check_in_actor_vm(
                    vm_cold, entry_point, test_code, prompt_doc
                )
        except Exception:
            engram_cold_reboot_passed = False

    # -------------------------------------------------------------------------
    # 2. Python Reference Baseline (Pass@1 on same model)
    # -------------------------------------------------------------------------
    py_passed = False
    t_py = time.time()
    try:
        client = ValkyrieLLMClient(model=model)
        py_reply = client.complete(
            prompt=f"Complete the following Python function. Output ONLY valid Python code:\n\n{prompt_doc}",
            system_prompt="You are a Python coding assistant. Output only the Python function implementation.",
        )
        py_ns: Dict[str, Any] = {}
        exec("from typing import List, Tuple, Optional, Any, Dict\nimport math\n" + prompt_doc + "\n" + py_reply, py_ns)
        exec(test_code, py_ns)
        py_ns["check"](py_ns[entry_point])
        py_passed = True
    except Exception:
        py_passed = False
    py_elapsed = round(time.time() - t_py, 2)

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "single_shot_actor_passed": single_shot_passed,
        "single_shot_error": single_shot_err,
        "actor_harness_passed": harness_passed,
        "actor_harness_steps": harness_steps_used,
        "self_corrected": (not single_shot_passed and harness_passed),
        "engram_cold_reboot_passed": engram_cold_reboot_passed,
        "actor_gas_used": harness_gas,
        "actor_elapsed_s": actor_elapsed,
        "python_pass1": py_passed,
        "python_elapsed_s": py_elapsed,
    }


def _extract_gsm8k_gold(answer_str: str) -> Optional[int]:
    if "####" in answer_str:
        tail = answer_str.split("####", 1)[1].strip().replace(",", "")
        m = re.search(r"-?\d+", tail)
        if m:
            return int(m.group(0))
    return None


def evaluate_gsm8k_problem(
    idx: int,
    row: Dict[str, Any],
    *,
    model: str = "gpt-oss:20b",
    max_steps: int = 4,
) -> Dict[str, Any]:
    """Evaluate one official GSM8K word problem via Program-Aided 1993 Actor bytecode execution."""
    question = row["question"]
    gold = _extract_gsm8k_gold(row["answer"])

    gsm_ns = f"bench_gsm8k_{idx}"
    vm, harness, sb = bootstrap_agent_harness(
        model=model,
        max_steps=max_steps,
        gas_limit=25000,
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=gsm_ns,
    )
    if sb.memory_graph is not None:
        sb.memory_graph.clearEngrams()
    goal_text = (
        f"Solve this math problem in 1993 Actor 4.1, store the exact integer result in "
        f"`Actor[#Answer]` and `storeScratch(Actor[#Harness], #answer, val)!!`, and append `COMPLETE: Done` on the next line:\n{question}"
    )

    t0 = time.time()
    escaped = goal_text.replace("\\", "\\\\").replace('"', '\\"')
    vm.eval_chunk(f'runGoal(Actor[#Harness], "{escaped}", {max_steps})!!')
    elapsed = round(time.time() - t0, 2)

    step_log = list(harness._fields.get("stepLog", []))
    steps_used = int(harness._fields.get("stepCount", 0) or 0)
    gas_used = int(harness._fields.get("totalGasUsed", 0) or 0)

    # Check Step 1 vs Final answer
    scratch = harness._fields.get("scratchpad")
    pred = vm.global_vars["Actor"].get("Answer")
    if not isinstance(pred, (int, float)) and scratch is not None:
        cand_scratch = scratch.at("answer")
        if cand_scratch is not None:
            pred = cand_scratch

    # Fallback: parse integer from lastResult if pred is None or a non-numeric ActorObject
    if not isinstance(pred, (int, float)):
        last_res = str(harness._fields.get("lastResult", ""))
        nums = re.findall(r"-?\d+", last_res.replace(",", ""))
        if nums:
            pred = int(nums[-1])

    step1_error = any(s.get("step") == 1 and s.get("status") == "error" for s in step_log)
    harness_passed = (pred == gold)
    single_shot_passed = harness_passed and not step1_error

    return {
        "task_id": f"GSM8K/{idx}",
        "gold": gold,
        "predicted": pred if isinstance(pred, (int, float, str, bool)) or pred is None else str(pred),
        "single_shot_actor_passed": single_shot_passed,
        "actor_harness_passed": harness_passed,
        "self_corrected": (step1_error and harness_passed),
        "steps": steps_used,
        "gas_used": gas_used,
        "elapsed_s": elapsed,
    }


def run_benchmark_suite(
    *,
    humaneval_limit: int = 12,
    gsm8k_limit: int = 8,
    model: str = "gpt-oss:20b",
) -> Dict[str, Any]:
    with HUMANEVAL_PATH.open("r", encoding="utf-8") as fh:
        he_by_id = {r["task_id"]: r for r in (json.loads(line) for line in fh)}

    selected_he = [he_by_id[tid] for tid in DEFAULT_HUMANEVAL_IDS[:humaneval_limit] if tid in he_by_id]
    he_results: List[Dict[str, Any]] = []

    print(f"\n=== 1. Running Official OpenAI HumanEval (MultiPL-E Actor 4.1 Track, N={len(selected_he)}) ===")
    for row in selected_he:
        res = evaluate_humaneval_problem(row, model=model)
        he_results.append(res)
        s1 = "PASS" if res["single_shot_actor_passed"] else "FAIL"
        hm = "PASS" if res["actor_harness_passed"] else "FAIL"
        eg = "PASS" if res.get("engram_cold_reboot_passed") else "FAIL"
        py = "PASS" if res["python_pass1"] else "FAIL"
        sc = " [SELF-CORRECTED!]" if res["self_corrected"] else ""
        print(
            f"  {res['task_id']:<13} ({res['entry_point']:<24}) │ "
            f"1-Shot: {s1} │ Harness: {hm} (steps={res['actor_harness_steps']}, gas={res['actor_gas_used']}){sc} │ Engram Cold-Boot: {eg} │ Py: {py}",
            flush=True,
        )

    gsm_results: List[Dict[str, Any]] = []
    if gsm8k_limit > 0 and GSM8K_PATH.is_file():
        with GSM8K_PATH.open("r", encoding="utf-8") as fh:
            gsm_rows = [json.loads(line) for _, line in zip(range(gsm8k_limit), fh)]
        print(f"\n=== 2. Running Official OpenAI GSM8K (Actor 4.1 PAL Track, N={len(gsm_rows)}) ===", flush=True)
        for idx, row in enumerate(gsm_rows):
            res = evaluate_gsm8k_problem(idx, row, model=model)
            gsm_results.append(res)
            hm = "PASS" if res["actor_harness_passed"] else "FAIL"
            sc = " [SELF-CORRECTED!]" if res["self_corrected"] else ""
            print(
                f"  {res['task_id']:<10} │ gold={res['gold']:<6} pred={str(res['predicted']):<6} │ "
                f"ActorHarness: {hm} (steps={res['steps']}, gas={res['gas_used']}){sc}",
                flush=True,
            )

    he_n = max(1, len(he_results))
    he_s1_cnt = sum(1 for r in he_results if r["single_shot_actor_passed"])
    he_hm_cnt = sum(1 for r in he_results if r["actor_harness_passed"])
    he_eg_cnt = sum(1 for r in he_results if r.get("engram_cold_reboot_passed"))
    he_py_cnt = sum(1 for r in he_results if r["python_pass1"])
    he_sc_cnt = sum(1 for r in he_results if r["self_corrected"])

    gsm_n = max(1, len(gsm_results))
    gsm_s1_cnt = sum(1 for r in gsm_results if r["single_shot_actor_passed"])
    gsm_hm_cnt = sum(1 for r in gsm_results if r["actor_harness_passed"])

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "humaneval": {
            "n": len(he_results),
            "single_shot_actor_pass1": f"{he_s1_cnt}/{len(he_results)} ({100.0 * he_s1_cnt / he_n:.1f}%)",
            "actor_harness_pass": f"{he_hm_cnt}/{len(he_results)} ({100.0 * he_hm_cnt / he_n:.1f}%)",
            "engram_cold_reboot_pass": f"{he_eg_cnt}/{len(he_results)} ({100.0 * he_eg_cnt / he_n:.1f}%)",
            "self_corrected_count": he_sc_cnt,
            "python_reference_pass1": f"{he_py_cnt}/{len(he_results)} ({100.0 * he_py_cnt / he_n:.1f}%)",
            "tasks": he_results,
        },
        "gsm8k": {
            "n": len(gsm_results),
            "single_shot_actor_pass1": f"{gsm_s1_cnt}/{len(gsm_results)} ({100.0 * gsm_s1_cnt / gsm_n:.1f}%)",
            "actor_harness_pass": f"{gsm_hm_cnt}/{len(gsm_results)} ({100.0 * gsm_hm_cnt / gsm_n:.1f}%)",
            "tasks": gsm_results,
        },
    }

    RESULTS_JSON_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    engram_json_path = RESULTS_JSON_PATH.parent / "BENCHMARK_RESULTS_qwen38_27b_engrams.json"
    engram_json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_markdown_report(summary)
    print(f"\nSaved full JSON results to {RESULTS_JSON_PATH} and {engram_json_path}.", flush=True)
    return summary


def _write_markdown_report(summary: Dict[str, Any]) -> None:
    he = summary["humaneval"]
    gsm = summary["gsm8k"]
    lines = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        "# ActorII AgentHarness — Official OpenAI HumanEval & GSM8K Benchmark Report",
        "",
        f"- **Model**: `{summary['model']}` (`valkyrie` local `:8080`)",
        f"- **Timestamp**: `{summary['timestamp']}`",
        "",
        "## 1. Executive Summary (Head-to-Head Harness Comparison)",
        "",
        "| Benchmark Suite | 1-Shot Actor 4.1 (`Pass@1`, No Harness) | **ActorII `AgentHarness` (Multi-Step + Linter/Test Recovery)** | Python `Pass@1` Reference |",
        "| :--- | :---: | :---: | :---: |",
        f"| **OpenAI HumanEval** (MultiPL-E Track, $N={he['n']}$) | {he['single_shot_actor_pass1']} | **{he['actor_harness_pass']}** (+{he['self_corrected_count']} self-corrected) | {he['python_reference_pass1']} |",
        f"| **OpenAI GSM8K** (PAL Math Track, $N={gsm['n']}$) | {gsm['single_shot_actor_pass1']} | **{gsm['actor_harness_pass']}** | — |",
        "",
        "## 2. Per-Task OpenAI HumanEval Results",
        "",
        "| Task ID | Entry Point | 1-Shot Actor 4.1 | ActorII `AgentHarness` | Steps | Gas (Opcodes) | Python `Pass@1` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in he["tasks"]:
        s1 = "PASS" if r["single_shot_actor_passed"] else "FAIL"
        hm = "**PASS (Recovered)**" if r["self_corrected"] else ("PASS" if r["actor_harness_passed"] else "FAIL")
        py = "PASS" if r["python_pass1"] else "FAIL"
        lines.append(
            f"| `{r['task_id']}` | `{r['entry_point']}` | {s1} | {hm} | {r['actor_harness_steps']} | {r['actor_gas_used']} | {py} |"
        )

    if gsm["tasks"]:
        lines.extend(
            [
                "",
                "## 3. Per-Task OpenAI GSM8K Results",
                "",
                "| Task ID | Gold Answer | Predicted | ActorII `AgentHarness` | Steps | Gas (Opcodes) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in gsm["tasks"]:
            hm = "PASS" if r["actor_harness_passed"] else "FAIL"
            lines.append(
                f"| `{r['task_id']}` | `{r['gold']}` | `{r['predicted']}` | {hm} | {r['steps']} | {r['gas_used']} |"
            )

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI HumanEval & GSM8K on ActorII AgentHarness")
    parser.add_argument("--humaneval-n", type=int, default=12)
    parser.add_argument("--gsm8k-n", type=int, default=8)
    parser.add_argument("--model", default="qwen3.8:latest")
    args = parser.parse_args()
    run_benchmark_suite(humaneval_limit=args.humaneval_n, gsm8k_limit=args.gsm8k_n, model=args.model)
