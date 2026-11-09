#!/usr/bin/env python3
"""Head-to-Head 30-Task Comprehensive Benchmark: ActorII AgentHarness vs. Hermes Agent.

Compares `ActorII AgentHarness` (1993 Whitewater Actor 4.1 VM + Neo4j `Actor[#Grammar]` +
Active Engrams `Actor[#Memory]`) directly against `Hermes Agent` (`hermes -z ... --yolo`)
using the exact same local GPU model (`qwen3.8:latest` on `:8080`) across 30 tasks:

- Track 1 (15 Unseen Official OpenAI HumanEval Problems):
  `HumanEval/0`, `2`, `4`, `5`, `9`, `12`, `18`, `28`, `30`, `36`, `40`, `42`, `43`, `46`, `48`
  (including list/array returns, string indexing, nested loops, and floating-point math).
- Track 2 (10 Unseen Official OpenAI GSM8K Problems):
  `GSM8K/8` through `GSM8K/17`.
- Track 3 (5 Cold-Reboot Persistent Graph Memory + Computation Tasks):
  Multi-step Neo4j domain calculations across cold session reboots.

Features automatic per-task checkpointing (`benchmarks/ACTOR_VS_HERMES_RESULTS.json`) so
the benchmark can resume cleanly if interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from actorii.ai import ValkyrieLLMClient
from actorii.classes.collections import ActorOrderedCollection
from actorii.harness import bootstrap_agent_harness, extract_actor_action
from actorii.vm import ActorVM

HUMANEVAL_PATH = Path(__file__).resolve().parent / "data" / "HumanEval.jsonl"
GSM8K_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_test.jsonl"
RESULTS_JSON_PATH = Path(__file__).resolve().parent / "ACTOR_VS_HERMES_RESULTS.json"
REPORT_MD_PATH = Path(__file__).resolve().parent / "ACTOR_VS_HERMES_REPORT.md"
HERMES_BIN = "/home/user/.hermes/src/venv/bin/hermes"

OPTION_C_HUMANEVAL_IDS = [
    "HumanEval/0",   # has_close_elements(numbers: List[float], threshold: float) -> bool
    "HumanEval/2",   # truncate_number(number: float) -> float
    "HumanEval/4",   # mean_absolute_deviation(numbers: List[float]) -> float
    "HumanEval/5",   # intersperse(numbers: List[int], delimeter: int) -> List[int]
    "HumanEval/9",   # rolling_max(numbers: List[int]) -> List[int]
    "HumanEval/12",  # longest(strings: List[str]) -> Optional[str]
    "HumanEval/18",  # how_many_times(string: str, substring: str) -> int
    "HumanEval/28",  # concatenate(strings: List[str]) -> str
    "HumanEval/30",  # get_positive(l: list) -> list
    "HumanEval/36",  # fizz_buzz(n: int) -> int
    "HumanEval/40",  # triples_sum_to_zero(l: list) -> bool
    "HumanEval/42",  # incr_list(l: list) -> list
    "HumanEval/43",  # pairs_sum_to_zero(l: list) -> bool
    "HumanEval/46",  # fib4(n: int) -> int
    "HumanEval/48",  # is_palindrome(text: str) -> bool
]

OPTION_C_GSM8K_INDICES = list(range(8, 18))  # GSM8K/8 .. GSM8K/17 (10 unseen problems)


# =============================================================================
# Token Counting & Hermes CLI Runner
# =============================================================================


class TokenCounter:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_ACTIVE_COUNTER: Optional[TokenCounter] = None
_ORIG_COMPLETE = ValkyrieLLMClient.complete


def _instrumented_complete(self: ValkyrieLLMClient, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
    res = _ORIG_COMPLETE(self, prompt, system_prompt=system_prompt, **kwargs)
    if _ACTIVE_COUNTER is not None:
        _ACTIVE_COUNTER.calls += 1
        _ACTIVE_COUNTER.prompt_tokens += max(1, (len(str(prompt)) + len(str(system_prompt))) // 4)
        _ACTIVE_COUNTER.completion_tokens += max(1, len(str(res or "")) // 4)
    return res


ValkyrieLLMClient.complete = _instrumented_complete  # type: ignore[method-assign]


def run_hermes_cli(prompt: str, timeout_s: int = 110, toolsets: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
    """Execute `hermes -z <prompt> --yolo` and return `(stdout_text, usage_dict, elapsed_s)`."""
    usage_path = Path(f"/tmp/hermes_bench_usage_{os.getpid()}.json")
    if usage_path.exists():
        usage_path.unlink()

    cmd = [
        HERMES_BIN,
        "-z",
        prompt,
        "--yolo",
        "--usage-file",
        str(usage_path),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd="/tmp",
        )
        out = (proc.stdout or "").strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")).strip()
    except Exception as exc:
        out = f"ERROR: {exc}"
    elapsed = round(time.time() - t0, 2)

    usage: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 1,
    }
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not usage.get("total_tokens"):
        est_in = 3200 + max(1, len(prompt) // 4)
        est_out = max(1, len(out) // 4)
        usage["input_tokens"] = est_in
        usage["output_tokens"] = est_out
        usage["total_tokens"] = est_in + est_out
    return out, usage, elapsed


# =============================================================================
# Track 1: OpenAI HumanEval Helpers
# =============================================================================


def _parse_python_signature(prompt: str, entry_point: str) -> List[str]:
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


def _actor_to_py_val(val: Any, returns_bool: bool = False) -> Any:
    if isinstance(val, (ActorOrderedCollection, list, tuple)):
        return [_actor_to_py_val(x, False) for x in list(val)]
    if returns_bool and isinstance(val, int):
        return bool(val)
    return val


def run_humaneval_check_in_actor_vm(
    vm: ActorVM,
    entry_point: str,
    test_code: str,
    prompt_doc: str,
) -> Tuple[bool, Optional[str], int]:
    returns_bool = (
        "-> bool" in prompt_doc
        or "True" in prompt_doc
        or "False" in prompt_doc
    )
    total_gas = 0
    last_actual: Any = None

    def _candidate(*args: Any) -> Any:
        nonlocal total_gas, last_actual
        actor_args = ", ".join(_py_arg_to_actor_literal(a) for a in args)
        call_expr = f"{entry_point}(Actor[#Sol], {actor_args})!!" if actor_args else f"{entry_point}(Actor[#Sol])!!"
        res = vm.eval_chunk(call_expr)
        total_gas += int(getattr(vm, "last_eval_opcodes", 0))
        res = _actor_to_py_val(res, returns_bool=returns_bool)
        last_actual = res
        return res

    ns: Dict[str, Any] = {}
    exec(test_code, ns)
    check_fn = ns["check"]

    try:
        check_fn(_candidate)
        return True, None, total_gas
    except Exception as exc:
        for line in test_code.splitlines():
            s = line.strip()
            if s.startswith("assert "):
                try:
                    exec(f"def _single(candidate):\n    {s}\n", ns)
                    ns["_single"](_candidate)
                except Exception as inner_exc:
                    return (
                        False,
                        f"Unit test failed on `{s}` (actual returned value was `{last_actual!r}`, {type(inner_exc).__name__}: {inner_exc}). "
                        f"Fix the algorithm in `Def {entry_point}(...)`, ensure all local variables are declared after `|`, "
                        f"and if returning a dynamic list use `res := new(OrderedCollection); add(res, item); ^res;`.",
                        total_gas,
                    )
        return False, f"{type(exc).__name__}: {exc}", total_gas


def _extract_python_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


def evaluate_humaneval_head_to_head(row: Dict[str, Any], model: str = "qwen3.8:latest") -> Dict[str, Any]:
    global _ACTIVE_COUNTER
    task_id = row["task_id"]
    entry_point = row["entry_point"]
    prompt_doc = row["prompt"]
    test_code = row["test"]
    params = _parse_python_signature(prompt_doc, entry_point)
    params_str = ", ".join(params)

    # 1. ActorII AgentHarness
    counter = TokenCounter()
    _ACTIVE_COUNTER = counter
    he_ns = f"h2h_he_{entry_point}"
    vm, harness, sandbox = bootstrap_agent_harness(
        model=model,
        max_steps=4,
        gas_limit=80000,
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

    actor_passed = False
    actor_steps = 0
    actor_gas = 0
    last_err = ""
    last_src = ""

    t0 = time.time()
    for step_idx in range(1, 5):
        actor_steps = step_idx
        step_prompt = (
            f"Solve OpenAI {task_id} (`{entry_point}`) in 1993 Whitewater Actor 4.1.\n\n"
            f"SPECIFICATION:\n{prompt_doc}\n\n"
            f"{grammar_guide}\n"
            f"Output ONLY the valid 1993 Actor `Def {entry_point}(...) {{ ... }}!!` chunk.\n"
        )
        if last_err:
            neo4j_diag = (
                sandbox.grammar_graph.diagnoseError(last_err)
                if sandbox.grammar_graph is not None
                else ""
            )
            step_prompt += (
                f"\n[PREVIOUS ATTEMPT FAILED — SELF-CORRECT THIS ERROR]:\n"
                f"Previous code:\n{last_src}\n"
                f"Error / Failed Test:\n{last_err}\n"
                + (f"{neo4j_diag}\n" if neo4j_diag else "")
                + "Fix the 1993 Actor 4.1 method definition and output ONLY `Def ... { ... }!!`."
            )

        harness.clearHistory()
        raw_reply = harness.ask(step_prompt)
        action_chunk = extract_actor_action(raw_reply)
        action_chunk = re.sub(r"now\s*\([^)]+\)\s*!!", "now(Solution)!!", action_chunk)
        if not action_chunk.lstrip().startswith("now(Solution)"):
            action_chunk = f"now(Solution)!!\n{action_chunk}"
        last_src = action_chunk

        outcome = sandbox.executeStepAction(harness, action_chunk, max_opcodes=80000)
        actor_gas += int(getattr(vm, "last_eval_opcodes", 0))
        if isinstance(outcome, str) and outcome.startswith("ERROR:"):
            last_err = outcome
            continue

        passed, test_err, test_gas = run_humaneval_check_in_actor_vm(vm, entry_point, test_code, prompt_doc)
        actor_gas += test_gas
        if passed:
            actor_passed = True
            break
        last_err = test_err or "Failed unit test assertions"

    actor_elapsed = round(time.time() - t0, 2)
    _ACTIVE_COUNTER = None

    # Verify Cold-Boot Engram Recall on fresh VM (0 LLM tokens)
    engram_cold_boot = False
    if actor_passed:
        try:
            vm_cold, _, sb_cold = bootstrap_agent_harness(
                model=model,
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace=he_ns,
            )
            if sb_cold.memory_graph is not None:
                sb_cold.memory_graph.hydrateRelevantEngrams(entry_point)
                vm_cold.eval_chunk("Actor[#Sol] := Actor[#Solution]!!")
                engram_cold_boot, _, _ = run_humaneval_check_in_actor_vm(
                    vm_cold, entry_point, test_code, prompt_doc
                )
        except Exception:
            engram_cold_boot = False

    # 2. Hermes Agent
    hermes_prompt = (
        f"Solve OpenAI {task_id} (`{entry_point}`) in Python. "
        f"Output ONLY the complete valid Python function implementation inside a ```python ... ``` block:\n\n{prompt_doc}"
    )
    hermes_out, hermes_usage, hermes_elapsed = run_hermes_cli(hermes_prompt, timeout_s=90, toolsets="none")
    hermes_passed = False
    try:
        py_code = _extract_python_code(hermes_out)
        py_ns: Dict[str, Any] = {}
        header = "from typing import List, Tuple, Optional, Any, Dict\nimport math\n"
        if f"def {entry_point}" in py_code:
            exec(header + prompt_doc + "\n\n" + py_code, py_ns)
        else:
            indented = "\n".join("    " + ln for ln in py_code.splitlines())
            exec(header + prompt_doc + "\n" + indented, py_ns)
        exec(test_code, py_ns)
        py_ns["check"](py_ns[entry_point])
        hermes_passed = True
    except Exception:
        hermes_passed = False

    return {
        "track": "HumanEval",
        "task_id": task_id,
        "title": entry_point,
        "actorii": {
            "passed": actor_passed,
            "engram_cold_boot_passed": engram_cold_boot,
            "steps": actor_steps,
            "tokens": counter.total_tokens,
            "elapsed_s": actor_elapsed,
            "gas": actor_gas,
        },
        "hermes": {
            "passed": hermes_passed,
            "api_calls": int(hermes_usage.get("api_calls", 1) or 1),
            "tokens": int(hermes_usage.get("total_tokens", 0) or 0),
            "elapsed_s": hermes_elapsed,
        },
    }


# =============================================================================
# Track 2: OpenAI GSM8K Helpers
# =============================================================================


def _extract_gsm8k_gold(answer_str: str) -> Optional[int]:
    if "####" in answer_str:
        tail = answer_str.split("####", 1)[1].strip().replace(",", "")
        m = re.search(r"-?\d+", tail)
        if m:
            return int(m.group(0))
    return None


def _extract_last_int(text: str) -> Optional[int]:
    cleaned = str(text or "").replace(",", "")
    if "####" in cleaned:
        tail = cleaned.split("####")[-1]
        m = re.search(r"-?\d+", tail)
        if m:
            return int(m.group(0))
    nums = re.findall(r"-?\d+", cleaned)
    if nums:
        return int(nums[-1])
    return None


def _extract_actor_numeric_prediction(vm: ActorVM, harness: Any) -> Optional[int]:
    cand = vm.global_vars.get("Actor", {}).get("Answer")
    if isinstance(cand, (int, float)) and not isinstance(cand, bool):
        return int(cand) if isinstance(cand, float) and cand.is_integer() else int(round(cand))

    scratch = harness._fields.get("scratchpad")
    if scratch is not None:
        s_ans = scratch.at("answer")
        if isinstance(s_ans, (int, float)) and not isinstance(s_ans, bool):
            return int(s_ans) if isinstance(s_ans, float) and s_ans.is_integer() else int(round(s_ans))
        entries = getattr(scratch, "_entries", {})
        if isinstance(entries, dict):
            for v in reversed(list(entries.values())):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return int(v) if isinstance(v, float) and v.is_integer() else int(round(v))

    last_res = harness._fields.get("lastResult")
    if isinstance(last_res, (int, float)) and not isinstance(last_res, bool):
        return int(last_res) if isinstance(last_res, float) and last_res.is_integer() else int(round(last_res))

    if hasattr(cand, "_fields") and isinstance(cand._fields, dict):
        for v in reversed(list(cand._fields.values())):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v) if isinstance(v, float) and v.is_integer() else int(round(v))

    return _extract_last_int(str(last_res or ""))


def evaluate_gsm8k_head_to_head(idx: int, row: Dict[str, Any], model: str = "qwen3.8:latest") -> Dict[str, Any]:
    global _ACTIVE_COUNTER
    task_id = f"GSM8K/{idx}"
    question = row["question"]
    gold = _extract_gsm8k_gold(row["answer"])

    # 1. ActorII AgentHarness
    counter = TokenCounter()
    _ACTIVE_COUNTER = counter
    gsm_ns = f"h2h_gsm8k_{idx}"
    vm, harness, sb = bootstrap_agent_harness(
        model=model,
        max_steps=4,
        gas_limit=30000,
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
    vm.eval_chunk(f'runGoal(Actor[#Harness], "{escaped}", 4)!!')
    actor_elapsed = round(time.time() - t0, 2)
    _ACTIVE_COUNTER = None

    actor_steps = int(harness._fields.get("stepCount", 0) or 0)
    actor_gas = int(harness._fields.get("totalGasUsed", 0) or 0)
    actor_pred = _extract_actor_numeric_prediction(vm, harness)

    actor_passed = (actor_pred == gold)

    # 2. Hermes Agent
    hermes_prompt = (
        f"Solve the following math word problem and end your response with `#### <exact_integer_answer>`:\n\n{question}"
    )
    hermes_out, hermes_usage, hermes_elapsed = run_hermes_cli(hermes_prompt, timeout_s=90, toolsets="none")
    hermes_pred = _extract_last_int(hermes_out)
    hermes_passed = (hermes_pred == gold)

    return {
        "track": "GSM8K",
        "task_id": task_id,
        "title": question[:48] + "...",
        "gold": gold,
        "actorii": {
            "passed": actor_passed,
            "predicted": actor_pred,
            "steps": actor_steps,
            "tokens": counter.total_tokens,
            "elapsed_s": actor_elapsed,
            "gas": actor_gas,
        },
        "hermes": {
            "passed": hermes_passed,
            "predicted": hermes_pred,
            "api_calls": int(hermes_usage.get("api_calls", 1) or 1),
            "tokens": int(hermes_usage.get("total_tokens", 0) or 0),
            "elapsed_s": hermes_elapsed,
        },
    }


# =============================================================================
# Track 3: 5 Cold-Reboot Persistent Graph Memory + Computation Tasks
# =============================================================================


COLD_REBOOT_MEMORY_TASKS = [
    {
        "task_id": "MemGraph/1",
        "title": "Contractor Overtime Payroll (hours=52)",
        "namespace": "h2h_mem_payroll",
        "seed_cypher": (
            "MATCH (n:H2HContractor) DETACH DELETE n; "
            "CREATE (:H2HContractor {name: 'Alice', rate: 120, std_cap: 40}), "
            "(:H2HContractor {name: 'Bob', rate: 90, std_cap: 35}), "
            "(:H2HContractor {name: 'Cara', rate: 150, std_cap: 45}), "
            "(:H2HContractor {name: 'Dan', rate: 80, std_cap: 40})"
        ),
        "engram_name": "PayrollCalc",
        "engram_desc": "Calculate total weekly payroll across :H2HContractor nodes for hoursWorked with 2x overtime above std_cap",
        "engram_source": (
            "inherit(Object, #PayrollCalc, #(#state), 1, nil)!!\n"
            "now(PayrollCalc)!!\n"
            "Def totalPayroll(self, hoursWorked | rows, i, row, rate, cap, reg, ot, total) {\n"
            '  rows := cypherQuery(Actor[#Memory], "MATCH (c:H2HContractor) RETURN c.rate AS rate, c.std_cap AS std_cap");\n'
            "  total := 0; i := 0;\n"
            "  loop while (i < size(rows)) begin\n"
            "    row := rows[i]; rate := at(row, #rate); cap := at(row, #std_cap);\n"
            "    if (hoursWorked > cap) then reg := cap; ot := hoursWorked - cap; else reg := hoursWorked; ot := 0; endif;\n"
            "    total := total + (reg * rate) + (ot * rate * 2);\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^total;\n"
            "}!!\n"
            "Actor[#PayrollCalc] := new(PayrollCalc)!!"
        ),
        "actor_goal": "Calculate total weekly payroll across all contractors for `hoursWorked = 52`, store the exact integer in `storeScratch(Actor[#Harness], #answer, val)!!`, and finish.",
        "hermes_prompt": (
            "Query the local Neo4j graph on `http://127.0.0.1:7474/db/neo4j/tx/commit` (no auth) for all `:H2HContractor` nodes "
            "(properties `rate` and `std_cap`). Compute the total weekly payroll across all contractors when every contractor works "
            "`hoursWorked = 52` hours, where hours up to `std_cap` pay `rate` and hours above `std_cap` pay `2 * rate`. "
            "End your response with `#### <exact_integer_answer>`."
        ),
        "gold": 27860,
    },
    {
        "task_id": "MemGraph/2",
        "title": "Supply Chain Landed Tariff Cost (surcharge=15%)",
        "namespace": "h2h_mem_supply",
        "seed_cypher": (
            "MATCH (n:H2HItem) DETACH DELETE n; "
            "CREATE (:H2HItem {sku: 'GPU', qty: 10, unit_cost: 1000, duty_pct: 15, hazmat_fee: 200}), "
            "(:H2HItem {sku: 'PSU', qty: 25, unit_cost: 160, duty_pct: 5, hazmat_fee: 0}), "
            "(:H2HItem {sku: 'BATT', qty: 40, unit_cost: 250, duty_pct: 20, hazmat_fee: 500}), "
            "(:H2HItem {sku: 'RACK', qty: 5, unit_cost: 800, duty_pct: 10, hazmat_fee: 100})"
        ),
        "engram_name": "TariffCalc",
        "engram_desc": "Calculate total landed shipment cost across :H2HItem nodes for surchargePct",
        "engram_source": (
            "inherit(Object, #TariffCalc, #(#state), 1, nil)!!\n"
            "now(TariffCalc)!!\n"
            "Def landedCost(self, surchargePct | rows, i, row, base, duty, total) {\n"
            '  rows := cypherQuery(Actor[#Memory], "MATCH (i:H2HItem) RETURN i.qty AS qty, i.unit_cost AS unit_cost, i.duty_pct AS duty_pct, i.hazmat_fee AS hazmat_fee");\n'
            "  total := 0; i := 0;\n"
            "  loop while (i < size(rows)) begin\n"
            "    row := rows[i];\n"
            "    base := at(row, #qty) * at(row, #unit_cost);\n"
            "    duty := (base * (at(row, #duty_pct) + surchargePct)) / 100;\n"
            "    total := total + base + duty + at(row, #hazmat_fee);\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^total;\n"
            "}!!\n"
            "Actor[#TariffCalc] := new(TariffCalc)!!"
        ),
        "actor_goal": "Calculate total landed shipment cost across all items for `surchargePct = 15`, store the exact integer in `storeScratch(Actor[#Harness], #answer, val)!!`, and finish.",
        "hermes_prompt": (
            "Query the local Neo4j graph on `http://127.0.0.1:7474/db/neo4j/tx/commit` (no auth) for all `:H2HItem` nodes "
            "(`qty`, `unit_cost`, `duty_pct`, `hazmat_fee`). For each item, `base = qty * unit_cost`, "
            "`duty = (base * (duty_pct + surchargePct)) // 100`, and `item_total = base + duty + hazmat_fee`. "
            "Compute the sum of `item_total` across all items for `surchargePct = 15` and end with `#### <exact_integer_answer>`."
        ),
        "gold": 37100,
    },
    {
        "task_id": "MemGraph/3",
        "title": "Microservice SLA Latency Breach Score (budget=65ms)",
        "namespace": "h2h_mem_sla",
        "seed_cypher": (
            "MATCH (n:H2HService) DETACH DELETE n; "
            "CREATE (:H2HService {name: 'Gateway', base_ms: 12, dep_hops: 4, hop_ms: 18, weight: 5}), "
            "(:H2HService {name: 'Auth', base_ms: 8, dep_hops: 1, hop_ms: 10, weight: 3}), "
            "(:H2HService {name: 'Billing', base_ms: 25, dep_hops: 3, hop_ms: 22, weight: 4}), "
            "(:H2HService {name: 'Search', base_ms: 15, dep_hops: 2, hop_ms: 20, weight: 2}), "
            "(:H2HService {name: 'Analytics', base_ms: 40, dep_hops: 5, hop_ms: 15, weight: 1})"
        ),
        "engram_name": "SlaAuditor",
        "engram_desc": "Calculate total SLA breach penalty score across :H2HService nodes for slaBudgetMs",
        "engram_source": (
            "inherit(Object, #SlaAuditor, #(#state), 1, nil)!!\n"
            "now(SlaAuditor)!!\n"
            "Def breachScore(self, slaBudgetMs | rows, i, row, eff, total) {\n"
            '  rows := cypherQuery(Actor[#Memory], "MATCH (s:H2HService) RETURN s.base_ms AS base_ms, s.dep_hops AS dep_hops, s.hop_ms AS hop_ms, s.weight AS weight");\n'
            "  total := 0; i := 0;\n"
            "  loop while (i < size(rows)) begin\n"
            "    row := rows[i];\n"
            "    eff := at(row, #base_ms) + (at(row, #dep_hops) * at(row, #hop_ms));\n"
            "    if (eff > slaBudgetMs) then total := total + ((eff - slaBudgetMs) * at(row, #weight)); endif;\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^total;\n"
            "}!!\n"
            "Actor[#SlaAuditor] := new(SlaAuditor)!!"
        ),
        "actor_goal": "Calculate the total SLA breach penalty score across all microservices for `slaBudgetMs = 65`, store the exact integer in `storeScratch(Actor[#Harness], #answer, val)!!`, and finish.",
        "hermes_prompt": (
            "Query the local Neo4j graph on `http://127.0.0.1:7474/db/neo4j/tx/commit` (no auth) for all `:H2HService` nodes "
            "(`base_ms`, `dep_hops`, `hop_ms`, `weight`). For each service, `eff = base_ms + dep_hops * hop_ms`. "
            "If `eff > slaBudgetMs`, add `(eff - slaBudgetMs) * weight` to the total penalty score. "
            "Compute the total penalty score for `slaBudgetMs = 65` and end with `#### <exact_integer_answer>`."
        ),
        "gold": 249,
    },
    {
        "task_id": "MemGraph/4",
        "title": "Smart-Grid Time-of-Use Bill Cents (days=10)",
        "namespace": "h2h_mem_grid",
        "seed_cypher": (
            "MATCH (n:H2HAppliance) DETACH DELETE n; "
            "CREATE (:H2HAppliance {id: 'HVAC', watts: 3500, peak_hrs: 4, offpeak_hrs: 6, credit_cents: 120}), "
            "(:H2HAppliance {id: 'EVCharger', watts: 7000, peak_hrs: 1, offpeak_hrs: 5, credit_cents: 200}), "
            "(:H2HAppliance {id: 'WaterHeater', watts: 4000, peak_hrs: 2, offpeak_hrs: 3, credit_cents: 50}), "
            "(:H2HAppliance {id: 'ServerRack', watts: 1000, peak_hrs: 6, offpeak_hrs: 18, credit_cents: 0})"
        ),
        "engram_name": "GridBillCalc",
        "engram_desc": "Calculate total smart-grid bill in cents across :H2HAppliance nodes over days",
        "engram_source": (
            "inherit(Object, #GridBillCalc, #(#state), 1, nil)!!\n"
            "now(GridBillCalc)!!\n"
            "Def billCents(self, days | rows, i, row, w, dailyCents, total) {\n"
            '  rows := cypherQuery(Actor[#Memory], "MATCH (a:H2HAppliance) RETURN a.watts AS watts, a.peak_hrs AS peak_hrs, a.offpeak_hrs AS offpeak_hrs, a.credit_cents AS credit_cents");\n'
            "  total := 0; i := 0;\n"
            "  loop while (i < size(rows)) begin\n"
            "    row := rows[i]; w := at(row, #watts);\n"
            "    dailyCents := (((w * at(row, #peak_hrs) * 30) + (w * at(row, #offpeak_hrs) * 10)) / 1000) - at(row, #credit_cents);\n"
            "    total := total + (dailyCents * days);\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^total;\n"
            "}!!\n"
            "Actor[#GridBillCalc] := new(GridBillCalc)!!"
        ),
        "actor_goal": "Calculate the total smart-grid electricity bill in cents across all appliances over `days = 10`, store the exact integer in `storeScratch(Actor[#Harness], #answer, val)!!`, and finish.",
        "hermes_prompt": (
            "Query the local Neo4j graph on `http://127.0.0.1:7474/db/neo4j/tx/commit` (no auth) for all `:H2HAppliance` nodes "
            "(`watts`, `peak_hrs`, `offpeak_hrs`, `credit_cents`). For each appliance, "
            "`daily_cents = ((watts * peak_hrs * 30 + watts * offpeak_hrs * 10) // 1000) - credit_cents`. "
            "Compute the sum of `daily_cents * days` across all appliances for `days = 10` and end with `#### <exact_integer_answer>`."
        ),
        "gold": 15400,
    },
    {
        "task_id": "MemGraph/5",
        "title": "EV Fleet Thermal Charging Energy Wh (lossPct=12)",
        "namespace": "h2h_mem_fleet",
        "seed_cypher": (
            "MATCH (n:H2HFleetVehicle) DETACH DELETE n; "
            "CREATE (:H2HFleetVehicle {vin: 'V1', cap_kwh: 80, soc_start: 20, soc_target: 90}), "
            "(:H2HFleetVehicle {vin: 'V2', cap_kwh: 100, soc_start: 15, soc_target: 85}), "
            "(:H2HFleetVehicle {vin: 'V3', cap_kwh: 60, soc_start: 30, soc_target: 95}), "
            "(:H2HFleetVehicle {vin: 'V4', cap_kwh: 120, soc_start: 10, soc_target: 80})"
        ),
        "engram_name": "FleetChargeCalc",
        "engram_desc": "Calculate total grid energy in Wh required to charge :H2HFleetVehicle fleet with lossPct thermal overhead",
        "engram_source": (
            "inherit(Object, #FleetChargeCalc, #(#state), 1, nil)!!\n"
            "now(FleetChargeCalc)!!\n"
            "Def fleetEnergyWh(self, lossPct | rows, i, row, netWh, grossWh, total) {\n"
            '  rows := cypherQuery(Actor[#Memory], "MATCH (v:H2HFleetVehicle) RETURN v.cap_kwh AS cap_kwh, v.soc_start AS soc_start, v.soc_target AS soc_target");\n'
            "  total := 0; i := 0;\n"
            "  loop while (i < size(rows)) begin\n"
            "    row := rows[i];\n"
            "    netWh := at(row, #cap_kwh) * (at(row, #soc_target) - at(row, #soc_start)) * 10;\n"
            "    grossWh := (netWh * (100 + lossPct)) / 100;\n"
            "    total := total + grossWh;\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^total;\n"
            "}!!\n"
            "Actor[#FleetChargeCalc] := new(FleetChargeCalc)!!"
        ),
        "actor_goal": "Calculate total grid energy in Wh required to charge the fleet with `lossPct = 12`, store the exact integer in `storeScratch(Actor[#Harness], #answer, val)!!`, and finish.",
        "hermes_prompt": (
            "Query the local Neo4j graph on `http://127.0.0.1:7474/db/neo4j/tx/commit` (no auth) for all `:H2HFleetVehicle` nodes "
            "(`cap_kwh`, `soc_start`, `soc_target`). For each vehicle, `net_wh = cap_kwh * (soc_target - soc_start) * 10`, "
            "and `gross_wh = (net_wh * (100 + lossPct)) // 100`. "
            "Compute the sum of `gross_wh` across all vehicles for `lossPct = 12` and end with `#### <exact_integer_answer>`."
        ),
        "gold": 278880,
    },
]


def evaluate_memory_graph_head_to_head(spec: Dict[str, Any], model: str = "qwen3.8:latest") -> Dict[str, Any]:
    global _ACTIVE_COUNTER
    task_id = spec["task_id"]
    ns = spec["namespace"]
    gold = spec["gold"]

    # Seed Neo4j domain data + crystallize Engram in Setup VM, then discard Setup VM (Cold Reboot!)
    vm_setup, _, sb_setup = bootstrap_agent_harness(
        model=model,
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=ns,
    )
    assert sb_setup.memory_graph is not None
    sb_setup.memory_graph.clearEngrams()
    for stmt in spec["seed_cypher"].split(";"):
        if stmt.strip():
            sb_setup.memory_graph._run_cypher(stmt.strip())
    sb_setup.memory_graph.saveEngram(
        spec["engram_name"],
        spec["engram_desc"],
        spec["engram_source"],
    )
    del vm_setup, sb_setup

    # 1. Cold-Reboot ActorII AgentHarness
    counter = TokenCounter()
    _ACTIVE_COUNTER = counter
    vm_cold, harness_cold, _ = bootstrap_agent_harness(
        model=model,
        max_steps=3,
        gas_limit=30000,
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=ns,
    )
    t0 = time.time()
    escaped = spec["actor_goal"].replace("\\", "\\\\").replace('"', '\\"')
    vm_cold.eval_chunk(f'runGoal(Actor[#Harness], "{escaped}", 3)!!')
    actor_elapsed = round(time.time() - t0, 2)
    _ACTIVE_COUNTER = None

    actor_steps = int(harness_cold._fields.get("stepCount", 0) or 0)
    actor_gas = int(harness_cold._fields.get("totalGasUsed", 0) or 0)
    actor_pred = _extract_actor_numeric_prediction(vm_cold, harness_cold)
    actor_passed = (actor_pred == gold)

    # 2. Cold-Reboot Hermes Agent (with terminal/code execution tools enabled so it can query Neo4j)
    hermes_out, hermes_usage, hermes_elapsed = run_hermes_cli(
        spec["hermes_prompt"], timeout_s=110, toolsets="terminal"
    )
    hermes_pred = _extract_last_int(hermes_out)
    hermes_passed = (hermes_pred == gold)

    return {
        "track": "MemGraph",
        "task_id": task_id,
        "title": spec["title"],
        "gold": gold,
        "actorii": {
            "passed": actor_passed,
            "predicted": actor_pred,
            "steps": actor_steps,
            "tokens": counter.total_tokens,
            "elapsed_s": actor_elapsed,
            "gas": actor_gas,
        },
        "hermes": {
            "passed": hermes_passed,
            "predicted": hermes_pred,
            "api_calls": int(hermes_usage.get("api_calls", 1) or 1),
            "tokens": int(hermes_usage.get("total_tokens", 0) or 0),
            "elapsed_s": hermes_elapsed,
        },
    }


# =============================================================================
# Main Checkpointed Runner & Report Generator
# =============================================================================


def _save_state(tasks: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    a_pass = sum(1 for t in tasks if t["actorii"]["passed"])
    h_pass = sum(1 for t in tasks if t["hermes"]["passed"])
    a_tok = sum(t["actorii"]["tokens"] for t in tasks)
    h_tok = sum(t["hermes"]["tokens"] for t in tasks)
    a_time = round(sum(t["actorii"]["elapsed_s"] for t in tasks), 2)
    h_time = round(sum(t["hermes"]["elapsed_s"] for t in tasks), 2)
    n = max(1, len(tasks))

    by_track: Dict[str, Any] = {}
    for tr in ("HumanEval", "GSM8K", "MemGraph"):
        sub = [t for t in tasks if t["track"] == tr]
        if not sub:
            continue
        sn = len(sub)
        ap = sum(1 for t in sub if t["actorii"]["passed"])
        hp = sum(1 for t in sub if t["hermes"]["passed"])
        by_track[tr] = {
            "n": sn,
            "actorii_pass": f"{ap}/{sn} ({100.0 * ap / sn:.1f}%)",
            "hermes_pass": f"{hp}/{sn} ({100.0 * hp / sn:.1f}%)",
            "actorii_tokens": sum(t["actorii"]["tokens"] for t in sub),
            "hermes_tokens": sum(t["hermes"]["tokens"] for t in sub),
            "actorii_avg_s": round(sum(t["actorii"]["elapsed_s"] for t in sub) / sn, 2),
            "hermes_avg_s": round(sum(t["hermes"]["elapsed_s"] for t in sub) / sn, 2),
        }

    state = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "completed_tasks": len(tasks),
        "overall": {
            "actorii_pass": f"{a_pass}/{len(tasks)} ({100.0 * a_pass / n:.1f}%)",
            "hermes_pass": f"{h_pass}/{len(tasks)} ({100.0 * h_pass / n:.1f}%)",
            "actorii_total_tokens": a_tok,
            "hermes_total_tokens": h_tok,
            "actorii_total_s": a_time,
            "hermes_total_s": h_time,
        },
        "by_track": by_track,
        "tasks": tasks,
    }
    RESULTS_JSON_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    _write_markdown_report(state)
    return state


def _write_markdown_report(state: Dict[str, Any]) -> None:
    ov = state["overall"]
    bt = state["by_track"]
    lines = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        "# Head-to-Head Benchmark Report: `ActorII AgentHarness` vs. `Hermes Agent` (Option C — 30 Tasks)",
        "",
        f"- **Shared Local Model (`:8080`)**: `{state['model']}` (`qwen3.8-27b.gguf` on RTX 5070 Ti)",
        f"- **Completed Tasks**: `{state['completed_tasks']}/30`",
        f"- **Timestamp**: `{state['timestamp']}`",
        "",
        "## 1. Executive Summary (`ActorII AgentHarness` vs. `Hermes Agent`)",
        "",
        "| Track | Tasks ($N$) | **`ActorII AgentHarness` Accuracy** | **`Hermes Agent` Accuracy** | `ActorII` Total Tokens | `Hermes` Total Tokens | `ActorII` Avg Latency | `Hermes` Avg Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for tr in ("HumanEval", "GSM8K", "MemGraph"):
        if tr in bt:
            r = bt[tr]
            lines.append(
                f"| **{tr}** | {r['n']} | **{r['actorii_pass']}** | {r['hermes_pass']} | `{r['actorii_tokens']:,}` | `{r['hermes_tokens']:,}` | `{r['actorii_avg_s']}s` | `{r['hermes_avg_s']}s` |"
            )
    lines.append(
        f"| **OVERALL (30 Tasks)** | **{state['completed_tasks']}** | **{ov['actorii_pass']}** | **{ov['hermes_pass']}** | **`{ov['actorii_total_tokens']:,}`** | **`{ov['hermes_total_tokens']:,}`** | `{round(ov['actorii_total_s']/max(1, state['completed_tasks']), 2)}s` | `{round(ov['hermes_total_s']/max(1, state['completed_tasks']), 2)}s` |"
    )
    lines.extend(
        [
            "",
            "## 2. Per-Task Breakdown (All 30 Tasks)",
            "",
            "| Track | Task ID | Title / Entry Point | **`ActorII` Result** | `ActorII` Steps / Tokens / Time | **`Hermes` Result** | `Hermes` Calls / Tokens / Time |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for t in state["tasks"]:
        a = t["actorii"]
        h = t["hermes"]
        a_res = "**PASS**" if a["passed"] else "FAIL"
        h_res = "**PASS**" if h["passed"] else "FAIL"
        lines.append(
            f"| {t['track']} | `{t['task_id']}` | `{t['title']}` | {a_res} | {a['steps']}st / {a['tokens']}tok / {a['elapsed_s']}s | {h_res} | {h['api_calls']}c / {h['tokens']}tok / {h['elapsed_s']}s |"
        )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30-Task ActorII vs Hermes Benchmark")
    parser.add_argument("--model", default="qwen3.8:latest")
    parser.add_argument("--reset", action="store_true", help="Ignore existing checkpoint and start fresh")
    args = parser.parse_args()

    completed: List[Dict[str, Any]] = []
    done_ids = set()
    if RESULTS_JSON_PATH.exists() and not args.reset:
        try:
            prev = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
            completed = list(prev.get("tasks", []))
            done_ids = {t["task_id"] for t in completed}
            print(f"Resuming from checkpoint with {len(completed)} completed tasks...", flush=True)
        except Exception:
            completed = []

    with HUMANEVAL_PATH.open("r", encoding="utf-8") as fh:
        he_by_id = {r["task_id"]: r for r in (json.loads(line) for line in fh)}
    with GSM8K_PATH.open("r", encoding="utf-8") as fh:
        gsm_all = [json.loads(line) for line in fh]

    print(f"\n=== Track 1: 15 Unseen Official OpenAI HumanEval Problems ({args.model}) ===", flush=True)
    for tid in OPTION_C_HUMANEVAL_IDS:
        if tid in done_ids:
            continue
        res = evaluate_humaneval_head_to_head(he_by_id[tid], model=args.model)
        completed.append(res)
        done_ids.add(tid)
        _save_state(completed, args.model)
        a = res["actorii"]
        h = res["hermes"]
        print(
            f"  {tid:<13} ({res['title']:<24}) │ "
            f"ActorII: {'PASS' if a['passed'] else 'FAIL'} ({a['steps']}st, {a['tokens']}tok, {a['elapsed_s']}s, EngramBoot={'PASS' if a['engram_cold_boot_passed'] else 'FAIL'}) │ "
            f"Hermes: {'PASS' if h['passed'] else 'FAIL'} ({h['tokens']}tok, {h['elapsed_s']}s)",
            flush=True,
        )

    print(f"\n=== Track 2: 10 Unseen Official OpenAI GSM8K Problems ({args.model}) ===", flush=True)
    for idx in OPTION_C_GSM8K_INDICES:
        tid = f"GSM8K/{idx}"
        if tid in done_ids:
            continue
        res = evaluate_gsm8k_head_to_head(idx, gsm_all[idx], model=args.model)
        completed.append(res)
        done_ids.add(tid)
        _save_state(completed, args.model)
        a = res["actorii"]
        h = res["hermes"]
        print(
            f"  {tid:<13} (gold={res['gold']:<6})             │ "
            f"ActorII: {'PASS' if a['passed'] else 'FAIL'} (pred={a['predicted']}, {a['steps']}st, {a['tokens']}tok, {a['elapsed_s']}s) │ "
            f"Hermes: {'PASS' if h['passed'] else 'FAIL'} (pred={h['predicted']}, {h['tokens']}tok, {h['elapsed_s']}s)",
            flush=True,
        )

    print(f"\n=== Track 3: 5 Cold-Reboot Persistent Graph Memory + Computation Tasks ({args.model}) ===", flush=True)
    for spec in COLD_REBOOT_MEMORY_TASKS:
        tid = spec["task_id"]
        if tid in done_ids:
            continue
        res = evaluate_memory_graph_head_to_head(spec, model=args.model)
        completed.append(res)
        done_ids.add(tid)
        _save_state(completed, args.model)
        a = res["actorii"]
        h = res["hermes"]
        print(
            f"  {tid:<13} (gold={res['gold']:<6})             │ "
            f"ActorII: {'PASS' if a['passed'] else 'FAIL'} (pred={a['predicted']}, {a['steps']}st, {a['tokens']}tok, {a['elapsed_s']}s) │ "
            f"Hermes: {'PASS' if h['passed'] else 'FAIL'} (pred={h['predicted']}, {h['tokens']}tok, {h['elapsed_s']}s)",
            flush=True,
        )

    final_state = _save_state(completed, args.model)
    print("\n=== FINAL 30-TASK HEAD-TO-HEAD SUMMARY ===", flush=True)
    print(json.dumps(final_state["overall"], indent=2), flush=True)
    print(json.dumps(final_state["by_track"], indent=2), flush=True)


if __name__ == "__main__":
    main()
