"""In-Image Agentic Benchmark Suite for ActorII (`actorii.harness.benchmark`).

Exposes `ActorBenchSuite` into the live VM object pool as `Actor[#Bench]` via
`@actor_bridge`, providing 5 objective, ground-truth verifiable agentic tasks
modeled on HumanEval (algorithm synthesis), AgentBench (stateful OOP modeling),
SWE-bench (live hot-patch bug-fixing without losing instance state), and GAIA
(in-memory data scrubbing):

1. `alg_primes`       : Synthesize prime summation class & compute sum of primes <= 30 (129).
2. `oop_account`      : Build `BankAccount` with overdraft protection & run transaction sequence (120).
3. `bugfix_hotpatch`  : Inspect a broken live `BuggyThermostat` instance & hot-patch `needsCooling` in RAM.
4. `data_pipeline`    : Filter sensor outliers (<0 or >100) from `Actor[#SensorFeed]` & compute clean mean (22).
5. `vault_state`      : Extend `VaultBox` with `unlockWith(self, pin)` & unlock live `Actor[#Vault]`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorObject, Symbol


@dataclass
class BenchTaskSpec:
    task_id: str
    category: str
    title: str
    goal_prompt: str
    max_steps: int = 5


BENCHMARK_TASKS: Dict[str, BenchTaskSpec] = {
    "alg_primes": BenchTaskSpec(
        task_id="alg_primes",
        category="HumanEval / Algorithm Synthesis",
        title="Prime Summation up to N=30",
        goal_prompt=(
            "Define a 1993 Actor class `PrimeCalc` with `inherit(Object, #PrimeCalc, #(#limit), 1, nil)!!`, "
            "compute the sum of all prime numbers <= 30 (which are 2,3,5,7,11,13,17,19,23,29), "
            "store the exact integer sum in `Actor[#PrimeSum]` and `storeScratch(Actor[#Harness], #primeSum, sum)!!`, and finish."
        ),
        max_steps=5,
    ),
    "oop_account": BenchTaskSpec(
        task_id="oop_account",
        category="AgentBench / Stateful OOP",
        title="BankAccount Overdraft Protection",
        goal_prompt=(
            "Define class `BankAccount` with ivars `#(#balance #txCount)`, `Def init(self, b) { self.balance := b; self.txCount := 0; ^self; }!!`, "
            "`Def deposit(self, amt) { self.balance := self.balance + amt; self.txCount := self.txCount + 1; ^self.balance; }!!`, "
            "and `Def withdraw(self, amt) { if amt <= self.balance then self.balance := self.balance - amt; self.txCount := self.txCount + 1; endif; ^self.balance; }!!`. "
            "Instantiate `Actor[#Acct] := init(new(BankAccount), 100)!!`, call `deposit(Actor[#Acct], 50)!!`, "
            "call `withdraw(Actor[#Acct], 200)!!` (must be rejected), call `withdraw(Actor[#Acct], 30)!!`, "
            "store `Actor[#Acct]`'s final balance (120) in `storeScratch(Actor[#Harness], #finalBalance, 120)!!`, and finish."
        ),
        max_steps=5,
    ),
    "bugfix_hotpatch": BenchTaskSpec(
        task_id="bugfix_hotpatch",
        category="SWE-bench / Live Image Hot-Patching",
        title="Hot-Patch Broken Thermostat Method in RAM",
        goal_prompt=(
            "A live instance `Actor[#RoomStat]` of class `BuggyThermostat` (ivars `#(#currentTemp #targetTemp)`) "
            "is already in the pool with `currentTemp=260` and `targetTemp=220`, but its `needsCooling(self)` method "
            "has an inverted bug (`self.currentTemp < self.targetTemp`). Without recreating `Actor[#RoomStat]`, "
            "hot-patch `BuggyThermostat` using `now(BuggyThermostat)!! Def needsCooling(self) { ^(self.currentTemp > self.targetTemp); }!!`, "
            "verify `needsCooling(Actor[#RoomStat])` returns `true`, store it via `storeScratch(Actor[#Harness], #coolingFixed, needsCooling(Actor[#RoomStat]))!!`, and finish."
        ),
        max_steps=5,
    ),
    "data_pipeline": BenchTaskSpec(
        task_id="data_pipeline",
        category="GAIA / In-Memory Data Scrubbing",
        title="Scrub Sensor Outliers & Compute Mean",
        goal_prompt=(
            "`Actor[#SensorFeed]` is an Array in the pool: `#(18, 22, 999, 24, -50, 20, 26, 22)`. "
            "Valid sensor readings are between 0 and 100 inclusive (so 999 and -50 are invalid outliers). "
            "Compute the integer average of the 6 valid readings ((18+22+24+20+26+22)/6 = 22), "
            "store it in `Actor[#CleanAvg] := 22!!` and `storeScratch(Actor[#Harness], #cleanAvg, 22)!!`, and finish."
        ),
        max_steps=5,
    ),
    "vault_state": BenchTaskSpec(
        task_id="vault_state",
        category="OOP Capability & Guardrail Compliance",
        title="Extend Live VaultBox Class & Unlock Instance",
        goal_prompt=(
            "Class `VaultBox` (ivars `#(#pin #locked)`) and live instance `Actor[#Vault]` (`pin=7749`, `locked=true`) "
            "are in the pool. Add method `now(VaultBox)!! Def unlockWith(self, attempt) { if attempt == self.pin then self.locked := false; endif; ^self.locked; }!!`, "
            "call `unlockWith(Actor[#Vault], 7749)!!`, store `locked(Actor[#Vault])` (`false`) via "
            "`storeScratch(Actor[#Harness], #vaultLocked, locked(Actor[#Vault]))!!`, and finish."
        ),
        max_steps=5,
    ),
}


def _norm_id(task_id: Any) -> str:
    if isinstance(task_id, Symbol):
        return task_id.name
    return str(task_id or "").strip().lstrip("#")


@actor_bridge(name="ActorBenchSuite")
class ActorBenchSuite:
    """In-image benchmark harness exposed at `Actor[#Bench]`."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm
        self.results: Dict[str, Dict[str, Any]] = {}

    @actor_query
    def listTasks(self) -> str:
        """List all 5 benchmark tasks and their categories."""
        lines = ["ActorII Agentic Benchmark Suite (5 tasks):"]
        for tid, spec in BENCHMARK_TASKS.items():
            status = self.results.get(tid, {}).get("passed")
            badge = "PASS" if status is True else ("FAIL" if status is False else "PENDING")
            lines.append(f"  - #{tid} [{spec.category}]: {spec.title} ({badge})")
        return "\n".join(lines)

    @actor_query
    def scoreboard(self) -> str:
        """Return ActorII / Bilby's official verified benchmark results from `benchmarks/BENCHMARK_RESULTS.md` and `scoreboard.json`."""
        import json
        from pathlib import Path

        bench_dir = Path(__file__).resolve().parents[3] / "benchmarks"
        out_lines: List[str] = []
        md_path = bench_dir / "BENCHMARK_RESULTS.md"
        sb_path = bench_dir / "scoreboard.json"

        if md_path.is_file():
            try:
                md_text = md_path.read_text(encoding="utf-8", errors="replace").strip()
                out_lines.append(md_text[:3500])
            except Exception:
                pass

        if sb_path.is_file():
            try:
                sb_data = json.loads(sb_path.read_text(encoding="utf-8", errors="replace"))
                runs = sb_data.get("runs") if isinstance(sb_data, dict) else sb_data
                if isinstance(runs, list) and runs:
                    out_lines.append("\nRecent Recorded Scoreboard Runs (`scoreboard.json`):")
                    for r in runs[-8:]:
                        if isinstance(r, dict):
                            suite = r.get("suite") or r.get("benchmark") or "suite"
                            model = r.get("model") or ""
                            score = r.get("score_pct") or r.get("pass_rate") or r.get("accuracy") or r.get("summary") or ""
                            ts = r.get("timestamp") or r.get("date") or ""
                            out_lines.append(f"  - {suite} ({model}) [{ts}]: {score}")
            except Exception:
                pass

        if self.results:
            out_lines.append("\n" + self.scoreReport())

        if not out_lines:
            return self.listTasks()
        return "\n".join(out_lines)

    @actor_query
    def officialResults(self) -> str:
        """Alias for `scoreboard(Actor[#Bench])`."""
        return self.scoreboard()

    @actor_query
    def summary(self) -> str:
        """Return a concise summary of both official ActorII benchmarks and the 5 in-image kata tasks."""
        return self.scoreboard()

    @actor_query
    def scoreReport(self) -> str:
        """Return a summary table of all benchmark task results recorded so far."""
        if not self.results:
            return "No benchmark tasks executed yet. Run `/bench` or `runAll(Actor[#Bench])!!`."
        passed_cnt = sum(1 for r in self.results.values() if r.get("passed"))
        total_cnt = len(self.results)
        total_gas = sum(int(r.get("gas_used", 0)) for r in self.results.values())
        lines = [
            f"=== ActorII Agentic Benchmark Report: {passed_cnt}/{total_cnt} PASSED (totalGas={total_gas} ops) ==="
        ]
        for tid, r in self.results.items():
            mark = "PASS" if r.get("passed") else "FAIL"
            lines.append(
                f"  [{mark}] #{tid:<16} | steps={r.get('steps', 0)} | gas={r.get('gas_used', 0):<5} | {r.get('detail', '')}"
            )
        return "\n".join(lines)

    @actor_command
    def setupTask(self, task_id: Any) -> str:
        """Seed the live Actor object pool with the initial state required for `task_id`."""
        tid = _norm_id(task_id)
        if tid not in BENCHMARK_TASKS:
            raise ValueError(f"Unknown benchmark task '{tid}'. Available: {sorted(BENCHMARK_TASKS.keys())}")

        if tid == "bugfix_hotpatch":
            self.vm.eval_chunk(
                """
                inherit(Object, #BuggyThermostat, #(#currentTemp #targetTemp), 2, nil)!!
                now(BuggyThermostat)!!
                Def init(self, cur, tgt) {
                    self.currentTemp := cur;
                    self.targetTemp := tgt;
                    ^self;
                }!!
                Def needsCooling(self) {
                    /* BUG: inverted comparison operator */
                    ^(self.currentTemp < self.targetTemp);
                }!!
                Actor[#RoomStat] := init(new(BuggyThermostat), 260, 220)!!
                """
            )
        elif tid == "data_pipeline":
            self.vm.eval_chunk("Actor[#SensorFeed] := #(18, 22, 999, 24, -50, 20, 26, 22)!!")
        elif tid == "vault_state":
            self.vm.eval_chunk(
                """
                inherit(Object, #VaultBox, #(#pin #locked), 2, nil)!!
                now(VaultBox)!!
                Def init(self, p) {
                    self.pin := p;
                    self.locked := true;
                    ^self;
                }!!
                Def locked(self) { ^self.locked; }!!
                Actor[#Vault] := init(new(VaultBox), 7749)!!
                """
            )
        return f"READY:{tid}"

    @actor_query
    def verifyTask(self, task_id: Any) -> str:
        """Verify ground-truth postconditions in the live object pool for `task_id`."""
        tid = _norm_id(task_id)
        actor_dict = self.vm.global_vars.get("Actor", {})
        harness = actor_dict.get("Harness")
        scratch: Dict[str, Any] = {}
        if harness is not None:
            sc_obj = harness._fields.get("scratchpad")
            if isinstance(sc_obj, Dictionary):
                for k, v in sc_obj._entries.items():
                    nk = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                    scratch[nk] = v

        passed = False
        detail = ""

        if tid == "alg_primes":
            val = scratch.get("primeSum") or actor_dict.get("PrimeSum")
            passed = (val == 129)
            detail = f"primeSum={val!r} (expected 129)"

        elif tid == "oop_account":
            acct = actor_dict.get("Acct")
            bal = acct._fields.get("balance") if isinstance(acct, ActorObject) else scratch.get("finalBalance")
            passed = (bal == 120)
            detail = f"Actor[#Acct].balance={bal!r} (expected 120)"

        elif tid == "bugfix_hotpatch":
            stat = actor_dict.get("RoomStat")
            try:
                cooling = self.vm.eval_chunk("needsCooling(Actor[#RoomStat])!!")
            except Exception:
                cooling = None
            cur_t = stat._fields.get("currentTemp") if isinstance(stat, ActorObject) else None
            passed = (cooling is True and cur_t == 260)
            detail = f"needsCooling(Actor[#RoomStat])={cooling!r}, currentTemp={cur_t!r}"

        elif tid == "data_pipeline":
            val = scratch.get("cleanAvg") or actor_dict.get("CleanAvg")
            passed = (val == 22)
            detail = f"cleanAvg={val!r} (expected 22)"

        elif tid == "vault_state":
            vault = actor_dict.get("Vault")
            locked_val = vault._fields.get("locked") if isinstance(vault, ActorObject) else None
            passed = locked_val is not None and (locked_val is False or locked_val == 0)
            detail = f"Actor[#Vault].locked={locked_val!r} (unlocked)"

        else:
            detail = f"Unknown task {tid}"

        return f"{'PASS' if passed else 'FAIL'}:{detail}"

    @actor_command
    def runTask(self, task_id: Any) -> str:
        """Seed, execute via `Actor[#Harness]`, and verify a single benchmark task."""
        tid = _norm_id(task_id)
        if tid not in BENCHMARK_TASKS:
            raise ValueError(f"Unknown benchmark task '{tid}'")
        spec = BENCHMARK_TASKS[tid]
        self.setupTask(tid)

        harness = self.vm.global_vars["Actor"]["Harness"]
        outer_goal = harness._fields.get("goal", "")
        outer_max_steps = harness._fields.get("maxSteps", 8)
        if hasattr(harness, "clearHistory"):
            harness.clearHistory()
        gas_before = int(harness._fields.get("totalGasUsed", 0) or 0)
        t0 = time.time()

        escaped_goal = spec.goal_prompt.replace("\\", "\\\\").replace('"', '\\"')
        self.vm.eval_chunk(f'runGoal(Actor[#Harness], "{escaped_goal}", {spec.max_steps})!!')

        elapsed = time.time() - t0
        gas_used = int(harness._fields.get("totalGasUsed", 0) or 0) - gas_before
        steps = int(harness._fields.get("stepCount", 0) or 0)

        verdict = self.verifyTask(tid)
        passed = verdict.startswith("PASS:")
        detail = verdict.split(":", 1)[1] if ":" in verdict else verdict

        self.results[tid] = {
            "task_id": tid,
            "title": spec.title,
            "category": spec.category,
            "passed": passed,
            "steps": steps,
            "gas_used": gas_used,
            "elapsed_s": round(elapsed, 2),
            "detail": detail,
        }
        summary_line = f"[{'PASS' if passed else 'FAIL'}] #{tid}: {detail} (steps={steps}, gas={gas_used} ops, {elapsed:.2f}s)"
        harness._fields["goal"] = outer_goal
        harness._fields["maxSteps"] = outer_max_steps
        harness._fields["status"] = Symbol("complete")
        harness._fields["lastResult"] = summary_line
        return summary_line

    @actor_command
    def runAll(self) -> str:
        """Run all 5 benchmark tasks sequentially through `Actor[#Harness]` and return the score report."""
        harness = self.vm.global_vars["Actor"]["Harness"]
        outer_goal = harness._fields.get("goal", "")
        outer_max_steps = harness._fields.get("maxSteps", 8)
        for tid in BENCHMARK_TASKS.keys():
            self.runTask(tid)
        report = self.scoreReport()
        harness._fields["goal"] = outer_goal
        harness._fields["maxSteps"] = outer_max_steps
        harness._fields["status"] = Symbol("complete")
        harness._fields["lastResult"] = report
        return report

    @actor_query
    def selfEvoSuites(self) -> str:
        """Return the catalogue of built-in deterministic self-evolution benchmark specs."""
        from actorii.harness.self_evolution_bench import list_self_evolution_suites

        return list_self_evolution_suites()

    @actor_command
    def runSelfEvo(self, spec_id: Any, mode: Any = "scripted") -> str:
        """Run a self-evolution benchmark spec (`specId`) in `scripted` or `live` mode inside an isolated sandbox VM."""
        from actorii.harness.self_evolution_bench import run_self_evolution_spec, summarize_result

        sid = _norm_id(spec_id)
        run_mode = _norm_id(mode) or "scripted"
        result = run_self_evolution_spec(sid, mode=run_mode)
        return summarize_result(result)

    @actor_query
    def selfEvoScoreboard(self) -> str:
        """Run the scripted self-evolution benchmark suite in an isolated sandbox and return a Markdown scoreboard."""
        from actorii.harness.self_evolution_bench import render_markdown, run_self_evolution_suite

        report = run_self_evolution_suite(specs="all", mode="scripted")
        return render_markdown(report)
