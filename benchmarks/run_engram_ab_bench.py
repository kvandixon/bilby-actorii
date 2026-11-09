#!/usr/bin/env python3
"""Empirical A/B/C Benchmark: Active Engrams (Memory+Skill) vs Passive Graph-RAG vs RAM-Only OOP.

Evaluates whether storing executable 1993 Actor 4.1 bytecode skills (`:ActorEngram`) alongside
Neo4j subgraphs produces a measurable Gain of Function (accuracy, step reduction, and amortized
token/latency savings across cold VM reboots) compared to:
- Condition A: RAM-Only OOP Harness (no Neo4j persistence across cold VM restarts)
- Condition B: Passive Graph-RAG (`cypherQuery` / `cypherWrite` enabled, NO stored `:ActorEngram` skills)
- Condition C: Active Engrams (`cypherQuery` / `cypherWrite` + `:ActorEngram` synthesis & cold-boot hydration)

Covers 4 multi-step procedural graph tasks x 3 parameterized queries after a cold VM reboot (12 queries/condition):
  1. `recipe_efficiency` (Workflow 1: Existing Graph)  — Multi-constraint protein-per-dollar recipe scoring
  2. `service_cascade`   (Workflow 1: Existing Graph)  — Hardware/service cluster spike risk rollup
  3. `thermal_cooling`   (Workflow 2: Chat-Ingested)   — Server zone iterative cooling curve simulation
  4. `contractor_ledger` (Workflow 2: Chat-Ingested)   — Tiered overtime payroll rollup from chat facts
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from actorii.core import Symbol
from actorii.harness import bootstrap_agent_harness
from actorii.harness.memory_graph import ActorMemoryGraph
from actorii.vm import ActorVM

RESULTS_PATH = Path(__file__).resolve().parent / "ENGRAM_AB_RESULTS.json"
REPORT_PATH = Path(__file__).resolve().parent / "ENGRAM_AB_REPORT.md"
BENCH_NAMESPACE = "engram_ab_bench"


@dataclass
class QuerySpec:
    query_id: str
    prompt: str
    expected: int


@dataclass
class TaskSpec:
    task_id: str
    workflow: str
    title: str
    seed_cypher: Optional[str]
    ingest_prompt_ram: str
    ingest_prompt_passive_graph: str
    ingest_prompt_active_engram: str
    queries: List[QuerySpec]


TASKS: List[TaskSpec] = [
    # -------------------------------------------------------------------------
    # Task 1 (Workflow 1: Existing Neo4j Graph -> Procedural Multi-Constraint Scoring)
    # -------------------------------------------------------------------------
    TaskSpec(
        task_id="recipe_efficiency",
        workflow="Workflow 1: Existing Graph",
        title="Multi-Constraint Recipe Efficiency Scorer",
        seed_cypher=(
            "CREATE "
            "(:BenchRecipe {name: 'LentilBowl', protein: 24, cost: 4, prep_mins: 20, allergen: 'none'}), "
            "(:BenchRecipe {name: 'PeanutNoodles', protein: 18, cost: 3, prep_mins: 15, allergen: 'nuts'}), "
            "(:BenchRecipe {name: 'WheyShake', protein: 30, cost: 5, prep_mins: 5, allergen: 'dairy'}), "
            "(:BenchRecipe {name: 'SeitanStirFry', protein: 28, cost: 4, prep_mins: 25, allergen: 'gluten'}), "
            "(:BenchRecipe {name: 'TofuCurry', protein: 20, cost: 5, prep_mins: 35, allergen: 'none'}), "
            "(:BenchRecipe {name: 'BeanChili', protein: 21, cost: 3, prep_mins: 40, allergen: 'none'})"
        ),
        ingest_prompt_ram=(
            "Store these 6 recipes in an Actor object or class in RAM: "
            "LentilBowl(protein=24, cost=4, prep_mins=20, allergen='none'), "
            "PeanutNoodles(protein=18, cost=3, prep_mins=15, allergen='nuts'), "
            "WheyShake(protein=30, cost=5, prep_mins=5, allergen='dairy'), "
            "SeitanStirFry(protein=28, cost=4, prep_mins=25, allergen='gluten'), "
            "TofuCurry(protein=20, cost=5, prep_mins=35, allergen='none'), "
            "BeanChili(protein=21, cost=3, prep_mins=40, allergen='none'). "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_passive_graph="",  # Already seeded in Neo4j; Passive Graph-RAG queries on demand
        ingest_prompt_active_engram=(
            "Neo4j contains `:BenchRecipe` nodes with properties `name`, `protein`, `cost`, `prep_mins`, and `allergen`. "
            "Define a 1993 Actor class `RecipeScorer` with method `Def scoreRecipes(self, maxPrep, badAllergen | rows, i, row, p, c, prep, alg, total)` "
            "that queries `:BenchRecipe` nodes from `Actor[#Memory]` via `cypherQuery`, sums `((protein * 10) / cost)` for all recipes where `prep_mins <= maxPrep` and `allergen <> badAllergen`, "
            "instantiates `Actor[#RecipeScorer] := new(RecipeScorer)!!`, and persists it to Neo4j via `persistEngram(Actor[#Memory], #RecipeScorer, \"Score BenchRecipe efficiency by maxPrep and badAllergen\")!!`. "
            "Verify `scoreRecipes(Actor[#RecipeScorer], 30, \"nuts\")`, store it in `storeScratch(Actor[#Harness], #ready, scoreRecipes(Actor[#RecipeScorer], 30, \"nuts\"))!!`, and reply COMPLETE."
        ),
        queries=[
            # Matching for (maxPrep=30, badAllergen='nuts'):
            # LentilBowl (20<=30, 'none'<>'nuts'): (24*10)/4 = 60
            # WheyShake (5<=30, 'dairy'<>'nuts'): (30*10)/5 = 60
            # SeitanStirFry (25<=30, 'gluten'<>'nuts'): (28*10)/4 = 70
            # Total = 60 + 60 + 70 = 190
            QuerySpec(
                query_id="q1_prep30_no_nuts",
                prompt=(
                    "Using the `:BenchRecipe` nodes in Neo4j (properties: `protein`, `cost`, `prep_mins`, `allergen`) or any hydrated Engram in the pool, "
                    "compute the sum of `((protein * 10) / cost)` (integer division) for all recipes where `prep_mins <= 30` and `allergen <> \"nuts\"`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=190,
            ),
            # Matching for (maxPrep=20, badAllergen='dairy'):
            # LentilBowl (20<=20, 'none'): 60
            # PeanutNoodles (15<=20, 'nuts'): (18*10)/3 = 60
            # Total = 120
            QuerySpec(
                query_id="q2_prep20_no_dairy",
                prompt=(
                    "Using the `:BenchRecipe` nodes in Neo4j (properties: `protein`, `cost`, `prep_mins`, `allergen`) or any hydrated Engram in the pool, "
                    "compute the sum of `((protein * 10) / cost)` (integer division) for all recipes where `prep_mins <= 20` and `allergen <> \"dairy\"`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=120,
            ),
            # Matching for (maxPrep=40, badAllergen='gluten'):
            # LentilBowl (60) + PeanutNoodles (60) + WheyShake (60) + TofuCurry ((20*10)/5 = 40) + BeanChili ((21*10)/3 = 70) = 290
            QuerySpec(
                query_id="q3_prep40_no_gluten",
                prompt=(
                    "Using the `:BenchRecipe` nodes in Neo4j (properties: `protein`, `cost`, `prep_mins`, `allergen`) or any hydrated Engram in the pool, "
                    "compute the sum of `((protein * 10) / cost)` (integer division) for all recipes where `prep_mins <= 40` and `allergen <> \"gluten\"`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=290,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Task 2 (Workflow 1: Existing Neo4j Graph -> Conditional Spike Risk Rollup)
    # -------------------------------------------------------------------------
    TaskSpec(
        task_id="service_cascade",
        workflow="Workflow 1: Existing Graph",
        title="Service Cluster Spike Risk Calculator",
        seed_cypher=(
            "CREATE "
            "(:BenchService {name: 'AuthAPI', base_ms: 40, weight: 3, cluster: 'core'}), "
            "(:BenchService {name: 'LedgerDB', base_ms: 60, weight: 4, cluster: 'core'}), "
            "(:BenchService {name: 'CacheNode', base_ms: 20, weight: 2, cluster: 'core'}), "
            "(:BenchService {name: 'EdgeProxy', base_ms: 35, weight: 2, cluster: 'edge'}), "
            "(:BenchService {name: 'CDNWorker', base_ms: 55, weight: 3, cluster: 'edge'})"
        ),
        ingest_prompt_ram=(
            "Store these 5 services in RAM: "
            "AuthAPI(base_ms=40, weight=3, cluster='core'), "
            "LedgerDB(base_ms=60, weight=4, cluster='core'), "
            "CacheNode(base_ms=20, weight=2, cluster='core'), "
            "EdgeProxy(base_ms=35, weight=2, cluster='edge'), "
            "CDNWorker(base_ms=55, weight=3, cluster='edge'). "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_passive_graph="",
        ingest_prompt_active_engram=(
            "Neo4j contains `:BenchService` nodes with properties `name`, `base_ms`, `weight`, and `cluster`. "
            "Define a 1993 Actor class `ClusterRisk` with method `Def evalRisk(self, clusterName, spikeFactor | rows, i, row, b, w, c, spiked, total)` "
            "that queries `:BenchService` nodes via `cypherQuery(Actor[#Memory], ...)`, and for each service where `cluster == clusterName`, lets `spiked := base_ms * spikeFactor` and adds `(spiked * weight)` if `spiked > 100` else `(base_ms * weight)`. "
            "Instantiate `Actor[#ClusterRisk] := new(ClusterRisk)!!`, persist it to Neo4j via `persistEngram(Actor[#Memory], #ClusterRisk, \"Compute cluster spike risk\")!!`, "
            "store `evalRisk(Actor[#ClusterRisk], \"core\", 2)` in `storeScratch(Actor[#Harness], #ready, evalRisk(Actor[#ClusterRisk], \"core\", 2))!!`, and reply COMPLETE."
        ),
        queries=[
            # core with spikeFactor=2:
            # AuthAPI: 40*2=80 (<=100) -> 40*3 = 120
            # LedgerDB: 60*2=120 (>100) -> 120*4 = 480
            # CacheNode: 20*2=40 (<=100) -> 20*2 = 40
            # Total = 640
            QuerySpec(
                query_id="q1_core_spike2",
                prompt=(
                    "For all `:BenchService` nodes in Neo4j (properties: `base_ms`, `weight`, `cluster`) where `cluster == \"core\"` with `spikeFactor = 2`: "
                    "for each service, let `spiked = base_ms * spikeFactor`. If `spiked > 100`, add `(spiked * weight)` to the total; otherwise add `(base_ms * weight)`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=640,
            ),
            # core with spikeFactor=3:
            # AuthAPI: 40*3=120 (>100) -> 120*3 = 360
            # LedgerDB: 60*3=180 (>100) -> 180*4 = 720
            # CacheNode: 20*3=60 (<=100) -> 20*2 = 40
            # Total = 1120
            QuerySpec(
                query_id="q2_core_spike3",
                prompt=(
                    "For all `:BenchService` nodes in Neo4j (properties: `base_ms`, `weight`, `cluster`) where `cluster == \"core\"` with `spikeFactor = 3`: "
                    "for each service, let `spiked = base_ms * spikeFactor`. If `spiked > 100`, add `(spiked * weight)` to the total; otherwise add `(base_ms * weight)`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=1120,
            ),
            # edge with spikeFactor=2:
            # EdgeProxy: 35*2=70 (<=100) -> 35*2 = 70
            # CDNWorker: 55*2=110 (>100) -> 110*3 = 330
            # Total = 400
            QuerySpec(
                query_id="q3_edge_spike2",
                prompt=(
                    "For all `:BenchService` nodes in Neo4j (properties: `base_ms`, `weight`, `cluster`) where `cluster == \"edge\"` with `spikeFactor = 2`: "
                    "for each service, let `spiked = base_ms * spikeFactor`. If `spiked > 100`, add `(spiked * weight)` to the total; otherwise add `(base_ms * weight)`. "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=400,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Task 3 (Workflow 2: Chat-Ingested Facts + Iterative Cooling Simulation)
    # -------------------------------------------------------------------------
    TaskSpec(
        task_id="thermal_cooling",
        workflow="Workflow 2: Chat-Ingested Memory+Skill",
        title="Server Zone Iterative Cooling Curve Memory",
        seed_cypher=None,
        ingest_prompt_ram=(
            "Remember these 3 server zones in RAM: "
            "ZoneA(start_temp=92, drop_per_min=3), "
            "ZoneB(start_temp=85, drop_per_min=2), "
            "ZoneC(start_temp=104, drop_per_min=4). "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_passive_graph=(
            "Persist these 3 server thermal zones in Neo4j using `cypherWrite(Actor[#Memory], ...)` as `:BenchZone` nodes: "
            "`CREATE (:BenchZone {name: 'ZoneA', start_temp: 92, drop_per_min: 3}), "
            "(:BenchZone {name: 'ZoneB', start_temp: 85, drop_per_min: 2}), "
            "(:BenchZone {name: 'ZoneC', start_temp: 104, drop_per_min: 4})`. "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_active_engram=(
            "1) Persist these 3 server thermal zones in Neo4j via `cypherWrite(Actor[#Memory], "
            "\"CREATE (:BenchZone {name: 'ZoneA', start_temp: 92, drop_per_min: 3}), "
            "(:BenchZone {name: 'ZoneB', start_temp: 85, drop_per_min: 2}), "
            "(:BenchZone {name: 'ZoneC', start_temp: 104, drop_per_min: 4})\")!!`. "
            "2) Define a 1993 Actor class `ThermalSim` with method `Def minutesToCool(self, zoneName, targetTemp | rows, i, row, cur, drop, mins)` "
            "that queries `:BenchZone` nodes via `cypherQuery(Actor[#Memory], ...)` and counts how many integer minutes it takes `zoneName` to drop from `start_temp` by `drop_per_min` each minute until `cur <= targetTemp`. "
            "Instantiate `Actor[#ThermalSim] := new(ThermalSim)!!`, persist it to Neo4j via `persistEngram(Actor[#Memory], #ThermalSim, \"Simulate server zone cooling minutes\")!!`, "
            "store `minutesToCool(Actor[#ThermalSim], \"ZoneA\", 74)` in `storeScratch(Actor[#Harness], #ready, minutesToCool(Actor[#ThermalSim], \"ZoneA\", 74))!!`, and reply COMPLETE."
        ),
        queries=[
            # ZoneA: start=92, drop=3, target=74 -> (92-74)/3 = 6 minutes
            QuerySpec(
                query_id="q1_zoneA_to_74",
                prompt=(
                    "Using our remembered `:BenchZone` thermal data (properties: `name`, `start_temp`, `drop_per_min`) or hydrated `#ThermalSim` Engram, "
                    "how many integer minutes does it take for `\"ZoneA\"` to cool until its temperature is `<= 74` (dropping by `drop_per_min` each minute from `start_temp`)? "
                    "Store the exact integer minutes in `storeScratch(Actor[#Harness], #answer, mins)!!` and reply COMPLETE."
                ),
                expected=6,
            ),
            # ZoneB: start=85, drop=2, target=75 -> (85-75)/2 = 5 minutes
            QuerySpec(
                query_id="q2_zoneB_to_75",
                prompt=(
                    "Using our remembered `:BenchZone` thermal data (properties: `name`, `start_temp`, `drop_per_min`) or hydrated `#ThermalSim` Engram, "
                    "how many integer minutes does it take for `\"ZoneB\"` to cool until its temperature is `<= 75` (dropping by `drop_per_min` each minute from `start_temp`)? "
                    "Store the exact integer minutes in `storeScratch(Actor[#Harness], #answer, mins)!!` and reply COMPLETE."
                ),
                expected=5,
            ),
            # ZoneC: start=104, drop=4, target=80 -> (104-80)/4 = 6 minutes
            QuerySpec(
                query_id="q3_zoneC_to_80",
                prompt=(
                    "Using our remembered `:BenchZone` thermal data (properties: `name`, `start_temp`, `drop_per_min`) or hydrated `#ThermalSim` Engram, "
                    "how many integer minutes does it take for `\"ZoneC\"` to cool until its temperature is `<= 80` (dropping by `drop_per_min` each minute from `start_temp`)? "
                    "Store the exact integer minutes in `storeScratch(Actor[#Harness], #answer, mins)!!` and reply COMPLETE."
                ),
                expected=6,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Task 4 (Workflow 2: Chat-Ingested Ledger + Tiered Overtime Rollup)
    # -------------------------------------------------------------------------
    TaskSpec(
        task_id="contractor_ledger",
        workflow="Workflow 2: Chat-Ingested Memory+Skill",
        title="Contractor Tiered Overtime Payroll Memory",
        seed_cypher=None,
        ingest_prompt_ram=(
            "Remember these 3 contractors in RAM: "
            "Alice(rate=80, std_cap=40), Bob(rate=60, std_cap=35), Carol(rate=100, std_cap=40). "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_passive_graph=(
            "Persist these 3 contractors in Neo4j via `cypherWrite(Actor[#Memory], ...)` as `:BenchContractor` nodes: "
            "`CREATE (:BenchContractor {name: 'Alice', rate: 80, std_cap: 40}), "
            "(:BenchContractor {name: 'Bob', rate: 60, std_cap: 35}), "
            "(:BenchContractor {name: 'Carol', rate: 100, std_cap: 40})`. "
            "Store `true` in `storeScratch(Actor[#Harness], #ready, true)!!` and finish."
        ),
        ingest_prompt_active_engram=(
            "1) Persist these 3 contractors in Neo4j via `cypherWrite(Actor[#Memory], "
            "\"CREATE (:BenchContractor {name: 'Alice', rate: 80, std_cap: 40}), "
            "(:BenchContractor {name: 'Bob', rate: 60, std_cap: 35}), "
            "(:BenchContractor {name: 'Carol', rate: 100, std_cap: 40})\")!!`. "
            "2) Define a 1993 Actor class `PayrollCalc` with method `Def totalPayroll(self, hoursWorked | rows, i, row, r, cap, over, total)` "
            "that queries `:BenchContractor` nodes via `cypherQuery(Actor[#Memory], ...)` and computes combined total pay across all contractors for `hoursWorked` (hours up to `std_cap` pay `rate`; overtime hours above `std_cap` pay `rate * 2`). "
            "Instantiate `Actor[#PayrollCalc] := new(PayrollCalc)!!`, persist it to Neo4j via `persistEngram(Actor[#Memory], #PayrollCalc, \"Compute tiered overtime payroll\")!!`, "
            "store `totalPayroll(Actor[#PayrollCalc], 40)` in `storeScratch(Actor[#Harness], #ready, totalPayroll(Actor[#PayrollCalc], 40))!!`, and reply COMPLETE."
        ),
        queries=[
            # hoursWorked = 40:
            # Alice (cap=40, rate=80): 40*80 = 3200
            # Bob (cap=35, rate=60): 35*60 + 5*120 = 2100 + 600 = 2700
            # Carol (cap=40, rate=100): 40*100 = 4000
            # Total = 9900
            QuerySpec(
                query_id="q1_hours40",
                prompt=(
                    "Using our remembered `:BenchContractor` roster (properties: `name`, `rate`, `std_cap`) or hydrated `#PayrollCalc` Engram, "
                    "compute the combined total payroll across all 3 contractors if each works `40` hours "
                    "(hours up to `std_cap` pay `rate`; overtime hours above `std_cap` pay `rate * 2`). "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=9900,
            ),
            # hoursWorked = 45:
            # Alice: 40*80 + 5*160 = 4000
            # Bob: 35*60 + 10*120 = 3300
            # Carol: 40*100 + 5*200 = 5000
            # Total = 12300
            QuerySpec(
                query_id="q2_hours45",
                prompt=(
                    "Using our remembered `:BenchContractor` roster (properties: `name`, `rate`, `std_cap`) or hydrated `#PayrollCalc` Engram, "
                    "compute the combined total payroll across all 3 contractors if each works `45` hours "
                    "(hours up to `std_cap` pay `rate`; overtime hours above `std_cap` pay `rate * 2`). "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=12300,
            ),
            # hoursWorked = 30:
            # Alice: 30*80 = 2400
            # Bob: 30*60 = 1800
            # Carol: 30*100 = 3000
            # Total = 7200
            QuerySpec(
                query_id="q3_hours30",
                prompt=(
                    "Using our remembered `:BenchContractor` roster (properties: `name`, `rate`, `std_cap`) or hydrated `#PayrollCalc` Engram, "
                    "compute the combined total payroll across all 3 contractors if each works `30` hours "
                    "(hours up to `std_cap` pay `rate`; overtime hours above `std_cap` pay `rate * 2`). "
                    "Store the exact integer total in `storeScratch(Actor[#Harness], #answer, total)!!` and reply COMPLETE."
                ),
                expected=7200,
            ),
        ],
    ),
]


def _clean_bench_graph(mem: ActorMemoryGraph) -> None:
    """Clean all benchmark domain nodes and `:ActorEngram` nodes in `BENCH_NAMESPACE`."""
    mem._run_cypher(
        "MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'Bench') DETACH DELETE n"
    )
    mem.clearEngrams()


def _extract_answer_from_vm(vm: ActorVM, harness: Any, expected: int) -> Tuple[bool, Any]:
    """Check `Actor[#Harness].scratchpad` (`#answer` or any stored value) and `lastResult` for `expected`."""
    scratch = harness._fields.get("scratchpad")
    candidates: List[Any] = []
    if scratch is not None and hasattr(scratch, "_entries"):
        for k, v in scratch._entries.items():
            norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
            if norm_k == "answer":
                candidates.insert(0, v)
            elif norm_k != "ready":
                candidates.append(v)

    last_res = harness._fields.get("lastResult")
    if last_res is not None:
        candidates.append(last_res)

    step_log = harness._fields.get("stepLog")
    if step_log is not None:
        for entry in reversed(list(step_log)):
            if isinstance(entry, dict) and entry.get("status") == "ok":
                candidates.append(entry.get("result"))

    for c in candidates:
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            if int(round(c)) == expected:
                return True, int(round(c))
        if isinstance(c, str):
            nums = [int(m) for m in re.findall(r"-?\d+", c)]
            if expected in nums:
                return True, expected

    return False, (candidates[0] if candidates else None)


def _run_harness_turn_with_telemetry(
    vm: ActorVM,
    harness: Any,
    prompt: str,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """Execute `runGoal(Actor[#Harness], prompt, max_steps)!!` while measuring tokens, steps, gas, and latency."""
    prompt_chars = 0
    completion_chars = 0
    llm_calls = 0

    orig_ask = harness.ask

    def _instrumented_ask(p: Any, context_obj: Any = None) -> str:
        nonlocal prompt_chars, completion_chars, llm_calls
        llm_calls += 1
        sys_p = harness.buildContextPrompt()
        prompt_chars += len(str(p)) + len(sys_p)
        reply = orig_ask(p, context_obj=context_obj)
        completion_chars += len(str(reply or ""))
        return reply

    harness.ask = _instrumented_ask
    t0 = time.time()
    try:
        vm.dispatch_message(harness, "runGoal", [prompt, int(max_steps)])
    except Exception as exc:
        print(f"      [TURN EXCEPTION]: {type(exc).__name__}: {exc}")
    finally:
        harness.ask = orig_ask

    elapsed_s = round(time.time() - t0, 2)
    steps_used = int(harness._fields.get("stepCount", 0) or 0)
    gas_used = int(harness._fields.get("totalGasUsed", 0) or 0)
    est_tokens = (prompt_chars + completion_chars) // 4
    return {
        "steps": steps_used,
        "llm_calls": llm_calls,
        "gas_used": gas_used,
        "elapsed_s": elapsed_s,
        "est_tokens": est_tokens,
        "prompt_chars": prompt_chars,
        "completion_chars": completion_chars,
    }


def evaluate_condition(
    condition_id: str,
    condition_label: str,
    *,
    enable_memory_graph: bool,
    enable_engrams: bool,
    model: str = "qwen3.8:latest",
    max_steps: int = 3,
) -> Dict[str, Any]:
    """Run all 4 tasks (Ingest Session + 3 Cold-Start Reboot Queries) under a single condition."""
    print(f"\n======================================================================")
    print(f"RUNNING {condition_id}: {condition_label} (model={model})")
    print(f"======================================================================")

    helper_vm = ActorVM()
    helper_mem = ActorMemoryGraph(helper_vm, enable_engrams=True, namespace=BENCH_NAMESPACE)

    task_reports: List[Dict[str, Any]] = []
    total_queries = 0
    total_passed = 0
    total_query_steps = 0
    total_query_tokens = 0
    total_query_time_s = 0.0
    total_ingest_tokens = 0
    total_ingest_time_s = 0.0

    for task in TASKS:
        _clean_bench_graph(helper_mem)
        if task.seed_cypher:
            helper_mem._run_cypher(task.seed_cypher)

        # ---------------------------------------------------------------------
        # Phase 1: Session 1 (Ingest / Skill Synthesis Before Cold Reboot)
        # ---------------------------------------------------------------------
        if not enable_memory_graph:
            ingest_prompt = task.ingest_prompt_ram
        elif not enable_engrams:
            ingest_prompt = task.ingest_prompt_passive_graph
        else:
            ingest_prompt = task.ingest_prompt_active_engram

        ingest_metrics = {
            "steps": 0,
            "llm_calls": 0,
            "gas_used": 0,
            "elapsed_s": 0.0,
            "est_tokens": 0,
        }
        if ingest_prompt:
            vm_ingest, h_ingest, _sb_ingest = bootstrap_agent_harness(
                model=model,
                max_steps=max_steps,
                enable_memory_graph=enable_memory_graph,
                enable_engrams=enable_engrams,
                memory_namespace=BENCH_NAMESPACE,
            )
            ingest_metrics = _run_harness_turn_with_telemetry(
                vm_ingest, h_ingest, ingest_prompt, max_steps=max_steps
            )
            print(
                f"  [{task.task_id}] Session 1 Ingest/Setup: steps={ingest_metrics['steps']}, "
                f"tokens={ingest_metrics['est_tokens']}, time={ingest_metrics['elapsed_s']}s"
            )

        total_ingest_tokens += ingest_metrics["est_tokens"]
        total_ingest_time_s += ingest_metrics["elapsed_s"]

        # ---------------------------------------------------------------------
        # Phase 2: Session 2 (Cold-Start VM Reboot for Each of the 3 Queries)
        # ---------------------------------------------------------------------
        query_results: List[Dict[str, Any]] = []
        for q in task.queries:
            total_queries += 1
            # Brand-new cold-start ActorVM! All RAM objects from Session 1 are wiped.
            vm_q, h_q, _sb_q = bootstrap_agent_harness(
                model=model,
                max_steps=max_steps,
                enable_memory_graph=enable_memory_graph,
                enable_engrams=enable_engrams,
                memory_namespace=BENCH_NAMESPACE,
            )
            q_metrics = _run_harness_turn_with_telemetry(
                vm_q, h_q, q.prompt, max_steps=max_steps
            )
            passed, actual_val = _extract_answer_from_vm(vm_q, h_q, q.expected)
            if passed:
                total_passed += 1
            total_query_steps += q_metrics["steps"]
            total_query_tokens += q_metrics["est_tokens"]
            total_query_time_s += q_metrics["elapsed_s"]

            badge = "PASS" if passed else "FAIL"
            print(
                f"    -> [{badge}] {task.task_id}.{q.query_id}: expected={q.expected}, actual={actual_val!r} "
                f"| steps={q_metrics['steps']}, tokens={q_metrics['est_tokens']}, time={q_metrics['elapsed_s']}s"
            )
            query_results.append(
                {
                    "query_id": q.query_id,
                    "expected": q.expected,
                    "actual": actual_val,
                    "passed": passed,
                    **q_metrics,
                }
            )

        task_reports.append(
            {
                "task_id": task.task_id,
                "workflow": task.workflow,
                "title": task.title,
                "ingest": ingest_metrics,
                "queries": query_results,
                "task_passed": sum(1 for qr in query_results if qr["passed"]),
                "task_total": len(query_results),
            }
        )

    _clean_bench_graph(helper_mem)

    return {
        "condition_id": condition_id,
        "condition_label": condition_label,
        "model": model,
        "enable_memory_graph": enable_memory_graph,
        "enable_engrams": enable_engrams,
        "summary": {
            "passed": total_passed,
            "total": total_queries,
            "pass_rate_pct": round((total_passed / max(1, total_queries)) * 100.0, 1),
            "avg_steps_per_query": round(total_query_steps / max(1, total_queries), 2),
            "avg_tokens_per_query": round(total_query_tokens / max(1, total_queries), 1),
            "avg_time_s_per_query": round(total_query_time_s / max(1, total_queries), 2),
            "total_query_tokens": total_query_tokens,
            "total_query_time_s": round(total_query_time_s, 2),
            "total_ingest_tokens": total_ingest_tokens,
            "total_ingest_time_s": round(total_ingest_time_s, 2),
            "amortized_total_tokens": total_ingest_tokens + total_query_tokens,
            "amortized_total_time_s": round(total_ingest_time_s + total_query_time_s, 2),
        },
        "tasks": task_reports,
    }


def generate_markdown_report(all_results: List[Dict[str, Any]]) -> str:
    lines = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        "# Empirical A/B/C Benchmark Report: Active Engrams (Memory+Skill) vs Passive Graph-RAG vs RAM-Only OOP",
        "",
        "## 1. Executive Summary Comparison Table",
        "",
        "| Condition | Cold-Start Accuracy | Avg Steps / Query | Avg Tokens / Query | Avg Latency / Query | Upfront Ingest Tokens | Total Amortized Tokens (Setup + 12 Queries) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c in all_results:
        s = c["summary"]
        lines.append(
            f"| **{c['condition_id']}: {c['condition_label']}** | **{s['passed']}/{s['total']} ({s['pass_rate_pct']}%)** | "
            f"`{s['avg_steps_per_query']}` | `{s['avg_tokens_per_query']}` | `{s['avg_time_s_per_query']}s` | "
            f"`{s['total_ingest_tokens']}` | `{s['amortized_total_tokens']}` ({s['amortized_total_time_s']}s) |"
        )

    lines.extend(
        [
            "",
            "## 2. Per-Task Breakdown Across All 3 Conditions",
            "",
            "| Task | Workflow | Condition A (RAM-Only) | Condition B (Passive Graph-RAG) | Condition C (Active Engrams) |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    num_tasks = len(all_results[0]["tasks"])
    for t_idx in range(num_tasks):
        t_meta = all_results[0]["tasks"][t_idx]
        cells = []
        for cond in all_results:
            tr = cond["tasks"][t_idx]
            q_steps = sum(q["steps"] for q in tr["queries"])
            q_toks = sum(q["est_tokens"] for q in tr["queries"])
            cells.append(f"**{tr['task_passed']}/{tr['task_total']}** ({q_steps} steps, {q_toks} tok)")
        lines.append(
            f"| `{t_meta['task_id']}` ({t_meta['title']}) | {t_meta['workflow']} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    model = "qwen3.8:latest"
    cond_a = evaluate_condition(
        "Condition A",
        "RAM-Only OOP (No Neo4j Bridge)",
        enable_memory_graph=False,
        enable_engrams=False,
        model=model,
    )
    cond_b = evaluate_condition(
        "Condition B",
        "Passive Graph-RAG (Cypher-in-Actor Only, No Stored Engrams)",
        enable_memory_graph=True,
        enable_engrams=False,
        model=model,
    )
    cond_c = evaluate_condition(
        "Condition C",
        "Active Engrams (Neo4j Data + Stored & Hydrated Actor 4.1 Bytecode Skills)",
        enable_memory_graph=True,
        enable_engrams=True,
        model=model,
    )

    all_results = [cond_a, cond_b, cond_c]
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    md_report = generate_markdown_report(all_results)
    REPORT_PATH.write_text(md_report, encoding="utf-8")
    print("\n" + md_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
