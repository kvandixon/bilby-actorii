"""Leap 3 Integration Tests (`tests/test_leap3_integration.py`).

Verifies the integration of Potassco `clingo` ASP Declarative Constraint Solver
(`Actor[#Solver]`) and MAC/FAC Structural Analogy Engine (`Actor[#Analogy]`):
1. Live Actor VM dispatch for `solveProgram`, `solveDAGSchedule`, `alignGraphs`,
   `extractStructure`, `findAnalogies`, `describe(Actor[#Solver])`, `describe(Actor[#Analogy])`.
2. Static AST & bytecode linter guardrails protecting `#Solver` and `#Analogy`
   against illegal reassignment, mutation (`atPut`), and deletion (`removeKey`).
3. Lazy Step-1 prompt schema injection: schemas for `#Solver` and `#Analogy`
   are omitted on unrelated goals (e.g. "What is 2+2?") keeping prompts < 4k tokens,
   and automatically surfaced when relevant keywords match.
4. Minimal Unsatisfiable Core (MUC) error diagnostic surfacing via `formatErrorDiagnostic`.
"""

import json
from typing import Any

import pytest

from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.guardrails.linter import ActorGuardrailError, ActorLinter
from actorii.harness import bootstrap_agent_harness
from actorii.vm import ActorVM


@pytest.fixture
def harness_env():
    """Create a bootstrapped agent harness with guardrails enabled."""
    vm, harness, sandbox = bootstrap_agent_harness(
        name="BilbyTest",
        enable_guardrails=True,
        enable_memory_graph=False,
    )
    return vm, harness, sandbox


# =============================================================================
# 1. Registration & Live Reflection
# =============================================================================

def test_bootstrap_registers_solver_and_analogy_singletons(harness_env):
    """Verify Actor[#Solver] and Actor[#Analogy] are exposed with both string and Symbol keys."""
    vm, _, sandbox = harness_env
    actor_pool = vm.global_vars.get("Actor", {})

    assert "Solver" in actor_pool
    assert Symbol("Solver") in actor_pool
    assert "Analogy" in actor_pool
    assert Symbol("Analogy") in actor_pool

    # Verify describe(Actor[#Solver])!!
    solver_desc = vm.eval_chunk("describe(Actor[#Solver])!!")
    assert isinstance(solver_desc, str)
    assert "ASPSolverBridge" in solver_desc or "Actor[#Solver]" in solver_desc
    assert "solveProgram" in solver_desc
    assert "solveDAGSchedule" in solver_desc
    assert "verifyInvariants" in solver_desc

    # Verify describe(Actor[#Analogy])!!
    analogy_desc = vm.eval_chunk("describe(Actor[#Analogy])!!")
    assert isinstance(analogy_desc, str)
    assert "AnalogyEngineBridge" in analogy_desc or "Actor[#Analogy]" in analogy_desc
    assert "alignGraphs" in analogy_desc
    assert "extractStructure" in analogy_desc

    # Verify describeBridge via Sandbox
    sb_solver_desc = sandbox.describeBridge("Solver")
    assert "Actor[#Solver]" in sb_solver_desc
    assert "solveProgram" in sb_solver_desc

    sb_analogy_desc = sandbox.describeBridge("Analogy")
    assert "Actor[#Analogy]" in sb_analogy_desc
    assert "alignGraphs" in sb_analogy_desc


# =============================================================================
# 2. Live Actor VM Dispatch: ASPSolverBridge
# =============================================================================

def test_vm_dispatch_solve_program_sat(harness_env):
    """Test live solveProgram dispatch for a satisfiable ASP program."""
    vm, _, _ = harness_env
    chunk = (
        'res := solveProgram(Actor[#Solver], "node(1..3). 1 { color(X, r; g; b) } 1 :- node(X). '
        ':- color(X, C), color(Y, C), X != Y.", 1, 5.0)!!'
    )
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert res.at(Symbol("satisfiable")) is True or res.at("satisfiable") is True
    models = res.at(Symbol("models")) or res.at("models")
    assert isinstance(models, (list, ActorArray))
    assert len(models) >= 1


def test_vm_dispatch_solve_program_unsat_with_muc(harness_env):
    """Test live solveProgram dispatch for UNSAT program extracting exact MUC."""
    vm, _, _ = harness_env
    chunk = (
        'res := solveProgram(Actor[#Solver], "a. b. % @label(no_ab)\\n:- a, b.", 1, 5.0)!!'
    )
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert res.at(Symbol("satisfiable")) is False or res.at("satisfiable") is False
    unsat_core = res.at(Symbol("unsatCore")) or res.at("unsatCore")
    assert isinstance(unsat_core, (list, ActorArray))
    core_items = [str(x) for x in unsat_core]
    assert "no_ab" in core_items


def test_vm_dispatch_verify_invariants(harness_env):
    """Test live verifyInvariants dispatch."""
    vm, _, _ = harness_env
    chunk = (
        'res := verifyInvariants(Actor[#Solver], "port(8080).", "% [no_8080]\\n:- port(8080).", 5.0)!!'
    )
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert res.at(Symbol("ok")) is False or res.at("ok") is False
    unsat_core = res.at(Symbol("unsatCore")) or res.at("unsatCore")
    assert len(unsat_core) >= 1


def test_vm_dispatch_dag_schedule_valid(harness_env):
    """Test live solveDAGSchedule dispatch for a valid DAG."""
    vm, _, _ = harness_env
    tasks = json.dumps([
        {"id": "A", "duration": 2, "deps": []},
        {"id": "B", "duration": 3, "deps": ["A"]},
        {"id": "C", "duration": 1, "deps": ["A"]},
        {"id": "D", "duration": 2, "deps": ["B", "C"]},
    ])
    chunk = f"res := solveDAGSchedule(Actor[#Solver], {json.dumps(tasks)}, 2, 5.0)!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert res.at(Symbol("satisfiable")) is True or res.at("satisfiable") is True
    makespan = res.at(Symbol("makespan")) or res.at("makespan")
    assert makespan == 7
    schedule = res.at(Symbol("schedule")) or res.at("schedule")
    assert len(schedule) == 4


def test_vm_dispatch_dag_schedule_cyclic_unsat(harness_env):
    """Test live solveDAGSchedule dispatch for a cyclic DAG returning conflict MUC."""
    vm, _, _ = harness_env
    cyclic_tasks = json.dumps([
        {"id": "A", "duration": 1, "deps": ["B"]},
        {"id": "B", "duration": 1, "deps": ["A"]},
    ])
    chunk = f"res := solveDAGSchedule(Actor[#Solver], {json.dumps(cyclic_tasks)}, 1, 5.0)!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert res.at(Symbol("satisfiable")) is False or res.at("satisfiable") is False
    unsat_core = res.at(Symbol("unsatCore")) or res.at("unsatCore")
    assert len(unsat_core) >= 1


# =============================================================================
# 3. Live Actor VM Dispatch: AnalogyEngineBridge
# =============================================================================

def test_vm_dispatch_analogy_extract_structure(harness_env):
    """Test live extractStructure dispatch."""
    vm, _, _ = harness_env
    graph = json.dumps({
        "nodes": [{"id": "n1", "type": "State"}, {"id": "n2", "type": "Resource"}],
        "edges": [{"src": "n1", "dst": "n2", "rel": "REQUIRES"}],
    })
    chunk = f"res := extractStructure(Actor[#Analogy], {json.dumps(graph)})!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    assert (res.at(Symbol("nodeCount")) or res.at("nodeCount")) == 2
    assert (res.at(Symbol("edgeCount")) or res.at("edgeCount")) == 1


def test_vm_dispatch_analogy_align_graphs(harness_env):
    """Test live alignGraphs dispatch projecting cross-domain analogical mapping."""
    vm, _, _ = harness_env
    base_graph = json.dumps({
        "nodes": ["A", "B", "C"],
        "edges": [
            {"src": "A", "dst": "B", "rel": "CAUSES"},
            {"src": "B", "dst": "C", "rel": "BLOCKS"},
        ],
    })
    target_graph = json.dumps({
        "nodes": ["X", "Y", "Z"],
        "edges": [
            {"src": "X", "dst": "Y", "rel": "CAUSES"},
            {"src": "Y", "dst": "Z", "rel": "BLOCKS"},
        ],
    })
    chunk = f"res := alignGraphs(Actor[#Analogy], {json.dumps(base_graph)}, {json.dumps(target_graph)})!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    score = float(res.at(Symbol("score")) or res.at("score"))
    assert score > 0.8
    mapping = res.at(Symbol("mapping")) or res.at("mapping")
    assert isinstance(mapping, Dictionary)
    assert (mapping.at(Symbol("A")) or mapping.at("A")) == "X"
    assert (mapping.at(Symbol("B")) or mapping.at("B")) == "Y"
    assert (mapping.at(Symbol("C")) or mapping.at("C")) == "Z"


# =============================================================================
# 4. Singleton Overwrite Protection in ActorLinter
# =============================================================================

def test_linter_protects_solver_and_analogy_singletons(harness_env):
    """Verify ActorLinter statically blocks reassigning #Solver and #Analogy."""
    _, _, sandbox = harness_env

    # Assignment via Actor[#Solver] := ...
    rep1 = sandbox.lintChunk("Actor[#Solver] := 123!!")
    assert rep1.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep1}"
    assert "Actor[#Solver]" in rep1

    rep2 = sandbox.lintChunk('Actor[#Analogy] := "corrupt"!!')
    assert rep2.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep2}"
    assert "Actor[#Analogy]" in rep2

    # Mutation via atPut
    rep3 = sandbox.lintChunk("atPut(Actor, #Solver, nil)!!")
    assert rep3.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep3}"
    assert "Actor[#Solver]" in rep3

    rep4 = sandbox.lintChunk("atPut(Actor, #Analogy, nil)!!")
    assert rep4.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep4}"
    assert "Actor[#Analogy]" in rep4

    # Deletion via removeKey
    rep5 = sandbox.lintChunk("removeKey(Actor, #Solver)!!")
    assert rep5.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep5}"
    assert "Actor[#Solver]" in rep5

    rep6 = sandbox.lintChunk("removeKey(Actor, #Analogy)!!")
    assert rep6.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep6}"
    assert "Actor[#Analogy]" in rep6


def test_runtime_guardrail_blocks_solver_overwrite(harness_env):
    """Verify runtime evaluation blocks overwriting #Solver and #Analogy when guardrails enabled."""
    vm, _, _ = harness_env

    with pytest.raises(ActorGuardrailError):
        vm.eval_chunk("Actor[#Solver] := 999!!")

    with pytest.raises(ActorGuardrailError):
        vm.eval_chunk("Actor[#Analogy] := 888!!")


# =============================================================================
# 5. Lazy Step-1 Prompt Schema Injection
# =============================================================================

def test_lazy_schema_injection_omitted_on_unrelated_goal(harness_env):
    """Verify Step-1 prompt and poolSummary omit #Solver and #Analogy detailed signatures on unrelated goals."""
    vm, harness, sandbox = harness_env
    unrelated_goal = "What is 2+2?"

    summary = sandbox.poolSummary(unrelated_goal)

    # Should NOT have the detailed signatures in 'Relevant twin methods:'
    assert "solveProgram(Actor[#Solver]" not in summary
    assert "alignGraphs(Actor[#Analogy]" not in summary

    # Should list them compactly under 'Other live twins'
    assert "#Solver" in summary
    assert "#Analogy" in summary

    # Verify Step-1 prompt stays compact (< 4k chars)
    harness._fields["goal"] = unrelated_goal
    prompt = str(vm.eval_chunk('buildStepPrompt(Actor[#Harness], "", "", "")!!'))
    assert len(prompt) < 4000
    assert "solveProgram(Actor[#Solver]" not in prompt
    assert "alignGraphs(Actor[#Analogy]" not in prompt


def test_lazy_schema_injection_surfaces_solver_on_constraint_goal(harness_env):
    """Verify #Solver signature is injected when goal contains constraint/dag/asp/schedule keywords."""
    _, _, sandbox = harness_env

    goals = [
        "Solve a combinatorial constraint satisfaction problem with clingo",
        "Compute an optimal dag schedule with minimal makespan",
        "Verify system safety invariants using asp rules",
    ]

    for g in goals:
        summary = sandbox.poolSummary(g)
        assert "`Actor[#Solver]`" in summary, f"Expected #Solver in summary for goal: {g}"
        assert "solveProgram" in summary
        assert "`Actor[#Analogy]`" not in summary, f"Did not expect #Analogy for goal: {g}"


def test_lazy_schema_injection_surfaces_analogy_on_analogy_goal(harness_env):
    """Verify #Analogy signature is injected when goal contains analogy/isomorphic/transfer keywords."""
    _, _, sandbox = harness_env

    goals = [
        "Find structural analogy and isomorphic mapping between two failure graphs",
        "Perform causal schema transfer from hydraulic flow to heat diffusion",
    ]

    for g in goals:
        summary = sandbox.poolSummary(g)
        assert "`Actor[#Analogy]`" in summary, f"Expected #Analogy in summary for goal: {g}"
        assert "alignGraphs" in summary
        assert "`Actor[#Solver]`" not in summary, f"Did not expect #Solver for goal: {g}"


def test_lazy_schema_injection_surfaces_both_when_both_relevant(harness_env):
    """Verify both #Solver and #Analogy are injected when goal mentions both domains."""
    _, _, sandbox = harness_env
    goal = "Retrieve an isomorphic analogy and solve the resulting constraint schedule with clingo"
    summary = sandbox.poolSummary(goal)

    assert "`Actor[#Solver]`" in summary
    assert "`Actor[#Analogy]`" in summary


# =============================================================================
# 6. MUC Error Diagnostic Surfacing
# =============================================================================

def test_format_error_diagnostic_surfaces_muc_from_context(harness_env):
    """Verify formatErrorDiagnostic cleanly surfaces MUC from lastErrorContext."""
    _, harness, sandbox = harness_env

    err_ctx = Dictionary(8)
    unsat_arr = ActorArray(2)
    unsat_arr.atPut(0, "no_8080_rule")
    unsat_arr.atPut(1, "service_port_clash")
    err_ctx.atPut(Symbol("unsatCore"), unsat_arr)
    err_ctx.atPut("unsatCore", unsat_arr)
    err_ctx.atPut(Symbol("errorKind"), Symbol("InvariantViolation"))

    harness._fields["lastError"] = "InvariantViolationError: program unsatisfiable"
    harness._fields["lastErrorContext"] = err_ctx

    diag = sandbox.formatErrorDiagnostic(harness)

    assert "Minimal Unsatisfiable Core (MUC): [no_8080_rule, service_port_clash]" in diag
    assert "Conflict Resolution Hint" in diag
    assert "no_8080_rule" in diag


def test_format_error_diagnostic_surfaces_muc_from_error_string(harness_env):
    """Verify formatErrorDiagnostic surfaces MUC when present in lastError text."""
    _, harness, sandbox = harness_env

    harness._fields["lastError"] = "ASP solver failed (unsatCore: ['constraint_alpha', 'constraint_beta'])"
    harness._fields["lastErrorContext"] = None

    diag = sandbox.formatErrorDiagnostic(harness)

    assert "Minimal Unsatisfiable Core (MUC): [constraint_alpha, constraint_beta]" in diag
    assert "Conflict Resolution Hint" in diag
