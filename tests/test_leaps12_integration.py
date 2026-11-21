"""Leaps 1 & 2 Integration Tests (`tests/test_leaps12_integration.py`).

Verifies the integration of Active-Inference Counterfactual VM Tree Search
(`Actor[#Planner]`) and Wake-Sleep Neocortical Library Compression & Production
Compilation (`Actor[#Consolidator]`):
1. Live Actor VM dispatch for:
   - computeEFE, recommendMode, simulateBranches, recordDeadEnd, recallDeadEnds
   - compressEpisodes, proveEquivalence, registerReflex, matchReflex, describe
2. Static AST & bytecode linter guardrails protecting `#Planner` and `#Consolidator`
   against illegal reassignment, mutation (`atPut`), and deletion (`removeKey`).
3. Static protection of `ActivePlannerBridge` and `ConsolidatorBridge` system classes.
4. Lazy Step-1 prompt schema injection: schemas for `#Planner` and `#Consolidator`
   are omitted on unrelated goals (keeping prompts < 4k tokens), and automatically
   surfaced when relevant keywords match.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.guardrails.linter import ActorGuardrailError, ActorLinter
from actorii.harness import (
    ActivePlannerBridge,
    ConsolidatorBridge,
    bootstrap_agent_harness,
)
from actorii.vm import ActorVM


@pytest.fixture
def harness_env():
    """Create a bootstrapped agent harness with guardrails enabled."""
    vm, harness, sandbox = bootstrap_agent_harness(
        name="BilbyLeapsTest",
        enable_guardrails=True,
        enable_memory_graph=False,
    )
    return vm, harness, sandbox


# =============================================================================
# 1. Registration & Live Reflection
# =============================================================================

def test_bootstrap_registers_planner_and_consolidator_singletons(harness_env):
    """Verify Actor[#Planner] and Actor[#Consolidator] are exposed with both string and Symbol keys."""
    vm, _, sandbox = harness_env
    actor_pool = vm.global_vars.get("Actor", {})

    assert "Planner" in actor_pool
    assert Symbol("Planner") in actor_pool
    assert "Consolidator" in actor_pool
    assert Symbol("Consolidator") in actor_pool

    # Verify describe(Actor[#Planner])!!
    planner_desc = vm.eval_chunk("describe(Actor[#Planner])!!")
    assert isinstance(planner_desc, str)
    assert "ActivePlannerBridge" in planner_desc or "Actor[#Planner]" in planner_desc
    assert "computeEFE" in planner_desc
    assert "simulateBranches" in planner_desc
    assert "recommendMode" in planner_desc

    # Verify describe(Actor[#Consolidator])!!
    consolidator_desc = vm.eval_chunk("describe(Actor[#Consolidator])!!")
    assert isinstance(consolidator_desc, str)
    assert "ConsolidatorBridge" in consolidator_desc or "Actor[#Consolidator]" in consolidator_desc
    assert "compressEpisodes" in consolidator_desc
    assert "proveEquivalence" in consolidator_desc
    assert "registerReflex" in consolidator_desc
    assert "matchReflex" in consolidator_desc

    # Verify describeBridge via Sandbox
    sb_planner_desc = sandbox.describeBridge("Planner")
    assert "Actor[#Planner]" in sb_planner_desc
    assert "computeEFE" in sb_planner_desc

    sb_cons_desc = sandbox.describeBridge("Consolidator")
    assert "Actor[#Consolidator]" in sb_cons_desc
    assert "compressEpisodes" in sb_cons_desc


# =============================================================================
# 2. Live Actor VM Dispatch: ActivePlannerBridge
# =============================================================================

def test_vm_dispatch_planner_compute_efe(harness_env):
    """Test live computeEFE dispatch via Actor VM."""
    vm, _, _ = harness_env
    belief = json.dumps({"qs": [0.6, 0.4], "C": [2.0, -2.0]})
    actions = json.dumps([
        {"name": "inspect", "kind": "epistemic", "obsAccuracy": 0.95},
        {"name": "execute", "kind": "pragmatic", "expectedSuccess": 0.92},
    ])
    chunk = f"res := computeEFE(Actor[#Planner], {json.dumps(belief)}, {json.dumps(actions)})!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    best_action = res.at(Symbol("bestAction")) or res.at("bestAction")
    assert best_action in ("inspect", "execute")
    mode = res.at(Symbol("mode")) or res.at("mode")
    assert mode in ("epistemic_probe", "pragmatic_exploit")
    backend = res.at(Symbol("backend")) or res.at("backend")
    assert backend in ("pymdp", "numpy_efe")


def test_vm_dispatch_planner_recommend_mode(harness_env):
    """Test live recommendMode dispatch via Actor VM."""
    vm, _, _ = harness_env
    chunk = "res := recommendMode(Actor[#Planner], 0.85, 5, true)!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    mode = res.at(Symbol("mode")) or res.at("mode")
    assert mode == "epistemic_probe"
    should_search = res.at(Symbol("shouldTreeSearch"))
    if should_search is None:
        should_search = res.at("shouldTreeSearch")
    assert should_search is True


def test_vm_dispatch_planner_simulate_branches_shadow_isolation(harness_env):
    """Test live simulateBranches dispatch leaving live VM state 100% untouched."""
    vm, _, _ = harness_env
    vm.eval_chunk("Actor[#liveBase] := 500!!")

    candidates = json.dumps([
        {"name": "branch_fail", "code": "Actor[#liveBase] := 999!!\nActor[#out] := 10!!", "goal": "solve"},
        {"name": "branch_win", "code": "Actor[#liveBase] := 777!!\nActor[#out] := 42!!", "goal": "solve"},
    ])
    chunk = f"res := simulateBranches(Actor[#Planner], {json.dumps(candidates)}, \"Actor[#out] == 42!!\", 2)!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    best_act = res.at(Symbol("bestAction")) or res.at("bestAction")
    assert best_act == "branch_win"
    best_idx = res.at(Symbol("bestIndex")) or res.at("bestIndex")
    assert best_idx == 1
    verified = res.at(Symbol("verified")) or res.at("verified")
    assert verified is True
    isolated = res.at(Symbol("shadowIsolated")) or res.at("shadowIsolated")
    assert isolated is True

    # Critical: live VM state untouched
    assert vm.eval_chunk("Actor[#liveBase]!!") == 500
    assert "out" not in vm.global_vars.get("Actor", {})


def test_vm_dispatch_planner_dead_end_lifecycle(harness_env):
    """Test live recordDeadEnd and recallDeadEnds via Actor VM."""
    vm, _, _ = harness_env
    rec_chunk = 'res := recordDeadEnd(Actor[#Planner], "deploy_spec", "exec(Actor[#Jetson], ...)", "TimeoutError")!!'
    rec = vm.eval_chunk(rec_chunk)
    assert isinstance(rec, Dictionary)
    assert (rec.at(Symbol("ok")) or rec.at("ok")) is True

    recall_chunk = 'res := recallDeadEnds(Actor[#Planner], "deploy_spec", 5)!!'
    recalled = vm.eval_chunk(recall_chunk)
    assert isinstance(recalled, Dictionary)
    assert (recalled.at(Symbol("count")) or recalled.at("count")) >= 1
    dead_ends = recalled.at(Symbol("deadEnds")) or recalled.at("deadEnds")
    assert isinstance(dead_ends, (list, ActorArray))
    assert len(dead_ends) >= 1


# =============================================================================
# 3. Live Actor VM Dispatch: ConsolidatorBridge
# =============================================================================

def test_vm_dispatch_consolidator_compress_episodes(harness_env):
    """Test live compressEpisodes dispatch via Actor VM."""
    vm, _, _ = harness_env
    episodes = json.dumps([
        {"session_id": "ep1", "actions": ['queryRows(Actor[#GraphService], "MATCH (n:A) RETURN n")!!', 'atPut(x, #val, 1)!!']},
        {"session_id": "ep2", "actions": ['queryRows(Actor[#GraphService], "MATCH (n:B) RETURN n")!!', 'atPut(x, #val, 2)!!']},
    ])
    chunk = f"res := compressEpisodes(Actor[#Consolidator], {json.dumps(episodes)}, 2, 5)!!"
    res = vm.eval_chunk(chunk)
    assert isinstance(res, Dictionary)
    err = res.at(Symbol("error")) or res.at("error") or res.at("#error")
    assert err == ""
    ratio = res.at(Symbol("compressionRatio")) or res.at("compressionRatio") or res.at("#compressionRatio")
    assert ratio >= 1.0
    backend = res.at(Symbol("backend")) or res.at("backend") or res.at("#backend")
    assert backend in ("stitch-core", "python-anti-unification")


def test_vm_dispatch_consolidator_prove_equivalence(harness_env):
    """Test live proveEquivalence dispatch for equivalent & non-equivalent expressions."""
    vm, _, _ = harness_env

    # Equivalent expressions
    chunk_eq = 'res := proveEquivalence(Actor[#Consolidator], "x + 0", "x")!!'
    res_eq = vm.eval_chunk(chunk_eq)
    assert isinstance(res_eq, Dictionary)
    eq_val = res_eq.at(Symbol("equivalent")) or res_eq.at("equivalent") or res_eq.at("#equivalent")
    assert eq_val is True
    canon = res_eq.at(Symbol("canonicalForm")) or res_eq.at("canonicalForm") or res_eq.at("#canonicalForm")
    assert isinstance(canon, str) and len(canon) > 0

    # Non-equivalent expressions
    chunk_neq = 'res := proveEquivalence(Actor[#Consolidator], "x + 1", "x")!!'
    res_neq = vm.eval_chunk(chunk_neq)
    assert isinstance(res_neq, Dictionary)
    neq_val = res_neq.at(Symbol("equivalent")) or res_neq.at("equivalent") or res_neq.at("#equivalent")
    assert neq_val is False


def test_vm_dispatch_consolidator_register_and_match_reflex(harness_env):
    """Test live reflex promotion, shadow self-test validation, and sub-millisecond match."""
    vm, _, _ = harness_env
    src = (
        "inherit(Object, #FastCalcReflex, #(), 0, nil)!!\n"
        "now(FastCalcReflex)!!\n"
        "Def calculate(self, x) { ^(x * 2); }!!\n"
        "Actor[#FastCalcReflex] := new(FastCalcReflex)!!"
    )
    self_test = "(calculate(Actor[#FastCalcReflex], 21) == 42)!!"

    reg_chunk = (
        f'res := registerReflex(Actor[#Consolidator], "FastCalcReflex", "compute double of number", '
        f'{json.dumps(src)}, {json.dumps(self_test)})!!'
    )
    res = vm.eval_chunk(reg_chunk)
    assert isinstance(res, Dictionary)
    promoted = res.at(Symbol("promoted")) or res.at("promoted") or res.at("#promoted")
    assert promoted is True
    name = res.at(Symbol("name")) or res.at("name") or res.at("#name")
    assert name == "FastCalcReflex"

    # Verify shadow isolation: class was not permanently injected into live hierarchy
    assert vm.hierarchy.get("FastCalcReflex") is None

    # Match reflex (< 5 ms lookup)
    match_chunk = 'res := matchReflex(Actor[#Consolidator], "please compute double of number for me")!!'
    hit = vm.eval_chunk(match_chunk)
    assert isinstance(hit, Dictionary)
    matched = hit.at(Symbol("matched")) or hit.at("matched") or hit.at("#matched")
    assert matched is True
    hit_name = hit.at(Symbol("name")) or hit.at("name") or hit.at("#name")
    assert hit_name == "FastCalcReflex"
    elapsed = float(hit.at(Symbol("elapsedMs")) or hit.at("elapsedMs") or hit.at("#elapsedMs") or 0.0)
    assert elapsed < 5.0


# =============================================================================
# 4. Singleton Overwrite Protection in ActorLinter
# =============================================================================

def test_linter_protects_planner_and_consolidator_singletons(harness_env):
    """Verify ActorLinter statically blocks reassigning, mutating, or removing #Planner and #Consolidator."""
    _, _, sandbox = harness_env

    # Assignment via Actor[#Planner] := ...
    rep1 = sandbox.lintChunk("Actor[#Planner] := 123!!")
    assert rep1.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep1}"
    assert "Actor[#Planner]" in rep1

    rep2 = sandbox.lintChunk('Actor[#Consolidator] := "corrupt"!!')
    assert rep2.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep2}"
    assert "Actor[#Consolidator]" in rep2

    # Mutation via atPut
    rep3 = sandbox.lintChunk("atPut(Actor, #Planner, nil)!!")
    assert rep3.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep3}"
    assert "Actor[#Planner]" in rep3

    rep4 = sandbox.lintChunk("atPut(Actor, #Consolidator, nil)!!")
    assert rep4.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep4}"
    assert "Actor[#Consolidator]" in rep4

    # Deletion via removeKey
    rep5 = sandbox.lintChunk("removeKey(Actor, #Planner)!!")
    assert rep5.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep5}"
    assert "Actor[#Planner]" in rep5

    rep6 = sandbox.lintChunk("removeKey(Actor, #Consolidator)!!")
    assert rep6.startswith("ERROR:[SYS_PROTECTED_SINGLETON]"), f"Unexpected: {rep6}"
    assert "Actor[#Consolidator]" in rep6


def test_runtime_guardrails_block_planner_and_consolidator_overwrites(harness_env):
    """Verify runtime evaluation blocks overwriting #Planner and #Consolidator when guardrails enabled."""
    vm, _, _ = harness_env

    with pytest.raises(ActorGuardrailError):
        vm.eval_chunk("Actor[#Planner] := 999!!")

    with pytest.raises(ActorGuardrailError):
        vm.eval_chunk("Actor[#Consolidator] := 888!!")


# =============================================================================
# 5. Lazy Step-1 Prompt Schema Injection & Keyword Sharpening
# =============================================================================

def test_lazy_schema_injection_omitted_on_unrelated_goal(harness_env):
    """Verify Step-1 prompt and poolSummary omit #Planner and #Consolidator signatures on unrelated goals."""
    vm, harness, sandbox = harness_env
    unrelated_goal = "What is the capital of France?"

    summary = sandbox.poolSummary(unrelated_goal)

    # Should NOT have the detailed signatures in 'Relevant twin methods:'
    assert "computeEFE(Actor[#Planner]" not in summary
    assert "simulateBranches(" not in summary
    assert "compressEpisodes(Actor[#Consolidator]" not in summary
    assert "proveEquivalence(" not in summary

    # Should list them compactly under 'Other live twins'
    assert "#Planner" in summary
    assert "#Consolidator" in summary

    # Verify Step-1 prompt stays compact (< 4k chars)
    harness._fields["goal"] = unrelated_goal
    prompt = str(vm.eval_chunk('buildStepPrompt(Actor[#Harness], "", "", "")!!'))
    assert len(prompt) < 4000
    assert "computeEFE(Actor[#Planner]" not in prompt
    assert "compressEpisodes(Actor[#Consolidator]" not in prompt


def test_lazy_schema_injection_surfaces_planner_on_planning_goal(harness_env):
    """Verify #Planner signature is injected when goal contains planning/branch/counterfactual keywords."""
    _, _, sandbox = harness_env

    goals = [
        "Simulate branches and plan tree search under epistemic uncertainty",
        "Perform counterfactual evaluation and compute efe for candidate actions",
        "Record deadend branches and prune failure paths",
    ]

    for g in goals:
        summary = sandbox.poolSummary(g)
        assert "`Actor[#Planner]`" in summary, f"Expected #Planner in summary for goal: {g}"
        assert "computeEFE" in summary
        assert "`Actor[#Consolidator]`" not in summary, f"Did not expect #Consolidator for goal: {g}"


def test_lazy_schema_injection_surfaces_consolidator_on_dream_goal(harness_env):
    """Verify #Consolidator signature is injected when goal contains sleep/dream/compress/reflex keywords."""
    _, _, sandbox = harness_env

    goals = [
        "Run neocortical dream consolidation cycle over recent episodes",
        "Compress episode motifs into reusable engram abstractions",
        "Compile reflex rules and prove equivalence via egraph",
    ]

    for g in goals:
        summary = sandbox.poolSummary(g)
        assert "`Actor[#Consolidator]`" in summary, f"Expected #Consolidator in summary for goal: {g}"
        assert "compressEpisodes" in summary
        assert "`Actor[#Planner]`" not in summary, f"Did not expect #Planner for goal: {g}"


def test_lazy_schema_injection_surfaces_both_when_both_relevant(harness_env):
    """Verify both #Planner and #Consolidator are injected when goal mentions both domains."""
    _, _, sandbox = harness_env
    goal = "Simulate branch counterfactuals during dream consolidation and register reflex engrams"
    summary = sandbox.poolSummary(goal)

    assert "`Actor[#Planner]`" in summary
    assert "`Actor[#Consolidator]`" in summary


def test_autonomous_reflex_shortcut_fires_with_zero_llm_calls(harness_env):
    """Promoted Reflex Engrams execute in <1ms with 0 LLM calls via AgentHarness>>runGoal."""
    vm, harness, _ = harness_env
    llm_calls = []

    def _boom_provider(*args, **kwargs):
        llm_calls.append(args)
        raise AssertionError("LLM should NOT be called when a promoted Reflex matches!")

    harness.custom_provider = _boom_provider

    reg = vm.eval_chunk(
        'registerReflex(Actor[#Consolidator], "fast_square_7", "compute fast square of seven", "(7 * 7)!!", "(7 * 7) == 49!!")!!'
    )
    assert reg.at(Symbol("promoted")) is True

    out = vm.eval_chunk('runGoal(Actor[#Harness], "compute fast square of seven", 5)!!')
    assert out == 49
    assert llm_calls == []
    assert str(harness._fields.get("status")) == "completed"
    entries = list(harness._fields["stepLog"])
    assert len(entries) == 1
    assert entries[0]["tier"] == "REFLEX_0TOK"


def test_autonomous_dead_end_recall_injected_in_step1_messages(harness_env):
    """Dead ends recorded on Actor[#Planner] are automatically surfaced in Step-1 prompt."""
    vm, harness, sandbox = harness_env
    vm.eval_chunk(
        'recordDeadEnd(Actor[#Planner], "Fix flaky parser bug", "badCall(Actor[#VM])!!", "KeyError: foo")!!'
    )
    vm.eval_chunk('setGoal(Actor[#Harness], "Fix flaky parser bug")!!')
    llm_ref = vm.global_vars["Actor"]["LLM"]
    llm_py = getattr(llm_ref, "_target", getattr(llm_ref, "target", llm_ref))
    msgs = llm_py._build_append_only_messages(harness, sandbox)
    user_msg = msgs[-1]["content"]
    assert "[KNOWN DEAD ENDS TO AVOID]:" in user_msg
    assert "badCall(Actor[#VM])!!" in user_msg
