"""Unit tests for Leap #2 Active-Inference Counterfactual VM Tree Search (`ActivePlannerBridge`)."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from actorii.core import Symbol
from actorii.harness import active_planner
from actorii.harness.active_planner import ActivePlannerBridge, _actor_to_py_value
from actorii.vm import ActorVM


class _FakeHTTPResponse:
    """Minimal context-manager mock for `urllib.request.urlopen`."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_compute_efe_epistemic_vs_pragmatic_switching() -> None:
    """High belief uncertainty favors epistemic probing; low uncertainty favors pragmatic exploit."""
    bridge = ActivePlannerBridge(enable_neo4j=False, namespace="pytest_planner")

    # 2 hidden states: state 0 = safe/valid, state 1 = broken/unsafe
    # 2 observations: obs 0 = success (+2.5 pref), obs 1 = failure (-2.5 pref)
    # Action 1 (probe): sharp observation likelihood across both states, neutral preference (no mutation)
    # Action 2 (commit): succeeds in state 0, fails badly in state 1 (risky when uncertain)
    actions = [
        {
            "name": "probe_schema",
            "kind": "epistemic",
            "A": [[0.96, 0.04], [0.04, 0.96]],
            "C": [0.0, 0.0],
        },
        {
            "name": "commit_mutation",
            "kind": "pragmatic",
            "A": [[0.92, 0.10], [0.08, 0.90]],
            "C": [2.5, -2.5],
        },
    ]

    # Case A: Maximum uncertainty q(s) = [0.5, 0.5] -> epistemic probe wins (lower EFE)
    high_u_actor = bridge.computeEFE(json.dumps({"qs": [0.5, 0.5]}), json.dumps(actions))
    assert high_u_actor.at(Symbol("bestAction")) == "probe_schema"
    assert high_u_actor.at(Symbol("mode")) == "epistemic_probe"
    high_u = _actor_to_py_value(high_u_actor)
    assert high_u["backend"] in ("pymdp", "numpy_efe")
    assert high_u["beliefEntropy"] == pytest.approx(1.0, abs=1e-4)
    probe_score = next(s for s in high_u["scores"] if s["action"] == "probe_schema")
    commit_score = next(s for s in high_u["scores"] if s["action"] == "commit_mutation")
    assert probe_score["epistemicValue"] > 0.4
    assert probe_score["efe"] < commit_score["efe"]

    # Case B: High certainty q(s) = [0.97, 0.03] -> pragmatic exploit wins (lower EFE)
    low_u_actor = bridge.computeEFE(json.dumps({"qs": [0.97, 0.03]}), json.dumps(actions))
    low_u = _actor_to_py_value(low_u_actor)
    assert low_u["bestAction"] == "commit_mutation"
    assert low_u["mode"] == "pragmatic_exploit"
    assert low_u["beliefEntropy"] < 0.25
    probe_low = next(s for s in low_u["scores"] if s["action"] == "probe_schema")
    commit_low = next(s for s in low_u["scores"] if s["action"] == "commit_mutation")
    assert commit_low["efe"] < probe_low["efe"]


def test_compute_efe_pymdp_and_numpy_fallback_agree() -> None:
    """`pymdp` backend and `numpy_efe` fallback compute consistent EFE decompositions."""
    b_pymdp = ActivePlannerBridge(enable_neo4j=False, prefer_pymdp=True)
    b_numpy = ActivePlannerBridge(enable_neo4j=False, prefer_pymdp=False)

    belief = json.dumps({"qs": [0.6, 0.4], "C": [2.0, -2.0]})
    actions = json.dumps(
        [
            {"name": "inspect", "kind": "epistemic", "obsAccuracy": 0.95},
            {"name": "execute", "kind": "pragmatic", "expectedSuccess": 0.92},
        ]
    )

    res_p = _actor_to_py_value(b_pymdp.computeEFE(belief, actions))
    res_n = _actor_to_py_value(b_numpy.computeEFE(belief, actions))

    assert res_n["backend"] == "numpy_efe"
    if active_planner._PYMDP_AVAILABLE:
        assert res_p["backend"] == "pymdp"

    assert res_p["bestAction"] == res_n["bestAction"]
    assert res_p["mode"] == res_n["mode"]
    for sp, sn in zip(res_p["scores"], res_n["scores"]):
        assert sp["efe"] == pytest.approx(sn["efe"], abs=1e-3)
        assert sp["epistemicValue"] == pytest.approx(sn["epistemicValue"], abs=1e-3)
        assert sp["pragmaticValue"] == pytest.approx(sn["pragmaticValue"], abs=1e-3)


def test_recommend_mode_thresholds_and_final_step() -> None:
    bridge = ActivePlannerBridge(enable_neo4j=False)

    # High uncertainty with writes -> epistemic_probe
    r1 = _actor_to_py_value(bridge.recommendMode(0.75, 5, True))
    assert r1["mode"] == "epistemic_probe"
    assert r1["shouldTreeSearch"] is True

    # Medium uncertainty (0.50): epistemic when writes=True, pragmatic when writes=False
    r_write = _actor_to_py_value(bridge.recommendMode(0.50, 4, True))
    r_read = _actor_to_py_value(bridge.recommendMode(0.50, 4, False))
    assert r_write["mode"] == "epistemic_probe"
    assert r_read["mode"] == "pragmatic_exploit"

    # Final step forces pragmatic_exploit even under high uncertainty
    r_last = _actor_to_py_value(bridge.recommendMode(0.95, 1, True))
    assert r_last["mode"] == "pragmatic_exploit"
    assert r_last["shouldTreeSearch"] is False


def test_simulate_branches_shadow_isolation_winner_and_dead_end_pruning() -> None:
    """Shadow simulation must pick verified winner, prune dead ends, and leave live VM untouched."""
    vm = ActorVM()
    vm.eval_chunk("Actor[#liveCounter] := 100!!")
    vm.eval_chunk('Actor[#liveTag] := "original"!!')

    bridge = ActivePlannerBridge(vm=vm, enable_neo4j=False, namespace="pytest_sim")

    # Record a known dead-end action before simulating branches
    bad_code = "Actor[#liveCounter] := 1 / 0!!"
    rec = _actor_to_py_value(
        bridge.recordDeadEnd("compute answer", bad_code, "ZeroDivisionError")
    )
    assert rec["ok"] is True

    candidates = [
        {"name": "dead_end_branch", "code": bad_code, "goal": "compute answer"},
        {
            "name": "wrong_answer",
            "code": 'Actor[#liveCounter] := 999!!\nActor[#liveTag] := "mutated"!!\nActor[#ans] := 41!!',
            "goal": "compute answer",
        },
        {
            "name": "winning_answer",
            "code": 'Actor[#liveCounter] := 777!!\nActor[#liveTag] := "shadow"!!\nActor[#ans] := 42!!',
            "goal": "compute answer",
        },
        {
            "name": "runtime_error_branch",
            "code": "Actor[#ans] := 5 / 0!!",
            "goal": "compute answer",
        },
    ]

    sim_actor = bridge.simulateBranches(
        json.dumps(candidates),
        verifier_chunk="Actor[#ans] == 42!!",
        max_branches=4,
    )
    # Verify native Actor Symbol lookup
    assert sim_actor.at(Symbol("bestAction")) == "winning_answer"
    assert sim_actor.at(Symbol("bestIndex")) == 2
    assert sim_actor.at(Symbol("verified")) is True

    sim = _actor_to_py_value(sim_actor)
    assert sim["shadowIsolated"] is True
    assert len(sim["branches"]) == 4
    assert len(sim["prunedDeadEnds"]) == 1
    assert sim["prunedDeadEnds"][0]["name"] == "dead_end_branch"
    assert sim["branches"][0]["pruned"] is True
    assert sim["branches"][1]["verifierPassed"] is False
    assert sim["branches"][2]["verifierPassed"] is True
    assert sim["branches"][3]["ok"] is False

    # Critical guardrail: live VM state MUST be 100% untouched after shadow simulation!
    assert vm.eval_chunk("Actor[#liveCounter]!!") == 100
    assert vm.eval_chunk("Actor[#liveTag]!!") == "original"
    assert "ans" not in vm.global_vars.get("Actor", {})


def test_simulate_branches_systemone_prior_optional_and_offline_safe() -> None:
    """Optional :11434 System-1 prior boosts score when reachable and degrades cleanly when offline."""
    vm = ActorVM()
    bridge = ActivePlannerBridge(
        vm=vm,
        enable_neo4j=False,
        enable_systemone=True,
        systemone_url="http://127.0.0.1:19999/v1/systemone",
    )

    candidates = [
        {"name": "cand_a", "code": "Actor[#v] := 10!!"},
        {"name": "cand_b", "code": "Actor[#v] := 10!!"},
    ]

    # Mock reachable System-1 giving cand_b a higher prior
    responses = [
        _FakeHTTPResponse({"answers": {"viable": 0.2}}),
        _FakeHTTPResponse({"answers": {"viable": 0.95}}),
    ]
    with patch("urllib.request.urlopen", side_effect=responses):
        sim_online = _actor_to_py_value(
            bridge.simulateBranches(json.dumps(candidates), "Actor[#v] == 10!!", 4)
        )
    assert sim_online["bestAction"] == "cand_b"
    assert sim_online["branches"][1]["systemOnePrior"] == pytest.approx(0.95)

    # Offline failure degrades gracefully without raising
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        sim_offline = _actor_to_py_value(
            bridge.simulateBranches(json.dumps(candidates), "Actor[#v] == 10!!", 4)
        )
    assert sim_offline["verified"] is True
    assert sim_offline["branches"][0]["systemOnePrior"] is None


def test_record_and_recall_dead_ends_memory_and_neo4j_mock() -> None:
    """`:ActorDeadEnd` record/recall works offline, respects namespaces, and syncs to Neo4j."""
    bridge = ActivePlannerBridge(enable_neo4j=False, namespace="ns_alpha")

    r1 = _actor_to_py_value(
        bridge.recordDeadEnd("solve dag schedule", "schedule(a, b)!!", "UnsatCoreError")
    )
    assert r1["ok"] is True
    assert r1["hitCount"] == 1
    assert r1["namespace"] == "ns_alpha"

    # Recording the same dead end increments hitCount
    r2 = _actor_to_py_value(
        bridge.recordDeadEnd("solve dag schedule", "schedule(a, b)!!", "UnsatCoreError")
    )
    assert r2["hitCount"] == 2

    # Different namespace should not leak
    bridge.setNamespace("ns_beta")
    assert bridge.getNamespace() == "ns_beta"
    beta_recall = _actor_to_py_value(bridge.recallDeadEnds("solve dag schedule", 5))
    assert beta_recall["count"] == 0

    bridge.setNamespace("ns_alpha")
    alpha_recall = _actor_to_py_value(bridge.recallDeadEnds("dag schedule", 5))
    assert alpha_recall["count"] == 1
    assert alpha_recall["deadEnds"][0]["hitCount"] == 2
    assert alpha_recall["deadEnds"][0]["errorSig"] == "UnsatCoreError"

    # Verify Neo4j HTTP Cypher path with mocked response
    bridge_neo = ActivePlannerBridge(enable_neo4j=True, namespace="ns_neo")
    captured_cypher: List[str] = []

    def _fake_urlopen(req, timeout=2.5):
        body = json.loads(req.data.decode("utf-8"))
        for stmt in body.get("statements", []):
            captured_cypher.append(stmt["statement"])
        if any("RETURN d.id" in s["statement"] for s in body.get("statements", [])):
            return _FakeHTTPResponse(
                {
                    "results": [
                        {
                            "columns": ["id", "goal", "actionSig", "errorSig", "hitCount", "updatedAt"],
                            "data": [
                                {
                                    "row": [
                                        "de_neo_1",
                                        "fix memory leak",
                                        "freeAll()!!",
                                        "DoubleFreeError",
                                        3,
                                        1700000000.0,
                                    ]
                                }
                            ],
                        }
                    ],
                    "errors": [],
                }
            )
        return _FakeHTTPResponse({"results": [], "errors": []})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        rec_neo = _actor_to_py_value(
            bridge_neo.recordDeadEnd("fix memory leak", "freeAll()!!", "DoubleFreeError")
        )
        assert rec_neo["neo4jSynced"] is True
        recall_neo = _actor_to_py_value(bridge_neo.recallDeadEnds("fix memory leak", 5))
        assert recall_neo["source"] == "neo4j+memory"
        assert recall_neo["count"] >= 1
        assert any("ActorDeadEnd" in c for c in captured_cypher)


def test_describe_and_actor_vm_exposure() -> None:
    """`ActivePlannerBridge` exposes cleanly to `ActorVM` and responds to `describe()`."""
    vm = ActorVM()
    bridge = ActivePlannerBridge(vm=vm, enable_neo4j=False, namespace="pytest_vm")
    vm.expose_object(bridge, global_key="#Planner")

    doc = vm.eval_chunk("describe(Actor[#Planner])!!")
    assert isinstance(doc, str)
    assert "ActivePlannerBridge" in doc
    assert "computeEFE" in doc
    assert "simulateBranches" in doc
    assert "recordDeadEnd" in doc
    assert "0 MB VRAM" in doc

    # Exercise end-to-end Actor 4.1 calls on Actor[#Planner]
    rec = vm.eval_chunk("recommendMode(Actor[#Planner], 0.85, 4, true)!!")
    assert rec.at(Symbol("mode")) == "epistemic_probe"

    sim = vm.eval_chunk(
        'simulateBranches(Actor[#Planner], "[\\"Actor[#k] := 1!!\\", \\"Actor[#k] := 2!!\\"]", "Actor[#k] == 2!!", 4)!!'
    )
    assert sim.at(Symbol("bestIndex")) == 1
    assert sim.at(Symbol("verified")) is True
    assert "k" not in vm.global_vars.get("Actor", {})
