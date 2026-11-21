"""Unit tests for `ASPSolverBridge` (`actorii.harness.asp_solver`)."""

from __future__ import annotations

import json
import pytest

from actorii.bridge import ActorBridgeObject, expose_python_object
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.harness import asp_solver as asp_mod
from actorii.harness.asp_solver import ASPSolverBridge


def _to_py(val):
    """Recursively unwrap Actor `Dictionary` and `ActorArray` to Python builtins for assertions."""
    if isinstance(val, Dictionary):
        out = {}
        for k, v in val._entries.items():
            k_str = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
            if k_str not in out:
                out[k_str] = _to_py(v)
        return out
    if isinstance(val, ActorArray):
        return [_to_py(val.at(i)) for i in range(val.size())]
    return val


@pytest.fixture
def solver() -> ASPSolverBridge:
    return ASPSolverBridge()


class TestSolveProgram:
    def test_sat_basic_graph_coloring(self, solver: ASPSolverBridge) -> None:
        prog = """
        node(a; b; c).
        edge(a, b). edge(b, c).
        color(red; blue).
        1 { assign(N, C) : color(C) } 1 :- node(N).
        :- edge(U, V), assign(U, C), assign(V, C).
        #show assign/2.
        """
        res = solver.solveProgram(prog, max_models=2, timeout_s=5.0)
        assert isinstance(res, Dictionary)
        assert res.at("#satisfiable") is True
        assert res.at(Symbol("satisfiable")) is True
        assert res.at("#optimum") is False
        assert res.at("#cost") is None
        assert res.at("#error") == ""
        assert res.at("#elapsedMs") >= 0.0
        models = _to_py(res.at("#models"))
        assert len(models) == 2
        for m in models:
            assert len(m) == 3
            assert all(atom.startswith("assign(") for atom in m)
            # Ensure no internal assumption atoms leak into models
            assert not any("__asp_c_" in atom for atom in m)
        assert _to_py(res.at("#unsatCore")) == []

    def test_minimize_optimization(self, solver: ASPSolverBridge) -> None:
        prog = """
        item(a, 10). item(b, 4). item(c, 7). item(d, 2).
        2 { pick(I) : item(I, _) } 2.
        #minimize { W, I : pick(I), item(I, W) }.
        #show pick/1.
        """
        res = solver.solveProgram(prog, max_models=1, timeout_s=5.0)
        assert res.at("#satisfiable") is True
        assert res.at("#optimum") is True
        assert _to_py(res.at("#cost")) == [6]
        models = _to_py(res.at("#models"))
        assert len(models) == 1
        assert sorted(models[0]) == ["pick(b)", "pick(d)"]

    def test_unsat_exact_muc_labeled_constraints(self, solver: ASPSolverBridge) -> None:
        prog = """
        a.
        b.
        c.
        % @label(conflict_ab)
        :- a, b.
        % @label(irrelevant_cd)
        :- c, d.
        % [irrelevant_ef]
        :- e, f.
        """
        res = solver.solveProgram(prog, max_models=1, timeout_s=5.0)
        assert res.at("#satisfiable") is False
        assert res.at("#error") == ""
        assert _to_py(res.at("#models")) == []
        core = _to_py(res.at("#unsatCore"))
        assert core == ["conflict_ab"]

    def test_unsat_exact_muc_explicit_externals(self, solver: ASPSolverBridge) -> None:
        prog = """
        #external rule_x.
        #external rule_y.
        #external rule_unrelated.
        p :- rule_x.
        q :- rule_y.
        r :- rule_unrelated.
        :- p, q.
        """
        res = solver.solveProgram(prog, max_models=1, timeout_s=5.0)
        assert res.at("#satisfiable") is False
        core = set(_to_py(res.at("#unsatCore")))
        assert "rule_x" in core
        assert "rule_y" in core
        assert "rule_unrelated" not in core

    def test_blocks_forbidden_io_directives(self, solver: ASPSolverBridge) -> None:
        for bad in (
            '#include "/etc/passwd".',
            "#script (python)\nimport os\n#end.",
            "p(@evil(1)).",
        ):
            res = solver.solveProgram(bad, max_models=1, timeout_s=2.0)
            assert res.at("#satisfiable") is False
            assert "blocked" in res.at("#error").lower()

    def test_syntax_error_reported_cleanly(self, solver: ASPSolverBridge) -> None:
        res = solver.solveProgram("this is not valid :- :- asp", max_models=1, timeout_s=2.0)
        assert res.at("#satisfiable") is False
        assert "error" in res.at("#error").lower()

    def test_timeout_guard(self, solver: ASPSolverBridge) -> None:
        # Hard pigeonhole problem PHP(11, 10) is UNSAT and takes far longer than 0.08s
        php = """
        pigeon(1..11).
        hole(1..10).
        1 { in(P, H) : hole(H) } 1 :- pigeon(P).
        :- in(P1, H), in(P2, H), P1 < P2.
        """
        res = solver.solveProgram(php, max_models=1, timeout_s=0.08)
        assert res.at("#satisfiable") is False
        assert "timed out" in res.at("#error").lower()
        assert res.at("#elapsedMs") < 3000.0


class TestVerifyInvariants:
    def test_invariants_pass(self, solver: ASPSolverBridge) -> None:
        facts = "service(llama, 8080). vram_mb(14400). vram_limit(16300)."
        rules = """
        % @label(vram_budget)
        :- vram_mb(U), vram_limit(L), U > L.
        % @label(no_forbidden_port)
        :- service(_, 22).
        """
        res = solver.verifyInvariants(facts, rules, timeout_s=5.0)
        assert res.at("#ok") is True
        assert _to_py(res.at("#violated")) == []
        assert _to_py(res.at("#unsatCore")) == []
        assert res.at("#error") == ""

    def test_invariants_violation_with_muc(self, solver: ASPSolverBridge) -> None:
        facts = "service(llama, 8080). vram_mb(17500). vram_limit(16300)."
        rules = """
        % @label(vram_budget_exceeded)
        :- vram_mb(U), vram_limit(L), U > L.
        % @label(no_forbidden_port)
        :- service(_, 22).
        """
        res = solver.verifyInvariants(facts, rules, timeout_s=5.0)
        assert res.at("#ok") is False
        assert _to_py(res.at("#violated")) == ["vram_budget_exceeded"]
        assert _to_py(res.at("#unsatCore")) == ["vram_budget_exceeded"]
        assert res.at("#error") == ""

    def test_invariants_violated_predicate(self, solver: ASPSolverBridge) -> None:
        facts = "action(rm_rf_root)."
        rules = "violated(destructive_rm) :- action(rm_rf_root)."
        res = solver.verifyInvariants(facts, rules, timeout_s=5.0)
        assert res.at("#ok") is False
        assert _to_py(res.at("#violated")) == ["violated(destructive_rm)"]
        assert res.at("#error") == ""


class TestSolveDAGSchedule:
    def test_valid_dag_parallel_optimal_makespan(self, solver: ASPSolverBridge) -> None:
        # Diamond DAG:
        #   A (2) -> B (3), C (2) -> D (1)
        # With max_parallel=2: A=[0,2], B=[2,5], C=[2,4], D=[5,6] => makespan=6
        # With max_parallel=1: makespan=8
        tasks = [
            {"id": "A", "duration": 2, "deps": []},
            {"id": "B", "duration": 3, "deps": ["A"], "priority": 2},
            {"id": "C", "duration": 2, "deps": ["A"], "priority": 1},
            {"id": "D", "duration": 1, "deps": ["B", "C"]},
        ]
        res_par = solver.solveDAGSchedule(json.dumps(tasks), max_parallel=2, timeout_s=5.0)
        assert res_par.at("#satisfiable") is True
        assert res_par.at("#optimum") is True
        assert res_par.at("#makespan") == 6
        assert res_par.at("#error") == ""
        sched = {e["id"]: e for e in _to_py(res_par.at("#schedule"))}
        assert sched["A"]["start"] == 0 and sched["A"]["end"] == 2
        assert sched["B"]["start"] >= sched["A"]["end"]
        assert sched["C"]["start"] >= sched["A"]["end"]
        assert sched["D"]["start"] >= max(sched["B"]["end"], sched["C"]["end"])
        assert sched["D"]["end"] == 6

        res_seq = solver.solveDAGSchedule(json.dumps(tasks), max_parallel=1, timeout_s=5.0)
        assert res_seq.at("#satisfiable") is True
        assert res_seq.at("#optimum") is True
        assert res_seq.at("#makespan") == 8

    def test_cyclic_dag_unsat_exact_cycle_muc(self, solver: ASPSolverBridge) -> None:
        # Cycle: B -> C -> B, plus unrelated acyclic edge A -> D
        tasks = [
            {"id": "A", "duration": 1, "deps": []},
            {"id": "B", "duration": 2, "deps": ["C"]},
            {"id": "C", "duration": 2, "deps": ["B"]},
            {"id": "D", "duration": 1, "deps": ["A"]},
        ]
        res = solver.solveDAGSchedule(json.dumps(tasks), max_parallel=2, timeout_s=5.0)
        assert res.at("#satisfiable") is False
        assert res.at("#makespan") is None
        assert res.at("#error") == ""
        core = set(_to_py(res.at("#unsatCore")))
        assert core == {"dep:B->C", "dep:C->B"}

    def test_empty_dag(self, solver: ASPSolverBridge) -> None:
        res = solver.solveDAGSchedule("[]", max_parallel=1, timeout_s=5.0)
        assert res.at("#satisfiable") is True
        assert res.at("#makespan") == 0
        assert res.at("#optimum") is True
        assert _to_py(res.at("#schedule")) == []

    def test_missing_dependency_reported(self, solver: ASPSolverBridge) -> None:
        tasks = [{"id": "A", "duration": 1, "deps": ["GHOST"]}]
        res = solver.solveDAGSchedule(json.dumps(tasks), max_parallel=1, timeout_s=5.0)
        assert res.at("#satisfiable") is False
        assert "missing_dep:GHOST->A" in _to_py(res.at("#unsatCore"))
        assert "GHOST" in res.at("#error")


class TestBridgeAndDegradation:
    def test_describe_and_bridge_exposure(self, solver: ASPSolverBridge) -> None:
        doc = solver.describe()
        assert "ASPSolverBridge" in doc
        assert "solveProgram" in doc
        assert "verifyInvariants" in doc
        assert "solveDAGSchedule" in doc
        assert "0 MB GPU VRAM" in doc

        bridge_obj = ActorBridgeObject(target=solver)
        caps = bridge_obj.bridgeCapabilities()
        assert caps["solveProgram"] == 0
        assert caps["verifyInvariants"] == 0
        assert caps["solveDAGSchedule"] == 0
        assert caps["describe"] == 0

    def test_graceful_degradation_when_clingo_unavailable(
        self, solver: ASPSolverBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(asp_mod, "clingo", None)
        res1 = solver.solveProgram("a.", max_models=1, timeout_s=1.0)
        assert res1.at("#satisfiable") is False
        assert "clingo is not installed" in res1.at("#error")

        res2 = solver.verifyInvariants("a.", ":- a.", timeout_s=1.0)
        assert res2.at("#ok") is False
        assert "clingo is not installed" in res2.at("#error")

        res3 = solver.solveDAGSchedule('[{"id":"A","duration":1,"deps":[]}]', max_parallel=1, timeout_s=1.0)
        assert res3.at("#satisfiable") is False
        assert "clingo is not installed" in res3.at("#error")

        assert "unavailable" in solver.describe()
