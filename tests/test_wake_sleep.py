"""Unit tests for `ConsolidatorBridge` (`actorii.harness.wake_sleep`)."""

from __future__ import annotations

import json
import pytest

from actorii.bridge import expose_python_object
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.harness import wake_sleep as ws_mod
from actorii.harness.wake_sleep import ConsolidatorBridge
from actorii.vm import ActorVM


def _to_py(val):
    """Recursively unwrap Actor `Dictionary` and `ActorArray` to Python builtins."""
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
def consolidator() -> ConsolidatorBridge:
    # Use offline mode (neo4j_url="") so unit tests never mutate live Neo4j
    return ConsolidatorBridge(vm=ActorVM(), neo4j_url="", namespace="test_ns")


SAMPLE_EPISODES = [
    {
        "session_id": "s1",
        "actions": [
            'queryRows(Actor[#GraphService], "MATCH (n:Task) RETURN n.name")!!',
            'atPut(stats, #count, 1)!!',
        ],
    },
    {
        "session_id": "s2",
        "actions": [
            'queryRows(Actor[#GraphService], "MATCH (n:DevPlan) RETURN n.name")!!',
            'atPut(stats, #count, 2)!!',
        ],
    },
    {
        "session_id": "s3",
        "actions": [
            'queryRows(Actor[#GraphService], "MATCH (n:ActorEngram) RETURN n.name")!!',
            'atPut(stats, #count, 3)!!',
        ],
    },
]

BILBY_CONTENT_EPISODES = [
    {
        "session_id": "live_bilby",
        "turn_range": "Turn 1",
        "content": (
            'User: Check tasks\n'
            'Bilby [Actions: queryRows(Actor[#GraphService], "MATCH (t:Task) RETURN t.name")!!; '
            'countActive(Actor[#DreamingCycle])!!]: Done.'
        ),
    },
    {
        "session_id": "live_bilby",
        "turn_range": "Turn 2",
        "content": (
            'User: Check plans\n'
            'Bilby [Actions: queryRows(Actor[#GraphService], "MATCH (p:DevPlan) RETURN p.name")!!; '
            'countActive(Actor[#DreamingCycle])!!]: Done.'
        ),
    },
    {
        "session_id": "live_bilby",
        "turn_range": "Turn 3",
        "content": (
            'User: Check engrams\n'
            'Bilby [Actions: queryRows(Actor[#GraphService], "MATCH (e:ActorEngram) RETURN e.name")!!; '
            'countActive(Actor[#DreamingCycle])!!]: Done.'
        ),
    },
]


class TestCompressEpisodes:
    def test_compress_structured_episodes_native_or_fallback(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        res = consolidator.compressEpisodes(
            json.dumps(SAMPLE_EPISODES), min_support=2, max_abstractions=5
        )
        assert isinstance(res, Dictionary)
        assert res.at("#error") == ""
        assert res.at(Symbol("episodeCount")) == 3
        assert res.at("#compressionRatio") >= 1.0
        backend = res.at("#backend")
        assert backend in ("stitch-core", "python-anti-unification")
        if ws_mod.stitch_core is not None:
            assert backend == "stitch-core"

        abstractions = _to_py(res.at("#abstractions"))
        assert len(abstractions) >= 1
        top = abstractions[0]
        assert top["id"].startswith("fn_")
        assert isinstance(top["template"], str) and len(top["template"]) > 0
        assert isinstance(top["params"], list)
        assert top["support"] >= 2
        assert top["estimatedTokensSaved"] > 0

    def test_compress_bilby_episode_content_format(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        res = consolidator.compressEpisodes(
            json.dumps(BILBY_CONTENT_EPISODES), min_support=2, max_abstractions=3
        )
        assert res.at("#error") == ""
        abstractions = _to_py(res.at("#abstractions"))
        assert len(abstractions) >= 1
        assert any(a["support"] >= 2 for a in abstractions)

    def test_compress_python_anti_unification_fallback(
        self, consolidator: ConsolidatorBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ws_mod, "stitch_core", None)
        res = consolidator.compressEpisodes(
            json.dumps(SAMPLE_EPISODES), min_support=2, max_abstractions=5
        )
        assert res.at("#backend") == "python-anti-unification"
        assert res.at("#error") == ""
        abstractions = _to_py(res.at("#abstractions"))
        assert len(abstractions) >= 1
        assert abstractions[0]["support"] >= 2
        assert res.at("#compressionRatio") > 1.0

    def test_compress_empty_input(self, consolidator: ConsolidatorBridge) -> None:
        res = consolidator.compressEpisodes("[]", min_support=2, max_abstractions=5)
        assert res.at("#error") == ""
        assert _to_py(res.at("#abstractions")) == []
        assert res.at("#compressionRatio") == 1.0


class TestProveEquivalence:
    @pytest.mark.parametrize(
        "expr_a,expr_b",
        [
            ("x + 0", "x"),
            ("a + b", "b + a"),
            ("x + x", "2 * x"),
            ("--x", "x"),
            ("a * (b + c)", "a * b + a * c"),
            ("x - x", "0"),
            ("!(a && b)", "(!a) || (!b)"),
            ("!(a || b)", "(!a) && (!b)"),
            ("!!flag", "flag"),
            ("p && true", "p"),
            ("p || !p", "true"),
        ],
    )
    def test_equivalent_expressions(
        self, consolidator: ConsolidatorBridge, expr_a: str, expr_b: str
    ) -> None:
        res = consolidator.proveEquivalence(expr_a, expr_b)
        assert isinstance(res, Dictionary)
        assert res.at("#error") == ""
        assert res.at("#equivalent") is True
        assert isinstance(res.at("#canonicalForm"), str) and len(res.at("#canonicalForm")) > 0
        if ws_mod._EgglogEGraph is not None:
            assert res.at("#backend") == "egglog"

    @pytest.mark.parametrize(
        "expr_a,expr_b",
        [
            ("x + 1", "x"),
            ("a - b", "b - a"),
            ("a && b", "a || b"),
            ("x * 2", "x * 3"),
        ],
    )
    def test_non_equivalent_expressions(
        self, consolidator: ConsolidatorBridge, expr_a: str, expr_b: str
    ) -> None:
        res = consolidator.proveEquivalence(expr_a, expr_b)
        assert res.at("#error") == ""
        assert res.at("#equivalent") is False

    def test_python_egraph_fallback(
        self, consolidator: ConsolidatorBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ws_mod, "_EgglogEGraph", None)
        eq_res = consolidator.proveEquivalence("!(a && b)", "(!a) || (!b)")
        assert eq_res.at("#backend") == "python-egraph"
        assert eq_res.at("#equivalent") is True
        neq_res = consolidator.proveEquivalence("x + 1", "x + 2")
        assert neq_res.at("#backend") == "python-egraph"
        assert neq_res.at("#equivalent") is False

    def test_parse_error_handled_cleanly(self, consolidator: ConsolidatorBridge) -> None:
        res = consolidator.proveEquivalence("x + + * )", "x")
        assert res.at("#equivalent") is False
        assert "ParseError" in res.at("#error")


class TestReflexPromotionAndMatch:
    def test_register_reflex_passing_selftest_and_shadow_rollback(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        src = (
            "inherit(Object, #QuickSumReflex, #(), 0, nil)!!\n"
            "now(QuickSumReflex)!!\n"
            "Def addPair(self, a, b) { ^(a + b); }!!\n"
            "Actor[#QuickSumReflex] := new(QuickSumReflex)!!"
        )
        self_test = "(addPair(Actor[#QuickSumReflex], 19, 23) == 42)!!"
        res = consolidator.registerReflex(
            "QuickSumReflex", "compute quick sum of two numbers", src, self_test
        )
        assert res.at("#promoted") is True
        assert res.at("#name") == "QuickSumReflex"
        assert res.at("#error") == ""
        assert res.at("#elapsedMs") >= 0.0
        # Verify shadow_only=True rolled back VM state (no leaked class in live VM)
        assert consolidator.vm.hierarchy.get("QuickSumReflex") is None

    def test_register_reflex_failing_selftest_rejected(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        src = (
            "inherit(Object, #BadReflex, #(), 0, nil)!!\n"
            "now(BadReflex)!!\n"
            "Def alwaysWrong(self) { ^1; }!!\n"
            "Actor[#BadReflex] := new(BadReflex)!!"
        )
        self_test = "(alwaysWrong(Actor[#BadReflex]) == 999)!!"
        res = consolidator.registerReflex("BadReflex", "broken reflex", src, self_test)
        assert res.at("#promoted") is False
        assert "ShadowSelfTestFailed" in res.at("#error")
        # Must not be registered in reflex cache
        hit = consolidator.matchReflex("broken reflex")
        assert hit.at("#matched") is False

    def test_register_reflex_blocks_tier2_host_calls(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        res = consolidator.registerReflex(
            "UnsafeReflex",
            "danger shell",
            'runShell(Actor[#Host], "rm -rf /")!!',
            "true!!",
        )
        assert res.at("#promoted") is False
        assert "Forbidden" in res.at("#error")

    def test_match_reflex_sub_millisecond_lookup(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        res = consolidator.registerReflex(
            "PlanStatusReflex",
            "check dev plan status",
            "1 + 1!!",
            "(1 + 1 == 2)!!",
        )
        assert res.at("#promoted") is True

        # Exact / substring / token-overlap lookups
        hit = consolidator.matchReflex("please check dev plan status now")
        assert hit.at("#matched") is True
        assert hit.at("#name") == "PlanStatusReflex"
        assert hit.at("#actorSource") == "1 + 1!!"
        assert hit.at("#elapsedMs") < 1.0

        miss = consolidator.matchReflex("completely unrelated astronomy query")
        assert miss.at("#matched") is False
        assert miss.at("#name") == ""
        assert miss.at("#elapsedMs") < 1.0

    def test_match_reflex_regex_pattern(self, consolidator: ConsolidatorBridge) -> None:
        res = consolidator.registerReflex(
            "DreamMetricReflex",
            "re:dreaming\\s+cycle\\s+(status|metrics)",
            "42!!",
            "true!!",
        )
        assert res.at("#promoted") is True
        hit = consolidator.matchReflex("show dreaming cycle metrics")
        assert hit.at("#matched") is True
        assert hit.at("#name") == "DreamMetricReflex"
        assert hit.at("#elapsedMs") < 1.0


class TestConsolidationCycleAndDescribe:
    def test_run_consolidation_cycle_end_to_end(
        self, consolidator: ConsolidatorBridge
    ) -> None:
        for ep in BILBY_CONTENT_EPISODES:
            consolidator.recordEpisodeTrace(ep)

        dry_res = consolidator.runConsolidationCycle(max_episodes=20, dry_run=True)
        assert dry_res.at("#ok") is True
        assert dry_res.at("#dryRun") is True
        assert dry_res.at("#epoch") == 0
        assert dry_res.at("#episodesAnalyzed") == 3
        assert dry_res.at("#tracesExtracted") == 3
        assert len(_to_py(dry_res.at("#abstractions"))) >= 1
        assert "Wake-Sleep Consolidation" in dry_res.at("#summary")

        live_res = consolidator.runConsolidationCycle(max_episodes=20, dry_run=False)
        assert live_res.at("#ok") is True
        assert live_res.at("#dryRun") is False
        assert live_res.at("#epoch") == 1

    def test_describe_and_actor_vm_bridge_integration(self) -> None:
        vm = ActorVM()
        bridge_py = ConsolidatorBridge(vm=vm, neo4j_url="", namespace="test_ns")
        expose_python_object(vm, bridge_py, global_key="Consolidator")

        doc = vm.eval_transactional("describe(Actor[#Consolidator])!!", shadow_only=True)
        assert isinstance(doc, str)
        assert "ConsolidatorBridge" in doc
        assert "DreamingCycle" in doc
        assert "0 MB GPU VRAM" in doc

        eq = vm.eval_transactional(
            'proveEquivalence(Actor[#Consolidator], "x + 0", "x")!!', shadow_only=True
        )
        assert isinstance(eq, Dictionary)
        assert eq.at("#equivalent") is True
