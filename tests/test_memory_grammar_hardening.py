"""Unit tests for Phase 2 memory_graph, grammar_graph, linter, judge, and fileout hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from actorii.core import Symbol
from actorii.guardrails import (
    PROTECTED_POOL_SINGLETONS,
    ActorCodeJudge,
    ActorFileOutManager,
    ActorGuardrailError,
    ActorLinter,
    JudgeVerdict,
    LintReport,
    RiskTier,
    ZeroVramSystemOneJudge,
)
from actorii.guardrails.judge import _parse_judge_json_object
from actorii.harness.grammar_graph import ActorGrammarGraph, SEED_ERROR_PATTERNS
from actorii.harness.memory_graph import (
    ActorMemoryGraph,
    _BUILTIN_POOL_KEYS,
    _EMBED_LRU,
    _extract_engram_metadata,
    _split_actor_chunks_safe,
    _strip_cypher_for_guard,
    _topological_sort_engram_deps,
)
from actorii.vm import ActorVM


# =============================================================================
# 1. FileOut Hardening
# =============================================================================


def test_fileout_escapes_newlines_preserves_symbols_and_atomic_write(tmp_path: Path) -> None:
    vm = ActorVM()
    vm.eval_chunk(
        """
        inherit(Object, #NoteHolder, #(#body #tag #count), 3, nil)!!
        now(NoteHolder)!!
        Def getBody(self) { ^self.body; }!!
        Actor[#NoteHolder] := new(NoteHolder)!!
        """
    )
    inst = vm.global_vars["Actor"]["NoteHolder"]
    inst._fields["body"] = 'line1\nline2\t"quoted"\r\n\\end'
    inst._fields["tag"] = Symbol("active")
    inst._fields["count"] = 7

    # In-memory formatting should NOT create output_dir
    lazy_dir = tmp_path / "never_created"
    mgr_lazy = ActorFileOutManager(lazy_dir)
    src = mgr_lazy.format_class_source(vm, "NoteHolder")
    assert not lazy_dir.exists()
    assert '#active' in src
    assert '\\n' in src and '\\t' in src and '\\r' in src

    # Round-trip reload into a fresh ActorVM must succeed cleanly
    vm2 = ActorVM()
    vm2.eval_chunk(src)
    assert vm2.eval_chunk("getBody(Actor[#NoteHolder])!!") == 'line1\nline2\t"quoted"\r\n\\end'
    reloaded = vm2.global_vars["Actor"]["NoteHolder"]
    tag_val = reloaded._fields["tag"]
    assert (isinstance(tag_val, Symbol) and tag_val.name == "active") or tag_val == "#active"

    # Atomic write creates the directory and rejects path traversal
    out_dir = tmp_path / "exported"
    mgr = ActorFileOutManager(out_dir)
    path = mgr.file_out_class(vm, "NoteHolder")
    assert path == out_dir / "NoteHolder.act"
    assert path.is_file()
    with pytest.raises(ValueError, match="Unsafe class name"):
        mgr.file_out_class(vm, "../evil")


# =============================================================================
# 2. Judge Hardening
# =============================================================================


def test_judge_8082_fail_closed_and_json_parsing() -> None:
    assert _parse_judge_json_object('{"approved": true, "confidence": 0.92, "reason": "ok"}')["approved"] is True
    assert _parse_judge_json_object("I do not approve this method") is None

    vm = ActorVM()
    vm.eval_chunk("inherit(Object, #DemoJ, #(#x), 1, nil)!!\nnow(DemoJ)!!\nDef setX(self, v) { self.x := v; ^self.x; }!!")
    cm = vm.hierarchy.get("DemoJ").get_method("setX")
    report = ActorLinter(vm).lint_method(cm, class_name="DemoJ")

    judge = ZeroVramSystemOneJudge(prefer_fast_8082=True, timeout=0.5)

    # Case 1: Malformed text containing "approve" without "false" MUST fail closed!
    fake_resp_bad = mock.MagicMock()
    fake_resp_bad.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "I cannot approve this unsafe mutation."}}]}
    ).encode("utf-8")
    fake_resp_bad.__enter__.return_value = fake_resp_bad
    with mock.patch("urllib.request.urlopen", return_value=fake_resp_bad):
        v_bad = judge.score_method(cm, report)
        assert v_bad.approved is False
        assert v_bad.confidence == 0.0

    # Case 2: Valid JSON approval respects confidence
    fake_resp_ok = mock.MagicMock()
    fake_resp_ok.read.return_value = json.dumps(
        {"choices": [{"message": {"content": '{"approved": true, "confidence": 0.93, "reason": "safe"}'}}]}
    ).encode("utf-8")
    fake_resp_ok.__enter__.return_value = fake_resp_ok
    with mock.patch("urllib.request.urlopen", return_value=fake_resp_ok):
        v_ok = judge.score_method(cm, report)
        assert v_ok.approved is True
        assert v_ok.confidence == pytest.approx(0.93)

    # Case 3: Both endpoints offline -> fail closed cleanly without raising
    with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        v_off = judge.score_method(cm, report)
        assert v_off.approved is False
        assert v_off.engine == "offline_fail_closed"


def test_judge_word_boundary_mutator_check() -> None:
    vm = ActorVM()
    judge = ActorCodeJudge(vm)
    dummy_cm = mock.MagicMock()

    # Query selectors containing substrings like "on", "set", "add" must NOT be treated as mutators
    for query_sel in ("getJson", "isConnected", "getVersion", "getAddress", "getOffset"):
        rep = LintReport(
            class_name="Demo",
            selector=query_sel,
            risk_tier=RiskTier.TIER_1_LOCAL_MUTATION,
            written_fields=["x"],
        )
        verdict = judge.evaluate(dummy_cm, rep, is_synthesized=True)
        assert verdict.approved is False, f"Expected {query_sel} mutation to be blocked"

    # Explicit mutator selectors MUST still be allowed
    for mut_sel in ("setJson", "onClick", "updateBalance", "addItem", "recordAlert"):
        rep = LintReport(
            class_name="Demo",
            selector=mut_sel,
            risk_tier=RiskTier.TIER_1_LOCAL_MUTATION,
            written_fields=["x"],
        )
        verdict = judge.evaluate(dummy_cm, rep, is_synthesized=True)
        assert verdict.approved is True, f"Expected {mut_sel} mutation to be allowed"


# =============================================================================
# 3. Linter Hardening (Singleton Bypass, False Tier-2, Compile-Time Diagnostics)
# =============================================================================


def test_linter_blocks_long_expr_and_var_key_singleton_overwrite_without_false_positives() -> None:
    vm = ActorVM()
    vm.lock_system_singletons = True
    linter = ActorLinter(vm, strict_ivars=True, protect_system_classes=True)

    vm.eval_chunk(
        """
        inherit(Object, #PoolTester, #(#x), 1, nil)!!
        now(PoolTester)!!
        Def bypassLongExpr(self) {
            Actor[#Host] := (1 + 2 + 3 + 4 + 5);
            ^true;
        }!!
        Def bypassVarKey(self | k) {
            k := #Host;
            Actor[k] := nil;
            ^true;
        }!!
        Def innocentLocalMut(self | arr, h) {
            h := Actor[#Host];
            arr := new(Array, 2);
            arr[0] := 42;
            ^arr[0];
        }!!
        Def innocentLocalClear(self | d, h) {
            h := Actor[#Host];
            d := new(Dictionary, 4);
            clear(d);
            ^d;
        }!!
        """
    )
    cls = vm.hierarchy.get("PoolTester")

    rep_long = linter.lint_method(cls.get_method("bypassLongExpr"), class_name="PoolTester")
    assert not rep_long.passed
    assert any(i.code == "SYS_PROTECTED_SINGLETON" for i in rep_long.errors)

    rep_var = linter.lint_method(cls.get_method("bypassVarKey"), class_name="PoolTester")
    assert not rep_var.passed
    assert any(i.code == "SYS_PROTECTED_SINGLETON" for i in rep_var.errors)

    # Innocent local array mutation in a method that reads Actor[#Host] must stay Tier 1 and pass!
    rep_local = linter.lint_method(cls.get_method("innocentLocalMut"), class_name="PoolTester")
    assert rep_local.passed, rep_local.errors
    assert rep_local.risk_tier == RiskTier.TIER_1_LOCAL_MUTATION

    # Innocent clear(localDict) in a method that reads Actor[#Host] must pass!
    rep_clear = linter.lint_method(cls.get_method("innocentLocalClear"), class_name="PoolTester")
    assert rep_clear.passed, rep_clear.errors


def test_linter_catches_atput_reassign_trap_and_symbol_array_index() -> None:
    vm = ActorVM()
    linter = ActorLinter(vm, strict_ivars=True, protect_system_classes=False)

    vm.eval_chunk(
        """
        inherit(Object, #TrapTester, #(), 0, nil)!!
        now(TrapTester)!!
        Def badAtPut(self | d) {
            d := new(Dictionary, 8);
            d := atPut(d, #k, 123);
            ^d;
        }!!
        Def goodAtPut(self | d) {
            d := new(Dictionary, 8);
            atPut(d, #k, 123);
            ^d;
        }!!
        Def badArraySymIndex(self | arr) {
            arr := new(Array, 3);
            ^arr[#score];
        }!!
        """
    )
    cls = vm.hierarchy.get("TrapTester")

    rep_bad_atput = linter.lint_method(cls.get_method("badAtPut"), class_name="TrapTester")
    assert any(i.code == "LINT_ATPUT_REASSIGN_TRAP" for i in rep_bad_atput.errors)

    rep_good_atput = linter.lint_method(cls.get_method("goodAtPut"), class_name="TrapTester")
    assert rep_good_atput.passed, rep_good_atput.errors

    rep_sym_idx = linter.lint_method(cls.get_method("badArraySymIndex"), class_name="TrapTester")
    assert any(i.code == "LINT_SYMBOL_ARRAY_INDEX" for i in rep_sym_idx.errors)


# =============================================================================
# 4. GrammarGraph Hardening
# =============================================================================


def test_grammar_graph_domain_filtering_and_offline_diagnosis() -> None:
    gg = ActorGrammarGraph(tx_url="http://127.0.0.1:1/unreachable", auto_seed=False)

    # Generic coding task should NOT inject Jetson/RTSP/Tailscale/ProductLearner docs
    coding_ctx = gg.queryConstructsForTask("Implement an Array sorting algorithm with Dictionary counts")
    assert "JetsonEdgeTwin" not in coding_ctx
    assert "ProtocolSuite" not in coding_ctx
    assert "NetworkTopology" not in coding_ctx
    assert "web_and_streaming_protocols" not in coding_ctx

    # Removed "== " false positive pattern
    assert not any(ep["pattern"] == "== " for ep in SEED_ERROR_PATTERNS)

    # Offline diagnoseError should still match known SEED_ERROR_PATTERNS locally
    diag = gg.diagnoseError("ValueError: invalid literal for int() with base 10: '#score'")
    assert "OFFLINE" in diag
    assert "atPut" in diag or "Dictionary" in diag

    diag_pass = gg.diagnoseError("ParseError: Unexpected token ('IDENT', 'pass')")
    assert "pass" in diag_pass


# =============================================================================
# 5. MemoryGraph Hardening (Guards, Chunk Splitting, Topo Sort, Cache Stability)
# =============================================================================


def test_memory_graph_cypher_guards_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    vm = ActorVM()
    mg = ActorMemoryGraph(vm, tx_url="http://127.0.0.1:1/unreachable", namespace="default")

    # Unified builtin pool keys include Jetson, Protocols, Network, Threads
    for key in ("Jetson", "Protocols", "Network", "Threads", "ProductLearner"):
        assert key in _BUILTIN_POOL_KEYS
        assert key in PROTECTED_POOL_SINGLETONS

    # Backtick-escaped protected labels must be caught
    assert "ActorEngram" in _strip_cypher_for_guard("MATCH (e:`ActorEngram`) DETACH DELETE e")
    with pytest.raises(PermissionError, match="MEMORY_SELF_PRESERVATION_GUARD"):
        mg._guard_cypher_write("MATCH (e:`ActorEngram`) DETACH DELETE e")

    # Unlabeled variable in DELETE alongside a dummy labeled node must be blocked
    with pytest.raises(PermissionError, match="Unlabeled `DELETE` target"):
        mg._guard_cypher_write("MATCH (x:Dummy), (n) DETACH DELETE n")

    # Cross-namespace read via WHERE or $param or missing namespace under lockdown must be blocked
    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    with pytest.raises(PermissionError, match="MEMORY_NAMESPACE_ISOLATION_GUARD"):
        mg._guard_cypher_read("MATCH (e:ActorEpisode) WHERE e.namespace = 'other' RETURN e")
    with pytest.raises(PermissionError, match="MEMORY_NAMESPACE_ISOLATION_GUARD"):
        mg._guard_cypher_read("MATCH (e:ActorEpisode {namespace: $p1}) RETURN e", param_map={"p1": "other"})
    with pytest.raises(PermissionError, match="MEMORY_NAMESPACE_ISOLATION_GUARD"):
        mg._guard_cypher_read("MATCH (e:ActorEpisode) RETURN e")

    # Safe chunk splitting ignores "!!" inside strings
    src = 'inherit(Object, #BangTest, #(), 0, nil)!!\nnow(BangTest)!!\nDef shout(self) { ^"Hello!! World!!"; }!!'
    chunks = _split_actor_chunks_safe(src)
    assert len(chunks) == 3
    target_cls, sigs, _ = _extract_engram_metadata("BangTest", src)
    assert target_cls == "BangTest"
    assert any("shout(Actor[#BangTest])" in s for s in sigs)

    # Topological sort handles 3+ level Engram chains regardless of input order
    deps = [
        {"name": "Level3", "superclass": "Level2", "actor_source": "x", "dep_names": ["Level2"]},
        {"name": "Level1", "superclass": "Object", "actor_source": "x", "dep_names": []},
        {"name": "Level2", "superclass": "Level1", "actor_source": "x", "dep_names": ["Level1"]},
    ]
    sorted_names = [d["name"] for d in _topological_sort_engram_deps(deps)]
    assert sorted_names == ["Level1", "Level2", "Level3"]


def test_recall_episodes_does_not_invalidate_caches() -> None:
    vm = ActorVM()
    mg = ActorMemoryGraph(vm, tx_url="http://127.0.0.1:1/unreachable", namespace="pytest_cache_check")

    calls: List[str] = []

    def fake_run_batch(statements, *, read_only=False, invalidate_cache=True, timeout=5.0):
        stmt = statements[0]["statement"]
        calls.append(stmt.strip().splitlines()[0])
        if not read_only and invalidate_cache:
            mg._invalidate_caches_for_statement(stmt)
        if "MATCH (ep:ActorEpisode" in stmt and "RETURN" in stmt:
            return [(
                ["episode_hash", "session_id", "session_date", "turn_range", "content", "embedding", "created_at"],
                [{
                    "episode_hash": "h1",
                    "session_id": "s1",
                    "session_date": "2026-11-20",
                    "turn_range": "T1",
                    "content": "User: Jetson TX1 uses omxh264enc | Assistant: Noted.",
                    "embedding": [],
                    "created_at": 1000,
                }],
            )]
        return [([], [])]

    mg._run_cypher_batch = fake_run_batch  # type: ignore[method-assign]
    epoch_before = mg._write_epoch
    engram_epoch_before = mg._engram_epoch

    out1 = mg.recallEpisodes("Jetson omxh264enc", top_k=3)
    assert "omxh264enc" in out1
    assert mg._write_epoch == epoch_before
    assert mg._engram_epoch == engram_epoch_before
    assert "pytest_cache_check" in mg._episode_rows_cache

    # Second recall hits `_episode_rows_cache` without re-querying Neo4j for episode rows
    match_calls_before = sum(1 for c in calls if "MATCH (ep:ActorEpisode" in c)
    out2 = mg.recallEpisodes("Jetson omxh264enc", top_k=3)
    assert out2 == out1
    match_calls_after = sum(1 for c in calls if "MATCH (ep:ActorEpisode" in c)
    assert match_calls_after == match_calls_before
