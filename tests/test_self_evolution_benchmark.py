"""Tests for the Bilby Self-Evolution Loop Benchmark (`actorii.harness.self_evolution_bench`).

Covers: spec schema/JSON loading, Phase 0 baseline oracle, deterministic multi-generation
self-correction (every state-machine transition), rollback gates, holdout anti-reward-hacking,
persistence/reload gate, CLI JSON/Markdown output, and optional live (`:8080`) / engram runs.

Run (isolated):
    cd /home/skippy/projects/ActorII && ACTORII_STATE_DIR=/tmp/actorii_evo_test \
      ACTORII_MEMORY_NAMESPACE=pytest_evo PYTHONPATH=src pytest tests/test_self_evolution_benchmark.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import urllib.request
from pathlib import Path
from typing import List

import pytest

from actorii.harness import self_evolution_bench as seb
from actorii.harness.self_evolution_bench import (
    DEFAULT_EVOLUTION_SPECS,
    DIAGNOSTICS_MARKER,
    EvolutionSpec,
    EvolutionTest,
    SelfEvolutionRunner,
    definitional_chunks,
    load_spec_file,
    normalize_value,
    render_markdown,
    run_oracle,
    run_self_evolution_spec,
    values_equal,
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DOUBLER_GOOD = """
inherit(Object, #Doubler, #(#unused), 1, nil)!!
now(Doubler)!!
Def twice(self, x) { ^(x * 2); }!!
"""
DOUBLER_LOOKUP = """
inherit(Object, #Doubler, #(#unused), 1, nil)!!
now(Doubler)!!
Def twice(self, x) {
    if x == 4 then ^8; endif;
    ^0;
}!!
"""
DOUBLER_BROKEN_SYNTAX = "inherit(Object, #Doubler, #(#unused), 1, nil)!!\nnow(Doubler)!!\nDef twice(self, x { ^(x * 2) }!!\n"


def _dbl(x: int) -> str:
    return f"Actor[#d] := new(Doubler)!! twice(Actor[#d], {x})!!"


def mini_spec(gens: List[str], **overrides) -> EvolutionSpec:
    base = dict(
        spec_id="mini_doubler",
        name="Doubler",
        difficulty=seb.TIER_CLASS,
        goal="Create class Doubler with twice(self, x) -> x * 2.",
        spec_prompt="`Doubler>>twice(self, x)` returns `x * 2` for any integer.",
        visible_tests=[EvolutionTest(_dbl(4), 8, "visible")],
        holdout_tests=[
            EvolutionTest(_dbl(5), 10, "positive"),
            EvolutionTest(_dbl(-3), -6, "negative"),
            EvolutionTest(_dbl(0), 0, "zero"),
        ],
        max_generations=3,
        required_classes=["Doubler"],
        scripted_generations=gens,
    )
    base.update(overrides)
    return EvolutionSpec(**base)


@pytest.fixture()
def runner(tmp_path):
    with SelfEvolutionRunner("scripted", work_dir=tmp_path / "evo") as r:
        yield r


# ---------------------------------------------------------------------------
# schema, normalisation, oracle
# ---------------------------------------------------------------------------


def test_default_suite_schema_tiers_and_catalogue() -> None:
    specs = DEFAULT_EVOLUTION_SPECS
    assert len(specs) >= 5
    assert {s.difficulty for s in specs.values()} == set(seb.VALID_TIERS)
    for sid, s in specs.items():
        assert sid == s.spec_id
        assert s.visible_tests and len(s.holdout_tests) > len(s.visible_tests) - 1
        assert len(s.scripted_generations) >= 2, "scripted solver needs a flawed gen1 and a fixed gen2"
        assert s.max_generations >= 2 and s.max_steps_per_gen >= 2
        # holdout expressions must never leak into the prompt shown to Bilby
        prompt = SelfEvolutionRunner("scripted").build_prompt(s, 1, [], "")
        for h in s.holdout_tests:
            assert h.expr.strip() not in prompt
        for v in s.visible_tests:
            assert v.expr.strip() in prompt
        assert EvolutionSpec.from_dict(json.loads(json.dumps(s.to_dict()))).to_dict() == s.to_dict()
    cat = seb.list_self_evolution_suites()
    for sid in specs:
        assert f"#{sid}" in cat


def test_spec_validation_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        mini_spec([DOUBLER_GOOD], holdout_tests=[])
    with pytest.raises(ValueError):
        mini_spec([DOUBLER_GOOD], difficulty="tier9")
    with pytest.raises(ValueError):
        mini_spec([DOUBLER_GOOD], spec_id="bad id!")
    with pytest.raises(ValueError):
        mini_spec([DOUBLER_GOOD], required_classes=[])  # persistence requires classes
    with pytest.raises(ValueError):
        EvolutionSpec.from_dict({"spec_id": "x"})


def test_normalize_equal_and_oracle_helpers() -> None:
    from actorii.core import Symbol

    assert normalize_value(Symbol("open")) == "#open"
    assert normalize_value([1, Symbol("a"), [2]]) == [1, "#a", [2]]
    assert values_equal(1, True) and values_equal(0, False)
    assert values_equal(0.1 + 0.2, 0.3)
    assert values_equal([1, 2], [1, 2]) and not values_equal([1, 2], [1, 2, 3])
    assert not values_equal(None, 0)
    assert run_oracle("result == sorted(result) and len(result) == 3", [1, 2, 3])
    with pytest.raises(Exception):
        run_oracle("__import__('os').getcwd()", 1)  # no unrestricted builtins


def test_definitional_chunks_drop_probes_and_side_effects() -> None:
    src = DOUBLER_GOOD + '\nActor[#d] := new(Doubler)!!\ntwice(Actor[#d], 4)!!\nstoreScratch(Actor[#Harness], #k, 1)!!'
    chunks = definitional_chunks(src)
    assert len(chunks) == 3 and all(c.endswith("!!") for c in chunks)
    assert not any("Actor[#d]" in c for c in chunks)


# ---------------------------------------------------------------------------
# Phase 0 - baseline oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", sorted(DEFAULT_EVOLUTION_SPECS))
def test_phase0_baseline_not_fully_passing_for_every_builtin(sid: str, runner) -> None:
    spec = DEFAULT_EVOLUTION_SPECS[sid]
    vm, _, _ = runner._boot_vm(spec, "baseline_check")
    vis, hold = runner._score(vm, spec)
    passed = sum(o.passed for o in vis) + sum(o.passed for o in hold)
    assert passed < len(vis) + len(hold)
    assert sum(o.passed for o in vis) < len(vis), "visible tests must fail before evolution"


def test_phase0_invalid_baseline_is_reported_not_run(runner) -> None:
    trivial = mini_spec(
        [DOUBLER_GOOD],
        visible_tests=[EvolutionTest("1 + 1!!", 2)],
        holdout_tests=[EvolutionTest("2 + 2!!", 4)],
    )
    res = runner.run_spec(trivial)
    assert res["status"] == "invalid_baseline"
    assert res["generations"] == [] and not res["passed"]


# ---------------------------------------------------------------------------
# Deterministic state machine: self-correction, every built-in spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", sorted(DEFAULT_EVOLUTION_SPECS))
def test_scripted_builtin_spec_fails_gen1_then_self_corrects_and_persists(sid: str, tmp_path) -> None:
    res = run_self_evolution_spec(sid, "scripted", work_dir=tmp_path / sid)
    assert res["status"] == "passed", res
    assert res["pass_at_g"] is True and res["pass_at_1"] is False
    assert res["generations_used"] == 2
    g1, g2 = res["generations"]
    assert g1["diagnostics_given"] is False and g2["diagnostics_given"] is True
    assert not g1["all_tests_passed"] and g2["all_tests_passed"]
    assert g1["visible_passed"] < g1["visible_total"]
    assert g2["visible_passed"] == g2["visible_total"] and g2["holdout_passed"] == g2["holdout_total"]
    assert res["holdout_pass_rate"] == 1.0 and res["visible_pass_rate"] == 1.0
    assert res["persistence_verified"] is True
    assert res["steps_used"] == g1["steps"] + g2["steps"] == 4
    assert res["gas_used"] > 0 and res["latency_s"] > 0
    assert res["reward_hacking_suspected"] is False
    assert "Def " in res["final_source"]
    assert res["baseline"]["visible_passed"] < res["baseline"]["visible_total"]


def test_self_correction_requires_diagnostics_to_reach_the_model(tmp_path, monkeypatch) -> None:
    """If the harness stops feeding diagnostics, the scripted solver cannot self-correct (wiring check)."""
    orig = SelfEvolutionRunner.build_prompt

    def no_diag(self, spec, generation, committed, diagnostics):
        return orig(self, spec, generation, committed, "")

    monkeypatch.setattr(SelfEvolutionRunner, "build_prompt", no_diag)
    res = run_self_evolution_spec("sensor_window_median", "scripted", work_dir=tmp_path / "nd")
    assert res["status"] == "failed" and res["generations_used"] == 3
    assert all(not g["diagnostics_given"] or g["generation"] > 1 for g in res["generations"])


def test_diagnostics_contain_visible_failures_but_hide_holdout_details(runner) -> None:
    spec = DEFAULT_EVOLUTION_SPECS["circuit_breaker_policy"]
    runner.max_generations_override = 1
    res = runner.run_spec(spec)
    assert res["status"] == "failed" and res["generations_used"] == 1
    g1 = res["generations"][0]
    assert g1["failed_visible"], "visible failures must be recorded for diagnostics"
    prompt_diag = runner._diagnostics(
        1, g1, [], [], [], [], []
    )
    assert prompt_diag.startswith(DIAGNOSTICS_MARKER)


def test_prompt_diagnostics_text_for_generation_two(tmp_path) -> None:
    seen: List[str] = []
    spec = DEFAULT_EVOLUTION_SPECS["ring_buffer_telemetry"]
    orig = SelfEvolutionRunner._scripted_provider

    def spy(self, spec_, generation):
        inner = orig(self, spec_, generation)

        def provider(prompt, sp, opts):
            seen.append(prompt)
            return inner(prompt, sp, opts)

        return provider

    SelfEvolutionRunner._scripted_provider = spy  # type: ignore[assignment]
    try:
        res = run_self_evolution_spec(spec, "scripted", work_dir=tmp_path / "spy")
    finally:
        SelfEvolutionRunner._scripted_provider = orig  # type: ignore[assignment]
    assert res["passed"]
    gen2_prompts = [p for p in seen if "generation 2 of" in p]
    assert gen2_prompts
    p2 = gen2_prompts[0]
    assert DIAGNOSTICS_MARKER in p2 and "visible tests: 1/4 passed" in p2
    assert "hidden holdout" in p2 and "details are withheld" in p2
    assert "CURRENT COMMITTED SOURCE" in p2
    for h in spec.holdout_tests:  # never leaked, not even via diagnostics
        assert h.expr.strip() not in p2


# ---------------------------------------------------------------------------
# Rollback & anti-regression gates
# ---------------------------------------------------------------------------


def test_syntax_error_generation_is_rolled_back_cleanly_then_recovers(runner) -> None:
    res = runner.run_spec(mini_spec([DOUBLER_BROKEN_SYNTAX, DOUBLER_GOOD]))
    assert res["status"] == "passed"
    g1, g2 = res["generations"]
    assert g1["outcome"] == "rolled_back" and g1["rollback_reason"] == "no_definitions"
    assert g1["definitional_chunks"] <= 2 and g1["visible_passed"] == 0  # malformed Def never became a method
    assert g2["outcome"] == "accepted" and g2["diagnostics_given"]
    assert res["rollbacks"] == 1
    assert res["final_source"].count("Def twice") == 1  # broken gen leaked nothing into committed source


def test_regression_generation_is_rolled_back(runner) -> None:
    # gen1 correct on everything but visible-wrong? -> use: gen1 good (accepted, all pass) is terminal, so craft:
    # gen1 partially good (lookup: passes visible only), gen2 strictly worse (always 0) -> regression rollback
    worse = DOUBLER_LOOKUP.replace("^8;", "^1;")
    spec = mini_spec([DOUBLER_LOOKUP, worse, DOUBLER_GOOD], max_generations=3)
    # scripted solver only emits gen>=2 code when diagnostics are present, which they will be.
    res = runner.run_spec(spec)
    outs = [g["outcome"] for g in res["generations"]]
    assert outs[0] == "accepted"
    assert outs[1] == "rolled_back" and res["generations"][1]["rollback_reason"] == "regression"
    assert res["status"] == "passed" and res["generations"][2]["outcome"] == "accepted"
    assert "^1;" not in res["final_source"]  # the regressed generation leaked nothing into committed source
    assert res["final_source"].rstrip().endswith("Def twice(self, x) { ^(x * 2); }!!")


def test_gas_budget_violation_rolls_back_generation(runner) -> None:
    res = runner.run_spec(mini_spec([DOUBLER_GOOD], gas_budget=5, max_generations=1))
    g = res["generations"][0]
    assert g["outcome"] == "rolled_back" and g["rollback_reason"] == "invariant_violation"
    assert any(v.startswith("gas:") for v in g["invariant_violations"])
    assert res["status"] == "failed" and res["final_source"] == ""


def test_tamper_invariant_detects_harness_mutation(runner) -> None:
    spec = mini_spec([DOUBLER_GOOD])
    vm, _, _ = runner._boot_vm(spec, "tamper")
    base = seb.vm_fingerprint(vm)
    assert seb.check_invariants(vm, base, spec) == []
    vm.eval_chunk('now(AgentHarness)!! Def policyNote(self) { ^"TAMPERED"; }!!')
    viol = seb.check_invariants(vm, base, spec)
    assert any("AgentHarness>>policyNote" in v for v in viol)
    spec.allow_harness_mutation = True  # Ring-1 mutable selectors may be allowed explicitly
    assert seb.check_invariants(vm, base, spec) == []


# ---------------------------------------------------------------------------
# Holdout anti-cheating
# ---------------------------------------------------------------------------


def test_hardcoded_lookup_passes_visible_but_fails_holdout(runner) -> None:
    res = runner.run_spec(mini_spec([DOUBLER_LOOKUP], max_generations=2))
    assert res["status"] == "failed" and not res["passed"]
    assert res["reward_hacking_suspected"] is True
    g1 = res["generations"][0]
    assert g1["visible_passed"] == g1["visible_total"] == 1
    assert g1["holdout_passed"] < g1["holdout_total"]
    assert g1.get("visible_overfit") is True
    assert g1["failed_holdout_tags"], "failing holdout categories are reported by tag only"
    assert res["persistence_verified"] is False


def test_holdout_expressions_never_shown_to_the_model(tmp_path) -> None:
    seen: List[str] = []
    orig = SelfEvolutionRunner._scripted_provider

    def spy(self, spec_, generation):
        inner = orig(self, spec_, generation)

        def provider(prompt, sp, opts):
            seen.append(prompt)
            return inner(prompt, sp, opts)

        return provider

    SelfEvolutionRunner._scripted_provider = spy  # type: ignore[assignment]
    try:
        spec = mini_spec([DOUBLER_LOOKUP, DOUBLER_GOOD])
        with SelfEvolutionRunner("scripted", work_dir=tmp_path / "hold") as r:
            r.run_spec(spec)
    finally:
        SelfEvolutionRunner._scripted_provider = orig  # type: ignore[assignment]
    assert seen
    for prompt in seen:
        for h in spec.holdout_tests:
            assert h.expr.strip() not in prompt


# ---------------------------------------------------------------------------
# Phase 3 - persistence / reload gate
# ---------------------------------------------------------------------------

STATEFUL_GEN = """
Actor[#Cfg] := 10!!
inherit(Object, #Doubler, #(#unused), 1, nil)!!
now(Doubler)!!
Def twice(self, x) { ^(x * Actor[#Cfg]) / 5; }!!
"""


def test_capability_depending_on_unpersisted_global_state_fails_reload_gate(runner) -> None:
    # In the generation VM `Actor[#Cfg]` exists (set by a non-definitional chunk Bilby ran), so every test passes
    # in place - but it is not part of the persisted class, so a brand-new VM cannot reproduce the capability.
    res = runner.run_spec(mini_spec([STATEFUL_GEN], max_generations=1))
    g = res["generations"][0]
    assert g["visible_passed"] == g["visible_total"] and g["holdout_passed"] == g["holdout_total"]
    assert g["persistence_verified"] is False and "reloaded VM passed" in g["persistence_error"]
    assert g["outcome"] == "rolled_back" and g["rollback_reason"] == "persistence_failed"
    assert res["status"] == "failed" and res["persistence_verified"] is False
    assert res["final_source"] == ""


def test_persistence_reload_uses_fileout_into_fresh_vm(tmp_path) -> None:
    res = run_self_evolution_spec("ring_buffer_telemetry", "scripted", work_dir=tmp_path / "persist")
    assert res["persistence_verified"] is True
    out = tmp_path / "persist" / "fileout" / "ring_buffer_telemetry" / "RingTelemetry.act"
    text = out.read_text(encoding="utf-8")
    assert "inherit(Object, #RingTelemetry" in text and "Def percentile" in text
    # an independent brand-new VM can load that file-out and use the capability
    from actorii.vm import ActorVM

    vm = ActorVM()
    vm.eval_chunk(text)
    vm.eval_chunk("Actor[#r] := initCapacity(new(RingTelemetry), 3)!!")
    for v in (1, 2, 3, 4, 5):
        vm.eval_chunk(f"record(Actor[#r], {v})!!")
    assert vm.eval_chunk("windowSum(Actor[#r])!!") == 12


def test_require_persistence_false_skips_reload_gate(runner) -> None:
    spec = mini_spec([DOUBLER_GOOD], require_persistence=False, required_classes=[], max_generations=1)
    res = runner.run_spec(spec)
    assert res["passed"] and res["persistence_required"] is False
    assert res["persistence_verified"] is False
    assert not (runner.work_dir / "fileout").exists()


# ---------------------------------------------------------------------------
# JSON spec files & CLI
# ---------------------------------------------------------------------------


def _write_custom_spec(path: Path) -> None:
    spec = mini_spec([DOUBLER_LOOKUP, DOUBLER_GOOD], spec_id="custom_json_doubler")
    path.write_text(json.dumps({"specs": [spec.to_dict()]}), encoding="utf-8")


def test_custom_json_spec_file_round_trip_and_run(tmp_path) -> None:
    p = tmp_path / "spec.json"
    _write_custom_spec(p)
    (spec,) = load_spec_file(p)
    assert spec.spec_id == "custom_json_doubler" and len(spec.holdout_tests) == 3
    # compact tuple form `[expr, expected]` is accepted too
    compact = {
        "spec_id": "compact", "name": "c", "difficulty": seb.TIER_METHOD, "goal": "g", "spec_prompt": "p",
        "visible_tests": [["1 + 1!!", 3]], "holdout_tests": [{"expr": "2 + 2!!", "oracle": "result == 5"}],
        "require_persistence": False, "scripted_generations": ["now(Object)!!"],
    }
    (c,) = [EvolutionSpec.from_dict(compact)]
    assert c.visible_tests[0].expected == 3 and c.holdout_tests[0].oracle == "result == 5"
    with pytest.raises(ValueError):
        EvolutionSpec.from_dict({**compact, "bogus": 1})
    res = run_self_evolution_spec(spec, "scripted", work_dir=tmp_path / "w")
    assert res["status"] == "passed" and res["generations_used"] == 2


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_self_evolution_benchmark", ROOT / "scripts" / "run_self_evolution_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_scripted_all_specs_writes_json_and_markdown(tmp_path, capsys) -> None:
    cli = _load_cli()
    out_json, out_md = tmp_path / "r" / "out.json", tmp_path / "r" / "out.md"
    spec_file = tmp_path / "custom.json"
    _write_custom_spec(spec_file)
    rc = cli.main([
        "--mode", "scripted", "--specs", "all", "--spec-file", str(spec_file), "--strict",
        "--output-json", str(out_json), "--output-md", str(out_md), "--quiet",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["mode"] == "scripted" and data["temperature"] == 0.0 and data["seed"] == 42
    ids = [s["spec_id"] for s in data["specs"]]
    assert set(DEFAULT_EVOLUTION_SPECS) <= set(ids) and "custom_json_doubler" in ids
    summ = data["summary"]
    assert summ["specs"] == len(ids) and summ["pass_at_g"] == 1.0 and summ["pass_at_1"] == 0.0
    assert summ["persistence_verified"] == len(ids) and summ["errors"] == 0
    for key in ("pass_at_1", "pass_at_g", "generations_used", "steps_used", "gas_used", "holdout_pass_rate",
                "persistence_verified", "latency_s"):
        assert all(key in s for s in data["specs"])
    md = out_md.read_text(encoding="utf-8")
    assert md.startswith("# ActorII Bilby Self-Evolution Benchmark Results")
    assert "| `ring_buffer_telemetry` |" in md and "pass@G" in md and "## Per-generation detail" in md
    assert md == render_markdown(data)


def test_cli_list_and_error_exit_codes(tmp_path, capsys) -> None:
    cli = _load_cli()
    assert cli.main(["--list"]) == 0
    assert "ring_buffer_telemetry" in capsys.readouterr().out
    assert cli.main(["--specs", "no_such_spec"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli.main(["--spec-file", str(bad)]) == 2


def test_cli_strict_exit_code_on_failed_spec(tmp_path) -> None:
    cli = _load_cli()
    spec = mini_spec([DOUBLER_LOOKUP], spec_id="cheater", max_generations=1)
    f = tmp_path / "cheat.json"
    f.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert cli.main(["--spec-file", str(f), "--specs", "cheater", "--strict", "--quiet"]) == 1
    assert cli.main(["--spec-file", str(f), "--specs", "cheater", "--quiet"]) == 0


def test_runner_scopes_env_and_does_not_touch_real_actorii_dir(tmp_path) -> None:
    before = {k: os.environ.get(k) for k in ("ACTORII_STATE_DIR", "ACTORII_MEMORY_NAMESPACE", "ACTORII_HOST_LOCKDOWN")}
    home_state = Path.home() / ".actorii"
    mtime = home_state.stat().st_mtime if home_state.exists() else None
    run_self_evolution_spec("circuit_breaker_policy", "scripted", work_dir=tmp_path / "env")
    after = {k: os.environ.get(k) for k in before}
    assert before == after
    assert (home_state.stat().st_mtime if home_state.exists() else None) == mtime


# ---------------------------------------------------------------------------
# Optional: live Bilby (:8080) and engram persistence
# ---------------------------------------------------------------------------


def _endpoint_up(url: str = "http://127.0.0.1:8080/v1/models") -> bool:
    try:
        urllib.request.urlopen(url, timeout=2.0).read()
        return True
    except Exception:
        return False


def _neo4j_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 7687), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    os.environ.get("ACTORII_RUN_LIVE_EVO") != "1" or not _endpoint_up(),
    reason="set ACTORII_RUN_LIVE_EVO=1 with a live Bilby endpoint on :8080 to run the live self-evolution test",
)
def test_live_bilby_self_evolution_records_full_metrics(tmp_path) -> None:
    res = run_self_evolution_spec(
        "sensor_window_median", "live", max_generations=2, work_dir=tmp_path / "live", llm_timeout=240
    )
    assert res["mode"] == "live" and res["status"] in ("passed", "failed")
    assert res["baseline"]["visible_passed"] < res["baseline"]["visible_total"]
    assert res["generations_used"] >= 1 and res["generations"][0]["llm"] is not None
    assert "llm_error" not in res["generations"][0], res["generations"][0].get("llm_error")
    for key in ("pass_at_1", "pass_at_g", "steps_used", "gas_used", "holdout_pass_rate", "persistence_verified", "latency_s"):
        assert key in res


@pytest.mark.skipif(not _neo4j_up(), reason="Neo4j (bolt :7687) not reachable for engram persistence")
def test_engram_persistence_backend_reloads_capability(tmp_path) -> None:
    res = run_self_evolution_spec(
        "circuit_breaker_policy", "scripted", work_dir=tmp_path / "engram", persistence_backend="engram"
    )
    assert res["status"] == "passed", res.get("generations")
    assert res["persistence_verified"] is True
