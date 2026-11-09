"""Tests for Milestone M15: True Metacircular Self-Mutable AgentHarness & Peer Bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from actorii.core import Symbol
from actorii.harness import bootstrap_agent_harness

TEST_NS = "pytest_m15_metacircular"


def test_metacircular_hotpatch_policy_and_transform_action() -> None:
    """Hot-patching `policyNote` and `transformAction` on `AgentHarness` alters `step` in live bytecode."""
    vm, harness, _ = bootstrap_agent_harness(
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=TEST_NS,
    )
    vm.eval_chunk("clearEngrams(Actor[#Memory])!!")

    captured_prompts = []

    def provider(prompt: str, system_prompt: str, metadata: dict) -> str:
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            return "storeScratch(Actor[#Harness], #val, 10)!!"
        return "COMPLETE: Finished"

    vm.set_ai_provider(provider)

    # Hot-patch `AgentHarness>>policyNote` and `AgentHarness>>transformAction` in 1993 Actor 4.1 bytecode
    vm.eval_chunk(
        """
        now(AgentHarness)!!
        Def policyNote(self) {
            ^"ALWAYS_USE_INTEGER_MATH_ONLY";
        }!!
        Def transformAction(self, rawAction) {
            if rawAction == "storeScratch(Actor[#Harness], #val, 10)!!" then
                ^"storeScratch(Actor[#Harness], #val, 42)!!";
            endif;
            ^rawAction;
        }!!
        """
    )

    out = vm.eval_chunk('runGoal(Actor[#Harness], "Compute value", 4)!!')
    assert out == "Finished"
    assert len(captured_prompts) >= 1
    assert "ALWAYS_USE_INTEGER_MATH_ONLY" in captured_prompts[0]
    assert vm.eval_chunk("readScratch(Actor[#Harness], #val)!!") == 42


def test_metacircular_build_step_prompt_and_direct_llm_complete() -> None:
    """Hot-patching `buildStepPrompt` and calling `complete(Actor[#LLM], ...)` works directly in Actor bytecode."""
    vm, harness, _ = bootstrap_agent_harness(
        enable_memory_graph=False,
        enable_engrams=False,
    )

    seen_prompts = []

    def provider(prompt: str, system_prompt: str, metadata: dict) -> str:
        seen_prompts.append(prompt)
        return "COMPLETE: Direct bytecode prompt verified"

    vm.set_ai_provider(provider)

    # 1. Direct `complete(Actor[#LLM], ...)` call from Actor bytecode
    direct_out = vm.eval_chunk('complete(Actor[#LLM], "Ping from Actor bytecode", Actor[#Harness])!!')
    assert "COMPLETE: Direct bytecode prompt verified" in str(direct_out)
    assert seen_prompts[-1] == "Ping from Actor bytecode"

    # 2. Hot-patch `buildStepPrompt` on `AgentHarness` in Actor bytecode to use a completely custom prompt format
    vm.eval_chunk(
        """
        now(AgentHarness)!!
        Def buildStepPrompt(self, engramCtx, grammarCtx, policyTxt) {
            ^"CUSTOM_BYTECODE_PROMPT_FOR: " + asString(self.goal);
        }!!
        """
    )

    res = vm.eval_chunk('runGoal(Actor[#Harness], "TestDirectPrompt", 3)!!')
    assert res == "Direct bytecode prompt verified"
    assert seen_prompts[-1] == "CUSTOM_BYTECODE_PROMPT_FOR: TestDirectPrompt"


def test_metacircular_verify_outcome_rejects_and_forces_self_correction() -> None:
    """Hot-patching `AgentHarness>>verifyOutcome` rejects an invalid result and triggers self-correction on Step 2."""
    vm, harness, _ = bootstrap_agent_harness(
        enable_memory_graph=False,
        enable_engrams=False,
    )

    step_calls = []

    def provider(prompt: str, system_prompt: str, metadata: dict) -> str:
        step_calls.append(prompt)
        if len(step_calls) == 1:
            # Step 1 returns negative number -5, which our custom verifyOutcome will reject
            return "storeScratch(Actor[#Harness], #score, -5)!!"
        if len(step_calls) == 2:
            # Step 2 sees OutcomeVerificationError in prompt and corrects to +25
            assert "OutcomeVerificationError" in prompt
            return "storeScratch(Actor[#Harness], #score, 25)!!"
        return "COMPLETE: Verified positive score"

    vm.set_ai_provider(provider)

    # Hot-patch `verifyOutcome` in 1993 Actor bytecode to reject negative numbers
    vm.eval_chunk(
        """
        now(AgentHarness)!!
        Def verifyOutcome(self, outcome) {
            if outcome == -5 then
                ^false;
            endif;
            ^true;
        }!!
        """
    )

    res = vm.eval_chunk('runGoal(Actor[#Harness], "Store a positive score", 5)!!')
    assert res == "Verified positive score"
    assert len(step_calls) == 3
    assert vm.eval_chunk("readScratch(Actor[#Harness], #score)!!") == 25
    assert harness._fields["status"] == Symbol("complete")


def test_evolve_harness_shadow_validation_and_neo4j_rollback() -> None:
    """`evolve` blocks a broken self-patch in shadow mode, commits a valid self-patch, and rolls back via Neo4j."""
    vm, harness, _ = bootstrap_agent_harness(
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=TEST_NS,
    )
    vm.eval_chunk("clearEngrams(Actor[#Memory])!!")

    # Baseline `policyNote(Actor[#Harness])` is ""
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == ""

    # 1. Attempt a self-patch that fails the validation chunk -> must be rejected in shadow mode!
    bad_res = vm.eval_chunk(
        'evolve(Actor[#Harness], "Def policyNote(self) { ^\\"BROKEN\\"; }!!", "policyNote(Actor[#Harness]) == \\"EXPECTED_OTHER\\"!!")!!'
    )
    assert str(bad_res).startswith("EVOLVE_REJECTED(validation_failed)")
    # Live AgentHarness must be untouched!
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == ""

    # 2. Apply a valid self-patch that passes the validation chunk -> committed + persisted in Neo4j!
    good_res = vm.eval_chunk(
        'evolve(Actor[#Harness], "Def policyNote(self) { ^\\"EVOLVED_V2_POLICY\\"; }!!", "policyNote(Actor[#Harness]) == \\"EVOLVED_V2_POLICY\\"!!")!!'
    )
    assert str(good_res).startswith("EVOLVE_COMMITTED")
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == "EVOLVED_V2_POLICY"

    # 3. Roll back `AgentHarness` to baseline v1 from Neo4j `:ActorEngramVersion`
    rb_res = vm.eval_chunk("rollbackHarness(Actor[#Harness])!!")
    assert "ENGRAM_ROLLED_BACK: Actor[#AgentHarness]" in str(rb_res)
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == ""

    vm.eval_chunk("clearEngrams(Actor[#Memory])!!")


_ASK_ACTORII_BIN = Path.home() / "bin" / "ask-actorii"


@pytest.mark.skipif(
    not _ASK_ACTORII_BIN.is_file(),
    reason="Optional local host CLI wrapper (~/bin/ask-actorii) not installed",
)
def test_ask_actorii_peer_cli_bridge() -> None:
    """Verify `~/bin/ask-actorii` executes 1993 Actor bytecode and dumps harness source."""
    proc = subprocess.run(
        [str(_ASK_ACTORII_BIN), "--local", "--json", "--eval", "21 * 2!!"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert data["result"] == 42

    proc_h = subprocess.run(
        [str(_ASK_ACTORII_BIN), "--local", "--harness"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert "Def step(self" in proc_h.stdout
    assert "Def evolve(self" in proc_h.stdout


def test_live_llm_agent_evolves_its_own_harness() -> None:
    """Verify `Actor[#Harness]` can use `evolve(Actor[#Harness], ...)` with live `qwen3.8:latest` to hot-patch its own `AgentHarness` bytecode."""
    import urllib.request

    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2.0).read()
    except Exception:
        return

    vm, harness, _ = bootstrap_agent_harness(
        model="qwen3.8:latest",
        max_steps=4,
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=TEST_NS,
    )
    vm.eval_chunk("clearEngrams(Actor[#Memory])!!")

    goal = (
        "Evolve your own `AgentHarness` class in live bytecode by calling "
        '`evolve(Actor[#Harness], "Def policyNote(self) { ^\\"SELF_MUTATED_BY_AGENT\\"; }!!", "policyNote(Actor[#Harness]) == \\"SELF_MUTATED_BY_AGENT\\"!!")!!` '
        "and store `policyNote(Actor[#Harness])` in `storeScratch(Actor[#Harness], #answer, policyNote(Actor[#Harness]))!!`."
    )
    escaped = goal.replace("\\", "\\\\").replace('"', '\\"')
    vm.eval_chunk(f'runGoal(Actor[#Harness], "{escaped}", 4)!!')

    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == "SELF_MUTATED_BY_AGENT"
    assert vm.eval_chunk("readScratch(Actor[#Harness], #answer)!!") == "SELF_MUTATED_BY_AGENT"

    # Roll back AgentHarness to baseline v1
    vm.eval_chunk("rollbackHarness(Actor[#Harness])!!")
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == ""
    vm.eval_chunk("clearEngrams(Actor[#Memory])!!")

