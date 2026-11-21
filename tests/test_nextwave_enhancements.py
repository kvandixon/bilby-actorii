"""Unit tests for Next-Wave Harness Enhancements:

1. Step-1 Prompt Trimming, `[Agent Instance Fields]` Leak Fix & Lazy-Load Twin Method Schemas
2. Workspace Verification-Before-Complete & Zero-Diff Guard (`WorkspaceVerificationGuard`)
3. Smarter `DuplicateStepWarning` Escalation (nil result, non-nil result, identical error)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from actorii.core import Symbol
from actorii.harness import (
    DEFAULT_HARNESS_SYSTEM_PROMPT,
    _is_workspace_coding_goal,
    _trim_engram_context,
    bootstrap_agent_harness,
)


class TestStep1PromptTrimmingAndLazyTwins:
    def test_system_prompt_is_lean_and_lazy_loads_twins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACTORII_DISABLE_PERSONA_OVERLAY", "1")
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        llm = vm.global_vars["Actor"]["LLM"].target

        # Trimmed static system prompt should be well under 500 tokens
        assert llm.countTokens(DEFAULT_HARNESS_SYSTEM_PROMPT) < 500
        assert "describe(Actor[#Twin])!!" in DEFAULT_HARNESS_SYSTEM_PROMPT
        # Unrelated twin full method tables should NOT be statically baked into system prompt
        assert "exposeLanPort" not in DEFAULT_HARNESS_SYSTEM_PROMPT
        assert "detonateSandbox" not in DEFAULT_HARNESS_SYSTEM_PROMPT
        assert "writeAndRun" not in DEFAULT_HARNESS_SYSTEM_PROMPT

        # Casual Q&A goal: no irrelevant twin signatures injected, 1-line catalog present
        vm.eval_chunk('setGoal(Actor[#Harness], "What is 2 + 2?")!!')
        qa_summary = sandbox.poolSummary("What is 2 + 2?")
        assert "Relevant twin methods:" not in qa_summary
        assert "Other live twins (call describe(Actor[#Name]) for methods):" in qa_summary
        assert "#Jetson" in qa_summary
        assert "#Valkyrie" in qa_summary

        # Jetson goal: injects Jetson signatures, keeps unrelated twins in catalog
        jetson_goal = "Check Jetson camera stream status via SSH"
        vm.eval_chunk(f'setGoal(Actor[#Harness], "{jetson_goal}")!!')
        j_summary = sandbox.poolSummary(jetson_goal)
        assert "Relevant twin methods:" in j_summary
        assert "writeAndRun" in j_summary
        assert "cameraStatus" in j_summary
        assert "Other live twins (call describe(Actor[#Name]) for methods):" in j_summary
        assert "detonateSandbox" not in j_summary

    def test_append_only_messages_no_field_leak_and_under_4k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACTORII_DISABLE_PERSONA_OVERLAY", "1")
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        llm = vm.global_vars["Actor"]["LLM"].target

        # Simulate a large hydrated engram/episode context cached on harness_obj
        huge_engram = "\n".join(
            f"- Episode {i}: detailed archived memory line with context payload {i * 17}"
            for i in range(250)
        )
        assert len(huge_engram) > 12000

        vm.eval_chunk('setGoal(Actor[#Harness], "Fix the bug in /workspace/app.py and run pytest")!!')
        harness._fields["cachedEngramCtx"] = huge_engram
        harness._fields["cachedGrammarCtx"] = "/* grammar snippet */"

        msgs_coding = llm._build_append_only_messages(harness, sandbox)
        sys_content = msgs_coding[0]["content"]
        usr_content = msgs_coding[-1]["content"]

        # Critical KV-cache fix: cachedEngramCtx/cachedGrammarCtx must NEVER leak into sys_msg!
        assert "Episode 0:" not in sys_content
        assert "cachedEngramCtx" not in sys_content
        assert "cachedGrammarCtx" not in sys_content

        # Step-1 engram context is cleanly capped
        assert "[truncated" in usr_content
        assert len(_trim_engram_context(huge_engram, max_chars=2800)) < 3100

        # Total Step-1 tokens stay well below 4,000 tokens
        total_coding_tokens = llm.countTokens(msgs_coding)
        assert total_coding_tokens < 4000, f"Expected <4000 tokens, got {total_coding_tokens}"

        # sys_msg stays 100% invariant across different goals (preserves KV cache f_keep=1.0)
        vm.eval_chunk('setGoal(Actor[#Harness], "Check GPU VRAM on Valkyrie")!!')
        harness._fields["cachedEngramCtx"] = "Different engram payload"
        msgs_valk = llm._build_append_only_messages(harness, sandbox)
        assert msgs_valk[0]["content"] == sys_content

    def test_on_demand_describe_reflection_works_for_any_twin(self) -> None:
        vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

        # Direct 1-arg `describe(Actor[#Twin])!!` works on both Python bridges and .act twins
        j_desc = str(vm.eval_chunk("describe(Actor[#Jetson])!!"))
        assert "Actor[#Jetson]" in j_desc
        assert "writeAndRun" in j_desc
        assert "Live Methods:" in j_desc

        v_desc = str(vm.eval_chunk("describe(Actor[#Valkyrie])!!"))
        assert "Actor[#Valkyrie]" in v_desc
        assert "gpuStatus" in v_desc

        w_desc = str(vm.eval_chunk("describe(Actor[#Web])!!"))
        assert "Actor[#Web]" in w_desc
        assert "search" in w_desc

        # Also works via Actor[#Sandbox] `describeBridge` and `describe`
        n_desc = str(vm.eval_chunk("describeBridge(Actor[#Sandbox], #Network)!!"))
        assert "exposeLanPort" in n_desc
        p_desc = str(vm.eval_chunk("describe(Actor[#Sandbox], Actor[#Pantry])!!"))
        assert "recipeCount" in p_desc


class TestWorkspaceVerificationGuard:
    def test_goal_classifier_precision(self) -> None:
        # True positives: clear workspace coding/file-modification tasks
        assert _is_workspace_coding_goal("Fix the bug in /workspace/app.py and run pytest") is True
        assert _is_workspace_coding_goal("Edit src/utils.py to handle empty lists") is True
        assert _is_workspace_coding_goal("Create a new script report.py in the workspace") is True
        assert _is_workspace_coding_goal("Implement retry logic in the repo's client.py") is True
        assert _is_workspace_coding_goal("Patch config.json file to set timeout=30") is True

        # True negatives: casual Q&A, read-only inspection, pure Actor VM calculations
        assert _is_workspace_coding_goal("Hi Bilby, what is 2 + 2?") is False
        assert _is_workspace_coding_goal("Explain how to fix a KeyError in app.py") is False
        assert _is_workspace_coding_goal("Check git status in workspace") is False
        assert _is_workspace_coding_goal("Read /workspace/README.md and summarize it") is False
        assert _is_workspace_coding_goal("Compute factorial of 10 in Actor VM") is False
        assert _is_workspace_coding_goal("Define an Actor class Counter with an increment method") is False
        assert _is_workspace_coding_goal("What is the current GPU temperature on Valkyrie?") is False

    def test_intercepts_once_then_allows_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "0")
        monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Fix the bug in /workspace/app.py and run pytest")!!')

        # Step 1: Bilby emits COMPLETE: with zero edits and zero verification commands -> intercepted ONCE
        r1 = str(sandbox.executeStepAction(harness, "COMPLETE: I looked at it and fixed the bug."))
        assert "WorkspaceVerificationGuard:" in r1
        assert harness._fields["status"] != Symbol("complete")
        assert harness._fields["failures"] == 0

        # Step 2: If Bilby emits COMPLETE: again in the same turn -> NEVER loops, completes cleanly
        harness._fields["stepCount"] = 2
        r2 = str(sandbox.executeStepAction(harness, "COMPLETE: Verified no file change is needed."))
        assert "WorkspaceVerificationGuard:" not in r2
        assert harness._fields["status"] == Symbol("complete")

    def test_does_not_trigger_after_edit_or_exec_shell(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "0")
        monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Fix the bug in /workspace/app.py and run pytest")!!')

        # Simulate a successful verification step in stepLog
        harness._fields["stepLog"].add(
            {
                "step": 1,
                "action": 'execShell(Actor[#Host], "python3 -m pytest -q", 15)!!',
                "status": "ok",
                "result": {"exitCode": 0},
            }
        )
        harness._fields["stepCount"] = 2
        r = str(sandbox.executeStepAction(harness, "COMPLETE: All tests pass."))
        assert "WorkspaceVerificationGuard:" not in r
        assert harness._fields["status"] == Symbol("complete")

    def test_never_triggers_on_casual_qa_or_readonly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "0")
        monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)

        for goal in (
            "What is 2 + 2?",
            "Read /workspace/README.md and summarize it",
            "Compute 15 * 23 in Actor VM",
        ):
            vm.eval_chunk(f'setGoal(Actor[#Harness], "{goal}")!!')
            res = str(sandbox.executeStepAction(harness, "COMPLETE: Done."))
            assert "WorkspaceVerificationGuard:" not in res
            assert harness._fields["status"] == Symbol("complete")


class TestSmarterDuplicateStepEscalation:
    def test_escalates_for_nil_result_on_second_repeat(self) -> None:
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Check empty lookup")!!')

        # Step 1: succeeds returning nil/None
        r1 = sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #missingKey)!!")
        assert r1 is None

        # Step 2 (repeat #1): standard DuplicateStepWarning
        harness._fields["stepCount"] = 2
        r2 = str(sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #missingKey)!!"))
        assert "DuplicateStepWarning:" in r2
        assert "[ESCALATED" not in r2

        # Step 3 (repeat #2): escalated nil-aware pivot guidance
        harness._fields["stepCount"] = 3
        r3 = str(sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #missingKey)!!"))
        assert "DuplicateStepWarning [ESCALATED #2]" in r3
        assert "nil" in r3.lower()
        assert "side-effect only" in r3.lower() or "empty lookup" in r3.lower()
        assert "COMPLETE:" in r3

    def test_escalates_for_non_nil_result_on_second_repeat(self) -> None:
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Compute value")!!')

        r1 = sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #ans, 42)!!')
        assert r1 == 42

        harness._fields["stepCount"] = 2
        r2 = str(sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #ans, 42)!!'))
        assert "DuplicateStepWarning:" in r2

        harness._fields["stepCount"] = 3
        r3 = str(sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #ans, 42)!!'))
        assert "DuplicateStepWarning [ESCALATED #2]" in r3
        assert "LAST RESULT" in r3
        assert "42" in r3
        assert "COMPLETE:" in r3

    def test_escalates_for_identical_error_on_repeat(self) -> None:
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Trigger error")!!')

        # Step 1: fails with ZeroDivisionError
        r1 = str(sandbox.executeStepAction(harness, "(10 / 0)!!"))
        assert "ZeroDivisionError" in r1

        # Step 2 (repeat #1 of identical error): DuplicateStepWarning referencing original error & diagnoseError
        harness._fields["stepCount"] = 2
        r2 = str(sandbox.executeStepAction(harness, "(10 / 0)!!"))
        assert "DuplicateStepWarning" in r2
        assert "ZeroDivisionError" in r2
        assert "diagnoseError" in r2

        # Step 3 (repeat #2 of identical error): escalated with concrete selector/diagnoseError guidance
        harness._fields["stepCount"] = 3
        r3 = str(sandbox.executeStepAction(harness, "(10 / 0)!!"))
        assert "DuplicateStepWarning [ESCALATED #2 — IDENTICAL ERROR]" in r3
        assert "methodsOf" in r3 or "describe" in r3
        assert "diagnoseError" in r3

    def test_intervening_state_mutation_allows_rerun_and_resets_consecutive_dups(self) -> None:
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Iterate on script")!!')

        # Step 1: read scratchpad -> nil
        harness._fields["stepCount"] = 1
        assert sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #k)!!") is None

        # Step 2: mutate state via storeScratch -> breaks duplicate lookback
        harness._fields["stepCount"] = 2
        assert sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #k, "v2")!!') == "v2"

        # Step 3: re-running exact same readScratch after mutation MUST NOT trigger DuplicateStepWarning
        harness._fields["stepCount"] = 3
        r3 = sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #k)!!")
        assert r3 == "v2"

        # Step 4: non-consecutive single dup warning does not hard-block when followed by progress
        harness._fields["stepCount"] = 4
        r4 = str(sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #k)!!"))
        assert "DuplicateStepWarning:" in r4
        assert str(harness._fields.get("status")) != "blocked"

        harness._fields["stepCount"] = 5
        assert sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #k, "v3")!!') == "v3"
        harness._fields["stepCount"] = 6
        assert sandbox.executeStepAction(harness, "readScratch(Actor[#Harness], #k)!!") == "v3"
        assert str(harness._fields.get("status")) != "blocked"

    def test_nonzero_shell_exit_dup_warning_and_sampling_breakout(self, monkeypatch) -> None:
        from unittest.mock import MagicMock
        import json as _json
        from actorii.ai import ValkyrieLLMClient
        from actorii.harness import Dictionary, Symbol

        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Run failing shell")!!')

        # Seed a Step 1 ok entry whose Dictionary result has exitCode=1 + stderr traceback
        fail_dict = Dictionary()
        fail_dict.atPut(Symbol("exitCode"), 1)
        fail_dict.atPut(Symbol("stdout"), "")
        fail_dict.atPut(Symbol("stderr"), "Traceback: KeyError: 'foo'")
        harness._fields["stepLog"].add({
            "step": 1,
            "action": 'execShell(Actor[#Host], "python3 bad.py", 10)!!',
            "status": "ok",
            "result": fail_dict,
        })
        harness._fields["stepCount"] = 2
        r2 = str(sandbox.executeStepAction(harness, 'execShell(Actor[#Host], "python3 bad.py", 10)!!'))
        assert "FAILED with exitCode=1" in r2
        assert "already succeeded" not in r2
        assert "KeyError" in r2

        # Verify ValkyrieLLMClient bumps temperature & presence_penalty on DuplicateStepWarning retry
        captured = {}

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return _json.dumps({"choices": [{"message": {"content": "COMPLETE: ok"}}], "usage": {}}).encode()

        def _fake_urlopen(req, timeout=None):
            captured["payload"] = _json.loads(req.data.decode("utf-8"))
            return _FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        client = ValkyrieLLMClient(endpoint="http://127.0.0.1:8080/v1/chat/completions", model="qwen3.8:latest")
        client.complete_messages(
            [
                {"role": "system", "content": "You are Bilby."},
                {"role": "user", "content": "Error: DuplicateStepWarning: You just re-emitted..."},
            ],
            temperature=0.1,
        )
        assert captured["payload"]["temperature"] >= 0.55
        assert captured["payload"]["presence_penalty"] == 0.5

    def test_readfile_spec_cap_line_hint_and_midpoint_checkpoint(self, tmp_path) -> None:
        from actorii.harness.host_bridge import HostWorkspaceBridge

        spec = tmp_path / "problem_spec.md"
        # Create a ~14 KB markdown spec (~200 lines of 75 chars)
        lines = [f"Line {i:03d}: " + ("x" * 65) for i in range(1, 201)]
        spec.write_text("\n".join(lines), encoding="utf-8")

        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        hb = HostWorkspaceBridge(vm, allowed_roots=[tmp_path], max_output_chars=4000)
        out = hb.readFile(str(spec), 1, 200)
        # Should allow > 4000 chars for .md spec files and report exact line resume hint if truncated
        assert len(out) > 8000
        assert "truncated at line " in out
        assert "read next from line " in out

        # Midpoint implementation checkpoint fires at maxSteps//2 when 0 files written on coding goal
        vm.eval_chunk('setGoal(Actor[#Harness], "Write /workspace/output/results.json using Python")!!')
        harness._fields["maxSteps"] = 32
        ok_probes = [{"step": i, "action": 'execShell(Actor[#Host], "ls", 5)!!', "status": "ok"} for i in range(1, 17)]
        nudge = sandbox._midpoint_implementation_nudge(harness, ok_probes, 16)
        assert "MIDPOINT IMPLEMENTATION CHECKPOINT" in nudge
        assert "16/32" in nudge
        # Does not fire if a file was already written
        with_write = ok_probes + [{"step": 15, "action": 'writeFile(Actor[#Host], "/workspace/s.py", "x")!!', "status": "ok"}]
        assert sandbox._midpoint_implementation_nudge(harness, with_write, 16) == ""


class TestUnaryOperatorCompilation:
    def test_unary_minus_and_not_in_actor_vm(self) -> None:
        from actorii.vm import ActorVM

        vm = ActorVM()
        vm.eval_chunk(
            """
            inherit(Object, #UnaryMath, #(), 0, nil)!!
            now(UnaryMath)!!
            Def neg(self, x) { ^-x; }!!
            Def absVal(self, d) {
                if (d < 0) then
                    d := -d;
                endif;
                ^d;
            }!!
            Def negExpr(self, a, b) { ^-(a + b); }!!
            Def invert(self, v) { ^not v; }!!
            Def notGt(self, a, b) { ^not (a > b); }!!
            Actor[#UM] := new(UnaryMath)!!
            """
        )
        assert vm.eval_chunk("neg(Actor[#UM], 5)!!") == -5
        assert vm.eval_chunk("neg(Actor[#UM], -2)!!") == 2
        assert vm.eval_chunk("neg(Actor[#UM], 0)!!") == 0
        assert vm.eval_chunk("neg(Actor[#UM], 2.5)!!") == -2.5
        assert vm.eval_chunk("absVal(Actor[#UM], -7)!!") == 7
        assert vm.eval_chunk("absVal(Actor[#UM], 4)!!") == 4
        assert vm.eval_chunk("absVal(Actor[#UM], 0)!!") == 0
        assert vm.eval_chunk("negExpr(Actor[#UM], 3, 4)!!") == -7

        assert vm.eval_chunk("invert(Actor[#UM], true)!!") is False
        assert vm.eval_chunk("invert(Actor[#UM], false)!!") is True
        assert vm.eval_chunk("invert(Actor[#UM], nil)!!") is True
        assert vm.eval_chunk("invert(Actor[#UM], 0)!!") is True
        assert vm.eval_chunk("invert(Actor[#UM], 1)!!") is False
        assert vm.eval_chunk("notGt(Actor[#UM], 5, 3)!!") is False
        assert vm.eval_chunk("notGt(Actor[#UM], 2, 3)!!") is True
