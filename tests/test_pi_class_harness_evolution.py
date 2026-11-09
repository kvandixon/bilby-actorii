"""Verification suite for Milestone M16: Pi-Class Self-Mutable Harness Evolution.

Tests:
1. `Actor[#Host]` (`HostWorkspaceBridge`): `readFile`, `writeFile`, `editFile`, `execShell`,
   and Copy-on-Write VFS isolation during `dryRun` (`shadow_only=True`).
2. `Actor[#VM]` (`VMReflectionBridge`): `classes`, `ringOf`, `methodsOf`, `sourceOf`, `ivarsOf`,
   plus structured `lastErrorContext` (`#errorKind`, `#failedClass`, `#failedSelector`,
   `#bytecodeOffset`, `#localBindings`).
3. `Actor[#LLM]` (`LLMBridge`): `countTokens`, `shadow_only=True` dry-run stubbing, and
   automatic `compactContext` token-budget compaction in `AgentHarness>>step`.
4. Task-scoped `#taskContract` (`setTaskContract` / `clearTaskContract`) post-condition
   verification in Ring 2 without mutating Ring 1 `AgentHarness`.
5. `spawnSubGoal` / `forkHarness`: Copy-on-Write isolated child `AgentHarness` execution
   that returns only the final result without polluting the parent's transcript or scratchpad.
6. Hardened Neo4j `:ActorEngram` lifecycle: runtime Cypher call-stack `[:OPERATES_ON]`
   domain tracking, warm-image instance reconciliation, `initializeClass` hook, execution
   telemetry (`recordEngramUsage`), and scratch/contract class crystallization filtering.
"""

from __future__ import annotations

from pathlib import Path

from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.harness import bootstrap_agent_harness


def test_host_workspace_bridge_and_cow_shadow_vfs(tmp_path: Path) -> None:
    """Verify `Actor[#Host]` file I/O, shell execution, and Copy-on-Write VFS during `dryRun`."""
    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)
    sample_file = tmp_path / "sample.txt"

    # 1. Committed writeFile & readFile
    w_res = vm.eval_chunk(
        f'writeFile(Actor[#Host], {str(sample_file)!r}, "alpha\\nbeta\\ngamma\\n")!!'
    )
    assert "WRITE_OK:" in str(w_res)
    assert sample_file.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

    r_res = vm.eval_chunk(f"readFile(Actor[#Host], {str(sample_file)!r}, 1, 3)!!")
    assert "1: alpha" in str(r_res)
    assert "2: beta" in str(r_res)

    # 2. Speculative `dryRun` (`shadow_only=True`) stages edits in CoW VFS without touching disk
    shadow_script = (
        f'editFile(Actor[#Host], {str(sample_file)!r}, "beta", "BETA_SHADOW")!!\n'
        f"readFile(Actor[#Host], {str(sample_file)!r}, 1, 3)!!"
    )
    shadow_out = vm.eval_chunk(f"dryRun(Actor[#Harness], {shadow_script!r})!!")
    assert "(CoW Shadow VFS)" in str(shadow_out)
    assert "BETA_SHADOW" in str(shadow_out)

    # Physical file on disk must be untouched and shadow VFS cleared after rollback
    assert sample_file.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"
    assert vm.eval_chunk("vfsOverlayCount(Actor[#Host])!!") == 0

    # 3. `execShell` is blocked during `dryRun` (`shadow_only=True`) but allowed in committed mode
    blocked_out = vm.eval_chunk('dryRun(Actor[#Harness], "execShell(Actor[#Host], \\"echo hi\\", 5)!!")!!')
    assert "ActorBridgeSecurityError" in str(blocked_out)

    shell_res = vm.eval_chunk('execShell(Actor[#Host], "echo pi_harness_ok", 5)!!')
    assert isinstance(shell_res, Dictionary)
    assert shell_res.at(Symbol("exitCode")) == 0
    assert "pi_harness_ok" in str(shell_res.at(Symbol("stdout")))


def test_vm_reflection_bridge_and_structured_error_context() -> None:
    """Verify `Actor[#VM]` reflection primitives and structured `lastErrorContext` with frame locals."""
    vm, harness, _ = bootstrap_agent_harness(enable_memory_graph=False)

    assert vm.eval_chunk("ringOf(Actor[#VM], #Object)!!") == 0
    assert vm.eval_chunk("ringOf(Actor[#VM], #LLMBridge)!!") == 0
    assert vm.eval_chunk("ringOf(Actor[#VM], #AgentHarness)!!") == 1

    vm.eval_chunk(
        """
        inherit(Object, #DiagWorker, #(#factor), 1, nil)!!
        now(DiagWorker)!!
        Def computeBad(self, inputVal | scaled) {
            scaled := inputVal * 3;
            ^error(self, "diagnostic failure");
        }!!
        Actor[#Diag] := new(DiagWorker)!!
        """
    )
    assert vm.eval_chunk("ringOf(Actor[#VM], #DiagWorker)!!") == 2

    ivars = vm.eval_chunk("ivarsOf(Actor[#VM], #AgentHarness)!!")
    ivar_list = [str(x) for x in ivars]
    for expected_ivar in ("chatLog", "taskContract", "lastErrorContext", "tokenBudget"):
        assert expected_ivar in ivar_list

    src = vm.eval_chunk("sourceOf(Actor[#VM], #AgentHarness, #verifyOutcome)!!")
    assert "self.taskContract" in str(src)

    # Trigger a step failure inside `DiagWorker>>computeBad` and inspect `lastErrorContext`
    replies = iter(["computeBad(Actor[#Diag], 14)!!"])
    harness.custom_provider = lambda prompt, sys_p, opts: next(replies)
    vm.eval_chunk('setGoal(Actor[#Harness], "Trigger diagnostic error")!!')
    out = vm.eval_chunk("step(Actor[#Harness])!!")
    assert str(out).startswith("ERROR:")

    err_ctx = vm.eval_chunk("lastErrorContext(Actor[#Harness])!!")
    assert isinstance(err_ctx, Dictionary)
    assert err_ctx.at(Symbol("failedClass")) == Symbol("DiagWorker")
    assert err_ctx.at(Symbol("failedSelector")) == Symbol("computeBad")
    locals_dict = err_ctx.at(Symbol("localBindings"))
    assert isinstance(locals_dict, Dictionary)
    assert locals_dict.at(Symbol("inputVal")) == "14"
    assert locals_dict.at(Symbol("scaled")) == "42"


def test_llm_token_counting_shadow_stub_and_auto_compaction() -> None:
    """Verify `countTokens(Actor[#LLM], ...)`, shadow stubbing, and automatic `compactContext` in `step`."""
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)

    tok_count = vm.eval_chunk('countTokens(Actor[#LLM], "Hello from Actor bytecode!")!!')
    assert tok_count > 0

    # `complete` inside `dryRun` returns the deterministic stub without calling `custom_provider`
    provider_calls = []
    harness.custom_provider = lambda p, s, o: (provider_calls.append(p) or "COMPLETE: live")
    stub_res = vm.eval_chunk(
        'dryRun(Actor[#Harness], "complete(Actor[#LLM], \\"test prompt\\", Actor[#Harness])!!")!!'
    )
    assert "COMPLETE: Shadow dry-run stub" in str(stub_res)
    assert len(provider_calls) == 0

    # Seed 5 turns into `chatLog`, set a tight tokenBudget, and verify `step` auto-compacts `chatLog`
    for i in range(1, 6):
        vm.eval_chunk(f'setGoal(Actor[#Harness], "Historical question {i}")!!')
        sandbox._record_completed_turn(harness, f"Historical answer {i} " + ("x" * 120))
    assert len(harness._fields["chatLog"]) == 5

    vm.eval_chunk("setTokenBudget(Actor[#Harness], 80)!!")
    vm.eval_chunk('setGoal(Actor[#Harness], "Final question after compaction")!!')
    vm.eval_chunk("step(Actor[#Harness])!!")

    # 5 turns compacted with keepTurns=2 -> 1 summary turn + 2 kept turns + 1 newly completed turn = 4
    chat_log = list(harness._fields["chatLog"])
    assert len(chat_log) == 4
    assert chat_log[0]["user"] == "[COMPACTED PRIOR TURNS]"


def test_task_scoped_contract_verification_and_self_correction() -> None:
    """Verify `#taskContract` enforces task post-conditions in Ring 2 without mutating Ring 1 `AgentHarness`."""
    vm, harness, _ = bootstrap_agent_harness(enable_memory_graph=False)

    vm.eval_chunk(
        """
        inherit(Object, #TaskContractHundred, #(#minVal), 1, nil)!!
        now(TaskContractHundred)!!
        Def check(self, outcome | val) {
            val := readScratch(Actor[#Harness], #score);
            if val == nil then
                ^false;
            endif;
            ^(val >= 100);
        }!!
        Actor[#HundredContract] := new(TaskContractHundred)!!
        setTaskContract(Actor[#Harness], Actor[#HundredContract])!!
        """
    )

    replies = iter(
        [
            "storeScratch(Actor[#Harness], #score, 50)!!\nCOMPLETE: Tried 50",
            "storeScratch(Actor[#Harness], #score, 125)!!\nCOMPLETE: Reached 125",
        ]
    )
    harness.custom_provider = lambda p, s, o: next(replies)

    final_res = vm.eval_chunk('runGoal(Actor[#Harness], "Compute a score >= 100", 4)!!')
    assert final_res == "Reached 125"
    assert vm.eval_chunk("readScratch(Actor[#Harness], #score)!!") == 125
    assert vm.eval_chunk("stepCount(Actor[#Harness])!!") == 2


def test_spawn_subgoal_cow_fork_isolation() -> None:
    """Verify `spawnSubGoal` runs an isolated CoW child harness without polluting parent transcript or scratchpad."""
    vm, harness, _ = bootstrap_agent_harness(enable_memory_graph=False)

    vm.eval_chunk('storeScratch(Actor[#Harness], #parentKey, "parent_intact")!!')

    def _router(prompt: str, sys_p: str, opts: dict) -> str:
        if "Sub-task: compute 6 * 7" in prompt:
            if "LAST RESULT: 42" not in prompt:
                return "storeScratch(Actor[#Harness], #childSecret, 999)!!\n(6 * 7)!!"
            return "COMPLETE: 42"
        return "COMPLETE: parent done"

    harness.custom_provider = _router

    child_result = vm.eval_chunk(
        'spawnSubGoal(Actor[#Harness], "Sub-task: compute 6 * 7", "Child policy active", 4)!!'
    )
    assert child_result == "42"

    # Parent scratchpad & chatLog must be completely untouched by child's `#childSecret` or turns
    assert vm.eval_chunk("readScratch(Actor[#Harness], #parentKey)!!") == "parent_intact"
    assert vm.eval_chunk("readScratch(Actor[#Harness], #childSecret)!!") is None
    assert len(harness._fields["chatLog"]) == 0


def test_engram_runtime_cypher_tracking_warm_reconciliation_and_telemetry() -> None:
    """Verify runtime Cypher call-stack `[:OPERATES_ON]` edges, warm-image reconciliation, `initializeClass`, and scratch filtering."""
    ns = "test_m16_pi_evolution"
    vm, _, _ = bootstrap_agent_harness(
        enable_memory_graph=True,
        enable_engrams=True,
        memory_namespace=ns,
    )
    mem = vm.global_vars["Actor"]["Memory"]._target
    mem.clearEngrams()

    try:
        # 1. Scratch & TaskContract classes must be ignored by `auto_crystallize_from_chunk`
        scratch_code = """
        inherit(Object, #TmpScratchHelper, #(#x), 1, nil)!!
        now(TmpScratchHelper)!!
        Def ping(self) { ^1; }!!
        """
        vm.eval_chunk(scratch_code)
        crystallized = mem.auto_crystallize_from_chunk(scratch_code, "temp test")
        assert crystallized == []

        # 2. Save an Engram with `initializeClass` hook and a dynamic Cypher query whose label is built at runtime
        engram_code = """
        inherit(Object, #TelemetryProbeSkill, #(#bootCount #customNote), 2, nil)!!
        now(TelemetryProbeSkill)!!
        Def initializeClass(self) {
            if self.bootCount == nil then
                self.bootCount := 1;
            else
                self.bootCount := self.bootCount + 1;
            endif;
            ^self.bootCount;
        }!!
        Def probeDomain(self | q) {
            q := "MATCH (n:" + "RuntimeMetricNode" + ") RETURN count(n) AS cnt";
            ^cypherQuery(Actor[#Memory], q);
        }!!
        Actor[#TelemetryProbeSkill] := new(TelemetryProbeSkill)!!
        """
        save_msg = vm.eval_chunk(
            f'saveEngram(Actor[#Memory], #TelemetryProbeSkill, "Runtime telemetry probe", {engram_code!r})!!'
        )
        assert "ENGRAM_SAVED" in str(save_msg)
        # `initializeClass` ran on initial hydration
        assert vm.eval_chunk("Actor[#TelemetryProbeSkill].bootCount!!") == 1

        # Set warm instance state (`customNote`) and execute `probeDomain` so runtime call-stack records `:RuntimeMetricNode`
        vm.eval_chunk('Actor[#TelemetryProbeSkill].customNote := "warm_preserved"!!')
        vm.eval_chunk("probeDomain(Actor[#TelemetryProbeSkill])!!")
        vm.eval_chunk("recordEngramUsage(Actor[#Memory], #TelemetryProbeSkill, true)!!")

        # Verify Neo4j recorded the runtime `[:OPERATES_ON]` edge to `:RuntimeMetricNode` and telemetry counts
        _, rows = mem._run_cypher(
            """
            MATCH (e:ActorEngram {name: 'TelemetryProbeSkill', namespace: $ns})
            OPTIONAL MATCH (e)-[:OPERATES_ON]->(d:ActorGraphDomain {namespace: $ns})
            RETURN e.call_count AS calls, e.success_count AS succ, collect(d.label) AS labels
            """,
            {"ns": ns},
        )
        assert len(rows) == 1
        assert rows[0]["calls"] == 1
        assert rows[0]["succ"] == 1
        assert "RuntimeMetricNode" in (rows[0]["labels"] or [])

        # Re-hydrate into the warm VM and verify `customNote` is preserved and `bootCount` incremented to 2
        vm.eval_chunk("loadEngram(Actor[#Memory], #TelemetryProbeSkill)!!")
        assert vm.eval_chunk("Actor[#TelemetryProbeSkill].customNote!!") == "warm_preserved"
        assert vm.eval_chunk("Actor[#TelemetryProbeSkill].bootCount!!") == 2
    finally:
        mem.clearEngrams()


def test_search_file_and_kv_prefix_prompt_ordering(tmp_path: Path) -> None:
    """Verify `searchFile(Actor[#Host], ...)` on large files and KV-cache prefix stability across steps."""
    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)
    host_bridge = vm.global_vars["Actor"]["Host"]._target
    host_bridge.allowed_roots.append(tmp_path.resolve())

    large_file = tmp_path / "large_sympy.py"
    lines = [f"# line {i}" for i in range(1, 2001)]
    lines[1705] = "class Derivative:"
    lines[1706] = "    def free_symbols(self):"
    lines[1707] = "        return set()"
    large_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = str(
        vm.eval_chunk(
            f'searchFile(Actor[#Host], "{str(large_file)}", "class Derivative", 5)!!'
        )
    )
    assert "Match at line 1706" in res
    assert "def free_symbols(self):" in res

    # Verify static instructions precede dynamic `STEP:` in `buildStepPrompt` for KV-cache prefix reuse
    vm.eval_chunk('setGoal(Actor[#Harness], "Fix Derivative.kind", 6)!!')
    p1 = str(vm.eval_chunk('buildStepPrompt(Actor[#Harness], "ENGRAM_CTX", "GRAMMAR_CTX", "POLICY_TXT")!!'))
    assert p1.index("INSTRUCTIONS:") < p1.index("STEP:")

