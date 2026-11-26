"""Focused hardening tests for Phase 2 core-harness upgrades and bug fixes."""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path
from typing import Any, Dict, List

import pytest

from actorii.ai import (
    ValkyrieLLMClient,
    _extract_def_chunk_balanced,
    extract_and_normalize_actor_method,
)
from actorii.bridge import actor_bridge, actor_query
from actorii.core import Symbol
from actorii.harness import (
    LLMBridge,
    _is_harness_method_overridden,
    _transpile_xml_host_tool_call,
    bootstrap_agent_harness,
)
from actorii.harness.telemetry import (
    DualChannelLogger,
    markdown_to_telegram_html,
    render_telegram_chunks,
    split_markdown_fence_aware,
)
from actorii.telegram_gateway import BilbyTelegramGateway
from actorii.tui import ActorTUISession


@actor_bridge(name="CountingMemory")
class _CountingMemory:
    enable_engrams = True
    namespace = "test"

    def __init__(self) -> None:
        self.hydrate_calls = 0

    @actor_query
    def hydrateContext(self, goal: str) -> str:
        self.hydrate_calls += 1
        return f"[ENGRAM_CTX #{self.hydrate_calls} for {goal}]"

    def auto_crystallize_from_chunk(self, *_args: Any, **_kwargs: Any) -> List[str]:
        return []


def test_per_turn_engram_caching_avoids_redundant_calls_on_steps_2_plus() -> None:
    vm, harness, sandbox = bootstrap_agent_harness(
        enable_memory_graph=False,
        max_steps=5,
        max_failures=3,
    )
    mem = _CountingMemory()
    sandbox.memory_graph = mem
    vm.lock_system_singletons = False
    vm.expose_object(mem, global_key="#Memory")
    vm.lock_system_singletons = True

    replies = iter([
        "storeScratch(Actor[#Harness], #a, 1)!!",
        "storeScratch(Actor[#Harness], #b, 2)!!",
        "COMPLETE: done",
    ])
    harness.custom_provider = lambda _p, _s, _m: next(replies)

    res = vm.eval_chunk('runGoal(Actor[#Harness], "compute two scratch items", 5)!!')
    assert res == "done"
    # Step 1 hydrates once; Steps 2 and 3 reuse cachedEngramCtx
    assert mem.hydrate_calls == 1

    # Starting a new goal resets cachedEngramCtx and hydrates once for the new goal
    replies2 = iter(["COMPLETE: second"])
    harness.custom_provider = lambda _p, _s, _m: next(replies2)
    res2 = vm.eval_chunk('runGoal(Actor[#Harness], "second goal", 3)!!')
    assert res2 == "second"
    assert mem.hydrate_calls == 2


def test_append_only_messages_honors_evolved_ring1_hooks_and_enforces_token_budget() -> None:
    vm, harness, sandbox = bootstrap_agent_harness(
        enable_memory_graph=False,
        max_steps=10,
    )
    # Evolve instructionBlock and policyNote
    evolve_res = sandbox.evolveHarness(
        harness,
        'Def instructionBlock(self) { ^"CUSTOM_RING1_INSTRUCTIONS"; }!!',
        'instructionBlock(Actor[#Harness]) == "CUSTOM_RING1_INSTRUCTIONS"!!',
    )
    assert evolve_res.startswith("EVOLVE_COMMITTED"), evolve_res
    assert _is_harness_method_overridden(harness, "instructionBlock")

    vm.eval_chunk('setGoal(Actor[#Harness], "budget test")!!')
    harness._fields["cachedEngramCtx"] = "[CACHED_ENGRAM_MARKER]"
    harness._fields["tokenBudget"] = 600

    # Populate 10 bulky step entries to trigger token-budget sliding
    step_log = harness._fields["stepLog"]
    for i in range(1, 11):
        step_log.add(
            {
                "step": i,
                "action": f'storeScratch(Actor[#Harness], #k{i}, "{"x" * 600}")!!',
                "status": "ok",
                "gas_used": 10,
                "result": "y" * 600,
            }
        )

    llm_bridge = LLMBridge(vm)
    msgs = llm_bridge._build_append_only_messages(harness, sandbox)
    assert "CUSTOM_RING1_INSTRUCTIONS" in msgs[0]["content"]
    assert "[CACHED_ENGRAM_MARKER]" in msgs[1]["content"]
    # pair_offset must have advanced to respect tokenBudget while keeping at least the last 2 step pairs
    cache = getattr(harness, "_kv_turn_cache")
    assert cache["pair_offset"] > 0
    assert len(msgs) < 2 + 20

    # clearChatHistory resets KV caches cleanly
    sandbox.clearChatHistory(harness)
    assert getattr(harness, "_kv_turn_cache", None) is None
    assert getattr(harness, "_kv_history_anchor", None) is None


def test_evolve_harness_liveness_guard_and_ast_validation_chunk() -> None:
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)

    # 1. Reject transformAction returning nil (self-brick prevention)
    bad = sandbox.evolveHarness(
        harness,
        "Def transformAction(self, rawAction) { ^nil; }!!",
        "",
    )
    assert "EVOLVE_REJECTED(liveness_failed)" in bad

    # 2. Reject transformAction returning non-executable garbage for VM actions
    bad2 = sandbox.evolveHarness(
        harness,
        'Def transformAction(self, rawAction) { ^"garbage prose without bangs"; }!!',
        "",
    )
    assert "EVOLVE_REJECTED(liveness_failed)" in bad2

    # 3. Reject validation_chunk containing only a Def with no executable assertion
    no_exec = sandbox.evolveHarness(
        harness,
        'Def policyNote(self) { ^"ok"; }!!',
        'Def helperCheck(self) { ^true; }!!',
    )
    assert "EVOLVE_REJECTED(validation_not_executed)" in no_exec

    # 4. Accept multi-chunk validation_chunk with braces (previously broken by greedy \{.*\})
    good = sandbox.evolveHarness(
        harness,
        'Def policyNote(self) { ^"MULTI_BLOCK_OK"; }!!',
        'if policyNote(Actor[#Harness]) == "MULTI_BLOCK_OK" then true; else false; endif;!!',
    )
    assert good.startswith("EVOLVE_COMMITTED"), good


def test_fork_harness_preserves_evolved_hooks_and_cleans_up_threadpool() -> None:
    vm, harness, sandbox = bootstrap_agent_harness(
        enable_memory_graph=False,
        max_failures=4,
    )
    ev = sandbox.evolveHarness(
        harness,
        'Def policyNote(self) { ^"PARENT_EVOLVED_HOOK"; }!!',
        'policyNote(Actor[#Harness]) == "PARENT_EVOLVED_HOOK"!!',
    )
    assert ev.startswith("EVOLVE_COMMITTED"), ev

    seen_prompts: List[str] = []

    def _prov(prompt: str, _sys: str, _meta: Dict[str, Any]) -> str:
        seen_prompts.append(prompt)
        return "COMPLETE: sub-done"

    harness.custom_provider = _prov
    out = sandbox.forkHarness(harness, "child task", "", 3, 25000)
    assert out == "sub-done"
    assert any("PARENT_EVOLVED_HOOK" in p for p in seen_prompts)


def test_skippy_preflight_blocks_comment_and_string_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTORII_ENFORCE_SKIPPY_PREFLIGHT", "1")
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)

    # Comment mentioning askSkippy must NOT bypass preflight on detached thread spawn!
    reason1 = sandbox._check_skippy_preflight_required(
        harness,
        '/* askSkippy */ spawnChunk(Actor[#Threads], "1 + 1!!", 5000)!!',
    )
    assert reason1 is not None and "SKIPPY_PREFLIGHT_REQUIRED" in reason1

    # String literal mentioning askSkippy must NOT bypass preflight on detached thread spawn!
    reason2 = sandbox._check_skippy_preflight_required(
        harness,
        'spawnChunk(Actor[#Threads], "/* askSkippy */ 1 + 1!!", 5000)!!',
    )
    assert reason2 is not None and "SKIPPY_PREFLIGHT_REQUIRED" in reason2

    # Genuine prior askSkippy call in stepLog DOES satisfy preflight
    harness._fields["stepLog"].add(
        {
            "step": 1,
            "status": "ok",
            "action": 'askSkippy(Actor[#Valkyrie], "PREFLIGHT MONITOR: safe detached worker")!!',
            "result": "APPROVED",
        }
    )
    reason3 = sandbox._check_skippy_preflight_required(
        harness,
        'spawnChunk(Actor[#Threads], "1 + 1!!", 5000)!!',
    )
    assert reason3 is None


def test_xml_transpiler_handles_non_integer_params() -> None:
    out1 = _transpile_xml_host_tool_call(
        "<function=execShell><parameter=command>echo hi</parameter><parameter=timeout>20s</parameter></function>"
    )
    assert out1 == 'execShell(Actor[#Host], "echo hi", 20)!!'

    out2 = _transpile_xml_host_tool_call(
        "<function=readFile><parameter=path>/tmp/a.txt</parameter><parameter=start_line>abc</parameter><parameter=end_line>50lines</parameter></function>"
    )
    assert out2 == 'readFile(Actor[#Host], "/tmp/a.txt", 1, 50)!!'


def test_ai_balanced_def_extraction_and_restricted_fallback() -> None:
    nested_src = (
        "Here is the method:\n"
        "Def compute(self, x) {\n"
        '    if x > 0 then ^"{ok}"; else ^"{zero}"; endif;\n'
        "}!!\n"
        "Hope that helps!"
    )
    extracted = _extract_def_chunk_balanced(nested_src)
    assert extracted is not None
    assert extracted.startswith("Def compute(self, x) {")
    assert extracted.endswith("}!!")
    assert '"{zero}"' in extracted

    norm = extract_and_normalize_actor_method(nested_src, "compute", ["x"])
    assert norm.endswith("}!!")

    # Fallback eligibility: HTTP 400 and TimeoutError must NOT fall back; ConnectionRefused / 503 SHOULD
    http_400 = urllib.error.HTTPError("http://127.0.0.1:8080/v1/chat/completions", 400, "Bad Request", {}, None)
    http_503 = urllib.error.HTTPError("http://127.0.0.1:8080/v1/chat/completions", 503, "Unavailable", {}, None)
    assert ValkyrieLLMClient._is_fallback_eligible_error(http_400) is False
    assert ValkyrieLLMClient._is_fallback_eligible_error(TimeoutError("timed out")) is False
    assert ValkyrieLLMClient._is_fallback_eligible_error(http_503) is True
    assert ValkyrieLLMClient._is_fallback_eligible_error(ConnectionRefusedError("refused")) is True


def test_telemetry_fence_long_line_html_cap_and_markdown_edge_cases(tmp_path: Path) -> None:
    # 1. Long single line inside code fence stays properly fenced and ordered
    long_code = "header\n```python\n" + ("a" * 500) + "\n```\nfooter"
    chunks = split_markdown_fence_aware(long_code, max_len=120)
    assert chunks[0].startswith("header")
    assert chunks[-1].endswith("footer")
    for ch in chunks:
        assert ch.count("```") % 2 == 0, f"Unbalanced fence in chunk: {ch!r}"

    # 2. Heavy HTML entity expansion stays <= 4096 chars per Telegram HTML chunk
    entity_heavy = "```xml\n" + ("<tag attr=\"&\">\n" * 260) + "```"
    rendered = render_telegram_chunks(entity_heavy, max_md_chunk=3400, max_html_len=4096)
    assert len(rendered) >= 2
    for html_part, plain_part in rendered:
        assert len(html_part) <= 4096
        assert len(plain_part) <= 4096

    # 3. Markdown link with balanced parentheses & (__init__) identifier preservation
    md = "See [Actor](https://en.wikipedia.org/wiki/Actor_(programming_language)) and (__init__) in /tmp/_a_/_b_."
    html_out = markdown_to_telegram_html(md)
    assert 'href="https://en.wikipedia.org/wiki/Actor_(programming_language)"' in html_out
    assert "(__init__)" in html_out
    assert "<i>_init_</i>" not in html_out

    # 4. DualChannelLogger cross-process flock write
    logger = DualChannelLogger(state_dir=tmp_path)
    logger.write_event("test_evt", {"k": 1}, verbose_block="hello")
    assert logger.agent_trace_path.is_file()
    assert logger.verbose_log_path.is_file()


def test_telegram_gateway_halted_ux_atomic_state_and_inbox(tmp_path: Path) -> None:
    session = ActorTUISession(max_steps=1, max_failures=2, use_color=False)
    # Force step budget exhaustion on step 1 without COMPLETE:
    session.harness.custom_provider = lambda _p, _s, _m: "storeScratch(Actor[#Harness], #x, 1)!!"

    sent: List[Dict[str, Any]] = []

    def _fake_api(_tok: str, method: str, payload: Any, _timeout: float) -> Dict[str, Any]:
        sent.append({"method": method, "payload": payload})
        return {"ok": True, "result": {"message_id": len(sent)}}

    gw = BilbyTelegramGateway(
        token="dummy",
        allowed_users=[123],
        default_chat_id="123",
        state_path=tmp_path / "gw_state.json",
        session=session,
        api_caller=_fake_api,
    )
    gw.inbox_path = tmp_path / "supervisor_inbox.jsonl"

    reply = gw.handle_text_command_or_goal("do something that hits max steps")
    assert "[TURN HALTED INCOMPLETE" not in reply
    assert "paused after" in reply.lower() or "stopped" in reply.lower()

    # Atomic state file persisted with no leftover .tmp files
    assert gw.state_path.is_file()
    saved = json.loads(gw.state_path.read_text(encoding="utf-8"))
    assert "offset" in saved
    assert not list(tmp_path.glob("*.tmp"))

    # Busy status snapshot works lock-free without VM eval
    snap = gw._format_busy_status_snapshot()
    assert "Bilby Active Status" in snap

    # Atomic supervisor inbox processes directives cleanly
    gw.inbox_path.write_text('{"text": "/controls", "notify_telegram": true}\n', encoding="utf-8")
    processed = gw._process_supervisor_inbox()
    assert processed == 1
    assert not gw.inbox_path.exists()


def test_tui_multiline_prompt_reset_vs_clear_evolve_split_and_transcript_cap() -> None:
    session = ActorTUISession(max_steps=3, max_failures=3, use_color=False)
    session.harness.custom_provider = lambda _p, _s, _m: "COMPLETE: multiline ok"

    # Multi-line prompt with quotes and backslashes must not cause ParseError
    res = session._run_harness_goal('Line 1 with "quotes"\nLine 2 with \\backslash\\ and !! bangs')
    assert res == "multiline ok"
    assert len(session.get_chat_history(local_only=True)) == 1

    # /reset resets breaker WITHOUT clearing chat history
    session.harness._fields["failures"] = 3
    session.harness._fields["status"] = Symbol("blocked")
    session.handle_input("/reset")
    assert session.harness._fields["failures"] == 0
    assert len(session.get_chat_history(local_only=True)) == 1

    # /clear clears chat history
    session.handle_input("/clear")
    assert len(session.get_chat_history(local_only=True)) == 0

    # /evolve splits `Def ...!! | <validation!!>` cleanly
    before = len(session.transcript)
    session.handle_input('/evolve Def policyNote(self) { ^"TUI_EVOLVED"; }!! | policyNote(Actor[#Harness]) == "TUI_EVOLVED"!!')
    out = "\n".join(session.transcript[before:])
    assert "EVOLVE_COMMITTED" in out

    # Transcript stays bounded under heavy emission
    session._suppress_stdout = True
    for i in range(1500):
        session._emit(f"line {i}")
    assert len(session.transcript) <= 1000


def test_disable_demo_twins_skips_thermometer_ble_scan_on_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setenv("ACTORII_DISABLE_DEMO_TWINS", "1")
    monkeypatch.delenv("ACTORII_THERMO_MAC", raising=False)
    t0 = time.perf_counter()
    vm, _harness, _sandbox = bootstrap_agent_harness(enable_memory_graph=False)
    elapsed = time.perf_counter() - t0
    actor_pool = vm.global_vars.get("Actor", {})
    assert "Thermometer" not in actor_pool
    assert "ThermoUI" not in actor_pool
    assert elapsed < 3.0, f"VM boot took {elapsed:.2f}s (expected <3s without BLE scan)"



def test_incomplete_thought_prose_detector_catches_dangling_colon_and_narration():
    from actorii.harness import _is_incomplete_thought_prose

    # Exact Bilby Step-16 Asteroids dead response must be caught
    bilby_dead = (
        "`exposeLanPort` correctly refuses to bridge valkyrie→itself. The phone is already on "
        "the same LAN (`192.168.68.0/16` is in the proxy's allowed CIDRs), so I just need a "
        "local HTTP server bound to `0.0.0.0`. Checking what serving primitives exist on the "
        "available twins before trying anything:"
    )
    assert _is_incomplete_thought_prose(bilby_dead) is True
    assert _is_incomplete_thought_prose("Checking the file contents now.") is True
    assert _is_incomplete_thought_prose("Next I'll inspect the logs:") is True

    # Normal conversational answers, COMPLETE: replies, and Actor code must NOT trigger
    assert _is_incomplete_thought_prose("Absolutely! I'm all ears. What's the fun idea?") is False
    assert _is_incomplete_thought_prose("COMPLETE: Checking done — game is ready.") is False
    assert _is_incomplete_thought_prose("describe(Actor[#Proxy])!!") is False
