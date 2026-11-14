"""Comprehensive test suite for ActorII & Bilby Conversational Polish (Phases 1-4).

Covers:
1. Dual-Channel Output Architecture (`agent_trace.jsonl`, `verbose_steps.log`, TUI single-print, Telegram Option A).
2. Markdown code-fence preservation inside `COMPLETE:` replies (`bash`/`python` blocks & `# Heading` lines).
3. Mobile-first Telegram HTML rendering, wide-table bullet-card conversion, fence-aware chunking, and HTTP 400 fallback.
4. Append-only KV-cache multi-message array prefix invariance across steps (`msgs_2[:len(msgs_1)] == msgs_1`).
5. All-step `DuplicateReplyError` anti-parrot protection & identical-error stagnation circuit breaker.
6. Unknown selector (`unhandled_selector`) diagnostic hint on native `Actor[...]` receivers.
7. Lockdown-aware `defaultInstructions()`, `ACTORII_DISABLE_DEMO_TWINS=1`, real `scoreboard(Actor[#Bench])!!`,
   `writeAndRun(Actor[#Jetson], ...)`, and `pytest` state-directory isolation.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from actorii.ai import _strip_thinking_only
from actorii.core import Symbol
from actorii.harness import LLMBridge, _split_code_and_complete, bootstrap_agent_harness
from actorii.harness.telemetry import (
    DualChannelLogger,
    convert_markdown_tables_for_mobile,
    is_test_environment,
    markdown_to_telegram_html,
    render_telegram_chunks,
    resolve_actorii_state_dir,
    split_markdown_fence_aware,
)
from actorii.telegram_gateway import BilbyTelegramGateway
from actorii.tui import ActorTUISession


def test_complete_preserves_markdown_fences_and_headings() -> None:
    """`COMPLETE:` replies containing ```bash / ```python blocks or `# Heading` lines must never be parsed as Actor code."""
    raw_reply = (
        "COMPLETE: # Jetson Camera Setup\n\n"
        "Run these commands on `jetson.local`:\n\n"
        "```bash\n"
        "gst-launch-1.0 nvarguscamerasrc ! fakesink\n"
        "```\n\n"
        "And verify in Python:\n"
        "```python\n"
        "import cv2\n"
        "print(cv2.__version__)\n"
        "```"
    )
    code_src, comp_summary = _split_code_and_complete(raw_reply)
    assert code_src == ""
    assert comp_summary is not None
    assert "```bash\ngst-launch-1.0 nvarguscamerasrc ! fakesink\n```" in comp_summary
    assert "```python\nimport cv2" in comp_summary
    assert "# Jetson Camera Setup" in comp_summary


def test_unprefixed_markdown_heading_and_negative_bullet_not_treated_as_actor_code() -> None:
    """Unprefixed prose starting with `# Summary` or `-10 dB` must be treated as conversational prose, not Actor code."""
    prose = "# Summary\n-5 dB noise floor reduction achieved.\n\n```json\n{\"ok\": true}\n```"
    code_src, comp_summary = _split_code_and_complete(prose)
    assert code_src == ""
    assert comp_summary == prose


def test_strip_thinking_only_preserves_fences_and_strips_unclosed_think() -> None:
    """`_strip_thinking_only` strips both closed and truncated `<think>` blocks while keeping Markdown code fences intact."""
    with_closed = "<think>internal chain of thought</think>\nHere is the script:\n```python\nx = 42\n```"
    cleaned = _strip_thinking_only(with_closed)
    assert "internal chain of thought" not in cleaned
    assert "```python\nx = 42\n```" in cleaned

    with_unclosed = "<think>ran out of max_tokens mid-thought without closing tag..."
    assert _strip_thinking_only(with_unclosed) == ""


def test_mobile_table_conversion_and_telegram_html_rendering() -> None:
    """Wide Markdown pipe tables convert to mobile bullet cards, and HTML rendering escapes entities safely."""
    md = (
        "## Stack Health\n\n"
        "| Service | Port | VRAM |\n"
        "| --- | --- | --- |\n"
        "| **llama-primary** | 8080 | **14.1 GB** |\n"
        "| clef-flash | 8082 | 0 MB |\n\n"
        "Use `bilby_status` & check `<gpu>` tags:\n"
        "```bash\n"
        "nvidia-smi --query-gpu=memory.used --format=csv\n"
        "```"
    )
    card_md = convert_markdown_tables_for_mobile(md)
    assert "| Service |" not in card_md
    assert "• **llama-primary**" in card_md
    assert "Port: 8080" in card_md
    assert "VRAM: **14.1 GB**" in card_md

    html_out = markdown_to_telegram_html(md)
    assert "<b>Stack Health</b>" in html_out
    assert "• <b>llama-primary</b> — Port: 8080 · VRAM: <b>14.1 GB</b>" in html_out
    assert "<b>**" not in html_out
    assert "</b><b>" not in html_out
    assert "<code>bilby_status</code>" in html_out
    assert "&lt;gpu&gt;" in html_out
    assert "<pre><code>nvidia-smi --query-gpu=memory.used --format=csv</code></pre>" in html_out


def test_fence_aware_chunking_and_telegram_400_fallback(tmp_path: Path) -> None:
    """Long fenced code blocks stay balanced across chunks, and HTTP 400 parse errors retry cleanly as plain text."""
    body_lines = "\n".join(f"echo line_{i}" for i in range(60))
    long_md = f"Header intro\n```bash\n{body_lines}\n```\nFooter note"
    chunks = split_markdown_fence_aware(long_md, max_len=350)
    assert len(chunks) >= 2
    for ch in chunks:
        # Every chunk that opens a fence must also close it
        fence_count = sum(1 for ln in ch.splitlines() if ln.lstrip().startswith("```"))
        assert fence_count % 2 == 0, f"Unbalanced fence in chunk:\n{ch}"

    calls: List[Dict[str, Any]] = []

    def flaky_html_api(token: str, method: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
        if method == "sendMessage" and payload:
            calls.append(dict(payload))
            if payload.get("parse_mode") == "HTML":
                return {"ok": False, "error_code": 400, "description": "Bad Request: can't parse entities"}
        return {"ok": True, "result": {"message_id": 1}}

    gw = BilbyTelegramGateway(
        token="123:TEST",
        allowed_users=[1001],
        default_chat_id="1001",
        state_path=tmp_path / "gw_state.json",
        api_caller=flaky_html_api,
    )
    res = gw.send_message(1001, "**Bold** reply with `code`")
    assert res[0]["ok"] is True
    assert len(calls) == 2
    assert calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in calls[1]


def test_append_only_kv_cache_prefix_invariance() -> None:
    """Multi-step turns must produce strictly append-only message prefixes so llama.cpp reuses 100% of prior step KV cache."""
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False, enable_engrams=False)
    llm_bridge: LLMBridge = vm.global_vars["Actor"]["LLM"].target  # type: ignore[assignment]

    vm.eval_chunk('setGoal(Actor[#Harness], "Inspect Valkyrie health and scratchpad")!!')
    harness._fields["stepCount"] = 1
    msgs_1 = llm_bridge._build_append_only_messages(harness, sandbox)

    # Simulate Step 1 executing an action
    sandbox.executeStepAction(harness, "storeScratch(Actor[#Harness], #k1, 100)!!", 25000)
    harness._fields["stepCount"] = 2
    msgs_2 = llm_bridge._build_append_only_messages(harness, sandbox)

    # Simulate Step 2 executing a second action
    sandbox.executeStepAction(harness, "storeScratch(Actor[#Harness], #k2, 200)!!", 25000)
    harness._fields["stepCount"] = 3
    msgs_3 = llm_bridge._build_append_only_messages(harness, sandbox)

    assert len(msgs_2) == len(msgs_1) + 2
    assert msgs_2[: len(msgs_1)] == msgs_1
    assert len(msgs_3) == len(msgs_2) + 2
    assert msgs_3[: len(msgs_2)] == msgs_2


def test_all_step_duplicate_reply_anti_parrot_and_unhandled_selector_hint() -> None:
    """Parroted prose on Step 2+ is rejected once with `DuplicateReplyError`, and unknown selectors surface a hint."""
    session = ActorTUISession(use_color=False)
    long_speech = (
        "Valkyrie is currently operating within all thermal and VRAM envelopes: "
        "GPU utilization is 12%, physical VRAM allocation is 14.1 GB out of 16.3 GB on port 8080, "
        "and zero-VRAM CPU specialists on port 8082 are healthy and standing by."
    )

    state: Dict[str, Any] = {"turn2_calls": 0, "step2_prompt": "", "step3_prompt": ""}

    def parrot_then_fix_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
        if "GOAL: Give me the initial status report" in prompt:
            return f"COMPLETE: {long_speech}"
        if "GOAL: Now check unknown method and give a new answer" in prompt:
            state["turn2_calls"] += 1
            if state["turn2_calls"] == 1:
                # Call a nonexistent selector on Actor[#Thermometer] (pure Actor object -> returns nil + unhandled_selector)
                return "nonExistentProbeSelector(Actor[#Thermometer])!!"
            if state["turn2_calls"] == 2:
                state["step2_prompt"] = prompt
                # Attempt to parrot Turn 1's long speech on Step 2!
                return f"COMPLETE: {long_speech}"
            state["step3_prompt"] = prompt
            return "COMPLETE: Fresh Turn 2 answer after inspecting Thermometer."
        return "COMPLETE: ok"

    session.harness.custom_provider = parrot_then_fix_brain
    session.handle_input("Give me the initial status report")
    session.handle_input("Now check unknown method and give a new answer")

    assert "UnknownSelectorWarning" in state["step2_prompt"]
    assert "nonExistentProbeSelector" in state["step2_prompt"]
    assert "DuplicateReplyError" in state["step3_prompt"]
    turns = session.get_chat_history(local_only=True)
    assert len(turns) == 2
    assert state["turn2_calls"] == 3
    assert turns[1]["reply"] == "Fresh Turn 2 answer after inspecting Thermometer."


def test_lockdown_instructions_demo_twin_gate_scoreboard_and_jetson_write_and_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify lockdown instructions, `ACTORII_DISABLE_DEMO_TWINS=1`, `scoreboard(Actor[#Bench])`, and `JetsonEdgeTwin.writeAndRun`."""
    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    monkeypatch.setenv("ACTORII_DISABLE_DEMO_TWINS", "1")

    session = ActorTUISession(use_color=False)
    pool = session.get_pool_objects()
    assert "Hue" not in pool
    assert "Sam" not in pool

    instructions = session.sandbox.defaultInstructions()
    assert "[HOST LOCKDOWN ACTIVE" in instructions
    assert "python3 -m pytest -q" not in instructions

    # Real benchmark scoreboard lookup
    board = str(session.vm.eval_chunk("scoreboard(Actor[#Bench])!!"))
    assert "LongMemEval" in board or "Benchmark" in board

    # Jetson writeAndRun base64 encodes multi-line scripts safely and enforces guardrails
    jetson = session.vm.global_vars["Actor"]["Jetson"].target
    blocked = jetson.writeAndRun("/tmp/bad.sh", "#!/bin/bash\nsudo apt upgrade -y\n")
    assert blocked.at("ok") is False
    assert "JETSON_GUARDRAIL_BLOCKED" in str(blocked.at("stderr"))

    captured_cmds: List[str] = []

    def fake_exec(cmd: str, timeout_sec: int = 15) -> Any:
        from actorii.harness.twins import _to_actor_dict

        captured_cmds.append(cmd)
        return _to_actor_dict({"ok": True, "exit_code": 0, "stdout": "JETSON_SCRIPT_OK", "stderr": ""})

    monkeypatch.setattr(jetson, "exec", fake_exec)
    script_body = '#!/usr/bin/env python3\nprint("Hello \'quotes\' & $vars")\n'
    ok_res = jetson.writeAndRun("/tmp/probe_cam.py", script_body, 20)
    assert ok_res.at("ok") is True
    assert ok_res.at("stdout") == "JETSON_SCRIPT_OK"
    assert len(captured_cmds) == 1
    expected_b64 = base64.b64encode(script_body.encode("utf-8")).decode("ascii")
    assert expected_b64 in captured_cmds[0]
    assert "/tmp/probe_cam.py" in captured_cmds[0]


def test_pytest_state_isolation_and_dual_channel_logger(tmp_path: Path) -> None:
    """Verify `pytest` automatically isolates state away from `~/.actorii` and `DualChannelLogger` writes JSONL + verbose logs."""
    assert is_test_environment() is True
    active_dir = resolve_actorii_state_dir()
    assert active_dir != (Path.home() / ".actorii")

    logger = DualChannelLogger(state_dir=tmp_path)
    tid = logger.new_turn_id()
    logger.log_turn_start(turn_id=tid, surface="telegram", goal="Check GPU", model="qwen3.8:latest", max_steps=25)
    logger.log_step_exec(
        turn_id=tid,
        entry={"step": 1, "status": "ok", "tier": "TIER_0", "gas_used": 42, "action": "gpuStatus(Actor[#Valkyrie])!!", "result": "14.1 GB"},
    )
    logger.log_turn_end(
        turn_id=tid,
        surface="telegram",
        status="complete",
        steps=1,
        gas_used=42,
        elapsed_s=1.23,
        reply="GPU VRAM is 14.1 GB.",
        actions=["gpuStatus(Actor[#Valkyrie])!!"],
    )

    trace_lines = (tmp_path / "logs" / "agent_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 3
    events = [json.loads(ln)["event"] for ln in trace_lines]
    assert events == ["turn_start", "step_exec", "turn_end"]

    verbose_txt = (tmp_path / "logs" / "verbose_steps.log").read_text(encoding="utf-8")
    assert "gpuStatus(Actor[#Valkyrie])!!" in verbose_txt
    assert "GPU VRAM is 14.1 GB." in verbose_txt


def test_recurring_sensor_query_no_false_duplicate_and_no_80char_truncation() -> None:
    """Updated sensor numbers after a valid tool step must NEVER trigger `DuplicateReplyError`, and rejected actions are never truncated at 80 chars."""
    session = ActorTUISession(use_color=False)
    rep_t1 = (
        "The cabin is currently at **18.6°C (65.5°F)** with **55% humidity** — right in the comfort zone. "
        "The ThermoPro TP358S sensor is reporting a dew point of 9.3°C and the battery is at 50%."
    )
    rep_t2 = (
        "The cabin is currently at **19.5°C (67.1°F)** with **46% humidity** — right in the comfort zone. "
        "The ThermoPro TP358S sensor is reporting a dew point of 7.6°C and the battery is at 50%."
    )
    state = {"t1": 0, "t2": 0, "t3": 0}

    def cabin_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
        if "GOAL: Whats the cabin temp" in prompt:
            state["t1"] += 1
            if state["t1"] == 1:
                return "summary(Actor[#Thermometer])!!"
            return f"COMPLETE: {rep_t1}"
        if "GOAL: Whats the temperature in the cabin?" in prompt:
            state["t2"] += 1
            if state["t2"] == 1:
                return "summary(Actor[#Thermometer])!!"
            return f"COMPLETE: {rep_t2}"
        if "GOAL: Unrelated question with lazy parrot" in prompt:
            state["t3"] += 1
            if state["t3"] == 1:
                return f"COMPLETE: {rep_t2}"
            return "COMPLETE: Fresh answer without truncation."
        return "COMPLETE: ok"

    session.harness.custom_provider = cabin_brain
    session.handle_input("Whats the cabin temp")
    session.handle_input("Whats the temperature in the cabin?")

    turns = session.get_chat_history(local_only=True)
    assert len(turns) == 2
    assert state["t2"] == 2
    assert turns[1]["reply"] == rep_t2
    assert not turns[1]["reply"].endswith("...")

    # Now verify that when a genuine lazy parrot occurs on Turn 3, stepLog action is NOT truncated with `[:80]...`
    session.handle_input("Unrelated question with lazy parrot")
    step_entries = list(session.harness._fields["stepLog"])
    err_entries = [e for e in step_entries if isinstance(e, dict) and e.get("status") == "error"]
    assert len(err_entries) == 1
    assert err_entries[0]["action"] == f"COMPLETE: {rep_t2}"
    assert "right in t..." not in err_entries[0]["action"]


def test_cross_turn_epoch_kv_prefix_invariance_and_static_system_prompt() -> None:
    """`sys_msg` stays 100% goal-invariant across turns and `prior_msgs` appends monotonically without sliding-window shifts."""
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False, enable_engrams=False)
    llm_bridge: LLMBridge = vm.global_vars["Actor"]["LLM"].target  # type: ignore[assignment]

    # Seed 8 completed turns in chatLog
    for i in range(1, 9):
        harness._fields["goal"] = f"Historical question {i}"
        harness._fields["stepCount"] = 1
        sandbox._record_completed_turn(harness, f"Historical answer {i}")

    vm.eval_chunk('setGoal(Actor[#Harness], "How you feeling Bilby?")!!')
    harness._fields["stepCount"] = 1
    msgs_t9 = llm_bridge._build_append_only_messages(harness, sandbox)
    sandbox._record_completed_turn(harness, "Feeling great!")

    vm.eval_chunk('setGoal(Actor[#Harness], "Whats the temperature in the cabin?")!!')
    harness._fields["stepCount"] = 1
    msgs_t10 = llm_bridge._build_append_only_messages(harness, sandbox)
    sandbox._record_completed_turn(harness, "19.5C and 46% humidity.")

    vm.eval_chunk('setGoal(Actor[#Harness], "Inspect Neo4j cypher graphSchema on Actor[#Memory]")!!')
    harness._fields["stepCount"] = 1
    msgs_t11 = llm_bridge._build_append_only_messages(harness, sandbox)

    # 1. System prompt is byte-for-byte identical even when Neo4j/Cypher keywords appear!
    assert msgs_t9[0] == msgs_t10[0] == msgs_t11[0]
    # 2. Prior-turn history prefix grows strictly by +2 messages per turn without shifting message #1!
    prior_9 = msgs_t9[1:-1]
    prior_10 = msgs_t10[1:-1]
    prior_11 = msgs_t11[1:-1]
    assert len(prior_10) == len(prior_9) + 2
    assert prior_10[: len(prior_9)] == prior_9
    assert len(prior_11) == len(prior_10) + 2
    assert prior_11[: len(prior_10)] == prior_10
    # 3. Task-specific Neo4j grammar constructs appear in Turn 11's user message, NOT mutating sys_msg
    assert "ActorMemoryGraph" in msgs_t11[-1]["content"] or "Neo4j" in msgs_t11[-1]["content"]


def test_thermopro_passive_repr_and_stale_auto_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """`repr(twin)` never scans BLE, while stale `lastSyncTime` (>180s) triggers `syncLive` even with non-empty history."""
    from actorii.classes.thermopro import ThermoProTwin

    twin = ThermoProTwin(mac_address="AA:BB:CC:DD:EE:FF", device_name="TP358S")
    twin._apply_reading(temp_c=18.5, humidity=50, battery=50, sync_time="2026-01-01T00:00:00")
    twin._append_record_dict({"tempC": 18.5, "humidity": 50, "timestamp": "2026-01-01T00:00:00"})

    sync_calls: List[float] = []

    def fake_sync(timeout_sec: float = 4.5) -> ThermoProTwin:
        sync_calls.append(timeout_sec)
        twin._apply_reading(temp_c=20.2, humidity=44, battery=50)
        return twin

    monkeypatch.setattr(twin, "syncLive", fake_sync)
    monkeypatch.setenv("ACTORII_FORCE_AUTO_SYNC", "1")

    # 1. Passive repr MUST NOT call syncLive
    r_txt = repr(twin)
    assert "Temp=18.5C" in r_txt
    assert len(sync_calls) == 0

    # 2. Active `summary()` refreshes stale reading even though `historyCount() == 1`
    s_txt = twin.summary()
    assert len(sync_calls) == 1
    assert "Temp=20.2C" in s_txt
    assert "Synced=" in s_txt


def test_telegram_live_step_status_in_place_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live progress posts a status message on Step 1 and edits it in-place (`editMessageText`) with the final HTML reply."""
    monkeypatch.setenv("ACTORII_MEMORY_NAMESPACE", "pytest_polish")
    session = ActorTUISession(use_color=False)
    step_state = {"calls": 0}

    def two_step_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
        step_state["calls"] += 1
        if step_state["calls"] == 1:
            return "storeScratch(Actor[#Bilby], #cabinTemp, 67)!!"
        return "COMPLETE: Cabin is **67°F** and comfortable."

    session.harness.custom_provider = two_step_brain
    api_events: List[tuple[str, dict]] = []

    def fake_tg_api(token: str, method: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
        p = dict(payload or {})
        api_events.append((method, p))
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 777}}
        if method in ("editMessageText", "deleteMessage", "sendChatAction"):
            return {"ok": True, "result": True}
        return {"ok": True, "result": []}

    gw = BilbyTelegramGateway(
        token="123:LIVE",
        allowed_users=[888],
        default_chat_id="888",
        state_path=tmp_path / "live_gw.json",
        session=session,
        api_caller=fake_tg_api,
    )
    out = gw._run_update_sync(1, 888, 12, 50, "Check cabin temp", keep_typing=False, live_progress=True)
    assert "67°F" in out

    send_msgs = [p for m, p in api_events if m == "sendMessage"]
    edit_msgs = [p for m, p in api_events if m == "editMessageText"]
    assert len(send_msgs) == 1
    assert "Step 1:" in send_msgs[0]["text"]
    assert len(edit_msgs) >= 1
    assert edit_msgs[-1]["message_id"] == 777
    assert "<b>67°F</b>" in edit_msgs[-1]["text"]


def test_telegram_bot_twin_multipart_media_and_lockdown_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`TelegramBotTwin` uploads photos/documents from safe staging paths while blocking symlinks, credentials, and host paths."""
    from actorii.harness.twins import TelegramBotTwin, _encode_multipart_formdata

    stage_dir = tmp_path / "actorii_media"
    stage_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ACTORII_MEDIA_STAGE_DIR", str(stage_dir))

    jpg_bytes = b"\xff\xd8\xff\xe0" + (b"\x42" * 39400) + b"\xff\xd9"
    jpg_file = stage_dir / "back_window.jpg"
    jpg_file.write_bytes(jpg_bytes)

    body, ctype = _encode_multipart_formdata(
        {"chat_id": 8492834492, "message_thread_id": 15996, "caption": "Back window"},
        "photo",
        "back_window.jpg",
        jpg_bytes,
    )
    assert ctype.startswith("multipart/form-data; boundary=----ActorIITelegramBoundary")
    assert b'name="photo"; filename="back_window.jpg"' in body
    assert b"Content-Type: image/jpeg" in body
    assert jpg_bytes in body

    uploads: List[dict] = []

    def fake_multipart(
        token: str,
        method: str,
        fields: dict | None,
        file_field: str,
        file_name: str,
        file_bytes: bytes,
        content_type: str | None = None,
        timeout: float = 40.0,
    ) -> dict:
        uploads.append(
            {
                "token": token,
                "method": method,
                "fields": dict(fields or {}),
                "file_field": file_field,
                "file_name": file_name,
                "size": len(file_bytes),
            }
        )
        return {"ok": True, "result": {"message_id": 901}}

    twin = TelegramBotTwin(
        token="999:SECRET_TOKEN",
        default_chat_id="8492834492",
        multipart_caller=fake_multipart,
    )
    twin._set_active_route(8492834492, 15996)

    # 1. Bare filename or `/tmp/back_window.jpg` resolves into `ACTORII_MEDIA_STAGE_DIR` and routes to thread 15996
    res_photo = twin.sendPhoto("/tmp/back_window.jpg", "Jetson OV5693 snapshot")
    assert "Sent Telegram photo #901" in res_photo
    assert len(uploads) == 1
    assert uploads[0]["method"] == "sendPhoto"
    assert uploads[0]["fields"]["chat_id"] == 8492834492
    assert uploads[0]["fields"]["message_thread_id"] == 15996
    assert uploads[0]["fields"]["caption"] == "Jetson OV5693 snapshot"
    assert uploads[0]["size"] == len(jpg_bytes)

    # 2. `sendDocument` works for arbitrary staged files
    doc_file = stage_dir / "telemetry_report.csv"
    doc_file.write_text("ts,temp_c\n1,39.5\n", encoding="utf-8")
    res_doc = twin.sendDocument(str(doc_file), "Jetson CSV")
    assert "Sent Telegram document #901" in res_doc
    assert uploads[-1]["method"] == "sendDocument"
    assert uploads[-1]["file_field"] == "document"

    # 3. Missing file gives actionable `pullFile(Actor[#Jetson], ...)` hint
    res_missing = twin.sendPhoto("/tmp/not_pulled_yet.jpg")
    assert "pullFile(Actor[#Jetson]" in res_missing

    # 4. Security guards block host files outside staging, `.env`/`.key` credentials, and symlinks
    res_etc = twin.sendDocument("/etc/hosts")
    assert "blocked by host lockdown" in res_etc

    env_stage = stage_dir / ".env"
    env_stage.write_text("SECRET=1\n", encoding="utf-8")
    res_env = twin.sendDocument(str(env_stage))
    assert "sensitive credential/state filename" in res_env

    sym_stage = stage_dir / "escape.jpg"
    sym_stage.symlink_to("/etc/hosts")
    res_sym = twin.sendDocument(str(sym_stage))
    assert "symlinks are not permitted" in res_sym


def test_jetson_pull_file_binary_integrity_and_ring2_engram_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pullFile(Actor[#Jetson], ...)` transfers >39 KB binary payloads without `[:8000]` clipping and composes cleanly in Actor 4.1."""
    import subprocess as sp
    from actorii.harness.twins import JetsonEdgeTwin

    stage_dir = tmp_path / "actorii_media"
    monkeypatch.setenv("ACTORII_MEDIA_STAGE_DIR", str(stage_dir))
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ACTORII_MEMORY_NAMESPACE", "pytest_polish")

    jetson = JetsonEdgeTwin(state_dir=tmp_path / "jetson_state")
    monkeypatch.setattr(
        jetson,
        "_resolve_live_host",
        lambda auto_discover=True: ("192.168.68.55", {"host": "192.168.68.55", "user": "kilog", "port": 22}, "SSH-2.0-OpenSSH_7.6p1 Ubuntu"),
    )

    binary_jpeg = b"\xff\xd8\xff\xe0" + bytes((i % 256 for i in range(42000))) + b"\xff\xd9"

    def fake_ssh_run(argv, capture_output=True, text=False, timeout=30.0):
        assert text is False
        return sp.CompletedProcess(argv, 0, stdout=binary_jpeg, stderr=b"")

    monkeypatch.setattr(sp, "run", fake_ssh_run)

    # 1. Sensitive remote paths on Jetson are blocked before SSH runs
    blocked = jetson.pullFile("/home/kilog/.ssh/id_ed25519")
    assert blocked.at("ok") is False
    assert "JETSON_GUARDRAIL_BLOCKED" in blocked.at("stderr")

    # 2. End-to-end Ring-2 Actor 4.1 class composing `Actor[#Jetson]` + `Actor[#Telegram]` inside `BilbyTelegramGateway`
    session = ActorTUISession(use_color=False)
    session.vm.expose_object(jetson, global_key="#Jetson")

    uploaded_media: List[dict] = []

    def fake_multipart(token, method, fields, file_field, file_name, file_bytes, content_type=None, timeout=40.0):
        uploaded_media.append(
            {
                "method": method,
                "fields": dict(fields or {}),
                "file_name": file_name,
                "bytes": file_bytes,
            }
        )
        return {"ok": True, "result": {"message_id": 502}}

    tg_twin = session.vm.global_vars["Actor"]["Telegram"].target
    tg_twin._token = "123:TEST"
    tg_twin._multipart_api = fake_multipart

    api_events: List[tuple[str, dict]] = []

    def fake_tg_api(token: str, method: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
        p = dict(payload or {})
        api_events.append((method, p))
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 500 + len(api_events)}}
        return {"ok": True, "result": True}

    brain_state = {"step": 0}

    def media_builder_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
        brain_state["step"] += 1
        if brain_state["step"] == 1:
            return (
                "inherit(DomainEntity, #JetsonTelegramBridge, #(#defaultCaption), 1, nil)!!\n"
                "now(JetsonTelegramBridge)!!\n"
                "Def sendRemotePhoto(self, remotePath, caption | pulled, localPath) {\n"
                "  pulled := pullFile(Actor[#Jetson], remotePath, \"back_window.jpg\", 30);\n"
                "  if (at(pulled, #ok)) then\n"
                "    localPath := at(pulled, #localPath);\n"
                "    ^sendPhoto(Actor[#Telegram], localPath, caption);\n"
                "  endif;\n"
                "  ^at(pulled, #stderr);\n"
                "}!!\n"
                "Actor[#JetsonMedia] := initEntity(new(JetsonTelegramBridge), \"jm1\", \"Jetson Media\")!!\n"
                "sendRemotePhoto(Actor[#JetsonMedia], \"/tmp/back_window.jpg\", \"Back window from TX1\")!!"
            )
        return "COMPLETE: Sent `/tmp/back_window.jpg` (42 KB) from the Jetson TX1 to Telegram!"

    session.harness.custom_provider = media_builder_brain
    gw = BilbyTelegramGateway(
        token="123:TEST",
        allowed_users=[8492834492],
        default_chat_id="8492834492",
        state_path=tmp_path / "gw_media.json",
        session=session,
        api_caller=fake_tg_api,
    )

    reply = gw._run_update_sync(
        99, 8492834492, 15996, 200, "Send the Jetson photo on Telegram", keep_typing=False, live_progress=True
    )
    assert "42 KB" in reply
    assert len(uploaded_media) == 1
    assert uploaded_media[0]["method"] == "sendPhoto"
    assert uploaded_media[0]["fields"]["chat_id"] == 8492834492
    assert uploaded_media[0]["fields"]["message_thread_id"] == 15996
    assert uploaded_media[0]["bytes"] == binary_jpeg

    # Because media was sent mid-turn, temporary status bubble is deleted and final reply is posted below the photo!
    methods_called = [m for m, _ in api_events]
    assert "deleteMessage" in methods_called
    assert methods_called[-1] == "sendMessage"


def test_grammar_diagnose_inherit_singleton_hint() -> None:
    """`diagnoseError` guides Bilby to compose Ring-1 bridges inside `DomainEntity`/`Object` instead of `inherit(Actor[#Telegram], ...)`."""
    from actorii.harness.grammar_graph import ActorGrammarGraph

    gg = ActorGrammarGraph(auto_seed=False)
    diag = gg.diagnoseError("ActorGuardrailError: ParseError: Expected class name in inherit(), got ('LBRACKET', '[')")
    assert "NEVER pass `Actor[#Singleton]`" in diag
    assert "pullFile(Actor[#Jetson]" in diag


def test_valkyrie_ask_skippy_timeout_and_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ValkyrieSystemTwin.askSkippy` invokes `ask-skippy` with a >=110s default timeout and surfaces full Cloud Skippy answers."""
    import shutil
    import subprocess as sp
    from actorii.harness.twins import ValkyrieSystemTwin

    fake_script = tmp_path / "ask-skippy"
    fake_script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    fake_script.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake_script) if name == "ask-skippy" else None)

    recorded: dict = {}

    def fake_run(argv, timeout=None, capture_output=True, text=True, **kwargs):
        recorded["argv"] = argv
        recorded["timeout"] = timeout
        return sp.CompletedProcess(argv, 0, stdout="Visual analysis: wooden deck and grill.\n", stderr="")

    monkeypatch.setattr(sp, "run", fake_run)
    twin = ValkyrieSystemTwin()
    ans = twin.askSkippy("Describe /tmp/actorii_media/back_window_2.jpg")
    assert ans == "Visual analysis: wooden deck and grill."
    assert recorded["timeout"] == 110.0
    assert recorded["argv"] == [str(fake_script), "Describe /tmp/actorii_media/back_window_2.jpg"]


def test_jetson_write_file_and_cuda_write_and_run_and_string_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`JetsonEdgeTwin.writeFile` stages source without running bash, and `writeAndRun` compiles `.cu` via `nvcc -arch=sm_53 -ccbin /usr/bin/gcc-7`."""
    from actorii.harness.grammar_graph import ActorGrammarGraph
    from actorii.harness.twins import JetsonEdgeTwin

    jetson = JetsonEdgeTwin(state_dir=tmp_path / "jetson_cu")
    captured_cmds: List[tuple[str, float]] = []

    def fake_ssh_exec_raw(remote_cmd: str, timeout_sec: float = 15.0) -> dict:
        captured_cmds.append((remote_cmd, timeout_sec))
        return {"ok": True, "host": "192.168.68.60", "exitCode": 0, "stdout": "GPU_MS=14.2\n", "stderr": ""}

    monkeypatch.setattr(jetson, "_ssh_exec_raw", fake_ssh_exec_raw)

    cu_src = '#include <stdio.h>\nint main() { printf("P6\\n1920 1080\\n255\\n"); return 0; }\n'
    wf = jetson.writeFile("/tmp/julia.cu", cu_src, 20)
    assert wf.at("ok") is True
    assert wf.at("remotePath") == "/tmp/julia.cu"
    assert wf.at("bytes") == len(cu_src.encode("utf-8"))
    assert "bash " not in captured_cmds[-1][0]

    war = jetson.writeAndRun("/tmp/julia.cu", cu_src, 30)
    assert war.at("ok") is True
    assert "/usr/local/cuda/bin/nvcc -O3 -arch=sm_53" in captured_cmds[-1][0]
    assert "/usr/bin/gcc-7" in captured_cmds[-1][0]
    assert "/tmp/julia.bin" in captured_cmds[-1][0]
    assert " -lm && " in captured_cmds[-1][0]
    assert captured_cmds[-1][1] >= 45.0

    # Guardrail still blocks writing outside `/tmp/`
    bad = jetson.writeFile("/etc/evil.cu", cu_src)
    assert bad.at("ok") is False
    assert "JETSON_GUARDRAIL_BLOCKED" in bad.at("stderr")

    gg = ActorGrammarGraph(auto_seed=False)
    hint = gg.diagnoseError("ActorGuardrailError: ParseError: Unterminated string literal starting at 54")
    assert "NEVER generate raw base64" in hint
    assert "writeFile(Actor[#Jetson]" in hint

    # Ensure multi-KB `writeAndRun` steps stay intact inside KV-cached assistant messages (no `... <N chars total> ...` corruption)
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False, enable_engrams=False)
    llm_bridge: LLMBridge = vm.global_vars["Actor"]["LLM"].target  # type: ignore[assignment]
    vm.eval_chunk('setGoal(Actor[#Harness], "Render CUDA Julia set")!!')
    harness._fields["stepCount"] = 1
    _ = llm_bridge._build_append_only_messages(harness, sandbox)
    long_cu_act = 'writeAndRun(Actor[#Jetson], "/tmp/julia_hd.cu", """\n' + ("float x = sinf(1.0f);\n" * 100) + '""", 60)!!'
    harness._fields["stepLog"].add(
        {"step": 1, "action": long_cu_act, "status": "ok", "result": "ok", "gas_used": 8}
    )
    harness._fields["stepCount"] = 2
    kv_msgs = llm_bridge._build_append_only_messages(harness, sandbox)
    asst_contents = [m["content"] for m in kv_msgs if m["role"] == "assistant"]
    assert any("float x = sinf(1.0f);" in c and "chars total>" not in c for c in asst_contents)

    # And `executeStepAction` rejects literal `<N chars total>` placeholders before touching disk
    err_res = sandbox.executeStepAction(
        harness,
        'writeAndRun(Actor[#Jetson], "/tmp/julia_hd.cu", """#include <cud ... <2302 chars total> ... return 0;""", 60)!!',
    )
    assert "TruncatedPlaceholderError" in str(err_res)


def test_ask_skippy_cheapest_model_cascade_and_active_route_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ask-skippy` routes to cheapest capable Flash tier with cross-pool failover, and `askSkippy` forwards active Telegram route."""
    import importlib.machinery
    import importlib.util
    import subprocess as sp

    from actorii.harness.twins import TelegramBotTwin, ValkyrieSystemTwin

    loader = importlib.machinery.SourceFileLoader("ask_skippy_mod", "/home/skippy/bin/ask-skippy")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)

    # 1. Routine Q&A + multimodal vision -> gemini-3.8-flash-low first, with cross-pool failover
    t_low, c_low = mod.select_model_cascade(
        "Inspect /tmp/actorii_media/julia_hd.jpg and describe colors",
        exec_mode=False,
        explicit_tier=None,
        explicit_model=None,
    )
    assert t_low == "low"
    assert c_low[0] == "gemini-3.8-flash-low"
    assert "claude-sonnet-5-5-low" in c_low

    # 2. Deep root-cause / deadlock query -> gemini-3.8-flash-medium first
    t_med, c_med = mod.select_model_cascade(
        "Diagnose deadlock root cause in ExecutionFiber stack",
        exec_mode=False,
        explicit_tier=None,
        explicit_model=None,
    )
    assert t_med == "medium"
    assert c_med[0] == "gemini-3.8-flash-medium"

    # 3. Tool-execution escalation -> gemini-3.8-flash-high first
    t_high, c_high = mod.select_model_cascade(
        "Apply hotfix",
        exec_mode=True,
        explicit_tier=None,
        explicit_model=None,
    )
    assert t_high == "high"
    assert c_high[0] == "gemini-3.8-flash-high"

    # 4. Active Telegram `(chat_id, thread_id)` propagates into `ValkyrieSystemTwin.askSkippy` subprocess env
    monkeypatch.delenv("ASK_SKIPPY_CHAT_ID", raising=False)
    monkeypatch.delenv("ASK_SKIPPY_THREAD_ID", raising=False)
    tg = TelegramBotTwin(token="dummy:tok", default_chat_id="111", api_caller=lambda *a, **k: {"ok": True})
    tg._set_active_route(8492834492, 15996)

    captured_env: dict = {}

    def fake_run(cmd, timeout=None, capture_output=False, text=False, env=None, **kw):
        if env is not None:
            captured_env.update(env)
        return sp.CompletedProcess(cmd, 0, stdout="Skippy reply", stderr="")

    monkeypatch.setattr("actorii.harness.twins.subprocess.run", fake_run)
    sys_twin = ValkyrieSystemTwin()
    ans = sys_twin.askSkippy("Quick question")
    assert ans == "Skippy reply"
    assert captured_env.get("ASK_SKIPPY_CHAT_ID") == "8492834492"
    assert captured_env.get("ASK_SKIPPY_THREAD_ID") == "15996"
    tg._set_active_route(None, None)
