"""Unit tests for Bilby `:8080` dual-engine switching, `reasoning_budget_tokens`, `COMPLETE:` truncation stitching, and `/new` / `/nes` slash-command guards."""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from actorii import ai
from actorii.ai import ValkyrieLLMClient
from actorii.harness.twins import ValkyrieSystemTwin
from actorii.telegram_gateway import BilbyTelegramGateway


def test_engine_mode_file_overrides_stale_boot_env_and_respects_manual_override(tmp_path: Path, monkeypatch) -> None:
    mode_file = tmp_path / "engine_mode.json"
    monkeypatch.setenv("ACTORII_ENGINE_MODE_PATH", str(mode_file))
    monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)
    monkeypatch.delenv("ACTORII_IN_DOCKER", raising=False)
    # Simulate daemon booted earlier with stale env "auto" before bilby-engine wrote "medium"
    monkeypatch.setenv("ACTORII_THINKING_MODE", "auto")
    monkeypatch.setenv("ACTORII_ENABLE_THINKING", "1")
    ai._LAST_ENGINE_MODE_MTIME = 0.0
    ai._MANUAL_THINK_SET_AT = 0.0

    mode_file.write_text(json.dumps({"engine": "strata", "thinking_mode": "medium"}), encoding="utf-8")
    assert ai.get_thinking_mode() == "medium"

    # Manual `/think high` override stays in effect until engine_mode.json gets a newer mtime
    assert ai.set_thinking_mode("high") == "high"
    assert ai.get_thinking_mode() == "high"

    # Simulate external CLI `bilby-engine llama` writing a newer timestamp
    new_mtime = os.path.getmtime(mode_file) + 5.0
    mode_file.write_text(json.dumps({"engine": "llama", "thinking_mode": "auto"}), encoding="utf-8")
    os.utime(mode_file, (new_mtime, new_mtime))
    assert ai.get_thinking_mode() == "auto"


def test_thinking_request_uses_reasoning_budget_and_stitches_truncated_complete(monkeypatch) -> None:
    monkeypatch.setenv("ACTORII_BENCH_STRICT", "1")
    monkeypatch.setenv("ACTORII_THINKING_MODE", "medium")
    monkeypatch.setenv("ACTORII_ENABLE_THINKING", "1")

    captured_payloads = []
    responses = [
        # Pass 1: hit finish_reason="length" mid-sentence inside a COMPLETE: reply
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "reasoning_content": "Planning response.",
                        "content": "COMPLETE: It stopped seeing ActorII as a list and started seeing it as",
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 768, "total_tokens": 868},
        },
        # Pass 2: seamless continuation
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "a living graph of digital twins.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 140, "completion_tokens": 12, "total_tokens": 152},
        },
    ]

    class _FakeResp:
        def __init__(self, obj):
            self._raw = json.dumps(obj).encode("utf-8")

        def read(self):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResp(responses.pop(0))

    client = ValkyrieLLMClient(model="qwen3.8:latest", endpoint="http://127.0.0.1:8080/v1/chat/completions")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = client.complete_messages(
            [
                {"role": "system", "content": "You are Bilby."},
                {"role": "user", "content": "step: 1 of 12\nShare your reflection."},
            ],
            max_tokens=4096,
        )

    assert len(captured_payloads) == 2
    assert captured_payloads[0]["reasoning_budget_tokens"] == 768
    assert captured_payloads[0]["max_tokens"] >= 4096
    assert out == "COMPLETE: It stopped seeing ActorII as a list and started seeing it as a living graph of digital twins."


def test_telegram_new_and_unknown_slash_typo_guard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ACTORII_BENCH_STRICT", "1")
    monkeypatch.setenv("ACTORII_THINKING_MODE", "medium")
    state_file = tmp_path / "telegram_gateway_state.json"
    gw = BilbyTelegramGateway(token="dummy", state_path=state_file, api_caller=lambda *a, **k: {"ok": True, "result": []})

    # Seed a stale turn and verify `/new` clears and persists empty history
    gw.session.harness._fields["chatLog"].add({"turn": 1, "user": "hi", "reply": "hello", "actions": []})
    res_new = gw.handle_text_command_or_goal("/new")
    assert "New session started" in res_new
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["chat_history"] == []

    # Verify `/nes` typo does NOT invoke LLM and suggests `/new`
    res_typo = gw.handle_text_command_or_goal("/nes")
    assert "Unknown command `/nes`" in res_typo
    assert "`/new`" in res_typo


def test_switch_engine_sandbox_and_whitelist_guards(monkeypatch) -> None:
    twin = ValkyrieSystemTwin()

    monkeypatch.setenv("ACTORII_BENCH_STRICT", "1")
    res_blocked = twin.switchEngine("strata")
    assert res_blocked.startswith("BLOCKED:")

    monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)
    monkeypatch.delenv("ACTORII_IN_DOCKER", raising=False)
    with patch.object(twin, "isSandboxed", return_value=False):
        bad = twin.switchEngine("rm -rf /")
        assert bad.startswith("ERROR: Unsupported engine target")

        fake_payload = {
            "active_engine": "strata",
            "active_model": "qwen3.8-flash-next-coder-iq1_m",
            "thinking_mode": "medium",
            "switch_elapsed_s": 18.4,
            "services": {"clef-flash-slot-8082.service": "inactive"},
            "vram": "15537 MiB, 388 MiB, 16303 MiB",
            "ram": "26.4 GiB used / 4.8 GiB avail",
        }
        fake_proc = subprocess.CompletedProcess(
            args=["bilby-engine", "strata", "--json"],
            returncode=0,
            stdout=json.dumps(fake_payload),
            stderr="",
        )
        with patch("os.path.isfile", return_value=True), patch("subprocess.run", return_value=fake_proc):
            ok_txt = twin.switchEngine("strata")
            assert "Active Engine: STRATA in 18.4s" in ok_txt
            assert os.environ.get("ACTORII_THINKING_MODE") == "medium"


def test_methods_of_exposes_string_primitives_and_bridge_singletons_and_evolve_blocks_def_only_validation() -> None:
    from actorii.vm import ActorVM
    from actorii.harness import bootstrap_agent_harness

    vm, harness, sandbox = bootstrap_agent_harness(
        enable_memory_graph=False,
        enable_engrams=False,
    )
    str_methods = list(vm.eval_chunk("methodsOf(Actor[#VM], #String)!!"))
    assert "startswith" in str_methods
    assert "endswith" in str_methods
    assert "count" in str_methods
    assert "strip" in str_methods

    gram_methods = list(vm.eval_chunk("methodsOf(Actor[#VM], #Grammar)!!"))
    assert "lookupSyntax" in gram_methods
    assert "diagnoseError" in gram_methods

    src_info = str(vm.eval_chunk("sourceOf(Actor[#VM], #String, #startswith)!!"))
    assert "Builtin String primitive" in src_info

    # Reject `Def selfTest(self) { ^true; }!!` passed as `validationCode` without executing an assertion
    rej = sandbox.evolveHarness(
        harness,
        'Def policyNote(self) { ^"NOOP"; }!!',
        "Def selfTest(self) { ^true; }!!",
    )
    assert rej.startswith("EVOLVE_REJECTED(validation_not_executed):")

    # Also reject a validation expression returning `nil` (None)
    rej_nil = sandbox.evolveHarness(
        harness,
        'Def policyNote(self) { ^"NOOP"; }!!',
        'nonExistentSelector("abc")!!',
    )
    assert rej_nil.startswith("EVOLVE_REJECTED(validation_failed):")

    # Ring-0 liveness invariant: reject `transformAction` that bricks normal actions / clean COMPLETE:
    rej_brick = sandbox.evolveHarness(
        harness,
        "Def transformAction(self, rawAction) { ^nil; }!!",
        'transformAction(Actor[#Harness], "COMPLETE: cut,") == nil!!',
    )
    assert rej_brick.startswith("EVOLVE_REJECTED(liveness_failed):"), rej_brick

    # Valid selective `transformAction` guard still commits cleanly
    ok_guard = sandbox.evolveHarness(
        harness,
        'Def transformAction(self, rawAction) { if (endswith(rawAction, ",")) then ^nil; endif; ^rawAction; }!!',
        'transformAction(Actor[#Harness], "COMPLETE: cut,") == nil!!',
    )
    assert ok_guard.startswith("EVOLVE_COMMITTED:"), ok_guard
    assert vm.eval_chunk('transformAction(Actor[#Harness], "COMPLETE: ok.")!!') == "COMPLETE: ok."
    assert vm.eval_chunk('transformAction(Actor[#Harness], "COMPLETE: cut,")!!') is None

