"""Unit & integration tests for LocalCoderBridge (`Actor[#Coder]`, 0 MB GPU VRAM)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from actorii.core import Symbol
from actorii.guardrails.linter import ActorLinter, PROTECTED_POOL_SINGLETONS
from actorii.harness import LocalCoderBridge, bootstrap_agent_harness


def _fake_chat_response(content: str, reasoning: str = "", c_tok: int = 24):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": c_tok},
            }).encode("utf-8")

    return _Resp()


def test_local_coder_primary_and_deep_think_toggle(monkeypatch):
    seen = []

    def _fake_urlopen(req, timeout=None):
        seen.append((req.full_url, json.loads(req.data.decode("utf-8"))))
        return _fake_chat_response("<think>checking null saucer</think>Fix: add `else if` on line 74.")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    coder = LocalCoderBridge()

    r1 = coder.reviewCode("if (saucer.life <= 0) saucer = null; saucer.x++;", "null crash", False)
    assert r1.at(Symbol("ok")) is True
    assert "else if" in str(r1.at(Symbol("reply")))
    assert "<think>" not in str(r1.at(Symbol("reply")))
    assert seen[0][0].endswith(":8083/v1/chat/completions")
    assert seen[0][1]["chat_template_kwargs"]["enable_thinking"] is False

    r2 = coder.diagnoseBug("x = 1/0", "ZeroDivisionError", True)
    assert r2.at(Symbol("ok")) is True
    assert seen[1][1]["chat_template_kwargs"]["enable_thinking"] is True


def test_local_coder_falls_back_to_ollama_cpu_when_8083_offline(monkeypatch):
    calls = []

    def _fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if ":8083" in req.full_url:
            raise urllib.error.URLError("Connection refused")
        return _fake_chat_response("Fallback review OK from qwen2.5-coder-cpu:7b.")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    coder = LocalCoderBridge()
    res = coder.askCoder("Check wrap() logic", False)
    assert res.at(Symbol("ok")) is True
    assert "Fallback review OK" in str(res.at(Symbol("reply")))
    assert res.at(Symbol("model")) == "qwen2.5-coder-cpu:7b"
    assert len(calls) == 2


def test_local_coder_graceful_when_both_backends_offline(monkeypatch):
    def _fail(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    coder = LocalCoderBridge()
    res = coder.askCoder("hi", False)
    assert res.at(Symbol("ok")) is False
    assert "unreachable" in str(res.at(Symbol("error"))).lower()


def test_actor_coder_wired_protected_and_lazy_schema_injected(monkeypatch):
    monkeypatch.setenv("ACTORII_DISABLE_DEMO_TWINS", "1")
    vm, _, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
    actor = vm.global_vars["Actor"]
    assert "Coder" in actor
    assert "Coder" in PROTECTED_POOL_SINGLETONS

    desc = str(vm.eval_chunk("describe(Actor[#Coder])!!"))
    assert "reviewCode" in desc
    assert "diagnoseBug" in desc

    # Lazy Step-1 schema injection: shown on coding/debug goals, omitted on trivial math
    assert "`Actor[#Coder]`" in sandbox.poolSummary("Debug the HTML5 Asteroids JS bug and review the patch")
    assert "`Actor[#Coder]`" not in sandbox.poolSummary("What is 2+2?")

    # Linter blocks overwriting Actor[#Coder]
    lint_res = sandbox.lintChunk("Actor[#Coder] := 123!!")
    assert lint_res.startswith("ERROR:")
