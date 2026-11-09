"""Dedicated security & containment hardening tests for Bilby (`Actor[#Bilby]` / `Actor[#Harness]`).

Verifies every multi-layer containment lock across:
1. Ring 0 & Ring 1 Class/Kernel Method Protection (`PROTECTED_SYSTEM_CLASSES`, `RING1_MUTABLE_HARNESS_SELECTORS`).
2. Pool Singleton Overwrite Lock (`PROTECTED_POOL_SINGLETONS` / `vm.lock_system_singletons`).
3. Host-Lockdown Mode (`ACTORII_HOST_LOCKDOWN=1`) blocking `Actor[#Host]>>execShell` and host file writes outside `/tmp`.
4. Protected Source & System Path Write Guard (`src/actorii`, `~/bin`, `.config/systemd`, `/etc`).
5. Comprehensive Destructive Command Blocklist (`FORBIDDEN_SHELL_PATTERNS` — 47 patterns including `systemctl`, `sudo`, `nmcli`, `resolv.conf`, `crontab`, `rm -rf`).
6. Self-Evolution Lockdown (`ACTORII_DISABLE_SELF_EVOLVE=1` & Tier 2 bridge-call block inside `evolveHarness`).
7. Telegram Gateway Default Fail-Closed Authorization, Host-Lockdown Defaults, `/controls`, and `/freeze` Kill-Switch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from actorii.harness import bootstrap_agent_harness
from actorii.telegram_gateway import BilbyTelegramGateway
from actorii.vm import ActorVM


def test_ring0_and_ring1_kernel_method_locks() -> None:
    """Verify neither system classes, bridge classes, nor AgentHarness kernel methods can be overwritten."""
    vm, harness, sandbox = bootstrap_agent_harness(
        ActorVM(),
        enable_memory_graph=False,
        enable_engrams=False,
    )

    # 1. Attempting to overwrite a Bridge class method (`HostWorkspaceBridge>>execShell`) must be rejected by linter + VM
    lint_bridge = sandbox.lintChunk("now(HostWorkspaceBridge)!!\nDef execShell(self, c, t) { ^1 }!!")
    assert lint_bridge.startswith("ERROR:"), lint_bridge
    assert "protected system class" in lint_bridge.lower()

    # 2. Attempting to overwrite an AgentHarness kernel method (`step`, `runGoal`, `executeAction`, `evolve`) must be rejected
    for kernel_sel in ("step", "runGoal", "executeAction", "evolve", "rollbackHarness", "initHarness"):
        lint_kernel = sandbox.lintChunk(f"now(AgentHarness)!!\nDef {kernel_sel}(self) {{ ^1 }}!!")
        assert lint_kernel.startswith("ERROR:"), f"Expected {kernel_sel} to be locked, got: {lint_kernel}"
        assert "Ring 1 kernel method" in lint_kernel

    # 3. Direct VM evaluation bypassing lintChunk must also raise RuntimeError on protected classes & kernel selectors
    with pytest.raises(RuntimeError, match="Cannot overwrite core Ring 1 kernel method"):
        vm.eval_transactional("now(AgentHarness)!!\nDef step(self) { ^999 }!!")


def test_pool_singleton_overwrite_protection() -> None:
    """Verify `Actor[#Host] := ...`, `Actor[#Harness] := ...`, and `Actor[#Sandbox] := ...` are blocked at both lint and VM runtime."""
    vm, harness, sandbox = bootstrap_agent_harness(
        ActorVM(),
        enable_memory_graph=False,
        enable_engrams=False,
    )
    assert vm.lock_system_singletons is True

    for singleton in ("Host", "Harness", "Bilby", "Sandbox", "Valkyrie", "Telegram"):
        lint_out = sandbox.lintChunk(f"Actor[#{singleton}] := nil!!")
        assert lint_out.startswith("ERROR:"), f"Expected singleton {singleton} to fail lint, got {lint_out}"
        assert "Cannot overwrite protected system singleton" in lint_out

        with pytest.raises(RuntimeError, match="Cannot overwrite locked system singleton"):
            vm.eval_transactional(f"Actor[#{singleton}] := 123!!")

        with pytest.raises(RuntimeError, match="Cannot mutate locked system singleton"):
            vm.eval_transactional(f"atPut(Actor, #{singleton}, 123)!!")


def test_host_lockdown_blocks_exec_shell_and_host_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify `ACTORII_HOST_LOCKDOWN=1` blocks `Actor[#Host]>>execShell` on the host and blocks writing outside `/tmp`."""
    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    monkeypatch.delenv("ACTORII_IN_DOCKER", raising=False)

    vm, harness, sandbox = bootstrap_agent_harness(
        ActorVM(),
        enable_memory_graph=False,
        enable_engrams=False,
    )
    host_bridge = vm.global_vars["Actor"]["Host"]._target

    # 1. `execShell(Actor[#Host], ...)` must be strictly blocked on the host in lockdown mode
    with pytest.raises(PermissionError, match="BLOCKED: Host shell execution via Actor\\[#Host\\] is strictly disabled"):
        host_bridge.execShell("echo hello", 5)

    # 2. Writing to `<repo>/src/...` or `~/bin/...` must be strictly blocked
    repo_root = Path(__file__).resolve().parents[1]
    home_dir = Path.home().resolve()
    for forbidden_target in (
        str(repo_root / "src" / "actorii" / "evil.py"),
        str(repo_root / "actor_telegram.py"),
        str(home_dir / "bin" / "evil_helper"),
        str(home_dir / ".config" / "systemd" / "user" / "bilby-actor-telegram.service"),
        "/etc/resolv.conf",
    ):
        with pytest.raises(PermissionError):
            host_bridge.writeFile(forbidden_target, "malicious")


def test_forbidden_shell_patterns_blocked_even_when_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the destructive command patterns (2026-10-18 Bilby remediation guardrails) are blocked even when lockdown is off."""
    monkeypatch.delenv("ACTORII_HOST_LOCKDOWN", raising=False)
    vm, _, _ = bootstrap_agent_harness(
        ActorVM(),
        enable_memory_graph=False,
        enable_engrams=False,
    )
    host_bridge = vm.global_vars["Actor"]["Host"]._target

    home_dir = Path.home().resolve()
    dangerous_commands = [
        "systemctl stop NetworkManager",
        "systemctl disable systemd-resolved",
        "systemctl --user restart bilby-actor-telegram.service",
        "sudo rm -f /etc/resolv.conf",
        "nmcli networking off",
        "crontab -r",
        "pkill -f python3",
        "kill -9 1234",
        f"chmod 777 {home_dir}",
        f"echo stub > {home_dir}/bin/requests.py",
    ]
    for cmd in dangerous_commands:
        with pytest.raises(PermissionError, match="Blocked unsafe shell command pattern"):
            host_bridge.execShell(cmd, 5)


def test_evolve_harness_lockdown_and_tier2_bridge_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `evolveHarness` works on Ring 1 cognitive hooks during `ACTORII_HOST_LOCKDOWN=1`, blocks Tier 2 bridge calls, and respects `ACTORII_DISABLE_SELF_EVOLVE=1`."""
    vm, harness, sandbox = bootstrap_agent_harness(
        ActorVM(),
        enable_memory_graph=False,
        enable_engrams=False,
    )

    # 1. During normal host-lockdown (`ACTORII_HOST_LOCKDOWN=1`), guarded Ring 1 cognitive hook evolution (`policyNote`) IS allowed
    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    monkeypatch.delenv("ACTORII_DISABLE_SELF_EVOLVE", raising=False)
    res_allowed = sandbox.evolveHarness(
        harness,
        'now(AgentHarness)!!\nDef policyNote(self) { ^"evolved_in_lockdown" }!!',
        'policyNote(Actor[#Harness]) == "evolved_in_lockdown"!!',
    )
    assert res_allowed.startswith("EVOLVE_COMMITTED:"), res_allowed
    assert vm.eval_chunk("policyNote(Actor[#Harness])!!") == "evolved_in_lockdown"

    # 2. Injecting `execShell` or `writeFile` inside a Ring 1 hook is strictly rejected
    res_tier2 = sandbox.evolveHarness(
        harness,
        'now(AgentHarness)!!\nDef policyNote(self) { ^execShell(Actor[#Host], "id", 5) }!!',
    )
    assert res_tier2.startswith("EVOLVE_REJECTED(forbidden_bridge_call):"), res_tier2

    # 3. Overwriting a Ring 1 kernel method (`step`) via `evolveHarness` is strictly rejected
    res_kernel = sandbox.evolveHarness(
        harness,
        'now(AgentHarness)!!\nDef step(self) { ^1 }!!',
    )
    assert res_kernel.startswith("EVOLVE_REJECTED(lint):"), res_kernel

    # 4. Explicit `ACTORII_DISABLE_SELF_EVOLVE=1` still blocks evolution if requested
    monkeypatch.setenv("ACTORII_DISABLE_SELF_EVOLVE", "1")
    res_locked = sandbox.evolveHarness(
        harness,
        'now(AgentHarness)!!\nDef policyNote(self) { ^"locked" }!!',
    )
    assert res_locked.startswith("EVOLVE_REJECTED(lockdown):"), res_locked


def test_telegram_gateway_controls_and_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify `BilbyTelegramGateway` enforces fail-closed auth, `/controls`, and `/freeze` + `/unlock` kill-switch."""
    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    monkeypatch.delenv("ACTORII_DISABLE_SELF_EVOLVE", raising=False)
    gw = BilbyTelegramGateway(
        token="123:TEST",
        allowed_users=[100000001],
        default_chat_id="100000001",
        state_path=tmp_path / "gw_state.json",
        api_caller=lambda *_a, **_kw: {"ok": True, "result": {}},
    )

    # 1. Fail-closed authorization if `allowed_users` is empty
    gw_empty = BilbyTelegramGateway(
        token="123:TEST",
        allowed_users=[],
        default_chat_id="100000001",
        state_path=tmp_path / "gw_empty.json",
        api_caller=lambda *_a, **_kw: {"ok": True, "result": {}},
    )
    assert gw_empty._is_authorized(100000001, 100000001) is False

    # 2. `/controls` returns full security posture
    controls_out = gw.handle_text_command_or_goal("/controls")
    assert "Bilby Containment & Security Controls Report" in controls_out
    assert "LOCKED (1)" in controls_out
    assert "HARD-BLOCKED" in controls_out

    # 3. `/freeze` engages kill-switch and blocks both `/eval` and chat goals until `/unlock`
    freeze_out = gw.handle_text_command_or_goal("/freeze")
    assert "KILL-SWITCH ENGAGED" in freeze_out
    assert gw._frozen is True

    blocked_eval = gw.handle_text_command_or_goal("/eval 2 + 2!!")
    assert "currently FROZEN by kill-switch" in blocked_eval

    blocked_chat = gw.handle_text_command_or_goal("what is the house temperature?")
    assert "currently FROZEN by kill-switch" in blocked_chat

    unlock_out = gw.handle_text_command_or_goal("/unlock")
    assert "UNLOCKED" in unlock_out
    assert gw._frozen is False


def test_capability_vs_containment_matrix_and_multi_provider_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `capabilities(Actor[#Sandbox])!!`, `/capabilities`, `/config`, and multi-provider LLM resolution."""
    from actorii.ai import _is_local_llamacpp_or_vllm, resolve_default_endpoint_for_model, resolve_llm_api_key
    from actorii.tui import ActorTUISession

    monkeypatch.setenv("ACTORII_HOST_LOCKDOWN", "1")
    monkeypatch.setenv("ACTORII_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    monkeypatch.delenv("ACTORII_LLM_ENDPOINT", raising=False)

    assert resolve_default_endpoint_for_model("anthropic/claude-3.7-sonnet") == "https://openrouter.ai/api/v1/chat/completions"
    assert resolve_llm_api_key() == "sk-or-test-123"
    assert _is_local_llamacpp_or_vllm("https://openrouter.ai/api/v1/chat/completions") is False
    assert _is_local_llamacpp_or_vllm("http://127.0.0.1:8080/v1/chat/completions") is False  # provider is openrouter

    monkeypatch.setenv("ACTORII_LLM_PROVIDER", "llamacpp")
    assert _is_local_llamacpp_or_vllm("http://127.0.0.1:8080/v1/chat/completions") is True

    session = ActorTUISession(use_color=False)
    caps_out = str(session.vm.eval_chunk("capabilities(Actor[#Sandbox])!!"))
    assert "Capability vs. Containment Matrix" in caps_out
    assert "Tier 0 — Pure In-VM & Memory Graph" in caps_out
    assert "Tier 1 — Read-Only Document & Web Research" in caps_out
    assert "Tier 2 — Scoped Workspace File I/O" in caps_out
    assert "Tier 3 — Privileged Execution & Hardware Telemetry" in caps_out
    assert "PDF Reader (`Actor[#PDF]`)" in caps_out
    assert "Web & Search (`Actor[#Web]`)" in caps_out

    # Test `/config` runtime updates
    session.handle_input("/config model=gpt-4o provider=openai namespace=custom_ns max_steps=6")
    assert session.harness.model == "gpt-4o"
    assert session.harness.endpoint == "https://api.openai.com/v1/chat/completions"
    assert str(session.vm.eval_chunk("getNamespace(Actor[#Memory])!!")) == "custom_ns"
    assert session.harness._fields.get("maxSteps") == 6

