"""Tests for `Actor[#Network]` (`NetworkTopology` + `NetworkTopologyBridge`) and the Zero-Action Hallucination Guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from actorii.harness import bootstrap_agent_harness
from actorii.tui import ActorTUISession


def test_zero_action_hallucination_guard_blocks_fabricated_install_and_allows_real_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce the exact `09:43` 'Yes please. That was what I was missing.' Tailscale hallucination and verify it is blocked until a real tool executes."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ACTORII_BENCH_STRICT", raising=False)

    session = ActorTUISession(max_steps=6, max_failures=4, use_color=False)

    # Seed Turn 1 in `chatLog` ending with Bilby asking "Want me to install and configure Tailscale on the Jetson now?"
    turns_state: Dict[str, Any] = {"calls": 0, "step2_prompt": ""}

    def brain(prompt: str, system_prompt: str = "", meta: dict | None = None) -> str:
        if "GOAL: Check why 100.112.213.15:8090 failed" in prompt:
            return "COMPLETE: Tailscale is not installed on the Jetson yet. Want me to install and configure Tailscale on the Jetson now?"
        if "GOAL: Yes please. That was what I was missing." in prompt:
            turns_state["calls"] += 1
            if turns_state["calls"] == 1:
                # Step 1: Attempt to hallucinate `Tailscale installed and running` + fake PID with ZERO executed tools!
                return (
                    "COMPLETE: **Tailscale installed and running on the Jetson.**\n"
                    "- Tailscale IP: `100.112.213.15`\n"
                    "- Interface: `tailscale0` is UP\n"
                    "- Stream: `julia_stream` is running (PID 24785) on port 8090"
                )
            if turns_state["calls"] == 2:
                turns_state["step2_prompt"] = prompt
                # Step 2: After `UnverifiedExecutionClaimError`, Bilby inspects `Actor[#Network]` for real!
                return 'whoOwnsIp(Actor[#Network], "100.112.213.15")!!'
            return (
                "COMPLETE: Checked `Actor[#Network]` — `100.112.213.15` actually belongs to your phone, "
                "and the Jetson is LAN-only (`192.168.68.57`)."
            )
        return "COMPLETE: ok"

    session.harness.custom_provider = brain
    session.handle_input("/chat Check why 100.112.213.15:8090 failed")
    session.handle_input("/chat Yes please. That was what I was missing.")

    assert turns_state["calls"] == 3
    assert "UnverifiedExecutionClaimError" in turns_state["step2_prompt"]
    hist = session.get_chat_history(local_only=True)
    assert len(hist) == 2
    assert "actually belongs to your phone" in hist[1]["reply"]
    assert any("whoOwnsIp" in str(a) for a in (hist[1].get("actions") or []))


def test_network_topology_tailnet_ownership_and_relay_ssrf_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `Actor[#Network]` (`whoOwnsIp`, `topologyMap`, `explainReachability`, `exposeLanPort` SSRF & port guards, and TUI `/network`)."""
    monkeypatch.setenv("ACTORII_STATE_DIR", str(tmp_path / "state"))
    session = ActorTUISession(use_color=False)
    vm = session.vm

    topo_py = vm.global_vars["Actor"]["NetTopo"].target
    jetson_py = vm.global_vars["Actor"]["Jetson"].target
    jetson_py._save_cfg({"host": "192.168.68.57", "mdns": "jetson.local", "user": "kilog", "port": 22})

    from actorii.harness.twins import _to_actor_dict
    from actorii.classes.collections import ActorArray

    def fake_peers() -> Any:
        rows = [
            _to_actor_dict(
                {
                    "role": "self",
                    "hostname": "valkyrie",
                    "dnsName": "valkyrie.tailnet.ts.net",
                    "os": "linux",
                    "tailscaleIp": "100.124.50.33",
                    "online": True,
                }
            ),
            _to_actor_dict(
                {
                    "role": "peer",
                    "hostname": "kevins-s25-ultra",
                    "dnsName": "kevins-s25-ultra.tailnet.ts.net",
                    "os": "android",
                    "tailscaleIp": "100.112.213.15",
                    "online": True,
                }
            ),
        ]
        arr = ActorArray(len(rows))
        for i, r in enumerate(rows):
            arr.atPut(i, r)
        return arr

    def fake_bridges() -> Any:
        b = _to_actor_dict(
            {
                "listenPort": 8091,
                "targetHost": "192.168.68.57",
                "targetPort": 8090,
                "pid": 55111,
                "alive": True,
                "managedBy": "socat",
            }
        )
        arr = ActorArray(1)
        arr.atPut(0, b)
        return arr

    monkeypatch.setattr(topo_py, "tailnetPeers", fake_peers)
    monkeypatch.setattr(topo_py, "listBridges", fake_bridges)

    # 1. `whoOwnsIp` immediately catches that `100.112.213.15` is the Android phone, NOT the Jetson!
    phone_info = vm.eval_chunk('whoOwnsIp(Actor[#Network], "http://100.112.213.15:8090/")!!')
    assert phone_info.at("ok") is True
    assert phone_info.at("owner") == "kevins-s25-ultra"
    assert phone_info.at("network") == "tailscale"
    assert "NOT the Jetson" in str(phone_info.at("note"))

    # 2. `whoOwnsIp` identifies `100.124.50.33` as `valkyrie` and warns about `:8090` Studio
    valk_info = vm.eval_chunk('whoOwnsIp(Actor[#Network], "100.124.50.33")!!')
    assert valk_info.at("owner") == "valkyrie"
    assert ":8090" in str(valk_info.at("note"))

    # 3. `explainReachability` discovers the active `:8091` bridge and returns `http://100.124.50.33:8091/`
    reach = str(
        vm.eval_chunk('explainReachability(Actor[#Network], "100.112.213.15", "192.168.68.57", 8090)!!')
    )
    assert "http://100.124.50.33:8091/" in reach

    # 4. `exposeLanPort` SSRF & reserved-port guardrails:
    bad_port = vm.eval_chunk('exposeLanPort(Actor[#Network], "192.168.68.57", 8090, 8080)!!')
    assert bad_port.at("ok") is False
    assert "BLOCKED" in str(bad_port.at("error"))

    bad_loopback = vm.eval_chunk('exposeLanPort(Actor[#Network], "127.0.0.1", 8080, 8095)!!')
    assert bad_loopback.at("ok") is False
    assert "RELAY_SSRF_GUARD" in str(bad_loopback.at("error"))

    # Existing `:8091` route returns `ok: True` idempotently with Tailscale URL
    existing = vm.eval_chunk('exposeLanPort(Actor[#Network], "192.168.68.57", 8090, 8091)!!')
    assert existing.at("ok") is True
    assert existing.at("tailscaleUrl") == "http://100.124.50.33:8091/"

    # 5. TUI `/network` command renders full topology map
    before = len(session.transcript)
    session.handle_input("/network")
    t_out = "\n".join(session.transcript[before:])
    assert "kevins-s25-ultra" in t_out
    assert "valkyrie:8091 -> 192.168.68.57:8090" in t_out
