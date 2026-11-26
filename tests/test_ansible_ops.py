"""Tests for `Actor[#Ops]` (`AnsibleOpsBridge`) — OS-agnostic guarded host ops & Ansible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actorii.core import Symbol
from actorii.guardrails.linter import PROTECTED_POOL_SINGLETONS
from actorii.harness import bootstrap_agent_harness
from actorii.harness.ansible_ops import AnsibleOpsBridge, _is_protected_service, _scrub_secrets


def _d(actor_dict: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in actor_dict._entries.items():
        key = k.name if isinstance(k, Symbol) else str(k)
        out[key] = _d(v) if hasattr(v, "_entries") else v
    return out


def test_secret_redaction_and_protected_services():
    raw = "token=abc123456789012345678901234567890 sk-abcdefghijklmnopqrstuvwx 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
    scrubbed = _scrub_secrets(raw)
    assert "abc123456" not in scrubbed
    assert "sk-abcdef" not in scrubbed
    assert "[TELEGRAM_TOKEN_REDACTED]" in scrubbed

    for svc in (
        "NetworkManager.service",
        "systemd-resolved",
        "tailscaled.service",
        "gdm.service",
        "firewalld",
        "sshd.service",
        "bilby-actor-telegram.service",
        "llama-primary.service",
    ):
        assert _is_protected_service(svc) is True

    assert _is_protected_service("bilby-coder-8083.service") is False
    assert _is_protected_service("honeypot-dashboard.service") is False


def test_http_check_blocks_external_hosts():
    ops = AnsibleOpsBridge()
    res = _d(ops.httpCheck("https://example.com/", 3))
    assert res["ok"] is False
    assert "localhost/LAN" in res["error"]


def test_playbook_guardrails_and_mandatory_dry_run_token(tmp_path: Path, monkeypatch):
    pb_dir = tmp_path / "playbooks" / "bilby"
    pb_dir.mkdir(parents=True)
    for name in ("restart_user_service.yml", "backup_jetson.yml", "gather_facts.yml"):
        (pb_dir / name).write_text("---\n- hosts: valkyrie\n  tasks: []\n", encoding="utf-8")
    (tmp_path / "ansible-bilby.cfg").write_text("[defaults]\n", encoding="utf-8")

    ops = AnsibleOpsBridge(ansible_dir=tmp_path, audit_log=tmp_path / "audit.jsonl")

    # 1. Unknown playbook blocked
    bad_pb = _d(ops.planPlaybook("evil_shell.yml", "valkyrie", "{}"))
    assert bad_pb["ok"] is False
    assert "OPS_GUARDRAIL_BLOCKED" in bad_pb["error"]

    # 2. Protected OS service blocked even in curated restart_user_service.yml
    bad_svc = _d(ops.planPlaybook("restart_user_service.yml", "valkyrie", json.dumps({"service_name": "NetworkManager.service"})))
    assert bad_svc["ok"] is False
    assert "OPS_GUARDRAIL_BLOCKED" in bad_svc["error"]

    # 3. Apply without planToken blocked
    no_tok = _d(ops.applyPlaybook("backup_jetson.yml", "valkyrie", "{}", ""))
    assert no_tok["ok"] is False
    assert "planToken" in no_tok["error"]

    # 4. Plan -> Apply with matching token succeeds (mocked ansible-playbook exec)
    calls: list[bool] = []

    def _fake_exec(pb_path, target_host, extra_vars, *, check_mode, timeout_s=90.0):
        calls.append(check_mode)
        return {"ok": True, "exitCode": 0, "elapsedMs": 12.0, "stdout": "ok=1 changed=0", "stderr": ""}

    monkeypatch.setattr(ops, "_exec_ansible_playbook", _fake_exec)
    plan = _d(ops.planPlaybook("backup_jetson.yml", "valkyrie", "{}"))
    assert plan["ok"] is True
    assert len(plan["planToken"]) == 16
    assert calls == [True]

    # Mismatched vars rejected
    bad_apply = _d(ops.applyPlaybook("backup_jetson.yml", "valkyrie", json.dumps({"keep_snapshots": 5}), plan["planToken"]))
    assert bad_apply["ok"] is False
    assert "does not match" in bad_apply["error"]

    # Exact matching token succeeds and consumes token (single-use)
    applied = _d(ops.applyPlaybook("backup_jetson.yml", "valkyrie", "{}", plan["planToken"]))
    assert applied["ok"] is True
    assert calls == [True, False]

    # Replay of same token blocked
    replay = _d(ops.applyPlaybook("backup_jetson.yml", "valkyrie", "{}", plan["planToken"]))
    assert replay["ok"] is False
    assert (tmp_path / "audit.jsonl").is_file()


def test_actor_ops_wired_protected_and_lazy_schema_injected(monkeypatch):
    monkeypatch.setenv("ACTORII_DISABLE_DEMO_TWINS", "1")
    vm, _, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
    actor = vm.global_vars["Actor"]
    assert "Ops" in actor
    assert "Ops" in PROTECTED_POOL_SINGLETONS

    desc = str(vm.eval_chunk("describe(Actor[#Ops])!!"))
    assert "readLogs" in desc
    assert "planPlaybook" in desc

    # Lazy Step-1 schema injection
    assert "`Actor[#Ops]`" in sandbox.poolSummary("Check the system logs and run a Jetson backup playbook")
    assert "`Actor[#Ops]`" not in sandbox.poolSummary("What is 2+2?")

    # Linter blocks overwriting Actor[#Ops]
    lint_res = sandbox.lintChunk("Actor[#Ops] := 123!!")
    assert lint_res.startswith("ERROR:")
