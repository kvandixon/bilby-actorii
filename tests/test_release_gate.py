"""Unit tests for the ActorII 3-Tier DevOps Release & Benchmark Gate (`scripts/release_gate.py`)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys


def _load_release_gate():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("release_gate", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["release_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_src_tree_hash_deterministic_and_sensitive(tmp_path: Path) -> None:
    rg = _load_release_gate()
    src = tmp_path / "src" / "actorii"
    src.mkdir(parents=True)
    (src / "a.py").write_text("x = 1\n", encoding="utf-8")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"\x00\x01")

    h1 = rg.compute_src_tree_hash(src)
    h2 = rg.compute_src_tree_hash(src)
    assert h1 == h2

    # Modifying `.pyc` is ignored, modifying `.py` changes the hash
    (src / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"\x99\x99")
    assert rg.compute_src_tree_hash(src) == h1

    (src / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert rg.compute_src_tree_hash(src) != h1


def test_canary_hard_block_and_frontier_auto_ratchet() -> None:
    rg = _load_release_gate()
    baseline = {
        "self_evolution": {
            "canaries": ["sensor_window_median", "harness_step_budget_advisor"],
            "frontier": {
                "circuit_breaker_policy": {
                    "best_holdout_pass_rate": 0.30,
                    "best_visible_pass_rate": 0.20,
                    "last_status": "failed",
                }
            },
        }
    }

    # Case 1: Canary fails -> hard block (`passed == False`)
    bad_report = {
        "results": [
            {"spec_id": "sensor_window_median", "passed": False, "holdout_pass_rate": 0.5},
            {"spec_id": "harness_step_budget_advisor", "passed": True, "holdout_pass_rate": 1.0},
        ]
    }
    res_bad = rg.evaluate_self_evo_against_baseline(bad_report, copy.deepcopy(baseline), ratchet=True)
    assert res_bad["passed"] is False
    assert res_bad["canary_failures"] == ["sensor_window_median"]

    # Case 2: Canaries pass AND frontier spec newly passes -> auto-ratcheted into Canaries!
    good_report = {
        "results": [
            {"spec_id": "sensor_window_median", "passed": True, "holdout_pass_rate": 1.0},
            {"spec_id": "harness_step_budget_advisor", "passed": True, "holdout_pass_rate": 1.0},
            {
                "spec_id": "circuit_breaker_policy",
                "passed": True,
                "persistence_verified": True,
                "holdout_pass_rate": 1.0,
                "visible_pass_rate": 1.0,
            },
        ]
    }
    b2 = copy.deepcopy(baseline)
    res_good = rg.evaluate_self_evo_against_baseline(good_report, b2, ratchet=True)
    assert res_good["passed"] is True
    assert res_good["ratcheted_to_canary"] == ["circuit_breaker_policy"]
    assert "circuit_breaker_policy" in b2["self_evolution"]["canaries"]
    assert "circuit_breaker_policy" not in b2["self_evolution"]["frontier"]

    # Case 3: Frontier holdout drops > 5% -> warning recorded without blocking unrelated bugfix
    drop_report = {
        "results": [
            {"spec_id": "sensor_window_median", "passed": True, "holdout_pass_rate": 1.0},
            {"spec_id": "harness_step_budget_advisor", "passed": True, "holdout_pass_rate": 1.0},
            {
                "spec_id": "circuit_breaker_policy",
                "passed": False,
                "persistence_verified": False,
                "holdout_pass_rate": 0.10,
                "visible_pass_rate": 0.20,
            },
        ]
    }
    res_drop = rg.evaluate_self_evo_against_baseline(drop_report, copy.deepcopy(baseline), ratchet=True)
    assert res_drop["passed"] is True
    assert len(res_drop["frontier_warnings"]) == 1


def test_verify_stamp_tier_and_hash_enforcement(tmp_path: Path, monkeypatch) -> None:
    rg = _load_release_gate()
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "vm.py").write_text("v = 1\n", encoding="utf-8")
    stamp_file = tmp_path / ".gate_stamp.json"

    monkeypatch.setattr(rg, "SRC_DIR", fake_src)
    monkeypatch.setattr(rg, "STAMP_PATH", stamp_file)
    monkeypatch.setattr(
        rg,
        "classify_change_scope",
        lambda *a, **kw: {
            "scope": "major",
            "required_tier": "live",
            "core_touched": ["src/actorii/vm.py"],
            "src_diff_lines": 120,
        },
    )

    # Missing stamp -> blocked
    ok, msg = rg.verify_stamp()
    assert ok is False
    assert "No verification stamp" in msg

    # `fast` stamp when `live` is required -> blocked
    cur_hash = rg.compute_src_tree_hash(fake_src)
    stamp_file.write_text(
        json.dumps({"passed": True, "tier": "fast", "src_tree_hash": cur_hash, "timestamp": "now"}),
        encoding="utf-8",
    )
    ok2, msg2 = rg.verify_stamp()
    assert ok2 is False
    assert "requires `--tier live`" in msg2

    # `live` stamp with matching hash -> verified!
    stamp_file.write_text(
        json.dumps({"passed": True, "tier": "live", "src_tree_hash": cur_hash, "timestamp": "now"}),
        encoding="utf-8",
    )
    ok3, msg3 = rg.verify_stamp()
    assert ok3 is True
    assert "GATE VERIFIED" in msg3

    # Editing `vm.py` invalidates the stamp immediately -> blocked!
    (fake_src / "vm.py").write_text("v = 2\n", encoding="utf-8")
    ok4, msg4 = rg.verify_stamp()
    assert ok4 is False
    assert "changed since last verified gate stamp" in msg4
