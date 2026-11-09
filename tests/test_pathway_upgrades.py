"""Verification suite for the 3 Self-Implemented Novel Pathway Upgrades in ActorII.

- Spec 1 (Pathway 3 — Virtual Paging): L1/L2 Step Result Paging in `VMSandbox.formatStepHistory`.
- Spec 2 (Pathway 4 — CodeAct / LLM-Native ACI): AST Symbol Inspector `readSymbol(Actor[#Host], path, symbol)`.
- Spec 3 (Pathway 1 — Darwin/Huxley-Gödel Ratchet): Test-Score Regression Warning & `undoLastEdit(Actor[#Host], path)`.
"""

from __future__ import annotations

from pathlib import Path

from actorii.harness import bootstrap_agent_harness


def test_spec1_l1_result_paging() -> None:
    """Verify `formatStepHistory` pages out step results older than the 2 most recent steps (`idx_s < n_recent - 2`)."""
    vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
    step_log = harness._fields["stepLog"]
    step_log.clear()

    long_old_res = "OLD_STEP_OUTPUT_" + ("A" * 400)
    long_new_res = "RECENT_STEP_OUTPUT_" + ("B" * 400)

    step_log.add({"step": 1, "action": 'readFile(Actor[#Host], "/workspace/a.py", 1, 50)!!', "status": "ok", "gas_used": 5, "result": long_old_res})
    step_log.add({"step": 2, "action": 'readFile(Actor[#Host], "/workspace/b.py", 1, 50)!!', "status": "ok", "gas_used": 5, "result": "short_res"})
    step_log.add({"step": 3, "action": 'readFile(Actor[#Host], "/workspace/c.py", 1, 50)!!', "status": "ok", "gas_used": 5, "result": "recent_1"})
    step_log.add({"step": 4, "action": 'readFile(Actor[#Host], "/workspace/d.py", 1, 50)!!', "status": "ok", "gas_used": 5, "result": long_new_res})

    formatted = sandbox.formatStepHistory(harness, page_old_results=True)
    assert "[paged out]" in formatted
    assert ("A" * 350) not in formatted
    assert ("B" * 350) in formatted


def test_spec2_ast_read_symbol(tmp_path: Path) -> None:
    """Verify `readSymbol(Actor[#Host], path, symbol_name)` extracts exact AST function/class ranges with line numbers."""
    sample_py = tmp_path / "sample_mod.py"
    sample_py.write_text(
        "# Header comment\n"
        "X = 10\n\n"
        "def compute_alpha(a: int, b: int) -> int:\n"
        "    total = a + b\n"
        "    return total * 2\n\n"
        "class WidgetEngine:\n"
        "    def run(self) -> str:\n"
        "        return 'ok'\n",
        encoding="utf-8",
    )

    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

    fn_out = vm.eval_chunk(f'readSymbol(Actor[#Host], "{sample_py}", "compute_alpha")!!')
    assert "symbol `compute_alpha` lines 4-6 of 10" in str(fn_out)
    assert "   4: def compute_alpha(a: int, b: int) -> int:" in str(fn_out)
    assert "   6:     return total * 2" in str(fn_out)
    assert "WidgetEngine" not in str(fn_out)

    cls_out = vm.eval_chunk(f'readSymbol(Actor[#Host], "{sample_py}", "WidgetEngine")!!')
    assert "symbol `WidgetEngine` lines 8-10 of 10" in str(cls_out)
    assert "   8: class WidgetEngine:" in str(cls_out)

    missing_out = vm.eval_chunk(f'readSymbol(Actor[#Host], "{sample_py}", "nonexistent_fn")!!')
    assert "not found" in str(missing_out)


def test_spec3_godel_test_ratchet_and_undo(tmp_path: Path) -> None:
    """Verify `_auto_verify_python_edit` flags regressions when passing tests drop and `undoLastEdit` restores the file."""
    calc_py = tmp_path / "calc.py"
    calc_test = tmp_path / "calc_test.py"
    calc_test.write_text(
        "from calc import add, mul\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_mul():\n"
        "    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )

    vm, _, _ = bootstrap_agent_harness(enable_memory_graph=False)

    # 1. Write passing implementation (2 passed)
    res1 = vm.eval_chunk(
        f'writeFile(Actor[#Host], "{calc_py}", """def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n""")!!'
    )
    assert "[AUTO-TEST PASS" in str(res1)

    # 2. Introduce regression (only 1 passed, 1 failed)
    res2 = vm.eval_chunk(
        f'editFile(Actor[#Host], "{calc_py}", """def mul(a, b):\n    return a * b""", """def mul(a, b):\n    return a - b""")!!'
    )
    assert "[AUTO-TEST FAIL" in str(res2)
    assert "[REGRESSION WARNING: passing tests dropped from 2 to 1" in str(res2)

    # 3. Call `undoLastEdit(Actor[#Host], path)` from Actor bytecode and verify restoration
    undo_res = vm.eval_chunk(f'undoLastEdit(Actor[#Host], "{calc_py}")!!')
    assert "UNDO_OK:" in str(undo_res)
    assert "return a * b" in calc_py.read_text(encoding="utf-8")
