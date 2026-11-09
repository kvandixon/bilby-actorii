# -*- coding: utf-8 -*-
"""Milestone M2 Unit Tests: Bytecode Compiler & Opcode Extensions.

Tests the authentic 1993 Whitewater Actor 4.1 bytecode compiler:
- Opcode set extensions, non-colliding numeric assignments, and aliases
- Local variable slot mapping (self at 0, params at 1..N, locals at N+1..M)
- Implicit return of self vs explicit caret return (^)
- Assignment compilation (STORE_LOCAL, STORE_GLOBAL, SET_FIELD, SET_INDEX)
- Collection indexing compilation (GET_INDEX, SET_INDEX)
- Block closure compilation (BlockDescriptor, MAKE_BLOCK, RETURN_BLOCK)
- Control flow branching and loops (JMP_IF_FALSE, JMP, backpatching)
- Early-bound qualification (CALL_EARLY, EarlyBoundDescriptor)
- Built-in primitives (INVOKE_PRIM for max, sqrt, asInt, dec, fill)
- Full compilation of historical benchmarks (SIEVE.ACT, SIEVE1.ACT, FRACTAL.ACT)
- ActorVM execution of compiled bytecodes
"""

import os
import pytest

from actorii.vm import ActorVM, Opcode, OpcodeValue
from actorii.core import ActorInheritanceHierarchy
from actorii.parser import (
    parse_actor,
    compile_to_bytecode,
    BlockDescriptor,
    EarlyBoundDescriptor,
    Scope,
    MethodDefNode,
    ClassDefNode,
    NowNode,
    BlockNode,
    IfNode,
    LoopNode,
    EarlyBoundNode,
    IndexGetNode,
    IndexSetNode,
    ReturnNode,
    AssignNode,
)

HISTORICAL_ACT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "historical_actor_41_source",
    "actact",
)


# ==============================================================================
# 1. Opcode Enum & Encoding Tests
# ==============================================================================

class TestOpcodeSetExtensions:
    """Verifies that all M2 opcodes and aliases are defined correctly."""

    def test_baseline_opcodes_retained(self):
        """All 25 baseline opcodes retain their exact numeric values."""
        assert Opcode.PUSH_NULL == 0x00
        assert Opcode.PUSH_INT == 0x01
        assert Opcode.PUSH_STR == 0x02
        assert Opcode.PUSH_OBJ == 0x03
        assert Opcode.PUSH_GLOBAL == 0x04
        assert Opcode.PUSH_LOCAL == 0x05
        assert Opcode.NEW_OBJ == 0x10
        assert Opcode.CALL_METHOD == 0x20
        assert Opcode.CALL_GLOBAL == 0x21
        assert Opcode.GET_FIELD == 0x30
        assert Opcode.SET_FIELD == 0x31
        assert Opcode.INHERIT == 0x40
        assert Opcode.BINARY_ADD == 0x50
        assert Opcode.BINARY_SUB == 0x51
        assert Opcode.BINARY_MUL == 0x52
        assert Opcode.BINARY_DIV == 0x53
        assert Opcode.BINARY_EQ == 0x54
        assert Opcode.BINARY_NE == 0x55
        assert Opcode.POP == 0x56
        assert Opcode.POP_N == 0x57
        assert Opcode.JMP == 0x60
        assert Opcode.JMP_IF_TRUE == 0x61
        assert Opcode.JMP_IF_FALSE == 0x62
        assert Opcode.RETURN == 0x70
        assert Opcode.HALT == 0xFF

    def test_new_variable_store_opcodes(self):
        """Variable store and load opcodes have expected non-colliding values."""
        assert Opcode.STORE_LOCAL == 0x06
        assert Opcode.STORE_GLOBAL == 0x07
        assert Opcode.STORE_NAME == 0x08
        assert Opcode.LOAD_NAME == 0x09
        assert Opcode.LOAD_CONST == 0x0A

    def test_opcode_aliases(self):
        """Standard aliases map to identical underlying opcode values."""
        assert Opcode.LOAD_LOCAL == Opcode.PUSH_LOCAL == 0x05
        assert Opcode.LOAD_GLOBAL == Opcode.PUSH_GLOBAL == 0x04
        assert Opcode.JUMP_ABSOLUTE == Opcode.JMP == 0x60
        assert Opcode.JUMP_IF_FALSE == Opcode.JMP_IF_FALSE == 0x62
        assert Opcode.JUMP_IF_TRUE == Opcode.JMP_IF_TRUE == 0x61

    def test_new_indexing_opcodes(self):
        """Indexing opcodes GET_INDEX and SET_INDEX."""
        assert Opcode.GET_INDEX == 0x32
        assert Opcode.SET_INDEX == 0x33

    def test_new_relational_and_stack_opcodes(self):
        """Relational comparisons and stack duplication opcodes."""
        assert Opcode.BINARY_LT == 0x58
        assert Opcode.BINARY_LE == 0x59
        assert Opcode.BINARY_GT == 0x5A
        assert Opcode.BINARY_GE == 0x5B
        assert Opcode.DUP == 0x5C

    def test_new_jump_opcodes(self):
        """Control flow jump additions."""
        assert Opcode.JUMP_FORWARD == 0x63
        assert Opcode.POP_JUMP_IF_FALSE == 0x64
        assert Opcode.POP_JUMP_IF_TRUE == 0x65

    def test_new_closure_and_early_bound_opcodes(self):
        """Closure, return, early-binding, and primitive opcodes."""
        assert Opcode.MAKE_BLOCK == 0x24
        assert Opcode.CALL_BLOCK == 0x25
        assert Opcode.CALL_EARLY == 0x22
        assert Opcode.INVOKE_PRIM == 0x23
        assert Opcode.RETURN_VALUE == 0x71
        assert Opcode.RETURN_BLOCK == 0x72

    def test_opcode_string_representation(self):
        """Opcode values contain their mnemonic in str() for test assertions."""
        assert "BLOCK" in str(Opcode.MAKE_BLOCK)
        assert str(Opcode.MAKE_BLOCK) == "MAKE_BLOCK"
        assert str(Opcode.CALL_EARLY) == "CALL_EARLY"
        assert str(Opcode.GET_INDEX) == "GET_INDEX"
        assert str(Opcode.SET_INDEX) == "SET_INDEX"


# ==============================================================================
# 2. Method Definition & Slot Allocation Compilation Tests
# ==============================================================================

class TestMethodCompilationAndSlots:
    """Tests compilation of methods, parameter binding, and local slots."""

    def test_method_slot_allocation_order(self):
        """Slot 0 is self, parameters are 1..N, declared locals are N+1..M."""
        source = "now(Tester)!! Def compute(self, a, b | x y) { x := a + b; ^x; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        # compute should be registered in constants
        assert "compute" in consts
        assert "Tester" in consts

        # Look for STORE_LOCAL for x: x is declared first local after a (slot 1), b (slot 2) -> slot 3
        store_ops = [(op, operand) for op, operand in insts if op == Opcode.STORE_LOCAL]
        assert len(store_ops) >= 1
        assert store_ops[0][1] == 3  # slot 3 is x

        # Look for LOAD_LOCAL 3 before return
        load_ops = [(op, operand) for op, operand in insts if op == Opcode.LOAD_LOCAL]
        slots_loaded = [operand for _, operand in load_ops]
        assert 1 in slots_loaded  # a
        assert 2 in slots_loaded  # b
        assert 3 in slots_loaded  # x

    def test_implicit_return_of_self(self):
        """Methods without an explicit ^ return self (LOAD_LOCAL 0, RETURN_VALUE 0)."""
        source = "now(Worker)!! Def doWork(self, x | temp) { temp := x * 2; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)

        # Final two instructions of the method must load slot 0 and return value
        assert (Opcode.LOAD_LOCAL, 0) in insts
        assert (Opcode.RETURN_VALUE, 0) in insts

    def test_bare_caret_returns_self(self):
        """A bare caret return (^;) explicitly returns self."""
        source = "now(Worker)!! Def finish(self) { ^; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)
        assert (Opcode.LOAD_LOCAL, 0) in insts
        assert (Opcode.RETURN_VALUE, 0) in insts

    def test_explicit_value_return(self):
        """Explicit ^expr returns the evaluated expression."""
        source = "now(Calc)!! Def answer(self) { ^42; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)
        assert (Opcode.PUSH_INT, 42) in insts
        assert (Opcode.RETURN_VALUE, 0) in insts

    def test_empty_method_body(self):
        """Method with completely empty body safely returns self."""
        source = "now(Stub)!! Def noop(self) {}!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)
        assert len(insts) >= 2
        assert (Opcode.LOAD_LOCAL, 0) in insts
        assert (Opcode.RETURN_VALUE, 0) in insts


# ==============================================================================
# 3. Assignment & Indexing Compilation Tests
# ==============================================================================

class TestAssignmentAndIndexingCompilation:
    """Tests compilation of assignments, fields, and collection indexing."""

    def test_assign_to_local_slot(self):
        """Assigning to declared local variable emits STORE_LOCAL."""
        source = "Def testAssign(self, val | target) { target := val; ^target; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)
        opcodes = [op for op, _ in insts]
        assert Opcode.STORE_LOCAL in opcodes

    def test_assign_to_field(self):
        """Assigning to an object field (self.x := 10) emits SET_FIELD."""
        source = "Def setField(self) { self.x := 10; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)
        opcodes = [op for op, _ in insts]
        assert Opcode.SET_FIELD in opcodes
        assert "x" in consts

    def test_collection_index_read(self):
        """Collection read coll[i] emits GET_INDEX."""
        source = "Def readItem(self, arr, i) { ^arr[i]; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)
        opcodes = [op for op, _ in insts]
        assert Opcode.GET_INDEX in opcodes

    def test_collection_index_write(self):
        """Collection write coll[i] := val emits SET_INDEX."""
        source = "Def writeItem(self, arr, i, val) { arr[i] := val; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)
        opcodes = [op for op, _ in insts]
        assert Opcode.SET_INDEX in opcodes

    def test_global_actor_dictionary_write(self):
        """Actor[#Key] := val top-level assignment emits SET_INDEX."""
        source = "Actor[#Sam] := 100!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)
        opcodes = [op for op, _ in insts]
        assert Opcode.SET_INDEX in opcodes
        assert "Actor" in consts
        assert "#Sam" in consts


# ==============================================================================
# 4. Block Closure Compilation Tests
# ==============================================================================

class TestBlockClosureCompilation:
    """Tests block closure compilation into BlockDescriptor and MAKE_BLOCK."""

    def test_simple_block_closure_emission(self):
        """{using(x) x + 1;} emits MAKE_BLOCK and stores BlockDescriptor."""
        source = "{using(x) x + 1;}"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        # MAKE_BLOCK instruction
        opcodes = [op for op, _ in insts]
        assert Opcode.MAKE_BLOCK in opcodes

        # BlockDescriptor in constants
        block_descs = [c for c in consts if isinstance(c, BlockDescriptor)]
        assert len(block_descs) == 1
        desc = block_descs[0]
        assert desc.params == ["x"]
        assert len(desc.instructions) > 0
        # Block body must end in RETURN_BLOCK
        assert desc.instructions[-1][0] == Opcode.RETURN_BLOCK

    def test_block_capturing_enclosing_variables(self):
        """Block accessing enclosing method parameters and locals."""
        source = "Def doFilter(self, limit | count) { count := 0; do(items, {using(x) if x < limit then count := count + 1; endif;}); ^count; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        block_descs = [c for c in consts if isinstance(c, BlockDescriptor)]
        assert len(block_descs) >= 1
        desc = block_descs[0]
        assert "x" in desc.params
        block_ops = [op for op, _ in desc.instructions]
        assert Opcode.BINARY_LT in block_ops
        assert Opcode.JMP_IF_FALSE in block_ops

    def test_nested_block_descriptors(self):
        """Nested blocks produce nested BlockDescriptor objects in constants."""
        source = "Def nested(self) { do(outer, {using(i) do(inner, {using(j) j + i;});}); }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        block_descs = [c for c in consts if isinstance(c, BlockDescriptor)]
        assert len(block_descs) >= 2


# ==============================================================================
# 5. Control Flow Branching & Loop Compilation Tests
# ==============================================================================

class TestControlFlowCompilation:
    """Tests conditional branching and iterative loops with jump backpatching."""

    def test_if_then_without_else(self):
        """if cond then body endif; emits JMP_IF_FALSE with backpatched target."""
        source = "Def testIf(self, x) { if x > 0 then x := 1; endif; ^x; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)

        jmp_false = [(i, op, operand) for i, (op, operand) in enumerate(insts) if op == Opcode.JMP_IF_FALSE]
        assert len(jmp_false) == 1
        idx, _, target = jmp_false[0]
        # Jump target must be strictly ahead of the jump instruction
        assert target > idx
        assert target <= len(insts)

    def test_if_then_else_branching(self):
        """if cond then a else b endif; emits JMP_IF_FALSE and unconditional JMP."""
        source = "Def testIfElse(self, x) { if x > 0 then ^1; else ^2; endif; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)

        opcodes = [op for op, _ in insts]
        assert Opcode.JMP_IF_FALSE in opcodes
        assert Opcode.JMP in opcodes

    def test_loop_pre_test(self):
        """loop while cond begin body endLoop; emits JMP_IF_FALSE and backward JMP."""
        source = "Def runLoop(self, n | i) { i := 0; loop while i < n begin i := i + 1; endLoop; ^i; }!!"
        ast = parse_actor(source)
        insts, _, _ = compile_to_bytecode(ast)

        opcodes = [op for op, _ in insts]
        assert Opcode.JMP_IF_FALSE in opcodes
        assert Opcode.JMP in opcodes

        # Backward jump points to or before condition
        jmps = [(i, target) for i, (op, target) in enumerate(insts) if op == Opcode.JMP]
        assert any(target < i for i, target in jmps)


# ==============================================================================
# 6. Early Binding & Primitives Compilation Tests
# ==============================================================================

class TestEarlyBindingAndPrimitives:
    """Tests early-bound method calls and built-in runtime primitives."""

    def test_early_bound_method_call(self):
        """target:Class.method() compiles to CALL_EARLY with EarlyBoundDescriptor."""
        source = "now(Sieve)!! Def testEarly(self) { ^self:Object.class(); }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        opcodes = [op for op, _ in insts]
        assert Opcode.CALL_EARLY in opcodes

        early_descs = [c for c in consts if isinstance(c, EarlyBoundDescriptor)]
        assert len(early_descs) == 1
        assert early_descs[0].target_class == "Object"
        assert early_descs[0].method == "class"
        assert early_descs[0].args == 0

    def test_early_bound_array_read(self):
        """flags:Object[i] registers early-bound descriptor and emits GET_INDEX."""
        source = "Def readEarly(self, flags, i) { ^flags:Object[i]; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        opcodes = [op for op, _ in insts]
        assert Opcode.GET_INDEX in opcodes
        early_descs = [c for c in consts if isinstance(c, EarlyBoundDescriptor)]
        assert len(early_descs) == 1
        assert early_descs[0].target_class == "Object"
        assert early_descs[0].method == "at"

    def test_early_bound_array_write(self):
        """flags:Object[j] := nil registers early-bound descriptor and emits SET_INDEX."""
        source = "Def writeEarly(self, flags, j) { flags:Object[j] := nil; }!!"
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        opcodes = [op for op, _ in insts]
        assert Opcode.SET_INDEX in opcodes
        early_descs = [c for c in consts if isinstance(c, EarlyBoundDescriptor)]
        assert len(early_descs) == 1
        assert early_descs[0].target_class == "Object"
        assert early_descs[0].method == "put"

    def test_invoke_prim_primitives(self):
        """max, sqrt, asInt, dec, fill emit INVOKE_PRIM."""
        source = """
Def testPrims(self, x | m s a d) {
    m := max(x, 10);
    s := sqrt(x);
    a := asInt(s);
    d := dec(x);
    fill(m, true);
}!!
"""
        ast = parse_actor(source)
        insts, consts, _ = compile_to_bytecode(ast)

        prim_insts = [op for op, _ in insts if op == Opcode.INVOKE_PRIM]
        assert len(prim_insts) == 5  # max, sqrt, asInt, dec, fill


# ==============================================================================
# 7. Authentic Target Benchmarks Compilation Tests
# ==============================================================================

class TestHistoricalBenchmarksCompilation:
    """Verifies end-to-end compilation of authentic Whitewater 4.1 benchmarks."""

    def test_compile_sieve_act(self):
        """SIEVE.ACT compiles cleanly to valid instructions with nested blocks."""
        sieve_path = os.path.join(HISTORICAL_ACT_ROOT, "SIEVE.ACT")
        assert os.path.isfile(sieve_path), f"Missing {sieve_path}"
        with open(sieve_path, "r", encoding="latin-1") as f:
            content = f.read()

        ast = parse_actor(content)
        assert len(ast) == 4  # ClassDefNode, NowNode, MethodDefNode, IndexSetNode

        insts, consts, strings = compile_to_bytecode(ast)
        assert len(insts) >= 30
        assert "Sieve" in consts
        assert "sieve" in consts

        opcodes = [op for op, _ in insts]
        assert Opcode.INVOKE_PRIM in opcodes
        assert Opcode.MAKE_BLOCK in opcodes
        assert Opcode.CALL_METHOD in opcodes
        assert Opcode.STORE_LOCAL in opcodes
        assert Opcode.SET_INDEX in opcodes

        # Verify inner and outer block descriptors
        block_descs = [c for c in consts if isinstance(c, BlockDescriptor)]
        assert len(block_descs) == 2
        param_sets = [b.params for b in block_descs]
        assert ["i"] in param_sets
        assert ["j"] in param_sets

    def test_compile_sieve1_act(self):
        """SIEVE1.ACT (with early-bound :Object syntax) compiles cleanly."""
        sieve1_path = os.path.join(HISTORICAL_ACT_ROOT, "SIEVE1.ACT")
        assert os.path.isfile(sieve1_path), f"Missing {sieve1_path}"
        with open(sieve1_path, "r", encoding="latin-1") as f:
            content = f.read()

        ast = parse_actor(content)
        insts, consts, strings = compile_to_bytecode(ast)
        assert len(insts) >= 30

        # Verify EarlyBoundDescriptor exists for :Object
        early_descs = [c for c in consts if isinstance(c, EarlyBoundDescriptor)]
        assert len(early_descs) >= 2
        assert all(d.target_class == "Object" for d in early_descs)

    def test_compile_fractal_act(self):
        """FRACTAL.ACT compiles cleanly into local allocations and method calls."""
        fractal_path = os.path.join(HISTORICAL_ACT_ROOT, "FRACTAL.ACT")
        assert os.path.isfile(fractal_path), f"Missing {fractal_path}"
        with open(fractal_path, "r", encoding="latin-1") as f:
            content = f.read()

        ast = parse_actor(content)
        insts, consts, strings = compile_to_bytecode(ast)
        assert len(insts) == 17
        assert "Int" in consts
        assert "point3D" in consts

        opcodes = [op for op, _ in insts]
        assert Opcode.STORE_LOCAL in opcodes
        assert Opcode.CALL_METHOD in opcodes
        assert Opcode.RETURN_VALUE in opcodes


# ==============================================================================
# 8. ActorVM Execution of Compiled Bytecodes
# ==============================================================================

class TestVMExecutionOfCompiledBytecode:
    """Verifies that bytecode compiled by the M2 compiler executes on ActorVM."""

    def test_vm_executes_compiled_arithmetic(self):
        """ActorVM executes compiled addition and returns expected value."""
        source = "Def add(self, a, b | res) { res := a + b; ^res; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(insts, consts, strings)
        # Pre-populate frame locals: self (0), a (1), b (2)
        vm.local_vars = [None, 15, 27, None]
        vm.run()
        assert vm.halted is True
        # Local slot 3 (res) must be 42
        assert vm.local_vars[3] == 42
        # Return value on stack
        assert len(vm.stack) >= 1
        assert vm.stack[-1] == 42

    def test_vm_executes_compiled_conditionals(self):
        """ActorVM correctly branches on compiled IfNode."""
        source = "Def sign(self, x | res) { if x > 0 then res := 1; else res := -1; endif; ^res; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        hierarchy = ActorInheritanceHierarchy()

        # Branch positive: x = 10
        vm_pos = ActorVM(hierarchy)
        vm_pos.load(insts, consts, strings)
        vm_pos.local_vars = [None, 10, None]
        vm_pos.run()
        assert vm_pos.halted is True
        assert vm_pos.local_vars[2] == 1

        # Branch negative: x = -5
        vm_neg = ActorVM(hierarchy)
        vm_neg.load(insts, consts, strings)
        vm_neg.local_vars = [None, -5, None]
        vm_neg.run()
        assert vm_neg.halted is True
        assert vm_neg.local_vars[2] == -1

    def test_vm_executes_primitive_max(self):
        """ActorVM executes compiled INVOKE_PRIM max."""
        source = "Def testMax(self, a, b | m) { m := max(a, b); ^m; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(insts, consts, strings)
        vm.local_vars = [None, 42, 99, None]
        vm.run()
        assert vm.halted is True
        assert vm.local_vars[3] == 99
        assert vm.stack[-1] == 99

    def test_vm_executes_primitive_fill(self):
        """ActorVM executes compiled INVOKE_PRIM fill."""
        source = "Def testFill(self, arr) { fill(arr, 7); ^arr; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(insts, consts, strings)
        my_arr = [0, 0, 0]
        vm.local_vars = [None, my_arr]
        vm.run()
        assert vm.halted is True
        assert my_arr == [7, 7, 7]


# ==============================================================================
# 9. Milestone M2 Remediation Tests (Gate Review & Challenge Invariants)
# ==============================================================================

class TestMilestoneM2Remediations:
    """Tests verified remediation fixes for M2 compiler and VM invariants."""

    def test_loop_condition_stack_balance_no_leaks(self):
        """Loop condition operand is popped on each iteration (0 leaked booleans after 100 iterations)."""
        source = "now(LoopTester)!! Def loopCount(self | i) { i := 0; loop while i < 100 begin i := i + 1; endLoop; ^i; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, 0]
        vm.run()

        assert vm.halted is True
        assert vm.local_vars[1] == 100
        # Stack must contain ONLY the return value, with 0 leaked booleans
        assert len(vm.stack) == 1
        assert vm.stack[0] == 100

    def test_return_of_assignment_expression_intact(self):
        """Return of assignment expression (^left := lft;) leaves assigned value on stack."""
        source = "now(ListNode)!! Def setLeft(self, lft) { ^left := lft; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = ["self_node", 42, None]
        vm.run()

        assert vm.halted is True
        assert vm.local_vars[2] == 42
        assert vm.stack == [42]

    def test_chained_assignment_execution(self):
        """Chained assignment (a := b := 5) sets both variables and leaves assigned value on stack."""
        source = "now(ChainTester)!! Def testChained(self | a b) { a := b := 5; ^a; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, None, None]
        vm.run()

        assert vm.halted is True
        assert vm.local_vars[1] == 5
        assert vm.local_vars[2] == 5
        assert vm.stack == [5]

    def test_unused_expression_statement_popped(self):
        """Unused expression statements (arithmetic, literals) are cleanly popped from stack."""
        source = "now(UnusedTester)!! Def testUnused(self | a) { a := 10; a + 20; 999; ^a; }!!"
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)

        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, 0]
        vm.run()

        assert vm.halted is True
        assert vm.local_vars[1] == 10
        assert vm.stack == [10]

    def test_cand_cor_short_circuiting(self):
        """cand and cor short-circuit evaluation without evaluating right-hand side when determined."""
        # 1. Truth table verification for cand
        for a, b, expected in [(1, 1, 1), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]:
            source = "Def testCand(self, a, b | res) { res := 0; if a > 0 cand b > 0 then res := 1; endif; ^res; }!!"
            insts, consts, strings = compile_to_bytecode(parse_actor(source))
            assert Opcode.DUP in [op for op, _ in insts]
            assert Opcode.POP_JUMP_IF_FALSE in [op for op, _ in insts]

            vm = ActorVM(ActorInheritanceHierarchy())
            vm.load(insts, consts, strings)
            vm.local_vars = [None, a, b, None]
            vm.run()
            assert vm.local_vars[3] == expected

        # 2. Truth table verification for cor
        for a, b, expected in [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 0)]:
            source = "Def testCor(self, a, b | res) { res := 0; if a > 0 cor b > 0 then res := 1; endif; ^res; }!!"
            insts, consts, strings = compile_to_bytecode(parse_actor(source))
            assert Opcode.DUP in [op for op, _ in insts]
            assert Opcode.POP_JUMP_IF_TRUE in [op for op, _ in insts]

            vm = ActorVM(ActorInheritanceHierarchy())
            vm.load(insts, consts, strings)
            vm.local_vars = [None, a, b, None]
            vm.run()
            assert vm.local_vars[3] == expected

        # 3. Short-circuit skips division-by-zero on false left operand for cand
        source_cand_div0 = "Def testCandDiv0(self, a | res) { res := 0; if a > 0 cand 10 / 0 > 0 then res := 1; endif; ^res; }!!"
        insts, consts, strings = compile_to_bytecode(parse_actor(source_cand_div0))
        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, -1, None]
        vm.run()  # Must not raise ZeroDivisionError
        assert vm.local_vars[2] == 0

        # 4. Short-circuit skips division-by-zero on true left operand for cor
        source_cor_div0 = "Def testCorDiv0(self, a | res) { res := 0; if a > 0 cor 10 / 0 > 0 then res := 1; endif; ^res; }!!"
        insts, consts, strings = compile_to_bytecode(parse_actor(source_cor_div0))
        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, 1, None]
        vm.run()  # Must not raise ZeroDivisionError
        assert vm.local_vars[2] == 1

    def test_binary_mod_arithmetic_execution(self):
        """Opcode.BINARY_MOD (0x5D) compiles for % and mod, and executes cleanly on VM."""
        assert Opcode.BINARY_MOD == 0x5D
        assert str(Opcode.BINARY_MOD) == "BINARY_MOD"

        # Operator %
        source_pct = "Def testModPct(self, a, b | res) { res := a % b; ^res; }!!"
        insts, consts, strings = compile_to_bytecode(parse_actor(source_pct))
        assert Opcode.BINARY_MOD in [op for op, _ in insts]

        vm = ActorVM(ActorInheritanceHierarchy())
        vm.load(insts, consts, strings)
        vm.local_vars = [None, 14, 5, None]
        vm.run()
        assert vm.local_vars[3] == 4

        # Keyword mod
        source_kw = "Def testModKw(self, a, b | res) { res := a mod b; ^res; }!!"
        insts2, consts2, strings2 = compile_to_bytecode(parse_actor(source_kw))
        assert Opcode.BINARY_MOD in [op for op, _ in insts2]

        vm2 = ActorVM(ActorInheritanceHierarchy())
        vm2.load(insts2, consts2, strings2)
        vm2.local_vars = [None, 27, 4, None]
        vm2.run()
        assert vm2.local_vars[3] == 3

        # Division by zero check
        vm_zero = ActorVM(ActorInheritanceHierarchy())
        vm_zero.load(insts, consts, strings)
        vm_zero.local_vars = [None, 10, 0, None]
        with pytest.raises(ZeroDivisionError):
            vm_zero.run()

    def test_call_early_execution_with_early_bound_descriptor(self):
        """_handle_call_early extracts attributes from EarlyBoundDescriptor and dispatches correctly."""
        from actorii.core import ActorClass
        hierarchy = ActorInheritanceHierarchy()
        obj_cls = ActorClass("Object")
        obj_cls.add_method("greet", lambda self, who: f"Hello, {who}!")
        hierarchy.register(obj_cls)

        desc = EarlyBoundDescriptor("Object", "greet", 1)
        vm = ActorVM(hierarchy)
        vm.constants = [desc]
        vm.stack = ["receiver_obj", "Actor"]
        vm._handle_call_early(0)

        assert vm.stack == ["Hello, Actor!"]

        # Dict descriptor backward compatibility
        desc_dict = {"class": "Object", "method": "greet", "args": 1}
        vm2 = ActorVM(hierarchy)
        vm2.constants = [desc_dict]
        vm2.stack = ["receiver_obj", "Whitewater"]
        vm2._handle_call_early(0)

        assert vm2.stack == ["Hello, Whitewater!"]

    def test_nested_block_parameter_slot_isolation_and_numbering(self):
        """Nested block scopes isolate parameters sequentially to slots 1..N without parent slot leakage."""
        source = """
        now(BlockParamTester)!!
        Def testBlockParams(self) {
            b := {using(x, y)
                {using(z, x)
                    z + x;
                };
            };
        }!!
        """
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)
        blocks = [c for c in consts if isinstance(c, BlockDescriptor)]

        inner_block = next((b for b in blocks if b.params == ["z", "x"]), None)
        assert inner_block is not None
        loads = [oper for op, oper in inner_block.instructions if op == Opcode.LOAD_LOCAL]
        # z is parameter 1 -> slot 1, x is parameter 2 -> slot 2
        assert loads == [1, 2]

    def test_block_descriptor_captured_vars_populated(self):
        """Scope.resolve registers captured variables into BlockDescriptor.captured_vars without duplicates."""
        source = """
        now(LexicalTester)!!
        Def testLexical(self | outerVar) {
            outerVar := 42;
            b := {using(elem)
                elem + outerVar + outerVar;
            };
        }!!
        """
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)
        blocks = [c for c in consts if isinstance(c, BlockDescriptor)]

        assert len(blocks) == 1
        b = blocks[0]
        assert len(b.captured_vars) == 1
        assert b.captured_vars[0] == ("outerVar", 1, 1)

    def test_unique_block_naming_nested_blocks(self):
        """Block counter threads across parent and child compilers, giving unique names to nested blocks."""
        source = """
        now(BlockTester)!!
        Def nestedBlocks(self) {
            b := {using(a)
                {using(b)
                    {using(c)
                        {using(d)
                            {using(e)
                                a + b + c + d + e;
                            };
                        };
                    };
                };
            };
            ^b;
        }!!
        """
        ast = parse_actor(source)
        insts, consts, strings = compile_to_bytecode(ast)
        blocks = [c for c in consts if isinstance(c, BlockDescriptor)]

        assert len(blocks) == 5
        block_names = [b.name for b in blocks]
        assert len(set(block_names)) == 5
        assert set(block_names) == {"Block_1", "Block_2", "Block_3", "Block_4", "Block_5"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
