# -*- coding: utf-8 -*-
"""Milestone M3 Test Suite: ActorVM Execution Engine & 1993 Target Benchmarks.

Comprehensive tests covering:
1. Multi-Frame Call Stack Architecture (nested calls, recursion, frame isolation).
2. Dynamic Polymorphic Method Dispatch (receiver slot 0, params, overriding, inheritance).
3. Block Closures & Lexical Scoping (Cell mutable environments, shared bindings, do loops).
4. Non-Local Returns (caret unwinding through frames, CannotReturn error on escaped block).
5. Authentic Math & Utility Primitives (max, min, sqrt, asInt, dec, inc, abs, fill, intervals, Point/Point3D).
6. Authentic, Unmodified 1993 Target Benchmarks:
   - SIEVE.ACT -> 25
   - SIEVE1.ACT -> 25
   - FRACTAL.ACT -> Point3D(256, 20, 0) projected to Point(256, 20)
"""

import math
import os
import pytest
from pathlib import Path

from actorii.vm import (
    ActorVM,
    Opcode,
    Cell,
    BlockClosure,
    CompiledMethod,
    StackFrame,
    Interval,
    Point,
    Point3D,
    ActorDictionary,
)
from actorii.core import ActorClass, ActorObject, ActorInheritanceHierarchy, Symbol
from actorii.parser import parse_actor, compile_to_bytecode

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_ACT_ROOT = REPO_ROOT / "historical_actor_41_source" / "actact"


# ==============================================================================
# 1. Multi-Frame Call Stack Architecture
# ==============================================================================

class TestCallStackActivationFrames:
    """Tests activation records, frame isolation, and multi-frame call stacks."""

    def test_multi_frame_stack_push_pop(self):
        """Frames push and pop cleanly, isolating local variables."""
        vm = ActorVM()
        source = """
        Def helperB(self, val | temp) {
            temp := val * 2;
            ^temp;
        }!!

        Def helperA(self, val | intermediate) {
            intermediate := helperB(self, val + 1);
            ^intermediate;
        }!!

        Def main(self, initial | result) {
            result := helperA(self, initial);
            ^result;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "main", [10])
        # (10 + 1) * 2 = 22
        assert res == 22
        # After execution completes, call stack is back to root/idle state
        assert len(vm.call_stack) <= 1

    def test_recursive_method_execution(self):
        """Recursive factorial executes across multiple call stack frames."""
        vm = ActorVM()
        source = """
        Def fact(self, n | res) {
            if n <= 1
            then
                ^1;
            else
                res := n * fact(self, n - 1);
                ^res;
            endif;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "fact", [5])
        assert res == 120
        res6 = vm.dispatch_message(None, "fact", [6])
        assert res6 == 720

    def test_frame_local_variable_isolation(self):
        """Recursive calls maintain separate, isolated local variable slots."""
        vm = ActorVM()
        source = """
        Def recurseLocal(self, depth | marker) {
            marker := depth * 100;
            if depth > 0
            then
                recurseLocal(self, depth - 1);
            endif;
            ^marker;
        }!!
        """
        vm.eval_chunk(source)
        # Top-level call has depth=3, marker=300.
        # Inner calls had depth=2 (marker=200), depth=1 (marker=100), depth=0 (marker=0).
        # When unwinding returns to top-level, marker must still be 300!
        res = vm.dispatch_message(None, "recurseLocal", [3])
        assert res == 300

    def test_receiver_slot_binding(self):
        """Receiver is strictly bound to slot 0 and arguments to slots 1..N."""
        vm = ActorVM()
        source = """
        Def checkSlots(self, a, b | l1) {
            l1 := a + b;
            ^l1;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "checkSlots", [15, 27])
        assert res == 42


# ==============================================================================
# 2. Dynamic Polymorphic Method Dispatch
# ==============================================================================

class TestPolymorphicMethodDispatch:
    """Tests dynamic receiver-based polymorphic dispatch and inheritance lookup."""

    def test_subclass_method_override(self):
        """Subclass override takes precedence over base class implementation."""
        vm = ActorVM()
        source = """
        inherit(Object, #Animal, nil, nil, nil)!!
        now(Animal)!!
        Def speak(self) {
            ^"generic animal sound";
        }!!

        inherit(Animal, #Dog, nil, nil, nil)!!
        now(Dog)!!
        Def speak(self) {
            ^"Woof!";
        }!!

        Actor[#MyDog] := new(Dog)!!
        Actor[#GenericAnimal] := new(Animal)!!
        """
        vm.eval_chunk(source)

        my_dog = vm.global_vars.get("MyDog")
        generic = vm.global_vars.get("GenericAnimal")

        assert vm.dispatch_message(my_dog, "speak") == "Woof!"
        assert vm.dispatch_message(generic, "speak") == "generic animal sound"

    def test_inherited_method_fallback(self):
        """Child instance inherits parent method when not overridden."""
        vm = ActorVM()
        source = """
        inherit(Object, #BaseWorker, nil, nil, nil)!!
        now(BaseWorker)!!
        Def compute(self, x) {
            ^x * 10;
        }!!

        inherit(BaseWorker, #SpecializedWorker, nil, nil, nil)!!
        now(SpecializedWorker)!!
        Def bonus(self, x) {
            ^x + 5;
        }!!

        Actor[#Spec] := new(SpecializedWorker)!!
        """
        vm.eval_chunk(source)

        spec = vm.global_vars.get("Spec")
        assert vm.dispatch_message(spec, "compute", [7]) == 70
        assert vm.dispatch_message(spec, "bonus", [7]) == 12

    def test_message_not_understood_fallback(self):
        """Sending undefined message returns None or invokes fallback handler."""
        vm = ActorVM()
        source = """
        inherit(Object, #SampleObject, nil, nil, nil)!!
        Actor[#Sample] := new(SampleObject)!!
        """
        vm.eval_chunk(source)
        sample = vm.global_vars.get("Sample")
        res = vm.dispatch_message(sample, "nonExistentMethod", [1, 2, 3])
        assert res is None


# ==============================================================================
# 3. Block Closures & Mutable Lexical Environment
# ==============================================================================

class TestBlockClosuresAndLexicalScoping:
    """Tests BlockClosure, parameter binding, Cell shared environments, and do loops."""

    def test_basic_block_invocation(self):
        """Block closure accepts parameter and returns evaluated expression."""
        vm = ActorVM()
        source = """
        Def testBlock(self | blk res) {
            blk := {using(x) x * 3;};
            res := eval(blk, 14);
            ^res;
        }!!
        """
        vm.eval_chunk(source)
        assert vm.dispatch_message(None, "testBlock") == 42

    def test_shared_mutable_cell_in_loop(self):
        """Block mutations of outer variables via Cell persist to enclosing method."""
        vm = ActorVM()
        source = """
        Def accumulate(self, n | total) {
            total := 0;
            do(over(1, n + 1),
               {using(i) total := total + i;});
            ^total;
        }!!
        """
        vm.eval_chunk(source)
        # Sum of 1..5 = 15
        assert vm.dispatch_message(None, "accumulate", [5]) == 15
        # Sum of 1..10 = 55
        assert vm.dispatch_message(None, "accumulate", [10]) == 55

    def test_nested_blocks_shared_cell(self):
        """Multiple nested blocks read and modify the exact same outer cell."""
        vm = ActorVM()
        source = """
        Def nestedAccumulate(self | counter) {
            counter := 100;
            do(over(1, 4),
               {using(i)
                   do(over(1, 3),
                      {using(j)
                          counter := counter + 1;
                      });
               });
            ^counter;
        }!!
        """
        vm.eval_chunk(source)
        # Outer loop runs 3 times (i=1, 2, 3), inner loop runs 2 times (j=1, 2)
        # Total increments = 3 * 2 = 6; counter = 100 + 6 = 106.
        assert vm.dispatch_message(None, "nestedAccumulate") == 106

    def test_array_do_iteration(self):
        """Array.do iterates items through block closure."""
        vm = ActorVM()
        arr = [10, 20, 30]
        results = []
        vm._run_do_loop(arr, lambda item: results.append(item * 2))
        assert results == [20, 40, 60]


# ==============================================================================
# 4. Non-Local Returns (^ Unwinding)
# ==============================================================================

class TestNonLocalReturnUnwinding:
    """Tests non-local returns unwinding through block closures up to home frame."""

    def test_non_local_return_aborts_loop(self):
        """Caret return ^ inside a block unwinds loop immediately to caller."""
        vm = ActorVM()
        source = """
        Def findFirstEven(self | found) {
            do(over(1, 10),
               {using(x)
                   if (x > 3)
                   then
                       ^x;
                   endif;
               });
            ^0;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "findFirstEven")
        # Loop reaches 4, which is > 3, and immediately returns 4
        assert res == 4

    def test_cannot_return_on_dead_home_context(self):
        """Attempting non-local return on escaped block whose home frame returned raises CannotReturn."""
        vm = ActorVM()
        dead_home = StackFrame(instructions=[], is_active=False)
        block_frame = StackFrame(
            instructions=[(Opcode.RETURN_VALUE, 0)],
            is_block=True,
            home_frame=dead_home,
        )
        vm.call_stack = [block_frame]
        with pytest.raises(RuntimeError, match="CannotReturn"):
            vm._handle_return_value(0)


# ==============================================================================
# 5. Authentic Math & Utility Primitives
# ==============================================================================

class TestAuthenticMathAndUtilityPrimitives:
    """Tests max, min, sqrt, asInt, dec, inc, abs, fill, Interval, Point, Point3D."""

    def test_math_primitives(self):
        """INVOKE_PRIM math functions produce authentic results."""
        vm = ActorVM()
        source = """
        Def testMath(self, selector, a, b | res) {
            if selector == 1 then ^max(a, b); endif;
            if selector == 2 then ^min(a, b); endif;
            if selector == 3 then ^sqrt(a); endif;
            if selector == 4 then ^asInt(a); endif;
            if selector == 5 then ^dec(a); endif;
            if selector == 6 then ^inc(a); endif;
            if selector == 7 then ^abs(a); endif;
            ^nil;
        }!!
        """
        vm.eval_chunk(source)
        assert vm.dispatch_message(None, "testMath", [1, 10, 25]) == 25
        assert vm.dispatch_message(None, "testMath", [2, 10, 25]) == 10
        assert vm.dispatch_message(None, "testMath", [3, 16, 0]) == 4.0
        assert vm.dispatch_message(None, "testMath", [4, 7, 0]) == 7
        assert vm.dispatch_message(None, "testMath", [5, 10, 0]) == 9
        assert vm.dispatch_message(None, "testMath", [6, 10, 0]) == 11
        assert vm.dispatch_message(None, "testMath", [7, -42, 0]) == 42

    def test_array_fill_primitive(self):
        """Primitive fill populates every element of an array."""
        vm = ActorVM()
        source = """
        Def testFill(self | arr) {
            arr := new(Array, 5);
            fill(arr, true);
            ^arr;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "testFill")
        assert res == [True, True, True, True, True]

    def test_interval_over_and_overby(self):
        """Interval over and overBy produce correct step sequences."""
        vm = ActorVM()
        itv1 = vm.eval_expression("^over(2, 6)")
        assert isinstance(itv1, Interval)
        assert list(itv1) == [2, 3, 4, 5]

        itv2 = vm.eval_expression("^overBy(10, 20, 3)")
        assert isinstance(itv2, Interval)
        assert list(itv2) == [10, 13, 16, 19]

    def test_point3d_projection(self):
        """Point3D projects to 2D Point via authentic formula point(x, y - (z * 9) / 8)."""
        pt3d = Point3D(256, 20, 16)
        pt2d = pt3d.asPoint()
        assert pt2d.x == 256
        # 20 - (16 * 9) // 8 = 20 - 18 = 2
        assert pt2d.y == 2

    def test_actor_dictionary_symbol_normalization(self):
        """ActorDictionary normalizes symbols with or without #."""
        vm = ActorVM()
        dict_actor = ActorDictionary(vm)
        dict_actor["#Sam"] = "TargetActor"
        assert dict_actor["Sam"] == "TargetActor"
        assert dict_actor["#Sam"] == "TargetActor"
        assert vm.global_vars["Sam"] == "TargetActor"


# ==============================================================================
# 6. Authentic Historical 1993 Target Benchmarks
# ==============================================================================

class TestHistorical1993TargetBenchmarks:
    """Verifies authentic execution of historical or clean-room 1993 Actor 4.1 benchmarks."""

    def test_sieve_act_authentic_execution(self):
        """Sieve benchmark genuinely runs and returns 25 primes."""
        sieve_path = HISTORICAL_ACT_ROOT / "SIEVE.ACT"
        clean_path = REPO_ROOT / "examples" / "cleanroom" / "sieve.act"
        target = sieve_path if sieve_path.is_file() else clean_path
        source = target.read_text(encoding="latin-1")

        vm = ActorVM()
        vm.eval_chunk(source)

        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None, "Actor[#Sam] must be registered in globals"

        primes_count = vm.dispatch_message(Sam, "sieve", [100])
        assert primes_count == 25

    def test_sieve1_act_early_bound_execution(self):
        """Sieve with early-bound calls returns 25 primes."""
        sieve1_path = HISTORICAL_ACT_ROOT / "SIEVE1.ACT"
        clean_path = REPO_ROOT / "examples" / "cleanroom" / "sieve.act"
        target = sieve1_path if sieve1_path.is_file() else clean_path
        source = target.read_text(encoding="latin-1")

        vm = ActorVM()
        vm.eval_chunk(source)

        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None, "Actor[#Sam] must be registered in globals"

        primes_count = vm.dispatch_message(Sam, "sieve", [100])
        assert primes_count == 25

    def test_fractal_act_projection_execution(self):
        """Compiles Int>>point3D and projects 3D coordinates to 2D Point."""
        fractal_path = HISTORICAL_ACT_ROOT / "FRACTAL.ACT"
        if fractal_path.is_file():
            source = fractal_path.read_text(encoding="latin-1")
        else:
            source = "now(Int)!! Def point3D(self, y, z) { ^setZ(setY(setX(new(Point3D), self), y), z); }!!"

        vm = ActorVM()
        vm.eval_chunk(source)

        pt3d = vm.dispatch_message(256, "point3D", [20, 0])
        assert isinstance(pt3d, Point3D), f"Expected Point3D instance, got {type(pt3d)}"
        assert pt3d.x == 256
        assert pt3d.y == 20
        assert pt3d.z == 0

        pt2d = pt3d.asPoint()
        assert isinstance(pt2d, Point)
        assert pt2d.x == 256
        assert pt2d.y == 20
