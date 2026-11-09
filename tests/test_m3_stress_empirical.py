# -*- coding: utf-8 -*-
"""Empirical Stress Test Suite for Milestone M3 VM Execution Engine.

Author: Challenger 1 (Milestone M3 Gate)
Scope:
1. Deep call stack recursion (depth 50+) verifying stack unwinding & memory neutrality.
2. Deeply nested closures (3+ levels) reading and writing outer variables through Cell.
3. Non-local returns from deeply nested blocks and cross-method invocation frames.
4. Sieve benchmarks (SIEVE.ACT and SIEVE1.ACT) with varying N (10 -> 4, 100 -> 25, 200 -> 46, up to 1000 -> 168).
5. Fractal 3D-to-2D projection benchmarks.
"""

from pathlib import Path
import pytest
from actorii.vm import ActorVM, Point, Point3D

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_ACT_ROOT = REPO_ROOT / "historical_actor_41_source" / "actact"


class TestEmpiricalDeepRecursion:
    """Stress tests call stack depth up to depth 80 and verifies memory neutrality."""

    def test_factorial_depth_50_memory_neutrality(self):
        """Recursive factorial to depth 50 unwinds cleanly with empty stack."""
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
        res = vm.dispatch_message(None, "fact", [50])
        expected = 30414093201713378043612608166064768844377641568960512000000000000
        assert res == expected
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_fibonacci_depth_15(self):
        """Tree recursion verifies branching activation frames pop correctly."""
        vm = ActorVM()
        source = """
        Def fib(self, n | a b) {
            if n <= 1
            then
                ^n;
            else
                a := fib(self, n - 1);
                b := fib(self, n - 2);
                ^a + b;
            endif;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "fib", [15])
        assert res == 610
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_deep_linear_recursion_depth_80(self):
        """Linear recursion to depth 80 verifies deep activation records."""
        vm = ActorVM()
        source = """
        Def countDepth(self, depth, maxDepth | nextVal) {
            if depth >= maxDepth
            then
                ^depth;
            else
                nextVal := countDepth(self, depth + 1, maxDepth);
                ^nextVal;
            endif;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "countDepth", [0, 80])
        assert res == 80
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_recursion_100_iterations_leak_check(self):
        """100 repeated runs of depth 50 recursion exhibit strict memory neutrality."""
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
        for i in range(100):
            res = vm.dispatch_message(None, "fact", [50])
            assert len(vm.call_stack) <= 1, f"Call stack leak on iteration {i}"
            assert len(vm.stack) == 0, f"Operand stack leak on iteration {i}"


class TestEmpiricalNestedClosures:
    """Stress tests deeply nested closures (3+ levels) with Cell mutation."""

    def test_three_level_nested_closure_cell_mutation(self):
        """3 levels of nested closures mutate shared outer variable via Cell."""
        vm = ActorVM()
        source = """
        Def nested3Closures(self | a blk1 blk2 blk3 res) {
            a := 100;
            blk1 := {using(x)
                a := a + x;
                blk2 := {using(y)
                    a := a + y;
                    blk3 := {using(z)
                        a := a + z;
                        a;
                    };
                    eval(blk3, 3);
                };
                eval(blk2, 20);
            };
            res := eval(blk1, 1);
            ^res;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "nested3Closures")
        assert res == 124  # 100 + 1 + 20 + 3
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_multi_scope_cell_read_write(self):
        """Variables defined at distinct lexical levels are captured and modified correctly."""
        vm = ActorVM()
        source = """
        Def multiScopeClosure(self | a blk1 res) {
            a := 10;
            blk1 := {using(x | b blk2)
                b := 20;
                a := a + x;
                blk2 := {using(y | c blk3)
                    c := 30;
                    a := a + y;
                    b := b + y;
                    blk3 := {using(z)
                        a := a + z;
                        b := b + z;
                        c := c + z;
                        a + b + c;
                    };
                    eval(blk3, 1);
                };
                eval(blk2, 2);
            };
            res := eval(blk1, 3);
            ^res;
        }!!
        """
        vm.eval_chunk(source)
        # a = 10 + 3 + 2 + 1 = 16
        # b = 20 + 2 + 1 = 23
        # c = 30 + 1 = 31
        # sum = 16 + 23 + 31 = 70
        res = vm.dispatch_message(None, "multiScopeClosure")
        assert res == 70
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_three_level_nested_do_loops(self):
        """Triple nested do loop iterates 3 * 3 * 3 = 27 times mutating cell."""
        vm = ActorVM()
        source = """
        Def tripleLoopDo(self | total) {
            total := 0;
            do(over(1, 4),
               {using(i)
                   do(over(1, 4),
                      {using(j)
                          do(over(1, 4),
                             {using(k)
                                 total := total + 1;
                             });
                      });
               });
            ^total;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "tripleLoopDo")
        assert res == 27
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_four_level_nested_do_loops(self):
        """Quadruple nested do loop iterates 2 * 2 * 2 * 2 = 16 times."""
        vm = ActorVM()
        source = """
        Def quadLoopDo(self | total) {
            total := 0;
            do(over(1, 3),
               {using(i)
                   do(over(1, 3),
                      {using(j)
                          do(over(1, 3),
                             {using(k)
                                 do(over(1, 3),
                                    {using(l)
                                        total := total + 1;
                                    });
                             });
                      });
               });
            ^total;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "quadLoopDo")
        assert res == 16
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_sibling_closures_share_identical_cell(self):
        """Sibling closures created in the same frame share the identical Cell."""
        vm = ActorVM()
        source = """
        Def siblingClosures(self | counter incBlk getBlk) {
            counter := 10;
            incBlk := {using(x) counter := counter + x;};
            getBlk := {using(dummy) counter;};
            eval(incBlk, 5);
            eval(incBlk, 7);
            ^eval(getBlk, 0);
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "siblingClosures")
        assert res == 22
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_escaped_closure_cell_mutation_across_calls(self):
        """Escaped block closure retains Cell across multiple invocations."""
        vm = ActorVM()
        source = """
        Def makeCounter(self | count blk) {
            count := 0;
            blk := {using(step)
                count := count + step;
                count;
            };
            ^blk;
        }!!

        Def testCounter(self | counter c1 c2 c3) {
            counter := makeCounter(self);
            c1 := eval(counter, 5);
            c2 := eval(counter, 10);
            c3 := eval(counter, 1);
            ^c1 + c2 + c3;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "testCounter")
        # c1 = 5, c2 = 15, c3 = 16 -> sum = 36
        assert res == 36
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0


class TestEmpiricalNonLocalReturns:
    """Stress tests non-local caret (^) returns unwinding through nested contexts."""

    def test_three_level_nested_non_local_return(self):
        """Caret return inside 3-level nested loop unwinds all intermediate loops."""
        vm = ActorVM()
        source = """
        Def nestedReturn3(self | dummy) {
            do(over(1, 10),
               {using(i)
                   do(over(1, 10),
                      {using(j)
                          do(over(1, 10),
                             {using(k)
                                 if (i == 2 cand j == 3 cand k == 4)
                                 then
                                     ^(i * 100 + j * 10 + k);
                                 endif;
                             });
                      });
               });
            ^0;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "nestedReturn3")
        assert res == 234
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_four_level_nested_non_local_return(self):
        """Caret return inside 4-level nested loop unwinds cleanly."""
        vm = ActorVM()
        source = """
        Def nestedReturn4(self | dummy) {
            do(over(1, 5),
               {using(i)
                   do(over(1, 5),
                      {using(j)
                          do(over(1, 5),
                             {using(k)
                                 do(over(1, 5),
                                    {using(l)
                                        if (i == 3 cand j == 2 cand k == 4 cand l == 1)
                                        then
                                            ^9999;
                                        endif;
                                    });
                             });
                      });
               });
            ^0;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "nestedReturn4")
        assert res == 9999
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_cross_method_non_local_return_unwinding(self):
        """Non-local return unwinds across intermediate method activation records."""
        vm = ActorVM()
        source = """
        Def deepCallee(self, blk) {
            eval(blk, 0);
            ^"should_not_reach_here_deep";
        }!!

        Def invokeCallee(self, blk) {
            deepCallee(self, blk);
            ^"should_not_reach_here_invoke";
        }!!

        Def methodA(self | blk res) {
            blk := {using(x) ^999;};
            res := invokeCallee(self, blk);
            ^res;
        }!!

        Def topCaller(self | ans) {
            ans := methodA(self);
            ^ans + 1;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "topCaller")
        assert res == 1000
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_escaped_block_cannot_return_error(self):
        """Non-local return in escaped block raises CannotReturn RuntimeError."""
        vm = ActorVM()
        source = """
        Def createEscapedBlock(self | blk) {
            blk := {using(x) ^42;};
            ^blk;
        }!!

        Def testEscaped(self | escapedBlk res) {
            escapedBlk := createEscapedBlock(self);
            res := eval(escapedBlk, 0);
            ^res;
        }!!
        """
        vm.eval_chunk(source)
        with pytest.raises(RuntimeError, match="CannotReturn"):
            vm.dispatch_message(None, "testEscaped")


class TestEmpiricalSieveScaling:
    """Stress tests SIEVE.ACT and SIEVE1.ACT across varying input sizes N."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (10, 4),
            (25, 9),
            (50, 15),
            (100, 25),
            (200, 46),
            (300, 62),
            (500, 95),
            (1000, 168),
        ],
    )
    def test_sieve_act_varying_n(self, n, expected):
        """SIEVE.ACT produces authentic prime counts for varying N."""
        sieve_path = HISTORICAL_ACT_ROOT / "SIEVE.ACT"
        source = sieve_path.read_text(encoding="latin-1")
        vm = ActorVM()
        vm.eval_chunk(source)
        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None

        actual = vm.dispatch_message(Sam, "sieve", [n])
        assert actual == expected
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (10, 4),
            (25, 9),
            (50, 15),
            (100, 25),
            (200, 46),
            (300, 62),
            (500, 95),
            (1000, 168),
        ],
    )
    def test_sieve1_act_varying_n(self, n, expected):
        """SIEVE1.ACT (early-bound messages) produces authentic prime counts for varying N."""
        sieve1_path = HISTORICAL_ACT_ROOT / "SIEVE1.ACT"
        source = sieve1_path.read_text(encoding="latin-1")
        vm = ActorVM()
        vm.eval_chunk(source)
        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None

        actual = vm.dispatch_message(Sam, "sieve", [n])
        assert actual == expected
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0

    def test_sieve_act_perf_scale_n2000(self):
        """SIEVE.ACT executes N=2000 primes computation swiftly."""
        sieve_path = HISTORICAL_ACT_ROOT / "SIEVE.ACT"
        source = sieve_path.read_text(encoding="latin-1")
        vm = ActorVM()
        vm.eval_chunk(source)
        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")

        actual = vm.dispatch_message(Sam, "sieve", [2000])
        assert actual == 303  # 303 primes <= 2000
        assert len(vm.call_stack) <= 1
        assert len(vm.stack) == 0


class TestEmpiricalFractalScaling:
    """Stress tests FRACTAL.ACT 3D-to-2D coordinates and projection."""

    def test_fractal_act_multi_point_projection(self):
        """FRACTAL.ACT point3D projection across multiple coordinates."""
        fractal_path = HISTORICAL_ACT_ROOT / "FRACTAL.ACT"
        source = fractal_path.read_text(encoding="latin-1")
        vm = ActorVM()
        vm.eval_chunk(source)

        coords = [
            (256, 20, 0, 256, 20),
            (256, 20, 16, 256, 2),
            (100, 50, 8, 100, 41),
            (0, 0, 0, 0, 0),
        ]

        for x, y, z, exp_px, exp_py in coords:
            pt3d = vm.dispatch_message(x, "point3D", [y, z])
            assert isinstance(pt3d, Point3D)
            assert pt3d.x == x and pt3d.y == y and pt3d.z == z
            pt2d = pt3d.asPoint()
            assert isinstance(pt2d, Point)
            assert pt2d.x == exp_px and pt2d.y == exp_py
            assert len(vm.call_stack) <= 1
            assert len(vm.stack) == 0
