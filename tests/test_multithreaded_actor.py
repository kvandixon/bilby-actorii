"""Multithreaded ActorII test suite (opt-in fibers + `Actor[#Threads]` primitives).

Covers:
- Legacy single-threaded semantics unchanged (primary fiber).
- Concurrent `eval_chunk` / `dispatch_message` stress from raw threads (auto-fibers).
- `TaskPool` / `ActorThreadPoolTwin` futures: values, errors, timeouts, gas limits, cancel.
- `parallelAll` wall-clock speedup + deterministic ordering, `parallelSettle`, `parallelMap`.
- `Channel` producer/consumer, `Mutex`, `AtomicCounter` race-freedom.
- Shadow-run inheritance (Tier 2 side-effects stay blocked in spawned tasks).
- Nested fan-out deadlock avoidance, concurrent class definition, shutdown.
"""

from __future__ import annotations

import threading
import time

import pytest

from actorii.bridge import ActorBridgeObject, ActorBridgeSecurityError, actor_bridge, actor_command, actor_query
from actorii.classes.concurrency import (
    ActorChannel,
    ActorChannelClosedError,
    ActorFuture,
    ActorMutex,
    ActorTaskCancelledError,
    ActorTaskTimeoutError,
    ActorThreadPoolTwin,
    AtomicCounter,
    TaskPool,
    THREADS_TWIN_SELECTORS,
)
from actorii.guardrails.linter import OpcodeBudgetExceededError
from actorii.vm import ActorVM, ExecutionFiber

SUM_TO = (
    "Def sumTo(self, n | i, s) { i := 0; s := 0; "
    "loop while i < n begin i := i + 1; s := s + i; endLoop; ^s; }!!"
)
FIB = "Def fib(self) { if self < 2 then ^self; endif; ^fib(self - 1) + fib(self - 2); }!!"
SPIN = "loop while true begin 1 + 1; endLoop!!"


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@pytest.fixture
def vm():
    v = ActorVM()
    v.eval_chunk(SUM_TO)
    v.eval_chunk(FIB)
    yield v


@pytest.fixture
def threads_vm(vm):
    twin = ActorThreadPoolTwin(vm, max_workers=8)
    vm.expose_object(twin, global_key="Threads")
    yield vm, twin
    twin.pool.shutdown(wait=True)


# =============================================================================
# 1. Fiber architecture & backward compatibility
# =============================================================================


class TestFiberArchitecture:
    def test_primary_fiber_is_default_and_legacy_attrs_work(self, vm):
        assert vm.current_fiber is vm.primary_fiber
        assert isinstance(vm.primary_fiber, ExecutionFiber)
        assert not vm.in_fiber()
        assert vm.eval_chunk("3 + 4!!") == 7
        vm.max_opcodes = 1234
        assert vm.primary_fiber.max_opcodes == 1234
        vm.halted = True
        assert vm.halted is True and vm.primary_fiber.halted is True

    def test_fiber_isolates_execution_state(self, vm):
        vm.eval_chunk("1 + 1!!")
        outer_stack = vm.stack
        outer_budget = vm.max_opcodes
        with vm.fiber(max_opcodes=777, name="t") as fib:
            assert vm.in_fiber()
            assert vm.current_fiber is fib
            assert vm.max_opcodes == 777
            assert vm.call_stack == []
            assert vm.eval_chunk("6 * 7!!") == 42
        assert not vm.in_fiber()
        assert vm.max_opcodes == outer_budget
        assert vm.stack is outer_stack

    def test_fibers_nest_and_restore(self, vm):
        with vm.fiber(name="a") as a:
            with vm.fiber(name="b") as b:
                assert vm.current_fiber is b
            assert vm.current_fiber is a
        assert vm.current_fiber is vm.primary_fiber
        assert vm.active_fiber_count == 0

    def test_isolated_gas_budget_does_not_leak(self, vm):
        with pytest.raises(OpcodeBudgetExceededError):
            vm.eval_isolated(SPIN, max_opcodes=500)
        assert vm.max_opcodes == 500_000
        assert vm.eval_chunk("sumTo(0, 100)!!") == 5050

    def test_sequential_cross_thread_handoff_uses_primary(self, vm):
        """Legacy pattern (e.g. Telegram worker threads): sequential use from another thread."""
        out = {}

        def worker():
            out["r"] = vm.eval_chunk("legacyGlobal := 99!! legacyGlobal + 1!!")
            out["fiber"] = vm.current_fiber

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert out["r"] == 100
        assert out["fiber"] is vm.primary_fiber
        assert vm.eval_chunk("legacyGlobal!!") == 99


# =============================================================================
# 2. Concurrent eval_chunk / dispatch_message stress (raw threads, auto-fibers)
# =============================================================================


class TestConcurrentStress:
    def test_16_concurrent_eval_chunk_no_stack_corruption(self, vm):
        n_threads = 16
        rounds = 3
        barrier = threading.Barrier(n_threads)
        errors: list = []
        results: dict = {}

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                for r in range(rounds):
                    n = 200 + idx * 37 + r
                    got = vm.eval_chunk(f"sumTo(0, {n})!!")
                    results[(idx, r)] = (got, n * (n + 1) // 2)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        assert not errors, errors
        assert len(results) == n_threads * rounds
        for key, (got, want) in results.items():
            assert got == want, (key, got, want)
        assert vm.active_fiber_count == 0
        assert vm.eval_chunk("sumTo(0, 10)!!") == 55

    def test_concurrent_dispatch_message(self, vm):
        errors: list = []
        out: dict = {}
        barrier = threading.Barrier(12)

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=10)
                out[i] = vm.dispatch_message(i + 5, "fib", [])
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        assert not errors, errors
        assert out == {i: _fib(i + 5) for i in range(12)}

    def test_16_concurrent_pool_tasks(self, vm):
        with TaskPool(vm, max_workers=16) as pool:
            futs = [pool.spawn_chunk(f"sumTo(0, {300 + i})!!") for i in range(16)]
            vals = [f.result(30) for f in futs]
        assert vals == [(300 + i) * (301 + i) // 2 for i in range(16)]
        assert vm.concurrency_stats["fibers_created"] >= 16

    def test_concurrent_method_definition_and_calls(self, vm):
        errors: list = []

        def worker(i: int) -> None:
            try:
                vm.eval_chunk(f"Def times{i}(self) {{ ^self * {i}; }}!!")
                assert vm.eval_chunk(f"times{i}(3)!!") == 3 * i
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(1, 13)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        assert not errors, errors
        for i in range(1, 13):
            assert vm.eval_chunk(f"times{i}(2)!!") == 2 * i

    def test_concurrent_global_stores_are_consistent(self, vm):
        def worker(i: int) -> None:
            vm.eval_chunk(f"slot{i} := {i * 10}!!")

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        for i in range(16):
            assert vm.global_vars[f"slot{i}"] == i * 10
            assert vm.global_vars["Actor"][f"slot{i}"] == i * 10


# =============================================================================
# 3. Futures: values, errors, timeouts, gas limits, cancellation
# =============================================================================


class TestFutures:
    def test_future_value_and_status(self, vm):
        with TaskPool(vm, max_workers=2) as pool:
            fut = pool.spawn_chunk("fib(12)!!")
            assert fut.result(10) == 144
            assert fut.status() == "#completed"
            assert fut.isDone() is True
            assert fut.error() is None
            assert fut.thread_name.startswith("actorii-task")

    def test_future_error_propagation(self, vm):
        with TaskPool(vm, max_workers=2) as pool:
            fut = pool.spawn_chunk("1 / 0!!")
            with pytest.raises(ZeroDivisionError):
                fut.result(10)
            assert fut.status() == "#failed"
            assert "ZeroDivisionError" in fut.error()

    def test_future_timeout(self, vm):
        with TaskPool(vm, max_workers=1) as pool:
            fut = pool.submit(lambda: time.sleep(0.3) or "late", label="slow")
            t0 = time.perf_counter()
            with pytest.raises(ActorTaskTimeoutError):
                fut.result(0.02)
            assert time.perf_counter() - t0 < 0.2
            assert fut.status() == "#pending"
            assert fut.result(5) == "late"

    def test_gas_limit_propagates_per_task(self, vm):
        with TaskPool(vm, max_workers=4) as pool:
            spin = pool.spawn_chunk(SPIN, gas=2_000)
            ok = pool.spawn_chunk("sumTo(0, 500)!!", gas=200_000)
            with pytest.raises(OpcodeBudgetExceededError):
                spin.result(10)
            assert ok.result(10) == 125250
            assert spin.status() == "#failed"
        assert vm.max_opcodes == 500_000

    def test_cancel_queued_task(self, vm):
        gate = threading.Event()
        with TaskPool(vm, max_workers=1) as pool:
            blocker = pool.submit(lambda: gate.wait(5), label="blocker")
            queued = pool.spawn_chunk("1 + 1!!")
            assert queued.request_cancel() is True
            gate.set()
            blocker.result(5)
            with pytest.raises(ActorTaskCancelledError):
                queued.result(5)
            assert queued.status() == "#cancelled"

    def test_cooperative_cancel_of_running_actor_loop(self, vm):
        with TaskPool(vm, max_workers=1) as pool:
            fut = pool.spawn_chunk(SPIN, gas=10_000_000)
            deadline = time.time() + 5
            while not fut.isRunning() and time.time() < deadline:
                time.sleep(0.005)
            time.sleep(0.02)
            assert fut.request_cancel() is True
            with pytest.raises(ActorTaskCancelledError):
                fut.result(5)
            assert fut.status() == "#cancelled"


# =============================================================================
# 4. Actor[#Threads] twin from Actor 4.1 code
# =============================================================================


class TestThreadsTwinActorCode:
    def test_spawn_chunk_and_await(self, threads_vm):
        vm, _ = threads_vm
        assert vm.eval_chunk('f := spawnChunk(Actor[#Threads], "fib(15)!!", 200000)!! await(f, 5000)!!') == 610
        assert vm.eval_chunk("status(f)!!") == "#completed"
        assert vm.eval_chunk("isDone(f)!!") is True
        assert str(vm.eval_chunk("threadName(f)!!")).startswith("actorii-threads")

    def test_spawn_message_send(self, threads_vm):
        vm, _ = threads_vm
        assert vm.eval_chunk("g := spawn(Actor[#Threads], 10, #fib, #())!! await(g, 5000)!!") == 55
        assert vm.eval_chunk("h := spawn(Actor[#Threads], 0, #sumTo, #(100))!! await(h, 5000)!!") == 5050

    def test_future_error_and_gas_from_actor(self, threads_vm):
        vm, _ = threads_vm
        res = vm.eval_chunk(f'h := spawnChunk(Actor[#Threads], "{SPIN}", 1000)!! awaitOr(h, 5000, #gasOut)!!')
        assert res == "#gasOut"
        assert vm.eval_chunk("status(h)!!") == "#failed"
        assert "OpcodeBudgetExceededError" in vm.eval_chunk("error(h)!!")
        with pytest.raises(ZeroDivisionError):
            vm.eval_chunk('await(spawnChunk(Actor[#Threads], "1 / 0!!"), 2000)!!')

    def test_await_timeout_from_actor(self, threads_vm):
        vm, _ = threads_vm
        with pytest.raises(ActorTaskTimeoutError):
            vm.eval_chunk('t := spawnChunk(Actor[#Threads], "sleepMs(Actor[#Threads], 300)!!")!! await(t, 20)!!')
        assert vm.eval_chunk("await(t, 5000)!!") == 300

    def test_cancel_from_actor(self, threads_vm):
        vm, _ = threads_vm
        vm.eval_chunk(f'k := spawnChunk(Actor[#Threads], "{SPIN}", 50000000)!!')
        vm.eval_chunk("sleepMs(Actor[#Threads], 30)!!")
        assert vm.eval_chunk("cancel(k)!!") is True
        vm.eval_chunk("awaitOr(k, 3000, nil)!!")
        assert vm.eval_chunk("status(k)!!") == "#cancelled"

    def test_parallel_all_order_and_values(self, threads_vm):
        vm, _ = threads_vm
        rs = vm.eval_chunk('parallelAll(Actor[#Threads], #("1 + 1!!" "2 * 21!!" "fib(10)!!"), 3000)!!')
        assert isinstance(rs, ActorBridgeObject) is False
        assert vm.eval_chunk("size(" "parallelAll(Actor[#Threads], #(\"1!!\" \"2!!\"), 1000))!!") == 2
        assert list(rs) == [2, 42, 55]

    def test_parallel_all_speedup_4x50ms(self, threads_vm):
        vm, _ = threads_vm
        src = (
            "parallelAll(Actor[#Threads], #("
            + " ".join(['"sleepMs(Actor[#Threads], 50)!!"'] * 4)
            + "), 2000)!!"
        )
        vm.eval_chunk(src)  # warm the worker threads
        t0 = time.perf_counter()
        rs = vm.eval_chunk(src)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert list(rs) == [50, 50, 50, 50]
        assert elapsed_ms < 110.0, f"parallelAll took {elapsed_ms:.1f}ms (serial would be >=200ms)"

    def test_parallel_all_deterministic_order_with_skewed_latency(self, threads_vm):
        vm, _ = threads_vm
        rs = vm.eval_chunk(
            'parallelAll(Actor[#Threads], #("sleepMs(Actor[#Threads], 60)!!" '
            '"sleepMs(Actor[#Threads], 30)!!" "sleepMs(Actor[#Threads], 1)!!"), 3000)!!'
        )
        assert list(rs) == [60, 30, 1]

    def test_parallel_all_timeout_raises(self, threads_vm):
        vm, _ = threads_vm
        t0 = time.perf_counter()
        with pytest.raises(ActorTaskTimeoutError):
            vm.eval_chunk('parallelAll(Actor[#Threads], #("sleepMs(Actor[#Threads], 400)!!" "1!!"), 50)!!')
        assert time.perf_counter() - t0 < 0.3

    def test_parallel_settle_never_raises(self, threads_vm):
        vm, _ = threads_vm
        rs = list(vm.eval_chunk('parallelSettle(Actor[#Threads], #("7!!" "1 / 0!!"), 2000)!!'))
        assert list(rs[0]) == ["#completed", 7, None]
        assert rs[1][0] == "#failed" and "ZeroDivisionError" in rs[1][2]

    def test_parallel_map(self, threads_vm):
        vm, _ = threads_vm
        assert list(vm.eval_chunk("parallelMap(Actor[#Threads], #(5 6 7), nil, #fib)!!")) == [5, 8, 13]
        assert list(vm.eval_chunk("parallelMap(Actor[#Threads], #(10 20), 0, #sumTo)!!")) == [55, 210]
        assert list(vm.eval_chunk("parallelMap(Actor[#Threads], #(1 2 3), {using(x) x * x})!!")) == [1, 4, 9]

    def test_spawn_block(self, threads_vm):
        vm, _ = threads_vm
        assert vm.eval_chunk("b := spawnBlock(Actor[#Threads], {using(x) x * 3}, #(14))!! await(b, 2000)!!") == 42

    def test_channel_mutex_counter_from_actor(self, threads_vm):
        vm, _ = threads_vm
        assert vm.eval_chunk("ch := newChannel(Actor[#Threads], 4)!! send(ch, 42)!! size(ch)!!") == 1
        assert vm.eval_chunk("receive(ch, 100)!!") == 42
        assert vm.eval_chunk("receive(ch, 10)!!") is None
        assert vm.eval_chunk("tryReceive(ch)!!") is None
        assert vm.eval_chunk("m := newMutex(Actor[#Threads])!! withLock(m, {1 + 41})!!") == 42
        assert vm.eval_chunk("isLocked(m)!!") is False
        tasks = " ".join(['"increment(c)!!"'] * 24)
        assert vm.eval_chunk(f"c := newCounter(Actor[#Threads], 0)!! parallelAll(Actor[#Threads], #({tasks}), 5000)!! get(c)!!") == 24

    def test_new_from_class_constructors(self, threads_vm):
        vm, _ = threads_vm
        assert vm.eval_chunk("x := new(Channel, 2)!! send(x, 7)!! tryReceive(x)!!") == 7
        assert vm.eval_chunk("ac := new(AtomicCounter, 5)!! add(ac, 10)!!") == 15
        assert vm.eval_chunk('p := new(ThreadPool, 2)!! await(spawnChunk(p, "40 + 2!!"), 2000)!!') == 42
        vm.eval_chunk("shutdown(p)!!")

    def test_summary_and_stats(self, threads_vm):
        vm, twin = threads_vm
        vm.eval_chunk('parallelAll(Actor[#Threads], #("1!!" "2!!"), 1000)!!')
        assert "ThreadPool(max=8" in vm.eval_chunk("summary(Actor[#Threads])!!")
        st = twin.stats()
        assert st["submitted"] >= 2 and st["maxWorkers"] == 8
        assert vm.eval_chunk("maxWorkers(Actor[#Threads])!!") == 8


# =============================================================================
# 5. Channels, mutexes, counters (Python API, real contention)
# =============================================================================


class TestSyncPrimitives:
    def test_channel_producer_consumer(self, vm):
        ch = ActorChannel(8)
        n_prod, per = 4, 250
        consumed: list = []
        lock = threading.Lock()

        def producer(p: int) -> None:
            for i in range(per):
                assert ch.send(p * 10_000 + i, 5000) is True

        def consumer() -> None:
            while True:
                v = ch.receive(2000)
                if v is None:
                    return
                with lock:
                    consumed.append(v)

        with TaskPool(vm, max_workers=6) as pool:
            cons = [pool.submit(consumer, label="consumer", isolate=False) for _ in range(2)]
            prods = [pool.submit(lambda p=p: producer(p), label="producer", isolate=False) for p in range(n_prod)]
            for f in prods:
                f.result(30)
            ch.close()
            for f in cons:
                f.result(30)
        assert len(consumed) == n_prod * per
        assert sorted(consumed) == sorted(p * 10_000 + i for p in range(n_prod) for i in range(per))
        assert ch.size() == 0

    def test_channel_bounded_and_closed(self):
        ch = ActorChannel(1)
        assert ch.send("a", 10) is True
        t0 = time.perf_counter()
        assert ch.send("b", 30) is False  # full -> timeout
        assert time.perf_counter() - t0 >= 0.025
        ch.close()
        assert ch.isClosed()
        with pytest.raises(ActorChannelClosedError):
            ch.send("c", 10)
        assert ch.receive(10) == "a"  # drains after close
        assert ch.receive(10) is None

    def test_atomic_counter_race_free(self, vm):
        c = AtomicCounter(0)
        with TaskPool(vm, max_workers=8) as pool:
            futs = [pool.submit(lambda: [c.increment() for _ in range(2000)], isolate=False) for _ in range(8)]
            for f in futs:
                f.result(30)
        assert c.get() == 16_000
        assert c.compareAndSet(16_000, 1) is True and c.get() == 1
        assert c.compareAndSet(99, 2) is False

    def test_mutex_protects_actor_global_read_modify_write(self, threads_vm):
        vm, _ = threads_vm
        vm.eval_chunk("total := 0!! mtx := newMutex(Actor[#Threads])!!")
        task = '"withLock(mtx, {total := total + 1})!!"'
        tasks = " ".join([task] * 40)
        vm.eval_chunk(f"parallelAll(Actor[#Threads], #({tasks}), 10000)!!")
        assert vm.eval_chunk("total!!") == 40

    def test_mutex_lock_unlock_timeout(self):
        m = ActorMutex()
        assert m.lock() is True
        got = {}

        def other():
            got["try"] = m.tryLock()
            got["timed"] = m.lock(20)

        t = threading.Thread(target=other)
        t.start()
        t.join(5)
        assert got == {"try": False, "timed": False}
        assert m.unlock() is True
        assert m.isLocked() is False


# =============================================================================
# 6. Safety: shadow-run inheritance, nested fan-out, shutdown, contract
# =============================================================================


@actor_bridge(name="Relay")
class _Relay:
    def __init__(self) -> None:
        self.fired = 0

    @actor_command
    def fire(self) -> int:
        self.fired += 1
        return self.fired

    @actor_query
    def count(self) -> int:
        return self.fired


class TestSafety:
    def test_shadow_run_blocks_tier2_inside_spawned_tasks(self, threads_vm):
        vm, _ = threads_vm
        relay = _Relay()
        vm.expose_object(relay, global_key="Relay")
        res = vm.eval_transactional(
            'f := spawnChunk(Actor[#Threads], "fire(Actor[#Relay])!!")!! awaitOr(f, 3000, #blocked)!!',
            shadow_only=True,
        )
        assert res == "#blocked"
        assert relay.fired == 0
        # Outside a shadow run the same task is allowed.
        assert vm.eval_chunk('await(spawnChunk(Actor[#Threads], "fire(Actor[#Relay])!!"), 3000)!!') == 1

    def test_shadow_flag_captured_python_api(self, vm):
        with TaskPool(vm, max_workers=1) as pool:
            vm._in_shadow_run = True
            try:
                fut = pool.submit(lambda: vm._in_shadow_run, label="probe")
            finally:
                vm._in_shadow_run = False
            assert fut.result(5) is True

    def test_nested_fan_out_does_not_deadlock(self, vm):
        twin = ActorThreadPoolTwin(vm, max_workers=2)
        vm.expose_object(twin, global_key="Threads")
        try:
            inner = '"parallelAll(Actor[#Threads], #(\\"1!!\\" \\"2!!\\" \\"3!!\\" \\"4!!\\"), 5000)!!"'
            fut = twin.pool.spawn_chunk(
                f"parallelAll(Actor[#Threads], #({inner} {inner} {inner}), 5000)!!"
            )
            outer = fut.result(15)
            assert [list(x) for x in outer] == [[1, 2, 3, 4]] * 3
        finally:
            twin.pool.shutdown()

    def test_shutdown_rejects_new_work(self, vm):
        pool = TaskPool(vm, max_workers=1)
        assert pool.spawn_chunk("1!!").result(5) == 1
        pool.shutdown()
        with pytest.raises(Exception):
            pool.spawn_chunk("2!!")

    def test_twin_selector_contract(self):
        expected = {
            "spawn", "spawnChunk", "spawnBlock", "parallelAll", "parallelSettle",
            "parallelMap", "awaitAll", "newChannel", "newMutex", "newCounter",
            "sleepMs", "currentThread", "maxWorkers", "activeTasks", "stats",
            "summary", "shutdown",
        }
        assert expected <= set(THREADS_TWIN_SELECTORS)
        assert THREADS_TWIN_SELECTORS["shutdown"] == 2
        fut_specs = ActorFuture.__actor_bridge_specs__
        assert {"await", "isDone", "status", "error", "cancel"} <= set(fut_specs)

    def test_vm_usable_after_heavy_concurrency(self, threads_vm):
        vm, _ = threads_vm
        tasks = " ".join([f'"sumTo(0, {50 + i})!!"' for i in range(16)])
        rs = vm.eval_chunk(f"parallelAll(Actor[#Threads], #({tasks}), 20000)!!")
        assert list(rs) == [(50 + i) * (51 + i) // 2 for i in range(16)]
        assert vm.active_fiber_count == 0
        assert vm.current_fiber is vm.primary_fiber
        assert vm.eval_chunk("fib(10)!!") == 55


# =============================================================================
# 7. VM Bytecode Return-Nil Regressions (_pop_frame _UNSET sentinel)
# =============================================================================


class TestVMNilReturnSemantics:
    def test_preassigned_local_overwritten_by_nil_return(self, vm):
        """Regression (#9/#17): `x := f(self)` must overwrite pre-assigned `x` when `f` returns `nil`."""
        vm.eval_chunk(
            """
            Def retNil(self) { ^nil; }!!
            Def assignLocalFromNil(self | x) {
                x := 42;
                x := retNil(self);
                ^x;
            }!!
            """
        )
        assert vm.eval_chunk("assignLocalFromNil(0)!!") is None

    def test_equality_check_on_nil_return_from_loop_body(self, vm):
        """Regression (#9/#17): `^(f(self) == nil)` must not underflow caller stack when `f` returns `^nil` inside `loop`."""
        vm.eval_chunk(
            """
            Def retNilInLoop(self, target | i) {
                i := 0;
                loop while i < 5 begin
                    if i == target then ^nil; endif;
                    i := i + 1;
                endLoop;
                ^i;
            }!!
            Def checkIsNil(self, target) {
                ^(retNilInLoop(self, target) == nil);
            }!!
            Def checkNotNil(self, target) {
                ^(retNilInLoop(self, target) != nil);
            }!!
            """
        )
        assert vm.eval_chunk("checkIsNil(0, 2)!!") is True
        assert vm.eval_chunk("checkNotNil(0, 2)!!") is False
        assert vm.eval_chunk("checkIsNil(0, 99)!!") is False
        assert vm.eval_chunk("checkNotNil(0, 99)!!") is True

    def test_multi_chunk_and_concurrent_nil_returns(self, threads_vm):
        """Multi-chunk global assignment to `nil` and concurrent worker `nil` returns preserve stack integrity."""
        vm, _ = threads_vm
        vm.eval_chunk(
            """
            Def maybeVal(self, flag) {
                if flag == 0 then ^nil; endif;
                ^flag * 10;
            }!!
            Def checkPair(self, flag | v) {
                v := 999;
                v := maybeVal(self, flag);
                if v == nil then ^-1; endif;
                ^v;
            }!!
            """
        )
        assert vm.eval_chunk("gSlot := 77!! gSlot := maybeVal(0, 0)!!") is None
        assert vm.eval_chunk("gSlot!!") is None
        rs = vm.eval_chunk(
            'parallelAll(Actor[#Threads], #("checkPair(0, 0)!!" "checkPair(0, 3)!!" "maybeVal(0, 0) == nil!!"), 5000)!!'
        )
        assert list(rs) == [-1, 30, True]

