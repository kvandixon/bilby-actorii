"""ActorII Opt-In Multithreaded Concurrency Primitives (`actorii.classes.concurrency`).

Modern structured concurrency for the 1993 Actor 4.1 language WITHOUT turning
every `ActorObject` into a thread. Work is scheduled onto a small bounded pool
of OS worker threads; every task executes inside its own `ExecutionFiber`
(isolated operand stack, call stack and gas budget) against the shared,
lock-guarded `ActorVM` image.

Python-side primitives
----------------------
- `TaskPool`      : bounded worker pool (`spawn`, `spawn_chunk`, `spawn_block`,
                    `parallel_all`, `parallel_settle`, `parallel_map`, `shutdown`).
- `ActorFuture`   : task handle (`await_`, `is_done`, `status`, `error`, `cancel`).
- `ActorChannel`  : bounded, closable, thread-safe mailbox queue.
- `ActorMutex`    : re-entrant mutex with `with_lock(block)` helper.
- `AtomicCounter` : lock-protected integer cell (`add`, `compare_and_set`, ...).

Actor-side (mounted by the harness as `Actor[#Threads]`)
--------------------------------------------------------
`ActorThreadPoolTwin` is an `@actor_bridge` twin; every handle it returns is a
capability-gated `ActorBridgeObject`, e.g.::

    f := spawnChunk(Actor[#Threads], "fib(20)!!", 200000)!!
    await(f, 5000)!!                       /* -> 6765                       */
    rs := parallelAll(Actor[#Threads], #("1 + 1!!" "2 * 21!!"), 2000)!!
    ch := newChannel(Actor[#Threads], 16)!! send(ch, 42)!! receive(ch, 100)!!
"""

from __future__ import annotations

import collections
import concurrent.futures
import os
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from actorii.bridge import ActorBridgeObject, actor_bridge, actor_command, actor_query

__all__ = [
    "ActorTaskError",
    "ActorTaskTimeoutError",
    "ActorTaskCancelledError",
    "ActorChannelClosedError",
    "ActorFuture",
    "ActorChannel",
    "ActorMutex",
    "AtomicCounter",
    "TaskPool",
    "ActorThreadPoolTwin",
    "wrap_for_actor",
    "unwrap_actor_value",
    "CONCURRENCY_BRIDGE_CLASSES",
    "THREADS_TWIN_SELECTORS",
]

STATUS_PENDING = "#pending"
STATUS_RUNNING = "#running"  # internal; reported to Actor code as #pending
STATUS_COMPLETED = "#completed"
STATUS_FAILED = "#failed"
STATUS_CANCELLED = "#cancelled"


# =============================================================================
# Errors
# =============================================================================


class ActorTaskError(RuntimeError):
    """Base class for ActorII concurrency errors."""


class ActorTaskTimeoutError(ActorTaskError, TimeoutError):
    """Raised when awaiting a task / fan-out exceeds its timeout."""


class ActorTaskCancelledError(ActorTaskError):
    """Raised when awaiting a cancelled task."""


class ActorChannelClosedError(ActorTaskError):
    """Raised when sending on a closed channel."""


# =============================================================================
# Helpers
# =============================================================================


def _norm_selector(selector: Any) -> str:
    sel = str(selector).strip()
    return sel[1:] if sel.startswith("#") else sel


def _timeout_seconds(timeout_ms: Any) -> Optional[float]:
    """Actor timeouts are milliseconds; nil / negative => wait forever."""
    if timeout_ms is None:
        return None
    try:
        ms = float(timeout_ms)
    except (TypeError, ValueError):
        return None
    if ms < 0:
        return None
    return ms / 1000.0


def unwrap_actor_value(value: Any) -> Any:
    """Unwrap an `ActorBridgeObject` handle back to its Python target."""
    if isinstance(value, ActorBridgeObject):
        return value.target
    return value


def _as_list(value: Any) -> List[Any]:
    value = unwrap_actor_value(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    items = getattr(value, "_items", None)
    if isinstance(items, list):
        return list(items)
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _make_array(values: List[Any]) -> Any:
    """Return an Actor `Array` (ActorArray) holding `values` in order."""
    try:
        from actorii.classes.collections import ActorArray

        return ActorArray(list(values))
    except Exception:  # pragma: no cover - defensive fallback
        return list(values)


def wrap_for_actor(vm: Any, obj: Any) -> Any:
    """Wrap a concurrency primitive as a capability-gated `ActorBridgeObject`."""
    if obj is None or isinstance(obj, ActorBridgeObject):
        return obj
    specs = getattr(type(obj), "__actor_bridge_specs__", None)
    if specs is None:
        return obj
    name = getattr(type(obj), "__actor_bridge_name__", type(obj).__name__)
    existing = getattr(obj, "_actor_handle", None)
    if existing is not None:
        return existing
    handle = ActorBridgeObject(
        target=obj,
        class_name=name,
        specs=dict(specs),
        allow_commands=True,
        vm=vm,
        obj_id=f"{name.lower()}:{id(obj)}",
    )
    try:
        obj._actor_handle = handle
    except Exception:
        pass
    return handle


def _active_vm() -> Any:
    from actorii.vm import ActorVM

    return ActorVM._active_vm


# =============================================================================
# ActorFuture
# =============================================================================


@actor_bridge(name="Future")
class ActorFuture:
    """Handle to a task running (or queued) on a `TaskPool`."""

    _QUEUED = "queued"

    def __init__(self, label: str = "", pool: Optional["TaskPool"] = None) -> None:
        self._cond = threading.Condition()
        self._state = self._QUEUED
        self._result: Any = None
        self._exc: Optional[BaseException] = None
        self._fn: Optional[Callable[["ActorFuture"], Any]] = None
        self._fiber: Any = None
        self._cancel_requested = False
        self._pool = pool
        self._callbacks: List[Callable[["ActorFuture"], None]] = []
        self.label = str(label)
        self.submitted_at = time.perf_counter()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.thread_name: str = ""

    # ---------------------------------------------------------------- internals
    def _claim(self) -> bool:
        with self._cond:
            if self._state != self._QUEUED:
                return False
            self._state = STATUS_RUNNING
            self.started_at = time.perf_counter()
            self.thread_name = threading.current_thread().name
            return True

    def _attach_fiber(self, fiber: Any) -> None:
        with self._cond:
            self._fiber = fiber
            if self._cancel_requested and fiber is not None:
                fiber.max_opcodes = 0

    def _finish(self, state: str, result: Any = None, exc: Optional[BaseException] = None) -> None:
        with self._cond:
            if self._state in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
                return
            self._state = state
            self._result = result
            self._exc = exc
            self.finished_at = time.perf_counter()
            self._fiber = None
            callbacks = list(self._callbacks)
            self._cond.notify_all()
        for cb in callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def _run(self) -> None:
        """Execute the task body if not already claimed (by a worker or an inline helper)."""
        if not self._claim():
            return
        fn = self._fn
        try:
            result = fn(self) if fn is not None else None
        except BaseException as exc:  # noqa: BLE001 - propagate via future
            if self._cancel_requested:
                self._finish(
                    STATUS_CANCELLED,
                    exc=ActorTaskCancelledError(f"Task '{self.label}' was cancelled while running"),
                )
            else:
                self._finish(STATUS_FAILED, exc=exc)
        else:
            self._finish(STATUS_COMPLETED, result=result)

    def add_done_callback(self, cb: Callable[["ActorFuture"], None]) -> None:
        with self._cond:
            if self._state not in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
                self._callbacks.append(cb)
                return
        cb(self)

    # --------------------------------------------------------------- Python API
    def is_done(self) -> bool:
        with self._cond:
            return self._state in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED)

    def raw_status(self) -> str:
        with self._cond:
            return STATUS_PENDING if self._state == self._QUEUED else self._state

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until done (help-run inline when awaited from a saturated worker)."""
        pool = self._pool
        if pool is not None and pool._is_worker_thread():
            # Deadlock avoidance for nested fan-out: run a still-queued task inline.
            self._run()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._state not in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def result(self, timeout: Optional[float] = None) -> Any:
        """Return the task value, re-raising its failure (or timeout / cancellation)."""
        if not self.wait(timeout):
            raise ActorTaskTimeoutError(
                f"ActorTaskTimeoutError: task '{self.label}' not finished within {timeout:.3f}s"
            )
        with self._cond:
            if self._state == STATUS_CANCELLED:
                raise self._exc or ActorTaskCancelledError(f"Task '{self.label}' was cancelled")
            if self._state == STATUS_FAILED and self._exc is not None:
                raise self._exc
            return self._result

    def exception(self) -> Optional[BaseException]:
        with self._cond:
            return self._exc

    def request_cancel(self) -> bool:
        """Cancel a queued task, or cooperatively abort a running Actor task (gas → 0)."""
        with self._cond:
            if self._state == self._QUEUED:
                self._state = STATUS_RUNNING  # transient; finished below
                queued = True
            elif self._state == STATUS_RUNNING:
                self._cancel_requested = True
                if self._fiber is not None:
                    self._fiber.max_opcodes = 0
                return True
            else:
                return False
        if queued:
            self._finish(
                STATUS_CANCELLED,
                exc=ActorTaskCancelledError(f"Task '{self.label}' was cancelled before starting"),
            )
        return True

    def elapsed_ms(self) -> float:
        start = self.started_at or self.submitted_at
        end = self.finished_at or time.perf_counter()
        return round((end - start) * 1000.0, 3)

    # ------------------------------------------------------------ Actor API
    @actor_query(name="await")
    def await_(self, timeoutMs: Any = None) -> Any:
        """await(fut, timeoutMs)!! — block for the value; re-raises task errors / timeout."""
        return self.result(_timeout_seconds(timeoutMs))

    @actor_query
    def awaitOr(self, timeoutMs: Any = None, default: Any = None) -> Any:
        """awaitOr(fut, timeoutMs, default)!! — value, or `default` on timeout/failure/cancel."""
        try:
            return self.result(_timeout_seconds(timeoutMs))
        except Exception:
            return default

    @actor_query
    def isDone(self) -> bool:
        """isDone(fut)!! — true once completed, failed or cancelled."""
        return self.is_done()

    @actor_query
    def status(self) -> str:
        """status(fut)!! — #pending / #completed / #failed / #cancelled."""
        st = self.raw_status()
        return STATUS_PENDING if st == STATUS_RUNNING else st

    @actor_query
    def isRunning(self) -> bool:
        """isRunning(fut)!! — true while a worker is executing the task."""
        return self.raw_status() == STATUS_RUNNING

    @actor_query
    def error(self) -> Optional[str]:
        """error(fut)!! — "ErrorType: message" for failed/cancelled tasks, else nil."""
        exc = self.exception()
        if exc is None:
            return None
        return f"{type(exc).__name__}: {exc}"

    @actor_query
    def value(self) -> Any:
        """value(fut)!! — non-blocking: the result if completed, else nil."""
        with self._cond:
            return self._result if self._state == STATUS_COMPLETED else None

    @actor_command(shadow_safe=True)
    def cancel(self) -> bool:
        """cancel(fut)!! — cancel queued task / cooperatively abort running Actor code."""
        return self.request_cancel()

    @actor_query
    def elapsedMs(self) -> float:
        """elapsedMs(fut)!! — wall-clock runtime in milliseconds."""
        return self.elapsed_ms()

    @actor_query
    def threadName(self) -> str:
        """threadName(fut)!! — worker thread that executed the task."""
        return self.thread_name

    @actor_query
    def taskLabel(self) -> str:
        """taskLabel(fut)!! — human-readable task description."""
        return self.label

    def __repr__(self) -> str:
        return f"<ActorFuture {self.label!r} {self.raw_status()}>"



# `await` is a Python keyword: expose the Actor selector via a class attribute alias
# so ActorBridgeObject's `getattr(target, "await")` resolves to `await_`.
setattr(ActorFuture, "await", ActorFuture.await_)


# =============================================================================
# ActorChannel
# =============================================================================


@actor_bridge(name="Channel")
class ActorChannel:
    """Bounded, closable, thread-safe FIFO mailbox for message passing between tasks.

    `capacity <= 0` means unbounded.
    """

    def __init__(self, capacity: Any = 0) -> None:
        try:
            cap = int(capacity or 0)
        except (TypeError, ValueError):
            cap = 0
        self._capacity = max(0, cap)
        self._items: Deque[Any] = collections.deque()
        self._closed = False
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self.sent_count = 0
        self.received_count = 0

    # Python API
    def put(self, value: Any, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_full:
            while True:
                if self._closed:
                    raise ActorChannelClosedError("ActorChannelClosedError: send on closed channel")
                if self._capacity <= 0 or len(self._items) < self._capacity:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._not_full.wait(remaining)
            self._items.append(value)
            self.sent_count += 1
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Tuple[bool, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._items:
                if self._closed:
                    return False, None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False, None
                self._not_empty.wait(remaining)
            val = self._items.popleft()
            self.received_count += 1
            self._not_full.notify()
            return True, val

    # Actor API
    @actor_command(shadow_safe=True)
    def send(self, value: Any, timeoutMs: Any = None) -> bool:
        """send(ch, value[, timeoutMs])!! — enqueue (blocks while full); false on timeout."""
        return self.put(value, _timeout_seconds(timeoutMs))

    @actor_command(shadow_safe=True)
    def receive(self, timeoutMs: Any = None) -> Any:
        """receive(ch, timeoutMs)!! — dequeue next value; nil on timeout or closed+drained."""
        _ok, val = self.get(_timeout_seconds(timeoutMs))
        return val

    @actor_command(shadow_safe=True)
    def tryReceive(self) -> Any:
        """tryReceive(ch)!! — non-blocking dequeue; nil when empty."""
        _ok, val = self.get(0.0)
        return val

    @actor_query
    def size(self) -> int:
        """size(ch)!! — number of buffered messages."""
        with self._lock:
            return len(self._items)

    @actor_query
    def capacity(self) -> int:
        """capacity(ch)!! — max buffered messages (0 = unbounded)."""
        return self._capacity

    @actor_command(shadow_safe=True)
    def close(self) -> bool:
        """close(ch)!! — reject further sends; receivers drain then get nil."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()
        return True

    @actor_query
    def isClosed(self) -> bool:
        """isClosed(ch)!!"""
        with self._lock:
            return self._closed

    @actor_command(shadow_safe=True)
    def drain(self) -> Any:
        """drain(ch)!! — remove and return all buffered messages as an Array."""
        with self._lock:
            vals = list(self._items)
            self._items.clear()
            self.received_count += len(vals)
            self._not_full.notify_all()
        return _make_array(vals)

    def __repr__(self) -> str:
        return f"<ActorChannel size={len(self._items)} cap={self._capacity} closed={self._closed}>"


# =============================================================================
# ActorMutex & AtomicCounter
# =============================================================================


@actor_bridge(name="Mutex")
class ActorMutex:
    """Re-entrant mutex for synchronized updates of shared Actor state."""

    def __init__(self, vm: Any = None) -> None:
        self._vm = vm
        self._lock = threading.RLock()
        self._owner: Optional[int] = None
        self._depth = 0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        ok = self._lock.acquire(timeout=-1 if timeout is None else max(0.0, timeout))
        if ok:
            self._owner = threading.get_ident()
            self._depth += 1
        return ok

    def release(self) -> bool:
        if self._owner != threading.get_ident():
            return False
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._owner = None
        self._lock.release()
        return True

    @actor_command(shadow_safe=True)
    def lock(self, timeoutMs: Any = None) -> bool:
        """lock(m[, timeoutMs])!! — acquire; false on timeout."""
        return self.acquire(_timeout_seconds(timeoutMs))

    @actor_command(shadow_safe=True)
    def tryLock(self) -> bool:
        """tryLock(m)!! — non-blocking acquire."""
        return self.acquire(0.0)

    @actor_command(shadow_safe=True)
    def unlock(self) -> bool:
        """unlock(m)!! — release (only by the owning thread)."""
        return self.release()

    @actor_query
    def isLocked(self) -> bool:
        """isLocked(m)!!"""
        return self._owner is not None

    @actor_command(shadow_safe=True)
    def withLock(self, block: Any, timeoutMs: Any = None) -> Any:
        """withLock(m, {...})!! — run a block (or callable) while holding the mutex."""
        if not self.acquire(_timeout_seconds(timeoutMs)):
            raise ActorTaskTimeoutError("ActorTaskTimeoutError: withLock could not acquire mutex")
        try:
            return _invoke_callable_inline(self._vm or _active_vm(), block, [])
        finally:
            self.release()

    def __enter__(self) -> "ActorMutex":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


@actor_bridge(name="AtomicCounter")
class AtomicCounter:
    """Lock-protected integer cell (race-free counters for parallel tasks)."""

    def __init__(self, initial: Any = 0) -> None:
        self._value = int(initial or 0)
        self._lock = threading.Lock()

    @actor_command(shadow_safe=True)
    def add(self, delta: Any = 1) -> int:
        """add(c, delta)!! — atomically add, returning the new value."""
        with self._lock:
            self._value += int(delta)
            return self._value

    @actor_command(shadow_safe=True)
    def increment(self) -> int:
        """increment(c)!! — atomically +1, returning the new value."""
        return self.add(1)

    @actor_command(shadow_safe=True)
    def decrement(self) -> int:
        """decrement(c)!! — atomically -1, returning the new value."""
        return self.add(-1)

    @actor_query
    def get(self) -> int:
        """get(c)!! — current value."""
        with self._lock:
            return self._value

    @actor_command(shadow_safe=True)
    def set(self, value: Any) -> int:
        """set(c, v)!! — store v, returning the previous value."""
        with self._lock:
            prev, self._value = self._value, int(value)
            return prev

    @actor_command(shadow_safe=True)
    def compareAndSet(self, expected: Any, new: Any) -> bool:
        """compareAndSet(c, expected, new)!! — CAS; true if swapped."""
        with self._lock:
            if self._value == int(expected):
                self._value = int(new)
                return True
            return False

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


# =============================================================================
# Task bodies
# =============================================================================


def _invoke_callable_inline(vm: Any, fn: Any, args: List[Any]) -> Any:
    """Invoke a BlockClosure / bridged callable / Python callable in the current fiber."""
    from actorii.vm import BlockClosure

    fn = unwrap_actor_value(fn)
    if isinstance(fn, BlockClosure):
        if vm is None:
            raise ActorTaskError("No ActorVM available to evaluate block")
        return vm._call_block_closure(fn, list(args))
    if callable(fn):
        return fn(*args)
    raise TypeError(f"Object {fn!r} is not callable as a task")


class TaskPool:
    """Bounded worker pool executing Actor work in isolated execution fibers.

    Not thread-per-object: `max_workers` OS threads multiplex any number of tasks.
    """

    def __init__(
        self,
        vm: Any = None,
        max_workers: int = 4,
        *,
        default_gas: Optional[int] = None,
        name: str = "actorii-task",
    ) -> None:
        self.vm = vm
        self.max_workers = max(1, int(max_workers or 1))
        self.default_gas = default_gas
        self.name = name
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._shutdown = False
        self._worker_idents: set = set()
        self._live: "set[ActorFuture]" = set()
        self.stats: Dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "inline_helped": 0,
        }

    # ------------------------------------------------------------ lifecycle
    def _get_vm(self) -> Any:
        return self.vm if self.vm is not None else _active_vm()

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._shutdown:
                raise ActorTaskError("ActorTaskError: TaskPool has been shut down")
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self.name,
                    initializer=self._register_worker,
                )
            return self._executor

    def _register_worker(self) -> None:
        with self._lock:
            self._worker_idents.add(threading.get_ident())

    def _is_worker_thread(self) -> bool:
        return threading.get_ident() in self._worker_idents

    def shutdown(self, wait: bool = True, cancel_pending: bool = True) -> None:
        """Stop accepting work; cancel queued tasks and abort running Actor code."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            executor = self._executor
            live = list(self._live)
        if cancel_pending:
            for fut in live:
                fut.request_cancel()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def cancel_all(self) -> int:
        """Cancel all queued or running tasks without permanently shutting down the pool."""
        with self._lock:
            live = list(self._live)
        cancelled = 0
        for fut in live:
            if fut.request_cancel():
                cancelled += 1
        return cancelled

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    def __enter__(self) -> "TaskPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._live if f.raw_status() == STATUS_RUNNING)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._live)

    # ----------------------------------------------------------- submission
    def _on_done(self, fut: ActorFuture) -> None:
        st = fut.raw_status()
        with self._lock:
            self._live.discard(fut)
            if st == STATUS_COMPLETED:
                self.stats["completed"] += 1
            elif st == STATUS_FAILED:
                self.stats["failed"] += 1
            elif st == STATUS_CANCELLED:
                self.stats["cancelled"] += 1

    def submit(
        self,
        body: Callable[[], Any],
        *,
        label: str = "task",
        gas: Optional[int] = None,
        isolate: bool = True,
    ) -> ActorFuture:
        """Submit a Python callable that runs inside a fresh VM fiber on a worker thread."""
        executor = self._ensure_executor()
        vm = self._get_vm()
        # Capture the submitter's shadow-run flag so dry-run tasks stay side-effect free.
        shadow = bool(getattr(vm, "_in_shadow_run", False)) if vm is not None else False
        budget = gas if gas is not None else self.default_gas
        fut = ActorFuture(label=label, pool=self)

        def _task(f: ActorFuture) -> Any:
            if vm is None or not isolate or not hasattr(vm, "fiber"):
                return body()
            kwargs: Dict[str, Any] = {"name": f"{self.name}:{label}"[:80], "shadow": shadow}
            if budget is not None:
                kwargs["max_opcodes"] = int(budget)
            with vm.fiber(**kwargs) as fib:
                f._attach_fiber(fib)
                return body()

        fut._fn = _task
        fut.add_done_callback(self._on_done)
        with self._lock:
            self._live.add(fut)
            self.stats["submitted"] += 1
        executor.submit(fut._run)
        return fut

    def spawn(
        self, receiver: Any, selector: Any, args: Any = None, *, gas: Optional[int] = None
    ) -> ActorFuture:
        """Concurrently send `selector` to `receiver` with `args`."""
        sel = _norm_selector(selector)
        arg_list = _as_list(args)
        vm = self._get_vm()
        rcv = receiver

        def _body() -> Any:
            return vm.dispatch_message(rcv, sel, arg_list)

        return self.submit(_body, label=f"send #{sel}", gas=gas)

    def spawn_chunk(self, source: str, gas: Optional[int] = None) -> ActorFuture:
        """Concurrently evaluate an Actor 4.1 chunk with an isolated stack + gas budget."""
        src = str(source)
        vm = self._get_vm()

        def _body() -> Any:
            return vm.eval_chunk(src)

        label = src.strip().replace("\n", " ")
        return self.submit(_body, label=f"chunk {label[:48]}", gas=gas)

    def spawn_block(self, block: Any, args: Any = None, gas: Optional[int] = None) -> ActorFuture:
        """Concurrently evaluate a BlockClosure (or Python callable)."""
        vm = self._get_vm()
        arg_list = _as_list(args)
        target = unwrap_actor_value(block)

        def _body() -> Any:
            return _invoke_callable_inline(vm, target, arg_list)

        return self.submit(_body, label="block", gas=gas)

    def spawn_task(self, task: Any, gas: Optional[int] = None) -> ActorFuture:
        """Spawn a polymorphic task spec.

        - `str`                         -> Actor chunk
        - `BlockClosure` / callable     -> invoked with no args
        - `[receiver, #sel, args?]`     -> message send
        - `ActorFuture`                 -> returned as-is (composition)
        """
        from actorii.vm import BlockClosure

        task = unwrap_actor_value(task)
        if isinstance(task, ActorFuture):
            return task
        if isinstance(task, str):
            return self.spawn_chunk(task, gas)
        if isinstance(task, BlockClosure) or callable(task):
            return self.spawn_block(task, None, gas)
        spec = _as_list(task)
        if len(spec) >= 2:
            return self.spawn(spec[0], spec[1], spec[2] if len(spec) > 2 else None, gas=gas)
        raise TypeError(f"Unsupported task spec for parallel execution: {task!r}")

    # -------------------------------------------------------------- fan-out
    def _gather(
        self, futures: List[ActorFuture], timeout: Optional[float], settle: bool
    ) -> List[Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        results: List[Any] = []
        try:
            for fut in futures:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if not fut.wait(remaining):
                    if settle:
                        fut.request_cancel()
                        results.append(_make_array([STATUS_PENDING, None, "ActorTaskTimeoutError"]))
                        continue
                    raise ActorTaskTimeoutError(
                        f"ActorTaskTimeoutError: fan-out of {len(futures)} tasks exceeded "
                        f"{timeout * 1000.0 if timeout is not None else 0:.0f}ms "
                        f"(stalled on '{fut.label}')"
                    )
                if settle:
                    st = fut.raw_status()
                    exc = fut.exception()
                    results.append(
                        _make_array(
                            [
                                st,
                                fut.value(),
                                None if exc is None else f"{type(exc).__name__}: {exc}",
                            ]
                        )
                    )
                else:
                    results.append(fut.result(0))
        except BaseException:
            for fut in futures:
                if not fut.is_done():
                    fut.request_cancel()
            raise
        return results

    def parallel_all(self, tasks: Any, timeout_ms: Any = None, gas: Optional[int] = None) -> List[Any]:
        """Run all tasks concurrently; results in input order; first error (in order) re-raised."""
        futures = [self.spawn_task(t, gas) for t in _as_list(tasks)]
        return self._gather(futures, _timeout_seconds(timeout_ms), settle=False)

    def parallel_settle(
        self, tasks: Any, timeout_ms: Any = None, gas: Optional[int] = None
    ) -> List[Any]:
        """Like `parallel_all` but never raises: each entry is `[status, value, error]`."""
        futures = [self.spawn_task(t, gas) for t in _as_list(tasks)]
        return self._gather(futures, _timeout_seconds(timeout_ms), settle=True)

    def parallel_map(
        self,
        items: Any,
        receiver: Any,
        selector: Any = None,
        timeout_ms: Any = None,
        gas: Optional[int] = None,
    ) -> List[Any]:
        """Apply `selector(receiver, item)` (or a block / `selector(item)` when receiver is nil)."""
        from actorii.vm import BlockClosure

        rcv = unwrap_actor_value(receiver)
        futures: List[ActorFuture] = []
        for item in _as_list(items):
            if isinstance(rcv, BlockClosure) or (selector is None and callable(rcv)):
                futures.append(self.spawn_block(rcv, [item], gas))
            elif rcv is None:
                futures.append(self.spawn(item, selector, [], gas=gas))
            else:
                futures.append(self.spawn(rcv, selector, [item], gas=gas))
        return self._gather(futures, _timeout_seconds(timeout_ms), settle=False)

    def await_all(self, futures: Any, timeout_ms: Any = None) -> List[Any]:
        futs = [unwrap_actor_value(f) for f in _as_list(futures)]
        futs = [f for f in futs if isinstance(f, ActorFuture)]
        return self._gather(futs, _timeout_seconds(timeout_ms), settle=False)


# =============================================================================
# Actor[#Threads] twin
# =============================================================================


@actor_bridge(name="ThreadPool")
class ActorThreadPoolTwin:
    """`Actor[#Threads]`: opt-in parallel execution for Actor 4.1 code (bounded worker pool)."""

    def __init__(
        self,
        vm: Any = None,
        max_workers: int = 8,
        default_gas: Optional[int] = None,
        name: str = "actorii-threads",
    ) -> None:
        # Tolerate `new(ThreadPool, 4)` from Actor code (first arg = worker count).
        if isinstance(vm, int) and not isinstance(vm, bool):
            max_workers, vm = vm, None
        self.vm = vm
        self.pool = TaskPool(vm, max_workers=max_workers, default_gas=default_gas, name=name)

    def _vm(self) -> Any:
        return self.vm if self.vm is not None else _active_vm()

    def _wrap(self, obj: Any) -> Any:
        return wrap_for_actor(self._vm(), obj)

    _INFINITE_LOOP_RE = re.compile(
        r"\bloop\s+while\s*\(?\s*(?:true|1|1\s*==\s*1)\s*\)?",
        re.IGNORECASE,
    )
    _LOCKDOWN_MAX_TASK_GAS = 500_000
    _LOCKDOWN_MAX_SLEEP_MS = 15_000

    @classmethod
    def _gas(cls, gasLimit: Any) -> Optional[int]:
        if gasLimit is None:
            return cls._LOCKDOWN_MAX_TASK_GAS if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1" else None
        try:
            g = int(gasLimit)
        except (TypeError, ValueError):
            return cls._LOCKDOWN_MAX_TASK_GAS if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1" else None
        if g <= 0:
            return cls._LOCKDOWN_MAX_TASK_GAS if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1" else None
        if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1":
            return min(g, cls._LOCKDOWN_MAX_TASK_GAS)
        return g

    @classmethod
    def _guard_background_chunk(cls, src: str) -> None:
        if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1" and cls._INFINITE_LOOP_RE.search(src):
            raise PermissionError(
                "BLOCKED [THREAD_SELF_PRESERVATION_GUARD]: Unbounded `loop while (true)` daemon loops are forbidden "
                "on `Actor[#Threads]` so worker threads can never starve Bilby's foreground self-correction turns! "
                "Use bounded single-pass batches (`parallelMap`, `parallelSettle`, or bounded `spawnChunk`) instead."
            )

    # ------------------------------------------------------------ spawning
    @actor_command(shadow_safe=True)
    def spawn(self, receiver: Any, selector: Any, argsArray: Any = None, gasLimit: Any = None) -> Any:
        """spawn(Actor[#Threads], receiver, #selector, argsArray[, gasLimit])!! -> Future"""
        return self._wrap(self.pool.spawn(receiver, selector, argsArray, gas=self._gas(gasLimit)))

    @actor_command(shadow_safe=True)
    def spawnChunk(self, source: Any, gasLimit: Any = None) -> Any:
        """spawnChunk(Actor[#Threads], "expr!!", gasLimit)!! -> Future (isolated stack + gas)"""
        src = str(source)
        self._guard_background_chunk(src)
        return self._wrap(self.pool.spawn_chunk(src, self._gas(gasLimit)))

    @actor_command(shadow_safe=True)
    def spawnBlock(self, block: Any, argsArray: Any = None, gasLimit: Any = None) -> Any:
        """spawnBlock(Actor[#Threads], {using(x) ...}, argsArray[, gasLimit])!! -> Future"""
        return self._wrap(self.pool.spawn_block(block, argsArray, self._gas(gasLimit)))

    @actor_command(shadow_safe=True)
    def parallelAll(self, tasksArray: Any, timeoutMs: Any = None, gasLimit: Any = None) -> Any:
        """parallelAll(Actor[#Threads], #("a!!" "b!!"), timeoutMs)!! -> Array (input order)"""
        return _make_array(self.pool.parallel_all(tasksArray, timeoutMs, self._gas(gasLimit)))

    @actor_command(shadow_safe=True)
    def parallelSettle(self, tasksArray: Any, timeoutMs: Any = None, gasLimit: Any = None) -> Any:
        """parallelSettle(Actor[#Threads], tasks, timeoutMs)!! -> Array of [status, value, error]"""
        return _make_array(self.pool.parallel_settle(tasksArray, timeoutMs, self._gas(gasLimit)))

    @actor_command(shadow_safe=True)
    def parallelMap(
        self, itemsArray: Any, receiver: Any, selector: Any = None, timeoutMs: Any = None
    ) -> Any:
        """parallelMap(Actor[#Threads], items, receiver, #selector[, timeoutMs])!! -> Array"""
        return _make_array(self.pool.parallel_map(itemsArray, receiver, selector, timeoutMs))

    @actor_query
    def awaitAll(self, futuresArray: Any, timeoutMs: Any = None) -> Any:
        """awaitAll(Actor[#Threads], futures, timeoutMs)!! -> Array of results"""
        return _make_array(self.pool.await_all(futuresArray, timeoutMs))

    # ------------------------------------------------------------ factories
    @actor_query
    def newChannel(self, capacity: Any = 0) -> Any:
        """newChannel(Actor[#Threads], capacity)!! -> Channel (0 = unbounded)"""
        return self._wrap(ActorChannel(capacity))

    @actor_query
    def newMutex(self) -> Any:
        """newMutex(Actor[#Threads])!! -> Mutex"""
        return self._wrap(ActorMutex(self._vm()))

    @actor_query
    def newCounter(self, initial: Any = 0) -> Any:
        """newCounter(Actor[#Threads], initial)!! -> AtomicCounter"""
        return self._wrap(AtomicCounter(initial))

    # ------------------------------------------------------------ utilities
    @actor_query
    def sleepMs(self, ms: Any = 0) -> int:
        """sleepMs(Actor[#Threads], ms)!! — block the calling task (releases the GIL)."""
        try:
            val = max(0, int(ms or 0))
        except (TypeError, ValueError):
            val = 0
        if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1":
            val = min(val, self._LOCKDOWN_MAX_SLEEP_MS)
        time.sleep(val / 1000.0)
        return val

    @actor_query
    def currentThread(self) -> str:
        """currentThread(Actor[#Threads])!! — name of the executing OS thread."""
        return threading.current_thread().name

    @actor_query
    def maxWorkers(self) -> int:
        """maxWorkers(Actor[#Threads])!!"""
        return self.pool.max_workers

    @actor_query
    def activeTasks(self) -> int:
        """activeTasks(Actor[#Threads])!! — tasks currently executing."""
        return self.pool.active_count()

    @actor_query
    def stats(self) -> Dict[str, int]:
        """stats(Actor[#Threads])!! — submitted/completed/failed/cancelled counters."""
        out = dict(self.pool.stats)
        out["maxWorkers"] = self.pool.max_workers
        out["active"] = self.pool.active_count()
        out["live"] = self.pool.pending_count()
        return out

    @actor_query
    def summary(self) -> str:
        """summary(Actor[#Threads])!! — one-line pool status."""
        s = self.pool.stats
        return (
            f"ThreadPool(max={self.pool.max_workers}, active={self.pool.active_count()}, "
            f"submitted={s['submitted']}, completed={s['completed']}, failed={s['failed']}, "
            f"cancelled={s['cancelled']}, shutdown={self.pool.is_shutdown})"
        )

    @actor_command
    def cancelAll(self) -> int:
        """cancelAll(Actor[#Threads])!! — cancel queued/running tasks without destroying the pool."""
        return self.pool.cancel_all()

    @actor_command
    def shutdown(self) -> bool:
        """shutdown(Actor[#Threads])!! — cancel pending work and stop the workers."""
        if os.environ.get("ACTORII_HOST_LOCKDOWN") == "1":
            # In host lockdown on the shared singleton pool, drain/cancel tasks instead of permanently killing Actor[#Threads]
            vm = self._vm()
            shared_threads = None
            if vm is not None and isinstance(getattr(vm, "global_vars", None), dict):
                actor_dict = vm.global_vars.get("Actor", {})
                if isinstance(actor_dict, dict):
                    shared_obj = actor_dict.get("Threads")
                    shared_threads = getattr(shared_obj, "_target", shared_obj)
            if shared_threads is self:
                self.pool.cancel_all()
                return True
        self.pool.shutdown(wait=False)
        return True


CONCURRENCY_BRIDGE_CLASSES: Dict[str, type] = {
    "Future": ActorFuture,
    "Channel": ActorChannel,
    "Mutex": ActorMutex,
    "AtomicCounter": AtomicCounter,
    "ThreadPool": ActorThreadPoolTwin,
}

THREADS_TWIN_SELECTORS: Dict[str, int] = {
    name: spec.tier
    for name, spec in getattr(ActorThreadPoolTwin, "__actor_bridge_specs__", {}).items()
}
