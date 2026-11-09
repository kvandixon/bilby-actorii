"""1993 Whitewater Actor 4.1 Interval Collection.

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/INTERVAL.CLS
- historical_actor_41_source/actcls/INT.CLS
- historical_actor_41_source/actact/SIEVE.ACT
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional
from actorii.classes.collections import ActorCollection, _eval_block


class Interval(ActorCollection):
    """Authentic 1993 Whitewater Actor Interval collection.

    Represents an arithmetic progression from start to stop (exclusive) with step.
    Source: historical_actor_41_source/actcls/INTERVAL.CLS
    """

    def __init__(
        self, start: int = 0, stop: int = 0, step: int = 1, obj_id: Optional[str] = None
    ):
        super().__init__(obj_id=obj_id)
        self._class_name = "Interval"
        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step)
        if self._step == 0:
            raise ValueError("Interval step cannot be zero")

    @classmethod
    def over(cls, start: int, stop: int) -> "Interval":
        """Factory: create interval from start to stop with step 1."""
        return cls(start, stop, 1)

    @classmethod
    def overBy(cls, start: int, stop: int, step: int) -> "Interval":
        """Factory: create interval from start to stop with step."""
        return cls(start, stop, step)

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def step(self) -> int:
        return self._step

    def get_start(self) -> int:
        return self._start

    def get_stop(self) -> int:
        return self._stop

    def get_step(self) -> int:
        return self._step

    def size(self) -> int:
        """Authentic 1993 Actor 4.1 size formula.

        Source: INTERVAL.CLS:88
        Def size(self) { ^max((stop - start + step - (step / abs(step)))/step, 0); }!!
        """
        if self._step > 0:
            if self._stop <= self._start:
                return 0
            return (self._stop - self._start - 1) // self._step + 1
        else:
            if self._start <= self._stop:
                return 0
            return (self._start - self._stop - 1) // (-self._step) + 1

    def __len__(self) -> int:
        return self.size()

    def first(self) -> Optional[int]:
        """First element in interval progression, or raises IndexError if empty."""
        if self.size() <= 0:
            raise IndexError("Empty interval")
        return self._start

    def last(self) -> Optional[int]:
        """Last element in interval progression, or raises IndexError if empty."""
        sz = self.size()
        if sz <= 0:
            raise IndexError("Empty interval")
        return self._start + (sz - 1) * self._step

    def at(self, idx: int) -> int:
        """0-based indexed access into interval progression."""
        i = int(idx)
        sz = self.size()
        if i < 0 or i >= sz:
            raise IndexError(f"Interval index out of range: {idx} (size {sz})")
        return self._start + i * self._step

    def __getitem__(self, idx: int) -> int:
        return self.at(idx)

    def in_interval(self, val: int) -> bool:
        """Authentic 1993 Interval 'in' test.

        Source: INTERVAL.CLS:95
        """
        v = int(val)
        if self._step > 0:
            if self._start <= v < self._stop:
                return (v - self._start) % self._step == 0
        else:
            if self._stop < v <= self._start:
                return (self._start - v) % (-self._step) == 0
        return False

    def __contains__(self, val: Any) -> bool:
        if isinstance(val, (int, float)):
            return self.in_interval(int(val))
        return False

    def __iter__(self) -> Iterator[int]:
        val = self._start
        if self._step > 0:
            while val < self._stop:
                yield val
                val += self._step
        else:
            while val > self._stop:
                yield val
                val += self._step

    def do(self, aBlock: Callable[[int], Any]) -> Any:
        """Evaluate aBlock over each element. Returns the last number evaluated or self."""
        last_val = None
        for val in self:
            last_val = _eval_block(aBlock, val)
        return last_val if last_val is not None else self

    def collect(self, aBlock: Callable[[int], Any]) -> List[Any]:
        """Evaluate aBlock over each element and return collected elements."""
        from actorii.classes.collections import ActorArray

        res = [_eval_block(aBlock, val) for val in self]
        arr = ActorArray(len(res))
        for i, v in enumerate(res):
            arr[i] = v
        return arr

    def asArray(self) -> Any:
        """Convert interval to Array."""
        from actorii.classes.collections import ActorArray

        items = list(self)
        arr = ActorArray(len(items))
        for i, v in enumerate(items):
            arr[i] = v
        return arr

    def species(self) -> str:
        """Species of Interval is Array."""
        return "Array"

    def reverse(self) -> "Interval":
        """Reverse interval: start and stop swapped, step negated."""
        sz = self.size()
        if sz <= 0:
            return Interval(self._stop, self._start, -self._step)
        last_elem = self.last()
        first_elem = self.first()
        assert last_elem is not None and first_elem is not None
        # stop exclusive bound: beyond first_elem in direction of -step
        new_stop = first_elem - self._step
        return Interval(last_elem, new_stop, -self._step)

    def asString(self) -> str:
        return f"{self._start}..{self._stop} by {self._step}"

    def __repr__(self) -> str:
        return f"Interval({self._start}, {self._stop}, {self._step})"


__all__ = ["Interval"]
