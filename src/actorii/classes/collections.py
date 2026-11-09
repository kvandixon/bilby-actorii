"""1993 Whitewater Actor 4.1 Collection Hierarchy.

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/COLLECTI.CLS
- historical_actor_41_source/actcls/INDEXEDC.CLS
- historical_actor_41_source/actcls/ARRAY.CLS
- historical_actor_41_source/actcls/ORDEREDC.CLS
- historical_actor_41_source/actact/LIST.ACT

Hierarchy:
    ActorObject
        └── ActorCollection
                └── ActorIndexedCollection (subclasses Python list)
                        ├── ActorArray (fixed size array)
                        └── ActorOrderedCollection (dynamically resizable stack/queue)
                                └── ActorList (dynamic sequential list)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Union
from actorii.core import ActorObject, Symbol


def _eval_block(block: Any, item: Any) -> Any:
    """Evaluate either an Actor BlockClosure or a Python callable with item."""
    if block is None:
        return item
    from actorii.vm import BlockClosure, ActorVM

    if isinstance(block, BlockClosure):
        active_vm = getattr(ActorVM, "_active_vm", None)
        if active_vm is not None:
            return active_vm._call_block_closure(block, [item])
        if hasattr(block, "eval"):
            return block.eval(item)
    elif callable(block):
        return block(item)
    return item


def _is_int(v: Any) -> bool:
    """Return True if v is an integer and not a boolean."""
    return isinstance(v, int) and not isinstance(v, bool)


# =============================================================================
# 1. ActorCollection (Abstract Base Collection)
# =============================================================================


class ActorCollection(ActorObject):
    """Abstract formal superclass for all Actor collection objects.

    Source: historical_actor_41_source/actcls/COLLECTI.CLS
    """

    def __init__(self, obj_id: Optional[str] = None):
        super().__init__(obj_id=obj_id)
        self._class_name = "Collection"

    def size(self) -> int:
        """Return number of elements in the collection."""
        if isinstance(self, list):
            return list.__len__(self)
        if hasattr(self, "_elements") and isinstance(
            self._elements, (list, set, tuple)
        ):
            return len(self._elements)
        if hasattr(self, "_entries") and isinstance(self._entries, dict):
            return len(self._entries)
        try:
            return sum(1 for _ in self)  # type: ignore
        except TypeError:
            return 0

    def __len__(self) -> int:
        return self.size()

    def isEmpty(self) -> bool:
        """Return True if collection has 0 elements, False otherwise."""
        return self.size() == 0

    def hasElements(self) -> bool:
        """Return True if collection has elements; raises IndexError if empty."""
        if self.isEmpty():
            raise IndexError("Empty collection error")
        return True

    def do(self, aBlock: Callable[[Any], Any]) -> Any:
        """Evaluate aBlock with each element of the collection.

        Authentic Whitewater Actor behavior: returns last evaluated value or self.
        """
        last_val = None
        for item in self:  # type: ignore
            last_val = _eval_block(aBlock, item)
        return last_val if last_val is not None else self

    def collect(self, aBlock: Callable[[Any], Any]) -> ActorOrderedCollection:
        """Map receiver to a new collection using a one-argument block.

        Source: COLLECTI.CLS:89 Def collect(self, aBlock | coll)
        """
        result = ActorOrderedCollection()
        for item in self:  # type: ignore
            result.add(_eval_block(aBlock, item))
        return result

    def select(self, aBlock: Callable[[Any], Any]) -> ActorOrderedCollection:
        """Filter collection: return subset where aBlock evaluates to truthy.

        Smalltalk 'select' / Actor 'extract'.
        """
        result = ActorOrderedCollection()
        for item in self:  # type: ignore
            val = _eval_block(aBlock, item)
            if val is not None and val is not False and val != 0:
                result.add(item)
        return result

    def extract(self, aBlock: Callable[[Any], Any]) -> ActorOrderedCollection:
        """Authentic 1993 Whitewater Actor method name for select.

        Source: COLLECTI.CLS:116 Def extract(self, aBlock | coll)
        """
        return self.select(aBlock)

    def reject(self, aBlock: Callable[[Any], Any]) -> ActorOrderedCollection:
        """Filter collection: return subset where aBlock evaluates to falsy."""
        result = ActorOrderedCollection()
        for item in self:  # type: ignore
            val = _eval_block(aBlock, item)
            if val is None or val is False or val == 0:
                result.add(item)
        return result

    def asArray(self) -> ActorArray:
        """Convert receiver elements into an authentic fixed-size Array.

        Source: COLLECTI.CLS:175 Def asArray(self | arr, x)
        """
        arr = ActorArray(self.size())
        for idx, item in enumerate(self):  # type: ignore
            arr[idx] = item
        return arr

    def asOrderedCollection(self) -> ActorOrderedCollection:
        """Convert receiver elements into an authentic dynamic OrderedCollection.

        Source: COLLECTI.CLS:155 Def asOrderedCollection(self | oc)
        """
        return ActorOrderedCollection(self)  # type: ignore

    def first(self) -> Any:
        """Return first element without removing it. Raises IndexError if empty."""
        self.hasElements()
        return next(iter(self))  # type: ignore

    def last(self) -> Any:
        """Return last element without removing it. Raises IndexError if empty."""
        self.hasElements()
        it = iter(self)  # type: ignore
        last_item = next(it)
        for item in it:
            last_item = item
        return last_item


# =============================================================================
# 2. ActorIndexedCollection (0-Based Indexed Base Collection)
# =============================================================================


class ActorIndexedCollection(ActorCollection, list):
    """Abstract formal superclass for 0-based indexed collections.

    Source: historical_actor_41_source/actcls/INDEXEDC.CLS
    Subclasses Python's built-in list to guarantee 100% compatibility with
    native VM index bytecode (GET_INDEX/SET_INDEX), slices, and existing tests.
    """

    def __init__(self, initial_elements: Optional[Iterable[Any]] = None):
        if initial_elements is not None:
            list.__init__(self, initial_elements)
        else:
            list.__init__(self)
        ActorObject.__init__(self)
        self._class_name = "IndexedCollection"

    def at(self, index: int) -> Any:
        """Return element at 0-based index. Raises IndexError if out of bounds.

        Source: ARRAY.CLS:146 Prim at(self, index)
        """
        idx = int(index)
        if 0 <= idx < len(self):
            return self[idx]
        raise IndexError(f"Index out of bounds: {index} (size: {len(self)})")

    def atPut(self, index: int, val: Any) -> Any:
        """Store val at 0-based index. Raises IndexError if out of bounds."""
        idx = int(index)
        if 0 <= idx < len(self):
            self[idx] = val
            return val
        raise IndexError(f"Index out of bounds: {index} (size: {len(self)})")

    def put(self, val: Any, index: int) -> Any:
        """Authentic 1993 Actor indexing write: put(coll, val, index).

        Source: ARRAY.CLS:193 Prim put(self, anObject, index)
        """
        return self.atPut(index, val)

    def fill(
        self, anObject: Any, count: Optional[int] = None
    ) -> ActorIndexedCollection:
        """Fill elements with anObject. If count specified, fill up to count elements."""
        limit = count if count is not None else len(self)
        for i in range(min(limit, len(self))):
            self[i] = anObject
        return self

    def copyFrom(self, begIdx: int, endIdx: int) -> ActorIndexedCollection:
        """Return new collection with elements from begIdx to endIdx - 1 inclusive."""
        b = int(begIdx)
        e = int(endIdx)
        if b > e or b < 0 or e > len(self):
            raise IndexError(
                f"Range error: copyFrom({begIdx}, {endIdx}) on size {len(self)}"
            )
        return self.__class__(self[b:e])

    def indexOf(self, target: Any) -> int:
        """Return 0-based index of target using identity (== in Actor, is in Python), or -1."""
        for i, elem in enumerate(self):
            if elem is target:
                return i
        return -1

    def find(self, target: Any) -> int:
        """Return 0-based index of target using equality (= in Actor, == in Python), or -1."""
        for i, elem in enumerate(self):
            if elem == target:
                return i
        return -1

    def keysDo(self, aBlock: Callable[[int], Any]) -> Any:
        """Evaluate aBlock with each 0-based index from 0 to size - 1.

        Source: INDEXEDC.CLS:88 Def keysDo(self, aBlock)
        """
        last_val = None
        for i in range(len(self)):
            last_val = _eval_block(aBlock, i)
        return last_val

    def reverse(self) -> ActorIndexedCollection:
        """Reverse collection in place and return self.

        Source: INDEXEDC.CLS:126 Def reverse(self)
        """
        super().reverse()
        return self

    def __repr__(self) -> str:
        elems = " ".join(repr(x) for x in self)
        return f"{self._class_name}({elems})"

    def asString(self) -> str:
        elems = " ".join(str(x) for x in self)
        return f"{self._class_name}({elems})"


# =============================================================================
# 3. ActorArray (Fixed-Size Sequential Collection)
# =============================================================================


class ActorArray(ActorIndexedCollection):
    """Fixed-size sequential indexed collection.

    Source: historical_actor_41_source/actcls/ARRAY.CLS
    """

    def __init__(self, size_or_elements: Any = 0, initial_element: Any = None):
        if isinstance(
            size_or_elements, (list, tuple, ActorIndexedCollection)
        ) and not isinstance(size_or_elements, (int, str)):
            super().__init__(size_or_elements)
        else:
            sz = int(size_or_elements) if size_or_elements is not None else 0
            super().__init__([initial_element] * sz)
        self._class_name = "Array"

    def fill(self, anObject: Any) -> ActorArray:
        """Fill all elements of the array with anObject.

        Source: ARRAY.CLS:160 Prim fill(self, anObject):Array!!
        """
        for i in range(len(self)):
            self[i] = anObject
        return self

    def copyFrom(self, begIdx: int, endIdx: int) -> ActorArray:
        """Return new Array with elements from begIdx to endIdx - 1 inclusive.

        Source: ARRAY.CLS:153 Prim copyFrom(self, begIdx, endIdx):Array!!
        """
        b = int(begIdx)
        e = int(endIdx)
        if b > e or b < 0 or e > len(self):
            raise ValueError(f"Bad range in copyFrom: {b} to {e}")
        sub = self[b:e]
        arr = ActorArray(len(sub))
        for i, v in enumerate(sub):
            arr[i] = v
        return arr

    def writeSize(self) -> int:
        """Return writeable size (identical to size)."""
        return len(self)

    def readSize(self) -> int:
        """Return readable size (identical to size)."""
        return len(self)


# =============================================================================
# 4. ActorOrderedCollection (Dynamically Resizable Stack/Queue)
# =============================================================================


class ActorOrderedCollection(ActorIndexedCollection):
    """Dynamically growing sequential collection with chronological order and stack/queue methods.

    Source: historical_actor_41_source/actcls/ORDEREDC.CLS
    """

    def __init__(self, initial_elements: Optional[Iterable[Any]] = None):
        super().__init__(initial_elements)
        self._class_name = "OrderedCollection"

    def add(self, anElement: Any) -> Any:
        """Add anElement to the end of the collection. Returns anElement.

        Source: ORDEREDC.CLS:265 Def add(self, anElement)
        """
        self.append(anElement)
        return anElement

    def push(self, anObject: Any) -> Any:
        """Synonym for add. Adds anObject to top/end of stack.

        Source: ORDEREDC.CLS:179 Def push(self, anObj)
        """
        return self.add(anObject)

    def addLast(self, anElement: Any) -> Any:
        """Add anElement to the end of collection. Synonym for add."""
        return self.add(anElement)

    def addFirst(self, anElement: Any) -> Any:
        """Insert anElement at index 0, shifting existing elements right."""
        super().insert(0, anElement)
        return anElement

    def first(self) -> Any:
        """Return first element without removing it. Raises IndexError if empty.

        Source: ORDEREDC.CLS:130 Def first(self)
        """
        self.hasElements()
        return self[0]

    def last(self) -> Any:
        """Return last element without removing it. Raises IndexError if empty.

        Source: ORDEREDC.CLS:138 Def last(self)
        """
        self.hasElements()
        return self[-1]

    def removeLast(self) -> Any:
        """Remove and return last element. Raises IndexError if empty.

        Source: ORDEREDC.CLS:162 Def removeLast(self | val)
        """
        self.hasElements()
        return self.pop()

    def pop(self, *args) -> Any:
        """Remove and return last element (or specified index). Raises IndexError if empty.

        Source: ORDEREDC.CLS:174 Def pop(self)
        """
        self.hasElements()
        return super().pop(*args)

    def removeFirst(self) -> Any:
        """Remove and return first element. Raises IndexError if empty.

        Source: ORDEREDC.CLS:247 Def removeFirst(self)
        """
        self.hasElements()
        return self.pop(0)

    def remove(self, item_or_index: Any, by_index: bool = False) -> Any:
        """Remove item by value or index. Raises IndexError if empty or not found.

        Source: ORDEREDC.CLS:229 Def remove(self, idx)
        """
        self.hasElements()
        if by_index:
            return self.removeAt(item_or_index)
        if item_or_index in self:
            super().remove(item_or_index)
            return item_or_index
        elif _is_int(item_or_index) and 0 <= item_or_index < len(self):
            return self.pop(item_or_index)
        raise IndexError(f"Item not found or index out of range: {item_or_index}")

    def removeAt(self, index: int) -> Any:
        """Remove and return element at index. Raises IndexError if out of bounds."""
        self.hasElements()
        idx = int(index)
        if 0 <= idx < len(self):
            return self.pop(idx)
        raise IndexError(f"Index out of range: {index}")

    def insert(self, *args: Any, **kwargs: Any) -> Any:
        """Insert element at index.

        Source: ORDEREDC.CLS:195 Def insert(self, elem, idx)
        Supports both Actor insert(elem, idx) and Python insert(idx, elem).
        """
        if "elem" in kwargs or "idx" in kwargs:
            elem = kwargs.get("elem")
            idx = kwargs.get("idx", len(self))
        elif "item" in kwargs or "index" in kwargs:
            elem = kwargs.get("item")
            idx = kwargs.get("index", len(self))
        elif len(args) == 1:
            elem = args[0]
            idx = len(self)
        elif len(args) >= 2:
            a, b = args[0], args[1]
            if _is_int(a) and _is_int(b):
                # Both are integers
                if 0 <= a <= len(self) and 0 <= b <= len(self):
                    # Both within bounds: default to authentic Actor signature insert(elem, idx)
                    elem, idx = a, b
                elif 0 <= a <= len(self) and not (0 <= b <= len(self)):
                    # a is valid index, b is out-of-bounds as index (e.g. insert(1, 150))
                    idx, elem = a, b
                elif 0 <= b <= len(self) and not (0 <= a <= len(self)):
                    # b is valid index, a is out-of-bounds as index (e.g. insert(150, 1))
                    elem, idx = a, b
                else:
                    # Both out of bounds: default to Actor (elem, idx)
                    elem, idx = a, b
            elif _is_int(a) and not _is_int(b):
                # a is int (index), b is element (e.g. insert(0, 'a'), insert(len, 'd'), insert(-1, 'bad'))
                idx, elem = a, b
            elif _is_int(b) and not _is_int(a):
                # b is int (index), a is element (e.g. insert('x', 2))
                elem, idx = a, b
            else:
                try:
                    idx = int(a)
                    elem = b
                except (ValueError, TypeError):
                    raise IndexError(f"Insert requires integer index, got {a} and {b}")
        else:
            raise TypeError("insert() requires at least 1 argument")

        if not _is_int(idx) or idx < 0 or idx > len(self):
            raise IndexError(f"Insert range error at {idx}")

        super().insert(idx, elem)
        return elem


# =============================================================================
# 5. ActorList (Dynamic Sequential List)
# =============================================================================


class ActorList(ActorOrderedCollection):
    """Dynamic sequential list collection (Smalltalk/Actor List equivalent).

    Source: historical_actor_41_source/actact/LIST.ACT
    """

    def __init__(self, initial_elements: Optional[Iterable[Any]] = None):
        super().__init__(initial_elements)
        self._class_name = "List"


# =============================================================================
# 6. ActorSet (Authentic 1993 Actor Unique Set Collection)
# =============================================================================


class ActorSet(ActorCollection):
    """Authentic 1993 Whitewater Actor Set collection.

    A Set is a collection of unique objects with fast membership testing.
    Source: historical_actor_41_source/actcls/SET.CLS
    """

    def __init__(
        self,
        initial_elements: Optional[Any] = None,
        capacity: Optional[int] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(obj_id=obj_id)
        self._class_name = "Set"
        self._elements: List[Any] = []
        self._set: set[Any] = set()

        if initial_elements is not None:
            if isinstance(initial_elements, int):
                pass
            elif isinstance(initial_elements, (Iterable, ActorCollection)):
                for elem in initial_elements:
                    self.add(elem)

    def add(self, anElement: Any) -> Any:
        """Add anElement to the set if not already present. Returns anElement."""
        try:
            if anElement not in self._set:
                self._set.add(anElement)
                self._elements.append(anElement)
        except TypeError:
            if anElement not in self._elements:
                self._elements.append(anElement)
        return anElement

    def remove(self, anElement: Any) -> Any:
        """Remove anElement from the set. Returns anElement."""
        try:
            if anElement in self._set:
                self._set.remove(anElement)
                if anElement in self._elements:
                    self._elements.remove(anElement)
        except TypeError:
            if anElement in self._elements:
                self._elements.remove(anElement)
        return anElement

    def includes(self, anElement: Any) -> bool:
        """Return True if anElement is in the receiver Set, False otherwise."""
        try:
            return anElement in self._set
        except TypeError:
            return anElement in self._elements

    def in_(self, anElement: Any) -> bool:
        return self.includes(anElement)

    def __contains__(self, anElement: Any) -> bool:
        return self.includes(anElement)

    def size(self) -> int:
        return len(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[Any]:
        return iter(list(self._elements))

    def do(self, aBlock: Callable[[Any], Any]) -> Any:
        """Evaluate aBlock with each element of the Set."""
        last_val = None
        for item in list(self._elements):
            last_val = _eval_block(aBlock, item)
        return last_val if last_val is not None else self

    def copy(self) -> ActorSet:
        return ActorSet(list(self._elements))

    def asSet(self) -> ActorSet:
        return self

    def clear(self) -> None:
        self._elements.clear()
        self._set.clear()

    def __repr__(self) -> str:
        return f"Set({' '.join(repr(x) for x in self._elements)})"


# Aliases for export
Collection = ActorCollection
IndexedCollection = ActorIndexedCollection
Array = ActorArray
OrderedCollection = ActorOrderedCollection
ListClass = ActorList
Set = ActorSet

__all__ = [
    "ActorCollection",
    "ActorIndexedCollection",
    "ActorArray",
    "ActorOrderedCollection",
    "ActorList",
    "ActorSet",
    "Collection",
    "IndexedCollection",
    "Array",
    "OrderedCollection",
    "ListClass",
    "Set",
    "_eval_block",
]
