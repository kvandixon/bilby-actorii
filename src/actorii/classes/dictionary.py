"""1993 Whitewater Actor 4.1 Dictionary & KeyedCollection Hierarchy.

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/KEYEDCOL.CLS
- historical_actor_41_source/actcls/DICTIONA.CLS

Hierarchy:
    ActorObject
        └── ActorCollection
                └── KeyedCollection
                        └── Dictionary
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from actorii.core import ActorObject, Symbol
from actorii.classes.collections import ActorCollection, _eval_block

# =============================================================================
# 1. KeyedCollection (Abstract Base Keyed Collection)
# =============================================================================


class KeyedCollection(ActorCollection):
    """Abstract formal class for keyed collections.

    Source: historical_actor_41_source/actcls/KEYEDCOL.CLS
    """

    def __init__(self, obj_id: Optional[str] = None):
        super().__init__(obj_id=obj_id)
        self._class_name = "KeyedCollection"

    def includesKey(self, key: Any) -> bool:
        """Check if key exists in collection."""
        raise NotImplementedError

    def keys(self) -> List[Any]:
        """Return collection of keys."""
        raise NotImplementedError

    def values(self) -> List[Any]:
        """Return collection of values."""
        raise NotImplementedError


# =============================================================================
# 2. Dictionary (Authentic 1993 Actor Dictionary)
# =============================================================================


class Dictionary(KeyedCollection):
    """Authentic 1993 Whitewater Actor Dictionary (KeyedCollection).

    Associative key-value store supporting #Symbol keys, string keys,
    and arbitrary object keys with transparent symbol normalization.

    Source: historical_actor_41_source/actcls/DICTIONA.CLS
    """

    def __init__(self, capacity: int = 16, obj_id: Optional[str] = None):
        super().__init__(obj_id=obj_id)
        self._class_name = "Dictionary"
        self.capacity: int = int(capacity) if capacity is not None else 16
        self._entries: Dict[Any, Any] = {}
        self._key_map: Dict[str, Any] = {}  # normalized string -> canonical key

    def _norm(self, key: Any) -> Any:
        if isinstance(key, str):
            return key[1:] if key.startswith("#") else key
        return key

    def at(self, key: Any) -> Any:
        """Lookup element by key. Returns None if key not found.

        Source: DICTIONA.CLS:230 Def at(self, aKey)
        """
        norm = self._norm(key)
        if isinstance(norm, str) and norm in self._key_map:
            canon = self._key_map[norm]
            return self._entries.get(canon)
        if key in self._entries:
            return self._entries[key]
        return None

    def atPut(self, key: Any, val: Any) -> Any:
        """Store val at key. Returns val."""
        norm = self._norm(key)
        if isinstance(norm, str):
            if norm in self._key_map:
                canon = self._key_map[norm]
                self._entries[canon] = val
            else:
                self._key_map[norm] = key
                self._entries[key] = val
        else:
            self._entries[key] = val
        return val

    def put(self, val: Any, key: Any) -> Any:
        """Authentic 1993 Actor order: put(self, anElement, aKey).

        Source: DICTIONA.CLS:195 Def put(self, anElement, aKey)
        """
        return self.atPut(key, val)

    def add(self, key: Any, val: Any) -> Any:
        """Authentic 1993 Actor order: add(self, aKey, anElement).

        Source: DICTIONA.CLS:214 Def add(self, aKey, anElement)
        """
        return self.atPut(key, val)

    def includesKey(self, key: Any) -> bool:
        """Check if key exists in dictionary."""
        norm = self._norm(key)
        if isinstance(norm, str) and norm in self._key_map:
            return True
        return key in self._entries

    def __contains__(self, key: Any) -> bool:
        return self.includesKey(key)

    def keys(self) -> List[Any]:
        """Return list of keys."""
        return list(self._entries.keys())

    def values(self) -> List[Any]:
        """Return list of values."""
        return list(self._entries.values())

    def removeKey(self, key: Any) -> Any:
        """Remove entry with given key. Returns the removed value or None.

        Source: DICTIONA.CLS:72
        """
        norm = self._norm(key)
        if isinstance(norm, str) and norm in self._key_map:
            canon = self._key_map.pop(norm)
            return self._entries.pop(canon, None)
        if key in self._entries:
            return self._entries.pop(key, None)
        return None

    def remove(self, key: Any) -> Any:
        return self.removeKey(key)

    def removeAt(self, key: Any) -> Any:
        return self.removeKey(key)

    def size(self) -> int:
        """Number of active associations (tally).

        Source: KEYEDCOL.CLS:78
        """
        return len(self._entries)

    def isEmpty(self) -> bool:
        """True if dictionary has 0 elements."""
        return len(self._entries) == 0

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._key_map.clear()

    def do(self, aBlock: Callable[[Any], Any]) -> Any:
        """Evaluate aBlock over the VALUES of the dictionary.

        Authentic 1993 Whitewater Actor behavior: iterates over values, not keys!
        Source: DICTIONA.CLS:107 Def do(self, aBlock | stop, val, i)
        """
        last_val = None
        for val in list(self._entries.values()):
            last_val = _eval_block(aBlock, val)
        return last_val if last_val is not None else self

    def keysDo(self, aBlock: Callable[[Any], Any]) -> Any:
        """Evaluate aBlock over the KEYS of the dictionary.

        Source: DICTIONA.CLS:145 Def keysDo(self, aBlock)
        """
        last_val = None
        for k in list(self._entries.keys()):
            last_val = _eval_block(aBlock, k)
        return last_val

    def assocsDo(self, aBlock: Callable[[Any], Any]) -> Any:
        """Evaluate aBlock over each key-value pair [key, value].

        Source: DICTIONA.CLS:177 Def assocsDo(self, aBlock)
        """
        last_val = None
        for k, v in list(self._entries.items()):
            last_val = _eval_block(aBlock, [k, v])
        return last_val

    def __getitem__(self, key: Any) -> Any:
        return self.at(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.atPut(key, value)

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[Any]:
        # By default in Python iteration, yields keys
        return iter(self.keys())

    def asString(self) -> str:
        items = [f"{k!r}: {v!r}" for k, v in self._entries.items()]
        return f"Dictionary({', '.join(items)})"

    def __repr__(self) -> str:
        return self.asString()


# =============================================================================
# 3. ActorDictionary (Live Global Actor Runtime Dictionary)
# =============================================================================


class ActorDictionary(dict):
    """Authentic 1993 Actor global dictionary.

    Contains all top-level symbols, classes, and globals:
    - Supports Actor[#Key] and Actor['Key'] access
    - Supports Actor[#Symbol] := value assignment
    - Synchronizes bidirectionally with ActorVM.global_vars and ActorInheritanceHierarchy
    - Transparently normalizes #Symbol and str keys
    - keys() returns deduplicated Symbol instances
    - Implements Actor collection protocol (at, put, atPut, includesKey, size, do, keysDo, assocsDo)
    """

    def __init__(self, vm: Any = None, hierarchy: Any = None, global_vars: Any = None):
        super().__init__()
        if vm is None:

            class _VMHolder:
                pass

            holder = _VMHolder()
            holder.hierarchy = hierarchy
            holder.global_vars = global_vars if global_vars is not None else {}
            self.vm = holder
        else:
            self.vm = vm
            if hierarchy is not None and not hasattr(self.vm, "hierarchy"):
                self.vm.hierarchy = hierarchy
            if global_vars is not None and not hasattr(self.vm, "global_vars"):
                self.vm.global_vars = global_vars
        self._storage: Dict[str, Any] = {}

    def _norm(self, key: Any) -> str:
        s = str(key)
        return s[1:] if s.startswith("#") else s

    def __getitem__(self, key: Any) -> Any:
        norm = self._norm(key)
        raw = str(key)
        if norm in self._storage:
            return self._storage[norm]
        if raw in self._storage:
            return self._storage[raw]
        if norm in self.vm.global_vars:
            return self.vm.global_vars[norm]
        if raw in self.vm.global_vars:
            return self.vm.global_vars[raw]
        if hasattr(self.vm, "hierarchy") and hasattr(self.vm.hierarchy, "get"):
            cls = self.vm.hierarchy.get(norm)
            if cls is not None:
                return cls
        if norm in self:
            return super().__getitem__(norm)
        if raw in self:
            return super().__getitem__(raw)
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        norm = self._norm(key)
        raw = str(key)
        self._storage[norm] = value
        super().__setitem__(norm, value)
        super().__setitem__(raw, value)
        self.vm.global_vars[norm] = value
        self.vm.global_vars[raw] = value
        if (
            hasattr(value, "name")
            and hasattr(self.vm, "hierarchy")
            and hasattr(self.vm.hierarchy, "register")
        ):
            from actorii.core import ActorClass

            if isinstance(value, ActorClass):
                self.vm.hierarchy.register(value)

    def __contains__(self, key: Any) -> bool:
        norm = self._norm(key)
        raw = str(key)
        if norm in self._storage or raw in self._storage:
            return True
        if norm in self.vm.global_vars or raw in self.vm.global_vars:
            return True
        if hasattr(self.vm, "hierarchy") and hasattr(self.vm.hierarchy, "get"):
            if self.vm.hierarchy.get(norm) is not None:
                return True
        return super().__contains__(norm) or super().__contains__(raw)

    def get(self, key: Any, default: Any = None) -> Any:
        val = self[key]
        return val if val is not None else default

    def keys(self) -> List[Symbol]:
        """Return deduplicated list of Symbol keys for all globals, storage, and classes."""
        all_keys: set[str] = set(self._storage.keys())
        for k in self.vm.global_vars.keys():
            if k != "Actor":
                all_keys.add(self._norm(k))
        if hasattr(self.vm, "hierarchy") and hasattr(self.vm.hierarchy, "_classes"):
            for k in self.vm.hierarchy._classes.keys():
                all_keys.add(self._norm(k))
        for k in super().keys():
            all_keys.add(self._norm(k))
        return [Symbol(k) for k in sorted(all_keys)]

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[Tuple[Symbol, Any]]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.keys())

    # Actor collection protocol methods
    def at(self, key: Any) -> Any:
        return self[key]

    def put(self, val: Any, key: Any) -> Any:
        self[key] = val
        return val

    def atPut(self, key: Any, val: Any) -> Any:
        self[key] = val
        return val

    def add(self, key: Any, val: Any) -> Any:
        self[key] = val
        return val

    def includesKey(self, key: Any) -> bool:
        return key in self

    def removeKey(self, key: Any) -> Any:
        norm = self._norm(key)
        raw = str(key)
        val = self._storage.pop(norm, None)
        self._storage.pop(raw, None)
        self.vm.global_vars.pop(norm, None)
        self.vm.global_vars.pop(raw, None)
        super().pop(norm, None)
        super().pop(raw, None)
        return val

    def remove(self, key: Any) -> Any:
        return self.removeKey(key)

    def removeAt(self, key: Any) -> Any:
        return self.removeKey(key)

    def size(self) -> int:
        return len(self)

    def isEmpty(self) -> bool:
        return len(self) == 0

    def do(self, aBlock: Callable[[Any], Any]) -> Any:
        last_val = None
        for val in self.values():
            last_val = _eval_block(aBlock, val)
        return last_val if last_val is not None else self

    def keysDo(self, aBlock: Callable[[Any], Any]) -> Any:
        last_val = None
        for k in self.keys():
            last_val = _eval_block(aBlock, k)
        return last_val

    def assocsDo(self, aBlock: Callable[[Any], Any]) -> Any:
        last_val = None
        for k, v in self.items():
            last_val = _eval_block(aBlock, [k, v])
        return last_val


__all__ = [
    "KeyedCollection",
    "Dictionary",
    "ActorDictionary",
]
