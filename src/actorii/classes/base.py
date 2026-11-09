"""1993 Whitewater Actor 4.1 Base Object and Symbol Protocols.

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/OBJECT.CLS
- historical_actor_41_source/actcls/SYMBOL.CLS
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional, Union
from actorii.core import ActorObject, ActorClass, ActorInheritanceHierarchy, Symbol


def is_kind_of(
    obj: Any, target: Any, hierarchy: Optional[ActorInheritanceHierarchy] = None
) -> bool:
    """Authentic kindOf check across Actor classes, Symbols, strings, and Python classes.

    Source: historical_actor_41_source/actcls/OBJECT.CLS:16
    Def kindOf(self, class) { ^isAncestor(class(self), class); }
    """
    if obj is None:
        return False

    target_name = ""
    if isinstance(target, (str, Symbol)):
        target_name = str(target)
        if target_name.startswith("#"):
            target_name = target_name[1:]
    elif isinstance(target, ActorClass):
        target_name = target.name
    elif isinstance(target, type):
        if isinstance(obj, target):
            return True
        target_name = target.__name__.replace("Actor", "")
    else:
        target_name = str(target)

    # 1. Direct class name match
    obj_class_name = getattr(obj, "class_name", "")
    if callable(obj_class_name):
        obj_class_name = obj_class_name()
    if not obj_class_name:
        obj_class_name = obj.__class__.__name__.replace("Actor", "")

    if obj_class_name == target_name:
        return True

    # 2. Universal Object ancestor check
    if target_name == "Object":
        return True

    # 3. Standard built-in type mappings
    if target_name == "Array":
        # Any list that is not OrderedCollection or List counts as Array in ActorII
        from actorii.classes.collections import (
            ActorArray,
            ActorOrderedCollection,
            ActorList,
        )

        return isinstance(obj, ActorArray) or (
            isinstance(obj, list)
            and not isinstance(obj, (ActorOrderedCollection, ActorList))
        )

    if target_name in ("OrderedCollection", "List"):
        from actorii.classes.collections import ActorOrderedCollection, ActorList

        if target_name == "OrderedCollection":
            return isinstance(obj, ActorOrderedCollection)
        return isinstance(obj, ActorList)

    if target_name == "IndexedCollection":
        from actorii.classes.collections import ActorIndexedCollection

        return isinstance(obj, (ActorIndexedCollection, list, tuple, str))

    if target_name == "Collection":
        from actorii.classes.collections import ActorCollection
        from actorii.classes.interval import Interval
        from actorii.classes.dictionary import Dictionary, KeyedCollection

        return isinstance(
            obj,
            (
                ActorCollection,
                Interval,
                Dictionary,
                KeyedCollection,
                list,
                tuple,
                dict,
                set,
            ),
        )

    if target_name == "KeyedCollection":
        from actorii.classes.dictionary import KeyedCollection, Dictionary

        return isinstance(obj, (KeyedCollection, Dictionary, dict))

    if target_name == "Dictionary":
        from actorii.classes.dictionary import Dictionary

        return isinstance(obj, (Dictionary, dict))

    if target_name == "Interval":
        from actorii.classes.interval import Interval

        return isinstance(obj, Interval)

    if target_name == "Point":
        from actorii.classes.geometry import Point

        return isinstance(obj, Point)

    if target_name == "Point3D":
        from actorii.classes.geometry import Point3D

        return isinstance(obj, Point3D)

    if target_name in (
        "WindowsObject",
        "Window",
        "Dialog",
        "Control",
        "Button",
        "TextField",
        "Checkbox",
    ):
        from actorii.classes.gui import (
            WindowsObject,
            Window,
            Dialog,
            Control,
            Button,
            TextField,
            Checkbox,
        )

        gui_type_map = {
            "WindowsObject": WindowsObject,
            "Window": Window,
            "Dialog": Dialog,
            "Control": Control,
            "Button": Button,
            "TextField": TextField,
            "Checkbox": Checkbox,
        }
        expected_type = gui_type_map.get(target_name)
        if expected_type and isinstance(obj, expected_type):
            return True

    if target_name == "Int":
        return isinstance(obj, int) and not isinstance(obj, bool)

    if target_name == "Float":
        return isinstance(obj, float)

    if target_name == "String":
        return isinstance(obj, str)

    if target_name == "Agent":
        from actorii.classes.agent import Agent

        if isinstance(obj, Agent):
            return True

    if target_name == "ThermoProTwin":
        from actorii.classes.thermopro import ThermoProTwin

        if isinstance(obj, ThermoProTwin):
            return True

    if target_name == "ThermometerUI":
        from actorii.classes.thermopro import ThermometerUI

        if isinstance(obj, ThermometerUI):
            return True

    # 4. Check class inheritance hierarchy if available
    if hierarchy is not None:
        curr = obj_class_name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            if curr == target_name:
                return True
            cls_obj = hierarchy.get(curr)
            if cls_obj and cls_obj._bases:
                curr = cls_obj._bases[0]
            elif cls_obj and cls_obj._parent:
                curr = cls_obj._parent.name
            else:
                break

    return False


def object_print(obj: Any) -> Any:
    """Print authentic representation to console and return obj.

    Source: historical_actor_41_source/actcls/OBJECT.CLS:136 Prim print(self):self!!
    """
    print(object_as_string(obj))
    return obj


def object_as_string(obj: Any) -> str:
    """Return authentic 1993 Actor string representation.

    Source: historical_actor_41_source/actcls/OBJECT.CLS:418 Def asString(self | strm)
    """
    if hasattr(obj, "asString") and callable(getattr(obj, "asString")):
        return obj.asString()
    if hasattr(obj, "class_name"):
        cn = obj.class_name
        name = cn() if callable(cn) else str(cn)
        if isinstance(obj, (list, tuple)):
            elems = " ".join(str(x) for x in obj)
            return f"{name}({elems})"
        return f"<{name}>"
    return str(obj)


def object_clone(obj: Any) -> Any:
    """Return shallow copy of object.

    Source: historical_actor_41_source/actcls/OBJECT.CLS:127 Prim copy(self):Object!!
    """
    return copy.copy(obj)


def object_error(obj: Any, message: Any) -> None:
    """Raise authentic Actor runtime error.

    Source: historical_actor_41_source/actcls/OBJECT.CLS:231 Def error(self, stackTop, sym | str, id)
    """
    raise RuntimeError(f"Actor Error in {obj}: {message}")


def object_init(obj: Any) -> Any:
    """Object initialization (default does nothing and returns self).

    Source: historical_actor_41_source/actcls/OBJECT.CLS:113 Prim init(self):self!!
    """
    return obj


def object_is_equal(self: Any, other: Any) -> bool:
    """Equality comparison (= in Actor)."""
    return self == other


def object_is_identical(self: Any, other: Any) -> bool:
    """Identity/equivalence comparison (== in Actor)."""
    return self is other


def object_is_not_equal(self: Any, other: Any) -> bool:
    """Non-equality comparison (<> and ~= in Actor)."""
    return self != other


__all__ = [
    "Symbol",
    "is_kind_of",
    "object_print",
    "object_as_string",
    "object_clone",
    "object_error",
    "object_init",
    "object_is_equal",
    "object_is_identical",
    "object_is_not_equal",
]
