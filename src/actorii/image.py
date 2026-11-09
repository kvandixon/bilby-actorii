"""1993 Whitewater Actor 4.1 Live Image Snapshot & Restoration Engine.

Implements whole-image persistence modeled on authentic 1993 Actor Professional 4.1
image snapshots (`ACTOR.IMG`, `ACTORAPP.CLS:snapshot`, and `OBJSTORE.ACT:storeStructureOn`).

Capabilities:
- Full serialization of `ActorInheritanceHierarchy` classes, instance variable schemas,
  and dynamically compiled bytecode methods (`CompiledMethod`, `BlockDescriptor`,
  `EarlyBoundDescriptor`, `CapturedVar`, `MethodRef`).
- Reference-preserving object graph serialization (`object_table`) supporting shared
  object identity and cyclic references across `Actor[#Key]`, `global_vars`, collections,
  custom `ActorObject` instances, and GUI hierarchies (`Window` <-> `Control`).
- Two-pass deterministic restoration onto a live `ActorVM` instance.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from actorii.classes.collections import (
    ActorArray,
    ActorList,
    ActorOrderedCollection,
    ActorSet,
)
from actorii.classes.dictionary import ActorDictionary, Dictionary
from actorii.classes.geometry import Point, Point3D, Rect
from actorii.classes.gui import (
    Button,
    Checkbox,
    Control,
    Dialog,
    TextField,
    Window,
    WindowsObject,
)
from actorii.classes.agent import Agent
from actorii.classes.thermopro import ThermoProTwin, ThermometerUI
from actorii.classes.interval import Interval
from actorii.core import ActorClass, ActorObject, Symbol
from actorii.parser import (
    BlockDescriptor,
    CapturedVar,
    EarlyBoundDescriptor,
    MethodRef,
)

IMAGE_FORMAT = "ActorII-Image-4.1"
IMAGE_VERSION = "1.0"


def _build_opcode_map() -> Dict[int, Any]:
    """Build a mapping from integer opcode value to canonical OpcodeValue instance."""
    from actorii.vm import Opcode, OpcodeValue

    mapping: Dict[int, Any] = {}
    for attr_name in dir(Opcode):
        if attr_name.startswith("_"):
            continue
        val = getattr(Opcode, attr_name)
        if isinstance(val, OpcodeValue):
            # Keep first canonical name or prefer primary names
            if int(val) not in mapping:
                mapping[int(val)] = val
    return mapping


_OPCODE_MAP: Optional[Dict[int, Any]] = None


def _get_opcode(code: int) -> Any:
    global _OPCODE_MAP
    if _OPCODE_MAP is None:
        _OPCODE_MAP = _build_opcode_map()
    if code in _OPCODE_MAP:
        return _OPCODE_MAP[code]
    from actorii.vm import OpcodeValue

    return OpcodeValue(int(code), f"OP_{int(code):02X}")


# =============================================================================
# 1. Bytecode & Constant Table Serialization
# =============================================================================


def _serialize_constant(val: Any, parent_constants: Optional[List[Any]] = None) -> Any:
    """Serialize a constant table entry (including descriptors and symbols)."""
    if val is None or isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, Symbol):
        return {"__const_type__": "Symbol", "name": val.name}
    if isinstance(val, MethodRef):
        return {
            "__const_type__": "MethodRef",
            "name": str(val),
            "arg_count": int(val.arg_count),
        }
    if isinstance(val, str):
        return val
    if isinstance(val, CapturedVar):
        return {
            "__const_type__": "CapturedVar",
            "name": val.name,
            "depth": int(val.depth),
            "outer_slot": int(val.outer_slot),
            "inner_slot": int(getattr(val, "inner_slot", val.outer_slot)),
        }
    if isinstance(val, EarlyBoundDescriptor):
        return {
            "__const_type__": "EarlyBoundDescriptor",
            "target_class": val.target_class,
            "method": val.method,
            "args": int(val.args),
        }
    if isinstance(val, BlockDescriptor):
        shares_parent = (
            parent_constants is not None and val.constants is parent_constants
        )
        return {
            "__const_type__": "BlockDescriptor",
            "name": val.name,
            "params": list(val.params),
            "locals": list(val.locals),
            "instructions": [[int(op), int(arg)] for op, arg in val.instructions],
            "uses_parent_constants": shares_parent,
            "constants": (
                []
                if shares_parent
                else [_serialize_constant(c, val.constants) for c in val.constants]
            ),
            "string_table": list(val.string_table),
            "captured_vars": [
                _serialize_constant(cv, parent_constants) for cv in val.captured_vars
            ],
            "has_nonlocal_return": bool(val.has_nonlocal_return),
        }
    if isinstance(val, tuple):
        return {
            "__const_type__": "tuple",
            "items": [_serialize_constant(x, parent_constants) for x in val],
        }
    if isinstance(val, list):
        return [_serialize_constant(x, parent_constants) for x in val]
    if isinstance(val, dict):
        return {str(k): _serialize_constant(v, parent_constants) for k, v in val.items()}
    return str(val)


def _deserialize_constant(
    data: Any, parent_constants: Optional[List[Any]] = None
) -> Any:
    """Reconstruct a constant table entry from serialized form."""
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [_deserialize_constant(x, parent_constants) for x in data]
    if isinstance(data, dict):
        ctype = data.get("__const_type__")
        if ctype == "Symbol":
            return Symbol(data["name"])
        if ctype == "MethodRef":
            return MethodRef(data["name"], int(data.get("arg_count", 0)))
        if ctype == "CapturedVar":
            return CapturedVar(
                name=data["name"],
                depth=int(data["depth"]),
                outer_slot=int(data["outer_slot"]),
                inner_slot=int(data.get("inner_slot", data["outer_slot"])),
            )
        if ctype == "EarlyBoundDescriptor":
            return EarlyBoundDescriptor(
                target_class=data["target_class"],
                method=data["method"],
                args=int(data.get("args", 0)),
            )
        if ctype == "BlockDescriptor":
            insts = [
                (_get_opcode(int(op)), int(arg))
                for op, arg in data.get("instructions", [])
            ]
            if data.get("uses_parent_constants") and parent_constants is not None:
                consts = parent_constants
            else:
                consts: List[Any] = []
                for c in data.get("constants", []):
                    consts.append(_deserialize_constant(c, consts))
            cap_vars = [
                _deserialize_constant(cv, consts)
                for cv in data.get("captured_vars", [])
            ]
            return BlockDescriptor(
                name=data.get("name", "Block"),
                params=list(data.get("params", [])),
                locals=list(data.get("locals", [])),
                instructions=insts,
                constants=consts,
                string_table=list(data.get("string_table", [])),
                captured_vars=cap_vars,
                has_nonlocal_return=bool(data.get("has_nonlocal_return", False)),
            )
        if ctype == "tuple":
            return tuple(
                _deserialize_constant(x, parent_constants)
                for x in data.get("items", [])
            )
        return {k: _deserialize_constant(v, parent_constants) for k, v in data.items()}
    return data


def serialize_compiled_method(cm: Any) -> Dict[str, Any]:
    """Serialize a CompiledMethod object to a JSON-compatible dictionary."""
    return {
        "__type__": "CompiledMethod",
        "name": cm.name,
        "class_name": cm.class_name,
        "params": list(cm.params),
        "locals": list(cm.locals),
        "instructions": [[int(op), int(arg)] for op, arg in cm.instructions],
        "constants": [_serialize_constant(c, cm.constants) for c in cm.constants],
        "string_table": list(cm.string_table),
        "receiver": getattr(cm, "receiver", "self") or "self",
        "is_action": bool(getattr(cm, "is_action", False)),
        "source": getattr(cm, "source", None),
        "synthesized": bool(getattr(cm, "synthesized", False)),
        "action_selectors": _serialize_constant(
            getattr(cm, "action_selectors", None), cm.constants
        ),
    }


def deserialize_compiled_method(data: Dict[str, Any]) -> Any:
    """Reconstruct a CompiledMethod instance from a dictionary."""
    from actorii.vm import CompiledMethod

    insts = [
        (_get_opcode(int(op)), int(arg)) for op, arg in data.get("instructions", [])
    ]
    consts: List[Any] = []
    for c in data.get("constants", []):
        consts.append(_deserialize_constant(c, consts))
    cm = CompiledMethod(
        name=data["name"],
        class_name=data.get("class_name", "Object"),
        params=list(data.get("params", [])),
        locals=list(data.get("locals", [])),
        instructions=insts,
        constants=consts,
        string_table=list(data.get("string_table", [])),
        receiver=data.get("receiver", "self"),
        is_action=bool(data.get("is_action", False)),
        source=data.get("source"),
        synthesized=bool(data.get("synthesized", False)),
    )
    if data.get("action_selectors") is not None:
        cm.action_selectors = _deserialize_constant(data["action_selectors"], consts)
    return cm


# =============================================================================
# 2. Reference-Preserving Object Graph Serializer
# =============================================================================


class _ImageSerializer:
    """Serializes live ActorVM classes, globals, and object graph with identity preservation."""

    def __init__(self, vm: Any):
        self.vm = vm
        self.object_table: Dict[str, Dict[str, Any]] = {}
        self._id_to_ref: Dict[int, str] = {}
        self._keep_alive: List[Any] = []
        self._counter: int = 0

    def _alloc_ref(self, obj: Any) -> str:
        self._counter += 1
        ref_id = f"obj_{self._counter}"
        self._id_to_ref[id(obj)] = ref_id
        self._keep_alive.append(obj)
        return ref_id

    def encode_value(self, val: Any) -> Any:
        """Encode any runtime value into a JSON-compatible representation or object ref."""
        from actorii.vm import BlockClosure, Cell, CompiledMethod

        if val is None or isinstance(val, (bool, int, float)):
            return val
        if isinstance(val, Symbol):
            return {"__val_type__": "Symbol", "name": val.name}
        if isinstance(val, str):
            return val
        if isinstance(val, ActorClass):
            return {"__class_ref__": val.name}
        if isinstance(val, type):
            return {"__builtin_type__": val.__name__}
        if isinstance(val, CompiledMethod):
            return serialize_compiled_method(val)
        if val is self.vm.global_vars.get("Actor"):
            return {"__global_actor_dict__": True}

        # Check if object is already registered in object_table
        obj_id = id(val)
        if obj_id in self._id_to_ref:
            return {"__ref__": self._id_to_ref[obj_id]}

        # Complex / mutable objects stored in object_table
        if isinstance(val, Point3D):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Point3D",
                "x": self.encode_value(val.x),
                "y": self.encode_value(val.y),
                "z": self.encode_value(val.z),
                "fields": self._encode_dict(getattr(val, "_fields", {})),
            }
            return {"__ref__": ref_id}

        if isinstance(val, Point):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Point",
                "x": self.encode_value(val.x),
                "y": self.encode_value(val.y),
                "fields": self._encode_dict(getattr(val, "_fields", {})),
            }
            return {"__ref__": ref_id}

        if isinstance(val, Rect):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Rect",
                "left": self.encode_value(val.left),
                "top": self.encode_value(val.top),
                "right": self.encode_value(val.right),
                "bottom": self.encode_value(val.bottom),
                "fields": self._encode_dict(getattr(val, "_fields", {})),
            }
            return {"__ref__": ref_id}

        if isinstance(val, Interval):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Interval",
                "start": self.encode_value(val.start),
                "stop": self.encode_value(val.stop),
                "step": self.encode_value(val.step),
            }
            return {"__ref__": ref_id}

        if isinstance(val, ActorArray):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {"type": "ActorArray", "items": []}
            self.object_table[ref_id]["items"] = [self.encode_value(x) for x in val]
            return {"__ref__": ref_id}

        if isinstance(val, ActorList):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {"type": "ActorList", "items": []}
            self.object_table[ref_id]["items"] = [self.encode_value(x) for x in val]
            return {"__ref__": ref_id}

        if isinstance(val, ActorOrderedCollection):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {"type": "ActorOrderedCollection", "items": []}
            self.object_table[ref_id]["items"] = [self.encode_value(x) for x in val]
            return {"__ref__": ref_id}

        if isinstance(val, ActorSet):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "ActorSet",
                "capacity": int(getattr(val, "capacity", 16)),
                "items": [],
            }
            self.object_table[ref_id]["items"] = [
                self.encode_value(x) for x in val.asArray()
            ]
            return {"__ref__": ref_id}

        if isinstance(val, Dictionary) and not isinstance(val, ActorDictionary):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Dictionary",
                "capacity": int(getattr(val, "capacity", 16)),
                "entries": [],
            }
            entries = []
            for k, v in val._entries.items():
                entries.append([self.encode_value(k), self.encode_value(v)])
            self.object_table[ref_id]["entries"] = entries
            return {"__ref__": ref_id}

        if isinstance(val, WindowsObject):
            return self._encode_gui_object(val)

        if isinstance(val, Cell):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "Cell",
                "value": self.encode_value(val.value),
            }
            return {"__ref__": ref_id}

        if isinstance(val, BlockClosure):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {
                "type": "BlockClosure",
                "descriptor": _serialize_constant(val.descriptor),
                "receiver": self.encode_value(val.receiver),
                "captured_cells": {
                    str(slot): self.encode_value(cell)
                    for slot, cell in val.captured_cells.items()
                },
            }
            return {"__ref__": ref_id}

        if isinstance(val, ThermoProTwin):
            ref_id = self._alloc_ref(val)
            cls_name = getattr(val, "_class_name", "ThermoProTwin")
            inst_methods = {}
            for m_name, m_fn in getattr(val, "_methods", {}).items():
                if isinstance(m_fn, CompiledMethod):
                    inst_methods[m_name] = serialize_compiled_method(m_fn)
            self.object_table[ref_id] = {
                "type": "ThermoProTwin",
                "class_name": cls_name,
                "obj_id": str(getattr(val, "_obj_id", ref_id)),
                "fields": {},
                "instance_methods": inst_methods,
            }
            self.object_table[ref_id]["fields"] = self._encode_dict(
                getattr(val, "_fields", {})
            )
            return {"__ref__": ref_id}

        if isinstance(val, Agent):
            ref_id = self._alloc_ref(val)
            cls_name = getattr(val, "_class_name", "Agent")
            inst_methods = {}
            for m_name, m_fn in getattr(val, "_methods", {}).items():
                if isinstance(m_fn, CompiledMethod):
                    inst_methods[m_name] = serialize_compiled_method(m_fn)
            self.object_table[ref_id] = {
                "type": "Agent",
                "class_name": cls_name,
                "obj_id": str(getattr(val, "_obj_id", ref_id)),
                "fields": {},
                "instance_methods": inst_methods,
            }
            self.object_table[ref_id]["fields"] = self._encode_dict(
                getattr(val, "_fields", {})
            )
            return {"__ref__": ref_id}

        if isinstance(val, ActorObject):
            ref_id = self._alloc_ref(val)
            cls_name = getattr(val, "_class_name", val.__class__.__name__)
            inst_methods = {}
            for m_name, m_fn in getattr(val, "_methods", {}).items():
                if isinstance(m_fn, CompiledMethod):
                    # Check if this method is distinct from or overrides class method
                    inst_methods[m_name] = serialize_compiled_method(m_fn)
            self.object_table[ref_id] = {
                "type": "ActorObject",
                "class_name": cls_name,
                "obj_id": str(getattr(val, "_obj_id", ref_id)),
                "fields": {},
                "instance_methods": inst_methods,
            }
            self.object_table[ref_id]["fields"] = self._encode_dict(
                getattr(val, "_fields", {})
            )
            return {"__ref__": ref_id}

        if isinstance(val, list):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {"type": "list", "items": []}
            self.object_table[ref_id]["items"] = [self.encode_value(x) for x in val]
            return {"__ref__": ref_id}

        if isinstance(val, tuple):
            return {
                "__val_type__": "tuple",
                "items": [self.encode_value(x) for x in val],
            }

        if isinstance(val, dict):
            ref_id = self._alloc_ref(val)
            self.object_table[ref_id] = {"type": "dict", "entries": []}
            self.object_table[ref_id]["entries"] = [
                [self.encode_value(k), self.encode_value(v)] for k, v in val.items()
            ]
            return {"__ref__": ref_id}

        return str(val)

    def _encode_dict(self, d: Dict[Any, Any]) -> Dict[str, Any]:
        return {str(k): self.encode_value(v) for k, v in d.items()}

    def _encode_gui_object(self, obj: WindowsObject) -> Dict[str, str]:
        from actorii.vm import CompiledMethod

        ref_id = self._alloc_ref(obj)
        cls_name = getattr(obj, "_class_name", obj.__class__.__name__)
        rect_obj = getattr(obj, "rect", None)
        rect_tuple = (
            [rect_obj.left, rect_obj.top, rect_obj.right, rect_obj.bottom]
            if rect_obj is not None
            else [0, 0, 100, 30]
        )

        inst_methods = {}
        for m_name, m_fn in getattr(obj, "_methods", {}).items():
            if isinstance(m_fn, CompiledMethod):
                inst_methods[m_name] = serialize_compiled_method(m_fn)

        entry: Dict[str, Any] = {
            "type": "GUIObject",
            "gui_class": obj.__class__.__name__,
            "class_name": cls_name,
            "obj_id": str(getattr(obj, "_obj_id", ref_id)),
            "title": getattr(obj, "title", ""),
            "id": self.encode_value(getattr(obj, "id", None)),
            "identity": (
                getattr(obj, "identity").name
                if getattr(obj, "identity", None) is not None
                else None
            ),
            "rect": rect_tuple,
            "visible": bool(getattr(obj, "visible", True)),
            "enabled": bool(getattr(obj, "enabled", True)),
            "fields": {},
            "instance_methods": inst_methods,
        }
        self.object_table[ref_id] = entry

        # Populate class-specific attributes and references
        entry["fields"] = self._encode_dict(getattr(obj, "_fields", {}))
        entry["parent"] = self.encode_value(getattr(obj, "parent", None))

        if isinstance(obj, Window):
            entry["modal"] = bool(getattr(obj, "modal", False))
            entry["children"] = [
                self.encode_value(c) for c in getattr(obj, "children", [])
            ]
            entry["focus_child"] = self.encode_value(getattr(obj, "focus_child", None))
            entry["action_handlers"] = [
                [str(k[0]), str(k[1]), str(v)]
                for k, v in getattr(obj, "action_handlers", {}).items()
            ]
            if isinstance(obj, Dialog):
                entry["is_modal"] = bool(getattr(obj, "is_modal", False))
                entry["result_code"] = self.encode_value(
                    getattr(obj, "result_code", None)
                )
                entry["ok_button"] = self.encode_value(getattr(obj, "ok_button", None))
                entry["cancel_button"] = self.encode_value(
                    getattr(obj, "cancel_button", None)
                )

        if isinstance(obj, Control):
            entry["data"] = self.encode_value(getattr(obj, "data", None))
            entry["selection"] = self.encode_value(getattr(obj, "selection", None))
            if isinstance(obj, Button):
                entry["label"] = getattr(obj, "label", "")
                act_sel = getattr(obj, "action_selector", None)
                entry["action_selector"] = str(act_sel) if act_sel is not None else None
                entry["is_default"] = bool(getattr(obj, "is_default", False))
                entry["pressed"] = bool(getattr(obj, "pressed", False))
            if isinstance(obj, Checkbox):
                entry["checked"] = bool(getattr(obj, "checked", False))
            if isinstance(obj, TextField):
                entry["text"] = getattr(obj, "text", "")
                entry["max_length"] = getattr(obj, "max_length", None)
                entry["read_only"] = bool(getattr(obj, "read_only", False))

        return {"__ref__": ref_id}

    def serialize_vm(self) -> Dict[str, Any]:
        """Serialize the entire live ActorVM state into a portable snapshot dictionary."""
        from actorii.vm import CompiledMethod

        classes_data: List[Dict[str, Any]] = []
        for cls_name, cls in self.vm.hierarchy._classes.items():
            compiled_methods: Dict[str, Any] = {}
            for m_name, m_fn in cls._class_methods.items():
                if isinstance(m_fn, CompiledMethod):
                    compiled_methods[m_name] = serialize_compiled_method(m_fn)
            for m_name, m_fn in cls._methods.items():
                if isinstance(m_fn, CompiledMethod) and m_name not in compiled_methods:
                    compiled_methods[m_name] = serialize_compiled_method(m_fn)

            class_fields = self._encode_dict(getattr(cls, "_class_fields", {}))
            action_handlers = [
                [str(k[0]), str(k[1]), str(v)]
                for k, v in getattr(cls, "action_handlers", {}).items()
            ]
            classes_data.append(
                {
                    "name": cls_name,
                    "bases": list(cls._bases),
                    "ivars": list(getattr(cls, "ivars", getattr(cls, "_ivars", []))),
                    "jit_synthesis": bool(getattr(cls, "jit_synthesis", False)),
                    "compiled_methods": compiled_methods,
                    "class_fields": class_fields,
                    "action_handlers": action_handlers,
                }
            )

        actor_dict = self.vm.global_vars.get("Actor", {})
        actor_entries: List[List[Any]] = []
        if hasattr(actor_dict, "_entries"):
            items = list(actor_dict._entries.items())
        elif isinstance(actor_dict, dict):
            items = list(actor_dict.items())
        else:
            items = []

        for k, v in items:
            if callable(v) and not isinstance(v, (ActorObject, ActorClass, type)):
                continue
            actor_entries.append([self.encode_value(k), self.encode_value(v)])

        global_entries: Dict[str, Any] = {}
        for k, v in self.vm.global_vars.items():
            if k == "Actor":
                continue
            if callable(v) and not isinstance(v, (ActorObject, ActorClass, type)):
                continue
            global_entries[str(k)] = self.encode_value(v)

        return {
            "format": IMAGE_FORMAT,
            "version": IMAGE_VERSION,
            "timestamp": time.time(),
            "current_compilation_class": getattr(
                self.vm, "_current_compilation_class", ""
            ),
            "jit_synthesis_enabled": bool(
                getattr(self.vm, "jit_synthesis_enabled", False)
            ),
            "synthesized_methods": list(getattr(self.vm, "synthesized_methods", [])),
            "classes": classes_data,
            "object_table": self.object_table,
            "actor_entries": actor_entries,
            "global_entries": global_entries,
        }


# =============================================================================
# 3. Two-Pass Deterministic Image Deserializer
# =============================================================================


class _ImageDeserializer:
    """Restores a serialized ActorII image snapshot onto a live ActorVM."""

    _BUILTIN_TYPES = {
        "ActorSet": ActorSet,
        "ActorArray": ActorArray,
        "ActorOrderedCollection": ActorOrderedCollection,
        "ActorList": ActorList,
        "Dictionary": Dictionary,
        "Interval": Interval,
        "Point": Point,
        "Point3D": Point3D,
        "Rect": Rect,
        "Window": Window,
        "Dialog": Dialog,
        "Button": Button,
        "Checkbox": Checkbox,
        "TextField": TextField,
        "Control": Control,
        "WindowsObject": WindowsObject,
        "ActorObject": ActorObject,
        "Symbol": Symbol,
    }

    def __init__(self, vm: Any, data: Dict[str, Any]):
        self.vm = vm
        self.data = data
        self.instances: Dict[str, Any] = {}

    def restore(self, clear_existing: bool = True) -> Dict[str, Any]:
        fmt = self.data.get("format")
        if fmt != IMAGE_FORMAT:
            raise ValueError(
                f"Invalid ActorII image format: expected '{IMAGE_FORMAT}', got '{fmt}'"
            )

        if clear_existing:
            self.vm.hierarchy._classes.clear()
            self.vm.hierarchy._root = None
            self.vm.global_vars.clear()
            self.vm.global_vars["Actor"] = ActorDictionary(self.vm)
            self.vm._init_builtins()

        # Step 1: Restore class definitions & attach compiled methods
        classes_data = self.data.get("classes", [])
        # First pass of classes: ensure all ActorClass shells exist in hierarchy
        for cdata in classes_data:
            cname = cdata["name"]
            bases = tuple(cdata.get("bases", []))
            cls = self.vm.hierarchy.get(cname)
            if cls is None:
                cls = ActorClass(cname, bases=bases)
                self.vm.hierarchy.register(cls)
                self.vm.global_vars[cname] = cls
                self.vm.global_vars["Actor"][cname] = cls
            else:
                cls._bases = bases
            cls.ivars = list(cdata.get("ivars", []))
            cls.jit_synthesis = bool(cdata.get("jit_synthesis", False))
            if cdata.get("action_handlers"):
                if not hasattr(cls, "action_handlers"):
                    cls.action_handlers = {}
                for child_k, ev_k, m_name in cdata["action_handlers"]:
                    cls.action_handlers[(child_k, ev_k)] = m_name

        # Attach compiled methods onto classes
        for cdata in classes_data:
            cname = cdata["name"]
            cls = self.vm.hierarchy.get(cname)
            if cls is None:
                continue
            for m_name, cm_data in cdata.get("compiled_methods", {}).items():
                cm = deserialize_compiled_method(cm_data)
                cls.add_method(m_name, cm)

        # Step 2: Two-pass Object Table reconstruction
        obj_table = self.data.get("object_table", {})
        for ref_id, entry in obj_table.items():
            self.instances[ref_id] = self._create_shell(ref_id, entry)

        for ref_id, entry in obj_table.items():
            self._populate_shell(self.instances[ref_id], entry)

        # Populate class_fields now that object_table instances exist
        for cdata in classes_data:
            cls = self.vm.hierarchy.get(cdata["name"])
            if cls is not None and cdata.get("class_fields"):
                for fk, fv in cdata["class_fields"].items():
                    cls._class_fields[fk] = self.decode_value(fv)

        # Step 3: Restore Actor dictionary & VM globals
        actor_dict = self.vm.global_vars["Actor"]
        if clear_existing:
            actor_dict.clear()

        for k_enc, v_enc in self.data.get("actor_entries", []):
            key = self.decode_value(k_enc)
            val = self.decode_value(v_enc)
            actor_dict[key] = val

        for gk, gv_enc in self.data.get("global_entries", {}).items():
            self.vm.global_vars[gk] = self.decode_value(gv_enc)

        self.vm._current_compilation_class = self.data.get(
            "current_compilation_class", ""
        )
        self.vm.jit_synthesis_enabled = bool(
            self.data.get("jit_synthesis_enabled", False)
        )
        self.vm.synthesized_methods = list(self.data.get("synthesized_methods", []))

        return {
            "classes_count": len(self.vm.hierarchy._classes),
            "objects_count": len(self.instances),
            "actor_keys_count": len(actor_dict),
        }

    def _create_shell(self, ref_id: str, entry: Dict[str, Any]) -> Any:
        from actorii.vm import BlockClosure, Cell

        otype = entry.get("type")
        if otype == "Point3D":
            return Point3D(0, 0, 0)
        if otype == "Point":
            return Point(0, 0)
        if otype == "Rect":
            return Rect(0, 0, 0, 0)
        if otype == "Interval":
            return Interval(
                int(self.decode_value(entry.get("start", 0))),
                int(self.decode_value(entry.get("stop", 0))),
                int(self.decode_value(entry.get("step", 1))),
            )
        if otype == "ActorArray":
            return ActorArray(0)
        if otype == "ActorList":
            return ActorList()
        if otype == "ActorOrderedCollection":
            return ActorOrderedCollection()
        if otype == "ActorSet":
            return ActorSet(int(entry.get("capacity", 16)))
        if otype == "Dictionary":
            return Dictionary(int(entry.get("capacity", 16)))
        if otype == "GUIObject":
            gcls_name = entry.get("gui_class", "WindowsObject")
            r = entry.get("rect", [0, 0, 100, 30])
            rect = Rect(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            ident_str = entry.get("identity")
            ident = Symbol(ident_str) if ident_str is not None else None
            oid = entry.get("obj_id", ref_id)
            if gcls_name == "ThermometerUI":
                inst: WindowsObject = ThermometerUI(
                    title=entry.get("title", "ThermoPro TP358W Digital Twin"),
                    rect=rect,
                    obj_id=oid,
                    class_name=entry.get("class_name", "ThermometerUI"),
                    vm=self.vm,
                )
            elif gcls_name == "Dialog":
                inst = Dialog(
                    title=entry.get("title", "Dialog"),
                    rect=rect,
                    obj_id=oid,
                    modal=bool(entry.get("modal", False)),
                )
            elif gcls_name == "Window":
                inst = Window(
                    title=entry.get("title", "Window"),
                    rect=rect,
                    obj_id=oid,
                    identity=ident,
                )
            elif gcls_name == "Checkbox":
                inst = Checkbox(
                    label=entry.get("label", entry.get("title", "")),
                    checked=bool(entry.get("checked", False)),
                    rect=rect,
                    identity=ident,
                    obj_id=oid,
                )
            elif gcls_name == "Button":
                act_sel = entry.get("action_selector")
                inst = Button(
                    label=entry.get("label", entry.get("title", "OK")),
                    action_selector=Symbol(act_sel) if act_sel else None,
                    rect=rect,
                    identity=ident,
                    is_default=bool(entry.get("is_default", False)),
                    obj_id=oid,
                )
            elif gcls_name == "TextField":
                inst = TextField(
                    text=entry.get("text", ""),
                    rect=rect,
                    identity=ident,
                    max_length=int(entry.get("max_length") or 255),
                    obj_id=oid,
                )
                if "read_only" in entry:
                    inst.read_only = bool(entry.get("read_only", False))
            elif gcls_name == "Control":
                inst = Control(
                    identity=ident,
                    rect=rect,
                    title=entry.get("title", ""),
                    obj_id=oid,
                )
            else:
                inst = WindowsObject(
                    title=entry.get("title", ""),
                    identity=ident,
                    obj_id=oid,
                )
            cls_name = entry.get("class_name", gcls_name)
            inst._class_name = cls_name
            cls = self.vm.hierarchy.get(cls_name)
            if cls is not None:
                inst.inherit_from(cls)
            return inst

        if otype == "Cell":
            return Cell(None)
        if otype == "BlockClosure":
            desc = _deserialize_constant(entry.get("descriptor"))
            return BlockClosure(descriptor=desc)
        if otype == "ThermoProTwin":
            cls_name = entry.get("class_name", "ThermoProTwin")
            oid = entry.get("obj_id", ref_id)
            tp_obj = ThermoProTwin(obj_id=oid, class_name=cls_name, vm=self.vm)
            cls = self.vm.hierarchy.get(cls_name)
            if cls is not None:
                tp_obj.inherit_from(cls)
            return tp_obj

        if otype == "Agent":
            cls_name = entry.get("class_name", "Agent")
            oid = entry.get("obj_id", ref_id)
            agent_obj = Agent(obj_id=oid, class_name=cls_name, vm=self.vm)
            cls = self.vm.hierarchy.get(cls_name)
            if cls is not None:
                agent_obj.inherit_from(cls)
            return agent_obj

        if otype == "ActorObject":
            cls_name = entry.get("class_name", "Object")
            oid = entry.get("obj_id", ref_id)
            inst_obj = ActorObject(obj_id=oid)
            inst_obj._class_name = cls_name
            cls = self.vm.hierarchy.get(cls_name)
            if cls is not None:
                inst_obj.inherit_from(cls)
            return inst_obj
        if otype == "list":
            return []
        if otype == "dict":
            return {}
        return ActorObject(obj_id=ref_id)

    def _populate_shell(self, obj: Any, entry: Dict[str, Any]) -> None:
        otype = entry.get("type")

        if otype == "Point3D":
            obj.setX(self.decode_value(entry.get("x", 0)))
            obj.setY(self.decode_value(entry.get("y", 0)))
            obj.setZ(self.decode_value(entry.get("z", 0)))
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            return

        if otype == "Point":
            obj.setX(self.decode_value(entry.get("x", 0)))
            obj.setY(self.decode_value(entry.get("y", 0)))
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            return

        if otype == "Rect":
            obj.left = self.decode_value(entry.get("left", 0))
            obj.top = self.decode_value(entry.get("top", 0))
            obj.right = self.decode_value(entry.get("right", 0))
            obj.bottom = self.decode_value(entry.get("bottom", 0))
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            return

        if otype in ("ActorArray", "ActorList", "ActorOrderedCollection", "list"):
            items = [self.decode_value(x) for x in entry.get("items", [])]
            obj.clear()
            obj.extend(items)
            return

        if otype == "ActorSet":
            for x in entry.get("items", []):
                obj.add(self.decode_value(x))
            return

        if otype == "Dictionary":
            for k_enc, v_enc in entry.get("entries", []):
                obj.atPut(self.decode_value(k_enc), self.decode_value(v_enc))
            return

        if otype == "dict":
            for k_enc, v_enc in entry.get("entries", []):
                obj[self.decode_value(k_enc)] = self.decode_value(v_enc)
            return

        if otype == "Cell":
            obj.value = self.decode_value(entry.get("value"))
            return

        if otype == "BlockClosure":
            obj.receiver = self.decode_value(entry.get("receiver"))
            for slot_str, cell_enc in entry.get("captured_cells", {}).items():
                obj.captured_cells[int(slot_str)] = self.decode_value(cell_enc)
            return

        if otype == "GUIObject":
            obj.id = self.decode_value(entry.get("id"))
            obj.visible = bool(entry.get("visible", True))
            obj.enabled = bool(entry.get("enabled", True))
            obj.parent = self.decode_value(entry.get("parent"))
            obj.vm = self.vm
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            for m_name, cm_data in entry.get("instance_methods", {}).items():
                obj.set_method(m_name, deserialize_compiled_method(cm_data))

            if isinstance(obj, Window):
                obj.children = [
                    self.decode_value(c) for c in entry.get("children", [])
                ]
                for child in obj.children:
                    if hasattr(child, "parent") and child.parent is None:
                        child.parent = obj
                obj.focus_child = self.decode_value(entry.get("focus_child"))
                for child_k, ev_k, m_name in entry.get("action_handlers", []):
                    obj.action_handlers[(child_k, ev_k)] = m_name
                if isinstance(obj, Dialog):
                    obj.is_modal = bool(entry.get("is_modal", False))
                    obj.result_code = self.decode_value(entry.get("result_code"))
                    obj.ok_button = self.decode_value(entry.get("ok_button"))
                    obj.cancel_button = self.decode_value(entry.get("cancel_button"))
                if isinstance(obj, ThermometerUI):
                    for ch in obj.children:
                        ident_s = str(getattr(ch, "identity", getattr(ch, "id", ""))).lstrip("#")
                        if ident_s == "tempField" and isinstance(ch, TextField):
                            obj.temp_field = ch
                        elif ident_s == "humidityField" and isinstance(ch, TextField):
                            obj.humidity_field = ch
                        elif ident_s == "comfortField" and isinstance(ch, TextField):
                            obj.comfort_field = ch
                        elif ident_s == "statsField" and isinstance(ch, TextField):
                            obj.stats_field = ch
                        elif ident_s == "btnSync" and isinstance(ch, Button):
                            obj.btn_sync = ch
                            ch.target = obj
                            ch.callback = lambda _b, o=obj: o.onSyncClicked()
                        elif ident_s == "btnHistory" and isinstance(ch, Button):
                            obj.btn_history = ch
                            ch.target = obj
                            ch.callback = lambda _b, o=obj: o.onHistoryClicked()
                        elif ident_s == "chkFahrenheit" and isinstance(ch, Checkbox):
                            obj.chk_unit = ch

            if isinstance(obj, Control):
                obj.data = self.decode_value(entry.get("data"))
                obj.selection = self.decode_value(entry.get("selection"))
                if isinstance(obj, Button):
                    obj.pressed = bool(entry.get("pressed", False))
                if isinstance(obj, Checkbox):
                    obj.checked = bool(entry.get("checked", False))
                    obj.data = obj.checked
                if isinstance(obj, TextField):
                    obj.text = str(entry.get("text", ""))
                    obj.data = obj.text
            return

        if otype == "ThermoProTwin":
            obj.vm = self.vm
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            if isinstance(obj._fields.get("history"), ActorOrderedCollection):
                obj._history = obj._fields["history"]
            if isinstance(obj._fields.get("observers"), ActorOrderedCollection):
                obj._observers = obj._fields["observers"]
            for m_name, cm_data in entry.get("instance_methods", {}).items():
                obj.set_method(m_name, deserialize_compiled_method(cm_data))
            return

        if otype == "Agent":
            obj.vm = self.vm
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            if isinstance(obj._fields.get("memory"), Dictionary):
                obj._memory = obj._fields["memory"]
            if isinstance(obj._fields.get("history"), ActorOrderedCollection):
                obj._history = obj._fields["history"]
            if isinstance(obj._fields.get("tools"), Dictionary):
                obj._tools = obj._fields["tools"]
            for m_name, cm_data in entry.get("instance_methods", {}).items():
                obj.set_method(m_name, deserialize_compiled_method(cm_data))
            return

        if otype == "ActorObject":
            for k, v in entry.get("fields", {}).items():
                obj._fields[k] = self.decode_value(v)
            for m_name, cm_data in entry.get("instance_methods", {}).items():
                obj.set_method(m_name, deserialize_compiled_method(cm_data))
            return

    def decode_value(self, data: Any) -> Any:
        if data is None or isinstance(data, (bool, int, float, str)):
            return data
        if isinstance(data, list):
            return [self.decode_value(x) for x in data]
        if isinstance(data, dict):
            if "__ref__" in data:
                return self.instances.get(data["__ref__"])
            if "__class_ref__" in data:
                cname = data["__class_ref__"]
                return self.vm.hierarchy.get(cname)
            if "__builtin_type__" in data:
                tname = data["__builtin_type__"]
                return self._BUILTIN_TYPES.get(tname, ActorObject)
            if "__global_actor_dict__" in data:
                return self.vm.global_vars.get("Actor")
            if data.get("__type__") == "CompiledMethod":
                return deserialize_compiled_method(data)
            vtype = data.get("__val_type__")
            if vtype == "Symbol":
                return Symbol(data["name"])
            if vtype == "tuple":
                return tuple(self.decode_value(x) for x in data.get("items", []))
            return {k: self.decode_value(v) for k, v in data.items()}
        return data


# =============================================================================
# 4. Public Image Snapshot & Restore API
# =============================================================================


def snapshot_to_dict(vm: Any) -> Dict[str, Any]:
    """Capture the complete state of `vm` into a JSON-serializable dictionary."""
    serializer = _ImageSerializer(vm)
    return serializer.serialize_vm()


def restore_from_dict(
    vm: Any, snapshot_data: Dict[str, Any], clear_existing: bool = True
) -> Dict[str, Any]:
    """Restore the state of `vm` from a previously captured snapshot dictionary."""
    deserializer = _ImageDeserializer(vm, snapshot_data)
    return deserializer.restore(clear_existing=clear_existing)


def save_image(vm: Any, path: Union[str, Path]) -> str:
    """Save the live `ActorVM` image to disk as an atomic `.img` JSON snapshot.

    Returns the resolved absolute file path.
    """
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_to_dict(vm)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    os.replace(tmp_path, target_path)
    return str(target_path)


def load_image(
    vm: Any, path: Union[str, Path], clear_existing: bool = True
) -> Dict[str, Any]:
    """Load an `.img` snapshot file from disk into `vm`.

    Returns summary statistics dictionary (`path`, `classes_count`, `objects_count`).
    """
    target_path = Path(path).expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"ActorII image file not found: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    stats = restore_from_dict(vm, snapshot, clear_existing=clear_existing)
    stats["path"] = str(target_path)
    return stats
