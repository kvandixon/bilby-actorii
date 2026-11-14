"""ActorII 1993 Whitewater Actor Standard Class Library Package.

Exports all standard library classes and provides the canonical hierarchy
registration function `register_classes`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from actorii.core import (
    ActorClass,
    ActorInheritanceHierarchy,
    ActorObject,
    ActorMessage,
    Symbol,
)
from actorii.classes.base import (
    is_kind_of,
    object_as_string,
    object_clone,
    object_error,
    object_init,
    object_is_equal,
    object_is_identical,
    object_is_not_equal,
    object_print,
)
from actorii.classes.collections import (
    ActorCollection,
    ActorIndexedCollection,
    ActorArray,
    ActorOrderedCollection,
    ActorList,
    ActorSet,
    Collection,
    IndexedCollection,
    Array,
    OrderedCollection,
    ListClass,
    Set,
)
from actorii.classes.dictionary import (
    KeyedCollection,
    Dictionary,
    ActorDictionary,
)
from actorii.classes.interval import Interval
from actorii.classes.geometry import Point, Point3D, Rect
from actorii.classes.gui import (
    WindowsObject,
    Window,
    Dialog,
    Control,
    Button,
    TextField,
    Checkbox,
)
from actorii.classes.agent import Agent
from actorii.classes.thermopro import ThermoProTwin, ThermometerUI
from actorii.classes.concurrency import (
    ActorFuture,
    ActorChannel,
    ActorMutex,
    AtomicCounter,
    TaskPool,
    ActorThreadPoolTwin,
    CONCURRENCY_BRIDGE_CLASSES,
    wrap_for_actor,
)


def register_classes(
    hierarchy: ActorInheritanceHierarchy,
    actor_dict: Optional[Any] = None,
    global_vars: Optional[Dict[str, Any]] = None,
) -> None:
    """Register all 1993 Actor standard library classes into hierarchy and globals."""
    # 1. Authentic 1993 Actor single-inheritance hierarchy definitions
    specs = [
        ("Object", ()),
        ("Collection", ("Object",)),
        ("IndexedCollection", ("Collection",)),
        ("Array", ("IndexedCollection",)),
        ("OrderedCollection", ("IndexedCollection",)),
        ("List", ("OrderedCollection",)),
        ("KeyedCollection", ("Collection",)),
        ("Dictionary", ("KeyedCollection",)),
        ("Interval", ("Collection",)),
        ("Point", ("Object",)),
        ("Point3D", ("Point",)),
        ("Rect", ("Object",)),
        ("WindowsObject", ("Object",)),
        ("Window", ("WindowsObject",)),
        ("Dialog", ("Window",)),
        ("Control", ("WindowsObject",)),
        ("Button", ("Control",)),
        ("TextField", ("Control",)),
        ("Checkbox", ("Button",)),
        ("Int", ("Object",)),
        ("Float", ("Object",)),
        ("String", ("IndexedCollection",)),
        ("Set", ("Collection",)),
        ("Agent", ("Object",)),
        ("ThermoProTwin", ("Object",)),
        ("ThermometerUI", ("Window",)),
        # Opt-in multithreaded concurrency primitives (actorii.classes.concurrency)
        ("Future", ("Object",)),
        ("Channel", ("Object",)),
        ("Mutex", ("Object",)),
        ("AtomicCounter", ("Object",)),
        ("ThreadPool", ("Object",)),
    ]

    for name, bases in specs:
        cls = hierarchy.get(name)
        if cls is None:
            cls = ActorClass(name, bases=bases)
            hierarchy.register(cls)
        else:
            cls._bases = bases
        if global_vars is not None:
            global_vars[name] = cls
        if actor_dict is not None:
            actor_dict[name] = cls

    if actor_dict is not None:
        actor_dict["Set"] = ActorSet
        try:
            actor_dict[Symbol("Set")] = ActorSet
        except Exception:
            pass

    # 2. Attach concrete constructors (_default_constructor) to ActorClass nodes
    set_cls = hierarchy.get("Set")
    if set_cls:
        set_cls._default_constructor = lambda *args: ActorSet(*args)
        set_cls.set_method("new", lambda self, *args: ActorSet(*args))

    arr_cls = hierarchy.get("Array")
    if arr_cls:
        arr_cls._default_constructor = lambda *args: ActorArray(args[0] if args else 0)
        arr_cls.set_method("new", lambda self, size=0: ActorArray(size))

    ord_cls = hierarchy.get("OrderedCollection")
    if ord_cls:
        ord_cls._default_constructor = lambda *args: ActorOrderedCollection(
            args[0] if args else None
        )
        ord_cls.set_method("new", lambda self, *args: ActorOrderedCollection())

    lst_cls = hierarchy.get("List")
    if lst_cls:
        lst_cls._default_constructor = lambda *args: ActorList(
            args[0] if args else None
        )
        lst_cls.set_method("new", lambda self, *args: ActorList())

    dict_cls = hierarchy.get("Dictionary")
    if dict_cls:
        dict_cls._default_constructor = lambda *args: Dictionary(
            int(args[0]) if args else 16
        )
        dict_cls.set_method("new", lambda self, cap=16: Dictionary(cap))

    itv_cls = hierarchy.get("Interval")
    if itv_cls:
        itv_cls._default_constructor = lambda *args: Interval(
            args[0] if len(args) > 0 else 0,
            args[1] if len(args) > 1 else 0,
            args[2] if len(args) > 2 else 1,
        )
        itv_cls.set_method(
            "new", lambda self, start=0, stop=0, step=1: Interval(start, stop, step)
        )

    pt_cls = hierarchy.get("Point")
    if pt_cls:
        pt_cls._default_constructor = lambda *args: Point(
            args[0] if len(args) > 0 else 0,
            args[1] if len(args) > 1 else 0,
        )
        pt_cls.set_method("new", lambda self, x=0, y=0: Point(x, y))

    p3d_cls = hierarchy.get("Point3D")
    if p3d_cls:
        p3d_cls._default_constructor = lambda *args: Point3D(
            args[0] if len(args) > 0 else 0,
            args[1] if len(args) > 1 else 0,
            args[2] if len(args) > 2 else 0,
        )
        p3d_cls.set_method("new", lambda self, x=0, y=0, z=0: Point3D(x, y, z))

    rect_cls = hierarchy.get("Rect")
    if rect_cls:
        rect_cls._default_constructor = lambda *args: Rect(*args)
        rect_cls.set_method("new", lambda self, *args: Rect(*args))

    win_cls = hierarchy.get("Window")
    if win_cls:
        win_cls._default_constructor = lambda *args, **kw: Window(*args, **kw)
        win_cls.set_method("new", lambda self, *args, **kw: Window(*args, **kw))

    dlg_cls = hierarchy.get("Dialog")
    if dlg_cls:
        dlg_cls._default_constructor = lambda *args, **kw: Dialog(*args, **kw)
        dlg_cls.set_method("new", lambda self, *args, **kw: Dialog(*args, **kw))

    ctrl_cls = hierarchy.get("Control")
    if ctrl_cls:
        ctrl_cls._default_constructor = lambda *args, **kw: Control(*args, **kw)
        ctrl_cls.set_method("new", lambda self, *args, **kw: Control(*args, **kw))

    btn_cls = hierarchy.get("Button")
    if btn_cls:
        btn_cls._default_constructor = lambda *args, **kw: Button(*args, **kw)
        btn_cls.set_method("new", lambda self, *args, **kw: Button(*args, **kw))

    txt_cls = hierarchy.get("TextField")
    if txt_cls:
        txt_cls._default_constructor = lambda *args, **kw: TextField(*args, **kw)
        txt_cls.set_method("new", lambda self, *args, **kw: TextField(*args, **kw))

    chk_cls = hierarchy.get("Checkbox")
    if chk_cls:
        chk_cls._default_constructor = lambda *args, **kw: Checkbox(*args, **kw)
        chk_cls.set_method("new", lambda self, *args, **kw: Checkbox(*args, **kw))

    # 3. Attach standard Object root methods
    obj_cls = hierarchy.get("Object")
    if obj_cls:
        obj_cls.add_method("print", object_print)
        obj_cls.add_method("asString", object_as_string)
        obj_cls.add_method("clone", object_clone)
        obj_cls.add_method("copy", object_clone)
        obj_cls.add_method("error", object_error)
        obj_cls.add_method("init", object_init)
        obj_cls.add_method(
            "isKindOf", lambda self, target: is_kind_of(self, target, hierarchy)
        )
        obj_cls.add_method(
            "kindOf", lambda self, target: is_kind_of(self, target, hierarchy)
        )
        obj_cls.add_method("==", object_is_identical)
        obj_cls.add_method("!=", object_is_not_equal)
        obj_cls.add_method("=", object_is_equal)
        obj_cls.add_method("<>", object_is_not_equal)

    # 4. Attach Collection methods
    col_cls = hierarchy.get("Collection")
    if col_cls:
        col_cls.add_method(
            "size", lambda self: self.size() if hasattr(self, "size") else len(self)
        )
        col_cls.add_method(
            "isEmpty",
            lambda self: (
                self.isEmpty() if hasattr(self, "isEmpty") else (len(self) == 0)
            ),
        )
        col_cls.add_method(
            "do", lambda self, block: self.do(block) if hasattr(self, "do") else None
        )
        col_cls.add_method(
            "collect",
            lambda self, block: (
                self.collect(block) if hasattr(self, "collect") else None
            ),
        )
        col_cls.add_method(
            "select",
            lambda self, block: self.select(block) if hasattr(self, "select") else None,
        )
        col_cls.add_method(
            "extract",
            lambda self, block: self.select(block) if hasattr(self, "select") else None,
        )
        col_cls.add_method(
            "reject",
            lambda self, block: self.reject(block) if hasattr(self, "reject") else None,
        )
        col_cls.add_method(
            "asArray",
            lambda self: (
                self.asArray() if hasattr(self, "asArray") else ActorArray(self)
            ),
        )
        col_cls.add_method(
            "asOrderedCollection",
            lambda self: (
                self.asOrderedCollection()
                if hasattr(self, "asOrderedCollection")
                else ActorOrderedCollection(self)
            ),
        )
        col_cls.add_method(
            "first", lambda self: self.first() if hasattr(self, "first") else self[0]
        )
        col_cls.add_method(
            "last", lambda self: self.last() if hasattr(self, "last") else self[-1]
        )

    # 5. Attach IndexedCollection methods
    idx_cls = hierarchy.get("IndexedCollection")
    if idx_cls:
        idx_cls.add_method(
            "at",
            lambda self, idx: self.at(idx) if hasattr(self, "at") else self[int(idx)],
        )
        idx_cls.add_method(
            "atPut",
            lambda self, idx, val: (
                self.atPut(idx, val)
                if hasattr(self, "atPut")
                else (self.__setitem__(int(idx), val), val)[1]
            ),
        )
        idx_cls.add_method(
            "put",
            lambda self, val, idx: (
                self.put(val, idx)
                if hasattr(self, "put")
                else (self.__setitem__(int(idx), val), val)[1]
            ),
        )
        idx_cls.add_method(
            "indexOf",
            lambda self, target: (
                self.indexOf(target)
                if hasattr(self, "indexOf")
                else (self.index(target) if target in self else None)
            ),
        )
        idx_cls.add_method(
            "find",
            lambda self, target: (
                self.find(target)
                if hasattr(self, "find")
                else (self.index(target) if target in self else None)
            ),
        )
        idx_cls.add_method(
            "keysDo",
            lambda self, block: self.keysDo(block) if hasattr(self, "keysDo") else None,
        )
        idx_cls.add_method(
            "reverse",
            lambda self: (
                self.reverse()
                if hasattr(self, "reverse")
                else (self.reverse(), self)[1]
            ),
        )

    # 6. Attach Array methods
    if arr_cls:
        arr_cls.add_method(
            "fill", lambda self, val: self.fill(val) if hasattr(self, "fill") else None
        )
        arr_cls.add_method(
            "copyFrom",
            lambda self, b, e: (
                self.copyFrom(b, e)
                if hasattr(self, "copyFrom")
                else ActorArray(self[int(b) : int(e)])
            ),
        )

    # 7. Attach OrderedCollection & List methods
    for c in (ord_cls, lst_cls):
        if c:
            c.add_method(
                "add",
                lambda self, item: (
                    self.add(item)
                    if hasattr(self, "add")
                    else (self.append(item), item)[1]
                ),
            )
            c.add_method(
                "push",
                lambda self, item: (
                    self.push(item)
                    if hasattr(self, "push")
                    else (self.append(item), item)[1]
                ),
            )
            c.add_method(
                "addLast",
                lambda self, item: (
                    self.addLast(item)
                    if hasattr(self, "addLast")
                    else (self.append(item), item)[1]
                ),
            )
            c.add_method(
                "addFirst",
                lambda self, item: (
                    self.addFirst(item)
                    if hasattr(self, "addFirst")
                    else (self.insert(0, item), item)[1]
                ),
            )
            c.add_method(
                "first",
                lambda self: self.first() if hasattr(self, "first") else self[0],
            )
            c.add_method(
                "last", lambda self: self.last() if hasattr(self, "last") else self[-1]
            )
            c.add_method(
                "removeLast",
                lambda self: (
                    self.removeLast() if hasattr(self, "removeLast") else self.pop()
                ),
            )
            c.add_method(
                "pop",
                lambda self, *args: (
                    self.pop(*args) if hasattr(self, "pop") else self.pop(*args)
                ),
            )
            c.add_method(
                "removeFirst",
                lambda self: (
                    self.removeFirst() if hasattr(self, "removeFirst") else self.pop(0)
                ),
            )
            c.add_method(
                "remove",
                lambda self, x: (
                    self.remove(x) if hasattr(self, "remove") else self.pop(x)
                ),
            )
            c.add_method(
                "removeAt",
                lambda self, idx: (
                    self.removeAt(idx)
                    if hasattr(self, "removeAt")
                    else self.pop(int(idx))
                ),
            )
            c.add_method(
                "insert",
                lambda self, elem, idx: (
                    self.insert(elem, idx)
                    if hasattr(self, "insert")
                    else (self.insert(int(idx), elem), elem)[1]
                ),
            )

    # 7b. Attach Set methods
    if set_cls:
        set_cls.add_method("add", lambda self, item: self.add(item))
        set_cls.add_method("remove", lambda self, item: self.remove(item))
        set_cls.add_method("includes", lambda self, item: self.includes(item))
        set_cls.add_method("in", lambda self, item: self.includes(item))
        set_cls.add_method("size", lambda self: self.size())
        set_cls.add_method("do", lambda self, block: self.do(block))
        set_cls.add_method("copy", lambda self: self.copy())

    # 8. Attach Dictionary methods
    if dict_cls:
        dict_cls.add_method(
            "at", lambda self, k: self.at(k) if hasattr(self, "at") else self.get(k)
        )
        dict_cls.add_method(
            "atPut",
            lambda self, k, v: (
                self.atPut(k, v)
                if hasattr(self, "atPut")
                else (self.__setitem__(k, v), v)[1]
            ),
        )
        dict_cls.add_method(
            "put",
            lambda self, v, k: (
                self.put(v, k)
                if hasattr(self, "put")
                else (self.__setitem__(k, v), v)[1]
            ),
        )
        dict_cls.add_method(
            "add",
            lambda self, k, v: (
                self.add(k, v)
                if hasattr(self, "add")
                else (self.__setitem__(k, v), v)[1]
            ),
        )
        dict_cls.add_method(
            "includesKey",
            lambda self, k: (
                self.includesKey(k) if hasattr(self, "includesKey") else (k in self)
            ),
        )
        dict_cls.add_method(
            "keys",
            lambda self: self.keys() if hasattr(self, "keys") else list(self.keys()),
        )
        dict_cls.add_method(
            "values",
            lambda self: (
                self.values() if hasattr(self, "values") else list(self.values())
            ),
        )
        dict_cls.add_method(
            "removeKey",
            lambda self, k: (
                self.removeKey(k) if hasattr(self, "removeKey") else self.pop(k, None)
            ),
        )
        dict_cls.add_method(
            "remove",
            lambda self, k: (
                self.remove(k) if hasattr(self, "remove") else self.pop(k, None)
            ),
        )
        dict_cls.add_method(
            "removeAt",
            lambda self, k: (
                self.removeAt(k) if hasattr(self, "removeAt") else self.pop(k, None)
            ),
        )
        dict_cls.add_method(
            "size", lambda self: self.size() if hasattr(self, "size") else len(self)
        )
        dict_cls.add_method(
            "isEmpty",
            lambda self: (
                self.isEmpty() if hasattr(self, "isEmpty") else (len(self) == 0)
            ),
        )
        dict_cls.add_method(
            "clear", lambda self: self.clear() if hasattr(self, "clear") else None
        )
        dict_cls.add_method(
            "do", lambda self, block: self.do(block) if hasattr(self, "do") else None
        )
        dict_cls.add_method(
            "keysDo",
            lambda self, block: self.keysDo(block) if hasattr(self, "keysDo") else None,
        )
        dict_cls.add_method(
            "assocsDo",
            lambda self, block: (
                self.assocsDo(block) if hasattr(self, "assocsDo") else None
            ),
        )

    # 9. Attach Interval methods
    if itv_cls:
        itv_cls.add_method("over", lambda self, start, stop: Interval.over(start, stop))
        itv_cls.add_method(
            "overBy",
            lambda self, start, stop, step=1: Interval.overBy(start, stop, step),
        )
        itv_cls.add_method(
            "start",
            lambda self: (
                self.start
                if not callable(getattr(self, "start", None))
                else self.start()
            ),
        )
        itv_cls.add_method(
            "stop",
            lambda self: (
                self.stop if not callable(getattr(self, "stop", None)) else self.stop()
            ),
        )
        itv_cls.add_method(
            "step",
            lambda self: (
                self.step if not callable(getattr(self, "step", None)) else self.step()
            ),
        )
        itv_cls.add_method("first", lambda self: self.first())
        itv_cls.add_method("last", lambda self: self.last())
        itv_cls.add_method("size", lambda self: self.size())
        itv_cls.add_method("at", lambda self, idx: self.at(idx))
        itv_cls.add_method("in", lambda self, val: self.in_interval(val))
        itv_cls.add_method("asArray", lambda self: self.asArray())
        itv_cls.add_method("reverse", lambda self: self.reverse())
        itv_cls.add_method("do", lambda self, block: self.do(block))
        itv_cls.add_method("collect", lambda self, block: self.collect(block))

    # 10. Attach Point & Point3D methods
    if pt_cls:
        pt_cls.add_method(
            "x",
            lambda self: self.x if not callable(getattr(self, "x", None)) else self.x(),
        )
        pt_cls.add_method(
            "y",
            lambda self: self.y if not callable(getattr(self, "y", None)) else self.y(),
        )
        pt_cls.add_method("setX", lambda self, x: self.setX(x))
        pt_cls.add_method("setY", lambda self, y: self.setY(y))
        pt_cls.add_method("set", lambda self, x, y: self.set(x, y))
        pt_cls.add_method("asPoint", lambda self: self.asPoint())
        pt_cls.add_method("round", lambda self: self.round())
        pt_cls.add_method("+", lambda self, other: self + other)
        pt_cls.add_method("-", lambda self, other: self - other)
        pt_cls.add_method("*", lambda self, s: self * s)
        pt_cls.add_method("//", lambda self, s: self // s)
        pt_cls.add_method("/", lambda self, s: self / s)
        pt_cls.add_method("=", lambda self, other: self == other)
        pt_cls.add_method("==", lambda self, other: self == other)
        pt_cls.add_method("!=", lambda self, other: self != other)
        pt_cls.add_method("<>", lambda self, other: self != other)

    if p3d_cls:
        p3d_cls.add_method(
            "z",
            lambda self: self.z if not callable(getattr(self, "z", None)) else self.z(),
        )
        p3d_cls.add_method("setZ", lambda self, z: self.setZ(z))
        p3d_cls.add_method("asPoint", lambda self: self.asPoint())
        p3d_cls.add_method(
            "middlePoint", lambda self, other, rng=0: self.middlePoint(other, rng)
        )
        p3d_cls.add_method("+", lambda self, other: self + other)
        p3d_cls.add_method("-", lambda self, other: self - other)
        p3d_cls.add_method("*", lambda self, s: self * s)
        p3d_cls.add_method("//", lambda self, s: self // s)
        p3d_cls.add_method("/", lambda self, s: self / s)
        p3d_cls.add_method("=", lambda self, other: self == other)
        p3d_cls.add_method("==", lambda self, other: self == other)
        p3d_cls.add_method("!=", lambda self, other: self != other)
        p3d_cls.add_method("<>", lambda self, other: self != other)

    # 11. Attach Int methods
    int_cls = hierarchy.get("Int")
    if int_cls:
        int_cls.add_method("over", lambda self, stop: Interval(self, stop, 1))
        int_cls.add_method(
            "overBy", lambda self, stop, step=1: Interval(self, stop, step)
        )
        int_cls.add_method("point3D", lambda self, yv, zv: Point3D(self, yv, zv))

    # 12. Attach Agent methods
    agent_cls = hierarchy.get("Agent")
    if agent_cls:
        agent_cls.ivars = [
            "name",
            "model",
            "endpoint",
            "systemPrompt",
            "memory",
            "history",
            "tools",
            "lastResponse",
            "autoSynthesize",
        ]
        agent_cls._default_constructor = lambda *args, **kw: Agent(*args, **kw)
        agent_cls.set_method("new", lambda self, *args, **kw: Agent(*args, **kw))
        agent_cls.add_method("name", lambda self: self.getName() if hasattr(self, "getName") else getattr(self, "name", "Agent"))
        agent_cls.add_method("getName", lambda self: self.getName())
        agent_cls.add_method("setName", lambda self, n: self.setName(n))
        agent_cls.add_method("model", lambda self: self.getModel() if hasattr(self, "getModel") else getattr(self, "model", "qwen3.8:latest"))
        agent_cls.add_method("getModel", lambda self: self.getModel())
        agent_cls.add_method("setModel", lambda self, m: self.setModel(m))
        agent_cls.add_method("endpoint", lambda self: self.getEndpoint() if hasattr(self, "getEndpoint") else getattr(self, "endpoint", ""))
        agent_cls.add_method("getEndpoint", lambda self: self.getEndpoint())
        agent_cls.add_method("setEndpoint", lambda self, ep: self.setEndpoint(ep))
        agent_cls.add_method("systemPrompt", lambda self: self.systemPrompt())
        agent_cls.add_method("setSystemPrompt", lambda self, p: self.setSystemPrompt(p))
        agent_cls.add_method("autoSynthesize", lambda self: self.autoSynthesize())
        agent_cls.add_method("setAutoSynthesize", lambda self, f: self.setAutoSynthesize(f))
        agent_cls.add_method("lastResponse", lambda self: self.lastResponse())
        agent_cls.add_method("memory", lambda self: self.memory())
        agent_cls.add_method("history", lambda self: self.history())
        agent_cls.add_method("tools", lambda self: self.tools())
        agent_cls.add_method("remember", lambda self, k, v: self.remember(k, v))
        agent_cls.add_method("recall", lambda self, k: self.recall(k))
        agent_cls.add_method("forget", lambda self, k: self.forget(k))
        agent_cls.add_method("clearHistory", lambda self: self.clearHistory())
        agent_cls.add_method("registerTool", lambda self, k, t: self.registerTool(k, t))
        agent_cls.add_method("useTool", lambda self, k, *args: self.useTool(k, *args))
        agent_cls.add_method("invokeTool", lambda self, k, *args: self.invokeTool(k, *args))
        agent_cls.add_method("inspectObject", lambda self, target: self.inspectObject(target))
        agent_cls.add_method("ask", lambda self, prompt, ctx=None: self.ask(prompt, ctx))
        agent_cls.add_method("think", lambda self, prompt, ctx=None: self.think(prompt, ctx))
        agent_cls.add_method("delegateTo", lambda self, other, prompt: self.delegateTo(other, prompt))
        agent_cls.add_method(
            "synthesizeMethod",
            lambda self, target, sel, spec="", arity=0: self.synthesizeMethod(target, sel, spec, arity),
        )

    # 13. Attach ThermoProTwin methods (Digital Twin for ThermoPro TP358W / TP358S)
    tp_cls = hierarchy.get("ThermoProTwin")
    if tp_cls:
        tp_cls.ivars = [
            "macAddress",
            "deviceName",
            "model",
            "temperatureC",
            "temperatureC10",
            "temperatureF",
            "temperatureF10",
            "humidity",
            "battery",
            "rssi",
            "comfortLevel",
            "dewPointC",
            "lastSyncTime",
            "history",
            "observers",
        ]
        tp_cls._default_constructor = lambda *args, **kw: ThermoProTwin(*args, **kw)
        tp_cls.set_method("new", lambda self, *args, **kw: ThermoProTwin(*args, **kw))
        tp_cls.add_method("macAddress", lambda self: self.macAddress())
        tp_cls.add_method("setMacAddress", lambda self, m: self.setMacAddress(m))
        tp_cls.add_method("deviceName", lambda self: self.deviceName())
        tp_cls.add_method("setDeviceName", lambda self, n: self.setDeviceName(n))
        tp_cls.add_method("model", lambda self: self.model())
        tp_cls.add_method("temperatureC", lambda self: self.temperatureC())
        tp_cls.add_method("temperatureC10", lambda self: self.temperatureC10())
        tp_cls.add_method("temperatureF", lambda self: self.temperatureF())
        tp_cls.add_method("temperatureF10", lambda self: self.temperatureF10())
        tp_cls.add_method("humidity", lambda self: self.humidity())
        tp_cls.add_method("battery", lambda self: self.battery())
        tp_cls.add_method("rssi", lambda self: self.rssi())
        tp_cls.add_method("comfortLevel", lambda self: self.comfortLevel())
        tp_cls.add_method("dewPointC", lambda self: self.dewPointC())
        tp_cls.add_method("lastSyncTime", lambda self: self.lastSyncTime())
        tp_cls.add_method("syncLive", lambda self, t=12.0: self.syncLive(t))
        tp_cls.add_method("readLive", lambda self, t=12.0: self.readLive(t))
        tp_cls.add_method("downloadHistory", lambda self, n=60: self.downloadHistory(n))
        tp_cls.add_method("ingestAdvertisementHex", lambda self, h, r=None: self.ingestAdvertisementHex(h, r))
        tp_cls.add_method("ingestHistoryHex", lambda self, h, ts=None: self.ingestHistoryHex(h, ts))
        tp_cls.add_method("history", lambda self: self.history())
        tp_cls.add_method("historyCount", lambda self: self.historyCount())
        tp_cls.add_method("recordAt", lambda self, idx: self.recordAt(idx))
        tp_cls.add_method("latestRecord", lambda self: self.latestRecord())
        tp_cls.add_method("oldestRecord", lambda self: self.oldestRecord())
        tp_cls.add_method("minTempC", lambda self: self.minTempC())
        tp_cls.add_method("maxTempC", lambda self: self.maxTempC())
        tp_cls.add_method("avgTempC", lambda self: self.avgTempC())
        tp_cls.add_method("minTempF", lambda self: self.minTempF())
        tp_cls.add_method("maxTempF", lambda self: self.maxTempF())
        tp_cls.add_method("avgTempF", lambda self: self.avgTempF())
        tp_cls.add_method("minTempC10", lambda self: self.minTempC10())
        tp_cls.add_method("maxTempC10", lambda self: self.maxTempC10())
        tp_cls.add_method("minHumidity", lambda self: self.minHumidity())
        tp_cls.add_method("maxHumidity", lambda self: self.maxHumidity())
        tp_cls.add_method("avgHumidity", lambda self: self.avgHumidity())
        tp_cls.add_method("addObserver", lambda self, obs: self.addObserver(obs))
        tp_cls.add_method("removeObserver", lambda self, obs: self.removeObserver(obs))
        tp_cls.add_method("notifyObservers", lambda self: self.notifyObservers())
        tp_cls.add_method("summary", lambda self: self.summary())
        tp_cls.add_method(
            "renderOnSurface",
            lambda self, surf, x=30, y=30, w=740, h=520: self.renderOnSurface(surf, x, y, w, h),
        )

    # 14. Attach ThermometerUI methods (Actor GUI Window for ThermoProTwin)
    tui_cls = hierarchy.get("ThermometerUI")
    if tui_cls:
        tui_cls.ivars = [
            "twin",
            "tempText",
            "humidityText",
            "comfortText",
            "statsText",
            "useFahrenheit",
            "updateCount",
        ]
        tui_cls._default_constructor = lambda *args, **kw: ThermometerUI(*args, **kw)
        tui_cls.set_method("new", lambda self, *args, **kw: ThermometerUI(*args, **kw))
        tui_cls.add_method("twin", lambda self: self.twin())
        tui_cls.add_method("attachTwin", lambda self, t: self.attachTwin(t))
        tui_cls.add_method("attachSurface", lambda self, s: self.attachSurface(s))
        tui_cls.add_method("refreshFromTwin", lambda self, t=None: self.refreshFromTwin(t))
        tui_cls.add_method("onSensorUpdate", lambda self, t: self.onSensorUpdate(t))
        tui_cls.add_method("onSyncClicked", lambda self, msg=None: self.onSyncClicked(msg))
        tui_cls.add_method("onHistoryClicked", lambda self, msg=None: self.onHistoryClicked(msg))
        tui_cls.add_method("onUnitToggled", lambda self, msg=None: self.onUnitToggled(msg))
        tui_cls.add_method("tempText", lambda self: self.tempText())
        tui_cls.add_method("humidityText", lambda self: self.humidityText())
        tui_cls.add_method("comfortText", lambda self: self.comfortText())
        tui_cls.add_method("statsText", lambda self: self.statsText())
        tui_cls.add_method("updateCount", lambda self: self.updateCount())
        tui_cls.add_method("summary", lambda self: self.summary())
        tui_cls.add_method(
            "addButton",
            lambda self, label, action="#selected", btn_id=None: self.addButton(label, action, btn_id),
        )
        tui_cls.add_method("onCustomButtonClicked", lambda self, msg=None: self.onCustomButtonClicked(msg))
        tui_cls.add_method("renderOnSurface", lambda self, s=None: self.renderOnSurface(s))
        tui_cls.add_method("draw", lambda self: self.draw())
        tui_cls.add_method("render", lambda self: self.render())

    # 15. Concurrency primitives: Future / Channel / Mutex / AtomicCounter / ThreadPool.
    # Instances are capability-gated ActorBridgeObjects bound to the active VM.
    _register_concurrency_classes(hierarchy)


def _register_concurrency_classes(hierarchy: ActorInheritanceHierarchy) -> None:
    """Mark concurrency classes as bridge classes and attach `new(...)` constructors."""

    def _active_vm() -> Any:
        from actorii.vm import ActorVM

        return ActorVM._active_vm

    factories = {
        "Channel": lambda *args: ActorChannel(*args[:1]),
        "Mutex": lambda *args: ActorMutex(_active_vm()),
        "AtomicCounter": lambda *args: AtomicCounter(*args[:1]),
        "ThreadPool": lambda *args: ActorThreadPoolTwin(
            _active_vm(), *(args[:1] or (4,))
        ),
    }
    for cname, py_cls in CONCURRENCY_BRIDGE_CLASSES.items():
        cls = hierarchy.get(cname)
        if cls is None:
            continue
        specs = dict(getattr(py_cls, "__actor_bridge_specs__", {}) or {})
        cls.is_bridge_class = True
        cls.bridge_specs = specs
        cls.ivars = list(specs.keys())
        # Class-level delegates so bridge selectors win over inherited Object
        # selectors of the same name (e.g. `error(fut)!!` vs Object>>error).
        for sel in specs:

            def _delegate(self: Any, *args: Any, _sel: str = sel) -> Any:
                invoke = getattr(self, "invoke_bridge_member", None)
                if invoke is None:
                    raise TypeError(
                        f"'{_sel}' requires a live concurrency handle, got {self!r}"
                    )
                return invoke(_sel, *args)

            cls.add_method(sel, _delegate)
        factory = factories.get(cname)
        if factory is None:
            continue

        def _ctor(*args: Any, _f: Any = factory) -> Any:
            return wrap_for_actor(_active_vm(), _f(*args))

        cls._default_constructor = _ctor
        cls.set_method("new", lambda self, *args, _c=_ctor: _c(*args))


__all__ = [
    "ActorObject",
    "ActorClass",
    "ActorInheritanceHierarchy",
    "ActorMessage",
    "Symbol",
    "is_kind_of",
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
    "KeyedCollection",
    "Dictionary",
    "ActorDictionary",
    "Interval",
    "Point",
    "Point3D",
    "Rect",
    "WindowsObject",
    "Window",
    "Dialog",
    "Control",
    "Button",
    "TextField",
    "Checkbox",
    "Agent",
    "ThermoProTwin",
    "ThermometerUI",
    "ActorFuture",
    "ActorChannel",
    "ActorMutex",
    "AtomicCounter",
    "TaskPool",
    "ActorThreadPoolTwin",
    "register_classes",
]
