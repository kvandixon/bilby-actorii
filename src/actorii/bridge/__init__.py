"""Capability-Gated Python-to-Actor FFI Bridge (`actorii.bridge`).

Allows any Python driver, sensor client, or service class to be exposed as a
first-class 1993 Actor object (`ActorBridgeObject`) and class (`ActorClass`)
with zero boilerplate while enforcing strict capability boundaries:

- `@actor_query`: Marks a Python method or property as **Tier 0 (Read-Only)**.
  Safe for unrestricted Actor evaluation and JIT method synthesis.
- `@actor_command`: Marks a Python method as **Tier 2 (Side-Effecting / Mutating)**.
  Logged in the bridge audit trail and gated by `allow_commands` and transactional
  shadow-run policies (blocked during dry-run shadow executions so hardware
  relays/lights are never triggered during speculative tests).
- Blocks arbitrary Python `getattr` / dunder escapes from Actor code.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type

from actorii.core import ActorClass, ActorObject, Symbol


class ActorBridgeSecurityError(PermissionError):
    """Raised when Actor code attempts an unauthorized or blocked Python bridge call."""


@dataclass(frozen=True)
class BridgeMemberSpec:
    """Metadata for a single exposed Python method or property on an Actor bridge."""

    name: str
    tier: int  # 0 = Read-Only Query, 2 = Side-Effecting Command
    is_property: bool = False
    arity: Optional[int] = None
    doc: str = ""
    shadow_safe: bool = False


@dataclass
class BridgeAuditEntry:
    """Audit record for a bridge invocation from the Actor VM."""

    timestamp: float
    class_name: str
    obj_id: str
    selector: str
    tier: int
    args_repr: str
    shadow_blocked: bool = False


def actor_query(func_or_prop: Any = None, *, name: Optional[str] = None) -> Any:
    """Decorator marking a Python method or `@property` as a Tier 0 Read-Only Actor query."""

    def _decorate(target: Any) -> Any:
        if isinstance(target, property) and target.fget is not None:
            setattr(target.fget, "__actor_tier__", 0)
            setattr(target.fget, "__actor_name__", name or target.fget.__name__)
            setattr(target.fget, "__actor_is_property__", True)
            return target
        setattr(target, "__actor_tier__", 0)
        setattr(target, "__actor_name__", name or getattr(target, "__name__", ""))
        setattr(target, "__actor_is_property__", False)
        return target

    if func_or_prop is not None:
        return _decorate(func_or_prop)
    return _decorate


def actor_command(
    func: Any = None, *, name: Optional[str] = None, shadow_safe: bool = False
) -> Any:
    """Decorator marking a Python method as a Tier 2 Side-Effecting Actor command."""

    def _decorate(target: Any) -> Any:
        setattr(target, "__actor_tier__", 2)
        setattr(target, "__actor_name__", name or getattr(target, "__name__", ""))
        setattr(target, "__actor_is_property__", False)
        setattr(target, "__actor_shadow_safe__", bool(shadow_safe))
        return target

    if func is not None:
        return _decorate(func)
    return _decorate


def _infer_callable_arity(fn: Callable[..., Any], is_bound: bool = True) -> Optional[int]:
    """Infer external Actor argument count (excluding Python `self`) for a callable."""
    try:
        sig = inspect.signature(fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not is_bound and params and params[0].name in ("self", "cls"):
            params = params[1:]
        return len(params)
    except Exception:
        return None


def introspect_bridge_specs(
    py_cls_or_obj: Any,
    queries: Optional[Iterable[str]] = None,
    commands: Optional[Iterable[str]] = None,
) -> Dict[str, BridgeMemberSpec]:
    """Collect allowed `BridgeMemberSpec` entries from decorators or explicit allowlists."""
    py_cls = py_cls_or_obj if isinstance(py_cls_or_obj, type) else py_cls_or_obj.__class__
    specs: Dict[str, BridgeMemberSpec] = {}

    explicit_queries: Set[str] = set(queries or ())
    explicit_commands: Set[str] = set(commands or ())

    for cls in reversed(py_cls.__mro__):
        if cls is object:
            continue
        for attr_name, attr_val in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if isinstance(attr_val, property) and attr_val.fget is not None:
                fget = attr_val.fget
                tier = getattr(fget, "__actor_tier__", None)
                exposed_name = getattr(fget, "__actor_name__", attr_name)
                if tier is not None or attr_name in explicit_queries:
                    specs[exposed_name] = BridgeMemberSpec(
                        name=exposed_name,
                        tier=0,
                        is_property=True,
                        arity=0,
                        doc=inspect.getdoc(fget) or "",
                    )
            elif callable(attr_val):
                tier = getattr(attr_val, "__actor_tier__", None)
                exposed_name = getattr(attr_val, "__actor_name__", attr_name)
                if attr_name in explicit_commands:
                    tier = 2
                elif attr_name in explicit_queries and tier is None:
                    tier = 0
                if tier is not None:
                    specs[exposed_name] = BridgeMemberSpec(
                        name=exposed_name,
                        tier=int(tier),
                        is_property=False,
                        arity=_infer_callable_arity(attr_val, is_bound=False),
                        doc=inspect.getdoc(attr_val) or "",
                        shadow_safe=bool(getattr(attr_val, "__actor_shadow_safe__", False)),
                    )

    return specs


class ActorBridgeObject(ActorObject):
    """Safe, capability-gated Actor wrapper around a live Python object.

    Only methods/properties registered in `_bridge_specs` (or standard `ActorObject`
    protocol selectors) may be invoked from Actor bytecode. Any attempt to call
    unexposed Python methods raises `ActorBridgeSecurityError`.
    """

    _STANDARD_OBJECT_SELECTORS = frozenset(
        {
            "asString",
            "print",
            "error",
            "isKindOf",
            "kindOf",
            "clone",
            "copy",
            "init",
            "class",
            "isNil",
            "notNil",
            "bridgeCapabilities",
            "bridgeAuditLog",
        }
    )

    def __init__(
        self,
        target: Any,
        class_name: Optional[str] = None,
        specs: Optional[Dict[str, BridgeMemberSpec]] = None,
        allow_commands: bool = True,
        vm: Optional[Any] = None,
        obj_id: Optional[str] = None,
    ) -> None:
        resolved_name = class_name or getattr(
            target.__class__, "__actor_bridge_name__", target.__class__.__name__
        )
        super().__init__(obj_id=obj_id or f"bridge:{resolved_name}:{id(target)}")
        self._class_name = resolved_name
        self._target = target
        self.vm = vm
        self.allow_commands = bool(allow_commands)
        self._bridge_specs: Dict[str, BridgeMemberSpec] = (
            dict(specs) if specs is not None else introspect_bridge_specs(target)
        )
        self.audit_log: List[BridgeAuditEntry] = []

        # Sync initial read-only properties into `_fields` for Actor inspector visibility
        self._sync_fields_from_target()

        # Register wrapper methods for each exposed spec
        for spec_name, spec in self._bridge_specs.items():
            self._register_spec_handler(spec_name, spec)

    @property
    def target(self) -> Any:
        """Return the underlying wrapped Python object."""
        return self._target

    def _sync_fields_from_target(self) -> None:
        """Mirror non-private scalar attributes and `@actor_query` properties into `_fields`."""
        if hasattr(self._target, "__dict__"):
            for k, v in self._target.__dict__.items():
                if not k.startswith("_") and isinstance(
                    v, (int, float, str, bool, Symbol, type(None))
                ):
                    self._fields[k] = v
        for spec_name, spec in self._bridge_specs.items():
            if spec.is_property:
                try:
                    val = getattr(self._target, spec_name)
                    if isinstance(val, (int, float, str, bool, Symbol, type(None))):
                        self._fields[spec_name] = val
                except Exception:
                    pass

    def _register_spec_handler(self, selector: str, spec: BridgeMemberSpec) -> None:
        def _handler(*args: Any) -> Any:
            return self.invoke_bridge_member(selector, *args)

        setattr(_handler, "__actor_tier__", spec.tier)
        self.set_method(selector, _handler)

    def get_member_tier(self, selector: str) -> Optional[int]:
        """Return the capability tier (0 = query, 2 = command) for `selector`, or None."""
        spec = self._bridge_specs.get(selector)
        return spec.tier if spec is not None else None

    def invoke_bridge_member(self, selector: str, *args: Any) -> Any:
        """Execute a capability-checked call against the underlying Python target."""
        # If called via `self._invoke_python_callable`, strip leading `self` receiver if passed
        if args and args[0] is self:
            args = args[1:]

        spec = self._bridge_specs.get(selector)
        if spec is None:
            raise ActorBridgeSecurityError(
                f"Selector '{selector}' is not exposed on ActorBridgeObject '{self._class_name}'. "
                f"Allowed selectors: {sorted(self._bridge_specs.keys())}"
            )

        active_vm = self.vm or getattr(ActorObject, "_active_vm", None)
        if active_vm is None:
            from actorii.vm import ActorVM

            active_vm = ActorVM._active_vm

        in_shadow = bool(getattr(active_vm, "_in_shadow_run", False))

        if spec.tier >= 2:
            if not self.allow_commands or not bool(getattr(active_vm, "allow_bridge_commands", True)):
                self.audit_log.append(
                    BridgeAuditEntry(
                        timestamp=time.time(),
                        class_name=self._class_name,
                        obj_id=self.obj_id,
                        selector=selector,
                        tier=spec.tier,
                        args_repr=repr(args),
                        shadow_blocked=True,
                    )
                )
                raise ActorBridgeSecurityError(
                    f"Tier 2 command '{self._class_name}>>{selector}' blocked: bridge commands are disabled."
                )
            if in_shadow and not spec.shadow_safe:
                self.audit_log.append(
                    BridgeAuditEntry(
                        timestamp=time.time(),
                        class_name=self._class_name,
                        obj_id=self.obj_id,
                        selector=selector,
                        tier=spec.tier,
                        args_repr=repr(args),
                        shadow_blocked=True,
                    )
                )
                raise ActorBridgeSecurityError(
                    f"Tier 2 side-effecting command '{self._class_name}>>{selector}' blocked during transactional shadow run."
                )

        self.audit_log.append(
            BridgeAuditEntry(
                timestamp=time.time(),
                class_name=self._class_name,
                obj_id=self.obj_id,
                selector=selector,
                tier=spec.tier,
                args_repr=repr(args),
                shadow_blocked=False,
            )
        )

        if spec.is_property:
            res = getattr(self._target, selector)
        else:
            attr = getattr(self._target, selector)
            res = attr(*args)

        self._sync_fields_from_target()
        return res

    def bridgeCapabilities(self, *_args: Any) -> Dict[str, int]:
        """Return a mapping of exposed selector -> risk tier (0=query, 2=command)."""
        return {k: v.tier for k, v in self._bridge_specs.items()}

    def bridgeAuditLog(self, *_args: Any) -> int:
        """Return the number of audited calls executed on this bridge object."""
        return len(self.audit_log)

    def message_not_understood(self, selector: str, *args: Any) -> Any:
        """Explicitly reject unexposed selectors on bridged Python objects."""
        raise ActorBridgeSecurityError(
            f"ActorBridgeSecurityError: '{self._class_name}' blocks unexposed selector '{selector}'. "
            f"Decorate with @actor_query or @actor_command to expose."
        )


def actor_bridge(
    py_cls: Optional[Type[Any]] = None,
    *,
    name: Optional[str] = None,
    queries: Optional[Iterable[str]] = None,
    commands: Optional[Iterable[str]] = None,
) -> Any:
    """Class decorator that prepares a Python class for registration in ActorII."""

    def _decorate(cls: Type[Any]) -> Type[Any]:
        bridge_name = name or cls.__name__
        specs = introspect_bridge_specs(cls, queries=queries, commands=commands)
        setattr(cls, "__actor_bridge_name__", bridge_name)
        setattr(cls, "__actor_bridge_specs__", specs)
        return cls

    if py_cls is not None:
        return _decorate(py_cls)
    return _decorate


def register_bridge_class(
    vm: Any,
    py_cls: Type[Any],
    *,
    name: Optional[str] = None,
    queries: Optional[Iterable[str]] = None,
    commands: Optional[Iterable[str]] = None,
    allow_commands: bool = True,
) -> ActorClass:
    """Register a Python class in `vm.hierarchy` so `new(ClassName, ...)` creates an `ActorBridgeObject`."""
    bridge_name = name or getattr(py_cls, "__actor_bridge_name__", py_cls.__name__)
    specs = getattr(py_cls, "__actor_bridge_specs__", None)
    if specs is None or queries is not None or commands is not None:
        specs = introspect_bridge_specs(py_cls, queries=queries, commands=commands)

    actor_cls = ActorClass(bridge_name, bases=("Object",))
    actor_cls.is_bridge_class = True
    actor_cls.bridge_specs = dict(specs)
    actor_cls.ivars = list(specs.keys())

    def _factory(*args: Any, **kwargs: Any) -> ActorBridgeObject:
        py_inst = py_cls(*args, **kwargs)
        bridge_obj = ActorBridgeObject(
            target=py_inst,
            class_name=bridge_name,
            specs=specs,
            allow_commands=allow_commands,
            vm=vm,
        )
        bridge_obj.inherit_from(actor_cls)
        return bridge_obj

    actor_cls._default_constructor = _factory
    actor_cls.set_method("new", _factory)

    # Register method placeholders on the ActorClass so hierarchy arity/method resolution succeeds
    for sel, spec in specs.items():
        def _make_dispatcher(s: str, sp: BridgeMemberSpec) -> Callable[..., Any]:
            def _dispatch(receiver: Any, *a: Any) -> Any:
                if isinstance(receiver, ActorBridgeObject):
                    return receiver.invoke_bridge_member(s, *a)
                raise ActorBridgeSecurityError(f"Receiver for {s} is not an ActorBridgeObject")

            setattr(_dispatch, "__actor_tier__", sp.tier)
            setattr(_dispatch, "__actor_arity__", sp.arity or 0)
            return _dispatch

        actor_cls.add_method(sel, _make_dispatcher(sel, spec))

    vm.hierarchy.register(actor_cls)
    vm.global_vars[bridge_name] = actor_cls
    if "Actor" in vm.global_vars and isinstance(vm.global_vars["Actor"], dict):
        vm.global_vars["Actor"][bridge_name] = actor_cls
    return actor_cls


def expose_python_object(
    vm: Any,
    py_obj: Any,
    *,
    global_key: Optional[str] = None,
    class_name: Optional[str] = None,
    queries: Optional[Iterable[str]] = None,
    commands: Optional[Iterable[str]] = None,
    allow_commands: bool = True,
) -> ActorBridgeObject:
    """Wrap a live Python object as an `ActorBridgeObject` and optionally bind `Actor[#key]`."""
    py_cls = py_obj.__class__
    resolved_cls_name = class_name or getattr(py_cls, "__actor_bridge_name__", py_cls.__name__)
    actor_cls = vm.hierarchy.get(resolved_cls_name)
    if actor_cls is None or not getattr(actor_cls, "is_bridge_class", False):
        actor_cls = register_bridge_class(
            vm,
            py_cls,
            name=resolved_cls_name,
            queries=queries,
            commands=commands,
            allow_commands=allow_commands,
        )

    specs = getattr(actor_cls, "bridge_specs", None) or introspect_bridge_specs(
        py_obj, queries=queries, commands=commands
    )
    bridge_obj = ActorBridgeObject(
        target=py_obj,
        class_name=resolved_cls_name,
        specs=specs,
        allow_commands=allow_commands,
        vm=vm,
    )
    bridge_obj.inherit_from(actor_cls)

    if global_key:
        clean_key = global_key[1:] if global_key.startswith("#") else global_key
        if "Actor" in vm.global_vars and isinstance(vm.global_vars["Actor"], dict):
            vm.global_vars["Actor"][clean_key] = bridge_obj
            vm.global_vars["Actor"][Symbol(clean_key)] = bridge_obj
    return bridge_obj


__all__ = [
    "ActorBridgeObject",
    "ActorBridgeSecurityError",
    "BridgeAuditEntry",
    "BridgeMemberSpec",
    "actor_bridge",
    "actor_command",
    "actor_query",
    "expose_python_object",
    "introspect_bridge_specs",
    "register_bridge_class",
]
