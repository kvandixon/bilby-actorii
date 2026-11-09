"""Core OOP foundation for the Actor programming language.

Implements clean-room reverse-engineered constructs from the 1993
Whitewater Group/Genesis Development Systems 'actor' language,
adapted for modern Linux/Wayland/HTML5 deployment.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ActorObject:
    """Base class for all actor language objects.

    Implements the core OOP primitives identified from the 1993
    reverse-engineered actor language architecture:
    - Object identity and encapsulation
    - Message passing (method dispatch)
    - Single inheritance with single-root hierarchy
    - Dynamic method resolution
    """

    def __init__(self, obj_id: str | None = None):
        self._obj_id = obj_id or id(self)
        self._methods: dict[str, callable] = {}
        self._instance_methods: set[str] = set()
        self._fields: dict[str, object] = {}
        self._parent: ActorObject | None = None
        self._class_name: str = self.__class__.__name__

    @property
    def obj_id(self) -> str:
        return str(self._obj_id)

    @property
    def class_name(self) -> str:
        return self._class_name

    def has_method(self, name: str) -> bool:
        """Check if object responds to a message/method."""
        if name in self._methods:
            return True
        if self._parent is not None and self._parent.has_method(name):
            return True
        return False

    def get_method(self, name: str) -> callable | None:
        """Retrieve a method by name (message passing), honoring live class hot-updates."""
        if name in self._instance_methods and name in self._methods:
            return self._methods.get(name)
        if self._parent is not None and self._parent.has_method(name):
            parent_m = self._parent.get_method(name)
            if parent_m is not None:
                self._methods[name] = parent_m
                return parent_m
        return self._methods.get(name)

    def set_method(self, name: str, func: callable) -> None:
        """Register a method (message handler)."""
        self._methods[name] = func
        if not isinstance(self, ActorClass):
            self._instance_methods.add(name)

    def inherit_from(self, parent: ActorObject) -> None:
        """Establish single inheritance relationship."""
        self._parent = parent
        # Inherit parent methods if not already defined
        if self._parent:
            for name, method in self._parent._methods.items():
                if name not in self._methods:
                    self._methods[name] = method

    def dispatch(self, method_name: str, *args, **kwargs) -> object:
        """Send a message to this object (method dispatch)."""
        method = self.get_method(method_name)
        if method is not None:
            return method(*args, **kwargs)
        elif self._parent is not None:
            return self._parent.dispatch(method_name, *args, **kwargs)
        else:
            raise AttributeError(
                f"ActorObject {self.class_name} has no method '{method_name}'"
            )

    def define_field(self, name: str, value: object) -> None:
        """Define an instance field/attribute."""
        self._fields[name] = value

    def field(self, name: str, default: object = None) -> object:
        """Get an instance field value."""
        return self._fields.get(name, default)

    def undefine_field(self, name: str) -> None:
        """Remove an instance field."""
        self._fields.pop(name, None)

    def __repr__(self) -> str:
        return f"<{self.class_name} {self.obj_id}>"

    # Standard Object Protocol (1993 Whitewater Actor OBJECT.CLS)
    def print(self, stream: Optional[Any] = None) -> "ActorObject":
        s = self.asString()
        if stream is not None and hasattr(stream, "write"):
            stream.write(s)
        else:
            print(s)
        return self

    def asString(self) -> str:
        return f"<{self.class_name}>"

    def error(self, message: object) -> None:
        raise RuntimeError(f"Actor Error in {self}: {message}")

    def isKindOf(self, target: object, hierarchy: Optional[Any] = None) -> bool:
        from actorii.classes.base import is_kind_of

        return is_kind_of(self, target, hierarchy)

    def kindOf(self, target: object, hierarchy: Optional[Any] = None) -> bool:
        return self.isKindOf(target, hierarchy)

    def clone(self) -> "ActorObject":
        return copy.copy(self)

    def copy(self) -> "ActorObject":
        return self.clone()

    def init(self) -> "ActorObject":
        return self


class ActorClass(ActorObject):
    """Meta-class for defining actor language classes.

    Handles class-level definitions including:
    - Class methods and class fields
    - Inheritance hierarchy
    - Constructor registration (pNEW from 1993 architecture)
    """

    def __init__(self, name: str, bases: tuple = ()):
        super().__init__(obj_id=f"class:{name}")
        self._name = name
        self._bases = bases
        self._class_methods: dict[str, callable] = {}
        self._class_fields: dict[str, object] = {}
        self._implements: list[str] = []

        # Register the implicit 'new' constructor from 1993 pNEW convention
        self.set_method("new", self._default_constructor)

    @property
    def name(self) -> str:
        return self._name

    def _default_constructor(self, *args, **kwargs) -> ActorObject:
        """Default constructor for actor classes."""
        instance = ActorObject(obj_id=str(id(self)))
        instance._class_name = self.name
        instance.inherit_from(self)
        return instance

    def add_method(self, name: str, func: callable) -> None:
        """Add a method to the class definition."""
        self._class_methods[name] = func
        # Also register on the base ActorObject for instances
        ActorObject.set_method(self, name, func)  # type: ignore

    def add_class_method(self, name: str, func: callable) -> None:
        """Add a class method (bound to the class, not instances)."""
        self._class_methods[name] = func

    def has_method(self, name: str) -> bool:
        """Check if this class defines a method."""
        return name in self._class_methods

    def get_method(self, name: str) -> callable | None:
        """Get a method from the class definition."""
        return self._class_methods.get(name)


class ActorInheritanceHierarchy:
    """Manages the single-inheritance hierarchy for actor language classes.

    The 1993 actor language used a single-root inheritance model
    (similar to Smalltalk), where all classes ultimately inherit
    from a common Object base class.
    """

    def __init__(self, register_defaults: bool = True):
        self._classes: dict[str, ActorClass] = {}
        self._root: ActorClass | None = None
        if register_defaults:
            try:
                from actorii.classes import register_classes

                register_classes(self)
            except ImportError:
                pass

    def register(self, cls: ActorClass) -> None:
        """Register a class in the hierarchy."""
        self._classes[cls.name] = cls
        # If no parent specified and this is the first class, make it root
        if self._root is None and not cls._bases:
            self._root = cls

    def get(self, name: str) -> ActorClass | None:
        """Get a class by name, traversing inheritance if needed."""
        if name in self._classes:
            return self._classes[name]
        # Check parent classes
        for cls in self._classes.values():
            if cls._bases:
                for base in cls._bases:
                    if base == name:
                        return self._classes.get(base)
        return None

    def resolve_method(self, class_name: str, method_name: str) -> callable | None:
        """Resolve a method through the inheritance chain."""
        cls = self.get(class_name)
        if cls is None:
            return None
        # Check own methods first
        if cls.has_method(method_name):
            return cls.get_method(method_name)
        # Walk up the inheritance chain
        for base in cls._bases:
            result = self.resolve_method(base, method_name)
            if result is not None:
                return result
        # Check parent class
        if cls._parent:
            return self.resolve_method(cls._parent.name, method_name)
        return None


class ActorMessage:
    """Represents a message (method call) in the actor language.

    From the 1993 architecture, messages are the primary mechanism
    for object interaction - sending a method name with arguments
    to an object, which dispatches via its method table.
    """

    def __init__(self, receiver: ActorObject, method_name: str, *args, **kwargs):
        self.receiver = receiver
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def execute(self) -> object:
        """Execute the message on the receiver."""
        return self.receiver.dispatch(self.method_name, *self.args, **self.kwargs)


class Symbol(str):
    """Authentic 1993 Symbol literal representation (#symbol)."""

    def __new__(cls, name: Any):
        if isinstance(name, str) and name.startswith("#"):
            name = name[1:]
        elif isinstance(name, Symbol):
            name = str(name)
        obj = super().__new__(cls, str(name))
        return obj

    @property
    def name(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"#{self}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Symbol):
            return super().__eq__(other)
        if isinstance(other, str):
            if other.startswith("#"):
                return super().__eq__(other[1:])
            return super().__eq__(other)
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


# Export core symbols
__all__ = [
    "ActorObject",
    "ActorClass",
    "ActorInheritanceHierarchy",
    "ActorMessage",
    "Symbol",
]


def __getattr__(name: str) -> Any:
    """Lazy resolution of standard library classes to prevent circular imports."""
    import actorii.classes

    if hasattr(actorii.classes, name):
        return getattr(actorii.classes, name)
    raise AttributeError(f"module 'actorii.core' has no attribute '{name}'")
