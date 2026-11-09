"""1993 Whitewater Actor 4.1 GUI Widget Hierarchy.

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/WINDOWSO.CLS
- historical_actor_41_source/actcls/WINDOW.CLS
- historical_actor_41_source/actcls/DIALOG.CLS
- historical_actor_41_source/actcls/CONTROL.CLS
- historical_actor_41_source/actcls/BUTTON.CLS
- historical_actor_41_source/actcls/CHECKBOX.CLS
- historical_actor_41_source/actcls/EDIT.CLS

Hierarchy:
    ActorObject
        └── WindowsObject
                ├── Window
                │     └── Dialog
                └── Control
                        ├── Button
                        │     └── Checkbox
                        └── TextField
"""

from __future__ import annotations

import builtins
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from actorii.core import ActorObject, Symbol
from actorii.classes.geometry import Rect

# =============================================================================
# 1. WindowsObject (Root GUI Class)
# =============================================================================


class WindowsObject(ActorObject):
    """Authentic 1993 Actor WindowsObject root class.

    Source: historical_actor_41_source/actcls/WINDOWSO.CLS
    """

    def __init__(
        self,
        title: str = "",
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(obj_id=obj_id or f"winobj_{builtins.id(self)}")
        self._class_name = "WindowsObject"
        self.title: str = str(title) if title is not None else ""
        self.id: Optional[Any] = id
        self.identity: Optional[Symbol] = (
            Symbol(identity) if identity is not None else None
        )
        self.visible: bool = True
        self.enabled: bool = True
        self.parent: Optional[Any] = None
        self.vm: Optional[Any] = None

    def show(self) -> None:
        """Make object visible (WINDOWSO.CLS:1645 ShowWindow)."""
        self.visible = True

    def hide(self) -> None:
        """Hide object."""
        self.visible = False

    def is_visible(self) -> bool:
        return self.visible

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def draw(self) -> str:
        return ""

    def render(self) -> str:
        return self.draw()

    def to_html(self) -> str:
        return self.draw()


# =============================================================================
# 2. Window (Window Container)
# =============================================================================


class Window(WindowsObject):
    """Authentic 1993 Actor Window class (WINDOW.CLS).

    Root container for windows and controls in the 1993 Whitewater Actor hierarchy.
    """

    def __init__(
        self,
        title: str = "Window",
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 0,
        top: int = 0,
        width: int = 640,
        height: int = 480,
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(title=title, id=id, identity=identity, obj_id=obj_id)
        self._class_name = "Window"
        if isinstance(rect, Rect):
            self.rect: Rect = rect
        elif isinstance(rect, (tuple, list)) and len(rect) == 4:
            self.rect = Rect(rect[0], rect[1], rect[2], rect[3])
        else:
            self.rect = Rect(left, top, left + width, top + height)

        self.children: List[WindowsObject] = []
        self.focus_child: Optional[WindowsObject] = None
        self.modal: bool = False
        self.action_handlers: Dict[Tuple[str, str], str] = {}

    @property
    def left(self) -> int:
        return self.rect.left

    @property
    def top(self) -> int:
        return self.rect.top

    @property
    def width(self) -> int:
        return self.rect.width

    @property
    def height(self) -> int:
        return self.rect.height

    def add(self, child: WindowsObject) -> WindowsObject:
        """Add child widget to window (WINDOW.CLS:242 add)."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            if self.vm and not getattr(child, "vm", None):
                child.vm = self.vm
        return child

    def add_child(self, child: WindowsObject) -> WindowsObject:
        """Alias for add()."""
        return self.add(child)

    def addChild(self, child: WindowsObject) -> WindowsObject:
        """Alias for add()."""
        return self.add(child)

    def remove(self, child: Any) -> Optional[WindowsObject]:
        """Remove child widget by instance, ID, or Symbol (WINDOW.CLS remove)."""
        target = (
            self.find_child(child) if not isinstance(child, WindowsObject) else child
        )
        if target in self.children:
            self.children.remove(target)
            target.parent = None
            if self.focus_child is target:
                self.focus_child = None
            return target
        return None

    def remove_child(self, child: Any) -> Optional[WindowsObject]:
        """Alias for remove()."""
        return self.remove(child)

    def removeChild(self, child: Any) -> Optional[WindowsObject]:
        """Alias for remove()."""
        return self.remove(child)

    def add_action_handler(self, child_id: Any, event: str, method_name: str) -> None:
        """Register an action handler for child widget events."""
        child_key = str(child_id)
        if child_key.startswith("#"):
            child_key = child_key[1:]
        event_key = str(event)
        if event_key.startswith("#"):
            event_key = event_key[1:]
        self.action_handlers[(child_key, event_key)] = method_name

    def set_focus(self, child: WindowsObject) -> None:
        """Set focus to a child widget."""
        self.focus_child = child

    def find_child(self, id_or_sym: Any) -> Optional[WindowsObject]:
        """Find a child by ID, identity Symbol, or title."""
        sym_str = str(id_or_sym)
        norm_sym = sym_str[1:] if sym_str.startswith("#") else sym_str
        for child in self.children:
            if child is id_or_sym:
                return child
            c_id = getattr(child, "id", None)
            c_ident = getattr(child, "identity", None)
            if c_id is not None:
                raw_id = str(c_id)
                norm_id = raw_id[1:] if raw_id.startswith("#") else raw_id
                if norm_id == norm_sym or raw_id == sym_str:
                    return child
            if c_ident is not None:
                raw_ident = str(c_ident)
                norm_ident = raw_ident[1:] if raw_ident.startswith("#") else raw_ident
                if norm_ident == norm_sym or raw_ident == sym_str:
                    return child
            if (
                getattr(child, "title", None) == sym_str
                or getattr(child, "label", None) == sym_str
            ):
                return child
        return None

    def __getitem__(self, key: Any) -> Optional[WindowsObject]:
        """Access child by Symbol or ID (win[#btn] in WINDOW.CLS:58)."""
        return self.find_child(key)

    def __setitem__(self, key: Any, child: WindowsObject) -> None:
        """Set child by Symbol/ID (win[#btn] := child)."""
        norm_key = Symbol(str(key))
        child.identity = norm_key
        child.id = norm_key
        existing = self.find_child(key)
        if existing and existing in self.children:
            idx = self.children.index(existing)
            self.children[idx] = child
            child.parent = self
        else:
            self.add(child)

    def register_action(self, child_id: Any, event_name: Any, method_name: str) -> None:
        """Map (child_id, event_name) to an Action method on this window."""
        norm_child = (
            str(child_id)[1:] if str(child_id).startswith("#") else str(child_id)
        )
        norm_event = (
            str(event_name)[1:] if str(event_name).startswith("#") else str(event_name)
        )
        self.action_handlers[(norm_child, norm_event)] = str(method_name)

    def draw(self) -> str:
        """Render window and children as structured HTML5-ready markup."""
        win_id = f"window_{id(self)}"
        child_html = "".join(
            [c.draw() for c in self.children if getattr(c, "visible", True)]
        )
        return (
            f'<div id="{win_id}" class="actor-window" data-title="{self.title}" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f"width: {self.rect.width}px; height: {self.rect.height}px; "
            f'background: #ECE9D8; border: 2px outset #FFFFFF; box-shadow: 2px 2px 6px rgba(0,0,0,0.4);">'
            f'<div class="actor-window-caption" style="height: 22px; background: #000080; color: #FFFFFF; '
            f"padding: 2px 6px; font-family: sans-serif; font-size: 12px; font-weight: bold; "
            f'display: flex; justify-content: space-between; align-items: center;">'
            f"<span>{self.title}</span>"
            f"</div>"
            f'<div class="actor-window-client" style="position: relative; width: 100%; height: calc(100% - 22px);">'
            f"{child_html}"
            f"</div>"
            f"</div>"
        )

    def handle_event(self, event: dict) -> bool:
        """Handle incoming event, dispatching to focus, children, or self."""
        for child in self.children:
            if hasattr(child, "handle_event") and child.handle_event(event):
                return True
        return False

    def __repr__(self) -> str:
        return f"{self._class_name}('{self.title}', {self.rect!r})"


# =============================================================================
# 3. Dialog (Modal / Modeless Dialog)
# =============================================================================


class Dialog(Window):
    """Authentic 1993 Actor Dialog class (inherits Window, modal support).

    Source: historical_actor_41_source/actcls/DIALOG.CLS
    """

    IDOK = 1
    IDCANCEL = 2

    def __init__(
        self,
        title: str = "Dialog",
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 100,
        top: int = 100,
        width: int = 400,
        height: int = 250,
        obj_id: Optional[str] = None,
        modal: bool = False,
    ):
        super().__init__(
            title=title,
            rect=rect,
            left=left,
            top=top,
            width=width,
            height=height,
            obj_id=obj_id,
        )
        self._class_name = "Dialog"
        self.modal: bool = modal
        self.is_modal: bool = False
        self.result_code: Optional[Union[int, Symbol]] = None
        self.ok_button: Optional[Button] = None
        self.cancel_button: Optional[Button] = None

    def end(self, result: Union[int, Symbol] = IDOK) -> None:
        """Close dialog and store result code (BUTTON.CLS line 67)."""
        self.result_code = result
        self.is_modal = False
        self.hide()

    def on_ok(self, msg: Any = None) -> None:
        """Default action when OK button is pressed."""
        self.end(self.IDOK)

    def on_cancel(self, msg: Any = None) -> None:
        """Default action when Cancel button is pressed."""
        self.end(self.IDCANCEL)

    def create_standard_buttons(
        self, ok_label: str = "OK", cancel_label: str = "Cancel"
    ) -> Tuple[Button, Button]:
        """Helper to create and add standard OK and Cancel buttons."""
        btn_y = max(10, self.rect.height - 60)
        btn_ok = Button(
            label=ok_label,
            action_selector=Symbol("okClicked"),
            left=self.rect.width - 200,
            top=btn_y,
            width=80,
            height=28,
            id=self.IDOK,
            identity=Symbol("btnOk"),
        )
        btn_cancel = Button(
            label=cancel_label,
            action_selector=Symbol("cancelClicked"),
            left=self.rect.width - 100,
            top=btn_y,
            width=80,
            height=28,
            id=self.IDCANCEL,
            identity=Symbol("btnCancel"),
        )
        self.ok_button = btn_ok
        self.cancel_button = btn_cancel
        self.add(btn_ok)
        self.add(btn_cancel)
        return btn_ok, btn_cancel

    def run_modal(self, vm: Optional[Any] = None) -> Optional[Union[int, Symbol]]:
        """Simulate modal execution; displays window and returns result code upon end()."""
        self.is_modal = True
        self.show()
        if vm:
            self.vm = vm
        return self.result_code

    def draw(self) -> str:
        """Render modal dialog with distinct modal frame styling."""
        win_id = f"dialog_{id(self)}"
        child_html = "".join(
            [c.draw() for c in self.children if getattr(c, "visible", True)]
        )
        return (
            f'<div id="{win_id}" class="actor-dialog actor-modal" data-title="{self.title}" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f"width: {self.rect.width}px; height: {self.rect.height}px; "
            f'background: #ECE9D8; border: 3px outset #D4D0C8; box-shadow: 4px 4px 12px rgba(0,0,0,0.5); z-index: 1000;">'
            f'<div class="actor-dialog-caption" style="height: 24px; background: #000080; color: #FFFFFF; '
            f'padding: 3px 8px; font-family: sans-serif; font-size: 13px; font-weight: bold;">'
            f"<span>{self.title}</span>"
            f"</div>"
            f'<div class="actor-dialog-client" style="position: relative; width: 100%; height: calc(100% - 24px); padding: 8px;">'
            f"{child_html}"
            f"</div>"
            f"</div>"
        )


# =============================================================================
# 4. Control (Base Interactive Control)
# =============================================================================


class Control(WindowsObject):
    """Authentic 1993 Actor Control class (CONTROL.CLS).

    Base class for interactive controls without children.
    """

    def __init__(
        self,
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 0,
        top: int = 0,
        width: int = 100,
        height: int = 30,
        title: str = "",
        obj_id: Optional[str] = None,
    ):
        super().__init__(title=title, id=id, identity=identity, obj_id=obj_id)
        self._class_name = "Control"
        if isinstance(rect, Rect):
            self.rect: Rect = rect
        elif isinstance(rect, (tuple, list)) and len(rect) == 4:
            self.rect = Rect(rect[0], rect[1], rect[2], rect[3])
        else:
            self.rect = Rect(left, top, left + width, top + height)

        self.data: Any = None
        self.selection: Any = None

    @property
    def left(self) -> int:
        return self.rect.left

    @property
    def top(self) -> int:
        return self.rect.top

    @property
    def width(self) -> int:
        return self.rect.width

    @property
    def height(self) -> int:
        return self.rect.height

    def should_have_children(self) -> bool:
        """Controls should never have children (CONTROL.CLS:96)."""
        return False

    def add(self, child: Any) -> Any:
        """Override add: controls do not accept children."""
        raise TypeError(f"Controls of type {self._class_name} cannot have children.")

    def get_value(self) -> Any:
        return self.data

    def set_value(self, val: Any) -> Any:
        self.data = val
        return val


# =============================================================================
# 5. Button (Push Button)
# =============================================================================


class Button(Control):
    """Authentic 1993 Actor Button class (BUTTON.CLS)."""

    def __init__(
        self,
        label: str = "OK",
        action_selector: Optional[Union[Symbol, str]] = None,
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 0,
        top: int = 0,
        width: int = 100,
        height: int = 30,
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        is_default: bool = False,
        target: Optional[Any] = None,
        callback: Optional[Callable] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            identity=identity,
            rect=rect,
            left=left,
            top=top,
            width=width,
            height=height,
            title=label,
            obj_id=obj_id,
        )
        self._class_name = "Button"
        self.label: str = str(label)
        self.action_selector: Symbol = (
            Symbol(action_selector)
            if action_selector is not None
            else Symbol("selected")
        )
        self.is_default: bool = is_default
        self.target: Optional[Any] = target
        self.callback: Optional[Callable] = callback
        self.pressed: bool = False

    def on_click(self, vm: Optional[Any] = None) -> Any:
        """Handle click event by dispatching action message into ActorVM or calling callback."""
        self.pressed = True
        self.data = Symbol("checked")
        active_vm = (
            vm
            or getattr(self, "vm", None)
            or (getattr(self.parent, "vm", None) if self.parent else None)
        )
        receiver = self.target or self.parent

        result = None
        # 1. Dispatch into ActorVM if active
        if active_vm is not None and receiver is not None:
            selector = str(self.action_selector)
            msg_obj = {
                "source": self,
                "action": Symbol(selector),
                "id": self.id or self.identity,
                "label": self.label,
            }
            handler_method = None
            if hasattr(receiver, "action_handlers"):
                child_key = str(self.identity or self.id or "")
                child_norm = child_key[1:] if child_key.startswith("#") else child_key
                event_norm = selector[1:] if selector.startswith("#") else selector
                handler_method = receiver.action_handlers.get((child_norm, event_norm))

            target_selector = handler_method or selector
            if hasattr(active_vm, "dispatch_message"):
                try:
                    result = active_vm.dispatch_message(
                        receiver, target_selector, [msg_obj]
                    )
                except Exception:
                    result = active_vm.dispatch_message(receiver, target_selector, [])

        # 2. Invoke Python callback if present
        if self.callback is not None:
            cb_res = self.callback(self)
            result = result if result is not None else cb_res

        # 3. Handle modal dialog termination (BUTTON.CLS:60-70)
        if isinstance(self.parent, Dialog):
            if self.id == Dialog.IDOK or self.action_selector == Symbol("okClicked"):
                self.parent.end(Dialog.IDOK)
            elif self.id == Dialog.IDCANCEL or self.action_selector == Symbol(
                "cancelClicked"
            ):
                self.parent.end(Dialog.IDCANCEL)

        self.pressed = False
        return result

    def press(self) -> Any:
        """Simulate programmatic button press."""
        return self.on_click()

    def draw(self) -> str:
        """Render HTML5 push button."""
        btn_id = f"btn_{id(self)}"
        default_cls = " actor-default-button" if self.is_default else ""
        return (
            f'<button id="{btn_id}" class="actor-button{default_cls}" data-label="{self.label}" '
            f'data-action="{self.action_selector}" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f"width: {self.rect.width}px; height: {self.rect.height}px; "
            f'font-family: sans-serif; font-size: 12px; font-weight: {("bold" if self.is_default else "normal")}; '
            f"background: #ECE9D8; border: 2px outset #FFFFFF; cursor: pointer; "
            f'display: inline-flex; align-items: center; justify-content: center;">'
            f"{self.label}"
            f"</button>"
        )


# =============================================================================
# 6. TextField (Text Input Control)
# =============================================================================


class TextField(Control):
    """Authentic 1993 Actor TextField class (EDIT.CLS)."""

    def __init__(
        self,
        text: str = "",
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 0,
        top: int = 0,
        width: int = 160,
        height: int = 24,
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        max_length: int = 255,
        obj_id: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            identity=identity,
            rect=rect,
            left=left,
            top=top,
            width=width,
            height=height,
            title=text,
            obj_id=obj_id,
        )
        self._class_name = "TextField"
        self.text: str = str(text) if text is not None else ""
        self.max_length: int = int(max_length)
        self.data: str = self.text

    def setText(self, new_text: str) -> None:
        """Set text content."""
        self.text = str(new_text)[: self.max_length]
        self.data = self.text

    def getText(self) -> str:
        """Retrieve text content."""
        return self.text

    def clear(self) -> None:
        """Clear text."""
        self.text = ""
        self.data = ""

    def draw(self) -> str:
        """Render HTML5 text input."""
        txt_id = f"txt_{id(self)}"
        return (
            f'<input type="text" id="{txt_id}" class="actor-textfield" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f"width: {self.rect.width}px; height: {self.rect.height}px; "
            f"font-family: monospace; font-size: 12px; background: #FFFFFF; "
            f'border: 2px inset #D4D0C8; padding: 2px 4px; box-sizing: border-box;" '
            f'value="{self.text}" maxlength="{self.max_length}">'
        )


# =============================================================================
# 7. Checkbox (Boolean Toggle Control)
# =============================================================================


class Checkbox(Button):
    """Authentic 1993 Actor Checkbox class (CHECKBOX.CLS)."""

    def __init__(
        self,
        label: str = "",
        checked: bool = False,
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 0,
        top: int = 0,
        width: int = 120,
        height: int = 24,
        id: Optional[Any] = None,
        identity: Optional[Symbol] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(
            label=label,
            action_selector=Symbol("toggled"),
            id=id,
            identity=identity,
            rect=rect,
            left=left,
            top=top,
            width=width,
            height=height,
            obj_id=obj_id,
        )
        self._class_name = "Checkbox"
        self.label: str = str(label)
        self.checked: bool = bool(checked)
        self.data: Symbol = Symbol("checked") if self.checked else Symbol("unChecked")

    def toggle(self) -> bool:
        """Toggle checked state and update data (CHECKBOX.CLS:80)."""
        self.checked = not self.checked
        self.data = Symbol("checked") if self.checked else Symbol("unChecked")
        return self.checked

    def on_click(self, msg: Optional[Any] = None, **kwargs: Any) -> Any:
        """Handle click/press event by toggling state and updating data (CHECKBOX.CLS:73)."""
        self.toggle()
        vm = kwargs.get("vm")
        if vm is None and msg is not None and not isinstance(msg, dict):
            vm = msg
        result = super().on_click(vm=vm)
        self.data = Symbol("checked") if self.checked else Symbol("unChecked")
        return result

    def check(self) -> None:
        """Check box."""
        self.checked = True
        self.data = Symbol("checked")

    def unCheck(self) -> None:
        """Uncheck box."""
        self.checked = False
        self.data = Symbol("unChecked")

    def value(self) -> bool:
        """Return boolean checked state."""
        return self.checked

    def setValue(self, val: Any) -> bool:
        """Set checked state from boolean or Symbol (CHECKBOX.CLS:16)."""
        if isinstance(val, (str, Symbol)):
            s = str(val).lower()
            self.checked = s in ("checked", "#checked", "true", "1")
        else:
            self.checked = bool(val)
        self.data = Symbol("checked") if self.checked else Symbol("unChecked")
        return self.checked

    def draw(self) -> str:
        """Render HTML5 checkbox input with label."""
        chk_id = f"chk_{id(self)}"
        checked_attr = " checked" if self.checked else ""
        return (
            f'<label id="{chk_id}_label" class="actor-checkbox" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f'display: inline-flex; align-items: center; font-family: sans-serif; font-size: 12px; cursor: pointer;">'
            f'<input type="checkbox" id="{chk_id}"{checked_attr} '
            f'style="margin-right: 6px;">'
            f"<span>{self.label}</span>"
            f"</label>"
        )


__all__ = [
    "WindowsObject",
    "Window",
    "Dialog",
    "Control",
    "Button",
    "TextField",
    "Checkbox",
]
