#!/usr/bin/env python3
"""WebSocket server for ActorII HTML5 canvas remote rendering and live workspace studio.

Provides an interactive 1993 Whitewater Actor 4.1 execution environment over WebSocket:
- 'eval': Compile and evaluate Actor chunks (ending in !!) on live ActorVM.
- 'get_classes': Query live ActorInheritanceHierarchy for registered classes, supers, methods, ivars.
- 'inspect_object': Inspect objects in Actor dictionary (Actor[#Name]), returning properties/fields.
- 'ui_event': Forward UI events (#selected, #clicked) to ActorVM.dispatch_action.
- Canvas graphics commands: set_pixel, draw_stroke, damage, clear, resize, render_widget.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

from actorii.classes.gui import (
    Button,
    Checkbox,
    Control,
    Dialog,
    TextField,
    Window,
    WindowsObject,
)
from actorii.core import ActorClass, ActorInheritanceHierarchy, ActorObject, Symbol
from actorii.parser import (
    ClassDefNode,
    MethodDefNode,
    NowNode,
    compile_to_bytecode,
    parse_actor,
)
from actorii.vm import ActorVM, Opcode


class WebSocketSurface:
    """HTML5 canvas surface driven via WebSocket commands (sparse pixel store)."""

    def __init__(
        self, width: int = 800, height: int = 600, title: str = "Actor Application"
    ):
        self._width = width
        self._height = height
        self._title = title
        self._canvas_id = f"actor_canvas_{id(self)}"
        self._commands: list[dict] = []
        self._dirty = True
        self._bg_color: str = "#FFFFFF"
        self._sparse_pixels: dict[tuple[int, int], str] = {}
        self._event_queue: list[dict] = []

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def stride(self) -> int:
        return self._width * 4

    @property
    def format(self) -> str:
        return "argb32"

    def post_event(self, event: Any) -> None:
        if isinstance(event, dict):
            self._event_queue.append(event)

    def poll_event(self) -> Optional[dict]:
        if self._event_queue:
            return self._event_queue.pop(0)
        return None

    def clear_pixels(self, color: str = "#FFFFFF") -> None:
        self._bg_color = color
        self._sparse_pixels.clear()
        self._dirty = True

    def fill_rect(self, x: int, y: int, w: int, h: int, color: str) -> int:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self._width, int(x + w))
        y1 = min(self._height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return 0
        count = (x1 - x0) * (y1 - y0)
        if x0 == 0 and y0 == 0 and x1 == self._width and y1 == self._height:
            self._bg_color = color
            self._sparse_pixels.clear()
            self._dirty = True
            return count
        if color == self._bg_color:
            if self._sparse_pixels:
                for py in range(y0, y1):
                    for px in range(x0, x1):
                        self._sparse_pixels.pop((px, py), None)
        else:
            for py in range(y0, y1):
                for px in range(x0, x1):
                    self._sparse_pixels[(px, py)] = color
        self._dirty = True
        return count

    def get_pixel(self, x: int, y: int) -> str:
        if 0 <= x < self._width and 0 <= y < self._height:
            return self._sparse_pixels.get((x, y), self._bg_color)
        return "#000000"

    def set_pixel(self, x: int, y: int, color: str) -> None:
        if 0 <= x < self._width and 0 <= y < self._height:
            if color == self._bg_color:
                self._sparse_pixels.pop((x, y), None)
            else:
                self._sparse_pixels[(x, y)] = color
            self._commands.append({"type": "pixel", "x": x, "y": y, "color": color})
            self._dirty = True

    def flush(self) -> None:
        self._dirty = True

    def damage(
        self, x: int = 0, y: int = 0, w: Optional[int] = None, h: Optional[int] = None
    ) -> None:
        if w is None:
            w = self._width - x
        if h is None:
            h = self._height - y
        self._commands.append({"type": "damage", "x": x, "y": y, "w": w, "h": h})
        self._dirty = True

    def copy_from(
        self,
        source: Any,
        x: int = 0,
        y: int = 0,
        w: Optional[int] = None,
        h: Optional[int] = None,
        dst_x: int = 0,
        dst_y: int = 0,
    ) -> None:
        if w is None:
            w = min(self._width - x, source.width - dst_x)
        if h is None:
            h = min(self._height - y, source.height - dst_y)
        self._commands.append(
            {
                "type": "copy",
                "src": getattr(source, "_canvas_id", "source"),
                "src_x": x,
                "src_y": y,
                "dst_x": dst_x,
                "dst_y": dst_y,
                "w": w,
                "h": h,
            }
        )
        self._dirty = True

    def flush_commands(self) -> list[dict]:
        cmds = list(self._commands)
        self._commands.clear()
        return cmds

    def _serialize_commands(self) -> str:
        if not self._commands:
            return json.dumps({"type": "noop"})
        cmds = list(self._commands)
        self._commands.clear()
        return json.dumps(cmds)


class WebSocketDriver:
    """Concrete WebSocket display driver for ActorII supporting multi-client broadcast."""

    def __init__(self, surface: WebSocketSurface, websocket: Any = None):
        self._surface = surface
        self._clients: set[Any] = set()
        self._primary_ws: Any = websocket
        if websocket is not None:
            self._clients.add(websocket)
        self._command_buffer: list[dict] = []

    @property
    def _ws(self) -> Any:
        if self._primary_ws is not None and (not self._clients or self._primary_ws in self._clients):
            return self._primary_ws
        if self._clients:
            return next(iter(self._clients))
        return None

    @_ws.setter
    def _ws(self, websocket: Any) -> None:
        self._primary_ws = websocket
        if websocket is not None:
            self._clients.add(websocket)
        else:
            self._clients.clear()

    def register_client(self, websocket: Any) -> None:
        if websocket is not None:
            self._clients.add(websocket)
            self._primary_ws = websocket

    def unregister_client(self, websocket: Any) -> None:
        self._clients.discard(websocket)
        if self._primary_ws is websocket:
            self._primary_ws = next(iter(self._clients), None)

    def _active_targets(self) -> list[Any]:
        if self._clients:
            return list(self._clients)
        return [self._primary_ws] if self._primary_ws is not None else []

    def _schedule_send(self, msg: str) -> None:
        targets = self._active_targets()
        if not targets:
            return
        try:
            loop = asyncio.get_running_loop()
            for client in targets:
                async def _safe_send(ws_client: Any) -> None:
                    try:
                        await ws_client.send(msg)
                    except Exception:
                        self.unregister_client(ws_client)

                loop.create_task(_safe_send(client))
        except RuntimeError:
            pass

    def present(self) -> None:
        if self._active_targets():
            cmd = self._surface._serialize_commands()
            msg = json.dumps({"type": "flush", "commands": cmd})
            self._schedule_send(msg)
        else:
            self._surface._dirty = True

    def resize(self, width: int, height: int) -> None:
        self._surface._width = width
        self._surface._height = height
        self._surface.clear_pixels("#FFFFFF")
        self._surface._commands.append(
            {"type": "resize", "width": width, "height": height}
        )
        self._surface._dirty = True

    def handle_event(self, event: Any) -> bool:
        msg_type = event.get("type", "") if isinstance(event, dict) else ""
        if msg_type == "event":
            event_data = event.get("data", {})
            etype = event_data.get("type", "")
            if etype == "keyboard":
                key = event_data.get("key", "")
                down = event_data.get("down", True)
                self._surface.post_event({"type": "keyboard", "key": key, "down": down})
                return True
            elif etype == "mousebutton":
                pressed = event_data.get("pressed", False)
                x = event_data.get("x", 0)
                y = event_data.get("y", 0)
                self._surface.post_event(
                    {"type": "mousebutton", "pressed": pressed, "pos": (x, y)}
                )
                return True
            elif etype == "mousemove":
                x = event_data.get("x", 0)
                y = event_data.get("y", 0)
                self._surface.post_event({"type": "mousemove", "pos": (x, y)})
                return True
        elif msg_type == "resize":
            width = event.get("width", self._surface.width)
            height = event.get("height", self._surface.height)
            self.resize(width, height)
            return True
        return False

    def clear(self, color: str = "#FFFFFF") -> None:
        self._surface.clear_pixels(color)
        self._surface._commands.append({"type": "clear", "color": color})
        self._surface._dirty = True

    def send_commands(self, commands: Any) -> None:
        if not self._active_targets():
            return
        if isinstance(commands, list):
            by_type: dict[str, list] = {}
            for cmd in commands:
                cmd_type = cmd.get("type", "unknown")
                by_type.setdefault(cmd_type, []).append(cmd)
            for cmd_type, cmds in by_type.items():
                msg = json.dumps({"type": cmd_type, "commands": cmds})
                self._schedule_send(msg)
        else:
            msg = json.dumps(
                {"type": commands.get("type", "unknown"), "data": commands}
            )
            self._schedule_send(msg)


def create_surface(
    width: int = 800, height: int = 600, title: str = "Actor Application"
) -> WebSocketSurface:
    return WebSocketSurface(width, height, title)


HIST_ROOT = os.path.join(os.path.dirname(__file__), "historical_actor_41_source")


def _solve_n_queens(n: int = 8, max_solutions: int = 92) -> list[list[int]]:
    solutions: list[list[int]] = []
    cols: set[int] = set()
    diag1: set[int] = set()
    diag2: set[int] = set()
    board: list[int] = [-1] * n

    def backtrack(row: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if row == n:
            solutions.append(list(board))
            return
        for col in range(n):
            d1 = row - col
            d2 = row + col
            if col in cols or d1 in diag1 or d2 in diag2:
                continue
            board[row] = col
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)

    backtrack(0)
    return solutions


def _run_sieve(limit: int = 200) -> dict:
    limit = max(10, min(5000, int(limit)))
    flags = [True] * (limit + 1)
    flags[0] = flags[1] = False
    p = 2
    while p * p <= limit:
        if flags[p]:
            for i in range(p * p, limit + 1, p):
                flags[i] = False
        p += 1
    primes = [i for i, is_p in enumerate(flags) if is_p]
    bucket_size = max(10, limit // 8)
    buckets = []
    for start in range(0, limit, bucket_size):
        end = min(limit, start + bucket_size - 1)
        cnt = sum(1 for pr in primes if start <= pr <= end)
        buckets.append({"label": f"{start}-{end}", "count": cnt})
    return {
        "limit": limit,
        "count": len(primes),
        "primes": primes[:60],
        "buckets": buckets,
    }


def _serialize_value(val: Any) -> Any:
    """Convert ActorVM runtime values into JSON-serializable primitives."""
    if val is None or isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, Symbol):
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, ActorClass):
        return {
            "__type__": "ActorClass",
            "name": val.name,
            "bases": list(val._bases),
            "ivars": list(getattr(val, "ivars", getattr(val, "_ivars", []))),
        }
    if isinstance(val, ActorObject):
        cls_name = getattr(val, "class_name", type(val).__name__)
        obj_ident = getattr(
            val, "identity", getattr(val, "id", getattr(val, "obj_id", id(val)))
        )
        return {
            "__type__": "ActorObject",
            "class": cls_name,
            "id": str(obj_ident),
            "repr": str(val),
        }
    if hasattr(val, "asArray") and callable(getattr(val, "asArray")):
        try:
            return _serialize_value(list(val.asArray()))
        except Exception:
            pass
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            return [_serialize_value(x) for x in list(val)[:100]]
        except Exception:
            pass
    return str(val)


class ActorWorkspace:
    """Interactive Actor 4.1 execution session maintaining live VM and display state."""

    def __init__(
        self,
        vm: Optional[ActorVM] = None,
        surface: Optional[WebSocketSurface] = None,
        title: str = "ActorII Remote",
    ):
        self.vm: ActorVM = vm if vm is not None else ActorVM()
        self.surface: WebSocketSurface = (
            surface if surface is not None else create_surface(800, 600, title)
        )
        self.driver: WebSocketDriver = WebSocketDriver(self.surface)
        self.widgets: dict[str, Any] = {}
        self._vm_lock: asyncio.Lock = asyncio.Lock()

    def _resolve_widget(self, widget_or_window: Any) -> Any:
        """Resolve a widget identifier or instance across workspace and VM scopes."""
        if isinstance(widget_or_window, (WindowsObject, ActorObject)):
            return widget_or_window
        if widget_or_window in self.widgets:
            return self.widgets[widget_or_window]
        norm_id = str(widget_or_window)
        if norm_id.startswith("#"):
            norm_id = norm_id[1:]
        target = self.vm.global_vars.get(norm_id) or self.vm.global_vars.get(
            "Actor", {}
        ).get(norm_id)
        if target is not None:
            return target
        for val in self.vm.global_vars.get("Actor", {}).values():
            if isinstance(val, Window):
                child = val.find_child(widget_or_window) or val.find_child(norm_id)
                if child is not None:
                    return child
        return None

    def _eval_source(self, source: str) -> Any:
        """Evaluate Actor code on VM, handling implicit return for final expression statement."""
        res = self.vm.eval_chunk(source)
        self._sync_windows_to_surface()
        return res

    def _sync_windows_to_surface(self) -> None:
        """Attach and render any live Window objects in `Actor[...]` onto `self.surface`."""
        actor_dict = self.vm.global_vars.get("Actor", {})
        if not isinstance(actor_dict, dict):
            return
        for k, val in list(actor_dict.items()):
            norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
            if isinstance(val, Window):
                self.widgets[norm_k] = val
                try:
                    self.driver.render_widget(val)
                except Exception:
                    pass
            elif hasattr(val, "renderOnSurface") and callable(getattr(val, "renderOnSurface")):
                try:
                    val.renderOnSurface(self.surface)
                except Exception:
                    pass

    async def handle_message(
        self, message: Union[str, dict], websocket: Any = None
    ) -> dict:
        """Dispatch an incoming JSON protocol message serialized under `_vm_lock`."""
        if isinstance(message, str):
            raw = json.loads(message)
        else:
            raw = message
        msg_type = raw.get("type", "")
        if msg_type in ("stop_goal", "cancel_goal"):
            actor_dict = self.vm.global_vars.get("Actor", {})
            harness_obj = actor_dict.get("Harness")
            if harness_obj is not None and hasattr(harness_obj, "_fields"):
                harness_obj._fields["status"] = Symbol("halted")
                step_cnt = int(harness_obj._fields.get("stepCount", 0) or 0)
                return {
                    "type": "goal_stopped",
                    "success": True,
                    "steps_count": step_cnt,
                    "message": f"Interrupted active AgentHarness loop at step {step_cnt}.",
                }
            return {
                "type": "goal_stopped",
                "success": True,
                "steps_count": 0,
                "message": "No active AgentHarness loop.",
            }
        async with self._vm_lock:
            return await self._handle_message_unlocked(raw, websocket=websocket)

    async def _handle_message_unlocked(
        self, raw: dict, websocket: Any = None
    ) -> dict:
        msg_type = raw.get("type", "")
        payload = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        print(f"[WS-MSG] {msg_type}", flush=True)

        # =====================================================================
        # 1. Milestone M5 Core Handlers
        # =====================================================================

        if msg_type == "eval":
            source = payload.get("source")
            if source is None:
                return {
                    "type": "eval_result",
                    "success": False,
                    "error": "Missing 'source' parameter in eval request",
                }
            try:
                res = self._eval_source(str(source))
                thermo_ui = self.vm.global_vars.get("Actor", {}).get("ThermoUI")
                thermo_html = (
                    thermo_ui.draw()
                    if thermo_ui is not None and hasattr(thermo_ui, "draw")
                    else None
                )
                return {
                    "type": "eval_result",
                    "result": _serialize_value(res),
                    "thermometer_html": thermo_html,
                    "success": True,
                }
            except Exception as e:
                return {
                    "type": "eval_result",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "get_classes":
            classes_info: list[dict] = []
            for name, cls in sorted(self.vm.hierarchy._classes.items()):
                super_name = cls._bases[0] if cls._bases else None
                methods = sorted(
                    list(
                        set(cls._class_methods.keys())
                        | {m for m in cls._methods.keys() if not m.startswith("_")}
                    )
                )
                ivars = list(getattr(cls, "ivars", getattr(cls, "_ivars", [])))
                classes_info.append(
                    {
                        "name": name,
                        "super": super_name,
                        "methods": methods,
                        "ivars": ivars,
                    }
                )
            return {
                "type": "classes_list",
                "classes": classes_info,
                "count": len(classes_info),
            }

        elif msg_type == "inspect_object":
            key = payload.get("key") or payload.get("name") or payload.get("id")
            if key is None:
                return {
                    "type": "object_details",
                    "found": False,
                    "error": "Missing 'key' parameter in inspect_object request",
                }

            skey = str(key)
            norm_key = skey[1:] if skey.startswith("#") else skey
            sym_key = Symbol(norm_key)

            obj = None
            actor_dict = self.vm.global_vars.get("Actor", {})
            if norm_key in actor_dict:
                obj = actor_dict[norm_key]
            elif sym_key in actor_dict:
                obj = actor_dict[sym_key]
            elif norm_key in self.vm.global_vars:
                obj = self.vm.global_vars[norm_key]
            elif norm_key in self.widgets:
                obj = self.widgets[norm_key]
            else:
                # Search open windows in Actor dictionary for child widgets
                for val in list(actor_dict.values()):
                    if isinstance(val, Window):
                        child = val.find_child(norm_key) or val.find_child(sym_key)
                        if child is not None:
                            obj = child
                            break

            if obj is None:
                return {
                    "type": "object_details",
                    "key": skey,
                    "found": False,
                    "error": f"Object '{skey}' not found in Actor dictionary",
                }

            cls_name = getattr(
                obj, "class_name", getattr(obj, "_class_name", type(obj).__name__)
            )
            fields: dict[str, Any] = {}

            if isinstance(obj, ActorClass):
                instance_methods = sorted(list(obj._methods.keys()))
                class_methods = sorted(list(obj._class_methods.keys()))
                all_methods = sorted(list(set(instance_methods) | set(class_methods)))
                fields = {
                    "name": obj.name,
                    "bases": list(obj._bases),
                    "methods": all_methods,
                    "instance_methods": instance_methods,
                    "class_methods": class_methods,
                    "ivars": list(getattr(obj, "ivars", getattr(obj, "_ivars", []))),
                }
            elif isinstance(obj, ActorObject):
                fields.update(dict(getattr(obj, "_fields", {})))
                for attr in [
                    "id",
                    "identity",
                    "title",
                    "label",
                    "text",
                    "checked",
                    "visible",
                    "enabled",
                    "data",
                    "action_selector",
                    "is_default",
                ]:
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if val is not None:
                            fields[attr] = _serialize_value(val)
                if hasattr(obj, "rect") and getattr(obj, "rect") is not None:
                    r = obj.rect
                    fields["rect"] = [r.left, r.top, r.right, r.bottom]
            elif isinstance(obj, (dict, list, tuple)):
                fields = {"content": _serialize_value(obj)}
            else:
                fields = {"value": _serialize_value(obj)}

            return {
                "type": "object_details",
                "key": skey,
                "class": cls_name,
                "fields": fields,
                "found": True,
            }

        elif msg_type == "ui_event":
            widget_id = payload.get("widget_id") or payload.get("id")
            action = payload.get("action", "selected")
            msg_data = payload.get("msg")
            args = payload.get("args")

            if widget_id is None:
                return {
                    "type": "ui_event_result",
                    "handled": False,
                    "error": "Missing 'widget_id' parameter in ui_event request",
                }

            target = self._resolve_widget(widget_id)
            if target is None:
                return {
                    "type": "ui_event_result",
                    "widget_id": str(widget_id),
                    "action": str(action),
                    "handled": False,
                    "error": f"Widget '{widget_id}' could not be resolved",
                }

            try:
                res = self.vm.dispatch_action(
                    target,
                    selector=str(action),
                    msg=msg_data,
                    args=args,
                )
                resp_payload: Dict[str, Any] = {
                    "type": "ui_event_result",
                    "widget_id": str(widget_id),
                    "action": str(action),
                    "handled": True,
                    "result": _serialize_value(res),
                }
                thermo_ui = self.vm.global_vars.get("Actor", {}).get("ThermoUI")
                if thermo_ui is not None and hasattr(thermo_ui, "draw"):
                    resp_payload["thermometer_html"] = thermo_ui.draw()
                return resp_payload
            except Exception as e:
                return {
                    "type": "ui_event_result",
                    "widget_id": str(widget_id),
                    "action": str(action),
                    "handled": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        # =====================================================================
        # 2. Rendering & Canvas Handlers
        # =====================================================================

        elif msg_type == "clear":
            color = payload.get("color", "#FFFFFF")
            self.driver.clear(color)
            return {"type": "cleared", "color": color}

        elif msg_type == "set_pixel":
            x = int(payload.get("x", 0))
            y = int(payload.get("y", 0))
            color = payload.get("color", "#FF0000")
            self.surface.set_pixel(x, y, color)
            return {"type": "pixel_color", "x": x, "y": y, "color": color}

        elif msg_type == "draw_stroke":
            pts = payload.get("points", [])
            color = payload.get("color", "#e94560")
            size = int(payload.get("size", 3))
            for pt in pts:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    self.surface.set_pixel(int(pt[0]), int(pt[1]), color)
            return {
                "type": "stroke_ack",
                "count": len(pts),
                "color": color,
                "size": size,
            }

        elif msg_type == "damage":
            x = int(payload.get("x", 0))
            y = int(payload.get("y", 0))
            w = payload.get("w")
            h = payload.get("h")
            self.surface.damage(x, y, w, h)
            return {"type": "damaged", "x": x, "y": y, "w": w, "h": h}

        elif msg_type == "resize":
            w = int(payload.get("width", self.surface.width))
            h = int(payload.get("height", self.surface.height))
            self.driver.resize(w, h)
            return {"type": "resized", "width": w, "height": h}

        elif msg_type == "flush":
            self.driver.present()
            return {"type": "flush_ack"}

        elif msg_type == "post_event":
            event = payload.get("event", payload)
            self.surface.post_event(event)
            return {"type": "event_ack", "event": event}

        elif msg_type == "get_surface_info":
            info = {
                "width": self.surface.width,
                "height": self.surface.height,
                "canvas_id": self.surface._canvas_id,
                "format": self.surface.format,
            }
            return {"type": "surface_info", "data": info}

        elif msg_type == "get_pixel":
            x = int(payload.get("x", 0))
            y = int(payload.get("y", 0))
            color = self.surface.get_pixel(x, y)
            return {"type": "pixel_color", "x": x, "y": y, "color": color}

        elif msg_type == "render_widget":
            widget_type = str(payload.get("widget_type", "button")).lower()
            x = int(payload.get("x", 40))
            y = int(payload.get("y", 80))
            label = str(payload.get("label", "Actor Control"))
            width = int(payload.get("width", 120))
            height = int(payload.get("height", 32))
            widget_id = str(payload.get("id", f"widget_{len(self.widgets) + 1}"))

            left_pct = max(1.0, min(82.0, (x / max(1, self.surface.width)) * 100.0))
            top_pct = max(6.0, min(88.0, (y / max(1, self.surface.height)) * 100.0))

            if widget_type == "button":
                btn = Button(
                    label=label,
                    action_selector=Symbol("selected"),
                    identity=Symbol(widget_id),
                    id=widget_id,
                    left=x,
                    top=y,
                    width=width,
                    height=height,
                )
                self.vm.global_vars["Actor"][widget_id] = btn
                self.widgets[widget_id] = btn
                html = (
                    f'<button data-widget="button" data-id="{widget_id}" data-label="{label}" '
                    f'style="position: absolute; left: {left_pct:.1f}%; top: {top_pct:.1f}%; '
                    f"padding: 6px 12px; font-family: sans-serif; "
                    f"font-size: 13px; font-weight: 600; background: #16213e; color: #fff; "
                    f"border: 2px solid #e94560; border-radius: 6px; cursor: pointer; "
                    f'box-shadow: 0 2px 6px rgba(0,0,0,0.35);">{label}</button>'
                )
            elif widget_type == "checkbox":
                checked = bool(payload.get("checked", False))
                chk = Checkbox(
                    label=label,
                    checked=checked,
                    identity=Symbol(widget_id),
                    id=widget_id,
                    left=x,
                    top=y,
                    width=width,
                    height=height,
                )
                self.vm.global_vars["Actor"][widget_id] = chk
                self.widgets[widget_id] = chk
                html = (
                    f'<label data-widget="checkbox" data-id="{widget_id}" data-label="{label}" '
                    f'style="position: absolute; left: {left_pct:.1f}%; top: {top_pct:.1f}%; '
                    f"background: rgba(22,33,62,0.95); color: #fff; padding: 6px 10px; "
                    f"border: 1px solid #1b9aaa; border-radius: 6px; font-family: sans-serif; "
                    f'font-size: 13px; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">'
                    f'<input type="checkbox" {"checked" if checked else ""}> {label}</label>'
                )
            elif widget_type == "textfield":
                txt = TextField(
                    text=label,
                    identity=Symbol(widget_id),
                    id=widget_id,
                    left=x,
                    top=y,
                    width=width,
                    height=height,
                )
                self.vm.global_vars["Actor"][widget_id] = txt
                self.widgets[widget_id] = txt
                html = (
                    f'<input data-widget="textfield" data-id="{widget_id}" data-label="{label}" type="text" placeholder="{label}" '
                    f'style="position: absolute; left: {left_pct:.1f}%; top: {top_pct:.1f}%; '
                    f"width: {width}px; padding: 5px 8px; background: #fff; color: #111; "
                    f"border: 2px solid #0f3460; border-radius: 5px; "
                    f'font-family: monospace; font-size: 13px;" value="">'
                )
            else:
                html = (
                    f'<div data-widget="{widget_type}" data-id="{widget_id}" style="position: absolute; '
                    f"left: {left_pct:.1f}%; top: {top_pct:.1f}%; "
                    f'padding: 6px 10px; background: #fff; border: 1px solid #000;">{label}</div>'
                )

            return {
                "type": "rendered_widget",
                "html": html,
                "x": x,
                "y": y,
                "label": label,
                "widget_type": widget_type,
                "widget_id": widget_id,
            }

        elif msg_type == "render_thermometer":
            from pathlib import Path
            from actorii.classes.thermopro import ThermoProTwin, ThermometerUI

            actor_dict = self.vm.global_vars.get("Actor", {})
            if "Thermometer" not in actor_dict or "ThermoUI" not in actor_dict:
                act_file = Path(__file__).resolve().parent / "examples" / "tp358w_digital_twin.act"
                if act_file.is_file():
                    self.vm.eval_chunk(act_file.read_text(encoding="utf-8"))
                else:
                    twin_obj = ThermoProTwin(vm=self.vm)
                    ui_obj = ThermometerUI(twin=twin_obj, vm=self.vm)
                    actor_dict["Thermometer"] = twin_obj
                    actor_dict["ThermoUI"] = ui_obj

            twin_obj = actor_dict["Thermometer"]
            ui_obj = actor_dict["ThermoUI"]
            ui_obj.attachSurface(self.surface)
            self.widgets["ThermoUI"] = ui_obj
            for ch in getattr(ui_obj, "children", []):
                cid = str(getattr(ch, "identity", getattr(ch, "id", ""))).lstrip("#")
                if cid:
                    self.widgets[cid] = ch

            if payload.get("advertisement_hex"):
                twin_obj.ingestAdvertisementHex(
                    str(payload["advertisement_hex"]),
                    int(payload.get("rssi", -75)),
                )
            if payload.get("history_hex"):
                twin_obj.ingestHistoryHex(
                    str(payload["history_hex"]),
                    payload.get("fetch_time"),
                )
            if payload.get("sync_live"):
                try:
                    twin_obj.syncLive(float(payload.get("timeout", 6.0)))
                except Exception:
                    pass
            if payload.get("download_history"):
                try:
                    twin_obj.downloadHistory(int(payload.get("count", 60)))
                except Exception:
                    pass

            surface_stats = ui_obj.renderOnSurface(self.surface)
            return {
                "type": "thermometer_rendered",
                "html": ui_obj.draw(),
                "surface_stats": surface_stats,
                "temperature_c": twin_obj.temperatureC(),
                "temperature_f": twin_obj.temperatureF(),
                "humidity": twin_obj.humidity(),
                "comfort_level": str(twin_obj.comfortLevel()),
                "dew_point_c": twin_obj.dewPointC(),
                "battery": twin_obj.battery(),
                "rssi": twin_obj.rssi(),
                "history_count": twin_obj.historyCount(),
                "temp_text": ui_obj.tempText(),
                "humidity_text": ui_obj.humidityText(),
                "comfort_text": ui_obj.comfortText(),
                "stats_text": ui_obj.statsText(),
            }

        # =====================================================================
        # 3. Benchmark & Historical Exploration Handlers
        # =====================================================================

        elif msg_type == "solve_queens":
            n = int(payload.get("n", 8))
            idx = int(payload.get("index", 0))
            sols = _solve_n_queens(n)
            chosen = sols[idx % len(sols)] if sols else []
            return {
                "type": "queens_solution",
                "n": n,
                "index": idx % max(1, len(sols)),
                "total_solutions": len(sols),
                "board": chosen,
            }

        elif msg_type == "run_sieve":
            limit = int(payload.get("limit", 200))
            res = _run_sieve(limit)
            return {"type": "sieve_result", "data": res}

        elif msg_type == "list_actor_files":
            folder = str(payload.get("folder", "actcls"))
            allowed = {
                "actcls",
                "actact",
                "drawcls",
                "oglcls",
                "sqlcls",
                "vidcls",
                "actres",
            }
            if folder not in allowed:
                folder = "actcls"
            target_dir = os.path.join(HIST_ROOT, folder)
            files = sorted(os.listdir(target_dir)) if os.path.isdir(target_dir) else []
            return {
                "type": "actor_files",
                "folder": folder,
                "folders": sorted(allowed),
                "files": files,
            }

        elif msg_type == "read_actor_file":
            folder = os.path.basename(str(payload.get("folder", "actcls")))
            filename = os.path.basename(str(payload.get("file", "OBJECT.CLS")))
            fpath = os.path.join(HIST_ROOT, folder, filename)
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="latin-1", errors="replace") as f:
                    content = f.read(16000)
                return {
                    "type": "actor_file_content",
                    "folder": folder,
                    "file": filename,
                    "content": content,
                }
            else:
                return {
                    "type": "error",
                    "message": f"File not found: {folder}/{filename}",
                }

        elif msg_type == "run_actor_vm":
            source = str(payload.get("source", ""))
            ast_nodes = parse_actor(source)
            instructions, constants, strings = compile_to_bytecode(ast_nodes)
            classes_summary = [
                {
                    "name": getattr(c, "name", "Anonymous"),
                    "bases": list(getattr(c, "bases", ())),
                    "methods": [
                        {"name": m.name, "params": list(getattr(m, "params", []))}
                        for m in getattr(c, "methods", [])
                    ],
                }
                for c in ast_nodes
                if hasattr(c, "methods")
            ]
            return {
                "type": "vm_result",
                "classes": classes_summary,
                "instructions": [[int(op), int(arg)] for op, arg in instructions],
                "constants": [str(c) for c in constants],
                "strings": strings,
            }

        # =====================================================================
        # 4. Image Persistence Handlers (save_image, load_image, list_images)
        # =====================================================================

        elif msg_type == "save_image":
            raw_path = payload.get("path") or payload.get("filename") or "ACTOR.IMG"
            try:
                saved_path = self.vm.save_image(str(raw_path))
                return {
                    "type": "image_saved",
                    "success": True,
                    "path": saved_path,
                    "classes_count": len(self.vm.hierarchy._classes),
                }
            except Exception as e:
                return {
                    "type": "image_saved",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "load_image":
            raw_path = payload.get("path") or payload.get("filename") or "ACTOR.IMG"
            try:
                stats = self.vm.load_image(str(raw_path))
                # Sync workspace widget registry with restored Actor dictionary
                self.widgets.clear()
                actor_dict = self.vm.global_vars.get("Actor", {})
                for k, v in list(actor_dict.items()):
                    if isinstance(v, WindowsObject):
                        norm_k = str(k)[1:] if str(k).startswith("#") else str(k)
                        self.widgets[norm_k] = v
                return {
                    "type": "image_loaded",
                    "success": True,
                    "path": stats.get("path", str(raw_path)),
                    "classes_count": stats.get("classes_count", 0),
                    "objects_count": stats.get("objects_count", 0),
                }
            except Exception as e:
                return {
                    "type": "image_loaded",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "jit_synthesize":
            cls_name = str(payload.get("class_name", "Object"))
            selector = str(payload.get("selector", "synthesizedMethod"))
            spec = str(payload.get("spec", ""))
            arity = int(payload.get("arity", 0))
            try:
                from actorii.ai import ActorMethodSynthesizer

                cls_obj = self.vm.hierarchy.get(cls_name)
                receiver = (
                    cls_obj._default_constructor() if cls_obj else ActorObject()
                )
                synth = ActorMethodSynthesizer(self.vm)
                cm = synth.synthesize_method(
                    receiver=receiver,
                    class_name=cls_name,
                    selector=selector,
                    args=[0] * arity,
                    spec_prompt=spec,
                )
                return {
                    "type": "jit_synthesized",
                    "success": True,
                    "class_name": cls_name,
                    "selector": selector,
                    "source": getattr(cm, "source", ""),
                }
            except Exception as e:
                return {
                    "type": "jit_synthesized",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "list_synthesized_methods":
            return {
                "type": "synthesized_methods",
                "methods": list(getattr(self.vm, "synthesized_methods", [])),
            }

        elif msg_type == "eval_transactional":
            source = payload.get("source")
            if source is None:
                return {
                    "type": "eval_result",
                    "success": False,
                    "error": "Missing 'source' parameter in eval_transactional request",
                }
            max_ops = payload.get("max_opcodes")
            shadow_only = bool(payload.get("shadow_only", False))
            try:
                res = self.vm.eval_transactional(
                    str(source),
                    max_opcodes=int(max_ops) if max_ops is not None else None,
                    shadow_only=shadow_only,
                )
                return {
                    "type": "eval_result",
                    "result": _serialize_value(res),
                    "success": True,
                    "transactional": True,
                    "shadow_only": shadow_only,
                }
            except Exception as e:
                return {
                    "type": "eval_result",
                    "success": False,
                    "transactional": True,
                    "shadow_only": shadow_only,
                    "rolled_back": True,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "enable_guardrails":
            enabled = bool(payload.get("enabled", True))
            protect_sys = bool(payload.get("protect_system_classes", True))
            strict_ivars = bool(payload.get("strict_ivars", True))
            max_ops = payload.get("max_opcodes", 100_000)
            fileout_dir = payload.get("fileout_dir")
            self.vm.enable_guardrails(
                enabled,
                protect_system_classes=protect_sys,
                strict_ivars=strict_ivars,
                max_opcodes=int(max_ops) if max_ops is not None else None,
                fileout_dir=str(fileout_dir) if fileout_dir else self.vm.fileout_dir,
            )
            return {
                "type": "guardrails_status",
                "success": True,
                "guardrails_enabled": self.vm.guardrails_enabled,
                "protect_system_classes": self.vm.protect_system_classes,
                "strict_ivars": self.vm.strict_ivars,
                "max_opcodes": self.vm.max_opcodes,
                "fileout_dir": self.vm.fileout_dir,
            }

        elif msg_type == "get_guardrail_audit":
            return {
                "type": "guardrail_audit",
                "success": True,
                "guardrails_enabled": self.vm.guardrails_enabled,
                "protect_system_classes": self.vm.protect_system_classes,
                "strict_ivars": self.vm.strict_ivars,
                "max_opcodes": self.vm.max_opcodes,
                "fileout_dir": self.vm.fileout_dir,
                "audit_log": list(getattr(self.vm, "guardrail_audit_log", [])),
            }

        elif msg_type == "file_out_class":
            cls_name = str(payload.get("class_name", ""))
            if not cls_name:
                return {
                    "type": "class_filed_out",
                    "success": False,
                    "error": "Missing 'class_name' parameter",
                }
            try:
                from pathlib import Path
                from actorii.guardrails.fileout import ActorFileOutManager

                out_dir = (
                    self.vm.fileout_dir
                    or str(Path(__file__).resolve().parent / "examples" / "fileout")
                )
                mgr = ActorFileOutManager(out_dir)
                out_path = mgr.file_out_class(self.vm, cls_name)
                return {
                    "type": "class_filed_out",
                    "success": True,
                    "class_name": cls_name,
                    "path": str(out_path),
                    "source": out_path.read_text(encoding="utf-8"),
                }
            except Exception as e:
                return {
                    "type": "class_filed_out",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type in ("run_goal", "agent_chat"):
            goal_str = str(
                payload.get("goal", "")
                or payload.get("objective", "")
                or payload.get("message", "")
            ).strip()
            raw_ms = payload.get("max_steps", 0)
            max_steps = 0 if raw_ms is None else int(raw_ms)
            if not goal_str:
                return {
                    "type": "goal_result",
                    "success": False,
                    "error": "Missing 'goal', 'objective', or 'message' parameter in request",
                }
            try:
                from actorii.harness import bootstrap_agent_harness

                actor_dict = self.vm.global_vars.get("Actor", {})
                if "Harness" not in actor_dict:
                    _, self.harness, self.sandbox = bootstrap_agent_harness(
                        self.vm, max_steps=25, max_failures=6
                    )
                harness_obj = self.vm.global_vars["Actor"]["Harness"]
                if max_steps <= 0:
                    max_steps = int(harness_obj._fields.get("maxSteps", 25) or 25)
                escaped = goal_str.replace("\\", "\\\\").replace('"', '\\"')
                loop = asyncio.get_running_loop()

                def _on_step(entry: dict) -> None:
                    if websocket is not None:
                        msg = json.dumps(
                            {
                                "type": "goal_step",
                                "step": entry.get("step"),
                                "action": entry.get("action"),
                                "status": entry.get("status"),
                                "tier": entry.get("tier"),
                                "gas_used": entry.get("gas_used"),
                                "result": _serialize_value(entry.get("result")),
                                "auto_engrams": entry.get("auto_engrams", []),
                                "error": entry.get("error"),
                            }
                        )
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(websocket.send(msg))
                        )

                sb = getattr(self, "sandbox", None)
                if sb is not None and hasattr(sb, "step_callbacks"):
                    sb.step_callbacks.append(_on_step)
                try:
                    res = await asyncio.to_thread(
                        self.vm.eval_chunk,
                        f'runGoal(Actor[#Harness], "{escaped}", {max_steps})!!',
                    )
                finally:
                    if sb is not None and hasattr(sb, "step_callbacks") and _on_step in sb.step_callbacks:
                        sb.step_callbacks.remove(_on_step)
                self._sync_windows_to_surface()
                status_str = str(harness_obj._fields.get("status", "#complete"))
                step_count = int(harness_obj._fields.get("stepCount", 0) or 0)
                last_res = _serialize_value(harness_obj._fields.get("lastResult"))
                raw_logs = _serialize_value(
                    list(harness_obj._fields.get("stepLog", []))
                )
                chat_hist = _serialize_value(
                    sb.getChatHistory(harness_obj)
                    if sb is not None and hasattr(sb, "getChatHistory")
                    else []
                )
                scratch_raw = harness_obj._fields.get("scratchpad")
                scratch_dict = (
                    {
                        (k.name if isinstance(k, Symbol) else str(k).lstrip("#")): _serialize_value(v)
                        for k, v in getattr(scratch_raw, "_entries", {}).items()
                    }
                    if scratch_raw is not None
                    else {}
                )
                thermo_ui = self.vm.global_vars.get("Actor", {}).get("ThermoUI")
                thermo_html = (
                    thermo_ui.draw()
                    if thermo_ui is not None and hasattr(thermo_ui, "draw")
                    else None
                )
                return {
                    "type": "goal_result",
                    "success": "complete" in status_str.lower(),
                    "goal": goal_str,
                    "status": status_str,
                    "steps": step_count,
                    "steps_count": step_count,
                    "gas_used": int(harness_obj._fields.get("totalGasUsed", 0) or 0),
                    "failures": int(harness_obj._fields.get("failures", 0) or 0),
                    "max_failures": int(harness_obj._fields.get("maxFailures", 3) or 3),
                    "model": getattr(harness_obj, "model", "qwen3.8:latest"),
                    "result": _serialize_value(res),
                    "last_result": last_res,
                    "final_summary": str(last_res or _serialize_value(res)),
                    "step_log": raw_logs,
                    "chat_history": chat_hist,
                    "scratchpad": scratch_dict,
                    "thermometer_html": thermo_html,
                    "commands": self.surface.flush_commands(),
                }
            except Exception as e:
                return {
                    "type": "goal_result",
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                }

        elif msg_type == "get_chat_history":
            from actorii.harness import bootstrap_agent_harness

            actor_dict = self.vm.global_vars.get("Actor", {})
            if "Harness" not in actor_dict:
                _, self.harness, self.sandbox = bootstrap_agent_harness(self.vm)
            harness_obj = self.vm.global_vars["Actor"]["Harness"]
            sb = getattr(self, "sandbox", None)
            chat_hist = _serialize_value(
                sb.getChatHistory(harness_obj)
                if sb is not None and hasattr(sb, "getChatHistory")
                else []
            )
            scratch_raw = harness_obj._fields.get("scratchpad")
            scratch_dict = (
                {
                    (k.name if isinstance(k, Symbol) else str(k).lstrip("#")): _serialize_value(v)
                    for k, v in getattr(scratch_raw, "_entries", {}).items()
                }
                if scratch_raw is not None
                else {}
            )
            return {
                "type": "chat_history",
                "success": True,
                "chat_history": chat_hist,
                "scratchpad": scratch_dict,
                "status": str(harness_obj._fields.get("status", "#idle")),
                "model": getattr(harness_obj, "model", "qwen3.8:latest"),
                "endpoint": getattr(harness_obj, "endpoint", ""),
                "gas_used": int(harness_obj._fields.get("totalGasUsed", 0) or 0),
                "failures": int(harness_obj._fields.get("failures", 0) or 0),
                "max_failures": int(harness_obj._fields.get("maxFailures", 3) or 3),
            }

        elif msg_type == "clear_chat_history":
            from actorii.harness import bootstrap_agent_harness

            actor_dict = self.vm.global_vars.get("Actor", {})
            if "Harness" not in actor_dict:
                _, self.harness, self.sandbox = bootstrap_agent_harness(self.vm)
            self.vm.eval_chunk("clearChat(Actor[#Harness])!!")
            self.vm.eval_chunk("resetBreaker(Actor[#Harness])!!")
            harness_obj = self.vm.global_vars["Actor"]["Harness"]
            scratch_raw = harness_obj._fields.get("scratchpad")
            scratch_dict = (
                {
                    (k.name if isinstance(k, Symbol) else str(k).lstrip("#")): _serialize_value(v)
                    for k, v in getattr(scratch_raw, "_entries", {}).items()
                }
                if scratch_raw is not None
                else {}
            )
            return {
                "type": "chat_history",
                "success": True,
                "cleared": True,
                "chat_history": [],
                "scratchpad": scratch_dict,
                "status": str(harness_obj._fields.get("status", "#ready")),
                "model": getattr(harness_obj, "model", "qwen3.8:latest"),
                "gas_used": int(harness_obj._fields.get("totalGasUsed", 0) or 0),
                "failures": 0,
                "max_failures": int(harness_obj._fields.get("maxFailures", 3) or 3),
            }

        elif msg_type == "get_engrams":
            from actorii.harness import bootstrap_agent_harness

            actor_dict = self.vm.global_vars.get("Actor", {})
            if "Harness" not in actor_dict:
                _, self.harness, self.sandbox = bootstrap_agent_harness(self.vm)
            summary = self.vm.eval_chunk("listEngrams(Actor[#Memory])!!")
            return {
                "type": "engrams_list",
                "success": True,
                "summary": str(summary),
            }

        elif msg_type == "get_graph_schema":
            from actorii.harness import bootstrap_agent_harness

            actor_dict = self.vm.global_vars.get("Actor", {})
            if "Harness" not in actor_dict:
                _, self.harness, self.sandbox = bootstrap_agent_harness(self.vm)
            schema_txt = self.vm.eval_chunk("graphSchema(Actor[#Memory])!!")
            return {
                "type": "graph_schema",
                "success": True,
                "schema": str(schema_txt),
            }

        else:
            return {"type": "echo", "received": raw}


_DEFAULT_LISTEN_HOST = (
    "0.0.0.0" if os.environ.get("ACTORII_IN_DOCKER") == "1" else "127.0.0.1"
)
LISTEN_HOST = os.environ.get("ACTOR_SURFACE_HOST", _DEFAULT_LISTEN_HOST)
LISTEN_PORT = int(os.environ.get("ACTOR_SURFACE_PORT", "8765"))
_SHARED_WORKSPACE: Optional[ActorWorkspace] = None


def get_shared_workspace() -> ActorWorkspace:
    """Return the persistent shared ActorWorkspace for HTTP + WebSocket clients."""
    global _SHARED_WORKSPACE
    if _SHARED_WORKSPACE is None:
        _SHARED_WORKSPACE = ActorWorkspace()
        try:
            import threading
            from pathlib import Path

            act_file = (
                Path(__file__).resolve().parent / "examples" / "tp358w_digital_twin.act"
            )
            fileout_dir = Path(__file__).resolve().parent / "examples" / "fileout"
            if act_file.is_file():
                _SHARED_WORKSPACE.vm.eval_chunk(act_file.read_text(encoding="utf-8"))
            from actorii.harness import bootstrap_agent_harness

            _, _SHARED_WORKSPACE.harness, _SHARED_WORKSPACE.sandbox = (
                bootstrap_agent_harness(
                    _SHARED_WORKSPACE.vm,
                    max_steps=25,
                    max_failures=6,
                    fileout_dir=str(fileout_dir),
                )
            )
            actor_dict = _SHARED_WORKSPACE.vm.global_vars.get("Actor", {})
            twin_obj = actor_dict.get("Thermometer")
            ui_obj = actor_dict.get("ThermoUI")
            if ui_obj is not None:
                ui_obj.attachSurface(_SHARED_WORKSPACE.surface)
                _SHARED_WORKSPACE.widgets["ThermoUI"] = ui_obj
                for ch in getattr(ui_obj, "children", []):
                    cid = str(getattr(ch, "identity", getattr(ch, "id", ""))).lstrip(
                        "#"
                    )
                    if cid:
                        _SHARED_WORKSPACE.widgets[cid] = ch

            def _warm_ble() -> None:
                try:
                    if twin_obj is not None:
                        try:
                            twin_obj.downloadHistory(60)
                        except Exception:
                            pass
                        try:
                            twin_obj.syncLive(10.0)
                        except Exception:
                            pass
                    if ui_obj is not None and _SHARED_WORKSPACE is not None:
                        ui_obj.renderOnSurface(_SHARED_WORKSPACE.surface)
                except Exception as e:
                    print(f"[ActorSurface] Background BLE sync note: {e}", flush=True)

            threading.Thread(target=_warm_ble, daemon=True).start()
        except Exception as exc:
            print(f"[ActorSurface] Initial twin warm-up warning: {exc}", flush=True)
    return _SHARED_WORKSPACE


def _build_mobile_thermometer_page(ws_app: ActorWorkspace) -> str:
    """Render standalone mobile-friendly HTML5 page for Actor[#ThermoUI] with live WebSocket actions."""
    actor_dict = ws_app.vm.global_vars.get("Actor", {})
    ui_obj = actor_dict.get("ThermoUI")
    card_html = ui_obj.draw() if ui_obj is not None and hasattr(ui_obj, "draw") else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>ActorII 4.1 — ThermoPro TP358W Surface</title>
<style>
  body {{
    margin: 0;
    padding: 12px;
    background: #0b0f19;
    color: #f8fafc;
    font-family: system-ui, -apple-system, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  #thermo-mount {{
    width: 100%;
    max-width: 760px;
    overflow-x: auto;
  }}
  #thermo-mount > .actor-thermometer-ui {{
    position: relative !important;
    left: 0 !important;
    top: 0 !important;
    margin: 0 auto;
  }}
  .topbar {{
    width: 100%;
    max-width: 760px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    font-size: 12px;
    color: #94a3b8;
  }}
  .topbar a {{
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
  }}
</style>
</head>
<body>
  <div class="topbar">
    <span>ActorII 4.1 Surface (<span id="ws-state">Connecting...</span>)</span>
    <a href="/">Open Full Actor Studio &rarr;</a>
  </div>
  <div id="thermo-mount">{card_html}</div>
  <script>
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${{proto}}//${{window.location.host}}/${{window.location.search || ''}}`;
    let ws = null;

    function bindThermoControls() {{
      const mount = document.getElementById('thermo-mount');
      if (!mount) return;
      mount.querySelectorAll('[data-widget="button"]').forEach(btn => {{
        btn.onclick = (e) => {{
          e.stopPropagation();
          const href = btn.getAttribute('data-href');
          if (href) {{
            window.location.href = href;
            return;
          }}
          const id = btn.getAttribute('data-id');
          const action = btn.getAttribute('data-action') || 'selected';
          btn.style.opacity = '0.6';
          if (ws && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{ type: 'ui_event', widget_id: id, action: action }}));
          }}
        }};
      }});
      mount.querySelectorAll('[data-widget="checkbox"]').forEach(lbl => {{
        const inp = lbl.querySelector('input[type="checkbox"]');
        if (inp) {{
          inp.onchange = (e) => {{
            e.stopPropagation();
            const id = lbl.getAttribute('data-id');
            if (ws && ws.readyState === WebSocket.OPEN) {{
              ws.send(JSON.stringify({{ type: 'ui_event', widget_id: id, action: 'toggle' }}));
            }}
          }};
        }}
      }});
    }}

    function connectWs() {{
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {{
        document.getElementById('ws-state').textContent = 'Live Connected';
        document.getElementById('ws-state').style.color = '#10b981';
      }};
      ws.onmessage = (ev) => {{
        try {{
          const data = JSON.parse(ev.data);
          if (data.thermometer_html) {{
            document.getElementById('thermo-mount').innerHTML = data.thermometer_html;
            bindThermoControls();
          }} else if (data.type === 'thermometer_rendered' && data.html) {{
            document.getElementById('thermo-mount').innerHTML = data.html;
            bindThermoControls();
          }}
        }} catch (err) {{}}
      }};
      ws.onclose = () => {{
        document.getElementById('ws-state').textContent = 'Reconnecting...';
        document.getElementById('ws-state').style.color = '#f59e0b';
        setTimeout(connectWs, 2000);
      }};
    }}

    bindThermoControls();
    connectWs();
  </script>
</body>
</html>"""


def _http_response(
    status_code: int,
    reason: str,
    content_type: str,
    body: bytes,
    extra_headers: Optional[List[tuple[str, str]]] = None,
) -> Any:
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    hdrs = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-cache, no-store, must-revalidate"),
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "SAMEORIGIN"),
        ("Connection", "close"),
    ]
    if extra_headers:
        hdrs.extend(extra_headers)
    return Response(status_code, reason, Headers(hdrs), body)


def _is_request_origin_allowed(request: Any) -> bool:
    """Validate HTTP/WebSocket `Origin` and `Sec-Fetch-Site` headers against cross-site CSRF."""
    import urllib.parse

    allowed_extra = {
        o.strip().rstrip("/")
        for o in os.environ.get("ACTOR_ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    }
    origin = (request.headers.get("Origin") or "").strip()
    sec_site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()

    if sec_site == "cross-site" and origin.rstrip("/") not in allowed_extra:
        return False

    if not origin:
        return True

    norm_origin = origin.rstrip("/")
    if norm_origin in allowed_extra:
        return True
    if origin.startswith("file://"):
        return True
    if origin.lower() == "null":
        return os.environ.get("ACTOR_ALLOW_NULL_ORIGIN") == "1"

    try:
        parsed = urllib.parse.urlparse(origin)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host_only = (parsed.hostname or "").lower()
    if host_only in ("127.0.0.1", "localhost", "::1"):
        return True

    req_host = (request.headers.get("Host") or "").strip().lower()
    if req_host:
        if parsed.netloc.lower() == req_host:
            return True
        req_hostname = req_host.split(":", 1)[0]
        if host_only == req_hostname and sec_site != "cross-site":
            return True

    return False


def _is_request_authorized(request: Any, *, allow_query_token: bool = False) -> bool:
    """Verify `ACTORII_STUDIO_TOKEN` via `Authorization: Bearer <token>` (or `?token=` on WS/HTML)."""
    import hmac
    import urllib.parse

    expected = os.environ.get("ACTORII_STUDIO_TOKEN", "").strip()
    if not expected:
        return True

    auth_hdr = (request.headers.get("Authorization") or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        provided = auth_hdr[7:].strip()
        if hmac.compare_digest(provided, expected):
            return True

    if allow_query_token and "?" in (request.path or ""):
        qs = urllib.parse.parse_qs((request.path or "").split("?", 1)[1])
        q_tok = (qs.get("token") or [""])[0].strip()
        if q_tok and hmac.compare_digest(q_tok, expected):
            return True

    return False


def _has_csrf_proof_header(request: Any) -> bool:
    """Require a non-CORS-safelisted header (`X-Actor-Request: 1` or `Authorization`) on mutating HTTP calls."""
    x_hdr = (request.headers.get("X-Actor-Request") or "").strip()
    if x_hdr == "1" or x_hdr.lower() == "true":
        return True
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    return auth_hdr.lower().startswith("bearer ")


_MUTATING_HTTP_PATHS = frozenset(
    {
        "/api/eval",
        "/api/goal",
        "/api/chat",
        "/api/chat/clear",
        "/api/harness/evolve",
    }
)


async def process_http_request(connection: Any, request: Any) -> Any:
    """Serve HTML5 Actor Surface and hardened HTTP API on the same port as WebSocket."""
    import urllib.parse
    from pathlib import Path

    remote = getattr(connection, "remote_address", "?")
    method = (getattr(request, "method", "GET") or "GET").upper()
    upgrade = request.headers.get("Upgrade", "")

    if upgrade.lower() == "websocket":
        if not _is_request_origin_allowed(request):
            print(f"[WS-REJECT-ORIGIN] {request.path} from {remote}", flush=True)
            return _http_response(
                403,
                "Forbidden",
                "application/json; charset=utf-8",
                json.dumps({"status": "error", "error": "Forbidden cross-origin WebSocket upgrade"}).encode("utf-8"),
            )
        if not _is_request_authorized(request, allow_query_token=True):
            print(f"[WS-REJECT-AUTH] {request.path} from {remote}", flush=True)
            return _http_response(
                401,
                "Unauthorized",
                "application/json; charset=utf-8",
                json.dumps({"status": "error", "error": "Missing or invalid ACTORII_STUDIO_TOKEN"}).encode("utf-8"),
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        print(f"[WS-UPGRADE] {request.path} from {remote}", flush=True)
        return None

    req_path = (request.path or "/").split("?", 1)[0]
    print(f"[HTTP] {method} {req_path} from {remote}", flush=True)
    root_dir = Path(__file__).resolve().parent

    if method == "OPTIONS":
        return _http_response(403, "Forbidden", "text/plain; charset=utf-8", b"CORS preflight rejected")

    if not _is_request_origin_allowed(request):
        return _http_response(
            403,
            "Forbidden",
            "application/json; charset=utf-8",
            json.dumps({"status": "error", "error": "Forbidden cross-origin request"}).encode("utf-8"),
        )

    if req_path == "/favicon.ico":
        return _http_response(204, "No Content", "image/x-icon", b"")

    if req_path in ("/", "/index.html", "/browser_client.html"):
        if not _is_request_authorized(request, allow_query_token=True):
            return _http_response(
                401,
                "Unauthorized",
                "text/plain; charset=utf-8",
                b"Unauthorized: append ?token=<ACTORII_STUDIO_TOKEN> or supply Bearer token.",
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        client_file = root_dir / "browser_client.html"
        if client_file.is_file():
            body = client_file.read_bytes()
            return _http_response(200, "OK", "text/html; charset=utf-8", body)

    if req_path in ("/benchmarks", "/benchmarks.html", "/scoreboard.html"):
        if not _is_request_authorized(request, allow_query_token=True):
            return _http_response(
                401,
                "Unauthorized",
                "text/plain; charset=utf-8",
                b"Unauthorized: append ?token=<ACTORII_STUDIO_TOKEN> or supply Bearer token.",
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        sb_html = root_dir / "benchmarks" / "scoreboard.html"
        if sb_html.is_file():
            return _http_response(200, "OK", "text/html; charset=utf-8", sb_html.read_bytes())

    if req_path == "/api/benchmarks":
        if not _is_request_authorized(request, allow_query_token=True):
            return _http_response(
                401,
                "Unauthorized",
                "application/json; charset=utf-8",
                json.dumps({"status": "error", "error": "Unauthorized"}).encode("utf-8"),
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        sb_json = root_dir / "benchmarks" / "scoreboard.json"
        if sb_json.is_file():
            return _http_response(200, "OK", "application/json; charset=utf-8", sb_json.read_bytes())
        return _http_response(
            404,
            "Not Found",
            "application/json; charset=utf-8",
            json.dumps({"status": "error", "error": "scoreboard.json not found"}).encode("utf-8"),
        )

    if req_path in ("/thermometer", "/thermometer_surface.html"):
        if not _is_request_authorized(request, allow_query_token=True):
            return _http_response(
                401,
                "Unauthorized",
                "text/plain; charset=utf-8",
                b"Unauthorized",
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            html = _build_mobile_thermometer_page(ws_app)
        return _http_response(
            200, "OK", "text/html; charset=utf-8", html.encode("utf-8")
        )

    if req_path == "/health":
        ws_app = get_shared_workspace()
        actor_dict = ws_app.vm.global_vars.get("Actor", {})
        twin = actor_dict.get("Thermometer")
        harness = actor_dict.get("Harness")
        status = {
            "status": "ok",
            "host": LISTEN_HOST,
            "ports": [LISTEN_PORT, 8090],
            "auth_required": bool(os.environ.get("ACTORII_STUDIO_TOKEN", "").strip()),
            "harness_model": getattr(harness, "model", None) if harness else None,
            "temperature_c": twin.temperatureC() if twin else None,
            "temperature_f": twin.temperatureF() if twin else None,
            "humidity": twin.humidity() if twin else None,
            "history_count": twin.historyCount() if twin else 0,
        }
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(status).encode("utf-8"),
        )

    # All `/api/*` routes enforce Bearer auth (when configured) and mutating routes enforce POST + anti-CSRF header
    if req_path.startswith("/api/"):
        if not _is_request_authorized(request, allow_query_token=False):
            return _http_response(
                401,
                "Unauthorized",
                "application/json; charset=utf-8",
                json.dumps({"status": "error", "error": "Missing or invalid Authorization Bearer token"}).encode("utf-8"),
                extra_headers=[("WWW-Authenticate", 'Bearer realm="ActorII Studio"')],
            )
        if req_path in _MUTATING_HTTP_PATHS:
            if method != "POST":
                return _http_response(
                    405,
                    "Method Not Allowed",
                    "application/json; charset=utf-8",
                    json.dumps(
                        {
                            "status": "error",
                            "error": f"Endpoint {req_path} requires HTTP POST with X-Actor-Request: 1 header",
                        }
                    ).encode("utf-8"),
                    extra_headers=[("Allow", "POST")],
                )
            if not _has_csrf_proof_header(request):
                return _http_response(
                    403,
                    "Forbidden",
                    "application/json; charset=utf-8",
                    json.dumps(
                        {
                            "status": "error",
                            "error": "Missing required X-Actor-Request: 1 anti-CSRF header",
                        }
                    ).encode("utf-8"),
                )

    if req_path == "/api/pool":
        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            sandbox = getattr(ws_app, "sandbox", None)
            summary_str = (
                sandbox.poolSummary()
                if sandbox and hasattr(sandbox, "poolSummary")
                else ""
            )
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps({"status": "ok", "pool_summary": summary_str}).encode("utf-8"),
        )

    if req_path == "/api/eval":
        qs = urllib.parse.parse_qs(
            (request.path or "").split("?", 1)[1] if "?" in (request.path or "") else ""
        )
        code = (qs.get("code") or [""])[0]
        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            try:
                res = ws_app._eval_source(code)
                payload = {"status": "ok", "success": True, "result": _serialize_value(res)}
            except Exception as exc:
                payload = {"status": "error", "success": False, "error": f"{type(exc).__name__}: {exc}"}
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    if req_path == "/api/goal/stop":
        ws_app = get_shared_workspace()
        harness_obj = ws_app.vm.global_vars.get("Actor", {}).get("Harness")
        if harness_obj is not None and hasattr(harness_obj, "_fields"):
            harness_obj._fields["status"] = Symbol("halted")
            step_cnt = int(harness_obj._fields.get("stepCount", 0) or 0)
            payload = {"status": "ok", "success": True, "steps_count": step_cnt}
        else:
            payload = {"status": "ok", "success": True, "steps_count": 0}
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    if req_path in ("/api/goal", "/api/chat"):
        qs = urllib.parse.parse_qs(
            (request.path or "").split("?", 1)[1] if "?" in (request.path or "") else ""
        )
        goal_q = (qs.get("q") or qs.get("goal") or qs.get("message") or [""])[0]
        raw_qms = (qs.get("max_steps") or ["0"])[0]
        max_steps = int(raw_qms) if str(raw_qms).strip().lstrip("-").isdigit() else 0
        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            try:
                from actorii.harness import bootstrap_agent_harness

                if "Harness" not in ws_app.vm.global_vars.get("Actor", {}):
                    _, ws_app.harness, ws_app.sandbox = bootstrap_agent_harness(
                        ws_app.vm, max_steps=25, max_failures=6
                    )
                harness_obj = ws_app.vm.global_vars["Actor"]["Harness"]
                if max_steps <= 0:
                    max_steps = int(harness_obj._fields.get("maxSteps", 25) or 25)
                sb = getattr(ws_app, "sandbox", None)
                escaped = goal_q.replace("\\", "\\\\").replace('"', '\\"')
                res = ws_app._eval_source(f'runGoal(Actor[#Harness], "{escaped}", {max_steps})!!')
                status_str = str(harness_obj._fields.get("status", "#complete"))
                step_count = int(harness_obj._fields.get("stepCount", 0) or 0)
                last_res = _serialize_value(harness_obj._fields.get("lastResult"))
                raw_logs = _serialize_value(list(harness_obj._fields.get("stepLog", [])))
                chat_hist = _serialize_value(
                    sb.getChatHistory(harness_obj)
                    if sb is not None and hasattr(sb, "getChatHistory")
                    else []
                )
                payload = {
                    "status": "ok",
                    "success": "complete" in status_str.lower(),
                    "goal": goal_q,
                    "harness_status": status_str,
                    "steps_count": step_count,
                    "gas_used": int(harness_obj._fields.get("totalGasUsed", 0) or 0),
                    "result": _serialize_value(res),
                    "last_result": last_res,
                    "final_summary": str(last_res or _serialize_value(res)),
                    "step_log": raw_logs,
                    "chat_history": chat_hist,
                }
            except Exception as exc:
                payload = {"status": "error", "success": False, "error": f"{type(exc).__name__}: {exc}"}
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    if req_path == "/api/chat/history":
        from actorii.harness import bootstrap_agent_harness

        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            if "Harness" not in ws_app.vm.global_vars.get("Actor", {}):
                _, ws_app.harness, ws_app.sandbox = bootstrap_agent_harness(ws_app.vm)
            harness_obj = ws_app.vm.global_vars["Actor"]["Harness"]
            sb = getattr(ws_app, "sandbox", None)
            chat_hist = _serialize_value(
                sb.getChatHistory(harness_obj)
                if sb is not None and hasattr(sb, "getChatHistory")
                else []
            )
            payload = {
                "status": "ok",
                "harness_status": str(harness_obj._fields.get("status", "#idle")),
                "chat_history": chat_hist,
                "turns_count": len(chat_hist),
            }
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    if req_path == "/api/chat/clear":
        from actorii.harness import bootstrap_agent_harness

        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            if "Harness" not in ws_app.vm.global_vars.get("Actor", {}):
                _, ws_app.harness, ws_app.sandbox = bootstrap_agent_harness(ws_app.vm)
            ws_app._eval_source("clearChat(Actor[#Harness])!!")
            ws_app._eval_source("resetBreaker(Actor[#Harness])!!")
        payload = {"status": "ok", "cleared": True, "chat_history": []}
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    if req_path in ("/api/engrams", "/api/schema", "/api/harness/source", "/api/harness/evolve"):
        from actorii.guardrails.fileout import ActorFileOutManager
        from actorii.harness import bootstrap_agent_harness

        ws_app = get_shared_workspace()
        async with ws_app._vm_lock:
            if "Harness" not in ws_app.vm.global_vars.get("Actor", {}):
                _, ws_app.harness, ws_app.sandbox = bootstrap_agent_harness(ws_app.vm)
            if req_path == "/api/engrams":
                out = ws_app._eval_source("listEngrams(Actor[#Memory])!!")
                payload = {"status": "ok", "engrams": str(out)}
            elif req_path == "/api/schema":
                out = ws_app._eval_source("graphSchema(Actor[#Memory])!!")
                payload = {"status": "ok", "schema": str(out)}
            elif req_path == "/api/harness/source":
                mgr = ActorFileOutManager("/tmp/actorii_harness_inspect")
                src = mgr.format_class_source(ws_app.vm, "AgentHarness")
                payload = {"status": "ok", "class_name": "AgentHarness", "source": src}
            else:
                qs = urllib.parse.parse_qs(
                    (request.path or "").split("?", 1)[1] if "?" in (request.path or "") else ""
                )
                patch_code = (qs.get("patch") or qs.get("code") or [""])[0]
                val_code = (qs.get("validate") or qs.get("validation") or [""])[0]
                p_esc = patch_code.replace("\\", "\\\\").replace('"', '\\"')
                v_esc = val_code.replace("\\", "\\\\").replace('"', '\\"')
                out = ws_app._eval_source(f'evolve(Actor[#Harness], "{p_esc}", "{v_esc}")!!')
                payload = {
                    "status": "ok" if str(out).startswith("EVOLVE_COMMITTED") else "rejected",
                    "result": str(out),
                }
        return _http_response(
            200,
            "OK",
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    return _http_response(404, "Not Found", "text/plain; charset=utf-8", b"Not Found")


async def handler(
    websocket: Any,
    path: Optional[str] = None,
    workspace: Optional[ActorWorkspace] = None,
):
    """Main WebSocket connection loop with multi-client driver registration."""
    ws_app = workspace if workspace is not None else get_shared_workspace()
    ws_app.driver.register_client(websocket)

    try:
        async for message in websocket:
            try:
                resp = await ws_app.handle_message(message, websocket=websocket)
                await websocket.send(json.dumps(resp))
            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
    except Exception:
        pass
    finally:
        ws_app.driver.unregister_client(websocket)
    print("Client disconnected", flush=True)


async def main():
    import websockets

    get_shared_workspace()
    ports = sorted({LISTEN_PORT, 8090})
    bind_host = None if LISTEN_HOST in ("", "0.0.0.0", "::") else LISTEN_HOST
    servers = []
    for port in ports:
        srv = await websockets.serve(
            handler,
            bind_host,
            port,
            process_request=process_http_request,
        )
        servers.append(srv)
        host_desc = "dual-stack [::] / 0.0.0.0" if bind_host is None else bind_host
        print(
            f"ActorII HTML5 & WebSocket Surface listening on {host_desc}:{port}",
            flush=True,
        )
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down", flush=True)
        sys.exit(0)
