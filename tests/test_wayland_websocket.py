"""Tests for Wayland display surfaces, WebSocket surfaces, and interactive workspace protocol."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.classes.geometry import Point3D
from actorii.classes.gui import Button, Checkbox, TextField, Window
from actorii.core import Symbol
from actorii.display.wayland import (
    WaylandDisplay,
    WaylandDriver,
    WaylandSurface,
    create_driver,
)
from actorii.display.websocket import WebSocketSurface
from websocket_server import ActorWorkspace, handler


def _resolve_hist_root() -> str:
    cur = os.path.dirname(os.path.abspath(__file__))
    while cur != "/" and not os.path.isdir(
        os.path.join(cur, "historical_actor_41_source")
    ):
        cur = os.path.dirname(cur)
    return os.path.join(cur, "historical_actor_41_source")


HIST_ROOT = _resolve_hist_root()
SIEVE_ACT_PATH = os.path.join(HIST_ROOT, "actact", "SIEVE.ACT")


# =============================================================================
# 1. Wayland Surface & Driver Tests (Headless)
# =============================================================================


def test_wayland_headless_surface():
    display = WaylandDisplay(headless=True)
    surface = display.create_surface(200, 100)
    assert surface.width == 200
    assert surface.height == 100
    surface.set_pixel(10, 10, "#FF0000")
    assert surface.get_pixel(10, 10) == "#ff0000"


def test_wayland_driver_headless():
    driver = create_driver("wayland", width=300, height=200, headless=True)
    assert driver.surface.width == 300
    driver.surface.set_pixel(5, 5, "#00FF00")
    assert driver.surface.get_pixel(5, 5) == "#00ff00"
    driver.present()


def test_wayland_surface_clear_and_fill():
    display = WaylandDisplay(headless=True)
    surface = display.create_surface(50, 50)
    surface.clear("#123456")
    assert surface.get_pixel(0, 0) == "#123456"
    assert surface.get_pixel(25, 25) == "#123456"
    assert surface.get_pixel(49, 49) == "#123456"


def test_wayland_surface_damage_and_flush():
    display = WaylandDisplay(headless=True)
    surface = display.create_surface(100, 100)
    surface.damage(10, 10, 50, 50)
    surface.flush()
    assert surface.width == 100


def test_wayland_surface_copy_from():
    display = WaylandDisplay(headless=True)
    src = display.create_surface(50, 50)
    dst = display.create_surface(50, 50)
    src.clear("#FFFFFF")
    dst.clear("#000000")
    src.set_pixel(5, 5, "#FF0000")
    dst.copy_from(src, x=0, y=0, w=20, h=20, dst_x=10, dst_y=10)
    assert dst.get_pixel(15, 15) == "#ff0000"
    assert dst.get_pixel(0, 0) == "#000000"


def test_wayland_surface_bounds_safety():
    display = WaylandDisplay(headless=True)
    surface = display.create_surface(20, 20)
    # Writing out of bounds should not crash
    surface.set_pixel(-5, 10, "#FF0000")
    surface.set_pixel(100, 100, "#FF0000")
    # Reading out of bounds should safely return black
    assert surface.get_pixel(-1, 0) == "#000000"
    assert surface.get_pixel(20, 20) == "#000000"


def test_wayland_driver_pointer_event_mapping():
    driver = create_driver("wayland", width=100, height=100, headless=True)
    handled_press = driver.handle_event(
        {"type": "pointer", "action": "press", "x": 10, "y": 20}
    )
    assert handled_press is True
    assert len(driver.surface._posted_events) == 1
    ev = driver.surface._posted_events[-1]
    assert ev["type"] == "mousebutton"
    assert ev["pressed"] is True
    assert ev["pos"] == (10, 20)

    handled_move = driver.handle_event(
        {"type": "pointer", "action": "move", "x": 15, "y": 25}
    )
    assert handled_move is True
    assert driver.surface._posted_events[-1]["type"] == "mousemove"

    handled_release = driver.handle_event(
        {"type": "pointer", "action": "release", "x": 15, "y": 25}
    )
    assert handled_release is True
    assert driver.surface._posted_events[-1]["pressed"] is False


def test_wayland_driver_keyboard_event_mapping():
    driver = create_driver("wayland", width=100, height=100, headless=True)
    handled = driver.handle_event({"type": "key", "key": "Return", "down": True})
    assert handled is True
    assert driver.surface._posted_events[-1] == {
        "type": "keyboard",
        "key": "Return",
        "down": True,
    }


def test_websocket_driver_resize():
    from websocket_server import WebSocketDriver

    ws_surface = WebSocketSurface(width=100, height=100)
    driver = WebSocketDriver(ws_surface)
    driver.resize(250, 180)
    assert ws_surface.width == 250
    assert ws_surface.height == 180


# =============================================================================
# 2. WebSocket Surface Buffer Unit Tests
# =============================================================================


def test_websocket_surface_buffer():
    ws_surface = WebSocketSurface(width=100, height=50)
    assert ws_surface.width == 100
    assert ws_surface.height == 50
    ws_surface.set_pixel(2, 2, "#0000FF")
    assert ws_surface.get_pixel(2, 2) == "#0000FF"
    assert ws_surface._dirty is True
    ws_surface.flush()


# =============================================================================
# 3. Interactive Workspace Protocol Unit Tests (In-Memory)
# =============================================================================


def test_workspace_initialization():
    """Verify ActorWorkspace initializes ActorVM with all core standard library classes."""
    ws = ActorWorkspace()
    assert ws.vm is not None
    assert ws.surface is not None
    assert ws.driver is not None
    assert len(ws.vm.hierarchy._classes) >= 20
    assert "Object" in ws.vm.hierarchy._classes
    assert "Array" in ws.vm.hierarchy._classes
    assert "Window" in ws.vm.hierarchy._classes
    assert "Button" in ws.vm.hierarchy._classes


def test_eval_arithmetic_expression():
    """Verify eval handler computes arithmetic expression ending in !!."""

    async def _run():
        ws = ActorWorkspace()
        res = await ws.handle_message({"type": "eval", "source": "3 + 4!!"})
        assert res["type"] == "eval_result"
        assert res["success"] is True
        assert res["result"] == 7

    asyncio.run(_run())


def test_eval_with_caret_return():
    """Verify eval handler computes expressions with explicit return operator (^)."""

    async def _run():
        ws = ActorWorkspace()
        res = await ws.handle_message({"type": "eval", "source": "^(10 * 5) - 8!!"})
        assert res["type"] == "eval_result"
        assert res["success"] is True
        assert res["result"] == 42

    asyncio.run(_run())


def test_eval_multiple_chunks_and_variables():
    """Verify eval handler maintains global state across multiple evaluation chunks."""

    async def _run():
        ws = ActorWorkspace()
        r1 = await ws.handle_message({"type": "eval", "source": "a := 100!!"})
        assert r1["success"] is True

        r2 = await ws.handle_message({"type": "eval", "source": "b := 250!!"})
        assert r2["success"] is True

        r3 = await ws.handle_message({"type": "eval", "source": "^a + b!!"})
        assert r3["success"] is True
        assert r3["result"] == 350

    asyncio.run(_run())


def test_eval_sieve_benchmark_historical_act():
    """Verify loading authentic SIEVE.ACT and evaluating sieve(Sam, 100) returns 25."""

    async def _run():
        ws = ActorWorkspace()
        assert os.path.isfile(
            SIEVE_ACT_PATH
        ), f"SIEVE.ACT not found at {SIEVE_ACT_PATH}"
        with open(SIEVE_ACT_PATH, "r", encoding="latin-1") as f:
            sieve_code = f.read()

        r_load = await ws.handle_message({"type": "eval", "source": sieve_code})
        assert r_load["success"] is True

        r_eval = await ws.handle_message(
            {"type": "eval", "source": "sieve(Sam, 100)!!"}
        )
        assert r_eval["type"] == "eval_result"
        assert r_eval["success"] is True
        assert r_eval["result"] == 25

    asyncio.run(_run())


def test_eval_dynamic_class_and_method_definition():
    """Verify dynamic class definition (inherit) and method definition via eval."""

    async def _run():
        ws = ActorWorkspace()
        r_inherit = await ws.handle_message(
            {
                "type": "eval",
                "source": "inherit(Object, #PointWorkspace, #(x y), 0, nil)!!",
            }
        )
        assert r_inherit["success"] is True

        r_classes = await ws.handle_message({"type": "get_classes"})
        point_cls = next(
            (c for c in r_classes["classes"] if c["name"] == "PointWorkspace"), None
        )
        assert point_cls is not None
        assert point_cls["super"] == "Object"
        assert "x" in point_cls["ivars"]
        assert "y" in point_cls["ivars"]

    asyncio.run(_run())


def test_eval_syntax_error_resilience():
    """Verify eval handles syntax errors gracefully without crashing the workspace."""

    async def _run():
        ws = ActorWorkspace()
        r_bad = await ws.handle_message({"type": "eval", "source": "Def bad( { }!!"})
        assert r_bad["type"] == "eval_result"
        assert r_bad["success"] is False
        assert "error" in r_bad

        # Verify workspace remains alive and healthy after error
        r_good = await ws.handle_message({"type": "eval", "source": "1 + 1!!"})
        assert r_good["success"] is True
        assert r_good["result"] == 2

    asyncio.run(_run())


def test_get_classes_hierarchy():
    """Verify get_classes returns all registered classes, supers, methods, and ivars."""

    async def _run():
        ws = ActorWorkspace()
        res = await ws.handle_message({"type": "get_classes"})
        assert res["type"] == "classes_list"
        assert res["count"] >= 20

        classes_by_name = {c["name"]: c for c in res["classes"]}
        assert "Object" in classes_by_name
        assert "Collection" in classes_by_name
        assert "Array" in classes_by_name
        assert "Window" in classes_by_name
        assert "Button" in classes_by_name

        # Verify inheritance chain
        assert classes_by_name["Object"]["super"] is None
        assert classes_by_name["Collection"]["super"] == "Object"
        assert classes_by_name["IndexedCollection"]["super"] == "Collection"
        assert classes_by_name["Array"]["super"] == "IndexedCollection"

        # Verify standard methods
        assert "new" in classes_by_name["Object"]["methods"]
        assert "print" in classes_by_name["Object"]["methods"]
        assert "size" in classes_by_name["Collection"]["methods"]

    asyncio.run(_run())


def test_inspect_object_existing_widget():
    """Verify inspect_object returns class and field attributes of registered objects."""

    async def _run():
        ws = ActorWorkspace()
        btn = Button(label="Submit Form", id="btnSubmit", identity=Symbol("btnSubmit"))
        ws.vm.global_vars["Actor"]["btnSubmit"] = btn

        res = await ws.handle_message({"type": "inspect_object", "key": "btnSubmit"})
        assert res["type"] == "object_details"
        assert res["found"] is True
        assert res["class"] == "Button"
        assert res["fields"]["label"] == "Submit Form"
        assert res["fields"]["id"] == "btnSubmit"

    asyncio.run(_run())


def test_inspect_object_not_found():
    """Verify inspect_object returns found=False for non-existent keys."""

    async def _run():
        ws = ActorWorkspace()
        res = await ws.handle_message({"type": "inspect_object", "key": "GhostObject"})
        assert res["type"] == "object_details"
        assert res["found"] is False
        assert "error" in res

    asyncio.run(_run())


def test_ui_event_button_click_dispatch():
    """Verify ui_event dispatches to button callback and records execution."""

    async def _run():
        ws = ActorWorkspace()
        clicks = []
        btn = Button(
            label="OK",
            id="btnTest",
            identity=Symbol("btnTest"),
            callback=lambda b: clicks.append(True),
        )
        ws.vm.global_vars["Actor"]["btnTest"] = btn

        res = await ws.handle_message(
            {
                "type": "ui_event",
                "widget_id": "btnTest",
                "action": "#selected",
            }
        )
        assert res["type"] == "ui_event_result"
        assert res["handled"] is True
        assert len(clicks) == 1

    asyncio.run(_run())


def test_ui_event_window_action_handler():
    """Verify ui_event triggers window Action method registered in action_handlers."""

    async def _run():
        ws = ActorWorkspace()
        win = Window(title="AppWindow")
        btn = Button(label="ClickMe", id=201, identity=Symbol("btnTarget"))
        win.add(btn)
        win.action_handlers[("btnTarget", "selected")] = "onTargetClicked"

        handled_events = []
        win.set_method("onTargetClicked", lambda self, msg: handled_events.append(msg))
        ws.vm.global_vars["Actor"]["appWindow"] = win

        res = await ws.handle_message(
            {
                "type": "ui_event",
                "widget_id": "btnTarget",
                "action": "#selected",
            }
        )
        assert res["type"] == "ui_event_result"
        assert res["handled"] is True
        assert len(handled_events) == 1
        assert handled_events[0]["action"] == Symbol("selected")

    asyncio.run(_run())


def test_render_widget_html_generation():
    """Verify render_widget generates HTML and registers widget in Actor dictionary."""

    async def _run():
        ws = ActorWorkspace()
        res = await ws.handle_message(
            {
                "type": "render_widget",
                "widget_type": "button",
                "id": "btnDynamic",
                "label": "Click Here",
                "x": 100,
                "y": 150,
            }
        )
        assert res["type"] == "rendered_widget"
        assert 'data-widget="button"' in res["html"]
        assert 'data-id="btnDynamic"' in res["html"]
        assert "Click Here" in res["html"]
        assert "btnDynamic" in ws.vm.global_vars["Actor"]

    asyncio.run(_run())


def test_canvas_drawing_commands():
    """Verify canvas pixel manipulation commands: clear, set_pixel, get_pixel."""

    async def _run():
        ws = ActorWorkspace()
        r1 = await ws.handle_message({"type": "clear", "color": "#000000"})
        assert r1["type"] == "cleared"

        r2 = await ws.handle_message(
            {"type": "set_pixel", "x": 10, "y": 20, "color": "#FF0000"}
        )
        assert r2["type"] == "pixel_color"
        assert r2["color"] == "#FF0000"

        r3 = await ws.handle_message({"type": "get_pixel", "x": 10, "y": 20})
        assert r3["color"] == "#FF0000"

    asyncio.run(_run())


# =============================================================================
# 4. Live Loopback WebSocket Network Integration Test
# =============================================================================


def test_websocket_e2e_network_session():
    """Full end-to-end integration test over live loopback WebSocket connection."""
    import websockets

    async def _e2e():
        ws_app = ActorWorkspace()
        with open(SIEVE_ACT_PATH, "r", encoding="latin-1") as f:
            sieve_code = f.read()

        # Pre-load Sieve on workspace VM
        await ws_app.handle_message({"type": "eval", "source": sieve_code})

        async def _client_session():
            async with websockets.serve(
                lambda ws: handler(ws, workspace=ws_app), "127.0.0.1", 0
            ) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
                    # 1. Test get_classes
                    await client.send(json.dumps({"type": "get_classes"}))
                    resp1 = json.loads(await client.recv())
                    assert resp1["type"] == "classes_list"
                    assert resp1["count"] >= 20

                    # 2. Test eval: sieve(Sam, 100)!! -> 25
                    await client.send(
                        json.dumps({"type": "eval", "source": "sieve(Sam, 100)!!"})
                    )
                    resp2 = json.loads(await client.recv())
                    assert resp2["type"] == "eval_result"
                    assert resp2["success"] is True
                    assert resp2["result"] == 25

                    # 3. Test inspect_object: Sam
                    await client.send(
                        json.dumps({"type": "inspect_object", "key": "Sam"})
                    )
                    resp3 = json.loads(await client.recv())
                    assert resp3["type"] == "object_details"
                    assert resp3["found"] is True
                    assert resp3["class"] == "Sieve"

                    # 4. Test render_widget
                    await client.send(
                        json.dumps(
                            {
                                "type": "render_widget",
                                "widget_type": "button",
                                "id": "btnLive",
                                "label": "Live Button",
                            }
                        )
                    )
                    resp4 = json.loads(await client.recv())
                    assert resp4["type"] == "rendered_widget"
                    assert 'data-id="btnLive"' in resp4["html"]

                    # 5. Test ui_event on newly created widget
                    await client.send(
                        json.dumps(
                            {
                                "type": "ui_event",
                                "widget_id": "btnLive",
                                "action": "#selected",
                            }
                        )
                    )
                    resp5 = json.loads(await client.recv())
                    assert resp5["type"] == "ui_event_result"
                    assert resp5["handled"] is True

                    # 6. Test canvas command
                    await client.send(json.dumps({"type": "get_surface_info"}))
                    resp6 = json.loads(await client.recv())
                    assert resp6["type"] == "surface_info"
                    assert resp6["data"]["width"] == 800

        await _client_session()

    asyncio.run(_e2e())
