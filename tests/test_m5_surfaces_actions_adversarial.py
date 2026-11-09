"""Adversarial stress tests for Milestone M5: UI Action Event Dispatch and Display Surfaces.

Authored by Challenger M5-2:
- UI Action event dispatching (ui_event -> ActorVM.dispatch_action):
  - Button clicks (callbacks, parent window handlers, compiled Actor action methods).
  - Checkbox toggle states (#checked / #unChecked, toggling via #selected, direct check/unCheck, setValue).
  - Custom widget selectors and action routing.
  - Unhandled widget IDs (missing, nonexistent, None, numeric, Symbol).
- Wayland headless surface stress tests:
  - Boundary coordinates (corners, exact width/height limits).
  - Negative coordinates (-1, -100).
  - Out-of-bounds pixel operations (safe bounds checking, get_pixel returns #000000).
  - Zero-size damage regions (0x0, 0xN, Nx0).
  - Multi-region damage flushes (rapid multi-region damage queuing and commits).
  - Region copy_from edge cases (partial, overlapping, out of bounds).
  - Color formats (#RRGGBB, lowercase, mixed case, non-hex strings) and buffer layout (xrgb8888, stride).
- render_widget DOM generation and binding to the Actor global dictionary:
  - button, checkbox, textfield, custom widget tags and data attributes.
  - Percentage coordinate clamping (negative, 0, extreme coordinates).
  - Global Actor dictionary binding and subsequent inspect_object / ui_event roundtrip.
- demo_wayland.py headless execution.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.classes.geometry import Rect
from actorii.classes.gui import Button, Checkbox, Dialog, TextField, Window
from actorii.core import ActorClass, ActorObject, Symbol
from actorii.display.wayland import (
    WaylandDisplay,
    WaylandDriver,
    WaylandSurface,
    create_driver,
)
from actorii.vm import ActorVM
from websocket_server import ActorWorkspace

# =============================================================================
# 1. UI Action Event Dispatching Stress Tests
# =============================================================================


class TestUIActionEventDispatchAdversarial:
    """Stress-test ui_event -> ActorVM.dispatch_action event routing."""

    def test_button_click_callback_and_pressed_state(self):
        """Button click via ui_event invokes callback and restores pressed state."""

        async def _run():
            ws = ActorWorkspace()
            clicked = []
            btn = Button(
                label="Submit",
                id="btn_submit",
                identity=Symbol("btn_submit"),
                callback=lambda b: clicked.append(b.label),
            )
            ws.vm.global_vars["Actor"]["btn_submit"] = btn

            res = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "btn_submit",
                    "action": "#selected",
                }
            )
            assert res["type"] == "ui_event_result"
            assert res["handled"] is True
            assert res["widget_id"] == "btn_submit"
            assert len(clicked) == 1
            assert clicked[0] == "Submit"
            assert btn.pressed is False  # Restored after click

            # Click again with different selector aliases
            for action in ["selected", "#clicked", "clicked", "press"]:
                res2 = await ws.handle_message(
                    {
                        "type": "ui_event",
                        "widget_id": "btn_submit",
                        "action": action,
                    }
                )
                assert res2["handled"] is True
            assert len(clicked) == 5

        asyncio.run(_run())

    def test_button_click_parent_window_action_routing(self):
        """Button inside Window routes action to window.action_handlers mapping."""

        async def _run():
            ws = ActorWorkspace()
            win = Window(title="ParentWindow")
            btn = Button(label="ActionBtn", id=101, identity=Symbol("actBtn"))
            win.add(btn)

            win.action_handlers[("actBtn", "selected")] = "onActBtnSelected"
            received_payloads = []
            win.set_method(
                "onActBtnSelected", lambda self, msg: received_payloads.append(msg)
            )
            ws.vm.global_vars["Actor"]["mainWin"] = win

            res = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "actBtn",
                    "action": "#selected",
                }
            )
            assert res["handled"] is True
            assert len(received_payloads) == 1
            payload = received_payloads[0]
            assert payload["action"] == Symbol("selected")
            assert payload["source"] is btn

        asyncio.run(_run())

    def test_modal_dialog_ok_cancel_action_dispatch(self):
        """Clicking standard OK and Cancel buttons closes Dialog with appropriate code."""

        async def _run():
            ws = ActorWorkspace()
            dlg = Dialog(title="ConfirmDialog", modal=True)
            ok_btn, cancel_btn = dlg.create_standard_buttons()
            ws.vm.global_vars["Actor"]["myDialog"] = dlg

            dlg.is_modal = True
            dlg.show()
            assert dlg.is_visible() is True

            # Click OK button via ui_event
            res_ok = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": Dialog.IDOK,
                    "action": "#selected",
                }
            )
            assert res_ok["handled"] is True
            assert dlg.result_code == Dialog.IDOK
            assert dlg.is_modal is False

            # Reset dialog and click Cancel button
            dlg.is_modal = True
            dlg.show()
            res_cancel = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": Dialog.IDCANCEL,
                    "action": "#selected",
                }
            )
            assert res_cancel["handled"] is True
            assert dlg.result_code == Dialog.IDCANCEL
            assert dlg.is_modal is False

        asyncio.run(_run())

    def test_checkbox_toggle_states_checked_and_unchecked(self):
        """Checkbox toggles between #checked and #unChecked states on successive clicks."""

        async def _run():
            ws = ActorWorkspace()
            chk = Checkbox(
                label="EnableFeature",
                checked=False,
                id="featureChk",
                identity=Symbol("featureChk"),
            )
            ws.vm.global_vars["Actor"]["featureChk"] = chk

            # Initial state
            assert chk.checked is False
            assert chk.data == Symbol("unChecked")
            assert chk.value() is False

            # 1. First click: toggles to True / #checked
            r1 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "featureChk",
                    "action": "#selected",
                }
            )
            assert r1["handled"] is True
            assert chk.checked is True
            assert chk.data == Symbol("checked")
            assert chk.value() is True

            # 2. Second click: toggles to False / #unChecked
            r2 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "featureChk",
                    "action": "selected",
                }
            )
            assert r2["handled"] is True
            assert chk.checked is False
            assert chk.data == Symbol("unChecked")

            # 3. Third click: toggles back to True
            r3 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "featureChk",
                    "action": "#clicked",
                }
            )
            assert chk.checked is True
            assert chk.data == Symbol("checked")

            # 4. Direct method actions: check() and unCheck()
            await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "featureChk",
                    "action": "unCheck",
                }
            )
            assert chk.checked is False
            assert chk.data == Symbol("unChecked")

            await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "featureChk",
                    "action": "check",
                }
            )
            assert chk.checked is True
            assert chk.data == Symbol("checked")

            # 5. setValue with symbol/boolean values
            chk.setValue(Symbol("unChecked"))
            assert chk.checked is False
            assert chk.data == Symbol("unChecked")

            chk.setValue(Symbol("checked"))
            assert chk.checked is True
            assert chk.data == Symbol("checked")

        asyncio.run(_run())

    def test_checkbox_parent_window_action_handlers(self):
        """Checkbox inside Window dispatches checked and unChecked action events to parent."""

        async def _run():
            ws = ActorWorkspace()
            win = Window(title="SettingsWin")
            chk = Checkbox(
                label="AutoSave",
                checked=False,
                id="autoSaveChk",
                identity=Symbol("autoSaveChk"),
            )
            win.add(chk)

            win.action_handlers[("autoSaveChk", "checked")] = "onAutoSaveChecked"
            win.action_handlers[("autoSaveChk", "unChecked")] = "onAutoSaveUnchecked"

            events = []
            win.set_method(
                "onAutoSaveChecked", lambda self, msg: events.append(("checked", msg))
            )
            win.set_method(
                "onAutoSaveUnchecked",
                lambda self, msg: events.append(("unChecked", msg)),
            )
            ws.vm.global_vars["Actor"]["settingsWin"] = win

            # Dispatch #checked
            r_chk = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "autoSaveChk",
                    "action": "#checked",
                }
            )
            assert r_chk["handled"] is True
            assert len(events) == 1
            assert events[0][0] == "checked"
            assert events[0][1]["action"] == Symbol("checked")

            # Dispatch #unChecked
            r_unchk = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "autoSaveChk",
                    "action": "#unChecked",
                }
            )
            assert r_unchk["handled"] is True
            assert len(events) == 2
            assert events[1][0] == "unChecked"
            assert events[1][1]["action"] == Symbol("unChecked")

        asyncio.run(_run())

    def test_custom_widget_selector_dispatch(self):
        """Custom widget selector invokes matching widget method with arguments."""

        async def _run():
            ws = ActorWorkspace()
            txt = TextField(text="Initial", id="txtField1", max_length=50)
            ws.vm.global_vars["Actor"]["txtField1"] = txt

            # Invoke setText with args
            r1 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "txtField1",
                    "action": "setText",
                    "args": ["Modified Content"],
                }
            )
            assert r1["handled"] is True
            assert txt.getText() == "Modified Content"

            # Invoke clear without args
            r2 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "txtField1",
                    "action": "clear",
                }
            )
            assert r2["handled"] is True
            assert txt.getText() == ""

        asyncio.run(_run())

    def test_unhandled_and_missing_widget_ids(self):
        """Stress-test unhandled widget IDs, missing IDs, None, and numeric ID lookup."""

        async def _run():
            ws = ActorWorkspace()

            # 1. Missing widget_id entirely
            r_missing = await ws.handle_message({"type": "ui_event"})
            assert r_missing["type"] == "ui_event_result"
            assert r_missing["handled"] is False
            assert "Missing 'widget_id'" in r_missing["error"]

            # 2. None widget_id
            r_none = await ws.handle_message({"type": "ui_event", "widget_id": None})
            assert r_none["handled"] is False

            # 3. Nonexistent widget ID does not crash server; returns handled: False with error
            r_nonexistent = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "nonexistent_control_xyz",
                    "action": "#selected",
                }
            )
            assert r_nonexistent["type"] == "ui_event_result"
            assert r_nonexistent["handled"] is False
            assert "could not be resolved" in r_nonexistent["error"]

            # 4. Numeric ID lookup in window
            win = Window(title="NumericWin")
            btn = Button(label="NumBtn", id=505, identity=Symbol("btn505"))
            win.add(btn)
            ws.vm.global_vars["Actor"]["numWin"] = win

            called = []
            btn.callback = lambda b: called.append(b.id)

            # Access by integer ID
            r_num = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": 505,
                    "action": "#selected",
                }
            )
            assert r_num["handled"] is True
            assert len(called) == 1
            assert called[0] == 505

            # Access by string representation of ID
            r_num_str = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "505",
                    "action": "#selected",
                }
            )
            assert r_num_str["handled"] is True
            assert len(called) == 2

            # Access by Symbol string "#btn505"
            r_sym = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "#btn505",
                    "action": "#selected",
                }
            )
            assert r_sym["handled"] is True
            assert len(called) == 3

        asyncio.run(_run())


# =============================================================================
# 2. Wayland Headless Surface Stress Tests
# =============================================================================


class TestWaylandHeadlessSurfaceAdversarial:
    """Stress-test Wayland surface pixel ops, coordinates, damage, and buffer constraints."""

    def test_surface_boundary_coordinates(self):
        """Pixel operations exactly at edges (0,0), (w-1, 0), (0, h-1), (w-1, h-1), (w, h)."""
        display = WaylandDisplay(headless=True)
        w, h = 320, 240
        surface = display.create_surface(w, h)

        # 4 extreme valid corners
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]

        for (x, y), c in zip(corners, colors):
            surface.set_pixel(x, y, c)
            assert surface.get_pixel(x, y) == c.lower()

        # Center pixel
        surface.set_pixel(w // 2, h // 2, "#123456")
        assert surface.get_pixel(w // 2, h // 2) == "#123456"

        # Boundary off-by-one: (w, 0), (0, h), (w, h) are out-of-bounds
        oob_points = [(w, 0), (0, h), (w, h), (w + 1, h + 1)]
        for ox, oy in oob_points:
            surface.set_pixel(ox, oy, "#FF0000")
            assert surface.get_pixel(ox, oy) == "#000000"

    def test_negative_coordinates_safety(self):
        """Negative coordinates do not throw exceptions and get_pixel returns #000000."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(100, 100)

        neg_coords = [(-1, 0), (0, -1), (-1, -1), (-100, -100), (-9999, -9999)]
        for nx, ny in neg_coords:
            # Writing to negative coords should safely drop without IndexError
            surface.set_pixel(nx, ny, "#FF0000")
            # Reading from negative coords should return black
            assert surface.get_pixel(nx, ny) == "#000000"

    def test_out_of_bounds_pixel_operations(self):
        """Extreme positive coordinates do not trigger buffer overflow."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(50, 50)

        huge_coords = [(50, 50), (1000, 1000), (999999, 999999)]
        for hx, hy in huge_coords:
            surface.set_pixel(hx, hy, "#00FF00")
            assert surface.get_pixel(hx, hy) == "#000000"

    def test_zero_size_and_boundary_damage_regions(self):
        """Damage regions with 0 width/height, negative values, or exceeding boundaries."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(200, 150)

        # Zero dimensions
        surface.damage(0, 0, 0, 0)
        surface.damage(50, 50, 0, 100)
        surface.damage(50, 50, 100, 0)

        # Default w and h (None -> edge of surface)
        surface.damage(10, 20, None, None)

        # Negative and oversized damage coordinates
        surface.damage(-50, -50, 1000, 1000)
        surface.damage(100, 100, -10, -10)

        surface.flush()

    def test_multi_region_damage_and_flush_sequence(self):
        """Multiple sequential and interleaved damage marks followed by repeated flushes."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(400, 300)

        # Emit 50 damage regions across surface
        for i in range(50):
            x = (i * 7) % 350
            y = (i * 5) % 250
            surface.set_pixel(x, y, "#AABBCC")
            surface.damage(x, y, 10, 10)
            if i % 10 == 0:
                surface.flush()

        # Multiple flush calls in a row
        for _ in range(5):
            surface.flush()

        assert surface.get_pixel(0, 0) == "#aabbcc"

    def test_copy_from_stress_and_edge_cases(self):
        """Surface copy_from between different surfaces, partial rects, and boundary offsets."""
        display = WaylandDisplay(headless=True)
        src = display.create_surface(100, 100)
        dst = display.create_surface(100, 100)

        src.clear("#FF0000")
        dst.clear("#000000")

        # Copy 20x20 block from (10, 10) to (50, 50)
        dst.copy_from(src, x=10, y=10, w=20, h=20, dst_x=50, dst_y=50)
        assert dst.get_pixel(55, 55) == "#ff0000"
        assert dst.get_pixel(10, 10) == "#000000"
        assert dst.get_pixel(80, 80) == "#000000"

        # Copy with None width and height (full remainder of surface)
        src.clear("#00FF00")
        dst.copy_from(src, x=0, y=0, w=None, h=None, dst_x=0, dst_y=0)
        assert dst.get_pixel(0, 0) == "#00ff00"
        assert dst.get_pixel(99, 99) == "#00ff00"

    def test_color_formats_and_buffer_layout(self):
        """Verify hex colors, lowercase/mixed case, non-hex fallback, and buffer stride."""
        display = WaylandDisplay(headless=True)
        w, h = 64, 32
        surface = display.create_surface(w, h)

        # Buffer geometry properties
        assert surface.width == w
        assert surface.height == h
        assert surface.stride == w * 4
        assert surface.format == "xrgb8888"
        assert len(surface._buffer_data) == w * h * 4

        # Initial buffer is white
        assert surface.get_pixel(0, 0) == "#ffffff"

        # Lowercase hex
        surface.set_pixel(1, 1, "#abcdef")
        assert surface.get_pixel(1, 1) == "#abcdef"

        # Mixed-case hex
        surface.set_pixel(2, 2, "#AbCdEf")
        assert surface.get_pixel(2, 2) == "#abcdef"

        # Non-hex string without leading # falls back to black (#000000)
        surface.set_pixel(3, 3, "blue")
        assert surface.get_pixel(3, 3) == "#000000"

        surface.set_pixel(4, 4, "")
        assert surface.get_pixel(4, 4) == "#000000"

    def test_wayland_driver_lifecycle_and_resize(self):
        """WaylandDriver wrapping surface, handling resize, and event dispatch."""
        driver = create_driver("wayland", width=250, height=180, headless=True)
        surface = driver.surface
        assert surface.width == 250
        assert surface.height == 180

        # Pixel op through surface
        surface.set_pixel(15, 15, "#112233")
        assert surface.get_pixel(15, 15) == "#112233"
        driver.present()

        # Handle events: key and pointer
        assert (
            driver.handle_event({"type": "key", "key": "Enter", "down": True}) is True
        )
        assert (
            driver.handle_event(
                {"type": "pointer", "action": "press", "x": 100, "y": 50}
            )
            is True
        )
        assert (
            driver.handle_event(
                {"type": "pointer", "action": "move", "x": 110, "y": 55}
            )
            is True
        )
        assert (
            driver.handle_event(
                {"type": "pointer", "action": "release", "x": 110, "y": 55}
            )
            is True
        )
        assert driver.handle_event({"type": "unknown_event"}) is False

        # Clear surface
        driver.clear("#998877")
        assert surface.get_pixel(0, 0) == "#998877"

    def test_wayland_driver_resize_defect_documentation(self):
        """Verify that WaylandDriver.resize cleanly delegates to WaylandSurface.resize."""
        driver = create_driver("wayland", width=250, height=180, headless=True)
        driver.resize(300, 200)
        assert driver.surface._width == 300
        assert driver.surface._height == 200
        assert driver.surface._stride == 300 * 4


# =============================================================================
# 3. render_widget DOM Generation and Actor Dictionary Binding
# =============================================================================


class TestRenderWidgetDOMAndBindingAdversarial:
    """Stress-test render_widget HTML generation, clamping, and global Actor binding."""

    def test_render_button_dom_and_binding(self):
        """render_widget creates button with correct attributes and binds to Actor."""

        async def _run():
            ws = ActorWorkspace()
            res = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "button",
                    "id": "calcBtn",
                    "label": "Calculate",
                    "x": 50,
                    "y": 90,
                    "width": 140,
                    "height": 36,
                }
            )
            assert res["type"] == "rendered_widget"
            html = res["html"]
            assert "<button" in html
            assert 'data-widget="button"' in html
            assert 'data-id="calcBtn"' in html
            assert 'data-label="Calculate"' in html
            assert "Calculate" in html

            # Verify binding in Actor global dictionary
            assert "calcBtn" in ws.vm.global_vars["Actor"]
            btn = ws.vm.global_vars["Actor"]["calcBtn"]
            assert isinstance(btn, Button)
            assert btn.label == "Calculate"
            assert btn.id == "calcBtn"

            # Verify inspect_object succeeds for this widget
            insp = await ws.handle_message({"type": "inspect_object", "key": "calcBtn"})
            assert insp["found"] is True
            assert insp["class"] == "Button"
            assert insp["fields"]["label"] == "Calculate"

        asyncio.run(_run())

    def test_render_checkbox_dom_and_binding(self):
        """render_widget creates checkbox with checked attribute when True and binds."""

        async def _run():
            ws = ActorWorkspace()

            # 1. Checked checkbox
            res_chk = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "checkbox",
                    "id": "chkEnabled",
                    "label": "Enabled",
                    "checked": True,
                    "x": 30,
                    "y": 60,
                }
            )
            assert "<label" in res_chk["html"]
            assert 'data-widget="checkbox"' in res_chk["html"]
            assert "checked" in res_chk["html"]
            chk1 = ws.vm.global_vars["Actor"]["chkEnabled"]
            assert isinstance(chk1, Checkbox)
            assert chk1.checked is True
            assert chk1.data == Symbol("checked")

            # 2. Unchecked checkbox
            res_unchk = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "checkbox",
                    "id": "chkDisabled",
                    "label": "Disabled",
                    "checked": False,
                    "x": 30,
                    "y": 100,
                }
            )
            assert 'data-widget="checkbox"' in res_unchk["html"]
            # Should NOT contain 'checked>' for input tag
            assert (
                '<input type="checkbox" >' in res_unchk["html"]
                or "checked" not in res_unchk["html"].split("<input")[1].split(">")[0]
            )
            chk2 = ws.vm.global_vars["Actor"]["chkDisabled"]
            assert isinstance(chk2, Checkbox)
            assert chk2.checked is False
            assert chk2.data == Symbol("unChecked")

        asyncio.run(_run())

    def test_render_textfield_dom_and_binding(self):
        """render_widget creates textfield input and binds to Actor global dictionary."""

        async def _run():
            ws = ActorWorkspace()
            res = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "textfield",
                    "id": "usernameInput",
                    "label": "Enter Username",
                    "width": 180,
                    "height": 28,
                    "x": 40,
                    "y": 120,
                }
            )
            html = res["html"]
            assert "<input" in html
            assert 'data-widget="textfield"' in html
            assert 'data-id="usernameInput"' in html
            assert 'placeholder="Enter Username"' in html

            txt = ws.vm.global_vars["Actor"]["usernameInput"]
            assert isinstance(txt, TextField)
            assert txt.text == "Enter Username"

        asyncio.run(_run())

    def test_render_widget_coordinate_percentage_clamping(self):
        """render_widget clamps coordinates between 1.0%-82.0% (left) and 6.0%-88.0% (top)."""

        async def _run():
            ws = ActorWorkspace()

            # Extreme negative coordinates
            res_neg = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "button",
                    "id": "negBtn",
                    "x": -500,
                    "y": -500,
                }
            )
            assert "left: 1.0%" in res_neg["html"]
            assert "top: 6.0%" in res_neg["html"]

            # Extreme positive coordinates
            res_huge = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "button",
                    "id": "hugeBtn",
                    "x": 50000,
                    "y": 50000,
                }
            )
            assert "left: 82.0%" in res_huge["html"]
            assert "top: 88.0%" in res_huge["html"]

        asyncio.run(_run())

    def test_render_widget_custom_fallback_markup(self):
        """render_widget handles unsupported widget_type gracefully with generic div."""

        async def _run():
            ws = ActorWorkspace()
            res = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "custom_slider",
                    "id": "mySlider",
                    "label": "Volume",
                    "x": 80,
                    "y": 120,
                }
            )
            assert res["type"] == "rendered_widget"
            assert '<div data-widget="custom_slider"' in res["html"]
            assert 'data-id="mySlider"' in res["html"]
            assert "Volume" in res["html"]

        asyncio.run(_run())

    def test_render_widget_e2e_event_and_inspect_integration(self):
        """End-to-end flow: render button -> inspect -> click via ui_event -> check state."""

        async def _run():
            ws = ActorWorkspace()

            # 1. Render button
            r1 = await ws.handle_message(
                {
                    "type": "render_widget",
                    "widget_type": "button",
                    "id": "e2eBtn",
                    "label": "Test E2E",
                }
            )
            assert r1["type"] == "rendered_widget"

            # 2. Inspect button
            r2 = await ws.handle_message({"type": "inspect_object", "key": "e2eBtn"})
            assert r2["found"] is True
            assert r2["class"] == "Button"

            # 3. Attach callback to button and click via ui_event
            btn = ws.vm.global_vars["Actor"]["e2eBtn"]
            calls = []
            btn.callback = lambda b: calls.append("invoked")

            r3 = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "e2eBtn",
                    "action": "#selected",
                }
            )
            assert r3["handled"] is True
            assert len(calls) == 1

        asyncio.run(_run())


# =============================================================================
# 4. Demo Wayland Execution Verification
# =============================================================================


def test_demo_wayland_subprocess_clean_exit():
    """Verify demo_wayland.py executes cleanly to completion with exit code 0."""
    demo_path = os.path.join(_PROJECT_ROOT, "demo_wayland.py")
    proc = subprocess.run(
        [sys.executable, demo_path],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"demo_wayland.py failed:\n{proc.stderr}"
    stdout = proc.stdout
    assert "=== ActorII Wayland Display Demo ===" in stdout
    assert "[1/6] Creating Wayland display connection..." in stdout
    assert "[2/6] Creating Wayland surface (800x600)..." in stdout
    assert "[3/6] Clearing surface to white background..." in stdout
    assert "[4/6] Rendering widgets on Wayland surface..." in stdout
    assert "[5/6] Marking damage region and flushing..." in stdout
    assert "[6/6] Demonstrating pixel operations..." in stdout
    assert "=== Demo complete ===" in stdout
