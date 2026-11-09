# -*- coding: utf-8 -*-
"""Empirical Adversarial Test Suite for Milestone M4 (Geometry, GUI & Action Dispatch).

Author: Challenger 2 (Empirical Challenger)
Covers:
1. Packaging & Import Graph Integrity (circular import detection)
2. Point & Point3D Vector Arithmetic, Boundary Math & Isometric Projection
3. Rect Bounding Box Math, Containment, Intersections, Unions & Dialog Units
4. GUI Widget Hierarchy (WindowsObject, Window, Dialog, Control, Button, TextField, Checkbox)
5. Checkbox Interaction & Toggle Mechanics under on_click / dispatch_action
6. Modal Dialog Flow, Return Values & Lifecycle
7. Action Event Dispatching & Symbol Lookup
8. Historical Benchmark Regressions (SIEVE.ACT, SIEVE1.ACT, FRACTAL.ACT)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

# Ensure actorii.core is loaded before actorii.classes in this module so the module can load
import actorii.core
from actorii.core import ActorObject, Symbol
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
from actorii.vm import ActorVM

HISTORICAL_ACT_ROOT = (
    Path(__file__).resolve().parent.parent / "historical_actor_41_source" / "actact"
)


# ==============================================================================
# 1. Packaging & Import Graph Integrity (Adversarial Import Isolation)
# ==============================================================================


class TestPackagingAndImports:
    """Stress tests module isolation to ensure modules can be imported directly."""

    def test_direct_import_actorii_classes(self):
        """Verify 'import actorii.classes' succeeds in a fresh Python process."""
        cmd = [sys.executable, "-c", "import actorii.classes"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"'import actorii.classes' failed with code {res.returncode}:\n{res.stderr}"

    def test_direct_import_geometry_point(self):
        """Verify 'from actorii.classes.geometry import Point' succeeds in isolation."""
        cmd = [sys.executable, "-c", "from actorii.classes.geometry import Point"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"'from actorii.classes.geometry import Point' failed with code {res.returncode}:\n{res.stderr}"

    def test_direct_import_gui_window(self):
        """Verify 'from actorii.classes.gui import Window' succeeds in isolation."""
        cmd = [sys.executable, "-c", "from actorii.classes.gui import Window"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"'from actorii.classes.gui import Window' failed with code {res.returncode}:\n{res.stderr}"


# ==============================================================================
# 2. Point & Point3D Vector Arithmetic & Isometric Projection
# ==============================================================================


class TestPointAdversarial:
    """Adversarial stress tests for 2D Point."""

    def test_point_coordinates_and_accessors(self):
        p = Point(15, -25)
        assert p.x == 15
        assert p.y == -25
        assert p.get_x() == 15
        assert p.get_y() == -25

        p.setX(100)
        assert p.x == 100
        p.setY(-200)
        assert p.y == -200

        p.set(0, 0)
        assert p.x == 0 and p.y == 0

    def test_point_vector_addition(self):
        p1 = Point(10, 20)
        p2 = Point(-5, 15)
        res = p1 + p2
        assert res == Point(5, 35)

        # Scalar addition
        res_scalar = p1 + 5
        assert res_scalar == Point(15, 25)
        # Reverse scalar addition
        res_radd = 5 + p1
        assert res_radd == Point(15, 25)

        with pytest.raises(TypeError):
            _ = p1 + "invalid"

    def test_point_vector_subtraction(self):
        p1 = Point(10, 20)
        p2 = Point(3, 5)
        assert (p1 - p2) == Point(7, 15)

        # Scalar subtraction
        assert (p1 - 4) == Point(6, 16)

        with pytest.raises(TypeError):
            _ = p1 - [1, 2]

    def test_point_scalar_multiplication(self):
        p = Point(7, -3)
        assert (p * 3) == Point(21, -9)
        assert (3 * p) == Point(21, -9)
        assert (p * 0) == Point(0, 0)

        with pytest.raises(TypeError):
            _ = p * Point(1, 1)

    def test_point_division_and_zero_division(self):
        p = Point(20, -10)
        assert (p // 2) == Point(10, -5)
        assert (p / 2) == Point(10, -5)

        with pytest.raises(ZeroDivisionError):
            _ = p / 0

        with pytest.raises(ZeroDivisionError):
            _ = p // 0

    def test_point_equality_and_hashing(self):
        p1 = Point(42, 99)
        p2 = Point(42, 99)
        p3 = Point(42, 100)

        assert p1 == p2
        assert p1 != p3
        assert p1 != (42, 99)
        assert hash(p1) == hash(p2)

        s = {p1}
        assert p2 in s
        assert p3 not in s

    def test_point_as_string_and_representation(self):
        p = Point(10, 20)
        assert p.asString() == "10@20"
        assert repr(p) == "Point(10, 20)"
        assert p.asPoint() is p

    def test_point_midpoint(self):
        p1 = Point(0, 0)
        p2 = Point(10, 20)
        assert p1.middlePoint(p2) == Point(5, 10)

        # Odd coordinates floored
        p3 = Point(3, 5)
        assert p1.middlePoint(p3) == Point(1, 2)


class TestPoint3DAdversarial:
    """Adversarial stress tests for 3D Point and isometric projection."""

    def test_point3d_coordinates_and_accessors(self):
        p3 = Point3D(10, 20, 30)
        assert p3.x == 10
        assert p3.y == 20
        assert p3.z == 30
        assert p3.get_z() == 30

        p3.setZ(99)
        assert p3.z == 99

    def test_point3d_authentic_isometric_projection(self):
        """Authentic 1993 Whitewater Actor projection formula: Point(x, y - (z * 9) // 8)."""
        # Historical baseline: FRACTAL.ACT uses Point3D(256, 20, 0) -> Point(256, 20)
        p_base = Point3D(256, 20, 0)
        assert p_base.asPoint() == Point(256, 20)

        # z = 8 -> y - (8 * 9) // 8 = y - 9
        assert Point3D(100, 200, 8).asPoint() == Point(100, 191)

        # z = 80 -> y - (80 * 9) // 8 = y - 90
        assert Point3D(100, 200, 80).asPoint() == Point(100, 110)

        # Negative z: z = -8 -> y - (-8 * 9) // 8 = y - (-9) = y + 9
        assert Point3D(100, 200, -8).asPoint() == Point(100, 209)

        # Large coordinates
        p_large = Point3D(10000, 20000, 8000)
        assert p_large.asPoint() == Point(10000, 20000 - 9000)

    def test_point3d_vector_arithmetic(self):
        p3a = Point3D(10, 20, 30)
        p3b = Point3D(1, 2, 3)

        assert (p3a + p3b) == Point3D(11, 22, 33)
        assert (p3a - p3b) == Point3D(9, 18, 27)
        assert (p3a * 2) == Point3D(20, 40, 60)
        assert (2 * p3a) == Point3D(20, 40, 60)
        assert (p3a // 2) == Point3D(5, 10, 15)
        assert (p3a / 2) == Point3D(5, 10, 15)

        # Add 2D Point to 3D Point (preserves z)
        p2 = Point(5, 5)
        assert (p3a + p2) == Point3D(15, 25, 30)
        assert (p3a - p2) == Point3D(5, 15, 30)

        # Division by zero
        with pytest.raises(ZeroDivisionError):
            _ = p3a / 0

    def test_point3d_equality_and_hashing(self):
        p3a = Point3D(1, 2, 3)
        p3b = Point3D(1, 2, 3)
        p3c = Point3D(1, 2, 4)

        assert p3a == p3b
        assert p3a != p3c
        assert p3a != Point(1, 2)
        assert hash(p3a) == hash(p3b)

    def test_point3d_middlepoint(self):
        p1 = Point3D(0, 0, 0)
        p2 = Point3D(10, 20, 30)
        mid = p1.middlePoint(p2, rng=0)
        assert mid == Point3D(5, 10, 15)


# ==============================================================================
# 3. Rect Bounding Box Math, Containment, Intersections & Unions
# ==============================================================================


class TestRectAdversarial:
    """Adversarial stress tests for Rect geometry."""

    def test_rect_construction_modes(self):
        # 1. 4 integers
        r1 = Rect(10, 20, 110, 120)
        assert r1.left == 10 and r1.top == 20
        assert r1.right == 110 and r1.bottom == 120
        assert r1.width == 100 and r1.height == 100
        assert r1.origin == Point(10, 20)
        assert r1.corner == Point(110, 120)

        # 2. 2 Point instances (origin, corner)
        r2 = Rect(Point(10, 20), Point(110, 120))
        assert r2 == r1
        assert r2.width == 100
        assert r2.height == 100

    def test_rect_mutators(self):
        r = Rect(0, 0, 100, 50)
        r.width = 200
        assert r.right == 200
        assert r.width == 200

        r.height = 150
        assert r.bottom == 150
        assert r.height == 150

        r.setOrigin(Point(25, 35))
        assert r.left == 25 and r.top == 35

        r.setCorner(Point(225, 185))
        assert r.right == 225 and r.bottom == 185

    def test_rect_containment(self):
        r = Rect(10, 20, 110, 120)

        # Interior
        assert r.contains(50, 50)
        assert r.containsPoint(Point(50, 50))

        # Boundaries
        assert r.contains(10, 20)
        assert r.contains(110, 120)
        assert r.contains(10, 120)
        assert r.contains(110, 20)

        # Exterior
        assert not r.contains(9, 50)
        assert not r.contains(111, 50)
        assert not r.contains(50, 19)
        assert not r.contains(50, 121)

    def test_rect_containment_inverted_bounds(self):
        """Negative / inverted coordinates (left > right, top > bottom)."""
        r_inv = Rect(100, 100, 0, 0)
        assert r_inv.width == 100
        assert r_inv.height == 100
        assert r_inv.contains(50, 50)
        assert not r_inv.contains(150, 50)

    def test_rect_intersections(self):
        r1 = Rect(0, 0, 100, 100)

        # Overlapping
        r_overlap = Rect(50, 50, 150, 150)
        assert r1.intersects(r_overlap)

        # Touching boundary edge
        r_touch = Rect(100, 0, 200, 100)
        assert r1.intersects(r_touch)

        # Disjoint
        r_disjoint = Rect(101, 101, 200, 200)
        assert not r1.intersects(r_disjoint)

        # Enclosing
        r_enclosed = Rect(20, 20, 40, 40)
        assert r1.intersects(r_enclosed)

    def test_rect_union(self):
        r1 = Rect(10, 20, 50, 60)
        r2 = Rect(40, 50, 100, 120)
        u = r1.union(r2)
        assert u == Rect(10, 20, 100, 120)

        # Disjoint union bounding box
        r_disj = Rect(200, 300, 250, 350)
        u_disj = r1.union(r_disj)
        assert u_disj == Rect(10, 20, 250, 350)

    def test_rect_dialog_units_conversion(self):
        # 4 px per horiz dlg unit, 8 px per vert dlg unit
        r = Rect.from_dlg_units(10, 20, 30, 40)
        assert r.left == 40
        assert r.top == 160
        assert r.width == 120
        assert r.height == 320

        # Round trip
        assert r.to_dlg_units() == (10, 20, 30, 40)


# ==============================================================================
# 4. GUI Hierarchy (WindowsObject, Window, Dialog, Control, Button, TextField)
# ==============================================================================


class TestGUIHierarchyAdversarial:
    """Stress tests GUI Widget hierarchy constraints and behaviors."""

    def test_control_rejects_children(self):
        """Authentic 1993 constraint: Controls cannot have children (CONTROL.CLS:96)."""
        btn = Button("OK")
        assert not btn.should_have_children()
        with pytest.raises(TypeError, match="cannot have children"):
            btn.add(Button("Child"))

        tf = TextField("Input")
        assert not tf.should_have_children()
        with pytest.raises(TypeError, match="cannot have children"):
            tf.add(TextField("Child"))

    def test_window_child_registration_and_removal(self):
        win = Window("Main Window")
        btn = Button("Click Me", identity=Symbol("myButton"), id=101)
        tf = TextField("Hello", identity=Symbol("myText"), id=102)

        win.add(btn)
        win.add(tf)
        assert len(win.children) == 2
        assert btn.parent is win
        assert tf.parent is win

        # Duplicate addition is a no-op
        win.add(btn)
        assert len(win.children) == 2

        # Access via indexing
        assert win[Symbol("#myButton")] is btn
        assert win[Symbol("myText")] is tf
        assert win[101] is btn
        assert win["Click Me"] is btn

        # Removal
        assert win.remove(btn) is btn
        assert btn.parent is None
        assert len(win.children) == 1

        # Removal by symbol
        assert win.remove(Symbol("myText")) is tf
        assert len(win.children) == 0

    def test_textfield_max_length_and_clear(self):
        tf = TextField("Initial", max_length=10)
        assert tf.getText() == "Initial"

        tf.setText("1234567890EXTRA")
        assert tf.getText() == "1234567890"

        tf.clear()
        assert tf.getText() == ""
        assert tf.get_value() == ""


# ==============================================================================
# 5. Checkbox Interaction & Toggle Mechanics (CRITICAL VERIFICATION)
# ==============================================================================


class TestCheckboxAdversarial:
    """Adversarially tests Checkbox behavior on click, press, and dispatch_action."""

    def test_checkbox_manual_toggle(self):
        chk = Checkbox("Option", checked=False)
        assert not chk.value()
        assert chk.data == Symbol("unChecked")

        res = chk.toggle()
        assert res is True
        assert chk.value() is True
        assert chk.data == Symbol("checked")

        chk.toggle()
        assert chk.value() is False
        assert chk.data == Symbol("unChecked")

    def test_checkbox_click_should_toggle_state(self):
        """Empirically test whether on_click() toggles Checkbox.checked.

        In authentic Actor 1993 CHECKBOX.CLS:73:
            Action pressed(self, msg) #[#selected] { toggle(self); }
        When a user clicks a checkbox, its checked state must flip!
        """
        chk = Checkbox("Option", checked=False)
        assert chk.checked is False
        assert chk.value() is False

        # Simulate user click
        chk.on_click()
        assert (
            chk.checked is True
        ), f"Expected checkbox to toggle to True on click, but remained {chk.checked}!"
        assert chk.value() is True

        # Click again to uncheck
        chk.on_click()
        assert (
            chk.checked is False
        ), f"Expected checkbox to toggle back to False on click, but remained {chk.checked}!"

    def test_checkbox_dispatch_action_selected_should_toggle(self):
        """Empirically test whether vm.dispatch_action(chk, 'selected') toggles the checkbox."""
        vm = ActorVM()
        chk = Checkbox("EnableSound", checked=False)

        vm.dispatch_action(chk, "selected")
        assert (
            chk.checked is True
        ), f"vm.dispatch_action(chk, 'selected') failed to toggle checkbox state (checked={chk.checked})!"


# ==============================================================================
# 6. Modal Dialog Flow & Return Codes
# ==============================================================================


class TestDialogModalFlowAdversarial:
    """Tests Modal Dialog lifecycle, standard buttons, and result codes."""

    def test_dialog_standard_buttons_and_modal_closure(self):
        dlg = Dialog("Confirm Action", modal=True)
        btn_ok, btn_cancel = dlg.create_standard_buttons()

        assert btn_ok.parent is dlg
        assert btn_cancel.parent is dlg
        assert btn_ok.id == Dialog.IDOK
        assert btn_cancel.id == Dialog.IDCANCEL

        # Start modal
        dlg.run_modal()
        assert dlg.is_modal is True
        assert dlg.visible is True

        # Simulate clicking Cancel
        btn_cancel.on_click()
        assert dlg.is_modal is False
        assert dlg.visible is False
        assert dlg.result_code == Dialog.IDCANCEL

    def test_dialog_ok_button_click(self):
        dlg = Dialog("Prompt", modal=True)
        btn_ok, _ = dlg.create_standard_buttons()
        dlg.run_modal()

        btn_ok.on_click()
        assert dlg.is_modal is False
        assert dlg.result_code == Dialog.IDOK


# ==============================================================================
# 7. Action Event Dispatching & Symbol Routing
# ==============================================================================


class TestActionEventDispatchAdversarial:
    """Stress tests ActorVM.dispatch_action event routing."""

    def test_dispatch_action_to_parent_window_handler(self):
        vm = ActorVM()
        win = Window("Main")
        btn = Button("Save", identity=Symbol("btnSave"))
        win.add(btn)

        received_events = []

        def save_handler(msg):
            received_events.append(msg)
            return "SAVED_OK"

        win.set_method("onSave", save_handler)
        win.add_action_handler("btnSave", "selected", "onSave")

        res = vm.dispatch_action(btn, "selected")
        assert res == "SAVED_OK"
        assert len(received_events) == 1
        assert received_events[0]["source"] is btn

    def test_dispatch_action_by_symbol_string_in_global_window(self):
        vm = ActorVM()
        win = Window("Editor")
        btn = Button("Execute", identity=Symbol("runButton"))
        win.add(btn)

        # Store window in Actor dictionary
        vm.global_vars["Actor"]["EditorWin"] = win

        clicked = []
        btn.callback = lambda b: clicked.append("CLICKED")

        vm.dispatch_action("#runButton", "selected")
        assert clicked == ["CLICKED"]


# ==============================================================================
# 8. Historical Benchmark Regressions (SIEVE.ACT, SIEVE1.ACT, FRACTAL.ACT)
# ==============================================================================


class TestHistoricalBenchmarkRegressions:
    """Verifies that M4 changes introduced zero regressions to authentic 1993 benchmarks."""

    def test_sieve_act_returns_25_primes(self):
        sieve_path = HISTORICAL_ACT_ROOT / "SIEVE.ACT"
        assert sieve_path.is_file(), f"Missing SIEVE.ACT at {sieve_path}"
        source = sieve_path.read_text(encoding="latin-1")

        vm = ActorVM()
        vm.eval_chunk(source)

        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None, "Sam must be defined in Actor dictionary"

        result = vm.dispatch_message(Sam, "sieve", [100])
        assert result == 25, f"Expected 25 primes <= 100, got {result}"

    def test_sieve1_act_early_bound_returns_25_primes(self):
        sieve1_path = HISTORICAL_ACT_ROOT / "SIEVE1.ACT"
        assert sieve1_path.is_file(), f"Missing SIEVE1.ACT at {sieve1_path}"
        source = sieve1_path.read_text(encoding="latin-1")

        vm = ActorVM()
        vm.eval_chunk(source)

        Sam = vm.global_vars.get("Sam") or vm.global_vars["Actor"].get("Sam")
        assert Sam is not None

        result = vm.dispatch_message(Sam, "sieve", [100])
        assert result == 25, f"Expected 25 primes <= 100, got {result}"

    def test_fractal_act_compiles_and_projects_point3d(self):
        fractal_path = HISTORICAL_ACT_ROOT / "FRACTAL.ACT"
        assert fractal_path.is_file(), f"Missing FRACTAL.ACT at {fractal_path}"
        source = fractal_path.read_text(encoding="latin-1")

        vm = ActorVM()
        vm.eval_chunk(source)

        pt3d = vm.dispatch_message(256, "point3D", [20, 0])
        assert isinstance(pt3d, Point3D)
        assert pt3d.x == 256 and pt3d.y == 20 and pt3d.z == 0

        pt2d = pt3d.asPoint()
        assert isinstance(pt2d, Point)
        assert pt2d.x == 256 and pt2d.y == 20
