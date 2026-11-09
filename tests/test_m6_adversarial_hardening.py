"""Milestone M6 White-Box Adversarial Coverage Hardening Test Suite.

Verifies hardening fixes specified by Challengers M6-1 and M6-2:
1. Display Surfaces:
   - WaylandSurface.resize reallocates buffer bytearray, initializes pixels to white,
     preserves overlapping region contents, updates dimensions/stride.
   - WaylandDriver.resize cleanly delegates to WaylandSurface.resize.
2. Workspace Server:
   - ActorWorkspace._eval_source delegates to ActorVM.eval_chunk, fixing statement
     expressions (e.g. if-then-else-endif returning evaluated value).
   - ActorWorkspace._resolve_widget resolves widgets across instances, workspace widgets,
     vm.global_vars, Actor dictionary, and window child hierarchies.
   - ActorWorkspace.handle_message for ui_event returns handled=False with error message
     when widget cannot be resolved.
   - ActorWorkspace.inspect_object on ActorClass returns instance_methods, class_methods,
     and combined methods.
3. Collections & Set Class:
   - ActorCollection safe size() and __len__() without mutual recursion.
   - ActorSet collection ported from historical SET.CLS with add, remove, includes,
     do, size, copy, registered in Actor dictionary as both string and Symbol.
4. Virtual Machine Hardening:
   - Non-local return resumption: unwinding NonLocalReturn allows caller frames
     to resume execution rather than prematurely halting the VM.
   - Call stack cleanup on unhandled exception: unwinds zombie frames.
   - Recursion guard: max_stack_depth (500) prevents infinite recursion hangs.
   - Field access: _handle_get_field and _handle_set_field resolve field names from
     constants and obj._fields, maintaining operand stack neutrality.
   - Receiver guard: dispatching new or methods on Python classes subclassing
     ActorObject avoids unbound method TypeErrors.
5. Parser & Compiler Hardening:
   - Unrecognized binary operators raise explicit NotImplementedError instead of
     falling through to BINARY_ADD.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.core import ActorClass, ActorObject, Symbol
from actorii.classes.collections import ActorCollection, ActorSet, Set
from actorii.classes.gui import Button, Window
from actorii.display.wayland import (
    WaylandDisplay,
    WaylandDriver,
    WaylandSurface,
    create_driver,
)
from actorii.vm import ActorVM, NonLocalReturn
from websocket_server import ActorWorkspace

# =============================================================================
# 1. Display Surfaces Hardening Tests
# =============================================================================


class TestWaylandDisplayHardening:
    """Adversarial tests for WaylandSurface.resize and WaylandDriver.resize."""

    def test_wayland_surface_resize_buffer_reallocation(self):
        """Verify that WaylandSurface.resize reallocates the buffer correctly."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(width=200, height=150)
        assert surface._width == 200
        assert surface._height == 150
        assert surface._stride == 800
        assert len(surface._buffer_data) == 200 * 150 * 4

        # Resize to larger dimensions
        surface.resize(400, 300)
        assert surface._width == 400
        assert surface._height == 300
        assert surface._stride == 1600
        assert len(surface._buffer_data) == 400 * 300 * 4
        assert surface._buffer["stride"] == 1600

    def test_wayland_surface_resize_pixel_preservation(self):
        """Verify that resize preserves existing pixel contents in overlapping regions."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(width=100, height=100)

        # Set specific pixel colors within 50x50 area
        surface.set_pixel(10, 10, "#FF0000")
        surface.set_pixel(25, 25, "#00FF00")
        surface.set_pixel(40, 40, "#0000FF")

        assert surface.get_pixel(10, 10) == "#ff0000"
        assert surface.get_pixel(25, 25) == "#00ff00"
        assert surface.get_pixel(40, 40) == "#0000ff"

        # Resize up: pixels must be preserved, new pixels must be white
        surface.resize(200, 200)
        assert surface.get_pixel(10, 10) == "#ff0000"
        assert surface.get_pixel(25, 25) == "#00ff00"
        assert surface.get_pixel(40, 40) == "#0000ff"
        # Newly exposed pixel
        assert surface.get_pixel(150, 150) == "#ffffff"

        # Resize down (still containing the pixels)
        surface.resize(50, 50)
        assert surface.get_pixel(10, 10) == "#ff0000"
        assert surface.get_pixel(25, 25) == "#00ff00"
        assert surface.get_pixel(40, 40) == "#0000ff"

    def test_wayland_surface_resize_clamping_and_boundaries(self):
        """Verify boundary inputs (zero, negative) are clamped to at least 1x1."""
        display = WaylandDisplay(headless=True)
        surface = display.create_surface(width=50, height=50)

        # Zero or negative dimensions clamped to 1
        surface.resize(0, -10)
        assert surface._width == 1
        assert surface._height == 1
        assert surface._stride == 4
        assert len(surface._buffer_data) == 4
        assert surface.get_pixel(0, 0) == "#ffffff"

    def test_wayland_driver_resize_delegation(self):
        """Verify WaylandDriver.resize cleanly delegates to WaylandSurface.resize."""
        driver = create_driver("wayland", width=120, height=80, headless=True)
        assert driver.surface._width == 120
        assert driver.surface._height == 80

        # Resize via driver
        driver.resize(320, 240)
        assert driver.surface._width == 320
        assert driver.surface._height == 240
        assert driver.surface._stride == 1280


# =============================================================================
# 2. Workspace Server Hardening Tests
# =============================================================================


class TestWorkspaceServerHardening:
    """Adversarial tests for ActorWorkspace expression eval, widget resolution, and introspection."""

    def test_eval_source_statement_expression_handling(self):
        """Verify _eval_source delegates to eval_chunk and returns statement expression values."""
        ws = ActorWorkspace()
        # Statement expression (if-then-else-endif) previously returned None in _eval_source
        res = ws._eval_source("if 1 then 42 else 99 endif!!")
        assert res == 42

        # Direct mathematical expression
        res_math = ws._eval_source("100 + 250!!")
        assert res_math == 350

        # Empty/whitespace source returns None safely
        assert ws._eval_source("") is None
        assert ws._eval_source("   ") is None

    def test_eval_source_historical_benchmark_execution(self):
        """Verify _eval_source accurately evaluates benchmark code."""
        ws = ActorWorkspace()
        sieve_path = Path(_PROJECT_ROOT) / "historical_actor_41_source/actact/SIEVE.ACT"
        if sieve_path.exists():
            ws._eval_source(sieve_path.read_text(encoding="latin-1"))
            res = ws._eval_source("sieve(Sam, 100)!!")
            assert res == 25

    def test_resolve_widget_scopes(self):
        """Verify _resolve_widget resolves widgets across all valid scopes."""
        ws = ActorWorkspace()

        # 1. Existing instance
        btn_direct = Button(label="Direct", id=101)
        assert ws._resolve_widget(btn_direct) is btn_direct

        # 2. Workspace widgets dict
        btn_ws = Button(label="InWs", id=102)
        ws.widgets["btn_ws"] = btn_ws
        assert ws._resolve_widget("btn_ws") is btn_ws

        # 3. Global vars
        btn_glob = Button(label="InGlob", id=103)
        ws.vm.global_vars["btn_glob"] = btn_glob
        assert ws._resolve_widget("btn_glob") is btn_glob
        assert ws._resolve_widget("#btn_glob") is btn_glob

        # 4. Global Actor dictionary
        btn_actor = Button(label="InActor", id=104)
        ws.vm.global_vars["Actor"]["btn_actor"] = btn_actor
        assert ws._resolve_widget("btn_actor") is btn_actor
        assert ws._resolve_widget("#btn_actor") is btn_actor

        # 5. Window child hierarchy
        win = Window(title="ParentWin")
        child_btn = Button(label="ChildBtn", id=105, identity=Symbol("childBtn"))
        win.add(child_btn)
        ws.vm.global_vars["Actor"]["mainWin"] = win
        assert ws._resolve_widget(105) is child_btn
        assert ws._resolve_widget("childBtn") is child_btn

        # 6. Unresolvable widget returns None
        assert ws._resolve_widget("nonexistent_id_xyz") is None

    def test_ui_event_nonexistent_widget_returns_handled_false(self):
        """Verify ui_event returns handled: False with error for nonexistent widgets."""

        async def _run():
            ws = ActorWorkspace()
            resp = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "ghost_widget_123",
                    "action": "#selected",
                }
            )
            assert resp["type"] == "ui_event_result"
            assert resp["handled"] is False
            assert "could not be resolved" in resp["error"]

        asyncio.run(_run())

    def test_inspect_object_actor_class_method_introspection(self):
        """Verify inspect_object returns instance_methods, class_methods, and combined methods."""

        async def _run():
            ws = ActorWorkspace()
            resp = await ws.handle_message({"type": "inspect_object", "key": "Button"})
            assert resp["type"] == "object_details"
            assert resp["found"] is True
            fields = resp["fields"]
            assert "methods" in fields
            assert "instance_methods" in fields
            assert "class_methods" in fields
            assert "new" in fields["methods"]

        asyncio.run(_run())


# =============================================================================
# 3. Collections & Set Class Hardening Tests
# =============================================================================


class TestCollectionsAndSetHardening:
    """Adversarial tests for ActorCollection safe protocols and ActorSet."""

    def test_actor_collection_safe_len_and_size(self):
        """Verify that ActorCollection direct instantiation supports len() and size() safely."""
        coll = ActorCollection()
        assert coll.size() == 0
        assert len(coll) == 0
        assert coll.isEmpty() is True

    def test_actor_set_basic_operations(self):
        """Verify Set uniqueness, add, remove, and membership testing."""
        s = ActorSet()
        assert s.size() == 0
        assert len(s) == 0
        assert s.isEmpty() is True

        # Add elements
        s.add("apple")
        s.add("banana")
        s.add("cherry")
        assert s.size() == 3

        # Duplicate addition has no effect
        s.add("apple")
        assert s.size() == 3

        # Membership testing
        assert s.includes("banana") is True
        assert "banana" in s
        assert s.in_("banana") is True
        assert s.includes("orange") is False
        assert "orange" not in s

        # Remove element
        removed = s.remove("banana")
        assert removed == "banana"
        assert s.size() == 2
        assert "banana" not in s

        # Remove non-existent element returns safely
        assert s.remove("nonexistent") == "nonexistent"
        assert s.size() == 2

    def test_actor_set_iteration_and_copy(self):
        """Verify Set iteration, do() block execution, and copy."""
        s = ActorSet([1, 2, 3, 4, 5])
        assert s.size() == 5

        # Iteration
        items = list(s)
        assert items == [1, 2, 3, 4, 5]

        # Block execution via do()
        collected = []
        s.do(lambda x: collected.append(x * 2))
        assert collected == [2, 4, 6, 8, 10]

        # Copy is independent
        s2 = s.copy()
        s2.add(99)
        assert s.includes(99) is False
        assert s2.includes(99) is True

    def test_actor_set_registered_in_actor_dictionary(self):
        """Verify Set is registered in Actor dictionary and VM global scope."""
        vm = ActorVM()
        actor_dict = vm.global_vars["Actor"]
        assert "Set" in actor_dict or actor_dict.get("Set") is not None
        assert actor_dict["Set"] is ActorSet
        assert actor_dict[Symbol("Set")] is ActorSet

    def test_actor_set_execution_in_vm(self):
        """Verify Set instantiation and operations directly in Actor bytecode."""
        vm = ActorVM()
        src = """
        Def testSet(self | s) {
            s := new(Set, 10);
            add(s, 10);
            add(s, 20);
            add(s, 10);
            ^size(s);
        }!!
        """
        vm.eval_chunk(src)
        res = vm.eval_expression("testSet(nil)!!")
        assert res == 2


# =============================================================================
# 4. Virtual Machine Hardening Tests
# =============================================================================


class TestVMHardening:
    """Adversarial tests for VM call stack, non-local return, and field access."""

    def test_non_local_return_caller_resumption(self):
        """Verify non-local return allows caller activation records to resume."""
        vm = ActorVM()
        src = """
        Def helperWithBlock(self | blk) {
            blk := { ^42; };
            eval(blk);
            ^999;
        }!!

        Def callerMethod(self | res) {
            res := helperWithBlock(self);
            ^res + 10;
        }!!
        """
        vm.eval_chunk(src)
        # helperWithBlock evaluates { ^42; }, which unwinds helperWithBlock
        # and returns 42 to callerMethod. callerMethod resumes and returns 42 + 10 = 52.
        res = vm.eval_expression("callerMethod(nil)!!")
        assert res == 52

    def test_call_stack_cleanup_on_unhandled_exception(self):
        """Verify zombie activation frames are cleaned up when an exception occurs."""
        vm = ActorVM()
        src = """
        Def crashChild(self) {
            ^10 / 0;
        }!!

        Def crashParent(self) {
            ^crashChild(self);
        }!!
        """
        vm.eval_chunk(src)
        with pytest.raises(ZeroDivisionError):
            vm.eval_expression("crashParent(nil)!!")

        # After unhandled exception, callee frames must be cleaned up
        assert len(vm.call_stack) <= 1

    def test_recursion_guard_raises_stack_overflow(self):
        """Verify deep recursion hits max_stack_depth limit rather than hanging."""
        vm = ActorVM()
        assert hasattr(vm, "max_stack_depth")
        assert vm.max_stack_depth == 500

        src = """
        Def infiniteRec(self, n) {
            ^infiniteRec(self, n + 1);
        }!!
        """
        vm.eval_chunk(src)
        with pytest.raises(RuntimeError) as exc_info:
            vm.eval_expression("infiniteRec(nil, 0)!!")
        assert "StackOverflowError" in str(exc_info.value)

    def test_field_access_dot_syntax_and_stack_neutrality(self):
        """Verify dot instance variable access and mutation on ActorObject."""
        vm = ActorVM()
        src = """
        inherit(Object, #Person, #(name age city), nil, nil)!!

        Def setCity(self, c) {
            self.city := c;
            ^self.city;
        }!!
        """
        vm.eval_chunk(src)
        res = vm.eval_expression('setCity(new(Person), "Paris")!!')
        assert res == "Paris"

    def test_receiver_guard_on_python_class(self):
        """Verify passing a Python class subclassing ActorObject does not crash with unbound method error."""
        vm = ActorVM()

        class CustomActorObj(ActorObject):
            pass

        vm.global_vars["CustomObj"] = CustomActorObj
        inst = vm.eval_expression("new(CustomObj)!!")
        assert isinstance(inst, CustomActorObj)


# =============================================================================
# 5. Parser & Compiler Hardening Tests
# =============================================================================


class TestParserCompilerHardening:
    """Adversarial tests for unrecognized operator rejection in compiler."""

    def test_unrecognized_binary_operator_raises_not_implemented(self):
        """Verify unrecognized binary operator raises explicit NotImplementedError."""
        vm = ActorVM()
        with pytest.raises(NotImplementedError) as exc_info:
            vm.eval_expression("3 in #(1 2 3)!!")
        assert "Unsupported binary operator" in str(exc_info.value)

    def test_supported_binary_operators_still_work(self):
        """Verify standard arithmetic and comparison operators compile and execute accurately."""
        vm = ActorVM()
        assert vm.eval_expression("10 + 20!!") == 30
        assert vm.eval_expression("50 - 15!!") == 35
        assert vm.eval_expression("6 * 7!!") == 42
        assert vm.eval_expression("100 / 4!!") == 25
        assert vm.eval_expression("17 % 5!!") == 2
        assert vm.eval_expression("10 < 20!!") in (True, 1)
        assert vm.eval_expression("20 > 10!!") in (True, 1)
        assert vm.eval_expression("15 == 15!!") in (True, 1)
        assert vm.eval_expression("15 != 20!!") in (True, 1)
