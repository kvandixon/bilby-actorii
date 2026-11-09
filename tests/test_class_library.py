# -*- coding: utf-8 -*-
"""Milestone M4 Comprehensive Test Suite: 1993 Whitewater Actor Standard Class Library.

Exhaustive tests covering:
1. Object root protocol & Symbol: isKindOf, asString, print, clone, error, init, ==, !=, Symbol.
2. Collection hierarchy: Collection, IndexedCollection, Array, OrderedCollection, List.
   - 0-based indexing, fill, copyFrom, indexOf, find, reverse, asArray, asOrderedCollection.
   - Higher-order methods: do, collect, select, reject, keysDo.
   - Dynamic growth, push/pop, first/last, addFirst, removeFirst, removeLast, remove, removeAt.
   - Python list protocol compatibility (len, [], append, pop, slices).
3. KeyedCollection & Dictionary:
   - Key normalization: #Symbol vs str transparent access, single canonical key.
   - at, atPut, put, add, includesKey, keys, values, removeKey, size, isEmpty, clear.
   - Authentic 1993 Whitewater iteration: do(block) iterates values, keysDo iterates keys, assocsDo iterates [k,v].
   - ActorDictionary live synchronization with VM globals and hierarchy.
4. Interval:
   - over, overBy, start, stop, step, first, last, authentic 1993 size formula, in_interval, do, collect, asArray, reverse.
5. Geometry (Point, Point3D, Rect):
   - Coordinate accessors, vector arithmetic (+, -, *, //), midpoint calculation.
   - Authentic 1993 Whitewater 2D isometric projection: asPoint() -> Point(x, y - (z * 9) // 8).
   - Rect origin, corner, coordinates, containsPoint, intersects, union.
6. GUI hierarchy (WindowsObject, Window, Dialog, Control, Button, TextField, Checkbox):
   - Inheritance, parent/child tree, modal lifecycle (run_modal, end, IDOK, IDCANCEL).
   - Action event binding, state toggling, HTML5 rendering (to_html / draw).
7. VM Integration:
   - Hierarchy and globals registration, CALL_METHOD, GET_INDEX, SET_INDEX.
   - dispatch_action for GUI event routing.
   - Higher-order block closures in VM.
"""

from __future__ import annotations

import io
import pytest

from actorii.core import ActorObject, ActorClass, ActorInheritanceHierarchy, Symbol
from actorii.classes import (
    register_classes,
    is_kind_of,
    ActorCollection,
    ActorIndexedCollection,
    ActorArray,
    ActorOrderedCollection,
    ActorList,
    Collection,
    IndexedCollection,
    Array,
    OrderedCollection,
    ListClass,
    KeyedCollection,
    Dictionary,
    ActorDictionary,
    Interval,
    Point,
    Point3D,
    Rect,
    WindowsObject,
    Window,
    Dialog,
    Control,
    Button,
    TextField,
    Checkbox,
)
from actorii.vm import ActorVM, BlockClosure

# ==============================================================================
# 1. Object Root Protocol & Symbol
# ==============================================================================


class TestActorObjectProtocol:
    """Tests the root ActorObject protocol and Symbol."""

    def test_object_creation_and_attributes(self):
        obj = ActorObject(obj_id="test_obj_1")
        assert obj.obj_id == "test_obj_1"
        assert obj.class_name == "ActorObject"
        assert repr(obj).startswith("<ActorObject")

    def test_symbol_creation_and_equality(self):
        s1 = Symbol("mySymbol")
        s2 = Symbol("#mySymbol")
        s3 = Symbol("mySymbol")
        s4 = Symbol("otherSymbol")

        assert str(s1) == "mySymbol"
        assert repr(s1) == "#mySymbol"
        assert repr(s2) == "#mySymbol"
        assert s1.name == "mySymbol"
        assert s1 == s2
        assert s1 == s3
        assert s1 != s4
        # Symbol compares equal to string and #string
        assert s1 == "mySymbol"
        assert s1 == "#mySymbol"
        assert hash(s1) == hash(s2)
        # Usable in set / dict
        test_set = {s1, s4}
        assert s2 in test_set
        assert Symbol("otherSymbol") in test_set

    def test_is_kind_of_hierarchy(self):
        """Verify is_kind_of across standard classes and inheritance chains."""
        pt = Point(10, 20)
        p3d = Point3D(10, 20, 30)
        lst = ActorList([1, 2, 3])
        arr = ActorArray(4)
        oc = ActorOrderedCollection([10, 20])
        d = Dictionary()
        iv = Interval(1, 10, 2)
        rect = Rect(Point(0, 0), Point(10, 10))
        win = Window("Main")
        dlg = Dialog("Modal")
        btn = Button("OK")
        tf = TextField("Input")
        chk = Checkbox("Option")

        # Point & Point3D
        assert is_kind_of(pt, "Point")
        assert is_kind_of(pt, "Object")
        assert not is_kind_of(pt, "Point3D")
        assert not is_kind_of(pt, "Collection")
        assert is_kind_of(p3d, "Point3D")
        assert is_kind_of(p3d, "Point")
        assert is_kind_of(p3d, "Object")
        assert not is_kind_of(p3d, "Rect")

        # Collection hierarchy
        assert is_kind_of(arr, "Array")
        assert is_kind_of(arr, "IndexedCollection")
        assert is_kind_of(arr, "Collection")
        assert is_kind_of(arr, "Object")
        assert not is_kind_of(arr, "OrderedCollection")

        assert is_kind_of(oc, "OrderedCollection")
        assert is_kind_of(oc, "IndexedCollection")
        assert is_kind_of(oc, "Collection")
        assert is_kind_of(oc, "Object")

        assert is_kind_of(lst, "List")
        assert is_kind_of(lst, "OrderedCollection")
        assert is_kind_of(lst, "IndexedCollection")
        assert is_kind_of(lst, "Collection")
        assert is_kind_of(lst, "Object")

        assert is_kind_of(d, "Dictionary")
        assert is_kind_of(d, "KeyedCollection")
        assert is_kind_of(d, "Collection")
        assert is_kind_of(d, "Object")
        assert not is_kind_of(d, "IndexedCollection")

        assert is_kind_of(iv, "Interval")
        assert is_kind_of(iv, "Collection")
        assert is_kind_of(iv, "Object")
        assert not is_kind_of(iv, "Dictionary")

        # GUI hierarchy
        assert is_kind_of(win, "Window")
        assert is_kind_of(win, "WindowsObject")
        assert is_kind_of(win, "Object")

        assert is_kind_of(dlg, "Dialog")
        assert is_kind_of(dlg, "Window")
        assert is_kind_of(dlg, "WindowsObject")
        assert is_kind_of(dlg, "Object")

        assert is_kind_of(btn, "Button")
        assert is_kind_of(btn, "Control")
        assert is_kind_of(btn, "WindowsObject")
        assert is_kind_of(btn, "Object")

        assert is_kind_of(tf, "TextField")
        assert is_kind_of(tf, "Control")
        assert is_kind_of(tf, "WindowsObject")

        assert is_kind_of(chk, "Checkbox")
        assert is_kind_of(chk, "Button")
        assert is_kind_of(chk, "Control")
        assert is_kind_of(chk, "WindowsObject")
        assert not is_kind_of(chk, "TextField")

    def test_as_string_and_print(self):
        pt = Point(12, 34)
        assert pt.asString() == "12@34"
        p3d = Point3D(1, 2, 3)
        assert p3d.asString() == "Point3D(1, 2, 3)"
        iv = Interval(1, 5, 2)
        assert iv.asString() == "1..5 by 2"

        # Test print method streaming
        buf = io.StringIO()
        pt.print(buf)
        assert buf.getvalue() == "12@34"

    def test_clone_and_copy(self):
        pt = Point(10, 20)
        c = pt.clone()
        assert c is not pt
        assert c.x == 10 and c.y == 20
        c.setX(99)
        assert pt.x == 10  # original unmodified

    def test_error_and_init(self):
        obj = ActorObject()
        assert obj.init() is obj
        with pytest.raises(RuntimeError) as exc_info:
            obj.error("Fatal actor failure")
        assert "Fatal actor failure" in str(exc_info.value)


# ==============================================================================
# 2. Collection Hierarchy (IndexedCollection, Array, OrderedCollection, List)
# ==============================================================================


class TestCollectionHierarchy:
    """Tests Collection, IndexedCollection, Array, OrderedCollection, List."""

    def test_indexed_collection_basics(self):
        col = ActorIndexedCollection([10, 20, 30, 40])
        assert col.size() == 4
        assert not col.isEmpty()
        assert col.at(0) == 10
        assert col.at(3) == 40
        with pytest.raises(IndexError):
            col.at(4)
        with pytest.raises(IndexError):
            col.at(-1)

        # atPut and put
        col.atPut(1, 99)
        assert col.at(1) == 99
        col.put(77, 1)
        assert col.at(1) == 77

        # fill
        col.fill(0, 2)
        assert list(col[:2]) == [0, 0]

        # copyFrom
        sub = col.copyFrom(1, 3)
        assert isinstance(sub, ActorIndexedCollection)
        assert list(sub) == [col[1], col[2]]

        # indexOf / find
        col2 = ActorIndexedCollection(["apple", "banana", "cherry"])
        assert col2.indexOf("banana") == 1
        assert col2.indexOf("grape") == -1
        assert col2.find("cherry") == 2

        # reverse
        rev = col2.reverse()
        assert list(rev) == ["cherry", "banana", "apple"]

    def test_higher_order_methods(self):
        col = ActorIndexedCollection([1, 2, 3, 4, 5])

        # collect (map)
        doubled = col.collect(lambda x: x * 2)
        assert list(doubled) == [2, 4, 6, 8, 10]

        # select (filter True)
        evens = col.select(lambda x: x % 2 == 0)
        assert list(evens) == [2, 4]

        # reject (filter False)
        odds = col.reject(lambda x: x % 2 == 0)
        assert list(odds) == [1, 3, 5]

        # do (each)
        accum = []
        col.do(lambda x: accum.append(x * 10))
        assert accum == [10, 20, 30, 40, 50]

        # keysDo (0-based indices)
        indices = []
        col.keysDo(lambda idx: indices.append(idx))
        assert indices == [0, 1, 2, 3, 4]

    def test_array_capacity_and_bounds(self):
        arr = ActorArray(5)
        assert arr.size() == 5
        assert len(arr) == 5
        assert list(arr) == [None, None, None, None, None]

        arr.atPut(0, "first")
        arr.atPut(4, "last")
        assert arr.at(0) == "first"
        assert arr.at(4) == "last"

        with pytest.raises(IndexError):
            arr.at(5)

    def test_ordered_collection_and_list_operations(self):
        oc = ActorOrderedCollection()
        assert oc.isEmpty()
        assert oc.size() == 0

        # add / push
        oc.add(10)
        oc.add(20)
        oc.push(30)
        assert list(oc) == [10, 20, 30]
        assert oc.first() == 10
        assert oc.last() == 30

        # pop (LIFO)
        popped = oc.pop()
        assert popped == 30
        assert oc.last() == 20

        # addFirst
        oc.addFirst(5)
        assert oc.first() == 5
        assert list(oc) == [5, 10, 20]

        # removeFirst / removeLast
        rf = oc.removeFirst()
        assert rf == 5
        rl = oc.removeLast()
        assert rl == 20
        assert list(oc) == [10]

        # Empty error cases
        empty_oc = ActorOrderedCollection()
        with pytest.raises(IndexError):
            empty_oc.first()
        with pytest.raises(IndexError):
            empty_oc.last()
        with pytest.raises(IndexError):
            empty_oc.removeFirst()
        with pytest.raises(IndexError):
            empty_oc.removeLast()

        # ActorList inherits full ordered collection protocol
        lst = ActorList([100, 200, 300])
        assert is_kind_of(lst, "List")
        assert is_kind_of(lst, "OrderedCollection")
        lst.insert(1, 150)
        assert list(lst) == [100, 150, 200, 300]
        lst.remove(200)
        assert list(lst) == [100, 150, 300]
        lst.removeAt(0)
        assert list(lst) == [150, 300]

    def test_python_list_compatibility(self):
        """Existing tests expect ActorIndexedCollection to behave 100% as Python list."""
        col = ActorIndexedCollection([1, 2, 3])
        # Direct list operations
        assert len(col) == 3
        assert col[0] == 1
        assert col[-1] == 3
        assert col[1:] == [2, 3]
        col.append(4)
        assert col == [1, 2, 3, 4]
        assert 3 in col
        assert list(reversed(col)) == [4, 3, 2, 1]


# ==============================================================================
# 3. KeyedCollection & Dictionary
# ==============================================================================


class TestDictionary:
    """Tests KeyedCollection, Dictionary, and ActorDictionary."""

    def test_dictionary_basic_operations(self):
        d = Dictionary()
        assert d.isEmpty()
        assert d.size() == 0

        d.atPut("a", 1)
        d.atPut("b", 2)
        d.put(3, "c")
        d.add("d", 4)

        assert d.size() == 4
        assert not d.isEmpty()
        assert d.at("a") == 1
        assert d.at("b") == 2
        assert d.at("c") == 3
        assert d.at("d") == 4
        assert d["a"] == 1
        assert d["d"] == 4

        assert d.includesKey("a")
        assert "b" in d
        assert not d.includesKey("z")

        # removeKey
        rem = d.removeKey("c")
        assert rem == 3
        assert not d.includesKey("c")
        assert d.size() == 3

        d.clear()
        assert d.size() == 0
        assert d.isEmpty()

    def test_symbol_string_key_normalization(self):
        """Symbol and string keys must resolve transparently to the same entry."""
        d = Dictionary()
        d.atPut(Symbol("count"), 42)

        # Lookup with string, symbol, #string
        assert d.at("count") == 42
        assert d.at("#count") == 42
        assert d.at(Symbol("count")) == 42
        assert d["count"] == 42
        assert d[Symbol("count")] == 42
        assert d.includesKey("count")
        assert d.includesKey("#count")
        assert d.includesKey(Symbol("count"))

        # Modifying with string key overwrites symbol entry without duplicating key
        d.atPut("count", 99)
        assert d.at(Symbol("count")) == 99
        assert d.size() == 1
        assert len(d.keys()) == 1

        # Remove with #count
        assert d.removeKey("#count") == 99
        assert d.size() == 0

    def test_1993_authentic_iteration_semantics(self):
        """In 1993 Whitewater Actor (DICTIONA.CLS:107), do iterates over VALUES!"""
        d = Dictionary()
        d.atPut("first", 10)
        d.atPut("second", 20)
        d.atPut("third", 30)

        # do(block) iterates values
        observed_values = []
        d.do(lambda val: observed_values.append(val))
        assert sorted(observed_values) == [10, 20, 30]

        # keysDo(block) iterates keys
        observed_keys = []
        d.keysDo(lambda k: observed_keys.append(str(k)))
        assert sorted(observed_keys) == ["first", "second", "third"]

        # assocsDo(block) iterates [key, value] pairs
        observed_assocs = []
        d.assocsDo(lambda assoc: observed_assocs.append(assoc))
        assert len(observed_assocs) == 3
        assert sorted(observed_assocs, key=lambda a: a[1]) == [
            ["first", 10],
            ["second", 20],
            ["third", 30],
        ]

    def test_actor_dictionary_live_sync(self):
        """ActorDictionary provides live sync with globals and hierarchy."""
        hierarchy = ActorInheritanceHierarchy()
        globals_dict = {}
        ad = ActorDictionary(hierarchy=hierarchy, global_vars=globals_dict)

        # Insert a new class into ActorDictionary
        test_cls = ActorClass("CustomWidget", bases=("WindowsObject",))
        ad["CustomWidget"] = test_cls

        assert "CustomWidget" in globals_dict
        assert globals_dict["CustomWidget"] is test_cls
        assert hierarchy.get("CustomWidget") is test_cls

        # Transparent symbol key access
        assert ad[Symbol("CustomWidget")] is test_cls
        assert ad["#CustomWidget"] is test_cls
        assert ad.includesKey("#CustomWidget")


# ==============================================================================
# 4. Interval
# ==============================================================================


class TestInterval:
    """Tests the Interval collection and authentic 1993 size formula."""

    def test_interval_creation_and_properties(self):
        iv1 = Interval(1, 10, 2)
        assert iv1.start == 1
        assert iv1.stop == 10
        assert iv1.step == 2
        assert iv1.first() == 1
        assert iv1.last() == 9

        iv2 = Interval.over(0, 5)
        assert iv2.start == 0 and iv2.stop == 5 and iv2.step == 1
        assert iv2.first() == 0 and iv2.last() == 4

        with pytest.raises(ValueError):
            Interval(0, 10, 0)

    def test_1993_size_formula(self):
        # Forward intervals
        assert Interval(0, 5, 1).size() == 5  # 0, 1, 2, 3, 4
        assert Interval(1, 10, 2).size() == 5  # 1, 3, 5, 7, 9
        assert Interval(0, 10, 3).size() == 4  # 0, 3, 6, 9
        assert Interval(5, 5, 1).size() == 0
        assert Interval(10, 5, 1).size() == 0

        # Reverse intervals
        assert Interval(10, 0, -2).size() == 5  # 10, 8, 6, 4, 2
        assert Interval(5, 0, -1).size() == 5  # 5, 4, 3, 2, 1
        assert Interval(0, 5, -1).size() == 0

        # Empty interval first/last raises IndexError
        empty = Interval(5, 5, 1)
        with pytest.raises(IndexError):
            empty.first()
        with pytest.raises(IndexError):
            empty.last()

    def test_interval_indexing_and_containment(self):
        iv = Interval(10, 30, 5)  # 10, 15, 20, 25 (size 4)
        assert iv.size() == 4
        assert iv.at(0) == 10
        assert iv.at(1) == 15
        assert iv.at(2) == 20
        assert iv.at(3) == 25
        with pytest.raises(IndexError):
            iv.at(4)

        # in_interval / __contains__
        assert iv.in_interval(10)
        assert iv.in_interval(15)
        assert iv.in_interval(25)
        assert not iv.in_interval(30)  # stop is exclusive
        assert not iv.in_interval(12)  # off-step
        assert 20 in iv
        assert 99 not in iv

    def test_interval_higher_order_methods(self):
        iv = Interval(1, 5, 1)  # 1, 2, 3, 4
        mapped = iv.collect(lambda x: x * 10)
        assert list(mapped) == [10, 20, 30, 40]

        arr = iv.asArray()
        assert isinstance(arr, ActorArray)
        assert list(arr) == [1, 2, 3, 4]

        rev = iv.reverse()
        assert list(rev) == [4, 3, 2, 1]


# ==============================================================================
# 5. Geometry (Point, Point3D, Rect)
# ==============================================================================


class TestGeometry:
    """Tests 2D Point, 3D Point3D, and Rect."""

    def test_point_accessors_and_arithmetic(self):
        p1 = Point(10, 20)
        assert p1.x == 10 and p1.y == 20
        p1.setX(15)
        p1.setY(25)
        assert p1.x == 15 and p1.y == 25
        p1.set(30, 40)
        assert p1.x == 30 and p1.y == 40

        # Vector addition / subtraction
        p2 = Point(5, 10)
        p_add = p1 + p2
        assert p_add.x == 35 and p_add.y == 50
        p_sub = p1 - p2
        assert p_sub.x == 25 and p_sub.y == 30

        # Scalar operations
        p_mul = p2 * 3
        assert p_mul.x == 15 and p_mul.y == 30
        p_div = p_mul // 3
        assert p_div.x == 5 and p_div.y == 10

        # Midpoint
        mid = Point(0, 0).middlePoint(Point(10, 20))
        assert mid.x == 5 and mid.y == 10

        # Projection (identity for 2D Point)
        assert p1.asPoint() is p1

    def test_point3d_and_1993_isometric_projection(self):
        """Authentic 1993 projection: point(x, y - (z * 9) / 8).

        Source: historical_actor_41_source/actcls/POINT3D.CLS:28
        """
        p3d = Point3D(100, 200, 80)
        assert p3d.x == 100 and p3d.y == 200 and p3d.z == 80

        # asPoint() projection formula: y - (z * 9) // 8
        # 200 - (80 * 9) // 8 = 200 - 90 = 110
        proj = p3d.asPoint()
        assert isinstance(proj, Point)
        assert proj.x == 100
        assert proj.y == 110

        # Benchmark reference check: Point3D(256, 20, 0).asPoint() == Point(256, 20)
        bmark = Point3D(256, 20, 0).asPoint()
        assert bmark.x == 256 and bmark.y == 20

    def test_point3d_arithmetic(self):
        p1 = Point3D(10, 20, 30)
        p2 = Point3D(5, 5, 5)

        p_add = p1 + p2
        assert p_add.x == 15 and p_add.y == 25 and p_add.z == 35

        p_sub = p1 - p2
        assert p_sub.x == 5 and p_sub.y == 15 and p_sub.z == 25

        p_scal = p1 * 2
        assert p_scal.x == 20 and p_scal.y == 40 and p_scal.z == 60

        p_div = p_scal // 2
        assert p_div.x == 10 and p_div.y == 20 and p_div.z == 30

        p_add_pt = p1 + Point(2, 3)
        assert p_add_pt.x == 12 and p_add_pt.y == 23 and p_add_pt.z == 30

    def test_rect_geometry(self):
        r1 = Rect(10, 20, 110, 120)
        assert r1.left == 10
        assert r1.top == 20
        assert r1.right == 110
        assert r1.bottom == 120
        assert r1.width == 100
        assert r1.height == 100

        # Contains point
        assert r1.containsPoint(Point(50, 50))
        assert r1.containsPoint(Point(10, 20))
        assert not r1.containsPoint(Point(5, 50))
        assert not r1.containsPoint(Point(150, 50))

        # Intersects & Union
        r2 = Rect(50, 50, 150, 150)
        assert r1.intersects(r2)
        r3 = Rect(200, 200, 300, 300)
        assert not r1.intersects(r3)

        u = r1.union(r2)
        assert u.left == 10 and u.top == 20 and u.right == 150 and u.bottom == 150


# ==============================================================================
# 6. GUI Hierarchy (WindowsObject, Window, Dialog, Control, Button, TextField, Checkbox)
# ==============================================================================


class TestGUIHierarchy:
    """Tests WindowsObject, Window, Dialog, Control, Button, TextField, Checkbox."""

    def test_window_and_child_management(self):
        win = Window("Test App", rect=Rect(0, 0, 640, 480))
        btn = Button(
            "Click", left=20, top=30, width=80, height=25, identity=Symbol("myButton")
        )
        win.addChild(btn)

        assert btn.parent is win
        assert btn in win.children
        assert win.find_child("myButton") is btn
        assert win.find_child(Symbol("myButton")) is btn

        win.removeChild(btn)
        assert btn.parent is None
        assert btn not in win.children

    def test_dialog_modal_lifecycle(self):
        dlg = Dialog("Confirm Action", modal=True)
        assert not dlg.is_modal
        assert dlg.result_code is None

        dlg.run_modal()
        assert dlg.is_modal

        dlg.end(Dialog.IDOK)
        assert not dlg.is_modal
        assert dlg.result_code == Dialog.IDOK

    def test_button_click_and_dialog_auto_close(self):
        dlg = Dialog("Prompt")
        ok_btn = Button("OK", id=Dialog.IDOK)
        dlg.addChild(ok_btn)
        dlg.run_modal()

        ok_btn.on_click()
        assert not dlg.is_modal
        assert dlg.result_code == Dialog.IDOK

    def test_button_callback(self):
        clicked = []
        btn = Button("Test", callback=lambda b: clicked.append(b.label))
        btn.on_click()
        assert clicked == ["Test"]

    def test_textfield_state(self):
        tf = TextField("Default Value", left=10, top=10, width=200, height=30)
        assert tf.text == "Default Value"
        assert tf.get_value() == "Default Value"

        tf.setText("Updated Content")
        assert tf.text == "Updated Content"
        assert tf.get_value() == "Updated Content"

        tf.clear()
        assert tf.text == ""

    def test_checkbox_state_and_toggling(self):
        cb = Checkbox("Enable Feature", checked=False)
        assert not cb.value()
        assert cb.data == Symbol("unChecked")

        cb.toggle()
        assert cb.value()
        assert cb.data == Symbol("checked")

        cb.unCheck()
        assert not cb.value()

        cb.check()
        assert cb.value()

        cb.setValue(False)
        assert not cb.value()

    def test_html5_rendering(self):
        win = Window("App Window", rect=Rect(10, 10, 400, 300))
        btn = Button("Submit", left=50, top=50, width=100, height=30)
        tf = TextField("Enter name", left=50, top=100, width=150, height=30)
        chk = Checkbox("Agree", checked=True, left=50, top=150)
        win.addChild(btn)
        win.addChild(tf)
        win.addChild(chk)

        html = win.to_html()
        assert 'class="actor-window"' in html
        assert "App Window" in html
        assert 'class="actor-button"' in html
        assert "Submit" in html
        assert 'class="actor-textfield"' in html
        assert "Enter name" in html
        assert 'class="actor-checkbox"' in html
        assert "checked" in html


# ==============================================================================
# 7. VM Integration & Action Dispatch
# ==============================================================================


class TestVMClassLibraryIntegration:
    """Tests VM class table registration, execution, and action event dispatch."""

    def test_vm_registers_all_standard_classes(self):
        vm = ActorVM()
        # Verify hierarchy entries
        expected_classes = [
            "Object",
            "Collection",
            "IndexedCollection",
            "Array",
            "OrderedCollection",
            "List",
            "KeyedCollection",
            "Dictionary",
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
        ]
        for cname in expected_classes:
            cls = vm.hierarchy.get(cname)
            assert cls is not None, f"Class {cname} missing in vm.hierarchy"
            assert cname in vm.global_vars, f"Class {cname} missing in vm.global_vars"
            assert (
                cname in vm.global_vars["Actor"]
            ), f"Class {cname} missing in vm.global_vars['Actor']"

    def test_vm_dispatch_message_on_standard_classes(self):
        vm = ActorVM()

        # Point methods via dispatch_message
        pt = Point(10, 20)
        res_x = vm.dispatch_message(pt, "x", [])
        assert res_x == 10

        p_sum = vm.dispatch_message(pt, "+", [Point(5, 5)])
        assert isinstance(p_sum, Point)
        assert p_sum.x == 15 and p_sum.y == 25

        # Point3D projection via dispatch_message
        p3d = Point3D(100, 200, 80)
        proj = vm.dispatch_message(p3d, "asPoint", [])
        assert isinstance(proj, Point)
        assert proj.x == 100 and proj.y == 110

        # Interval size via dispatch_message
        iv = Interval(1, 10, 2)
        sz = vm.dispatch_message(iv, "size", [])
        assert sz == 5

        # Dictionary operations via dispatch_message
        d = Dictionary()
        vm.dispatch_message(d, "atPut", [Symbol("score"), 100])
        val = vm.dispatch_message(d, "at", ["#score"])
        assert val == 100

    def test_vm_dispatch_action_event_routing(self):
        """Simulate UI action dispatch from button into registered window action handler."""
        vm = ActorVM()

        win = Window("Dashboard")
        handled_events = []

        # Define custom action handler on window
        def handle_submit(self_win, msg=None):
            handled_events.append(msg)
            return "SUCCESS"

        win.set_method("handleSubmit", handle_submit)
        win.add_action_handler("submitBtn", "selected", "handleSubmit")

        btn = Button("Submit", identity=Symbol("submitBtn"))
        win.addChild(btn)

        # Dispatch action through VM
        result = vm.dispatch_action(btn, "selected")
        assert result == "SUCCESS"
        assert len(handled_events) == 1
        assert handled_events[0]["source"] is btn

    def test_vm_collection_higher_order_closures(self):
        """Higher-order collection methods with VM block evaluation."""
        col = ActorIndexedCollection([1, 2, 3, 4])
        # Test with Python callable inside VM context
        vm = ActorVM()
        res = col.collect(lambda x: x * 3)
        assert list(res) == [3, 6, 9, 12]

    def test_vm_compiled_bytecode_collections(self):
        """Execute compiled Actor method utilizing OrderedCollection."""
        vm = ActorVM()
        source = """
        Def testOrdered(self | oc, f, l) {
            oc := new(OrderedCollection);
            add(oc, 10);
            add(oc, 20);
            add(oc, 30);
            f := first(oc);
            l := last(oc);
            ^f + l;
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "testOrdered")
        assert res == 40

    def test_vm_compiled_bytecode_dictionary(self):
        """Execute compiled Actor method utilizing Dictionary and Symbols."""
        vm = ActorVM()
        source = """
        Def testDict(self | d) {
            d := new(Dictionary);
            atPut(d, #item, 100);
            atPut(d, #tax, 15);
            ^at(d, #item) + at(d, #tax);
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "testDict")
        assert res == 115

    def test_vm_compiled_bytecode_point_projection(self):
        """Execute compiled Actor method projecting Point3D to Point."""
        vm = ActorVM()
        source = """
        Def testProject(self | p3d, p2d) {
            p3d := new(Point3D, 100, 200, 80);
            p2d := asPoint(p3d);
            ^y(p2d);
        }!!
        """
        vm.eval_chunk(source)
        res = vm.dispatch_message(None, "testProject")
        # 200 - (80 * 9) // 8 = 110
        assert res == 110
