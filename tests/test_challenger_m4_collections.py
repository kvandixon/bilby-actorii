# -*- coding: utf-8 -*-
"""Milestone M4 Challenger 1 Adversarial Empirical Stress Test Suite.

Adversarial testing targeting:
1. Collections (ActorArray, ActorOrderedCollection, ActorList, ActorIndexedCollection):
   - Huge arrays (100k items) and boundaries (0, size-1, size, negative indices)
   - copyFrom slices (empty slices, identity slices, full slices, out of bound ranges, inverted ranges)
   - fill on empty, full, and clamped counts
   - indexOf (identity) vs find (equality)
   - OrderedCollection / List stack (LIFO) and queue (FIFO) under 10k items
   - Boundary exceptions on empty collections (first, last, pop, removeFirst, removeLast, removeAt, hasElements)
   - insert heuristic compatibility (Actor vs Python)
   - higher-order methods: collect, select, reject, do, keysDo
2. Dictionary & ActorDictionary:
   - Symbol vs string key transparent normalization (#symbol, symbol, str, #str)
   - Canonical key deduplication (avoiding duplicate entries on symbol/string collisions)
   - 1993 Whitewater Actor iteration semantics: do (values), keysDo (keys), assocsDo ([key, value])
   - High volume (5,000 entries) addition, verification, and partial deletion
   - Safe removal during iteration
   - Missing key lookups (__getitem__, at, get, removeKey, includesKey)
   - Non-string keys (integers, tuples)
   - ActorDictionary live synchronization with VM globals and hierarchy
3. Interval:
   - Zero step exception
   - Positive steps (exact, non-divisible, step > range)
   - Negative steps (exact, non-divisible, step < -range)
   - Empty intervals (start >= stop for positive step, start <= stop for negative step)
   - at and [] index access matching iterator output for all valid indices
   - out of bounds indexing (negative, size, on empty)
   - first and last on empty and non-empty
   - in_interval containment with positive and negative steps
   - reverse produces inverted progression; double reverse restores original
   - Huge interval O(1) performance (1 billion range)
   - collect returning ActorArray and species == 'Array'
"""

from __future__ import annotations

import pytest
from actorii.core import ActorInheritanceHierarchy, ActorObject, Symbol
from actorii.classes import (
    ActorArray,
    ActorCollection,
    ActorDictionary,
    ActorIndexedCollection,
    ActorList,
    ActorOrderedCollection,
    Dictionary,
    Interval,
    KeyedCollection,
    register_classes,
)
from actorii.vm import ActorVM, BlockClosure

# ==============================================================================
# 1. ActorArray & IndexedCollection Adversarial Stress Tests
# ==============================================================================


class TestArrayAndIndexedStress:
    """Stress tests and boundary checks for ActorArray and ActorIndexedCollection."""

    def test_large_array_boundaries_and_indexing(self):
        """Stress test 100k element ActorArray: memory, boundary access, and exception safety."""
        size = 100_000
        arr = ActorArray(size)

        assert arr.size() == size
        assert len(arr) == size
        assert arr.at(0) is None
        assert arr.at(size - 1) is None

        # Boundary stores
        arr.atPut(0, "FIRST")
        arr.atPut(size - 1, "LAST")
        arr.put("MID", size // 2)

        assert arr.at(0) == "FIRST"
        assert arr[0] == "FIRST"
        assert arr.at(size - 1) == "LAST"
        assert arr[size - 1] == "LAST"
        assert arr.at(size // 2) == "MID"
        assert arr[size // 2] == "MID"

        # Out-of-bounds boundary access (both at() and atPut())
        with pytest.raises(IndexError):
            arr.at(size)
        with pytest.raises(IndexError):
            arr.at(-1)
        with pytest.raises(IndexError):
            arr.at(-size)
        with pytest.raises(IndexError):
            arr.atPut(size, "OUT")
        with pytest.raises(IndexError):
            arr.atPut(-1, "OUT")
        with pytest.raises(IndexError):
            arr.put("OUT", size)

    def test_array_fill_semantics(self):
        """Array.fill replaces all elements; empty array fill is safe."""
        # Empty array
        empty_arr = ActorArray(0)
        assert empty_arr.size() == 0
        empty_arr.fill("nothing")
        assert empty_arr.size() == 0

        # Non-empty array
        arr = ActorArray(10_000)
        arr.fill(42)
        assert arr.size() == 10_000
        assert arr.at(0) == 42
        assert arr.at(9999) == 42
        assert all(x == 42 for x in arr)

        # fill returns self
        res = arr.fill("updated")
        assert res is arr
        assert arr.at(500) == "updated"

    def test_indexed_collection_fill_with_count(self):
        """ActorIndexedCollection.fill supports optional count limit."""
        col = ActorIndexedCollection([1, 2, 3, 4, 5])
        col.fill(99, count=3)
        assert list(col) == [99, 99, 99, 4, 5]

        # Count exceeding length clamps to length without error
        col.fill(0, count=100)
        assert list(col) == [0, 0, 0, 0, 0]

        # Count 0 modifies nothing
        col.fill(77, count=0)
        assert list(col) == [0, 0, 0, 0, 0]

    def test_array_copy_from_exhaustive_boundaries(self):
        """copyFrom slices: empty slices, full slices, valid slices, invalid ranges."""
        arr = ActorArray(["a", "b", "c", "d", "e"])
        sz = arr.size()

        # Full copy
        full_copy = arr.copyFrom(0, sz)
        assert isinstance(full_copy, ActorArray)
        assert list(full_copy) == ["a", "b", "c", "d", "e"]
        assert full_copy is not arr

        # Sub slices
        sub = arr.copyFrom(1, 4)
        assert list(sub) == ["b", "c", "d"]
        assert sub.size() == 3

        # Zero-length slices at boundaries
        empty_start = arr.copyFrom(0, 0)
        assert empty_start.size() == 0
        empty_mid = arr.copyFrom(2, 2)
        assert empty_mid.size() == 0
        empty_end = arr.copyFrom(sz, sz)
        assert empty_end.size() == 0

        # Invalid bounds must raise ValueError
        with pytest.raises(ValueError):
            arr.copyFrom(-1, 3)
        with pytest.raises(ValueError):
            arr.copyFrom(3, 2)  # beg > end
        with pytest.raises(ValueError):
            arr.copyFrom(0, sz + 1)
        with pytest.raises(ValueError):
            arr.copyFrom(sz + 1, sz + 2)

    def test_indexed_collection_copy_from_boundaries(self):
        """ActorIndexedCollection.copyFrom boundary validation."""
        col = ActorIndexedCollection([10, 20, 30])
        # Slices
        assert list(col.copyFrom(0, 1)) == [10]
        assert list(col.copyFrom(1, 3)) == [20, 30]
        assert list(col.copyFrom(0, 0)) == []
        assert list(col.copyFrom(3, 3)) == []

        # Out-of-bounds raises IndexError
        with pytest.raises(IndexError):
            col.copyFrom(-1, 2)
        with pytest.raises(IndexError):
            col.copyFrom(2, 1)
        with pytest.raises(IndexError):
            col.copyFrom(0, 4)

    def test_index_of_vs_find_identity_versus_equality(self):
        """indexOf must use identity (is) while find uses equality (==)."""

        # Create two distinct objects that compare equal
        class CustomObj:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                return isinstance(other, CustomObj) and self.val == other.val

        obj1 = CustomObj(42)
        obj2 = CustomObj(42)
        assert obj1 == obj2
        assert obj1 is not obj2

        col = ActorIndexedCollection([10, obj1, 20])

        # indexOf with obj2 should fail (-1) because obj2 is NOT obj1
        assert col.indexOf(obj2) == -1
        # indexOf with obj1 should succeed (1) because obj1 is obj1
        assert col.indexOf(obj1) == 1

        # find with obj2 should succeed (1) because obj2 == obj1
        assert col.find(obj2) == 1
        assert col.find(obj1) == 1

        # Non-existent target
        assert col.indexOf("missing") == -1
        assert col.find("missing") == -1

    def test_reverse_in_place_modifications(self):
        """reverse() mutates collection in place and returns self."""
        # Empty
        empty = ActorIndexedCollection([])
        assert empty.reverse() is empty
        assert list(empty) == []

        # Single element
        single = ActorIndexedCollection([1])
        assert single.reverse() is single
        assert list(single) == [1]

        # Even length
        even = ActorIndexedCollection([1, 2, 3, 4])
        assert even.reverse() is even
        assert list(even) == [4, 3, 2, 1]

        # Odd length
        odd = ActorIndexedCollection([10, 20, 30, 40, 50])
        assert odd.reverse() is odd
        assert list(odd) == [50, 40, 30, 20, 10]


# ==============================================================================
# 2. ActorOrderedCollection & ActorList Adversarial Stress Tests
# ==============================================================================


class TestOrderedCollectionAndListStress:
    """Stress tests and boundary checks for ActorOrderedCollection and ActorList."""

    def test_10k_elements_stack_and_queue_stress(self):
        """Push 10,000 items, verify stack (LIFO) and queue (FIFO) integrity."""
        oc = ActorOrderedCollection()

        # Add 10,000 items
        for i in range(10_000):
            oc.add(i)

        assert oc.size() == 10_000
        assert oc.first() == 0
        assert oc.last() == 9999

        # Pop 5,000 items (LIFO)
        for i in range(9999, 4999, -1):
            val = oc.pop()
            assert val == i

        assert oc.size() == 5000
        assert oc.last() == 4999

        # removeFirst 2,500 items (FIFO)
        for i in range(2500):
            val = oc.removeFirst()
            assert val == i

        assert oc.size() == 2500
        assert oc.first() == 2500
        assert oc.last() == 4999

    def test_empty_collection_boundary_exceptions(self):
        """Every retrieval/removal method on an empty collection must raise IndexError."""
        empty_oc = ActorOrderedCollection()
        empty_list = ActorList()

        for c in (empty_oc, empty_list):
            assert c.isEmpty()
            assert c.size() == 0
            with pytest.raises(IndexError):
                c.hasElements()
            with pytest.raises(IndexError):
                c.first()
            with pytest.raises(IndexError):
                c.last()
            with pytest.raises(IndexError):
                c.pop()
            with pytest.raises(IndexError):
                c.removeFirst()
            with pytest.raises(IndexError):
                c.removeLast()
            with pytest.raises(IndexError):
                c.removeAt(0)
            with pytest.raises(IndexError):
                c.remove("nonexistent")
            with pytest.raises(IndexError):
                c.at(0)
            with pytest.raises(IndexError):
                c.atPut(0, "val")

    def test_add_first_and_add_last(self):
        """addFirst prepends while addLast appends."""
        oc = ActorOrderedCollection()
        oc.addLast(10)
        oc.addFirst(5)
        oc.addLast(20)
        oc.addFirst(1)

        assert list(oc) == [1, 5, 10, 20]
        assert oc.first() == 1
        assert oc.last() == 20

    def test_insert_heuristics_and_boundaries(self):
        """insert(index, item) vs insert(item, index) edge cases."""
        oc = ActorOrderedCollection(["b", "c"])

        # Insert at start
        oc.insert(0, "a")
        assert list(oc) == ["a", "b", "c"]

        # Insert at end
        oc.insert(len(oc), "d")
        assert list(oc) == ["a", "b", "c", "d"]

        # Insert in middle with Actor style (elem, idx)
        oc.insert("x", 2)
        assert list(oc) == ["a", "b", "x", "c", "d"]

        # Out-of-bounds index must raise IndexError
        with pytest.raises(IndexError):
            oc.insert(-1, "bad")
        with pytest.raises(IndexError):
            oc.insert(100, "bad")

    def test_collection_mapping_and_filtering(self):
        """collect, select, reject, asArray, asOrderedCollection."""
        oc = ActorOrderedCollection([1, 2, 3, 4, 5, 6])

        # collect maps each element
        squares = oc.collect(lambda x: x * x)
        assert isinstance(squares, ActorOrderedCollection)
        assert list(squares) == [1, 4, 9, 16, 25, 36]

        # select filters truthy
        gt3 = oc.select(lambda x: x > 3)
        assert isinstance(gt3, ActorOrderedCollection)
        assert list(gt3) == [4, 5, 6]

        # extract is an alias for select
        extracted = oc.extract(lambda x: x > 3)
        assert list(extracted) == [4, 5, 6]

        # reject filters falsy
        not_gt3 = oc.reject(lambda x: x > 3)
        assert isinstance(not_gt3, ActorOrderedCollection)
        assert list(not_gt3) == [1, 2, 3]

        # asArray produces fixed ActorArray
        arr = oc.asArray()
        assert isinstance(arr, ActorArray)
        assert list(arr) == [1, 2, 3, 4, 5, 6]
        assert arr.size() == 6

        # asOrderedCollection
        new_oc = arr.asOrderedCollection()
        assert isinstance(new_oc, ActorOrderedCollection)
        assert list(new_oc) == [1, 2, 3, 4, 5, 6]

    def test_nested_collections(self):
        """Verify deeply nested collections preserve types and structure."""
        outer = ActorOrderedCollection()
        inner_arr = ActorArray([1, 2, 3])
        inner_list = ActorList(["x", "y"])
        inner_dict = Dictionary()
        inner_dict.atPut("key", "val")
        inner_itv = Interval(0, 5, 2)

        outer.add(inner_arr)
        outer.add(inner_list)
        outer.add(inner_dict)
        outer.add(inner_itv)

        assert outer.size() == 4
        assert isinstance(outer.at(0), ActorArray)
        assert isinstance(outer.at(1), ActorList)
        assert isinstance(outer.at(2), Dictionary)
        assert isinstance(outer.at(3), Interval)


# ==============================================================================
# 3. Dictionary & ActorDictionary Adversarial Stress Tests
# ==============================================================================


class TestDictionaryAdversarialStress:
    """Stress tests and edge cases for Dictionary and ActorDictionary."""

    def test_symbol_string_key_transparent_normalization_and_dedup(self):
        """Colliding keys like 'key', '#key', Symbol('key'), Symbol('#key') must map to the same entry."""
        d = Dictionary()

        # Add initial entry with Symbol
        d.atPut(Symbol("alpha"), 100)
        assert d.size() == 1
        assert len(d.keys()) == 1

        # Lookups via all variants
        assert d.at("alpha") == 100
        assert d.at("#alpha") == 100
        assert d.at(Symbol("alpha")) == 100
        assert d.at(Symbol("#alpha")) == 100
        assert d["alpha"] == 100
        assert d["#alpha"] == 100
        assert d[Symbol("alpha")] == 100
        assert d[Symbol("#alpha")] == 100

        # Updates via string must overwrite existing entry without creating duplicates
        d.atPut("alpha", 200)
        assert d.size() == 1
        assert len(d.keys()) == 1
        assert d.at(Symbol("alpha")) == 200

        # Updates via #string
        d["#alpha"] = 300
        assert d.size() == 1
        assert len(d.keys()) == 1
        assert d["alpha"] == 300

        # Remove via any variant clears the entry completely
        rem = d.removeKey(Symbol("#alpha"))
        assert rem == 300
        assert d.size() == 0
        assert not d.includesKey("alpha")
        assert not d.includesKey(Symbol("alpha"))
        assert d.at("alpha") is None

    def test_1993_authentic_iteration_values_vs_keys(self):
        """do must iterate over values; keysDo must iterate over keys; assocsDo over [k, v]."""
        d = Dictionary()
        data = {"one": 1, "two": 2, "three": 3, "four": 4}
        for k, v in data.items():
            d.atPut(k, v)

        # do() iterates VALUES
        vals = []
        last_ret = d.do(lambda v: vals.append(v) or v)
        assert sorted(vals) == [1, 2, 3, 4]
        # Return value of do is the last evaluated value
        assert last_ret in (1, 2, 3, 4)

        # keysDo() iterates KEYS
        keys = []
        d.keysDo(lambda k: keys.append(k))
        assert sorted(keys) == ["four", "one", "three", "two"]

        # assocsDo() iterates [k, v]
        assocs = []
        d.assocsDo(lambda pair: assocs.append(pair))
        assert len(assocs) == 4
        assert sorted(assocs, key=lambda p: p[1]) == [
            ["one", 1],
            ["two", 2],
            ["three", 3],
            ["four", 4],
        ]

    def test_dictionary_high_volume_stress_5000_entries(self):
        """Insert 5,000 key-value pairs, read all back, delete half, verify remainder."""
        d = Dictionary()
        count = 5000

        for i in range(count):
            d.atPut(f"key_{i}", i * 10)

        assert d.size() == count

        # Verify all entries
        for i in range(0, count, 50):
            assert d.at(f"key_{i}") == i * 10
            assert d.includesKey(f"key_{i}")

        # Delete even keys
        for i in range(0, count, 2):
            rem = d.removeKey(f"key_{i}")
            assert rem == i * 10

        assert d.size() == count // 2

        # Check deleted are gone and odd remain
        for i in range(count):
            if i % 2 == 0:
                assert d.at(f"key_{i}") is None
                assert not d.includesKey(f"key_{i}")
            else:
                assert d.at(f"key_{i}") == i * 10
                assert d.includesKey(f"key_{i}")

    def test_removal_during_iteration_safety(self):
        """Modifying dictionary while iterating with do, keysDo, assocsDo should not crash."""
        d = Dictionary()
        for i in range(20):
            d.atPut(f"k_{i}", i)

        # Remove keys inside keysDo callback
        def delete_even_keys(k):
            val = d.at(k)
            if val is not None and val % 2 == 0:
                d.removeKey(k)

        # Must not raise RuntimeError: dictionary changed size during iteration
        d.keysDo(delete_even_keys)
        assert d.size() == 10

    def test_missing_key_lookups(self):
        """Looking up or removing non-existent keys returns None or False cleanly."""
        d = Dictionary()
        assert d.at("ghost") is None
        assert d["ghost"] is None
        assert not d.includesKey("ghost")
        assert "ghost" not in d
        assert d.removeKey("ghost") is None
        assert d.remove("ghost") is None
        assert d.removeAt("ghost") is None

    def test_non_string_keys(self):
        """Dictionary supports integer, float, and tuple keys."""
        d = Dictionary()
        d.atPut(1, "one")
        d.atPut(2.5, "float_two_half")
        d.atPut((10, 20), "tuple_val")

        assert d.at(1) == "one"
        assert d.at(2.5) == "float_two_half"
        assert d.at((10, 20)) == "tuple_val"
        assert d.size() == 3

        assert d.removeKey(2.5) == "float_two_half"
        assert d.size() == 2
        assert d.at(2.5) is None

    def test_actor_dictionary_sync_and_stress(self):
        """ActorDictionary VM globals synchronization, symbol key extraction, and volume."""
        hierarchy = ActorInheritanceHierarchy()
        globals_dict: dict = {}
        ad = ActorDictionary(hierarchy=hierarchy, global_vars=globals_dict)

        # Adding 1,000 entries
        for i in range(1000):
            ad[f"var_{i}"] = i * 2

        assert len(ad) >= 1000
        assert ad["var_500"] == 1000
        assert ad[Symbol("var_500")] == 1000
        assert globals_dict["var_500"] == 1000

        # keys() returns Symbol objects
        keys = ad.keys()
        assert all(isinstance(k, Symbol) for k in keys)

        # Missing key lookup returns None
        assert ad["non_existent"] is None
        assert ad.get("non_existent", "fallback") == "fallback"

        # Removal syncs with globals_dict
        ad.removeKey("var_500")
        assert "var_500" not in globals_dict
        assert ad["var_500"] is None


# ==============================================================================
# 4. Interval Adversarial Stress Tests
# ==============================================================================


class TestIntervalAdversarialStress:
    """Stress tests and boundary checks for Interval."""

    def test_zero_step_rejection(self):
        """Step of zero is invalid and must raise ValueError."""
        with pytest.raises(ValueError):
            Interval(0, 10, 0)
        with pytest.raises(ValueError):
            Interval.overBy(0, 10, 0)

    def test_positive_step_exact_and_non_divisible(self):
        """Positive steps: compare list(itv), itv.size(), and itv.at(i)."""
        # Exact step
        itv1 = Interval(0, 10, 2)
        assert list(itv1) == [0, 2, 4, 6, 8]
        assert itv1.size() == 5
        assert len(itv1) == 5
        assert [itv1.at(i) for i in range(itv1.size())] == [0, 2, 4, 6, 8]
        assert itv1.first() == 0
        assert itv1.last() == 8

        # Non-divisible step
        itv2 = Interval(0, 10, 3)
        assert list(itv2) == [0, 3, 6, 9]
        assert itv2.size() == 4
        assert [itv2[i] for i in range(len(itv2))] == [0, 3, 6, 9]
        assert itv2.first() == 0
        assert itv2.last() == 9

        # Step larger than range
        itv3 = Interval(0, 10, 50)
        assert list(itv3) == [0]
        assert itv3.size() == 1
        assert itv3.first() == 0
        assert itv3.last() == 0

    def test_negative_step_exact_and_non_divisible(self):
        """Negative steps: progression down from start towards stop (exclusive)."""
        # Exact step
        itv1 = Interval(10, 0, -2)
        assert list(itv1) == [10, 8, 6, 4, 2]
        assert itv1.size() == 5
        assert len(itv1) == 5
        assert [itv1.at(i) for i in range(itv1.size())] == [10, 8, 6, 4, 2]
        assert itv1.first() == 10
        assert itv1.last() == 2

        # Non-divisible negative step
        itv2 = Interval(10, 0, -3)
        assert list(itv2) == [10, 7, 4, 1]
        assert itv2.size() == 4
        assert [itv2.at(i) for i in range(itv2.size())] == [10, 7, 4, 1]
        assert itv2.first() == 10
        assert itv2.last() == 1

        # Negative step larger than range
        itv3 = Interval(10, 0, -50)
        assert list(itv3) == [10]
        assert itv3.size() == 1
        assert itv3.first() == 10
        assert itv3.last() == 10

    def test_empty_intervals_both_directions(self):
        """Empty intervals return size 0, empty list, and raise IndexError on first/last/at."""
        empty_forward_equal = Interval(5, 5, 1)
        empty_forward_inverted = Interval(10, 5, 1)
        empty_backward_equal = Interval(5, 5, -1)
        empty_backward_inverted = Interval(5, 10, -1)

        empties = [
            empty_forward_equal,
            empty_forward_inverted,
            empty_backward_equal,
            empty_backward_inverted,
        ]

        for itv in empties:
            assert itv.size() == 0
            assert len(itv) == 0
            assert list(itv) == []
            assert itv.isEmpty()
            with pytest.raises(IndexError):
                itv.first()
            with pytest.raises(IndexError):
                itv.last()
            with pytest.raises(IndexError):
                itv.at(0)
            with pytest.raises(IndexError):
                itv.at(-1)

    def test_interval_indexing_boundaries(self):
        """Out-of-bounds indexing raises IndexError."""
        itv = Interval(1, 10, 2)  # [1, 3, 5, 7, 9] (size 5)
        assert itv.at(0) == 1
        assert itv.at(4) == 9

        with pytest.raises(IndexError):
            itv.at(5)
        with pytest.raises(IndexError):
            itv.at(100)
        with pytest.raises(IndexError):
            itv.at(-1)
        with pytest.raises(IndexError):
            itv.at(-10)

    def test_interval_in_containment(self):
        """in_interval and 'in' operator test membership for positive and negative steps."""
        itv_pos = Interval(0, 10, 2)  # [0, 2, 4, 6, 8]
        assert 0 in itv_pos
        assert 4 in itv_pos
        assert 8 in itv_pos
        assert 1 not in itv_pos
        assert 10 not in itv_pos  # stop is exclusive
        assert -2 not in itv_pos

        itv_neg = Interval(10, 0, -2)  # [10, 8, 6, 4, 2]
        assert 10 in itv_neg
        assert 6 in itv_neg
        assert 2 in itv_neg
        assert 0 not in itv_neg  # stop is exclusive
        assert 8 in itv_neg
        assert 7 not in itv_neg
        assert 12 not in itv_neg

    def test_interval_reverse_and_double_reverse(self):
        """reverse() produces inverted progression; reverse().reverse() reproduces original."""
        itv = Interval(2, 11, 3)  # [2, 5, 8]
        rev = itv.reverse()
        assert list(rev) == [8, 5, 2]
        assert rev.size() == 3

        double_rev = rev.reverse()
        assert list(double_rev) == [2, 5, 8]
        assert double_rev.size() == 3

        # Negative interval reversed
        neg_itv = Interval(20, 5, -4)  # [20, 16, 12, 8]
        rev_neg = neg_itv.reverse()
        assert list(rev_neg) == [8, 12, 16, 20]

        # Empty interval reversed
        empty = Interval(0, 0, 1)
        rev_empty = empty.reverse()
        assert rev_empty.size() == 0
        assert list(rev_empty) == []

    def test_huge_interval_o1_performance(self):
        """Interval size, indexing, and containment on huge range (1 billion) run in O(1)."""
        huge = Interval(0, 1_000_000_000, 100)
        # 1,000,000,000 / 100 = 10,000,000 elements
        assert huge.size() == 10_000_000
        assert huge.first() == 0
        assert huge.last() == 999_999_900

        # O(1) indexed lookups
        assert huge.at(0) == 0
        assert huge.at(5_000_000) == 500_000_000
        assert huge.at(9_999_999) == 999_999_900

        # O(1) containment
        assert 500_000_000 in huge
        assert 500_000_001 not in huge
        assert 1_000_000_000 not in huge

    def test_interval_collect_and_species(self):
        """Interval species is 'Array' and collect returns ActorArray."""
        itv = Interval(1, 6, 1)  # [1, 2, 3, 4, 5]
        assert itv.species() == "Array"

        collected = itv.collect(lambda x: x * 10)
        assert isinstance(collected, ActorArray)
        assert list(collected) == [10, 20, 30, 40, 50]

        arr = itv.asArray()
        assert isinstance(arr, ActorArray)
        assert list(arr) == [1, 2, 3, 4, 5]


# ==============================================================================
# 5. VM Bytecode Execution with Collections, Dictionary, and Interval
# ==============================================================================


class TestVMClassLibraryAdversarialIntegration:
    """Empirically challenge class library operations compiled and run on ActorVM."""

    def _make_vm(self) -> ActorVM:
        vm = ActorVM()
        register_classes(vm.hierarchy, vm.global_vars.get("Actor"), vm.global_vars)
        return vm

    def test_vm_array_indexing_and_fill(self):
        """ActorVM bytecode execution for Array instantiation, indexing, and fill."""
        vm = self._make_vm()
        code = """
        arr := new(Array, 4);
        arr[0] := 10;
        arr[1] := 20;
        arr[2] := 30;
        arr[3] := 40;
        fill(arr, 77);
        ^arr[2];
        !!
        """
        res = vm.eval_expression(code)
        assert res == 77

    def test_vm_dictionary_symbol_and_string_access(self):
        """ActorVM bytecode execution for Dictionary symbol keys, string keys, and lookup."""
        vm = self._make_vm()
        code = """
        d := new(Dictionary, 16);
        d[#mySymbol] := 42;
        d["anotherKey"] := 99;
        ^d[#mySymbol];
        !!
        """
        assert vm.eval_expression(code) == 42

        # Transparent access via string for symbol key
        code_str = """
        ^d["mySymbol"];
        !!
        """
        assert vm.eval_expression(code_str) == 42

    def test_vm_interval_iteration_with_block_closure(self):
        """ActorVM executes Interval do block closures with positive and negative steps."""
        vm = self._make_vm()

        # Forward interval do
        code_fwd = """
        sum := 0;
        do(over(1, 6), {using(x) sum := sum + x;});
        ^sum;
        !!
        """
        assert vm.eval_expression(code_fwd) == 15  # 1+2+3+4+5 = 15

        # Backward (negative step) interval do
        code_bwd = """
        bwdSum := 0;
        do(overBy(10, 0, -2), {using(x) bwdSum := bwdSum + x;});
        ^bwdSum;
        !!
        """
        assert vm.eval_expression(code_bwd) == 30  # 10+8+6+4+2 = 30

    def test_vm_collection_higher_order_collect_select_reject(self):
        """ActorVM executes collect, select, and reject blocks."""
        vm = self._make_vm()

        # collect
        code_collect = """
        arr := new(Array, 3);
        arr[0] := 2;
        arr[1] := 4;
        arr[2] := 6;
        res := collect(arr, {using(x) x * 3;});
        ^res[1];
        !!
        """
        assert vm.eval_expression(code_collect) == 12  # 4 * 3 = 12

        # select
        code_select = """
        selArr := new(Array, 5);
        selArr[0] := 1;
        selArr[1] := 2;
        selArr[2] := 3;
        selArr[3] := 4;
        selArr[4] := 5;
        selected := select(selArr, {using(x) (x > 2);});
        ^size(selected);
        !!
        """
        assert vm.eval_expression(code_select) == 3  # 3, 4, 5

        # reject
        code_reject = """
        rejected := reject(selArr, {using(x) (x > 2);});
        ^size(rejected);
        !!
        """
        assert vm.eval_expression(code_reject) == 2  # 1, 2

    def test_vm_dictionary_iteration_semantics(self):
        """ActorVM authentic iteration semantics: do (values), keysDo (keys), assocsDo ([k, v])."""
        vm = self._make_vm()
        code = """
        dictInst := new(Dictionary, 16);
        dictInst[#k1] := 10;
        dictInst[#k2] := 20;
        dictInst[#k3] := 30;

        valSum := 0;
        do(dictInst, {using(v) valSum := valSum + v;});

        keyCount := 0;
        keysDo(dictInst, {using(k) keyCount := keyCount + 1;});

        assocValSum := 0;
        assocsDo(dictInst, {using(pair) assocValSum := assocValSum + pair[1];});

        ^(valSum == 60 cand keyCount == 3 cand assocValSum == 60);
        !!
        """
        assert vm.eval_expression(code) is True

    def test_vm_ordered_collection_stack_ops(self):
        """ActorVM OrderedCollection add, pop, first, last."""
        vm = self._make_vm()
        code = """
        oc := new(OrderedCollection);
        add(oc, 100);
        add(oc, 200);
        add(oc, 300);
        f := first(oc);
        l := last(oc);
        p := pop(oc);
        ^(f == 100 cand l == 300 cand p == 300 cand last(oc) == 200);
        !!
        """
        assert vm.eval_expression(code) is True

    def test_vm_interval_reverse_and_as_array(self):
        """ActorVM Interval reverse and asArray."""
        vm = self._make_vm()
        code = """
        itv := over(0, 5);
        rev := reverse(itv);
        arr := asArray(itv);
        ^(rev[0] == 4 cand arr[0] == 0 cand arr[4] == 4);
        !!
        """
        assert vm.eval_expression(code) is True


# ==============================================================================
# 6. Packaging & Clean Subprocess Import Isolation Tests
# ==============================================================================


class TestCollectionPackagingAndImports:
    """Verify that collection modules can be imported directly in fresh Python subprocesses."""

    def test_direct_import_collections_isolation(self):
        """Direct import of actorii.classes.collections without prior actorii.core import."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-c",
            "from actorii.classes.collections import ActorArray, ActorOrderedCollection, ActorList; print('OK')",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"Direct import of actorii.classes.collections failed with code {res.returncode}:\n{res.stderr}"

    def test_direct_import_dictionary_isolation(self):
        """Direct import of actorii.classes.dictionary without prior actorii.core import."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-c",
            "from actorii.classes.dictionary import Dictionary, ActorDictionary; print('OK')",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"Direct import of actorii.classes.dictionary failed with code {res.returncode}:\n{res.stderr}"

    def test_direct_import_interval_isolation(self):
        """Direct import of actorii.classes.interval without prior actorii.core import."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-c",
            "from actorii.classes.interval import Interval; print('OK')",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"Direct import of actorii.classes.interval failed with code {res.returncode}:\n{res.stderr}"

    def test_direct_import_actorii_classes_isolation(self):
        """Direct import of actorii.classes without prior actorii.core import."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-c",
            "import actorii.classes; print('OK')",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            res.returncode == 0
        ), f"Direct import of actorii.classes failed with code {res.returncode}:\n{res.stderr}"
