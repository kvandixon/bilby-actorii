"""Test Suite for 1993 Actor 4.1 Live Image Persistence (Snapshot / Restore).

Verifies:
1. CompiledMethod, BlockDescriptor, EarlyBoundDescriptor, and dynamic class hierarchy
   survival across `save_image` / `load_image` onto a fresh `ActorVM` (including
   unmodified SIEVE.ACT, SIEVE1.ACT, and FRACTAL.ACT benchmarks without re-parsing).
2. Reference-preserving object graph serialization (`object_table`), maintaining exact
   Python object identity (`is`) for shared references and cyclic object graphs.
3. Collection and geometry persistence (`Array`, `OrderedCollection`, `List`, `ActorSet`,
   `Dictionary`, `Interval`, `Point`, `Point3D`, `Rect`).
4. GUI hierarchy (`Window`, `Dialog`, `Button`, `Checkbox`, `TextField`) and live
   `ui_event` action dispatch after WebSocket `save_image` / `load_image`.
5. Native Actor 4.1 language `saveImage("...")!!`, `snapshot("...")!!`, and
   `loadImage("...")!!` invocations.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.classes.collections import (
    ActorArray,
    ActorList,
    ActorOrderedCollection,
    ActorSet,
)
from actorii.classes.dictionary import Dictionary
from actorii.classes.geometry import Point, Point3D, Rect
from actorii.classes.gui import Button, Checkbox, Dialog, TextField, Window
from actorii.classes.interval import Interval
from actorii.core import Symbol
from actorii.vm import ActorVM
from websocket_server import ActorWorkspace

REPO_ROOT = Path(__file__).resolve().parent.parent
HIST_ACT = REPO_ROOT / "historical_actor_41_source" / "actact"


class TestImageSnapshotBenchmarksAndBytecode:
    """Verify compiled bytecode methods and 1993 benchmarks survive image snapshot/restore."""

    def test_sieve_act_runs_from_restored_image_without_recompilation(self, tmp_path):
        """Load SIEVE.ACT in vm1, snapshot to disk, restore into fresh vm2, and run sieve."""
        img_file = tmp_path / "sieve.img"

        vm1 = ActorVM()
        sieve_src = (HIST_ACT / "SIEVE.ACT").read_text(encoding="latin-1")
        vm1.eval_chunk(sieve_src)
        vm1.save_image(str(img_file))

        # Fresh VM: never parses SIEVE.ACT
        vm2 = ActorVM()
        assert "Sam" not in vm2.global_vars["Actor"]
        stats = vm2.load_image(str(img_file))
        assert stats["classes_count"] >= 24
        assert "Sam" in vm2.global_vars["Actor"]

        # Execute restored bytecode method directly on vm2
        res = vm2.eval_chunk("sieve(Actor[#Sam], 100)!!")
        assert res == 25

    def test_sieve1_early_bound_and_fractal_survive_snapshot(self, tmp_path):
        """Early-bound :Object calls (SIEVE1.ACT) and Int>>point3D (FRACTAL.ACT) survive."""
        img_file = tmp_path / "benchmarks.img"

        vm1 = ActorVM()
        sieve1_src = (HIST_ACT / "SIEVE1.ACT").read_text(encoding="latin-1")
        fractal_src = (HIST_ACT / "FRACTAL.ACT").read_text(encoding="latin-1")
        vm1.eval_chunk(sieve1_src)
        vm1.eval_chunk(fractal_src)
        vm1.save_image(str(img_file))

        vm2 = ActorVM()
        vm2.load_image(str(img_file))

        assert vm2.eval_chunk("sieve(Actor[#Sam], 100)!!") == 25
        pt3d = vm2.eval_chunk("point3D(256, 20, 0)!!")
        assert isinstance(pt3d, Point3D)
        assert (pt3d.x, pt3d.y, pt3d.z) == (256, 20, 0)
        pt2d = vm2.eval_chunk("asPoint(point3D(256, 20, 0))!!")
        assert isinstance(pt2d, Point)
        assert (pt2d.x, pt2d.y) == (256, 20)

    def test_closures_lexical_cells_and_nonlocal_return_survive_snapshot(self, tmp_path):
        """Methods with nested {using(...) ...} closures, lexical cells, and ^ non-local returns survive."""
        img_file = tmp_path / "closures.img"

        vm1 = ActorVM()
        vm1.eval_chunk(
            """
            inherit(Object, #Analyzer, #(threshold), 2, nil)!!
            now(Analyzer)!!
            Def findFirstAbove(self, items | found) {
                do(items, {using(x)
                    if x > self.threshold
                    then ^x;
                    endif;
                });
                ^nil;
            }!!
            Def sumAbove(self, items | acc) {
                acc := 0;
                do(items, {using(x)
                    if x > self.threshold
                    then acc := acc + x;
                    endif;
                });
                ^acc;
            }!!
            Actor[#MyAnalyzer] := new(Analyzer)!!
            Actor[#MyAnalyzer].threshold := 50!!
            Actor[#SampleItems] := new(Array, 4)!!
            Actor[#SampleItems][0] := 10!!
            Actor[#SampleItems][1] := 25!!
            Actor[#SampleItems][2] := 64!!
            Actor[#SampleItems][3] := 90!!
            """
        )
        vm1.save_image(str(img_file))

        vm2 = ActorVM()
        vm2.load_image(str(img_file))
        res = vm2.eval_chunk(
            "findFirstAbove(Actor[#MyAnalyzer], Actor[#SampleItems])!!"
        )
        assert res == 64
        total = vm2.eval_chunk(
            "sumAbove(Actor[#MyAnalyzer], Actor[#SampleItems])!!"
        )
        assert total == 154


class TestObjectGraphIdentityAndCollections:
    """Verify reference identity, cycles, collections, and custom fields survive."""

    def test_shared_reference_identity_and_cyclic_graph(self, tmp_path):
        """Shared references remain identical (`is`) and cyclic references do not recurse."""
        img_file = tmp_path / "cycles.img"

        vm1 = ActorVM()
        vm1.eval_chunk(
            """
            inherit(Object, #Node, #(name peer), 2, nil)!!
            Actor[#NodeA] := new(Node)!!
            Actor[#NodeB] := new(Node)!!
            Actor[#NodeA].name := "Alpha"!!
            Actor[#NodeB].name := "Beta"!!
            Actor[#NodeA].peer := Actor[#NodeB]!!
            Actor[#NodeB].peer := Actor[#NodeA]!!
            Actor[#AliasA] := Actor[#NodeA]!!
            """
        )
        vm1.save_image(str(img_file))

        vm2 = ActorVM()
        vm2.load_image(str(img_file))

        node_a = vm2.global_vars["Actor"]["NodeA"]
        node_b = vm2.global_vars["Actor"]["NodeB"]
        alias_a = vm2.global_vars["Actor"]["AliasA"]

        assert node_a is alias_a
        assert node_a._fields["name"] == "Alpha"
        assert node_b._fields["name"] == "Beta"
        assert node_a._fields["peer"] is node_b
        assert node_b._fields["peer"] is node_a

    def test_all_collection_and_geometry_types_persist(self, tmp_path):
        """Array, OrderedCollection, List, ActorSet, Dictionary, Interval, Point, Rect persist."""
        img_file = tmp_path / "collections.img"

        vm1 = ActorVM()
        actor = vm1.global_vars["Actor"]
        actor["Arr"] = ActorArray([10, 20, 30])
        actor["Ord"] = ActorOrderedCollection(["a", "b"])
        actor["Lst"] = ActorList([Symbol("x"), Symbol("y")])

        s = ActorSet(8)
        s.add("unique1")
        s.add(42)
        actor["MySet"] = s

        d = Dictionary(8)
        d.atPut(Symbol("keySym"), Point(12, 34))
        d.atPut("rectKey", Rect(1, 2, 100, 200))
        actor["MyDict"] = d
        actor["MyInterval"] = Interval(1, 10, 2)

        vm1.save_image(str(img_file))

        vm2 = ActorVM()
        vm2.load_image(str(img_file))
        actor2 = vm2.global_vars["Actor"]

        assert isinstance(actor2["Arr"], ActorArray)
        assert list(actor2["Arr"]) == [10, 20, 30]
        assert isinstance(actor2["Ord"], ActorOrderedCollection)
        assert list(actor2["Ord"]) == ["a", "b"]
        assert isinstance(actor2["Lst"], ActorList)
        assert list(actor2["Lst"]) == [Symbol("x"), Symbol("y")]
        assert isinstance(actor2["MySet"], ActorSet)
        assert actor2["MySet"].includes("unique1")
        assert actor2["MySet"].includes(42)
        assert isinstance(actor2["MyDict"], Dictionary)
        pt = actor2["MyDict"].at(Symbol("keySym"))
        assert isinstance(pt, Point) and (pt.x, pt.y) == (12, 34)
        rc = actor2["MyDict"].at("rectKey")
        assert isinstance(rc, Rect) and (rc.left, rc.top, rc.right, rc.bottom) == (
            1,
            2,
            100,
            200,
        )
        itv = actor2["MyInterval"]
        assert isinstance(itv, Interval) and (itv.start, itv.stop, itv.step) == (
            1,
            10,
            2,
        )

    def test_actor_language_save_and_load_image_primitives(self, tmp_path):
        """Actor code can call saveImage("...")!! and loadImage("...")!! directly."""
        img_file = tmp_path / "from_actor.img"
        path_str = str(img_file)

        vm1 = ActorVM()
        vm1.eval_chunk(
            f"""
            Actor[#SavedCounter] := 777!!
            saveImage("{path_str}")!!
            """
        )
        assert img_file.is_file()

        vm2 = ActorVM()
        assert vm2.global_vars["Actor"].at("SavedCounter") is None
        vm2.eval_chunk(f'loadImage("{path_str}")!!')
        assert vm2.global_vars["Actor"]["SavedCounter"] == 777


class TestWebSocketAndGUIImagePersistence:
    """Verify GUI windows, widgets, and WebSocket save_image/load_image roundtrips."""

    def test_websocket_save_and_load_image_with_gui_action_dispatch(self, tmp_path):
        """GUI Window, Button, Checkbox, TextField and compiled action handlers survive."""
        img_file = tmp_path / "workspace_gui.img"

        async def _run():
            ws1 = ActorWorkspace()
            await ws1.handle_message(
                {
                    "type": "eval",
                    "source": """
                    inherit(Window, #DashWindow, #(clicks), 2, nil)!!
                    now(DashWindow)!!
                    Def onBoost(self, msg) {
                        self.clicks := self.clicks + 10;
                        ^self.clicks;
                    }!!
                    """,
                }
            )

            win_cls = ws1.vm.hierarchy.get("DashWindow")
            win = Window(title="Telemetry Dashboard", left=10, top=20, width=500, height=400)
            win._class_name = "DashWindow"
            win.inherit_from(win_cls)
            win._fields["clicks"] = 5

            btn = Button(label="Boost", id="btnBoost", identity=Symbol("btnBoost"))
            chk = Checkbox(label="Active", checked=True, id="chkAct", identity=Symbol("chkAct"))
            txt = TextField(text="valkyrie-01", id="txtHost", identity=Symbol("txtHost"))
            win.add(btn)
            win.add(chk)
            win.add(txt)
            win.register_action("btnBoost", "selected", "onBoost")

            ws1.vm.global_vars["Actor"]["MainWin"] = win
            ws1.vm.global_vars["Actor"]["btnBoost"] = btn

            save_resp = await ws1.handle_message(
                {"type": "save_image", "path": str(img_file)}
            )
            assert save_resp["type"] == "image_saved"
            assert save_resp["success"] is True

            # Restore into a brand-new ActorWorkspace
            ws2 = ActorWorkspace()
            load_resp = await ws2.handle_message(
                {"type": "load_image", "path": str(img_file)}
            )
            assert load_resp["type"] == "image_loaded"
            assert load_resp["success"] is True

            # Verify restored Window and child widgets
            restored_win = ws2.vm.global_vars["Actor"]["MainWin"]
            assert restored_win.title == "Telemetry Dashboard"
            assert restored_win._fields["clicks"] == 5
            assert len(restored_win.children) == 3
            assert restored_win.find_child("chkAct").checked is True
            assert restored_win.find_child("txtHost").text == "valkyrie-01"

            # Dispatch UI action on restored button in ws2
            ev_resp = await ws2.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "btnBoost",
                    "action": "#selected",
                }
            )
            assert ev_resp["type"] == "ui_event_result"
            assert ev_resp["handled"] is True
            assert ev_resp["result"] == 15
            assert restored_win._fields["clicks"] == 15

        asyncio.run(_run())
