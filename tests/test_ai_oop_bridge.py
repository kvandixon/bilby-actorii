"""Track 2 Test Suite: The AI + OOP Bridge (`doesNotUnderstand` JIT Synthesis & `Agent` Class).

Verifies:
1. Authentic 1993 `doesNotUnderstand` / `fail` message interception in Actor bytecode.
2. Dynamic `doesNotUnderstand` JIT method synthesis (`ActorMethodSynthesizer`), compiling
   synthesized 1993 Actor `Def ... !!` source into `CompiledMethod` bytecode and caching it
   permanently on `ActorClass`.
3. Custom LLM provider (`vm.set_ai_provider`) single-shot synthesis (called once on first
   missing message, zero LLM calls on subsequent message sends).
4. First-class `Agent` class & 1993 Actor subclassing (`inherit(Agent, #SpecialistAgent, ...)`),
   including semantic `memory` (`remember`/`recall`), episodic `history`, `inspectObject`,
   OOP tool registration (`registerTool`/`useTool` with `BlockClosure` and `ActorObject`),
   multi-agent delegation (`delegateTo`), and explicit `synthesizeMethod`.
5. Live `valkyrie` LLM endpoint (`http://127.0.0.1:8080/v1/chat/completions`) method synthesis
   and `Agent` reasoning.
6. Whole-image `.img` persistence (`save_image` / `load_image`) preserving `Agent` state,
   episodic memory, tools, and JIT-synthesized `CompiledMethod` bytecode across VM restarts.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.ai import ActorMethodSynthesizer, inspect_actor_object
from actorii.classes.agent import Agent
from actorii.core import ActorClass, ActorObject, Symbol
from actorii.vm import ActorVM, CompiledMethod
from websocket_server import ActorWorkspace


def _is_valkyrie_8080_online() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


class TestDoesNotUnderstandAndJITSynthesis:
    """Tests `doesNotUnderstand` message interception and dynamic JIT bytecode synthesis."""

    def test_actor_level_does_not_understand_and_1993_fail_hooks(self):
        """Classes defining `doesNotUnderstand` or 1993 `fail` intercept unknown messages."""
        vm = ActorVM()
        vm.eval_chunk(
            """
            inherit(Object, #DynamicProxy, #(#lastSelector #lastArgCount), 2, nil)!!
            now(DynamicProxy)!!
            Def doesNotUnderstand(self, selector, args) {
                self.lastSelector := selector;
                self.lastArgCount := size(args);
                ^"Intercepted:" + selector;
            }!!

            inherit(Object, #Legacy1993Object, nil, 0, nil)!!
            now(Legacy1993Object)!!
            Def fail(self, stackTop, selector, args) {
                ^at(args, 0) + at(args, 1);
            }!!

            Actor[#Proxy] := new(DynamicProxy)!!
            Actor[#Legacy] := new(Legacy1993Object)!!
            """
        )

        res1 = vm.eval_chunk("unknownAction(Actor[#Proxy], 10, 20, 30)!!")
        assert res1 == "Intercepted:unknownAction"
        proxy = vm.global_vars["Actor"]["Proxy"]
        assert proxy._fields["lastSelector"] == Symbol("unknownAction")
        assert proxy._fields["lastArgCount"] == 3

        res2 = vm.eval_chunk("mysteryAdd(Actor[#Legacy], 19, 23)!!")
        assert res2 == 42

    def test_jit_method_synthesis_on_invoice_and_geometry(self):
        """JIT synthesis generates 1993 Actor methods from live fields and caches bytecode on the class."""
        vm = ActorVM()
        vm.eval_chunk(
            """
            inherit(Object, #Invoice, #(#subtotal #taxRate #customer), 3, nil)!!
            enableJitSynthesis(Invoice, true)!!

            Actor[#Inv1] := new(Invoice)!!
            setField(Actor[#Inv1], #subtotal, 200)!!
            setField(Actor[#Inv1], #taxRate, 15)!!
            setField(Actor[#Inv1], #customer, "Whitewater Group")!!
            """
        )

        # 1. Send unknown messages `totalWithTax`, `getSubtotal`, `setSubtotal`, and ` discountedTotal`
        total = vm.eval_chunk("totalWithTax(Actor[#Inv1])!!")
        assert total == 230

        # Verify `totalWithTax` is now a real CompiledMethod on `Invoice`
        inv_cls = vm.hierarchy.get("Invoice")
        assert inv_cls.has_method("totalWithTax")
        cm = inv_cls.get_method("totalWithTax")
        assert isinstance(cm, CompiledMethod)
        assert cm.synthesized is True
        assert "Def totalWithTax(self)" in cm.source

        # Verify Actor-level `synthesizedSource` reflection primitive
        src = vm.eval_chunk("synthesizedSource(Invoice, #totalWithTax)!!")
        assert "self.subtotal" in src

        # 2. Create a second Invoice instance and verify it uses the already-compiled method
        # without triggering any new synthesis!
        synth_count_before = len(vm.synthesized_methods)
        vm.eval_chunk(
            """
            Actor[#Inv2] := new(Invoice)!!
            setField(Actor[#Inv2], #subtotal, 500)!!
            setField(Actor[#Inv2], #taxRate, 10)!!
            """
        )
        total2 = vm.eval_chunk("totalWithTax(Actor[#Inv2])!!")
        assert total2 == 550
        assert len(vm.synthesized_methods) == synth_count_before

    def test_custom_ai_provider_called_once_then_cached_as_bytecode(self):
        """Custom LLM provider synthesizes a method once; subsequent calls run native bytecode."""
        vm = ActorVM()
        provider_calls = []

        def custom_llm(prompt: str, system_prompt: str, metadata: dict) -> str:
            provider_calls.append(metadata)
            return """
            Def computeWeightedScore(self, weight, bonus | base) {
                base := self.score * weight;
                ^base + bonus;
            }!!
            """

        vm.set_ai_provider(custom_llm)
        vm.eval_chunk(
            """
            inherit(Object, #Candidate, #(#score), 1, nil)!!
            enableJitSynthesis(Candidate, true)!!
            Actor[#Cand1] := new(Candidate)!!
            setField(Actor[#Cand1], #score, 12)!!
            Actor[#Cand2] := new(Candidate)!!
            setField(Actor[#Cand2], #score, 20)!!
            """
        )

        # First call triggers custom_llm to synthesize `computeWeightedScore`
        res1 = vm.eval_chunk("computeWeightedScore(Actor[#Cand1], 3, 6)!!")
        assert res1 == 42  # 12 * 3 + 6
        assert len(provider_calls) == 1
        assert provider_calls[0]["selector"] == "computeWeightedScore"

        # Second & third calls execute the compiled bytecode directly (0 additional LLM calls!)
        res2 = vm.eval_chunk("computeWeightedScore(Actor[#Cand2], 4, 10)!!")
        assert res2 == 90  # 20 * 4 + 10
        assert len(provider_calls) == 1


class TestFirstClassAgentAndSubclassing:
    """Tests the native `Agent` class, memory, tools, object inspection, and multi-agent delegation."""

    def test_agent_memory_tools_inspection_and_delegation_in_actor_syntax(self):
        """Agents remember/recall facts, inspect live objects, invoke closure tools, and delegate tasks."""
        vm = ActorVM()
        vm.set_ai_provider(
            lambda prompt, sys_prompt, meta: f"Reply[{meta.get('agent_name')}]: {prompt.splitlines()[0]}"
        )

        vm.eval_chunk(
            """
            inherit(Agent, #ArchitectAgent, #(#domain #clearance), 2, nil)!!
            now(ArchitectAgent)!!
            Def tagDomain(self, d) {
                self.domain := d;
                remember(self, #activeDomain, d);
                ^self.domain;
            }!!

            Actor[#SkippyAgent] := new(ArchitectAgent, "Skippy", "gpt-oss:20b")!!
            Actor[#BilbyAgent] := new(Agent, "Bilby", "clef-flash")!!
            """
        )

        skippy = vm.global_vars["Actor"]["SkippyAgent"]
        bilby = vm.global_vars["Actor"]["BilbyAgent"]
        assert isinstance(skippy, Agent)
        assert skippy.isKindOf("Agent", vm.hierarchy)
        assert skippy.isKindOf("ArchitectAgent", vm.hierarchy)
        assert skippy.isKindOf("Object", vm.hierarchy)

        # Test subclass method & semantic memory
        dom = vm.eval_chunk('tagDomain(Actor[#SkippyAgent], "VM Architecture")!!')
        assert dom == "VM Architecture"
        recalled = vm.eval_chunk("recall(Actor[#SkippyAgent], #activeDomain)!!")
        assert recalled == "VM Architecture"

        # Register a BlockClosure tool on SkippyAgent and invoke it from Actor code
        tool_res = vm.eval_chunk(
            """
            registerTool(Actor[#SkippyAgent], #tripler, {using(x) x * 3 + 1;})!!
            useTool(Actor[#SkippyAgent], #tripler, 10)!!
            """
        )
        assert tool_res == 31

        # Inspect a live object from Actor code
        inspection_json = vm.eval_chunk("inspectObject(Actor[#SkippyAgent], Actor[#BilbyAgent])!!")
        parsed_info = json.loads(inspection_json)
        assert parsed_info["class"] == "Agent"
        assert parsed_info["fields"]["name"] == "Bilby"

        # Multi-agent delegation from SkippyAgent to BilbyAgent
        delegated_reply = vm.eval_chunk(
            'delegateTo(Actor[#SkippyAgent], Actor[#BilbyAgent], "Run local diagnostics")!!'
        )
        assert "Reply[Bilby]:" in delegated_reply
        assert "Delegated from Skippy" in delegated_reply
        assert vm.eval_chunk('recall(Actor[#SkippyAgent], "delegated_Bilby")!!') == delegated_reply
        assert len(bilby.history()) == 2

        # Explicit method synthesis by an Agent onto another class
        vm.eval_chunk(
            """
            inherit(Object, #RectangleBox, #(#width #height), 2, nil)!!
            Actor[#Box] := new(RectangleBox)!!
            setField(Actor[#Box], #width, 12)!!
            setField(Actor[#Box], #height, 5)!!
            """
        )
        vm.set_ai_provider(None)  # Use structural synthesizer
        vm.eval_chunk('synthesizeMethod(Actor[#SkippyAgent], RectangleBox, #area, "Compute width * height", 0)!!')
        assert vm.eval_chunk("area(Actor[#Box])!!") == 60
        vm.eval_chunk('synthesizeMethod(Actor[#SkippyAgent], RectangleBox, #perimeter, "Compute perimeter", 0)!!')
        assert vm.eval_chunk("perimeter(Actor[#Box])!!") == 34


class TestAgentAndJITSynthesisImagePersistence:
    """Tests that JIT-synthesized bytecode methods and live `Agent` objects survive `.img` snapshots."""

    def test_synthesized_methods_and_agents_persist_across_image_save_and_load(
        self, tmp_path: Path
    ):
        """Save an image containing JIT-synthesized classes and stateful Agents; restore into a fresh VM."""
        img_path = tmp_path / "ai_oop_snapshot.img"

        vm1 = ActorVM()
        vm1.set_ai_provider(
            lambda prompt, sys_prompt, meta: "Def customScore(self, factor) { ^(self.base * factor) + self.offset; }!!"
            if meta.get("mode") == "method_synthesis"
            else f"Planned: {prompt}"
        )

        vm1.eval_chunk(
            """
            inherit(Object, #SensorNode, #(#base #offset), 2, nil)!!
            enableJitSynthesis(SensorNode, true)!!
            Actor[#Node1] := new(SensorNode)!!
            setField(Actor[#Node1], #base, 15)!!
            setField(Actor[#Node1], #offset, 7)!!

            inherit(Agent, #SupervisorAgent, #(#hostname), 1, nil)!!
            Actor[#Sup] := new(SupervisorAgent, "Skippy", "gpt-oss:20b")!!
            setField(Actor[#Sup], #hostname, "valkyrie")!!
            remember(Actor[#Sup], #vramBudget, 16)!!
            registerTool(Actor[#Sup], #scaleReading, {using(v) v * 10;})!!
            """
        )

        # Trigger JIT synthesis on vm1 and ask Agent a question
        assert vm1.eval_chunk("customScore(Actor[#Node1], 3)!!") == 52  # 15 * 3 + 7
        assert vm1.eval_chunk('ask(Actor[#Sup], "Check VRAM")!!') == "Planned: Check VRAM"

        # Save image to disk
        vm1.save_image(str(img_path))

        # Restore into a brand-new VM with NO ai_provider attached!
        vm2 = ActorVM()
        vm2.load_image(str(img_path))

        # 1. The JIT-synthesized `customScore` method executes from restored bytecode with no AI provider!
        assert vm2.eval_chunk("customScore(Actor[#Node1], 4)!!") == 67  # 15 * 4 + 7
        sensor_cls = vm2.hierarchy.get("SensorNode")
        restored_cm = sensor_cls.get_method("customScore")
        assert restored_cm.synthesized is True
        assert "self.base * factor" in restored_cm.source

        # 2. The restored `SupervisorAgent` retains its subclass fields, semantic memory, history, and closure tools!
        sup2 = vm2.global_vars["Actor"]["Sup"]
        assert isinstance(sup2, Agent)
        assert sup2.class_name == "SupervisorAgent"
        assert sup2.name == "Skippy"
        assert sup2._fields["hostname"] == "valkyrie"
        assert vm2.eval_chunk("recall(Actor[#Sup], #vramBudget)!!") == 16
        assert vm2.eval_chunk("lastResponse(Actor[#Sup])!!") == "Planned: Check VRAM"
        assert len(sup2.history()) == 2
        assert vm2.eval_chunk("useTool(Actor[#Sup], #scaleReading, 9)!!") == 90

    def test_websocket_jit_synthesize_protocol(self):
        """WebSocket `jit_synthesize` and `list_synthesized_methods` commands work end-to-end."""

        async def _run():
            ws = ActorWorkspace()
            ws.vm.eval_chunk(
                """
                inherit(Object, #Account, #(#balance), 1, nil)!!
                Actor[#Acc] := new(Account)!!
                setField(Actor[#Acc], #balance, 100)!!
                """
            )
            resp = await ws.handle_message(
                json.dumps(
                    {
                        "type": "jit_synthesize",
                        "class_name": "Account",
                        "selector": "deposit",
                        "spec": "Add amount to balance",
                        "arity": 1,
                    }
                )
            )
            assert resp["type"] == "jit_synthesized"
            assert resp["success"] is True
            assert "Def deposit" in resp["source"]

            new_bal = ws.vm.eval_chunk("deposit(Actor[#Acc], 55)!!")
            assert new_bal == 155

            list_resp = await ws.handle_message(
                json.dumps({"type": "list_synthesized_methods"})
            )
            assert list_resp["type"] == "synthesized_methods"
            assert len(list_resp["methods"]) == 1
            assert list_resp["methods"][0]["selector"] == "deposit"

        asyncio.run(_run())


@pytest.mark.skipif(
    not _is_valkyrie_8080_online(),
    reason="Local valkyrie port 8080 LLM server is not online",
)
class TestLiveValkyrieLLMIntegration:
    """Live integration test against valkyrie's local GPU model on port 8080 (`gpt-oss:20b`)."""

    def test_live_valkyrie_8080_jit_method_synthesis_and_agent_ask(self):
        """Synthesize a real 1993 Actor method and query an Agent using live inference on :8080."""
        vm = ActorVM()
        vm.use_live_llm_for_jit = True
        vm.ai_endpoint = "http://127.0.0.1:8080/v1/chat/completions"

        vm.eval_chunk(
            """
            inherit(Object, #MathBox, #(#factor), 1, nil)!!
            enableJitSynthesis(MathBox, true)!!
            Actor[#MBox] := new(MathBox)!!
            setField(Actor[#MBox], #factor, 10)!!
            """
        )

        # Synthesize `doublePlusOne(self, x)` live via :8080 LLM
        synth = ActorMethodSynthesizer(vm)
        cm = synth.synthesize_method(
            receiver=vm.global_vars["Actor"]["MBox"],
            class_name="MathBox",
            selector="doublePlusOne",
            args=[20],
            spec_prompt="Return arg1 * 2 + 1",
            use_llm=True,
        )
        assert isinstance(cm, CompiledMethod)
        assert cm.synthesized is True
        res = vm.eval_chunk("doublePlusOne(Actor[#MBox], 20)!!")
        assert res == 41

        # Live Agent `ask` against :8080
        vm.eval_chunk(
            """
            Actor[#LiveAgent] := new(Agent, "ValkyrieAgent", "gpt-oss:20b")!!
            remember(Actor[#LiveAgent], #secretCode, "WHITEWATER-1993")!!
            """
        )
        reply = vm.eval_chunk(
            'ask(Actor[#LiveAgent], "Reply with ONLY the exact secretCode from your semantic memory.")!!'
        )
        assert "WHITEWATER-1993" in reply
