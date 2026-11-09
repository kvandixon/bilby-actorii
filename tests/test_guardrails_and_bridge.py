"""Test Suite for ActorII Guardrails Subsystem & Python-to-Actor FFI Bridge.

Verifies:
1. Zero-boilerplate Python-to-Actor Bridge (`@actor_bridge`, `@actor_query`, `@actor_command`,
   `vm.expose_object`, `vm.register_bridge`) and strict capability enforcement (`ActorBridgeSecurityError`).
2. Static Bytecode Linter & Risk Tier Classification (`Tier 0`, `Tier 1`, `Tier 2`),
   strict `ivar` schema validation, and Protected System Class guards (`ProtectedClassError`).
3. Opcode Gas Metering (`OpcodeBudgetExceededError`) and Transactional Image Rollback
   (`vm.eval_transactional`, speculative `shadow_only=True` dry-runs blocking hardware side-effects).
4. Self-Correcting `doesNotUnderstand` JIT Synthesis with Judge gating and automatic `.act` File-Out.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.bridge import (
    ActorBridgeObject,
    ActorBridgeSecurityError,
    actor_bridge,
    actor_command,
    actor_query,
)
from actorii.guardrails import (
    ActorFileOutManager,
    ActorGuardrailError,
    OpcodeBudgetExceededError,
    ProtectedClassError,
    RiskTier,
)
from actorii.vm import ActorVM


# =============================================================================
# 1. Sample Python Hardware Driver for Bridge Testing
# =============================================================================


@actor_bridge(name="SmartRelayDriver")
class SmartRelayDriver:
    """Minimal Python hardware driver exposed to ActorII in <25 lines."""

    def __init__(self, room: str = "Lab", initial_temp_c10: int = 215) -> None:
        self.room = room
        self._temp_c10 = initial_temp_c10
        self._relay_on = False
        self._secret_token = "DO_NOT_EXPOSE_ME"

    @actor_query
    def roomName(self) -> str:
        return self.room

    @actor_query
    def tempC10(self) -> int:
        return self._temp_c10

    @actor_query
    def isRelayOn(self) -> bool:
        return self._relay_on

    @actor_command
    def setRelay(self, state: bool) -> bool:
        self._relay_on = bool(state)
        return self._relay_on

    def internalReset(self) -> None:
        """Unexposed internal Python method — must be blocked from Actor."""
        self._relay_on = False
        self._temp_c10 = 0


# =============================================================================
# 2. Python-to-Actor Bridge Tests
# =============================================================================


class TestPythonToActorBridge:
    """Tests capability-gated Python-to-Actor bridging."""

    def test_expose_python_object_queries_and_commands_in_actor(self):
        vm = ActorVM()
        driver = SmartRelayDriver(room="ValkyrieRack", initial_temp_c10=248)
        bridge_obj = vm.expose_object(driver, global_key="#RackRelay")

        assert isinstance(bridge_obj, ActorBridgeObject)
        assert vm.eval_chunk("roomName(Actor[#RackRelay])!!") == "ValkyrieRack"
        assert vm.eval_chunk("tempC10(Actor[#RackRelay])!!") == 248
        assert vm.eval_chunk("isRelayOn(Actor[#RackRelay])!!") is False

        # Trigger Tier 2 command from an Actor rule
        vm.eval_chunk(
            """
            if tempC10(Actor[#RackRelay]) > 240 then
                setRelay(Actor[#RackRelay], true);
            endif!!
            """
        )
        assert driver._relay_on is True
        assert vm.eval_chunk("isRelayOn(Actor[#RackRelay])!!") is True
        assert vm.eval_chunk("bridgeAuditLog(Actor[#RackRelay])!!") >= 3

    def test_bridge_blocks_unexposed_python_methods_and_dunders(self):
        vm = ActorVM()
        driver = SmartRelayDriver()
        vm.expose_object(driver, global_key="#Relay")

        with pytest.raises(ActorBridgeSecurityError, match="unexposed selector 'internalReset'"):
            vm.eval_chunk("internalReset(Actor[#Relay])!!")

        # Ensure internal state was not touched
        assert driver._temp_c10 == 215

    def test_register_bridge_class_instantiation_from_actor(self):
        vm = ActorVM()
        vm.register_bridge(SmartRelayDriver)

        vm.eval_chunk('Actor[#LabSensor] := new(SmartRelayDriver, "CleanRoom", 195)!!')
        assert vm.eval_chunk("roomName(Actor[#LabSensor])!!") == "CleanRoom"
        assert vm.eval_chunk("tempC10(Actor[#LabSensor])!!") == 195


# =============================================================================
# 3. Static Linter, Risk Tiers, Protected Classes & Opcode Gas Tests
# =============================================================================


class TestGuardrailsLinterAndSandbox:
    """Tests static bytecode linting, risk tiers, opcode gas limits, and transactions."""

    def test_bytecode_risk_tier_classification(self):
        vm = ActorVM()
        vm.register_bridge(SmartRelayDriver)
        vm.enable_guardrails(True, protect_system_classes=True, strict_ivars=True)

        vm.eval_chunk(
            """
            inherit(Object, #SensorNode, #(#reading #alerts), 2, nil)!!
            now(SensorNode)!!

            Def getReading(self) {
                ^self.reading;
            }!!

            Def recordAlert(self, val) {
                self.reading := val;
                self.alerts := self.alerts + 1;
                ^self.alerts;
            }!!

            Def triggerBridgeRelay(self, relayObj) {
                setRelay(relayObj, true);
                ^true;
            }!!
            """
        )

        cls = vm.hierarchy.get("SensorNode")
        m_read = cls.get_method("getReading")
        m_mut = cls.get_method("recordAlert")
        m_bridge = cls.get_method("triggerBridgeRelay")

        assert m_read.risk_tier == RiskTier.TIER_0_READ_ONLY
        assert m_mut.risk_tier == RiskTier.TIER_1_LOCAL_MUTATION
        assert m_bridge.risk_tier == RiskTier.TIER_2_SYSTEM_OR_BRIDGE

    def test_strict_ivar_linter_catches_undeclared_fields(self):
        vm = ActorVM()
        vm.enable_guardrails(True, strict_ivars=True)

        vm.eval_chunk("inherit(Object, #StrictBox, #(#width #height), 2, nil)!!")
        vm.eval_chunk("now(StrictBox)!!")

        with pytest.raises(ActorGuardrailError, match="LINT_UNDECLARED_IVAR"):
            vm.eval_chunk(
                """
                Def badArea(self) {
                    ^self.width * self.typoHeight;
                }!!
                """
            )

    def test_protected_system_class_prevents_overwriting_core_methods(self):
        vm = ActorVM()
        vm.enable_guardrails(True, protect_system_classes=True)

        with pytest.raises(ProtectedClassError, match="protected system class 'Object'"):
            vm.eval_chunk(
                """
                now(Object)!!
                Def asString(self) { ^"HACKED"; }!!
                """
            )

    def test_opcode_gas_budget_halts_infinite_loops(self):
        vm = ActorVM()
        vm.enable_guardrails(True, max_opcodes=2_500)

        with pytest.raises(OpcodeBudgetExceededError, match="exceeded max_opcodes budget"):
            vm.eval_chunk(
                """
                Actor[#Counter] := 0;
                loop while true begin
                    Actor[#Counter] := Actor[#Counter] + 1;
                endLoop!!
                """
            )

    def test_transactional_eval_rolls_back_state_on_error_and_blocks_shadow_commands(self):
        vm = ActorVM()
        driver = SmartRelayDriver(room="ServerRoom", initial_temp_c10=220)
        vm.expose_object(driver, global_key="#Relay")

        vm.eval_chunk(
            """
            inherit(Object, #Vault, #(#balance), 1, nil)!!
            now(Vault)!!
            Def init(self, b) { self.balance := b; ^self; }!!
            Def balance(self) { ^self.balance; }!!
            Def setBalance(self, b) { self.balance := b; ^self; }!!
            Actor[#MyVault] := init(new(Vault), 1000)!!
            """
        )
        assert vm.eval_chunk("balance(Actor[#MyVault])!!") == 1000

        # 1. Mutate state and then hit an opcode gas limit inside a transaction -> full rollback!
        with pytest.raises(OpcodeBudgetExceededError):
            vm.eval_transactional(
                """
                setBalance(Actor[#MyVault], 999999);
                loop while true begin
                    setBalance(Actor[#MyVault], balance(Actor[#MyVault]) + 1);
                endLoop!!
                """,
                max_opcodes=1_500,
            )

        # Balance must be rolled back to 1000!
        assert vm.eval_chunk("balance(Actor[#MyVault])!!") == 1000

        # 2. Shadow dry-run blocks Tier 2 hardware bridge commands from firing
        with pytest.raises(ActorBridgeSecurityError, match="blocked during transactional shadow run"):
            vm.eval_transactional(
                "setRelay(Actor[#Relay], true)!!",
                shadow_only=True,
            )
        assert driver._relay_on is False


# =============================================================================
# 4. Self-Correcting JIT Synthesis, Judge Gating & Automatic `.act` File-Out
# =============================================================================


class TestSelfCorrectingJitAndFileOut:
    """Tests LLM self-correction on lint errors, Judge gating, and `.act` File-Out."""

    def test_jit_synthesis_self_corrects_when_first_attempt_uses_undeclared_ivar(self, tmp_path):
        vm = ActorVM()
        fileout_dir = tmp_path / "fileout"
        vm.enable_guardrails(True, strict_ivars=True, fileout_dir=str(fileout_dir))

        vm.eval_chunk(
            """
            inherit(Object, #BatteryPack, #(#voltageMv #capacityMah), 2, nil)!!
            now(BatteryPack)!!
            Def init(self, v, c) {
                self.voltageMv := v;
                self.capacityMah := c;
                ^self;
            }!!
            Actor[#Pack] := init(new(BatteryPack), 3700, 2000)!!
            """
        )

        prompts_received: List[str] = []

        def self_correcting_llm(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            prompts_received.append(prompt)
            if len(prompts_received) == 1:
                # Attempt 1: Hallucinates undeclared field `self.volts`
                return "Def energyScore(self) { ^self.volts * self.capacityMah; }!!"
            # Attempt 2: Sees LINT_UNDECLARED_IVAR feedback in prompt and uses `self.voltageMv`
            assert "LINT_UNDECLARED_IVAR" in prompt
            return "Def energyScore(self) { ^(self.voltageMv * self.capacityMah) / 1000; }!!"

        vm.set_ai_provider(self_correcting_llm)
        vm.enable_jit_synthesis(True, target="BatteryPack")

        score = vm.eval_chunk("energyScore(Actor[#Pack])!!")
        assert score == 7400
        assert len(prompts_received) == 2

        # Verify automatic `.act` File-Out wrote `BatteryPack.act` to disk
        exported_file = fileout_dir / "BatteryPack.act"
        assert exported_file.is_file()
        exported_text = exported_file.read_text(encoding="utf-8")
        assert "inherit(Object, #BatteryPack, #(#voltageMv #capacityMah), 2, nil)!!" in exported_text
        assert "Def energyScore(self)" in exported_text

        # Verify that a brand-new ActorVM can load the exported `.act` file directly from Git/disk
        vm_fresh = ActorVM()
        vm_fresh.eval_chunk(exported_text)
        vm_fresh.eval_chunk("Actor[#Pack2] := init(new(BatteryPack), 4000, 1500)!!")
        assert vm_fresh.eval_chunk("energyScore(Actor[#Pack2])!!") == 6000

    def test_judge_blocks_synthesized_query_that_secretly_mutates_state(self):
        vm = ActorVM()
        vm.enable_guardrails(True, strict_ivars=True)

        vm.eval_chunk(
            """
            inherit(Object, #Account, #(#balance), 1, nil)!!
            now(Account)!!
            Def init(self, b) { self.balance := b; ^self; }!!
            Actor[#Acc] := init(new(Account), 500)!!
            """
        )

        def sneaky_mutating_llm(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            # Query selector `getBalance` tries to zero out `self.balance`!
            return "Def getBalance(self) { self.balance := 0; ^self.balance; }!!"

        vm.set_ai_provider(sneaky_mutating_llm)
        vm.enable_jit_synthesis(True, target="Account")

        with pytest.raises(ActorGuardrailError, match="unexpectedly mutates state"):
            vm.eval_chunk("getBalance(Actor[#Acc])!!")

        # Balance remains intact
        assert vm.global_vars["Actor"]["Acc"]._fields["balance"] == 500

    def test_websocket_workspace_guardrail_and_fileout_protocol(self, tmp_path):
        import asyncio
        from websocket_server import ActorWorkspace

        ws = ActorWorkspace()
        fout_dir = tmp_path / "ws_fileout"

        # 1. Enable guardrails over WebSocket message protocol
        res_en = asyncio.run(
            ws.handle_message(
                {
                    "type": "enable_guardrails",
                    "enabled": True,
                    "protect_system_classes": True,
                    "strict_ivars": True,
                    "max_opcodes": 5000,
                    "fileout_dir": str(fout_dir),
                }
            )
        )
        assert res_en["success"] is True
        assert res_en["guardrails_enabled"] is True

        # 2. Define a class and method via eval -> verify audit log and file-out
        res_ev = asyncio.run(
            ws.handle_message(
                {
                    "type": "eval",
                    "source": """
                    inherit(Object, #Meter, #(#watts), 1, nil)!!
                    now(Meter)!!
                    Def init(self, w) { self.watts := w; ^self; }!!
                    Def watts(self) { ^self.watts; }!!
                    Actor[#M1] := init(new(Meter), 450)!!
                    """,
                }
            )
        )
        assert res_ev["success"] is True

        res_audit = asyncio.run(ws.handle_message({"type": "get_guardrail_audit"}))
        assert res_audit["success"] is True
        assert len(res_audit["audit_log"]) >= 2

        # 3. Run failing transactional eval over WebSocket -> verify automatic rollback
        res_tx = asyncio.run(
            ws.handle_message(
                {
                    "type": "eval_transactional",
                    "max_opcodes": 800,
                    "source": """
                    loop while true begin
                        Actor[#M1] := nil;
                    endLoop!!
                    """,
                }
            )
        )
        assert res_tx["success"] is False
        assert res_tx["rolled_back"] is True

        # Verify Actor[#M1] was NOT wiped out
        res_check = asyncio.run(
            ws.handle_message({"type": "eval", "source": "watts(Actor[#M1])!!"})
        )
        assert res_check["result"] == 450

        # 4. File out class over WebSocket
        res_fo = asyncio.run(
            ws.handle_message({"type": "file_out_class", "class_name": "Meter"})
        )
        assert res_fo["success"] is True
        assert "inherit(Object, #Meter, #(#watts), 1, nil)!!" in res_fo["source"]

    def test_bridged_device_twin_showcase_script(self):
        from examples.bridged_device_twin import run_demo

        out = run_demo()
        assert out["location"] == "ValkyrieRack"
        assert out["fan_duty"] == 80
        assert out["audit_entries"] >= 2
        assert out["fileout_lines"] >= 15

