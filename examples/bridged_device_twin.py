"""Showcase: Zero-Boilerplate Python-to-Actor Hardware Twin (`examples/bridged_device_twin.py`).

Demonstrates how a standard Python hardware/service driver can be exposed to the
ActorII VM in ~30 lines using `@actor_bridge`, `@actor_query`, and `@actor_command`,
with full static bytecode linting, transactional shadow runs, and automatic `.act`
source File-Out.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.vm import ActorVM


@actor_bridge(name="EnvironmentNode")
class EnvironmentNodeDriver:
    """Python hardware driver wrapped for ActorII with capability-gated selectors."""

    def __init__(self, location: str = "ValkyrieOffice", temp_c10: int = 212, rh: int = 48) -> None:
        self.location = location
        self._temp_c10 = temp_c10
        self._rh = rh
        self._fan_duty = 0

    @actor_query
    def locationName(self) -> str:
        return self.location

    @actor_query
    def tempC10(self) -> int:
        return self._temp_c10

    @actor_query
    def humidity(self) -> int:
        return self._rh

    @actor_query
    def fanDuty(self) -> int:
        return self._fan_duty

    @actor_command
    def setFanDuty(self, duty_pct: int) -> int:
        self._fan_duty = max(0, min(100, int(duty_pct)))
        return self._fan_duty


def run_demo() -> dict:
    """Run an end-to-end guarded bridge & file-out workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = ActorVM()
        vm.enable_guardrails(
            True,
            protect_system_classes=True,
            strict_ivars=True,
            max_opcodes=50_000,
            fileout_dir=tmpdir,
        )

        # 1. Expose the live Python driver in 1 line
        node = EnvironmentNodeDriver(location="ValkyrieRack", temp_c10=255, rh=58)
        vm.expose_object(node, global_key="#RackNode")

        # 2. Define a reactive Actor controller class and wire it to the bridged Python object
        vm.eval_chunk(
            """
            inherit(Object, #RackFanPolicy, #(#thresholdC10 #lastDuty), 2, nil)!!
            now(RackFanPolicy)!!

            Def init(self, thresh) {
                self.thresholdC10 := thresh;
                self.lastDuty := 0;
                ^self;
            }!!

            Def evaluate(self, sensor) {
                if tempC10(sensor) > self.thresholdC10 then
                    self.lastDuty := setFanDuty(sensor, 80);
                else
                    self.lastDuty := setFanDuty(sensor, 0);
                endif;
                ^self.lastDuty;
            }!!

            Actor[#FanPolicy] := init(new(RackFanPolicy), 240)!!
            """
        )

        duty = vm.eval_chunk("evaluate(Actor[#FanPolicy], Actor[#RackNode])!!")
        exported_act = (Path(tmpdir) / "RackFanPolicy.act").read_text(encoding="utf-8")
        return {
            "location": vm.eval_chunk("locationName(Actor[#RackNode])!!"),
            "fan_duty": duty,
            "audit_entries": vm.eval_chunk("bridgeAuditLog(Actor[#RackNode])!!"),
            "fileout_lines": len(exported_act.splitlines()),
        }


if __name__ == "__main__":
    print(run_demo())
