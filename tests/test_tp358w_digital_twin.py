"""Test Suite for ThermoPro TP358W / TP358S Bluetooth Digital Twin (`ThermoProTwin`).

Verifies:
1. Deterministic BLE advertisement decoding (`0xD0C2` / `0xC2` frames, positive & negative temperatures)
   and active GATT `0xCCCC` historical stream assembly & decoding.
2. Full 1993 Actor OOP Digital Twin (`examples/tp358w_digital_twin.act`):
   - `ThermoProTwin` message protocol (`temperatureC`, `temperatureF`, `humidity`, `comfortLevel`,
     `dewPointC`, `historyCount`, `minTempF`, `maxTempF`, `avgHumidity`, `recordAt`, `summary`).
   - Automatic Actor observer notifications (`addObserver` -> `onSensorUpdate(observer, twin)`)
     driving `ComfortMonitor` and `HvacController`.
3. AI + OOP integration (`ClimateAgent` inspection + `doesNotUnderstand` JIT method synthesis on
   `ThermoProTwin`) and whole-image `.img` snapshot persistence (`save_image` / `load_image`).
4. Live physical BLE execution on `valkyrie` against `TP358S (Demo)` (`00:11:22:33:44:55`).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from actorii.classes.thermopro import (
    BlueZThermoProTransport,
    ThermoProTwin,
    decode_advertisement_bytes,
    decode_history_stream,
    make_datetime_sync_cmd,
    make_history_request_cmds,
)
from actorii.core import Symbol
from actorii.vm import ActorVM, CompiledMethod


REAL_ADV_HEX = "c2d00034211b01"
# Real 30-minute GATT history stream captured from 00:11:22:33:44:55 on valkyrie
REAL_HISTORY_HEX = (
    "cccc015b00"
    "0000d00034cf0034ce0034ce0034cf0034cf0034cf0034cf0034d00034d00034"
    "d00034cf0034cf0034cf0034cf0034cf0034cf0034d00034cf0034d00034"
    "cf0034cf0034d00034d00034d00034d10034d00034cf0034cf0034cf0034"
    "036666"
)


def _is_bluez_adapter_powered() -> bool:
    """Check whether BlueZ adapter `/org/bluez/hci0` is powered on."""
    try:
        from gi.repository import Gio

        bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        objs = bus.call_sync(
            "org.bluez",
            "/",
            "org.freedesktop.DBus.ObjectManager",
            "GetManagedObjects",
            None,
            None,
            Gio.DBusCallFlags.NONE,
            5000,
            None,
        ).unpack()[0]
        adapter = objs.get("/org/bluez/hci0", {}).get("org.bluez.Adapter1", {})
        return bool(adapter.get("Powered", False))
    except Exception:
        return False


class TestThermoProWireProtocol:
    """Tests deterministic decoding of TP358W BLE advertisements and GATT history streams."""

    def test_decode_real_advertisement_and_subzero_packets(self):
        reading = decode_advertisement_bytes(bytes.fromhex(REAL_ADV_HEX), rssi=-77)
        assert reading["temperature_c"] == 20.8
        assert reading["temperature_c10"] == 208
        assert reading["temperature_f"] == 69.4
        assert reading["temperature_f10"] == 694
        assert reading["humidity"] == 52
        assert reading["battery"] == 50
        assert reading["rssi"] == -77
        assert reading["comfort_level"] == Symbol("Comfort")
        assert reading["dew_point_c"] == 10.6

        # Test negative temperature (-5.0 C = -50 = 0xFFCE little-endian: ce ff) and 22% RH (#Dry)
        neg_raw = bytes.fromhex("c2ceff16221b02")
        neg = decode_advertisement_bytes(neg_raw, rssi=-65)
        assert neg["temperature_c"] == -5.0
        assert neg["temperature_c10"] == -50
        assert neg["temperature_f"] == 23.0
        assert neg["humidity"] == 22
        assert neg["battery"] == 100
        assert neg["comfort_level"] == Symbol("Dry")

    def test_decode_real_gatt_history_stream_with_status_frame(self):
        # Include the cmd2 status frame (`cccc0400000000046666`) before the `cccc01...` history frame
        combined = bytes.fromhex("cccc0400000000046666" + REAL_HISTORY_HEX)
        ft = datetime(2026, 10, 27, 1, 5, 0)
        records = decode_history_stream(combined, fetch_time=ft)
        assert len(records) == 30

        temps_c = [r["tempC"] for r in records]
        humidities = [r["humidity"] for r in records]
        assert all(20.0 <= t <= 21.5 for t in temps_c)
        assert all(h == 52 for h in humidities)
        assert min(temps_c) == 20.6
        assert max(temps_c) == 20.9
        # Chronological ordering (oldest -> newest)
        assert records[0]["timestamp"] < records[-1]["timestamp"]
        assert records[-1]["minutesAgo"] == 0
        assert records[0]["minutesAgo"] == 29

    def test_command_builders_checksums(self):
        dt = datetime(2026, 10, 27, 1, 5, 30)
        sync_cmd = make_datetime_sync_cmd(now=dt)
        assert sync_cmd[0] == 0xA5
        assert len(sync_cmd) == 9
        assert sync_cmd[-1] == (sum(sync_cmd[:-1]) & 0xFF)

        cmds = make_history_request_cmds(count=30, now=dt)
        assert len(cmds) == 3
        assert cmds[0] == bytes.fromhex("cccc0201000001046666")
        assert cmds[1] == bytes.fromhex("cccc0400000000046666")
        assert cmds[2][:2] == b"\xcc\xcc"
        assert cmds[2][-2:] == b"\x66\x66"


class TestActorDigitalTwinAndCollaborators:
    """Tests the 1993 Actor Digital Twin program (`examples/tp358w_digital_twin.act`)."""

    def test_digital_twin_observer_pattern_and_history_in_actor(self):
        vm = ActorVM()
        act_path = Path(_PROJECT_ROOT) / "examples" / "tp358w_digital_twin.act"
        vm.eval_chunk(act_path.read_text(encoding="utf-8"))

        # 1. Verify Digital Twin class & identity in Actor
        assert vm.eval_chunk("kindOf(Actor[#Thermometer], ThermoProTwin)!!") is True
        assert vm.eval_chunk("kindOf(Actor[#Thermometer], Object)!!") is True
        assert vm.eval_chunk("macAddress(Actor[#Thermometer])!!") == "00:11:22:33:44:55"
        assert vm.eval_chunk("model(Actor[#Thermometer])!!") == "TP358W"

        # 2. Ingest real advertisement packet via Actor message passing
        twin = vm.eval_chunk(f'ingestAdvertisementHex(Actor[#Thermometer], "{REAL_ADV_HEX}", -77)!!')
        assert isinstance(twin, ThermoProTwin)
        assert vm.eval_chunk("temperatureC(Actor[#Thermometer])!!") == 20.8
        assert vm.eval_chunk("temperatureC10(Actor[#Thermometer])!!") == 208
        assert vm.eval_chunk("temperatureF10(Actor[#Thermometer])!!") == 694
        assert vm.eval_chunk("humidity(Actor[#Thermometer])!!") == 52
        assert vm.eval_chunk("battery(Actor[#Thermometer])!!") == 50
        assert vm.eval_chunk("comfortLevel(Actor[#Thermometer])!!") == Symbol("Comfort")

        # 3. Verify that `ComfortMonitor` and `HvacController` were notified via `onSensorUpdate`!
        assert vm.eval_chunk("lastTempC10(Actor[#Monitor])!!") == 208
        assert vm.eval_chunk("lastHumidity(Actor[#Monitor])!!") == 52
        assert vm.eval_chunk("lastStatus(Actor[#Monitor])!!") == Symbol("Comfort")
        assert vm.eval_chunk("alertCount(Actor[#Monitor])!!") == 0
        assert vm.eval_chunk("mode(Actor[#Hvac])!!") == "IDLE"
        assert vm.eval_chunk("updates(Actor[#Hvac])!!") == 1

        # 4. Simulate a hot + humid spike (26.0 C = 260 = 0x0104 -> 04 01, 68% RH = 0x44 -> #Wet)
        vm.eval_chunk('ingestAdvertisementHex(Actor[#Thermometer], "c2040144211b01", -72)!!')
        assert vm.eval_chunk("temperatureC10(Actor[#Thermometer])!!") == 260
        assert vm.eval_chunk("comfortLevel(Actor[#Thermometer])!!") == Symbol("Wet")
        assert vm.eval_chunk("alertCount(Actor[#Monitor])!!") == 1
        assert vm.eval_chunk("mode(Actor[#Hvac])!!") == "COOLING"
        assert vm.eval_chunk("updates(Actor[#Hvac])!!") == 2

        # 5. Ingest real GATT history stream and query historical statistics in Actor
        count = vm.eval_chunk(
            f'ingestHistoryHex(Actor[#Thermometer], "{REAL_HISTORY_HEX}", "2026-10-27T01:05:00")!!'
        )
        assert count == 30
        assert vm.eval_chunk("historyCount(Actor[#Thermometer])!!") == 30
        assert vm.eval_chunk("minTempC10(Actor[#Thermometer])!!") == 206
        assert vm.eval_chunk("maxTempC10(Actor[#Thermometer])!!") == 209
        assert vm.eval_chunk("avgHumidity(Actor[#Thermometer])!!") == 52

        latest = vm.eval_chunk("latestRecord(Actor[#Thermometer])!!")
        assert latest.at("tempC") == 20.8
        assert latest.at("humidity") == 52

        # 6. Verify `ClimateAgent` tool execution & report on the Digital Twin
        report = vm.eval_chunk("twinReport(Actor[#ClimateBot])!!")
        assert "HistoryRecords=30" in report
        assert "MinC10=206" in report
        assert "MaxC10=209" in report

    def test_jit_method_synthesis_and_image_persistence_on_digital_twin(self, tmp_path):
        """Tests synthesizing a new method onto ThermoProTwin and saving/restoring `.img`."""
        vm = ActorVM()
        act_path = Path(_PROJECT_ROOT) / "examples" / "tp358w_digital_twin.act"
        vm.eval_chunk(act_path.read_text(encoding="utf-8"))
        vm.eval_chunk(f'ingestAdvertisementHex(Actor[#Thermometer], "{REAL_ADV_HEX}", -77)!!')
        vm.eval_chunk(
            f'ingestHistoryHex(Actor[#Thermometer], "{REAL_HISTORY_HEX}", "2026-10-27T01:05:00")!!'
        )

        # Register deterministic AI provider for JIT synthesis of `tempRangeC10`
        def custom_llm(prompt: str, system_prompt: str = "") -> str:
            return "Def tempRangeC10(self) { ^maxTempC10(self) - minTempC10(self); }!!"

        vm.set_ai_provider(custom_llm)
        vm.eval_chunk("enableJitSynthesis(ThermoProTwin, true)!!")

        # Send unknown message `tempRangeC10` -> synthesized, compiled to bytecode, and executed!
        spread = vm.eval_chunk("tempRangeC10(Actor[#Thermometer])!!")
        assert spread == 3  # 209 - 206 = 3 (0.3 C)
        tp_cls = vm.hierarchy.get("ThermoProTwin")
        assert isinstance(tp_cls.get_method("tempRangeC10"), CompiledMethod)

        # Save live image to disk
        img_file = tmp_path / "tp358w_twin.img"
        vm.eval_chunk(f'saveImage("{img_file}")!!')
        assert img_file.is_file()

        # Restore into a brand-new ActorVM
        vm2 = ActorVM()
        vm2.load_image(img_file)

        assert vm2.eval_chunk("macAddress(Actor[#Thermometer])!!") == "00:11:22:33:44:55"
        assert vm2.eval_chunk("historyCount(Actor[#Thermometer])!!") == 30
        assert vm2.eval_chunk("tempRangeC10(Actor[#Thermometer])!!") == 3
        assert vm2.eval_chunk("lastTempC10(Actor[#Monitor])!!") == 208
        assert vm2.eval_chunk("tempText(Actor[#ThermoUI])!!") == "69.4 °F (20.8 °C)"

    def test_thermometer_ui_window_and_surface_rendering(self):
        """Tests `ThermometerUI` / `ThermometerWindow` child controls, Wayland/WebSocket surface rendering, and HTML5 output."""
        from actorii.display.wayland import WaylandDisplay
        from websocket_server import create_surface

        vm = ActorVM()
        act_path = Path(_PROJECT_ROOT) / "examples" / "tp358w_digital_twin.act"
        vm.eval_chunk(act_path.read_text(encoding="utf-8"))

        # 1. Verify ThermometerWindow hierarchy (`ThermometerWindow` -> `ThermometerUI` -> `Window` -> `WindowsObject`)
        assert vm.eval_chunk("kindOf(Actor[#ThermoUI], ThermometerWindow)!!") is True
        assert vm.eval_chunk("kindOf(Actor[#ThermoUI], ThermometerUI)!!") is True
        assert vm.eval_chunk("kindOf(Actor[#ThermoUI], Window)!!") is True
        assert vm.eval_chunk("kindOf(Actor[#ThermoUI], WindowsObject)!!") is True

        # 2. Attach a WebSocketSurface to Actor[#ThermoUI] and ingest sensor data
        ws_surface = create_surface(800, 600, "TP358W Surface")
        ui_obj = vm.global_vars["Actor"]["ThermoUI"]
        ui_obj.attachSurface(ws_surface)

        vm.eval_chunk(f'ingestAdvertisementHex(Actor[#Thermometer], "{REAL_ADV_HEX}", -77)!!')
        vm.eval_chunk(
            f'ingestHistoryHex(Actor[#Thermometer], "{REAL_HISTORY_HEX}", "2026-10-27T01:05:00")!!'
        )

        # Verify Actor[#ThermoUI] automatically updated its child TextField controls via observer notification
        assert vm.eval_chunk("tempText(Actor[#ThermoUI])!!") == "69.4 °F (20.8 °C)"
        assert vm.eval_chunk("humidityText(Actor[#ThermoUI])!!") == "52% RH | Dew Point: 10.6 °C"
        assert "Comfort: #Comfort" in vm.eval_chunk("comfortText(Actor[#ThermoUI])!!")
        assert "History: 30m" in vm.eval_chunk("statsText(Actor[#ThermoUI])!!")
        assert "69.4 °F (20.8 °C)" in vm.eval_chunk("banner(Actor[#ThermoUI])!!")

        # 3. Verify pixel-level rendering on WebSocketSurface & headless WaylandSurface
        stats_ws = ui_obj.renderOnSurface(ws_surface)
        assert stats_ws["pixels_drawn"] > 50000
        assert stats_ws["mercury_color"] == "#E94560"
        assert stats_ws["humidity_color"] == "#10B981"
        assert stats_ws["history_points_plotted"] == 30
        # Verify caption bar pixel (#000080) and chart background pixel (#0F172A) on surface
        assert ws_surface.get_pixel(50, 40) == "#000080"
        assert ws_surface.get_pixel(250, 100) == "#0F172A"

        wl_display = WaylandDisplay(headless=True)
        wl_surface = wl_display.create_surface(800, 600)
        stats_wl = ui_obj.renderOnSurface(wl_surface)
        assert stats_wl["pixels_drawn"] > 50000
        assert wl_surface.get_pixel(50, 40).lower() == "#000080"

        # 4. Verify HTML5 + SVG draw() output from Actor
        html = vm.eval_chunk("draw(Actor[#ThermoUI])!!")
        assert "actor-thermometer-ui" in html
        assert "69.4°F" in html
        assert "52% RH" in html
        assert "<polyline" in html

    def test_websocket_workspace_thermometer_ui_and_actions(self):
        """Tests WebSocket `render_thermometer` and interactive `ui_event` unit toggle on `ActorWorkspace`."""
        import asyncio
        from websocket_server import ActorWorkspace

        async def _run():
            ws = ActorWorkspace()
            resp = await ws.handle_message(
                {
                    "type": "render_thermometer",
                    "advertisement_hex": REAL_ADV_HEX,
                    "history_hex": REAL_HISTORY_HEX,
                    "fetch_time": "2026-10-27T01:05:00",
                    "rssi": -77,
                }
            )
            assert resp["type"] == "thermometer_rendered"
            assert resp["temperature_c"] == 20.8
            assert resp["temperature_f"] == 69.4
            assert resp["humidity"] == 52
            assert resp["history_count"] == 30
            assert resp["temp_text"] == "69.4 °F (20.8 °C)"
            assert "actor-thermometer-ui" in resp["html"]
            assert ">69.4°F</text>" in resp["html"]

            # 1. Click the Fahrenheit/Celsius checkbox (`chkFahrenheit`) via `ui_event` -> switches to Celsius
            toggle_resp = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "chkFahrenheit",
                    "action": "#selected",
                }
            )
            assert toggle_resp["handled"] is True
            assert toggle_resp["result"] == "20.8 °C (69.4 °F)"
            assert ">20.8°C</text>" in toggle_resp["thermometer_html"]

            # 2. Click `chkFahrenheit` again -> switches back to Fahrenheit (including the bulb!)
            toggle_back_resp = await ws.handle_message(
                {
                    "type": "ui_event",
                    "widget_id": "chkFahrenheit",
                    "action": "#selected",
                }
            )
            assert toggle_back_resp["handled"] is True
            assert toggle_back_resp["result"] == "69.4 °F (20.8 °C)"
            assert ">69.4°F</text>" in toggle_back_resp["thermometer_html"]

        asyncio.run(_run())


@pytest.mark.skipif(
    not _is_bluez_adapter_powered(),
    reason="BlueZ adapter /org/bluez/hci0 is not powered on",
)
class TestLivePhysicalTP358WHardware:
    """Live BLE hardware test against 00:11:22:33:44:55 (`TP358S (Demo)`) on valkyrie."""

    def test_live_ble_sync_and_history_download_in_actor(self):
        vm = ActorVM()
        act_path = Path(_PROJECT_ROOT) / "examples" / "tp358w_digital_twin.act"
        vm.eval_chunk(act_path.read_text(encoding="utf-8"))

        # 1. Read live temperature & humidity from the physical thermometer via BLE advertisement
        vm.eval_chunk("syncLive(Actor[#Thermometer], 8)!!")
        live_temp_c = vm.eval_chunk("temperatureC(Actor[#Thermometer])!!")
        live_temp_f = vm.eval_chunk("temperatureF(Actor[#Thermometer])!!")
        live_hum = vm.eval_chunk("humidity(Actor[#Thermometer])!!")
        live_comfort = vm.eval_chunk("comfortLevel(Actor[#Thermometer])!!")

        assert isinstance(live_temp_c, float)
        assert -20.0 <= live_temp_c <= 60.0
        assert 0 <= live_hum <= 100
        assert live_comfort in (Symbol("Comfort"), Symbol("Dry"), Symbol("Wet"))

        # Verify Actor observers (`ComfortMonitor` & `HvacController`) received the live hardware reading
        assert vm.eval_chunk("lastTempC10(Actor[#Monitor])!!") == int(round(live_temp_c * 10))
        assert vm.eval_chunk("lastHumidity(Actor[#Monitor])!!") == live_hum
        assert vm.eval_chunk("updates(Actor[#Hvac])!!") >= 1

        # 2. Download historical temperature & humidity records over GATT (`0xCCCC` protocol)
        hist_count = vm.eval_chunk("downloadHistory(Actor[#Thermometer], 30)!!")
        assert hist_count >= 1
        assert vm.eval_chunk("historyCount(Actor[#Thermometer])!!") == hist_count

        min_f = vm.eval_chunk("minTempF(Actor[#Thermometer])!!")
        max_f = vm.eval_chunk("maxTempF(Actor[#Thermometer])!!")
        avg_h = vm.eval_chunk("avgHumidity(Actor[#Thermometer])!!")
        assert -4.0 <= min_f <= max_f <= 140.0
        assert 0 <= avg_h <= 100
