"""ThermoPro TP358 / TP358S / TP358W BLE Digital Twin for ActorII.

Implements deterministic Bluetooth Low Energy (BLE) telemetry & GATT history
acquisition for the ThermoPro TP358W / TP358S hygrometer-thermometer, wrapped
as a first-class 1993 Whitewater Actor object (`ThermoProTwin`) that other
Actor objects, GUI windows, and AI `Agent`s can message, observe, inspect,
and persist into `.img` image snapshots (`ACTOR.IMG`).

Hardware Protocol Specification (TP358W / TP358S):
1. Passive BLE Advertisement (`ManufacturerData` key `0xD0C2` / prefix `0xC2`):
   - Byte 0: `0xC2` magic prefix
   - Bytes 1..2: Signed 16-bit little-endian temperature * 10 (°C)
   - Byte 3: Relative humidity (% RH, uint8)
   - Byte 4: Battery indicator in bits 0..1 (`0` -> 1%, `1` -> 50%, `2` -> 100%)
2. Active GATT History Download:
   - Service UUID: `00010203-0405-0607-0809-0a0b0c0d1910`
   - Notify UUID:  `00010203-0405-0607-0809-0a0b0c0d2b10`
   - Write UUID:   `00010203-0405-0607-0809-0a0b0c0d2b11`
   - Step 1 (Clock Sync Handshake): `a5 YY MM DD HH MM SS DOW CS` (`CS = sum & 0xff`)
   - Step 2 (Session Init):         `cc cc 02 01 00 00 01 04 66 66`
   - Step 3 (Device Status Query):  `cc cc 04 00 00 00 00 04 66 66`
   - Step 4 (History Request):      `cc cc 01 09 00 00 00 YY MM DD HH MM SS NL NH CS 66 66`
   - Response Stream (`cc cc 01 [N2 N1 N0] 00 [TL TH HH]... [CS] 66 66`):
     3 bytes per 1-minute historical record (most-recent record first).
"""

from __future__ import annotations

import datetime
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from actorii.classes.collections import ActorOrderedCollection
from actorii.classes.dictionary import Dictionary
from actorii.classes.geometry import Rect
from actorii.classes.gui import Button, Checkbox, Control, TextField, Window
from actorii.core import ActorObject, Symbol

UUID_SERVICE = "00010203-0405-0607-0809-0a0b0c0d1910"
UUID_NOTIFY = "00010203-0405-0607-0809-0a0b0c0d2b10"
UUID_WRITE = "00010203-0405-0607-0809-0a0b0c0d2b11"

DEFAULT_TP358W_MAC = os.environ.get("ACTORII_THERMOPRO_MAC", "00:11:22:33:44:55")
DEFAULT_TP358W_NAME = os.environ.get("ACTORII_THERMOPRO_NAME", "TP358S (Demo)")

BATTERY_LEVEL_MAP = {
    0: 1,
    1: 50,
    2: 100,
    3: 100,
}


# =============================================================================
# 1. Deterministic Wire Protocol Encoders & Decoders
# =============================================================================


def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit rounded to 1 decimal place."""
    return round(temp_c * 9.0 / 5.0 + 32.0, 1)


def compute_dew_point_c(temp_c: float, humidity: int) -> float:
    """Compute dew point (°C) using the Magnus-Tetens approximation."""
    rh = max(1.0, min(100.0, float(humidity)))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    return round((b * alpha) / (a - alpha), 1)


def compute_comfort_symbol(humidity: int) -> Symbol:
    """Return authentic ThermoPro TP358W LCD comfort indicator (#Dry, #Comfort, #Wet)."""
    if humidity < 30:
        return Symbol("Dry")
    if humidity <= 60:
        return Symbol("Comfort")
    return Symbol("Wet")


def decode_advertisement_bytes(raw: bytes, rssi: Optional[int] = None) -> Dict[str, Any]:
    """Deterministically decode a 7-byte ThermoPro TP35x BLE advertisement payload.

    Example payload: `c2 d0 00 34 21 1b 01` -> 20.8°C (69.4°F), 52% RH, 50% battery.
    """
    if len(raw) < 5:
        raise ValueError(f"ThermoPro advertisement payload too short ({len(raw)} bytes): {raw.hex()}")
    temp_humi = raw[1:4]
    if temp_humi == b"\xff\xff\xff":
        raise ValueError("ThermoPro advertisement contains invalid sensor reading (0xffffff)")

    temp_c10 = int.from_bytes(raw[1:3], byteorder="little", signed=True)
    temp_c = round(temp_c10 / 10.0, 1)
    temp_f = c_to_f(temp_c)
    temp_f10 = int(round(temp_f * 10))
    humidity = int(raw[3])
    batt_code = int(raw[4]) & 0x03
    battery = BATTERY_LEVEL_MAP.get(batt_code, 100)

    return {
        "temperature_c": temp_c,
        "temperature_c10": temp_c10,
        "temperature_f": temp_f,
        "temperature_f10": temp_f10,
        "humidity": humidity,
        "battery": battery,
        "rssi": rssi,
        "comfort_level": compute_comfort_symbol(humidity),
        "dew_point_c": compute_dew_point_c(temp_c, humidity),
        "raw_hex": raw.hex(),
    }


def make_datetime_sync_cmd(now: Optional[datetime.datetime] = None) -> bytes:
    """Build the required `0xA5` datetime-sync handshake packet."""
    if now is None:
        now = datetime.datetime.now()
    body = bytes(
        [
            0xA5,
            now.year % 100,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            now.weekday() + 1,
        ]
    )
    return body + bytes([sum(body) & 0xFF])


def make_history_request_cmds(
    count: int = 60, now: Optional[datetime.datetime] = None
) -> Tuple[bytes, bytes, bytes]:
    """Build the 3-command sequence (`0xCCCC` framed) to request `count` 1-minute records."""
    if now is None:
        now = datetime.datetime.now()
    count_clamped = max(1, min(65535, int(count)))
    lo = count_clamped & 0xFF
    hi = (count_clamped >> 8) & 0xFF

    cmd1 = bytes.fromhex("cccc0201000001046666")
    cmd2 = bytes.fromhex("cccc0400000000046666")
    body = bytes(
        [
            0x01,
            0x09,
            0x00,
            0x00,
            0x00,
            now.year % 100,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            lo,
            hi,
        ]
    )
    cs = sum(body) & 0xFF
    cmd3 = b"\xcc\xcc" + body + bytes([cs]) + b"\x66\x66"
    return cmd1, cmd2, cmd3


def decode_history_stream(
    buf: bytes,
    fetch_time: Optional[datetime.datetime] = None,
    interval_minutes: int = 1,
) -> List[Dict[str, Any]]:
    """Deterministically decode a `cccc 01 ... 6666` GATT history stream into timestamped records.

    Returns records ordered oldest-to-newest.
    """
    idx = buf.find(b"\xcc\xcc\x01")
    if idx < 0:
        return []
    buf = buf[idx:]
    if len(buf) < 10:
        return []

    payload = buf[2:]
    if payload[-2:] == b"\x66\x66":
        payload = payload[:-2]

    # Skip 5-byte header (01 [N2 N1 N0] 00) and trailing 1-byte checksum
    triplets_raw = payload[5:-1]
    num_records = len(triplets_raw) // 3

    if fetch_time is None:
        fetch_time = datetime.datetime.now().replace(second=0, microsecond=0)

    records_newest_first: List[Dict[str, Any]] = []
    valid_idx = 0
    for i in range(num_records):
        triplet = triplets_raw[i * 3 : i * 3 + 3]
        if triplet in (b"\x00\x00\x00", b"\xff\xff\xff"):
            continue
        temp_c10 = int.from_bytes(triplet[0:2], byteorder="little", signed=True)
        hum = int(triplet[2])
        if temp_c10 > 1000 or temp_c10 < -500 or hum <= 0 or hum > 100:
            continue
        temp_c = round(temp_c10 / 10.0, 1)
        temp_f = c_to_f(temp_c)
        temp_f10 = int(round(temp_f * 10))
        ts = fetch_time - datetime.timedelta(minutes=valid_idx * interval_minutes)
        valid_idx += 1
        records_newest_first.append(
            {
                "timestamp": ts.isoformat(timespec="minutes"),
                "minutesAgo": (valid_idx - 1) * interval_minutes,
                "tempC": temp_c,
                "tempC10": temp_c10,
                "tempF": temp_f,
                "tempF10": temp_f10,
                "humidity": hum,
            }
        )

    records_newest_first.reverse()
    return records_newest_first


# =============================================================================
# 2. BlueZ D-Bus Hardware Transport for Linux (`valkyrie`)
# =============================================================================


class BlueZThermoProTransport:
    """Pure-Python BlueZ D-Bus transport for reading BLE advertisements and GATT history."""

    def __init__(
        self,
        mac_address: str = DEFAULT_TP358W_MAC,
        adapter_path: str = "/org/bluez/hci0",
    ) -> None:
        self.mac_address = mac_address.upper()
        self.adapter_path = adapter_path

    def scan_advertisement(self, timeout_sec: float = 12.0) -> Dict[str, Any]:
        """Scan BlueZ for the TP358W/TP358S device and decode its live advertisement."""
        from gi.repository import Gio, GLib

        bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        try:
            bus.call_sync(
                "org.bluez",
                self.adapter_path,
                "org.bluez.Adapter1",
                "SetDiscoveryFilter",
                GLib.Variant("(a{sv})", ({"Transport": GLib.Variant("s", "le")},)),
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )
            bus.call_sync(
                "org.bluez",
                self.adapter_path,
                "org.bluez.Adapter1",
                "StartDiscovery",
                None,
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )
        except Exception:
            pass

        deadline = time.time() + max(12.0, float(timeout_sec))
        found_info: Optional[Dict[str, Any]] = None
        fallback_dev_path: Optional[str] = None
        fallback_dev_name: str = DEFAULT_TP358W_NAME

        try:
            while time.time() < deadline:
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
                for path, ifaces in objs.items():
                    dev = ifaces.get("org.bluez.Device1")
                    if not dev:
                        continue
                    addr = str(dev.get("Address", "")).upper()
                    name = str(dev.get("Name", dev.get("Alias", "")))
                    if addr == self.mac_address or name.startswith("TP35"):
                        fallback_dev_path = path
                        if name:
                            fallback_dev_name = name
                        mfg = dev.get("ManufacturerData", {})
                        rssi = dev.get("RSSI")
                        if mfg:
                            last_key = list(mfg.keys())[-1]
                            raw = int(last_key).to_bytes(2, "little") + bytes(mfg[last_key])
                            decoded = decode_advertisement_bytes(raw, rssi=rssi)
                            decoded["device_path"] = path
                            decoded["mac_address"] = addr
                            decoded["device_name"] = name
                            found_info = decoded
                            break
                if found_info is not None:
                    break
                time.sleep(0.3)
        finally:
            try:
                bus.call_sync(
                    "org.bluez",
                    self.adapter_path,
                    "org.bluez.Adapter1",
                    "StopDiscovery",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    5000,
                    None,
                )
            except Exception:
                pass

        if found_info is None:
            raise RuntimeError(
                f"ThermoPro sensor ({self.mac_address}) not seen advertising within {timeout_sec}s"
            )
        return found_info

    def download_history(
        self, count: int = 60, scan_timeout: float = 12.0, transfer_timeout: float = 15.0
    ) -> Dict[str, Any]:
        """Connect to the TP358W/TP358S over GATT, sync clock, and download `count` history records."""
        from gi.repository import Gio, GLib

        bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)

        # 1. Check if BlueZ already knows the device path or scan for it
        adv_info: Dict[str, Any] = {}
        dev_path: Optional[str] = None
        try:
            adv_info = self.scan_advertisement(timeout_sec=scan_timeout)
            dev_path = adv_info.get("device_path")
        except Exception:
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
            expected_suffix = "dev_" + self.mac_address.replace(":", "_")
            for p in objs:
                if p.endswith(expected_suffix):
                    dev_path = p
                    break
            if dev_path is None:
                raise

        # 2. Connect to GATT server (with retry for BlueZ StopDiscovery -> Connect InProgress race)
        last_conn_err: Optional[Exception] = None
        for _attempt in range(6):
            try:
                bus.call_sync(
                    "org.bluez",
                    dev_path,
                    "org.bluez.Device1",
                    "Connect",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    20000,
                    None,
                )
                last_conn_err = None
                break
            except Exception as conn_exc:
                err_msg = str(conn_exc)
                if "AlreadyConnected" in err_msg:
                    last_conn_err = None
                    break
                last_conn_err = conn_exc
                if "InProgress" in err_msg or "Failed" in err_msg:
                    time.sleep(0.5)
                    continue
                raise
        if last_conn_err is not None:
            raise last_conn_err

        notify_path: Optional[str] = None
        write_path: Optional[str] = None
        for _ in range(50):
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
            for path, ifaces in objs.items():
                if path.startswith(dev_path):
                    ch = ifaces.get("org.bluez.GattCharacteristic1")
                    if ch:
                        u = str(ch.get("UUID", "")).lower()
                        if u == UUID_NOTIFY:
                            notify_path = path
                        elif u == UUID_WRITE:
                            write_path = path
            if notify_path and write_path:
                break
            time.sleep(0.2)

        if not notify_path or not write_path:
            try:
                bus.call_sync(
                    "org.bluez",
                    dev_path,
                    "org.bluez.Device1",
                    "Disconnect",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    5000,
                    None,
                )
            except Exception:
                pass
            raise RuntimeError(f"Failed to resolve GATT characteristics on {dev_path}")

        ctx = GLib.MainContext.default()
        history_chunks: List[bytes] = []
        live_notification: Dict[str, Any] = {}
        stream_complete = False

        def on_props_changed(conn, sender, path, iface, sig, params):
            nonlocal stream_complete
            if path != notify_path:
                return
            changed = params.unpack()[1]
            if "Value" not in changed:
                return
            d = bytes(changed["Value"])
            if not d:
                return
            # 0xC2 live reading notification
            if d[0] == 0xC2 and len(d) >= 6:
                tc10 = int.from_bytes(d[3:5], "little", signed=True)
                tc = round(tc10 / 10.0, 1)
                live_notification["temperature_c"] = tc
                live_notification["temperature_c10"] = tc10
                live_notification["temperature_f"] = c_to_f(tc)
                live_notification["temperature_f10"] = int(round(c_to_f(tc) * 10))
                live_notification["humidity"] = int(d[5])
                return

            # 0xCCCC 0x01 history stream (ignore 0xCCCC 0x04 status echo)
            if len(d) >= 3 and d[:2] == b"\xcc\xcc":
                if d[2] == 0x01:
                    history_chunks.clear()
                    history_chunks.append(d)
                else:
                    return
            elif history_chunks:
                history_chunks.append(d)

            if history_chunks and history_chunks[-1][-2:] == b"\x66\x66":
                stream_complete = True

        sub_id = bus.signal_subscribe(
            "org.bluez",
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            notify_path,
            None,
            Gio.DBusSignalFlags.NONE,
            on_props_changed,
        )

        def pump(duration_sec: float) -> None:
            end = time.time() + duration_sec
            while time.time() < end:
                while ctx.pending():
                    ctx.iteration(False)
                if stream_complete:
                    break
                time.sleep(0.005)

        def write_gatt(payload: bytes) -> None:
            bus.call_sync(
                "org.bluez",
                write_path,
                "org.bluez.GattCharacteristic1",
                "WriteValue",
                GLib.Variant(
                    "(aya{sv})", (list(payload), {"type": GLib.Variant("s", "command")})
                ),
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )

        now = datetime.datetime.now()
        fetch_time = now.replace(second=0, microsecond=0)

        try:
            bus.call_sync(
                "org.bluez",
                notify_path,
                "org.bluez.GattCharacteristic1",
                "StartNotify",
                None,
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )

            # Step 1: 0xA5 Datetime handshake
            write_gatt(make_datetime_sync_cmd(now))
            pump(1.0)

            # Step 2: 3-command 0xCCCC history request
            for cmd in make_history_request_cmds(count=count, now=now):
                write_gatt(cmd)
                pump(0.22)

            # Step 3: Pump event loop until history stream ends with 0x6666
            pump(transfer_timeout)
        finally:
            try:
                bus.call_sync(
                    "org.bluez",
                    notify_path,
                    "org.bluez.GattCharacteristic1",
                    "StopNotify",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    5000,
                    None,
                )
            except Exception:
                pass
            bus.signal_unsubscribe(sub_id)
            try:
                bus.call_sync(
                    "org.bluez",
                    dev_path,
                    "org.bluez.Device1",
                    "Disconnect",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    5000,
                    None,
                )
            except Exception:
                pass

        raw_stream = b"".join(history_chunks)
        records = decode_history_stream(raw_stream, fetch_time=fetch_time)
        return {
            "advertisement": adv_info,
            "live_notification": live_notification,
            "raw_stream_hex": raw_stream.hex(),
            "records": records,
            "fetch_time": fetch_time.isoformat(timespec="seconds"),
        }


# =============================================================================
# 3. First-Class `ThermoProTwin` Actor Object (1993 Actor Digital Twin)
# =============================================================================


class ThermoProTwin(ActorObject):
    """Digital Twin of the ThermoPro TP358W / TP358S Bluetooth Thermometer in ActorII."""

    def __init__(
        self,
        mac_address: str = DEFAULT_TP358W_MAC,
        device_name: str = DEFAULT_TP358W_NAME,
        model: str = "TP358W",
        obj_id: Optional[str] = None,
        class_name: str = "ThermoProTwin",
        vm: Optional[Any] = None,
    ) -> None:
        super().__init__(obj_id=obj_id)
        self._class_name = class_name
        self.vm = vm

        self._history = ActorOrderedCollection()
        self._observers = ActorOrderedCollection()

        self._fields["macAddress"] = str(mac_address).upper()
        self._fields["deviceName"] = str(device_name)
        self._fields["model"] = str(model)
        self._fields["temperatureC"] = 0.0
        self._fields["temperatureC10"] = 0
        self._fields["temperatureF"] = 32.0
        self._fields["temperatureF10"] = 320
        self._fields["humidity"] = 0
        self._fields["battery"] = 100
        self._fields["rssi"] = -70
        self._fields["comfortLevel"] = Symbol("Comfort")
        self._fields["dewPointC"] = 0.0
        self._fields["lastSyncTime"] = None
        self._fields["lastRawHex"] = ""
        self._fields["history"] = self._history
        self._fields["observers"] = self._observers

    # =========================================================================
    # Synchronized Collections & State Helpers
    # =========================================================================

    def history(self) -> ActorOrderedCollection:
        h = self._fields.get("history")
        if isinstance(h, ActorOrderedCollection):
            self._history = h
        return self._history

    def observers(self) -> ActorOrderedCollection:
        obs = self._fields.get("observers")
        if isinstance(obs, ActorOrderedCollection):
            self._observers = obs
        return self._observers

    def _apply_reading(
        self,
        temp_c: float,
        humidity: int,
        battery: Optional[int] = None,
        rssi: Optional[int] = None,
        sync_time: Optional[str] = None,
        raw_hex: Optional[str] = None,
    ) -> None:
        tc = round(float(temp_c), 1)
        tc10 = int(round(tc * 10))
        tf = c_to_f(tc)
        tf10 = int(round(tf * 10))
        hum = int(humidity)

        self._fields["temperatureC"] = tc
        self._fields["temperatureC10"] = tc10
        self._fields["temperatureF"] = tf
        self._fields["temperatureF10"] = tf10
        self._fields["humidity"] = hum
        self._fields["comfortLevel"] = compute_comfort_symbol(hum)
        self._fields["dewPointC"] = compute_dew_point_c(tc, hum)
        if battery is not None:
            self._fields["battery"] = int(battery)
        if rssi is not None:
            self._fields["rssi"] = int(rssi)
        self._fields["lastSyncTime"] = sync_time or datetime.datetime.now().isoformat(
            timespec="seconds"
        )
        if raw_hex is not None:
            self._fields["lastRawHex"] = str(raw_hex)

    def _append_record_dict(self, rec: Dict[str, Any]) -> Dictionary:
        d = Dictionary(16)
        for k, v in rec.items():
            d.atPut(k, v)
            d.atPut(Symbol(k), v)
        self.history().add(d)
        return d

    # =========================================================================
    # Deterministic Ingestion & Hardware Sync Methods
    # =========================================================================

    def ingestAdvertisementHex(
        self, hex_str: str, rssi: int = -75, timestamp: Optional[str] = None
    ) -> "ThermoProTwin":
        """Deterministically decode and apply a raw BLE advertisement hex frame."""
        raw = bytes.fromhex(str(hex_str).strip().replace(" ", ""))
        decoded = decode_advertisement_bytes(raw, rssi=rssi)
        self._apply_reading(
            temp_c=decoded["temperature_c"],
            humidity=decoded["humidity"],
            battery=decoded["battery"],
            rssi=decoded["rssi"],
            sync_time=timestamp,
            raw_hex=decoded["raw_hex"],
        )
        self.notifyObservers()
        return self

    def ingestHistoryHex(
        self, hex_str: str, fetch_time_iso: Optional[str] = None
    ) -> int:
        """Deterministically decode a `cccc01...6666` GATT history hex stream into `self.history`."""
        raw = bytes.fromhex(str(hex_str).strip().replace(" ", ""))
        ft = (
            datetime.datetime.fromisoformat(fetch_time_iso)
            if fetch_time_iso
            else datetime.datetime.now().replace(second=0, microsecond=0)
        )
        records = decode_history_stream(raw, fetch_time=ft)
        self.history().clear()
        for r in records:
            self._append_record_dict(r)
        if records:
            latest = records[-1]
            self._apply_reading(
                temp_c=latest["tempC"],
                humidity=latest["humidity"],
                sync_time=latest["timestamp"],
                raw_hex=raw.hex(),
            )
            self.notifyObservers()
        return len(records)

    def syncLive(self, timeout_sec: float = 8.0) -> "ThermoProTwin":
        """Scan BlueZ for live BLE advertisement from the physical TP358W and update twin state."""
        transport = BlueZThermoProTransport(mac_address=self.macAddress())
        adv = transport.scan_advertisement(timeout_sec=float(timeout_sec))
        if adv.get("device_name"):
            self._fields["deviceName"] = adv["device_name"]
        if adv.get("mac_address") and self._fields.get("macAddress") in ("", "00:11:22:33:44:55"):
            if os.environ.get("ACTORII_THERMOPRO_ADOPT_SCAN_MAC") == "1":
                self._fields["macAddress"] = adv["mac_address"]
        self._apply_reading(
            temp_c=adv["temperature_c"],
            humidity=adv["humidity"],
            battery=adv["battery"],
            rssi=adv["rssi"],
            raw_hex=adv["raw_hex"],
        )
        self.notifyObservers()
        return self

    def readLive(self, timeout_sec: float = 8.0) -> "ThermoProTwin":
        """Alias for `syncLive`."""
        return self.syncLive(timeout_sec=timeout_sec)

    def downloadHistory(
        self, count: int = 60, scan_timeout: float = 8.0, transfer_timeout: float = 15.0
    ) -> int:
        """Connect to the physical TP358W via GATT, download `count` minute records, and notify observers."""
        transport = BlueZThermoProTransport(mac_address=self.macAddress())
        result = transport.download_history(
            count=int(count),
            scan_timeout=float(scan_timeout),
            transfer_timeout=float(transfer_timeout),
        )
        adv = result.get("advertisement", {})
        if adv.get("device_name"):
            self._fields["deviceName"] = adv["device_name"]
        if adv.get("mac_address") and self._fields.get("macAddress") in ("", "00:11:22:33:44:55"):
            if os.environ.get("ACTORII_THERMOPRO_ADOPT_SCAN_MAC") == "1":
                self._fields["macAddress"] = adv["mac_address"]

        records = result.get("records", [])
        self.history().clear()
        for r in records:
            self._append_record_dict(r)

        live_notif = result.get("live_notification", {})
        if live_notif:
            self._apply_reading(
                temp_c=live_notif["temperature_c"],
                humidity=live_notif["humidity"],
                battery=adv.get("battery"),
                rssi=adv.get("rssi"),
                sync_time=result.get("fetch_time"),
                raw_hex=result.get("raw_stream_hex"),
            )
        elif records:
            latest = records[-1]
            self._apply_reading(
                temp_c=latest["tempC"],
                humidity=latest["humidity"],
                battery=adv.get("battery"),
                rssi=adv.get("rssi"),
                sync_time=result.get("fetch_time"),
                raw_hex=result.get("raw_stream_hex"),
            )
        elif adv:
            self._apply_reading(
                temp_c=adv["temperature_c"],
                humidity=adv["humidity"],
                battery=adv["battery"],
                rssi=adv["rssi"],
                sync_time=result.get("fetch_time"),
            )

        self.notifyObservers()
        return len(records)

    # =========================================================================
    # 1993 Actor Object Interface (Accessors, Analytics & Observer Pattern)
    # =========================================================================

    def macAddress(self) -> str:
        return str(self._fields.get("macAddress", DEFAULT_TP358W_MAC))

    def setMacAddress(self, mac: str) -> "ThermoProTwin":
        self._fields["macAddress"] = str(mac).upper()
        return self

    def deviceName(self) -> str:
        return str(self._fields.get("deviceName", DEFAULT_TP358W_NAME))

    def model(self) -> str:
        return str(self._fields.get("model", "TP358W"))

    def _ensure_initial_sync(self, timeout_sec: float = 5.0, max_age_sec: float = 300.0) -> None:
        """Auto-sync via BlueZ BLE if no live reading/history has been ingested yet or if the reading is stale (>5 min)."""
        last_sync = self._fields.get("lastSyncTime")
        is_stale = False
        if last_sync and len(self.history()) == 0:
            try:
                dt = datetime.datetime.fromisoformat(str(last_sync))
                if (datetime.datetime.now() - dt).total_seconds() > max_age_sec:
                    is_stale = True
            except Exception:
                pass
        if (
            last_sync is None
            and len(self.history()) == 0
            and float(self._fields.get("temperatureC", 0.0)) == 0.0
            and int(self._fields.get("humidity", 0)) == 0
        ) or is_stale:
            try:
                self.syncLive(timeout_sec=timeout_sec)
            except Exception:
                pass

    def temperatureC(self) -> float:
        self._ensure_initial_sync()
        return float(self._fields.get("temperatureC", 0.0))

    def temperatureC10(self) -> int:
        self._ensure_initial_sync()
        return int(self._fields.get("temperatureC10", 0))

    def temperatureF(self) -> float:
        self._ensure_initial_sync()
        return float(self._fields.get("temperatureF", 32.0))

    def temperatureF10(self) -> int:
        self._ensure_initial_sync()
        return int(self._fields.get("temperatureF10", 320))

    def humidity(self) -> int:
        self._ensure_initial_sync()
        return int(self._fields.get("humidity", 0))

    def battery(self) -> int:
        return int(self._fields.get("battery", 100))

    def rssi(self) -> int:
        return int(self._fields.get("rssi", -70))

    def comfortLevel(self) -> Symbol:
        val = self._fields.get("comfortLevel")
        return val if isinstance(val, Symbol) else Symbol(str(val or "Comfort"))

    def dewPointC(self) -> float:
        return float(self._fields.get("dewPointC", 0.0))

    def lastSyncTime(self) -> Optional[str]:
        val = self._fields.get("lastSyncTime")
        return str(val) if val is not None else None

    def historyCount(self) -> int:
        return len(self.history())

    def recordAt(self, index: int) -> Optional[Dictionary]:
        h = self.history()
        idx = int(index)
        if 0 <= idx < len(h):
            return h[idx]
        return None

    def latestRecord(self) -> Optional[Dictionary]:
        h = self.history()
        return h[-1] if len(h) > 0 else None

    def oldestRecord(self) -> Optional[Dictionary]:
        h = self.history()
        return h[0] if len(h) > 0 else None

    def _extract_history_metric(self, key: str, fallback: Any) -> List[Any]:
        vals = []
        for rec in self.history():
            if isinstance(rec, Dictionary):
                v = rec.at(key)
                if v is not None:
                    vals.append(v)
            elif isinstance(rec, dict) and key in rec:
                vals.append(rec[key])
        return vals if vals else [fallback]

    def minTempC(self) -> float:
        return float(min(self._extract_history_metric("tempC", self.temperatureC())))

    def maxTempC(self) -> float:
        return float(max(self._extract_history_metric("tempC", self.temperatureC())))

    def avgTempC(self) -> float:
        vals = self._extract_history_metric("tempC", self.temperatureC())
        return round(sum(vals) / len(vals), 2)

    def minTempF(self) -> float:
        return float(min(self._extract_history_metric("tempF", self.temperatureF())))

    def maxTempF(self) -> float:
        return float(max(self._extract_history_metric("tempF", self.temperatureF())))

    def avgTempF(self) -> float:
        vals = self._extract_history_metric("tempF", self.temperatureF())
        return round(sum(vals) / len(vals), 2)

    def minTempC10(self) -> int:
        return int(min(self._extract_history_metric("tempC10", self.temperatureC10())))

    def maxTempC10(self) -> int:
        return int(max(self._extract_history_metric("tempC10", self.temperatureC10())))

    def minHumidity(self) -> int:
        return int(min(self._extract_history_metric("humidity", self.humidity())))

    def maxHumidity(self) -> int:
        return int(max(self._extract_history_metric("humidity", self.humidity())))

    def avgHumidity(self) -> int:
        vals = self._extract_history_metric("humidity", self.humidity())
        return int(round(sum(vals) / len(vals)))

    def addObserver(self, observer: Any) -> "ThermoProTwin":
        """Register an Actor object to receive `onSensorUpdate(observer, self)` messages."""
        obs = self.observers()
        if observer not in obs:
            obs.add(observer)
        return self

    def removeObserver(self, observer: Any) -> "ThermoProTwin":
        obs = self.observers()
        if observer in obs:
            obs.remove(observer)
        return self

    def notifyObservers(self) -> int:
        """Send `onSensorUpdate(observer, self)` to all registered Actor observers."""
        from actorii.vm import ActorVM, BlockClosure

        active_vm = self.vm or getattr(ActorVM, "_active_vm", None)
        notified = 0
        for obs in list(self.observers()):
            if obs is None:
                continue
            if isinstance(obs, BlockClosure) and active_vm is not None:
                active_vm._call_block_closure(obs, [self])
                notified += 1
            elif active_vm is not None:
                active_vm.dispatch_message(obs, "onSensorUpdate", [self])
                notified += 1
            elif hasattr(obs, "onSensorUpdate") and callable(obs.onSensorUpdate):
                obs.onSensorUpdate(self)
                notified += 1
        return notified

    def summary(self) -> str:
        self._ensure_initial_sync()
        return (
            f"<{self.class_name} {self.deviceName()} [{self.macAddress()}] "
            f"Temp={self.temperatureC():.1f}C/{self.temperatureF():.1f}F "
            f"Hum={self.humidity()}% ({self.comfortLevel()}) "
            f"DewPoint={self.dewPointC():.1f}C Batt={self.battery()}% "
            f"History={self.historyCount()}pts>"
        )

    def renderOnSurface(
        self,
        surface: Any,
        x: int = 30,
        y: int = 30,
        width: int = 740,
        height: int = 520,
    ) -> Dict[str, Any]:
        """Render this ThermoProTwin's thermometer, hygrometer, and history graph onto an Actor surface."""
        return render_thermometer_on_surface(
            self, surface, x=int(x), y=int(y), width=int(width), height=int(height)
        )

    def asString(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# 4. Actor Surface Renderer & `ThermometerUI` Window Class
# =============================================================================


def _fill_rect(
    surface: Any, x: int, y: int, w: int, h: int, color: str
) -> int:
    """Fill a rectangle on any surface supporting `set_pixel(x, y, color)`."""
    if not hasattr(surface, "set_pixel"):
        return 0
    sw = int(getattr(surface, "width", 800))
    sh = int(getattr(surface, "height", 600))
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(sw, int(x + w))
    y1 = min(sh, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return 0

    # Fast path if surface has _pixel_data (WebSocketSurface) or _pixels (HTML5Surface)
    pdata = getattr(surface, "_pixel_data", None) or getattr(surface, "_pixels", None)
    count = (x1 - x0) * (y1 - y0)
    if isinstance(pdata, list) and len(pdata) >= y1:
        for py in range(y0, y1):
            row = pdata[py]
            for px in range(x0, x1):
                row[px] = color
        if hasattr(surface, "_dirty"):
            surface._dirty = True
        return count

    for py in range(y0, y1):
        for px in range(x0, x1):
            surface.set_pixel(px, py, color)
    return count


def _draw_line(
    surface: Any, x0: int, y0: int, x1: int, y1: int, color: str, thickness: int = 2
) -> int:
    """Bresenham line drawing onto an Actor surface."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cx, cy = x0, y0
    drawn = 0
    while True:
        drawn += _fill_rect(surface, cx, cy, thickness, thickness, color)
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            cx += sx
        if e2 <= dx:
            err += dx
            cy += sy
    return drawn


def render_thermometer_on_surface(
    twin: ThermoProTwin,
    surface: Any,
    x: int = 30,
    y: int = 30,
    width: int = 740,
    height: int = 520,
) -> Dict[str, Any]:
    """Render the ThermoProTwin thermometer tube, humidity gauge, and history plot onto `surface`."""
    pixels_drawn = 0

    # 1. Outer 1993 Actor Window Frame & Title Bar
    pixels_drawn += _fill_rect(surface, x, y, width, height, "#ECE9D8")
    pixels_drawn += _fill_rect(surface, x, y, width, 2, "#FFFFFF")
    pixels_drawn += _fill_rect(surface, x, y + height - 2, width, 2, "#404040")
    pixels_drawn += _fill_rect(surface, x, y, 2, height, "#FFFFFF")
    pixels_drawn += _fill_rect(surface, x + width - 2, y, 2, height, "#404040")
    # Caption bar (#000080)
    pixels_drawn += _fill_rect(surface, x + 3, y + 3, width - 6, 24, "#000080")

    # 2. Left Instrument: Mercury Thermometer Tube & Bulb
    tube_x = x + 42
    tube_y = y + 52
    tube_w = 28
    tube_h = 260
    # Glass tube background & border
    pixels_drawn += _fill_rect(surface, tube_x - 2, tube_y - 2, tube_w + 4, tube_h + 4, "#334155")
    pixels_drawn += _fill_rect(surface, tube_x, tube_y, tube_w, tube_h, "#F8FAFC")

    # Scale temperature (-10C .. 45C) into mercury height
    temp_c = twin.temperatureC()
    norm_t = max(0.0, min(1.0, (temp_c - (-10.0)) / 55.0))
    merc_h = max(8, int(round(norm_t * (tube_h - 8))))
    merc_y = tube_y + tube_h - merc_h
    merc_color = "#38BDF8" if temp_c < 18.0 else ("#EF4444" if temp_c > 25.0 else "#E94560")
    pixels_drawn += _fill_rect(surface, tube_x + 4, merc_y, tube_w - 8, merc_h, merc_color)

    # Thermometer bulb at bottom
    bulb_x = tube_x - 8
    bulb_y = tube_y + tube_h - 4
    bulb_size = tube_w + 16
    pixels_drawn += _fill_rect(surface, bulb_x - 2, bulb_y - 2, bulb_size + 4, bulb_size + 4, "#334155")
    pixels_drawn += _fill_rect(surface, bulb_x, bulb_y, bulb_size, bulb_size, merc_color)

    # Tick marks along the thermometer tube
    for tick_i in range(6):
        ty = tube_y + int(tick_i * (tube_h / 5))
        pixels_drawn += _fill_rect(surface, tube_x + tube_w + 3, ty, 8, 2, "#1E293B")

    # 3. Middle Instrument: Hygrometer Relative Humidity Bar (0% .. 100% RH)
    hum_x = x + 118
    hum_y = y + 52
    hum_w = 28
    hum_h = 260
    pixels_drawn += _fill_rect(surface, hum_x - 2, hum_y - 2, hum_w + 4, hum_h + 4, "#334155")
    pixels_drawn += _fill_rect(surface, hum_x, hum_y, hum_w, hum_h, "#F8FAFC")

    hum_val = max(0, min(100, twin.humidity()))
    hum_fill_h = max(6, int(round((hum_val / 100.0) * (hum_h - 4))))
    hum_fill_y = hum_y + hum_h - hum_fill_h
    comfort_sym = str(twin.comfortLevel())
    if "Dry" in comfort_sym:
        hum_color = "#F59E0B"
    elif "Wet" in comfort_sym:
        hum_color = "#3B82F6"
    else:
        hum_color = "#10B981"
    pixels_drawn += _fill_rect(surface, hum_x + 4, hum_fill_y, hum_w - 8, hum_fill_h, hum_color)

    # Comfort Indicator Badge below gauges
    badge_x = x + 34
    badge_y = bulb_y + bulb_size + 14
    pixels_drawn += _fill_rect(surface, badge_x, badge_y, 124, 28, hum_color)

    # 4. Right Panel: Historical Temperature & Humidity Chart Box
    chart_x = x + 185
    chart_y = y + 52
    chart_w = max(120, width - 210)
    chart_h = 260
    pixels_drawn += _fill_rect(surface, chart_x - 2, chart_y - 2, chart_w + 4, chart_h + 4, "#475569")
    pixels_drawn += _fill_rect(surface, chart_x, chart_y, chart_w, chart_h, "#0F172A")

    # Horizontal reference grid lines in chart
    for g_i in range(1, 4):
        gy = chart_y + int(g_i * (chart_h / 4))
        pixels_drawn += _fill_rect(surface, chart_x + 4, gy, chart_w - 8, 1, "#1E293B")

    records = list(twin.history())
    if records:
        t_vals = [float(r.at("tempC") if hasattr(r, "at") else r["tempC"]) for r in records]
        h_vals = [int(r.at("humidity") if hasattr(r, "at") else r["humidity"]) for r in records]
        t_min = min(t_vals) - 0.5
        t_max = max(t_vals) + 0.5
        if t_max <= t_min:
            t_max = t_min + 1.0

        n_pts = len(records)
        prev_tx, prev_ty = None, None
        prev_hx, prev_hy = None, None
        for idx, (tv, hv) in enumerate(zip(t_vals, h_vals)):
            px = chart_x + 10 + int(round((idx / max(1, n_pts - 1)) * (chart_w - 20)))
            ty = chart_y + chart_h - 12 - int(round(((tv - t_min) / (t_max - t_min)) * (chart_h - 24)))
            hy = chart_y + chart_h - 12 - int(round((hv / 100.0) * (chart_h - 24)))
            if prev_tx is not None and prev_ty is not None:
                pixels_drawn += _draw_line(surface, prev_tx, prev_ty, px, ty, "#E94560", thickness=2)
                pixels_drawn += _draw_line(surface, prev_hx, prev_hy, px, hy, "#1B9AAA", thickness=2)
            else:
                pixels_drawn += _fill_rect(surface, px - 1, ty - 1, 3, 3, "#E94560")
                pixels_drawn += _fill_rect(surface, px - 1, hy - 1, 3, 3, "#1B9AAA")
            prev_tx, prev_ty = px, ty
            prev_hx, prev_hy = px, hy

    if hasattr(surface, "damage"):
        surface.damage(x, y, width, height)
    if hasattr(surface, "flush"):
        surface.flush()

    return {
        "pixels_drawn": pixels_drawn,
        "mercury_color": merc_color,
        "mercury_height": merc_h,
        "humidity_color": hum_color,
        "humidity_fill_height": hum_fill_h,
        "history_points_plotted": len(records),
        "bounds": [x, y, width, height],
    }


class ThermometerUI(Window):
    """First-Class 1993 Actor GUI Window for the ThermoPro TP358W Digital Twin (`ThermoProTwin`).

    Combines:
    1. Authentic 1993 Actor child controls (`TextField` digital readouts, `Button` BLE triggers,
       `Checkbox` °F/°C unit selector).
    2. Automatic Actor Observer synchronization (`onSensorUpdate(self, twin)`).
    3. Pixel-level surface rendering (`renderOnSurface`) + HTML5/SVG visual gauge rendering (`draw`).
    """

    def __init__(
        self,
        twin: Optional[ThermoProTwin] = None,
        title: str = "ThermoPro TP358W Digital Twin",
        rect: Union[Rect, Tuple[int, int, int, int], None] = None,
        left: int = 30,
        top: int = 30,
        width: int = 740,
        height: int = 520,
        obj_id: Optional[str] = None,
        class_name: str = "ThermometerUI",
        vm: Optional[Any] = None,
    ) -> None:
        if isinstance(twin, str) and not isinstance(title, ThermoProTwin):
            # Support `new(ThermometerUI, "Custom Title", rect)` signature
            title = twin
            twin = None
        super().__init__(
            title=title,
            rect=rect,
            left=left,
            top=top,
            width=width,
            height=height,
            identity=Symbol("ThermoUI"),
            obj_id=obj_id,
        )
        self._class_name = class_name
        self.vm = vm
        self.surface: Optional[Any] = None
        self.use_fahrenheit: bool = True

        # Create standard 1993 Actor child controls inside this Window
        self.temp_field = TextField(
            text="--.- C / --.- F",
            left=185,
            top=330,
            width=250,
            height=28,
            id="tempField",
            identity=Symbol("tempField"),
        )
        self.humidity_field = TextField(
            text="--% RH (DewPt --.- C)",
            left=450,
            top=330,
            width=255,
            height=28,
            id="humidityField",
            identity=Symbol("humidityField"),
        )
        self.comfort_field = TextField(
            text="STATUS: #Comfort",
            left=185,
            top=366,
            width=520,
            height=26,
            id="comfortField",
            identity=Symbol("comfortField"),
        )
        self.stats_field = TextField(
            text="History: 0 pts",
            left=185,
            top=400,
            width=520,
            height=26,
            id="statsField",
            identity=Symbol("statsField"),
        )
        self.btn_sync = Button(
            label="Sync Live BLE",
            action_selector=Symbol("syncClicked"),
            left=185,
            top=440,
            width=155,
            height=32,
            id="btnSync",
            identity=Symbol("btnSync"),
            is_default=True,
            target=self,
            callback=lambda _b: self.onSyncClicked(),
        )
        self.btn_history = Button(
            label="Download 60m History",
            action_selector=Symbol("historyClicked"),
            left=355,
            top=440,
            width=175,
            height=32,
            id="btnHistory",
            identity=Symbol("btnHistory"),
            target=self,
            callback=lambda _b: self.onHistoryClicked(),
        )
        self.chk_unit = Checkbox(
            label="Fahrenheit Primary",
            checked=True,
            left=545,
            top=444,
            width=160,
            height=26,
            id="chkFahrenheit",
            identity=Symbol("chkFahrenheit"),
        )

        self.add(self.temp_field)
        self.add(self.humidity_field)
        self.add(self.comfort_field)
        self.add(self.stats_field)
        self.add(self.btn_sync)
        self.add(self.btn_history)
        self.add(self.chk_unit)

        self.register_action("btnSync", "selected", "onSyncClicked")
        self.register_action("btnSync", "syncClicked", "onSyncClicked")
        self.register_action("btnHistory", "selected", "onHistoryClicked")
        self.register_action("btnHistory", "historyClicked", "onHistoryClicked")
        self.register_action("chkFahrenheit", "toggled", "onUnitToggled")
        self.register_action("chkFahrenheit", "checked", "onUnitToggled")
        self.register_action("chkFahrenheit", "unChecked", "onUnitToggled")

        self._fields["twin"] = None
        self._fields["tempText"] = self.temp_field.getText()
        self._fields["humidityText"] = self.humidity_field.getText()
        self._fields["comfortText"] = self.comfort_field.getText()
        self._fields["statsText"] = self.stats_field.getText()
        self._fields["useFahrenheit"] = True
        self._fields["updateCount"] = 0

        if isinstance(twin, ThermoProTwin):
            self.attachTwin(twin)

    def twin(self) -> Optional[ThermoProTwin]:
        t = self._fields.get("twin")
        return t if isinstance(t, ThermoProTwin) else None

    def attachTwin(self, twin: ThermoProTwin) -> "ThermometerUI":
        """Bind this UI window to a `ThermoProTwin` and subscribe to sensor updates."""
        self._fields["twin"] = twin
        twin.addObserver(self)
        self.refreshFromTwin(twin)
        return self

    def attachSurface(self, surface: Any) -> "ThermometerUI":
        """Attach an Actor display surface (`WebSocketSurface`, `HTML5Surface`, `WaylandSurface`)."""
        self.surface = surface
        if hasattr(surface, "register_object"):
            try:
                surface.register_object(self)
            except Exception:
                pass
        t = self.twin()
        if t is not None:
            self.renderOnSurface(surface)
        return self

    def refreshFromTwin(self, twin: Optional[ThermoProTwin] = None) -> "ThermometerUI":
        """Update all child controls and fields from the bound `ThermoProTwin`."""
        t = twin or self.twin()
        if t is None:
            return self

        use_f = bool(self.chk_unit.value())
        self.use_fahrenheit = use_f
        self._fields["useFahrenheit"] = use_f

        tc = t.temperatureC()
        tf = t.temperatureF()
        hum = t.humidity()
        dew = t.dewPointC()
        comfort = t.comfortLevel()
        batt = t.battery()
        rssi = t.rssi()
        hcount = t.historyCount()

        if use_f:
            temp_str = f"{tf:.1f} °F ({tc:.1f} °C)"
        else:
            temp_str = f"{tc:.1f} °C ({tf:.1f} °F)"

        hum_str = f"{hum}% RH | Dew Point: {dew:.1f} °C"
        comfort_str = (
            f"Comfort: #{comfort} | Device: {t.deviceName()} [{t.macAddress()}] "
            f"| Batt: {batt}% | RSSI: {rssi} dBm"
        )
        stats_str = (
            f"History: {hcount}m | Min: {t.minTempC():.1f}°C ({t.minTempF():.1f}°F) "
            f"| Max: {t.maxTempC():.1f}°C ({t.maxTempF():.1f}°F) | Avg RH: {t.avgHumidity()}%"
        )

        self.temp_field.setText(temp_str)
        self.humidity_field.setText(hum_str)
        self.comfort_field.setText(comfort_str)
        self.stats_field.setText(stats_str)

        self._fields["tempText"] = temp_str
        self._fields["humidityText"] = hum_str
        self._fields["comfortText"] = comfort_str
        self._fields["statsText"] = stats_str
        self._fields["updateCount"] = int(self._fields.get("updateCount", 0)) + 1

        if self.surface is not None:
            self.renderOnSurface(self.surface)
        return self

    def onSensorUpdate(self, twin: ThermoProTwin) -> "ThermometerUI":
        """Observer callback invoked automatically whenever `twin` updates."""
        return self.refreshFromTwin(twin)

    def onSyncClicked(self, msg: Any = None) -> Any:
        """Action handler for the 'Sync Live BLE' button."""
        t = self.twin()
        if t is not None:
            t.syncLive(6.0)
            self.refreshFromTwin(t)
        return self.tempText()

    def onHistoryClicked(self, msg: Any = None) -> Any:
        """Action handler for the 'Download 60m History' button."""
        t = self.twin()
        if t is not None:
            cnt = t.downloadHistory(60)
            self.refreshFromTwin(t)
            return cnt
        return 0

    def onUnitToggled(self, msg: Any = None) -> str:
        """Action handler when the Fahrenheit/Celsius checkbox is toggled."""
        self.refreshFromTwin()
        return self.tempText()

    def tempText(self) -> str:
        return self.temp_field.getText()

    def humidityText(self) -> str:
        return self.humidity_field.getText()

    def comfortText(self) -> str:
        return self.comfort_field.getText()

    def statsText(self) -> str:
        return self.stats_field.getText()

    def updateCount(self) -> int:
        return int(self._fields.get("updateCount", 0))

    def summary(self) -> str:
        """Return a summary of this `ThermometerUI` window and its child controls."""
        child_labels = [
            f"{getattr(c, 'id', getattr(c, 'identity', 'ctrl'))}({getattr(c, 'label', getattr(c, 'text', ''))})"
            for c in getattr(self, "children", [])
        ]
        return (
            f"ThermometerUI title={self.title!r} | {self.tempText()} | {self.humidityText()} "
            f"| controls=[{', '.join(child_labels)}]"
        )

    def addButton(
        self,
        label: str,
        action_or_href: Any = "#selected",
        btn_id: Optional[str] = None,
    ) -> Button:
        """Add a dynamic `Button` control to `ThermometerUI` (rendered in the button bar on `/` and `/thermometer`)."""
        lbl = str(label or "Button")
        act_str = str(action_or_href or "#selected")
        clean_id = btn_id or ("btn_" + "".join(ch for ch in lbl if ch.isalnum())[:16] or "btnCustom")
        # Update existing button if one with same id or label already exists
        for ch in list(getattr(self, "children", [])):
            if isinstance(ch, Button) and (getattr(ch, "id", "") == clean_id or getattr(ch, "label", "") == lbl):
                ch.label = lbl
                ch._href = act_str if (act_str.startswith("/") or act_str.startswith("http")) else None
                return ch
        is_nav = (
            act_str.startswith("/")
            or act_str.startswith("http")
            or "studio" in lbl.lower()
            or "back" in lbl.lower()
        )
        sel_name = "onCustomButtonClicked" if is_nav else act_str.lstrip("#")
        btn = Button(
            label=lbl,
            action_selector=Symbol(sel_name),
            left=185,
            top=480,
            width=155,
            height=32,
            id=clean_id,
            identity=Symbol(clean_id),
            target=self,
            callback=lambda _b: self.onCustomButtonClicked(lbl),
        )
        if is_nav:
            btn._href = act_str if (act_str.startswith("/") or act_str.startswith("http")) else "/"
        self.add(btn)
        self.register_action(clean_id, "selected", "onCustomButtonClicked")
        return btn

    def onCustomButtonClicked(self, msg: Any = None) -> str:
        """Default action handler for dynamically added buttons on `ThermometerUI`."""
        return f"ThermometerUI button activated ({msg or 'ok'})"

    def renderOnSurface(self, surface: Optional[Any] = None) -> Dict[str, Any]:
        """Render the visual thermometer, hygrometer, and history chart onto an Actor surface."""
        target_surface = surface or self.surface
        if target_surface is None:
            return {"pixels_drawn": 0, "error": "No surface provided"}
        self.surface = target_surface
        t = self.twin()
        if t is None:
            return {"pixels_drawn": 0, "error": "No ThermoProTwin attached"}
        return render_thermometer_on_surface(
            t,
            target_surface,
            x=self.rect.left,
            y=self.rect.top,
            width=self.rect.width,
            height=self.rect.height,
        )

    def draw(self) -> str:
        """Render interactive HTML5 + SVG Thermometer & Hygrometer UI window for the Actor surface."""
        t = self.twin()
        tc = t.temperatureC() if t else 20.6
        tf = t.temperatureF() if t else 69.1
        hum = t.humidity() if t else 52
        dew = t.dewPointC() if t else 10.4
        batt = t.battery() if t else 50
        rssi = t.rssi() if t else -77
        comfort = str(t.comfortLevel()) if t else "Comfort"
        dev_name = t.deviceName() if t else DEFAULT_TP358W_NAME
        mac = t.macAddress() if t else DEFAULT_TP358W_MAC
        sync_time = (t.lastSyncTime() if t else None) or "Live"

        norm_t = max(0.0, min(1.0, (tc - (-10.0)) / 55.0))
        merc_h = int(round(norm_t * 170))
        merc_y = 195 - merc_h
        merc_color = "#38bdf8" if tc < 18.0 else ("#ef4444" if tc > 25.0 else "#e94560")

        hum_h = int(round((max(0, min(100, hum)) / 100.0) * 170))
        hum_y = 195 - hum_h
        hum_color = "#f59e0b" if "Dry" in comfort else ("#3b82f6" if "Wet" in comfort else "#10b981")

        # Build SVG polyline points for historical temperature & humidity
        records = list(t.history()) if t else []
        temp_pts = []
        hum_pts = []
        if records:
            t_vals = [float(r.at("tempC") if hasattr(r, "at") else r["tempC"]) for r in records]
            h_vals = [int(r.at("humidity") if hasattr(r, "at") else r["humidity"]) for r in records]
            t_min = min(t_vals) - 0.4
            t_max = max(t_vals) + 0.4
            if t_max <= t_min:
                t_max = t_min + 1.0
            n_pts = len(records)
            for idx, (tv, hv) in enumerate(zip(t_vals, h_vals)):
                px = 30 + int(round((idx / max(1, n_pts - 1)) * 410))
                ty = 175 - int(round(((tv - t_min) / (t_max - t_min)) * 145))
                hy = 175 - int(round((hv / 100.0) * 145))
                temp_pts.append(f"{px},{ty}")
                hum_pts.append(f"{px},{hy}")

        temp_poly = (
            f'<polyline fill="none" stroke="#e94560" stroke-width="2.5" points="{" ".join(temp_pts)}"/>'
            if temp_pts
            else '<text x="235" y="105" fill="#8d99ae" font-size="12" text-anchor="middle">Click "Download 60m History" to plot GATT log</text>'
        )
        hum_poly = (
            f'<polyline fill="none" stroke="#1b9aaa" stroke-width="2" stroke-dasharray="4,2" points="{" ".join(hum_pts)}"/>'
            if hum_pts
            else ""
        )

        self.use_fahrenheit = bool(self.chk_unit.value())
        self._fields["useFahrenheit"] = self.use_fahrenheit
        primary_temp = f"{tf:.1f}°F" if self.use_fahrenheit else f"{tc:.1f}°C"
        secondary_temp = f"{tc:.1f}°C" if self.use_fahrenheit else f"{tf:.1f}°F"
        bulb_temp = f"{tf:.1f}°F" if self.use_fahrenheit else f"{tc:.1f}°C"
        dew_display = f"{c_to_f(dew):.1f}°F" if self.use_fahrenheit else f"{dew:.1f}°C"
        unit_label = "°F" if self.use_fahrenheit else "°C"

        return (
            f'<div id="thermo_ui_{id(self)}" class="actor-window actor-thermometer-ui" '
            f'data-widget="thermometer" data-id="ThermoUI" data-title="{self.title}" '
            f'style="position: absolute; left: {self.rect.left}px; top: {self.rect.top}px; '
            f"width: {self.rect.width}px; min-height: {self.rect.height}px; height: auto; "
            f"background: #16213e; color: #eaeaea; border: 2px solid #e94560; border-radius: 8px; "
            f'box-shadow: 0 8px 24px rgba(0,0,0,0.65); font-family: sans-serif; overflow: visible;">'
            # Caption bar
            f'<div class="actor-window-caption" style="height: 28px; background: linear-gradient(90deg, #0f3460, #e94560); '
            f'color: #fff; padding: 4px 12px; font-size: 13px; font-weight: 700; display: flex; justify-content: space-between; align-items: center;">'
            f"<span>🌡️ {self.title} — {dev_name} [{mac}]</span>"
            f'<span style="font-size: 11px; font-family: monospace;">🔋 {batt}% | 📶 {rssi} dBm | 🕒 {sync_time}</span>'
            f"</div>"
            # Main instrument body
            f'<div style="padding: 12px; display: grid; grid-template-columns: 220px 1fr; gap: 12px;">'
            # Left Column: Visual Mercury Thermometer + Hygrometer SVG + Digital LCD
            f'<div style="background: #080b12; border: 1px solid #284b63; border-radius: 6px; padding: 10px; display: flex; flex-direction: column; align-items: center;">'
            f'<svg width="190" height="235" viewBox="0 0 190 235">'
            # Thermometer Tube
            f'<text x="48" y="16" fill="#a8dadc" font-size="11" font-weight="bold" text-anchor="middle">TEMP ({unit_label})</text>'
            f'<rect x="34" y="25" width="28" height="172" rx="14" fill="#1e293b" stroke="#64748b" stroke-width="2"/>'
            f'<rect x="39" y="{merc_y}" width="18" height="{merc_h}" rx="6" fill="{merc_color}"/>'
            f'<circle cx="48" cy="204" r="20" fill="{merc_color}" stroke="#64748b" stroke-width="2"/>'
            f'<text id="thermo-bulb-text" x="48" y="208" fill="#fff" font-size="9.5" font-weight="bold" text-anchor="middle">{bulb_temp}</text>'
            # Hygrometer Bar
            f'<text x="138" y="16" fill="#a8dadc" font-size="11" font-weight="bold" text-anchor="middle">HUMIDITY</text>'
            f'<rect x="124" y="25" width="28" height="172" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="2"/>'
            f'<rect x="129" y="{hum_y}" width="18" height="{hum_h}" rx="4" fill="{hum_color}"/>'
            f'<circle cx="138" cy="204" r="20" fill="{hum_color}" stroke="#64748b" stroke-width="2"/>'
            f'<text x="138" y="208" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">{hum}%</text>'
            f"</svg>"
            # Digital LCD Readout Box
            f'<div style="width: 100%; background: #121629; border: 1px solid #284b63; border-radius: 6px; padding: 8px; text-align: center; margin-top: 4px;">'
            f'<div style="font-size: 24px; font-weight: 800; color: #80ed99; font-family: monospace;">{primary_temp} <span style="font-size: 14px; color: #a8dadc;">({secondary_temp})</span></div>'
            f'<div style="font-size: 16px; font-weight: 700; color: #ffb703; font-family: monospace; margin-top: 2px;">💧 {hum}% RH &bull; Dew {dew_display}</div>'
            f'<div style="margin-top: 6px;"><span style="background: {hum_color}; color: #fff; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 700;">#{comfort.upper()}</span></div>'
            f"</div>"
            f"</div>"
            # Right Column: 60-Minute Historical Chart + Actor Control Fields
            f'<div style="display: flex; flex-direction: column; gap: 8px;">'
            f'<div style="background: #080b12; border: 1px solid #284b63; border-radius: 6px; padding: 8px;">'
            f'<div style="display: flex; justify-content: space-between; font-size: 11px; color: #a8dadc; margin-bottom: 4px;">'
            f"<span><strong>GATT Historical Log ({len(records)} min)</strong></span>"
            f'<span><span style="color:#e94560;">■ Temp (°C)</span> &nbsp; <span style="color:#1b9aaa;">-- Humidity (%)</span></span>'
            f"</div>"
            f'<svg width="460" height="190" viewBox="0 0 460 190" style="background: #0f172a; border-radius: 4px;">'
            f'<line x1="25" y1="45" x2="445" y2="45" stroke="#1e293b" stroke-width="1"/>'
            f'<line x1="25" y1="95" x2="445" y2="95" stroke="#1e293b" stroke-width="1"/>'
            f'<line x1="25" y1="145" x2="445" y2="145" stroke="#1e293b" stroke-width="1"/>'
            f"{hum_poly}"
            f"{temp_poly}"
            f"</svg>"
            f"</div>"
            # Actor TextField Readout Rows
            f'<div style="background: #080b12; border: 1px solid #284b63; border-radius: 6px; padding: 8px; font-family: monospace; font-size: 11.5px; line-height: 1.5;">'
            f'<div><strong style="color:#ffb703;">#tempField:</strong> {self.temp_field.getText()} &nbsp;|&nbsp; <strong style="color:#ffb703;">#humidityField:</strong> {self.humidity_field.getText()}</div>'
            f'<div><strong style="color:#80ed99;">#comfortField:</strong> {self.comfort_field.getText()}</div>'
            f'<div><strong style="color:#a8dadc;">#statsField:</strong> {self.stats_field.getText()}</div>'
            f"</div>"
            # Interactive Actor Action Buttons
            f'<div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">'
            f'<button data-widget="button" data-id="btnSync" data-label="Sync Live BLE" '
            f'style="padding: 7px 14px; background: #e94560; color: #fff; border: none; border-radius: 5px; font-weight: 700; cursor: pointer;">🔄 Sync Live BLE</button>'
            f'<button data-widget="button" data-id="btnHistory" data-label="Download 60m History" '
            f'style="padding: 7px 14px; background: #0f3460; color: #80ed99; border: 1px solid #80ed99; border-radius: 5px; font-weight: 700; cursor: pointer;">📥 Download 60m History</button>'
            f'<button data-widget="button" data-id="chkFahrenheit" data-label="Toggle °F/°C" '
            f'style="padding: 7px 12px; background: #1e293b; color: #ffb703; border: 1px solid #ffb703; border-radius: 5px; font-weight: 600; cursor: pointer;">🌡️ Toggle °F / °C</button>'
            + "".join(
                (
                    f'<button data-widget="button" data-id="{getattr(ch, "id", "btnCustom")}" '
                    f'data-label="{getattr(ch, "label", "Button")}" '
                    + (
                        f'data-href="{getattr(ch, "_href", "/")}" onclick="window.location.href=\'{getattr(ch, "_href", "/")}\'" '
                        if getattr(ch, "_href", None)
                        or "studio" in str(getattr(ch, "label", "")).lower()
                        or "back" in str(getattr(ch, "label", "")).lower()
                        else ""
                    )
                    + f'style="padding: 7px 14px; background: #3b82f6; color: #fff; border: 1px solid #60a5fa; border-radius: 5px; font-weight: 700; cursor: pointer;">'
                    f'{getattr(ch, "label", "Button")}</button>'
                )
                for ch in getattr(self, "children", [])
                if isinstance(ch, Button)
                and str(getattr(ch, "id", "")) not in ("btnSync", "btnHistory", "chkFahrenheit")
            )
            + f"</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )

