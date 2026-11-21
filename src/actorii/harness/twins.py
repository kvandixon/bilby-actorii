"""Capability-Gated Digital-Twin Bridges (`actorii.harness.twins`).

Exposes real-world and host-level subsystems into the `Actor[...]` object pool
using `@actor_bridge`, `@actor_query` (Tier 0 read-only), and `@actor_command`
(Tier 2 audited state mutation):

- `ValkyrieSystemTwin` (`Actor[#Valkyrie]`): Read-only GPU VRAM, utilization,
  temperature, and local service port health telemetry.
- `SmartLightTwin` (`Actor[#Hue]`): Digital twin for smart lighting (`openhue`
  compatible with deterministic in-memory state + opt-in live hardware bridge).
- `PantryGraphTwin` (`Actor[#Pantry]`): Read-only Neo4j recipe & ingredient
  knowledge graph query twin.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from actorii.bridge import actor_bridge, actor_command, actor_query

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    os.environ.get("ACTORII_NEO4J_TX_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"),
)


def _enforce_safe_read_path(raw_path: str) -> Path:
    """Resolve `raw_path` and block reads from `FORBIDDEN_READ_PREFIXES`."""
    from actorii.harness.host_bridge import FORBIDDEN_READ_PREFIXES

    p = Path(str(raw_path or "").strip()).expanduser().resolve()
    p_str = str(p)
    for fp in FORBIDDEN_READ_PREFIXES:
        if p_str == fp or p_str.startswith(fp + "/"):
            raise PermissionError(f"Access to restricted path '{p_str}' is forbidden.")
    if p_str.startswith("/proc/") and p_str.endswith("/environ"):
        raise PermissionError(f"Access to restricted path '{p_str}' is forbidden.")
    return p


def _write_bytes_atomic(dest_path: Path, raw_bytes: bytes, mode: int = 0o600) -> None:
    """Atomically write `raw_bytes` to `dest_path` using `os.fdopen` (no partial `os.write`)."""
    dest_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.path.islink(dest_path):
        dest_path.unlink()
    part_path = dest_path.parent / f".{dest_path.name}.{os.getpid()}.{time.monotonic_ns()}.part"
    fd = os.open(str(part_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw_bytes)
        os.replace(str(part_path), str(dest_path))
    except Exception:
        try:
            if part_path.exists():
                part_path.unlink()
        except OSError:
            pass
        raise


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP redirect handler that re-validates every redirect destination against SSRF guards."""

    def __init__(self, validator: Callable[[str], Any]) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._validator(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_urlopen(
    req: urllib.request.Request | str,
    *,
    timeout: float,
    validator: Callable[[str], Any],
) -> Any:
    """Open an HTTP(S) request while validating all redirect hops with `validator`."""
    opener = urllib.request.build_opener(_SafeRedirectHandler(validator))
    return opener.open(req, timeout=timeout)


@actor_bridge(name="ValkyrieSystemTwin")
class ValkyrieSystemTwin:
    """Read-only host GPU & local service telemetry twin exposed at `Actor[#Valkyrie]`."""

    def __init__(self) -> None:
        self._last_gpu: Dict[str, Any] = {}
        self._last_gpu_ts: float = 0.0

    def _query_nvidia_smi(self, ttl_sec: float = 1.5) -> Dict[str, Any]:
        now = time.monotonic()
        if self._last_gpu and (now - self._last_gpu_ts) < ttl_sec:
            return self._last_gpu
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=2.5,
                    text=True,
                ).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 6:
                        self._last_gpu = {
                            "name": parts[0],
                            "used_mib": int(parts[1]),
                            "free_mib": int(parts[2]),
                            "total_mib": int(parts[3]),
                            "util_pct": int(parts[4]),
                            "temp_c": int(parts[5]),
                        }
                        self._last_gpu_ts = now
                        return self._last_gpu
            except Exception:
                pass
        return {
            "name": "Simulated-GPU",
            "used_mib": 0,
            "free_mib": 16303,
            "total_mib": 16303,
            "util_pct": 0,
            "temp_c": 40,
        }

    @actor_query
    def gpuStatus(self) -> str:
        """Return formatted GPU VRAM, utilization, and temperature status."""
        g = self._query_nvidia_smi()
        return (
            f"{g['name']} | VRAM: {g['used_mib']}/{g['total_mib']} MiB "
            f"({g['free_mib']} MiB free) | Util: {g['util_pct']}% | Temp: {g['temp_c']}°C"
        )

    @actor_query
    def vramUsedMiB(self) -> int:
        """Return integer MiB of GPU VRAM currently in use."""
        return int(self._query_nvidia_smi()["used_mib"])

    @actor_query
    def vramFreeMiB(self) -> int:
        """Return integer MiB of free GPU VRAM."""
        return int(self._query_nvidia_smi()["free_mib"])

    @actor_query
    def gpuTempC(self) -> int:
        """Return integer GPU core temperature in Celsius."""
        return int(self._query_nvidia_smi()["temp_c"])

    @actor_query
    def serviceStatus(self, port: int = 8080) -> str:
        """Check whether a local TCP service port (`8080`, `8082`, `11434`, `7687`, `8090`) is listening."""
        p = int(port)
        try:
            with socket.create_connection(("127.0.0.1", p), timeout=0.25):
                return "ONLINE"
        except Exception:
            return "OFFLINE"

    @actor_query
    def isSandboxed(self) -> bool:
        """Return True if running inside the isolated `actorii-sandbox` Docker container."""
        return os.environ.get("ACTORII_IN_DOCKER") == "1" or os.path.exists("/.dockerenv")

    def _active_8080_label(self) -> str:
        """Probe `:8080/health` quickly to distinguish Strata Coder from llama.cpp."""
        import urllib.request

        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=0.6) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                if '"service": "strata"' in raw or '"service":"strata"' in raw:
                    return "StrataCoderGPU"
        except Exception:
            pass
        return "LlamaPrimaryGPU"

    @actor_query
    def summary(self) -> str:
        """Return a 1-line telemetry summary of GPU VRAM, sandbox mode, and core Valkyrie stack ports."""
        gpu_str = self.gpuStatus()
        mode = "DOCKER-SANDBOX" if self.isSandboxed() else "HOST-RESIDENT"
        lbl8080 = self._active_8080_label()
        ports = [
            (8080, lbl8080),
            (8082, "SpecialistCPU"),
            (11434, "OllamaSystem1"),
            (8084, "Embeddings"),
            (8818, "Hindsight"),
            (7474, "Neo4jHTTP"),
            (7687, "Neo4jBolt"),
            (8090, "ActorStudio"),
            (8765, "ActorWS"),
        ]
        port_strs = [f":{p}({label})={self.serviceStatus(p)}" for p, label in ports]
        note = (
            "Note: Strata Coder IQ1_M pins ~15.5GB VRAM + 18.9GiB DDR5 RAM for 101 tok/s 64k context"
            if lbl8080 == "StrataCoderGPU"
            else "Note: ~14.4GB VRAM is intentionally pinned for qwen3.8:latest 64k context"
        )
        return f"[{mode}] {gpu_str} ({note}) | Services: {' '.join(port_strs)}"

    @actor_query
    def stackHealth(self) -> str:
        """Run `hermes-log-triage` (if available) and `summary()` to inspect full stack health."""
        import shutil
        from pathlib import Path

        parts: List[str] = [self.summary()]
        triage_bin = shutil.which("hermes-log-triage") or str(Path.home() / "bin" / "hermes-log-triage")
        if os.path.isfile(triage_bin):
            try:
                out = subprocess.check_output(
                    [triage_bin],
                    timeout=6.0,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out:
                    parts.append("LogTriage:\n" + out[:1500])
            except Exception as exc:
                parts.append(f"LogTriage error: {exc}")
        return "\n\n".join(parts)

    @actor_query
    def engineStatus(self) -> str:
        """Return active `:8080` inference engine (`strata` vs `llama`), model, thinking mode, VRAM, and RAM via `bilby-engine status`."""
        from pathlib import Path

        ctl = shutil.which("bilby-engine") or str(Path.home() / "bin" / "bilby-engine")
        if os.path.isfile(ctl):
            try:
                out = subprocess.check_output([ctl, "status"], timeout=8.0, text=True, stderr=subprocess.STDOUT).strip()
                if out:
                    return out
            except Exception as exc:
                return f"engineStatus error: {exc}"
        return self.summary()

    @actor_command
    def switchEngine(self, target: str = "status") -> str:
        """Safely switch `:8080` between `strata` (Qwen3.8-Flash-Next Coder IQ1_M @ ~101 tok/s, default) and `llama` (Qwen3.8-27B dense for Actor 4.1 multi-method Self-Evolution or `:8082` CPU specialist RAM)."""
        from pathlib import Path
        from actorii.ai import set_thinking_mode

        if self.isSandboxed() or os.environ.get("ACTORII_BENCH_STRICT") == "1":
            return "BLOCKED: switchEngine is disabled inside isolated benchmark containers."
        clean = str(target or "status").lstrip("#").strip().lower()
        allowed = {
            "status",
            "toggle",
            "strata",
            "strata-coder",
            "coder",
            "fast",
            "llama",
            "llamacpp",
            "llama.cpp",
            "qwen27b",
            "27b",
            "dense",
            "self-evo",
        }
        if clean not in allowed:
            return f"ERROR: Unsupported engine target {target!r}. Expected: 'strata', 'llama', 'toggle', or 'status'."
        ctl = shutil.which("bilby-engine") or str(Path.home() / "bin" / "bilby-engine")
        if not os.path.isfile(ctl):
            return "ERROR: bilby-engine helper not installed in ~/bin/bilby-engine."
        try:
            proc = subprocess.run(
                [ctl, clean, "--json"],
                timeout=115.0,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                detail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[:1200]
                return f"switchEngine error (exit {proc.returncode}): {detail}"
            data = json.loads(proc.stdout)
            tmode = str(data.get("thinking_mode") or "").strip().lower()
            if tmode in ("off", "auto", "medium", "high"):
                try:
                    set_thinking_mode(tmode)
                except ValueError:
                    pass
            svcs = data.get("services") or {}
            elapsed = f" in {data['switch_elapsed_s']}s" if "switch_elapsed_s" in data else ""
            return (
                f"Active Engine: {str(data.get('active_engine', '')).upper()}{elapsed} | "
                f"Model: {data.get('active_model')} | Thinking: {data.get('thinking_mode')} | "
                f"8082 Specialist: {svcs.get('clef-flash-slot-8082.service')} | "
                f"VRAM: {data.get('vram')} | RAM: {data.get('ram')}"
            )
        except subprocess.TimeoutExpired:
            return "switchEngine timed out after 115s waiting for :8080 health check."
        except Exception as exc:
            return f"switchEngine error: {type(exc).__name__}: {exc}"

    @actor_command
    def askSkippy(self, question: str, timeout_sec: int = 110) -> str:
        """Escalate a diagnostic question, multimodal image/PDF analysis (`/tmp/actorii_media/...`), or architectural consult to Cloud Skippy via `ask-skippy`."""
        import shutil
        from pathlib import Path

        q = str(question or "").strip()
        if not q:
            return "ERROR: Empty question for askSkippy"
        ask_bin = shutil.which("ask-skippy") or str(Path.home() / "bin" / "ask-skippy")
        if os.path.isfile(ask_bin):
            wait_s = max(15.0, min(180.0, float(timeout_sec or 110)))
            sub_env = dict(os.environ)
            active_cid = _ACTIVE_TELEGRAM_ROUTE.get("chat_id")
            active_tid = _ACTIVE_TELEGRAM_ROUTE.get("thread_id")
            if active_cid and not sub_env.get("ASK_SKIPPY_CHAT_ID"):
                sub_env["ASK_SKIPPY_CHAT_ID"] = str(active_cid)
            if active_tid and not sub_env.get("ASK_SKIPPY_THREAD_ID"):
                sub_env["ASK_SKIPPY_THREAD_ID"] = str(active_tid)
            try:
                proc = subprocess.run(
                    [ask_bin, q],
                    timeout=wait_s,
                    capture_output=True,
                    text=True,
                    env=sub_env,
                )
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()
                if proc.returncode == 0:
                    return out[:4000] if out else "Logged escalation to Cloud Skippy."
                detail = (err or out or f"exit code {proc.returncode}")[:1200]
                return f"askSkippy error (exit {proc.returncode}): {detail}"
            except subprocess.TimeoutExpired:
                return f"askSkippy timed out after {wait_s:.0f}s waiting for Cloud Skippy."
            except Exception as exc:
                return f"askSkippy error: {type(exc).__name__}: {exc}"
        return "ERROR: ask-skippy helper not installed on this host"

    @actor_query
    def findTools(self, keyword: str = "") -> str:
        """Search `/usr/local/bin` and `/usr/bin` for installed CLI utilities matching `keyword`."""
        kw = str(keyword or "").lstrip("#").strip().lower()
        found: List[str] = []
        for d in ("/usr/local/bin", "/usr/bin", "/bin"):
            if not os.path.isdir(d):
                continue
            try:
                for entry in sorted(os.listdir(d)):
                    if not kw or kw in entry.lower():
                        found.append(f"{d}/{entry}")
            except Exception:
                continue
        if not found:
            return f"No tools found matching {kw!r}"
        return ", ".join(found[:40])

    @actor_query
    def discoverLighting(self) -> str:
        """Discover RGB/fan/RAM lighting controllers and control software in the sandbox environment."""
        clues: List[str] = []
        ctl = shutil.which("valkyrie-rgb-ctl") or (
            "/usr/local/bin/valkyrie-rgb-ctl"
            if os.path.isfile("/usr/local/bin/valkyrie-rgb-ctl")
            else None
        )
        if ctl:
            try:
                out = subprocess.check_output(
                    [ctl, "status"], timeout=2.0, text=True
                ).strip()
                clues.append(f"CLI Controller (`{ctl}`): {out}")
            except Exception as exc:
                clues.append(f"CLI Controller (`{ctl}`): present ({exc})")
        for tool_name in ("openrgb", "liquidctl", "ectool"):
            path = shutil.which(tool_name)
            if path:
                clues.append(f"Detected binary: {path}")
        if os.path.isdir("/etc/valkyrie-rgb"):
            clues.append(f"Config dir `/etc/valkyrie-rgb`: {sorted(os.listdir('/etc/valkyrie-rgb'))}")
        if os.path.isdir("/sys/class/leds"):
            try:
                leds = sorted(os.listdir("/sys/class/leds"))[:15]
                if leds:
                    clues.append(f"/sys/class/leds: {leds}")
            except Exception:
                pass
        return " | ".join(clues) if clues else "No RGB/lighting controllers discovered."

    @actor_query
    def listDirectory(self, path: str = "/app") -> str:
        """List files in `path` inside the container sandbox."""
        if not self.isSandboxed():
            return "BLOCKED: Filesystem browsing requires the `actorii-sandbox` Docker container."
        target = str(path or "/app").strip()
        if not os.path.exists(target):
            return f"Path not found: {target}"
        if os.path.isfile(target):
            return f"File: {target} ({os.path.getsize(target)} bytes)"
        try:
            entries = sorted(os.listdir(target))[:60]
            return f"{target}: {', '.join(entries)}"
        except Exception as exc:
            return f"Error listing {target}: {exc}"

    @actor_query
    def readTextFile(self, path: str, max_chars: int = 3000) -> str:
        """Read up to `max_chars` from a text file inside the container sandbox."""
        if not self.isSandboxed():
            return "BLOCKED: File reading requires the `actorii-sandbox` Docker container."
        target = str(path or "").strip()
        try:
            safe_p = _enforce_safe_read_path(target)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        if not safe_p.is_file():
            return f"File not found: {target}"
        try:
            with open(safe_p, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(max(100, min(8000, int(max_chars or 3000))))
        except Exception as exc:
            return f"Error reading {target}: {exc}"

    @actor_command
    def runShell(self, command: str) -> str:
        """Execute a shell command inside the isolated `actorii-sandbox` Docker container."""
        if not self.isSandboxed():
            return (
                "BLOCKED: Host shell execution is strictly disabled outside the "
                "`actorii-sandbox` Docker container."
            )
        cmd_str = str(command or "").strip()
        if not cmd_str:
            return "ERROR: Empty command"
        from actorii.harness.host_bridge import _run_shell_with_pgkill, validate_safe_shell_command

        try:
            validate_safe_shell_command(cmd_str)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        # Quote bare #RRGGBB hex color arguments so /bin/sh does not treat `#` as a comment
        import re as _re

        cmd_str = _re.sub(r"(?<=\s)(#[0-9A-Fa-f]{3,8})(?=\s|$)", r"'\1'", cmd_str)
        try:
            rc, out_s, err_s = _run_shell_with_pgkill(cmd_str, cwd=os.getcwd(), timeout_sec=5.0)
            combined = ((out_s or "") + ("\n" + err_s if err_s else "")).strip()
            if len(combined) > 3000:
                combined = combined[:3000] + "...[truncated]"
            return f"[exit={rc}] {combined}" if combined else f"[exit={rc}] (no output)"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"


@actor_bridge(name="SmartLightTwin")
class SmartLightTwin:
    """Digital twin for smart lighting (`Actor[#Hue]`), with deterministic state and opt-in `openhue` bridge."""

    def __init__(self) -> None:
        self._lights: Dict[str, Dict[str, Any]] = {
            "Office": {"state": "on", "brightness": 85, "color_temp": "warm"},
            "LivingRoom": {"state": "on", "brightness": 70, "color_temp": "neutral"},
            "Workbench": {"state": "off", "brightness": 0, "color_temp": "cool"},
        }

    def _norm(self, name: Any) -> str:
        return str(name or "").lstrip("#").strip() or "Office"

    @actor_query
    def listLights(self) -> str:
        """Return a summary of all tracked light zones and their current state/brightness."""
        items = [
            f"{k}: {v['state']} ({v['brightness']}%)"
            for k, v in sorted(self._lights.items())
        ]
        return " | ".join(items)

    @actor_query
    def getRoomState(self, room: str) -> str:
        """Return the state (`on`/`off` and brightness %) of a room or light."""
        key = self._norm(room)
        info = self._lights.get(key)
        if not info:
            return f"{key}: unknown"
        return f"{key}: {info['state']} ({info['brightness']}%)"

    @actor_query
    def getBrightness(self, room: str) -> int:
        """Return integer brightness (0..100) for `room`."""
        key = self._norm(room)
        info = self._lights.get(key)
        return int(info["brightness"]) if info else 0

    @actor_command
    def setLight(self, room: str, state: str, brightness: int = 100) -> str:
        """Update digital twin light state (`on`/`off`) and brightness (0..100)."""
        key = self._norm(room)
        norm_state = str(state or "on").lstrip("#").lower()
        if norm_state not in ("on", "off"):
            norm_state = "on"
        b = max(0, min(100, int(brightness)))
        if norm_state == "off":
            b = 0
        self._lights[key] = {
            "state": norm_state,
            "brightness": b,
            "color_temp": self._lights.get(key, {}).get("color_temp", "neutral"),
        }
        return f"{key} -> {norm_state} ({b}%)"


@actor_bridge(name="PantryGraphTwin")
class PantryGraphTwin:
    """Read-only Neo4j recipe & ingredient digital twin exposed at `Actor[#Pantry]`."""

    def __init__(self, tx_url: Optional[str] = None) -> None:
        self.tx_url = tx_url or os.environ.get("ACTORII_NEO4J_HTTP_URL", NEO4J_HTTP_TX_URL)

    def _run_cypher(self, statement: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        payload = {"statements": [{"statement": statement, "parameters": parameters or {}}]}
        req = urllib.request.Request(
            self.tx_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        errors = body.get("errors") or []
        if errors:
            first_err = errors[0] if isinstance(errors[0], dict) else {"message": str(errors[0])}
            raise RuntimeError(f"Neo4j Cypher error: {first_err.get('message', first_err)}")
        results = body.get("results", [])
        if not results:
            return []
        cols = results[0].get("columns", [])
        out: List[Dict[str, Any]] = []
        for r in results[0].get("data", []):
            row_vals = r.get("row", [])
            out.append({cols[i]: row_vals[i] for i in range(min(len(cols), len(row_vals)))})
        return out

    @actor_query
    def recipeCount(self) -> int:
        """Return total number of `:Recipe` nodes in Neo4j."""
        try:
            rows = self._run_cypher("MATCH (r:Recipe) RETURN count(r) AS cnt")
            return int(rows[0]["cnt"]) if rows else 0
        except Exception:
            return 0

    @actor_query
    def listRecipes(self, limit: int = 10) -> str:
        """Return comma-separated titles of up to `limit` `:Recipe` nodes in Neo4j."""
        try:
            rows = self._run_cypher(
                "MATCH (r:Recipe) RETURN coalesce(r.title, r.name, 'Untitled') AS title ORDER BY title LIMIT $lim",
                {"lim": int(limit or 10)},
            )
            if rows:
                return ", ".join(str(r["title"]) for r in rows)
            return "No recipes found in Neo4j"
        except Exception as exc:
            return f"Neo4j Pantry Twin offline ({exc})"

    @actor_query
    def searchByIngredient(self, ingredient: str) -> str:
        """Find `:Recipe` nodes in Neo4j linked to an `:Ingredient` matching `ingredient`."""
        q = str(ingredient or "").lstrip("#").strip().lower()
        try:
            rows = self._run_cypher(
                """
                MATCH (r:Recipe)-[:CONTAINS|USES|HAS_INGREDIENT]->(i:Ingredient)
                WHERE toLower(coalesce(i.name, '')) CONTAINS $q
                RETURN DISTINCT coalesce(r.title, r.name, 'Untitled') AS title
                ORDER BY title
                LIMIT 10
                """,
                {"q": q},
            )
            if rows:
                return ", ".join(str(r["title"]) for r in rows)
            return f"No recipes matched ingredient {q!r}"
        except Exception as exc:
            return f"Neo4j Pantry Twin offline ({exc})"


@actor_bridge(name="ReverseEngineeringTwin")
class ReverseEngineeringTwin:
    """Capability-gated static reverse engineering (REA+Ghidra) & air-gapped behavioral sandbox twin (`Actor[#ReverseEngineer]`)."""

    def __init__(self) -> None:
        import shutil
        from pathlib import Path

        home_bin = Path.home() / "bin"
        self._rea_docker = shutil.which("rea-docker") or str(home_bin / "rea-docker")
        self._rea_batch = shutil.which("rea-ghidra-batch") or str(home_bin / "rea-ghidra-batch")
        self._vt_sandbox = shutil.which("vt-sandbox") or str(home_bin / "vt-sandbox")

    def _to_container_path(self, host_path: str) -> str:
        from pathlib import Path
        p = Path(str(host_path).strip()).resolve()
        home_dir = Path.home().resolve()
        try:
            rel = p.relative_to(home_dir)
            return f"/host/{home_dir.name}/{rel}"
        except ValueError:
            return str(p)

    @actor_query
    def summary(self) -> str:
        """Return readiness status of the REA+Ghidra static engine and air-gapped VT-Sandbox."""
        has_rea = os.path.isfile(self._rea_docker) and os.path.isfile(self._rea_batch)
        has_vt = os.path.isfile(self._vt_sandbox)
        return (
            f"ReverseEngineeringTwin | REA+Ghidra12.1.4={'READY' if has_rea else 'MISSING'} "
            f"| AirGapped-VT-Sandbox={'READY' if has_vt else 'MISSING'}"
        )

    @actor_query
    def fileIdentity(self, path: str) -> str:
        """Return read-only cryptographic SHA-256, byte size, and `file` magic description of `path`."""
        import hashlib
        try:
            safe_p = _enforce_safe_read_path(path)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        target = str(safe_p)
        if not os.path.isfile(target):
            return f"File not found: {target}"
        size = os.path.getsize(target)
        h = hashlib.sha256()
        with open(target, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
        try:
            magic = subprocess.check_output(["file", "-b", "--", target], timeout=5.0, text=True).strip()
        except Exception:
            magic = "unknown"
        return f"SHA256={sha} | Size={size} bytes | Type={magic}"

    @actor_query
    def inspectEntropy(self, path: str) -> str:
        """Run REA `inspect-byte-entropy` on `path` and return overall entropy and high-entropy block ratio."""
        try:
            safe_p = _enforce_safe_read_path(path)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        target = str(safe_p)
        if not os.path.isfile(target):
            return f"File not found: {target}"
        cpath = self._to_container_path(target)
        try:
            out = subprocess.check_output(
                [self._rea_docker, "inspect-byte-entropy", cpath, "--json"],
                timeout=30.0,
                text=True,
            )
            data = json.loads(out).get("normalized_result", {})
            overall = data.get("file_entropy")
            blocks = data.get("blocks", []) or []
            high = sum(1 for b in blocks if (b.get("entropy") or 0) > 7.2)
            return f"Entropy={overall}/8.0 | HighEntropyBlocks(>7.2)={high}/{len(blocks)}"
        except Exception as exc:
            return f"Entropy inspection failed: {exc}"

    @actor_query
    def staticAudit(self, path: str, procedures_csv: str = "") -> str:
        """Run persistent REA+Ghidra batch audit (`rea-ghidra-batch`) on `path` and return procedure/import/string summary."""
        try:
            safe_p = _enforce_safe_read_path(path)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        target = str(safe_p)
        if not os.path.isfile(target):
            return f"File not found: {target}"
        cmd = [self._rea_batch, target]
        procs = [
            p.strip()
            for p in str(procedures_csv or "").split(",")
            if p.strip() and not p.strip().startswith("-")
        ]
        if procs:
            cmd.extend(["--decompile"] + procs)
        try:
            out = subprocess.check_output(cmd, timeout=120.0, text=True)
            data = json.loads(out)
            counts = data.get("counts", {})
            decomp = list((data.get("decompiled_functions") or {}).keys())
            return (
                f"StaticAudit({os.path.basename(target)}): TotalProcs={counts.get('total_procedures')} "
                f"(External={counts.get('external_imports')}, Local={counts.get('local_functions')}), "
                f"Strings={counts.get('defined_strings')} (Matched={counts.get('keyword_matched_strings')}), "
                f"Decompiled={decomp}"
            )
        except Exception as exc:
            return f"Static audit failed: {exc}"

    @actor_query
    def decompileFunction(self, path: str, procedure: str) -> str:
        """Decompile a single procedure or address in `path` via REA+Ghidra and return C pseudocode."""
        try:
            safe_p = _enforce_safe_read_path(path)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        target = str(safe_p)
        proc_sel = str(procedure or "entry").lstrip("#").strip()
        if not proc_sel or proc_sel.startswith("-"):
            return f"Invalid procedure selector: {procedure!r}"
        if not os.path.isfile(target):
            return f"File not found: {target}"
        try:
            out = subprocess.check_output(
                [self._rea_batch, target, "--decompile", proc_sel],
                timeout=120.0,
                text=True,
            )
            data = json.loads(out)
            df = data.get("decompiled_functions", {})
            for _, info in df.items():
                if info.get("pseudocode"):
                    return str(info["pseudocode"]).strip()
            return f"No pseudocode returned for {proc_sel}"
        except Exception as exc:
            return f"Decompile failed: {exc}"

    @actor_command
    def detonateSandbox(self, path: str, duration_sec: int = 10, run_as: str = "sample.exe") -> str:
        """Detonate `path` inside the air-gapped VT-Sandbox (CAPA + Speakeasy + Wine64/Xvfb + strace) and return behavioral summary."""
        try:
            safe_p = _enforce_safe_read_path(path)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        target = str(safe_p)
        if not os.path.isfile(target):
            return f"File not found: {target}"
        dur = max(4, min(30, int(duration_sec or 10)))
        raw_rname = os.path.basename(str(run_as or "sample.exe").strip())
        rname = re.sub(r"[^A-Za-z0-9._-]", "_", raw_rname).lstrip(".-") or "sample.exe"
        out_dir = tempfile.mkdtemp(prefix="actor_vtsandbox_")
        try:
            out = subprocess.check_output(
                [self._vt_sandbox, target, out_dir, "--duration", str(dur), "--run-as", rname],
                timeout=dur + 90.0,
                text=True,
            )
            rep = json.loads(out)
            capa_cnt = rep.get("capa", {}).get("rule_count", 0)
            se_calls = rep.get("speakeasy", {}).get("total_api_calls", 0)
            live = rep.get("live_detonation", {})
            net_cnt = len(live.get("strace_af_inet_connects", []) or [])
            dropped = len(live.get("filesystem_diff", {}).get("workdir_created", []) or [])
            return (
                f"SandboxDetonation({os.path.basename(target)}): CAPA_Rules={capa_cnt} | "
                f"Speakeasy_APIs={se_calls} | LiveAlive={live.get('ran_until_timeout_alive')} | "
                f"NetworkConnects={net_cnt} | DroppedFiles={dropped} | OutDir={out_dir}"
            )
        except Exception as exc:
            return f"Sandbox detonation failed: {exc}"


_TELEGRAM_SEED_IPS = ("149.154.166.110", "149.154.167.220")


def load_telegram_credentials(env_path: Optional[str] = None) -> Dict[str, Any]:
    """Load Telegram token, allowed users, and home channel from `.env` or environment."""
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        or os.environ.get("BILBY_TELEGRAM_BOT_TOKEN", "").strip()
    )
    chat_id = (
        os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
        or os.environ.get("BILBY_TELEGRAM_HOME_CHANNEL", "").strip()
    )
    thread_id_raw = os.environ.get("TELEGRAM_HOME_CHANNEL_THREAD_ID", "").strip()
    allowed_raw = (
        os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
        or os.environ.get("BILBY_TELEGRAM_ALLOWED_CHATS", "").strip()
    )

    candidates = [
        env_path,
        os.environ.get("ACTORII_TELEGRAM_ENV_FILE"),
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/.actorii/.env"),
        os.path.expanduser("~/.hermes/profiles/bilby/.env"),
        os.path.expanduser("~/.hermes/.env"),
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.split("#", 1)[0].strip().strip("'\"")
                    if k in ("TELEGRAM_BOT_TOKEN", "BILBY_TELEGRAM_BOT_TOKEN") and v and not token:
                        token = v
                    elif k in ("TELEGRAM_HOME_CHANNEL", "BILBY_TELEGRAM_HOME_CHANNEL") and v and not chat_id:
                        chat_id = v
                    elif k == "TELEGRAM_HOME_CHANNEL_THREAD_ID" and v and not thread_id_raw:
                        thread_id_raw = v
                    elif k in ("TELEGRAM_ALLOWED_USERS", "BILBY_TELEGRAM_ALLOWED_CHATS") and v and not allowed_raw:
                        allowed_raw = v
        except Exception:
            pass

    allowed_users = [
        int(x.strip())
        for x in allowed_raw.split(",")
        if x.strip().lstrip("-").isdigit()
    ]
    if not chat_id and allowed_users:
        chat_id = str(allowed_users[0])

    thread_id: Optional[int] = int(thread_id_raw) if thread_id_raw.isdigit() else None
    return {
        "token": token,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "allowed_users": allowed_users,
    }


def _telegram_https_request(
    token: str,
    method: str,
    *,
    http_method: str,
    body_bytes: Optional[bytes],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    """Shared Telegram HTTPS transport with IPv4 SNI fallback and guaranteed socket cleanup."""
    import http.client
    import ssl

    if not token:
        return {"ok": False, "description": "Missing TELEGRAM_BOT_TOKEN"}

    path = f"/bot{token}/{method}"
    ctx = ssl.create_default_context()
    targets = ["api.telegram.org", *_TELEGRAM_SEED_IPS]
    last_err = "Unknown error"

    for target in targets:
        raw_sock = None
        conn = None
        try:
            raw_sock = socket.create_connection((target, 443), timeout=timeout)
            tls_sock = ctx.wrap_socket(raw_sock, server_hostname="api.telegram.org")
            raw_sock = None  # ownership transferred to tls_sock/conn
            conn = http.client.HTTPSConnection("api.telegram.org", 443, timeout=timeout, context=ctx)
            conn.sock = tls_sock
            conn.request(http_method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw_data = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw_data)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            if token:
                err_msg = err_msg.replace(token, "<REDACTED>")
            last_err = err_msg
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            elif raw_sock is not None:
                try:
                    raw_sock.close()
                except Exception:
                    pass

    return {"ok": False, "description": last_err}


def telegram_api_request(
    token: str,
    method: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Call `api.telegram.org/bot<token>/<method>` with automatic IPv4 SNI fallback (`149.154.166.110`)."""
    body_bytes = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    headers = {"Host": "api.telegram.org", "Accept": "application/json"}
    if body_bytes is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))
    return _telegram_https_request(
        token,
        method,
        http_method="POST" if body_bytes is not None else "GET",
        body_bytes=body_bytes,
        headers=headers,
        timeout=timeout,
    )


TELEGRAM_MEDIA_STAGE_DIR = "/tmp/actorii_media"
TELEGRAM_MAX_PHOTO_BYTES = 10 * 1024 * 1024
TELEGRAM_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024

_ACTIVE_TELEGRAM_ROUTE: Dict[str, Any] = {"chat_id": None, "thread_id": None}

_SENSITIVE_UPLOAD_BASENAMES = (
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "known_hosts",
    "authorized_keys",
    "shadow",
    "gshadow",
    "passwd",
    "sudoers",
    "credentials",
    "stack_state.json",
    "telegram_gateway_state.json",
)
_SENSITIVE_UPLOAD_EXTS = (".key", ".pem", ".p12", ".pfx", ".kdbx")


def _encode_multipart_formdata(
    fields: Optional[Dict[str, Any]],
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    content_type: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Encode RFC 7578 `multipart/form-data` payload and return `(body_bytes, content_type_header)`."""
    import mimetypes
    import uuid

    boundary = f"----ActorIITelegramBoundary{uuid.uuid4().hex}"
    safe_fname = os.path.basename(file_name or "upload.bin").replace('"', "_").replace("\r", "").replace("\n", "")
    guessed_mime = content_type or mimetypes.guess_type(safe_fname)[0] or "application/octet-stream"

    chunks: List[bytes] = []
    for k, v in (fields or {}).items():
        if v is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{v}\r\n".encode("utf-8"))

    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_fname}"\r\n'
            f"Content-Type: {guessed_mime}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def telegram_multipart_request(
    token: str,
    method: str,
    fields: Optional[Dict[str, Any]],
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    content_type: Optional[str] = None,
    timeout: float = 35.0,
) -> Dict[str, Any]:
    """Upload binary media (`sendPhoto`, `sendDocument`) to `api.telegram.org` via `multipart/form-data` with IPv4 SNI fallback."""
    body_bytes, ctype_header = _encode_multipart_formdata(
        fields=fields,
        file_field=file_field,
        file_name=file_name,
        file_bytes=file_bytes,
        content_type=content_type,
    )
    headers = {
        "Host": "api.telegram.org",
        "Accept": "application/json",
        "Content-Type": ctype_header,
        "Content-Length": str(len(body_bytes)),
    }
    return _telegram_https_request(
        token,
        method,
        http_method="POST",
        body_bytes=body_bytes,
        headers=headers,
        timeout=timeout,
    )


@actor_bridge(name="TelegramBotTwin")
class TelegramBotTwin:
    """Telegram Bot API Digital Twin exposed at `Actor[#Telegram]`."""

    def __init__(
        self,
        token: Optional[str] = None,
        default_chat_id: Optional[str] = None,
        default_thread_id: Optional[int] = None,
        env_path: Optional[str] = None,
        api_caller: Optional[Any] = None,
        multipart_caller: Optional[Any] = None,
    ) -> None:
        creds = load_telegram_credentials(env_path=env_path)
        self._token = (token if token is not None else creds["token"]).strip()
        self._chat_id = str(default_chat_id if default_chat_id is not None else creds["chat_id"]).strip()
        self._thread_id = default_thread_id if default_thread_id is not None else creds["thread_id"]
        self._active_chat_id: Optional[str] = None
        self._active_thread_id: Optional[int] = None
        self._allowed_users: List[int] = list(creds["allowed_users"])
        self._bot_info: Dict[str, Any] = {}
        self._sent_log: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._api = api_caller or telegram_api_request
        self._multipart_api = multipart_caller or telegram_multipart_request

    def _set_active_route(self, chat_id: Any, thread_id: Optional[int] = None) -> None:
        """Update the active inbound conversation `(chat_id, thread_id)` from `BilbyTelegramGateway`."""
        cid_s = str(chat_id or "").strip()
        self._active_chat_id = cid_s if cid_s else None
        if thread_id is not None and str(thread_id).lstrip("-").isdigit() and int(thread_id) > 0:
            self._active_thread_id = int(thread_id)
        else:
            self._active_thread_id = None
        _ACTIVE_TELEGRAM_ROUTE["chat_id"] = self._active_chat_id
        _ACTIVE_TELEGRAM_ROUTE["thread_id"] = self._active_thread_id

    def _resolve_route(self, chat_id: Any = None, thread_id: int = 0) -> Tuple[str, int]:
        """Resolve effective `(chat_id, thread_id)` preferring explicit args, then active turn route, then `.env` defaults."""
        explicit_cid = str(chat_id or "").strip() if chat_id not in (None, "", 0) else ""
        explicit_tid = int(thread_id or 0) if str(thread_id or 0).lstrip("-").isdigit() else 0
        default_cid = self._active_chat_id or self._chat_id
        default_tid = int(
            self._active_thread_id
            if self._active_thread_id is not None
            else (self._thread_id or 0)
        )
        if explicit_cid:
            if explicit_tid > 0:
                return explicit_cid, explicit_tid
            if explicit_cid == default_cid and default_tid > 0:
                return explicit_cid, default_tid
            return explicit_cid, 0
        return default_cid, (explicit_tid if explicit_tid > 0 else default_tid)

    @staticmethod
    def _resolve_safe_upload_path(file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Validate and resolve a local staging file path safe for Telegram upload under host lockdown."""
        raw_in = str(file_path or "").strip()
        if not raw_in or "\x00" in raw_in:
            return None, "Telegram upload error: missing or invalid file_path."

        stage_dir = os.environ.get("ACTORII_MEDIA_STAGE_DIR", TELEGRAM_MEDIA_STAGE_DIR)
        stage_real = os.path.realpath(stage_dir)
        tmp_real = os.path.realpath("/tmp")
        ws_real = os.path.realpath("/workspace")

        expanded = os.path.abspath(os.path.expanduser(raw_in))
        candidate = expanded
        # If Bilby passed `/tmp/foo.jpg` or bare `foo.jpg` for a file staged in `/tmp/actorii_media/foo.jpg`, resolve it cleanly
        if not os.path.lexists(candidate) and ("/" not in raw_in or expanded.startswith(tmp_real + os.sep)):
            alt = os.path.join(stage_dir, os.path.basename(raw_in))
            if os.path.lexists(alt):
                candidate = alt

        if not os.path.lexists(candidate):
            return (
                None,
                f"Telegram upload error: local file {raw_in!r} not found on valkyrie. "
                f"If the file is on the Jetson, pull it first via pullFile(Actor[#Jetson], {raw_in!r})!!",
            )

        if os.path.islink(candidate):
            return None, f"Telegram upload blocked: symlinks are not permitted ({raw_in!r})."

        real_p = os.path.realpath(candidate)
        in_stage = real_p == stage_real or real_p.startswith(stage_real + os.sep)
        in_tmp = real_p.startswith(tmp_real + os.sep)
        in_ws = os.path.isdir("/workspace") and real_p.startswith(ws_real + os.sep)

        if not (in_stage or in_tmp or in_ws):
            return (
                None,
                f"Telegram upload blocked by host lockdown: {raw_in!r} is outside allowed staging directories (`/tmp/actorii_media`, `/tmp`, `/workspace`).",
            )

        if not os.path.isfile(real_p):
            return None, f"Telegram upload error: {raw_in!r} is not a regular file."

        base_low = os.path.basename(real_p).lower()
        if any(base_low == b or base_low.endswith("." + b.lstrip(".")) for b in _SENSITIVE_UPLOAD_BASENAMES) or any(
            base_low.endswith(ext) for ext in _SENSITIVE_UPLOAD_EXTS
        ):
            return None, f"Telegram upload blocked: sensitive credential/state filename {os.path.basename(real_p)!r} is prohibited."

        if in_tmp and not in_stage:
            try:
                st = os.stat(real_p)
                if hasattr(os, "getuid") and st.st_uid != os.getuid():
                    return None, f"Telegram upload blocked: `/tmp` file {raw_in!r} is not owned by current user."
            except OSError as exc:
                return None, f"Telegram upload error: cannot stat {raw_in!r}: {exc}"

        return real_p, None

    @actor_query
    def summary(self) -> str:
        """Return Telegram bot configuration, active topic route, and media capabilities."""
        masked = "set" if self._token else "unset"
        uname = self._bot_info.get("username", "unconfigured")
        eff_cid, eff_tid = self._resolve_route()
        return (
            f"TelegramBotTwin(@{uname} | token={masked} | "
            f"home_chat={self._chat_id or 'none'} | active_chat={eff_cid or 'none'} | thread={eff_tid} | "
            f"methods=[sendMessage, sendToChat, sendPhoto, sendDocument] | allowed_users={self._allowed_users})"
        )

    @actor_query
    def botInfo(self) -> str:
        """Query Telegram `getMe` API and return bot identity."""
        res = self._api(self._token, "getMe", None, 8.0)
        if res.get("ok") and isinstance(res.get("result"), dict):
            self._bot_info = res["result"]
            r = self._bot_info
            return f"@{r.get('username')} (id={r.get('id')}, name={r.get('first_name')})"
        return f"Telegram getMe error: {res.get('description', 'unknown')}"

    @actor_command
    def sendMessage(self, text: str) -> str:
        """Send a message to the active Telegram conversation (or configured home channel)."""
        cid, tid = self._resolve_route()
        return self.sendToChat(cid, text, tid)

    @actor_command
    def sendToChat(self, chat_id: Any, text: str, thread_id: int = 0) -> str:
        """Send a Telegram message to `chat_id` (optionally in `thread_id` forum topic)."""
        cid, tid = self._resolve_route(chat_id, thread_id)
        msg = str(text or "").strip()
        if not cid:
            return "Telegram send error: missing chat_id"
        if not msg:
            return "Telegram send error: empty message"

        payload: Dict[str, Any] = {"chat_id": int(cid) if cid.lstrip("-").isdigit() else cid, "text": msg}
        if tid > 0:
            payload["message_thread_id"] = tid

        res = self._api(self._token, "sendMessage", payload, 12.0)
        if not res.get("ok") and "message_thread_id" in payload:
            fallback_payload = {k: v for k, v in payload.items() if k != "message_thread_id"}
            res = self._api(self._token, "sendMessage", fallback_payload, 12.0)
        if res.get("ok") and isinstance(res.get("result"), dict):
            msg_id = res["result"].get("message_id")
            self._sent_log.append({"type": "text", "chat_id": cid, "thread_id": tid, "message_id": msg_id, "text": msg})
            return f"Sent Telegram message #{msg_id} to chat {cid}"
        return f"Telegram send failed: {res.get('description', 'unknown')}"

    def _upload_media(
        self,
        *,
        api_method: str,
        file_field: str,
        max_bytes: int,
        file_path: str,
        caption: str = "",
        chat_id: Any = None,
        thread_id: int = 0,
    ) -> str:
        cid, tid = self._resolve_route(chat_id, thread_id)
        if not cid:
            return f"Telegram {api_method} error: missing chat_id"

        real_p, err = self._resolve_safe_upload_path(file_path)
        if err or not real_p:
            return err or f"Telegram {api_method} error: invalid file path"

        try:
            size_b = os.path.getsize(real_p)
        except OSError as exc:
            return f"Telegram {api_method} error: cannot stat {real_p!r}: {exc}"

        if size_b <= 0:
            return f"Telegram {api_method} error: file {os.path.basename(real_p)!r} is empty (0 bytes)."
        if size_b > max_bytes:
            lim_mb = max_bytes // (1024 * 1024)
            tip = " Use sendDocument for files up to 50 MB." if api_method == "sendPhoto" and size_b <= TELEGRAM_MAX_DOCUMENT_BYTES else ""
            return f"Telegram {api_method} error: file {os.path.basename(real_p)!r} is {size_b} bytes (exceeds {lim_mb} MB limit).{tip}"

        try:
            with open(real_p, "rb") as fh:
                file_bytes = fh.read(max_bytes + 1)
        except OSError as exc:
            return f"Telegram {api_method} error: cannot read {real_p!r}: {exc}"
        if len(file_bytes) > max_bytes:
            lim_mb = max_bytes // (1024 * 1024)
            return f"Telegram {api_method} error: file {os.path.basename(real_p)!r} grew past {lim_mb} MB limit during read."

        fields: Dict[str, Any] = {"chat_id": int(cid) if cid.lstrip("-").isdigit() else cid}
        clean_cap = str(caption or "").strip()
        if clean_cap:
            fields["caption"] = clean_cap[:1024]
        if tid > 0:
            fields["message_thread_id"] = tid

        fname = os.path.basename(real_p)
        res = self._multipart_api(
            self._token,
            api_method,
            fields,
            file_field,
            fname,
            file_bytes,
            None,
            40.0,
        )
        if not res.get("ok") and "message_thread_id" in fields:
            fallback_fields = {k: v for k, v in fields.items() if k != "message_thread_id"}
            res = self._multipart_api(
                self._token,
                api_method,
                fallback_fields,
                file_field,
                fname,
                file_bytes,
                None,
                40.0,
            )

        if res.get("ok") and isinstance(res.get("result"), dict):
            msg_id = res["result"].get("message_id")
            self._sent_log.append(
                {
                    "type": file_field,
                    "chat_id": cid,
                    "thread_id": tid,
                    "message_id": msg_id,
                    "file_name": fname,
                    "bytes": len(file_bytes),
                    "caption": clean_cap,
                }
            )
            return f"Sent Telegram {file_field} #{msg_id} ({fname}, {len(file_bytes)} bytes) to chat {cid}"

        desc = res.get("description", "unknown")
        tip = " (Tip: use sendDocument for non-image or uncompressed files up to 50 MB)" if api_method == "sendPhoto" else ""
        return f"Telegram {api_method} failed: {desc}{tip}"

    @actor_command
    def sendPhoto(self, file_path: str, caption: str = "", chat_id: Any = None, thread_id: int = 0) -> str:
        """Upload and send a local image (`<= 10 MB`, e.g. from `/tmp/actorii_media/...`) via Telegram `sendPhoto`."""
        return self._upload_media(
            api_method="sendPhoto",
            file_field="photo",
            max_bytes=TELEGRAM_MAX_PHOTO_BYTES,
            file_path=file_path,
            caption=caption,
            chat_id=chat_id,
            thread_id=thread_id,
        )

    @actor_command
    def sendPhotoToChat(self, chat_id: Any, file_path: str, caption: str = "", thread_id: int = 0) -> str:
        """Upload and send a local image (`<= 10 MB`) to a specific Telegram `chat_id` / `thread_id`."""
        return self.sendPhoto(file_path=file_path, caption=caption, chat_id=chat_id, thread_id=thread_id)

    @actor_command
    def sendDocument(self, file_path: str, caption: str = "", chat_id: Any = None, thread_id: int = 0) -> str:
        """Upload and send any local file (`<= 50 MB`, e.g. from `/tmp/actorii_media/...`) via Telegram `sendDocument`."""
        return self._upload_media(
            api_method="sendDocument",
            file_field="document",
            max_bytes=TELEGRAM_MAX_DOCUMENT_BYTES,
            file_path=file_path,
            caption=caption,
            chat_id=chat_id,
            thread_id=thread_id,
        )

    @actor_command
    def sendDocumentToChat(self, chat_id: Any, file_path: str, caption: str = "", thread_id: int = 0) -> str:
        """Upload and send a local file (`<= 50 MB`) to a specific Telegram `chat_id` / `thread_id`."""
        return self.sendDocument(file_path=file_path, caption=caption, chat_id=chat_id, thread_id=thread_id)

    @actor_command
    def sendFile(self, file_path: str, caption: str = "", chat_id: Any = None, thread_id: int = 0) -> str:
        """Alias for `sendDocument(Actor[#Telegram], file_path, caption)`."""
        return self.sendDocument(file_path=file_path, caption=caption, chat_id=chat_id, thread_id=thread_id)


def _to_actor_dict(mapping: Dict[str, Any]) -> Any:
    """Convert a Python dict into an authentic 1993 Actor `Dictionary`."""
    from actorii.classes.collections import ActorArray
    from actorii.classes.dictionary import Dictionary
    from actorii.core import Symbol

    d = Dictionary(max(8, len(mapping) * 2))
    for k, v in mapping.items():
        if isinstance(v, dict):
            conv = _to_actor_dict(v)
        elif isinstance(v, list):
            arr = ActorArray(len(v))
            for idx, item in enumerate(v):
                arr.atPut(idx, _to_actor_dict(item) if isinstance(item, dict) else item)
            conv = arr
        else:
            conv = v
        k_str = str(k).lstrip("#")
        d.atPut(k_str, conv)
        d.atPut(Symbol(k_str), conv)
    return d


def _validate_public_http_url(raw_url: str, *, label: str = "URL") -> str:
    """Block SSRF against loopback, link-local, private RFC1918, Docker host-gateway, and Neo4j/internal ports."""
    import ipaddress
    import urllib.parse

    u = str(raw_url or "").strip()
    if not u:
        raise ValueError(f"Empty {label}")
    parsed = urllib.parse.urlparse(u)
    if parsed.scheme.lower() not in ("http", "https"):
        raise PermissionError(
            f"BLOCKED [SSRF_URL_GUARD]: Only `http://` and `https://` schemes are allowed for {label} (got {parsed.scheme!r})."
        )
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError(f"Missing hostname in {label}: {u!r}")

    if os.environ.get("ACTORII_ALLOW_PRIVATE_URLS", "0") == "1":
        return u

    blocked_hosts = {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
        "gateway.docker.internal",
        "neo4j",
        "bilby-neo4j",
        "valkyrie",
        "valkyrie.local",
        "metadata.google.internal",
    }
    if host in blocked_hosts or host.endswith((".localhost", ".internal")):
        raise PermissionError(
            f"BLOCKED [SSRF_URL_GUARD]: {label} targeting internal/loopback host {host!r} is forbidden."
        )
    if parsed.port in (7474, 7687, 8080, 8082, 8084, 8818, 11434):
        raise PermissionError(
            f"BLOCKED [SSRF_URL_GUARD]: {label} targeting protected local stack port :{parsed.port} is forbidden."
        )

    def _check_ip(ip_str: str) -> None:
        addr = ipaddress.ip_address(ip_str)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
            or not addr.is_global
        ):
            raise PermissionError(
                f"BLOCKED [SSRF_URL_GUARD]: {label} targeting non-public IP {ip_str} ({host!r}) is forbidden."
            )

    try:
        _check_ip(host)
    except ValueError:
        # Hostname rather than literal IP: if DNS resolves locally right now, ensure none of its records are loopback/private
        try:
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                sockaddr = info[4]
                if sockaddr and sockaddr[0]:
                    _check_ip(str(sockaddr[0]).split("%", 1)[0])
        except socket.gaierror:
            pass
    return u


_LOCAL_HOST_IPS_CACHE: Tuple[float, set[str]] = (0.0, set())


def _discover_local_host_ips(ttl_sec: float = 30.0) -> set[str]:
    """Return all IPv4/IPv6 addresses bound on the local `valkyrie` host so LAN probes never SSRF the host itself."""
    global _LOCAL_HOST_IPS_CACHE
    now = time.monotonic()
    ts, cached = _LOCAL_HOST_IPS_CACHE
    if cached and (now - ts) < ttl_sec:
        return set(cached)

    local_ips = {"127.0.0.1", "::1", "0.0.0.0"}
    try:
        from actorii.reverse_proxy import discover_host_network_ips

        info = discover_host_network_ips()
        for v in (info or {}).values():
            if v and isinstance(v, str):
                local_ips.add(v.strip())
    except Exception:
        pass
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None):
            if res and res[4] and res[4][0]:
                local_ips.add(str(res[4][0]).split("%", 1)[0])
    except Exception:
        pass
    _LOCAL_HOST_IPS_CACHE = (now, set(local_ips))
    return local_ips


def _validate_probe_target_url(
    raw_url: str,
    *,
    allowed_schemes: Tuple[str, ...] = ("http", "https"),
    label: str = "Probe URL",
) -> urllib.parse.ParseResult:
    """Validate URLs for active protocol conformance probes (`probeHttp`, `probeMjpeg`, `probeRtsp`, etc.).

    Allows public hosts AND non-host LAN edge devices (`192.168.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`, `.local`)
    such as the NVIDIA Jetson (`192.168.68.55`), while strictly blocking loopback (`127.0.0.0/8`, `::1`),
    `valkyrie`'s own local IP addresses, cloud metadata (`169.254.169.254`), Docker internal gateways,
    and protected local stack ports (`:5432, :7474, :7687, :8080, :8082, :8084, :8818, :11434`).
    """
    import ipaddress

    u = str(raw_url or "").strip()
    if not u:
        raise ValueError(f"Empty {label}")
    parsed = urllib.parse.urlparse(u)
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        raise PermissionError(
            f"BLOCKED [PROBE_SCHEME_GUARD]: {label} requires scheme in {allowed_schemes} (got {scheme!r})."
        )
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError(f"Missing hostname in {label}: {u!r}")

    if os.environ.get("ACTORII_ALLOW_PRIVATE_URLS", "0") == "1":
        return parsed

    blocked_hosts = {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
        "gateway.docker.internal",
        "neo4j",
        "bilby-neo4j",
        "valkyrie",
        "valkyrie.local",
        "metadata.google.internal",
    }
    if host in blocked_hosts or host.endswith((".localhost", ".internal")):
        raise PermissionError(
            f"BLOCKED [SSRF_PROBE_GUARD]: {label} targeting internal/loopback host {host!r} is forbidden."
        )
    if parsed.port in (5432, 7474, 7687, 8080, 8082, 8084, 8818, 11434):
        raise PermissionError(
            f"BLOCKED [SSRF_PROBE_GUARD]: {label} targeting protected local stack port :{parsed.port} is forbidden."
        )

    local_host_ips = _discover_local_host_ips()

    def _check_probe_ip(ip_str: str) -> None:
        addr = ipaddress.ip_address(ip_str)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        norm_ip = str(addr)
        if (
            addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
            or ip_str in local_host_ips
            or norm_ip in local_host_ips
        ):
            raise PermissionError(
                f"BLOCKED [SSRF_PROBE_GUARD]: {label} targeting loopback/host/link-local IP {ip_str} ({host!r}) is forbidden."
            )

    try:
        _check_probe_ip(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                sockaddr = info[4]
                if sockaddr and sockaddr[0]:
                    _check_probe_ip(str(sockaddr[0]).split("%", 1)[0])
        except socket.gaierror:
            pass
    return parsed


def _parse_jpeg_dimensions(jpeg_bytes: bytes) -> Tuple[int, int]:
    """Extract `(width, height)` from a baseline or progressive JPEG (`SOF0`..`SOF3`) frame."""
    n = len(jpeg_bytes)
    if n < 4 or jpeg_bytes[:2] != b"\xff\xd8":
        return (0, 0)
    idx = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while idx + 4 < n:
        if jpeg_bytes[idx] != 0xFF:
            idx += 1
            continue
        while idx < n and jpeg_bytes[idx] == 0xFF:
            idx += 1
        if idx >= n:
            break
        marker = jpeg_bytes[idx]
        idx += 1
        if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7) or marker == 0x01:
            continue
        if idx + 2 > n:
            break
        seg_len = int.from_bytes(jpeg_bytes[idx : idx + 2], "big")
        if seg_len < 2 or idx + seg_len > n:
            break
        if marker in sof_markers and seg_len >= 7:
            height = int.from_bytes(jpeg_bytes[idx + 3 : idx + 5], "big")
            width = int.from_bytes(jpeg_bytes[idx + 5 : idx + 7], "big")
            return (width, height)
        idx += seg_len
    return (0, 0)


def _stage_media_sample(raw_bytes: bytes, sample_name: str, default_name: str) -> str:
    """Atomically write a verified media frame into `/tmp/actorii_media/` (`0600`) for `askSkippy`/`sendPhoto`."""
    raw_base = os.path.basename(str(sample_name or "").strip() or default_name)
    safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", raw_base).lstrip(".") or default_name
    stage_dir = Path(os.environ.get("ACTORII_MEDIA_STAGE_DIR", TELEGRAM_MEDIA_STAGE_DIR))
    dest_path = stage_dir / safe_base
    _write_bytes_atomic(dest_path, raw_bytes, mode=0o600)
    return str(dest_path)


@actor_bridge(name="NetTransportBridge")
class NetTransportBridge:
    """Low-level HTTP/HTML/MCP transport driver exposed at `Actor[#Net]` for the 1993 Actor 4.1 `WebScraper` (`Actor[#Web]`)."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        import shutil
        from pathlib import Path

        home_bin = Path.home() / "bin"
        self._firecrawl_key = (api_key if api_key is not None else os.environ.get("FIRECRAWL_API_KEY", "")).strip()
        self._web_clean_bin = shutil.which("web-clean") or str(home_bin / "web-clean")
        self._browser_fetch_bin = shutil.which("browser-fetch") or str(home_bin / "browser-fetch")

    @actor_query
    def summary(self) -> str:
        """Return status of low-level web transport drivers."""
        try:
            import trafilatura as _tf  # noqa: F401
            has_py_trafilatura = True
        except Exception:
            has_py_trafilatura = False
        has_trafilatura = has_py_trafilatura or os.path.isfile(self._web_clean_bin)
        has_playwright = os.path.isfile(self._browser_fetch_bin)
        return (
            f"NetTransportBridge(firecrawl_key={'set' if self._firecrawl_key else 'keyless'} | "
            f"trafilatura={has_trafilatura} | browser-fetch={has_playwright})"
        )

    @actor_query
    def firecrawlPostScrape(self, base_url: str, target_url: str, max_chars: int = 8000) -> Any:
        """Execute HTTP POST to Firecrawl `/v2/scrape` and return an Actor `Dictionary`."""
        from actorii.core import Symbol

        api_base = str(base_url or "https://api.firecrawl.dev").rstrip("/")
        url = str(target_url or "").strip()
        try:
            _validate_public_http_url(api_base, label="Firecrawl base_url")
            _validate_public_http_url(url, label="target_url")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("firecrawl"), "url": url, "title": "", "content": "", "error": str(exc)})
        limit = max(500, int(max_chars or 8000))
        payload = json.dumps({"url": url, "formats": ["markdown"]}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._firecrawl_key:
            headers["Authorization"] = f"Bearer {self._firecrawl_key}"

        last_err = "Firecrawl returned empty markdown content"
        for endpoint in (f"{api_base}/v2/scrape", f"{api_base}/v1/scrape"):
            try:
                req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
                with _safe_urlopen(req, timeout=15.0, validator=lambda u: _validate_public_http_url(u, label="Firecrawl redirect")) as resp:
                    raw = json.loads(resp.read().decode("utf-8", errors="replace"))
                data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
                md = str(data.get("markdown") or data.get("content") or data.get("html") or "").strip()
                meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                title = str(meta.get("title") or data.get("title") or "").strip()
                if md:
                    return _to_actor_dict(
                        {
                            "ok": True,
                            "provider": Symbol("firecrawl"),
                            "url": url,
                            "title": title,
                            "content": md[:limit],
                            "error": "",
                        }
                    )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
        return _to_actor_dict({"ok": False, "provider": Symbol("firecrawl"), "url": url, "title": "", "content": "", "error": last_err})

    @actor_query
    def firecrawlPostSearch(self, base_url: str, query: str, limit: int = 5, max_chars: int = 6000) -> Any:
        """Execute HTTP POST to Firecrawl `/v2/search` and return an Actor `Dictionary`."""
        from actorii.core import Symbol

        api_base = str(base_url or "https://api.firecrawl.dev").rstrip("/")
        q = str(query or "").strip()
        try:
            _validate_public_http_url(api_base, label="Firecrawl base_url")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("firecrawl"), "query": q, "hits": [], "error": str(exc)})
        lim = max(1, min(15, int(limit or 5)))
        payload = json.dumps({"query": q, "limit": lim}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._firecrawl_key:
            headers["Authorization"] = f"Bearer {self._firecrawl_key}"

        try:
            req = urllib.request.Request(f"{api_base}/v2/search", data=payload, headers=headers, method="POST")
            with _safe_urlopen(req, timeout=15.0, validator=lambda u: _validate_public_http_url(u, label="Firecrawl redirect")) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="replace"))
            data = raw.get("data")
            rows_raw = (
                data
                if isinstance(data, list)
                else (data.get("web") or data.get("results") if isinstance(data, dict) else raw.get("web") or raw.get("results") or [])
            )
            hits: List[Dict[str, str]] = []
            if isinstance(rows_raw, list):
                for r in rows_raw[:lim]:
                    if isinstance(r, dict) and (r.get("url") or r.get("title")):
                        hits.append(
                            {
                                "title": str(r.get("title") or ""),
                                "url": str(r.get("url") or ""),
                                "snippet": str(r.get("description") or r.get("snippet") or r.get("markdown") or "")[:400],
                            }
                        )
            if hits:
                return _to_actor_dict({"ok": True, "provider": Symbol("firecrawl"), "query": q, "hits": hits, "error": ""})
            return _to_actor_dict({"ok": False, "provider": Symbol("firecrawl"), "query": q, "hits": [], "error": "Empty Firecrawl results"})
        except Exception as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("firecrawl"), "query": q, "hits": [], "error": f"{type(exc).__name__}: {exc}"})

    @actor_query
    def trafilaturaExtract(self, target_url: str, max_chars: int = 8000) -> Any:
        """Extract clean boilerplate-free text via `trafilatura` (or `web-clean` CLI helper)."""
        from actorii.core import Symbol

        url = str(target_url or "").strip()
        try:
            _validate_public_http_url(url, label="target_url")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("trafilatura"), "url": url, "title": "", "content": "", "error": str(exc)})
        limit = max(500, int(max_chars or 8000))
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_links=True, output_format="txt")
                if extracted and extracted.strip():
                    return _to_actor_dict(
                        {
                            "ok": True,
                            "provider": Symbol("trafilatura"),
                            "url": url,
                            "title": "",
                            "content": extracted.strip()[:limit],
                            "error": "",
                        }
                    )
        except Exception:
            pass

        if os.path.isfile(self._web_clean_bin):
            try:
                out = subprocess.check_output([self._web_clean_bin, url], timeout=20.0, text=True, stderr=subprocess.PIPE).strip()
                if out:
                    return _to_actor_dict(
                        {
                            "ok": True,
                            "provider": Symbol("trafilatura"),
                            "url": url,
                            "title": "",
                            "content": out[:limit],
                            "error": "",
                        }
                    )
            except Exception as exc:
                return _to_actor_dict({"ok": False, "provider": Symbol("trafilatura"), "url": url, "title": "", "content": "", "error": str(exc)})

        return _to_actor_dict({"ok": False, "provider": Symbol("trafilatura"), "url": url, "title": "", "content": "", "error": "Trafilatura yielded no text"})

    @actor_query
    def keylessRingExtract(self, target_url: str, max_chars: int = 8000) -> Any:
        """Extract URL content via the Keyless MCP Ring (Parallel / Exa / Keenable)."""
        from actorii.core import Symbol

        url = str(target_url or "").strip()
        try:
            _validate_public_http_url(url, label="target_url")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("keyless"), "url": url, "title": "", "content": "", "error": str(exc)})
        limit = max(500, int(max_chars or 8000))
        last_err = "Keyless ring returned empty content"

        # 1. Try Keenable public fetch
        try:
            import urllib.parse

            k_url = f"https://api.keenable.ai/v1/fetch/public?url={urllib.parse.quote(url, safe='')}"
            req = urllib.request.Request(k_url, headers={"X-Keenable-Title": "hermes-agent", "Accept": "application/json"})
            with _safe_urlopen(req, timeout=15.0, validator=lambda u: _validate_public_http_url(u, label="Keenable redirect")) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            txt = str(data.get("content") or "").strip()
            if txt:
                return _to_actor_dict(
                    {
                        "ok": True,
                        "provider": Symbol("keenable"),
                        "url": url,
                        "title": str(data.get("title") or ""),
                        "content": txt[:limit],
                        "error": "",
                    }
                )
        except Exception as exc:
            last_err = str(exc)

        # 2. Try Exa MCP `web_fetch_exa`
        try:
            mcp_payload = json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "web_fetch_exa", "arguments": {"urls": [url]}}}
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://mcp.exa.ai/mcp",
                data=mcp_payload,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
            for line in raw_body.splitlines():
                cand = line[6:].strip() if line.startswith("data: ") else (line.strip() if line.strip().startswith("{") else "")
                if cand:
                    obj = json.loads(cand)
                    contents = (obj.get("result") or {}).get("content") or []
                    texts = [c.get("text", "") for c in contents if isinstance(c, dict) and c.get("text")]
                    if texts:
                        return _to_actor_dict(
                            {
                                "ok": True,
                                "provider": Symbol("exa"),
                                "url": url,
                                "title": "",
                                "content": "\n".join(texts).strip()[:limit],
                                "error": "",
                            }
                        )
        except Exception as exc:
            last_err = str(exc)

        return _to_actor_dict({"ok": False, "provider": Symbol("keyless"), "url": url, "title": "", "content": "", "error": last_err})

    @actor_query
    def playwrightExtract(self, target_url: str, max_chars: int = 8000) -> Any:
        """Render and extract JS-heavy page text using `browser-fetch` (Playwright Chromium)."""
        from actorii.core import Symbol

        url = str(target_url or "").strip()
        try:
            _validate_public_http_url(url, label="target_url")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("playwright"), "url": url, "title": "", "content": "", "error": str(exc)})
        limit = max(500, int(max_chars or 8000))
        if not os.path.isfile(self._browser_fetch_bin):
            return _to_actor_dict({"ok": False, "provider": Symbol("playwright"), "url": url, "title": "", "content": "", "error": "browser-fetch not installed"})
        try:
            out = subprocess.check_output([self._browser_fetch_bin, url], timeout=35.0, text=True, stderr=subprocess.PIPE).strip()
            if out:
                return _to_actor_dict(
                    {
                        "ok": True,
                        "provider": Symbol("playwright"),
                        "url": url,
                        "title": "",
                        "content": out[:limit],
                        "error": "",
                    }
                )
            return _to_actor_dict({"ok": False, "provider": Symbol("playwright"), "url": url, "title": "", "content": "", "error": "Empty body text"})
        except Exception as exc:
            return _to_actor_dict({"ok": False, "provider": Symbol("playwright"), "url": url, "title": "", "content": "", "error": str(exc)})

    @actor_query
    def keylessRingSearch(self, query: str, limit: int = 5, max_chars: int = 6000) -> Any:
        """Search via the Keyless MCP Ring (Keenable / Exa / Parallel)."""
        from actorii.core import Symbol

        q = str(query or "").strip()
        lim = max(1, min(15, int(limit or 5)))
        last_err = "Keyless ring exhausted"

        # 1. Try Keenable public search
        try:
            payload = json.dumps({"query": q, "max_results": lim}).encode("utf-8")
            req = urllib.request.Request(
                "https://api.keenable.ai/v1/search/public",
                data=payload,
                headers={"X-Keenable-Title": "hermes-agent", "Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            results = data.get("results") or []
            hits = [
                {
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("url") or ""),
                    "snippet": str(r.get("snippet") or r.get("description") or "")[:400],
                }
                for r in results[:lim]
                if isinstance(r, dict) and r.get("url")
            ]
            if hits:
                return _to_actor_dict({"ok": True, "provider": Symbol("keenable"), "query": q, "hits": hits, "error": ""})
        except Exception as exc:
            last_err = str(exc)

        # 2. Try Exa MCP `web_search_exa`
        try:
            mcp_payload = json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "web_search_exa", "arguments": {"query": q, "numResults": lim}}}
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://mcp.exa.ai/mcp",
                data=mcp_payload,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
            for line in raw_body.splitlines():
                cand = line[6:].strip() if line.startswith("data: ") else (line.strip() if line.strip().startswith("{") else "")
                if cand:
                    obj = json.loads(cand)
                    contents = (obj.get("result") or {}).get("content") or []
                    texts = [c.get("text", "") for c in contents if isinstance(c, dict) and c.get("text")]
                    if texts:
                        hits = [{"title": f"Exa Result #{i+1}", "url": "", "snippet": blk.strip()[:450]} for i, blk in enumerate(texts[0].split("\n---\n")[:lim]) if blk.strip()]
                        if hits:
                            return _to_actor_dict({"ok": True, "provider": Symbol("exa"), "query": q, "hits": hits, "error": ""})
        except Exception as exc:
            last_err = str(exc)

        return _to_actor_dict({"ok": False, "provider": Symbol("keyless"), "query": q, "hits": [], "error": last_err})

    @actor_query
    def ddgsSearch(self, query: str, limit: int = 5, max_chars: int = 6000) -> Any:
        """Fallback search via DuckDuckGo (`duckduckgo_search` / `ddgs` or HTML lite)."""
        from actorii.core import Symbol

        q = str(query or "").strip()
        lim = max(1, min(15, int(limit or 5)))
        last_err = "DuckDuckGo returned 0 results"
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                rows = list(ddgs.text(q, max_results=lim))
            hits = [
                {
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("href") or r.get("url") or ""),
                    "snippet": str(r.get("body") or r.get("snippet") or "")[:400],
                }
                for r in rows
                if isinstance(r, dict)
            ]
            if hits:
                return _to_actor_dict({"ok": True, "provider": Symbol("ddgs"), "query": q, "hits": hits, "error": ""})
        except Exception as exc:
            last_err = str(exc)

        return _to_actor_dict({"ok": False, "provider": Symbol("ddgs"), "query": q, "hits": [], "error": last_err})

    @actor_query
    def formatScrapeEntry(self, entry: Any, provider_sym: Any, max_chars: int = 8000) -> str:
        """Format an Actor `Dictionary` scrape response into clean Markdown with provider attribution."""
        from actorii.classes.dictionary import Dictionary

        prov = str(provider_sym).lstrip("#")
        limit = max(500, int(max_chars or 8000))
        if isinstance(entry, Dictionary):
            url = str(entry.at("url") or "")
            title = str(entry.at("title") or "")
            content = str(entry.at("content") or "")
        elif isinstance(entry, dict):
            url = str(entry.get("url") or "")
            title = str(entry.get("title") or "")
            content = str(entry.get("content") or "")
        else:
            url, title, content = "", "", str(entry or "")

        header = f"[WebScrape via #{prov}] {title + ' — ' if title else ''}{url}\n"
        body = content[:limit]
        return f"{header}{body}".strip()

    @actor_query
    def formatSearchHits(self, entry: Any, provider_sym: Any, max_chars: int = 6000) -> str:
        """Format an Actor `Dictionary` search response into numbered results."""
        from actorii.classes.dictionary import Dictionary

        prov = str(provider_sym).lstrip("#")
        limit = max(500, int(max_chars or 6000))
        if isinstance(entry, Dictionary):
            q = str(entry.at("query") or "")
            hits = entry.at("hits") or []
        elif isinstance(entry, dict):
            q = str(entry.get("query") or "")
            hits = entry.get("hits") or []
        else:
            q, hits = "", []

        lines = [f"[WebSearch via #{prov}] Query: {q!r} ({len(hits)} results)"]
        for idx, h in enumerate(hits, 1):
            if isinstance(h, Dictionary):
                t = str(h.at("title") or "")
                u = str(h.at("url") or "")
                s = str(h.at("snippet") or "")
            elif isinstance(h, dict):
                t = str(h.get("title") or "")
                u = str(h.get("url") or "")
                s = str(h.get("snippet") or "")
            else:
                t, u, s = str(h), "", ""
            lines.append(f"{idx}. {t} ({u})\n   {s}".strip())
        return "\n".join(lines)[:limit]

    @actor_query
    def extractUrls(self, text: str, max_urls: int = 3) -> Any:
        """Extract prioritized official documentation HTTP(S) URLs from search results as an Actor `Array`."""
        import re as _re
        from actorii.classes.collections import ActorArray

        raw_txt = str(text or "")
        lim = max(1, min(10, int(max_urls or 3)))
        raw_matches = _re.findall(r"https?://[^\s\)\]>\"']+", raw_txt)
        cleaned: List[str] = []
        seen = set()
        for m in raw_matches:
            u = m.rstrip(".,;:!?)")
            if not u or u in seen:
                continue
            low = u.lower()
            if any(skip in low for skip in ("duckduckgo.com", "google.com/search", "bing.com/search", "youtube.com/watch")):
                continue
            seen.add(u)
            cleaned.append(u)

        def _doc_priority(url_s: str) -> int:
            ul = url_s.lower()
            score = 0
            if any(k in ul for k in ("docs.", "developer.", "manual", "guide", "datasheet", "reference", ".pdf", "readthedocs", "github.com")):
                score -= 10
            if any(k in ul for k in ("reddit.com", "quora.com", "pinterest.", "facebook.com")):
                score += 10
            return score

        cleaned.sort(key=_doc_priority)
        top = cleaned[:lim]
        arr = ActorArray(len(top))
        for idx, u in enumerate(top):
            arr.atPut(idx, u)
        return arr

    @actor_query
    def isPdfSource(self, url_or_path: str) -> bool:
        """Return True if `url_or_path` (ignoring query string/fragment) ends in `.pdf`."""
        base = str(url_or_path or "").strip().split("?", 1)[0].split("#", 1)[0].lower()
        return base.endswith(".pdf")

    @actor_query
    def isHttpUrl(self, url_or_path: str) -> bool:
        """Return True if `url_or_path` starts with `http://` or `https://`."""
        low = str(url_or_path or "").strip().lower()
        return low.startswith("http://") or low.startswith("https://")

    @actor_query
    def readLocalManual(self, path_str: str, max_chars: int = 12000) -> str:
        """Read a local non-restricted text/markdown documentation file (`FORBIDDEN_READ_PREFIXES` enforced)."""
        from actorii.harness.host_bridge import FORBIDDEN_READ_PREFIXES

        raw = str(path_str or "").strip()
        if not raw:
            return "ERROR: Empty file path"
        p = Path(raw).expanduser().resolve()
        p_str = str(p)
        for fp in FORBIDDEN_READ_PREFIXES:
            if p_str == fp or p_str.startswith(fp + "/"):
                raise PermissionError(f"Access to restricted path '{p_str}' is forbidden.")
        if not p.is_file():
            return f"ERROR: Documentation file not found: {raw}"
        limit = max(500, min(30000, int(max_chars or 12000)))
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        return f"[LocalManual: {p.name}]\n{txt[:limit]}"

    # ------------------------------------------------------------------
    # Active Wire-Level Protocol Conformance & Stream Receive Probes
    # ------------------------------------------------------------------

    @staticmethod
    def _open_probe_socket(parsed: urllib.parse.ParseResult, timeout_sec: float) -> Tuple[Any, str, int, str]:
        import ssl

        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        default_port = 443 if scheme in ("https", "wss", "rtsps") else (554 if scheme == "rtsp" else 80)
        port = int(parsed.port or default_port)
        path_q = parsed.path or "/"
        if parsed.query:
            path_q = f"{path_q}?{parsed.query}"
        raw_sock = socket.create_connection((host, port), timeout=timeout_sec)
        raw_sock.settimeout(timeout_sec)
        if scheme in ("https", "wss", "rtsps"):
            try:
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(raw_sock, server_hostname=host)
                sock.settimeout(timeout_sec)
                return sock, host, port, path_q
            except Exception:
                try:
                    raw_sock.close()
                except Exception:
                    pass
                raise
        return raw_sock, host, port, path_q

    @staticmethod
    def _diagnose_non_http_prefix(head: bytes, expected_proto: str = "HTTP/1.1") -> str:
        hex_preview = " ".join(f"{b:02x}" for b in head[:16])
        if len(head) >= 6 and head[4:6] == b"\xff\xd8":
            frame_len = int.from_bytes(head[:4], "little")
            return (
                f"InvalidProtocolFraming: Endpoint emitted raw length-prefixed binary JPEG over TCP "
                f"(first 16 bytes: {hex_preview}; uint32_le={frame_len} followed by JPEG SOI ff d8) with NO {expected_proto} status line! "
                f"Web browsers, curl, and HTTP clients reject raw TCP sockets. Wrap frames in `{expected_proto} 200 OK` + "
                "`Content-Type: multipart/x-mixed-replace; boundary=frame` (for MJPEG `<img src=...>`) or serve HTML at `/`."
            )
        if head.startswith(b"\xff\xd8"):
            return (
                f"InvalidProtocolFraming: Endpoint emitted raw JPEG bytes (`ff d8 ...`) directly on TCP without an `{expected_proto} 200 OK` status line or `Content-Type` header!"
            )
        if head.upper().startswith(b"RTSP/1.0"):
            return (
                "ProtocolMismatch: Endpoint speaks `RTSP/1.0`, not HTTP! Web browsers cannot open `rtsp://` directly; "
                "use `probeRtsp(Actor[#Net], url, \"snap.jpg\", 10)!!` to inspect/receive RTSP, or bridge RTSP to HTTP MJPEG/HLS for browsers."
            )
        return (
            f"InvalidStatusLine: Expected `{expected_proto}` status line, but received raw bytes `{hex_preview}` ({head[:24]!r}). "
            "Ensure the server writes `HTTP/1.1 200 OK\\r\\nContent-Type: ...\\r\\n\\r\\n` before payload bytes."
        )

    @actor_query
    def probeHttp(self, url: str, timeout_sec: int = 8) -> Any:
        """Actively probe an HTTP(S) endpoint like a real browser/`curl`, catching raw-TCP non-HTTP sockets & reporting headers."""
        from actorii.core import Symbol

        u = str(url or "").strip()
        timeout_f = max(1.5, min(30.0, float(timeout_sec or 8)))
        try:
            parsed = _validate_probe_target_url(u, allowed_schemes=("http", "https"), label="HTTP Probe URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("http"), "url": u, "statusCode": 0, "statusLine": "", "contentType": "", "diagnosis": str(exc)})

        sock = None
        try:
            sock, host, port, path_q = self._open_probe_socket(parsed, timeout_f)
            host_hdr = f"{host}:{port}" if parsed.port else host
            req_bytes = (
                f"GET {path_q} HTTP/1.1\r\n"
                f"Host: {host_hdr}\r\n"
                "User-Agent: ActorII-ProtocolProbe/4.4\r\n"
                "Accept: text/html,application/json,multipart/x-mixed-replace,*/*\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(req_bytes)

            buf = bytearray()
            while b"\r\n\r\n" not in buf and len(buf) < 16384:
                chunk = sock.recv(2048)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= 12 and not bytes(buf[:5]).upper().startswith(b"HTTP/"):
                    break

            raw_head = bytes(buf)
            if not raw_head:
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("http"),
                        "url": u,
                        "statusCode": 0,
                        "statusLine": "",
                        "contentType": "",
                        "diagnosis": "EmptyResponse: Server accepted TCP connection and closed immediately without sending an HTTP status line.",
                    }
                )
            if not raw_head[:5].upper().startswith(b"HTTP/"):
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("raw_tcp"),
                        "url": u,
                        "statusCode": 0,
                        "statusLine": "",
                        "contentType": "",
                        "firstBytesHex": " ".join(f"{b:02x}" for b in raw_head[:16]),
                        "diagnosis": self._diagnose_non_http_prefix(raw_head, "HTTP/1.1"),
                    }
                )

            hdr_part, _, body_prefix = raw_head.partition(b"\r\n\r\n")
            lines = hdr_part.decode("iso-8859-1", errors="replace").split("\r\n")
            status_line = lines[0].strip() if lines else ""
            parts = status_line.split(None, 2)
            status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            headers_map: Dict[str, str] = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    headers_map[k.strip().lower()] = v.strip()

            while len(body_prefix) < 512:
                try:
                    more = sock.recv(512)
                    if not more:
                        break
                    body_prefix += more
                except socket.timeout:
                    break

            ctype = headers_map.get("content-type", "")
            cors = headers_map.get("access-control-allow-origin", "")
            preview_txt = body_prefix[:300].decode("utf-8", errors="replace").strip() if not ctype.lower().startswith(("image/", "multipart/", "video/", "application/octet")) else f"<{len(body_prefix)} binary bytes>"
            ok = 200 <= status_code < 400
            diag = (
                f"Valid {status_line} | Content-Type={ctype or '(missing)'} | CORS={cors or 'none'}"
                if ok
                else f"HTTP error status `{status_line}` (Content-Type={ctype or 'none'})."
            )
            return _to_actor_dict(
                {
                    "ok": ok,
                    "protocol": Symbol("http"),
                    "url": u,
                    "statusCode": status_code,
                    "statusLine": status_line,
                    "contentType": ctype,
                    "cors": cors,
                    "headers": headers_map,
                    "bodyPreview": preview_txt,
                    "diagnosis": diag,
                }
            )
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("http"),
                    "url": u,
                    "statusCode": 0,
                    "statusLine": "",
                    "contentType": "",
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    @actor_query
    def probeMjpeg(
        self,
        url: str,
        frames_to_read: int = 2,
        save_sample_as: str = "mjpeg_probe.jpg",
        timeout_sec: int = 12,
    ) -> Any:
        """Validate a browser-compatible HTTP `multipart/x-mixed-replace` MJPEG stream, decode `frames_to_read` frames, save a sample JPEG for `askSkippy`/`sendPhoto`, and verify server survival after disconnect."""
        from actorii.core import Symbol

        u = str(url or "").strip()
        want_frames = max(1, min(10, int(frames_to_read or 2)))
        timeout_f = max(2.0, min(30.0, float(timeout_sec or 12)))
        try:
            parsed = _validate_probe_target_url(u, allowed_schemes=("http", "https"), label="MJPEG Probe URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("mjpeg"), "url": u, "framesVerified": 0, "diagnosis": str(exc)})

        sock = None
        t_start = time.monotonic()
        try:
            sock, host, port, path_q = self._open_probe_socket(parsed, timeout_f)
            host_hdr = f"{host}:{port}" if parsed.port else host
            req_bytes = (
                f"GET {path_q} HTTP/1.1\r\n"
                f"Host: {host_hdr}\r\n"
                "User-Agent: Mozilla/5.0 ActorII-MJPEGProbe/4.4\r\n"
                "Accept: multipart/x-mixed-replace,image/jpeg,*/*\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(req_bytes)

            buf = bytearray()
            deadline = t_start + timeout_f
            while b"\r\n\r\n" not in buf and time.monotonic() < deadline and len(buf) < 16384:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= 12 and not bytes(buf[:5]).upper().startswith(b"HTTP/"):
                    break

            raw_init = bytes(buf)
            if not raw_init[:5].upper().startswith(b"HTTP/"):
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("mjpeg"),
                        "url": u,
                        "framesVerified": 0,
                        "diagnosis": self._diagnose_non_http_prefix(raw_init, "HTTP/1.1"),
                    }
                )

            hdr_bytes, _, remainder = raw_init.partition(b"\r\n\r\n")
            lines = hdr_bytes.decode("iso-8859-1", errors="replace").split("\r\n")
            status_line = lines[0].strip() if lines else ""
            parts = status_line.split(None, 2)
            status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            headers_map: Dict[str, str] = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    headers_map[k.strip().lower()] = v.strip()

            ctype = headers_map.get("content-type", "")
            if status_code != 200 or "multipart/x-mixed-replace" not in ctype.lower():
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("mjpeg"),
                        "url": u,
                        "statusCode": status_code,
                        "statusLine": status_line,
                        "contentType": ctype,
                        "framesVerified": 0,
                        "diagnosis": (
                            f"Expected `HTTP/1.1 200 OK` with `Content-Type: multipart/x-mixed-replace; boundary=...`, "
                            f"but got `{status_line}` with `Content-Type: {ctype or '(missing)'}`."
                        ),
                    }
                )

            boundary = ""
            for seg in ctype.split(";")[1:]:
                seg_s = seg.strip()
                if seg_s.lower().startswith("boundary="):
                    boundary = seg_s.split("=", 1)[1].strip().strip('"')
                    break
            if not boundary:
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("mjpeg"),
                        "url": u,
                        "statusCode": status_code,
                        "contentType": ctype,
                        "framesVerified": 0,
                        "diagnosis": f"Missing `boundary=...` parameter in `Content-Type: {ctype}`.",
                    }
                )

            stream_buf = bytearray(remainder)
            frames: List[bytes] = []
            frame_times: List[float] = []
            max_bytes = 12 * 1024 * 1024

            while len(frames) < want_frames and time.monotonic() < deadline and len(stream_buf) < max_bytes:
                soi = stream_buf.find(b"\xff\xd8")
                if soi >= 0:
                    eoi = stream_buf.find(b"\xff\xd9", soi + 2)
                    if eoi >= 0:
                        jpg = bytes(stream_buf[soi : eoi + 2])
                        del stream_buf[: eoi + 2]
                        frames.append(jpg)
                        frame_times.append(time.monotonic())
                        continue
                rem_t = deadline - time.monotonic()
                if rem_t <= 0:
                    break
                sock.settimeout(max(0.1, rem_t))
                chunk = sock.recv(16384)
                if not chunk:
                    break
                stream_buf.extend(chunk)
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("mjpeg"),
                    "url": u,
                    "framesVerified": 0,
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        if len(frames) < want_frames:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("mjpeg"),
                    "url": u,
                    "statusCode": 200,
                    "contentType": ctype,
                    "framesVerified": len(frames),
                    "diagnosis": f"Multipart stream headers matched, but only decoded {len(frames)}/{want_frames} complete JPEG (`FFD8..FFD9`) frame(s) before timeout/EOF.",
                }
            )

        width, height = _parse_jpeg_dimensions(frames[0])
        fps = 0.0
        if len(frame_times) >= 2 and (frame_times[-1] - frame_times[0]) > 0.001:
            fps = round((len(frame_times) - 1) / (frame_times[-1] - frame_times[0]), 2)

        sample_path = ""
        if save_sample_as:
            sample_path = _stage_media_sample(frames[0], save_sample_as, "mjpeg_probe.jpg")

        # Verify server survives client disconnect by opening a quick second connection
        survived_disconnect = False
        try:
            s2, _, _, _ = self._open_probe_socket(parsed, 2.5)
            s2.close()
            survived_disconnect = True
        except Exception:
            survived_disconnect = False

        ok = width > 0 and height > 0 and survived_disconnect
        diag = (
            f"Verified {len(frames)} live MJPEG frame(s) ({width}x{height} JPEG, ~{len(frames[0])} bytes/frame, "
            f"fps={fps}, boundary={boundary!r}, survivedDisconnect={survived_disconnect}"
            + (f", sample={sample_path}" if sample_path else "")
            + ")."
        )
        if not survived_disconnect:
            diag += " WARNING: Server process died or stopped accepting new clients after first client disconnect (check SIGPIPE / accept loop)!"
        return _to_actor_dict(
            {
                "ok": ok,
                "protocol": Symbol("mjpeg"),
                "url": u,
                "statusCode": 200,
                "contentType": ctype,
                "boundary": boundary,
                "framesVerified": len(frames),
                "width": width,
                "height": height,
                "frameBytes": len(frames[0]),
                "fps": fps,
                "survivedDisconnect": survived_disconnect,
                "samplePath": sample_path,
                "diagnosis": diag,
            }
        )

    @actor_query
    def probeRtsp(
        self,
        rtsp_url: str,
        capture_sample_as: str = "",
        timeout_sec: int = 10,
    ) -> Any:
        """Probe an `rtsp://` endpoint via RFC 2326 `OPTIONS` + `DESCRIBE` (SDP), inspect Send/Receive capabilities, and optionally receive 1 decoded JPEG frame via `-rtsp_transport tcp`."""
        from actorii.core import Symbol

        u = str(rtsp_url or "").strip()
        timeout_f = max(2.0, min(30.0, float(timeout_sec or 10)))
        try:
            parsed = _validate_probe_target_url(u, allowed_schemes=("rtsp", "rtsps"), label="RTSP Probe URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("rtsp"), "url": u, "statusCode": 0, "diagnosis": str(exc)})

        def _read_rtsp_response(s: Any) -> Tuple[int, str, Dict[str, str], str]:
            buf = bytearray()
            while b"\r\n\r\n" not in buf and len(buf) < 32768:
                ch = s.recv(4096)
                if not ch:
                    break
                buf.extend(ch)
            raw = bytes(buf)
            if not raw.upper().startswith(b"RTSP/"):
                return -1, raw[:32].decode("iso-8859-1", errors="replace"), {}, ""
            hdr_b, _, rest = raw.partition(b"\r\n\r\n")
            hdr_lines = hdr_b.decode("iso-8859-1", errors="replace").split("\r\n")
            st_line = hdr_lines[0].strip() if hdr_lines else ""
            st_parts = st_line.split(None, 2)
            code = int(st_parts[1]) if len(st_parts) >= 2 and st_parts[1].isdigit() else 0
            hmap: Dict[str, str] = {}
            for ln in hdr_lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    hmap[k.strip().lower()] = v.strip()
            clen = int(hmap.get("content-length", "0") or 0)
            body_b = bytearray(rest)
            while len(body_b) < clen and len(body_b) < 65536:
                ch = s.recv(min(4096, clen - len(body_b)))
                if not ch:
                    break
                body_b.extend(ch)
            return code, st_line, hmap, body_b[:clen].decode("utf-8", errors="replace") if clen else body_b.decode("utf-8", errors="replace")

        sock = None
        try:
            sock, host, port, _ = self._open_probe_socket(parsed, timeout_f)
            clean_uri = f"{parsed.scheme}://{host}:{port}{parsed.path or '/'}"
            if parsed.query:
                clean_uri = f"{clean_uri}?{parsed.query}"

            opt_req = (
                f"OPTIONS {clean_uri} RTSP/1.0\r\n"
                "CSeq: 1\r\n"
                "User-Agent: ActorII-RTSPProbe/4.4\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(opt_req)
            opt_code, opt_line, opt_hdrs, _ = _read_rtsp_response(sock)
            if opt_code == -1:
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("rtsp"),
                        "url": u,
                        "statusCode": 0,
                        "diagnosis": f"Endpoint did not reply with `RTSP/1.0` (got {opt_line!r}).",
                    }
                )

            public_methods = opt_hdrs.get("public", "")
            supports_receive = any(m in public_methods.upper() for m in ("DESCRIBE", "PLAY")) or not public_methods
            supports_send = any(m in public_methods.upper() for m in ("ANNOUNCE", "RECORD"))

            desc_req = (
                f"DESCRIBE {clean_uri} RTSP/1.0\r\n"
                "CSeq: 2\r\n"
                "Accept: application/sdp\r\n"
                "User-Agent: ActorII-RTSPProbe/4.4\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(desc_req)
            desc_code, desc_line, desc_hdrs, sdp_body = _read_rtsp_response(sock)
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("rtsp"),
                    "url": u,
                    "statusCode": 0,
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        codecs: List[str] = []
        media_tracks: List[str] = []
        has_level_idc_zero_bug = False
        for ln in sdp_body.splitlines():
            s_ln = ln.strip()
            if s_ln.startswith("m="):
                media_tracks.append(s_ln[2:])
            elif s_ln.startswith("a=rtpmap:"):
                codecs.append(s_ln.split(" ", 1)[-1] if " " in s_ln else s_ln)
            elif s_ln.startswith("a=fmtp:") and "profile-level-id=" in s_ln:
                plid = s_ln.split("profile-level-id=", 1)[1].split(";", 1)[0].strip()
                if len(plid) == 6 and plid.endswith("00"):
                    has_level_idc_zero_bug = True

        sample_path = ""
        capture_err = ""
        if desc_code == 200 and capture_sample_as:
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                stage_dir = Path(os.environ.get("ACTORII_MEDIA_STAGE_DIR", TELEGRAM_MEDIA_STAGE_DIR))
                stage_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
                safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(str(capture_sample_as))).lstrip(".") or "rtsp_probe.jpg"
                if not safe_base.lower().endswith((".jpg", ".jpeg", ".png")):
                    safe_base += ".jpg"
                dest = stage_dir / safe_base
                try:
                    proc = subprocess.run(
                        [
                            ffmpeg_bin,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-rtsp_transport",
                            "tcp",
                            "-i",
                            u,
                            "-frames:v",
                            "1",
                            "-y",
                            str(dest),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=min(20.0, timeout_f + 5.0),
                    )
                    if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                        sample_path = str(dest)
                    else:
                        capture_err = (proc.stderr or "").strip()[:300]
                except Exception as exc:
                    capture_err = f"{type(exc).__name__}: {exc}"
            else:
                capture_err = "Local `ffmpeg` binary not found for frame decode."

        if desc_code == 200:
            diag = (
                f"Active RTSP stream (`{desc_line}`) | Public=[{public_methods}] | "
                f"Tracks={media_tracks} | Codecs={codecs} | canReceive={supports_receive} | canPublishSend={supports_send}"
            )
            if has_level_idc_zero_bug:
                diag += " | WARNING: SDP `profile-level-id` ends in `00` (`level_idc=0`, classic Jetson TX1 `nvv4l2h264enc` bug — switch encoder to `omxh264enc insert-sps-pps=true`!)."
            if sample_path:
                diag += f" | Captured sample frame -> {sample_path}"
            elif capture_err:
                diag += f" | Frame capture warning: {capture_err}"
            ok = not has_level_idc_zero_bug and (not capture_sample_as or bool(sample_path))
        elif desc_code == 404 and opt_code == 200:
            ok = False
            diag = (
                f"RTSP server is online (`{opt_line}`, Public=[{public_methods}]), but mount returned `{desc_line}` (no active publisher on this path yet; "
                f"canPublishSend={supports_send})."
            )
        else:
            ok = False
            diag = f"RTSP DESCRIBE returned `{desc_line}` (OPTIONS=`{opt_line}`, Public=[{public_methods}])."

        return _to_actor_dict(
            {
                "ok": ok,
                "protocol": Symbol("rtsp"),
                "url": u,
                "optionsStatus": opt_code,
                "statusCode": desc_code,
                "statusLine": desc_line,
                "publicMethods": public_methods,
                "canReceive": supports_receive,
                "canSendPublish": supports_send,
                "tracks": media_tracks,
                "codecs": codecs,
                "sdp": sdp_body[:2000],
                "samplePath": sample_path,
                "diagnosis": diag,
            }
        )

    @actor_query
    def probeSse(self, url: str, max_events: int = 2, timeout_sec: int = 8) -> Any:
        """Validate an HTTP Server-Sent Events (`text/event-stream`) endpoint and read up to `max_events` events."""
        from actorii.core import Symbol

        u = str(url or "").strip()
        want_ev = max(1, min(10, int(max_events or 2)))
        timeout_f = max(1.5, min(25.0, float(timeout_sec or 8)))
        try:
            parsed = _validate_probe_target_url(u, allowed_schemes=("http", "https"), label="SSE Probe URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("sse"), "url": u, "eventsRead": 0, "diagnosis": str(exc)})

        sock = None
        try:
            sock, host, port, path_q = self._open_probe_socket(parsed, timeout_f)
            host_hdr = f"{host}:{port}" if parsed.port else host
            req = (
                f"GET {path_q} HTTP/1.1\r\n"
                f"Host: {host_hdr}\r\n"
                "Accept: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(req)

            buf = bytearray()
            deadline = time.monotonic() + timeout_f
            while b"\r\n\r\n" not in buf and time.monotonic() < deadline and len(buf) < 16384:
                ch = sock.recv(2048)
                if not ch:
                    break
                buf.extend(ch)
            raw_init = bytes(buf)
            if not raw_init[:5].upper().startswith(b"HTTP/"):
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("sse"),
                        "url": u,
                        "eventsRead": 0,
                        "diagnosis": self._diagnose_non_http_prefix(raw_init, "HTTP/1.1"),
                    }
                )
            hdr_b, _, rem = raw_init.partition(b"\r\n\r\n")
            lines = hdr_b.decode("iso-8859-1", errors="replace").split("\r\n")
            status_line = lines[0].strip() if lines else ""
            parts = status_line.split(None, 2)
            status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            hmap: Dict[str, str] = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    hmap[k.strip().lower()] = v.strip()
            ctype = hmap.get("content-type", "")
            if status_code != 200 or "text/event-stream" not in ctype.lower():
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("sse"),
                        "url": u,
                        "statusCode": status_code,
                        "contentType": ctype,
                        "eventsRead": 0,
                        "diagnosis": f"Expected `200 OK` + `Content-Type: text/event-stream`, got `{status_line}` (`{ctype}`).",
                    }
                )

            ev_buf = rem.decode("utf-8", errors="replace").replace("\r\n", "\n")
            events: List[str] = []
            while len(events) < want_ev and time.monotonic() < deadline and len(ev_buf) < 65536:
                while "\n\n" in ev_buf and len(events) < want_ev:
                    block, ev_buf = ev_buf.split("\n\n", 1)
                    data_lines = [ln[5:].lstrip() for ln in block.split("\n") if ln.startswith("data:")]
                    if data_lines:
                        events.append("\n".join(data_lines))
                if len(events) >= want_ev:
                    break
                rem_t = deadline - time.monotonic()
                if rem_t <= 0:
                    break
                try:
                    sock.settimeout(max(0.1, rem_t))
                    ch = sock.recv(2048)
                    if not ch:
                        break
                    ev_buf += ch.decode("utf-8", errors="replace").replace("\r\n", "\n")
                except socket.timeout:
                    break

            ok = len(events) > 0
            return _to_actor_dict(
                {
                    "ok": ok,
                    "protocol": Symbol("sse"),
                    "url": u,
                    "statusCode": status_code,
                    "contentType": ctype,
                    "eventsRead": len(events),
                    "events": events,
                    "diagnosis": (
                        f"Verified SSE `text/event-stream` ({len(events)} event(s) received)."
                        if ok
                        else "Valid `text/event-stream` headers, but 0 `data: ...\\n\\n` events arrived before timeout."
                    ),
                }
            )
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("sse"),
                    "url": u,
                    "eventsRead": 0,
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    @actor_query
    def probeWebSocket(self, ws_url: str, send_text: str = "", timeout_sec: int = 8) -> Any:
        """Perform a real RFC 6455 WebSocket `101 Switching Protocols` handshake, optionally send a masked text frame & read 1 frame."""
        import base64
        import hashlib

        from actorii.core import Symbol

        u = str(ws_url or "").strip()
        timeout_f = max(1.5, min(20.0, float(timeout_sec or 8)))
        try:
            parsed = _validate_probe_target_url(u, allowed_schemes=("ws", "wss"), label="WebSocket Probe URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("ws"), "url": u, "statusCode": 0, "diagnosis": str(exc)})

        sock = None
        try:
            sock, host, port, path_q = self._open_probe_socket(parsed, timeout_f)
            raw_nonce = os.urandom(16)
            ws_key = base64.b64encode(raw_nonce).decode("ascii")
            expected_accept = base64.b64encode(
                hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
            ).decode("ascii")
            host_hdr = f"{host}:{port}" if parsed.port else host
            req = (
                f"GET {path_q} HTTP/1.1\r\n"
                f"Host: {host_hdr}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {ws_key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii", errors="ignore")
            sock.sendall(req)

            buf = bytearray()
            while b"\r\n\r\n" not in buf and len(buf) < 16384:
                ch = sock.recv(2048)
                if not ch:
                    break
                buf.extend(ch)
            hdr_b, _, rem = bytes(buf).partition(b"\r\n\r\n")
            lines = hdr_b.decode("iso-8859-1", errors="replace").split("\r\n")
            status_line = lines[0].strip() if lines else ""
            parts = status_line.split(None, 2)
            status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            hmap: Dict[str, str] = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    hmap[k.strip().lower()] = v.strip()

            got_accept = hmap.get("sec-websocket-accept", "")
            if status_code != 101 or got_accept != expected_accept:
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("ws"),
                        "url": u,
                        "statusCode": status_code,
                        "statusLine": status_line,
                        "diagnosis": (
                            f"RFC 6455 handshake failed: expected `101 Switching Protocols` + valid `Sec-WebSocket-Accept`, "
                            f"got `{status_line}` (accept_match={got_accept == expected_accept})."
                        ),
                    }
                )

            if send_text:
                payload = str(send_text).encode("utf-8")[:125]
                mask = os.urandom(4)
                masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                frame = bytes([0x81, 0x80 | len(payload)]) + mask + masked
                sock.sendall(frame)

            recv_preview = ""
            sock.settimeout(min(2.0, timeout_f))
            try:
                fbuf = bytearray(rem)
                while len(fbuf) < 2:
                    ch = sock.recv(1024)
                    if not ch:
                        break
                    fbuf.extend(ch)
                if len(fbuf) >= 2:
                    opcode = fbuf[0] & 0x0F
                    plen = fbuf[1] & 0x7F
                    offset = 2
                    if plen == 126:
                        while len(fbuf) < 4:
                            ch = sock.recv(1024)
                            if not ch:
                                break
                            fbuf.extend(ch)
                        if len(fbuf) >= 4:
                            plen = int.from_bytes(fbuf[2:4], "big")
                            offset = 4
                        else:
                            plen = 0
                    elif plen == 127:
                        while len(fbuf) < 10:
                            ch = sock.recv(1024)
                            if not ch:
                                break
                            fbuf.extend(ch)
                        if len(fbuf) >= 10:
                            plen = int.from_bytes(fbuf[2:10], "big")
                            offset = 10
                        else:
                            plen = 0
                    while len(fbuf) < offset + min(plen, 4096):
                        ch = sock.recv(2048)
                        if not ch:
                            break
                        fbuf.extend(ch)
                    raw_msg = bytes(fbuf[offset : offset + min(plen, 4096)])
                    recv_preview = raw_msg.decode("utf-8", errors="replace")[:300] if opcode == 0x1 else f"<opcode={opcode} len={len(raw_msg)}>"
            except socket.timeout:
                pass

            # Send clean masked close frame (0x88)
            try:
                sock.sendall(b"\x88\x80" + os.urandom(4))
            except Exception:
                pass

            return _to_actor_dict(
                {
                    "ok": True,
                    "protocol": Symbol("ws"),
                    "url": u,
                    "statusCode": 101,
                    "statusLine": status_line,
                    "receivedMessage": recv_preview,
                    "diagnosis": f"Verified RFC 6455 WebSocket `101 Switching Protocols` handshake" + (f" | recv={recv_preview!r}" if recv_preview else "."),
                }
            )
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("ws"),
                    "url": u,
                    "statusCode": 0,
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    @actor_query
    def probeHls(self, m3u8_url: str, timeout_sec: int = 10) -> Any:
        """Validate an HTTP Live Streaming (`.m3u8`) playlist and verify its first `.ts` / `fMP4` media segment."""
        from actorii.core import Symbol

        u = str(m3u8_url or "").strip()
        timeout_f = max(2.0, min(25.0, float(timeout_sec or 10)))
        try:
            _validate_probe_target_url(u, allowed_schemes=("http", "https"), label="HLS Playlist URL")
        except (PermissionError, ValueError) as exc:
            return _to_actor_dict({"ok": False, "protocol": Symbol("hls"), "url": u, "diagnosis": str(exc)})

        try:
            req = urllib.request.Request(
                u,
                headers={"User-Agent": "ActorII-HLSProbe/4.4", "Accept": "application/vnd.apple.mpegurl,application/x-mpegURL,*/*"},
            )
            _hls_val = lambda redir_u: _validate_probe_target_url(redir_u, allowed_schemes=("http", "https"), label="HLS Redirect URL")
            with _safe_urlopen(req, timeout=timeout_f, validator=_hls_val) as resp:
                status_code = int(getattr(resp, "status", 200))
                ctype = resp.headers.get("Content-Type", "")
                playlist = resp.read(262144).decode("utf-8", errors="replace")

            if not playlist.lstrip().startswith("#EXTM3U"):
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("hls"),
                        "url": u,
                        "statusCode": status_code,
                        "contentType": ctype,
                        "diagnosis": "Invalid HLS playlist: response body does not start with `#EXTM3U`.",
                    }
                )

            entries = [ln.strip() for ln in playlist.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if not entries:
                return _to_actor_dict(
                    {
                        "ok": False,
                        "protocol": Symbol("hls"),
                        "url": u,
                        "statusCode": status_code,
                        "contentType": ctype,
                        "diagnosis": "HLS `#EXTM3U` header found, but playlist contains 0 segment or variant URIs yet.",
                    }
                )

            first_uri = urllib.parse.urljoin(u, entries[0])
            _validate_probe_target_url(first_uri, allowed_schemes=("http", "https"), label="HLS Segment URL")
            seg_req = urllib.request.Request(first_uri, headers={"User-Agent": "ActorII-HLSProbe/4.4"})
            with _safe_urlopen(seg_req, timeout=timeout_f, validator=_hls_val) as seg_resp:
                seg_status = int(getattr(seg_resp, "status", 200))
                seg_head = seg_resp.read(1024)

            is_sub_playlist = seg_head.lstrip().startswith(b"#EXTM3U")
            is_mpegts = len(seg_head) >= 189 and seg_head[0] == 0x47 and seg_head[188] == 0x47
            is_fmp4 = b"ftyp" in seg_head[:32] or b"moof" in seg_head[:64] or b"styp" in seg_head[:32]
            seg_ok = seg_status == 200 and (is_sub_playlist or is_mpegts or is_fmp4 or (len(seg_head) > 0 and seg_head[0] == 0x47))
            container = "variant-m3u8" if is_sub_playlist else ("mpeg-ts" if is_mpegts else ("fmp4" if is_fmp4 else "unknown"))
            return _to_actor_dict(
                {
                    "ok": seg_ok,
                    "protocol": Symbol("hls"),
                    "url": u,
                    "statusCode": status_code,
                    "contentType": ctype,
                    "entryCount": len(entries),
                    "firstSegmentUrl": first_uri,
                    "container": container,
                    "diagnosis": (
                        f"Verified HLS `#EXTM3U` playlist ({len(entries)} entries) and first `{container}` segment (`200 OK`)."
                        if seg_ok
                        else f"Playlist parsed ({len(entries)} entries), but first segment `{first_uri}` failed container check (status={seg_status}, container={container})."
                    ),
                }
            )
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "protocol": Symbol("hls"),
                    "url": u,
                    "diagnosis": f"{type(exc).__name__}: {exc}",
                }
            )


@actor_bridge(name="PDFDocumentTwin")
class PDFDocumentTwin:
    """Read-only PDF Document & URL Reader Twin exposed at `Actor[#PDF]`.

    Safely downloads remote `http://` / `https://` PDFs into `/tmp/actorii_pdf_cache/`
    (never mutating protected host directories) and extracts/searches bounded text
    pages via `pypdf` without requiring `execShell`.
    """

    def __init__(self, cache_dir: str = "/tmp/actorii_pdf_cache") -> None:
        from pathlib import Path

        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._url_cache: OrderedDict[str, str] = OrderedDict()
        self._reader_cache: OrderedDict[Tuple[str, int, int], Any] = OrderedDict()

    def _get_reader(self, local_path: str) -> Any:
        import pypdf

        st = os.stat(local_path)
        key = (local_path, st.st_mtime_ns, st.st_size)
        cached = self._reader_cache.get(key)
        if cached is not None:
            self._reader_cache.move_to_end(key)
            return cached
        reader = pypdf.PdfReader(local_path)
        self._reader_cache[key] = reader
        while len(self._reader_cache) > 3:
            self._reader_cache.popitem(last=False)
        return reader

    def _resolve_source(self, source: str) -> str:
        import hashlib
        from pathlib import Path

        raw = str(source or "").strip()
        if not raw:
            raise ValueError("Empty PDF path or URL")

        if raw.lower().startswith(("http://", "https://")):
            _validate_public_http_url(raw, label="PDF URL")
            if raw in self._url_cache and os.path.isfile(self._url_cache[raw]):
                return self._url_cache[raw]
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            url_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
            base_part = raw.split("?")[0].rstrip("/").split("/")[-1]
            if not base_part.lower().endswith(".pdf"):
                base_part = f"doc_{url_hash}.pdf"
            else:
                safe_stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base_part[:-4])[:60]
                base_part = f"{safe_stem}_{url_hash}.pdf"
            dest = self.cache_dir / base_part
            if not dest.is_file() or dest.stat().st_size == 0:
                req = urllib.request.Request(
                    raw,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ActorII-PDFTwin/4.1",
                        "Accept": "application/pdf,*/*",
                    },
                )
                with _safe_urlopen(req, timeout=45.0, validator=lambda u: _validate_public_http_url(u, label="PDF Redirect URL")) as resp:
                    data = resp.read(50 * 1024 * 1024)  # 50 MB safety cap
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"Downloaded URL does not appear to be a valid PDF (%PDF header missing): {raw}")
                _write_bytes_atomic(dest, data, mode=0o600)
            self._url_cache[raw] = str(dest)
            self._url_cache.move_to_end(raw)
            while len(self._url_cache) > 32:
                self._url_cache.popitem(last=False)
            return str(dest)

        p = Path(raw).expanduser().resolve()
        from actorii.harness.host_bridge import FORBIDDEN_READ_PREFIXES

        p_str = str(p)
        for fp in FORBIDDEN_READ_PREFIXES:
            if p_str == fp or p_str.startswith(fp + "/"):
                raise PermissionError(f"Access to restricted path '{p_str}' is forbidden.")
        if not p.is_file():
            # Also check inside `/tmp/actorii_pdf_cache` by exact basename or stem prefix
            bname = Path(raw).name
            cand = self.cache_dir / bname
            if cand.is_file():
                return str(cand)
            stem = bname[:-4] if bname.lower().endswith(".pdf") else bname
            if stem and self.cache_dir.is_dir():
                for m in sorted(self.cache_dir.glob(f"{stem}*.pdf")):
                    if m.is_file():
                        return str(m)
            raise FileNotFoundError(f"PDF file not found: {raw}")
        return p_str

    @actor_query
    def summary(self) -> str:
        """Return status of `Actor[#PDF]` and list cached PDF files in `/tmp/actorii_pdf_cache`."""
        try:
            import pypdf

            ver = getattr(pypdf, "__version__", "installed")
        except Exception:
            ver = "unavailable"
        cached = []
        if self.cache_dir.is_dir():
            for f in sorted(self.cache_dir.glob("*.pdf")):
                cached.append(f"{f.name} ({f.stat().st_size // 1024} KB)")
        return f"PDFDocumentTwin(pypdf={ver} | cache_dir={self.cache_dir} | cached={cached})"

    @actor_query
    def fetchUrl(self, url: str) -> str:
        """Download a remote PDF URL into `/tmp/actorii_pdf_cache/` and return its local path and page count."""
        local_path = self._resolve_source(url)
        reader = self._get_reader(local_path)
        size_kb = os.path.getsize(local_path) // 1024
        return f"[PDF Cached] path={local_path} | pages={len(reader.pages)} | size={size_kb} KB"

    @actor_query
    def info(self, path_or_url: str) -> str:
        """Return page count, file size, and PDF metadata for a local path or remote URL."""
        local_path = self._resolve_source(path_or_url)
        reader = self._get_reader(local_path)
        meta = reader.metadata or {}
        title = str(meta.get("/Title") or "")
        author = str(meta.get("/Author") or "")
        size_kb = os.path.getsize(local_path) // 1024
        return (
            f"[PDF Info] file={os.path.basename(local_path)} | path={local_path} | "
            f"pages={len(reader.pages)} | size={size_kb} KB | title={title!r} | author={author!r}"
        )

    @actor_query
    def pageCount(self, path_or_url: str) -> int:
        """Return integer total number of pages in `path_or_url`."""
        local_path = self._resolve_source(path_or_url)
        reader = self._get_reader(local_path)
        return len(reader.pages)

    @actor_query
    def readPages(
        self,
        path_or_url: str,
        start_page: int = 1,
        end_page: int = 5,
        max_chars: int = 6000,
    ) -> str:
        """Extract clean text from pages `start_page..end_page` (1-based inclusive) of a PDF path or URL."""
        local_path = self._resolve_source(path_or_url)
        reader = self._get_reader(local_path)
        total = len(reader.pages)
        if total == 0:
            return f"[{os.path.basename(local_path)}: 0 pages]"
        s_idx = max(1, int(start_page or 1))
        e_idx = min(total, max(s_idx, int(end_page or s_idx)))
        limit = max(500, min(20000, int(max_chars or 6000)))

        chunks: List[str] = [
            f"[{os.path.basename(local_path)} — Pages {s_idx}..{e_idx} of {total}]"
        ]
        chars_so_far = len(chunks[0])
        for p_num in range(s_idx, e_idx + 1):
            txt = (reader.pages[p_num - 1].extract_text() or "").strip()
            block = f"\n--- [Page {p_num}/{total}] ---\n{txt}"
            if chars_so_far + len(block) > limit:
                rem = max(0, limit - chars_so_far)
                chunks.append(block[:rem] + "\n... [truncated at max_chars]")
                break
            chunks.append(block)
            chars_so_far += len(block)
        return "\n".join(chunks)

    @actor_query
    def searchText(
        self,
        path_or_url: str,
        query: str,
        max_hits: int = 8,
        max_chars: int = 6000,
    ) -> str:
        """Search all pages of a PDF (`path_or_url`) for `query` (case-insensitive substring or regex) and return matching page excerpts."""
        import re as _re

        local_path = self._resolve_source(path_or_url)
        q = str(query or "").strip()
        if not q:
            return "ERROR: Empty search query"
        reader = self._get_reader(local_path)
        total = len(reader.pages)
        hit_limit = max(1, min(25, int(max_hits or 8)))
        char_limit = max(500, min(20000, int(max_chars or 6000)))

        try:
            rx = _re.compile(q, _re.IGNORECASE)
        except Exception:
            rx = _re.compile(_re.escape(q), _re.IGNORECASE)

        matches: List[str] = []
        for p_num in range(1, total + 1):
            txt = reader.pages[p_num - 1].extract_text() or ""
            if not txt:
                continue
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            matched_snippets: List[str] = []
            for idx, ln in enumerate(lines):
                if rx.search(ln):
                    s_l = max(0, idx - 1)
                    e_l = min(len(lines), idx + 3)
                    snippet = " / ".join(lines[s_l:e_l])
                    matched_snippets.append(f"  • {snippet[:320]}")
                    if len(matched_snippets) >= 3:
                        break
            if matched_snippets:
                matches.append(f"Page {p_num}/{total}:\n" + "\n".join(matched_snippets))
                if len(matches) >= hit_limit:
                    break

        if not matches:
            return f"[{os.path.basename(local_path)}: 0 page matches for {q!r} across {total} pages]"
        header = f"[{os.path.basename(local_path)}: {len(matches)} matching page(s) for {q!r} (total {total} pages)]"
        out = header + "\n" + "\n\n".join(matches)
        return out[:char_limit]


@actor_bridge(name="ReverseProxyTwin")
class ReverseProxyTwin:
    """Digital Twin (`Actor[#Proxy]`) enabling Bilby to inspect and manage the WLAN/Tailscale HTTP & WebSocket reverse proxy."""

    def _urls_for_host(self, host: str, port: int, token_param: str = "") -> str:
        suffix = f"?pass={token_param}" if token_param else ""
        return (
            f"  Studio     : http://{host}:{port}/{suffix}\n"
            f"  Benchmarks : http://{host}:{port}/benchmarks{suffix}\n"
            f"  ThermoUI   : http://{host}:{port}/thermometer{suffix}"
        )

    @actor_query
    def summary(self) -> str:
        """One-line operational summary of the Bilby Reverse Proxy."""
        from actorii.reverse_proxy import discover_host_network_ips, load_proxy_config

        cfg = load_proxy_config()
        ips = discover_host_network_ips()
        lan_ip = ips.get("lan", "127.0.0.1")
        ts_ip = ips.get("tailscale", "")
        port = int(cfg.get("http_port", 8090))
        auth_state = "AUTH-REQUIRED" if cfg.get("require_auth") else "CIDR-ALLOWLIST"
        ts_part = f" | tailscale=http://{ts_ip}:{port}" if ts_ip else ""
        return (
            f"ReverseProxyTwin(mode={cfg.get('mode', 'lan')} | gate={auth_state} | "
            f"wlan=http://{lan_ip}:{port}{ts_part})"
        )

    @actor_query
    def status(self) -> str:
        """Detailed proxy status, active mode, network URLs, CIDRs, and auth gate."""
        from actorii.reverse_proxy import discover_host_network_ips, load_proxy_config

        cfg = load_proxy_config()
        ips = discover_host_network_ips()
        port = int(cfg.get("http_port", 8090))
        ws_port = int(cfg.get("ws_port", 8765))
        req_auth = bool(cfg.get("require_auth", False))
        tok = str(cfg.get("access_token", "")) if req_auth else ""

        lines = [
            f"🌐 Bilby Reverse Proxy (`:{port}` HTTP+WS / `:{ws_port}` WS -> `127.0.0.1:{cfg.get('upstream_http_port', 18090)}`)",
            f"• Mode          : `{cfg.get('mode', 'lan')}` (`lan` | `tailscale_only` | `local_only` | `paused`)",
            f"• Auth Gate     : {'🔒 Passcode / Magic-Link Cookie ENABLED' if req_auth else '🟢 Open to Allowed Subnets (`require_auth=false`)'}",
            f"• Allowed CIDRs : {', '.join(cfg.get('allowed_cidrs') or [])}",
            f"• Blocked IPs   : {', '.join(cfg.get('blocked_ips') or []) or '(none)'}",
        ]
        if ips.get("lan"):
            lines.append(f"\n📡 WLAN (`{ips['lan']}`):\n" + self._urls_for_host(ips["lan"], port, tok))
        if ips.get("tailscale"):
            lines.append(f"\n🔐 Tailscale (`{ips['tailscale']}`):\n" + self._urls_for_host(ips["tailscale"], port, tok))
        lines.append(f"\n🖥️ Localhost:\n  http://127.0.0.1:{port}/  |  http://127.0.0.1:{port}/benchmarks")
        return "\n".join(lines)

    @actor_query
    def urls(self) -> str:
        """Return ready-to-open WLAN and Tailscale Studio & Benchmark URLs (including magic-link token if enabled)."""
        return self.status()

    @actor_command
    def setMode(self, mode: str) -> str:
        """Set reverse proxy access mode: `lan`, `tailscale_only`, `local_only`, or `paused`."""
        from actorii.reverse_proxy import VALID_MODES, load_proxy_config, save_proxy_config

        norm = str(mode or "").strip().lstrip("#").lower()
        aliases = {
            "wlan": "lan",
            "open": "lan",
            "on": "lan",
            "tailscale": "tailscale_only",
            "vpn": "tailscale_only",
            "local": "local_only",
            "localhost": "local_only",
            "off": "paused",
            "pause": "paused",
        }
        norm = aliases.get(norm, norm)
        if norm not in VALID_MODES:
            return f"ERROR: Unsupported mode {mode!r}. Expected one of: {', '.join(VALID_MODES)}"
        cfg = load_proxy_config()
        cfg["mode"] = norm
        save_proxy_config(cfg)
        return f"✔ Reverse proxy mode updated to `{norm}` (effective immediately).\n\n{self.status()}"

    @actor_command
    def setAuth(self, enabled: Any) -> str:
        """Enable or disable the passcode / 1-tap magic-link cookie gate (`true` / `false`)."""
        from actorii.reverse_proxy import load_proxy_config, save_proxy_config

        if isinstance(enabled, str):
            flag = enabled.strip().lower() in ("1", "true", "on", "yes", "enable", "enabled")
        else:
            flag = bool(enabled)
        cfg = load_proxy_config()
        cfg["require_auth"] = flag
        save_proxy_config(cfg)
        state_lbl = "ENABLED (1-tap magic links below)" if flag else "DISABLED (subnet allowlist active)"
        return f"✔ Proxy passcode gate {state_lbl}.\n\n{self.status()}"

    @actor_command
    def rotateToken(self) -> str:
        """Rotate the proxy access token (invalidating prior browser session cookies) and return fresh links."""
        import secrets
        from actorii.reverse_proxy import load_proxy_config, save_proxy_config

        cfg = load_proxy_config()
        cfg["access_token"] = secrets.token_urlsafe(18)
        save_proxy_config(cfg)
        return f"✔ Proxy access token rotated (previous session cookies invalidated).\n\n{self.status()}"

    @actor_command
    def allowCidr(self, cidr: str) -> str:
        """Add an IP or CIDR subnet (e.g. `192.168.68.0/22`) to `allowed_cidrs`."""
        import ipaddress
        from actorii.reverse_proxy import load_proxy_config, save_proxy_config

        raw = str(cidr or "").strip()
        try:
            net = str(ipaddress.ip_network(raw, strict=False))
        except ValueError as exc:
            return f"ERROR: Invalid IP/CIDR {raw!r}: {exc}"
        cfg = load_proxy_config()
        cidrs = list(cfg.get("allowed_cidrs") or [])
        if net not in cidrs:
            cidrs.append(net)
        cfg["allowed_cidrs"] = cidrs
        save_proxy_config(cfg)
        return f"✔ Allowed CIDR `{net}` added ({len(cidrs)} total rules)."

    @actor_command
    def blockIp(self, ip_or_cidr: str) -> str:
        """Immediately block an IP address or CIDR at the reverse proxy."""
        import ipaddress
        from actorii.reverse_proxy import load_proxy_config, save_proxy_config

        raw = str(ip_or_cidr or "").strip()
        try:
            norm = str(ipaddress.ip_network(raw, strict=False)) if "/" in raw else str(ipaddress.ip_address(raw))
        except ValueError as exc:
            return f"ERROR: Invalid IP/CIDR {raw!r}: {exc}"
        if norm.startswith("127.") or norm == "::1":
            return "ERROR: Refusing to block loopback."
        cfg = load_proxy_config()
        blocked = list(cfg.get("blocked_ips") or [])
        if norm not in blocked:
            blocked.append(norm)
        cfg["blocked_ips"] = blocked
        save_proxy_config(cfg)
        return f"✔ Blocked `{norm}` at reverse proxy."

    @actor_command
    def unblockIp(self, ip_or_cidr: str) -> str:
        """Remove an IP address or CIDR from `blocked_ips`."""
        from actorii.reverse_proxy import load_proxy_config, save_proxy_config

        raw = str(ip_or_cidr or "").strip()
        cfg = load_proxy_config()
        before = list(cfg.get("blocked_ips") or [])
        after = [x for x in before if x != raw]
        cfg["blocked_ips"] = after
        save_proxy_config(cfg)
        return f"✔ Removed `{raw}` from blocked IPs ({len(after)} remaining)."

    @actor_query
    def recentLogs(self, limit: int = 10) -> str:
        """Return the last `limit` external HTTP/WebSocket access events seen by the reverse proxy."""
        from actorii.reverse_proxy import read_recent_proxy_logs

        events = read_recent_proxy_logs(limit=max(1, min(50, int(limit or 10))))
        if not events:
            return "No external proxy requests logged yet."
        rows = [
            f"• {e.get('ts', '')[:19]}Z | {e.get('ip', '?'):15s} | :{e.get('port', '8090')} | "
            f"{e.get('method', 'GET'):4s} {e.get('path', '/'):20s} -> {e.get('status', '?')} ({e.get('reason', '')})"
            for e in events
        ]
        return f"Recent Proxy Access Log ({len(rows)} entries):\n" + "\n".join(rows)


JETSON_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "do-release-upgrade",
    "apt upgrade",
    "apt-get upgrade",
    "apt full-upgrade",
    "apt-get full-upgrade",
    "apt dist-upgrade",
    "apt-get dist-upgrade",
    "nvidia-l4t-initrd",
    "nvidia-l4t-bootloader",
    "/boot/initrd",
    "extlinux.conf",
    "/dev/mmcblk0",
    "/dev/mmcblk2",
    "mkfs",
    "fdisk",
    "parted",
    "sfdisk",
    "dd if=",
    "rm -rf /",
    "rm -rf /*",
    "shutdown",
    "poweroff",
    "halt",
    "reboot",
    "init 0",
    "init 6",
    "iptables -f",
    "ufw reset",
)

_JETSON_WORD_BOUNDARY_PATTERNS = frozenset({"halt", "reboot", "shutdown", "poweroff", "mkfs", "fdisk", "parted", "sfdisk"})


@actor_bridge(name="JetsonEdgeTwin")
class JetsonEdgeTwin:
    """Guarded Edge Digital Twin (`Actor[#Jetson]`) for the NVIDIA Jetson dev board on the local LAN.

    Uses a dedicated single-purpose SSH key (`~/.actorii/jetson/id_ed25519`), list-argv `ssh` invocation
    (never invoking a shell on `valkyrie`), dynamic mDNS/LAN discovery (`discover` / `scanLan`), and
    hard guardrails protecting the dual-boot SD/eMMC failsafe bootloader (`jetson-tx1-edge-ops`).
    """

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self._state_dir = state_dir or (Path.home() / ".actorii" / "jetson")
        self._host_file = self._state_dir / "host.json"
        self._key_file = self._state_dir / "id_ed25519"
        self._known_hosts = self._state_dir / "known_hosts"
        self._last_scan_miss_ts: float = 0.0

    def _load_cfg(self) -> Dict[str, Any]:
        default_cfg: Dict[str, Any] = {
            "host": os.environ.get("ACTORII_JETSON_HOST", "192.168.68.60"),
            "mdns": os.environ.get("ACTORII_JETSON_MDNS", "jetson.local"),
            "user": os.environ.get("ACTORII_JETSON_USER", "kilog"),
            "port": int(os.environ.get("ACTORII_JETSON_PORT", "22") or 22),
        }
        try:
            if self._host_file.is_file():
                raw = json.loads(self._host_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    default_cfg.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            pass
        return default_cfg

    def _save_cfg(self, cfg: Dict[str, Any]) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._host_file.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _probe_tcp_banner(ip: str, port: int = 22, timeout: float = 1.2) -> Optional[str]:
        try:
            with socket.create_connection((ip, int(port)), timeout=timeout) as sock:
                if int(port) == 22:
                    sock.settimeout(min(1.5, timeout + 0.3))
                    banner = sock.recv(128).decode("utf-8", errors="replace").strip()
                    return banner or "OPEN"
                return "OPEN"
        except Exception:
            return None

    def _resolve_live_host(self, auto_discover: bool = True) -> Tuple[str, Dict[str, Any], Optional[str]]:
        cfg = self._load_cfg()
        port = int(cfg.get("port", 22) or 22)
        cur_host = str(cfg.get("host", "192.168.68.60")).strip()
        banner = self._probe_tcp_banner(cur_host, port=port, timeout=1.4)
        if banner:
            return cur_host, cfg, banner

        mdns_name = str(cfg.get("mdns", "jetson.local")).strip()
        if mdns_name:
            try:
                resolved_ip = socket.gethostbyname(mdns_name)
                if resolved_ip and resolved_ip != cur_host:
                    b2 = self._probe_tcp_banner(resolved_ip, port=port, timeout=1.5)
                    if b2:
                        cfg["host"] = resolved_ip
                        self._save_cfg(cfg)
                        return resolved_ip, cfg, b2
            except Exception:
                pass

        if auto_discover:
            now = time.monotonic()
            if (now - self._last_scan_miss_ts) >= 30.0:
                found_ip, b3 = self._scan_for_jetson_ip(port=port)
                if found_ip:
                    self._last_scan_miss_ts = 0.0
                    cfg["host"] = found_ip
                    self._save_cfg(cfg)
                    return found_ip, cfg, b3
                self._last_scan_miss_ts = now

        return cur_host, cfg, None

    def _scan_for_jetson_ip(self, port: int = 22) -> Tuple[Optional[str], Optional[str]]:
        import concurrent.futures

        candidates = [f"192.168.68.{h}" for h in range(1, 255)]
        hits: List[Tuple[str, str]] = []

        def _check(ip: str) -> Optional[Tuple[str, str]]:
            b = self._probe_tcp_banner(ip, port=port, timeout=0.9)
            return (ip, b) if b else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            for item in pool.map(_check, candidates):
                if item:
                    hits.append(item)

        local_ips = _discover_local_host_ips()
        for ip, banner in hits:
            if ip not in local_ips and ("Ubuntu" in banner or "OpenSSH_7." in banner or "OpenSSH_8." in banner):
                return ip, banner
        return (hits[0][0], hits[0][1]) if len(hits) == 1 and hits[0][0] not in local_ips else (None, None)

    def _ssh_exec_raw(self, remote_cmd: str, timeout_sec: float = 15.0) -> Dict[str, Any]:
        host, cfg, banner = self._resolve_live_host(auto_discover=True)
        if banner is None:
            return {
                "ok": False,
                "host": host,
                "exitCode": -1,
                "stdout": "",
                "stderr": f"Jetson unreachable on {host}:{cfg.get('port', 22)} (SSH offline or asleep).",
            }
        ssh_bin = shutil.which("ssh") or "/usr/bin/ssh"
        if not os.path.isfile(ssh_bin):
            return {
                "ok": False,
                "host": host,
                "exitCode": -1,
                "stdout": "",
                "stderr": "Local `ssh` client binary not installed.",
            }
        user = str(cfg.get("user", "kilog")).strip() or "kilog"
        port = int(cfg.get("port", 22) or 22)
        argv = [
            ssh_bin,
            "-F",
            "/dev/null",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self._known_hosts}",
        ]
        if self._key_file.is_file():
            argv.extend(["-i", str(self._key_file), "-o", "IdentitiesOnly=yes"])
        safe_cmd = str(remote_cmd or "").replace("\x00", "\\0")
        argv.extend([f"{user}@{host}", safe_cmd])

        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=max(2.0, min(120.0, float(timeout_sec or 15.0))),
            )
            return {
                "ok": proc.returncode == 0,
                "host": host,
                "exitCode": int(proc.returncode),
                "stdout": (proc.stdout or "").strip()[:8000],
                "stderr": (proc.stderr or "").strip()[:4000],
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "host": host,
                "exitCode": 124,
                "stdout": "",
                "stderr": f"SSH command timed out after {timeout_sec}s on {user}@{host}.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "host": host,
                "exitCode": -1,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _validate_jetson_command(cmd: str, *, is_source_file: bool = False) -> Optional[str]:
        low = " ".join((cmd or "").lower().split())
        if not low:
            return "Empty command."
        for pat in JETSON_FORBIDDEN_PATTERNS:
            if is_source_file and pat in ("halt", "shutdown", "reboot", "poweroff"):
                continue
            if pat in _JETSON_WORD_BOUNDARY_PATTERNS:
                matched = bool(re.search(rf"(?<![a-z0-9_]){re.escape(pat)}(?![a-z0-9_])", low))
            else:
                matched = pat in low
            if matched:
                return (
                    f"JETSON_GUARDRAIL_BLOCKED: Command matches forbidden pattern `{pat}` "
                    "(protects Tegra L4T R32.7.6 OS, dual-boot SD/eMMC failsafe `/boot/initrd`, and power state)."
                )
        return None

    @actor_query
    def scanLan(self) -> Any:
        """Scan `192.168.68.0/22` for reachable SSH (`:22`) and RTSP (`:8554`) edge devices without host shell."""
        import concurrent.futures

        subnets = [f"192.168.{b}.{h}" for b in (68, 69, 70, 71) for h in range(1, 255)]
        found: List[Any] = []

        def _probe_host(ip: str) -> Optional[Dict[str, Any]]:
            ssh_b = self._probe_tcp_banner(ip, port=22, timeout=0.8)
            rtsp_b = self._probe_tcp_banner(ip, port=8554, timeout=0.45) if ssh_b else None
            if ssh_b or rtsp_b:
                return {
                    "ip": ip,
                    "sshPort22": bool(ssh_b),
                    "sshBanner": ssh_b or "",
                    "rtspPort8554": bool(rtsp_b),
                }
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=96) as pool:
            for res in pool.map(_probe_host, subnets):
                if res:
                    found.append(_to_actor_dict(res))
        from actorii.classes.collections import ActorArray

        arr = ActorArray(len(found))
        for idx, item in enumerate(found):
            arr.atPut(idx, item)
        return arr

    @actor_command
    def discover(self) -> str:
        """Locate the Jetson board via mDNS / LAN SSH banner scan and update `Actor[#Jetson]` host config."""
        host, cfg, banner = self._resolve_live_host(auto_discover=True)
        if banner:
            return f"✔ Jetson reachable at {cfg.get('user', 'kilog')}@{host}:{cfg.get('port', 22)} ({banner})"
        return f"✖ Jetson not currently reachable on LAN (last known host={host})."

    @actor_command
    def setHost(self, host: str, user: str = "kilog") -> str:
        """Point `Actor[#Jetson]` at a specific LAN IP/hostname (e.g. when switching between TX1 and TX2)."""
        clean_h = str(host or "").strip()
        clean_u = str(user or "kilog").strip() or "kilog"
        if not clean_h or any(ch in clean_h for ch in (" ", ";", "&", "|", "`", "$")):
            return f"ERROR: Invalid hostname or IP {host!r}"
        cfg = self._load_cfg()
        cfg["host"] = clean_h
        cfg["user"] = clean_u
        self._save_cfg(cfg)
        banner = self._probe_tcp_banner(clean_h, int(cfg.get("port", 22) or 22), timeout=1.5)
        state = f"ONLINE ({banner})" if banner else "OFFLINE / SSH closed"
        return f"✔ Updated `Actor[#Jetson]` target to `{clean_u}@{clean_h}` -> {state}"

    @actor_query
    def telemetry(self) -> Any:
        """Query live hardware, thermal, memory, storage, boot-source, and RTSP camera telemetry from the Jetson."""
        probe_script = (
            "echo MODEL=$(tr -d '\\0' </proc/device-tree/model 2>/dev/null || uname -m); "
            "echo HOSTNAME=$(hostname); "
            "echo KERNEL=$(uname -r); "
            "echo BOOT=$(cat /etc/jetson-boot-source 2>/dev/null || findmnt -n -o SOURCE / 2>/dev/null || echo unknown); "
            "echo UPTIME=$(uptime -p 2>/dev/null || uptime); "
            "echo MEM=$(free -m | awk '/^Mem:/ {print $2\",\"$3\",\"$7}'); "
            "echo DISK=$(df -h / | awk 'NR==2 {print $1\",\"$2\",\"$4\",\"$5}'); "
            "echo TEMP=$(cat /sys/devices/virtual/thermal/thermal_zone1/temp 2>/dev/null || echo 0); "
            "echo RTSP=$(systemctl is-active jetson-rtsp.service 2>/dev/null || echo inactive)"
        )
        res = self._ssh_exec_raw(probe_script, timeout_sec=10.0)
        host = res["host"]
        rtsp_open = bool(self._probe_tcp_banner(host, port=8554, timeout=0.8))
        if not res["ok"]:
            return _to_actor_dict(
                {
                    "online": False,
                    "ip": host,
                    "error": res["stderr"],
                    "rtspPort8554": rtsp_open,
                }
            )

        kv: Dict[str, str] = {}
        for ln in res["stdout"].splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                kv[k.strip()] = v.strip()

        mem_parts = (kv.get("MEM") or "0,0,0").split(",")
        mem_total = int(mem_parts[0]) if len(mem_parts) > 0 and mem_parts[0].isdigit() else 0
        mem_used = int(mem_parts[1]) if len(mem_parts) > 1 and mem_parts[1].isdigit() else 0
        mem_avail = int(mem_parts[2]) if len(mem_parts) > 2 and mem_parts[2].isdigit() else 0
        raw_temp = int(kv.get("TEMP") or 0) if (kv.get("TEMP") or "").isdigit() else 0
        temp_c = round(raw_temp / 1000.0, 1) if raw_temp > 200 else float(raw_temp)

        return _to_actor_dict(
            {
                "online": True,
                "ip": host,
                "hostname": kv.get("HOSTNAME", "jetson"),
                "model": kv.get("MODEL", "Jetson"),
                "kernel": kv.get("KERNEL", ""),
                "bootSource": kv.get("BOOT", ""),
                "uptime": kv.get("UPTIME", ""),
                "memTotalMiB": mem_total,
                "memUsedMiB": mem_used,
                "memAvailMiB": mem_avail,
                "diskRoot": kv.get("DISK", ""),
                "tempC": temp_c,
                "rtspService": kv.get("RTSP", "unknown"),
                "rtspPort8554": rtsp_open,
                "rtspUrl1080p": f"rtsp://{host}:8554/cam",
                "rtspUrl720p": f"rtsp://{host}:8554/cam720",
            }
        )

    @actor_query
    def summary(self) -> str:
        """Return a 1-line live status summary of `Actor[#Jetson]`."""
        t = self.telemetry()
        if not t.at("online"):
            return f"JetsonEdgeTwin[OFFLINE @ {t.at('ip')}] ({t.at('error')})"
        return (
            f"JetsonEdgeTwin[ONLINE @ {t.at('ip')}] {t.at('model')} ({t.at('hostname')}, {t.at('kernel')}) | "
            f"RAM: {t.at('memAvailMiB')}/{t.at('memTotalMiB')} MiB avail | Disk(/): {t.at('diskRoot')} | "
            f"Temp: {t.at('tempC')}°C | RTSP(:8554): {t.at('rtspService')} ({'OPEN' if t.at('rtspPort8554') else 'CLOSED'})"
        )

    @actor_query
    def cameraStatus(self) -> str:
        """Inspect the onboard OV5693 MIPI-CSI camera (`/dev/video0`) and `jetson-rtsp.service`."""
        res = self._ssh_exec_raw(
            "ls -l /dev/video* 2>/dev/null || echo 'no /dev/video*'; systemctl is-active jetson-rtsp.service 2>/dev/null || true",
            timeout_sec=8.0,
        )
        host = res["host"]
        rtsp_open = bool(self._probe_tcp_banner(host, port=8554, timeout=0.8))
        if not res["ok"]:
            return f"Jetson camera check failed ({host}): {res['stderr']}"
        return (
            f"Jetson Camera @ {host} | port :8554={'OPEN' if rtsp_open else 'CLOSED'} | "
            f"streams: rtsp://{host}:8554/cam (1080p30), rtsp://{host}:8554/cam720 (720p60)\n{res['stdout']}"
        )

    @actor_command
    def exec(self, command: str, timeout_sec: int = 20) -> Any:
        """Execute a guarded shell command on the Jetson over SSH (never runs on `valkyrie`)."""
        violation = self._validate_jetson_command(command)
        if violation:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": self._load_cfg().get("host", "192.168.68.60"),
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": violation,
                }
            )
        raw = self._ssh_exec_raw(str(command), timeout_sec=float(timeout_sec or 20))
        return _to_actor_dict(raw)

    @actor_command
    def writeFile(self, remote_path: str, content: str, timeout_sec: int = 20) -> Any:
        """Base64-upload any source/config/text file (`.cu`, `.c`, `.cpp`, `.py`, `.sh`, `.json`) to `/tmp/...` on the Jetson without executing it (zero quote-escaping issues)."""
        import base64
        import shlex

        raw_r = str(remote_path or "").strip()
        norm_r = os.path.normpath(raw_r) if raw_r else ""
        fallback_host = self._load_cfg().get("host", "192.168.68.60")
        if not norm_r.startswith("/tmp/") or norm_r == "/tmp" or "\x00" in raw_r:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": fallback_host,
                    "exitCode": 126,
                    "remotePath": raw_r,
                    "bytes": 0,
                    "stdout": "",
                    "stderr": f"JETSON_GUARDRAIL_BLOCKED: `writeFile` remote_path must be inside `/tmp/` (got {raw_r!r}).",
                }
            )
        body = str(content if content is not None else "")
        _, ext = os.path.splitext(norm_r)
        is_src = ext.lower() in (".cu", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".py", ".json", ".txt", ".md", ".yaml", ".yml", ".html", ".js", ".css")
        violation = self._validate_jetson_command(body, is_source_file=is_src)
        if violation:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": fallback_host,
                    "exitCode": 126,
                    "remotePath": norm_r,
                    "bytes": 0,
                    "stdout": "",
                    "stderr": violation,
                }
            )
        b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        q_rpath = shlex.quote(norm_r)
        safe_remote_cmd = f"printf '%s' '{b64}' | base64 -d > {q_rpath} && wc -c < {q_rpath}"
        raw = self._ssh_exec_raw(safe_remote_cmd, timeout_sec=float(timeout_sec or 20))
        written = len(body.encode("utf-8")) if raw.get("ok") else 0
        raw["remotePath"] = norm_r
        raw["bytes"] = written
        return _to_actor_dict(raw)

    @actor_command
    def writeAndRun(self, remote_path: str, script_content: str, timeout_sec: int = 35) -> Any:
        """Base64-upload a multi-line `.sh`, `.py`, `.cu` (CUDA `nvcc`), `.cpp`, or `.c` program to `/tmp/...` on the Jetson, compile if needed, and execute it cleanly (zero quote-escaping issues)."""
        import base64
        import shlex

        raw_r = str(remote_path or "/tmp/bilby_jetson_probe.sh").strip()
        norm_r = os.path.normpath(raw_r) if raw_r else ""
        if not norm_r.startswith("/tmp/") or norm_r == "/tmp" or "\x00" in raw_r:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": self._load_cfg().get("host", "192.168.68.60"),
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": f"JETSON_GUARDRAIL_BLOCKED: `writeAndRun` remote_path must be inside `/tmp/` (got {raw_r!r}).",
                }
            )
        body = str(script_content or "")
        violation = self._validate_jetson_command(body)
        if violation:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": self._load_cfg().get("host", "192.168.68.60"),
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": violation,
                }
            )
        b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        q_rpath = shlex.quote(norm_r)
        stem, ext = os.path.splitext(norm_r)
        ext_low = ext.lower()
        eff_timeout = int(timeout_sec or 35)
        if ext_low == ".cu":
            q_bin = shlex.quote(f"{stem}.bin")
            run_stage = (
                f"CCBIN=$([ -x /usr/bin/gcc-7 ] && echo '-ccbin /usr/bin/gcc-7' || echo ''); "
                f"/usr/local/cuda/bin/nvcc -O3 -arch=sm_53 $CCBIN -o {q_bin} {q_rpath} -lm && {q_bin}"
            )
            eff_timeout = max(eff_timeout, 45)
        elif ext_low in (".cpp", ".cc", ".cxx"):
            q_bin = shlex.quote(f"{stem}.bin")
            run_stage = f"g++ -O3 -o {q_bin} {q_rpath} -lm && {q_bin}"
            eff_timeout = max(eff_timeout, 35)
        elif ext_low == ".c":
            q_bin = shlex.quote(f"{stem}.bin")
            run_stage = f"gcc -O3 -o {q_bin} {q_rpath} -lm && {q_bin}"
            eff_timeout = max(eff_timeout, 35)
        elif ext_low == ".py":
            run_stage = f"chmod 700 {q_rpath} && python3 {q_rpath}"
        else:
            run_stage = f"chmod 700 {q_rpath} && bash {q_rpath}"

        safe_remote_cmd = f"printf '%s' '{b64}' | base64 -d > {q_rpath} && {run_stage}"
        return self.exec(safe_remote_cmd, eff_timeout)

    @actor_command
    def runCommand(self, command: str, timeout_sec: int = 20) -> Any:
        """Alias for `exec(Actor[#Jetson], command, timeout_sec)`."""
        return self.exec(command, timeout_sec)

    @actor_command
    def pullFile(self, remote_path: str, local_name: str = "", timeout_sec: int = 30) -> Any:
        """Binary-pull a regular file (`<= 50 MiB`) from the Jetson over SSH into `/tmp/actorii_media/` without stdout truncation."""
        import re
        import shlex

        r_path = str(remote_path or "").strip()
        cfg = self._load_cfg()
        fallback_host = cfg.get("host", "192.168.68.55")
        if not r_path or "\x00" in r_path or "\n" in r_path or "\r" in r_path:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": fallback_host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": "JETSON_PULL_ERROR: Missing or invalid remote_path.",
                }
            )

        low_r = posixpath.normpath(r_path).lower()
        blocked_remote = (
            "/.ssh/",
            "id_rsa",
            "id_ed25519",
            "id_ecdsa",
            "authorized_keys",
            "/etc/shadow",
            "/etc/gshadow",
            ".env",
            "/dev/",
            "/proc/",
            "/sys/",
        )
        if any(pat in low_r for pat in blocked_remote):
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": fallback_host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": f"JETSON_GUARDRAIL_BLOCKED: Remote path {r_path!r} is restricted (must be a regular non-credential file).",
                }
            )

        host, cfg, banner = self._resolve_live_host(auto_discover=True)
        if banner is None:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": f"Jetson unreachable on {host}:{cfg.get('port', 22)} (SSH offline or asleep).",
                }
            )

        ssh_bin = shutil.which("ssh") or "/usr/bin/ssh"
        if not os.path.isfile(ssh_bin):
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": "Local `ssh` client binary not installed.",
                }
            )

        raw_base = os.path.basename(str(local_name or "").strip() or r_path)
        safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", raw_base).lstrip(".") or "jetson_file.bin"
        if any(safe_base.lower() == b or safe_base.lower().endswith("." + b.lstrip(".")) for b in _SENSITIVE_UPLOAD_BASENAMES):
            safe_base = f"jetson_{safe_base}.dat"

        stage_dir = Path(os.environ.get("ACTORII_MEDIA_STAGE_DIR", TELEGRAM_MEDIA_STAGE_DIR))
        stage_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        dest_path = stage_dir / safe_base

        q_path = shlex.quote(r_path)
        remote_pull_cmd = (
            f"if [ ! -f {q_path} ]; then echo 'ERR_NOT_REGULAR_FILE' >&2; exit 2; fi; "
            f"if [ ! -r {q_path} ]; then echo 'ERR_NOT_READABLE' >&2; exit 3; fi; "
            f"sz=$(wc -c < {q_path}); "
            f"if [ \"$sz\" -gt {TELEGRAM_MAX_DOCUMENT_BYTES} ]; then echo \"ERR_TOO_LARGE:$sz\" >&2; exit 4; fi; "
            f"cat -- {q_path}"
        )

        user = str(cfg.get("user", "kilog")).strip() or "kilog"
        port = int(cfg.get("port", 22) or 22)
        argv = [
            ssh_bin,
            "-F",
            "/dev/null",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self._known_hosts}",
        ]
        if self._key_file.is_file():
            argv.extend(["-i", str(self._key_file), "-o", "IdentitiesOnly=yes"])
        argv.extend([f"{user}@{host}", remote_pull_cmd])

        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=False,
                timeout=max(3.0, min(120.0, float(timeout_sec or 30.0))),
            )
            if proc.returncode != 0:
                err_txt = (proc.stderr or b"").decode("utf-8", errors="replace").strip()[:2000]
                return _to_actor_dict(
                    {
                        "ok": False,
                        "host": host,
                        "remotePath": r_path,
                        "localPath": "",
                        "bytes": 0,
                        "stderr": err_txt or f"SSH pull exited with code {proc.returncode}",
                    }
                )
            raw_bytes = proc.stdout or b""
            _write_bytes_atomic(dest_path, raw_bytes, mode=0o600)
            return _to_actor_dict(
                {
                    "ok": True,
                    "host": host,
                    "remotePath": r_path,
                    "localPath": str(dest_path),
                    "bytes": len(raw_bytes),
                    "stderr": "",
                }
            )
        except subprocess.TimeoutExpired:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": f"SSH pullFile timed out after {timeout_sec}s on {user}@{host}.",
                }
            )
        except Exception as exc:
            return _to_actor_dict(
                {
                    "ok": False,
                    "host": host,
                    "remotePath": r_path,
                    "localPath": "",
                    "bytes": 0,
                    "stderr": f"{type(exc).__name__}: {exc}",
                }
            )

    @actor_command
    def fetchFile(self, remote_path: str, local_name: str = "", timeout_sec: int = 30) -> Any:
        """Alias for `pullFile(Actor[#Jetson], remote_path, local_name, timeout_sec)`."""
        return self.pullFile(remote_path=remote_path, local_name=local_name, timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # Persistent User-Space Service Lifecycle (`~/services/<name>`)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_service_name(name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_-]", "-", str(name or "").strip().lstrip("#")).strip("-").lower()
        if not clean or len(clean) > 48:
            raise ValueError(f"Invalid service name {name!r}; expected 1..48 alphanumeric/`-`/`_` chars.")
        if clean in ("ssh", "sshd", "systemd", "networkmanager", "gdm", "gdm3", "jetson-rtsp"):
            raise PermissionError(f"Service name {clean!r} is reserved for protected system daemons.")
        return clean

    @actor_command
    def deployService(
        self,
        service_name: str,
        work_dir: str,
        start_cmd: str,
        port: int = 0,
        timeout_sec: int = 25,
    ) -> Any:
        """Deploy/restart a persistent background user service in `~/services/<service_name>` on the Jetson without SSH FD hangs or `pkill -f` self-kills."""
        import base64
        import shlex

        fallback_host = self._load_cfg().get("host", "192.168.68.55")
        try:
            svc = self._sanitize_service_name(service_name)
        except (ValueError, PermissionError) as exc:
            return _to_actor_dict({"ok": False, "host": fallback_host, "service": str(service_name), "stderr": str(exc)})

        cmd_str = str(start_cmd or "").strip()
        wdir_str = str(work_dir or f"$HOME/services/{svc}").strip()
        if not cmd_str:
            return _to_actor_dict({"ok": False, "host": fallback_host, "service": svc, "stderr": "Empty `start_cmd`."})

        for check_txt in (cmd_str, wdir_str):
            violation = self._validate_jetson_command(check_txt)
            if violation:
                return _to_actor_dict({"ok": False, "host": fallback_host, "service": svc, "stderr": violation})

        if any(ch in wdir_str for ch in (";", "&", "|", "`", "\n", "\r", "$(", ">", "<")):
            return _to_actor_dict(
                {"ok": False, "host": fallback_host, "service": svc, "stderr": f"JETSON_GUARDRAIL_BLOCKED: Invalid shell metacharacters in `work_dir` ({wdir_str!r})."}
            )
        if wdir_str == "$HOME":
            work_dir_expr = '"$HOME"'
        elif wdir_str.startswith("$HOME/"):
            work_dir_expr = '"$HOME/"' + shlex.quote(wdir_str[len("$HOME/"):])
        else:
            work_dir_expr = shlex.quote(wdir_str)

        port_num = max(0, min(65535, int(port or 0)))
        runner_script = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"SVC_DIR=\"$HOME/services/{svc}\"\n"
            "mkdir -p \"$SVC_DIR\"\n"
            f"WORK_DIR={work_dir_expr}\n"
            "mkdir -p \"$WORK_DIR\"\n"
            "cd \"$WORK_DIR\"\n"
            f"exec {cmd_str}\n"
        )
        b64_runner = base64.b64encode(runner_script.encode("utf-8")).decode("ascii")

        deploy_sh = (
            f"SVC_DIR=\"$HOME/services/{svc}\"; "
            "mkdir -p \"$SVC_DIR\"; "
            "if [ -f \"$SVC_DIR/service.pid\" ]; then "
            "  OLD_PID=$(cat \"$SVC_DIR/service.pid\" 2>/dev/null || true); "
            "  if [ -n \"$OLD_PID\" ] && kill -0 \"$OLD_PID\" 2>/dev/null; then "
            "    kill -- -\"$OLD_PID\" 2>/dev/null || kill \"$OLD_PID\" 2>/dev/null || true; sleep 0.6; kill -9 -- -\"$OLD_PID\" 2>/dev/null || kill -9 \"$OLD_PID\" 2>/dev/null || true; "
            "  fi; "
            "fi; "
            f"printf '%s' '{b64_runner}' | base64 -d > \"$SVC_DIR/run.sh\"; "
            "chmod 700 \"$SVC_DIR/run.sh\"; "
            ": > \"$SVC_DIR/service.log\"; "
            "setsid nohup \"$SVC_DIR/run.sh\" </dev/null >>\"$SVC_DIR/service.log\" 2>&1 & "
            "NEW_PID=$!; echo \"$NEW_PID\" > \"$SVC_DIR/service.pid\"; "
        )
        if port_num > 0:
            deploy_sh += (
                f"for _i in 1 2 3 4 5 6 7 8 9 10; do "
                f"  if ss -tln 2>/dev/null | grep -q ':{port_num} '; then break; fi; "
                "  sleep 0.4; "
                "done; "
                f"LISTEN_OK=$(ss -tln 2>/dev/null | grep -c ':{port_num} ' || true); "
            )
        else:
            deploy_sh += "sleep 0.8; LISTEN_OK=1; "

        deploy_sh += (
            "ALIVE=$(kill -0 \"$NEW_PID\" 2>/dev/null && echo 1 || echo 0); "
            "echo \"PID=$NEW_PID ALIVE=$ALIVE LISTEN=$LISTEN_OK\"; "
            "tail -n 20 \"$SVC_DIR/service.log\" 2>/dev/null || true"
        )

        raw = self._ssh_exec_raw(deploy_sh, timeout_sec=float(timeout_sec or 25))
        out_txt = str(raw.get("stdout") or "")
        pid_val = 0
        alive = False
        listening = port_num == 0
        m = re.search(r"PID=(\d+)\s+ALIVE=(\d+)\s+LISTEN=(\d+)", out_txt)
        if m:
            pid_val = int(m.group(1))
            alive = int(m.group(2)) == 1
            listening = (int(m.group(3)) >= 1) if port_num > 0 else True
        ok = bool(raw.get("ok")) and alive and listening
        return _to_actor_dict(
            {
                "ok": ok,
                "host": raw.get("host", fallback_host),
                "service": svc,
                "pid": pid_val,
                "alive": alive,
                "port": port_num,
                "portListening": listening,
                "serviceDir": f"~/services/{svc}",
                "logPath": f"~/services/{svc}/service.log",
                "stdout": out_txt[:2000],
                "stderr": str(raw.get("stderr") or ""),
            }
        )

    @actor_query
    def serviceStatus(self, service_name: str, tail_lines: int = 25) -> Any:
        """Inspect PID health, listening sockets, and recent `service.log` lines for `~/services/<service_name>` on the Jetson."""
        fallback_host = self._load_cfg().get("host", "192.168.68.55")
        try:
            svc = self._sanitize_service_name(service_name)
        except (ValueError, PermissionError) as exc:
            return _to_actor_dict({"ok": False, "host": fallback_host, "service": str(service_name), "stderr": str(exc)})
        n_lines = max(5, min(100, int(tail_lines or 25)))
        check_sh = (
            f"SVC_DIR=\"$HOME/services/{svc}\"; "
            "if [ ! -d \"$SVC_DIR\" ]; then echo 'STATUS=NOT_DEPLOYED'; exit 0; fi; "
            "PID=$(cat \"$SVC_DIR/service.pid\" 2>/dev/null || echo 0); "
            "if [ -n \"$PID\" ] && [ \"$PID\" != \"0\" ] && kill -0 \"$PID\" 2>/dev/null; then "
            "  echo \"STATUS=RUNNING PID=$PID\"; ps -p \"$PID\" -o pid,etime,rss,args --no-headers 2>/dev/null || true; "
            "else "
            "  echo \"STATUS=STOPPED PID=$PID\"; "
            "fi; "
            "echo '---LOG---'; "
            f"tail -n {n_lines} \"$SVC_DIR/service.log\" 2>/dev/null || true"
        )
        raw = self._ssh_exec_raw(check_sh, timeout_sec=12.0)
        out_txt = str(raw.get("stdout") or "")
        running = "STATUS=RUNNING" in out_txt
        return _to_actor_dict(
            {
                "ok": bool(raw.get("ok")),
                "host": raw.get("host", fallback_host),
                "service": svc,
                "running": running,
                "stdout": out_txt[:2500],
                "stderr": str(raw.get("stderr") or ""),
            }
        )

    @actor_command
    def stopService(self, service_name: str) -> Any:
        """Cleanly stop a deployed user service in `~/services/<service_name>` via its recorded PID file (zero SSH self-kill risk)."""
        fallback_host = self._load_cfg().get("host", "192.168.68.55")
        try:
            svc = self._sanitize_service_name(service_name)
        except (ValueError, PermissionError) as exc:
            return _to_actor_dict({"ok": False, "host": fallback_host, "service": str(service_name), "stderr": str(exc)})
        stop_sh = (
            f"SVC_DIR=\"$HOME/services/{svc}\"; "
            "if [ -f \"$SVC_DIR/service.pid\" ]; then "
            "  PID=$(cat \"$SVC_DIR/service.pid\" 2>/dev/null || true); "
            "  if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then "
            "    kill -- -\"$PID\" 2>/dev/null || kill \"$PID\" 2>/dev/null || true; sleep 0.5; kill -9 -- -\"$PID\" 2>/dev/null || kill -9 \"$PID\" 2>/dev/null || true; "
            "    echo \"STOPPED PID=$PID\"; "
            "  else echo 'ALREADY_STOPPED'; fi; "
            "else echo 'NO_PID_FILE'; fi"
        )
        raw = self._ssh_exec_raw(stop_sh, timeout_sec=10.0)
        return _to_actor_dict(
            {
                "ok": bool(raw.get("ok")),
                "host": raw.get("host", fallback_host),
                "service": svc,
                "stdout": str(raw.get("stdout") or "").strip(),
                "stderr": str(raw.get("stderr") or ""),
            }
        )


# =============================================================================
# NetworkTopologyBridge (`Actor[#NetTopo]` backing Ring-1 `Actor[#Network]`)
# =============================================================================

_RESERVED_VALKYRIE_PORTS = frozenset({22, 80, 443, 5432, 7474, 7687, 8080, 8081, 8082, 8084, 8765, 8818, 8090, 11434, 18090, 18765})


@actor_bridge(name="NetworkTopologyBridge")
class NetworkTopologyBridge:
    """Ring-0 Network Topology, Tailnet Identity & Guarded LAN->Tailscale TCP Relay Bridge (`Actor[#NetTopo]`).

    Gives Bilby's Ring-1 `Actor[#Network]` object model ground-truth visibility into:
      1. Host network interfaces & subnets (`192.168.68.0/22` LAN vs `100.64.0.0/10` Tailscale CGNAT vs Docker bridges).
      2. Live Tailscale tailnet peers (`tailscale status --json`) so Bilby immediately knows which `100.x.x.x` IP belongs
         to `valkyrie` vs the user's phone (`kevins-s25-ultra`) vs whether an edge device (Jetson) is on the tailnet at all!
      3. Guarded user-space TCP port relays (`8091..8999`) bridging a Tailscale-reachable port on `valkyrie` to an
         authorized `192.168.x.x` LAN edge device (never allowing forwards to `127.0.0.1` or internal stack ports, and
         operating cleanly inside `ProtectSystem=strict` via `~/.actorii/port_bridges/`).
    """

    def __init__(self, state_dir: Optional[Path] = None, jetson_twin: Optional[JetsonEdgeTwin] = None) -> None:
        from actorii.harness.telemetry import resolve_actorii_state_dir

        self._state_dir = state_dir or (resolve_actorii_state_dir() / "port_bridges")
        self._registry_file = self._state_dir / "bridges.json"
        self._jetson_twin = jetson_twin
        self._managed_procs: Dict[int, subprocess.Popen[Any]] = {}
        self._ifaces_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])
        self._peers_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])

    def _reap_managed_procs(self) -> None:
        dead = [lp for lp, p in self._managed_procs.items() if p.poll() is not None]
        for lp in dead:
            self._managed_procs.pop(lp, None)

    @staticmethod
    def _pid_is_socat(pid_n: int) -> bool:
        try:
            cmdline = Path(f"/proc/{pid_n}/cmdline").read_bytes()
            return b"socat" in cmdline.lower()
        except OSError:
            return False

    @actor_query
    def hostInterfaces(self) -> Any:
        """Return live IPv4 interfaces on `valkyrie` (`ip -j -4 addr show`) as an Actor `Array` of `Dictionary` entries."""
        from actorii.classes.collections import ActorArray

        now = time.monotonic()
        ts, cached_rows = self._ifaces_cache
        if cached_rows and (now - ts) < 5.0:
            arr = ActorArray(len(cached_rows))
            for i, it in enumerate(cached_rows):
                arr.atPut(i, _to_actor_dict(it))
            return arr

        raw_rows: List[Dict[str, Any]] = []
        ip_bin = shutil.which("ip") or "/usr/sbin/ip"
        if os.path.isfile(ip_bin):
            try:
                proc = subprocess.run([ip_bin, "-j", "-4", "addr", "show"], capture_output=True, text=True, timeout=4.0)
                if proc.returncode == 0 and proc.stdout.strip():
                    raw_list = json.loads(proc.stdout)
                    for iface in raw_list:
                        ifname = str(iface.get("ifname") or "")
                        state = str(iface.get("operstate") or "UNKNOWN")
                        for addr in iface.get("addr_info") or []:
                            if addr.get("family") == "inet":
                                local_ip = str(addr.get("local") or "")
                                prefix = int(addr.get("prefixlen") or 32)
                                kind = (
                                    "loopback"
                                    if ifname == "lo" or local_ip.startswith("127.")
                                    else (
                                        "tailscale"
                                        if ifname.startswith("tailscale") or local_ip.startswith("100.")
                                        else ("docker" if ifname.startswith(("docker", "br-", "veth")) else "lan")
                                    )
                                )
                                raw_rows.append(
                                    {
                                        "interface": ifname,
                                        "ip": local_ip,
                                        "cidr": f"{local_ip}/{prefix}",
                                        "kind": kind,
                                        "state": state,
                                    }
                                )
            except Exception:
                pass
        if raw_rows:
            self._ifaces_cache = (now, raw_rows)
        arr = ActorArray(len(raw_rows))
        for i, it in enumerate(raw_rows):
            arr.atPut(i, _to_actor_dict(it))
        return arr

    @actor_query
    def tailnetPeers(self) -> Any:
        """Query `tailscale status --json` (read-only) and return all Tailnet nodes (`Self` + `Peer`s) with hostnames, OS, IPs, and online state."""
        from actorii.classes.collections import ActorArray

        now = time.monotonic()
        ts, cached_rows = self._peers_cache
        if cached_rows and (now - ts) < 5.0:
            arr = ActorArray(len(cached_rows))
            for i, it in enumerate(cached_rows):
                arr.atPut(i, _to_actor_dict(it))
            return arr

        raw_rows: List[Dict[str, Any]] = []
        ts_bin = shutil.which("tailscale") or "/usr/bin/tailscale"
        if os.path.isfile(ts_bin):
            try:
                proc = subprocess.run([ts_bin, "status", "--json"], capture_output=True, text=True, timeout=5.0)
                if proc.returncode == 0 and proc.stdout.strip():
                    data = json.loads(proc.stdout)
                    self_node = data.get("Self") or {}
                    self_ips = [str(x) for x in (self_node.get("TailscaleIPs") or []) if ":" not in str(x)]
                    if self_node:
                        raw_rows.append(
                            {
                                "role": "self",
                                "hostname": str(self_node.get("HostName") or "valkyrie"),
                                "dnsName": str(self_node.get("DNSName") or "").rstrip("."),
                                "os": str(self_node.get("OS") or "linux"),
                                "tailscaleIp": self_ips[0] if self_ips else "",
                                "online": bool(self_node.get("Online", True)),
                            }
                        )
                    for _, p in sorted((data.get("Peer") or {}).items()):
                        if not isinstance(p, dict):
                            continue
                        p_ips = [str(x) for x in (p.get("TailscaleIPs") or []) if ":" not in str(x)]
                        raw_rows.append(
                            {
                                "role": "peer",
                                "hostname": str(p.get("HostName") or "peer"),
                                "dnsName": str(p.get("DNSName") or "").rstrip("."),
                                "os": str(p.get("OS") or ""),
                                "tailscaleIp": p_ips[0] if p_ips else "",
                                "online": bool(p.get("Online", False)),
                            }
                        )
            except Exception:
                pass
        if raw_rows:
            self._peers_cache = (now, raw_rows)
        arr = ActorArray(len(raw_rows))
        for i, it in enumerate(raw_rows):
            arr.atPut(i, _to_actor_dict(it))
        return arr

    @actor_query
    def identifyIp(self, ip_or_host: str) -> Any:
        """Identify who owns `ip_or_host` across `valkyrie` interfaces, active Tailnet peers, and configured LAN Edge Twins (`Jetson`)."""
        raw = str(ip_or_host or "").strip()
        parsed_h = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}").hostname
        host_only = (parsed_h or raw.split("://", 1)[-1].split("/", 1)[0]).strip()
        host_low = host_only.lower()
        if not host_only:
            return _to_actor_dict({"ok": False, "ip": raw, "owner": "unknown", "network": "invalid", "note": "Empty IP/host."})

        # 1. Check Tailscale peers first (`100.x.x.x` confusion prevention!)
        ts_nodes = self.tailnetPeers()
        valkyrie_ts_ip = ""
        for node in ts_nodes:
            ts_ip = str(node.at("tailscaleIp") or "")
            hname = str(node.at("hostname") or "")
            role = str(node.at("role") or "")
            if role == "self" and ts_ip:
                valkyrie_ts_ip = ts_ip
            if host_low in (ts_ip.lower(), hname.lower(), str(node.at("dnsName") or "").lower()):
                note = (
                    "This is `valkyrie` (local Fedora host). Note: `:8090` on `valkyrie` is ActorII Web Studio, NOT the Jetson! Use a bridge port like `:8091` (`listBridges(Actor[#Network])!!`) to reach Jetson services over Tailscale."
                    if role == "self"
                    else f"This Tailscale IP belongs to `{hname}` ({node.at('os')}, role=client peer), NOT the Jetson! The Jetson is on LAN (`192.168.68.x`) unless Tailscale is installed on it or bridged via `valkyrie` (`{valkyrie_ts_ip}:<bridgePort>`)."
                )
                return _to_actor_dict(
                    {
                        "ok": True,
                        "ip": host_only,
                        "owner": hname,
                        "role": role,
                        "os": str(node.at("os") or ""),
                        "network": "tailscale",
                        "online": bool(node.at("online")),
                        "valkyrieTailscaleIp": valkyrie_ts_ip,
                        "note": note,
                    }
                )

        # 2. Check local `valkyrie` LAN / loopback / docker interfaces
        for iface in self.hostInterfaces():
            if str(iface.at("ip") or "") == host_only:
                return _to_actor_dict(
                    {
                        "ok": True,
                        "ip": host_only,
                        "owner": "valkyrie",
                        "role": "self",
                        "network": str(iface.at("kind") or "lan"),
                        "interface": str(iface.at("interface") or ""),
                        "valkyrieTailscaleIp": valkyrie_ts_ip,
                        "note": f"Local `valkyrie` `{iface.at('interface')}` address.",
                    }
                )

        # 3. Check configured Jetson LAN IP
        jetson_ip = ""
        if self._jetson_twin is not None:
            jetson_ip = str(self._jetson_twin._load_cfg().get("host") or "")
        if jetson_ip and host_low in (jetson_ip.lower(), "jetson", "jetson.local"):
            reachable = bool(JetsonEdgeTwin._probe_tcp_banner(jetson_ip, 22, 1.0))
            return _to_actor_dict(
                {
                    "ok": True,
                    "ip": jetson_ip,
                    "owner": "jetson-tx1",
                    "role": "edge_device",
                    "network": "lan_only",
                    "online": reachable,
                    "valkyrieTailscaleIp": valkyrie_ts_ip,
                    "note": (
                        f"NVIDIA Jetson TX1 on local Wi-Fi LAN (`{jetson_ip}`). Not directly reachable from remote Tailscale clients unless bridged via `valkyrie` (`exposeLanPort(Actor[#Network], {jetson_ip!r}, <remotePort>, <listenPort>)!!`)."
                    ),
                }
            )

        if host_only.startswith("100."):
            return _to_actor_dict(
                {
                    "ok": False,
                    "ip": host_only,
                    "owner": "unassigned_tailscale_ip",
                    "network": "tailscale",
                    "valkyrieTailscaleIp": valkyrie_ts_ip,
                    "note": f"`{host_only}` is in Tailscale CGNAT space (`100.64.0.0/10`) but is NOT registered to any peer on this Tailnet!",
                }
            )

        return _to_actor_dict(
            {
                "ok": True,
                "ip": host_only,
                "owner": "lan_or_external_host",
                "network": "lan" if host_only.startswith(("192.168.", "10.", "172.")) else "public",
                "valkyrieTailscaleIp": valkyrie_ts_ip,
                "note": "External or uncataloged LAN address.",
            }
        )

    def _load_bridges_meta(self) -> Dict[str, Dict[str, Any]]:
        try:
            if self._registry_file.is_file():
                data = json.loads(self._registry_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_bridges_meta(self, meta: Dict[str, Dict[str, Any]]) -> None:
        try:
            self._state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            self._registry_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    @actor_query
    def listBridges(self) -> Any:
        """Discover both systemd `socat` bridges (like `:8091 -> 192.168.68.57:8090`) and runtime `Actor[#Network]` port relays."""
        from actorii.classes.collections import ActorArray

        self._reap_managed_procs()
        bridges: List[Any] = []
        seen_ports: set[int] = set()

        # 1. Inspect active `socat TCP4-LISTEN:<port>... TCP4:<target_ip>:<target_port>` processes on `valkyrie`
        ps_bin = shutil.which("ps") or "/usr/bin/ps"
        if os.path.isfile(ps_bin):
            try:
                proc = subprocess.run([ps_bin, "-eo", "pid,args", "--no-headers"], capture_output=True, text=True, timeout=4.0)
                for ln in (proc.stdout or "").splitlines():
                    if "socat" not in ln or "TCP" not in ln:
                        continue
                    m = re.search(
                        r"^\s*(\d+)\s+\S*socat\s+TCP4?-LISTEN:(\d+)[^\s]*\s+TCP4?:([0-9a-zA-Z._-]+):(\d+)",
                        ln,
                    )
                    if m:
                        pid_n = int(m.group(1))
                        lport = int(m.group(2))
                        rip = m.group(3)
                        rport = int(m.group(4))
                        seen_ports.add(lport)
                        bridges.append(
                            _to_actor_dict(
                                {
                                    "listenPort": lport,
                                    "targetHost": rip,
                                    "targetPort": rport,
                                    "pid": pid_n,
                                    "alive": True,
                                    "managedBy": "socat",
                                }
                            )
                        )
            except Exception:
                pass

        # 2. Include metadata from `~/.actorii/port_bridges/bridges.json`
        meta = self._load_bridges_meta()
        for p_str, info in sorted(meta.items()):
            lport = int(p_str)
            if lport in seen_ports:
                continue
            pid_n = int(info.get("pid") or 0)
            alive = False
            if pid_n > 0:
                try:
                    os.kill(pid_n, 0)
                    alive = True
                except OSError:
                    alive = False
            bridges.append(
                _to_actor_dict(
                    {
                        "listenPort": lport,
                        "targetHost": str(info.get("targetHost") or ""),
                        "targetPort": int(info.get("targetPort") or 0),
                        "pid": pid_n,
                        "alive": alive,
                        "managedBy": "actorii",
                    }
                )
            )

        arr = ActorArray(len(bridges))
        for idx, b in enumerate(bridges):
            arr.atPut(idx, b)
        return arr

    @actor_command
    def exposeLanPort(self, target_host: str, target_port: int, listen_port: int = 8091) -> Any:
        """Create or verify a guarded user-space TCP relay on `valkyrie:<listen_port>` (`8091..8999`) -> `<target_host>:<target_port>` so Tailscale peers can reach a LAN edge service."""
        import ipaddress

        thost = str(target_host or "").strip()
        tport = int(target_port or 0)
        lport = int(listen_port or 8091)

        if lport in _RESERVED_VALKYRIE_PORTS or not (8091 <= lport <= 8999):
            return _to_actor_dict(
                {
                    "ok": False,
                    "listenPort": lport,
                    "error": f"BLOCKED: `listen_port` must be in safe relay range `8091..8999` and not a reserved stack port (got {lport}).",
                }
            )
        if not (1 <= tport <= 65535):
            return _to_actor_dict({"ok": False, "listenPort": lport, "error": f"Invalid `target_port` {tport}."})

        # Strictly require `target_host` to be a non-host RFC1918 LAN IP (e.g. `192.168.68.57`), NEVER loopback or `valkyrie` itself!
        local_ips = _discover_local_host_ips()
        try:
            addr = ipaddress.ip_address(thost)
        except ValueError:
            return _to_actor_dict(
                {
                    "ok": False,
                    "listenPort": lport,
                    "error": f"Invalid `target_host` {thost!r}: pass an explicit LAN IPv4 address (e.g. '192.168.68.57').",
                }
            )
        if (
            not isinstance(addr, ipaddress.IPv4Address)
            or not addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
            or thost in local_ips
        ):
            return _to_actor_dict(
                {
                    "ok": False,
                    "listenPort": lport,
                    "error": f"BLOCKED [RELAY_SSRF_GUARD]: `exposeLanPort` only bridges to external private RFC1918 IPv4 LAN edge devices, never loopback/link-local/IPv6 or `valkyrie` ({thost}).",
                }
            )

        # Check if an existing `socat` bridge is already serving this exact route!
        for b in self.listBridges():
            if int(b.at("listenPort") or 0) == lport and bool(b.at("alive")):
                if str(b.at("targetHost")) == thost and int(b.at("targetPort") or 0) == tport:
                    ts_ip = ""
                    for peer in self.tailnetPeers():
                        if str(peer.at("role")) == "self":
                            ts_ip = str(peer.at("tailscaleIp") or "")
                    return _to_actor_dict(
                        {
                            "ok": True,
                            "listenPort": lport,
                            "targetHost": thost,
                            "targetPort": tport,
                            "pid": int(b.at("pid") or 0),
                            "tailscaleUrl": f"http://{ts_ip}:{lport}/" if ts_ip else f"http://valkyrie:{lport}/",
                            "note": "Existing active TCP relay already forwarding this exact route.",
                        }
                    )
                if str(b.at("managedBy")) == "socat" and str(lport) not in self._load_bridges_meta():
                    return _to_actor_dict(
                        {
                            "ok": False,
                            "listenPort": lport,
                            "error": f"Port :{lport} is already bound by an external systemd socat unit targeting {b.at('targetHost')}:{b.at('targetPort')}. Pick another port in 8092..8999.",
                        }
                    )

        socat_bin = shutil.which("socat") or "/usr/bin/socat"
        if not os.path.isfile(socat_bin):
            return _to_actor_dict({"ok": False, "listenPort": lport, "error": "`socat` binary not installed on host."})

        # Stop any prior ActorII-managed relay on `lport`
        self.closeLanPort(lport)

        self._state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        log_file = self._state_dir / f"relay_{lport}.log"
        with log_file.open("ab") as lf:
            proc = subprocess.Popen(
                [
                    socat_bin,
                    f"TCP4-LISTEN:{lport},bind=0.0.0.0,reuseaddr,fork",
                    f"TCP4:{thost}:{tport}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        time.sleep(0.25)
        alive = proc.poll() is None
        if alive:
            self._managed_procs[lport] = proc
            meta = self._load_bridges_meta()
            meta[str(lport)] = {
                "listenPort": lport,
                "targetHost": thost,
                "targetPort": tport,
                "pid": proc.pid,
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._save_bridges_meta(meta)

        ts_ip = ""
        for peer in self.tailnetPeers():
            if str(peer.at("role")) == "self":
                ts_ip = str(peer.at("tailscaleIp") or "")

        return _to_actor_dict(
            {
                "ok": alive,
                "listenPort": lport,
                "targetHost": thost,
                "targetPort": tport,
                "pid": proc.pid if alive else 0,
                "tailscaleUrl": f"http://{ts_ip}:{lport}/" if ts_ip else "",
                "error": "" if alive else f"socat exited immediately (code={proc.returncode}); check if port {lport} is in use.",
            }
        )

    @actor_command
    def closeLanPort(self, listen_port: int) -> Any:
        """Stop an ActorII-managed user-space TCP relay on `listen_port` (never kills external systemd units)."""
        lport = int(listen_port or 0)
        meta = self._load_bridges_meta()
        entry = meta.pop(str(lport), None)
        proc = self._managed_procs.pop(lport, None)
        killed = False
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            killed = True
        elif entry and int(entry.get("pid") or 0) > 0:
            pid_n = int(entry["pid"])
            if self._pid_is_socat(pid_n):
                try:
                    os.kill(pid_n, 15)
                    killed = True
                except OSError:
                    pass
        self._save_bridges_meta(meta)
        return _to_actor_dict({"ok": True, "listenPort": lport, "stopped": killed})


__all__ = [
    "JetsonEdgeTwin",
    "NetTransportBridge",
    "NetworkTopologyBridge",
    "PDFDocumentTwin",
    "PantryGraphTwin",
    "ReverseEngineeringTwin",
    "ReverseProxyTwin",
    "SmartLightTwin",
    "TelegramBotTwin",
    "ValkyrieSystemTwin",
    "load_telegram_credentials",
    "telegram_api_request",
    "telegram_multipart_request",
]
