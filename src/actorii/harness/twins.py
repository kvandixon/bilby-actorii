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
import shutil
import socket
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional

from actorii.bridge import actor_bridge, actor_command, actor_query

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    os.environ.get("ACTORII_NEO4J_TX_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"),
)


@actor_bridge(name="ValkyrieSystemTwin")
class ValkyrieSystemTwin:
    """Read-only host GPU & local service telemetry twin exposed at `Actor[#Valkyrie]`."""

    def __init__(self) -> None:
        self._last_gpu: Dict[str, Any] = {}

    def _query_nvidia_smi(self) -> Dict[str, Any]:
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
            with socket.create_connection(("127.0.0.1", p), timeout=0.5):
                return "ONLINE"
        except Exception:
            return "OFFLINE"

    @actor_query
    def isSandboxed(self) -> bool:
        """Return True if running inside the isolated `actorii-sandbox` Docker container."""
        return os.environ.get("ACTORII_IN_DOCKER") == "1" or os.path.exists("/.dockerenv")

    @actor_query
    def summary(self) -> str:
        """Return a 1-line telemetry summary of GPU VRAM, sandbox mode, and core Valkyrie stack ports."""
        gpu_str = self.gpuStatus()
        mode = "DOCKER-SANDBOX" if self.isSandboxed() else "HOST-RESIDENT"
        ports = [
            (8080, "LlamaPrimaryGPU"),
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
        return f"[{mode}] {gpu_str} (Note: ~14.7GB/90.5% VRAM is intentionally pinned for qwen3.8:latest 64k context) | Services: {' '.join(port_strs)}"

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

    @actor_command
    def askSkippy(self, question: str) -> str:
        """Escalate a diagnostic question or architectural consult to Cloud Skippy via `ask-skippy`."""
        import shutil
        from pathlib import Path

        q = str(question or "").strip()
        if not q:
            return "ERROR: Empty question for askSkippy"
        ask_bin = shutil.which("ask-skippy") or str(Path.home() / "bin" / "ask-skippy")
        if os.path.isfile(ask_bin):
            try:
                out = subprocess.check_output(
                    [ask_bin, q],
                    timeout=15.0,
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip()
                return out[:2000] if out else "Logged escalation to Cloud Skippy."
            except Exception as exc:
                return f"askSkippy logged (note: {exc})"
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
        if not os.path.isfile(target):
            return f"File not found: {target}"
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
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
        from actorii.harness.host_bridge import validate_safe_shell_command

        try:
            validate_safe_shell_command(cmd_str)
        except PermissionError as exc:
            return f"BLOCKED: {exc}"
        # Quote bare #RRGGBB hex color arguments so /bin/sh does not treat `#` as a comment
        import re as _re

        cmd_str = _re.sub(r"(?<=\s)(#[0-9A-Fa-f]{3,8})(?=\s|$)", r"'\1'", cmd_str)
        try:
            proc = subprocess.run(
                cmd_str,
                shell=True,
                timeout=5.0,
                capture_output=True,
                text=True,
            )
            combined = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
            if len(combined) > 3000:
                combined = combined[:3000] + "...[truncated]"
            return f"[exit={proc.returncode}] {combined}" if combined else f"[exit={proc.returncode}] (no output)"
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
        target = str(path or "").strip()
        if not os.path.isfile(target):
            return f"File not found: {target}"
        size = os.path.getsize(target)
        h = hashlib.sha256()
        with open(target, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
        try:
            magic = subprocess.check_output(["file", "-b", target], timeout=5.0, text=True).strip()
        except Exception:
            magic = "unknown"
        return f"SHA256={sha} | Size={size} bytes | Type={magic}"

    @actor_query
    def inspectEntropy(self, path: str) -> str:
        """Run REA `inspect-byte-entropy` on `path` and return overall entropy and high-entropy block ratio."""
        target = str(path or "").strip()
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
        target = str(path or "").strip()
        if not os.path.isfile(target):
            return f"File not found: {target}"
        cmd = [self._rea_batch, target]
        procs = [p.strip() for p in str(procedures_csv or "").split(",") if p.strip()]
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
        target = str(path or "").strip()
        proc_sel = str(procedure or "entry").lstrip("#").strip()
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
        target = str(path or "").strip()
        if not os.path.isfile(target):
            return f"File not found: {target}"
        dur = max(4, min(30, int(duration_sec or 10)))
        rname = str(run_as or "sample.exe").strip()
        out_dir = f"/tmp/actor_vtsandbox_{int(os.path.getmtime(target))}"
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


def telegram_api_request(
    token: str,
    method: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Call `api.telegram.org/bot<token>/<method>` with automatic IPv4 SNI fallback (`149.154.166.110`)."""
    import http.client
    import ssl

    if not token:
        return {"ok": False, "description": "Missing TELEGRAM_BOT_TOKEN"}

    path = f"/bot{token}/{method}"
    body_bytes = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    headers = {"Host": "api.telegram.org", "Accept": "application/json"}
    if body_bytes is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))

    ctx = ssl.create_default_context()
    targets = ["api.telegram.org", *_TELEGRAM_SEED_IPS]
    last_err = "Unknown error"

    for target in targets:
        try:
            raw_sock = socket.create_connection((target, 443), timeout=timeout)
            tls_sock = ctx.wrap_socket(raw_sock, server_hostname="api.telegram.org")
            conn = http.client.HTTPSConnection("api.telegram.org", 443, timeout=timeout, context=ctx)
            conn.sock = tls_sock
            conn.request("POST" if body_bytes is not None else "GET", path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw_data = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return json.loads(raw_data)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

    return {"ok": False, "description": last_err}


@actor_bridge(name="TelegramBotTwin")
class TelegramBotTwin:
    """Telegram Bot API Digital Twin exposed at `Actor[#Telegram]`."""

    def __init__(
        self,
        token: Optional[str] = None,
        default_chat_id: Optional[str] = None,
        default_thread_id: Optional[int] = None,
        env_path: Optional[str] = None,
    ) -> None:
        creds = load_telegram_credentials(env_path=env_path)
        self._token = (token if token is not None else creds["token"]).strip()
        self._chat_id = str(default_chat_id if default_chat_id is not None else creds["chat_id"]).strip()
        self._thread_id = default_thread_id if default_thread_id is not None else creds["thread_id"]
        self._allowed_users: List[int] = list(creds["allowed_users"])
        self._bot_info: Dict[str, Any] = {}
        self._sent_log: List[Dict[str, Any]] = []

    @actor_query
    def summary(self) -> str:
        """Return Telegram bot configuration and status summary."""
        masked = "set" if self._token else "unset"
        uname = self._bot_info.get("username", "unconfigured")
        return (
            f"TelegramBotTwin(@{uname} | token={masked} | "
            f"home_chat={self._chat_id or 'none'} | thread={self._thread_id} | "
            f"allowed_users={self._allowed_users})"
        )

    @actor_query
    def botInfo(self) -> str:
        """Query Telegram `getMe` API and return bot identity."""
        res = telegram_api_request(self._token, "getMe", timeout=8.0)
        if res.get("ok") and isinstance(res.get("result"), dict):
            self._bot_info = res["result"]
            r = self._bot_info
            return f"@{r.get('username')} (id={r.get('id')}, name={r.get('first_name')})"
        return f"Telegram getMe error: {res.get('description', 'unknown')}"

    @actor_command
    def sendMessage(self, text: str) -> str:
        """Send a message to the configured default Telegram home channel."""
        return self.sendToChat(self._chat_id, text, 0)

    @actor_command
    def sendToChat(self, chat_id: Any, text: str, thread_id: int = 0) -> str:
        """Send a Telegram message to `chat_id` (optionally in `thread_id` forum topic)."""
        cid = str(chat_id or self._chat_id).strip()
        msg = str(text or "").strip()
        if not cid:
            return "Telegram send error: missing chat_id"
        if not msg:
            return "Telegram send error: empty message"

        payload: Dict[str, Any] = {"chat_id": int(cid) if cid.lstrip("-").isdigit() else cid, "text": msg}
        tid = int(thread_id or 0)
        if tid > 0:
            payload["message_thread_id"] = tid

        res = telegram_api_request(self._token, "sendMessage", payload=payload, timeout=12.0)
        if res.get("ok") and isinstance(res.get("result"), dict):
            msg_id = res["result"].get("message_id")
            self._sent_log.append({"chat_id": cid, "thread_id": tid, "message_id": msg_id, "text": msg})
            return f"Sent Telegram message #{msg_id} to chat {cid}"
        return f"Telegram send failed: {res.get('description', 'unknown')}"


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
        limit = max(500, int(max_chars or 8000))
        payload = json.dumps({"url": url, "formats": ["markdown"]}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._firecrawl_key:
            headers["Authorization"] = f"Bearer {self._firecrawl_key}"

        for endpoint in (f"{api_base}/v2/scrape", f"{api_base}/v1/scrape"):
            try:
                req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=15.0) as resp:
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
        lim = max(1, min(15, int(limit or 5)))
        payload = json.dumps({"query": q, "limit": lim}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._firecrawl_key:
            headers["Authorization"] = f"Bearer {self._firecrawl_key}"

        try:
            req = urllib.request.Request(f"{api_base}/v2/search", data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15.0) as resp:
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
        limit = max(500, int(max_chars or 8000))

        # 1. Try Keenable public fetch
        try:
            import urllib.parse

            k_url = f"https://api.keenable.ai/v1/fetch/public?url={urllib.parse.quote(url, safe='')}"
            req = urllib.request.Request(k_url, headers={"X-Keenable-Title": "hermes-agent", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15.0) as resp:
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
        self._url_cache: Dict[str, str] = {}

    def _resolve_source(self, source: str) -> str:
        import hashlib
        from pathlib import Path

        raw = str(source or "").strip()
        if not raw:
            raise ValueError("Empty PDF path or URL")

        if raw.lower().startswith(("http://", "https://")):
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
                with urllib.request.urlopen(req, timeout=45.0) as resp:
                    data = resp.read(50 * 1024 * 1024)  # 50 MB safety cap
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"Downloaded URL does not appear to be a valid PDF (%PDF header missing): {raw}")
                dest.write_bytes(data)
            self._url_cache[raw] = str(dest)
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
        import pypdf

        local_path = self._resolve_source(url)
        reader = pypdf.PdfReader(local_path)
        size_kb = os.path.getsize(local_path) // 1024
        return f"[PDF Cached] path={local_path} | pages={len(reader.pages)} | size={size_kb} KB"

    @actor_query
    def info(self, path_or_url: str) -> str:
        """Return page count, file size, and PDF metadata for a local path or remote URL."""
        import pypdf

        local_path = self._resolve_source(path_or_url)
        reader = pypdf.PdfReader(local_path)
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
        import pypdf

        local_path = self._resolve_source(path_or_url)
        reader = pypdf.PdfReader(local_path)
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
        import pypdf

        local_path = self._resolve_source(path_or_url)
        reader = pypdf.PdfReader(local_path)
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
        import pypdf

        local_path = self._resolve_source(path_or_url)
        q = str(query or "").strip()
        if not q:
            return "ERROR: Empty search query"
        reader = pypdf.PdfReader(local_path)
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


__all__ = [
    "NetTransportBridge",
    "PDFDocumentTwin",
    "PantryGraphTwin",
    "ReverseEngineeringTwin",
    "ReverseProxyTwin",
    "SmartLightTwin",
    "TelegramBotTwin",
    "ValkyrieSystemTwin",
    "load_telegram_credentials",
    "telegram_api_request",
]
