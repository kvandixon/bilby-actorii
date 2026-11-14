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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            err_msg = f"{type(exc).__name__}: {exc}"
            if token:
                err_msg = err_msg.replace(token, "<REDACTED>")
            last_err = err_msg
            continue

    return {"ok": False, "description": last_err}


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
    import http.client
    import ssl

    if not token:
        return {"ok": False, "description": "Missing TELEGRAM_BOT_TOKEN"}

    path = f"/bot{token}/{method}"
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

    ctx = ssl.create_default_context()
    targets = ["api.telegram.org", *_TELEGRAM_SEED_IPS]
    last_err = "Unknown error"

    for target in targets:
        try:
            raw_sock = socket.create_connection((target, 443), timeout=timeout)
            tls_sock = ctx.wrap_socket(raw_sock, server_hostname="api.telegram.org")
            conn = http.client.HTTPSConnection("api.telegram.org", 443, timeout=timeout, context=ctx)
            conn.sock = tls_sock
            conn.request("POST", path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw_data = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return json.loads(raw_data)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            if token:
                err_msg = err_msg.replace(token, "<REDACTED>")
            last_err = err_msg
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
        self._sent_log: List[Dict[str, Any]] = []
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


JETSON_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "do-release-upgrade",
    "apt upgrade",
    "apt-get upgrade",
    "apt full-upgrade",
    "apt dist-upgrade",
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
            found_ip, b3 = self._scan_for_jetson_ip(port=port)
            if found_ip:
                cfg["host"] = found_ip
                self._save_cfg(cfg)
                return found_ip, cfg, b3

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

        for ip, banner in hits:
            if "Ubuntu" in banner or "OpenSSH_7." in banner or "OpenSSH_8." in banner:
                return ip, banner
        return (hits[0][0], hits[0][1]) if len(hits) == 1 and hits[0][0] != "192.168.68.56" else (None, None)

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
    def _validate_jetson_command(cmd: str) -> Optional[str]:
        low = " ".join((cmd or "").lower().split())
        if not low:
            return "Empty command."
        for pat in JETSON_FORBIDDEN_PATTERNS:
            if pat in low:
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
        violation = self._validate_jetson_command(body)
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

        low_r = r_path.lower()
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
            if os.path.islink(dest_path):
                dest_path.unlink()
            part_path = stage_dir / f".{safe_base}.{os.getpid()}.part"
            fd = os.open(str(part_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, raw_bytes)
            finally:
                os.close(fd)
            os.replace(str(part_path), str(dest_path))
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


__all__ = [
    "JetsonEdgeTwin",
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
    "telegram_multipart_request",
]
