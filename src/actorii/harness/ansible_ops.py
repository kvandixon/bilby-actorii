"""Guarded Ansible Host Abstraction Bridge (`Actor[#Ops]`, `actorii.harness.ansible_ops`).

Applies ActorII's Ring-0 safety principles to multi-host operations across `valkyrie`,
`jetson` (with graceful offline degradation), and future OS targets:

1. **Fast Read-Only Path (`@actor_query`, <300ms, zero Ansible fork overhead):**
   - `listHosts()` — configured inventory hosts, roles, and criticality.
   - `hostFacts(host)` — cached OS/CPU/RAM/kernel facts from `/srv/ansible/facts/<host>.json`.
   - `readLogs(host, unit, lines, since, userScope)` — bounded, secret-redacted `journalctl`
     reader (`lines <= 120`) supporting both user and system units on `valkyrie` and `jetson`.
   - `serviceStatus(host, service, userScope)` — fast `systemctl` status probe.
   - `httpCheck(url, timeoutSec)` — fast read-only local/LAN HTTP GET healthcheck (so Bilby
     never needs the Jetson online just to verify a web server on `valkyrie`).
   - `listPlaybooks()` — curated, pre-audited playbooks under `/srv/ansible/playbooks/bilby/`.

2. **Controlled Mutation Path (`@actor_command`, mandatory `--check --diff` dry-run):**
   - `planPlaybook(playbook, targetHost, varsJson)` — validates playbook + variables against
     the protected OS daemon denylist, runs `ansible-playbook --check --diff`, and issues a
     short-lived `planToken`.
   - `applyPlaybook(playbook, targetHost, varsJson, planToken)` — requires a matching recent
     `planToken` (enforcing dry-run-before-apply), re-validates guardrails, executes the
     curated playbook non-interactively, and appends an audit record to `~/.actorii/ops_audit.jsonl`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol

DEFAULT_ANSIBLE_DIR = Path(os.environ.get("ACTORII_ANSIBLE_DIR", "/srv/ansible"))
DEFAULT_ANSIBLE_CFG = DEFAULT_ANSIBLE_DIR / "ansible-bilby.cfg"
DEFAULT_PLAYBOOK_DIR = DEFAULT_ANSIBLE_DIR / "playbooks" / "bilby"
DEFAULT_FACTS_DIR = DEFAULT_ANSIBLE_DIR / "facts"
DEFAULT_AUDIT_LOG = Path.home() / ".actorii" / "ops_audit.jsonl"

# Hard-protected OS/network/desktop/self daemons (codifying the 2026-10-18 incident guardrails)
PROTECTED_OS_SERVICES = frozenset({
    "networkmanager",
    "systemd-resolved",
    "tailscaled",
    "gdm",
    "gdm3",
    "firewalld",
    "crond",
    "cron",
    "chronyd",
    "containerd",
    "docker",
    "polkit",
    "sshd",
    "ssh",
    "systemd-journald",
    "systemd-logind",
    "systemd-udevd",
    "dbus",
    "bilby-actor-telegram",
    "llama-primary",
})

# Curated Tier-1 playbooks Bilby is permitted to plan/apply via `Actor[#Ops]`
CURATED_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "backup_jetson.yml": {
        "tier": 1,
        "description": "Pull critical Jetson TX1 configs, ActorII edge scripts, boot failsafe, and bilby-edge workspace to /home/skippy/backups/jetson/ (gracefully skips if Jetson is offline).",
        "allowed_hosts": ("valkyrie",),
        "allowed_vars": ("keep_snapshots",),
    },
    "gather_facts.yml": {
        "tier": 1,
        "description": "Refresh cached OS/hardware/RAM facts in /srv/ansible/facts/<host>.json across online hosts (gracefully skips offline edge nodes).",
        "allowed_hosts": ("valkyrie", "jetson", "workstations:edge_nodes"),
        "allowed_vars": ("target_hosts",),
    },
    "restart_user_service.yml": {
        "tier": 1,
        "description": "Restart an explicitly allowlisted user-scope service on valkyrie (bilby-coder-8083.service, clef-flash-slot-8082.service, honeypot-dashboard.service).",
        "allowed_hosts": ("valkyrie",),
        "allowed_vars": ("service_name",),
        "allowed_services": (
            "bilby-coder-8083.service",
            "clef-flash-slot-8082.service",
            "honeypot-dashboard.service",
        ),
    },
    "cleanup_tmp_workspace.yml": {
        "tier": 1,
        "description": "Remove stale scratch files older than max_age_days in approved temp dirs (/tmp/actorii_media or /home/bilby-edge/workspace/tmp).",
        "allowed_hosts": ("valkyrie", "jetson"),
        "allowed_vars": ("target_hosts", "max_age_days"),
    },
}

_SECRET_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(token|apikey|api_key|secret|password|passwd|authorization)\s*[:=]\s*[^\s,'\";]+"), r"\1=[REDACTED]"),
    (re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"), "[TELEGRAM_TOKEN_REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[API_KEY_REDACTED]"),
)

_SAFE_UNIT_RE = re.compile(r"^[A-Za-z0-9@._:-]{1,96}$")
_SAFE_SINCE_RE = re.compile(r"^[A-Za-z0-9 :._+-]{1,48}$")


def _to_actor_dict(d: Dict[str, Any]) -> Dictionary:
    out = Dictionary()
    for k, v in d.items():
        if isinstance(v, dict):
            out.atPut(Symbol(str(k)), _to_actor_dict(v))
        else:
            out.atPut(Symbol(str(k)), v)
    return out


def _scrub_secrets(text: str) -> str:
    s = str(text or "")
    for pat, repl in _SECRET_PATTERNS:
        s = pat.sub(repl, s)
    return s


def _norm_service(name: str) -> str:
    s = str(name or "").strip().lower()
    if s.endswith(".service"):
        s = s[:-8]
    return s


def _is_protected_service(name: str) -> bool:
    norm = _norm_service(name)
    if not norm:
        return True
    if norm in PROTECTED_OS_SERVICES:
        return True
    if norm.startswith("systemd-"):
        return True
    return False


@actor_bridge(name="AnsibleOpsBridge")
class AnsibleOpsBridge:
    """OS-agnostic, Ring-0 guarded host operations & Ansible bridge (`Actor[#Ops]`)."""

    def __init__(
        self,
        ansible_dir: Optional[Path] = None,
        audit_log: Optional[Path] = None,
        plan_ttl_s: float = 300.0,
    ) -> None:
        self.ansible_dir = Path(ansible_dir) if ansible_dir else DEFAULT_ANSIBLE_DIR
        self.ansible_cfg = self.ansible_dir / "ansible-bilby.cfg"
        self.playbook_dir = self.ansible_dir / "playbooks" / "bilby"
        self.facts_dir = self.ansible_dir / "facts"
        self.audit_log = Path(audit_log) if audit_log else DEFAULT_AUDIT_LOG
        self.plan_ttl_s = max(30.0, float(plan_ttl_s or 300.0))
        # plan_token -> (expires_monotonic, playbook, target_host, canonical_vars_json)
        self._plans: Dict[str, Tuple[float, str, str, str]] = {}

    def _jetson_ssh_target(self) -> Tuple[str, str, str]:
        host_json = Path.home() / ".actorii" / "jetson" / "host.json"
        host = "192.168.68.57"
        user = "bilby-edge"
        key = str(Path.home() / ".actorii" / "jetson" / "id_ed25519")
        try:
            if host_json.is_file():
                cfg = json.loads(host_json.read_text(encoding="utf-8"))
                host = str(cfg.get("host") or host)
                user = str(cfg.get("user") or user)
        except Exception:
            pass
        return host, user, key

    def _run_host_argv(self, host: str, argv: List[str], timeout_s: float = 8.0) -> Dict[str, Any]:
        norm_h = str(host or "valkyrie").strip().lower()
        if norm_h in ("valkyrie", "localhost", "local"):
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
                return {
                    "ok": proc.returncode == 0,
                    "host": "valkyrie",
                    "offline": False,
                    "exitCode": proc.returncode,
                    "stdout": _scrub_secrets(proc.stdout or ""),
                    "stderr": _scrub_secrets(proc.stderr or ""),
                }
            except Exception as exc:
                return {"ok": False, "host": "valkyrie", "offline": False, "exitCode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}

        if norm_h == "jetson":
            j_host, j_user, j_key = self._jetson_ssh_target()
            import shlex
            remote_cmd = " ".join(shlex.quote(a) for a in argv)
            ssh_argv = [
                shutil.which("ssh") or "/usr/bin/ssh",
                "-F", "/dev/null",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=3",
                "-o", "LogLevel=ERROR",
                "-o", "StrictHostKeyChecking=accept-new",
                "-i", j_key,
                "-o", "IdentitiesOnly=yes",
                f"{j_user}@{j_host}",
                remote_cmd,
            ]
            try:
                proc = subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout_s)
                offline = proc.returncode == 255
                return {
                    "ok": proc.returncode == 0,
                    "host": "jetson",
                    "offline": offline,
                    "exitCode": proc.returncode,
                    "stdout": _scrub_secrets(proc.stdout or ""),
                    "stderr": _scrub_secrets(proc.stderr or ""),
                }
            except subprocess.TimeoutExpired:
                return {"ok": False, "host": "jetson", "offline": True, "exitCode": -1, "stdout": "", "stderr": "Jetson offline or SSH timed out (degraded gracefully)."}
            except Exception as exc:
                return {"ok": False, "host": "jetson", "offline": True, "exitCode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}

        return {"ok": False, "host": norm_h, "offline": False, "exitCode": -1, "stdout": "", "stderr": f"Unknown host {host!r}; allowed: valkyrie, jetson."}

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        try:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "action": action, **details}
            with self.audit_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # =========================================================================
    # Tier 0: Fast Read-Only Host Operations (`@actor_query`)
    # =========================================================================

    @actor_query
    def listHosts(self) -> Any:
        """List managed hosts in Bilby's Ansible inventory along with roles and criticality."""
        j_host, j_user, _ = self._jetson_ssh_target()
        return _to_actor_dict({
            "ok": True,
            "hosts": {
                "valkyrie": {"connection": "local", "role": "primary_workstation", "critical": True},
                "jetson": {"address": j_host, "user": j_user, "role": "edge_sandbox", "critical": False},
            },
        })

    @actor_query
    def hostFacts(self, host: str = "valkyrie") -> Any:
        """Return cached OS/hardware/RAM facts for `host` from `/srv/ansible/facts/<host>.json`."""
        norm_h = str(host or "valkyrie").strip().lower()
        if norm_h not in ("valkyrie", "jetson"):
            return _to_actor_dict({"ok": False, "error": f"Unknown host {host!r}; allowed: valkyrie, jetson."})
        fact_file = self.facts_dir / f"{norm_h}.json"
        if not fact_file.is_file():
            return _to_actor_dict({
                "ok": False,
                "host": norm_h,
                "error": f"No cached facts at {fact_file}. Run `planPlaybook`/`applyPlaybook` with `gather_facts.yml`.",
            })
        try:
            data = json.loads(fact_file.read_text(encoding="utf-8"))
            return _to_actor_dict({"ok": True, "host": norm_h, "facts": data})
        except Exception as exc:
            return _to_actor_dict({"ok": False, "host": norm_h, "error": f"{type(exc).__name__}: {exc}"})

    @actor_query
    def readLogs(
        self,
        host: str = "valkyrie",
        unit: str = "",
        lines: int = 40,
        since: str = "1h",
        user_scope: bool = True,
    ) -> Any:
        """Fast, bounded, secret-redacted `journalctl` log reader (`lines <= 120`) for troubleshooting."""
        unit_s = str(unit or "").strip()
        if unit_s and not _SAFE_UNIT_RE.match(unit_s):
            return _to_actor_dict({"ok": False, "error": f"Invalid unit name {unit_s!r}."})
        since_s = str(since or "1h").strip()
        if since_s and not _SAFE_SINCE_RE.match(since_s):
            return _to_actor_dict({"ok": False, "error": f"Invalid since window {since_s!r}."})
        n_lines = max(1, min(120, int(lines or 40)))

        argv = ["journalctl", "--no-pager", "-n", str(n_lines)]
        if bool(user_scope) and str(host or "valkyrie").strip().lower() in ("valkyrie", "localhost", "local"):
            argv.append("--user")
        if unit_s:
            argv.extend(["-u", unit_s])
        if since_s:
            argv.extend(["--since", f"-{since_s}" if re.match(r"^\d+[smhd]$", since_s) else since_s])

        res = self._run_host_argv(host, argv, timeout_s=8.0)
        out_lines = (res.get("stdout") or "").splitlines()
        clipped = "\n".join(line[:240] for line in out_lines[-n_lines:])
        return _to_actor_dict({
            "ok": bool(res["ok"]),
            "host": res["host"],
            "offline": bool(res.get("offline", False)),
            "unit": unit_s or "(all)",
            "linesReturned": len(out_lines[-n_lines:]),
            "logs": clipped,
            "stderr": res.get("stderr", "")[:400],
        })

    @actor_query
    def serviceStatus(self, host: str = "valkyrie", service: str = "", user_scope: bool = True) -> Any:
        """Fast read-only systemd service status check on `valkyrie` or `jetson`."""
        svc = str(service or "").strip()
        if not svc or not _SAFE_UNIT_RE.match(svc):
            return _to_actor_dict({"ok": False, "error": f"Invalid service name {svc!r}."})
        argv = ["systemctl"]
        if bool(user_scope) and str(host or "valkyrie").strip().lower() in ("valkyrie", "localhost", "local"):
            argv.append("--user")
        argv.extend(["show", svc, "--property=ActiveState,SubState,MainPID,MemoryCurrent,Description", "--no-pager"])
        res = self._run_host_argv(host, argv, timeout_s=6.0)
        props: Dict[str, str] = {}
        for line in (res.get("stdout") or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
        return _to_actor_dict({
            "ok": bool(res["ok"]) and bool(props),
            "host": res["host"],
            "offline": bool(res.get("offline", False)),
            "service": svc,
            "activeState": props.get("ActiveState", "unknown"),
            "subState": props.get("SubState", "unknown"),
            "mainPid": props.get("MainPID", "0"),
            "description": props.get("Description", ""),
            "stderr": res.get("stderr", "")[:300],
        })

    @actor_query
    def httpCheck(self, url: str, timeout_sec: int = 5) -> Any:
        """Fast read-only HTTP GET healthcheck for loopback/LAN URLs (replaces needing Jetson to curl valkyrie)."""
        raw_url = str(url or "").strip()
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return _to_actor_dict({"ok": False, "url": raw_url, "error": "Only http:// or https:// URLs are allowed."})
        host = parsed.hostname.lower()
        allowed_net = (
            host in ("127.0.0.1", "localhost", "::1")
            or host.startswith("192.168.")
            or host.startswith("10.")
            or host.startswith("100.")
            or host.endswith(".local")
        )
        if not allowed_net:
            return _to_actor_dict({"ok": False, "url": raw_url, "error": f"httpCheck is scoped to localhost/LAN/Tailnet hosts (blocked {host!r})."})
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(raw_url, method="GET", headers={"User-Agent": "Bilby-ActorOps/1.0"})
            with urllib.request.urlopen(req, timeout=max(1.0, min(15.0, float(timeout_sec or 5)))) as resp:
                body = resp.read(8192).decode("utf-8", errors="replace")
                return _to_actor_dict({
                    "ok": True,
                    "url": raw_url,
                    "status": int(resp.status),
                    "bytesSampled": len(body),
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1),
                    "snippet": _scrub_secrets(body[:600]),
                })
        except urllib.error.HTTPError as exc:
            return _to_actor_dict({
                "ok": False,
                "url": raw_url,
                "status": int(exc.code),
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1),
                "error": f"HTTP {exc.code}: {exc.reason}",
            })
        except Exception as exc:
            return _to_actor_dict({
                "ok": False,
                "url": raw_url,
                "status": 0,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })

    @actor_query
    def listPlaybooks(self) -> Any:
        """List curated, pre-audited Ansible playbooks available to Bilby under `/srv/ansible/playbooks/bilby/`."""
        items: Dict[str, Any] = {}
        for name, meta in CURATED_PLAYBOOKS.items():
            pb_path = self.playbook_dir / name
            items[name] = {
                "exists": pb_path.is_file(),
                "tier": meta["tier"],
                "allowedHosts": list(meta["allowed_hosts"]),
                "allowedVars": list(meta["allowed_vars"]),
                "description": meta["description"],
            }
        return _to_actor_dict({"ok": True, "playbookDir": str(self.playbook_dir), "playbooks": items})

    # =========================================================================
    # Tier 1: Dry-Run-Gated Curated Ansible Playbooks (`@actor_command`)
    # =========================================================================

    def _validate_playbook_request(
        self,
        playbook: str,
        target_host: str,
        vars_json: str,
    ) -> Tuple[Optional[str], Optional[Path], str, Dict[str, Any]]:
        pb_name = os.path.basename(str(playbook or "").strip())
        if pb_name not in CURATED_PLAYBOOKS:
            return (
                f"OPS_GUARDRAIL_BLOCKED: Playbook {playbook!r} is not in curated catalog {sorted(CURATED_PLAYBOOKS.keys())}.",
                None,
                "",
                {},
            )
        meta = CURATED_PLAYBOOKS[pb_name]
        pb_path = (self.playbook_dir / pb_name).resolve()
        if not pb_path.is_file() or pb_path.parent != self.playbook_dir.resolve():
            return f"OPS_ERROR: Playbook file not found at {pb_path}.", None, "", {}

        host_s = str(target_host or meta["allowed_hosts"][0]).strip()
        if host_s not in meta["allowed_hosts"]:
            return (
                f"OPS_GUARDRAIL_BLOCKED: Host {host_s!r} not allowed for {pb_name} (allowed: {list(meta['allowed_hosts'])}).",
                None,
                "",
                {},
            )

        raw_v = str(vars_json or "{}").strip() or "{}"
        try:
            parsed_vars = json.loads(raw_v)
            if not isinstance(parsed_vars, dict):
                return "OPS_ERROR: varsJson must be a JSON object.", None, "", {}
        except Exception as exc:
            return f"OPS_ERROR: Invalid varsJson: {exc}", None, "", {}

        allowed_v = set(meta["allowed_vars"])
        for k in parsed_vars.keys():
            if k not in allowed_v:
                return f"OPS_GUARDRAIL_BLOCKED: Variable {k!r} not permitted for {pb_name} (allowed: {sorted(allowed_v)}).", None, "", {}

        if pb_name == "restart_user_service.yml":
            svc = str(parsed_vars.get("service_name") or "").strip()
            if _is_protected_service(svc) or svc not in meta.get("allowed_services", ()):
                return (
                    f"OPS_GUARDRAIL_BLOCKED: Service {svc!r} is protected or not in allowlist {list(meta.get('allowed_services', ()))}.",
                    None,
                    "",
                    {},
                )

        if "max_age_days" in parsed_vars:
            days = int(parsed_vars["max_age_days"])
            if days < 1 or days > 365:
                return "OPS_GUARDRAIL_BLOCKED: max_age_days must be between 1 and 365.", None, "", {}
            parsed_vars["max_age_days"] = days

        if "keep_snapshots" in parsed_vars:
            ks = int(parsed_vars["keep_snapshots"])
            if ks < 2 or ks > 30:
                return "OPS_GUARDRAIL_BLOCKED: keep_snapshots must be between 2 and 30.", None, "", {}
            parsed_vars["keep_snapshots"] = ks

        return None, pb_path, host_s, parsed_vars

    def _exec_ansible_playbook(
        self,
        pb_path: Path,
        target_host: str,
        extra_vars: Dict[str, Any],
        *,
        check_mode: bool,
        timeout_s: float = 90.0,
    ) -> Dict[str, Any]:
        ab_bin = shutil.which("ansible-playbook") or "/usr/bin/ansible-playbook"
        if not os.path.isfile(ab_bin):
            return {"ok": False, "exitCode": -1, "stdout": "", "stderr": "ansible-playbook binary not found."}

        argv = [ab_bin, str(pb_path), "--limit", target_host]
        if check_mode:
            argv.extend(["--check", "--diff"])
        if extra_vars:
            argv.extend(["-e", json.dumps(extra_vars, sort_keys=True)])

        env = dict(os.environ)
        env["ANSIBLE_CONFIG"] = str(self.ansible_cfg)
        env["ANSIBLE_ASK_BECOME_PASS"] = "false"
        env["ANSIBLE_NOCOLOR"] = "1"

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout_s)
            return {
                "ok": proc.returncode == 0,
                "exitCode": proc.returncode,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1),
                "stdout": _scrub_secrets(proc.stdout or "")[-3000:],
                "stderr": _scrub_secrets(proc.stderr or "")[-1000:],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "exitCode": -1, "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1), "stdout": "", "stderr": f"ansible-playbook timed out after {timeout_s}s."}
        except Exception as exc:
            return {"ok": False, "exitCode": -1, "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 1), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}

    @actor_command
    def planPlaybook(self, playbook: str, target_host: str = "valkyrie", vars_json: str = "{}") -> Any:
        """Run a mandatory `--check --diff` dry-run for a curated playbook and return a short-lived `planToken`."""
        err, pb_path, host_s, extra_vars = self._validate_playbook_request(playbook, target_host, vars_json)
        if err or pb_path is None:
            return _to_actor_dict({"ok": False, "error": err or "Validation failed."})

        res = self._exec_ansible_playbook(pb_path, host_s, extra_vars, check_mode=True)
        canon_vars = json.dumps(extra_vars, sort_keys=True)
        token = ""
        if res["ok"]:
            raw_sig = f"{pb_path.name}|{host_s}|{canon_vars}|{time.time():.3f}"
            token = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]
            self._plans[token] = (time.monotonic() + self.plan_ttl_s, pb_path.name, host_s, canon_vars)

        self._audit("planPlaybook", {"playbook": pb_path.name, "host": host_s, "vars": extra_vars, "ok": res["ok"], "token": token})
        return _to_actor_dict({
            "ok": bool(res["ok"]),
            "mode": "check_diff",
            "playbook": pb_path.name,
            "targetHost": host_s,
            "planToken": token,
            "expiresInSec": int(self.plan_ttl_s) if token else 0,
            "elapsedMs": res.get("elapsedMs", 0.0),
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
        })

    @actor_command
    def applyPlaybook(
        self,
        playbook: str,
        target_host: str = "valkyrie",
        vars_json: str = "{}",
        plan_token: str = "",
    ) -> Any:
        """Apply a curated playbook ONLY after a matching `planPlaybook` (`--check --diff`) dry-run token."""
        err, pb_path, host_s, extra_vars = self._validate_playbook_request(playbook, target_host, vars_json)
        if err or pb_path is None:
            return _to_actor_dict({"ok": False, "error": err or "Validation failed."})

        tok = str(plan_token or "").strip()
        canon_vars = json.dumps(extra_vars, sort_keys=True)
        rec = self._plans.get(tok)
        if not tok or rec is None:
            return _to_actor_dict({
                "ok": False,
                "error": "OPS_GUARDRAIL_BLOCKED: Missing or unknown planToken. Run `planPlaybook(Actor[#Ops], playbook, host, varsJson)!!` first to review `--check --diff`.",
            })
        exp_ts, p_pb, p_host, p_vars = rec
        if time.monotonic() > exp_ts:
            self._plans.pop(tok, None)
            return _to_actor_dict({"ok": False, "error": "OPS_GUARDRAIL_BLOCKED: planToken expired. Re-run `planPlaybook` first."})
        if (p_pb, p_host, p_vars) != (pb_path.name, host_s, canon_vars):
            return _to_actor_dict({"ok": False, "error": "OPS_GUARDRAIL_BLOCKED: planToken does not match playbook/host/vars."})

        self._plans.pop(tok, None)
        res = self._exec_ansible_playbook(pb_path, host_s, extra_vars, check_mode=False)
        self._audit("applyPlaybook", {"playbook": pb_path.name, "host": host_s, "vars": extra_vars, "ok": res["ok"], "exitCode": res.get("exitCode")})
        return _to_actor_dict({
            "ok": bool(res["ok"]),
            "mode": "apply",
            "playbook": pb_path.name,
            "targetHost": host_s,
            "exitCode": res.get("exitCode", -1),
            "elapsedMs": res.get("elapsedMs", 0.0),
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
        })

    @actor_query
    def describe(self) -> str:
        return (
            "Actor[#Ops] (AnsibleOpsBridge — OS-agnostic guarded host ops): "
            "Tier-0 fast queries: listHosts, hostFacts(host), readLogs(host, unit, lines, since, userScope), "
            "serviceStatus(host, service, userScope), httpCheck(url, timeoutSec), listPlaybooks; "
            "Tier-1 curated playbooks (mandatory --check --diff -> apply): "
            "planPlaybook(playbook, targetHost, varsJson), applyPlaybook(playbook, targetHost, varsJson, planToken)."
        )
