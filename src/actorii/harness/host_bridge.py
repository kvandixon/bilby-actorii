"""Minimal Transactional Host Workspace Bridge (`Actor[#Host]`) & VM Reflection Bridge (`Actor[#VM]`).

Implements Phase 1 and Phase 2 of the Pi-Class Harness Evolution Specification:
1. `HostWorkspaceBridge` (`Actor[#Host]`):
   - `readFile(Actor[#Host], path, startLine, endLine)` (`@actor_query`, Tier 0):
     Reads bounded file slices with line numbers; reads from the Copy-on-Write VFS
     overlay first when executing inside a `shadow_only=True` transaction.
   - `writeFile(Actor[#Host], path, content)` (`@actor_command`, Tier 2, shadow-safe):
     Writes to the in-memory CoW VFS overlay during `shadow_only=True` speculative
     runs; writes atomically to disk within allowed workspace roots during committed runs.
   - `editFile(Actor[#Host], path, oldText, newText)` (`@actor_command`, Tier 2, shadow-safe):
     Surgical exact-match replacement returning a unified diff preview; staged in the
     in-memory CoW VFS overlay during `shadow_only=True`.
   - `execShell(Actor[#Host], command, timeoutSec)` (`@actor_command`, Tier 2):
     Executes a shell command inside the workspace with a strict timeout and
     head/tail output truncation (`4,000` chars max) so verbose logs never overflow
     the 16 GB KV-cache. Blocked during `shadow_only=True`.
2. `VMReflectionBridge` (`Actor[#VM]`):
   - `classes(Actor[#VM])`: Returns Actor `Array` of `Dictionary` entries (`#name`, `#ring`, `#superclass`).
   - `ringOf(Actor[#VM], classSymbol)`: Returns Ring tier (`0`, `1`, or `2`).
   - `methodsOf(Actor[#VM], classSymbol)`: Returns Actor `Array` of method selector strings.
   - `sourceOf(Actor[#VM], classSymbol, selectorSymbol)`: Returns `.act` source or bytecode disassembly.
   - `ivarsOf(Actor[#VM], classSymbol)`: Returns Actor `Array` of declared instance variable names.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorClass, Symbol
from actorii.guardrails.linter import PROTECTED_SYSTEM_CLASSES


def _norm_sym(val: Any) -> str:
    if isinstance(val, Symbol):
        return val.name
    s = str(val or "").strip()
    return s[1:] if s.startswith("#") else s


RING_1_CLASSES: Set[str] = {
    "AgentHarness",
    "VMSandbox",
    "LLMBridge",
    "HostWorkspaceBridge",
    "VMReflectionBridge",
    "ActorGrammarGraph",
    "ActorMemoryGraph",
    "ActorBenchSuite",
    "ValkyrieSystemTwin",
    "SmartLightTwin",
    "PantryGraphTwin",
    "ThermoProTwin",
    "ThermometerUI",
    "ReverseEngineeringTwin",
    "ReverseProxyTwin",
    "JetsonEdgeTwin",
    "TelegramBotTwin",
    "NetTransportBridge",
    "PDFDocumentTwin",
    "WebScraper",
}


def classify_class_ring(cls_name: str, cls_obj: Optional[ActorClass] = None) -> int:
    """Return Ring 0 (Protected Kernel), Ring 1 (Harness/Bridge), or Ring 2 (User/Engram)."""
    norm = _norm_sym(cls_name)
    if norm == "AgentHarness":
        return 1
    if norm in PROTECTED_SYSTEM_CLASSES:
        return 0
    if norm in RING_1_CLASSES or (cls_obj is not None and getattr(cls_obj, "is_bridge_class", False)):
        return 1
    return 2


FORBIDDEN_SHELL_PATTERNS: Tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /home",
    "rm -rf *",
    "sudo ",
    "sudo\t",
    "su -",
    "su root",
    "pkexec",
    "systemctl disable",
    "systemctl stop",
    "systemctl mask",
    "systemctl kill",
    "systemctl restart",
    "systemctl reload",
    "systemctl daemon-reload",
    "systemctl reset-failed",
    "systemctl edit",
    "systemctl --user",
    "systemctl ",
    "service ",
    "nmcli",
    "ifconfig",
    "ip link",
    "ip addr",
    "ip route",
    "resolv.conf",
    "tailscale ",
    "tailscaled",
    "firewall-cmd",
    "firewalld",
    "iptables",
    "nft ",
    "ufw ",
    "crontab",
    "/etc/cron",
    "mkfs",
    "dd if=",
    "reboot",
    "shutdown",
    "poweroff",
    "init 0",
    "init 6",
    "telinit",
    "kill -9",
    "killall",
    "pkill",
    "chmod 777",
    "chown ",
    f"{Path.home().resolve()}/bin/",
    ".config/systemd",
)

_HOME_DIR = str(Path.home().resolve())
_REPO_ROOT = str(Path(__file__).resolve().parents[3])

FORBIDDEN_READ_PREFIXES: Tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "/etc/shadow",
            "/etc/sudoers",
            "/etc/gshadow",
            "/root",
            "/proc/self/environ",
            f"{_HOME_DIR}/.ssh",
            f"{_HOME_DIR}/.gnupg",
            f"{_HOME_DIR}/.hermes",
            f"{_HOME_DIR}/.actorii",
            f"{_REPO_ROOT}/.env",
        )
    )
)

FORBIDDEN_WRITE_PREFIXES: Tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "/etc",
            "/boot",
            "/usr",
            "/var",
            "/root",
            "/proc",
            "/sys",
            "/dev",
            f"{_HOME_DIR}/.ssh",
            f"{_HOME_DIR}/.gnupg",
            f"{_HOME_DIR}/.config",
            f"{_HOME_DIR}/.hermes",
            f"{_HOME_DIR}/.gemini",
            f"{_HOME_DIR}/.agents",
            f"{_HOME_DIR}/.actorii",
            f"{_HOME_DIR}/bin",
            f"{_HOME_DIR}/scripts",
            f"{_REPO_ROOT}/src",
            f"{_REPO_ROOT}/actor_telegram.py",
            f"{_REPO_ROOT}/docker",
        )
    )
)


def is_host_lockdown() -> bool:
    """Return True when running in production host-lockdown mode (`ACTORII_HOST_LOCKDOWN=1`)."""
    return os.environ.get("ACTORII_HOST_LOCKDOWN") == "1"


def validate_safe_shell_command(cmd_str: str) -> None:
    """Validate that `cmd_str` contains no forbidden host/service/network/cron patterns."""
    lower_cmd = cmd_str.lower()
    for banned in FORBIDDEN_SHELL_PATTERNS:
        if banned.lower() in lower_cmd:
            raise PermissionError(f"Blocked unsafe shell command pattern: {banned!r}")


@actor_bridge(name="HostWorkspaceBridge")
class HostWorkspaceBridge:
    """Transactional Host Workspace I/O & Shell Bridge exposed as `Actor[#Host]`."""

    def __init__(
        self,
        vm: Any,
        allowed_roots: Optional[List[str | Path]] = None,
        max_output_chars: int = 4000,
    ) -> None:
        self.vm = vm
        if os.environ.get("ACTORII_BENCH_STRICT") == "1":
            default_roots = [
                Path("/workspace").resolve() if Path("/workspace").exists() else Path("/workspace"),
                Path("/tmp").resolve(),
            ]
        elif is_host_lockdown():
            default_roots = [
                Path("/workspace").resolve() if Path("/workspace").exists() else Path("/workspace"),
                Path("/tmp").resolve(),
            ]
        else:
            default_roots = [
                Path.cwd().resolve(),
                Path("/workspace").resolve() if Path("/workspace").exists() else Path("/workspace"),
                Path(_REPO_ROOT).resolve(),
                Path("/app").resolve() if Path("/app").exists() else Path("/app"),
                Path("/tmp").resolve(),
            ]
        self.allowed_roots: List[Path] = [
            Path(r).expanduser().resolve() for r in (allowed_roots or default_roots)
        ]
        self.max_output_chars = int(max_output_chars)
        # In-memory Copy-on-Write VFS overlay used during `vm._in_shadow_run` (`shadow_only=True`)
        self._shadow_vfs: Dict[str, str] = {}
        self._last_file_backup: Dict[str, str] = {}
        self._best_pass_count: Dict[str, int] = {}

    def clear_shadow_vfs(self) -> None:
        """Discard all uncommitted Copy-on-Write VFS overlay buffers."""
        self._shadow_vfs.clear()

    def _resolve_safe_path(
        self,
        raw_path: str,
        allow_read_anywhere_in_project: bool = True,
        for_write: bool = False,
    ) -> Path:
        p = Path(str(raw_path or "").strip()).expanduser()
        if not p.is_absolute():
            p = (self.allowed_roots[0] / p).resolve()
        else:
            p = p.resolve()

        # Always forbid sensitive system/credential paths
        p_str = str(p)
        for fp in FORBIDDEN_READ_PREFIXES:
            if p_str == fp or p_str.startswith(fp + "/"):
                raise PermissionError(f"Access to restricted path '{p_str}' is forbidden by HostWorkspaceBridge.")

        if for_write and os.environ.get("ACTORII_ALLOW_SELF_EDIT") != "1":
            for wp in FORBIDDEN_WRITE_PREFIXES:
                if p_str == wp or p_str.startswith(wp + "/"):
                    raise PermissionError(
                        f"BLOCKED: Write/edit access to protected host/source path '{p_str}' is strictly forbidden by HostWorkspaceBridge."
                    )
            if is_host_lockdown():
                tmp_root = Path("/tmp").resolve()
                ws_root = Path("/workspace").resolve() if Path("/workspace").exists() else Path("/workspace")
                if not (
                    p == tmp_root
                    or tmp_root in p.parents
                    or p == ws_root
                    or ws_root in p.parents
                ):
                    raise PermissionError(
                        f"BLOCKED: Host filesystem mutation of '{p_str}' is disabled in production host-lockdown mode (`ACTORII_HOST_LOCKDOWN=1`)."
                    )

        if not any(p == root or root in p.parents for root in self.allowed_roots):
            if not allow_read_anywhere_in_project or for_write:
                roots_str = ", ".join(str(r) for r in self.allowed_roots)
                raise PermissionError(
                    f"Path '{p_str}' is outside allowed HostWorkspaceBridge roots ({roots_str})."
                )
        return p

    def _read_raw_text(self, target: Path) -> str:
        key = str(target)
        if getattr(self.vm, "_in_shadow_run", False) and key in self._shadow_vfs:
            return self._shadow_vfs[key]
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        return target.read_text(encoding="utf-8", errors="replace")

    @actor_query
    def readFile(self, path: str, start_line: int = 1, end_line: int = 120) -> str:
        """Read a bounded slice of `path` (`start_line..end_line`, 1-based inclusive) with line numbers."""
        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False)
        content = self._read_raw_text(target)
        lines = content.splitlines()
        total = len(lines)
        s_idx = max(1, int(start_line or 1))
        e_idx = min(total, int(end_line or max(s_idx, 120)))
        if total == 0:
            return f"[{target.name}: empty file]"
        sliced = [f"{ln_no:4d}: {lines[ln_no - 1]}" for ln_no in range(s_idx, e_idx + 1)]
        body = "\n".join(sliced)
        if len(body) > self.max_output_chars:
            body = body[: self.max_output_chars] + "\n... [truncated]"
        vfs_tag = " (CoW Shadow VFS)" if (getattr(self.vm, "_in_shadow_run", False) and str(target) in self._shadow_vfs) else ""
        return f"[{target}{vfs_tag} lines {s_idx}-{e_idx} of {total}]\n{body}"

    @actor_query
    def readSymbol(self, path: str, symbol_name: str) -> str:
        """Inspect a specific AST symbol (function/class) in `path` and return its source lines."""
        import ast as _ast

        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False)
        content = self._read_raw_text(target)
        lines = content.splitlines()
        total = len(lines)
        sym = str(symbol_name or "").strip()
        if not sym:
            return f"[{target.name}: empty symbol name]"
        try:
            tree = _ast.parse(content)
        except SyntaxError as exc:
            return f"[{target.name}: syntax error {exc}]"
        found = None
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)) and node.name == sym:
                found = node
                break
        if found is None:
            return f"[{target.name}: symbol {sym!r} not found]"
        s_idx = int(found.lineno)
        e_idx = int(getattr(found, "end_lineno", s_idx) or s_idx)
        sliced = [f"{ln_no:4d}: {lines[ln_no - 1]}" for ln_no in range(s_idx, e_idx + 1)]
        return f"[{target} symbol `{sym}` lines {s_idx}-{e_idx} of {total}]\n" + "\n".join(sliced)

    @actor_query
    def searchFile(self, path: str, pattern: str, context_lines: int = 12) -> str:
        """Search `path` for `pattern` (regex or substring) and return matching line windows with 1-based line numbers."""
        import re as _re

        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False)
        pat_str = str(pattern or "").strip()
        if not pat_str:
            return f"[{target.name}: empty search pattern]"

        if target.is_dir():
            proc = subprocess.run(
                ["grep", "-rnI", "--include=*.py", "-E", pat_str, "."],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=10,
            )
            raw_hits = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if not raw_hits:
                return f"[{target}: 0 directory matches for {pat_str!r}]"

            def _dir_hit_prio(ln: str) -> int:
                parts = ln.split(":", 2)
                f_part = parts[0] if parts else ""
                code_part = parts[2].lstrip() if len(parts) > 2 else ""
                is_test = 1 if ("/tests/" in f_part or "test_" in f_part) else 0
                if code_part.startswith("class "):
                    return is_test * 10 + 0
                if code_part.startswith("def "):
                    return is_test * 10 + 1
                return is_test * 10 + 2

            ranked = sorted(raw_hits, key=_dir_hit_prio)[:35]
            return f"[{target}: {len(raw_hits)} directory matches for {pat_str!r} (top {len(ranked)})]\n" + "\n".join(ranked)

        content = self._read_raw_text(target)
        lines = content.splitlines()
        total = len(lines)

        try:
            rx = _re.compile(pat_str)
        except Exception:
            rx = _re.compile(_re.escape(pat_str), _re.IGNORECASE)

        match_indices = [i for i, ln in enumerate(lines) if rx.search(ln) or pat_str.lower() in ln.lower()]
        if not match_indices:
            return f"[{target}: 0 matches for {pat_str!r} across {total} lines]"

        # Prioritize `class ` and `def ` definition lines ahead of comments/usages so large-file searches jump straight to definitions
        def _match_priority(i: int) -> Tuple[int, int]:
            s = lines[i].lstrip()
            if s.startswith("class "):
                return (0, i)
            if s.startswith("def "):
                return (1, i)
            if s.startswith("#"):
                return (3, i)
            return (2, i)

        ordered_indices = sorted(match_indices, key=_match_priority)

        # Pre-index all `def <func>(` definitions in the file for intra-file helper call-chain hints
        def_line_map: Dict[str, int] = {}
        for ln_i, ln_txt in enumerate(lines):
            dm = _re.match(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", ln_txt)
            if dm and dm.group(1) not in def_line_map:
                def_line_map[dm.group(1)] = ln_i + 1

        ctx = max(2, min(60, int(context_lines or 20)))
        windows: List[str] = []
        shown_ranges: List[Tuple[int, int]] = []
        for idx in ordered_indices:
            if len(windows) >= 6:
                break
            if any(r_s <= idx <= r_e for r_s, r_e in shown_ranges):
                continue
            s_ln = max(1, idx + 1 - 2)
            e_ln = min(total, idx + 1 + ctx)
            shown_ranges.append((s_ln - 1, e_ln - 1))
            win_lines = [f"{ln_no:4d}: {lines[ln_no - 1]}" for ln_no in range(s_ln, e_ln + 1)]
            block_str = f"--- Match at line {idx + 1} (showing {s_ln}..{e_ln} of {total}) ---\n" + "\n".join(win_lines)

            # If the matched line defines a Python class, append a compact outline of its base-class imports, `__slots__`, and `def` methods
            matched_line = lines[idx]
            cls_m = _re.match(r"^(\s*)class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?", matched_line)
            if cls_m:
                base_indent = len(cls_m.group(1))
                cls_name = cls_m.group(2)
                bases_raw = cls_m.group(3) or ""
                base_names = [b.strip().split("=")[-1].strip() for b in bases_raw.split(",") if b.strip()]
                import_hints: List[str] = []
                for k_ln in range(min(idx, 60)):
                    lk_s = lines[k_ln].strip()
                    if lk_s.startswith(("from ", "import ")) and any(bn and bn in lk_s for bn in base_names):
                        import_hints.append(f"  {k_ln + 1:4d}: {lk_s[:76]}")
                method_sigs: List[str] = []
                cls_end_ln = total
                for j in range(idx + 1, total):
                    lj = lines[j]
                    stripped = lj.lstrip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    cur_indent = len(lj) - len(stripped)
                    if cur_indent <= base_indent and not stripped.startswith((")", "]", "}", "@")):
                        cls_end_ln = j
                        break
                    if stripped.startswith(("def ", "@property", "__slots__")):
                        method_sigs.append(f"  {j + 1:4d}: {stripped[:72]}")
                if import_hints or method_sigs:
                    outline = "\n".join(import_hints + method_sigs[:35])
                    block_str += f"\n[Method Outline for `class {cls_name}` (lines {idx + 1}..{cls_end_ln})]:\n{outline}"

            windows.append(block_str)

        out = f"[{target}: {len(match_indices)} match(es) for {pat_str!r}]\n" + "\n\n".join(windows)
        if len(out) > self.max_output_chars:
            out = out[: self.max_output_chars] + "\n... [truncated]"
        return out

    @actor_query
    def vfsOverlayCount(self) -> int:
        """Return the number of files currently staged in the Copy-on-Write shadow VFS."""
        return len(self._shadow_vfs)

    @actor_query
    def previewEdit(self, path: str, old_text: str, new_text: str) -> str:
        """Speculatively stage `editFile` in the Copy-on-Write Shadow VFS (`_in_shadow_run=True`), read back the result, and roll back."""
        prev_shadow = getattr(self.vm, "_in_shadow_run", False)
        self.vm._in_shadow_run = True
        try:
            diff_out = self.editFile(path, old_text, new_text)
            read_out = self.readFile(path, 1, 120)
            return f"{diff_out}\n{read_out}"
        finally:
            self.clear_shadow_vfs()
            self.vm._in_shadow_run = prev_shadow

    def _auto_verify_python_edit(self, target: Path) -> str:
        """Automatically syntax-check modified `.py` files and run sibling `*_test.py` / `test_*.py` suites if present."""
        if target.suffix != ".py" or not target.is_file():
            return ""
        import py_compile

        try:
            py_compile.compile(str(target), doraise=True)
        except py_compile.PyCompileError as exc:
            return f"\n[AUTO-CHECK SYNTAX ERROR in {target.name}]: {exc.msg}"
        except Exception:
            pass

        if target.name.endswith("_test.py") or target.name.startswith("test_"):
            return ""

        candidate_tests = [
            target.with_name(f"{target.stem}_test.py"),
            target.with_name(f"test_{target.stem}.py"),
            target.parent / "tests" / f"test_{target.stem}.py",
            target.parent / "tests" / "test_outputs.py",
        ]
        test_file = next((p for p in candidate_tests if p.is_file()), None)
        if test_file is None:
            return ""

        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", "-q", "--tb=short", "-x", str(test_file)],
                cwd=str(target.parent),
                capture_output=True,
                text=True,
                timeout=6,
            )
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            import re as _re
            m_pass = _re.search(r"([0-9]+)\s+passed", combined)
            cur_passed = int(m_pass.group(1)) if m_pass else 0
            prev_best = self._best_pass_count.get(str(target), 0)
            if cur_passed > prev_best:
                self._best_pass_count[str(target)] = cur_passed
            lines = [ln for ln in combined.splitlines() if ln.strip()]
            if proc.returncode == 0:
                summary = lines[-1] if lines else "passed"
                return f"\n[AUTO-TEST PASS ({test_file.name})]: {summary} — All unit tests pass!"
            tail = "\n".join(lines[-22:])
            if len(tail) > 1200:
                tail = tail[-1200:]
            result = f"\n[AUTO-TEST FAIL ({test_file.name})]:\n{tail}"
            if prev_best > 0 and cur_passed < prev_best:
                result += f"\n[REGRESSION WARNING: passing tests dropped from {prev_best} to {cur_passed} — call undoLastEdit(Actor[#Host], \"{target}\") to restore]"
            return result
        except subprocess.TimeoutExpired:
            prev_best = self._best_pass_count.get(str(target), 0)
            reg_msg = (
                f"\n[REGRESSION WARNING: passing tests dropped from {prev_best} to TIMEOUT — call undoLastEdit(Actor[#Host], \"{target}\") to restore]"
                if prev_best > 0
                else ""
            )
            return (
                f"\n[AUTO-TEST FAIL ({test_file.name})]: Pytest timed out after 12s! "
                f"Your code in {target.name} has an infinite loop or self-referential cycle.{reg_msg}"
            )
        except Exception:
            return ""

    @staticmethod
    def _fuzzy_replace_block(original: str, old_s: str, new_s: str) -> Optional[str]:
        """Line-trimmed indentation-preserving fallback when exact substring match fails due to minor whitespace/indent drift."""
        orig_lines = original.splitlines(keepends=True)
        old_raw_lines = [ln for ln in old_s.splitlines() if ln.strip()]
        if not old_raw_lines:
            return None
        old_stripped = [ln.strip() for ln in old_raw_lines]

        non_empty_orig = [(idx, ln.strip()) for idx, ln in enumerate(orig_lines) if ln.strip()]
        n_target = len(old_stripped)
        if len(non_empty_orig) < n_target:
            return None

        matches: List[Tuple[int, int]] = []
        for k in range(len(non_empty_orig) - n_target + 1):
            if all(non_empty_orig[k + m][1] == old_stripped[m] for m in range(n_target)):
                start_line_idx = non_empty_orig[k][0]
                end_line_idx = non_empty_orig[k + n_target - 1][0]
                matches.append((start_line_idx, end_line_idx))

        if len(matches) != 1:
            return None

        start_idx, end_idx = matches[0]
        orig_first = orig_lines[start_idx]
        orig_indent = len(orig_first) - len(orig_first.lstrip(" "))
        old_first_indent = len(old_raw_lines[0]) - len(old_raw_lines[0].lstrip(" "))
        indent_delta = orig_indent - old_first_indent

        adj_new_lines: List[str] = []
        for nln in new_s.splitlines():
            if not nln.strip():
                adj_new_lines.append("\n")
            else:
                cur_ind = len(nln) - len(nln.lstrip(" "))
                new_ind = max(0, cur_ind + indent_delta)
                adj_new_lines.append((" " * new_ind) + nln.lstrip(" ") + "\n")

        replacement_block = "".join(adj_new_lines)
        if not orig_lines[end_idx].endswith(("\n", "\r")) and replacement_block.endswith("\n"):
            replacement_block = replacement_block[:-1]

        return "".join(orig_lines[:start_idx]) + replacement_block + "".join(orig_lines[end_idx + 1 :])

    @staticmethod
    def _sanitize_python_source(target: Path, text: str) -> str:
        """Escape raw null bytes (`\\x00`) and unescaped multi-line Python string literals that Actor single-backslash string literals produce when patching Python code."""
        if target.suffix != ".py":
            return text
        if "\x00" in text:
            text = (
                text.replace("'\r\n\x00'", "'\\r\\n\\0'")
                .replace("'\n\x00'", "'\\r\\n\\0'")
                .replace('"\r\n\x00"', '"\\r\\n\\0"')
                .replace('"\n\x00"', '"\\r\\n\\0"')
                .replace("\x00", "\\0")
            )
        # If Actor single-quoted/double-quoted string decoding turned `"...\n"` inside a Python single-line string into a literal newline before the closing quote, repair it so Python syntax stays valid
        import ast as _ast
        import re as _re

        try:
            _ast.parse(text)
            return text
        except SyntaxError:
            fixed = _re.sub(
                r'(["\'])([^\n"\']*?)\n(\s*)\1',
                lambda m: f"{m.group(1)}{m.group(2)}\\n{m.group(1)}\n{m.group(3)}",
                text,
            )
            try:
                _ast.parse(fixed)
                return fixed
            except SyntaxError:
                return text

    @actor_command(shadow_safe=True)
    def writeFile(self, path: str, content: str) -> str:
        """Write `content` to `path` (routes to in-memory CoW VFS during `shadow_only=True`)."""
        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False, for_write=True)
        text = self._sanitize_python_source(target, str(content if content is not None else ""))
        if getattr(self.vm, "_in_shadow_run", False):
            self._shadow_vfs[str(target)] = text
            return f"SHADOW_VFS_WRITE: {target} ({len(text)} chars staged in CoW overlay)"

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file(): self._last_file_backup[str(target)] = target.read_text(encoding="utf-8", errors="replace")
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, target)
        auto_check = self._auto_verify_python_edit(target)
        return f"WRITE_OK: {target} ({len(text)} chars){auto_check}"

    @actor_command(shadow_safe=True)
    def editFile(self, path: str, old_text: str, new_text: str) -> str:
        """Replace exact or whitespace-normalized `old_text` with `new_text` in `path` and return a compact unified diff."""
        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False, for_write=True)
        original = self._read_raw_text(target)
        old_s = str(old_text)
        new_s = self._sanitize_python_source(target, str(new_text))
        if not old_s:
            raise ValueError(f"editFile target string not found in {target.name} (empty old_text)")

        if old_s in original:
            updated = original.replace(old_s, new_s, 1)
        else:
            fuzzy_updated = self._fuzzy_replace_block(original, old_s, new_s)
            if fuzzy_updated is None:
                # Find the closest line in `original` to provide immediate line-numbered context
                orig_lines = original.splitlines()
                first_probe = next((ln.strip() for ln in old_s.splitlines() if ln.strip()), "")
                hint_str = ""
                if first_probe and orig_lines:
                    best_i = max(
                        range(len(orig_lines)),
                        key=lambda i: difflib.SequenceMatcher(None, first_probe, orig_lines[i].strip()).ratio(),
                    )
                    s_i = max(0, best_i - 4)
                    e_i = min(len(orig_lines), best_i + 8)
                    snippet = "\n".join(f"{idx + 1:4d}: {orig_lines[idx]}" for idx in range(s_i, e_i))
                    hint_str = f"\nClosest lines in {target.name} (lines {s_i + 1}..{e_i}):\n{snippet}"
                raise ValueError(f"editFile target string not found in {target.name}.{hint_str}")
            updated = fuzzy_updated
        updated = self._sanitize_python_source(target, updated)

        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{target.name}",
                tofile=f"b/{target.name}",
                lineterm="",
                n=2,
            )
        )
        diff_preview = "\n".join(diff_lines[:30])

        if getattr(self.vm, "_in_shadow_run", False):
            self._shadow_vfs[str(target)] = updated
            return f"SHADOW_VFS_EDIT: {target}\n{diff_preview}"

        self._last_file_backup[str(target)] = original
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(updated, encoding="utf-8")
        os.replace(tmp_path, target)
        auto_check = self._auto_verify_python_edit(target)
        return f"EDIT_OK: {target}\n{diff_preview}{auto_check}"

    @actor_command
    def undoLastEdit(self, path: str) -> str:
        target = self._resolve_safe_path(path, allow_read_anywhere_in_project=False, for_write=True)
        prev_text = self._last_file_backup.get(str(target))
        if prev_text is None:
            return f"UNDO_UNAVAILABLE: {target.name}"
        target.write_text(prev_text, encoding="utf-8")
        return f"UNDO_OK: {target} ({len(prev_text)} chars restored)"

    @actor_command
    def execShell(self, command: str, timeout_sec: int = 15) -> Dictionary:
        """Execute a bounded shell command inside the primary workspace root and return a Dictionary (`#exitCode`, `#stdout`, `#stderr`)."""
        cmd_str = self._sanitize_python_source(Path("cmd.py"), str(command or "").strip())
        if not cmd_str:
            raise ValueError("execShell requires a non-empty command string.")

        # Block host shell execution entirely when running in production host-lockdown outside Docker
        if is_host_lockdown() and os.environ.get("ACTORII_IN_DOCKER") != "1" and not os.path.exists("/.dockerenv"):
            raise PermissionError(
                "BLOCKED: Host shell execution via Actor[#Host] is strictly disabled in host-lockdown mode (`ACTORII_HOST_LOCKDOWN=1`). "
                "Use read-only Actor[#Valkyrie] telemetry twins or run inside the `actorii-sandbox` Docker container."
            )

        # Block dangerous destructive host/service/network/cron patterns
        validate_safe_shell_command(cmd_str)

        t_sec = max(1, min(60, int(timeout_sec or 15)))
        existing_roots = [r for r in self.allowed_roots if r.exists()]
        cwd = str(existing_roots[0] if existing_roots else Path.cwd())
        t_before = time.time()
        try:
            proc = subprocess.run(
                cmd_str,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=t_sec,
            )
            exit_code = int(proc.returncode)
            stdout_raw = proc.stdout or ""
            stderr_raw = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout_raw = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            stderr_raw = f"TimeoutExpired after {t_sec}s"

        if exit_code == 0 and not stdout_raw.strip():
            try:
                changed_artifacts = []
                for p in sorted(Path(cwd).iterdir()):
                    if p.is_file() and p.suffix in (".csv", ".json", ".parquet") and not p.name.startswith("."):
                        if p.stat().st_mtime >= t_before - 0.5:
                            if p.suffix == ".csv":
                                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                                changed_artifacts.append(f"{p.name} ({len(lines)} lines, head={lines[0] if lines else ''!r})")
                            else:
                                changed_artifacts.append(f"{p.name} ({p.stat().st_size} bytes)")
                if changed_artifacts:
                    stdout_raw = "[GENERATED OUTPUT FILES]: " + ", ".join(changed_artifacts)
            except Exception:
                pass

        def _clip(s: str) -> str:
            if len(s) <= self.max_output_chars:
                return s
            half = self.max_output_chars // 2
            return f"{s[:half]}\n... [truncated {len(s) - self.max_output_chars} chars] ...\n{s[-half:]}"

        out_dict = Dictionary(8)
        out_dict.atPut(Symbol("exitCode"), exit_code)
        out_dict.atPut(Symbol("stdout"), _clip(stdout_raw))
        out_dict.atPut(Symbol("stderr"), _clip(stderr_raw))
        return out_dict


@actor_bridge(name="VMReflectionBridge")
class VMReflectionBridge:
    """Tier 0 Deep VM Reflection Primitives exposed in Actor bytecode as `Actor[#VM]`."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    @actor_query
    def classes(self) -> ActorArray:
        """Return an Actor `Array` of `Dictionary` entries (`#name`, `#ring`, `#superclass`) for all loaded classes."""
        cls_map = getattr(self.vm.hierarchy, "_classes", {})
        names = sorted(cls_map.keys())
        arr = ActorArray(len(names))
        for i, cname in enumerate(names):
            c_obj = cls_map[cname]
            super_name = c_obj._bases[0] if getattr(c_obj, "_bases", None) else "Object"
            d = Dictionary(8)
            d.atPut(Symbol("name"), cname)
            d.atPut(Symbol("ring"), classify_class_ring(cname, c_obj))
            d.atPut(Symbol("superclass"), super_name)
            arr[i] = d
        return arr

    @actor_query
    def ringOf(self, class_symbol: Any) -> int:
        """Return the protection ring (`0`, `1`, or `2`) of `class_symbol`."""
        cname = _norm_sym(class_symbol)
        c_obj = self.vm.hierarchy.get(cname)
        return classify_class_ring(cname, c_obj)

    @actor_query
    def methodsOf(self, class_symbol: Any) -> ActorArray:
        """Return an Actor `Array` of method selector strings defined on `class_symbol`."""
        cname = _norm_sym(class_symbol)
        c_obj = self.vm.hierarchy.get(cname)
        if c_obj is None:
            return ActorArray(0)
        sels = sorted(set(list(getattr(c_obj, "_methods", {}).keys()) + list(getattr(c_obj, "_class_methods", {}).keys())))
        arr = ActorArray(len(sels))
        for i, s in enumerate(sels):
            arr[i] = s
        return arr

    @actor_query
    def sourceOf(self, class_symbol: Any, selector_symbol: Any) -> str:
        """Return the stored `.act` source chunk or bytecode disassembly for `class_symbol>>selector_symbol`."""
        cname = _norm_sym(class_symbol)
        sel = _norm_sym(selector_symbol)
        c_obj = self.vm.hierarchy.get(cname)
        if c_obj is None:
            return f"/* Class {cname} not found */"
        m_obj = c_obj._class_methods.get(sel) or c_obj._methods.get(sel) or self.vm.hierarchy.resolve_method(cname, sel)
        if m_obj is None:
            return f"/* Method {cname}>>{sel} not found */"
        src = getattr(m_obj, "source", None)
        if src:
            return str(src)
        insts = getattr(m_obj, "instructions", None)
        if insts is not None:
            params = getattr(m_obj, "params", [])
            locs = getattr(m_obj, "locals", [])
            ops_str = "; ".join(f"{op.name if hasattr(op, 'name') else op}({arg})" for op, arg in insts[:24])
            return f"/* CompiledMethod {cname}>>{sel} params={params} locals={locs}: {ops_str} */"
        return f"/* Builtin/Bridge callable {cname}>>{sel} */"

    @actor_query
    def ivarsOf(self, class_symbol: Any) -> ActorArray:
        """Return the ordered Actor `Array` of declared instance variable names for `class_symbol`."""
        cname = _norm_sym(class_symbol)
        ivars = self.vm._collect_class_ivars(cname) if hasattr(self.vm, "_collect_class_ivars") else []
        if not ivars:
            c_obj = self.vm.hierarchy.get(cname)
            ivars = [str(iv).lstrip("#") for iv in getattr(c_obj, "ivars", [])] if c_obj else []
        arr = ActorArray(len(ivars))
        for i, iv in enumerate(ivars):
            arr[i] = iv
        return arr
