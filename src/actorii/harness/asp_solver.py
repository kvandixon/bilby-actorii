"""Declarative Constraint & Answer Set Programming Solver Bridge (`actorii.harness.asp_solver`).

Exposes Potassco `clingo` as a zero-VRAM, pure-CPU Tier-0 Read-Only Actor bridge
(`ASPSolverBridge`, registered as `Actor[#Solver]`):

- `solveProgram(asp_code, max_models=1, timeout_s=5.0)`:
  Grounds and solves arbitrary safe ASP programs with bounded model count,
  hard CPU timeout, `#minimize`/`#maximize` optimization tracking, and
  assumption-based Minimal Unsatisfiable Core (MUC) extraction.
- `verifyInvariants(facts_asp, rules_asp, timeout_s=5.0)`:
  Verifies declarative facts against labeled integrity constraints or
  `violated(...)` rules, returning exact violated invariants and MUC.
- `solveDAGSchedule(tasks_json, max_parallel=1, timeout_s=5.0)`:
  Computes a makespan-optimal (and priority-aware) parallel DAG schedule
  via `clingo`, or extracts the exact dependency cycle / conflict MUC.
- `describe()`:
  Concise self-documentation and Actor 4.1 usage examples.

Guardrails:
- Strictly 0 MB GPU VRAM (pure CPU `clingo.Control`).
- Static pre-grounding sanitizer blocks `#include`, `#script`, and `@func(...)`
  external script calls (zero external file or shell I/O inside ASP).
- Hard timeout via `threading.Timer` -> `clingo.Control.interrupt()` plus
  `SolveHandle.wait(remaining)` so both grounding and solving are bounded.
- Graceful degradation when `clingo` is not installed (returns structured error
  `Dictionary` without crashing import).
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from actorii.bridge import actor_bridge, actor_query

try:
    import clingo  # type: ignore
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    clingo = None  # type: ignore[assignment]


MAX_ASP_CHARS = 262_144
MIN_TIMEOUT_S = 0.05
MAX_TIMEOUT_S = 30.0
MIN_MODELS = 1
MAX_MODELS = 100
INTERNAL_ASM_PREFIX = "__asp_c_"

_FORBIDDEN_DIRECTIVE_RE = re.compile(
    r"(?:^|[\s.;])(?:#include\b|#script\b|#end\b)",
    re.IGNORECASE,
)
_EXTERNAL_SCRIPT_CALL_RE = re.compile(r"@[A-Za-z_][A-Za-z0-9_]*\s*\(")
_OPT_DIRECTIVE_RE = re.compile(r"(?:^|[\s.;])(?:#minimize|#maximize)\b", re.IGNORECASE)
_EXTERNAL_STMT_RE = re.compile(
    r"^\s*#external\s+([a-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*(?::[^.]*)?\.\s*(?:\[\s*(true|false|free|release)\s*\])?\s*$",
    re.IGNORECASE,
)
_LABEL_COMMENT_RE = re.compile(
    r"%\s*(?:@(?:label|constraint)\s*\(\s*([^)]+?)\s*\)|\[\s*([^\]]+?)\s*\])\s*$",
    re.IGNORECASE,
)


def _to_actor_value(val: Any) -> Any:
    """Recursively convert Python dicts/lists/scalars into native Actor objects."""
    from actorii.classes.collections import ActorArray
    from actorii.classes.dictionary import Dictionary
    from actorii.core import Symbol

    if isinstance(val, dict):
        d = Dictionary(max(8, len(val) * 2))
        for k, v in val.items():
            conv_v = _to_actor_value(v)
            k_str = str(k).lstrip("#")
            d.atPut(k_str, conv_v)
            d.atPut(Symbol(k_str), conv_v)
        return d
    if isinstance(val, (list, tuple)):
        arr = ActorArray(len(val))
        for idx, item in enumerate(val):
            arr.atPut(idx, _to_actor_value(item))
        return arr
    return val


def _strip_comments_and_strings(code: str) -> str:
    """Remove ASP line/block comments and quoted string contents for static scanning."""
    out: List[str] = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        # Block comment %* ... *%
        if ch == "%" and i + 1 < n and code[i + 1] == "*":
            end = code.find("*%", i + 2)
            if end == -1:
                break
            out.append(" ")
            i = end + 2
            continue
        # Line comment % ... \n
        if ch == "%":
            nl = code.find("\n", i)
            if nl == -1:
                break
            out.append("\n")
            i = nl + 1
            continue
        # Double-quoted string "..."
        if ch == '"':
            j = i + 1
            while j < n:
                if code[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    break
                j += 1
            out.append('""')
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _validate_asp_safety(asp_code: str) -> Optional[str]:
    """Validate that `asp_code` contains no external file/shell/script I/O."""
    if not isinstance(asp_code, str):
        return "ASP program must be a string."
    if len(asp_code) > MAX_ASP_CHARS:
        return f"ASP program exceeds maximum length ({len(asp_code)} > {MAX_ASP_CHARS} chars)."
    cleaned = _strip_comments_and_strings(asp_code)
    if _FORBIDDEN_DIRECTIVE_RE.search(cleaned):
        return (
            "Forbidden ASP directive detected (#include/#script/#end are blocked for "
            "zero-I/O sandbox safety)."
        )
    if _EXTERNAL_SCRIPT_CALL_RE.search(cleaned):
        return (
            "Forbidden external script call (@func(...)) detected; external Python/Lua "
            "callbacks are blocked inside ASP."
        )
    return None


def _split_asp_statements(asp_code: str) -> List[Tuple[Optional[str], str]]:
    """Split ASP code into `(optional_label, statement_text)` pairs.

    Preserves `..` integer range operators, quoted strings, and associates
    immediately preceding `% @label(name)`, `% @constraint(name)`, or `% [name]`
    comments with the next statement.
    """
    statements: List[Tuple[Optional[str], str]] = []
    buf: List[str] = []
    pending_label: Optional[str] = None
    i = 0
    n = len(asp_code)

    while i < n:
        ch = asp_code[i]

        # Block comment %* ... *%
        if ch == "%" and i + 1 < n and asp_code[i + 1] == "*":
            end = asp_code.find("*%", i + 2)
            if end == -1:
                break
            i = end + 2
            continue

        # Line comment % ... \n
        if ch == "%":
            nl = asp_code.find("\n", i)
            line = asp_code[i:] if nl == -1 else asp_code[i:nl]
            m = _LABEL_COMMENT_RE.match(line.strip())
            if m:
                lbl = (m.group(1) or m.group(2) or "").strip()
                if lbl:
                    pending_label = lbl
            if nl == -1:
                break
            if buf:
                buf.append("\n")
            i = nl + 1
            continue

        # Quoted string
        if ch == '"':
            j = i + 1
            while j < n:
                if asp_code[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if asp_code[j] == '"':
                    j += 1
                    break
                j += 1
            buf.append(asp_code[i:j])
            i = j
            continue

        # Statement terminator `.` (but NOT `..` range operator)
        if ch == ".":
            if i + 1 < n and asp_code[i + 1] == ".":
                buf.append("..")
                i += 2
                continue
            buf.append(".")
            stmt = "".join(buf).strip()
            buf.clear()
            if stmt:
                statements.append((pending_label, stmt))
                pending_label = None
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append((pending_label, tail))
    return statements


def _instrument_constraints(
    asp_code: str,
    *,
    prefix: str = INTERNAL_ASM_PREFIX,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Wrap top-level integrity constraints (`:- body.`) with `#external` assumption atoms.

    Also detects user-declared `#external atom.` statements so explicit assumptions
    participate in MUC extraction.

    Returns:
        `(instrumented_code, assumption_specs)` where each entry in `assumption_specs`
        is `(atom_str, human_label)`.
    """
    stmts = _split_asp_statements(asp_code)
    out_lines: List[str] = []
    assumptions: List[Tuple[str, str]] = []
    c_idx = 0

    for label, stmt in stmts:
        ext_match = _EXTERNAL_STMT_RE.match(stmt)
        if ext_match:
            atom_str = ext_match.group(1).strip()
            truth = (ext_match.group(2) or "true").lower()
            out_lines.append(stmt)
            if truth != "false":
                assumptions.append((atom_str, label or atom_str))
            continue

        if stmt.startswith(":-"):
            body = stmt[2:].strip()
            if body.endswith("."):
                body = body[:-1].strip()
            asm_atom = f"{prefix}{c_idx}"
            c_idx += 1
            compact = " ".join(stmt.split())
            if len(compact) > 120:
                compact = compact[:117] + "..."
            human_label = label if label else f"constraint_{c_idx}: {compact}"
            out_lines.append(f"#external {asm_atom}. [free]")
            if body:
                out_lines.append(f":- {asm_atom}, {body}.")
            else:
                out_lines.append(f":- {asm_atom}.")
            assumptions.append((asm_atom, human_label))
            continue

        out_lines.append(stmt)

    return "\n".join(out_lines), assumptions


def _parse_atom_symbol(atom_str: str) -> Any:
    """Parse a ground atom string (e.g. `c1` or `dep(a,b)`) into a `clingo.Symbol`."""
    return clingo.parse_term(atom_str)


def _clamp_timeout(timeout_s: Any) -> float:
    try:
        val = float(timeout_s)
    except (TypeError, ValueError):
        val = 5.0
    return max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, val))


def _clamp_max_models(max_models: Any) -> int:
    try:
        val = int(max_models)
    except (TypeError, ValueError):
        val = 1
    return max(MIN_MODELS, min(MAX_MODELS, val))


def _shrink_to_muc(
    ctrl: Any,
    core_syms: List[Any],
    deadline: float,
) -> List[Any]:
    """Shrink an unsatisfiable core to an exact Minimal Unsatisfiable Core (MUC).

    Uses fast deletion-based filtering while within `deadline` and for compact cores.
    """
    if len(core_syms) <= 1 or len(core_syms) > 20:
        return core_syms

    muc = list(core_syms)
    idx = 0
    while idx < len(muc):
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            break
        candidate = muc[:idx] + muc[idx + 1 :]
        assumptions = [(s, True) for s in candidate]
        with ctrl.solve(assumptions=assumptions, async_=True) as handle:
            finished = handle.wait(remaining)
            if not finished:
                handle.cancel()
                break
            res = handle.get()
            if res.unsatisfiable:
                muc = candidate
                continue
        idx += 1
    return muc


def _run_clingo(
    asp_code: str,
    assumption_specs: Sequence[Tuple[str, str]],
    *,
    max_models: int,
    timeout_s: float,
) -> Dict[str, Any]:
    """Execute `clingo.Control` with bounded timeout, optimization tracking, and MUC."""
    t0 = time.monotonic()
    deadline = t0 + timeout_s

    if clingo is None:
        return {
            "satisfiable": False,
            "models": [],
            "optimum": False,
            "cost": None,
            "unsatCore": [],
            "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
            "error": "clingo is not installed; install with `python3 -m pip install --user clingo`.",
        }

    cleaned = _strip_comments_and_strings(asp_code)
    has_opt = bool(_OPT_DIRECTIVE_RE.search(cleaned))

    ctl_args: List[str] = []
    if has_opt:
        ctl_args.extend(["--opt-mode=optN", str(max_models)])
    else:
        ctl_args.append(str(max_models))

    clingo_errors: List[str] = []

    def _logger(code: Any, message: str) -> None:
        code_name = str(code)
        if "RuntimeError" in code_name or "Error" in code_name or "error:" in message.lower():
            clingo_errors.append(message.strip())

    ctrl = clingo.Control(ctl_args, logger=_logger)
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        try:
            ctrl.interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout_s, _on_timeout)
    timer.daemon = True
    timer.start()

    try:
        try:
            ctrl.add("base", [], asp_code)
            ctrl.ground([("base", [])])
        except RuntimeError as exc:
            err_detail = "; ".join(clingo_errors) if clingo_errors else str(exc)
            if timed_out.is_set() or "interrupted" in err_detail.lower():
                return {
                    "satisfiable": False,
                    "models": [],
                    "optimum": False,
                    "cost": None,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": f"ASP grounding timed out after {timeout_s:.2f}s.",
                }
            return {
                "satisfiable": False,
                "models": [],
                "optimum": False,
                "cost": None,
                "unsatCore": [],
                "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                "error": f"ASP syntax/grounding error: {err_detail}",
            }

        if timed_out.is_set():
            return {
                "satisfiable": False,
                "models": [],
                "optimum": False,
                "cost": None,
                "unsatCore": [],
                "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                "error": f"ASP solver timed out after {timeout_s:.2f}s.",
            }

        # Map ground assumption symbols -> solver literals and human labels
        assumptions: List[Tuple[Any, bool]] = []
        lit_to_sym: Dict[int, Any] = {}
        sym_to_label: Dict[Any, str] = {}
        for atom_str, label in assumption_specs:
            try:
                sym = _parse_atom_symbol(atom_str)
            except Exception:
                continue
            sa = ctrl.symbolic_atoms[sym]
            if sa is not None:
                # Ensure external atoms are free so positive assumptions don't conflict with default-false externals
                if sa.is_external:
                    ctrl.assign_external(sym, None)
                assumptions.append((sym, True))
                lit_to_sym[sa.literal] = sym
                sym_to_label[sym] = label

        collected_models: List[Tuple[bool, Optional[List[int]], List[str]]] = []
        raw_core_syms: List[Any] = []

        def _on_model(model: Any) -> None:
            atoms = sorted(
                str(sym)
                for sym in model.symbols(shown=True)
                if not str(sym).startswith(INTERNAL_ASM_PREFIX)
            )
            cost = list(model.cost) if model.cost else ([] if has_opt else None)
            collected_models.append((bool(model.optimality_proven), cost, atoms))

        def _on_core(core: Sequence[int]) -> None:
            seen_syms: set = set()
            for lit in core:
                sym = lit_to_sym.get(lit) or lit_to_sym.get(abs(lit))
                if sym is not None and sym not in seen_syms:
                    seen_syms.add(sym)
                    raw_core_syms.append(sym)

        remaining = max(0.01, deadline - time.monotonic())
        with ctrl.solve(
            assumptions=assumptions,
            on_model=_on_model,
            on_core=_on_core,
            async_=True,
        ) as handle:
            finished = handle.wait(remaining)
            if not finished:
                timed_out.set()
                handle.cancel()
                handle.wait(0.2)
            solve_res = handle.get()
            if solve_res.unsatisfiable and not raw_core_syms:
                try:
                    _on_core(handle.core())
                except Exception:
                    pass

        if timed_out.is_set() and not solve_res.satisfiable and not solve_res.unsatisfiable:
            return {
                "satisfiable": False,
                "models": [],
                "optimum": False,
                "cost": None,
                "unsatCore": [],
                "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                "error": f"ASP solver timed out after {timeout_s:.2f}s.",
            }

        if solve_res.satisfiable:
            optimum = False
            best_cost: Optional[List[int]] = None
            final_models: List[List[str]] = []

            if has_opt and collected_models:
                opt_models = [m for m in collected_models if m[0]]
                if opt_models:
                    optimum = True
                    best_cost = opt_models[-1][1]
                    chosen = opt_models
                else:
                    # Keep models matching the best (lowest) cost found before interruption
                    best_cost = min(
                        (m[1] for m in collected_models if m[1] is not None),
                        default=None,
                    )
                    chosen = [m for m in collected_models if m[1] == best_cost]
                seen_atom_sets: set = set()
                for _opt, _cost, atoms in chosen:
                    key = tuple(atoms)
                    if key not in seen_atom_sets:
                        seen_atom_sets.add(key)
                        final_models.append(atoms)
                    if len(final_models) >= max_models:
                        break
            else:
                for _opt, _cost, atoms in collected_models[:max_models]:
                    final_models.append(atoms)

            err_msg = ""
            if timed_out.is_set() and has_opt and not optimum:
                err_msg = f"ASP optimization timed out after {timeout_s:.2f}s; returning best suboptimal model."

            return {
                "satisfiable": True,
                "models": final_models,
                "optimum": optimum,
                "cost": best_cost,
                "unsatCore": [],
                "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                "error": err_msg,
            }

        # UNSAT path: shrink to exact MUC when possible
        muc_syms = _shrink_to_muc(ctrl, raw_core_syms, deadline) if raw_core_syms else []
        unsat_core = [sym_to_label[s] for s in muc_syms if s in sym_to_label]

        return {
            "satisfiable": False,
            "models": [],
            "optimum": False,
            "cost": None,
            "unsatCore": unsat_core,
            "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
            "error": "",
        }
    finally:
        timer.cancel()


@actor_bridge(name="ASPSolverBridge")
class ASPSolverBridge:
    """Zero-VRAM CPU Answer Set Programming (`clingo`) bridge exposed at `Actor[#Solver]`."""

    def __init__(self, default_timeout_s: float = 5.0) -> None:
        self._default_timeout_s = _clamp_timeout(default_timeout_s)

    @actor_query
    def solveProgram(
        self,
        asp_code: str,
        max_models: int = 1,
        timeout_s: float = 5.0,
    ) -> Any:
        """Solve an ASP program with bounded models/timeout and MUC extraction.

        Returns an Actor `Dictionary` with keys:
        `#satisfiable`, `#models`, `#optimum`, `#cost`, `#unsatCore`, `#elapsedMs`, `#error`.
        """
        t0 = time.monotonic()
        err = _validate_asp_safety(asp_code)
        if err:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "models": [],
                    "optimum": False,
                    "cost": None,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": err,
                }
            )

        instrumented, assumption_specs = _instrument_constraints(asp_code)
        res = _run_clingo(
            instrumented,
            assumption_specs,
            max_models=_clamp_max_models(max_models),
            timeout_s=_clamp_timeout(timeout_s),
        )
        return _to_actor_value(res)

    @actor_query
    def verifyInvariants(
        self,
        facts_asp: str,
        rules_asp: str,
        timeout_s: float = 5.0,
    ) -> Any:
        """Check declarative `facts_asp` against labeled constraints/rules in `rules_asp`.

        Supports both integrity constraints (`:- ... .` with optional `% @label(name)`
        or `% [name]` tags) and `violated(name) :- ... .` reporting predicates.

        Returns an Actor `Dictionary` with keys:
        `#ok`, `#violated`, `#unsatCore`, `#elapsedMs`, `#error`.
        """
        t0 = time.monotonic()
        for label, chunk in (("facts_asp", facts_asp), ("rules_asp", rules_asp)):
            err = _validate_asp_safety(chunk)
            if err:
                return _to_actor_value(
                    {
                        "ok": False,
                        "violated": [],
                        "unsatCore": [],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"{label}: {err}",
                    }
                )

        # Keep facts_asp un-instrumented (except explicit #external), instrument rules_asp constraints
        facts_stmts = _split_asp_statements(facts_asp)
        facts_assumptions: List[Tuple[str, str]] = []
        for lbl, stmt in facts_stmts:
            m = _EXTERNAL_STMT_RE.match(stmt)
            if m and (m.group(2) or "true").lower() != "false":
                atom_str = m.group(1).strip()
                facts_assumptions.append((atom_str, lbl or atom_str))

        inst_rules, rules_assumptions = _instrument_constraints(rules_asp)
        combined = f"{facts_asp}\n{inst_rules}"
        res = _run_clingo(
            combined,
            facts_assumptions + rules_assumptions,
            max_models=1,
            timeout_s=_clamp_timeout(timeout_s),
        )

        if res["error"]:
            return _to_actor_value(
                {
                    "ok": False,
                    "violated": [],
                    "unsatCore": res["unsatCore"],
                    "elapsedMs": res["elapsedMs"],
                    "error": res["error"],
                }
            )

        if not res["satisfiable"]:
            core = list(res["unsatCore"])
            return _to_actor_value(
                {
                    "ok": False,
                    "violated": core,
                    "unsatCore": core,
                    "elapsedMs": res["elapsedMs"],
                    "error": "",
                }
            )

        # SAT: also check for explicit `violated(...)` atoms in the model
        violated_atoms: List[str] = []
        if res["models"]:
            for atom in res["models"][0]:
                if atom == "violated" or atom.startswith("violated("):
                    violated_atoms.append(atom)

        return _to_actor_value(
            {
                "ok": len(violated_atoms) == 0,
                "violated": violated_atoms,
                "unsatCore": [],
                "elapsedMs": res["elapsedMs"],
                "error": "",
            }
        )

    @actor_query
    def solveDAGSchedule(
        self,
        tasks_json: str,
        max_parallel: int = 1,
        timeout_s: float = 5.0,
    ) -> Any:
        """Compute an optimal valid parallel DAG schedule or exact cycle/conflict MUC.

        `tasks_json`: JSON array of `{id, duration, deps: [...], priority?}`.
        `max_parallel`: Maximum number of concurrently running tasks (>= 1).

        Returns an Actor `Dictionary` with keys:
        `#satisfiable`, `#makespan`, `#schedule`, `#optimum`, `#unsatCore`, `#elapsedMs`, `#error`.
        """
        t0 = time.monotonic()
        if clingo is None:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": "clingo is not installed; install with `python3 -m pip install --user clingo`.",
                }
            )

        try:
            par = int(max_parallel)
        except (TypeError, ValueError):
            par = 1
        if par < 1:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": f"max_parallel must be >= 1, got {max_parallel!r}.",
                }
            )

        try:
            raw_tasks = json.loads(tasks_json) if isinstance(tasks_json, str) else tasks_json
        except Exception as exc:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": f"Invalid tasks_json: {exc}",
                }
            )

        if not isinstance(raw_tasks, list):
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": "tasks_json must be a JSON array of task objects.",
                }
            )

        if not raw_tasks:
            return _to_actor_value(
                {
                    "satisfiable": True,
                    "makespan": 0,
                    "schedule": [],
                    "optimum": True,
                    "unsatCore": [],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": "",
                }
            )

        # Normalize and validate tasks
        tasks: List[Dict[str, Any]] = []
        id_to_idx: Dict[str, int] = {}
        for idx, item in enumerate(raw_tasks):
            if not isinstance(item, dict) or "id" not in item:
                return _to_actor_value(
                    {
                        "satisfiable": False,
                        "makespan": None,
                        "schedule": [],
                        "optimum": False,
                        "unsatCore": [],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"Task at index {idx} must be an object with an 'id' field.",
                    }
                )
            tid = str(item["id"]).strip()
            if not tid:
                return _to_actor_value(
                    {
                        "satisfiable": False,
                        "makespan": None,
                        "schedule": [],
                        "optimum": False,
                        "unsatCore": [],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"Task at index {idx} has an empty 'id'.",
                    }
                )
            if tid in id_to_idx:
                return _to_actor_value(
                    {
                        "satisfiable": False,
                        "makespan": None,
                        "schedule": [],
                        "optimum": False,
                        "unsatCore": [f"duplicate_task:{tid}"],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"Duplicate task id '{tid}'.",
                    }
                )
            try:
                dur = int(item.get("duration", 1))
            except (TypeError, ValueError):
                dur = -1
            if dur < 1:
                return _to_actor_value(
                    {
                        "satisfiable": False,
                        "makespan": None,
                        "schedule": [],
                        "optimum": False,
                        "unsatCore": [f"invalid_duration:{tid}"],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"Task '{tid}' duration must be a positive integer, got {item.get('duration')!r}.",
                    }
                )
            deps = item.get("deps", [])
            if deps is None:
                deps = []
            if not isinstance(deps, list):
                return _to_actor_value(
                    {
                        "satisfiable": False,
                        "makespan": None,
                        "schedule": [],
                        "optimum": False,
                        "unsatCore": [],
                        "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                        "error": f"Task '{tid}' field 'deps' must be a list.",
                    }
                )
            try:
                prio = int(item.get("priority", 0) or 0)
            except (TypeError, ValueError):
                prio = 0
            id_to_idx[tid] = idx
            tasks.append(
                {
                    "idx": idx,
                    "id": tid,
                    "duration": dur,
                    "deps": [str(d).strip() for d in deps],
                    "priority": max(0, prio),
                }
            )

        # Check for unknown dependency references
        missing_deps: List[str] = []
        edges: List[Tuple[int, int, str]] = []  # (dep_idx, task_idx, label)
        for t in tasks:
            for dep_id in t["deps"]:
                if dep_id not in id_to_idx:
                    missing_deps.append(f"missing_dep:{dep_id}->{t['id']}")
                else:
                    edges.append((id_to_idx[dep_id], t["idx"], f"dep:{dep_id}->{t['id']}"))

        if missing_deps:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": missing_deps,
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": f"Unknown dependency reference(s): {', '.join(missing_deps)}",
                }
            )

        horizon = sum(t["duration"] for t in tasks)
        t_limit = _clamp_timeout(timeout_s)

        # Phase 1: Pure ASP cycle check with per-edge assumptions for exact cycle MUC
        # (fast even for large horizons; avoids grounding time slots on cyclic graphs)
        cycle_lines: List[str] = []
        edge_assumptions: List[Tuple[str, str]] = []
        for e_idx, (u, v, lbl) in enumerate(edges):
            asm = f"{INTERNAL_ASM_PREFIX}e_{e_idx}"
            cycle_lines.append(f"#external {asm}. [free]")
            cycle_lines.append(f"edge({u}, {v}) :- {asm}.")
            edge_assumptions.append((asm, lbl))
        cycle_lines.append("reach(X, Y) :- edge(X, Y).")
        cycle_lines.append("reach(X, Z) :- reach(X, Y), edge(Y, Z).")
        cycle_lines.append(":- reach(X, X).")

        cycle_res = _run_clingo(
            "\n".join(cycle_lines),
            edge_assumptions,
            max_models=1,
            timeout_s=t_limit,
        )
        if cycle_res["error"]:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": cycle_res["unsatCore"],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": cycle_res["error"],
                }
            )
        if not cycle_res["satisfiable"]:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": cycle_res["unsatCore"],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": "",
                }
            )

        # Phase 2: Makespan-optimal (+ priority tie-break) parallel DAG scheduling in clingo
        asp_lines: List[str] = [f"max_parallel({par}).", f"time(0..{horizon - 1})."]
        for t in tasks:
            asp_lines.append(f"task({t['idx']}, {t['duration']}, {t['priority']}).")
        for u, v, _lbl in edges:
            asp_lines.append(f"dep({u}, {v}).")

        asp_lines.extend(
            [
                f"1 {{ start(T, S) : S = 0..{horizon}-D }} 1 :- task(T, D, _).",
                ":- dep(U, V), start(U, SU), task(U, DU, _), start(V, SV), SV < SU + DU.",
                "running(T, Time) :- start(T, S), task(T, D, _), Time = S..S+D-1.",
                ":- time(Time), max_parallel(P), #count { T : running(T, Time) } > P.",
                "finish(T, S + D) :- start(T, S), task(T, D, _).",
                "makespan(M) :- M = #max { F : finish(_, F) }.",
                "#minimize { M@2 : makespan(M) }.",
                "#minimize { S*P@1, T : start(T, S), task(T, _, P), P > 0 }.",
                "#show start/2.",
                "#show makespan/1.",
            ]
        )

        remaining = max(MIN_TIMEOUT_S, t_limit - (time.monotonic() - t0))
        sched_res = _run_clingo(
            "\n".join(asp_lines),
            [],
            max_models=1,
            timeout_s=remaining,
        )

        if not sched_res["satisfiable"] or not sched_res["models"]:
            return _to_actor_value(
                {
                    "satisfiable": False,
                    "makespan": None,
                    "schedule": [],
                    "optimum": False,
                    "unsatCore": sched_res["unsatCore"],
                    "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": sched_res["error"],
                }
            )

        start_map: Dict[int, int] = {}
        makespan_val: Optional[int] = None
        for atom in sched_res["models"][0]:
            m_start = re.match(r"^start\((\d+),(\d+)\)$", atom)
            if m_start:
                start_map[int(m_start.group(1))] = int(m_start.group(2))
                continue
            m_ms = re.match(r"^makespan\((\d+)\)$", atom)
            if m_ms:
                makespan_val = int(m_ms.group(1))

        schedule_entries: List[Dict[str, Any]] = []
        for t in tasks:
            st = start_map.get(t["idx"], 0)
            end = st + t["duration"]
            schedule_entries.append(
                {
                    "id": t["id"],
                    "start": st,
                    "end": end,
                    "duration": t["duration"],
                    "priority": t["priority"],
                }
            )
        schedule_entries.sort(key=lambda e: (e["start"], e["id"]))
        if makespan_val is None and schedule_entries:
            makespan_val = max(e["end"] for e in schedule_entries)

        return _to_actor_value(
            {
                "satisfiable": True,
                "makespan": makespan_val,
                "schedule": schedule_entries,
                "optimum": bool(sched_res["optimum"]),
                "unsatCore": [],
                "elapsedMs": round((time.monotonic() - t0) * 1000.0, 2),
                "error": sched_res["error"],
            }
        )

    @actor_query
    def describe(self) -> str:
        """Return concise self-documentation of `Actor[#Solver]` (`ASPSolverBridge`)."""
        version = getattr(clingo, "__version__", "unavailable") if clingo is not None else "unavailable"
        return (
            f"ASPSolverBridge (Actor[#Solver]) | clingo={version} | 0 MB GPU VRAM (pure CPU)\n"
            "Methods (all Tier-0 @actor_query):\n"
            "1. solveProgram(asp_code, max_models=1, timeout_s=5.0) -> Dictionary\n"
            "   Keys: #satisfiable, #models, #optimum, #cost, #unsatCore, #elapsedMs, #error\n"
            "   Supports #minimize/#maximize and assumption-based MUC via `% @label(name)` or `#external`.\n"
            "   Example: solveProgram(Actor[#Solver], \"a. b. % @label(no_ab)\\n:- a, b.\", 1, 5.0)\n"
            "2. verifyInvariants(facts_asp, rules_asp, timeout_s=5.0) -> Dictionary\n"
            "   Keys: #ok, #violated, #unsatCore, #elapsedMs, #error\n"
            "   Example: verifyInvariants(Actor[#Solver], \"port(8080).\", \"% [no_8080]\\n:- port(8080).\", 5.0)\n"
            "3. solveDAGSchedule(tasks_json, max_parallel=1, timeout_s=5.0) -> Dictionary\n"
            "   Input JSON: [{\"id\":\"A\",\"duration\":2,\"deps\":[],\"priority\":1}, ...]\n"
            "   Keys: #satisfiable, #makespan, #schedule, #optimum, #unsatCore, #elapsedMs, #error\n"
            "4. describe() -> String\n"
            "Guardrails: #include, #script, and @func() are blocked; timeout clamped to [0.05, 30.0]s."
        )


__all__ = ["ASPSolverBridge"]
