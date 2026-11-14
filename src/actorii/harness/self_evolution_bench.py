"""Bilby Self-Evolution Loop Benchmark & Deterministic Evolution Runner (`actorii.harness.self_evolution_bench`).

Measures whether `Actor[#Bilby]` (the metacircular `AgentHarness`) can *grow a new
capability into a live `ActorVM` image* across several self-evolution generations,
and whether that capability survives a clean reload.  Every verdict is produced by a
deterministic oracle (Actor 4.1 test expressions + optional restricted Python
oracles), never by an LLM judge.

State machine (per `EvolutionSpec`)::

    Phase 0  Pre-flight baseline oracle   : fresh sandbox VM, tests MUST NOT all pass yet
    Phase 1  Generation g = 1..G          : fresh VM + replay of committed source, Bilby runs
                                            `runGoal` with goal + contract + VISIBLE tests
                                            (+ structured diagnostics from generation g-1)
    Phase 2  Verification / regression    : visible + HOLDOUT tests, core VM/harness invariants,
                                            opcode (gas) accounting; broken generations are
                                            rolled back (their code is never committed)
    Phase 3  Persistence / reload gate    : `.act` file-out (or `:ActorEngram`) -> brand-new
                                            `ActorVM` must pass 100% of tests

Rollback is replay based: the committed capability is a list of definitional Actor
chunks; each generation starts from a *fresh* bootstrapped VM and replays only the
accepted chunks, so a rejected generation cannot leak any state.

Modes:
    scripted : deterministic reference solver (intentionally flawed generation 1,
               diagnostic-aware self-correcting generation 2) - no LLM, <2s per spec.
    live     : real Bilby over an OpenAI-compatible endpoint (`temperature=0.0`, `seed=42`).

Public API (consumed by `Actor[#Bench]` / the CLI):
    EvolutionTest, EvolutionSpec, DEFAULT_EVOLUTION_SPECS, SelfEvolutionRunner,
    load_spec_file, list_self_evolution_suites, run_self_evolution_spec,
    run_self_evolution_suite, summarize_result, render_markdown.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

TIER_METHOD = "tier1_method"
TIER_CLASS = "tier2_class_engram"
TIER_HARNESS = "tier3_harness_policy"
VALID_TIERS = (TIER_METHOD, TIER_CLASS, TIER_HARNESS)

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8:latest"
DEFAULT_SEED = 42
DIAGNOSTICS_MARKER = "PREVIOUS GENERATION DIAGNOSTICS"

# ---------------------------------------------------------------------------
# Spec schema
# ---------------------------------------------------------------------------


@dataclass
class EvolutionTest:
    """One deterministic check: an Actor 4.1 expression (one or more `!!` chunks).

    The value of the *last* chunk is compared to `expected`, or - when `oracle` is set -
    a restricted Python expression evaluated with the normalized value bound to `result`.
    """

    expr: str
    expected: Any = None
    tag: str = ""
    oracle: Optional[str] = None

    @classmethod
    def from_any(cls, raw: Any) -> "EvolutionTest":
        if isinstance(raw, EvolutionTest):
            return raw
        if isinstance(raw, dict):
            if "expr" not in raw:
                raise ValueError(f"test entry missing 'expr': {raw!r}")
            return cls(
                expr=str(raw["expr"]),
                expected=raw.get("expected"),
                tag=str(raw.get("tag", "") or ""),
                oracle=raw.get("oracle"),
            )
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return cls(expr=str(raw[0]), expected=raw[1])
        raise ValueError(f"unsupported test entry: {raw!r}")


@dataclass
class EvolutionSpec:
    spec_id: str
    name: str
    difficulty: str
    goal: str
    spec_prompt: str
    visible_tests: List[EvolutionTest]
    holdout_tests: List[EvolutionTest]
    max_generations: int = 3
    max_steps_per_gen: int = 8
    gas_budget: int = 400_000
    require_persistence: bool = True
    required_classes: List[str] = field(default_factory=list)
    setup_source: str = ""  # environment loaded into baseline + every generation VM
    fixture_source: str = ""  # evaluated just before the tests run (never persisted)
    reload_setup_source: str = ""  # supporting classes for the Phase-3 reload VM only
    scripted_generations: List[str] = field(default_factory=list)  # scripted solver (gen1 flawed, gen2 fixed)
    allow_harness_mutation: bool = False
    test_opcode_cap: int = 60_000

    def __post_init__(self) -> None:
        self.visible_tests = [EvolutionTest.from_any(t) for t in self.visible_tests]
        self.holdout_tests = [EvolutionTest.from_any(t) for t in self.holdout_tests]
        self.validate()

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", self.spec_id or ""):
            raise ValueError(f"invalid spec_id {self.spec_id!r} (use letters, digits, _ or -)")
        if self.difficulty not in VALID_TIERS:
            raise ValueError(f"spec {self.spec_id}: difficulty must be one of {VALID_TIERS}, got {self.difficulty!r}")
        if not self.visible_tests:
            raise ValueError(f"spec {self.spec_id}: needs at least one visible test")
        if not self.holdout_tests:
            raise ValueError(f"spec {self.spec_id}: needs at least one holdout test (anti reward-hacking)")
        if self.max_generations < 1 or self.max_steps_per_gen < 1 or self.gas_budget < 1:
            raise ValueError(f"spec {self.spec_id}: max_generations/max_steps_per_gen/gas_budget must be >= 1")
        if self.require_persistence and not self.required_classes:
            raise ValueError(f"spec {self.spec_id}: require_persistence needs required_classes")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown spec fields: {sorted(unknown)}")
        for key in ("spec_id", "name", "difficulty", "goal", "spec_prompt", "visible_tests", "holdout_tests"):
            if key not in data:
                raise ValueError(f"spec missing required field {key!r}")
        payload = dict(data)
        sg = payload.get("scripted_generations") or []
        payload["scripted_generations"] = [str(s) for s in sg]
        return cls(**payload)


def load_spec_file(path: Union[str, Path]) -> List[EvolutionSpec]:
    """Load one or many specs from JSON: an object, a list of objects, or `{"specs": [...]}`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "specs" in raw:
        raw = raw["specs"]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a spec object, a list of specs, or {{'specs': [...]}}")
    return [EvolutionSpec.from_dict(item) for item in raw]


# ---------------------------------------------------------------------------
# Value normalisation & oracles
# ---------------------------------------------------------------------------


def normalize_value(value: Any, _depth: int = 0) -> Any:
    """Convert VM values into JSON-comparable Python values."""
    if _depth > 8:
        return "<deep>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    cls_name = type(value).__name__
    if cls_name == "Symbol":
        return "#" + str(getattr(value, "name", value)).lstrip("#")
    if isinstance(value, dict):
        return {str(normalize_value(k, _depth + 1)): normalize_value(v, _depth + 1) for k, v in value.items()}
    entries = getattr(value, "_entries", None)
    if isinstance(entries, dict):
        return {str(normalize_value(k, _depth + 1)): normalize_value(v, _depth + 1) for k, v in entries.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v, _depth + 1) for v in value]
    try:
        if hasattr(value, "__iter__") and not hasattr(value, "_fields"):
            return [normalize_value(v, _depth + 1) for v in list(value)]
    except Exception:
        pass
    cname = getattr(value, "class_name", None) or cls_name
    return f"<{cname}>"


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(actual) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(values_equal(a, e) for a, e in zip(actual, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(values_equal(actual[k], expected[k]) for k in expected)
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected  # VM booleans may surface as 1/0 (True == 1)
    return actual == expected


_ORACLE_BUILTINS = {
    n: getattr(builtins, n)
    for n in ("abs", "all", "any", "bool", "dict", "float", "int", "isinstance", "len", "list", "max", "min", "range",
              "round", "set", "sorted", "str", "sum", "tuple", "zip")
}


def run_oracle(expr: str, result: Any) -> bool:
    """Evaluate a restricted deterministic Python oracle (`result` is the normalized VM value).

    This is NOT a security sandbox: spec files are trusted input (like any test file).
    """
    return bool(eval(expr, {"__builtins__": _ORACLE_BUILTINS}, {"result": result}))  # noqa: S307


@dataclass
class TestOutcome:
    __test__ = False  # not a pytest class

    expr: str
    passed: bool
    expected: Any = None
    actual: Any = None
    error: str = ""
    tag: str = ""
    opcodes: int = 0

    def brief(self, max_len: int = 160) -> str:
        e = self.expr if len(self.expr) <= max_len else self.expr[: max_len - 3] + "..."
        if self.error:
            return f"{e}  ->  ERROR {self.error[:120]}"
        return f"{e}  ->  got {self.actual!r}, expected {self.expected!r}"


def run_tests(vm: Any, tests: Sequence[EvolutionTest], *, fixture: str = "", opcode_cap: int = 60_000) -> List[TestOutcome]:
    """Run deterministic tests against `vm` (state is isolated per test via unique `Actor[#..]` names)."""
    outcomes: List[TestOutcome] = []
    saved_cap = getattr(vm, "max_opcodes", None)
    try:
        vm.max_opcodes = int(opcode_cap)
        if fixture.strip():
            try:
                vm.eval_chunk(fixture)
            except Exception as exc:  # fixture failure is a spec bug or tampering: fail every test loudly
                msg = f"fixture failed: {type(exc).__name__}: {exc}"
                return [TestOutcome(t.expr, False, t.expected, None, msg, t.tag) for t in tests]
        for t in tests:
            src = t.expr.strip()
            if not src.endswith("!!"):
                src += "!!"
            try:
                raw = vm.eval_chunk(src)
                norm = normalize_value(raw)
                ops = int(getattr(vm, "last_eval_opcodes", 0) or 0)
                if t.oracle:
                    ok = run_oracle(t.oracle, norm)
                    expected: Any = f"oracle:{t.oracle}"
                else:
                    ok = values_equal(norm, t.expected)
                    expected = t.expected
                outcomes.append(TestOutcome(t.expr, ok, expected, norm, "", t.tag, ops))
            except Exception as exc:
                outcomes.append(
                    TestOutcome(t.expr, False, t.expected if not t.oracle else f"oracle:{t.oracle}", None,
                                f"{type(exc).__name__}: {exc}", t.tag,
                                int(getattr(vm, "last_eval_opcodes", 0) or 0))
                )
    finally:
        if saved_cap is not None:
            vm.max_opcodes = saved_cap
    return outcomes


def _pass_count(outs: Sequence[TestOutcome]) -> int:
    return sum(1 for o in outs if o.passed)


def _rate(outs: Sequence[TestOutcome]) -> float:
    return round(_pass_count(outs) / len(outs), 4) if outs else 0.0


# ---------------------------------------------------------------------------
# Source helpers (definitional chunks, fingerprints)
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r"^\s*(inherit\s*\(|now\s*\(|Def\s)", re.IGNORECASE)


def split_chunks(source: str) -> List[str]:
    """Split Actor source into `!!` terminated chunks (string/comment aware)."""
    from actorii.vm import ActorVM

    return [c for c in ActorVM()._split_chunks_safely(source) if c.strip()]  # noqa: SLF001


def definitional_chunks(source: str) -> List[str]:
    """Keep only class/method definition chunks (`inherit`, `now`, `Def`) - never probes/side effects."""
    out: List[str] = []
    for chunk in split_chunks(source):
        body = re.sub(r"/\*.*?\*/", "", chunk, flags=re.DOTALL).strip()
        if body and _DEF_RE.match(body):
            out.append(chunk.strip() if chunk.strip().endswith("!!") else chunk.strip() + "!!")
    return out


def _method_sig(m: Any) -> str:
    src = getattr(m, "source", None)
    if src:
        return hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:12]
    instr = getattr(m, "instructions", None) or getattr(m, "_instructions", None)
    return f"bc{len(instr) if instr is not None else 0}"


def vm_fingerprint(vm: Any) -> Dict[str, Dict[str, str]]:
    """Fingerprint `AgentHarness` + protected system classes (selector -> source hash)."""
    from actorii.guardrails.linter import PROTECTED_SYSTEM_CLASSES

    names = {"AgentHarness", *PROTECTED_SYSTEM_CLASSES}
    fp: Dict[str, Dict[str, str]] = {}
    for name in sorted(names):
        cls = vm.hierarchy.get(name)
        if cls is None:
            continue
        methods = getattr(cls, "_class_methods", {}) or {}
        fp[name] = {str(sel): _method_sig(m) for sel, m in sorted(methods.items())}
    return fp


def check_invariants(vm: Any, baseline_fp: Dict[str, Dict[str, str]], spec: EvolutionSpec) -> List[str]:
    """Core VM/harness sanity invariants + protected-class tamper check. Returns violation strings."""
    from actorii.guardrails.linter import RING1_MUTABLE_HARNESS_SELECTORS

    violations: List[str] = []
    try:
        if vm.eval_chunk("(6 * 7)!!") != 42:
            violations.append("sanity: (6 * 7) != 42")
        if vm.eval_chunk("size(#(1, 2, 3))!!") != 3:
            violations.append("sanity: size(#(1,2,3)) != 3")
    except Exception as exc:
        violations.append(f"sanity: core VM evaluation broke: {type(exc).__name__}: {exc}")
    harness = (vm.global_vars.get("Actor") or {}).get("Harness")
    if harness is None:
        violations.append("sanity: Actor[#Harness] missing")
    now_fp = vm_fingerprint(vm)
    for cname, base_methods in baseline_fp.items():
        cur = now_fp.get(cname)
        if cur is None:
            violations.append(f"tamper: protected class {cname} disappeared")
            continue
        for sel, sig in base_methods.items():
            if cname == "AgentHarness" and spec.allow_harness_mutation and sel in RING1_MUTABLE_HARNESS_SELECTORS:
                continue
            if cur.get(sel) != sig:
                violations.append(f"tamper: {cname}>>{sel} was modified or removed")
        if cname != "AgentHarness":
            added = sorted(set(cur) - set(base_methods))
            if added:
                violations.append(f"tamper: {cname} gained unexpected selectors {added}")
    return violations


# ---------------------------------------------------------------------------
# Live LLM provider (deterministic: temperature 0.0, fixed seed)
# ---------------------------------------------------------------------------


class LiveLLMError(RuntimeError):
    pass


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE) if "</think>" in text else text
    return text.strip()


def make_live_provider(
    harness: Any,
    *,
    endpoint: str,
    model: str,
    temperature: float,
    seed: int,
    timeout: float,
    max_tokens: int,
    stats: Dict[str, Any],
) -> Callable[[str, str, dict], str]:
    """Return a `(prompt, system_prompt, opts) -> reply` callable hitting an OpenAI-compatible endpoint."""
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    def provider(prompt: str, system_prompt: str, opts: dict) -> str:
        sys_txt = str(getattr(harness, "system_prompt", "") or system_prompt or "")
        payload = {
            "model": model,
            "messages": ([{"role": "system", "content": sys_txt}] if sys_txt else [])
            + [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            stats["llm_error"] = f"{type(exc).__name__}: {exc}"
            return f"COMPLETE: [LLM_ERROR] {type(exc).__name__}: {exc}"
        stats["llm_calls"] = stats.get("llm_calls", 0) + 1
        stats["llm_latency_s"] = stats.get("llm_latency_s", 0.0) + (time.perf_counter() - t0)
        usage = parsed.get("usage") or {}
        stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + int(usage.get("prompt_tokens") or 0)
        stats["completion_tokens"] = stats.get("completion_tokens", 0) + int(usage.get("completion_tokens") or 0)
        choices = parsed.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        return _strip_thinking(str(content))

    return provider


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _scoped_env(updates: Dict[str, str]) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SelfEvolutionRunner:
    """Deterministic Phase 0..3 self-evolution state machine."""

    def __init__(
        self,
        mode: str = "scripted",
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        seed: int = DEFAULT_SEED,
        max_generations: Optional[int] = None,
        work_dir: Optional[Union[str, Path]] = None,
        keep_artifacts: bool = False,
        host_lockdown: bool = True,
        persistence_backend: str = "fileout",
        llm_timeout: float = 300.0,
        llm_max_tokens: int = 2048,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        if mode not in ("scripted", "live"):
            raise ValueError("mode must be 'scripted' or 'live'")
        if persistence_backend not in ("fileout", "engram"):
            raise ValueError("persistence_backend must be 'fileout' or 'engram'")
        self.mode = mode
        self.endpoint = endpoint
        self.model = model
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_generations_override = max_generations
        self._own_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="actorii_selfevo_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.keep_artifacts = bool(keep_artifacts) or not self._own_work_dir
        self.host_lockdown = bool(host_lockdown)
        self.persistence_backend = persistence_backend
        self.llm_timeout = float(llm_timeout)
        self.llm_max_tokens = int(llm_max_tokens)
        self.on_event = on_event or (lambda msg: None)
        self._ns = f"selfevo_{os.getpid()}_{int(time.time())}"

    # -- environment ------------------------------------------------------

    def _env(self) -> Dict[str, str]:
        env = {
            "ACTORII_STATE_DIR": str(self.work_dir / "state"),
            "ACTORII_MEMORY_NAMESPACE": self._ns,
            "ACTORII_DISABLE_DEMO_TWINS": "1",
        }
        env["ACTORII_HOST_LOCKDOWN"] = "1" if self.host_lockdown else "0"
        return env

    def close(self) -> None:
        if self._own_work_dir and not self.keep_artifacts:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def __enter__(self) -> "SelfEvolutionRunner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- VM construction --------------------------------------------------

    def _boot_vm(self, spec: EvolutionSpec, tag: str, *, with_setup: bool = True, memory: bool = False) -> Tuple[Any, Any, Any]:
        from actorii.harness import bootstrap_agent_harness  # lazy: avoids import cycle with harness.benchmark

        fo_dir = self.work_dir / "fileout_runtime" / spec.spec_id / tag
        fo_dir.mkdir(parents=True, exist_ok=True)
        vm, harness, sandbox = bootstrap_agent_harness(
            model=self.model,
            endpoint=self.endpoint if self.mode == "live" else None,
            max_steps=spec.max_steps_per_gen,
            gas_limit=min(spec.gas_budget, 100_000),
            fileout_dir=str(fo_dir),
            enable_memory_graph=memory,
            enable_engrams=memory,
            memory_namespace=self._ns if memory else None,
        )
        if with_setup and spec.setup_source.strip():
            vm.eval_chunk(spec.setup_source)
        return vm, harness, sandbox

    def _score(self, vm: Any, spec: EvolutionSpec) -> Tuple[List[TestOutcome], List[TestOutcome]]:
        vis = run_tests(vm, spec.visible_tests, fixture=spec.fixture_source, opcode_cap=spec.test_opcode_cap)
        hold = run_tests(vm, spec.holdout_tests, fixture=spec.fixture_source, opcode_cap=spec.test_opcode_cap)
        return vis, hold

    # -- prompts ----------------------------------------------------------

    def build_prompt(
        self,
        spec: EvolutionSpec,
        generation: int,
        committed_chunks: Sequence[str],
        diagnostics: str,
    ) -> str:
        lines = [
            f"SELF-EVOLUTION TASK `{spec.spec_id}` ({spec.name}) - difficulty {spec.difficulty} - "
            f"generation {generation} of {self.max_gen(spec)} (at most {spec.max_steps_per_gen} steps).",
            "",
            f"CAPABILITY GOAL: {spec.goal}",
            "",
            "BEHAVIORAL CONTRACT (exact class / selector names are mandatory):",
            spec.spec_prompt.strip(),
            "",
            "VISIBLE ACCEPTANCE TESTS (each is an Actor 4.1 expression; run the same expressions yourself and "
            "compare with the expected value before you finish):",
        ]
        for i, t in enumerate(spec.visible_tests, 1):
            exp = f"oracle `{t.oracle}`" if t.oracle else repr(t.expected)
            lines.append(f"  {i}. {t.expr.strip()}   ==> expected {exp}")
        if spec.fixture_source.strip():
            lines += ["", "TEST FIXTURE (preloaded by the grader right before the tests; do not redefine it):",
                      spec.fixture_source.strip()]
        lines += [
            "",
            f"HIDDEN HOLDOUT TESTS: {len(spec.holdout_tests)} additional unseen edge-case checks also decide the "
            "verdict. Implement the general algorithm - lookup tables / hard-coded answers for the visible "
            "tests WILL fail the holdout.",
            "RULES: define the capability with `inherit(...)!!`, `now(Class)!!`, `Def selector(self, ...) { ... }!!` "
            "chunks. Never modify protected system classes or `AgentHarness`. The capability is then persisted "
            "to a `.act` file-out and reloaded into a brand-new VM, so it must be self-contained. "
            "ACTOR 4.1 SYNTAX REMINDER: instance variables are read/written ONLY as `self.ivar` and `self.ivar := expr;` "
            "(never `ivar(self) := ...`); call methods in prefix form `selector(receiver, arg)`; `now` is a reserved word "
            "(never use it as a parameter/variable name); declare locals after the parameters: `Def m(self, a | t i) { ... }!!`; "
            "use `if c then ... else ... endif;` and `loop while (c) begin ... endLoop;`; `/` is integer division on integers; "
            "an unknown selector silently returns nil, so a `got nil` test result usually means a misspelled or missing method. "
            "When the visible tests pass in your own verification, reply `COMPLETE: <summary>`.",
        ]
        if committed_chunks:
            lines += ["", "CURRENT COMMITTED SOURCE (already loaded in this VM; redefine a method with `now(Class)!! Def ...!!` to fix it):",
                      "\n".join(committed_chunks)]
        if diagnostics:
            lines += ["", diagnostics]
        return "\n".join(lines)

    def max_gen(self, spec: EvolutionSpec) -> int:
        return int(self.max_generations_override or spec.max_generations)

    def _diagnostics(
        self,
        generation: int,
        record: Dict[str, Any],
        vis: Sequence[TestOutcome],
        hold: Sequence[TestOutcome],
        violations: Sequence[str],
        step_errors: Sequence[str],
        lint_notes: Sequence[str],
    ) -> str:
        lines = [
            f"{DIAGNOSTICS_MARKER} (generation {generation}, outcome={record.get('outcome')}"
            + (f", rollback_reason={record['rollback_reason']}" if record.get("rollback_reason") else "")
            + "):",
            f"- visible tests: {_pass_count(vis)}/{len(vis)} passed",
        ]
        for o in vis:
            if not o.passed:
                lines.append(f"  FAIL {o.brief()}")
        failing_hold = [o for o in hold if not o.passed]
        if hold:
            tags = sorted({o.tag for o in failing_hold if o.tag})
            lines.append(
                f"- hidden holdout: {_pass_count(hold)}/{len(hold)} passed"
                + (f" (failing categories: {', '.join(tags)})" if tags else "")
                + " - details are withheld; generalize the algorithm"
            )
        for v in violations:
            lines.append(f"- INVARIANT VIOLATION: {v}")
        for e in list(step_errors)[-4:]:
            lines.append(f"- harness step error: {e[:300]}")
        for n in lint_notes[:4]:
            lines.append(f"- guardrail linter: {n[:300]}")
        if record.get("persistence_error"):
            lines.append(f"- persistence/reload gate FAILED: {record['persistence_error'][:400]}")
        return "\n".join(lines)

    # -- scripted solver --------------------------------------------------

    def _scripted_provider(self, spec: EvolutionSpec, generation: int) -> Callable[[str, str, dict], str]:
        gens = spec.scripted_generations
        if not gens:
            raise ValueError(f"spec {spec.spec_id} has no scripted_generations (needed for --mode scripted)")
        idx = min(generation - 1, len(gens) - 1)
        state = {"calls": 0}

        def provider(prompt: str, system_prompt: str, opts: dict) -> str:
            state["calls"] += 1
            use_idx = idx
            # Self-correction is only emitted when the harness actually fed back diagnostics.
            if generation > 1 and DIAGNOSTICS_MARKER not in prompt:
                use_idx = 0
            if state["calls"] == 1:
                return gens[use_idx].strip()
            return f"COMPLETE: scripted generation {generation} implemented {spec.spec_id}"

        return provider

    # -- one generation ---------------------------------------------------

    def _run_generation(
        self,
        spec: EvolutionSpec,
        generation: int,
        committed: List[str],
        diagnostics: str,
        baseline_fp: Dict[str, Dict[str, str]],
        best_passes: int,
    ) -> Tuple[Dict[str, Any], Any, List[str], Tuple[List[TestOutcome], List[TestOutcome]], str]:
        t0 = time.perf_counter()
        rec: Dict[str, Any] = {"generation": generation, "diagnostics_given": bool(diagnostics)}
        vm, harness, sandbox = self._boot_vm(spec, f"gen{generation}")
        for chunk in committed:
            vm.eval_chunk(chunk)  # replay of accepted capability (already validated once)

        prompt = self.build_prompt(spec, generation, committed, diagnostics)
        rec["prompt_chars"] = len(prompt)
        llm_stats: Dict[str, Any] = {}
        if self.mode == "scripted":
            harness.custom_provider = self._scripted_provider(spec, generation)
        else:
            harness.custom_provider = make_live_provider(
                harness,
                endpoint=self.endpoint,
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                timeout=self.llm_timeout,
                max_tokens=self.llm_max_tokens,
                stats=llm_stats,
            )
        vm.global_vars["Actor"]["EvoGoal"] = prompt
        try:
            vm.eval_chunk(f"runGoal(Actor[#Harness], Actor[#EvoGoal], {int(spec.max_steps_per_gen)})!!")
        except Exception as exc:
            rec["runner_error"] = f"{type(exc).__name__}: {exc}"

        step_log = list(harness._fields.get("stepLog") or [])
        steps = int(harness._fields.get("stepCount", 0) or 0)
        gas = int(harness._fields.get("totalGasUsed", 0) or 0)
        ok_actions = [str(e.get("action", "")) for e in step_log if isinstance(e, dict) and e.get("status") == "ok"]
        step_errors = [str(e.get("error", "")) for e in step_log if isinstance(e, dict) and e.get("status") == "error"]
        gen_chunks: List[str] = []
        for act in ok_actions:
            gen_chunks.extend(definitional_chunks(act))
        rec.update(
            steps=steps,
            step_errors=len(step_errors),
            definitional_chunks=len(gen_chunks),
            llm=llm_stats or None,
        )
        if llm_stats.get("llm_error"):
            rec["llm_error"] = llm_stats["llm_error"]

        lint_notes: List[str] = []
        for ch in gen_chunks:
            try:
                status = str(sandbox.lintChunk(ch))
                if status.startswith("ERROR:"):
                    lint_notes.append(status)
            except Exception:
                pass

        vis, hold = self._score(vm, spec)
        verify_ops = sum(o.opcodes for o in vis) + sum(o.opcodes for o in hold)
        total_gas = gas + verify_ops
        violations = check_invariants(vm, baseline_fp, spec)
        if total_gas > spec.gas_budget:
            violations.append(
                f"gas: harness {gas} + verification {verify_ops} = {total_gas} opcodes > gas_budget {spec.gas_budget} "
                "(capability is too expensive - use a cheaper algorithm)"
            )
        rec.update(
            visible_passed=_pass_count(vis),
            visible_total=len(vis),
            holdout_passed=_pass_count(hold),
            holdout_total=len(hold),
            harness_gas=gas,
            verify_opcodes=verify_ops,
            gas_used=total_gas,
            invariant_violations=violations,
        )
        passes = _pass_count(vis) + _pass_count(hold)

        reason = ""
        if violations:
            reason = "invariant_violation"
        elif not any(re.match(r"\s*(?:/\*.*?\*/\s*)*Def\s", c, re.DOTALL) for c in gen_chunks):
            reason = "no_definitions"
        elif passes < best_passes:
            reason = "regression"
        if reason:
            rec["outcome"] = "rolled_back"
            rec["rollback_reason"] = reason
        else:
            rec["outcome"] = "accepted"
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        return rec, vm, gen_chunks, (vis, hold), self._diagnostics(
            generation, rec, vis, hold, violations, step_errors, lint_notes
        )

    # -- persistence gate -------------------------------------------------

    def _persistence_gate(self, spec: EvolutionSpec, vm: Any) -> Tuple[bool, str, Optional[Tuple[List[TestOutcome], List[TestOutcome]]]]:
        from actorii.guardrails.fileout import ActorFileOutManager

        out_dir = self.work_dir / "fileout" / spec.spec_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        sources: Dict[str, str] = {}
        try:
            mgr = ActorFileOutManager(out_dir)
            for cname in spec.required_classes:
                path = mgr.file_out_class(vm, cname)
                sources[cname] = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            return False, f"file-out failed: {type(exc).__name__}: {exc}", None

        if self.persistence_backend == "engram":
            return self._engram_reload(spec, vm, sources)

        try:
            fresh, _, _ = self._boot_vm(spec, "reload", with_setup=False)
            if spec.reload_setup_source.strip():
                fresh.eval_chunk(spec.reload_setup_source)
            for cname, src in sources.items():
                fresh.eval_chunk(src)
            scores = self._score(fresh, spec)
        except Exception as exc:
            return False, f"reload into fresh ActorVM failed: {type(exc).__name__}: {exc}", None
        vis, hold = scores
        ok = _pass_count(vis) == len(vis) and _pass_count(hold) == len(hold)
        msg = "" if ok else (
            f"reloaded VM passed visible {_pass_count(vis)}/{len(vis)}, holdout {_pass_count(hold)}/{len(hold)}"
        )
        return ok, msg, scores

    def _engram_reload(self, spec: EvolutionSpec, vm: Any, sources: Dict[str, str]):
        """`:ActorEngram` persistence through `Actor[#Memory]` (requires Neo4j; isolated namespace)."""
        try:
            src_vm, _, _ = self._boot_vm(spec, "engram_src", memory=True)
            if spec.setup_source.strip():
                pass  # setup already applied by _boot_vm
            for ch in definitional_chunks("\n".join(sources.values())):
                src_vm.eval_chunk(ch)
            mem = src_vm.global_vars["Actor"]["Memory"]._target
            mem.clearEngrams()
            for cname, src in sources.items():
                lit = json.dumps(src)
                res = src_vm.eval_chunk(
                    f'saveEngram(Actor[#Memory], #{cname}, "selfevo {spec.spec_id}", {lit})!!'
                )
                if "ENGRAM_SAVED" not in str(res):
                    return False, f"saveEngram({cname}) returned {res!r}", None
            fresh, _, _ = self._boot_vm(spec, "engram_reload", with_setup=False, memory=True)
            if spec.reload_setup_source.strip():
                fresh.eval_chunk(spec.reload_setup_source)
            for cname in sources:
                res = fresh.eval_chunk(f"loadEngram(Actor[#Memory], #{cname})!!")
                if "ERROR" in str(res).upper()[:40]:
                    return False, f"loadEngram({cname}) returned {res!r}", None
            scores = self._score(fresh, spec)
            with contextlib.suppress(Exception):
                fresh.global_vars["Actor"]["Memory"]._target.clearEngrams()
        except Exception as exc:
            return False, f"engram persistence unavailable/failed: {type(exc).__name__}: {exc}", None
        vis, hold = scores
        ok = _pass_count(vis) == len(vis) and _pass_count(hold) == len(hold)
        return ok, "" if ok else "engram-reloaded VM failed tests", scores

    # -- spec driver ------------------------------------------------------

    def run_spec(self, spec: EvolutionSpec) -> Dict[str, Any]:
        with _scoped_env(self._env()):
            return self._run_spec(spec)

    def _run_spec(self, spec: EvolutionSpec) -> Dict[str, Any]:
        t_start = time.perf_counter()
        gmax = self.max_gen(spec)
        result: Dict[str, Any] = {
            "spec_id": spec.spec_id,
            "name": spec.name,
            "difficulty": spec.difficulty,
            "mode": self.mode,
            "model": self.model if self.mode == "live" else "scripted-reference-solver",
            "status": "error",
            "passed": False,
            "pass_at_1": False,
            "pass_at_g": False,
            "max_generations": gmax,
            "generations_used": 0,
            "steps_used": 0,
            "gas_used": 0,
            "visible_pass_rate": 0.0,
            "holdout_pass_rate": 0.0,
            "persistence_required": spec.require_persistence,
            "persistence_verified": False,
            "baseline": {},
            "reward_hacking_suspected": False,
            "rollbacks": 0,
            "latency_s": 0.0,
            "final_source": "",
            "generations": [],
            "error": "",
        }
        try:
            # ---- Phase 0: pre-flight baseline oracle --------------------
            base_vm, _, _ = self._boot_vm(spec, "baseline")
            baseline_fp = vm_fingerprint(base_vm)
            bvis, bhold = self._score(base_vm, spec)
            result["baseline"] = {
                "visible_passed": _pass_count(bvis),
                "visible_total": len(bvis),
                "holdout_passed": _pass_count(bhold),
                "holdout_total": len(bhold),
            }
            if _pass_count(bvis) == len(bvis) and _pass_count(bhold) == len(bhold):
                result["status"] = "invalid_baseline"
                result["error"] = "capability tests already pass before evolution - spec cannot measure evolution"
                return result
            best_passes = _pass_count(bvis) + _pass_count(bhold)
            self.on_event(f"[{spec.spec_id}] baseline ok: {best_passes}/{len(bvis) + len(bhold)} tests pass pre-evolution")

            # ---- Phase 1..G ---------------------------------------------
            committed: List[str] = []
            diagnostics = ""
            last_scores: Tuple[List[TestOutcome], List[TestOutcome]] = (bvis, bhold)
            for g in range(1, gmax + 1):
                rec, vm, gen_chunks, scores, diag = self._run_generation(
                    spec, g, committed, diagnostics, baseline_fp, best_passes
                )
                vis, hold = scores
                last_scores = scores
                result["generations_used"] = g
                result["steps_used"] += int(rec.get("steps", 0))
                result["gas_used"] += int(rec.get("gas_used", 0))
                if rec["outcome"] == "accepted":
                    committed = committed + gen_chunks
                    best_passes = max(best_passes, _pass_count(vis) + _pass_count(hold))
                else:
                    result["rollbacks"] += 1
                all_pass = (
                    rec["outcome"] == "accepted"
                    and _pass_count(vis) == len(vis)
                    and _pass_count(hold) == len(hold)
                )
                rec["all_tests_passed"] = all_pass
                if _pass_count(vis) == len(vis) and _pass_count(hold) < len(hold):
                    result["reward_hacking_suspected"] = True
                    rec["visible_overfit"] = True
                if all_pass and spec.require_persistence:
                    ok, msg, rscores = self._persistence_gate(spec, vm)
                    rec["persistence_verified"] = ok
                    if not ok:
                        rec["persistence_error"] = msg
                        rec["outcome"] = "rolled_back"
                        rec["rollback_reason"] = "persistence_failed"
                        result["rollbacks"] += 1
                        committed = committed[: len(committed) - len(gen_chunks)]
                        all_pass = False
                        diag = self._diagnostics(g, rec, vis, hold, [], [], [])
                    else:
                        result["persistence_verified"] = True
                rec["failed_visible"] = [o.brief() for o in vis if not o.passed]
                rec["failed_holdout_tags"] = sorted({o.tag or "holdout" for o in hold if not o.passed})
                result["generations"].append(rec)
                self.on_event(
                    f"[{spec.spec_id}] gen {g}: {rec['outcome']} visible {rec['visible_passed']}/{rec['visible_total']} "
                    f"holdout {rec['holdout_passed']}/{rec['holdout_total']} steps={rec['steps']} gas={rec['gas_used']}"
                )
                diagnostics = diag
                if all_pass:
                    result["passed"] = True
                    result["pass_at_g"] = True
                    result["pass_at_1"] = g == 1
                    break
            result["visible_pass_rate"] = _rate(last_scores[0]) if result["generations"] else 0.0
            result["holdout_pass_rate"] = _rate(last_scores[1]) if result["generations"] else 0.0
            # The final committed capability's scores (not a rolled-back generation's):
            accepted = [r for r in result["generations"] if r["outcome"] == "accepted"]
            if accepted:
                last = accepted[-1]
                result["visible_pass_rate"] = round(last["visible_passed"] / max(1, last["visible_total"]), 4)
                result["holdout_pass_rate"] = round(last["holdout_passed"] / max(1, last["holdout_total"]), 4)
            result["final_source"] = "\n\n".join(committed)
            result["status"] = "passed" if result["passed"] else "failed"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["latency_s"] = round(time.perf_counter() - t_start, 3)
        return result

    def run_suite(self, specs: Sequence[EvolutionSpec]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = [self.run_spec(s) for s in specs]
        report = {
            "benchmark": "actorii-bilby-self-evolution",
            "mode": self.mode,
            "model": self.model if self.mode == "live" else "scripted-reference-solver",
            "endpoint": self.endpoint if self.mode == "live" else None,
            "temperature": self.temperature,
            "seed": self.seed,
            "persistence_backend": self.persistence_backend,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "specs": results,
            "summary": summarize_suite(results),
        }
        report["summary"]["total_latency_s"] = round(time.perf_counter() - t0, 3)
        return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_suite(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    p1 = sum(1 for r in results if r.get("pass_at_1"))
    pg = sum(1 for r in results if r.get("pass_at_g"))
    used = [r.get("generations_used", 0) for r in results]
    hold = [r.get("holdout_pass_rate", 0.0) for r in results]
    return {
        "specs": n,
        "pass_at_1": round(p1 / n, 4) if n else 0.0,
        "pass_at_1_count": p1,
        "pass_at_g": round(pg / n, 4) if n else 0.0,
        "pass_at_g_count": pg,
        "mean_generations_used": round(sum(used) / n, 3) if n else 0.0,
        "steps_used": sum(r.get("steps_used", 0) for r in results),
        "gas_used": sum(r.get("gas_used", 0) for r in results),
        "mean_holdout_pass_rate": round(sum(hold) / n, 4) if n else 0.0,
        "persistence_verified": sum(1 for r in results if r.get("persistence_verified")),
        "reward_hacking_suspected": sum(1 for r in results if r.get("reward_hacking_suspected")),
        "rollbacks": sum(r.get("rollbacks", 0) for r in results),
        "errors": sum(1 for r in results if r.get("status") in ("error", "invalid_baseline")),
    }


def summarize_result(r: Dict[str, Any]) -> str:
    return (
        f"[{'PASS' if r.get('passed') else 'FAIL'}] #{r.get('spec_id')} ({r.get('difficulty')}, {r.get('mode')}): "
        f"pass@1={r.get('pass_at_1')} pass@G={r.get('pass_at_g')} gens={r.get('generations_used')}/{r.get('max_generations')} "
        f"steps={r.get('steps_used')} gas={r.get('gas_used')} holdout={r.get('holdout_pass_rate')} "
        f"persist={r.get('persistence_verified')} rollbacks={r.get('rollbacks')} {r.get('latency_s')}s"
        + (f" ERROR={r.get('error')}" if r.get("error") else "")
    )


def render_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# ActorII Bilby Self-Evolution Benchmark Results",
        "",
        f"- mode: `{report['mode']}` | model: `{report['model']}` | temperature: {report['temperature']} | seed: {report['seed']}",
        f"- timestamp: {report['timestamp']} | persistence backend: `{report.get('persistence_backend')}`",
        "",
        "| spec | tier | status | pass@1 | pass@G | gens | steps | gas | holdout | persist | rollbacks | latency |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["specs"]:
        lines.append(
            f"| `{r['spec_id']}` | {r['difficulty']} | {r['status']} | {'Y' if r['pass_at_1'] else 'N'} | "
            f"{'Y' if r['pass_at_g'] else 'N'} | {r['generations_used']}/{r['max_generations']} | {r['steps_used']} | "
            f"{r['gas_used']} | {r['holdout_pass_rate']:.0%} | "
            f"{'Y' if r['persistence_verified'] else ('n/a' if not r['persistence_required'] else 'N')} | "
            f"{r['rollbacks']} | {r['latency_s']}s |"
        )
    lines += [
        "",
        f"**Summary:** pass@1 = {s['pass_at_1_count']}/{s['specs']} ({s['pass_at_1']:.0%}), "
        f"pass@G = {s['pass_at_g_count']}/{s['specs']} ({s['pass_at_g']:.0%}), "
        f"mean generations = {s['mean_generations_used']}, mean holdout = {s['mean_holdout_pass_rate']:.0%}, "
        f"persistence verified = {s['persistence_verified']}, rollbacks = {s['rollbacks']}, "
        f"reward-hacking flags = {s['reward_hacking_suspected']}, errors = {s['errors']}, "
        f"steps = {s['steps_used']}, gas = {s['gas_used']}, total latency = {s.get('total_latency_s', 0)}s.",
        "",
        "## Per-generation detail",
    ]
    for r in report["specs"]:
        lines += ["", f"### `{r['spec_id']}` - {r['name']}"]
        b = r.get("baseline") or {}
        if b:
            lines.append(
                f"- Phase 0 baseline: visible {b['visible_passed']}/{b['visible_total']}, "
                f"holdout {b['holdout_passed']}/{b['holdout_total']} (must not be fully passing)"
            )
        if r.get("error"):
            lines.append(f"- ERROR: {r['error']}")
        for g in r["generations"]:
            extra = f", rollback={g['rollback_reason']}" if g.get("rollback_reason") else ""
            if g.get("llm_error"):
                extra += f", llm_error={g['llm_error']}"
            lines.append(
                f"- gen {g['generation']}: **{g['outcome']}**{extra} - visible {g['visible_passed']}/{g['visible_total']}, "
                f"holdout {g['holdout_passed']}/{g['holdout_total']}, steps {g['steps']}, gas {g['gas_used']}, "
                f"{g['latency_s']}s, diagnostics_given={g['diagnostics_given']}"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Curated built-in suite
# ---------------------------------------------------------------------------

_SORT_HELPER = """
Def sortedValues(self | out i j n tmp) {
    out := new(OrderedCollection);
    n := size(self.vals);
    i := 0;
    loop while (i < n) begin
        add(out, at(self.vals, i));
        i := i + 1;
    endLoop;
    i := 1;
    loop while (i < n) begin
        j := i;
        loop while (j > 0) begin
            if at(out, j - 1) > at(out, j) then
                tmp := at(out, j);
                atPut(out, j, at(out, j - 1));
                atPut(out, j - 1, tmp);
                j := j - 1;
            else
                j := 0;
            endif;
        endLoop;
        i := i + 1;
    endLoop;
    ^out;
}!!
"""

# ---- Tier 1: sensor_window_median -----------------------------------------------------------
_MEDIAN_SETUP = """
inherit(Object, #SensorWindow, #(#vals), 1, nil)!!
now(SensorWindow)!!
Def init(self) { self.vals := new(OrderedCollection); ^self; }!!
Def record(self, v) { add(self.vals, v); ^self; }!!
Def readingCount(self) { ^size(self.vals); }!!
Def valueAt(self, i) { ^at(self.vals, i); }!!
"""
_MEDIAN_GEN1 = """
now(SensorWindow)!!
Def median(self | n mid) {
    n := size(self.vals);
    if n == 0 then ^nil; endif;
    mid := n / 2;
    if (n mod 2) == 1 then ^at(self.vals, mid); endif;
    ^((at(self.vals, mid - 1) + at(self.vals, mid)) / 2);
}!!
"""
_MEDIAN_GEN2 = (
    "now(SensorWindow)!!\n"
    + _SORT_HELPER
    + """
Def median(self | n mid s) {
    n := size(self.vals);
    if n == 0 then ^nil; endif;
    s := sortedValues(self);
    mid := n / 2;
    if (n mod 2) == 1 then ^at(s, mid); endif;
    ^((at(s, mid - 1) + at(s, mid)) / 2);
}!!
"""
)


def _win(name: str, values: Sequence[int], tail: str) -> str:
    recs = " ".join(f"record(Actor[#{name}], {v})!!" for v in values)
    return f"Actor[#{name}] := init(new(SensorWindow))!! {recs} {tail}!!".replace("!!!!", "!!")


_MEDIAN_SPEC = EvolutionSpec(
    spec_id="sensor_window_median",
    name="Hot-add `median` to a live SensorWindow class",
    difficulty=TIER_METHOD,
    goal="Teach the live `SensorWindow` class a robust integer `median` method without breaking its existing behaviour.",
    spec_prompt=(
        "`SensorWindow` (ivar `#vals`, an OrderedCollection in insertion order) is already loaded with "
        "`init(new(SensorWindow))`, `record(w, v)`, `readingCount(w)` and `valueAt(w, i)`.\n"
        "Add `Def median(self)`: the median of the recorded values. Empty window -> `nil`. Odd count -> the middle "
        "value of the SORTED values. Even count -> integer average of the two middle sorted values "
        "(`(a + b) / 2`, Actor integer division). `median` MUST NOT reorder or modify the stored insertion-order "
        "values (`valueAt` must still return readings in the order they were recorded)."
    ),
    visible_tests=[
        EvolutionTest(_win("w", [9, 1, 5], "median(Actor[#w])"), 5, "visible"),
        EvolutionTest(_win("w", [10, 2, 8, 4], "median(Actor[#w])"), 6, "visible"),
        EvolutionTest("Actor[#w] := init(new(SensorWindow))!! median(Actor[#w])!!", None, "visible"),
    ],
    holdout_tests=[
        EvolutionTest(_win("h", [7], "median(Actor[#h])"), 7, "single"),
        EvolutionTest(_win("h", [3, 3, 3, 3, 3], "median(Actor[#h])"), 3, "duplicates"),
        EvolutionTest(_win("h", [50, 10, 40, 20, 30, 60, 70], "median(Actor[#h])"), 40, "odd-unsorted"),
        EvolutionTest(_win("h", [9, 1, 5], "median(Actor[#h])!! valueAt(Actor[#h], 0)"), 9, "no-mutation"),
        EvolutionTest(_win("h", [9, 1, 5], "median(Actor[#h])!! median(Actor[#h])"), 5, "idempotent"),
        EvolutionTest(_win("h", [-4, -8], "median(Actor[#h])"), -6, "negatives"),
        EvolutionTest(
            _win("h", [1, 2, 3], "record(Actor[#h], 100)!! median(Actor[#h])"), 2, "grows-after-record"
        ),
        EvolutionTest(
            "Actor[#h] := init(new(SensorWindow))!! Actor[#i] := 20!! "
            "loop while (Actor[#i] > 0) begin record(Actor[#h], Actor[#i]); Actor[#i] := Actor[#i] - 1; endLoop!! "
            "median(Actor[#h])!!",
            oracle="result == 10",
            tag="bulk-descending",
        ),
    ],
    required_classes=["SensorWindow"],
    setup_source=_MEDIAN_SETUP,
    scripted_generations=[_MEDIAN_GEN1, _MEDIAN_GEN2],
)

# ---- Tier 2: ring_buffer_telemetry ----------------------------------------------------------
_RING_BASE = """
inherit(Object, #RingTelemetry, #(#vals #cap), 2, nil)!!
now(RingTelemetry)!!
Def initCapacity(self, c) { self.cap := c; self.vals := new(OrderedCollection); ^self; }!!
Def count(self) { ^size(self.vals); }!!
Def oldest(self) {
    if size(self.vals) == 0 then ^nil; endif;
    ^at(self.vals, 0);
}!!
Def windowSum(self | i t) {
    t := 0;
    i := 0;
    loop while (i < size(self.vals)) begin
        t := t + at(self.vals, i);
        i := i + 1;
    endLoop;
    ^t;
}!!
Def windowMean(self) {
    if size(self.vals) == 0 then ^nil; endif;
    ^(windowSum(self) / size(self.vals));
}!!
Def windowMax(self | i b) {
    if size(self.vals) == 0 then ^nil; endif;
    b := at(self.vals, 0);
    i := 1;
    loop while (i < size(self.vals)) begin
        if at(self.vals, i) > b then b := at(self.vals, i); endif;
        i := i + 1;
    endLoop;
    ^b;
}!!
Def windowMin(self | i b) {
    if size(self.vals) == 0 then ^nil; endif;
    b := at(self.vals, 0);
    i := 1;
    loop while (i < size(self.vals)) begin
        if at(self.vals, i) < b then b := at(self.vals, i); endif;
        i := i + 1;
    endLoop;
    ^b;
}!!
"""
_RING_GEN1 = (
    _RING_BASE
    + """
Def record(self, v) {
    add(self.vals, v);
    ^size(self.vals);
}!!
Def percentile(self, p) {
    if size(self.vals) == 0 then ^nil; endif;
    ^windowMax(self);
}!!
Def isAnomaly(self, v, k) { ^false; }!!
"""
)
_RING_GEN2 = (
    _RING_BASE
    + """
Def record(self, v) {
    add(self.vals, v);
    if size(self.vals) > self.cap then removeFirst(self.vals); endif;
    ^size(self.vals);
}!!
"""
    + _SORT_HELPER.replace("Def sortedValues", "Def sortedValues")
    + """
Def percentile(self, p | s n rank) {
    n := size(self.vals);
    if n == 0 then ^nil; endif;
    s := sortedValues(self);
    rank := ((p * n) + 99) / 100;
    if rank < 1 then rank := 1; endif;
    if rank > n then rank := n; endif;
    ^at(s, rank - 1);
}!!
Def meanAbsDev(self | m i t d) {
    m := windowMean(self);
    t := 0;
    i := 0;
    loop while (i < size(self.vals)) begin
        d := at(self.vals, i) - m;
        if d < 0 then d := 0 - d; endif;
        t := t + d;
        i := i + 1;
    endLoop;
    ^(t / size(self.vals));
}!!
Def isAnomaly(self, v, k | m mad d) {
    if size(self.vals) < 3 then ^false; endif;
    m := windowMean(self);
    mad := meanAbsDev(self);
    d := v - m;
    if d < 0 then d := 0 - d; endif;
    if mad == 0 then ^(d > 0); endif;
    ^(d > (k * mad));
}!!
"""
)


def _ring(cap: int, values: Sequence[int], tail: str, name: str = "r") -> str:
    recs = " ".join(f"record(Actor[#{name}], {v})!!" for v in values)
    return f"Actor[#{name}] := initCapacity(new(RingTelemetry), {cap})!! {recs} {tail}!!".replace("!!!!", "!!")


_RING_SPEC = EvolutionSpec(
    spec_id="ring_buffer_telemetry",
    name="Bounded sliding-window percentile / anomaly tracker",
    difficulty=TIER_CLASS,
    goal="Create a new `RingTelemetry` class: a bounded sliding window of integer samples with statistics and anomaly detection.",
    spec_prompt=(
        "Define `inherit(Object, #RingTelemetry, #(...), n, nil)` with these selectors (all integer math, "
        "Actor integer division):\n"
        "- `initCapacity(self, cap)` -> self; creates an empty window that never holds more than `cap` samples.\n"
        "- `record(self, v)` -> number of samples now in the window. When full, evict the OLDEST sample.\n"
        "- `count(self)`, `oldest(self)` (nil if empty), `windowSum(self)`, `windowMean(self)` (integer mean, nil if empty), "
        "`windowMax(self)`, `windowMin(self)` (nil if empty).\n"
        "- `percentile(self, p)`: nearest-rank percentile, 0 <= p <= 100, over the window sorted ascending: "
        "`rank = (p * n + 99) / 100` (integer division), clamped to 1..n; returns the value at that 1-based rank; nil if empty.\n"
        "- `isAnomaly(self, v, k)`: false when the window holds fewer than 3 samples. Otherwise let `mean = windowMean`, "
        "`mad` = integer mean of `|x - mean|` over the window; if `mad == 0` the result is `|v - mean| > 0`, else "
        "`|v - mean| > k * mad`."
    ),
    visible_tests=[
        EvolutionTest(_ring(3, [1, 2, 3, 4, 5], "windowSum(Actor[#r])"), 12, "visible"),
        EvolutionTest(_ring(3, [1, 2, 3, 4, 5], "oldest(Actor[#r])"), 3, "visible"),
        EvolutionTest(_ring(5, [10, 20, 30, 40, 50], "percentile(Actor[#r], 50)"), 30, "visible"),
        EvolutionTest("Actor[#r] := initCapacity(new(RingTelemetry), 4)!! windowMean(Actor[#r])!!", None, "visible"),
    ],
    holdout_tests=[
        EvolutionTest(_ring(4, list(range(1, 11)), "windowMax(Actor[#h])", "h"), 10, "wraparound"),
        EvolutionTest(_ring(4, list(range(1, 11)), "windowMin(Actor[#h])", "h"), 7, "wraparound"),
        EvolutionTest(_ring(4, list(range(1, 11)), "windowMean(Actor[#h])", "h"), 8, "wraparound"),
        EvolutionTest(_ring(1, [5, 9], "oldest(Actor[#h])", "h"), 9, "cap-1"),
        EvolutionTest(_ring(1, [5, 9], "count(Actor[#h])", "h"), 1, "cap-1"),
        EvolutionTest(_ring(4, [40, 10, 30, 20], "percentile(Actor[#h], 0)", "h"), 10, "pct-clamp-low"),
        EvolutionTest(_ring(4, [40, 10, 30, 20], "percentile(Actor[#h], 100)", "h"), 40, "pct-clamp-high"),
        EvolutionTest(_ring(4, [40, 10, 30, 20], "percentile(Actor[#h], 25)", "h"), 10, "pct-rank"),
        EvolutionTest(_ring(4, [40, 10, 30, 20], "percentile(Actor[#h], 75)", "h"), 30, "pct-rank"),
        EvolutionTest(_ring(6, [10] * 6, "isAnomaly(Actor[#h], 50, 3)", "h"), True, "anomaly-flat"),
        EvolutionTest(_ring(6, [10] * 6, "isAnomaly(Actor[#h], 10, 3)", "h"), False, "anomaly-flat"),
        EvolutionTest(_ring(6, [10, 12, 8, 10, 12, 8], "isAnomaly(Actor[#h], 14, 3)", "h"), True, "anomaly-spread"),
        EvolutionTest(_ring(6, [10, 12, 8, 10, 12, 8], "isAnomaly(Actor[#h], 12, 3)", "h"), False, "anomaly-spread"),
        EvolutionTest(_ring(6, [1, 2], "isAnomaly(Actor[#h], 1000, 1)", "h"), False, "anomaly-min-samples"),
        EvolutionTest(
            "Actor[#h] := initCapacity(new(RingTelemetry), 50)!! Actor[#i] := 0!! "
            "loop while (Actor[#i] < 200) begin record(Actor[#h], Actor[#i]); Actor[#i] := Actor[#i] + 1; endLoop!! "
            "windowSum(Actor[#h])!!",
            8725,
            "bulk-eviction",
        ),
    ],
    required_classes=["RingTelemetry"],
    scripted_generations=[_RING_GEN1, _RING_GEN2],
)

# ---- Tier 2: circuit_breaker_policy ---------------------------------------------------------
_CB_BASE = """
inherit(Object, #CircuitBreaker, #(#state #failures #threshold #cooldown #openedAt #probing #trips), 7, nil)!!
now(CircuitBreaker)!!
Def initBreaker(self, th, cd) {
    self.threshold := th;
    self.cooldown := cd;
    self.state := #closed;
    self.failures := 0;
    self.openedAt := 0;
    self.probing := false;
    self.trips := 0;
    ^self;
}!!
Def state(self) { ^self.state; }!!
Def failureCount(self) { ^self.failures; }!!
Def tripCount(self) { ^self.trips; }!!
"""
_CB_GEN1 = (
    _CB_BASE
    + """
Def allow(self, tick) {
    if self.state == #open then ^false; endif;
    ^true;
}!!
Def recordSuccess(self) {
    self.failures := 0;
    self.state := #closed;
    ^self.state;
}!!
Def recordFailure(self, tick) {
    self.failures := self.failures + 1;
    if self.failures >= self.threshold then
        self.state := #open;
        self.openedAt := tick;
        self.trips := self.trips + 1;
    endif;
    ^self.state;
}!!
"""
)
_CB_GEN2 = (
    _CB_BASE
    + """
Def allow(self, tick) {
    if self.state == #closed then ^true; endif;
    if self.state == #open then
        if (tick - self.openedAt) >= self.cooldown then
            self.state := #halfOpen;
            self.probing := true;
            ^true;
        endif;
        ^false;
    endif;
    if self.probing == true then ^false; endif;
    self.probing := true;
    ^true;
}!!
Def recordSuccess(self) {
    if self.state == #halfOpen then self.state := #closed; endif;
    if self.state == #closed then self.failures := 0; endif;
    self.probing := false;
    ^self.state;
}!!
Def recordFailure(self, tick) {
    if self.state == #halfOpen then
        self.state := #open;
        self.openedAt := tick;
        self.probing := false;
        self.trips := self.trips + 1;
        ^self.state;
    endif;
    if self.state == #closed then
        self.failures := self.failures + 1;
        if self.failures >= self.threshold then
            self.state := #open;
            self.openedAt := tick;
            self.trips := self.trips + 1;
        endif;
    endif;
    ^self.state;
}!!
"""
)


def _cb(th: int, cd: int, steps: str, tail: str) -> str:
    return f"Actor[#b] := initBreaker(new(CircuitBreaker), {th}, {cd})!! {steps} {tail}!!".replace("!!!!", "!!")


_TRIP3 = "recordFailure(Actor[#b], 1)!! recordFailure(Actor[#b], 2)!! recordFailure(Actor[#b], 3)!!"
_CB_SPEC = EvolutionSpec(
    spec_id="circuit_breaker_policy",
    name="Stateful closed/open/half-open retry guard",
    difficulty=TIER_CLASS,
    goal="Create a new `CircuitBreaker` class implementing a deterministic closed/open/half-open retry guard with an explicit clock.",
    spec_prompt=(
        "Define `inherit(Object, #CircuitBreaker, #(...), n, nil)`. Time is passed explicitly as an integer tick argument named `tick` "
        "(never read a real clock; NOTE `now` is a reserved word in Actor - do not use it as an identifier). States are the symbols `#closed`, `#open`, `#halfOpen`.\n"
        "- `initBreaker(self, failThreshold, cooldownTicks)` -> self, starts `#closed` with 0 failures and 0 trips.\n"
        "- `state(self)`, `failureCount(self)` (consecutive failures while closed), `tripCount(self)` (times it opened).\n"
        "- `recordFailure(self, tick)` -> new state. Closed: failures + 1; when failures >= threshold it opens "
        "(remembers `tick`, trips + 1). HalfOpen: the probe failed -> re-open with the NEW `tick` (trips + 1). "
        "Open: no change.\n"
        "- `recordSuccess(self)` -> new state. HalfOpen: close the breaker. Closed: reset failures to 0. "
        "A late success while `#open` must NOT close the breaker. Always clears an in-flight probe.\n"
        "- `allow(self, tick)` -> true/false. Closed: always true. Open: false until `tick - openedAt >= cooldownTicks`, "
        "then the breaker becomes `#halfOpen` and exactly ONE probe call is allowed (true); while that probe is "
        "outstanding further `allow` calls return false."
    ),
    visible_tests=[
        EvolutionTest(_cb(3, 10, "recordFailure(Actor[#b], 1)!! recordFailure(Actor[#b], 2)!!", "state(Actor[#b])"), "#closed", "visible"),
        EvolutionTest(_cb(3, 10, _TRIP3, "state(Actor[#b])"), "#open", "visible"),
        EvolutionTest(_cb(3, 10, _TRIP3, "allow(Actor[#b], 5)"), False, "visible"),
        EvolutionTest(_cb(3, 10, _TRIP3 + " allow(Actor[#b], 13)!!", "state(Actor[#b])"), "#halfOpen", "visible"),
        EvolutionTest(_cb(3, 10, _TRIP3 + " allow(Actor[#b], 13)!! recordSuccess(Actor[#b])!!", "state(Actor[#b])"), "#closed", "visible"),
    ],
    holdout_tests=[
        EvolutionTest(
            _cb(3, 10, _TRIP3 + " allow(Actor[#b], 13)!! recordFailure(Actor[#b], 14)!!", "allow(Actor[#b], 20)"),
            False, "halfopen-reopen-new-clock"),
        EvolutionTest(
            _cb(3, 10, _TRIP3 + " allow(Actor[#b], 13)!! recordFailure(Actor[#b], 14)!!", "allow(Actor[#b], 24)"),
            True, "cooldown-boundary"),
        EvolutionTest(_cb(3, 10, _TRIP3 + " allow(Actor[#b], 13)!!", "allow(Actor[#b], 13)"), False, "single-probe"),
        EvolutionTest(
            _cb(3, 10, "recordFailure(Actor[#b], 1)!! recordFailure(Actor[#b], 2)!! recordSuccess(Actor[#b])!! "
                       "recordFailure(Actor[#b], 3)!! recordFailure(Actor[#b], 4)!!", "state(Actor[#b])"),
            "#closed", "success-resets-failures"),
        EvolutionTest(
            _cb(3, 10, "recordFailure(Actor[#b], 1)!! recordFailure(Actor[#b], 2)!! recordSuccess(Actor[#b])!! "
                       "recordFailure(Actor[#b], 3)!! recordFailure(Actor[#b], 4)!!", "failureCount(Actor[#b])"),
            2, "success-resets-failures"),
        EvolutionTest(
            _cb(2, 5, "recordFailure(Actor[#b], 1)!! recordFailure(Actor[#b], 2)!! allow(Actor[#b], 7)!! "
                      "recordSuccess(Actor[#b])!! recordFailure(Actor[#b], 8)!! recordFailure(Actor[#b], 9)!!",
                "tripCount(Actor[#b])"),
            2, "trip-count"),
        EvolutionTest(_cb(3, 10, "recordFailure(Actor[#b], 1)!!", "allow(Actor[#b], 2)"), True, "closed-allows"),
        EvolutionTest(_cb(3, 10, _TRIP3 + " recordSuccess(Actor[#b])!!", "state(Actor[#b])"), "#open", "late-success-open"),
        EvolutionTest(_cb(1, 4, "recordFailure(Actor[#b], 0)!!", "allow(Actor[#b], 3)"), False, "just-before-cooldown"),
        EvolutionTest(_cb(1, 4, "recordFailure(Actor[#b], 0)!! allow(Actor[#b], 4)!!", "state(Actor[#b])"), "#halfOpen", "cooldown-boundary"),
    ],
    required_classes=["CircuitBreaker"],
    scripted_generations=[_CB_GEN1, _CB_GEN2],
)

# ---- Tier 2: dag_task_scheduler -------------------------------------------------------------
_DAG_BASE = """
inherit(Object, #TaskScheduler, #(#ids #deps), 2, nil)!!
now(TaskScheduler)!!
Def initScheduler(self) { self.ids := new(OrderedCollection); self.deps := new(Dictionary, 16); ^self; }!!
Def inList(self, lst, id | i found) {
    found := false;
    i := 0;
    loop while (i < size(lst)) begin
        if at(lst, i) == id then
            found := true;
            i := size(lst);
        else
            i := i + 1;
        endif;
    endLoop;
    ^found;
}!!
Def hasTask(self, id) { ^inList(self, self.ids, id); }!!
Def addTask(self, id | known) {
    known := hasTask(self, id);
    if known == false then
        add(self.ids, id);
        atPut(self.deps, id, new(OrderedCollection));
    endif;
    ^self;
}!!
Def addDependency(self, task, pre) {
    addTask(self, task);
    addTask(self, pre);
    add(at(self.deps, task), pre);
    ^self;
}!!
Def taskCount(self) { ^size(self.ids); }!!
"""
_DAG_GEN1 = (
    _DAG_BASE
    + """
Def order(self) { ^self.ids; }!!
Def hasCycle(self) { ^false; }!!
Def depthOf(self, id) { ^0; }!!
"""
)
_DAG_GEN2 = (
    _DAG_BASE
    + """
Def prereqsDone(self, id, done | pres i ok seen) {
    pres := at(self.deps, id);
    ok := true;
    i := 0;
    loop while (i < size(pres)) begin
        seen := inList(self, done, at(pres, i));
        if seen == false then ok := false; endif;
        i := i + 1;
    endLoop;
    ^ok;
}!!
Def nextReady(self, done | i id best bestPos finished ready) {
    bestPos := 0 - 1;
    best := 0;
    i := 0;
    loop while (i < size(self.ids)) begin
        id := at(self.ids, i);
        finished := inList(self, done, id);
        if finished == false then
            ready := prereqsDone(self, id, done);
            if ready == true then
                if bestPos < 0 then
                    best := id;
                    bestPos := i;
                else
                    if id < best then
                        best := id;
                        bestPos := i;
                    endif;
                endif;
            endif;
        endif;
        i := i + 1;
    endLoop;
    ^bestPos;
}!!
Def order(self | done pos total stuck) {
    done := new(OrderedCollection);
    stuck := false;
    total := size(self.ids);
    loop while (size(done) < total) begin
        pos := nextReady(self, done);
        if pos < 0 then
            stuck := true;
            total := 0;
        else
            add(done, at(self.ids, pos));
        endif;
    endLoop;
    if stuck == true then ^nil; endif;
    ^done;
}!!
Def hasCycle(self | o) {
    o := order(self);
    ^(o == nil);
}!!
Def depthRec(self, id | pres i d best) {
    pres := at(self.deps, id);
    best := 0;
    i := 0;
    loop while (i < size(pres)) begin
        d := depthRec(self, at(pres, i)) + 1;
        if d > best then best := d; endif;
        i := i + 1;
    endLoop;
    ^best;
}!!
Def depthOf(self, id | known cyc) {
    known := hasTask(self, id);
    if known == false then ^nil; endif;
    cyc := hasCycle(self);
    if cyc == true then ^nil; endif;
    ^depthRec(self, id);
}!!
"""
)


def _dag(edges: Sequence[Tuple[int, int]], tail: str, extra_tasks: Sequence[int] = ()) -> str:
    pre = " ".join(f"addTask(Actor[#s], {t})!!" for t in extra_tasks)
    deps = " ".join(f"addDependency(Actor[#s], {a}, {b})!!" for a, b in edges)
    return f"Actor[#s] := initScheduler(new(TaskScheduler))!! {pre} {deps} {tail}!!".replace("!!!!", "!!")


_DAG_SPEC = EvolutionSpec(
    spec_id="dag_task_scheduler",
    name="Topological dependency resolver with cycle detection",
    difficulty=TIER_CLASS,
    goal="Create a new `TaskScheduler` class that resolves integer task dependencies into a deterministic execution order.",
    spec_prompt=(
        "Define `inherit(Object, #TaskScheduler, #(...), n, nil)`. Tasks are integer ids.\n"
        "- `initScheduler(self)` -> self (empty).\n"
        "- `addTask(self, id)` -> self; idempotent (adding twice does not duplicate).\n"
        "- `addDependency(self, task, pre)` -> self; `task` can only run AFTER `pre`; auto-registers both ids.\n"
        "- `taskCount(self)`.\n"
        "- `order(self)` -> an OrderedCollection of ALL task ids in a valid execution order; among tasks that are "
        "simultaneously ready always run the SMALLEST id first (Kahn's algorithm with smallest-id tie-break). "
        "Returns `nil` when a dependency cycle exists (including a task depending on itself).\n"
        "- `hasCycle(self)` -> true/false.\n"
        "- `depthOf(self, id)` -> length of the longest prerequisite chain below `id` (no prerequisites = 0); "
        "`nil` for an unknown id or when the graph has a cycle."
    ),
    visible_tests=[
        EvolutionTest(_dag([(3, 1), (2, 1), (4, 2), (4, 3)], "order(Actor[#s])"), [1, 2, 3, 4], "visible"),
        EvolutionTest(_dag([(2, 1), (1, 2)], "hasCycle(Actor[#s])"), True, "visible"),
        EvolutionTest(_dag([(2, 1), (1, 2)], "order(Actor[#s])"), None, "visible"),
        EvolutionTest(_dag([(3, 1), (2, 1), (4, 2), (4, 3)], "depthOf(Actor[#s], 4)"), 2, "visible"),
    ],
    holdout_tests=[
        EvolutionTest(_dag([(7, 9)], "order(Actor[#s])", extra_tasks=[9, 5, 7]), [5, 9, 7], "smallest-first"),
        EvolutionTest(_dag([], "order(Actor[#s])", extra_tasks=[3, 1, 2]), [1, 2, 3], "independent-sorted"),
        EvolutionTest(_dag([], "taskCount(Actor[#s])", extra_tasks=[3, 3, 3, 1]), 2, "idempotent-add"),
        EvolutionTest(_dag([(10, 20)], "taskCount(Actor[#s])"), 2, "auto-register"),
        EvolutionTest(_dag([(1, 1)], "hasCycle(Actor[#s])"), True, "self-cycle"),
        EvolutionTest(_dag([(1, 3), (2, 1), (3, 2)], "order(Actor[#s])", extra_tasks=[4]), None, "long-cycle"),
        EvolutionTest(_dag([(2, 1), (3, 2), (4, 3), (5, 4)], "depthOf(Actor[#s], 5)"), 4, "chain-depth"),
        EvolutionTest(_dag([(2, 1), (3, 1), (4, 2), (4, 3)], "depthOf(Actor[#s], 4)"), 2, "diamond-depth"),
        EvolutionTest(_dag([(2, 1)], "depthOf(Actor[#s], 99)"), None, "unknown-id"),
        EvolutionTest(_dag([(2, 1), (1, 2)], "depthOf(Actor[#s], 1)"), None, "cycle-depth"),
        EvolutionTest(_dag([(5, 2), (2, 8), (8, 1)], "order(Actor[#s])"), [1, 8, 2, 5], "reverse-chain"),
        EvolutionTest(_dag([(2, 1), (3, 1)], "hasCycle(Actor[#s])"), False, "no-cycle"),
        EvolutionTest(
            "Actor[#s] := initScheduler(new(TaskScheduler))!! Actor[#i] := 12!! "
            "loop while (Actor[#i] > 1) begin addDependency(Actor[#s], Actor[#i], Actor[#i] - 1); "
            "Actor[#i] := Actor[#i] - 1; endLoop!! order(Actor[#s])!!",
            oracle="result == list(range(1, 13))",
            tag="bulk-chain",
        ),
    ],
    required_classes=["TaskScheduler"],
    test_opcode_cap=250_000,
    scripted_generations=[_DAG_GEN1, _DAG_GEN2],
)

# ---- Tier 3: harness_step_budget_advisor ----------------------------------------------------
_ADV_FIXTURE = """
inherit(Object, #FakeHarness, #(#log #steps #limit), 3, nil)!!
now(FakeHarness)!!
Def initFake(self, lim) { self.log := new(OrderedCollection); self.steps := 0; self.limit := lim; ^self; }!!
Def push(self, act, st | d) {
    d := new(Dictionary, 4);
    atPut(d, #action, act);
    atPut(d, #status, st);
    add(self.log, d);
    self.steps := self.steps + 1;
    ^self;
}!!
Def stepLog(self) { ^self.log; }!!
Def stepCount(self) { ^self.steps; }!!
Def maxSteps(self) { ^self.limit; }!!
"""
_ADV_BASE = """
inherit(Object, #StepBudgetAdvisor, #(#unused), 1, nil)!!
now(StepBudgetAdvisor)!!
Def remainingBudget(self, h | r) {
    if maxSteps(h) <= 0 then ^(0 - 1); endif;
    r := maxSteps(h) - stepCount(h);
    if r < 0 then r := 0; endif;
    ^r;
}!!
Def consecutiveErrors(self, h | log i n c) {
    log := stepLog(h);
    n := size(log);
    c := 0;
    i := n - 1;
    loop while (i >= 0) begin
        if at(at(log, i), #status) == "error" then
            c := c + 1;
            i := i - 1;
        else
            i := 0 - 1;
        endif;
    endLoop;
    ^c;
}!!
"""
_ADV_GEN1 = (
    _ADV_BASE
    + """
Def isOscillating(self, h | log n a b c) {
    log := stepLog(h);
    n := size(log);
    if n < 3 then ^false; endif;
    a := at(at(log, n - 1), #action);
    b := at(at(log, n - 2), #action);
    c := at(at(log, n - 3), #action);
    ^((a == b) and (b == c));
}!!
Def verdict(self, h) {
    if remainingBudget(self, h) == 0 then ^#exhausted; endif;
    if isOscillating(self, h) == true then ^#oscillating; endif;
    ^#ok;
}!!
Def adaptiveBudget(self, h | r) {
    r := remainingBudget(self, h);
    if isOscillating(self, h) == true then
        if r > 2 then ^2; endif;
    endif;
    ^r;
}!!
"""
)
_ADV_GEN2 = (
    _ADV_BASE
    + """
Def isOscillating(self, h | log n a b c d) {
    log := stepLog(h);
    n := size(log);
    if n >= 3 then
        a := at(at(log, n - 1), #action);
        b := at(at(log, n - 2), #action);
        c := at(at(log, n - 3), #action);
        if (a == b) and (b == c) then ^true; endif;
    endif;
    if n >= 4 then
        d := at(at(log, n - 4), #action);
        if (a == c) and (b == d) and (a <> b) then ^true; endif;
    endif;
    ^false;
}!!
Def verdict(self, h) {
    if remainingBudget(self, h) == 0 then ^#exhausted; endif;
    if consecutiveErrors(self, h) >= 3 then ^#failing; endif;
    if isOscillating(self, h) == true then ^#oscillating; endif;
    ^#ok;
}!!
Def adaptiveBudget(self, h | r v) {
    r := remainingBudget(self, h);
    if r <= 0 then ^r; endif;
    v := verdict(self, h);
    if v == #failing then
        if r > 1 then ^1; endif;
    endif;
    if v == #oscillating then
        if r > 2 then ^2; endif;
    endif;
    ^r;
}!!
"""
)


def _fake(lim: int, pushes: Sequence[Tuple[str, str]], tail: str) -> str:
    ps = " ".join(f'push(Actor[#f], "{a}", "{s}")!!' for a, s in pushes)
    return f"Actor[#f] := initFake(new(FakeHarness), {lim})!! {ps} {tail}!!".replace("!!!!", "!!")


_ADV = "Actor[#adv] := new(StepBudgetAdvisor)!!"
_ADV_SPEC = EvolutionSpec(
    spec_id="harness_step_budget_advisor",
    name="Metacircular step-budget advisor (oscillation / failure detection)",
    difficulty=TIER_HARNESS,
    goal=(
        "Create a new `StepBudgetAdvisor` class that inspects an `AgentHarness`-shaped object (its `stepLog`, "
        "`stepCount`, `maxSteps`) to detect oscillation / repeated failure and compute an adaptive remaining step budget."
    ),
    spec_prompt=(
        "Define `inherit(Object, #StepBudgetAdvisor, #(...), n, nil)`. Every selector takes the harness `h` and may ONLY use "
        "the public reflective accessors `stepLog(h)`, `stepCount(h)`, `maxSteps(h)`. `stepLog(h)` is an OrderedCollection "
        "of entries; read an entry `e` with `at(e, #action)` (string) and `at(e, #status)` (string `\"ok\"` or `\"error\"`). "
        "It must work for the REAL `Actor[#Harness]` (whose entries are Python dicts) as well as any object providing those accessors.\n"
        "- `remainingBudget(self, h)`: `maxSteps - stepCount`, never below 0; `-1` when `maxSteps <= 0` (unlimited).\n"
        "- `consecutiveErrors(self, h)`: number of TRAILING entries whose status is `\"error\"`.\n"
        "- `isOscillating(self, h)`: true when the last 3 actions are identical (A,A,A) OR the last 4 actions ping-pong "
        "(A,B,A,B with A different from B).\n"
        "- `verdict(self, h)`: `#exhausted` if remainingBudget == 0; else `#failing` if consecutiveErrors >= 3; "
        "else `#oscillating` if isOscillating; else `#ok`.\n"
        "- `adaptiveBudget(self, h)`: the budget Bilby should keep: 0 if exhausted; `min(remaining, 1)` when `#failing`; "
        "`min(remaining, 2)` when `#oscillating`; otherwise `remainingBudget`."
    ),
    visible_tests=[
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "ok"), ("c", "ok")], "remainingBudget(Actor[#adv], Actor[#f])"), 7, "visible"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "ok"), ("a", "ok"), ("b", "ok")], "isOscillating(Actor[#adv], Actor[#f])"), True, "visible"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "error"), ("c", "error"), ("d", "error")], "verdict(Actor[#adv], Actor[#f])"), "#failing", "visible"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "ok"), ("a", "ok"), ("b", "ok")], "adaptiveBudget(Actor[#adv], Actor[#f])"), 2, "visible"),
    ],
    holdout_tests=[
        EvolutionTest(_ADV + _fake(10, [("x", "ok"), ("x", "ok"), ("x", "ok")], "isOscillating(Actor[#adv], Actor[#f])"), True, "triple-repeat"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "ok"), ("a", "ok"), ("c", "ok")], "isOscillating(Actor[#adv], Actor[#f])"), False, "not-pingpong"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("a", "ok"), ("b", "ok")], "isOscillating(Actor[#adv], Actor[#f])"), False, "double-only"),
        EvolutionTest(_ADV + _fake(10, [("a", "error"), ("b", "error"), ("c", "error"), ("d", "ok")], "consecutiveErrors(Actor[#adv], Actor[#f])"), 0, "trailing-only"),
        EvolutionTest(_ADV + _fake(10, [("a", "ok"), ("b", "error"), ("c", "error")], "consecutiveErrors(Actor[#adv], Actor[#f])"), 2, "trailing-count"),
        EvolutionTest(_ADV + _fake(4, [("a", "ok"), ("b", "ok"), ("a", "ok"), ("b", "ok")], "verdict(Actor[#adv], Actor[#f])"), "#exhausted", "exhausted-priority"),
        EvolutionTest(_ADV + _fake(4, [("a", "ok"), ("b", "ok"), ("a", "ok"), ("b", "ok")], "adaptiveBudget(Actor[#adv], Actor[#f])"), 0, "exhausted-budget"),
        EvolutionTest(_ADV + _fake(10, [("x", "error"), ("x", "error"), ("x", "error")], "verdict(Actor[#adv], Actor[#f])"), "#failing", "failing-over-oscillating"),
        EvolutionTest(_ADV + _fake(10, [("x", "error"), ("x", "error"), ("x", "error")], "adaptiveBudget(Actor[#adv], Actor[#f])"), 1, "failing-budget"),
        EvolutionTest(_ADV + _fake(0, [("a", "ok")], "remainingBudget(Actor[#adv], Actor[#f])"), -1, "unlimited"),
        EvolutionTest(_ADV + _fake(10, [], "verdict(Actor[#adv], Actor[#f])"), "#ok", "empty-log"),
        EvolutionTest(_ADV + _fake(10, [], "adaptiveBudget(Actor[#adv], Actor[#f])"), 10, "empty-budget"),
        EvolutionTest(_ADV + _fake(3, [("a", "ok")] * 1 + [("b", "ok")] * 1 + [("c", "ok")] * 1 + [("d", "ok")] * 2, "remainingBudget(Actor[#adv], Actor[#f])"), 0, "never-negative"),
        EvolutionTest(
            _ADV + "remainingBudget(Actor[#adv], Actor[#Harness])!!",
            oracle="isinstance(result, int) and (result >= 0 or result == -1)",
            tag="real-harness",
        ),
        EvolutionTest(
            _ADV + "verdict(Actor[#adv], Actor[#Harness])!!",
            oracle="result in ('#ok', '#oscillating', '#failing', '#exhausted')",
            tag="real-harness",
        ),
    ],
    required_classes=["StepBudgetAdvisor"],
    fixture_source=_ADV_FIXTURE,
    scripted_generations=[_ADV_GEN1, _ADV_GEN2],
)

DEFAULT_EVOLUTION_SPECS: Dict[str, EvolutionSpec] = {
    s.spec_id: s for s in (_MEDIAN_SPEC, _RING_SPEC, _CB_SPEC, _DAG_SPEC, _ADV_SPEC)
}


# ---------------------------------------------------------------------------
# Convenience API (used by `Actor[#Bench]` and the CLI)
# ---------------------------------------------------------------------------


def resolve_specs(
    selection: Union[str, Sequence[str], None] = "all",
    spec_files: Sequence[Union[str, Path]] = (),
) -> List[EvolutionSpec]:
    """Resolve `all` / comma list / list of ids (+ spec files) into `EvolutionSpec`s."""
    pool: Dict[str, EvolutionSpec] = dict(DEFAULT_EVOLUTION_SPECS)
    custom: List[EvolutionSpec] = []
    for f in spec_files:
        for s in load_spec_file(f):
            pool[s.spec_id] = s
            custom.append(s)
    if isinstance(selection, str):
        ids = [x.strip().lstrip("#") for x in selection.split(",") if x.strip()]
    else:
        ids = [str(x).strip().lstrip("#") for x in (selection or [])]
    if not ids or ids == ["all"]:
        return list(DEFAULT_EVOLUTION_SPECS.values()) + [s for s in custom if s.spec_id not in DEFAULT_EVOLUTION_SPECS]
    missing = [i for i in ids if i not in pool]
    if missing:
        raise KeyError(f"unknown self-evolution spec(s) {missing}; available: {sorted(pool)}")
    return [pool[i] for i in ids]


def list_self_evolution_suites() -> str:
    """Human-readable catalogue of built-in specs (for `selfEvoSuites(Actor[#Bench])`)."""
    lines = [f"ActorII Bilby Self-Evolution Suite ({len(DEFAULT_EVOLUTION_SPECS)} specs):"]
    for s in DEFAULT_EVOLUTION_SPECS.values():
        lines.append(
            f"  - #{s.spec_id} [{s.difficulty}] {s.name} "
            f"(visible={len(s.visible_tests)}, holdout={len(s.holdout_tests)}, G={s.max_generations}, "
            f"steps/gen={s.max_steps_per_gen}, persistence={'yes' if s.require_persistence else 'no'})"
        )
    return "\n".join(lines)


def run_self_evolution_spec(
    spec_id: Union[str, EvolutionSpec],
    mode: str = "scripted",
    *,
    max_generations: Optional[int] = None,
    **runner_kwargs: Any,
) -> Dict[str, Any]:
    """Run one spec (by id or object) in an isolated sandbox and return the result dict."""
    spec = spec_id if isinstance(spec_id, EvolutionSpec) else resolve_specs([str(spec_id)])[0]
    with SelfEvolutionRunner(mode, max_generations=max_generations, **runner_kwargs) as runner:
        return runner.run_spec(spec)


def run_self_evolution_suite(
    specs: Union[str, Sequence[str], Sequence[EvolutionSpec], None] = "all",
    mode: str = "scripted",
    *,
    spec_files: Sequence[Union[str, Path]] = (),
    max_generations: Optional[int] = None,
    **runner_kwargs: Any,
) -> Dict[str, Any]:
    if specs and not isinstance(specs, str) and all(isinstance(s, EvolutionSpec) for s in specs):
        resolved = list(specs)  # type: ignore[arg-type]
    else:
        resolved = resolve_specs(specs, spec_files)  # type: ignore[arg-type]
    with SelfEvolutionRunner(mode, max_generations=max_generations, **runner_kwargs) as runner:
        return runner.run_suite(resolved)


__all__ = [
    "DEFAULT_EVOLUTION_SPECS",
    "DIAGNOSTICS_MARKER",
    "EvolutionSpec",
    "EvolutionTest",
    "SelfEvolutionRunner",
    "TIER_CLASS",
    "TIER_HARNESS",
    "TIER_METHOD",
    "list_self_evolution_suites",
    "load_spec_file",
    "render_markdown",
    "resolve_specs",
    "run_self_evolution_spec",
    "run_self_evolution_suite",
    "summarize_result",
    "summarize_suite",
]
