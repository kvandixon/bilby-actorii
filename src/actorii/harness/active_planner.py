"""Active-Inference Counterfactual VM Tree Search Bridge (`actorii.harness.active_planner`).

Exposes Leap #2 (`Actor[#Planner]`) as a strictly **0 MB GPU VRAM**, pure-CPU
Actor bridge (`ActivePlannerBridge`) combining:

1. **Active Inference Expected Free Energy (EFE):**
   Computes $G(\\pi) = \\text{risk} + \\text{ambiguity} = -(\\text{pragmaticValue} + \\text{epistemicValue})$
   over discrete POMDP beliefs $q(s)$ and candidate action policies. Uses
   `inferactively-pymdp` (forced onto CPU via `JAX_PLATFORMS=cpu`) automatically
   when installed, with an exact NumPy discrete POMDP EFE engine (`numpy_efe`)
   as a first-class fallback.
2. **Counterfactual Shadow VM Branch Simulation:**
   Runs competing Actor 4.1 code candidates (plus optional verifier chunks) inside
   isolated `shadow_only=True` VM transactions (`vm.eval_transactional`), scoring
   branches with deterministic verifier pass + gas efficiency + optional `:11434`
   zero-VRAM System-1 prior while guaranteeing **zero mutation** of live VM state.
3. **Persistent Counterfactual Dead-End Memory (`:ActorDeadEnd`):**
   Records and recalls failed branch signatures in Neo4j (namespace-scoped, with
   in-memory fallback so offline/unit runs always pass) and prunes known dead ends
   prior to speculative expansion.

Guardrails:
- Strictly 0 MB GPU VRAM (`JAX_PLATFORMS=cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
- Zero live VM mutation during `simulateBranches` (enforced via `shadow_only=True`).
- Offline-safe Neo4j and `:11434` System-1 HTTP integrations with bounded timeouts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from actorii.bridge import actor_bridge, actor_command, actor_query

# Force CPU-only execution before any JAX/pymdp import to guarantee 0 MB GPU VRAM.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_PYMDP_CONTROL: Any = None
_PYMDP_JNP: Any = None
_PYMDP_AVAILABLE: bool = False
_PYMDP_MODE: str = "none"  # "jax_v1" | "numpy_v0" | "none"

try:
    import pymdp.control as _pymdp_ctrl  # type: ignore

    if hasattr(_pymdp_ctrl, "compute_info_gain") and hasattr(
        _pymdp_ctrl, "compute_expected_utility"
    ):
        try:
            import jax.numpy as _jnp  # type: ignore

            _PYMDP_CONTROL = _pymdp_ctrl
            _PYMDP_JNP = _jnp
            _PYMDP_AVAILABLE = True
            _PYMDP_MODE = "jax_v1"
        except Exception:
            _PYMDP_CONTROL = None
            _PYMDP_JNP = None
            _PYMDP_AVAILABLE = False
    elif hasattr(_pymdp_ctrl, "calc_states_info_gain") and hasattr(
        _pymdp_ctrl, "calc_expected_utility"
    ):
        _PYMDP_CONTROL = _pymdp_ctrl
        _PYMDP_AVAILABLE = True
        _PYMDP_MODE = "numpy_v0"
except Exception:
    _PYMDP_CONTROL = None
    _PYMDP_JNP = None
    _PYMDP_AVAILABLE = False
    _PYMDP_MODE = "none"


NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    os.environ.get("ACTORII_NEO4J_TX_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"),
)
DEFAULT_SYSTEMONE_URL = os.environ.get(
    "ACTORII_SYSTEMONE_URL", "http://127.0.0.1:11434/v1/systemone"
)
DEFAULT_SYSTEMONE_MODEL = os.environ.get("ACTORII_SYSTEMONE_MODEL", "clef-flash-cpu:latest")

_EPS = 1e-12
_INDEXED_PLANNER_NEO4J_URLS: set[str] = set()
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _norm_sym(value: Any) -> str:
    """Normalize an Actor Symbol or string into a plain Python string."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lstrip("#")
    return str(value).lstrip("#")


def _actor_to_py_value(val: Any) -> Any:
    """Recursively convert Actor Dictionary/Array/Symbol values to plain Python objects."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    sym_name = getattr(val, "name", None)
    if isinstance(sym_name, str) and type(val).__name__ == "Symbol":
        return sym_name.lstrip("#")
    if isinstance(val, dict):
        return {_norm_sym(k): _actor_to_py_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_actor_to_py_value(item) for item in val]
    data = getattr(val, "_data", None)
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            key = _norm_sym(k)
            if key not in out:
                out[key] = _actor_to_py_value(v)
        return out
    items = getattr(val, "_items", None)
    if isinstance(items, list):
        return [_actor_to_py_value(item) for item in items]
    return val


def _py_to_actor_value(val: Any) -> Any:
    """Recursively convert Python dicts/lists/scalars into native Actor objects."""
    from actorii.classes.collections import ActorArray
    from actorii.classes.dictionary import Dictionary
    from actorii.core import Symbol

    if isinstance(val, dict):
        d = Dictionary(max(8, len(val) * 2))
        for k, v in val.items():
            conv_v = _py_to_actor_value(v)
            k_str = str(k).lstrip("#")
            d.atPut(k_str, conv_v)
            d.atPut(Symbol(k_str), conv_v)
        return d
    if isinstance(val, (list, tuple)):
        arr = ActorArray(len(val))
        for idx, item in enumerate(val):
            arr.atPut(idx, _py_to_actor_value(item))
        return arr
    if isinstance(val, np.generic):
        return val.item()
    return val


def _parse_json_or_actor(raw: Any, label: str) -> Any:
    """Parse a JSON string or convert a native Actor object into Python data."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{label} must not be empty.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {label}: {exc}") from exc
    return _actor_to_py_value(raw)


def _normalize_prob_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Probability vector must be non-empty.")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.size, 1.0 / arr.size, dtype=np.float64)
    return arr / total


def _normalize_columns(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected 2D non-empty matrix, got shape {arr.shape}.")
    arr = np.clip(arr, 0.0, None)
    col_sums = arr.sum(axis=0, keepdims=True)
    zero_cols = col_sums <= 0.0
    if np.any(zero_cols):
        arr[:, zero_cols[0]] = 1.0 / arr.shape[0]
        col_sums = arr.sum(axis=0, keepdims=True)
    return arr / col_sums


def _softmax(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    shifted = arr - float(np.max(arr))
    exps = np.exp(shifted)
    return exps / float(np.sum(exps))


def _shannon_entropy(p: np.ndarray) -> float:
    probs = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _normalized_entropy(p: np.ndarray) -> float:
    n = p.size
    if n <= 1:
        return 0.0
    return float(min(1.0, max(0.0, _shannon_entropy(p) / math.log(n))))


def _binary_belief_from_uncertainty(uncertainty: float) -> np.ndarray:
    """Construct a 2-state belief vector whose normalized entropy matches `uncertainty`."""
    u = float(min(1.0, max(0.0, uncertainty)))
    if u <= 1e-6:
        return np.array([0.999, 0.001], dtype=np.float64)
    if u >= 1.0 - 1e-6:
        return np.array([0.5, 0.5], dtype=np.float64)
    target_h = u * math.log(2.0)
    lo, hi = 0.5, 0.999999
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        h = -(mid * math.log(mid) + (1.0 - mid) * math.log(1.0 - mid))
        if h > target_h:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return np.array([p, 1.0 - p], dtype=np.float64)


def _normalize_signature(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _dead_end_id(namespace: str, goal: str, action_sig: str, error_sig: str) -> str:
    payload = f"{namespace}|{goal.strip().lower()}|{_normalize_signature(action_sig)}|{error_sig.strip().lower()}"
    return "de_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _token_overlap(a: str, b: str) -> float:
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _ensure_chunk_terminated(source: str) -> str:
    chunk = str(source or "").strip()
    if not chunk:
        return ""
    if not chunk.endswith("!!"):
        chunk = chunk.rstrip("!") + "!!"
    return chunk


def _build_action_matrices(
    action_spec: Dict[str, Any],
    num_states: int,
    default_C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Build normalized `(A, B, C, kind)` matrices for one candidate action."""
    kind_raw = str(action_spec.get("kind") or action_spec.get("type") or "").strip().lower()

    # Transition matrix B: shape (S, S)
    if "B" in action_spec and action_spec["B"] is not None:
        B = _normalize_columns(np.asarray(action_spec["B"], dtype=np.float64))
        if B.shape != (num_states, num_states):
            raise ValueError(
                f"Action B matrix must have shape ({num_states}, {num_states}), got {B.shape}."
            )
    else:
        B = np.eye(num_states, dtype=np.float64)

    # Observation preferences C: shape (O,)
    if "C" in action_spec and action_spec["C"] is not None:
        C = np.asarray(action_spec["C"], dtype=np.float64).reshape(-1)
    else:
        C = default_C.copy()

    # Observation likelihood matrix A: shape (O, S)
    if "A" in action_spec and action_spec["A"] is not None:
        A = _normalize_columns(np.asarray(action_spec["A"], dtype=np.float64))
        if A.shape[1] != num_states:
            raise ValueError(
                f"Action A matrix must have {num_states} columns (states), got {A.shape[1]}."
            )
        if C.size != A.shape[0]:
            if C.size == 0:
                C = np.zeros(A.shape[0], dtype=np.float64)
            else:
                padded = np.zeros(A.shape[0], dtype=np.float64)
                n_copy = min(C.size, A.shape[0])
                padded[:n_copy] = C[:n_copy]
                C = padded
    else:
        # Synthesize POMDP likelihood matrix from ergonomic high-level hints.
        num_obs = max(2, C.size if C.size > 0 else num_states)
        if C.size != num_obs:
            padded = np.zeros(num_obs, dtype=np.float64)
            if C.size > 0:
                n_copy = min(C.size, num_obs)
                padded[:n_copy] = C[:n_copy]
            else:
                padded[0] = 2.0
                padded[-1] = -2.0
            C = padded

        is_epistemic_hint = kind_raw in ("epistemic", "probe", "epistemic_probe", "inspect", "read")
        if is_epistemic_hint:
            # High-precision observation likelihood (sharp diagonal -> high info gain),
            # neutral observation preferences (probes do not directly claim goal completion).
            acc = float(action_spec.get("obsAccuracy", action_spec.get("accuracy", 0.94)))
            acc = min(0.995, max(0.55, acc))
            A = np.full((num_obs, num_states), (1.0 - acc) / max(1, num_obs - 1), dtype=np.float64)
            for s in range(num_states):
                A[s % num_obs, s] = acc
            A = _normalize_columns(A)
            if "C" not in action_spec:
                C = np.zeros(num_obs, dtype=np.float64)
        else:
            # Pragmatic action: state 0 yields preferred obs 0 with high probability,
            # while wrong states yield failure obs -1 with high probability.
            p_succ = float(action_spec.get("expectedSuccess", action_spec.get("goalProb", 0.90)))
            p_succ = min(0.995, max(0.05, p_succ))
            p_fail_on_wrong = float(action_spec.get("wrongStateFailProb", 0.85))
            p_fail_on_wrong = min(0.995, max(0.05, p_fail_on_wrong))
            A = np.full((num_obs, num_states), 1.0 / num_obs, dtype=np.float64)
            # Good state (index 0) -> preferred observation (index 0)
            A[:, 0] = (1.0 - p_succ) / max(1, num_obs - 1)
            A[0, 0] = p_succ
            # Non-good states -> failure observation (last index)
            for s in range(1, num_states):
                A[:, s] = (1.0 - p_fail_on_wrong) / max(1, num_obs - 1)
                A[-1, s] = p_fail_on_wrong
            A = _normalize_columns(A)

    if not kind_raw:
        kind_raw = "epistemic" if float(np.max(np.abs(C))) < 1e-6 else "pragmatic"
    return A, B, C, kind_raw


def _compute_efe_single_numpy(
    qs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Dict[str, float]:
    """Exact NumPy discrete POMDP Expected Free Energy for a single 1-step policy.

    Computes:
    - `qs_next = B @ qs`
    - `qo = A @ qs_next`
    - `log_p_o = log(softmax(C))`
    - `pragmaticValue = sum(qo * log_p_o)` (expected utility)
    - `epistemicValue = H(qo) - sum_s qs_next[s] * H(A[:, s])` (Bayesian info gain)
    - `risk = D_KL(qo || p(o)) = sum(qo * (log(qo) - log_p_o))`
    - `ambiguity = sum_s qs_next[s] * H(A[:, s])`
    - `efe = risk + ambiguity = -(pragmaticValue + epistemicValue)`
    """
    qs_next = _normalize_prob_vector(B @ qs)
    qo = _normalize_prob_vector(A @ qs_next)
    p_pref = _softmax(C)
    log_p_pref = np.log(np.clip(p_pref, _EPS, 1.0))
    log_qo = np.log(np.clip(qo, _EPS, 1.0))

    # Ambiguity: E_{q(s|pi)}[H[P(o|s)]]
    A_safe = np.clip(A, _EPS, 1.0)
    H_A_cols = -np.sum(A_safe * np.log(A_safe), axis=0)
    ambiguity = float(np.dot(qs_next, H_A_cols))

    # Entropy of predicted observation distribution H[q(o|pi)]
    H_qo = float(-np.sum(qo * log_qo))

    # Epistemic value (information gain) = H[q(o|pi)] - ambiguity >= 0
    epistemic_value = max(0.0, H_qo - ambiguity)

    # Pragmatic value (expected utility under normalized log-preferences)
    pragmatic_value = float(np.dot(qo, log_p_pref))

    # Risk = D_KL(q(o|pi) || p(o)) >= 0
    risk = max(0.0, float(np.dot(qo, log_qo - log_p_pref)))

    # Expected Free Energy G(pi) = risk + ambiguity = -(pragmatic + epistemic)
    efe = risk + ambiguity

    return {
        "efe": float(efe),
        "risk": float(risk),
        "ambiguity": float(ambiguity),
        "pragmaticValue": float(pragmatic_value),
        "epistemicValue": float(epistemic_value),
    }


def _compute_efe_single_pymdp(
    qs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Dict[str, float]:
    """Compute Expected Free Energy via `inferactively-pymdp` (CPU-only)."""
    if not _PYMDP_AVAILABLE or _PYMDP_CONTROL is None:
        return _compute_efe_single_numpy(qs, A, B, C)

    qs_next_np = _normalize_prob_vector(B @ qs)
    p_pref = _softmax(C)
    log_p_pref = np.log(np.clip(p_pref, _EPS, 1.0))

    if _PYMDP_MODE == "jax_v1" and _PYMDP_JNP is not None:
        jnp = _PYMDP_JNP
        qs_list = [jnp.asarray(qs_next_np, dtype=jnp.float32)]
        A_list = [jnp.asarray(A, dtype=jnp.float32)]
        C_list = [jnp.asarray(log_p_pref, dtype=jnp.float32)]
        qo_list = _PYMDP_CONTROL.compute_expected_obs(qs_list, A_list, [[0]])
        epistemic_value = max(
            0.0, float(_PYMDP_CONTROL.compute_info_gain(qs_list, qo_list, A_list, [[0]]))
        )
        pragmatic_value = float(_PYMDP_CONTROL.compute_expected_utility(qo_list, C_list))
        qo_np = _normalize_prob_vector(np.asarray(qo_list[0], dtype=np.float64))
    else:
        # Legacy pymdp 0.0.x NumPy API
        qs_obj = np.empty(1, dtype=object)
        qs_obj[0] = qs_next_np
        A_obj = np.empty(1, dtype=object)
        A_obj[0] = A
        C_obj = np.empty(1, dtype=object)
        C_obj[0] = log_p_pref
        qo_np = _normalize_prob_vector(A @ qs_next_np)
        qo_obj = np.empty(1, dtype=object)
        qo_obj[0] = qo_np
        epistemic_value = max(0.0, float(_PYMDP_CONTROL.calc_states_info_gain(A_obj, qs_obj)))
        pragmatic_value = float(_PYMDP_CONTROL.calc_expected_utility(qo_obj, C_obj))

    H_qo = _shannon_entropy(qo_np)
    ambiguity = max(0.0, H_qo - epistemic_value)
    log_qo = np.log(np.clip(qo_np, _EPS, 1.0))
    risk = max(0.0, float(np.dot(qo_np, log_qo - log_p_pref)))
    efe = risk + ambiguity

    return {
        "efe": float(efe),
        "risk": float(risk),
        "ambiguity": float(ambiguity),
        "pragmaticValue": float(pragmatic_value),
        "epistemicValue": float(epistemic_value),
    }


@actor_bridge(name="ActivePlannerBridge")
class ActivePlannerBridge:
    """Active-Inference Counterfactual VM Tree Search bridge (`Actor[#Planner]`, 0 MB VRAM)."""

    def __init__(
        self,
        vm: Optional[Any] = None,
        tx_url: Optional[str] = None,
        namespace: Optional[str] = None,
        enable_neo4j: bool = True,
        enable_systemone: Optional[bool] = None,
        systemone_url: Optional[str] = None,
        systemone_timeout_s: float = 1.5,
        prefer_pymdp: bool = True,
        default_max_opcodes: int = 25000,
    ) -> None:
        self.vm = vm
        self.tx_url = tx_url or os.environ.get("ACTORII_NEO4J_HTTP_URL", NEO4J_HTTP_TX_URL)
        default_ns = os.environ.get("ACTORII_MEMORY_NAMESPACE", "default").strip() or "default"
        self.namespace = str(namespace or default_ns)
        self.enable_neo4j = bool(enable_neo4j)
        if enable_systemone is None:
            self.enable_systemone = os.environ.get("ACTORII_PLANNER_SYSTEMONE", "0") == "1"
        else:
            self.enable_systemone = bool(enable_systemone)
        self.systemone_url = systemone_url or DEFAULT_SYSTEMONE_URL
        self.systemone_timeout_s = max(0.1, float(systemone_timeout_s))
        self.prefer_pymdp = bool(prefer_pymdp)
        self.default_max_opcodes = max(1000, int(default_max_opcodes))

        # Namespace-scoped in-memory dead-end fallback store: (namespace, dead_end_id) -> record
        self._local_dead_ends: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def set_vm(self, vm: Any) -> None:
        """Bind or rebind the live `ActorVM` used for `shadow_only=True` rollouts."""
        self.vm = vm

    @property
    def backend_name(self) -> str:
        if self.prefer_pymdp and _PYMDP_AVAILABLE:
            return "pymdp"
        return "numpy_efe"

    def _neo4j_headers(self, *, read_only: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if read_only:
            headers["Access-Mode"] = "READ"
        neo_user = os.environ.get("ACTORII_NEO4J_USER", "").strip()
        neo_pass = os.environ.get("ACTORII_NEO4J_PASSWORD", "").strip()
        if neo_user and neo_pass:
            token = base64.b64encode(f"{neo_user}:{neo_pass}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers

    def _run_cypher_batch(
        self,
        statements: List[Dict[str, Any]],
        *,
        read_only: bool = False,
        timeout: float = 2.5,
    ) -> List[Tuple[List[str], List[Dict[str, Any]]]]:
        payload = {"statements": statements}
        req = urllib.request.Request(
            self.tx_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._neo4j_headers(read_only=read_only),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        if body.get("errors"):
            err_items = body["errors"]
            msg = "; ".join(
                f"{e.get('code', 'CypherError')}: {e.get('message', '')}" for e in err_items
            )
            raise RuntimeError(f"Neo4j Cypher error: {msg}")
        out: List[Tuple[List[str], List[Dict[str, Any]]]] = []
        for res in body.get("results", []):
            cols = res.get("columns", [])
            rows = [dict(zip(cols, item.get("row", []))) for item in res.get("data", [])]
            out.append((cols, rows))
        return out

    def _ensure_dead_end_indexes(self) -> None:
        if not self.enable_neo4j or self.tx_url in _INDEXED_PLANNER_NEO4J_URLS:
            return
        _INDEXED_PLANNER_NEO4J_URLS.add(self.tx_url)
        try:
            self._run_cypher_batch(
                [
                    {
                        "statement": (
                            "CREATE INDEX actor_dead_end_ns_goal IF NOT EXISTS "
                            "FOR (d:ActorDeadEnd) ON (d.namespace, d.goal)"
                        ),
                        "parameters": {},
                    },
                    {
                        "statement": (
                            "CREATE INDEX actor_dead_end_ns_id IF NOT EXISTS "
                            "FOR (d:ActorDeadEnd) ON (d.namespace, d.id)"
                        ),
                        "parameters": {},
                    },
                ],
                timeout=2.0,
            )
        except Exception:
            pass

    def _query_systemone_prior(self, candidate_code: str, result_repr: str) -> Optional[float]:
        """Query optional `:11434` zero-VRAM System-1 prior; strictly offline-safe."""
        if not self.enable_systemone:
            return None
        body = {
            "model": DEFAULT_SYSTEMONE_MODEL,
            "state": (
                f"Actor 4.1 shadow branch code:\n{candidate_code[:400]}\n"
                f"Shadow output: {result_repr[:200]}"
            ),
            "questions": {
                "viable": "Is this shadow branch clean, safe, and likely to satisfy the goal?"
            },
        }
        try:
            req = urllib.request.Request(
                self.systemone_url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.systemone_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            # Handle both {"answers": {"viable": 0.82}} and {"probabilities": ...}
            answers = data.get("answers") or data.get("probabilities") or data
            if isinstance(answers, dict):
                raw_p = answers.get("viable")
                if isinstance(raw_p, dict):
                    raw_p = raw_p.get("yes", raw_p.get("true", raw_p.get("prob")))
                if isinstance(raw_p, (int, float)):
                    return float(min(1.0, max(0.0, raw_p)))
        except Exception:
            return None
        return None

    def _find_matching_dead_end(self, code: str, goal: str = "") -> Optional[Dict[str, Any]]:
        """Check whether a candidate code/action matches a known `:ActorDeadEnd`."""
        norm_code = _normalize_signature(code)
        if not norm_code:
            return None

        # Check local cache first (fast, offline)
        for (ns, _did), rec in self._local_dead_ends.items():
            if ns != self.namespace:
                continue
            if goal and rec.get("goal") and rec["goal"].lower() != goal.lower():
                if _token_overlap(goal, rec["goal"]) < 0.25:
                    continue
            if _normalize_signature(rec.get("actionSig", "")) == norm_code:
                return rec

        # Check Neo4j if enabled
        if self.enable_neo4j:
            try:
                res = self._run_cypher_batch(
                    [
                        {
                            "statement": (
                                "MATCH (d:ActorDeadEnd {namespace: $ns, action_sig: $asig}) "
                                "RETURN d.id AS id, d.goal AS goal, d.action_sig AS actionSig, "
                                "d.error_sig AS errorSig, d.hit_count AS hitCount "
                                "LIMIT 1"
                            ),
                            "parameters": {"ns": self.namespace, "asig": norm_code},
                        }
                    ],
                    read_only=True,
                    timeout=1.5,
                )
                if res and res[0][1]:
                    return res[0][1][0]
            except Exception:
                pass
        return None

    @actor_query
    def getNamespace(self) -> str:
        """Return the active namespace for `:ActorDeadEnd` storage and recall."""
        return self.namespace

    @actor_command
    def setNamespace(self, namespace: str) -> str:
        """Switch the active namespace for `:ActorDeadEnd` storage and recall."""
        self.namespace = _norm_sym(namespace).strip() or "default"
        return f"NAMESPACE_SET:{self.namespace}"

    @actor_query
    def computeEFE(self, belief_json: str, actions_json: str) -> Any:
        """Compute Expected Free Energy $G(\\pi) = \\text{risk} + \\text{ambiguity}$ for candidate actions.

        Returns native Actor `Dictionary` with `#bestAction`, `#mode`
        (`"epistemic_probe"` vs `"pragmatic_exploit"`), `#scores`, `#backend`
        (`"pymdp"` or `"numpy_efe"`), `#beliefEntropy`, and `#bestEFE`.
        """
        belief_spec = _parse_json_or_actor(belief_json, "belief_json")
        actions_spec = _parse_json_or_actor(actions_json, "actions_json")

        # Parse belief state q(s) and optional default preference vector C
        default_C = np.array([2.0, -2.0], dtype=np.float64)
        if isinstance(belief_spec, list):
            qs = _normalize_prob_vector(belief_spec)
        elif isinstance(belief_spec, dict):
            if "qs" in belief_spec:
                qs = _normalize_prob_vector(belief_spec["qs"])
            elif "belief" in belief_spec:
                qs = _normalize_prob_vector(belief_spec["belief"])
            elif "probs" in belief_spec:
                qs = _normalize_prob_vector(belief_spec["probs"])
            elif "uncertainty" in belief_spec:
                qs = _binary_belief_from_uncertainty(float(belief_spec["uncertainty"]))
            else:
                raise ValueError(
                    "belief_json dict must include 'qs', 'belief', 'probs', or 'uncertainty'."
                )
            if "C" in belief_spec and belief_spec["C"] is not None:
                default_C = np.asarray(belief_spec["C"], dtype=np.float64).reshape(-1)
            elif "preferences" in belief_spec and belief_spec["preferences"] is not None:
                default_C = np.asarray(belief_spec["preferences"], dtype=np.float64).reshape(-1)
        else:
            raise ValueError("belief_json must be a JSON list of probabilities or dict.")

        if not isinstance(actions_spec, list) or len(actions_spec) == 0:
            raise ValueError("actions_json must be a non-empty JSON list of candidate actions.")

        num_states = qs.size
        use_pymdp = self.prefer_pymdp and _PYMDP_AVAILABLE
        backend = "pymdp" if use_pymdp else "numpy_efe"
        eval_fn = _compute_efe_single_pymdp if use_pymdp else _compute_efe_single_numpy

        scored: List[Dict[str, Any]] = []
        efe_values: List[float] = []

        for idx, raw_act in enumerate(actions_spec):
            if isinstance(raw_act, str):
                act_dict: Dict[str, Any] = {"name": raw_act}
            elif isinstance(raw_act, dict):
                act_dict = raw_act
            else:
                raise ValueError(f"Action at index {idx} must be a dict or string.")

            act_name = str(
                act_dict.get("name")
                or act_dict.get("action")
                or act_dict.get("id")
                or f"action_{idx}"
            )
            A, B, C, kind = _build_action_matrices(act_dict, num_states, default_C)
            metrics = eval_fn(qs, A, B, C)
            efe_values.append(metrics["efe"])
            scored.append(
                {
                    "action": act_name,
                    "kind": kind,
                    "efe": round(metrics["efe"], 6),
                    "risk": round(metrics["risk"], 6),
                    "ambiguity": round(metrics["ambiguity"], 6),
                    "pragmaticValue": round(metrics["pragmaticValue"], 6),
                    "epistemicValue": round(metrics["epistemicValue"], 6),
                    "posteriorProb": 0.0,
                }
            )

        # Posterior over policies: q(pi) = softmax(-gamma * G(pi))
        gamma = 4.0
        post_probs = _softmax(-gamma * np.asarray(efe_values, dtype=np.float64))
        for item, prob in zip(scored, post_probs):
            item["posteriorProb"] = round(float(prob), 6)

        best_idx = int(np.argmin(efe_values))
        best = scored[best_idx]

        # Classify mode: epistemic if best action is epistemic or epistemic info-gain dominates
        is_epistemic_mode = (
            best["kind"] in ("epistemic", "probe", "epistemic_probe", "inspect", "read")
            or (best["epistemicValue"] > 0.05 and best["epistemicValue"] >= abs(best["pragmaticValue"]) * 0.5)
        )
        mode = "epistemic_probe" if is_epistemic_mode else "pragmatic_exploit"

        result = {
            "bestAction": best["action"],
            "bestIndex": best_idx,
            "bestEFE": best["efe"],
            "mode": mode,
            "beliefEntropy": round(_normalized_entropy(qs), 6),
            "backend": backend,
            "scores": scored,
        }
        return _py_to_actor_value(result)

    @actor_query
    def recommendMode(
        self,
        uncertainty: float,
        steps_remaining: int,
        has_writes: bool = False,
    ) -> Any:
        """Fast epistemic-vs-pragmatic policy recommendation for Ring-1 step control."""
        u = float(min(1.0, max(0.0, float(uncertainty))))
        steps = max(0, int(steps_remaining))
        writes = bool(has_writes)

        # Mutating writes require higher certainty before committing.
        threshold = 0.45 if writes else 0.65

        if steps <= 1:
            mode = "pragmatic_exploit"
            should_tree = False
            reason = "Final step budget reached; must commit best pragmatic action."
        elif u >= threshold:
            mode = "epistemic_probe"
            should_tree = writes and steps >= 3
            reason = (
                f"Uncertainty ({u:.2f}) >= threshold ({threshold:.2f}) "
                f"{'with mutating writes ' if writes else ''}- forage information before committing."
            )
        else:
            mode = "pragmatic_exploit"
            should_tree = u >= 0.35 and steps >= 2
            reason = (
                f"Uncertainty ({u:.2f}) < threshold ({threshold:.2f}) - "
                f"sufficient belief certainty for pragmatic execution."
            )

        return _py_to_actor_value(
            {
                "mode": mode,
                "uncertainty": round(u, 6),
                "stepsRemaining": steps,
                "hasWrites": writes,
                "threshold": threshold,
                "shouldTreeSearch": should_tree,
                "reason": reason,
                "backend": self.backend_name,
            }
        )

    @actor_query
    def simulateBranches(
        self,
        candidates_json: str,
        verifier_chunk: str = "",
        max_branches: int = 4,
    ) -> Any:
        """Evaluate candidate Actor 4.1 code chunks inside isolated `shadow_only=True` transactions.

        Guarantees zero mutation of live VM state. Optionally executes `verifier_chunk`
        inside the same shadow transaction before rollback, prunes recorded `:ActorDeadEnd`
        branches, and returns the winning branch and per-branch telemetry.
        """
        raw_candidates = _parse_json_or_actor(candidates_json, "candidates_json")
        if not isinstance(raw_candidates, list) or len(raw_candidates) == 0:
            raise ValueError("candidates_json must be a non-empty JSON array of candidates.")

        cap = max(1, min(16, int(max_branches or 4)))
        candidates = raw_candidates[:cap]

        vm = self.vm
        if vm is None:
            from actorii.vm import ActorVM

            vm = ActorVM()

        verifier_src = _ensure_chunk_terminated(verifier_chunk) if str(verifier_chunk or "").strip() else ""
        max_opcodes = self.default_max_opcodes

        branch_results: List[Dict[str, Any]] = []
        pruned_dead_ends: List[Dict[str, Any]] = []

        best_index = -1
        best_score = float("-inf")
        best_action = ""
        best_code = ""
        best_verified = False

        for idx, cand in enumerate(candidates):
            if isinstance(cand, str):
                name = f"branch_{idx}"
                code_raw = cand
                goal = ""
            elif isinstance(cand, dict):
                name = str(cand.get("name") or cand.get("action") or f"branch_{idx}")
                code_raw = str(cand.get("code") or cand.get("chunk") or "")
                goal = str(cand.get("goal") or "")
            else:
                raise ValueError(f"Candidate at index {idx} must be a string or dict.")

            norm_code = _normalize_signature(code_raw)
            dead_end_hit = self._find_matching_dead_end(norm_code, goal=goal)
            if dead_end_hit is not None:
                prune_info = {
                    "index": idx,
                    "name": name,
                    "actionSig": norm_code,
                    "deadEndId": dead_end_hit.get("id", ""),
                    "errorSig": dead_end_hit.get("errorSig", ""),
                    "reason": "Matched recorded :ActorDeadEnd signature",
                }
                pruned_dead_ends.append(prune_info)
                branch_results.append(
                    {
                        "index": idx,
                        "name": name,
                        "code": code_raw,
                        "ok": False,
                        "verifierPassed": False,
                        "pruned": True,
                        "result": "",
                        "error": f"PRUNED_DEAD_END: {dead_end_hit.get('errorSig', '')}",
                        "gas": 0,
                        "systemOnePrior": None,
                        "score": -2.0,
                    }
                )
                continue

            cand_chunk = _ensure_chunk_terminated(code_raw)
            if not cand_chunk:
                branch_results.append(
                    {
                        "index": idx,
                        "name": name,
                        "code": code_raw,
                        "ok": False,
                        "verifierPassed": False,
                        "pruned": False,
                        "result": "",
                        "error": "Empty candidate code chunk.",
                        "gas": 0,
                        "systemOnePrior": None,
                        "score": -1.0,
                    }
                )
                continue

            combined_source = f"{cand_chunk}\n{verifier_src}" if verifier_src else cand_chunk

            ok = False
            verifier_passed = False
            result_str = ""
            error_str = ""
            gas_used = 0
            s1_prior: Optional[float] = None

            try:
                res = vm.eval_transactional(
                    combined_source,
                    max_opcodes=max_opcodes,
                    shadow_only=True,
                )
                gas_used = int(getattr(vm, "last_eval_opcodes", 0) or 0)
                ok = True
                result_str = str(res)
                if verifier_src:
                    if isinstance(res, bool):
                        verifier_passed = res
                    elif res is None:
                        verifier_passed = False
                    elif isinstance(res, (int, float)):
                        verifier_passed = res != 0
                    else:
                        verifier_passed = str(res).strip().lower() not in (
                            "",
                            "false",
                            "nil",
                            "none",
                            "0",
                            "error",
                        )
                else:
                    verifier_passed = True
                s1_prior = self._query_systemone_prior(code_raw, result_str)
            except Exception as exc:
                gas_used = int(getattr(vm, "last_eval_opcodes", 0) or 0)
                ok = False
                verifier_passed = False
                error_str = f"{type(exc).__name__}: {exc}"

            # Composite grounded branch score
            if not ok:
                score = -1.0
            else:
                gas_eff = max(0.0, 1.0 - (float(gas_used) / float(max_opcodes)))
                verifier_term = 0.50 if verifier_passed else -0.25
                s1_term = 0.10 * s1_prior if s1_prior is not None else 0.0
                score = 0.35 + verifier_term + 0.15 * gas_eff + s1_term

            branch_entry = {
                "index": idx,
                "name": name,
                "code": code_raw,
                "ok": ok,
                "verifierPassed": verifier_passed,
                "pruned": False,
                "result": result_str,
                "error": error_str,
                "gas": gas_used,
                "systemOnePrior": round(s1_prior, 6) if s1_prior is not None else None,
                "score": round(score, 6),
            }
            branch_results.append(branch_entry)

            if ok and score > best_score:
                best_score = score
                best_index = idx
                best_action = name
                best_code = code_raw
                best_verified = verifier_passed

        return _py_to_actor_value(
            {
                "bestIndex": best_index,
                "bestAction": best_action,
                "bestCode": best_code,
                "bestScore": round(best_score, 6) if best_index >= 0 else -1.0,
                "verified": best_verified,
                "shadowIsolated": True,
                "branches": branch_results,
                "prunedDeadEnds": pruned_dead_ends,
                "backend": self.backend_name,
            }
        )

    @actor_command
    def recordDeadEnd(self, goal: str, action_sig: str, error_sig: str) -> Any:
        """Persist a counterfactual `:ActorDeadEnd` node in Neo4j (with in-memory fallback)."""
        goal_str = str(goal or "").strip()
        action_str = _normalize_signature(action_sig)
        error_str = str(error_sig or "").strip()
        if not action_str:
            raise ValueError("action_sig must be non-empty.")

        did = _dead_end_id(self.namespace, goal_str, action_str, error_str)
        now_ts = time.time()
        key = (self.namespace, did)
        existing = self._local_dead_ends.get(key)
        hit_count = int(existing["hitCount"]) + 1 if existing else 1
        created_at = float(existing["createdAt"]) if existing else now_ts

        record = {
            "id": did,
            "namespace": self.namespace,
            "goal": goal_str,
            "actionSig": action_str,
            "errorSig": error_str,
            "hitCount": hit_count,
            "createdAt": created_at,
            "updatedAt": now_ts,
        }
        self._local_dead_ends[key] = record

        neo4j_synced = False
        neo4j_error = ""
        if self.enable_neo4j:
            try:
                self._ensure_dead_end_indexes()
                self._run_cypher_batch(
                    [
                        {
                            "statement": (
                                "MERGE (d:ActorDeadEnd {namespace: $ns, id: $id}) "
                                "ON CREATE SET d.goal = $goal, d.action_sig = $asig, "
                                "d.error_sig = $esig, d.hit_count = 1, "
                                "d.created_at = $ts, d.updated_at = $ts "
                                "ON MATCH SET d.hit_count = coalesce(d.hit_count, 0) + 1, "
                                "d.updated_at = $ts"
                            ),
                            "parameters": {
                                "ns": self.namespace,
                                "id": did,
                                "goal": goal_str,
                                "asig": action_str,
                                "esig": error_str,
                                "ts": now_ts,
                            },
                        }
                    ],
                    read_only=False,
                    timeout=2.5,
                )
                neo4j_synced = True
            except Exception as exc:
                neo4j_error = f"{type(exc).__name__}: {exc}"[:200]

        return _py_to_actor_value(
            {
                "ok": True,
                "id": did,
                "namespace": self.namespace,
                "goal": goal_str,
                "actionSig": action_str,
                "errorSig": error_str,
                "hitCount": hit_count,
                "neo4jSynced": neo4j_synced,
                "neo4jError": neo4j_error,
            }
        )

    @actor_query
    def recallDeadEnds(self, goal: str, limit: int = 5) -> Any:
        """Query `:ActorDeadEnd` nodes for `goal` (namespace-scoped, Neo4j + in-memory fallback)."""
        goal_str = str(goal or "").strip()
        cap = max(1, min(50, int(limit or 5)))

        merged: Dict[str, Dict[str, Any]] = {}
        neo4j_used = False

        if self.enable_neo4j:
            try:
                res = self._run_cypher_batch(
                    [
                        {
                            "statement": (
                                "MATCH (d:ActorDeadEnd {namespace: $ns}) "
                                "RETURN d.id AS id, d.goal AS goal, d.action_sig AS actionSig, "
                                "d.error_sig AS errorSig, coalesce(d.hit_count, 1) AS hitCount, "
                                "coalesce(d.updated_at, 0.0) AS updatedAt "
                                "ORDER BY d.updated_at DESC LIMIT 100"
                            ),
                            "parameters": {"ns": self.namespace},
                        }
                    ],
                    read_only=True,
                    timeout=2.5,
                )
                neo4j_used = True
                if res:
                    for row in res[0][1]:
                        rid = str(row.get("id") or "")
                        if rid:
                            merged[rid] = {
                                "id": rid,
                                "goal": str(row.get("goal") or ""),
                                "actionSig": str(row.get("actionSig") or ""),
                                "errorSig": str(row.get("errorSig") or ""),
                                "hitCount": int(row.get("hitCount") or 1),
                                "updatedAt": float(row.get("updatedAt") or 0.0),
                            }
            except Exception:
                neo4j_used = False

        for (ns, did), rec in self._local_dead_ends.items():
            if ns != self.namespace:
                continue
            if did not in merged:
                merged[did] = {
                    "id": rec["id"],
                    "goal": rec["goal"],
                    "actionSig": rec["actionSig"],
                    "errorSig": rec["errorSig"],
                    "hitCount": int(rec["hitCount"]),
                    "updatedAt": float(rec["updatedAt"]),
                }

        ranked: List[Tuple[float, int, float, Dict[str, Any]]] = []
        goal_lower = goal_str.lower()
        for rec in merged.values():
            rec_goal = rec["goal"]
            if not goal_lower:
                rel = 1.0
            elif rec_goal.lower() == goal_lower:
                rel = 1.0
            elif goal_lower in rec_goal.lower() or rec_goal.lower() in goal_lower:
                rel = 0.85
            else:
                rel = _token_overlap(goal_lower, rec_goal)
            if goal_lower and rel <= 0.0:
                continue
            entry = dict(rec)
            entry["relevance"] = round(rel, 6)
            ranked.append((rel, entry["hitCount"], entry["updatedAt"], entry))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        top_items = [item[3] for item in ranked[:cap]]

        return _py_to_actor_value(
            {
                "goal": goal_str,
                "namespace": self.namespace,
                "count": len(top_items),
                "deadEnds": top_items,
                "source": "neo4j+memory" if neo4j_used else "memory",
            }
        )

    @actor_query
    def describe(self) -> str:
        """Return concise self-documentation and Actor 4.1 usage examples."""
        return (
            f"ActivePlannerBridge (Actor[#Planner], backend={self.backend_name}, 0 MB VRAM)\n"
            "Active-Inference Expected Free Energy (EFE) controller + shadow_only VM counterfactual tree search.\n"
            "Methods:\n"
            "  - computeEFE(belief_json, actions_json) -> Dictionary(#bestAction, #mode, #scores, #backend, #beliefEntropy)\n"
            "  - recommendMode(uncertainty, steps_remaining, has_writes=false) -> Dictionary(#mode, #shouldTreeSearch, #reason)\n"
            "  - simulateBranches(candidates_json, verifier_chunk='', max_branches=4) -> Dictionary(#bestIndex, #bestAction, #branches, #prunedDeadEnds)\n"
            "  - recordDeadEnd(goal, action_sig, error_sig) -> Dictionary(#ok, #id, #hitCount, #neo4jSynced)\n"
            "  - recallDeadEnds(goal, limit=5) -> Dictionary(#count, #deadEnds, #source)\n"
            "  - getNamespace() / setNamespace(ns)\n"
            "Actor 4.1 Examples:\n"
            '  plan := computeEFE(Actor[#Planner], "{\\"qs\\": [0.5, 0.5]}", "[{\\"name\\": \\"probe\\", \\"kind\\": \\"epistemic\\"}, {\\"name\\": \\"commit\\", \\"kind\\": \\"pragmatic\\"}]")!!\n'
            "  rec := recommendMode(Actor[#Planner], 0.85, 5, true)!!\n"
            '  sim := simulateBranches(Actor[#Planner], "[\\"Actor[#x] := 41!!\\", \\"Actor[#x] := 42!!\\"]", "Actor[#x] == 42!!", 4)!!\n'
            '  recordDeadEnd(Actor[#Planner], "fix parser", "Actor[#x] := 1 / 0!!", "ZeroDivisionError")!!\n'
            '  recallDeadEnds(Actor[#Planner], "fix parser", 5)!!'
        )
