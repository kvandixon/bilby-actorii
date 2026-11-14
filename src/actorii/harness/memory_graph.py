"""Neo4j-Backed Computational Memory & Active Engram Subsystem (`Actor[#Memory]`).

Unifies Neo4j graph storage with live 1993 Whitewater Actor 4.1 OOP bytecode skills
(`:ActorEngram`), enabling:
1. Graph-First Skill Synthesis: Inspecting an existing Neo4j graph (`graphSchema`),
   querying it from Actor 4.1 (`cypherQuery`, `cypherQueryWith`), and saving reusable
   compiled Actor classes/methods (`persistEngram`, `saveEngram`) alongside the graph.
2. Chat-First Memory + Skill Creation: Writing structured facts from chat into Neo4j
   (`cypherWrite`, `cypherWriteWith`) together with the Actor 4.1 bytecode that reads
   and computes over those facts across future sessions and cold VM reboots.
3. Zero-Token Auto-Crystallization: Automatically filing out newly compiled user classes
   into Neo4j at the end of a turn with 0 extra LLM calls.
4. Graph-Linked Topology & Versioned Hot-Patching: Linking `(:ActorEngram)-[:OPERATES_ON]->(:ActorGraphDomain)`
   and `(:ActorEngram)-[:DEPENDS_ON]->(:ActorEngram)` for transitive multi-Engram hydration
   and `rollbackEngram` version recovery.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorObject, Symbol
from actorii.guardrails.fileout import ActorFileOutManager
from actorii.guardrails.linter import PROTECTED_SYSTEM_CLASSES
from actorii.parser import ClassDefNode, MethodDefNode, parse_actor
from actorii.vm import CompiledMethod

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"
)
EMBEDDING_API_URL = os.environ.get(
    "ACTORII_EMBED_ENDPOINT", "http://127.0.0.1:8084/v1/embeddings"
)
EMBEDDING_MODEL_NAME = os.environ.get("ACTORII_EMBED_MODEL", "bge-small-en-v1.5")

_SCRATCH_CLASS_PREFIXES = ("Tmp", "Scratch", "TaskContract", "Mock", "Ephemeral")
_EMBED_LRU: Dict[Tuple[str, str, str], List[float]] = {}


def _compute_embedding(text: str, timeout: float = 0.8) -> Optional[List[float]]:
    """Query the local (`bge-small-en-v1.5` or Ollama) or cloud embedding server; return None if offline."""
    clean = (text or "").strip()
    if not clean:
        return None
    emb_url = os.environ.get("ACTORII_EMBED_ENDPOINT", EMBEDDING_API_URL).strip()
    if not emb_url or emb_url.lower() in ("none", "off", "disabled"):
        return None
    emb_model = os.environ.get("ACTORII_EMBED_MODEL", EMBEDDING_MODEL_NAME).strip() or EMBEDDING_MODEL_NAME
    clipped = clean[:1000]
    lru_key = (emb_url, emb_model, clipped)
    cached_vec = _EMBED_LRU.get(lru_key)
    if cached_vec is not None:
        return cached_vec
    try:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        emb_key = (
            os.environ.get("ACTORII_EMBED_API_KEY", "").strip()
            or os.environ.get("ACTORII_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        if emb_key:
            headers["Authorization"] = f"Bearer {emb_key}"
        payload = json.dumps({"input": clipped, "model": emb_model}).encode("utf-8")
        req = urllib.request.Request(
            emb_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        emb = data.get("data", [{}])[0].get("embedding") or (data.get("embeddings") or [None])[0]
        if isinstance(emb, list) and emb:
            vec = [float(x) for x in emb]
            if len(_EMBED_LRU) >= 256:
                _EMBED_LRU.clear()
            _EMBED_LRU[lru_key] = vec
            return vec
    except Exception:
        return None
    return None


def _cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))

_INTERNAL_LABELS = frozenset(
    {
        "ActorSyntaxConstruct",
        "ActorBuiltinClass",
        "ActorPrimitive",
        "ActorErrorPattern",
        "ActorLearnedFix",
        "ActorEngram",
        "ActorEngramVersion",
        "ActorGraphDomain",
        "ActorEpisode",
        "STERule",
        "STEWord",
    }
)

_BUILTIN_POOL_KEYS = frozenset(
    {
        "Harness",
        "Bilby",
        "Sandbox",
        "Memory",
        "Grammar",
        "Bench",
        "Valkyrie",
        "System",
        "Thermometer",
        "ThermoUI",
        "Pantry",
        "ReverseEngineer",
        "REA",
        "Telegram",
        "Web",
        "Net",
        "PDF",
        "Hue",
        "Sam",
        "LLM",
        "Host",
        "VM",
    }
)

_CYPHER_LABEL_RE = re.compile(r"\(\s*[A-Za-z0-9_]*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")
_ACTOR_POOL_REF_RE = re.compile(r"Actor\s*\[\s*#([A-Za-z_][A-Za-z0-9_]*)\s*\]")
_INHERIT_PARSE_RE = re.compile(
    r"\binherit\s*\(\s*#?([A-Za-z_][A-Za-z0-9_]*)\s*,\s*#?([A-Za-z_][A-Za-z0-9_]*)\s*,\s*#\(\s*([^)]*)\)",
    re.DOTALL,
)
_IVAR_TOKEN_RE = re.compile(r"#?([A-Za-z_][A-Za-z0-9_]*)")
_SYSTEM_BASE_CLASSES = frozenset(PROTECTED_SYSTEM_CLASSES | {"Agent", "Object"})


def _extract_class_hierarchy_info(
    engram_name: str,
    actor_source: str,
    vm: Any = None,
) -> Tuple[str, List[str]]:
    """Extract `(superclass_name, declared_ivars)` from `actor_source` or the live VM hierarchy."""
    super_cls = "Object"
    ivars: List[str] = []
    if vm is not None and hasattr(vm, "hierarchy"):
        cls_obj = vm.hierarchy.get(engram_name)
        if cls_obj is not None:
            bases = getattr(cls_obj, "_bases", None)
            if bases:
                super_cls = _norm_sym(bases[0])
            raw_ivars = getattr(cls_obj, "ivars", None)
            if raw_ivars:
                ivars = [_norm_sym(x) for x in raw_ivars if _norm_sym(x)]
                return super_cls, ivars

    for m in _INHERIT_PARSE_RE.finditer(actor_source or ""):
        parent_cand, child_cand, ivar_blob = m.group(1), m.group(2), m.group(3)
        if child_cand == engram_name or super_cls == "Object":
            super_cls = _norm_sym(parent_cand) or "Object"
            ivars = [
                _norm_sym(tok)
                for tok in _IVAR_TOKEN_RE.findall(ivar_blob or "")
                if _norm_sym(tok)
            ]
            if child_cand == engram_name:
                break
    return super_cls, ivars


def _norm_sym(val: Any) -> str:
    if isinstance(val, Symbol):
        return val.name
    s = str(val or "").strip()
    return s[1:] if s.startswith("#") else s


def _py_to_actor_value(val: Any) -> Any:
    """Recursively convert Neo4j JSON values (dicts, lists, scalars) into native Actor objects."""
    if isinstance(val, dict):
        d = Dictionary(max(8, len(val) * 2))
        for k, v in val.items():
            conv_v = _py_to_actor_value(v)
            k_str = str(k)
            d.atPut(k_str, conv_v)
            d.atPut(Symbol(k_str), conv_v)
        return d
    if isinstance(val, list):
        arr = ActorArray(len(val))
        for idx, item in enumerate(val):
            arr.atPut(idx, _py_to_actor_value(item))
        return arr
    return val


def _actor_to_py_value(val: Any) -> Any:
    """Convert Actor `Dictionary`, `ActorArray`, or `Symbol` objects into JSON-serializable Python values."""
    if isinstance(val, Symbol):
        return val.name
    if isinstance(val, Dictionary):
        out: Dict[str, Any] = {}
        for k, v in val._entries.items():
            out[_norm_sym(k)] = _actor_to_py_value(v)
        return out
    if isinstance(val, ActorArray):
        return [_actor_to_py_value(x) for x in val]
    if isinstance(val, (list, tuple)):
        return [_actor_to_py_value(x) for x in val]
    if isinstance(val, dict):
        return {_norm_sym(k): _actor_to_py_value(v) for k, v in val.items()}
    return val


def _build_cypher_params(params: Tuple[Any, ...]) -> Dict[str, Any]:
    """Build a Neo4j parameter dictionary from either a `Dictionary`/`dict` or positional `$p1, $p2` args."""
    if not params:
        return {}
    if len(params) == 1 and isinstance(params[0], (Dictionary, dict)):
        raw_map = _actor_to_py_value(params[0])
        if isinstance(raw_map, dict):
            return raw_map
    out: Dict[str, Any] = {}
    for idx, p in enumerate(params, 1):
        py_val = _actor_to_py_value(p)
        out[f"p{idx}"] = py_val
        out[str(idx)] = py_val
        if idx == 1:
            out["param"] = py_val
            out["arg"] = py_val
    return out


def _extract_engram_graph_links(
    engram_name: str,
    actor_source: str,
) -> Tuple[List[str], List[str]]:
    """Extract referenced Neo4j domain node labels (`:Label`) and dependent `Actor[#OtherEngram]` keys."""
    labels: List[str] = []
    for m in _CYPHER_LABEL_RE.finditer(actor_source or ""):
        lbl = m.group(1)
        if lbl not in _INTERNAL_LABELS and lbl not in labels:
            labels.append(lbl)

    deps: List[str] = []
    for m in _ACTOR_POOL_REF_RE.finditer(actor_source or ""):
        ref_key = m.group(1)
        if ref_key != engram_name and ref_key not in _BUILTIN_POOL_KEYS and ref_key not in deps:
            deps.append(ref_key)

    return labels, deps


def _extract_engram_metadata(
    engram_name: str,
    actor_source: str,
) -> Tuple[str, List[str], str]:
    """Ensure `actor_source` instantiates `Actor[#engram_name]` and extract callable signatures."""
    src = (actor_source or "").strip()
    if src.startswith("```"):
        lines = [ln for ln in src.splitlines() if not ln.strip().startswith("```")]
        src = "\n".join(lines).strip()
    if not src.endswith("!!"):
        src = src.rstrip("!") + "!!"

    target_class = engram_name
    signatures: List[str] = []

    try:
        raw_chunks = [c.strip() + "!!" for c in src.split("!!") if c.strip()]
        for piece in raw_chunks:
            nodes = parse_actor(piece)
            for node in nodes:
                if isinstance(node, ClassDefNode):
                    target_class = _norm_sym(node.name)
                    for m in getattr(node, "methods", []):
                        sel = _norm_sym(m.name)
                        params = [_norm_sym(p) for p in getattr(m, "params", [])]
                        user_params = [p for p in params if p != "self"]
                        call_sig = (
                            f"{sel}(Actor[#{engram_name}], {', '.join(user_params)})"
                            if user_params
                            else f"{sel}(Actor[#{engram_name}])"
                        )
                        if call_sig not in signatures:
                            signatures.append(call_sig)
                elif hasattr(node, "target_class") and getattr(node, "target_class", None):
                    target_class = _norm_sym(getattr(node, "target_class"))
                elif isinstance(node, MethodDefNode):
                    sel = _norm_sym(node.name)
                    params = [_norm_sym(p) for p in getattr(node, "params", [])]
                    user_params = [p for p in params if p != "self"]
                    call_sig = (
                        f"{sel}(Actor[#{engram_name}], {', '.join(user_params)})"
                        if user_params
                        else f"{sel}(Actor[#{engram_name}])"
                    )
                    if call_sig not in signatures:
                        signatures.append(call_sig)
    except Exception:
        pass

    has_inherit = bool(re.search(r"\binherit\s*\(", src))
    has_now = bool(re.search(r"\bnow\s*\(", src))
    if not has_inherit and not has_now:
        src = (
            f"inherit(Object, #{engram_name}, #(#state), 1, nil)!!\n"
            f"now({engram_name})!!\n"
            f"{src}"
        )
        target_class = engram_name
    elif has_now and not has_inherit and target_class == engram_name:
        src = f"inherit(Object, #{engram_name}, #(#state), 1, nil)!!\n{src}"

    inst_pattern = rf"Actor\s*\[\s*#?{re.escape(engram_name)}\s*\]\s*:="
    if not re.search(inst_pattern, src) and engram_name != "AgentHarness":
        src = f"{src}\nActor[#{engram_name}] := new({target_class})!!"

    return target_class, signatures, src


@actor_bridge(name="ActorMemoryGraph")
class ActorMemoryGraph:
    """Neo4j-backed graph query & executable `:ActorEngram` persistence bridge (`Actor[#Memory]`)."""

    def __init__(
        self,
        vm: Any,
        tx_url: Optional[str] = None,
        enable_engrams: bool = True,
        auto_crystallize: bool = True,
        namespace: Optional[str] = None,
    ) -> None:
        self.vm = vm
        self.tx_url = tx_url or os.environ.get("ACTORII_NEO4J_HTTP_URL", NEO4J_HTTP_TX_URL)
        self.enable_engrams = bool(enable_engrams)
        self.auto_crystallize = bool(auto_crystallize)
        default_ns = os.environ.get("ACTORII_MEMORY_NAMESPACE", "default").strip() or "default"
        self.namespace = str(namespace or default_ns)
        self._write_epoch: int = 0
        self._engram_epoch: int = 0
        self._hydrate_cache: Dict[Tuple[str, str, int], str] = {}
        self._episode_rows_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._engram_rows_cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}
        self._product_rows_cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}

    @actor_query
    def getNamespace(self) -> str:
        """Return the active Neo4j memory namespace."""
        return self.namespace

    @actor_command
    def setNamespace(self, namespace: str) -> str:
        """Switch the active Neo4j memory namespace at runtime."""
        ns = _norm_sym(namespace).strip() or "default"
        self.namespace = ns
        self._write_epoch += 1
        self._engram_epoch += 1
        self._hydrate_cache.clear()
        self._episode_rows_cache.clear()
        self._engram_rows_cache.clear()
        self._product_rows_cache.clear()
        return f"NAMESPACE_SET:{self.namespace}"

    def _run_cypher(
        self,
        statement: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        stmt_upper = statement.upper()
        if any(kw in stmt_upper for kw in ("MERGE ", "MERGE(", "CREATE ", "CREATE(", "DELETE ", "SET ")):
            self._write_epoch += 1
            self._hydrate_cache.clear()
            if "MERGE (EP:ACTOREPISODE" not in stmt_upper:
                self._engram_epoch += 1
                self._episode_rows_cache.clear()
                self._engram_rows_cache.clear()
                self._product_rows_cache.clear()
        payload = {
            "statements": [
                {
                    "statement": statement,
                    "parameters": parameters or {},
                }
            ]
        }
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        neo_user = os.environ.get("ACTORII_NEO4J_USER", "").strip()
        neo_pass = os.environ.get("ACTORII_NEO4J_PASSWORD", "").strip()
        if neo_user and neo_pass:
            token = base64.b64encode(f"{neo_user}:{neo_pass}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            self.tx_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        if body.get("errors"):
            err_items = body["errors"]
            msg = "; ".join(f"{e.get('code', 'CypherError')}: {e.get('message', '')}" for e in err_items)
            raise RuntimeError(f"Neo4j Cypher error: {msg}")
        results = body.get("results", [])
        if not results:
            return [], []
        cols = results[0].get("columns", [])
        rows: List[Dict[str, Any]] = []
        for item in results[0].get("data", []):
            row_vals = item.get("row", [])
            rows.append(dict(zip(cols, row_vals)))
        return cols, rows

    def _rows_to_actor_array(self, raw_rows: List[Dict[str, Any]]) -> ActorArray:
        arr = ActorArray(len(raw_rows))
        for idx, row_dict in enumerate(raw_rows):
            actor_row = Dictionary(max(8, len(row_dict) * 4))
            for col_key, cell_val in row_dict.items():
                conv_val = _py_to_actor_value(cell_val)
                col_str = str(col_key)
                actor_row.atPut(col_str, conv_val)
                actor_row.atPut(Symbol(col_str), conv_val)
                if "." in col_str:
                    short_key = col_str.rsplit(".", 1)[-1].strip()
                    if short_key and actor_row.at(short_key) is None:
                        actor_row.atPut(short_key, conv_val)
                        actor_row.atPut(Symbol(short_key), conv_val)
                if isinstance(cell_val, dict):
                    for pk, pv in cell_val.items():
                        pk_str = str(pk)
                        if actor_row.at(pk_str) is None:
                            conv_pv = _py_to_actor_value(pv)
                            actor_row.atPut(pk_str, conv_pv)
                            actor_row.atPut(Symbol(pk_str), conv_pv)
            arr.atPut(idx, actor_row)
        return arr

    # =========================================================================
    # Tier 0: Read-Only Graph Introspection & Cypher Queries (`@actor_query`)
    # =========================================================================

    @actor_query
    def graphSchema(self) -> str:
        """Inspect domain node labels, sample properties, and relationship patterns in Neo4j."""
        try:
            _, label_rows = self._run_cypher(
                """
                MATCH (n)
                UNWIND labels(n) AS lbl
                WITH lbl, count(n) AS cnt, collect(keys(n))[0..3] AS key_samples, collect(properties(n))[0] AS sample_props
                RETURN lbl, cnt, key_samples, sample_props
                ORDER BY lbl
                """
            )
            domain_labels = [
                r for r in label_rows if str(r.get("lbl", "")) not in _INTERNAL_LABELS
            ]
            _, rel_rows = self._run_cypher(
                """
                MATCH (a)-[r]->(b)
                WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS dst, count(r) AS cnt, collect(keys(r))[0] AS rkeys
                WHERE NOT src IN $internal AND NOT dst IN $internal
                RETURN src, rel, dst, cnt, rkeys
                ORDER BY src, rel, dst
                """,
                {"internal": list(_INTERNAL_LABELS)},
            )
            lines = ["[Neo4j Domain Graph Schema]:"]
            if not domain_labels:
                lines.append("  Nodes: (no domain nodes found)")
            else:
                for r in domain_labels:
                    lbl = r["lbl"]
                    cnt = r["cnt"]
                    props = r.get("sample_props") or {}
                    prop_preview = ", ".join(f"{k}={v!r}" for k, v in list(props.items())[:6])
                    lines.append(f"  - (:{lbl}) [count={cnt}] sample properties: {{{prop_preview}}}")
            if rel_rows:
                lines.append("  Relationships:")
                for rr in rel_rows:
                    rkeys = rr.get("rkeys") or []
                    rk_str = f" props={rkeys}" if rkeys else ""
                    lines.append(
                        f"  - (:{rr['src']})-[:{rr['rel']}{rk_str}]->(:{rr['dst']}) [count={rr['cnt']}]"
                    )
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR: Neo4j schema introspection failed: {exc}"

    def _record_runtime_cypher_domains(self, cypher_text: str) -> None:
        """Inspect `self.vm.call_stack` at runtime to bind the calling Ring 2 Engram to queried `:ActorGraphDomain` labels."""
        if not self.enable_engrams or getattr(self.vm, "_in_shadow_run", False):
            return
        labels = [
            m.group(1)
            for m in _CYPHER_LABEL_RE.finditer(cypher_text or "")
            if m.group(1) not in _INTERNAL_LABELS
        ]
        if not labels:
            return
        unique_labels = list(dict.fromkeys(labels))

        caller_engram: Optional[str] = None
        for frame in reversed(getattr(self.vm, "call_stack", [])):
            self_obj = getattr(frame, "receiver", None)
            cname = _norm_sym(
                getattr(self_obj, "class_name", "")
                if isinstance(self_obj, ActorObject)
                else getattr(frame, "class_name", "")
            )
            if (
                cname
                and cname not in PROTECTED_SYSTEM_CLASSES
                and cname not in ("AgentHarness", "<Chunk>", "Object")
                and not getattr(self.vm.hierarchy.get(cname), "is_bridge_class", False)
            ):
                caller_engram = cname
                break

        if not caller_engram:
            return

        try:
            self._run_cypher(
                """
                MATCH (e:ActorEngram {name: $name, namespace: $ns})
                UNWIND $labels AS lbl
                MERGE (d:ActorGraphDomain {label: lbl, namespace: $ns})
                MERGE (e)-[:OPERATES_ON]->(d)
                """,
                {"name": caller_engram, "ns": self.namespace, "labels": unique_labels},
            )
        except Exception:
            pass

    def _hydrate_engram_source_safely(self, engram_name: str, actor_src: str) -> Any:
        """Compile `actor_src` into `self.vm` with warm-image instance reconciliation and `initializeClass` hook."""
        norm_name = _norm_sym(engram_name)
        actor_pool = self.vm.global_vars.get("Actor", {})
        existing_inst = actor_pool.get(norm_name) or actor_pool.get(Symbol(norm_name))
        saved_fields = (
            dict(getattr(existing_inst, "_fields", {}))
            if isinstance(existing_inst, ActorObject)
            else {}
        )

        prev_lock = getattr(self.vm, "lock_system_singletons", False)
        self.vm.lock_system_singletons = False
        try:
            self.vm.eval_chunk(actor_src)
        finally:
            self.vm.lock_system_singletons = prev_lock

        cls_obj = self.vm.hierarchy.get(norm_name)
        inst = actor_pool.get(norm_name) or actor_pool.get(Symbol(norm_name))

        # Reconcile warm instance state if an instance already existed before hydration
        if isinstance(existing_inst, ActorObject) and cls_obj is not None:
            declared_ivars = set(getattr(cls_obj, "ivars", []) or [])
            if isinstance(inst, ActorObject) and inst is not existing_inst:
                for k, v in saved_fields.items():
                    if v is not None and (not declared_ivars or k in declared_ivars) and inst._fields.get(k) is None:
                        inst._fields[k] = v
            else:
                existing_inst._parent = cls_obj
                existing_inst.inherit_from(cls_obj)
                for k, v in saved_fields.items():
                    if v is not None and (not declared_ivars or k in declared_ivars):
                        existing_inst._fields[k] = v
                actor_pool[norm_name] = existing_inst
                inst = existing_inst

        # Invoke optional `initializeClass` lifecycle hook if defined on the Engram class
        if cls_obj is not None and "initializeClass" in getattr(cls_obj, "_methods", {}):
            try:
                self.vm.eval_chunk(f"initializeClass(Actor[#{norm_name}])!!")
            except Exception:
                pass

        return inst

    @actor_query
    def cypherQuery(self, query: str) -> ActorArray:
        """Run a read Cypher query and return a 0-based Actor `Array` of `Dictionary` rows."""
        q_str = str(query or "").strip()
        if not q_str:
            return ActorArray(0)
        self._record_runtime_cypher_domains(q_str)
        _, raw_rows = self._run_cypher(q_str)
        return self._rows_to_actor_array(raw_rows)

    @actor_query
    def cypherQueryWith(self, query: str, *params: Any) -> ActorArray:
        """Run a parameterized read Cypher query (`$p1, $p2` or a `Dictionary`) and return an Actor `Array` of `Dictionary` rows."""
        q_str = str(query or "").strip()
        if not q_str:
            return ActorArray(0)
        self._record_runtime_cypher_domains(q_str)
        param_map = _build_cypher_params(params)
        _, raw_rows = self._run_cypher(q_str, param_map)
        return self._rows_to_actor_array(raw_rows)

    @actor_query
    def loadEngram(self, name: Any) -> Any:
        """Hydrate a saved `:ActorEngram` (and any transitive `:SUBCLASS_OF` / `:DEPENDS_ON` Engrams) from Neo4j into `self.vm`."""
        if not self.enable_engrams:
            raise RuntimeError("Active Engrams are disabled in this harness configuration.")
        norm_name = _norm_sym(name)
        _, rows = self._run_cypher(
            """
            MATCH (e:ActorEngram {name: $name, namespace: $ns})
            OPTIONAL MATCH (e)-[:SUBCLASS_OF|DEPENDS_ON*1..5]->(dep:ActorEngram {namespace: $ns})
            RETURN e.name AS name, e.actor_source AS actor_source,
                   collect(DISTINCT {name: dep.name, actor_source: dep.actor_source}) AS deps
            LIMIT 1
            """,
            {"name": norm_name, "ns": self.namespace},
        )
        if not rows:
            raise KeyError(f"No `:ActorEngram` named '{norm_name}' found in Neo4j (namespace={self.namespace!r}).")

        row = rows[0]
        # Hydrate prerequisite superclasses and dependencies first
        for dep in reversed(row.get("deps") or []):
            dep_name = dep.get("name")
            dep_src = dep.get("actor_source")
            if dep_name and dep_src:
                self._hydrate_engram_source_safely(str(dep_name), str(dep_src))

        actor_src = str(row.get("actor_source") or "")
        return self._hydrate_engram_source_safely(norm_name, actor_src)

    @actor_query
    def listEngrams(self) -> str:
        """List all saved `:ActorEngram` computational memories in Neo4j along with their superclass, ivars, domains, and dependencies."""
        if not self.enable_engrams:
            return "Active Engrams disabled."
        try:
            _, rows = self._run_cypher(
                """
                MATCH (e:ActorEngram {namespace: $ns})
                OPTIONAL MATCH (e)-[:OPERATES_ON]->(d:ActorGraphDomain {namespace: $ns})
                OPTIONAL MATCH (e)-[:DEPENDS_ON]->(dep:ActorEngram {namespace: $ns})
                OPTIONAL MATCH (e)-[:HAS_PREVIOUS_VERSION]->(v:ActorEngramVersion)
                RETURN e.name AS name, e.description AS description,
                       coalesce(e.superclass, 'Object') AS superclass,
                       coalesce(e.ivars, []) AS ivars,
                       e.signatures AS signatures, coalesce(e.version, 1) AS version,
                       collect(DISTINCT d.label) AS labels,
                       collect(DISTINCT dep.name) AS deps,
                       count(DISTINCT v) AS prev_versions
                ORDER BY e.name
                """,
                {"ns": self.namespace},
            )
            if not rows:
                return "No `:ActorEngram` skills saved in Neo4j."
            lines = ["[Neo4j Saved `:ActorEngram` Memory-Skills]:"]
            for r in rows:
                sigs = ", ".join(r.get("signatures") or [])
                lbls = [x for x in (r.get("labels") or []) if x]
                deps = [x for x in (r.get("deps") or []) if x]
                sup = str(r.get("superclass") or "Object")
                ivs = [x for x in (r.get("ivars") or []) if x]
                meta_parts = [f"v{r.get('version', 1)}", f"extends={sup}"]
                if ivs:
                    meta_parts.append(f"ivars={ivs}")
                if lbls:
                    meta_parts.append(f"domains={lbls}")
                if deps:
                    meta_parts.append(f"depends_on={deps}")
                if r.get("prev_versions", 0) > 0:
                    meta_parts.append(f"history={r['prev_versions']}")
                meta_str = ", ".join(meta_parts)
                lines.append(
                    f"  - Actor[#{r['name']}] [{meta_str}]: {r.get('description', '')} | Calls: {sigs}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR listing Engrams: {exc}"

    @actor_query
    def classHierarchyTree(self) -> str:
        """Return a multi-level OOP class hierarchy tree of all foundational & Neo4j `:ActorEngram` classes."""
        try:
            _, rows = self._run_cypher(
                """
                MATCH (e:ActorEngram {namespace: $ns})
                WHERE e.name <> 'AgentHarness'
                RETURN e.name AS name,
                       coalesce(e.superclass, 'Object') AS superclass,
                       coalesce(e.ivars, []) AS ivars,
                       coalesce(e.signatures, []) AS signatures
                ORDER BY superclass, name
                """,
                {"ns": self.namespace},
            )
        except Exception:
            rows = []

        children_map: Dict[str, List[Dict[str, Any]]] = {}
        known_names: Set[str] = set()
        for r in rows:
            nm = str(r.get("name") or "")
            if not nm:
                continue
            known_names.add(nm)
            sup = str(r.get("superclass") or "Object")
            children_map.setdefault(sup, []).append(r)

        # Ensure foundational base classes from live VM hierarchy are represented even if not yet queried
        for base_cls in ("DomainEntity", "GraphService", "MetricEvaluator", "CompositePipeline"):
            if base_cls not in known_names and self.vm.hierarchy.get(base_cls) is not None:
                cls_obj = self.vm.hierarchy.get(base_cls)
                sup_name = cls_obj._bases[0] if getattr(cls_obj, "_bases", None) else "Object"
                ivs = list(getattr(cls_obj, "ivars", []) or [])
                sels = [s for s in getattr(cls_obj, "_class_methods", {}).keys() if s != "new"]
                children_map.setdefault(sup_name, []).append(
                    {
                        "name": base_cls,
                        "superclass": sup_name,
                        "ivars": ivs,
                        "signatures": [f"{s}(self, ...)" for s in sels[:5]],
                    }
                )
                known_names.add(base_cls)

        lines = ["[LIVE 1993 ACTOR 4.1 OOP CLASS HIERARCHY (`Object` Root)]:"]

        def _render_branch(parent_name: str, depth: int, visited: Set[str]) -> None:
            kids = children_map.get(parent_name, [])
            # Sort so subclasses that themselves have children appear first
            kids_sorted = sorted(
                kids,
                key=lambda item: (0 if item["name"] in children_map else 1, item["name"]),
            )
            for item in kids_sorted:
                cname = item["name"]
                if cname in visited:
                    continue
                visited.add(cname)
                indent = "  " * depth
                ivs = [f"#{x}" for x in (item.get("ivars") or []) if x]
                iv_str = f"#({' '.join(ivs)})" if ivs else "#()"
                sigs = item.get("signatures") or []
                short_sigs = ", ".join(s.split("(")[0] for s in sigs[:5])
                sig_tag = f" | methods: {short_sigs}" if short_sigs else ""
                lines.append(f"{indent}└── {cname} (ivars: {iv_str}{sig_tag})")
                _render_branch(cname, depth + 1, visited)

        visited_nodes: Set[str] = set()
        _render_branch("Object", 1, visited_nodes)
        # Render any custom roots whose superclass isn't Object
        for sup_key in sorted(children_map.keys()):
            if sup_key != "Object" and sup_key not in visited_nodes:
                lines.append(f"  └── {sup_key}")
                _render_branch(sup_key, 2, visited_nodes)
        return "\n".join(lines)

    # =========================================================================
    # Tier 2: Graph Mutations, Engram Persistence & Version Rollback (`@actor_command`)
    # =========================================================================

    @actor_command
    def cypherWrite(self, query: str) -> int:
        """Execute a mutating Cypher statement (`MERGE`, `CREATE`, `SET`, `DELETE`) in Neo4j."""
        q_str = str(query or "").strip()
        if not q_str:
            return 0
        self._record_runtime_cypher_domains(q_str)
        _, rows = self._run_cypher(q_str)
        return len(rows) if rows else 1

    @actor_command
    def cypherWriteWith(self, query: str, *params: Any) -> int:
        """Execute a parameterized mutating Cypher statement (`$p1, $p2` or a `Dictionary`) in Neo4j."""
        q_str = str(query or "").strip()
        if not q_str:
            return 0
        self._record_runtime_cypher_domains(q_str)
        param_map = _build_cypher_params(params)
        _, rows = self._run_cypher(q_str, param_map)
        return len(rows) if rows else 1

    @actor_command
    def persistEngram(self, name: Any, description: str = "") -> str:
        """File-out an already-compiled live `ActorClass` from `self.vm.hierarchy` into Neo4j as an `:ActorEngram`."""
        if not self.enable_engrams:
            raise RuntimeError("Active Engrams are disabled in this harness configuration.")
        norm_name = _norm_sym(name)
        if isinstance(name, ActorObject):
            norm_name = _norm_sym(getattr(name, "class_name", norm_name))
        elif hasattr(name, "name") and not isinstance(name, Symbol):
            norm_name = _norm_sym(getattr(name, "name"))

        mgr = ActorFileOutManager("/tmp/actorii_engram_fileout")
        raw_cls_source = mgr.format_class_source(self.vm, norm_name)
        return self.saveEngram(norm_name, description or f"Engram skill {norm_name}", raw_cls_source)

    @actor_command
    def saveEngram(self, name: Any, description: str = "", actor_source: str = "") -> str:
        """Compile & verify a 1993 Actor 4.1 class/object in the VM and persist it as a versioned `:ActorEngram` in Neo4j."""
        if not self.enable_engrams:
            raise RuntimeError("Active Engrams are disabled in this harness configuration.")
        norm_name = _norm_sym(name)
        if isinstance(name, ActorObject):
            norm_name = _norm_sym(getattr(name, "class_name", norm_name))
        elif hasattr(name, "name") and not isinstance(name, Symbol):
            norm_name = _norm_sym(getattr(name, "name"))

        desc_str = str(description or f"Engram skill {norm_name}").strip()
        src_arg = str(actor_source or "").strip()
        if not src_arg and self.vm.hierarchy.get(norm_name) is not None:
            mgr = ActorFileOutManager("/tmp/actorii_engram_fileout")
            src_arg = mgr.format_class_source(self.vm, norm_name)

        target_cls, signatures, normalized_src = _extract_engram_metadata(norm_name, src_arg)
        domain_labels, dep_engrams = _extract_engram_graph_links(norm_name, normalized_src)

        # Compile and execute inside the live VM first so any syntax/lint error is caught immediately
        self._hydrate_engram_source_safely(norm_name, normalized_src)

        # If the Engram class defines `selfTest(self)`, automatically run it as a pre-commit verification gate!
        cls_obj = self.vm.hierarchy.get(target_cls)
        if cls_obj is not None and "selfTest" in getattr(cls_obj, "_class_methods", {}):
            try:
                test_res = self.vm.eval_transactional(
                    f"selfTest(Actor[#{norm_name}])!!",
                    max_opcodes=25000,
                    shadow_only=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"EngramSelfTestError: `{target_cls}>>selfTest` raised {type(exc).__name__}: {exc}. "
                    f"Fix the method bug in `{target_cls}` before calling `persistEngram` / `saveEngram`!"
                ) from exc
            if test_res is False or test_res is None or test_res == 0 or str(test_res).startswith("ERROR"):
                raise RuntimeError(
                    f"EngramSelfTestError: `{target_cls}>>selfTest` failed verification (returned {test_res!r}). "
                    f"Fix `{target_cls}` so `selfTest(Actor[#{norm_name}])!!` returns `true` before saving!"
                )

        super_cls, ivars = _extract_class_hierarchy_info(target_cls, normalized_src, self.vm)
        if super_cls not in _SYSTEM_BASE_CLASSES and super_cls != norm_name and super_cls not in dep_engrams:
            dep_engrams.append(super_cls)

        emb_text = (
            f"{norm_name} (extends {super_cls}): {desc_str} | "
            f"{' '.join(signatures)} | {' '.join(domain_labels)} | {' '.join(ivars)}"
        )
        embedding = _compute_embedding(emb_text) or []

        # Archive previous version if `actor_source` changed, then upsert `:ActorEngram` and graph edges
        self._run_cypher(
            """
            MERGE (e:ActorEngram {name: $name, namespace: $ns})
            WITH e, e.actor_source AS old_src, e.signatures AS old_sigs, coalesce(e.version, 0) AS old_ver
            FOREACH (_ IN CASE WHEN old_src IS NOT NULL AND old_src <> $src THEN [1] ELSE [] END |
                CREATE (v:ActorEngramVersion {
                    name: $name,
                    namespace: $ns,
                    version: old_ver,
                    actor_source: old_src,
                    signatures: coalesce(old_sigs, []),
                    archived_at: timestamp()
                })
                MERGE (e)-[:HAS_PREVIOUS_VERSION]->(v)
            )
            SET e.target_class = $target_cls,
                e.superclass = $super_cls,
                e.ivars = $ivars,
                e.description = CASE WHEN $desc <> '' THEN $desc ELSE coalesce(e.description, $name) END,
                e.actor_source = $src,
                e.signatures = $sigs,
                e.embedding = CASE WHEN size($embedding) > 0 THEN $embedding ELSE coalesce(e.embedding, []) END,
                e.call_count = coalesce(e.call_count, 0),
                e.success_count = coalesce(e.success_count, 0),
                e.failure_count = coalesce(e.failure_count, 0),
                e.version = CASE WHEN old_src IS NULL THEN 1 WHEN old_src <> $src THEN old_ver + 1 ELSE old_ver END,
                e.updated_at = timestamp()
            WITH e
            OPTIONAL MATCH (e)-[old_op:OPERATES_ON]->()
            DELETE old_op
            WITH e
            OPTIONAL MATCH (e)-[old_dep:DEPENDS_ON]->()
            DELETE old_dep
            WITH e
            OPTIONAL MATCH (e)-[old_sub:SUBCLASS_OF]->()
            DELETE old_sub
            """,
            {
                "name": norm_name,
                "ns": self.namespace,
                "target_cls": target_cls,
                "super_cls": super_cls,
                "ivars": ivars,
                "desc": desc_str,
                "src": normalized_src,
                "sigs": signatures,
                "embedding": embedding,
            },
        )

        if super_cls and super_cls not in _SYSTEM_BASE_CLASSES and super_cls != norm_name:
            self._run_cypher(
                """
                MATCH (e:ActorEngram {name: $name, namespace: $ns})
                MATCH (parent:ActorEngram {name: $parent_name, namespace: $ns})
                MERGE (e)-[:SUBCLASS_OF]->(parent)
                """,
                {"name": norm_name, "ns": self.namespace, "parent_name": super_cls},
            )

        if domain_labels:
            self._run_cypher(
                """
                MATCH (e:ActorEngram {name: $name, namespace: $ns})
                UNWIND $labels AS lbl
                MERGE (d:ActorGraphDomain {label: lbl, namespace: $ns})
                MERGE (e)-[:OPERATES_ON]->(d)
                """,
                {"name": norm_name, "ns": self.namespace, "labels": domain_labels},
            )

        if dep_engrams:
            self._run_cypher(
                """
                MATCH (e:ActorEngram {name: $name, namespace: $ns})
                UNWIND $deps AS dep_name
                MATCH (target:ActorEngram {name: dep_name, namespace: $ns})
                MERGE (e)-[:DEPENDS_ON]->(target)
                """,
                {"name": norm_name, "ns": self.namespace, "deps": dep_engrams},
            )

        sig_summary = ", ".join(signatures) if signatures else f"Actor[#{norm_name}]"
        return f"ENGRAM_SAVED: Actor[#{norm_name}] ({sig_summary})"

    @actor_command
    def recordEngramUsage(self, name: Any, success: bool = True) -> str:
        """Update execution telemetry (`call_count`, `success_count`, `failure_count`, `last_used_at`) on `:ActorEngram`."""
        norm_name = _norm_sym(name)
        is_ok = bool(success)
        self._run_cypher(
            """
            MATCH (e:ActorEngram {name: $name, namespace: $ns})
            SET e.call_count = coalesce(e.call_count, 0) + 1,
                e.success_count = coalesce(e.success_count, 0) + CASE WHEN $ok THEN 1 ELSE 0 END,
                e.failure_count = coalesce(e.failure_count, 0) + CASE WHEN $ok THEN 0 ELSE 1 END,
                e.last_used_at = timestamp()
            """,
            {"name": norm_name, "ns": self.namespace, "ok": is_ok},
        )
        return f"TELEMETRY_RECORDED: Actor[#{norm_name}] (success={is_ok})"

    @actor_command
    def rollbackEngram(self, name: Any) -> str:
        """Restore the most recent `:ActorEngramVersion` for `name` in Neo4j and re-compile it into `self.vm`."""
        if not self.enable_engrams:
            raise RuntimeError("Active Engrams are disabled in this harness configuration.")
        norm_name = _norm_sym(name)
        _, rows = self._run_cypher(
            """
            MATCH (e:ActorEngram {name: $name, namespace: $ns})-[:HAS_PREVIOUS_VERSION]->(v:ActorEngramVersion)
            WITH e, v ORDER BY v.version DESC, v.archived_at DESC LIMIT 1
            SET e.actor_source = v.actor_source,
                e.signatures = v.signatures,
                e.version = v.version,
                e.updated_at = timestamp()
            WITH e, v, v.actor_source AS restored_src, v.version AS restored_ver
            DETACH DELETE v
            RETURN restored_src, restored_ver
            """,
            {"name": norm_name, "ns": self.namespace},
        )
        if not rows:
            raise KeyError(f"No previous version found to roll back for `:ActorEngram` '{norm_name}'.")
        restored_src = str(rows[0]["restored_src"])
        restored_ver = rows[0]["restored_ver"]
        cls_obj = self.vm.hierarchy.get(norm_name)
        if cls_obj is not None and not getattr(cls_obj, "is_bridge_class", False):
            new_ctor = cls_obj._methods.get("new")
            cls_obj._methods.clear()
            cls_obj._class_methods.clear()
            if new_ctor is not None:
                cls_obj._methods["new"] = new_ctor
        prev_lock = getattr(self.vm, "lock_system_singletons", False)
        self.vm.lock_system_singletons = False
        try:
            self.vm.eval_chunk(restored_src)
        finally:
            self.vm.lock_system_singletons = prev_lock
        if norm_name == "AgentHarness":
            h_obj = self.vm.global_vars.get("Actor", {}).get("Harness")
            if h_obj is not None and cls_obj is not None:
                h_obj._methods = {
                    k: v for k, v in h_obj._methods.items() if k in getattr(h_obj, "_instance_methods", set())
                }
                h_obj.inherit_from(cls_obj)
        return f"ENGRAM_ROLLED_BACK: Actor[#{norm_name}] (version={restored_ver})"

    @actor_command
    def clearEngrams(self) -> int:
        """Delete all `:ActorEngram`, `:ActorEngramVersion`, and `:ActorGraphDomain` nodes in the active namespace."""
        _, rows = self._run_cypher(
            """
            MATCH (n {namespace: $ns})
            WHERE n:ActorEngram OR n:ActorEngramVersion OR n:ActorGraphDomain
            WITH n, count(n) AS c
            DETACH DELETE n
            RETURN c
            """,
            {"ns": self.namespace},
        )
        return int(rows[0]["c"]) if rows and "c" in rows[0] else 0

    @actor_command
    def recordEpisode(
        self,
        session_id: str,
        session_date: str,
        content: str,
        turn_range: str = "",
        embed_text: str = "",
    ) -> str:
        """Archive a compacted conversation episode into Neo4j (`:ActorEpisode`) with a `:8084` (`bge-small-en-v1.5`) vector embedding."""
        clean_text = str(content or "").strip()
        if not clean_text:
            return "EMPTY_EPISODE"
        s_id = str(session_id or "session_0").strip()
        s_date = str(session_date or "").strip()
        t_rng = str(turn_range or "").strip()
        raw_emb_src = str(embed_text or clean_text).strip()
        emb_input = f"[{s_date}] {raw_emb_src[:950]}" if s_date else raw_emb_src[:950]
        emb = _compute_embedding(emb_input) or []
        ep_hash = hashlib.sha256(f"{self.namespace}|{s_id}|{t_rng}|{clean_text}".encode("utf-8")).hexdigest()[:32]
        self._run_cypher(
            """
            MERGE (ep:ActorEpisode {namespace: $ns, episode_hash: $ehash})
            SET ep.session_id = $sid,
                ep.session_date = $sdate,
                ep.turn_range = $trng,
                ep.content = $content,
                ep.embedding = $emb,
                ep.created_at = coalesce(ep.created_at, timestamp()),
                ep.updated_at = timestamp()
            """,
            {
                "ns": self.namespace,
                "ehash": ep_hash,
                "sid": s_id,
                "sdate": s_date,
                "trng": t_rng,
                "content": clean_text,
                "emb": emb,
            },
        )
        cached_ep = self._episode_rows_cache.get(self.namespace)
        if cached_ep is not None:
            cached_ts, cached_list = cached_ep
            new_row = {
                "episode_hash": ep_hash,
                "session_id": s_id,
                "session_date": s_date,
                "turn_range": t_rng,
                "content": clean_text,
                "embedding": emb,
                "created_at": int(time.time() * 1000),
            }
            replaced = False
            for idx_r, r_existing in enumerate(cached_list):
                if r_existing.get("episode_hash") == ep_hash:
                    cached_list[idx_r] = new_row
                    replaced = True
                    break
            if not replaced:
                cached_list.append(new_row)
            self._episode_rows_cache[self.namespace] = (cached_ts, cached_list)
        return f"EPISODE_ARCHIVED:{s_id}"

    @actor_command
    def clearEpisodes(self) -> int:
        """Delete all `:ActorEpisode` nodes in the active namespace."""
        self._episode_rows_cache.pop(self.namespace, None)
        _, rows = self._run_cypher(
            """
            MATCH (ep:ActorEpisode {namespace: $ns})
            WITH ep, count(ep) AS c
            DETACH DELETE ep
            RETURN c
            """,
            {"ns": self.namespace},
        )
        self._episode_rows_cache.pop(self.namespace, None)
        return int(rows[0]["c"]) if rows and "c" in rows[0] else 0

    @actor_query
    def recallEpisodes(self, query_text: str, top_k: int = 5, min_score: float = 0.0) -> str:
        """Retrieve top-K dated `:ActorEpisode` memories from Neo4j via hybrid `:8084` vector + lexical + chronological ranking."""
        import os as _os

        if _os.environ.get("ACTORII_BENCH_STRICT") == "1" and self.namespace == "default":
            return ""
        q_str = str(query_text or "").strip()
        if not q_str:
            return ""
        try:
            now_mono = time.monotonic()
            cached_ep = self._episode_rows_cache.get(self.namespace)
            if cached_ep is not None and (now_mono - cached_ep[0]) <= 90.0:
                rows = cached_ep[1]
            else:
                _, rows = self._run_cypher(
                    """
                    MATCH (ep:ActorEpisode {namespace: $ns})
                    RETURN coalesce(ep.episode_hash, '') AS episode_hash,
                           ep.session_id AS session_id,
                           coalesce(ep.session_date, '') AS session_date,
                           coalesce(ep.turn_range, '') AS turn_range,
                           ep.content AS content,
                           ep.embedding AS embedding,
                           coalesce(ep.created_at, 0) AS created_at
                    ORDER BY created_at ASC
                    """,
                    {"ns": self.namespace},
                )
                self._episode_rows_cache[self.namespace] = (now_mono, list(rows))
            if not rows:
                return ""
            raw_q = q_str.rsplit(":", 1)[-1].strip() if ":" in q_str and len(q_str.rsplit(":", 1)[-1].strip()) >= 8 else q_str
            stop_w = {
                "the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "were",
                "what", "when", "where", "which", "who", "how", "many", "much", "did", "does", "have",
                "has", "had", "can", "you", "your", "our", "previous", "chat", "conversation", "tell",
                "remind", "about", "after", "before", "first", "last", "answer", "question", "based",
                "sessions", "fact", "changed", "across", "dates", "use", "most", "recent", "update",
                "says", "yesterday", "week", "compute", "exact", "calendar", "date", "session",
                "header", "questions", "multiple", "carefully", "count", "every", "distinct",
                "mentioned", "retrieved", "contains", "false", "premise", "someone", "something",
                "state", "correction", "clearly", "current", "would", "likely",
            }
            q_words = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9_]+", raw_q)
                if len(w) >= 3 and w.lower() not in stop_w
            }
            q_emb = _compute_embedding(raw_q)
            total_eps = max(1, len(rows))
            scored: List[Tuple[float, int, Dict[str, Any]]] = []
            for idx, r in enumerate(rows):
                c_txt = str(r.get("content") or "")
                s_date = str(r.get("session_date") or "")
                user_part = c_txt.split(" | Assistant:", 1)[0]
                hay_user = f"{s_date} {user_part}".lower()
                hay_full = f"{s_date} {c_txt}".lower()
                lex_hits = 0.0
                for w in q_words:
                    if w in hay_user or (len(w) >= 5 and w[:5] in hay_user):
                        lex_hits += 1.0
                    elif w in hay_full or (len(w) >= 5 and w[:5] in hay_full):
                        lex_hits += 0.45
                lex_score = (lex_hits / max(1, len(q_words))) if q_words else 0.0
                emb_score = _cosine_similarity(q_emb, r.get("embedding"))
                t_rng_str = str(r.get("turn_range") or "")
                t1_bonus = 0.035 if t_rng_str == "T1" else 0.0
                recency_bonus = 0.01 * (idx / total_eps)
                score = 0.52 * emb_score + 0.47 * lex_score + t1_bonus + recency_bonus
                scored.append((score, idx, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            k = max(1, int(top_k or 12))
            floor = float(min_score or 0.0)
            selected_map: Dict[int, Tuple[float, int, Dict[str, Any]]] = {}
            seen_sessions = set()
            # Ensure the highest-scoring episode from each distinct session with non-trivial relevance is represented
            for item_tuple in scored:
                sc_val, i_val, r_dict = item_tuple
                if sc_val < max(0.25, floor):
                    continue
                sid_val = str(r_dict.get("session_id") or "")
                if sid_val and sid_val not in seen_sessions and len(selected_map) < max(2, k // 2):
                    seen_sessions.add(sid_val)
                    selected_map[i_val] = item_tuple
            for item_tuple in scored:
                if len(selected_map) >= k:
                    break
                sc_val, i_val, _ = item_tuple
                if sc_val < floor:
                    continue
                if i_val not in selected_map:
                    selected_map[i_val] = item_tuple

            if not selected_map:
                return ""

            top_items = sorted(selected_map.values(), key=lambda x: x[1])
            lines = ["[ARCHIVED EPISODIC MEMORY RECALL (NEO4J `:ActorEpisode` + `:8084` VECTOR HYBRID, CHRONOLOGICAL ORDER — LATER DATES SUPERSEDE EARLIER FACTS)]:"]
            for score_val, _, item in top_items:
                s_id = item.get("session_id") or "session"
                s_dt = item.get("session_date") or "unknown-date"
                t_rg = item.get("turn_range")
                hdr = f"Session {s_id} [Date: {s_dt}]" + (f" ({t_rg})" if t_rg else "")
                lines.append(f"• {hdr} (relevance={score_val:.2f}):\n{item.get('content', '')}")
            return "\n".join(lines)
        except Exception:
            return ""

    def auto_crystallize_from_chunk(
        self,
        chunk_src: str,
        goal_description: str = "",
    ) -> List[str]:
        """Zero-token post-step hook: automatically persist any user class with compiled methods created/updated in `chunk_src`."""
        if not self.enable_engrams or not self.auto_crystallize:
            return []
        src = (chunk_src or "").strip()
        if not src:
            return []
        # Skip if the chunk already explicitly called `persistEngram` or `saveEngram`
        if "persistEngram" in src or "saveEngram" in src:
            return []

        candidate_classes: List[str] = []
        active_cls = self.vm._current_compilation_class or "Object"
        try:
            for raw_piece in self.vm._split_chunks_safely(src):
                nodes = parse_actor(raw_piece)
                for node in nodes:
                    if isinstance(node, ClassDefNode):
                        cname = _norm_sym(node.name)
                        active_cls = cname
                        if cname not in candidate_classes:
                            candidate_classes.append(cname)
                    elif hasattr(node, "target_class") and getattr(node, "target_class", None):
                        active_cls = _norm_sym(getattr(node, "target_class"))
                    elif isinstance(node, MethodDefNode):
                        raw_m_cls = _norm_sym(getattr(node, "class_name", ""))
                        t_cls = active_cls if raw_m_cls in ("", "Object") else raw_m_cls
                        if t_cls and t_cls not in candidate_classes:
                            candidate_classes.append(t_cls)
        except Exception:
            return []

        persisted: List[str] = []
        actor_pool = self.vm.global_vars.get("Actor", {})
        for cname in candidate_classes:
            if (
                not cname
                or cname in PROTECTED_SYSTEM_CLASSES
                or cname in ("AgentHarness", "<Chunk>", "Object")
                or cname.startswith(_SCRATCH_CLASS_PREFIXES)
            ):
                continue
            cls_obj = self.vm.hierarchy.get(cname)
            if cls_obj is None or getattr(cls_obj, "is_bridge_class", False):
                continue
            methods_dict = getattr(cls_obj, "_class_methods", {})
            compiled_sels = [
                sel
                for sel, m in methods_dict.items()
                if sel != "new" and isinstance(m, CompiledMethod) and getattr(m, "source", None)
            ]
            if not compiled_sels:
                continue
            try:
                if cname not in actor_pool and Symbol(cname) not in actor_pool:
                    self.vm.eval_chunk(f"Actor[#{cname}] := new({cname})!!")
                desc = (goal_description or f"Auto-crystallized Engram {cname}")[:180]
                self.persistEngram(cname, desc)
                persisted.append(cname)
            except Exception:
                continue
        return persisted

    def hydrateRelevantEngrams(self, goal_text: str = "") -> str:
        """Auto-hydrate saved `:ActorEngram` nodes (and transitive `:SUBCLASS_OF` / `:DEPENDS_ON` targets) from Neo4j into `self.vm`."""
        import os as _os

        if not self.enable_engrams:
            return ""
        if _os.environ.get("ACTORII_BENCH_STRICT") == "1" and self.namespace == "default":
            return ""
        try:
            now_mono = time.monotonic()
            eng_key = (self.namespace, self._engram_epoch)
            cached_eng = self._engram_rows_cache.get(eng_key)
            if cached_eng is not None and (now_mono - cached_eng[0]) <= 90.0:
                rows = cached_eng[1]
            else:
                _, rows = self._run_cypher(
                    """
                    MATCH (e:ActorEngram {namespace: $ns})
                    OPTIONAL MATCH (e)-[:OPERATES_ON]->(d:ActorGraphDomain {namespace: $ns})
                    OPTIONAL MATCH (e)-[:SUBCLASS_OF|DEPENDS_ON*1..5]->(dep:ActorEngram {namespace: $ns})
                    RETURN e.name AS name, e.description AS description,
                           coalesce(e.superclass, 'Object') AS superclass,
                           coalesce(e.ivars, []) AS ivars,
                           e.signatures AS signatures, e.actor_source AS actor_source,
                           e.embedding AS embedding,
                           coalesce(e.call_count, 0) AS call_count,
                           coalesce(e.success_count, 0) AS success_count,
                           coalesce(e.updated_at, 0) AS updated_at,
                           collect(DISTINCT d.label) AS domain_labels,
                           collect(DISTINCT {name: dep.name, actor_source: dep.actor_source, superclass: coalesce(dep.superclass, 'Object')}) AS deps
                    ORDER BY updated_at DESC
                    LIMIT 20
                    """,
                    {"ns": self.namespace},
                )
                self._engram_rows_cache[eng_key] = (now_mono, list(rows))
            if not rows:
                return ""

            stop_words = {
                "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
                "not", "all", "can", "you", "your", "has", "have", "only", "write", "read",
                "file", "files", "workspace", "code", "python", "function", "test", "tests",
                "pass", "fix", "issue", "edit", "implement", "resolve", "disk", "string",
                "return", "value", "find", "search", "inspect", "using", "complete", "working",
            }
            goal_words = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9_]+", str(goal_text or ""))
                if len(w) >= 3 and w.lower() not in stop_words
            }
            goal_emb = _compute_embedding(goal_text) if goal_text else None

            scored_rows: List[Tuple[float, Dict[str, Any]]] = []
            for r in rows:
                if r.get("name") == "AgentHarness":
                    continue
                domain_list = [str(x) for x in (r.get("domain_labels") or []) if x]
                domain_str = " ".join(domain_list)
                sup_name = str(r.get("superclass") or "Object")
                haystack = (
                    f"{r.get('name', '')} {sup_name} {r.get('description', '')} "
                    f"{' '.join(r.get('signatures') or [])} {domain_str}"
                ).lower()

                lexical_matches = sum(1 for w in goal_words if w in haystack)
                lexical_sim = (lexical_matches / max(1, len(goal_words))) if goal_words else 0.5
                emb_sim = _cosine_similarity(goal_emb, r.get("embedding"))
                sim_score = max(lexical_sim, emb_sim)

                domain_matches = sum(1 for d in domain_list if d.lower() in goal_words or any(w in d.lower() for w in goal_words))
                domain_overlap = min(1.0, domain_matches / max(1, len(domain_list))) if domain_list else 0.0

                calls = int(r.get("call_count") or 0)
                succ = int(r.get("success_count") or 0)
                reliability = (succ + 1.0) / (calls + 2.0)

                composite = 0.45 * sim_score + 0.35 * domain_overlap + 0.20 * reliability
                if (not goal_text) or domain_matches > 0 or lexical_matches >= 1 or emb_sim >= 0.62:
                    scored_rows.append((composite, r))

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            selected = [r for _, r in scored_rows[:6]]
            if not selected and not goal_text:
                selected = [r for r in rows if r.get("name") != "AgentHarness"][:4]
            if not selected:
                return ""

            lines = [
                "[NEO4J ACTIVE ENGRAMS & OOP CLASS HIERARCHY (HYDRATED INTO LIVE VM POOL)]:",
                "• Foundational Base Classes available to subclass (`inherit(SuperClass, #SubClass, #(#ivar), 1, nil)!!`):",
                "  - `GraphService` (ivars: `#(#namespace #defaultLabel #lastQueryCount)`) -> inherited methods: `initService(self, label)`, `queryRows(self, cypher)`, `writeGraph(self, cypher)`, `sumColumn(self, rows, #col)`, `avgColumn(self, rows, #col)`, `maxColumn(self, rows, #col)`",
                "  - `MetricEvaluator` (extends `GraphService`, ivars: `#(#metricName #threshold #weight)`) -> inherited methods: `initEvaluator(self, name, thresh, weight)`, `evaluateRow(self, row)` (override in subclasses!), `evaluateAll(self, rows)`, `weightedTotal(self, rows)`",
                "  - `CompositePipeline` (ivars: `#(#pipelineName #stages #lastReport)`) -> inherited methods: `initPipeline(self, name)`, `addStage(self, evaluatorObj)`, `runEvaluators(self, rows)`, `lastReport(self)`",
                "  - `DomainEntity` (ivars: `#(#id #name #metadata)`) -> inherited methods: `initEntity(self, id, name)`, `setMeta(self, #k, v)`, `getMeta(self, #k)`, `describe(self)`",
                "• Pre-verified Engram Objects (invoke directly in 1 step, OR subclass/compose them via `inherit(ExistingClass, #NewSubClass, ...)`):",
            ]
            for r in selected:
                # Hydrate prerequisite superclasses and dependencies first (in topological order: Object -> Base -> Child)
                deps_list = [d for d in (r.get("deps") or []) if d.get("name") and d.get("actor_source")]
                deps_list.sort(key=lambda d: 0 if d.get("superclass", "Object") in ("Object", "GraphService") else 1)
                for dep in deps_list:
                    dep_name = dep.get("name")
                    dep_src = dep.get("actor_source")
                    if dep_name and dep_src:
                        try:
                            self._hydrate_engram_source_safely(str(dep_name), str(dep_src))
                        except Exception:
                            pass

                ename = str(r["name"])
                sup_cls = str(r.get("superclass") or "Object")
                ivs = [f"#{x}" for x in (r.get("ivars") or []) if x]
                iv_tag = f" ivars:#({' '.join(ivs)})" if ivs else ""
                src = str(r.get("actor_source") or "")
                if src:
                    try:
                        self._hydrate_engram_source_safely(ename, src)
                    except Exception:
                        pass
                raw_sigs = list(r.get("signatures") or [])
                sigs = ", ".join(f"`{s}!!`" for s in raw_sigs)
                ex_call = raw_sigs[0] if raw_sigs else f"run(Actor[#{ename}])"
                lbls = [x for x in (r.get("domain_labels") or []) if x]
                lbl_tag = f" [Graph: {', '.join(':' + l for l in lbls)}]" if lbls else ""
                lines.append(
                    f"  • `Actor[#{ename}]` [extends `{sup_cls}`{iv_tag}]{lbl_tag} ({r.get('description', '')}) -> Signatures: {sigs} | "
                    f"1-Step Call: `storeScratch(Actor[#Harness], #answer, {ex_call})!!\\nCOMPLETE: Done`"
                )
            return "\n".join(lines)
        except Exception:
            return ""

    # =========================================================================
    # Dual-Mode Product Technical Knowledge Graph + `:8084` Vector Manual Chunks
    # (`Actor[#ProductLearner]` backing store on `Actor[#Memory]`)
    # =========================================================================

    @staticmethod
    def _slugify_product_id(name: str) -> str:
        raw = str(name or "").strip().lower()
        # Strip parenthetical version tags when generating canonical ID so follow-up ingests merge cleanly
        base = re.sub(r"\([^)]*\)", " ", raw)
        slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
        if not slug:
            slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-") or "unknown-product"
        return slug[:80]

    @staticmethod
    def _extract_json_dict(raw_text: str) -> Optional[Dict[str, Any]]:
        txt = str(raw_text or "").strip()
        if not txt:
            return None
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
        fence_m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
        if fence_m:
            try:
                parsed = json.loads(fence_m.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            cand = txt[start : end + 1]
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return None

    @classmethod
    def _heuristic_distill_corpus(
        cls,
        product_name: str,
        focus_topics: str,
        raw_corpus: str,
    ) -> Dict[str, Any]:
        """Deterministic manual parser & chunker (guarantees zero-loss extraction even if LLM is offline)."""
        p_name = str(product_name or "Unknown Product").strip()
        focus = str(focus_topics or "").strip()
        corpus = str(raw_corpus or "").strip()
        # Strip optional YAML frontmatter (`--- ... ---`) at top of markdown files/skills
        corpus = re.sub(r"^(?:\[LocalManual:[^\]]*\]\s*)?---\s*\n.*?\n---\s*\n", "", corpus, flags=re.DOTALL)

        # Extract URLs
        urls = []
        seen_urls = set()
        for u in re.findall(r"https?://[^\s\)\]>\"']+", corpus):
            cu = u.rstrip(".,;:!?)")
            if cu and cu not in seen_urls:
                seen_urls.add(cu)
                urls.append({"url": cu, "title": f"{p_name} Documentation"})

        # Split into topical sections / manual chunks
        sections: List[Tuple[str, str]] = []
        current_title = f"{p_name} Overview"
        current_lines: List[str] = []
        for ln in corpus.splitlines():
            hdr_m = re.match(r"^\s*#{1,4}\s+(.+?)\s*$", ln)
            scrape_m = re.match(r"^\[(WebScrape|LocalManual|PDF)[^\]]*\]\s*(.*)$", ln)
            if hdr_m:
                if current_lines:
                    body_txt = "\n".join(current_lines).strip()
                    if len(body_txt) >= 30:
                        sections.append((current_title, body_txt))
                    current_lines = []
                current_title = hdr_m.group(1).strip()[:120]
            elif scrape_m and current_lines:
                body_txt = "\n".join(current_lines).strip()
                if len(body_txt) >= 30:
                    sections.append((current_title, body_txt))
                current_lines = []
                current_title = (scrape_m.group(2).strip() or f"{p_name} Manual Section")[:120]
            else:
                current_lines.append(ln)
        if current_lines:
            body_txt = "\n".join(current_lines).strip()
            if body_txt:
                sections.append((current_title, body_txt))

        chunks: List[Dict[str, str]] = []
        for sec_title, sec_body in sections:
            if len(sec_body) <= 1000:
                chunks.append({"title": sec_title, "content": sec_body})
            else:
                # Split long sections on blank lines into <= 1000-char chunks
                paras = [p.strip() for p in re.split(r"\n\s*\n", sec_body) if p.strip()]
                buf = ""
                part_idx = 1
                for p in paras:
                    if len(buf) + len(p) + 2 <= 1000:
                        buf = f"{buf}\n\n{p}".strip() if buf else p
                    else:
                        if buf:
                            chunks.append({"title": f"{sec_title} (Part {part_idx})", "content": buf[:1000]})
                            part_idx += 1
                        buf = p[:1000]
                if buf:
                    chunks.append({"title": f"{sec_title}" if part_idx == 1 else f"{sec_title} (Part {part_idx})", "content": buf[:1000]})
            if len(chunks) >= 12:
                break

        if not chunks and corpus:
            chunks.append({"title": f"{p_name} Technical Notes", "content": corpus[:1000]})

        # Extract CLI procedures (backticked commands or shell lines)
        procedures: List[Dict[str, str]] = []
        seen_cmds = set()
        cmd_prefixes = (
            "sudo ", "systemctl ", "modprobe ", "nvstart-", "gst-launch", "v4l2-ctl",
            "weston", "xrandr", "export ", "docker ", "cargo ", "cmake ", "ssh ", "nvpmodel", "jetson_clocks",
        )
        for ln in corpus.splitlines():
            s = ln.strip().lstrip("•-*$# ").strip()
            bt_matches = re.findall(r"`([^`]{6,160})`", ln)
            candidates = list(bt_matches)
            if any(s.startswith(pref) for pref in cmd_prefixes):
                candidates.append(s[:180])
            for c in candidates:
                c_clean = c.strip()
                if any(c_clean.startswith(pref) or ".sh" in c_clean or "systemctl" in c_clean or "modprobe" in c_clean for pref in cmd_prefixes):
                    if c_clean not in seen_cmds:
                        seen_cmds.add(c_clean)
                        procedures.append(
                            {
                                "name": f"Command: {c_clean[:52]}",
                                "command": c_clean,
                                "notes": s[:200] if s != c_clean else "",
                            }
                        )
                        if len(procedures) >= 10:
                            break
            if len(procedures) >= 10:
                break

        # Extract critical gotchas / guardrails
        gotchas: List[Dict[str, str]] = []
        seen_rules = set()
        for ln in corpus.splitlines():
            s = ln.strip().lstrip("•-*> ").strip()
            low = s.lower()
            if len(s) >= 20 and any(k in low for k in ("never ", "do not ", "don't ", "warning", "critical", "caution", "important", "avoid ", "breaks ", "brick")):
                rule_txt = s[:260]
                if rule_txt not in seen_rules:
                    seen_rules.add(rule_txt)
                    sev = "CRITICAL" if any(k in low for k in ("never", "critical", "brick", "boot", "initrd")) else "WARNING"
                    gotchas.append({"rule": rule_txt, "severity": sev})
                    if len(gotchas) >= 8:
                        break

        # Extract subsystems from section titles
        subsystems: List[Dict[str, Any]] = []
        for sec_title, sec_body in sections[:6]:
            if sec_title.lower().endswith("overview") and len(sections) > 1:
                continue
            sub_cmds = [b.strip() for b in re.findall(r"`([^`]{4,120})`", sec_body)][:4]
            subsystems.append(
                {
                    "name": sec_title[:80],
                    "driver": "",
                    "description": " ".join(sec_body.split())[:260],
                    "commands": sub_cmds,
                }
            )

        kw_tokens = set()
        for tok in re.findall(r"[A-Za-z0-9_-]{3,}", f"{p_name} {focus} " + " ".join(t for t, _ in sections[:8])):
            tl = tok.lower()
            if tl not in {"the", "and", "for", "with", "from", "manual", "overview", "section", "part", "guide", "documentation"}:
                kw_tokens.add(tl)

        first_summary = " ".join((chunks[0]["content"] if chunks else corpus).split())[:360]
        return {
            "name": p_name,
            "category": "Hardware / Software Product",
            "version": "",
            "keywords": sorted(kw_tokens)[:25],
            "summary": first_summary or f"Technical reference for {p_name}.",
            "subsystems": subsystems[:8],
            "procedures": procedures[:10],
            "gotchas": gotchas[:8],
            "chunks": chunks[:10],
            "sources": urls[:6],
        }

    @actor_query
    def distillProductCorpus(
        self,
        product_name: str,
        focus_topics: str = "",
        raw_corpus: str = "",
    ) -> str:
        """Distill raw web/PDF manual text into structured Product Knowledge JSON via `ValkyrieLLMClient` + heuristic enrichment."""
        p_name = str(product_name or "").strip() or "Unknown Product"
        focus = str(focus_topics or "").strip()
        corpus = str(raw_corpus or "").strip()

        heuristic = self._heuristic_distill_corpus(p_name, focus, corpus)
        if not corpus:
            return json.dumps(heuristic, ensure_ascii=False)

        # Skip live LLM call during shadow dry-runs
        if getattr(self.vm, "_in_shadow_run", False):
            return json.dumps(heuristic, ensure_ascii=False)

        llm_parsed: Optional[Dict[str, Any]] = None
        try:
            from actorii.ai import ValkyrieLLMClient

            actor_pool = getattr(self.vm, "global_vars", {}).get("Actor", {})
            harness_obj = actor_pool.get("Harness") if isinstance(actor_pool, dict) else None
            model = getattr(harness_obj, "model", getattr(self.vm, "ai_model", "qwen3.8:latest"))
            endpoint = getattr(harness_obj, "endpoint", getattr(self.vm, "ai_endpoint", None))
            custom_prov = getattr(harness_obj, "custom_provider", None) or getattr(self.vm, "ai_provider", None)

            # Prefer zero-VRAM CPU specialist slot (`:8082`) when running locally so product distillation
            # never evicts `AgentHarness`'s warm KV-cache slot on `:8080`!
            specialist_ep = os.environ.get(
                "ACTORII_SPECIALIST_ENDPOINT",
                "http://127.0.0.1:8082/v1/chat/completions",
            ).strip()
            use_specialist_first = (
                custom_prov is None
                and bool(specialist_ep)
                and (not endpoint or ":8080" in str(endpoint))
            )
            distill_ep = specialist_ep if use_specialist_first else endpoint

            client = ValkyrieLLMClient(
                endpoint=distill_ep,
                model="default" if use_specialist_first else model,
                timeout=180.0,
                custom_provider=custom_prov,
            )
            sys_prompt = (
                "You are a Senior Systems & Hardware Technical Documentation Distiller. "
                "Given raw documentation/manual excerpts for a product, extract an accurate, high-density JSON technical sheet. "
                "Output ONLY a valid JSON object (no markdown commentary, no Actor `!!` code) matching this exact schema:\n"
                "{\n"
                '  "name": "Canonical Product Name",\n'
                '  "category": "Category (e.g. Embedded GPU SoM, Sensor, Library)",\n'
                '  "version": "OS / Firmware / SDK Version if mentioned",\n'
                '  "keywords": ["lowercase", "technical", "search", "terms", "drivers", "binaries"],\n'
                '  "summary": "Concise 2-3 sentence technical architecture summary",\n'
                '  "subsystems": [{"name": "Subsystem", "driver": "Driver/Daemon/Stack", "description": "How it works", "commands": ["exact command 1"]}],\n'
                '  "procedures": [{"name": "Task Name", "command": "exact CLI or API command", "notes": "Prerequisites or flags"}],\n'
                '  "gotchas": [{"rule": "Critical pitfall, conflict, or safety guardrail", "severity": "CRITICAL or WARNING"}],\n'
                '  "chunks": [{"title": "Section Title", "content": "Self-contained 3-6 sentence technical reference paragraph preserving exact paths, flags, ports, and driver names"}],\n'
                '  "sources": [{"url": "https://...", "title": "Source Title"}]\n'
                "}"
            )
            user_prompt = (
                f"PRODUCT: {p_name}\n"
                f"FOCUS TOPICS: {focus or 'Architecture, subsystems, drivers, CLI procedures, and critical gotchas'}\n\n"
                f"RAW DOCUMENTATION CORPUS:\n{corpus[:9500]}\n\n"
                "Return ONLY the valid JSON object:"
            )
            raw_llm = client.complete(
                user_prompt,
                system_prompt=sys_prompt,
                max_tokens=1800,
                temperature=0.1,
            )
            llm_parsed = self._extract_json_dict(raw_llm)
        except Exception:
            llm_parsed = None

        if not isinstance(llm_parsed, dict):
            return json.dumps(heuristic, ensure_ascii=False)

        # Merge LLM structured distillation with heuristic manual chunks & URLs so nothing in the raw manual is lost
        merged_kw = sorted(
            {
                str(k).strip().lower()
                for k in (list(llm_parsed.get("keywords") or []) + list(heuristic.get("keywords") or []))
                if str(k).strip()
            }
        )[:30]

        llm_chunks = [
            {"title": str(c.get("title") or "Manual Section"), "content": str(c.get("content") or "").strip()}
            for c in (llm_parsed.get("chunks") or [])
            if isinstance(c, dict) and str(c.get("content") or "").strip()
        ]
        # Append heuristic chunks whose content isn't already covered so `:8084` vector recall has full manual coverage
        existing_titles = {c["title"].lower() for c in llm_chunks}
        for hc in heuristic.get("chunks") or []:
            if len(llm_chunks) >= 12:
                break
            if hc["title"].lower() not in existing_titles:
                llm_chunks.append(hc)

        llm_sources = [
            {"url": str(s.get("url") or "").strip(), "title": str(s.get("title") or p_name)}
            for s in (llm_parsed.get("sources") or [])
            if isinstance(s, dict) and str(s.get("url") or "").strip()
        ]
        seen_s_urls = {s["url"] for s in llm_sources}
        for hs in heuristic.get("sources") or []:
            if hs["url"] not in seen_s_urls and len(llm_sources) < 8:
                llm_sources.append(hs)
                seen_s_urls.add(hs["url"])

        merged = {
            "name": str(llm_parsed.get("name") or p_name).strip(),
            "category": str(llm_parsed.get("category") or heuristic["category"]).strip(),
            "version": str(llm_parsed.get("version") or heuristic["version"]).strip(),
            "keywords": merged_kw,
            "summary": str(llm_parsed.get("summary") or heuristic["summary"]).strip(),
            "subsystems": list(llm_parsed.get("subsystems") or heuristic["subsystems"])[:10],
            "procedures": list(llm_parsed.get("procedures") or heuristic["procedures"])[:12],
            "gotchas": list(llm_parsed.get("gotchas") or heuristic["gotchas"])[:10],
            "chunks": llm_chunks[:12],
            "sources": llm_sources[:8],
        }
        return json.dumps(merged, ensure_ascii=False)

    @actor_command
    def storeProductKnowledge(self, product_name: str, spec_json_or_dict: Any) -> str:
        """Persist dual-mode `:Product` structured graph + `:8084` vector-embedded `:ProductManualChunk` nodes in Neo4j."""
        import hashlib

        if isinstance(spec_json_or_dict, Dictionary):
            spec: Dict[str, Any] = {}
            for k, v in spec_json_or_dict._entries.items():
                nk = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                spec[nk] = _actor_to_py_value(v)
        elif isinstance(spec_json_or_dict, dict):
            spec = dict(spec_json_or_dict)
        else:
            raw_s = str(spec_json_or_dict or "").strip()
            spec = self._extract_json_dict(raw_s) or self._heuristic_distill_corpus(product_name, "", raw_s)

        canon_name = str(spec.get("name") or product_name or "Unknown Product").strip()
        pid = self._slugify_product_id(product_name or canon_name)
        category = str(spec.get("category") or "Technical Product").strip()
        version = str(spec.get("version") or "").strip()
        summary = str(spec.get("summary") or "").strip()

        raw_kws = list(spec.get("keywords") or [])
        for tok in re.findall(r"[A-Za-z0-9_-]{3,}", f"{product_name} {canon_name} {version}"):
            raw_kws.append(tok.lower())
        keywords = sorted({str(k).strip().lower() for k in raw_kws if str(k).strip()})[:35]

        prod_emb_text = f"{canon_name} ({version}) [{category}]: {summary} Keywords: {' '.join(keywords)}"
        prod_emb = _compute_embedding(prod_emb_text)

        self._run_cypher(
            """
            MERGE (p:Product {id: $pid, namespace: $ns})
            SET p.name = $name,
                p.category = $category,
                p.version = CASE WHEN $version <> '' THEN $version ELSE coalesce(p.version, '') END,
                p.summary = CASE WHEN $summary <> '' THEN $summary ELSE coalesce(p.summary, '') END,
                p.keywords = reduce(acc = coalesce(p.keywords, []), k IN $keywords | CASE WHEN k IN acc THEN acc ELSE acc + k END),
                p.embedding = CASE WHEN size($emb) > 0 THEN $emb ELSE coalesce(p.embedding, []) END,
                p.updated_at = timestamp()
            """,
            {
                "pid": pid,
                "ns": self.namespace,
                "name": canon_name,
                "category": category,
                "version": version,
                "summary": summary,
                "keywords": keywords,
                "emb": prod_emb or [],
            },
        )

        sub_cnt = 0
        for sub in spec.get("subsystems") or []:
            if not isinstance(sub, dict):
                continue
            sname = str(sub.get("name") or "").strip()
            if not sname:
                continue
            sdriver = str(sub.get("driver") or "").strip()
            sdesc = str(sub.get("description") or "").strip()
            scmds_raw = sub.get("commands") or []
            scmds = [str(x).strip() for x in (scmds_raw if isinstance(scmds_raw, list) else [scmds_raw]) if str(x).strip()]
            self._run_cypher(
                """
                MATCH (p:Product {id: $pid, namespace: $ns})
                MERGE (s:Subsystem {product_id: $pid, name: $sname, namespace: $ns})
                SET s.driver = $sdriver,
                    s.description = $sdesc,
                    s.commands = $scmds,
                    s.updated_at = timestamp()
                MERGE (p)-[:HAS_SUBSYSTEM]->(s)
                """,
                {
                    "pid": pid,
                    "ns": self.namespace,
                    "sname": sname[:120],
                    "sdriver": sdriver[:160],
                    "sdesc": sdesc[:600],
                    "scmds": scmds[:10],
                },
            )
            sub_cnt += 1

        proc_cnt = 0
        for pr in spec.get("procedures") or []:
            if not isinstance(pr, dict):
                continue
            prname = str(pr.get("name") or "").strip()
            prcmd = str(pr.get("command") or "").strip()
            if not prname and not prcmd:
                continue
            prname = prname or f"Command: {prcmd[:50]}"
            prnotes = str(pr.get("notes") or "").strip()
            self._run_cypher(
                """
                MATCH (p:Product {id: $pid, namespace: $ns})
                MERGE (proc:Procedure {product_id: $pid, name: $prname, namespace: $ns})
                SET proc.command = $prcmd,
                    proc.notes = $prnotes,
                    proc.updated_at = timestamp()
                MERGE (p)-[:HAS_PROCEDURE]->(proc)
                """,
                {
                    "pid": pid,
                    "ns": self.namespace,
                    "prname": prname[:140],
                    "prcmd": prcmd[:500],
                    "prnotes": prnotes[:500],
                },
            )
            proc_cnt += 1

        gotcha_cnt = 0
        for g in spec.get("gotchas") or []:
            rule_txt = str(g.get("rule") if isinstance(g, dict) else g or "").strip()
            if not rule_txt:
                continue
            sev = str(g.get("severity", "WARNING") if isinstance(g, dict) else "WARNING").strip().upper()
            self._run_cypher(
                """
                MATCH (p:Product {id: $pid, namespace: $ns})
                MERGE (gt:Gotcha {product_id: $pid, rule: $rule, namespace: $ns})
                SET gt.severity = $sev,
                    gt.updated_at = timestamp()
                MERGE (p)-[:HAS_GOTCHA]->(gt)
                """,
                {
                    "pid": pid,
                    "ns": self.namespace,
                    "rule": rule_txt[:400],
                    "sev": sev[:32],
                },
            )
            gotcha_cnt += 1

        src_cnt = 0
        for src in spec.get("sources") or []:
            u = str(src.get("url") if isinstance(src, dict) else src or "").strip()
            if not u:
                continue
            t = str(src.get("title", canon_name) if isinstance(src, dict) else canon_name).strip()
            self._run_cypher(
                """
                MATCH (p:Product {id: $pid, namespace: $ns})
                MERGE (ds:DocSource {product_id: $pid, url: $url, namespace: $ns})
                SET ds.title = $title,
                    ds.updated_at = timestamp()
                MERGE (p)-[:CITES_SOURCE]->(ds)
                """,
                {
                    "pid": pid,
                    "ns": self.namespace,
                    "url": u[:500],
                    "title": t[:200],
                },
            )
            src_cnt += 1

        chunk_cnt = 0
        for ch in spec.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            ctitle = str(ch.get("title") or f"{canon_name} Section").strip()
            ccontent = str(ch.get("content") or "").strip()
            if not ccontent:
                continue
            cid = hashlib.sha1(f"{pid}::{ctitle}::{ccontent[:160]}".encode("utf-8")).hexdigest()[:16]
            c_emb = _compute_embedding(f"{canon_name} — {ctitle}: {ccontent}")
            self._run_cypher(
                """
                MATCH (p:Product {id: $pid, namespace: $ns})
                MERGE (mc:ProductManualChunk {chunk_id: $cid, namespace: $ns})
                SET mc.product_id = $pid,
                    mc.product_name = $pname,
                    mc.title = $title,
                    mc.content = $content,
                    mc.embedding = $emb,
                    mc.updated_at = timestamp()
                MERGE (p)-[:HAS_MANUAL_CHUNK]->(mc)
                """,
                {
                    "pid": pid,
                    "ns": self.namespace,
                    "cid": cid,
                    "pname": canon_name,
                    "title": ctitle[:160],
                    "content": ccontent[:1800],
                    "emb": c_emb or [],
                },
            )
            chunk_cnt += 1

        return (
            f"✔ Learned Product '{canon_name}' [id={pid}]: "
            f"{sub_cnt} subsystems, {proc_cnt} procedures, {gotcha_cnt} gotchas, "
            f"{chunk_cnt} `:8084` vector-embedded manual chunks, {src_cnt} sources persisted in Neo4j."
        )

    def _query_matching_products(
        self,
        query_str: str,
        topic_str: str = "",
        top_k_chunks: int = 5,
        strict_auto_hydrate: bool = False,
    ) -> str:
        """Shared hybrid Cypher + `:8084` vector retrieval engine for `recallProductKnowledge` and `hydrateContext`."""
        combined_q = f"{query_str} {topic_str}".strip()
        if not combined_q:
            return ""
        now_mono = time.monotonic()
        prod_key = (self.namespace, self._engram_epoch)
        cached_prod = self._product_rows_cache.get(prod_key)
        if cached_prod is not None and (now_mono - cached_prod[0]) <= 90.0:
            prod_rows = cached_prod[1]
        else:
            _, prod_rows = self._run_cypher(
                """
                MATCH (p:Product {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_SUBSYSTEM]->(s:Subsystem {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_PROCEDURE]->(pr:Procedure {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_GOTCHA]->(g:Gotcha {namespace: $ns})
                OPTIONAL MATCH (p)-[:CITES_SOURCE]->(ds:DocSource {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_MANUAL_CHUNK]->(mc:ProductManualChunk {namespace: $ns})
                RETURN p.id AS id,
                       p.name AS name,
                       coalesce(p.category, '') AS category,
                       coalesce(p.version, '') AS version,
                       coalesce(p.summary, '') AS summary,
                       coalesce(p.keywords, []) AS keywords,
                       p.embedding AS embedding,
                       collect(DISTINCT {name: s.name, driver: s.driver, description: s.description, commands: s.commands}) AS subsystems,
                       collect(DISTINCT {name: pr.name, command: pr.command, notes: pr.notes}) AS procedures,
                       collect(DISTINCT {rule: g.rule, severity: g.severity}) AS gotchas,
                       collect(DISTINCT {url: ds.url, title: ds.title}) AS sources,
                       collect(DISTINCT {chunk_id: mc.chunk_id, title: mc.title, content: mc.content, embedding: mc.embedding}) AS chunks
                """,
                {"ns": self.namespace},
            )
            self._product_rows_cache[prod_key] = (now_mono, list(prod_rows))
        if not prod_rows:
            return ""

        stop_w = {
            "the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "how",
            "what", "where", "when", "why", "can", "you", "your", "please", "check", "start",
            "run", "use", "using", "get", "make", "show", "tell", "about", "product", "manual",
        }
        q_words = {
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_-]+", combined_q)
            if len(w) >= 3 and w.lower() not in stop_w
        }
        q_slug = self._slugify_product_id(query_str) if query_str else ""
        q_emb = _compute_embedding(combined_q)

        scored_products: List[Tuple[float, Dict[str, Any], List[Tuple[float, Dict[str, Any]]]]] = []
        for prow in prod_rows:
            pid = str(prow.get("id") or "")
            pname = str(prow.get("name") or "")
            p_tokens = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9_-]+", f"{pid} {pname}")
                if len(w) >= 3 and w.lower() not in {"nvidia", "the", "pro", "plus"}
            }
            kws = {str(k).lower() for k in (prow.get("keywords") or []) if k}
            subs = [s for s in (prow.get("subsystems") or []) if isinstance(s, dict) and s.get("name")]
            procs = [p for p in (prow.get("procedures") or []) if isinstance(p, dict) and (p.get("name") or p.get("command"))]
            gotchas = [g for g in (prow.get("gotchas") or []) if isinstance(g, dict) and g.get("rule")]
            chunks = [c for c in (prow.get("chunks") or []) if isinstance(c, dict) and c.get("content")]

            haystack = (
                f"{pid} {pname} {prow.get('version', '')} {prow.get('summary', '')} "
                f"{' '.join(kws)} "
                + " ".join(f"{s.get('name', '')} {s.get('driver', '')} {s.get('description', '')}" for s in subs)
                + " "
                + " ".join(f"{p.get('name', '')} {p.get('command', '')}" for p in procs)
            ).lower()

            exact_id_match = bool(q_slug and (q_slug == pid or q_slug in pid or pid in q_slug))
            name_token_hits = sum(1 for w in q_words if w in p_tokens)
            lex_hits = sum(1 for w in q_words if w in haystack)
            lex_sim = (lex_hits / max(1, len(q_words))) if q_words else 0.0
            prod_emb_sim = _cosine_similarity(q_emb, prow.get("embedding"))

            scored_chunks: List[Tuple[float, Dict[str, Any]]] = []
            best_chunk_emb = 0.0
            for ch in chunks:
                ch_hay = f"{ch.get('title', '')} {ch.get('content', '')}".lower()
                ch_lex = (sum(1 for w in q_words if w in ch_hay) / max(1, len(q_words))) if q_words else 0.0
                ch_emb = _cosine_similarity(q_emb, ch.get("embedding"))
                if ch_emb > best_chunk_emb:
                    best_chunk_emb = ch_emb
                ch_score = 0.58 * ch_emb + 0.42 * ch_lex
                scored_chunks.append((ch_score, ch))
            scored_chunks.sort(key=lambda item: item[0], reverse=True)

            best_emb = max(prod_emb_sim, best_chunk_emb)
            prod_score = (
                (0.50 if exact_id_match else 0.0)
                + (0.30 if name_token_hits >= 1 else 0.0)
                + 0.45 * best_emb
                + 0.35 * lex_sim
            )

            if strict_auto_hydrate:
                # Only auto-inject into `hydrateContext` when the user's goal genuinely mentions the product or strongly matches its manual!
                is_relevant = (
                    name_token_hits >= 1
                    or exact_id_match
                    or (lex_hits >= 2 and lex_sim >= 0.45)
                    or best_emb >= 0.66
                )
                if not is_relevant:
                    continue
            else:
                if prod_score < 0.25 and not exact_id_match and name_token_hits == 0 and lex_hits == 0:
                    continue

            scored_products.append((prod_score, prow, scored_chunks, exact_id_match or name_token_hits >= 1))

        if not scored_products:
            return ""

        scored_products.sort(key=lambda item: item[0], reverse=True)
        # If any product explicitly matched by name/id, drop unrelated secondary products that didn't match by name/id!
        if any(item[3] for item in scored_products):
            scored_products = [
                item for item in scored_products if item[3] or item[0] >= 0.58
            ]
        max_prods = 2 if strict_auto_hydrate else 3
        header_tag = (
            "[AUTO-RECALLED PRODUCT TECHNICAL MANUAL (NEO4J `:Product` GRAPH + `:8084` VECTOR HYBRID — USE THESE VERIFIED COMMANDS & GUARDRAILS INSTEAD OF GUESSING!)]:"
            if strict_auto_hydrate
            else "[PRODUCT TECHNICAL MANUAL RECALL (`Actor[#ProductLearner]` — NEO4J GRAPH + `:8084` VECTOR HYBRID)]:"
        )
        out_lines = [header_tag]

        for p_score, prow, s_chunks, _named in scored_products[:max_prods]:
            pname = prow.get("name") or prow.get("id")
            ver = f" ({prow['version']})" if prow.get("version") else ""
            cat = f" [{prow['category']}]" if prow.get("category") else ""
            out_lines.append(f"• Product: {pname}{ver}{cat} (id=`{prow.get('id')}`, relevance={p_score:.2f})")
            if prow.get("summary"):
                out_lines.append(f"  Summary: {prow['summary']}")

            gotchas = [g for g in (prow.get("gotchas") or []) if isinstance(g, dict) and g.get("rule")]
            if gotchas:
                out_lines.append("  Critical Gotchas & Safety Rules:")
                for g in gotchas[:6]:
                    out_lines.append(f"    - [{g.get('severity', 'WARNING')}] {g['rule']}")

            subs = [s for s in (prow.get("subsystems") or []) if isinstance(s, dict) and s.get("name")]
            if subs:
                # Filter/rank subsystems toward `topic_str` / `q_words` if provided
                def _sub_rank(s_obj: Dict[str, Any]) -> int:
                    sh = f"{s_obj.get('name', '')} {s_obj.get('driver', '')} {s_obj.get('description', '')}".lower()
                    return sum(1 for w in q_words if w in sh)

                subs_sorted = sorted(subs, key=_sub_rank, reverse=True)
                out_lines.append("  Subsystems & Drivers:")
                for s in subs_sorted[:6]:
                    drv = f" (driver/stack: `{s['driver']}`)" if s.get("driver") else ""
                    cmds = s.get("commands") or []
                    cmd_str = f" | cmds: {', '.join('`' + str(c) + '`' for c in cmds[:3])}" if cmds else ""
                    out_lines.append(f"    - {s['name']}{drv}: {s.get('description', '')}{cmd_str}")

            procs = [p for p in (prow.get("procedures") or []) if isinstance(p, dict) and (p.get("name") or p.get("command"))]
            if procs:
                def _proc_rank(pr_obj: Dict[str, Any]) -> int:
                    ph = f"{pr_obj.get('name', '')} {pr_obj.get('command', '')} {pr_obj.get('notes', '')}".lower()
                    return sum(1 for w in q_words if w in ph)

                procs_sorted = sorted(procs, key=_proc_rank, reverse=True)
                out_lines.append("  Verified Procedures & Exact Commands:")
                for pr in procs_sorted[:6]:
                    note_part = f" — {pr['notes']}" if pr.get("notes") else ""
                    out_lines.append(f"    - {pr.get('name', 'Procedure')}: `{pr.get('command', '')}`{note_part}")

            if s_chunks:
                k_ch = max(1, min(8, int(top_k_chunks or 4)))
                out_lines.append("  Relevant Manual Excerpts (`:ProductManualChunk` vector recall):")
                for ch_sc, ch in s_chunks[:k_ch]:
                    out_lines.append(f"    - [{ch.get('title', 'Section')} | sim={ch_sc:.2f}]: {ch.get('content', '')}")

            sources = [s for s in (prow.get("sources") or []) if isinstance(s, dict) and s.get("url")]
            if sources and not strict_auto_hydrate:
                out_lines.append(
                    "  Sources: " + ", ".join(f"{s.get('title', 'Doc')} ({s['url']})" for s in sources[:5])
                )

        return "\n".join(out_lines)

    @actor_query
    def recallProductKnowledge(self, product_or_query: str, topic: str = "", top_k: int = 5) -> str:
        """Explicitly query learned `:Product` knowledge graph + `:8084` embedded manual chunks in Neo4j."""
        p_q = str(product_or_query or "").strip()
        t_q = str(topic or "").strip()
        if not p_q and not t_q:
            return self.listLearnedProducts()
        res = self._query_matching_products(p_q, t_q, top_k_chunks=int(top_k or 5), strict_auto_hydrate=False)
        if res:
            return res
        return (
            f"No learned `:Product` entry matched {p_q!r} (topic={t_q!r}) in namespace '{self.namespace}'. "
            f"Call `learnProduct(Actor[#ProductLearner], {p_q!r}, {t_q!r})!!` or "
            f"`ingestManual(Actor[#ProductLearner], {p_q!r}, \"https://...\", {t_q!r})!!` to research and store its manual first!"
        )

    @actor_query
    def listLearnedProducts(self) -> str:
        """List all `:Product` entries learned in the current Neo4j namespace."""
        _, rows = self._run_cypher(
            """
            MATCH (p:Product {namespace: $ns})
            OPTIONAL MATCH (p)-[:HAS_SUBSYSTEM]->(s:Subsystem {namespace: $ns})
            OPTIONAL MATCH (p)-[:HAS_PROCEDURE]->(pr:Procedure {namespace: $ns})
            OPTIONAL MATCH (p)-[:HAS_GOTCHA]->(g:Gotcha {namespace: $ns})
            OPTIONAL MATCH (p)-[:HAS_MANUAL_CHUNK]->(mc:ProductManualChunk {namespace: $ns})
            RETURN p.id AS id,
                   p.name AS name,
                   coalesce(p.version, '') AS version,
                   coalesce(p.category, '') AS category,
                   coalesce(p.summary, '') AS summary,
                   count(DISTINCT s) AS subsystems,
                   count(DISTINCT pr) AS procedures,
                   count(DISTINCT g) AS gotchas,
                   count(DISTINCT mc) AS chunks,
                   coalesce(p.updated_at, 0) AS updated_at
            ORDER BY updated_at DESC
            """,
            {"ns": self.namespace},
        )
        if not rows:
            return f"No learned `:Product` manuals in namespace '{self.namespace}' yet. Use `learnProduct(Actor[#ProductLearner], \"Product Name\", \"focus topics\")!!`."
        lines = [f"Learned Products in Neo4j (`namespace={self.namespace}`, {len(rows)} total):"]
        for r in rows:
            ver = f" ({r['version']})" if r.get("version") else ""
            cat = f" [{r['category']}]" if r.get("category") else ""
            lines.append(
                f"• `{r['id']}` — {r['name']}{ver}{cat}: "
                f"{r['subsystems']} subsystems, {r['procedures']} procedures, "
                f"{r['gotchas']} gotchas, {r['chunks']} vector manual chunks"
            )
        return "\n".join(lines)

    @actor_command
    def clearProducts(self) -> int:
        """Delete all `:Product` knowledge nodes and attached manual chunks in `self.namespace`."""
        _, rows = self._run_cypher(
            """
            MATCH (p:Product {namespace: $ns})
            OPTIONAL MATCH (p)-[r]->(child {namespace: $ns})
            WITH collect(DISTINCT p) AS prods, collect(DISTINCT child) AS children
            FOREACH (c IN children | DETACH DELETE c)
            FOREACH (pr IN prods | DETACH DELETE pr)
            RETURN size(prods) AS deleted_count
            """,
            {"ns": self.namespace},
        )
        return int(rows[0]["deleted_count"]) if rows and "deleted_count" in rows[0] else 0

    def _hydrate_relevant_product_manuals(self, goal_text: str) -> str:
        """Automatically inject learned `:Product` technical manuals when `goal_text` references a learned product or subsystem."""
        import os as _os

        if not self.enable_engrams:
            return ""
        if _os.environ.get("ACTORII_BENCH_STRICT") == "1" and self.namespace == "default":
            return ""
        q_str = str(goal_text or "").strip()
        if not q_str:
            return ""
        try:
            return self._query_matching_products(
                q_str,
                "",
                top_k_chunks=4,
                strict_auto_hydrate=True,
            )
        except Exception:
            return ""

    @actor_query
    def hydrateContext(self, goal_text: str = "") -> str:
        """1993 Actor-callable entry point for `AgentHarness>>selectEngramContext`."""
        q_txt = str(goal_text or "")
        cache_key = (self.namespace, q_txt, self._write_epoch)
        cached = self._hydrate_cache.get(cache_key)
        if cached is not None:
            return cached

        engram_txt = self.hydrateRelevantEngrams(q_txt)
        prod_txt = self._hydrate_relevant_product_manuals(q_txt)
        if self.enable_engrams:
            # In live conversational `default` namespace, apply a relevance floor (`>= 0.40`, `top_k=4`)
            # so casual chat isn't polluted by 12 unrelated past turns; isolated benchmark namespaces
            # (`longmem_*`, `locomo_*`) keep full `top_k=12` recall.
            if self.namespace == "default":
                ep_txt = self.recallEpisodes(q_txt, top_k=4, min_score=0.40)
            else:
                ep_txt = self.recallEpisodes(q_txt, top_k=12, min_score=0.0)
        else:
            ep_txt = ""
        parts = [blk for blk in (engram_txt, prod_txt, ep_txt) if blk]
        result = "\n\n".join(parts)
        if len(self._hydrate_cache) > 64:
            self._hydrate_cache.clear()
        self._hydrate_cache[cache_key] = result
        return result
