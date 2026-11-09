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


def _compute_embedding(text: str, timeout: float = 0.8) -> Optional[List[float]]:
    """Query the local (`bge-small-en-v1.5` or Ollama) or cloud embedding server; return None if offline."""
    clean = (text or "").strip()
    if not clean:
        return None
    emb_url = os.environ.get("ACTORII_EMBED_ENDPOINT", EMBEDDING_API_URL).strip()
    if not emb_url or emb_url.lower() in ("none", "off", "disabled"):
        return None
    emb_model = os.environ.get("ACTORII_EMBED_MODEL", EMBEDDING_MODEL_NAME).strip() or EMBEDDING_MODEL_NAME
    try:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        emb_key = (
            os.environ.get("ACTORII_EMBED_API_KEY", "").strip()
            or os.environ.get("ACTORII_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        if emb_key:
            headers["Authorization"] = f"Bearer {emb_key}"
        payload = json.dumps({"input": clean[:1000], "model": emb_model}).encode("utf-8")
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
            return [float(x) for x in emb]
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

    @actor_query
    def getNamespace(self) -> str:
        """Return the active Neo4j memory namespace."""
        return self.namespace

    @actor_command
    def setNamespace(self, namespace: str) -> str:
        """Switch the active Neo4j memory namespace at runtime."""
        ns = _norm_sym(namespace).strip() or "default"
        self.namespace = ns
        return f"NAMESPACE_SET:{self.namespace}"

    def _run_cypher(
        self,
        statement: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
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
        return f"EPISODE_ARCHIVED:{s_id}"

    @actor_command
    def clearEpisodes(self) -> int:
        """Delete all `:ActorEpisode` nodes in the active namespace."""
        _, rows = self._run_cypher(
            """
            MATCH (ep:ActorEpisode {namespace: $ns})
            WITH ep, count(ep) AS c
            DETACH DELETE ep
            RETURN c
            """,
            {"ns": self.namespace},
        )
        return int(rows[0]["c"]) if rows and "c" in rows[0] else 0

    @actor_query
    def recallEpisodes(self, query_text: str, top_k: int = 5) -> str:
        """Retrieve top-K dated `:ActorEpisode` memories from Neo4j via hybrid `:8084` vector + lexical + chronological ranking."""
        import os as _os

        if _os.environ.get("ACTORII_BENCH_STRICT") == "1" and self.namespace == "default":
            return ""
        q_str = str(query_text or "").strip()
        if not q_str:
            return ""
        try:
            _, rows = self._run_cypher(
                """
                MATCH (ep:ActorEpisode {namespace: $ns})
                RETURN ep.session_id AS session_id,
                       coalesce(ep.session_date, '') AS session_date,
                       coalesce(ep.turn_range, '') AS turn_range,
                       ep.content AS content,
                       ep.embedding AS embedding,
                       coalesce(ep.created_at, 0) AS created_at
                ORDER BY created_at ASC
                """,
                {"ns": self.namespace},
            )
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
            selected_map: Dict[int, Tuple[float, int, Dict[str, Any]]] = {}
            seen_sessions = set()
            # Ensure the highest-scoring episode from each distinct session with non-trivial relevance is represented
            for item_tuple in scored:
                sc_val, i_val, r_dict = item_tuple
                sid_val = str(r_dict.get("session_id") or "")
                if sid_val and sid_val not in seen_sessions and sc_val >= 0.25 and len(selected_map) < max(2, k // 2):
                    seen_sessions.add(sid_val)
                    selected_map[i_val] = item_tuple
            for item_tuple in scored:
                if len(selected_map) >= k:
                    break
                i_val = item_tuple[1]
                if i_val not in selected_map:
                    selected_map[i_val] = item_tuple

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

    @actor_query
    def hydrateContext(self, goal_text: str = "") -> str:
        """1993 Actor-callable entry point for `AgentHarness>>selectEngramContext`."""
        q_txt = str(goal_text or "")
        engram_txt = self.hydrateRelevantEngrams(q_txt)
        ep_txt = self.recallEpisodes(q_txt, top_k=12) if self.enable_engrams else ""
        if engram_txt and ep_txt:
            return f"{engram_txt}\n\n{ep_txt}"
        return engram_txt or ep_txt
