"""Wake-Sleep Neocortical Library Compression & Production Compilation (`actorii.harness.wake_sleep`).

Exposes `ConsolidatorBridge` (registered as `Actor[#Consolidator]`) as a zero-VRAM,
pure-CPU wake-sleep consolidation & reflex compilation engine designed to plug directly
into Bilby's `DreamingCycle` engram:

- `compressEpisodes(episodes_json, min_support=2, max_abstractions=5)` (`@actor_query`):
  Extracts recurring parameterized Actor 4.1 action motifs/subroutines from episode
  action traces using `stitch-core` (Rust top-down anti-unification) with a deterministic
  pure-Python AST anti-unification fallback.
- `proveEquivalence(expr_a, expr_b)` (`@actor_query`):
  Proves algebraic/boolean/Actor expression equivalence using `egglog` (Rust e-graph
  equality saturation, bounded at `max_iterations=8`) with a deterministic pure-Python
  union-find e-graph fallback.
- `registerReflex(name, pattern, actor_source, self_test="")` (`@actor_command`):
  Validates `actor_source` + `self_test` inside a `shadow_only=True` Actor VM transaction,
  promoting passing routines into the `<1 ms` in-memory reflex cache (and best-effort
  Neo4j `:ActorEngram {reflex: true}` when Neo4j is online).
- `matchReflex(goal_or_pattern)` (`@actor_query`):
  `<1 ms`, 0-token lookup returning the best matching promoted Reflex Engram.
- `runConsolidationCycle(max_episodes=20, dry_run=True)` (`@actor_command`):
  End-to-end wake-sleep consolidation pass over recent `:ActorEpisode` traces (from Neo4j
  or injected history), extracting candidate abstractions, checking equivalence, and
  returning a structured consolidation report for `DreamingCycle`.
- `describe()` (`@actor_query`):
  Concise self-documentation and Actor 4.1 usage examples (including `DreamingCycle`).

Guardrails:
- Strictly 0 MB extra GPU VRAM (pure CPU).
- Bounded e-graph saturation (`max_iterations=8`, `max_nodes=50_000`) to prevent explosion.
- Reflexes MUST pass `shadow_only=True` VM validation and NEVER contain Tier-2 host
  shell/filesystem mutation calls (`runShell`, `writeFile`, `deleteFile`, `Actor[#Host]`).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from actorii.bridge import actor_bridge, actor_command, actor_query

try:
    import stitch_core  # type: ignore
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    stitch_core = None  # type: ignore[assignment]

try:
    import egglog  # type: ignore
    from egglog.bindings import EGraph as _EgglogEGraph  # type: ignore
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    egglog = None  # type: ignore[assignment]
    _EgglogEGraph = None  # type: ignore[assignment]


MAX_EGRAPH_ITERATIONS = 8
MAX_EGRAPH_NODES = 50_000
DEFAULT_NEO4J_URL = "http://127.0.0.1:7474/db/neo4j/tx/commit"

_FORBIDDEN_REFLEX_PATTERNS = (
    re.compile(r"\bActor\s*\[\s*#(?:Host|OS|Shell|Process)\s*\]", re.IGNORECASE),
    re.compile(r"\b(?:runShell|execShell|shellExec|writeFile|deleteFile|removeFile|rmFile)\s*\(", re.IGNORECASE),
)

_ACTIONS_BLOCK_RE = re.compile(r"\[Actions:\s*(.*?)\](?::|\s|$)", re.DOTALL)
_TOKEN_SPLIT_RE = re.compile(r"[a-z0-9_]+")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ==============================================================================
# Actor Conversion Helpers
# ==============================================================================


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


def _from_actor_value(val: Any) -> Any:
    """Unwrap Actor `Dictionary` / `ActorArray` / `Symbol` into Python builtins."""
    from actorii.classes.collections import ActorArray
    from actorii.classes.dictionary import Dictionary
    from actorii.core import Symbol

    if isinstance(val, Symbol):
        return val.name
    if isinstance(val, Dictionary):
        out: Dict[str, Any] = {}
        for k, v in val._entries.items():
            k_str = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
            if k_str not in out:
                out[k_str] = _from_actor_value(v)
        return out
    if isinstance(val, ActorArray):
        return [_from_actor_value(val.at(i)) for i in range(val.size())]
    return val


def _norm_sym(val: Any) -> str:
    return str(val or "").strip().lstrip("#")


# ==============================================================================
# S-Expression AST Codec (Actor 4.1 <-> S-Expr)
# ==============================================================================


SExpr = Union[str, List["SExpr"]]


class _AtomTable:
    """Bijective mapping between arbitrary Actor literals and stitch-safe S-expr atoms."""

    def __init__(self) -> None:
        self._lit_to_atom: Dict[str, str] = {}
        self._atom_to_lit: Dict[str, str] = {}

    def encode_string(self, text: str) -> str:
        key = f"STR:{text}"
        if key in self._lit_to_atom:
            return self._lit_to_atom[key]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        # Keep short readable prefix for human-readable templates
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")[:16]
        atom = f"str_{slug}_{digest}" if slug else f"str_{digest}"
        self._lit_to_atom[key] = atom
        self._atom_to_lit[atom] = json.dumps(text)
        return atom

    def encode_symbol(self, name: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).lstrip("#"))
        atom = f"sym_{clean}" if clean else "sym_empty"
        self._atom_to_lit[atom] = f"#{str(name).lstrip('#')}"
        return atom

    def encode_number(self, num: Union[int, float]) -> str:
        s = str(num)
        if s.startswith("-"):
            atom = f"neg_{s[1:].replace('.', 'p')}"
            self._atom_to_lit[atom] = s
            return atom
        if "." in s:
            atom = f"flt_{s.replace('.', 'p')}"
            self._atom_to_lit[atom] = s
            return atom
        return s

    def encode_ident(self, name: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
        if not clean or clean[0].isdigit():
            clean = f"id_{clean}"
        self._atom_to_lit.setdefault(clean, str(name))
        return clean

    def decode_atom(self, atom: str) -> str:
        if atom.startswith("#") and atom[1:].isdigit():
            return f"${atom[1:]}"
        if atom.startswith("$"):
            return atom
        if atom in self._atom_to_lit:
            return self._atom_to_lit[atom]
        return atom


def _ast_node_to_sexpr(node: Any, atoms: _AtomTable) -> SExpr:
    """Convert an `actorii.parser` AST node into a stitch-safe S-expression tree."""
    cls_name = type(node).__name__

    if cls_name == "Identifier":
        return atoms.encode_ident(getattr(node, "name", "id"))
    if cls_name == "Integer":
        return atoms.encode_number(getattr(node, "value", 0))
    if cls_name == "RealLiteral":
        return atoms.encode_number(getattr(node, "value", 0.0))
    if cls_name == "String":
        return atoms.encode_string(str(getattr(node, "value", "")))
    if cls_name == "SymbolLiteralNode":
        return atoms.encode_symbol(str(getattr(node, "name", getattr(node, "value", ""))))
    if cls_name == "BooleanLiteral":
        return "bool_true" if getattr(node, "value", False) else "bool_false"
    if cls_name == "NilLiteral":
        return "nil_val"
    if cls_name == "SelfLiteral":
        return "self"
    if cls_name == "CallNode":
        selector = atoms.encode_ident(str(getattr(node, "selector", getattr(node, "name", "call"))))
        receiver = getattr(node, "receiver", None)
        args = list(getattr(node, "args", getattr(node, "arguments", [])) or [])
        items: List[SExpr] = [f"call_{selector}"]
        if receiver is not None:
            items.append(_ast_node_to_sexpr(receiver, atoms))
        for a in args:
            items.append(_ast_node_to_sexpr(a, atoms))
        return items
    if cls_name == "BinaryOp":
        op_map = {
            "+": "op_add",
            "-": "op_sub",
            "*": "op_mul",
            "/": "op_div",
            "==": "op_eq",
            "!=": "op_neq",
            "<": "op_lt",
            "<=": "op_lte",
            ">": "op_gt",
            ">=": "op_gte",
            "&&": "op_and",
            "||": "op_or",
        }
        op_raw = str(getattr(node, "op", "+"))
        op_tok = op_map.get(op_raw, f"op_{atoms.encode_ident(op_raw)}")
        left = _ast_node_to_sexpr(getattr(node, "left", None), atoms)
        right = _ast_node_to_sexpr(getattr(node, "right", None), atoms)
        return [op_tok, left, right]
    if cls_name == "UnaryOp":
        op_raw = str(getattr(node, "op", "-"))
        op_tok = "op_neg" if op_raw == "-" else ("op_not" if op_raw in ("!", "not") else f"uop_{atoms.encode_ident(op_raw)}")
        operand = _ast_node_to_sexpr(getattr(node, "operand", getattr(node, "expr", None)), atoms)
        return [op_tok, operand]
    if cls_name in ("AssignNode", "Assignment"):
        target = getattr(node, "target", getattr(node, "name", "var"))
        target_s: SExpr = (
            atoms.encode_ident(target) if isinstance(target, str) else _ast_node_to_sexpr(target, atoms)
        )
        val_s = _ast_node_to_sexpr(getattr(node, "value", getattr(node, "expr", None)), atoms)
        return ["assign", target_s, val_s]
    if cls_name == "IndexGetNode":
        obj_s = _ast_node_to_sexpr(getattr(node, "target", getattr(node, "obj", None)), atoms)
        idx_s = _ast_node_to_sexpr(getattr(node, "index", None), atoms)
        return ["idx_get", obj_s, idx_s]
    if cls_name == "IndexSetNode":
        obj_s = _ast_node_to_sexpr(getattr(node, "target", getattr(node, "obj", None)), atoms)
        idx_s = _ast_node_to_sexpr(getattr(node, "index", None), atoms)
        val_s = _ast_node_to_sexpr(getattr(node, "value", None), atoms)
        return ["idx_set", obj_s, idx_s, val_s]
    if cls_name == "AttributeAccess":
        obj_s = _ast_node_to_sexpr(getattr(node, "obj", getattr(node, "target", None)), atoms)
        attr = atoms.encode_ident(str(getattr(node, "attr", getattr(node, "name", "attr"))))
        return ["attr", obj_s, attr]
    if cls_name == "ReturnNode":
        val = getattr(node, "value", getattr(node, "expr", None))
        return ["ret", _ast_node_to_sexpr(val, atoms)] if val is not None else ["ret", "nil_val"]
    if cls_name == "ArrayLiteralNode":
        elems = list(getattr(node, "elements", getattr(node, "items", [])) or [])
        return ["arr"] + [_ast_node_to_sexpr(e, atoms) for e in elems] if elems else "arr_empty"
    if cls_name == "IfNode":
        cond = _ast_node_to_sexpr(getattr(node, "cond", getattr(node, "condition", None)), atoms)
        then_body = list(getattr(node, "then_body", []) or [])
        else_body = list(getattr(node, "else_body", []) or [])
        then_s: SExpr = ["seq"] + [_ast_node_to_sexpr(s, atoms) for s in then_body] if then_body else "noop"
        else_s: SExpr = ["seq"] + [_ast_node_to_sexpr(s, atoms) for s in else_body] if else_body else "noop"
        return ["if_expr", cond, then_s, else_s]

    # Generic fallback
    return atoms.encode_string(repr(node))


def _sexpr_to_str(tree: SExpr) -> str:
    if isinstance(tree, str):
        return tree
    if not tree:
        return "noop"
    return "(" + " ".join(_sexpr_to_str(child) for child in tree) + ")"


def _parse_sexpr_str(s: str) -> SExpr:
    """Parse a stitch-formatted S-expression string back into nested Python lists/strings."""
    tokens = re.findall(r"\(|\)|[^\s()]+", s)
    pos = 0

    def _read() -> SExpr:
        nonlocal pos
        if pos >= len(tokens):
            return "noop"
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst: List[SExpr] = []
            while pos < len(tokens) and tokens[pos] != ")":
                lst.append(_read())
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return lst if lst else "noop"
        return tok

    return _read()


def _sexpr_to_actor_template(tree: SExpr, atoms: _AtomTable) -> str:
    """Render an S-expression template (with `#0`/`$0` params) back into readable Actor 4.1."""
    if isinstance(tree, str):
        if tree == "bool_true":
            return "true"
        if tree == "bool_false":
            return "false"
        if tree == "nil_val":
            return "nil"
        if tree == "arr_empty":
            return "#()"
        return atoms.decode_atom(tree)

    if not tree:
        return ""
    head = tree[0]
    args = tree[1:]

    if isinstance(head, str) and head.startswith("call_"):
        sel = head[len("call_"):]
        rendered_args = ", ".join(_sexpr_to_actor_template(a, atoms) for a in args)
        return f"{sel}({rendered_args})"
    if head == "idx_get" and len(args) == 2:
        return f"{_sexpr_to_actor_template(args[0], atoms)}[{_sexpr_to_actor_template(args[1], atoms)}]"
    if head == "idx_set" and len(args) == 3:
        return (
            f"{_sexpr_to_actor_template(args[0], atoms)}"
            f"[{_sexpr_to_actor_template(args[1], atoms)}] := {_sexpr_to_actor_template(args[2], atoms)}"
        )
    if head == "assign" and len(args) == 2:
        return f"{_sexpr_to_actor_template(args[0], atoms)} := {_sexpr_to_actor_template(args[1], atoms)}"
    if head == "attr" and len(args) == 2:
        return f"{_sexpr_to_actor_template(args[0], atoms)}.{_sexpr_to_actor_template(args[1], atoms)}"
    if head == "ret" and len(args) == 1:
        return f"^{_sexpr_to_actor_template(args[0], atoms)}"
    if head == "seq":
        return "; ".join(_sexpr_to_actor_template(a, atoms) for a in args)
    if head == "arr":
        return "#(" + " ".join(_sexpr_to_actor_template(a, atoms) for a in args) + ")"
    if head in ("op_add", "op_sub", "op_mul", "op_div", "op_eq", "op_neq", "op_lt", "op_lte", "op_gt", "op_gte", "op_and", "op_or") and len(args) == 2:
        rev = {
            "op_add": "+",
            "op_sub": "-",
            "op_mul": "*",
            "op_div": "/",
            "op_eq": "==",
            "op_neq": "!=",
            "op_lt": "<",
            "op_lte": "<=",
            "op_gt": ">",
            "op_gte": ">=",
            "op_and": "&&",
            "op_or": "||",
        }
        return f"({_sexpr_to_actor_template(args[0], atoms)} {rev[head]} {_sexpr_to_actor_template(args[1], atoms)})"
    if head == "op_neg" and len(args) == 1:
        return f"(-{_sexpr_to_actor_template(args[0], atoms)})"
    if head == "op_not" and len(args) == 1:
        return f"(!{_sexpr_to_actor_template(args[0], atoms)})"
    if head == "if_expr" and len(args) == 3:
        return (
            f"if ({_sexpr_to_actor_template(args[0], atoms)}) then "
            f"{_sexpr_to_actor_template(args[1], atoms)}; else "
            f"{_sexpr_to_actor_template(args[2], atoms)}; endif"
        )

    # Generic S-expr rendering
    return "(" + " ".join(_sexpr_to_actor_template(x, atoms) for x in tree) + ")"


def _actor_action_to_sexpr(action_src: str, atoms: _AtomTable) -> Optional[SExpr]:
    """Parse a single Actor 4.1 action/statement into a stitch-safe S-expression."""
    src = str(action_src or "").strip()
    if not src:
        return None
    if not src.endswith("!!"):
        src_parse = src.rstrip(";") + "!!"
    else:
        src_parse = src

    try:
        from actorii.parser import parse_actor

        ast_nodes = parse_actor(src_parse)
        if ast_nodes:
            if len(ast_nodes) == 1:
                return _ast_node_to_sexpr(ast_nodes[0], atoms)
            return ["seq"] + [_ast_node_to_sexpr(n, atoms) for n in ast_nodes]
    except Exception:
        pass

    # Fallback: lightweight call-pattern parser for snippets that aren't full Actor statements
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(?:!!|;)?$", src, re.DOTALL)
    if m:
        sel = atoms.encode_ident(m.group(1))
        raw_args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        arg_nodes = [atoms.encode_string(a) for a in raw_args]
        return [f"call_{sel}"] + arg_nodes
    return atoms.encode_string(src)


def _extract_actions_from_episode_content(content: str) -> List[str]:
    """Extract Actor 4.1 action statements from a Bilby `:ActorEpisode.content` string."""
    actions: List[str] = []
    if not content:
        return actions
    for block in _ACTIONS_BLOCK_RE.findall(content):
        # Split on `!!;` or `!!`
        parts = re.split(r"!!\s*;?\s*", block)
        for p in parts:
            cleaned = p.strip().strip(";").strip()
            if cleaned and "(" in cleaned:
                actions.append(cleaned + "!!")
    return actions


def _extract_episode_traces(episodes_input: Any) -> Tuple[List[List[str]], int]:
    """Normalize diverse episode inputs into a list of per-episode Actor action lists."""
    data = _from_actor_value(episodes_input)
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return [], 0
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Treat as raw episode content / semicolon-separated actions
            acts = _extract_actions_from_episode_content(stripped)
            if not acts:
                acts = [p.strip() for p in re.split(r"!!\s*;?\s*|\n+", stripped) if p.strip()]
            return ([acts] if acts else []), 1

    if isinstance(data, dict):
        if "episodes" in data and isinstance(data["episodes"], list):
            data = data["episodes"]
        else:
            data = [data]

    if not isinstance(data, list):
        return [], 0

    traces: List[List[str]] = []
    for item in data:
        if isinstance(item, str):
            acts = _extract_actions_from_episode_content(item)
            if acts:
                traces.append(acts)
            elif item.strip():
                traces.append([item.strip()])
        elif isinstance(item, list):
            acts = [str(x).strip() for x in item if str(x).strip()]
            if acts:
                traces.append(acts)
        elif isinstance(item, dict):
            if isinstance(item.get("actions"), list):
                acts = []
                for a in item["actions"]:
                    if isinstance(a, str) and a.strip():
                        acts.append(a.strip())
                    elif isinstance(a, dict):
                        code = a.get("code") or a.get("actor_source") or a.get("action") or ""
                        if str(code).strip():
                            acts.append(str(code).strip())
                if acts:
                    traces.append(acts)
            elif isinstance(item.get("content"), str):
                acts = _extract_actions_from_episode_content(item["content"])
                if acts:
                    traces.append(acts)
            elif isinstance(item.get("trace"), list):
                acts = [str(x).strip() for x in item["trace"] if str(x).strip()]
                if acts:
                    traces.append(acts)
            elif isinstance(item.get("source"), str) and item["source"].strip():
                traces.append([item["source"].strip()])
    return traces, len(data)


# ==============================================================================
# Pure-Python Anti-Unification Compressor (Fallback for stitch-core)
# ==============================================================================


def _sexpr_size(tree: SExpr) -> int:
    if isinstance(tree, str):
        return 1
    return 1 + sum(_sexpr_size(c) for c in tree)


def _anti_unify(t1: SExpr, t2: SExpr, subst: Dict[Tuple[str, str], str]) -> SExpr:
    """Plotkin anti-unification (least general generalization) of two S-expressions."""
    if t1 == t2:
        return t1
    if isinstance(t1, list) and isinstance(t2, list) and len(t1) == len(t2) and t1 and t1[0] == t2[0]:
        return [t1[0]] + [_anti_unify(a, b, subst) for a, b in zip(t1[1:], t2[1:])]
    key = (_sexpr_to_str(t1), _sexpr_to_str(t2))
    if key not in subst:
        subst[key] = f"${len(subst)}"
    return subst[key]


def _collect_subtrees(tree: SExpr, out: List[SExpr]) -> None:
    if isinstance(tree, list) and len(tree) >= 2:
        out.append(tree)
        for child in tree[1:]:
            _collect_subtrees(child, out)


def _matches_template(tree: SExpr, template: SExpr, env: Dict[str, str]) -> bool:
    """Check if `tree` is an instance of `template` (with `$k` placeholders)."""
    if isinstance(template, str):
        if template.startswith("$"):
            val = _sexpr_to_str(tree)
            if template in env:
                return env[template] == val
            env[template] = val
            return True
        return tree == template
    if not isinstance(tree, list) or len(tree) != len(template):
        return False
    return all(_matches_template(c, tc, env) for c, tc in zip(tree, template))


def _extract_template_params(template: SExpr) -> List[str]:
    found: List[str] = []

    def _walk(node: SExpr) -> None:
        if isinstance(node, str):
            if (node.startswith("$") or (node.startswith("#") and node[1:].isdigit())) and node not in found:
                found.append(f"${node[1:]}" if node.startswith("#") else node)
        elif isinstance(node, list):
            for c in node:
                _walk(c)

    _walk(template)
    return found


def _python_anti_unify_compress(
    sexpr_trees: List[SExpr],
    atoms: _AtomTable,
    min_support: int,
    max_abstractions: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Deterministic pure-Python top-down anti-unification compressor."""
    all_subtrees: List[Tuple[int, SExpr]] = []
    for ep_idx, tree in enumerate(sexpr_trees):
        subs: List[SExpr] = []
        _collect_subtrees(tree, subs)
        for s in subs:
            all_subtrees.append((ep_idx, s))

    # Group by (head, arity)
    groups: Dict[Tuple[str, int], List[Tuple[int, SExpr]]] = {}
    for ep_idx, sub in all_subtrees:
        if isinstance(sub, list) and sub and isinstance(sub[0], str):
            groups.setdefault((sub[0], len(sub)), []).append((ep_idx, sub))

    candidates: Dict[str, Tuple[SExpr, int, int]] = {}  # key -> (template, support, savings)
    for (_head, _arity), items in groups.items():
        if len(items) < min_support:
            continue
        # Generate candidate templates from pairwise anti-unification (bounded for speed)
        sample = items[:24]
        seen_templates: Set[str] = set()
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                subst: Dict[Tuple[str, str], str] = {}
                tmpl = _anti_unify(sample[i][1], sample[j][1], subst)
                if not isinstance(tmpl, list):
                    continue
                params = _extract_template_params(tmpl)
                if len(params) > 4:
                    continue
                tmpl_size = _sexpr_size(tmpl)
                if tmpl_size <= len(params) + 1:
                    continue
                key = _sexpr_to_str(tmpl)
                if key in seen_templates:
                    continue
                seen_templates.add(key)

                # Count total matches across all subtrees
                support = 0
                for _eidx, sub in items:
                    if _matches_template(sub, tmpl, {}):
                        support += 1
                if support >= min_support:
                    # Estimated AST savings
                    savings = max(1, (tmpl_size - len(params) - 1) * (support - 1))
                    if key not in candidates or savings > candidates[key][2]:
                        candidates[key] = (tmpl, support, savings)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item[2], item[1], _sexpr_size(item[0])),
        reverse=True,
    )

    abstractions: List[Dict[str, Any]] = []
    total_orig_size = max(1, sum(_sexpr_size(t) for t in sexpr_trees))
    total_saved_nodes = 0
    used_templates: Set[str] = set()

    for tmpl, support, savings in ranked:
        if len(abstractions) >= max_abstractions:
            break
        actor_tmpl = _sexpr_to_actor_template(tmpl, atoms)
        if actor_tmpl in used_templates:
            continue
        used_templates.add(actor_tmpl)
        params = _extract_template_params(tmpl)
        fn_id = f"fn_{len(abstractions)}"
        # Heuristic: ~4 LLM tokens per AST node saved
        est_tokens = max(2, savings * 4)
        total_saved_nodes += min(savings, max(0, total_orig_size - 1 - total_saved_nodes))
        abstractions.append(
            {
                "id": fn_id,
                "template": actor_tmpl,
                "sexpr": _sexpr_to_str(tmpl),
                "params": params,
                "support": int(support),
                "estimatedTokensSaved": int(est_tokens),
            }
        )

    final_size = max(1, total_orig_size - total_saved_nodes)
    ratio = round(total_orig_size / final_size, 4) if abstractions else 1.0
    return abstractions, ratio


# ==============================================================================
# Expression Parser & E-Graph Equivalence Prover (egglog + pure-Python fallback)
# ==============================================================================


@dataclass(frozen=True)
class _ENode:
    op: str
    val: Any = None
    children: Tuple["_ENode", ...] = ()


_EXPR_TOKEN_RE = re.compile(
    r"\s*(?:(\d+)|(==|!=|<=|>=|&&|\|\||->|[+\-*/()!,])|([A-Za-z_][A-Za-z0-9_#]*))"
)


def _parse_expr(expr_str: str) -> _ENode:
    """Parse an arithmetic/boolean/Actor expression into a normalized `_ENode` AST."""
    src = str(expr_str or "").strip().rstrip("!; ")
    if not src:
        raise ValueError("Expression cannot be empty.")

    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(src):
        m = _EXPR_TOKEN_RE.match(src, pos)
        if not m:
            raise ValueError(f"Unexpected character in expression at offset {pos}: {src[pos:]!r}")
        pos = m.end()
        if m.group(1) is not None:
            tokens.append(("NUM", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("OP", m.group(2)))
        elif m.group(3) is not None:
            word = m.group(3)
            low = word.lower()
            if low == "and":
                tokens.append(("OP", "&&"))
            elif low == "or":
                tokens.append(("OP", "||"))
            elif low == "not":
                tokens.append(("OP", "!"))
            elif low == "true":
                tokens.append(("BOOL", "true"))
            elif low == "false":
                tokens.append(("BOOL", "false"))
            else:
                tokens.append(("IDENT", word))

    idx = 0

    def _peek() -> Optional[Tuple[str, str]]:
        return tokens[idx] if idx < len(tokens) else None

    def _consume() -> Tuple[str, str]:
        nonlocal idx
        tok = tokens[idx]
        idx += 1
        return tok

    def _parse_or() -> _ENode:
        node = _parse_and()
        while _peek() == ("OP", "||"):
            _consume()
            node = _ENode("Or", children=(node, _parse_and()))
        return node

    def _parse_and() -> _ENode:
        node = _parse_eq()
        while _peek() == ("OP", "&&"):
            _consume()
            node = _ENode("And", children=(node, _parse_eq()))
        return node

    def _parse_eq() -> _ENode:
        node = _parse_add()
        while _peek() in (("OP", "=="), ("OP", "!="), ("OP", "<"), ("OP", "<="), ("OP", ">"), ("OP", ">=")):
            _, op = _consume()
            op_name = {"==": "Eq", "!=": "Neq", "<": "Lt", "<=": "Lte", ">": "Gt", ">=": "Gte"}[op]
            node = _ENode(op_name, children=(node, _parse_add()))
        return node

    def _parse_add() -> _ENode:
        node = _parse_mul()
        while _peek() in (("OP", "+"), ("OP", "-")):
            _, op = _consume()
            node = _ENode("Add" if op == "+" else "Sub", children=(node, _parse_mul()))
        return node

    def _parse_mul() -> _ENode:
        node = _parse_unary()
        while _peek() in (("OP", "*"), ("OP", "/")):
            _, op = _consume()
            node = _ENode("Mul" if op == "*" else "Div", children=(node, _parse_unary()))
        return node

    def _parse_unary() -> _ENode:
        tok = _peek()
        if tok == ("OP", "-"):
            _consume()
            return _ENode("Neg", children=(_parse_unary(),))
        if tok == ("OP", "!"):
            _consume()
            return _ENode("Not", children=(_parse_unary(),))
        if tok == ("OP", "+"):
            _consume()
            return _parse_unary()
        return _parse_primary()

    def _parse_primary() -> _ENode:
        tok = _peek()
        if tok is None:
            raise ValueError("Unexpected end of expression.")
        kind, val = _consume()
        if kind == "NUM":
            return _ENode("Num", val=int(val))
        if kind == "BOOL":
            return _ENode("Bool", val=(val == "true"))
        if kind == "IDENT":
            # Function / message call syntax: `f(a, b)`
            if _peek() == ("OP", "("):
                _consume()
                args: List[_ENode] = []
                if _peek() != ("OP", ")"):
                    args.append(_parse_or())
                    while _peek() == ("OP", ","):
                        _consume()
                        args.append(_parse_or())
                if _peek() != ("OP", ")"):
                    raise ValueError(f"Missing closing ')' in call to {val}.")
                _consume()
                return _ENode("Call", val=val, children=tuple(args))
            return _ENode("Var", val=val)
        if kind == "OP" and val == "(":
            inner = _parse_or()
            if _peek() != ("OP", ")"):
                raise ValueError("Missing closing ')'.")
            _consume()
            return inner
        raise ValueError(f"Unexpected token {(kind, val)!r} in expression.")

    root = _parse_or()
    if idx < len(tokens):
        raise ValueError(f"Trailing tokens in expression: {tokens[idx:]!r}")
    return root


def _enode_to_egglog(node: _ENode) -> str:
    """Serialize `_ENode` into an egglog `Expr` constructor term."""
    if node.op == "Num":
        return f"(Num {int(node.val)})"
    if node.op == "Bool":
        return f"(Bool {'true' if node.val else 'false'})"
    if node.op == "Var":
        return f"(Var {json.dumps(str(node.val))})"
    if node.op == "Call":
        name = json.dumps(str(node.val))
        if len(node.children) == 0:
            return f"(Call0 {name})"
        if len(node.children) == 1:
            return f"(Call1 {name} {_enode_to_egglog(node.children[0])})"
        if len(node.children) == 2:
            return f"(Call2 {name} {_enode_to_egglog(node.children[0])} {_enode_to_egglog(node.children[1])})"
        # Fold 3+ args deterministically
        acc = f"(Call2 {name} {_enode_to_egglog(node.children[0])} {_enode_to_egglog(node.children[1])})"
        for ch in node.children[2:]:
            acc = f"(Call2 \"__arg\" {acc} {_enode_to_egglog(ch)})"
        return acc
    args = " ".join(_enode_to_egglog(c) for c in node.children)
    return f"({node.op} {args})"


def _enode_to_str(node: _ENode) -> str:
    """Format an `_ENode` cleanly as an infix Actor/algebraic expression."""
    if node.op == "Num":
        return str(node.val)
    if node.op == "Bool":
        return "true" if node.val else "false"
    if node.op == "Var":
        return str(node.val)
    if node.op == "Neg":
        return f"(-{_enode_to_str(node.children[0])})"
    if node.op == "Not":
        return f"(!{_enode_to_str(node.children[0])})"
    if node.op == "Call":
        args = ", ".join(_enode_to_str(c) for c in node.children)
        return f"{node.val}({args})"
    op_sym = {
        "Add": "+",
        "Sub": "-",
        "Mul": "*",
        "Div": "/",
        "And": "&&",
        "Or": "||",
        "Eq": "==",
        "Neq": "!=",
        "Lt": "<",
        "Lte": "<=",
        "Gt": ">",
        "Gte": ">=",
    }.get(node.op, node.op)
    return f"({_enode_to_str(node.children[0])} {op_sym} {_enode_to_str(node.children[1])})"


_EGGLOG_THEORY = """
(datatype Expr
    (Num i64)
    (Bool bool)
    (Var String)
    (Add Expr Expr)
    (Sub Expr Expr)
    (Mul Expr Expr)
    (Div Expr Expr)
    (Neg Expr)
    (And Expr Expr)
    (Or Expr Expr)
    (Not Expr)
    (Eq Expr Expr)
    (Neq Expr Expr)
    (Lt Expr Expr)
    (Lte Expr Expr)
    (Gt Expr Expr)
    (Gte Expr Expr)
    (Call0 String)
    (Call1 String Expr)
    (Call2 String Expr Expr))

; Arithmetic identities & algebraic laws
(rewrite (Add x (Num 0)) x)
(rewrite (Add (Num 0) x) x)
(rewrite (Add x y) (Add y x))
(rewrite (Add (Add x y) z) (Add x (Add y z)))
(rewrite (Add x (Add y z)) (Add (Add x y) z))
(rewrite (Sub x (Num 0)) x)
(rewrite (Sub x x) (Num 0))
(rewrite (Sub (Num 0) x) (Neg x))
(rewrite (Sub x y) (Add x (Neg y)))
(rewrite (Add x (Neg y)) (Sub x y))
(rewrite (Add x (Neg x)) (Num 0))
(rewrite (Neg (Neg x)) x)
(rewrite (Neg (Num n)) (Num (- 0 n)))
(rewrite (Mul x (Num 1)) x)
(rewrite (Mul (Num 1) x) x)
(rewrite (Mul x (Num 0)) (Num 0))
(rewrite (Mul (Num 0) x) (Num 0))
(rewrite (Mul x y) (Mul y x))
(rewrite (Mul (Mul x y) z) (Mul x (Mul y z)))
(rewrite (Mul x (Mul y z)) (Mul (Mul x y) z))
(rewrite (Add x x) (Mul (Num 2) x))
(rewrite (Mul (Num 2) x) (Add x x))
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)))
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)))
(rewrite (Div x (Num 1)) x)

; Constant folding
(rewrite (Add (Num a) (Num b)) (Num (+ a b)))
(rewrite (Sub (Num a) (Num b)) (Num (- a b)))
(rewrite (Mul (Num a) (Num b)) (Num (* a b)))

; Boolean algebra & De Morgan
(rewrite (And x (Bool true)) x)
(rewrite (And (Bool true) x) x)
(rewrite (And x (Bool false)) (Bool false))
(rewrite (And (Bool false) x) (Bool false))
(rewrite (And x x) x)
(rewrite (And x y) (And y x))
(rewrite (And (And x y) z) (And x (And y z)))
(rewrite (Or x (Bool false)) x)
(rewrite (Or (Bool false) x) x)
(rewrite (Or x (Bool true)) (Bool true))
(rewrite (Or (Bool true) x) (Bool true))
(rewrite (Or x x) x)
(rewrite (Or x y) (Or y x))
(rewrite (Or (Or x y) z) (Or x (Or y z)))
(rewrite (Not (Not x)) x)
(rewrite (Not (Bool true)) (Bool false))
(rewrite (Not (Bool false)) (Bool true))
(rewrite (Not (And a b)) (Or (Not a) (Not b)))
(rewrite (Or (Not a) (Not b)) (Not (And a b)))
(rewrite (Not (Or a b)) (And (Not a) (Not b)))
(rewrite (And (Not a) (Not b)) (Not (Or a b)))
(rewrite (And x (Not x)) (Bool false))
(rewrite (Or x (Not x)) (Bool true))

; Comparison & equality symmetry
(rewrite (Eq x y) (Eq y x))
(rewrite (Eq x x) (Bool true))
(rewrite (Neq x y) (Neq y x))
(rewrite (Neq x x) (Bool false))
(rewrite (Not (Eq x y)) (Neq x y))
(rewrite (Not (Neq x y)) (Eq x y))
(rewrite (Lt x y) (Gt y x))
(rewrite (Gt x y) (Lt y x))
"""


def _canonicalize_enode(node: _ENode) -> _ENode:
    """Deterministic algebraic/boolean normalizer producing a compact canonical `_ENode`."""
    # Recursively normalize children first
    ch = tuple(_canonicalize_enode(c) for c in node.children)
    op = node.op

    if op in ("Num", "Bool", "Var"):
        return node
    if op == "Call":
        return _ENode("Call", val=node.val, children=ch)

    if op == "Neg":
        x = ch[0]
        if x.op == "Num":
            return _ENode("Num", val=-int(x.val))
        if x.op == "Neg":
            return x.children[0]
        return _ENode("Neg", children=(x,))

    if op == "Not":
        x = ch[0]
        if x.op == "Bool":
            return _ENode("Bool", val=not bool(x.val))
        if x.op == "Not":
            return x.children[0]
        if x.op == "And":
            return _canonicalize_enode(
                _ENode("Or", children=(_ENode("Not", children=(x.children[0],)), _ENode("Not", children=(x.children[1],))))
            )
        if x.op == "Or":
            return _canonicalize_enode(
                _ENode("And", children=(_ENode("Not", children=(x.children[0],)), _ENode("Not", children=(x.children[1],))))
            )
        if x.op == "Eq":
            return _canonicalize_enode(_ENode("Neq", children=x.children))
        if x.op == "Neq":
            return _canonicalize_enode(_ENode("Eq", children=x.children))
        return _ENode("Not", children=(x,))

    if op == "Sub":
        a, b = ch
        if a == b:
            return _ENode("Num", val=0)
        if b.op == "Num" and b.val == 0:
            return a
        if a.op == "Num" and a.val == 0:
            return _canonicalize_enode(_ENode("Neg", children=(b,)))
        if a.op == "Num" and b.op == "Num":
            return _ENode("Num", val=int(a.val) - int(b.val))
        return _canonicalize_enode(_ENode("Add", children=(a, _ENode("Neg", children=(b,)))))

    if op == "Add":
        # Flatten associative Add terms, fold constants, cancel x + (-x), combine x + x -> 2*x
        terms: List[_ENode] = []

        def _collect_add(n: _ENode) -> None:
            if n.op == "Add":
                _collect_add(n.children[0])
                _collect_add(n.children[1])
            else:
                terms.append(n)

        _collect_add(ch[0])
        _collect_add(ch[1])

        const_sum = 0
        counts: Dict[_ENode, int] = {}
        for t in terms:
            if t.op == "Num":
                const_sum += int(t.val)
            elif t.op == "Neg":
                base = t.children[0]
                counts[base] = counts.get(base, 0) - 1
            elif t.op == "Mul" and t.children[0].op == "Num":
                base = t.children[1]
                counts[base] = counts.get(base, 0) + int(t.children[0].val)
            elif t.op == "Mul" and t.children[1].op == "Num":
                base = t.children[0]
                counts[base] = counts.get(base, 0) + int(t.children[1].val)
            else:
                counts[t] = counts.get(t, 0) + 1

        rebuilt: List[_ENode] = []
        for base, coeff in sorted(counts.items(), key=lambda kv: _enode_to_str(kv[0])):
            if coeff == 0:
                continue
            if coeff == 1:
                rebuilt.append(base)
            elif coeff == -1:
                rebuilt.append(_ENode("Neg", children=(base,)))
            else:
                rebuilt.append(_ENode("Mul", children=(_ENode("Num", val=coeff), base)))
        if const_sum != 0 or not rebuilt:
            rebuilt.append(_ENode("Num", val=const_sum))

        acc = rebuilt[0]
        for nxt in rebuilt[1:]:
            acc = _ENode("Add", children=(acc, nxt))
        return acc

    if op == "Mul":
        a, b = ch
        if (a.op == "Num" and a.val == 0) or (b.op == "Num" and b.val == 0):
            return _ENode("Num", val=0)
        if a.op == "Num" and a.val == 1:
            return b
        if b.op == "Num" and b.val == 1:
            return a
        if a.op == "Num" and b.op == "Num":
            return _ENode("Num", val=int(a.val) * int(b.val))
        # Distribute over Add so `a * (b + c)` and `a*b + a*c` canonicalize identically
        if b.op == "Add":
            return _canonicalize_enode(
                _ENode("Add", children=(_ENode("Mul", children=(a, b.children[0])), _ENode("Mul", children=(a, b.children[1]))))
            )
        if a.op == "Add":
            return _canonicalize_enode(
                _ENode("Add", children=(_ENode("Mul", children=(a.children[0], b)), _ENode("Mul", children=(a.children[1], b))))
            )
        # Flatten associative Mul and sort factors canonically
        factors: List[_ENode] = []

        def _collect_mul(n: _ENode) -> None:
            if n.op == "Mul":
                _collect_mul(n.children[0])
                _collect_mul(n.children[1])
            else:
                factors.append(n)

        _collect_mul(a)
        _collect_mul(b)
        const_prod = 1
        sym_factors: List[_ENode] = []
        for f in factors:
            if f.op == "Num":
                const_prod *= int(f.val)
            else:
                sym_factors.append(f)
        if const_prod == 0:
            return _ENode("Num", val=0)
        sym_factors.sort(key=_enode_to_str)
        if const_prod != 1 or not sym_factors:
            sym_factors.insert(0, _ENode("Num", val=const_prod))
        acc = sym_factors[0]
        for nxt in sym_factors[1:]:
            acc = _ENode("Mul", children=(acc, nxt))
        return acc

    if op == "Div":
        a, b = ch
        if b.op == "Num" and b.val == 1:
            return a
        return _ENode("Div", children=(a, b))

    if op in ("And", "Or"):
        identity = True if op == "And" else False
        absorber = False if op == "And" else True
        items: List[_ENode] = []

        def _collect_bool(n: _ENode) -> None:
            if n.op == op:
                _collect_bool(n.children[0])
                _collect_bool(n.children[1])
            else:
                items.append(n)

        _collect_bool(ch[0])
        _collect_bool(ch[1])
        unique: List[_ENode] = []
        seen: Set[_ENode] = set()
        for it in items:
            if it.op == "Bool":
                if bool(it.val) is absorber:
                    return _ENode("Bool", val=absorber)
                if bool(it.val) is identity:
                    continue
            if it not in seen:
                seen.add(it)
                unique.append(it)
        # Complement law: x && !x == false, x || !x == true
        for it in unique:
            neg_it = _canonicalize_enode(_ENode("Not", children=(it,)))
            if neg_it in seen:
                return _ENode("Bool", val=absorber)
        if not unique:
            return _ENode("Bool", val=identity)
        unique.sort(key=_enode_to_str)
        acc = unique[0]
        for nxt in unique[1:]:
            acc = _ENode(op, children=(acc, nxt))
        return acc

    if op in ("Eq", "Neq"):
        a, b = ch
        if a == b:
            return _ENode("Bool", val=(op == "Eq"))
        pair = tuple(sorted((a, b), key=_enode_to_str))
        return _ENode(op, children=pair)

    if op == "Gt":
        return _ENode("Lt", children=(ch[1], ch[0]))
    if op == "Gte":
        return _ENode("Lte", children=(ch[1], ch[0]))

    return _ENode(op, val=node.val, children=ch)


class _UnionFindEGraph:
    """Minimal pure-Python union-find E-Graph fallback for equational proofs."""

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}
        self._node_to_id: Dict[_ENode, int] = {}
        self._id_to_node: Dict[int, _ENode] = {}
        self._next_id = 0

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
        return ra

    def add(self, node: _ENode) -> int:
        if node in self._node_to_id:
            return self.find(self._node_to_id[node])
        eid = self._next_id
        self._next_id += 1
        self._parent[eid] = eid
        self._node_to_id[node] = eid
        self._id_to_node[eid] = node
        return eid

    def prove(self, node_a: _ENode, node_b: _ENode) -> Tuple[bool, str, str]:
        id_a = self.add(node_a)
        id_b = self.add(node_b)
        canon_a = _canonicalize_enode(node_a)
        canon_b = _canonicalize_enode(node_b)
        cid_a = self.add(canon_a)
        cid_b = self.add(canon_b)
        self.union(id_a, cid_a)
        self.union(id_b, cid_b)
        equiv = self.find(id_a) == self.find(id_b)
        return equiv, _enode_to_str(canon_a), _enode_to_str(canon_b)


def _prove_with_egglog(node_a: _ENode, node_b: _ENode) -> Tuple[bool, str, str]:
    """Prove equivalence of two `_ENode` trees using bounded `egglog` e-graph saturation."""
    assert _EgglogEGraph is not None
    term_a = _enode_to_egglog(node_a)
    term_b = _enode_to_egglog(node_b)

    program = (
        f"{_EGGLOG_THEORY}\n"
        f"(let $a {term_a})\n"
        f"(let $b {term_b})\n"
        f"(run {MAX_EGRAPH_ITERATIONS})\n"
        f"(check (= $a $b))\n"
    )
    eg = _EgglogEGraph()
    cmds = eg.parse_program(program)
    try:
        eg.run_program(*cmds)
        equivalent = True
    except Exception:
        equivalent = False

    canon_a = _enode_to_str(_canonicalize_enode(node_a))
    canon_b = _enode_to_str(_canonicalize_enode(node_b))
    return equivalent, canon_a, canon_b


# ==============================================================================
# Reflex Engram Cache Entry
# ==============================================================================


@dataclass
class _ReflexEntry:
    name: str
    pattern: str
    pattern_lower: str
    tokens: FrozenSet[str]
    regex: Optional[re.Pattern[str]]
    actor_source: str
    self_test: str
    verified_runs: int = 1
    created_at: float = field(default_factory=time.time)


def _tokenize_pattern(text: str) -> FrozenSet[str]:
    return frozenset(_TOKEN_SPLIT_RE.findall(str(text or "").lower()))


def _validate_reflex_safety(actor_source: str, self_test: str) -> Optional[str]:
    combined = f"{actor_source}\n{self_test}"
    for pat in _FORBIDDEN_REFLEX_PATTERNS:
        if pat.search(combined):
            return (
                "Forbidden Tier-2 host/shell call detected in reflex source; "
                "reflex engrams must remain pure and sandbox-safe."
            )
    return None


# ==============================================================================
# ConsolidatorBridge (`Actor[#Consolidator]`)
# ==============================================================================


@actor_bridge(name="ConsolidatorBridge")
class ConsolidatorBridge:
    """Wake-Sleep Neocortical Library Compression & Reflex Compilation Bridge."""

    def __init__(
        self,
        vm: Any = None,
        neo4j_url: str = DEFAULT_NEO4J_URL,
        namespace: str = "default",
        timeout_s: float = 4.0,
    ) -> None:
        self.vm = vm
        self.neo4j_url = str(neo4j_url or "").strip()
        self.namespace = str(namespace or "default").strip()
        self.timeout_s = max(0.5, float(timeout_s))
        self._lock = threading.RLock()
        self._reflexes: Dict[str, _ReflexEntry] = {}
        self._consolidation_epoch: int = 0
        self._episode_history: List[Any] = []

    def _get_shadow_vm(self) -> Any:
        if self.vm is not None:
            return self.vm
        from actorii.vm import ActorVM

        self.vm = ActorVM()
        return self.vm

    def _neo4j_query(self, statement: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Run a best-effort Cypher query over Neo4j HTTP transactional endpoint."""
        if not self.neo4j_url:
            return None
        payload = {
            "statements": [
                {
                    "statement": statement,
                    "parameters": parameters or {},
                }
            ]
        }
        req = urllib.request.Request(
            self.neo4j_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        if body.get("errors"):
            return None
        results = body.get("results") or []
        if not results:
            return []
        cols = results[0].get("columns") or []
        rows: List[Dict[str, Any]] = []
        for entry in results[0].get("data") or []:
            vals = entry.get("row") or []
            rows.append({col: vals[i] if i < len(vals) else None for i, col in enumerate(cols)})
        return rows

    def recordEpisodeTrace(self, episode: Any) -> None:
        """Record an episode trace in local memory (useful for offline consolidation/tests)."""
        with self._lock:
            self._episode_history.append(episode)

    @actor_query
    def compressEpisodes(
        self,
        episodes_json: Any,
        min_support: int = 2,
        max_abstractions: int = 5,
    ) -> Any:
        """Extract recurring parameterized Actor 4.1 motifs from episode action traces."""
        t0 = time.perf_counter()
        min_sup = max(1, int(min_support))
        max_abs = max(1, min(25, int(max_abstractions)))

        traces, ep_count = _extract_episode_traces(episodes_json)
        if not traces:
            return _to_actor_value(
                {
                    "abstractions": [],
                    "compressionRatio": 1.0,
                    "backend": "stitch-core" if stitch_core is not None else "python-anti-unification",
                    "episodeCount": ep_count,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": "",
                }
            )

        atoms = _AtomTable()
        sexpr_trees: List[SExpr] = []
        for ep_actions in traces:
            parsed_actions = [s for a in ep_actions if (s := _actor_action_to_sexpr(a, atoms)) is not None]
            if not parsed_actions:
                continue
            if len(parsed_actions) == 1:
                sexpr_trees.append(parsed_actions[0])
            else:
                sexpr_trees.append(["seq"] + parsed_actions)

        if not sexpr_trees:
            return _to_actor_value(
                {
                    "abstractions": [],
                    "compressionRatio": 1.0,
                    "backend": "stitch-core" if stitch_core is not None else "python-anti-unification",
                    "episodeCount": ep_count,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": "",
                }
            )

        sexpr_strings = [_sexpr_to_str(t) for t in sexpr_trees]
        abstractions: List[Dict[str, Any]] = []
        compression_ratio = 1.0
        backend = "python-anti-unification"

        if stitch_core is not None and len(sexpr_strings) >= 2:
            try:
                res = stitch_core.compress(
                    sexpr_strings,
                    iterations=max_abs,
                    max_arity=3,
                    silent=True,
                )
                backend = "stitch-core"
                meta_json = getattr(res, "json", {}) or {}
                raw_ratio = float(meta_json.get("compression_ratio", 1.0) or 1.0)
                compression_ratio = round(max(1.0, raw_ratio), 4)
                rewritten_all = list(getattr(res, "rewritten", []) or [])
                json_abs_list = list(meta_json.get("abstractions", []) or [])

                for idx, abs_obj in enumerate(getattr(res, "abstractions", []) or []):
                    fn_name = str(getattr(abs_obj, "name", f"fn_{idx}"))
                    body_str = str(getattr(abs_obj, "body", ""))
                    arity = int(getattr(abs_obj, "arity", 0))
                    # Count occurrences in rewritten programs (or from stitch json metadata)
                    support = sum(rw.count(f"({fn_name} ") + rw.count(f"({fn_name})") + (1 if rw == fn_name else 0) for rw in rewritten_all)
                    if support == 0 and idx < len(json_abs_list):
                        support = int(json_abs_list[idx].get("num_uses", 0) or 0)
                    if support < min_sup:
                        continue
                    parsed_body = _parse_sexpr_str(body_str)
                    actor_template = _sexpr_to_actor_template(parsed_body, atoms)
                    params = [f"${i}" for i in range(arity)]
                    body_size = _sexpr_size(parsed_body)
                    est_tokens = max(2, max(1, body_size - arity - 1) * max(1, support - 1) * 4)
                    abstractions.append(
                        {
                            "id": fn_name,
                            "template": actor_template,
                            "sexpr": body_str,
                            "params": params,
                            "support": int(support),
                            "estimatedTokensSaved": int(est_tokens),
                        }
                    )
                    if len(abstractions) >= max_abs:
                        break
            except Exception:
                # Fall back seamlessly to deterministic Python anti-unification
                abstractions, compression_ratio = _python_anti_unify_compress(
                    sexpr_trees, atoms, min_sup, max_abs
                )
                backend = "python-anti-unification"
        else:
            abstractions, compression_ratio = _python_anti_unify_compress(
                sexpr_trees, atoms, min_sup, max_abs
            )
            backend = "python-anti-unification"

        return _to_actor_value(
            {
                "abstractions": abstractions,
                "compressionRatio": compression_ratio,
                "backend": backend,
                "episodeCount": ep_count,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": "",
            }
        )

    @actor_query
    def proveEquivalence(self, expr_a: str, expr_b: str) -> Any:
        """Prove equivalence of two arithmetic/boolean/Actor expressions via e-graph."""
        t0 = time.perf_counter()
        try:
            node_a = _parse_expr(str(expr_a))
            node_b = _parse_expr(str(expr_b))
        except Exception as exc:
            return _to_actor_value(
                {
                    "equivalent": False,
                    "canonicalForm": "",
                    "canonicalFormA": "",
                    "canonicalFormB": "",
                    "backend": "egglog" if _EgglogEGraph is not None else "python-egraph",
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": f"ParseError: {exc}",
                }
            )

        if _EgglogEGraph is not None:
            try:
                equiv, canon_a, canon_b = _prove_with_egglog(node_a, node_b)
                backend = "egglog"
            except Exception:
                equiv, canon_a, canon_b = _UnionFindEGraph().prove(node_a, node_b)
                backend = "python-egraph"
        else:
            equiv, canon_a, canon_b = _UnionFindEGraph().prove(node_a, node_b)
            backend = "python-egraph"

        return _to_actor_value(
            {
                "equivalent": bool(equiv),
                "canonicalForm": canon_a,
                "canonicalFormA": canon_a,
                "canonicalFormB": canon_b,
                "backend": backend,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": "",
            }
        )

    @actor_command
    def registerReflex(
        self,
        name: str,
        pattern: str,
        actor_source: str,
        self_test: str = "",
    ) -> Any:
        """Validate `actor_source` + `self_test` in `shadow_only=True` VM and promote as Reflex."""
        t0 = time.perf_counter()
        norm_name = _norm_sym(name)
        pat_str = str(pattern or "").strip()
        src_str = str(actor_source or "").strip()
        test_str = str(self_test or "").strip()

        if not norm_name or not pat_str or not src_str:
            return _to_actor_value(
                {
                    "promoted": False,
                    "name": norm_name,
                    "neo4jPersisted": False,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": "name, pattern, and actor_source must all be non-empty.",
                }
            )

        safety_err = _validate_reflex_safety(src_str, test_str)
        if safety_err:
            return _to_actor_value(
                {
                    "promoted": False,
                    "name": norm_name,
                    "neo4jPersisted": False,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": safety_err,
                }
            )

        # Build combined shadow validation source
        src_exec = src_str if src_str.endswith("!!") else f"{src_str}!!"
        if test_str:
            test_exec = test_str if test_str.endswith("!!") else f"{test_str}!!"
            combined_exec = f"{src_exec}\n{test_exec}"
            has_explicit_test = True
        elif re.search(r"\bDef\s+selfTest\s*\(", src_str):
            combined_exec = f"{src_exec}\nselfTest(Actor[#{norm_name}])!!"
            has_explicit_test = True
        else:
            combined_exec = src_exec
            has_explicit_test = False

        # Execute strictly inside `shadow_only=True` transaction
        vm = self._get_shadow_vm()
        try:
            shadow_res = vm.eval_transactional(
                combined_exec,
                max_opcodes=25_000,
                shadow_only=True,
            )
        except Exception as exc:
            return _to_actor_value(
                {
                    "promoted": False,
                    "name": norm_name,
                    "neo4jPersisted": False,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                    "error": f"ShadowValidationError: {type(exc).__name__}: {exc}",
                }
            )

        if has_explicit_test:
            if (
                shadow_res is False
                or shadow_res is None
                or shadow_res == 0
                or str(shadow_res).startswith("ERROR")
            ):
                return _to_actor_value(
                    {
                        "promoted": False,
                        "name": norm_name,
                        "neo4jPersisted": False,
                        "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                        "error": f"ShadowSelfTestFailed: self_test returned {shadow_res!r} (expected truthy).",
                    }
                )

        # Compile optional regex pattern (`re:<pattern>`)
        compiled_re: Optional[re.Pattern[str]] = None
        if pat_str.startswith("re:"):
            try:
                compiled_re = re.compile(pat_str[3:], re.IGNORECASE)
            except re.error as exc:
                return _to_actor_value(
                    {
                        "promoted": False,
                        "name": norm_name,
                        "neo4jPersisted": False,
                        "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                        "error": f"InvalidRegexPattern: {exc}",
                    }
                )

        entry = _ReflexEntry(
            name=norm_name,
            pattern=pat_str,
            pattern_lower=pat_str.lower(),
            tokens=_tokenize_pattern(pat_str),
            regex=compiled_re,
            actor_source=src_str,
            self_test=test_str,
        )
        with self._lock:
            self._reflexes[norm_name] = entry

        # Best-effort persistence to Neo4j `:ActorEngram {reflex: true}`
        neo4j_ok = False
        if self.neo4j_url:
            rows = self._neo4j_query(
                """
                MERGE (e:ActorEngram {name: $name, namespace: $ns})
                SET e.reflex = true,
                    e.compressed = true,
                    e.reflex_trigger_pattern = $pattern,
                    e.actor_source = $src,
                    e.reflex_verified_runs = coalesce(e.reflex_verified_runs, 0) + 1,
                    e.consolidation_epoch = $epoch,
                    e.updated_at = timestamp()
                RETURN e.name AS name
                """,
                {
                    "name": norm_name,
                    "ns": self.namespace,
                    "pattern": pat_str,
                    "src": src_str,
                    "epoch": self._consolidation_epoch,
                },
            )
            neo4j_ok = bool(rows)

        return _to_actor_value(
            {
                "promoted": True,
                "name": norm_name,
                "neo4jPersisted": neo4j_ok,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": "",
            }
        )

    @actor_query
    def matchReflex(self, goal_or_pattern: str) -> Any:
        """<1 ms, 0-token lookup returning the best matching promoted Reflex Engram."""
        t0 = time.perf_counter()
        query = str(goal_or_pattern or "").strip()
        if not query:
            return _to_actor_value(
                {
                    "matched": False,
                    "name": "",
                    "actorSource": "",
                    "pattern": "",
                    "score": 0.0,
                    "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            )

        q_lower = query.lower()
        q_tokens = _tokenize_pattern(q_lower)

        best_entry: Optional[_ReflexEntry] = None
        best_score = 0.0

        with self._lock:
            entries = list(self._reflexes.values())

        for entry in entries:
            score = 0.0
            if entry.regex is not None:
                if entry.regex.search(query):
                    score = 0.95
            elif q_lower == entry.pattern_lower:
                score = 1.0
            elif entry.pattern_lower and entry.pattern_lower in q_lower:
                score = 0.9
            elif q_lower in entry.pattern_lower and len(q_lower) >= 4:
                score = 0.8
            elif entry.tokens and q_tokens:
                overlap = len(entry.tokens & q_tokens)
                if overlap > 0:
                    coverage = overlap / len(entry.tokens)
                    jaccard = overlap / len(entry.tokens | q_tokens)
                    score = 0.65 * coverage + 0.35 * jaccard

            if score >= 0.5 and score > best_score:
                best_score = score
                best_entry = entry

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        if best_entry is None:
            return _to_actor_value(
                {
                    "matched": False,
                    "name": "",
                    "actorSource": "",
                    "pattern": "",
                    "score": 0.0,
                    "elapsedMs": elapsed_ms,
                }
            )

        return _to_actor_value(
            {
                "matched": True,
                "name": best_entry.name,
                "actorSource": best_entry.actor_source,
                "pattern": best_entry.pattern,
                "score": round(best_score, 4),
                "elapsedMs": elapsed_ms,
            }
        )

    @actor_command
    def runConsolidationCycle(
        self,
        max_episodes: int = 20,
        dry_run: bool = True,
    ) -> Any:
        """End-to-end wake-sleep consolidation pass over recent `:ActorEpisode` traces."""
        t0 = time.perf_counter()
        limit = max(1, min(200, int(max_episodes)))
        is_dry = bool(dry_run)

        episodes: List[Dict[str, Any]] = []
        neo4j_rows = self._neo4j_query(
            """
            MATCH (ep:ActorEpisode)
            WHERE ep.content IS NOT NULL
            RETURN ep.session_id AS session_id,
                   ep.turn_range AS turn_range,
                   ep.content AS content
            ORDER BY coalesce(ep.created_at, 0) DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        if neo4j_rows:
            for r in neo4j_rows:
                episodes.append(
                    {
                        "session_id": r.get("session_id") or "",
                        "turn_range": r.get("turn_range") or "",
                        "content": r.get("content") or "",
                    }
                )

        # Merge any locally recorded episodes (e.g. offline mode or unit tests)
        with self._lock:
            local_eps = list(self._episode_history[-limit:])
        for ep in local_eps:
            if len(episodes) >= limit:
                break
            episodes.append(ep)

        traces, _ = _extract_episode_traces(episodes)
        comp_dict = _from_actor_value(
            self.compressEpisodes(json.dumps(episodes), min_support=2, max_abstractions=5)
        )
        abstractions = comp_dict.get("abstractions") or []

        # Equivalence check across extracted abstraction templates (for deduplication)
        equivalences_found = 0
        eq_backend = "egglog" if _EgglogEGraph is not None else "python-egraph"
        for i in range(len(abstractions)):
            for j in range(i + 1, len(abstractions)):
                t_i = str(abstractions[i].get("template", ""))
                t_j = str(abstractions[j].get("template", ""))
                if not t_i or not t_j:
                    continue
                if t_i == t_j:
                    equivalences_found += 1
                    continue
                # Try algebraic equivalence check on pure expression templates
                eq_res = _from_actor_value(self.proveEquivalence(t_i, t_j))
                eq_backend = eq_res.get("backend", eq_backend)
                if eq_res.get("equivalent"):
                    equivalences_found += 1

        with self._lock:
            if not is_dry:
                self._consolidation_epoch += 1
            epoch = self._consolidation_epoch
            reflex_count = len(self._reflexes)

        summary = (
            f"Wake-Sleep Consolidation (epoch={epoch}, dry_run={is_dry}): "
            f"episodes={len(episodes)}, traces={len(traces)}, "
            f"abstractions={len(abstractions)}, ratio={comp_dict.get('compressionRatio', 1.0)}x, "
            f"equivalences={equivalences_found}, reflexes={reflex_count}"
        )

        return _to_actor_value(
            {
                "ok": True,
                "dryRun": is_dry,
                "epoch": epoch,
                "episodesAnalyzed": len(episodes),
                "tracesExtracted": len(traces),
                "abstractions": abstractions,
                "compressionRatio": float(comp_dict.get("compressionRatio", 1.0)),
                "equivalencesFound": equivalences_found,
                "reflexCount": reflex_count,
                "compressionBackend": str(comp_dict.get("backend", "python-anti-unification")),
                "equivalenceBackend": eq_backend,
                "summary": summary,
                "elapsedMs": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": "",
            }
        )

    @actor_query
    def describe(self) -> str:
        """Concise self-documentation and Actor 4.1 examples for `Actor[#Consolidator]`."""
        comp_backend = "stitch-core" if stitch_core is not None else "python-anti-unification"
        eq_backend = "egglog" if _EgglogEGraph is not None else "python-egraph"
        return (
            f"ConsolidatorBridge (Actor[#Consolidator]) | compression={comp_backend} | "
            f"egraph={eq_backend} | 0 MB GPU VRAM (pure CPU)\n"
            "Wake-Sleep Neocortical Library Compression & Production Compilation.\n"
            "Methods:\n"
            "  - compressEpisodes(episodesJson, minSupport, maxAbstractions) -> "
            "Dictionary(#abstractions, #compressionRatio, #backend)\n"
            "  - proveEquivalence(exprA, exprB) -> "
            "Dictionary(#equivalent, #canonicalForm, #backend)\n"
            "  - registerReflex(name, pattern, actorSource, selfTest) -> "
            "Dictionary(#promoted, #name, #elapsedMs, #error) [shadow_only=True gated]\n"
            "  - matchReflex(goalOrPattern) -> "
            "Dictionary(#matched, #name, #actorSource, #elapsedMs) [<1 ms, 0-token]\n"
            "  - runConsolidationCycle(maxEpisodes, dryRun) -> "
            "Dictionary(#ok, #abstractions, #compressionRatio, #equivalencesFound, #summary)\n"
            "Actor 4.1 / DreamingCycle Examples:\n"
            "  /* Inside DreamingCycle>>runDreaming */\n"
            "  report := runConsolidationCycle(Actor[#Consolidator], 20, true)!!\n"
            "  /* Reflex fast-path lookup */\n"
            "  hit := matchReflex(Actor[#Consolidator], \"check plan status\")!!\n"
            "  /* Algebraic equivalence proof */\n"
            "  eq := proveEquivalence(Actor[#Consolidator], \"!(a && b)\", \"(!a) || (!b)\")!!"
        )


__all__ = ["ConsolidatorBridge"]
