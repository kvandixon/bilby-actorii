"""MAC/FAC Structural Analogy Engine (`Actor[#Analogy]`).

Pure-CPU (0 MB GPU VRAM) implementation of Gentner's Structure-Mapping Theory
(Gentner 1983; Falkenhainer, Forbus, & Gentner 1989, SME) and the two-stage
MAC/FAC retrieval architecture (Forbus, Gentner, & Law 1995) over directed
labeled causal/dependency graphs using `networkx`.

Stages:
1. **MAC (Many Are Called):** Fast $O(|V| + |E|)$ structural signature extraction
   (normalized relation-type histogram, sorted in/out-degree profiles, max causal
   chain depth) to filter candidate graphs without touching surface node names.
2. **FAC (Few Are Chosen — SME):** Relational subgraph alignment over directed
   labeled graphs (`nodes: [{id, type?}]`, `edges: [{src, dst, rel}]`) enforcing
   1-to-1 node mapping consistency, matching edge relations while ignoring
   arbitrary surface node names, scoring **systematicity** (connected multi-hop
   causal chains outrank isolated single-edge matches), and projecting unmapped
   base relations into target **candidate inferences**.
3. **Persistence:** Stores named `:CausalEntity` graphs (`:CAUSES`,
   `:DEPENDS_ON`, `:CONSTRAINS`, etc.) in Neo4j scoped to the active namespace,
   with an automatic in-memory fallback registry for offline/unit-test use.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import json
import math
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import isomorphism as nx_iso

from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.collections import ActorArray
from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    os.environ.get("ACTORII_NEO4J_TX_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"),
)

_REL_TYPE_SAFE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,47}$")
_INDEXED_ANAL_NEO4J_URLS: Set[str] = set()

# Type alias for canonical directed labeled edge: (src, dst, rel)
EdgeTuple = Tuple[str, str, str]


def _norm_sym(val: Any) -> str:
    if isinstance(val, Symbol):
        return val.name
    s = str(val or "").strip()
    return s[1:] if s.startswith("#") else s


def _sanitize_rel_type(raw_rel: Any) -> str:
    """Normalize a relation label into a safe uppercase Cypher/SME identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw_rel or "RELATES_TO").strip()).strip("_").upper()
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"REL_{cleaned}" if cleaned else "RELATES_TO"
    cleaned = cleaned[:48]
    if not _REL_TYPE_SAFE_RE.match(cleaned):
        return "RELATES_TO"
    return cleaned


def _py_to_actor_value(val: Any) -> Any:
    """Recursively convert Python dicts/lists/scalars into native Actor objects."""
    if isinstance(val, dict):
        d = Dictionary(max(8, len(val) * 2))
        for k, v in val.items():
            conv_v = _py_to_actor_value(v)
            k_str = str(k)
            d.atPut(Symbol(k_str), conv_v)
            d.atPut(k_str, conv_v)
        return d
    if isinstance(val, (list, tuple)):
        arr = ActorArray(len(val))
        for idx, item in enumerate(val):
            arr.atPut(idx, _py_to_actor_value(item))
        return arr
    return val


def _actor_to_py_value(val: Any) -> Any:
    """Convert Actor `Dictionary`, `ActorArray`, or `Symbol` objects into Python values."""
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


def _parse_json_or_actor(raw: Any, field_name: str) -> Any:
    """Accept either a JSON string or an already-parsed Python/Actor collection."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for {field_name}: {exc}") from exc
    return _actor_to_py_value(raw)


def _normalize_graph_spec(
    graph_input: Any,
    *,
    nodes_override: Any = None,
    edges_override: Any = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Normalize graph input into canonical `(nodes, edges)` lists of dicts.

    Accepts:
    - `{"nodes": [...], "edges": [...]}` (JSON string or dict/Dictionary)
    - Explicit `nodes_override` and `edges_override` lists (JSON strings or lists)
    - String node IDs or `{id, type?}` dicts
    - Edge keys `src`/`dst`/`rel` (plus `source`/`target`/`relation`/`type` aliases)
    """
    if nodes_override is not None or edges_override is not None:
        raw_nodes = _parse_json_or_actor(nodes_override, "nodes_json") if nodes_override is not None else []
        raw_edges = _parse_json_or_actor(edges_override, "edges_json") if edges_override is not None else []
    else:
        parsed = _parse_json_or_actor(graph_input, "graph_json")
        if not isinstance(parsed, dict):
            raise ValueError("graph_json must decode to a JSON object with 'nodes' and 'edges'.")
        raw_nodes = parsed.get("nodes", [])
        raw_edges = parsed.get("edges", [])

    if not isinstance(raw_nodes, list):
        raise ValueError("Graph 'nodes' must be a list.")
    if not isinstance(raw_edges, list):
        raise ValueError("Graph 'edges' must be a list.")

    node_types: Dict[str, str] = {}
    node_order: List[str] = []

    for item in raw_nodes:
        if isinstance(item, str):
            nid = item.strip()
            ntype = ""
        elif isinstance(item, dict):
            nid = str(item.get("id") or item.get("name") or "").strip()
            ntype = str(item.get("type") or item.get("role") or "").strip()
        else:
            continue
        if not nid:
            continue
        if nid not in node_types:
            node_order.append(nid)
            node_types[nid] = ntype
        elif ntype and not node_types[nid]:
            node_types[nid] = ntype

    edges: List[Dict[str, str]] = []
    seen_edges: Set[EdgeTuple] = set()
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or item.get("source") or item.get("from") or "").strip()
        dst = str(item.get("dst") or item.get("target") or item.get("to") or "").strip()
        rel_raw = item.get("rel") or item.get("relation") or item.get("type") or item.get("label") or ""
        if not src or not dst:
            continue
        rel = _sanitize_rel_type(rel_raw)
        for endpoint in (src, dst):
            if endpoint not in node_types:
                node_order.append(endpoint)
                node_types[endpoint] = ""
        key = (src, dst, rel)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"src": src, "dst": dst, "rel": rel})

    nodes = [{"id": nid, "type": node_types[nid]} for nid in node_order]
    return nodes, edges


def _build_nx_graph(
    nodes: List[Dict[str, str]], edges: List[Dict[str, str]]
) -> nx.MultiDiGraph:
    """Construct a `networkx.MultiDiGraph` from normalized nodes and edges."""
    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["id"], type=n.get("type", ""))
    for e in edges:
        g.add_edge(e["src"], e["dst"], rel=e["rel"])
    return g


def _max_causal_chain_depth(g: nx.MultiDiGraph, max_dfs_steps: int = 4000) -> int:
    """Compute the longest simple directed path length (in edges) in `g`.

    Uses exact linear-time `nx.dag_longest_path_length` on DAGs and a bounded
    DFS fallback on cyclic graphs so execution is always fast and finite.
    """
    if g.number_of_edges() == 0:
        return 0
    simple_dg = nx.DiGraph()
    simple_dg.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if u != v:
            simple_dg.add_edge(u, v)
    if simple_dg.number_of_edges() == 0:
        return 1  # self-loop only

    if nx.is_directed_acyclic_graph(simple_dg):
        return int(nx.dag_longest_path_length(simple_dg))

    # Cyclic graph: bounded longest simple path search
    best_depth = 0
    steps = 0

    # Prioritize nodes with smaller in-degree (roots / entry points)
    start_nodes = sorted(simple_dg.nodes(), key=lambda n: (simple_dg.in_degree(n), -simple_dg.out_degree(n), str(n)))
    for start in start_nodes:
        if steps >= max_dfs_steps:
            break
        stack: List[Tuple[str, Set[str], int]] = [(start, {start}, 0)]
        while stack and steps < max_dfs_steps:
            node, visited, depth = stack.pop()
            steps += 1
            if depth > best_depth:
                best_depth = depth
            for nxt in simple_dg.successors(node):
                if nxt not in visited:
                    stack.append((nxt, visited | {nxt}, depth + 1))
    return best_depth


def _extract_signature_dict(
    nodes: List[Dict[str, str]], edges: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Compute the MAC structural signature dictionary for a normalized graph."""
    g = _build_nx_graph(nodes, edges)
    rel_hist = dict(sorted(Counter(e["rel"] for e in edges).items()))
    in_profile = sorted((int(g.in_degree(n)) for n in g.nodes()), reverse=True)
    out_profile = sorted((int(g.out_degree(n)) for n in g.nodes()), reverse=True)
    max_depth = _max_causal_chain_depth(g)
    wcc = list(nx.weakly_connected_components(g)) if g.number_of_nodes() > 0 else []
    max_comp_edges = 0
    for comp in wcc:
        sub = g.subgraph(comp)
        if sub.number_of_edges() > max_comp_edges:
            max_comp_edges = sub.number_of_edges()

    simple_dg = nx.DiGraph()
    simple_dg.add_nodes_from(g.nodes())
    simple_dg.add_edges_from((u, v) for u, v in g.edges())
    has_cycles = not nx.is_directed_acyclic_graph(simple_dg) if simple_dg.number_of_nodes() > 0 else False

    return {
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "relationHistogram": rel_hist,
        "inDegreeProfile": in_profile,
        "outDegreeProfile": out_profile,
        "maxCausalChainDepth": max_depth,
        "connectedComponents": len(wcc),
        "maxComponentEdges": max_comp_edges,
        "hasCycles": has_cycles,
    }


def _cosine_counter(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    dot = sum(float(a[k]) * float(b[k]) for k in keys)
    norm_a = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    norm_b = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _profile_similarity(p1: List[int], p2: List[int]) -> float:
    if not p1 and not p2:
        return 1.0
    if not p1 or not p2:
        return 0.0
    n = max(len(p1), len(p2))
    a = p1 + [0] * (n - len(p1))
    b = p2 + [0] * (n - len(p2))
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _mac_similarity(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    """Fast MAC structural similarity in `[0.0, 1.0]` combining relations, degree profile, and chain depth."""
    rel_sim = _cosine_counter(
        sig_a.get("relationHistogram") or {},
        sig_b.get("relationHistogram") or {},
    )
    if rel_sim <= 0.0:
        return 0.0
    in_sim = _profile_similarity(
        sig_a.get("inDegreeProfile") or [],
        sig_b.get("inDegreeProfile") or [],
    )
    out_sim = _profile_similarity(
        sig_a.get("outDegreeProfile") or [],
        sig_b.get("outDegreeProfile") or [],
    )
    deg_sim = 0.5 * (in_sim + out_sim)
    d_a = int(sig_a.get("maxCausalChainDepth") or 0)
    d_b = int(sig_b.get("maxCausalChainDepth") or 0)
    depth_sim = 1.0 - abs(d_a - d_b) / float(max(1, d_a, d_b))
    score = 0.55 * rel_sim + 0.25 * deg_sim + 0.20 * depth_sim
    return round(max(0.0, min(1.0, score)), 6)


def _systematicity_weight(edge_tuples: List[EdgeTuple]) -> float:
    """Compute Gentner's systematicity weight $W(E)$ for a set of directed labeled edges.

    Rewards:
    - Each matched relational edge (`+1.0`)
    - Multi-hop causal/dependency continuations `u -> v -> w` (`+0.6 * in_deg(v) * out_deg(v)`)
    - Shared-entity fan-in / fan-out coherence (`+0.25 * C(in_deg, 2) + 0.25 * C(out_deg, 2)`)
    - Maximum directed causal chain depth beyond 1 (`+0.5 * max(0, depth - 1)`)

    Because $W(M)$ is super-additive for connected chains, a connected 3-hop
    chain strictly outranks 3 disconnected single edges.
    """
    if not edge_tuples:
        return 0.0
    in_deg: Counter[str] = Counter()
    out_deg: Counter[str] = Counter()
    g = nx.MultiDiGraph()
    for u, v, r in edge_tuples:
        in_deg[v] += 1
        out_deg[u] += 1
        g.add_edge(u, v, rel=r)

    weight = float(len(edge_tuples))
    all_nodes = set(in_deg.keys()) | set(out_deg.keys())
    for n in all_nodes:
        ind = in_deg[n]
        outd = out_deg[n]
        # Causal transit / multi-hop chain continuation
        weight += 0.6 * float(ind * outd)
        # Coherent branching / convergence around a shared entity
        if ind > 1:
            weight += 0.25 * float(ind * (ind - 1)) / 2.0
        if outd > 1:
            weight += 0.25 * float(outd * (outd - 1)) / 2.0

    depth = _max_causal_chain_depth(g)
    if depth > 1:
        weight += 0.5 * float(depth - 1)
    return weight


def _edge_rel_match(attrs_a: Dict[Any, Dict[str, Any]], attrs_b: Dict[Any, Dict[str, Any]]) -> bool:
    """NetworkX MultiDiGraph edge matcher comparing relation multisets."""
    rels_a = Counter(d.get("rel", "") for d in attrs_a.values())
    rels_b = Counter(d.get("rel", "") for d in attrs_b.values())
    return rels_a == rels_b


def _edge_rel_monomorphic_match(
    host_attrs: Dict[Any, Dict[str, Any]], sub_attrs: Dict[Any, Dict[str, Any]]
) -> bool:
    """NetworkX MultiDiGraph edge matcher for monomorphism (host edges >= sub edges per relation)."""
    host_rels = Counter(d.get("rel", "") for d in host_attrs.values())
    sub_rels = Counter(d.get("rel", "") for d in sub_attrs.values())
    return all(host_rels[r] >= cnt for r, cnt in sub_rels.items())


def _sme_align(
    base_nodes: List[Dict[str, str]],
    base_edges: List[Dict[str, str]],
    target_nodes: List[Dict[str, str]],
    target_edges: List[Dict[str, str]],
    *,
    max_search_steps: int = 8000,
) -> Dict[str, Any]:
    """Run Gentner's Structure-Mapping Engine (FAC) over two directed labeled graphs."""
    b_edge_tuples: List[EdgeTuple] = [(e["src"], e["dst"], e["rel"]) for e in base_edges]
    t_edge_tuples: List[EdgeTuple] = [(e["src"], e["dst"], e["rel"]) for e in target_edges]
    b_type_map = {n["id"]: n.get("type", "").lower() for n in base_nodes}
    t_type_map = {n["id"]: n.get("type", "").lower() for n in target_nodes}

    w_base = _systematicity_weight(b_edge_tuples)
    w_target = _systematicity_weight(t_edge_tuples)

    if not b_edge_tuples or not t_edge_tuples:
        return {
            "score": 0.0,
            "systematicity": 0.0,
            "edgeCoverage": 0.0,
            "mapping": {},
            "matchedRelations": [],
            "candidateInferences": [],
        }

    b_rel_counts = Counter(r for _, _, r in b_edge_tuples)
    t_rel_counts = Counter(r for _, _, r in t_edge_tuples)
    if not (set(b_rel_counts.keys()) & set(t_rel_counts.keys())):
        return {
            "score": 0.0,
            "systematicity": 0.0,
            "edgeCoverage": 0.0,
            "mapping": {},
            "matchedRelations": [],
            "candidateInferences": [],
        }

    g_base = _build_nx_graph(base_nodes, base_edges)
    g_target = _build_nx_graph(target_nodes, target_edges)

    best_mapping: Dict[str, str] = {}
    best_matched: List[Tuple[EdgeTuple, EdgeTuple]] = []
    best_sys: float = 0.0

    def _type_tiebreak(mapping: Dict[str, str]) -> int:
        return sum(
            1
            for b_id, t_id in mapping.items()
            if b_type_map.get(b_id) and b_type_map.get(b_id) == t_type_map.get(t_id)
        )

    def _evaluate_mapping(b2t: Dict[str, str]) -> Tuple[float, List[Tuple[EdgeTuple, EdgeTuple]]]:
        t_edge_set = set(t_edge_tuples)
        matched_pairs: List[Tuple[EdgeTuple, EdgeTuple]] = []
        for bu, bv, br in b_edge_tuples:
            tu = b2t.get(bu)
            tv = b2t.get(bv)
            if tu is not None and tv is not None and (tu, tv, br) in t_edge_set:
                matched_pairs.append(((bu, bv, br), (tu, tv, br)))
        sys_w = _systematicity_weight([p[0] for p in matched_pairs])
        return sys_w, matched_pairs

    def _consider_mapping(b2t: Dict[str, str]) -> None:
        nonlocal best_mapping, best_matched, best_sys
        sys_w, matched_pairs = _evaluate_mapping(b2t)
        if not matched_pairs:
            return
        # Prune mapping to only nodes participating in matched edges
        active_b_nodes: Set[str] = set()
        for (bu, bv, _), _ in matched_pairs:
            active_b_nodes.add(bu)
            active_b_nodes.add(bv)
        pruned_map = {k: b2t[k] for k in sorted(active_b_nodes)}
        cand_key = (sys_w, len(matched_pairs), _type_tiebreak(pruned_map))
        best_key = (best_sys, len(best_matched), _type_tiebreak(best_mapping))
        if cand_key > best_key:
            best_sys = sys_w
            best_matched = matched_pairs
            best_mapping = pruned_map

    # Fast Path 1: Exact VF2 isomorphism (all edges & relations match)
    found_exact = False
    if (
        len(b_edge_tuples) == len(t_edge_tuples)
        and b_rel_counts == t_rel_counts
        and g_base.number_of_nodes() == g_target.number_of_nodes()
    ):
        matcher = nx_iso.MultiDiGraphMatcher(g_base, g_target, edge_match=_edge_rel_match)
        checked = 0
        for iso_map in matcher.isomorphisms_iter():
            _consider_mapping(iso_map)
            checked += 1
            if checked >= 16:
                break
        if len(best_matched) == len(b_edge_tuples):
            found_exact = True

    # Fast Path 2: Full subgraph monomorphism (base embeds completely in target)
    if (
        not found_exact
        and len(b_edge_tuples) <= len(t_edge_tuples)
        and all(t_rel_counts[r] >= cnt for r, cnt in b_rel_counts.items())
    ):
        # Restrict base to edge-incident nodes so isolated unused nodes don't block monomorphism
        b_active = nx.MultiDiGraph()
        for bu, bv, br in b_edge_tuples:
            b_active.add_edge(bu, bv, rel=br)
        matcher = nx_iso.MultiDiGraphMatcher(
            g_target, b_active, edge_match=_edge_rel_monomorphic_match
        )
        checked = 0
        for t2b in matcher.subgraph_monomorphisms_iter():
            b2t = {b_n: t_n for t_n, b_n in t2b.items()}
            _consider_mapping(b2t)
            checked += 1
            if checked >= 16:
                break
        if len(best_matched) == len(b_edge_tuples):
            found_exact = True

    # Fast Path 3: Full subgraph monomorphism (target embeds completely in base -> candidate inferences!)
    if (
        not found_exact
        and len(t_edge_tuples) <= len(b_edge_tuples)
        and all(b_rel_counts[r] >= cnt for r, cnt in t_rel_counts.items())
    ):
        t_active = nx.MultiDiGraph()
        for tu, tv, tr in t_edge_tuples:
            t_active.add_edge(tu, tv, rel=tr)
        matcher = nx_iso.MultiDiGraphMatcher(
            g_base, t_active, edge_match=_edge_rel_monomorphic_match
        )
        checked = 0
        for b2t in matcher.subgraph_monomorphisms_iter():
            _consider_mapping(b2t)
            checked += 1
            if checked >= 16:
                break
        if len(best_matched) == len(t_edge_tuples):
            found_exact = True

    # General Path: Bounded SME branch-and-bound over local relation matches
    if not found_exact:
        target_by_rel: Dict[str, List[EdgeTuple]] = defaultdict(list)
        for te in t_edge_tuples:
            target_by_rel[te[2]].append(te)

        # Order base edges so higher-degree / connected chain edges are explored first
        b_deg = Counter()
        for bu, bv, _ in b_edge_tuples:
            b_deg[bu] += 1
            b_deg[bv] += 1
        matchable_base_edges = [e for e in b_edge_tuples if e[2] in target_by_rel]
        matchable_base_edges.sort(
            key=lambda e: (-(b_deg[e[0]] + b_deg[e[1]]), len(target_by_rel[e[2]]), e)
        )

        # Precompute suffix remaining matchable edge counts for fast pruning
        n_edges = len(matchable_base_edges)
        steps = 0
        b2t: Dict[str, str] = {}
        t2b: Dict[str, str] = {}
        used_t_edges: Set[EdgeTuple] = set()
        current_pairs: List[Tuple[EdgeTuple, EdgeTuple]] = []

        def _dfs(idx: int) -> None:
            nonlocal steps, best_mapping, best_matched, best_sys
            steps += 1
            if steps > max_search_steps:
                return
            if idx == n_edges:
                if current_pairs:
                    sys_w = _systematicity_weight([p[0] for p in current_pairs])
                    pruned = {
                        k: b2t[k]
                        for k in sorted(
                            {u for (u, _, _), _ in current_pairs}
                            | {v for (_, v, _), _ in current_pairs}
                        )
                    }
                    cand_key = (sys_w, len(current_pairs), _type_tiebreak(pruned))
                    best_key = (best_sys, len(best_matched), _type_tiebreak(best_mapping))
                    if cand_key > best_key:
                        best_sys = sys_w
                        best_matched = list(current_pairs)
                        best_mapping = pruned
                return

            # Optimistic edge-count pruning when we already have a strong connected match
            remaining = n_edges - idx
            if len(current_pairs) + remaining == 0:
                return

            be = matchable_base_edges[idx]
            bu, bv, br = be
            candidates = target_by_rel[br]

            # Prefer target edges that connect to already-mapped nodes (systematicity-first)
            ordered_cands: List[Tuple[int, EdgeTuple]] = []
            for te in candidates:
                if te in used_t_edges:
                    continue
                tu, tv, _ = te
                # Check 1-to-1 consistency
                if bu in b2t and b2t[bu] != tu:
                    continue
                if bu not in b2t and tu in t2b:
                    continue
                if bv in b2t and b2t[bv] != tv:
                    continue
                if bv not in b2t and tv in t2b:
                    continue
                if (bu == bv) != (tu == tv):
                    continue
                overlap = (1 if bu in b2t else 0) + (1 if bv in b2t else 0)
                ordered_cands.append((-overlap, te))
            ordered_cands.sort(key=lambda x: (x[0], x[1]))

            for _, te in ordered_cands:
                if steps > max_search_steps:
                    break
                tu, tv, _ = te
                added_u = bu not in b2t
                added_v = bv not in b2t and bv != bu
                if added_u:
                    b2t[bu] = tu
                    t2b[tu] = bu
                if added_v:
                    b2t[bv] = tv
                    t2b[tv] = bv
                used_t_edges.add(te)
                current_pairs.append((be, te))

                _dfs(idx + 1)

                current_pairs.pop()
                used_t_edges.remove(te)
                if added_v:
                    del b2t[bv]
                    del t2b[tv]
                if added_u:
                    del b2t[bu]
                    del t2b[tu]

            # Also allow skipping this base edge (partial subgraph alignment)
            if steps <= max_search_steps:
                _dfs(idx + 1)

        _dfs(0)

    # Compute normalized score in [0.0, 1.0]
    denom = w_base + w_target
    score = (2.0 * best_sys / denom) if denom > 1e-9 else 0.0
    score = round(max(0.0, min(1.0, score)), 6)
    edge_cov = round(len(best_matched) / float(max(1, len(b_edge_tuples))), 6)

    matched_relations = [
        {
            "baseSrc": be[0],
            "baseDst": be[1],
            "targetSrc": te[0],
            "targetDst": te[1],
            "rel": be[2],
        }
        for be, te in best_matched
    ]

    # Candidate Inference Projection:
    # Unmapped base relations whose endpoints map into target -> concrete hypotheses!
    matched_base_edge_set = {be for be, _ in best_matched}
    target_edge_set = set(t_edge_tuples)
    candidate_inferences: List[Dict[str, Any]] = []
    for bu, bv, br in b_edge_tuples:
        if (bu, bv, br) in matched_base_edge_set:
            continue
        tu = best_mapping.get(bu)
        tv = best_mapping.get(bv)
        if tu is not None and tv is not None:
            if (tu, tv, br) in target_edge_set:
                continue
            candidate_inferences.append(
                {
                    "src": tu,
                    "dst": tv,
                    "rel": br,
                    "baseSrc": bu,
                    "baseDst": bv,
                    "kind": "missing_relation",
                    "confidence": score,
                    "hypothesis": f"{tu} -[{br}]-> {tv}",
                }
            )
        elif tu is not None and tv is None:
            proj_dst = f"?{bv}"
            candidate_inferences.append(
                {
                    "src": tu,
                    "dst": proj_dst,
                    "rel": br,
                    "baseSrc": bu,
                    "baseDst": bv,
                    "kind": "projected_consequence",
                    "confidence": round(score * 0.85, 6),
                    "hypothesis": f"{tu} -[{br}]-> {proj_dst}",
                }
            )
        elif tu is None and tv is not None:
            proj_src = f"?{bu}"
            candidate_inferences.append(
                {
                    "src": proj_src,
                    "dst": tv,
                    "rel": br,
                    "baseSrc": bu,
                    "baseDst": bv,
                    "kind": "projected_precondition",
                    "confidence": round(score * 0.85, 6),
                    "hypothesis": f"{proj_src} -[{br}]-> {tv}",
                }
            )

    return {
        "score": score,
        "systematicity": round(best_sys, 6),
        "edgeCoverage": edge_cov,
        "mapping": best_mapping,
        "matchedRelations": matched_relations,
        "candidateInferences": candidate_inferences,
    }


@actor_bridge(name="AnalogyEngineBridge")
class AnalogyEngineBridge:
    """MAC/FAC Structural Analogy Engine bridge (`Actor[#Analogy]`, 0 MB VRAM)."""

    def __init__(
        self,
        tx_url: Optional[str] = None,
        namespace: Optional[str] = None,
        enable_neo4j: bool = True,
    ) -> None:
        self.tx_url = tx_url or os.environ.get("ACTORII_NEO4J_HTTP_URL", NEO4J_HTTP_TX_URL)
        default_ns = os.environ.get("ACTORII_MEMORY_NAMESPACE", "default").strip() or "default"
        self.namespace = str(namespace or default_ns)
        self.enable_neo4j = bool(enable_neo4j)
        # In-memory fallback registry keyed by (namespace, graph_name)
        self._local_registry: Dict[Tuple[str, str], Dict[str, Any]] = {}

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
        timeout: float = 3.0,
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
            msg = "; ".join(f"{e.get('code', 'CypherError')}: {e.get('message', '')}" for e in err_items)
            raise RuntimeError(f"Neo4j Cypher error: {msg}")
        out: List[Tuple[List[str], List[Dict[str, Any]]]] = []
        for res in body.get("results", []):
            cols = res.get("columns", [])
            rows = [dict(zip(cols, item.get("row", []))) for item in res.get("data", [])]
            out.append((cols, rows))
        return out

    def _ensure_causal_indexes(self) -> None:
        if not self.enable_neo4j or self.tx_url in _INDEXED_ANAL_NEO4J_URLS:
            return
        _INDEXED_ANAL_NEO4J_URLS.add(self.tx_url)
        try:
            self._run_cypher_batch(
                [
                    {
                        "statement": (
                            "CREATE INDEX causal_entity_ns_graph IF NOT EXISTS "
                            "FOR (c:CausalEntity) ON (c.namespace, c.graph_name)"
                        ),
                        "parameters": {},
                    },
                    {
                        "statement": (
                            "CREATE INDEX causal_entity_ns_role IF NOT EXISTS "
                            "FOR (c:CausalEntity) ON (c.namespace, c.role)"
                        ),
                        "parameters": {},
                    },
                ],
                timeout=2.0,
            )
        except Exception:
            pass

    @actor_query
    def getNamespace(self) -> str:
        """Return the active namespace for causal graph storage and retrieval."""
        return self.namespace

    @actor_command
    def setNamespace(self, namespace: str) -> str:
        """Switch the active causal graph namespace."""
        self.namespace = _norm_sym(namespace).strip() or "default"
        return f"NAMESPACE_SET:{self.namespace}"

    @actor_query
    def extractStructure(self, graph_json: str) -> Dictionary:
        """MAC stage: extract compact structural signature & topological stats from `graph_json`."""
        nodes, edges = _normalize_graph_spec(graph_json)
        sig = _extract_signature_dict(nodes, edges)
        return _py_to_actor_value(sig)

    @actor_query
    def alignGraphs(self, base_json: str, target_json: str) -> Dictionary:
        """FAC stage: run Gentner SME relational subgraph alignment & candidate inference projection."""
        base_nodes, base_edges = _normalize_graph_spec(base_json)
        target_nodes, target_edges = _normalize_graph_spec(target_json)
        result = _sme_align(base_nodes, base_edges, target_nodes, target_edges)
        return _py_to_actor_value(result)

    @actor_command
    def registerCausalGraph(
        self,
        name: str,
        nodes_json: str,
        edges_json: str,
        domain: str = "",
    ) -> Dictionary:
        """Store a named `:CausalEntity` graph in Neo4j (namespace-scoped) and local fallback registry."""
        graph_name = _norm_sym(name).strip()
        if not graph_name:
            raise ValueError("Graph name must be non-empty.")
        domain_str = _norm_sym(domain).strip() if domain is not None else ""
        nodes, edges = _normalize_graph_spec(
            None, nodes_override=nodes_json, edges_override=edges_json
        )
        sig = _extract_signature_dict(nodes, edges)
        now_ts = time.time()

        entry = {
            "name": graph_name,
            "domain": domain_str,
            "namespace": self.namespace,
            "nodes": nodes,
            "edges": edges,
            "signature": sig,
            "updated_at": now_ts,
        }
        self._local_registry[(self.namespace, graph_name)] = entry

        neo4j_synced = False
        neo4j_error = ""
        if self.enable_neo4j:
            try:
                self._ensure_causal_indexes()
                stmts: List[Dict[str, Any]] = [
                    {
                        "statement": (
                            "MATCH (c:CausalEntity {namespace: $ns, graph_name: $gname}) "
                            "DETACH DELETE c"
                        ),
                        "parameters": {"ns": self.namespace, "gname": graph_name},
                    }
                ]
                for node in nodes:
                    stmts.append(
                        {
                            "statement": (
                                "MERGE (c:CausalEntity {namespace: $ns, graph_name: $gname, entity_id: $eid}) "
                                "SET c.name = $eid, c.role = $role, c.domain = $domain, c.updated_at = $ts"
                            ),
                            "parameters": {
                                "ns": self.namespace,
                                "gname": graph_name,
                                "eid": node["id"],
                                "role": node.get("type", ""),
                                "domain": domain_str,
                                "ts": now_ts,
                            },
                        }
                    )
                for edge in edges:
                    rel_type = _sanitize_rel_type(edge["rel"])
                    stmts.append(
                        {
                            "statement": (
                                "MATCH (a:CausalEntity {namespace: $ns, graph_name: $gname, entity_id: $src}) "
                                "MATCH (b:CausalEntity {namespace: $ns, graph_name: $gname, entity_id: $dst}) "
                                f"MERGE (a)-[r:{rel_type}]->(b) "
                                "SET r.rel = $rel, r.namespace = $ns, r.graph_name = $gname"
                            ),
                            "parameters": {
                                "ns": self.namespace,
                                "gname": graph_name,
                                "src": edge["src"],
                                "dst": edge["dst"],
                                "rel": rel_type,
                            },
                        }
                    )
                self._run_cypher_batch(stmts, read_only=False, timeout=3.0)
                neo4j_synced = True
            except Exception as exc:
                neo4j_error = f"{type(exc).__name__}: {exc}"[:200]

        res = {
            "ok": True,
            "name": graph_name,
            "domain": domain_str,
            "namespace": self.namespace,
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "neo4jSynced": neo4j_synced,
            "neo4jError": neo4j_error,
            "signature": sig,
        }
        return _py_to_actor_value(res)

    def _load_neo4j_candidates(self) -> Dict[str, Dict[str, Any]]:
        """Best-effort fetch of `:CausalEntity` graphs from Neo4j for the active namespace."""
        if not self.enable_neo4j:
            return {}
        try:
            res = self._run_cypher_batch(
                [
                    {
                        "statement": (
                            "MATCH (c:CausalEntity {namespace: $ns}) "
                            "RETURN c.graph_name AS graph_name, c.entity_id AS entity_id, "
                            "c.role AS role, c.domain AS domain"
                        ),
                        "parameters": {"ns": self.namespace},
                    },
                    {
                        "statement": (
                            "MATCH (a:CausalEntity {namespace: $ns})-[r]->(b:CausalEntity {namespace: $ns}) "
                            "WHERE a.graph_name = b.graph_name "
                            "RETURN a.graph_name AS graph_name, a.entity_id AS src, "
                            "b.entity_id AS dst, type(r) AS rel"
                        ),
                        "parameters": {"ns": self.namespace},
                    },
                ],
                read_only=True,
                timeout=3.0,
            )
            node_rows = res[0][1] if len(res) > 0 else []
            edge_rows = res[1][1] if len(res) > 1 else []

            grouped_nodes: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            grouped_domain: Dict[str, str] = {}
            for row in node_rows:
                gname = str(row.get("graph_name") or "").strip()
                eid = str(row.get("entity_id") or "").strip()
                if not gname or not eid:
                    continue
                grouped_nodes[gname].append({"id": eid, "type": str(row.get("role") or "")})
                if row.get("domain") and gname not in grouped_domain:
                    grouped_domain[gname] = str(row.get("domain"))

            grouped_edges: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in edge_rows:
                gname = str(row.get("graph_name") or "").strip()
                src = str(row.get("src") or "").strip()
                dst = str(row.get("dst") or "").strip()
                rel = str(row.get("rel") or "").strip()
                if not gname or not src or not dst or not rel:
                    continue
                grouped_edges[gname].append(
                    {"src": src, "dst": dst, "rel": _sanitize_rel_type(rel)}
                )

            out: Dict[str, Dict[str, Any]] = {}
            for gname, raw_nodes in grouped_nodes.items():
                nodes, edges = _normalize_graph_spec(
                    None, nodes_override=raw_nodes, edges_override=grouped_edges.get(gname, [])
                )
                out[gname] = {
                    "name": gname,
                    "domain": grouped_domain.get(gname, ""),
                    "namespace": self.namespace,
                    "nodes": nodes,
                    "edges": edges,
                    "signature": _extract_signature_dict(nodes, edges),
                }
            return out
        except Exception:
            return {}

    @actor_query
    def findAnalogies(self, target_json: str, limit: int = 3) -> ActorArray:
        """MAC filter + FAC SME alignment across registered graphs (Neo4j + local registry)."""
        top_k = max(1, min(50, int(limit or 3)))
        target_nodes, target_edges = _normalize_graph_spec(target_json)
        target_sig = _extract_signature_dict(target_nodes, target_edges)

        # Merge Neo4j graphs with local registry (local registry takes precedence on same name)
        candidates: Dict[str, Dict[str, Any]] = self._load_neo4j_candidates()
        for (ns, gname), entry in self._local_registry.items():
            if ns == self.namespace:
                candidates[gname] = entry

        if not candidates:
            return ActorArray(0)

        # Stage 1: MAC fast structural filtering
        mac_scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for gname, entry in candidates.items():
            sig = entry.get("signature") or _extract_signature_dict(entry["nodes"], entry["edges"])
            m_score = _mac_similarity(sig, target_sig)
            if m_score > 0.0:
                mac_scored.append((m_score, gname, entry))
        mac_scored.sort(key=lambda x: (-x[0], x[1]))
        mac_shortlist = mac_scored[: max(25, top_k * 5)]

        # Stage 2: FAC SME relational subgraph alignment
        ranked: List[Dict[str, Any]] = []
        for m_score, gname, entry in mac_shortlist:
            fac = _sme_align(entry["nodes"], entry["edges"], target_nodes, target_edges)
            if fac["score"] <= 0.0:
                continue
            ranked.append(
                {
                    "name": gname,
                    "domain": entry.get("domain", ""),
                    "score": fac["score"],
                    "macScore": m_score,
                    "systematicity": fac["systematicity"],
                    "edgeCoverage": fac["edgeCoverage"],
                    "mapping": fac["mapping"],
                    "matchedRelations": fac["matchedRelations"],
                    "candidateInferences": fac["candidateInferences"],
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["systematicity"],
                -item["macScore"],
                item["name"],
            )
        )
        return _py_to_actor_value(ranked[:top_k])

    @actor_query
    def describe(self) -> str:
        """Concise self-documentation of `Actor[#Analogy]` (`AnalogyEngineBridge`)."""
        return (
            "AnalogyEngineBridge (`Actor[#Analogy]`) — Pure-CPU (0 MB VRAM) MAC/FAC Structural Analogy Engine "
            "(Gentner SME over directed labeled graphs using networkx).\n"
            "Graph JSON format: {\"nodes\": [{\"id\": \"A\", \"type\": \"Resource\"}], "
            "\"edges\": [{\"src\": \"A\", \"dst\": \"B\", \"rel\": \"CAUSES\"}]}\n"
            "Methods:\n"
            "  - extractStructure(Actor[#Analogy], graphJson)!! -> Dictionary (#nodeCount, #edgeCount, #relationHistogram, #inDegreeProfile, #outDegreeProfile, #maxCausalChainDepth)\n"
            "  - alignGraphs(Actor[#Analogy], baseJson, targetJson)!! -> Dictionary (#score, #systematicity, #mapping, #matchedRelations, #candidateInferences)\n"
            "  - registerCausalGraph(Actor[#Analogy], name, nodesJson, edgesJson, domain)!! -> Dictionary (stores :CausalEntity graph in Neo4j + local registry)\n"
            "  - findAnalogies(Actor[#Analogy], targetJson, limit)!! -> Array of top-K analogies (#name, #domain, #score, #mapping, #candidateInferences)\n"
            "  - getNamespace(Actor[#Analogy])!! / setNamespace(Actor[#Analogy], ns)!!\n"
            "  - describe(Actor[#Analogy])!! -> This summary."
        )
