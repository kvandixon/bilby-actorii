"""Unit tests for the MAC/FAC Structural Analogy Engine (`AnalogyEngineBridge`)."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List
from unittest.mock import patch
import urllib.error

import pytest

from actorii.bridge import ActorBridgeObject
from actorii.core import Symbol
from actorii.harness.analogy_engine import AnalogyEngineBridge, _actor_to_py_value


class _FakeHTTPResponse:
    """Minimal context-manager mock for `urllib.request.urlopen`."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_extract_structure_dag_and_cycle() -> None:
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")

    dag_graph = {
        "nodes": [
            {"id": "Sensor", "type": "Trigger"},
            {"id": "Buffer", "type": "Resource"},
            {"id": "Worker", "type": "Mechanism"},
            {"id": "Output", "type": "State"},
        ],
        "edges": [
            {"src": "Sensor", "dst": "Buffer", "rel": "CAUSES"},
            {"src": "Buffer", "dst": "Worker", "rel": "DEPENDS_ON"},
            {"src": "Worker", "dst": "Output", "rel": "CAUSES"},
            {"src": "Buffer", "dst": "Output", "rel": "CONSTRAINS"},
        ],
    }
    sig = _actor_to_py_value(bridge.extractStructure(json.dumps(dag_graph)))
    assert sig["nodeCount"] == 4
    assert sig["edgeCount"] == 4
    assert sig["relationHistogram"] == {"CAUSES": 2, "CONSTRAINS": 1, "DEPENDS_ON": 1}
    assert sig["maxCausalChainDepth"] == 3
    assert sig["inDegreeProfile"] == [2, 1, 1, 0]
    assert sig["outDegreeProfile"] == [2, 1, 1, 0]
    assert sig["connectedComponents"] == 1
    assert sig["hasCycles"] is False

    # Cyclic graph should terminate cleanly and report cycle + simple chain depth
    cyclic_graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [
            {"src": "A", "dst": "B", "rel": "CAUSES"},
            {"src": "B", "dst": "C", "rel": "CAUSES"},
            {"src": "C", "dst": "A", "rel": "CONSTRAINS"},
        ],
    }
    csig = _actor_to_py_value(bridge.extractStructure(json.dumps(cyclic_graph)))
    assert csig["hasCycles"] is True
    assert csig["maxCausalChainDepth"] == 2


def test_cross_domain_isomorphic_transfer_ignores_surface_names() -> None:
    """Two graphs from completely different domains with zero shared vocabulary align with score 1.0."""
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")

    # Domain 1: Hydraulic plumbing system
    hydraulic_base = {
        "nodes": [
            {"id": "WaterPump", "type": "Source"},
            {"id": "PipeValve", "type": "Gate"},
            {"id": "PressureTank", "type": "Reservoir"},
            {"id": "ReliefNozzle", "type": "Escape"},
        ],
        "edges": [
            {"src": "WaterPump", "dst": "PipeValve", "rel": "CAUSES"},
            {"src": "PipeValve", "dst": "PressureTank", "rel": "CONSTRAINS"},
            {"src": "PressureTank", "dst": "ReliefNozzle", "rel": "DEPENDS_ON"},
            {"src": "WaterPump", "dst": "PressureTank", "rel": "CAUSES"},
        ],
    }

    # Domain 2: LLM batching & KV-cache scheduler (zero lexical overlap in node IDs or types)
    scheduler_target = {
        "nodes": [
            {"id": "PromptIngress", "type": "Queue"},
            {"id": "TokenThrottle", "type": "Limiter"},
            {"id": "KVCacheArena", "type": "Memory"},
            {"id": "EvictionSweep", "type": "Reclaimer"},
        ],
        "edges": [
            {"src": "PromptIngress", "dst": "TokenThrottle", "rel": "CAUSES"},
            {"src": "TokenThrottle", "dst": "KVCacheArena", "rel": "CONSTRAINS"},
            {"src": "KVCacheArena", "dst": "EvictionSweep", "rel": "DEPENDS_ON"},
            {"src": "PromptIngress", "dst": "KVCacheArena", "rel": "CAUSES"},
        ],
    }

    res_actor = bridge.alignGraphs(json.dumps(hydraulic_base), json.dumps(scheduler_target))
    # Verify native Actor Dictionary #Symbol lookup works
    assert res_actor.at(Symbol("score")) == pytest.approx(1.0)
    res = _actor_to_py_value(res_actor)

    assert res["score"] == pytest.approx(1.0)
    assert res["mapping"] == {
        "WaterPump": "PromptIngress",
        "PipeValve": "TokenThrottle",
        "PressureTank": "KVCacheArena",
        "ReliefNozzle": "EvictionSweep",
    }
    assert len(res["matchedRelations"]) == 4
    assert res["candidateInferences"] == []


def test_partial_subgraph_alignment_and_candidate_inference_projection() -> None:
    """Unmapped base relations with mapped endpoints project concrete target hypotheses."""
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")

    # Complete solved base causal model:
    # TrafficSpike -[:CAUSES]-> QueueOverflow -[:CAUSES]-> TimeoutStorm
    # TrafficSpike -[:CONSTRAINS]-> TimeoutStorm (direct missing relation in target)
    # TimeoutStorm -[:RESOLVES]-> CircuitBreaker (projected consequence in target)
    base_graph = {
        "nodes": [
            {"id": "TrafficSpike"},
            {"id": "QueueOverflow"},
            {"id": "TimeoutStorm"},
            {"id": "CircuitBreaker"},
        ],
        "edges": [
            {"src": "TrafficSpike", "dst": "QueueOverflow", "rel": "CAUSES"},
            {"src": "QueueOverflow", "dst": "TimeoutStorm", "rel": "CAUSES"},
            {"src": "TrafficSpike", "dst": "TimeoutStorm", "rel": "CONSTRAINS"},
            {"src": "TimeoutStorm", "dst": "CircuitBreaker", "rel": "RESOLVES"},
        ],
    }

    # Partial target observation: only the 2-hop CAUSES chain is observed so far
    target_graph = {
        "nodes": [
            {"id": "BatchBurst"},
            {"id": "VRAMContention"},
            {"id": "InferenceStall"},
        ],
        "edges": [
            {"src": "BatchBurst", "dst": "VRAMContention", "rel": "CAUSES"},
            {"src": "VRAMContention", "dst": "InferenceStall", "rel": "CAUSES"},
        ],
    }

    res = _actor_to_py_value(bridge.alignGraphs(json.dumps(base_graph), json.dumps(target_graph)))
    assert 0.5 < res["score"] < 1.0
    assert res["mapping"] == {
        "TrafficSpike": "BatchBurst",
        "QueueOverflow": "VRAMContention",
        "TimeoutStorm": "InferenceStall",
    }
    assert len(res["matchedRelations"]) == 2

    inferences = res["candidateInferences"]
    assert len(inferences) == 2

    # 1. Missing relation between two mapped target nodes
    missing_rel = [inf for inf in inferences if inf["kind"] == "missing_relation"]
    assert len(missing_rel) == 1
    assert missing_rel[0]["src"] == "BatchBurst"
    assert missing_rel[0]["dst"] == "InferenceStall"
    assert missing_rel[0]["rel"] == "CONSTRAINS"

    # 2. Projected consequence from mapped tail node to skolemized remedy
    projected = [inf for inf in inferences if inf["kind"] == "projected_consequence"]
    assert len(projected) == 1
    assert projected[0]["src"] == "InferenceStall"
    assert projected[0]["dst"] == "?CircuitBreaker"
    assert projected[0]["rel"] == "RESOLVES"


def test_systematicity_ranking_connected_chain_beats_disconnected_edges() -> None:
    """Gentner's Systematicity Principle: a 3-edge connected causal chain outranks 3 isolated edges."""
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")

    # Target contains BOTH a 3-hop connected chain AND 3 disconnected single CAUSES edges
    target_graph = {
        "nodes": [
            {"id": "T1"}, {"id": "T2"}, {"id": "T3"}, {"id": "T4"},
            {"id": "U1"}, {"id": "U2"}, {"id": "U3"}, {"id": "U4"}, {"id": "U5"}, {"id": "U6"},
        ],
        "edges": [
            # Connected 3-hop chain
            {"src": "T1", "dst": "T2", "rel": "CAUSES"},
            {"src": "T2", "dst": "T3", "rel": "CAUSES"},
            {"src": "T3", "dst": "T4", "rel": "CAUSES"},
            # 3 disconnected single edges
            {"src": "U1", "dst": "U2", "rel": "CAUSES"},
            {"src": "U3", "dst": "U4", "rel": "CAUSES"},
            {"src": "U5", "dst": "U6", "rel": "CAUSES"},
        ],
    }

    # Candidate A: 3-edge connected causal chain (completely different names)
    chain_candidate = {
        "nodes": [{"id": "C1"}, {"id": "C2"}, {"id": "C3"}, {"id": "C4"}],
        "edges": [
            {"src": "C1", "dst": "C2", "rel": "CAUSES"},
            {"src": "C2", "dst": "C3", "rel": "CAUSES"},
            {"src": "C3", "dst": "C4", "rel": "CAUSES"},
        ],
    }

    # Candidate B: 3 disconnected single CAUSES edges (same edge count & relation histogram!)
    disconnected_candidate = {
        "nodes": [{"id": "D1"}, {"id": "D2"}, {"id": "D3"}, {"id": "D4"}, {"id": "D5"}, {"id": "D6"}],
        "edges": [
            {"src": "D1", "dst": "D2", "rel": "CAUSES"},
            {"src": "D3", "dst": "D4", "rel": "CAUSES"},
            {"src": "D5", "dst": "D6", "rel": "CAUSES"},
        ],
    }

    res_chain = _actor_to_py_value(bridge.alignGraphs(json.dumps(chain_candidate), json.dumps(target_graph)))
    res_disc = _actor_to_py_value(bridge.alignGraphs(json.dumps(disconnected_candidate), json.dumps(target_graph)))

    assert len(res_chain["matchedRelations"]) == 3
    assert len(res_disc["matchedRelations"]) == 3
    assert res_chain["systematicity"] > res_disc["systematicity"]
    assert res_chain["score"] > res_disc["score"]

    # Also verify findAnalogies ranks the connected chain #1 ahead of the disconnected graph
    bridge.registerCausalGraph(
        "disconnected_pairs",
        json.dumps(disconnected_candidate["nodes"]),
        json.dumps(disconnected_candidate["edges"]),
        "surface",
    )
    bridge.registerCausalGraph(
        "connected_chain",
        json.dumps(chain_candidate["nodes"]),
        json.dumps(chain_candidate["edges"]),
        "causal_deep",
    )
    ranked = _actor_to_py_value(bridge.findAnalogies(json.dumps(target_graph), 2))
    assert len(ranked) == 2
    assert ranked[0]["name"] == "connected_chain"
    assert ranked[1]["name"] == "disconnected_pairs"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_register_and_find_analogies_offline_fallback_and_namespace_isolation() -> None:
    """When Neo4j is unreachable, registration falls back cleanly to the local registry scoped by namespace."""
    bridge = AnalogyEngineBridge(
        tx_url="http://127.0.0.1:1/db/neo4j/tx/commit",
        namespace="pytest_ns_alpha",
        enable_neo4j=True,
    )

    nodes_relay = [{"id": "Ingress"}, {"id": "Backpressure"}, {"id": "DropPacket"}, {"id": "RateLimit"}]
    edges_relay = [
        {"src": "Ingress", "dst": "Backpressure", "rel": "CAUSES"},
        {"src": "Backpressure", "dst": "DropPacket", "rel": "CAUSES"},
        {"src": "RateLimit", "dst": "Backpressure", "rel": "CONSTRAINS"},
    ]

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        reg = _actor_to_py_value(
            bridge.registerCausalGraph(
                "PortRelayBackpressure",
                json.dumps(nodes_relay),
                json.dumps(edges_relay),
                "networking",
            )
        )
        assert reg["ok"] is True
        assert reg["neo4jSynced"] is False
        assert reg["namespace"] == "pytest_ns_alpha"
        assert reg["nodeCount"] == 4
        assert reg["edgeCount"] == 3

        # Register an unrelated graph with disjoint relations
        bridge.registerCausalGraph(
            "RecipePrep",
            json.dumps([{"id": "Flour"}, {"id": "Dough"}]),
            json.dumps([{"src": "Flour", "dst": "Dough", "rel": "TRANSFORMS_INTO"}]),
            "cooking",
        )

        # Query with a partial scheduling graph (missing the CONSTRAINS remedy)
        target = {
            "nodes": [{"id": "JobArrival"}, {"id": "QueueStarvation"}, {"id": "TaskTimeout"}],
            "edges": [
                {"src": "JobArrival", "dst": "QueueStarvation", "rel": "CAUSES"},
                {"src": "QueueStarvation", "dst": "TaskTimeout", "rel": "CAUSES"},
            ],
        }
        matches = _actor_to_py_value(bridge.findAnalogies(json.dumps(target), 3))
        assert len(matches) == 1
        assert matches[0]["name"] == "PortRelayBackpressure"
        assert matches[0]["domain"] == "networking"
        assert matches[0]["mapping"]["Backpressure"] == "QueueStarvation"
        # Candidate inference should project the missing RateLimit CONSTRAINS QueueStarvation!
        infs = matches[0]["candidateInferences"]
        assert any(
            inf["rel"] == "CONSTRAINS" and inf["dst"] == "QueueStarvation"
            for inf in infs
        )

        # Switching namespace isolates the registry
        bridge.setNamespace("pytest_ns_beta")
        assert bridge.getNamespace() == "pytest_ns_beta"
        other_matches = _actor_to_py_value(bridge.findAnalogies(json.dumps(target), 3))
        assert other_matches == []


def test_register_and_find_analogies_mocked_neo4j() -> None:
    """Verify Cypher write statements and Neo4j-backed `findAnalogies` retrieval via mocked HTTP TX."""
    captured_payloads: List[Dict[str, Any]] = []

    def _fake_urlopen(req, timeout=3.0):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        stmts = payload.get("statements", [])
        # Distinguish read query in `_load_neo4j_candidates` from write batch
        if stmts and "RETURN c.graph_name AS graph_name" in stmts[0].get("statement", ""):
            return _FakeHTTPResponse(
                {
                    "results": [
                        {
                            "columns": ["graph_name", "entity_id", "role", "domain"],
                            "data": [
                                {"row": ["Neo4jDAG", "JobA", "Task", "scheduler"]},
                                {"row": ["Neo4jDAG", "JobB", "Task", "scheduler"]},
                                {"row": ["Neo4jDAG", "JobC", "Task", "scheduler"]},
                            ],
                        },
                        {
                            "columns": ["graph_name", "src", "dst", "rel"],
                            "data": [
                                {"row": ["Neo4jDAG", "JobA", "JobB", "DEPENDS_ON"]},
                                {"row": ["Neo4jDAG", "JobB", "JobC", "DEPENDS_ON"]},
                                {"row": ["Neo4jDAG", "JobA", "JobC", "CONSTRAINS"]},
                            ],
                        },
                    ],
                    "errors": [],
                }
            )
        return _FakeHTTPResponse({"results": [], "errors": []})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        writer = AnalogyEngineBridge(
            tx_url="http://mock-neo4j:7474/db/neo4j/tx/commit",
            namespace="pytest_neo4j_ns",
            enable_neo4j=True,
        )
        reg = _actor_to_py_value(
            writer.registerCausalGraph(
                "Neo4jDAG",
                json.dumps([{"id": "JobA", "type": "Task"}, {"id": "JobB", "type": "Task"}, {"id": "JobC", "type": "Task"}]),
                json.dumps([
                    {"src": "JobA", "dst": "JobB", "rel": "DEPENDS_ON"},
                    {"src": "JobB", "dst": "JobC", "rel": "DEPENDS_ON"},
                    {"src": "JobA", "dst": "JobC", "rel": "CONSTRAINS"},
                ]),
                "scheduler",
            )
        )
        assert reg["ok"] is True
        assert reg["neo4jSynced"] is True

        # Confirm Cypher payload created :CausalEntity nodes and :DEPENDS_ON / :CONSTRAINS rels scoped to namespace
        all_stmts = [st["statement"] for p in captured_payloads for st in p.get("statements", [])]
        assert any("MERGE (c:CausalEntity {namespace: $ns" in s for s in all_stmts)
        assert any("MERGE (a)-[r:DEPENDS_ON]->(b)" in s for s in all_stmts)
        assert any("MERGE (a)-[r:CONSTRAINS]->(b)" in s for s in all_stmts)

        # Fresh bridge instance with empty local registry must discover Neo4jDAG from Neo4j!
        reader = AnalogyEngineBridge(
            tx_url="http://mock-neo4j:7474/db/neo4j/tx/commit",
            namespace="pytest_neo4j_ns",
            enable_neo4j=True,
        )
        target = {
            "nodes": [{"id": "BuildStep1"}, {"id": "BuildStep2"}, {"id": "DeployStep"}],
            "edges": [
                {"src": "BuildStep1", "dst": "BuildStep2", "rel": "DEPENDS_ON"},
                {"src": "BuildStep2", "dst": "DeployStep", "rel": "DEPENDS_ON"},
                {"src": "BuildStep1", "dst": "DeployStep", "rel": "CONSTRAINS"},
            ],
        }
        hits = _actor_to_py_value(reader.findAnalogies(json.dumps(target), 3))
        assert len(hits) == 1
        assert hits[0]["name"] == "Neo4jDAG"
        assert hits[0]["domain"] == "scheduler"
        assert hits[0]["score"] == pytest.approx(1.0)
        assert hits[0]["mapping"] == {
            "JobA": "BuildStep1",
            "JobB": "BuildStep2",
            "JobC": "DeployStep",
        }


def test_describe_and_actor_bridge_capabilities() -> None:
    """Verify `describe()` self-documentation and capability tiers via `ActorBridgeObject`."""
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")
    obj = ActorBridgeObject(bridge)

    caps = obj.bridgeCapabilities()
    assert caps["extractStructure"] == 0
    assert caps["alignGraphs"] == 0
    assert caps["findAnalogies"] == 0
    assert caps["describe"] == 0
    assert caps["registerCausalGraph"] == 2

    desc = obj.invoke_bridge_member("describe")
    assert isinstance(desc, str)
    for token in (
        "AnalogyEngineBridge",
        "Actor[#Analogy]",
        "extractStructure",
        "alignGraphs",
        "registerCausalGraph",
        "findAnalogies",
        "describe",
    ):
        assert token in desc


def test_general_partial_overlap_and_disjoint_graphs() -> None:
    """Exercise general SME branch-and-bound when neither graph embeds in the other, plus disjoint graphs."""
    bridge = AnalogyEngineBridge(enable_neo4j=False, namespace="pytest_analogy")

    # Neither graph is a subgraph of the other (base has RESOLVES, target has BLOCKS)
    base = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
        "edges": [
            {"src": "A", "dst": "B", "rel": "CAUSES"},
            {"src": "B", "dst": "C", "rel": "CAUSES"},
            {"src": "C", "dst": "D", "rel": "RESOLVES"},
        ],
    }
    target = {
        "nodes": [{"id": "W"}, {"id": "X"}, {"id": "Y"}, {"id": "Z"}],
        "edges": [
            {"src": "W", "dst": "X", "rel": "BLOCKS"},
            {"src": "X", "dst": "Y", "rel": "CAUSES"},
            {"src": "Y", "dst": "Z", "rel": "CAUSES"},
        ],
    }
    res = _actor_to_py_value(bridge.alignGraphs(json.dumps(base), json.dumps(target)))
    assert 0.5 < res["score"] < 1.0
    assert res["mapping"] == {"A": "X", "B": "Y", "C": "Z"}
    assert len(res["matchedRelations"]) == 2
    assert any(
        inf["src"] == "Z" and inf["dst"] == "?D" and inf["rel"] == "RESOLVES"
        for inf in res["candidateInferences"]
    )

    # Disjoint relation sets -> score 0.0, empty mapping & inferences
    disjoint = {
        "nodes": [{"id": "P"}, {"id": "Q"}],
        "edges": [{"src": "P", "dst": "Q", "rel": "OWNS"}],
    }
    res_zero = _actor_to_py_value(bridge.alignGraphs(json.dumps(base), json.dumps(disjoint)))
    assert res_zero["score"] == 0.0
    assert res_zero["mapping"] == {}
    assert res_zero["matchedRelations"] == []
    assert res_zero["candidateInferences"] == []
