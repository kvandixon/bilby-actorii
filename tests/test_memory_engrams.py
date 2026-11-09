"""Comprehensive Unit and Integration Tests for `ActorMemoryGraph` (`Actor[#Memory]`) and Active Engrams (Phases 1-4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from actorii.harness import bootstrap_agent_harness
from actorii.tui import ActorTUISession
from websocket_server import ActorWorkspace


class TestActorMemoryGraphAndEngrams:
    """Verify Neo4j graph read/write, zero-token auto-crystallization, transitive dependencies, versioning, and TUI/WS."""

    def test_cypher_write_and_query_from_actor_bytecode(self) -> None:
        vm, _harness, sandbox = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace="pytest_engrams",
        )
        assert sandbox.memory_graph is not None
        sandbox.memory_graph._run_cypher("MATCH (n:PytestSensor) DETACH DELETE n")

        try:
            vm.eval_chunk(
                'cypherWrite(Actor[#Memory], "MERGE (s:PytestSensor {id: \'S1\', temp: 72}) '
                'MERGE (s2:PytestSensor {id: \'S2\', temp: 88})")!!'
            )
            schema_txt = vm.eval_chunk("graphSchema(Actor[#Memory])!!")
            assert "PytestSensor" in str(schema_txt)

            avg_temp = vm.eval_chunk(
                """
                inherit(Object, #TempReader, #(#tag), 1, nil)!!
                now(TempReader)!!
                Def meanTemp(self | rows, i, n, total, row) {
                    rows := cypherQuery(Actor[#Memory], "MATCH (s:PytestSensor) RETURN s.id AS id, s.temp AS temp");
                    n := size(rows);
                    total := 0;
                    i := 0;
                    loop while (i < n) begin
                        row := rows[i];
                        total := total + at(row, #temp);
                        i := i + 1;
                    endLoop;
                    ^(total / n);
                }!!
                Actor[#Reader] := new(TempReader)!!
                meanTemp(Actor[#Reader])!!
                """
            )
            assert avg_temp == 80
        finally:
            sandbox.memory_graph.clearEngrams()
            sandbox.memory_graph._run_cypher("MATCH (n:PytestSensor) DETACH DELETE n")

    def test_engram_save_and_cold_reboot_hydration_across_vms(self) -> None:
        vm1, _h1, sb1 = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace="pytest_cold_boot",
        )
        assert sb1.memory_graph is not None
        sb1.memory_graph.clearEngrams()
        sb1.memory_graph._run_cypher("MATCH (n:PytestNode) DETACH DELETE n")

        try:
            vm1.eval_chunk(
                'cypherWrite(Actor[#Memory], "CREATE (:PytestNode {name: \'A\', weight: 15}), '
                '(:PytestNode {name: \'B\', weight: 25}), (:PytestNode {name: \'C\', weight: 5})")!!'
            )
            engram_src = (
                "inherit(Object, #WeightScorer, #(#tag), 1, nil)!!\n"
                "now(WeightScorer)!!\n"
                "Def sumAbove(self, minWeight | rows, i, row, w, acc) {\n"
                '  rows := cypherQuery(Actor[#Memory], "MATCH (n:PytestNode) RETURN n.weight AS weight");\n'
                "  acc := 0;\n"
                "  i := 0;\n"
                "  loop while (i < size(rows)) begin\n"
                "    row := rows[i];\n"
                "    w := at(row, #weight);\n"
                "    if (w >= minWeight) then acc := acc + w; endif;\n"
                "    i := i + 1;\n"
                "  endLoop;\n"
                "  ^acc;\n"
                "}!!\n"
                "Actor[#WeightScorer] := new(WeightScorer)!!"
            )
            save_res = sb1.memory_graph.saveEngram(
                "#WeightScorer",
                "Sum PytestNode weights >= minWeight",
                engram_src,
            )
            assert "ENGRAM_SAVED" in save_res
            assert "sumAbove(Actor[#WeightScorer], minWeight)" in save_res

            # Session 2: Brand-new cold-start VM (`vm2`) with empty RAM
            vm2, _h2, sb2 = bootstrap_agent_harness(
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace="pytest_cold_boot",
            )
            assert sb2.memory_graph is not None
            assert "WeightScorer" not in vm2.global_vars["Actor"]

            hydrated_info = sb2.memory_graph.hydrateRelevantEngrams("Calculate WeightScorer sumAbove")
            assert "Actor[#WeightScorer]" in hydrated_info
            assert ":PytestNode" in hydrated_info
            assert "WeightScorer" in vm2.global_vars["Actor"]

            assert vm2.eval_chunk("sumAbove(Actor[#WeightScorer], 10)!!") == 40
            assert vm2.eval_chunk("sumAbove(Actor[#WeightScorer], 20)!!") == 25
        finally:
            sb1.memory_graph.clearEngrams()
            sb1.memory_graph._run_cypher("MATCH (n:PytestNode) DETACH DELETE n")

    def test_phase1_zero_token_auto_crystallization_on_step_execution(self) -> None:
        """Verify `VMSandboxBridge.executeStepAction` auto-crystallizes user classes into Neo4j with 0 extra LLM calls."""
        vm1, h1, sb1 = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace="pytest_auto_cryst",
        )
        assert sb1.memory_graph is not None
        sb1.memory_graph.clearEngrams()
        sb1.memory_graph._run_cypher("MATCH (n:PytestMetric) DETACH DELETE n")

        try:
            h1._fields["goal"] = "Calculate metric score above threshold"
            # Execute a step that defines a class `MetricCalc` WITHOUT calling `persistEngram` or `saveEngram`!
            step_code = (
                'cypherWrite(Actor[#Memory], "CREATE (:PytestMetric {val: 10}), (:PytestMetric {val: 30})")!!\n'
                "inherit(Object, #MetricCalc, #(#state), 1, nil)!!\n"
                "now(MetricCalc)!!\n"
                "Def sumMetrics(self, minVal | rows, i, row, v, total) {\n"
                '    rows := cypherQuery(Actor[#Memory], "MATCH (m:PytestMetric) RETURN m.val AS val");\n'
                "    total := 0; i := 0;\n"
                "    loop while (i < size(rows)) begin\n"
                "        row := rows[i]; v := at(row, #val);\n"
                "        if (v >= minVal) then total := total + v; endif;\n"
                "        i := i + 1;\n"
                "    endLoop;\n"
                "    ^total;\n"
                "}!!\n"
                "Actor[#MetricCalc] := new(MetricCalc)!!\n"
                "storeScratch(Actor[#Harness], #answer, sumMetrics(Actor[#MetricCalc], 15))!!"
            )
            res = sb1.executeStepAction(h1, step_code, max_opcodes=25000)
            assert res == 30
            step_log = list(h1._fields["stepLog"])
            assert step_log[-1]["auto_engrams"] == ["MetricCalc"]

            # Cold-boot a second VM and verify `MetricCalc` is already persisted in Neo4j!
            vm2, _h2, sb2 = bootstrap_agent_harness(
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace="pytest_auto_cryst",
            )
            assert sb2.memory_graph is not None
            sb2.memory_graph.loadEngram("#MetricCalc")
            assert vm2.eval_chunk("sumMetrics(Actor[#MetricCalc], 5)!!") == 40

            # Also verify that adding a method to an already-existing class via `now(ExistingClass)!! Def ... !!`
            # (without `inherit(...)` in the same chunk) auto-crystallizes the updated class as v2!
            ext_code = "now(MetricCalc)!!\nDef doubleSum(self, minVal) { ^(sumMetrics(self, minVal) * 2); }!!"
            sb1.executeStepAction(h1, ext_code, max_opcodes=25000)
            vm3, _h3, sb3 = bootstrap_agent_harness(
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace="pytest_auto_cryst",
            )
            assert sb3.memory_graph is not None
            sb3.memory_graph.loadEngram("#MetricCalc")
            assert vm3.eval_chunk("doubleSum(Actor[#MetricCalc], 15)!!") == 60
        finally:
            sb1.memory_graph.clearEngrams()
            sb1.memory_graph._run_cypher("MATCH (n:PytestMetric) DETACH DELETE n")

    def test_phase2_transitive_engram_dependencies_and_version_rollback(self) -> None:
        """Verify Engram B depending on Engram A hydrates transitively on cold reboot, and `rollbackEngram` restores v1."""
        vm1, _h1, sb1 = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace="pytest_deps_rollback",
        )
        assert sb1.memory_graph is not None
        sb1.memory_graph.clearEngrams()
        sb1.memory_graph._run_cypher("MATCH (n:PytestRate) DETACH DELETE n")

        try:
            # 1. Define base Engram `BaseTax` (v1 returns x * 2)
            vm1.eval_chunk(
                """
                inherit(Object, #BaseTax, #(#state), 1, nil)!!
                now(BaseTax)!!
                Def applyTax(self, amt) { ^(amt * 2); }!!
                Actor[#BaseTax] := new(BaseTax)!!
                persistEngram(Actor[#Memory], #BaseTax, "Base tax multiplier")!!
                """
            )
            # 2. Define composite Engram `InvoiceTotal` that calls `applyTax(Actor[#BaseTax], amt)`
            vm1.eval_chunk(
                """
                inherit(Object, #InvoiceTotal, #(#state), 1, nil)!!
                now(InvoiceTotal)!!
                Def computeInvoice(self, baseAmt) {
                    ^(applyTax(Actor[#BaseTax], baseAmt) + 10);
                }!!
                Actor[#InvoiceTotal] := new(InvoiceTotal)!!
                persistEngram(Actor[#Memory], #InvoiceTotal, "Composite invoice calculator")!!
                """
            )

            # 3. Boot a brand-new cold VM (`vm2`) and load ONLY `#InvoiceTotal` -> must transitively hydrate `#BaseTax`!
            vm2, _h2, sb2 = bootstrap_agent_harness(
                enable_memory_graph=True,
                enable_engrams=True,
                memory_namespace="pytest_deps_rollback",
            )
            assert sb2.memory_graph is not None
            sb2.memory_graph.loadEngram("#InvoiceTotal")
            assert "BaseTax" in vm2.global_vars["Actor"]
            assert "InvoiceTotal" in vm2.global_vars["Actor"]
            assert vm2.eval_chunk("computeInvoice(Actor[#InvoiceTotal], 50)!!") == 110

            # 4. Hot-patch `BaseTax` to v2 (`amt * 10`) and persist -> then rollback to v1 (`amt * 2`)!
            vm2.eval_chunk(
                """
                now(BaseTax)!!
                Def applyTax(self, amt) { ^(amt * 10); }!!
                persistEngram(Actor[#Memory], #BaseTax, "Base tax multiplier v2")!!
                """
            )
            assert vm2.eval_chunk("computeInvoice(Actor[#InvoiceTotal], 50)!!") == 510

            rb_msg = vm2.eval_chunk("rollbackEngram(Actor[#Memory], #BaseTax)!!")
            assert "ENGRAM_ROLLED_BACK" in str(rb_msg)
            assert vm2.eval_chunk("computeInvoice(Actor[#InvoiceTotal], 50)!!") == 110
        finally:
            sb1.memory_graph.clearEngrams()

    def test_phase3_parameterized_cypher_query_with_and_write_with(self) -> None:
        """Verify `cypherWriteWith` and `cypherQueryWith` using positional `$p1, $p2` parameters."""
        vm, _h, sb = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace="pytest_param_cypher",
        )
        assert sb.memory_graph is not None
        sb.memory_graph._run_cypher("MATCH (n:PytestParamNode) DETACH DELETE n")

        try:
            vm.eval_chunk(
                """
                cypherWriteWith(Actor[#Memory], "CREATE (:PytestParamNode {code: $p1, score: $p2})", "ALPHA", 42)!!
                cypherWriteWith(Actor[#Memory], "CREATE (:PytestParamNode {code: $p1, score: $p2})", "BETA", 99)!!
                """
            )
            val = vm.eval_chunk(
                """
                rows := cypherQueryWith(Actor[#Memory], "MATCH (n:PytestParamNode {code: $p1}) RETURN n.score AS score", "BETA");
                at(rows[0], #score)!!
                """
            )
            assert val == 99
        finally:
            sb.memory_graph._run_cypher("MATCH (n:PytestParamNode) DETACH DELETE n")

    def test_phase4_tui_commands_and_websocket_engram_endpoints(self) -> None:
        """Verify `/engrams`, `/schema`, and `/hydrate` in `ActorTUISession` and `get_engrams` in `ActorWorkspace`."""
        session = ActorTUISession(use_color=False)
        assert session.sandbox.memory_graph is not None
        session.sandbox.memory_graph.namespace = "pytest_tui_ws"
        session.sandbox.memory_graph.clearEngrams()

        try:
            session.vm.eval_chunk(
                """
                inherit(Object, #TuiHelper, #(#state), 1, nil)!!
                now(TuiHelper)!!
                Def doubleIt(self, x) { ^(x * 2); }!!
                Actor[#TuiHelper] := new(TuiHelper)!!
                persistEngram(Actor[#Memory], #TuiHelper, "Double integer helper")!!
                """
            )
            assert session.handle_input("/engrams") is True
            assert any("TuiHelper" in ln for ln in session.transcript)

            assert session.handle_input("/schema") is True
            assert session.handle_input("/hydrate TuiHelper") is True
            assert session.vm.eval_chunk("doubleIt(Actor[#TuiHelper], 21)!!") == 42

            ws = ActorWorkspace()
            engram_resp = asyncio.run(ws.handle_message({"type": "get_engrams"}))
            assert engram_resp["type"] == "engrams_list"
            assert engram_resp["success"] is True

            schema_resp = asyncio.run(ws.handle_message({"type": "get_graph_schema"}))
            assert schema_resp["type"] == "graph_schema"
            assert schema_resp["success"] is True
        finally:
            session.sandbox.memory_graph.clearEngrams()
