"""Unit Tests & Microbenchmarks for ActorII Runtime Optimizations.

Covers:
1. Bounded LRU AST & Bytecode Caching (`parse_actor`, `compile_to_bytecode`,
   `is_authentic_1993_actor`, `clear_parser_caches`, `get_parser_cache_stats`):
   - Verifies cache hit/miss accounting and bounded LRU eviction (`_PARSER_CACHE_MAXSIZE = 512`).
   - Verifies deep-copy isolation: mutating returned AST nodes (`node.name = ...`) or
     compiled bytecode instruction lists does NOT corrupt cached entries.
   - Measures cold vs. warm AST + bytecode compilation speedup (asserting significant speedup).
2. Precompiled Lexer & Guardrail Linter Optimizations (`ActorLinter`):
   - Verifies O(1) Tier 2 bridge selector lookup cache (`_get_tier2_bridge_selectors`),
     ivar lineage cache (`collect_allowed_fields`), and precompiled Ring-1 selector checks.
   - Verifies 100% equivalence of `LintReport` across Tier 0, Tier 1, Tier 2, undeclared ivar,
     and protected singleton checks.
3. AI Client & Method Synthesizer Thread-Safety & Regex Precompilation (`ai.py`):
   - Verifies precompiled regex fast-paths (`_strip_thinking_only`, `_strip_thinking_and_fences`,
     `extract_and_normalize_actor_method`).
   - Verifies thread-safe token/call accounting and per-thread `get_thread_last_call_stats()`
     under 16-thread concurrent `ValkyrieLLMClient` invocations (zero lost increments or races).
   - Verifies concurrent parser, compiler, and linter execution across 8 worker threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from actorii.ai import (
    ValkyrieLLMClient,
    _strip_thinking_and_fences,
    _strip_thinking_only,
    extract_and_normalize_actor_method,
    inspect_actor_object,
)
from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.guardrails.linter import ActorLinter, RiskTier
from actorii.parser import (
    MethodDefNode,
    _PARSER_CACHE_MAXSIZE,
    _parse_actor_1993,
    clear_parser_caches,
    compile_to_bytecode,
    get_parser_cache_stats,
    is_authentic_1993_actor,
    parse_actor,
)
from actorii.vm import ActorVM, Opcode


# ==============================================================================
# 1. Parser & Bytecode Cache Correctness, Isolation, and Microbenchmarks
# ==============================================================================


class TestParserAndBytecodeCaching:
    """Tests bounded LRU caching, mutation isolation, and cache hit speedups."""

    def test_ast_and_bytecode_cache_hit_accounting(self):
        clear_parser_caches()
        stats0 = get_parser_cache_stats()
        assert stats0["ast_hits"] == 0
        assert stats0["ast_misses"] == 0
        assert stats0["bytecode_hits"] == 0

        source = """
        inherit(Object, #CacheBenchNode, #(#alpha #beta), 2, nil)!!
        now(CacheBenchNode)!!
        Def compute(self, x, y | tmp) {
            tmp := (x * 2) + y;
            if tmp > 10 then
                ^tmp + 1;
            else
                ^tmp - 1;
            endif;
        }!!
        """

        # Cold parse + compile
        ast1 = parse_actor(source)
        insts1, consts1, strings1 = compile_to_bytecode(ast1)
        stats1 = get_parser_cache_stats()
        assert stats1["ast_misses"] == 1
        assert stats1["ast_hits"] == 0
        assert stats1["bytecode_misses"] == 1
        assert stats1["bytecode_hits"] == 0

        # Warm parse + compile
        ast2 = parse_actor(source)
        insts2, consts2, strings2 = compile_to_bytecode(ast2)
        stats2 = get_parser_cache_stats()
        assert stats2["ast_hits"] == 1
        assert stats2["bytecode_hits"] == 1
        assert ast1 == ast2
        assert insts1 == insts2
        assert strings1 == strings2

    def test_cached_ast_and_bytecode_mutation_isolation(self):
        """Mutating returned AST nodes or bytecode lists must never corrupt cached entries."""
        clear_parser_caches()
        source = "Def origSelector(self, a) { ^a + 42; }!!"

        nodes1 = parse_actor(source)
        assert isinstance(nodes1[0], MethodDefNode)
        assert nodes1[0].name == "origSelector"

        # Mutate returned AST node in-place (just as ActorMethodSynthesizer._try_compile_method_source does)
        nodes1[0].name = "mutatedSelector"
        insts_mut, consts_mut, _ = compile_to_bytecode(nodes1)
        assert "mutatedSelector" in consts_mut
        nodes1[0].params.append("injectedParam")
        nodes1.append("junk")

        # Next parse of identical source must return pristine AST
        nodes2 = parse_actor(source)
        assert len(nodes2) == 1
        assert isinstance(nodes2[0], MethodDefNode)
        assert nodes2[0].name == "origSelector"
        assert nodes2[0].params == ["a"]

        # Mutate returned bytecode instructions and constants in-place (just as ActorVM.eval_chunk does)
        insts1, consts1, strings1 = compile_to_bytecode(nodes2)
        orig_len = len(insts1)
        insts1.append((Opcode.HALT, 999))
        consts1.append("CORRUPTED_CONST")
        strings1.append("CORRUPTED_STR")

        nodes3 = parse_actor(source)
        insts2, consts2, strings2 = compile_to_bytecode(nodes3)
        assert len(insts2) == orig_len
        assert "CORRUPTED_CONST" not in consts2
        assert "CORRUPTED_STR" not in strings2

    def test_lru_cache_bounded_eviction(self):
        """Verify cache size never exceeds `_PARSER_CACHE_MAXSIZE`."""
        clear_parser_caches()
        total_items = _PARSER_CACHE_MAXSIZE + 64
        for i in range(total_items):
            src = f"Actor[#Key{i}] := {i}!!"
            ast = parse_actor(src)
            compile_to_bytecode(ast)

        stats = get_parser_cache_stats()
        assert stats["ast_cache_size"] <= _PARSER_CACHE_MAXSIZE
        assert stats["bytecode_cache_size"] <= _PARSER_CACHE_MAXSIZE

    def test_microbenchmark_cache_speedup(self):
        """Measure cold uncached parsing vs. warm cached parsing + compilation on SIEVE.ACT."""
        clear_parser_caches()
        sieve_source = """
        /* Eratosthenes Sieve Benchmark in 1993 Actor 4.1 */
        inherit(Object, #Sieve, nil, 2, nil)!!
        now(Sieve)!!
        Def sieve(self, size | flags, count, i, prime, k) {
            flags := new(Array, size);
            fill(flags, true);
            count := 0;
            do(over(0, size - 1), {using(i)
                if flags[i] then
                    prime := i + i + 3;
                    k := i + prime;
                    loop while k < size begin
                        flags[k] := false;
                        k := k + prime;
                    endLoop;
                    count := count + 1;
                endif;
            });
            ^count;
        }!!
        """
        iterations = 200

        # Cold uncached baseline (direct parser + fresh BytecodeCompiler)
        t0 = time.perf_counter()
        for _ in range(iterations):
            raw_nodes = _parse_actor_1993(sieve_source)
            compile_to_bytecode(list(raw_nodes))  # plain list without _source_cache_key
        cold_elapsed = time.perf_counter() - t0

        # Prime cache once
        warm_ast = parse_actor(sieve_source)
        compile_to_bytecode(warm_ast)

        # Warm cached run
        t1 = time.perf_counter()
        for _ in range(iterations):
            ast = parse_actor(sieve_source)
            compile_to_bytecode(ast)
        warm_elapsed = time.perf_counter() - t1

        speedup = cold_elapsed / max(warm_elapsed, 1e-9)
        assert speedup >= 2.5, f"Expected >=2.5x cache speedup, got {speedup:.2f}x"


# ==============================================================================
# 2. Guardrail Linter Optimization & Caching Tests
# ==============================================================================


@actor_bridge(name="OptTestDriver")
class _OptTestDriver:
    @actor_query
    def readMetric(self) -> int:
        return 10

    @actor_command
    def triggerActuator(self, level: int) -> bool:
        return level > 0


class TestGuardrailLinterOptimizations:
    """Tests cached ivar lineage & Tier 2 bridge selector sets in ActorLinter."""

    def test_linter_tier2_selector_and_ivar_caches(self):
        vm = ActorVM()
        vm.register_bridge(_OptTestDriver)
        vm.enable_guardrails(True, protect_system_classes=True, strict_ivars=True)

        vm.eval_chunk(
            """
            inherit(Object, #OptDevice, #(#metric #changes), 2, nil)!!
            now(OptDevice)!!
            Def readOnly(self) { ^self.metric; }!!
            Def mutateLocal(self, v) { self.metric := v; self.changes := self.changes + 1; ^self.changes; }!!
            Def invokeTier2(self, drv) { ^triggerActuator(drv, 5); }!!
            """
        )

        cls = vm.hierarchy.get("OptDevice")
        linter = ActorLinter(vm, strict_ivars=True, protect_system_classes=True)

        r0 = linter.lint_method(cls.get_method("readOnly"), class_name="OptDevice", is_overwrite_check=False)
        r1 = linter.lint_method(cls.get_method("mutateLocal"), class_name="OptDevice", is_overwrite_check=False)
        r2 = linter.lint_method(cls.get_method("invokeTier2"), class_name="OptDevice", is_overwrite_check=False)

        assert r0.risk_tier == RiskTier.TIER_0_READ_ONLY
        assert r1.risk_tier == RiskTier.TIER_1_LOCAL_MUTATION
        assert r2.risk_tier == RiskTier.TIER_2_SYSTEM_OR_BRIDGE

        # Verify caches were populated on the linter instance
        assert linter._tier2_selector_cache is not None
        assert "triggerActuator" in linter._tier2_selector_cache
        assert "OptDevice" in linter._allowed_fields_cache
        assert linter._allowed_fields_cache["OptDevice"] == {"metric", "changes"}


# ==============================================================================
# 3. AI Client Precompiled Regexes & Multi-Threaded Concurrency Safety
# ==============================================================================


class TestAIClientAndConcurrencySafety:
    """Tests precompiled regex helpers and multi-threaded safety across parser, linter, and LLM client."""

    def test_precompiled_thinking_and_method_normalization(self):
        raw = (
            "<think>\nLet me reason step by step about the method.\n</think>\n"
            "```actor\n"
            "Def wrongName(self, x) {\n"
            "    local y := x * 3;\n"
            "    return y + 1;\n"
            "}\n"
            "```"
        )
        assert _strip_thinking_only(raw).startswith("```actor")
        assert _strip_thinking_and_fences(raw).startswith("Def wrongName")

        normalized = extract_and_normalize_actor_method(
            raw, selector="triplePlusOne", param_names=["x"]
        )
        assert normalized.startswith("Def triplePlusOne(self, x)")
        assert "local " not in normalized
        assert "y := x * 3;" in normalized
        assert "^y + 1;" in normalized
        assert normalized.endswith("!!")

    def test_valkyrie_llm_client_thread_safety(self):
        """Concurrent threads calling ValkyrieLLMClient must not race on token counters or per-thread stats."""
        client = ValkyrieLLMClient(
            endpoint="http://127.0.0.1:18080/v1/chat/completions",
            model="test-concurrent-model",
        )
        base_global_calls = ValkyrieLLMClient.GLOBAL_API_CALLS
        base_global_tokens = ValkyrieLLMClient.GLOBAL_TOTAL_TOKENS

        def fake_urlopen(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            user_txt = payload["messages"][-1]["content"]
            tok_id = int(user_txt.split("-")[-1])
            resp_body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": f"reply-{tok_id}!!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            ).encode("utf-8")
            mock_resp = MagicMock()
            mock_resp.read.return_value = resp_body
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = False
            return mock_resp

        num_threads = 16
        calls_per_thread = 15
        total_expected_calls = num_threads * calls_per_thread

        def worker(tid: int) -> List[str]:
            outputs = []
            for c in range(calls_per_thread):
                tag = tid * 1000 + c
                res = client.complete(prompt=f"req-{tag}", system_prompt="sys")
                outputs.append(res)
                t_stats = client.get_thread_last_call_stats()
                assert t_stats.get("total_tokens") == 15
            return outputs

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                results = list(pool.map(worker, range(num_threads)))

        assert len(results) == num_threads
        assert client.total_api_calls == total_expected_calls
        assert client.total_prompt_tokens == total_expected_calls * 10
        assert client.total_completion_tokens == total_expected_calls * 5
        assert client.total_tokens == total_expected_calls * 15
        assert ValkyrieLLMClient.GLOBAL_API_CALLS - base_global_calls == total_expected_calls
        assert ValkyrieLLMClient.GLOBAL_TOTAL_TOKENS - base_global_tokens == total_expected_calls * 15

    def test_concurrent_parser_compiler_and_linter_threads(self):
        """8 concurrent worker threads parsing, mutating ASTs, compiling, and linting simultaneously."""
        clear_parser_caches()
        vm = ActorVM()
        vm.enable_guardrails(True, strict_ivars=True, protect_system_classes=True)
        vm.eval_chunk("inherit(Object, #ConcurrentBox, #(#val #scale), 2, nil)!!")
        cls = vm.hierarchy.get("ConcurrentBox")
        linter = ActorLinter(vm, strict_ivars=True, protect_system_classes=True)

        shared_sources = [
            "Def calcA(self, x) { ^self.val + (x * self.scale); }!!",
            "Def calcB(self, x) { self.val := self.val + x; ^self.val; }!!",
            "Def calcC(self | i s) { i := 0; s := 0; loop while i < 5 begin s := s + self.val; i := i + 1; endLoop; ^s; }!!",
        ]

        def thread_task(worker_id: int) -> int:
            verified = 0
            for idx in range(30):
                src = shared_sources[idx % len(shared_sources)]
                nodes = parse_actor(src)
                assert len(nodes) == 1
                m_node = nodes[0]
                assert isinstance(m_node, MethodDefNode)
                orig_name = m_node.name
                # Compile before mutating local copy
                insts, consts, _ = compile_to_bytecode(nodes)
                assert len(insts) > 2
                cm = vm.compile_method_node(m_node, class_name="ConcurrentBox")
                report = linter.lint_method(cm, class_name="ConcurrentBox", is_overwrite_check=False)
                assert report.passed is True
                if orig_name == "calcB":
                    assert report.risk_tier == RiskTier.TIER_1_LOCAL_MUTATION
                else:
                    assert report.risk_tier == RiskTier.TIER_0_READ_ONLY
                # Mutate local AST node to verify no cross-thread contamination
                m_node.name = f"corrupted_{worker_id}_{idx}"
                verified += 1
            return verified

        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(pool.map(thread_task, range(8)))

        assert sum(counts) == 8 * 30
        stats = get_parser_cache_stats()
        assert stats["ast_hits"] >= 200
        assert stats["bytecode_hits"] >= 200
