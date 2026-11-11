"""Test Suite for the Metacircular Actor-Native AgentHarness (`agent_harness.act`) & TUI (`actorii.tui`).

Verifies:
1. Authentic 1993 Actor 4.1 `AgentHarness` bytecode compilation, `VMSandboxBridge` (`Actor[#Sandbox]`),
   and multi-step `runGoal` / `step` execution in the VM.
2. Self-correction across steps when Step 1 fails static linting or runtime execution (automatic
   transactional rollback + error feedback into Step 2).
3. 3-strike circuit breaker (`status == #blocked`) and `resetBreaker`.
4. Speculative copy-on-write shadow dry-runs (`dryRun` / `/shadow`) with opcode gas telemetry.
5. Metacircular hot-patching of `AgentHarness` methods mid-session without losing `scratchpad` state.
6. TUI slash commands (`/goal`, `/step`, `/shadow`, `/lint`, `/synth`, `/patch`, `/pool`, `/inspect`,
   `/classes`, `/scratch`, `/save`, `/load`, `/fileout`, `/status`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from actorii.core import Symbol
from actorii.harness import bootstrap_agent_harness
from actorii.tui import ActorTUISession, main as tui_main
from actorii.vm import ActorVM


class TestActorNativeAgentHarness:
    """Tests the 1993 Actor 4.1 `AgentHarness` class (`agent_harness.act`) inside `ActorVM`."""

    def test_bootstrap_and_single_step_and_dry_run_in_bytecode(self):
        vm, harness, sandbox = bootstrap_agent_harness()

        assert vm.eval_chunk("status(Actor[#Harness])!!") == Symbol("idle")
        assert vm.eval_chunk("stepCount(Actor[#Harness])!!") == 0

        # Speculative dry-run computes result and rolls back state
        shadow_out = vm.eval_chunk(
            'dryRun(Actor[#Harness], "Actor[#Ghost] := 123; Actor[#Ghost] * 2!!")!!'
        )
        assert str(shadow_out).startswith("SHADOW_OK")
        assert "246" in str(shadow_out)
        assert "Ghost" not in vm.global_vars["Actor"]

        # Static lint check from Actor bytecode
        lint_ok = vm.eval_chunk('lint(Actor[#Harness], "3 + 4 * 5!!")!!')
        assert lint_ok == "OK:TIER_0_READ_ONLY"

        lint_tier2 = vm.eval_chunk('lint(Actor[#Harness], "Actor[#X] := 10!!")!!')
        assert lint_tier2 == "OK:TIER_2_SYSTEM_OR_BRIDGE"

    def test_self_correcting_multi_step_run_goal_in_actor_bytecode(self):
        vm, harness, _ = bootstrap_agent_harness()

        prompts_seen: List[str] = []

        def scripted_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            prompts_seen.append(prompt)
            if len(prompts_seen) == 1:
                # Step 1: Intentionally references undeclared ivar `self.typoVal` -> triggers lint rollback!
                return (
                    "inherit(Object, #Calc, #(#val), 1, nil)!! "
                    "now(Calc)!! "
                    "Def badGet(self) { ^self.typoVal; }!!"
                )
            if len(prompts_seen) == 2:
                # Step 2: Sees LINT_UNDECLARED_IVAR in prompt, fixes class & stores answer in scratchpad
                assert "LINT_UNDECLARED_IVAR" in prompt
                return (
                    "inherit(Object, #Calc, #(#val), 1, nil)!! "
                    "now(Calc)!! "
                    "Def init(self, v) { self.val := v; ^self; }!! "
                    "Def tripled(self) { ^(self.val * 3); }!! "
                    "Actor[#C1] := init(new(Calc), 14)!! "
                    "storeScratch(Actor[#Harness], #tripledVal, tripled(Actor[#C1]))!!"
                )
            # Step 3: Sees #tripledVal == 42 in scratchpad and completes
            return "COMPLETE: Tripled value is 42"

        harness.custom_provider = scripted_brain

        final_answer = vm.eval_chunk(
            'runGoal(Actor[#Harness], "Create Calc(14) and store tripled value in #tripledVal", 5)!!'
        )
        assert final_answer == "Tripled value is 42"
        assert vm.eval_chunk("status(Actor[#Harness])!!") == Symbol("complete")
        assert vm.eval_chunk("stepCount(Actor[#Harness])!!") == 3
        assert vm.eval_chunk("readScratch(Actor[#Harness], #tripledVal)!!") == 42
        assert vm.eval_chunk("totalGasUsed(Actor[#Harness])!!") > 0

    def test_three_strike_circuit_breaker_and_reset_in_bytecode(self):
        vm, harness, _ = bootstrap_agent_harness()

        def always_broken_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            # Infinite loop that exceeds gasLimit (500 opcodes) on every attempt
            return "loop while true begin 1 + 1; endLoop!!"

        harness.custom_provider = always_broken_brain
        vm.eval_chunk("setGasLimit(Actor[#Harness], 400)!!")

        vm.eval_chunk('runGoal(Actor[#Harness], "Run broken infinite loop", 6)!!')
        assert vm.eval_chunk("status(Actor[#Harness])!!") == Symbol("blocked")
        assert vm.eval_chunk("failures(Actor[#Harness])!!") == 5
        assert vm.eval_chunk("stepCount(Actor[#Harness])!!") == 5
        assert "OpcodeBudgetExceededError" in vm.eval_chunk("lastError(Actor[#Harness])!!")

        # Resetting circuit breaker restores `#ready` and clears failures
        vm.eval_chunk("resetBreaker(Actor[#Harness])!!")
        assert vm.eval_chunk("status(Actor[#Harness])!!") == Symbol("ready")
        assert vm.eval_chunk("failures(Actor[#Harness])!!") == 0

    def test_metacircular_hot_patching_of_agent_harness_preserves_state(self):
        vm, harness, _ = bootstrap_agent_harness()

        vm.eval_chunk('storeScratch(Actor[#Harness], #sessionKey, "warm_state_123")!!')
        assert vm.eval_chunk("readScratch(Actor[#Harness], #sessionKey)!!") == "warm_state_123"

        # Hot-patch a new method onto `AgentHarness` while `Actor[#Harness]` is live in RAM
        vm.eval_chunk(
            """
            now(AgentHarness)!!
            Def inspectWarmState(self) {
                ^readScratch(self, #sessionKey) + "|steps=" + asString(self.stepCount);
            }!!
            """
        )

        patched_res = vm.eval_chunk("inspectWarmState(Actor[#Harness])!!")
        assert patched_res == "warm_state_123|steps=0"


class TestActorTUIWorkflow:
    """Tests the Hermes/Antigravity-style TUI session and slash commands."""

    def test_tui_slash_commands_and_cli_execution(self, tmp_path):
        session = ActorTUISession(use_color=False)

        # Deterministic brain for fast repeatable test
        def fast_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            return (
                "storeScratch(Actor[#Harness], #fib7, 13)!! "
                "COMPLETE: Stored fib7=13"
            )

        session.harness.custom_provider = fast_brain

        assert session.handle_input("/status") is True
        assert session.handle_input("/pool") is True
        assert session.handle_input("/classes") is True
        assert session.handle_input("/lint 10 + 20!!") is True
        assert session.handle_input("/shadow Actor[#ShadowVar] := 77; Actor[#ShadowVar] + 1!!") is True
        assert "ShadowVar" not in session.vm.global_vars["Actor"]

        # Run autonomous `/goal`
        assert session.handle_input("/goal Compute 7th Fibonacci number into #fib7") is True
        assert session.get_scratchpad_dict().get("fib7") == 13

        # Hot-patch a method on `AgentHarness` via `/patch`
        assert (
            session.handle_input(
                "/patch AgentHarness Def doubleFib(self) { ^(readScratch(self, #fib7) * 2); }!!"
            )
            is True
        )
        assert session.vm.eval_chunk("doubleFib(Actor[#Harness])!!") == 26

        # Save image & fileout class
        img_path = tmp_path / "tui_test.img"
        session.vm.fileout_dir = str(tmp_path)
        assert session.handle_input(f"/save {img_path}") is True
        assert img_path.is_file()

        assert session.handle_input("/fileout AgentHarness") is True
        assert (tmp_path / "AgentHarness.act").is_file()

        assert session.handle_input("/quit") is False

    def test_in_image_benchmark_suite_and_plain_english_routing(self):
        session = ActorTUISession(use_color=False)

        def bench_solver_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            if "GOAL: A live instance `Actor[#RoomStat]`" in prompt:
                return (
                    "now(BuggyThermostat)!! "
                    "Def needsCooling(self) { ^(self.currentTemp > self.targetTemp); }!! "
                    "storeScratch(Actor[#Harness], #coolingFixed, needsCooling(Actor[#RoomStat]))!! "
                    "COMPLETE: Fixed BuggyThermostat"
                )
            if "GOAL: Class `VaultBox`" in prompt:
                return (
                    "now(VaultBox)!! "
                    "Def unlockWith(self, attempt) { if attempt == self.pin then self.locked := false; endif; ^self.locked; }!! "
                    "unlockWith(Actor[#Vault], 7749)!! "
                    "storeScratch(Actor[#Harness], #vaultLocked, locked(Actor[#Vault]))!! "
                    "COMPLETE: Unlocked VaultBox"
                )
            return "storeScratch(Actor[#Harness], #plainEnglishActed, 99)!! COMPLETE: Acted on plain English"

        session.harness.custom_provider = bench_solver_brain

        # 1. Plain English input (no `/goal`) routes through `AgentHarness>>runGoal` and mutates scratchpad
        assert session.handle_input("Please set #plainEnglishActed to 99") is True
        assert session.get_scratchpad_dict().get("plainEnglishActed") == 99

        # 2. `/bench` commands execute in-image benchmark tasks and verify ground truth
        assert session.handle_input("/bench list") is True
        assert session.handle_input("/bench run bugfix_hotpatch") is True
        assert session.handle_input("/bench run vault_state") is True
        report = session.vm.eval_chunk("scoreReport(Actor[#Bench])!!")
        assert "2/2 PASSED" in report

        # 3. Bare `exit` (no slash) exits cleanly
        assert session.handle_input("exit") is False

    def test_tui_cli_entrypoint(self):
        rc = tui_main(["-c", "sieve(Actor[#Sam], 100)!!", "-c", "/status"])
        assert rc == 0

    def test_array_literals_with_negative_numbers_in_vm(self):
        vm, _, _ = bootstrap_agent_harness()
        arr = vm.eval_chunk("#(1, 2, -4, 5, -6.5)!!")
        assert arr == [1, 2, -4, 5, -6.5]

    def test_neo4j_actor_grammar_graph_and_tui_command(self):
        session = ActorTUISession(use_color=False)
        stats = session.vm.eval_chunk("graphStats(Actor[#Grammar])!!")
        assert "ActorGrammarGraph" in str(stats)
        diag = session.vm.eval_chunk(
            'diagnoseError(Actor[#Grammar], "Unit test failed on assert candidate(5, 3) == 7.5")!!'
        )
        assert "2.0" in str(diag)
        assert session.handle_input("/grammar loop") is True

    def test_digital_twins_and_tui_twins_command(self):
        session = ActorTUISession(use_color=False)
        valk_summary = session.vm.eval_chunk("summary(Actor[#Valkyrie])!!")
        assert "VRAM" in str(valk_summary)
        hue_lights = session.vm.eval_chunk("listLights(Actor[#Hue])!!")
        assert "Office" in str(hue_lights)
        hue_set = session.vm.eval_chunk('setLight(Actor[#Hue], "Office", "on", 90)!!')
        assert "90%" in str(hue_set)
        pantry_cnt = session.vm.eval_chunk("recipeCount(Actor[#Pantry])!!")
        assert isinstance(pantry_cnt, int) and pantry_cnt >= 0
        assert session.handle_input("/twins") is True

    def test_xml_tool_call_unwrap_and_thermo_ui_add_button(self):
        from actorii.harness import extract_actor_action

        xml_wrapped = (
            "<tool_call>\n<function=VMSandbox.execute\n<parameter=code>\n"
            'addButton(Actor[#ThermoUI], "← Back to Studio", "/")!!\n'
            "</parameter>\n</function>\n</tool_call>"
        )
        extracted = extract_actor_action(xml_wrapped)
        assert extracted == 'addButton(Actor[#ThermoUI], "← Back to Studio", "/")!!'

        vm, _, _ = bootstrap_agent_harness()
        vm.eval_chunk(
            'Actor[#Thermometer] := new(ThermoProTwin, "00:11:22:33:44:55", "TP358S")!!\n'
            'Actor[#ThermoUI] := new(ThermometerUI, Actor[#Thermometer], "ThermoPro UI")!!\n'
            'addButton(Actor[#ThermoUI], "← Back to Studio", "/")!!'
        )
        html = vm.eval_chunk("draw(Actor[#ThermoUI])!!")
        assert "← Back to Studio" in html
        assert 'data-href="/"' in html

    def test_conversational_prose_and_multi_turn_chat_memory_in_harness_and_tui(self):
        session = ActorTUISession(use_color=False)
        assert session.harness.name == "Bilby"
        assert session.vm.eval_chunk("Actor[#Bilby] == Actor[#Harness]!!") is True
        prompts_seen: List[str] = []

        def conversational_and_action_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            prompts_seen.append(prompt)
            # Turn 1: Pure conversational English reply WITHOUT "COMPLETE:" prefix
            if "GOAL: Hello, who are you?" in prompt:
                return "Hello! I am Bilby, a metacircular AI agent running inside the 1993 ActorII VM."
            # Turn 2: Action step 1 -> Step 2 sees STEPS EXECUTED IN CURRENT TURN SO FAR and CONVERSATION HISTORY
            if "GOAL: Store 21 in #baseVal" in prompt:
                if "STEPS EXECUTED IN CURRENT TURN SO FAR:" not in prompt:
                    assert "CONVERSATION HISTORY (RECENT TURNS):" in prompt
                    assert "Hello, who are you?" in prompt
                    return "storeScratch(Actor[#Harness], #baseVal, 21)!!"
                assert "storeScratch(Actor[#Harness], #baseVal, 21)!!" in prompt
                return "I have stored 21 in #baseVal."
            # Turn 3: Follow-up referencing prior turn without repeating instructions
            if "GOAL: Now double what you just stored into #doubledVal" in prompt:
                assert "Turn 2:" in prompt
                assert "storeScratch(Actor[#Harness], #baseVal, 21)!!" in prompt
                return (
                    "storeScratch(Actor[#Harness], #doubledVal, readScratch(Actor[#Harness], #baseVal) * 2)!!\n"
                    "COMPLETE: Doubled #baseVal (21) into #doubledVal (42)."
                )
            return "COMPLETE: Done"

        session.harness.custom_provider = conversational_and_action_brain

        # Turn 1: Conversational greeting without COMPLETE: completes in 1 step with 0 failures
        assert session.handle_input("Hello, who are you?") is True
        assert session.vm.eval_chunk("status(Actor[#Harness])!!") == Symbol("complete")
        assert session.vm.eval_chunk("failures(Actor[#Harness])!!") == 0
        assert session.vm.eval_chunk("stepCount(Actor[#Harness])!!") == 1

        # Turn 2: 2-step action + unprefixed English summary
        assert session.handle_input("/chat Store 21 in #baseVal") is True
        assert session.get_scratchpad_dict().get("baseVal") == 21
        assert session.vm.eval_chunk("stepCount(Actor[#Harness])!!") == 2

        # Turn 3: Follow-up using multi-turn conversation memory
        assert session.handle_input("/chat Now double what you just stored into #doubledVal") is True
        assert session.get_scratchpad_dict().get("doubledVal") == 42

        # Verify `/history` shows all 3 turns and `chatHistory(Actor[#Harness])!!` has 3 entries
        turns = session.vm.eval_chunk("chatHistory(Actor[#Harness])!!")
        assert len(turns) == 3
        assert turns[0]["user"] == "Hello, who are you?"
        assert turns[1]["actions"] == ["storeScratch(Actor[#Harness], #baseVal, 21)!!"]
        assert turns[2]["reply"] == "Doubled #baseVal (21) into #doubledVal (42)."

        assert session.handle_input("/history") is True

        # Verify `/clear` resets conversation turns while preserving VM scratchpad/pool
        assert session.handle_input("/clear") is True
        assert len(session.vm.eval_chunk("chatHistory(Actor[#Harness])!!")) == 0
        assert session.get_scratchpad_dict().get("doubledVal") == 42

        # Verify leading prose preamble stripping before Actor 4.1 code and backtick unwrapping
        from actorii.harness import _split_code_and_complete

        preamble_reply = (
            "The previous command failed because the CLI expects positional arguments, not --flag style.\n\n"
            'runShell(Actor[#Valkyrie], "valkyrie-rgb-ctl --help")!!'
        )
        code_part, comp_part = _split_code_and_complete(preamble_reply)
        assert code_part == 'runShell(Actor[#Valkyrie], "valkyrie-rgb-ctl --help")!!'
        assert comp_part is None

        backtick_comp, backtick_summary = _split_code_and_complete("`COMPLETE: Updated cpu_fan_ring to #FFD700`")
        assert backtick_comp == ""
        assert backtick_summary == "Updated cpu_fan_ring to #FFD700"

    def test_bilby_telegram_twin_and_gateway_roundtrip(self, tmp_path: Path):
        from actorii.telegram_gateway import BilbyTelegramGateway

        session = ActorTUISession(use_color=False)
        assert "TelegramBotTwin" in str(session.vm.eval_chunk("summary(Actor[#Telegram])!!"))

        def fake_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            if "GOAL: Compute 6 * 7 in #ans" in prompt:
                return "storeScratch(Actor[#Bilby], #ans, 6 * 7)!!\nCOMPLETE: Stored 42 in #ans."
            return "COMPLETE: Hello from Bilby on Telegram!"

        session.harness.custom_provider = fake_brain
        sent_payloads: List[dict] = []

        def fake_tg_api(token: str, method: str, payload: dict = None, timeout: float = 10.0) -> dict:
            if method == "sendMessage":
                sent_payloads.append(payload or {})
                return {"ok": True, "result": {"message_id": 100 + len(sent_payloads)}}
            if method == "sendChatAction":
                return {"ok": True}
            return {"ok": True, "result": []}

        gw = BilbyTelegramGateway(
            token="1000000001:test_token_placeholder",
            allowed_users=[100000001],
            default_chat_id="100000001",
            state_path=tmp_path / "tg_state.json",
            session=session,
            api_caller=fake_tg_api,
        )

        # Unauthorized user is ignored
        unauth = gw.process_update(
            {
                "update_id": 10,
                "message": {"message_id": 1, "from": {"id": 99999}, "chat": {"id": 99999}, "text": "hi"},
            }
        )
        assert unauth is None
        assert len(sent_payloads) == 0

        # Authorized user triggers Bilby ReAct loop + Actor 4.1 bytecode execution
        reply = gw.process_update(
            {
                "update_id": 11,
                "message": {
                    "message_id": 55,
                    "message_thread_id": 42,
                    "from": {"id": 100000001},
                    "chat": {"id": 100000001},
                    "text": "Compute 6 * 7 in #ans",
                },
            }
        )
        assert reply is not None
        assert "Stored 42 in #ans." in reply
        assert "storeScratch(Actor[#Bilby], #ans, 6 * 7)!!" in reply
        assert session.get_scratchpad_dict().get("ans") == 42
        assert len(sent_payloads) == 1
        assert sent_payloads[0]["chat_id"] == 100000001
        assert sent_payloads[0]["message_thread_id"] == 42
        assert gw.offset == 12

    def test_actor_native_web_scraper_failover_and_caching(self, monkeypatch):
        from actorii.harness.twins import _to_actor_dict

        vm, _, _ = bootstrap_agent_harness()
        summary = vm.eval_chunk("summary(Actor[#Web])!!")
        assert "WebScraper(firecrawl=https://api.firecrawl.dev" in summary

        # Verify WebScraper is a genuine compiled 1993 Actor 4.1 class whose methods are inspectable
        methods = vm.eval_chunk("methodsOf(Actor[#VM], #WebScraper)!!")
        assert "scrape" in methods
        assert "search" in methods
        assert "firecrawlScrape" in methods

        net_obj = vm.global_vars["Actor"]["Net"].target

        # Case 1: Firecrawl succeeds on Rung 1 -> cached in Actor Dictionary `pageCache`
        monkeypatch.setattr(
            net_obj,
            "firecrawlPostScrape",
            lambda base_url, target_url, max_chars=8000: _to_actor_dict(
                {
                    "ok": True,
                    "provider": Symbol("firecrawl"),
                    "url": target_url,
                    "title": "Whitewater Actor 4.1 Manual",
                    "content": "Welcome to 1993 Actor 4.1 Object-Oriented Programming.",
                    "error": "",
                }
            ),
        )
        scraped = vm.eval_chunk('scrape(Actor[#Web], "https://example.com/actor41")!!')
        assert "[WebScrape via #firecrawl] Whitewater Actor 4.1 Manual — https://example.com/actor41" in scraped
        assert vm.eval_chunk("lastProvider(Actor[#Web])!!") == Symbol("firecrawl")
        assert vm.eval_chunk("scrapeCount(Actor[#Web])!!") == 1
        assert vm.eval_chunk("cacheHits(Actor[#Web])!!") == 0

        # Calling `scrape` again hits the Actor 4.1 Dictionary `pageCache` without invoking transport
        cached_scraped = vm.eval_chunk('scrape(Actor[#Web], "https://example.com/actor41")!!')
        assert cached_scraped == scraped
        assert vm.eval_chunk("lastProvider(Actor[#Web])!!") == Symbol("cache")
        assert vm.eval_chunk("cacheHits(Actor[#Web])!!") == 1

        # Case 2: Firecrawl fails -> Actor 4.1 bytecode fails over to Rung 2 (`trafilatura`)
        monkeypatch.setattr(
            net_obj,
            "firecrawlPostScrape",
            lambda base_url, target_url, max_chars=8000: _to_actor_dict(
                {"ok": False, "provider": Symbol("firecrawl"), "url": target_url, "title": "", "content": "", "error": "HTTP 429"}
            ),
        )
        monkeypatch.setattr(
            net_obj,
            "trafilaturaExtract",
            lambda target_url, max_chars=8000: _to_actor_dict(
                {
                    "ok": True,
                    "provider": Symbol("trafilatura"),
                    "url": target_url,
                    "title": "Trafilatura Clean Page",
                    "content": "Extracted cleanly by local trafilatura.",
                    "error": "",
                }
            ),
        )
        fallback_scraped = vm.eval_chunk('scrape(Actor[#Web], "https://example.com/fallback")!!')
        assert "[WebScrape via #trafilatura]" in fallback_scraped
        assert "Extracted cleanly by local trafilatura." in fallback_scraped
        assert vm.eval_chunk("lastProvider(Actor[#Web])!!") == Symbol("trafilatura")

        # Case 3: Search fails over from Firecrawl to Keyless MCP Ring and caches in `searchCache`
        monkeypatch.setattr(
            net_obj,
            "firecrawlPostSearch",
            lambda base_url, query, limit=5, max_chars=6000: _to_actor_dict(
                {"ok": False, "provider": Symbol("firecrawl"), "query": query, "hits": [], "error": "No key"}
            ),
        )
        monkeypatch.setattr(
            net_obj,
            "keylessRingSearch",
            lambda query, limit=5, max_chars=6000: _to_actor_dict(
                {
                    "ok": True,
                    "provider": Symbol("keenable"),
                    "query": query,
                    "hits": [{"title": "Actor 4.1 Spec", "url": "https://example.org/spec", "snippet": "1993 Whitewater Group"}],
                    "error": "",
                }
            ),
        )
        search_out = vm.eval_chunk('search(Actor[#Web], "Whitewater Actor 4.1", 3)!!')
        assert "[WebSearch via #keenable]" in search_out
        assert "1. Actor 4.1 Spec (https://example.org/spec)" in search_out
        assert vm.eval_chunk("lastProvider(Actor[#Web])!!") == Symbol("keenable")
        assert vm.eval_chunk("searchCount(Actor[#Web])!!") == 1

    def test_oop_foundation_hierarchy_polymorphism_and_composition(self):
        vm, _, _ = bootstrap_agent_harness(memory_namespace="test_oop_kata")
        try:
            # 1. Verify foundational hierarchy is compiled in the VM
            assert vm.hierarchy.get("DomainEntity") is not None
            assert vm.hierarchy.get("GraphService") is not None
            assert vm.hierarchy.get("MetricEvaluator") is not None
            assert vm.hierarchy.get("CompositePipeline") is not None

            # 2. Subclass `MetricEvaluator` (which extends `GraphService` -> `Object`) with 2 polymorphic evaluators
            vm.eval_chunk(
                """
                inherit(MetricEvaluator, #UptimeEvaluator, #(#bonusPoints), 1, nil)!!
                now(UptimeEvaluator)!!
                Def initUptime(self, thresh, weight, bonus) {
                    initEvaluator(self, #uptime, thresh, weight);
                    self.bonusPoints := bonus;
                    ^self;
                }!!
                Def evaluateRow(self, row | up) {
                    up := at(row, #uptime);
                    if (up == nil) then ^0; endif;
                    if (up >= self.threshold) then ^(up + self.bonusPoints); endif;
                    ^0;
                }!!

                inherit(MetricEvaluator, #LatencyEvaluator, #(#penalty), 1, nil)!!
                now(LatencyEvaluator)!!
                Def initLatency(self, maxMs, weight, pen) {
                    initEvaluator(self, #latency, maxMs, weight);
                    self.penalty := pen;
                    ^self;
                }!!
                Def evaluateRow(self, row | ms) {
                    ms := at(row, #latency);
                    if (ms == nil) then ^0; endif;
                    if (ms <= self.threshold) then ^(100 - ms); endif;
                    ^(0 - self.penalty);
                }!!
                """
            )

            # 3. Compose both polymorphic evaluators inside a `CompositePipeline`
            res = vm.eval_chunk(
                """
                Actor[#UpEval] := initUptime(new(UptimeEvaluator), 99, 2, 10)!!
                Actor[#LatEval] := initLatency(new(LatencyEvaluator), 50, 1, 25)!!
                Actor[#SlaPipe] := initPipeline(new(CompositePipeline), "ClusterSLA")!!
                addStage(Actor[#SlaPipe], Actor[#UpEval])!!
                addStage(Actor[#SlaPipe], Actor[#LatEval])!!

                r1 := new(Dictionary, 8); atPut(r1, #uptime, 100); atPut(r1, #latency, 20);
                r2 := new(Dictionary, 8); atPut(r2, #uptime, 95);  atPut(r2, #latency, 80);
                rows := #(r1 r2);
                runEvaluators(Actor[#SlaPipe], rows)!!
                """
            )
            # r1 uptime: 100 >= 99 -> 100 + 10 = 110; r2 uptime: 95 < 99 -> 0. Uptime sum = 110
            # r1 latency: 20 <= 50 -> 100 - 20 = 80; r2 latency: 80 > 50 -> -25. Latency sum = 55
            # Total pipeline score = 110 + 55 = 165
            assert res == 165
            report = vm.eval_chunk("lastReport(Actor[#SlaPipe])!!")
            assert report.at(Symbol("uptime")) == 110
            assert report.at(Symbol("latency")) == 55
            assert report.at(Symbol("total")) == 165
        finally:
            try:
                vm.eval_chunk("clearEngrams(Actor[#Memory])!!")
            except Exception:
                pass

    def test_unlimited_retries_and_mid_loop_stop_breakout(self):
        session = ActorTUISession(use_color=False)
        assert session.harness._fields.get("maxSteps") == 0
        assert session.harness._fields.get("maxFailures") == 0
        assert "steps=0/∞" in session.status_ribbon_text()
        assert "strikes=0/∞" in session.status_ribbon_text()

        calls = {"n": 0}

        def resilient_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            calls["n"] += 1
            assert "(unlimited)" in prompt
            # Fail 8 consecutive times (> old 3 and 5 strike limits) with real runtime errors, then succeed on attempt 9!
            if calls["n"] <= 8:
                return f"({calls['n']} / 0)!!"
            return "COMPLETE: Recovered after 8 consecutive rollbacks."

        session.harness.custom_provider = resilient_brain
        session.handle_input("Recover from 8 errors")
        assert session.harness._fields.get("status") == Symbol("complete")
        assert session.harness._fields.get("stepCount") == 9
        assert "Recovered after 8 consecutive rollbacks" in str(session.harness._fields.get("lastResult"))

        # Verify mid-loop breakout (`/stop` setting status := #halted during invokeModel)
        loop_calls = {"n": 0}

        def infinite_brain(prompt: str, system_prompt: str = "", metadata: dict = None) -> str:
            loop_calls["n"] += 1
            if loop_calls["n"] >= 3:
                session.harness._fields["status"] = Symbol("halted")
            return "storeScratch(Actor[#Harness], #tick, 1)!!"

        session.harness.custom_provider = infinite_brain
        session.handle_input("Loop forever until halted")
        assert session.harness._fields.get("status") == Symbol("halted")
        assert loop_calls["n"] == 3

    def test_jetson_edge_twin_guardrails_and_pool_exposure(self):
        session = ActorTUISession(use_color=False)
        assert "Jetson" in session.get_pool_objects()
        blocked1 = session.vm.eval_chunk('exec(Actor[#Jetson], "sudo do-release-upgrade")!!')
        assert blocked1.at("ok") is False
        assert "JETSON_GUARDRAIL_BLOCKED" in str(blocked1.at("stderr"))
        blocked2 = session.vm.eval_chunk('exec(Actor[#Jetson], "cp foo /boot/initrd")!!')
        assert blocked2.at("ok") is False
        assert "JETSON_GUARDRAIL_BLOCKED" in str(blocked2.at("stderr"))









