"""Metacircular Actor-Native Agent Harness Subsystem (`actorii.harness`).

Loads `agent_harness.act` (written in authentic 1993 Whitewater Actor 4.1 syntax)
into the `ActorVM` bytecode image and bridges a capability-gated `VMSandboxBridge`
(`Actor[#Sandbox]`) so the Actor-native `AgentHarness` (`Actor[#Harness]`) can:
- Orchestrate multi-step ReAct loops (`step`, `runGoal`) in compiled VM bytecode.
- Run speculative copy-on-write dry-runs (`dryRun` -> `evalShadow`).
- Statically lint code chunks (`lint` -> `lintChunk`) into Tier 0 / Tier 1 / Tier 2.
- Enforce per-step opcode gas budgets (`gasLimit`) and a 3-strike circuit breaker.
- Self-correct across steps when a lint, gas, or runtime error rolls back a step.
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from actorii.ai import ValkyrieLLMClient
from actorii.bridge import actor_bridge, actor_command, actor_query
from actorii.classes.agent import Agent
from actorii.classes.collections import ActorOrderedCollection
from actorii.classes.concurrency import ActorThreadPoolTwin
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorClass, ActorObject, Symbol
from actorii.guardrails.fileout import ActorFileOutManager
from actorii.guardrails.linter import (
    PROTECTED_SYSTEM_CLASSES,
    RING1_MUTABLE_HARNESS_SELECTORS,
    ActorLinter,
    RiskTier,
)
from actorii.harness.benchmark import BENCHMARK_TASKS, ActorBenchSuite
from actorii.harness.grammar_graph import ActorGrammarGraph
from actorii.harness.host_bridge import HostWorkspaceBridge, VMReflectionBridge, is_host_lockdown
from actorii.harness.memory_graph import ActorMemoryGraph
from actorii.harness.telemetry import GLOBAL_TELEMETRY, summarize_actor_action_brief
from actorii.harness.twins import (
    JetsonEdgeTwin,
    NetTransportBridge,
    PDFDocumentTwin,
    PantryGraphTwin,
    ReverseEngineeringTwin,
    ReverseProxyTwin,
    SmartLightTwin,
    TelegramBotTwin,
    ValkyrieSystemTwin,
)
from actorii.parser import ClassDefNode, MethodDefNode, compile_to_bytecode, parse_actor
from actorii.vm import ActorVM, CompiledMethod, Opcode

HARNESS_ACT_PATH = Path(__file__).resolve().parent / "agent_harness.act"
WEB_SCRAPER_ACT_PATH = Path(__file__).resolve().parent / "web_scraper.act"
PRODUCT_LEARNER_ACT_PATH = Path(__file__).resolve().parent / "product_learner.act"
OOP_FOUNDATION_ACT_PATH = Path(__file__).resolve().parent / "oop_foundation.act"
DEFAULT_PERSONA_PATH = Path.home() / ".actorii" / "persona.md"

DEFAULT_HARNESS_SYSTEM_PROMPT = (
    "You are Bilby, a metacircular object-oriented AI Agent (`Actor[#Bilby]` / `Actor[#Harness]`) "
    "executing inside `ActorII` (a clean-room, Actor-inspired Linux bytecode VM & harness distinct from historical 1993 DOS/Win3.1 Whitewater Actor 4.1).\n"
    "CAPABILITIES & SHARED OBJECT POOL (`Actor[...]`):\n"
    "- OOP Base Classes (`inherit(SuperCls, #SubCls, #(#ivar), 1, nil)!! now(SubCls)!! Def m(self) { ... }!!`):\n"
    "  * `GraphService` (`initService(self, label)`, `queryRows(self, cypher)`, `queryRowsWith(self, cypher, p1)`, `writeGraph(self, cypher)`, `sumColumn`/`avgColumn`/`maxColumn(self, rows, #col)`)\n"
    "  * `MetricEvaluator` extends `GraphService` (`initEvaluator(self, name, thresh, weight)`, override `evaluateRow(self, row)`, `evaluateAll(self, rows)`, `weightedTotal(self, rows)`)\n"
    "  * `CompositePipeline` (`initPipeline(self, name)`, `addStage(self, evalObj)`, `runEvaluators(self, rows)`) & `DomainEntity` (`initEntity(self, id, name)`, `setMeta`/`getMeta`, `describe`)\n"
    "- Digital Twins & Bridges:\n"
    "  * `Actor[#Valkyrie]` (`#System`): `summary(Actor[#Valkyrie])!!`, `stackHealth(...)!!`, `gpuStatus(...)!!`, `vramFreeMiB(...)!!`, `serviceStatus(..., 8080)!!`, `askSkippy(..., questionOrImagePath)!!` (multimodal Cloud Skippy consult/vision!) (container-only when `isSandboxed`: `listDirectory`, `readTextFile`, `runShell`)\n"
    "  * `Actor[#Thermometer]`: `summary(Actor[#Thermometer])!!`, `temperatureF(...)!!`, `temperatureC(...)!!`, `humidity(...)!!`, `syncLive(..., 6)!!` | `Actor[#ThermoUI]`: `summary(...)!!`, `addButton(..., label, href)!!`, `draw(...)!!`\n"
    "  * `Actor[#Jetson]`: `summary(Actor[#Jetson])!!`, `telemetry(...)!!`, `discover(...)!!`, `scanLan(...)!!`, `setHost(..., ip)!!`, `cameraStatus(...)!!`, `exec(..., cmd, 20)!!`, `writeFile(..., \"/tmp/code.cu\", \"\"\"...\"\"\", 20)!!`, `writeAndRun(..., \"/tmp/code.cu\", \"\"\"...\"\"\", 45)!!` (supports `.sh`, `.py`, `.cu`, `.cpp`, `.c`!), `pullFile(..., remotePath, localName, 30)!!`\n"
    "  * `Actor[#Memory]`: `graphSchema(Actor[#Memory])!!`, `classHierarchyTree(...)!!`, `cypherQuery(..., q)!!`, `cypherQueryWith(..., q, p1)!!`, `cypherWrite(..., q)!!`, `recallEpisodes(..., q, 8)!!`, `persistEngram(..., #Cls, desc)!!`, `rollbackEngram(..., #Cls)!!`\n"
    "  * `Actor[#ProductLearner]`: `learnProduct(Actor[#ProductLearner], name, focus)!!`, `ingestManual(..., name, urlOrPath, focus)!!`, `recallProduct(..., name, topic)!!`, `listProducts(...)!!` (read/learn manuals before guessing unfamiliar drivers/CLI!)\n"
    "  * `Actor[#Web]`: `search(Actor[#Web], query, 5)!!`, `scrape(..., url)!!`, `extract(..., url)!!`, `cleanUrl(..., url)!!`, `renderBrowser(..., url)!!`\n"
    "  * `Actor[#PDF]`: `fetchUrl(Actor[#PDF], url)!!`, `info(..., p)!!`, `pageCount(..., p)!!`, `readPages(..., p, s, e, maxChars)!!`, `searchText(..., p, q, hits, maxChars)!!`\n"
    "  * `Actor[#Pantry]`: `recipeCount(...)!!`, `listRecipes(..., 5)!!`, `searchByIngredient(..., ing)!!` | `Actor[#Proxy]`: `status(...)!!`, `urls(...)!!`, `setMode(..., mode)!!`, `setAuth(..., bool)!!`, `rotateToken(...)!!`, `recentLogs(..., 10)!!`\n"
    "  * `Actor[#ReverseEngineer]`: `fileIdentity(..., p)!!`, `inspectEntropy(..., p)!!`, `staticAudit(..., p, fn)!!`, `decompileFunction(..., p, fn)!!`, `detonateSandbox(..., p, 10, name)!!`\n"
    "  * `Actor[#Telegram]`: `summary(...)!!`, `botInfo(...)!!`, `sendMessage(..., txt)!!`, `sendToChat(..., cid, txt, tid)!!`, `sendPhoto(..., localPath, caption)!!`, `sendDocument(..., localPath, caption)!!` | `Actor[#Bench]`: `scoreboard(Actor[#Bench])!!`, `listTasks(...)!!`, `setupTask(..., #id)!!`, `runTask(..., #id)!!`, `selfEvoSuites(...)!!`, `runSelfEvo(..., #specId, #scripted)!!`, `selfEvoScoreboard(...)!!`\n"
    "  * `Actor[#Threads]`: `spawn(Actor[#Threads], recv, #sel, argsArray)!!`, `spawnChunk(..., \"expr!!\", gas)!!`, `parallelAll(..., tasksArray, timeoutMs)!!`, `parallelSettle(..., tasksArray, timeoutMs)!!`, `parallelMap(..., itemsArray, recv, #sel)!!`, `newChannel(..., cap)!!`, `await(fut, timeoutMs)!!` (safe in lockdown for concurrent Actor 4.1 / Ring-1 twin calls)\n"
    "  * `Actor[#VM]` & `Actor[#Sandbox]`: `classes(Actor[#VM])!!`, `methodsOf(Actor[#VM], #Cls)!!`, `sourceOf(Actor[#VM], #Cls, #sel)!!`, `capabilities(Actor[#Sandbox])!!`\n"
    "  * `Actor[#Host]`: `searchFile`, `readFile`, `writeFile`, `editFile`, `execShell` (locked on host when `ACTORII_HOST_LOCKDOWN=1`; unlocked inside `/workspace` containers).\n"
    "- SECURITY & OUTPUT GUARDRAILS:\n"
    "  * Ring 0/1 classes, `AgentHarness` kernel methods, and `Actor[...]` singletons are locked. Never stop/disable OS/network/DNS/VPN/desktop services, touch `/etc/resolv.conf` or crontabs, or modify `~/bin/` or `ActorII/src/`.\n"
    "  * Casual questions: reply in 1 step (`COMPLETE: <answer>`). Structured domain facts/formulas: persist to Neo4j (`cypherWrite`) + define an Actor class (auto-crystallized as `:ActorEngram`).\n"
    "  * NEVER emit `<tool_call>` XML wrappers or wrap Actor code in markdown fences—output raw 1993 Actor 4.1 ending in `!!` or `COMPLETE: <answer>`."
)


def load_harness_system_prompt(persona_path: Optional[Path | str] = None) -> str:
    """Build the active harness system prompt, appending an optional local host persona overlay."""
    if os.environ.get("ACTORII_DISABLE_PERSONA_OVERLAY") == "1":
        return DEFAULT_HARNESS_SYSTEM_PROMPT
    target = (
        Path(persona_path).expanduser()
        if persona_path is not None
        else Path(os.environ.get("ACTORII_PERSONA_FILE", str(DEFAULT_PERSONA_PATH))).expanduser()
    )
    try:
        if target.is_file():
            overlay = target.read_text(encoding="utf-8", errors="replace").strip()
            if overlay:
                return f"{DEFAULT_HARNESS_SYSTEM_PROMPT}\nHOST ENVIRONMENT & PERSONA OVERLAY:\n{overlay}"
    except Exception:
        pass
    return DEFAULT_HARNESS_SYSTEM_PROMPT


HARNESS_SYSTEM_PROMPT = load_harness_system_prompt()

_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_NON_ACTOR_FENCE_LANGS = {
    "bash", "sh", "zsh", "shell", "console", "text", "txt", "json", "yaml", "yml",
    "toml", "ini", "xml", "html", "css", "sql", "cypher", "diff", "md", "markdown",
    "python", "py", "rust", "rs", "go", "cpp", "c", "java", "js", "javascript", "ts", "typescript",
}
_XML_PARAM_RE = re.compile(r"<parameter=(?:code|action|chunk|source)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
_XML_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_FUNC_NAME_RE = re.compile(r"<function=([A-Za-z0-9_.-]+)>?", re.IGNORECASE)
_XML_KV_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|(?=</function>)|(?=</tool_call>)|$)", re.DOTALL | re.IGNORECASE)


def _transpile_xml_host_tool_call(text: str) -> Optional[str]:
    """Transpile Qwen `<function=writeFile|execShell|readFile|editFile>` XML tool calls into 1993 Actor 4.1 `Actor[#Host]` expressions."""
    fn_m = _XML_FUNC_NAME_RE.search(text)
    if not fn_m:
        return None
    fn_name = fn_m.group(1).strip().split(".")[-1].lower()
    params = {k.lower(): v for k, v in _XML_KV_PARAM_RE.findall(text)}
    if not params:
        return None

    if fn_name in ("writefile", "write_file", "create_file"):
        path = (params.get("path") or params.get("file_path") or params.get("filepath") or "").strip()
        content = params.get("content") or params.get("text") or ""
        if path:
            return f"writeFile(Actor[#Host], {json.dumps(path)}, {json.dumps(content)})!!"
    elif fn_name in ("execshell", "exec_shell", "bash", "run_command", "shell", "terminal", "execute_command"):
        cmd = (params.get("command") or params.get("cmd") or "").strip()
        t_sec = int(params.get("timeout") or params.get("timeout_sec") or params.get("timeoutsec") or 20)
        if cmd:
            return f"execShell(Actor[#Host], {json.dumps(cmd)}, {t_sec})!!"
    elif fn_name in ("readfile", "read_file"):
        path = (params.get("path") or params.get("file_path") or "").strip()
        s_ln = int(params.get("startline") or params.get("start_line") or 1)
        e_ln = int(params.get("endline") or params.get("end_line") or 120)
        if path:
            return f"readFile(Actor[#Host], {json.dumps(path)}, {s_ln}, {e_ln})!!"
    elif fn_name in ("editfile", "edit_file", "replace_in_file"):
        path = (params.get("path") or params.get("file_path") or "").strip()
        old_t = params.get("oldtext") or params.get("old_text") or params.get("old_str") or ""
        new_t = params.get("newtext") or params.get("new_text") or params.get("new_str") or ""
        if path and old_t:
            return f"editFile(Actor[#Host], {json.dumps(path)}, {json.dumps(old_t)}, {json.dumps(new_t)})!!"
    return None


def _unwrap_xml_tool_call(text: str) -> str:
    """Strip Qwen/Hermes `<tool_call>` / `<parameter=code>` XML wrappers if emitted by the model, rejecting non-Actor GUI/OS tool calls."""
    has_xml_tool = bool(
        _XML_TOOL_RE.search(text)
        or "<tool_call" in text.lower()
        or "<function=" in text.lower()
    )
    pm = _XML_PARAM_RE.search(text)
    if pm:
        candidate = pm.group(1).strip()
        if _is_likely_actor_code(candidate):
            return candidate
        return (
            '__HALLUCINATED_XML_TOOL_CALL__("External XML tool calls are forbidden; output raw 1993 Actor 4.1 code ending in !!")!!'
        )
    if has_xml_tool:
        transpiled = _transpile_xml_host_tool_call(text)
        if transpiled:
            return transpiled
    tm = _XML_TOOL_RE.search(text)
    if tm:
        inner = tm.group(1).strip()
        lines = [
            ln
            for ln in inner.splitlines()
            if not re.match(r"^\s*</?(?:function|parameter|tool_call)\b", ln.strip(), flags=re.IGNORECASE)
        ]
        candidate = "\n".join(lines).strip()
        if _is_likely_actor_code(candidate):
            return candidate
        return (
            '__HALLUCINATED_XML_TOOL_CALL__("External XML tool calls are forbidden; output raw 1993 Actor 4.1 code ending in !!")!!'
        )
    if has_xml_tool:
        lines = [
            ln
            for ln in text.splitlines()
            if not re.match(r"^\s*</?(?:function|parameter|tool_call)\b", ln.strip(), flags=re.IGNORECASE)
        ]
        candidate = "\n".join(lines).strip()
        if _is_likely_actor_code(candidate):
            return candidate
        return (
            '__HALLUCINATED_XML_TOOL_CALL__("External XML tool calls are forbidden; output raw 1993 Actor 4.1 code ending in !!")!!'
        )
    return text


_ACTOR_CODE_MARKER_RE = re.compile(
    r"(?:\binherit\s*\(|\bnow\s*\(|\bDef\s+[A-Za-z_]|\bAction\s+[A-Za-z_]|:=|Actor\s*\[|#\s*\(|\bstoreScratch\s*\(|\breadScratch\s*\(|\bloop\b|\bendLoop\b|\bendif\b|\bnew\s*\(|\^[A-Za-z0-9_(\[\"#])"
)
_ACTOR_CALL_RE = re.compile(r"^[A-Za-z_]\w*\s*\(.*\)\s*;?\s*!*$", flags=re.DOTALL)


def _is_likely_actor_code(text: str) -> bool:
    """Return True if `text` is 1993 Actor 4.1 code (including code with syntax/lint errors) rather than conversational prose."""
    s = (text or "").strip()
    if not s:
        return False
    if _ACTOR_CODE_MARKER_RE.search(s):
        return True
    if _ACTOR_CALL_RE.match(s):
        return True
    if s.endswith("!!"):
        body = s[:-2].strip()
        if any(ch in body for ch in (";", "(", ")", "[", "]", "+", "-", "*", "/", "==")):
            return True
        try:
            nodes = parse_actor(s)
            return bool(nodes)
        except Exception:
            return False
    return False


_CODE_START_LINE_RE = re.compile(
    r"^\s*(?:`|(?:inherit|now|Def|Action|if|loop|Actor)\b|[A-Za-z_]\w*\s*(?:\(|:=|\[)|/\*|\^|#(?:\(|[A-Za-z_])|\(|\[|\d|\"|-\d)"
)


def _strip_leading_prose_before_code(text: str) -> str:
    """If an LLM response contains conversational preamble or trailing prose around Actor 4.1 code, keep only the code."""
    s = (text or "").strip()
    if "\n" not in s:
        return s
    candidate_full = s if s.endswith("!!") else (s.rstrip("!") + "!!")
    try:
        if parse_actor(candidate_full):
            return s
    except Exception:
        pass

    lines = s.splitlines()
    for i in range(1, len(lines)):
        first_ln = lines[i].strip()
        if not first_ln or first_ln.endswith(":") or not _CODE_START_LINE_RE.match(first_ln):
            continue
        sub = "\n".join(lines[i:]).strip()
        if len(sub) >= 2 and sub.startswith("`") and not sub.startswith("```"):
            if sub.endswith("`") and "`" not in sub[1:-1]:
                sub = sub[1:-1].strip()
            elif sub.endswith("`!!") and "`" not in sub[1:-3]:
                sub = sub[1:-3].strip().rstrip("!") + "!!"
        if not _is_likely_actor_code(sub):
            continue
        cand = sub if sub.endswith("!!") else (sub.rstrip("!") + "!!")
        try:
            if parse_actor(cand):
                return sub
        except Exception:
            continue

    # If there is trailing prose after `!!`, try prefixes ending at each `!!`
    if "!!" in s:
        bang_positions = [m.end() for m in re.finditer(r"!!", s)]
        for end_pos in reversed(bang_positions):
            prefix = s[:end_pos].strip()
            p_lines = prefix.splitlines()
            for i in range(len(p_lines)):
                first_ln = p_lines[i].strip()
                if not first_ln or first_ln.endswith(":") or not _CODE_START_LINE_RE.match(first_ln):
                    continue
                sub = "\n".join(p_lines[i:]).strip()
                if not sub or not _is_likely_actor_code(sub):
                    continue
                cand = sub if sub.endswith("!!") else (sub.rstrip("!") + "!!")
                try:
                    if parse_actor(cand):
                        return cand
                except Exception:
                    continue
    return s


def _clean_complete_summary(summary_raw: str) -> str:
    """Clean trailing Actor `/* comments */` or stray trailing `!!` code outside Markdown fences in a `COMPLETE:` reply."""
    s = (summary_raw or "").strip()
    if s.endswith("`") and not s.endswith("```"):
        s = s[:-1].strip()
    # Do not strip `!` if reply ends with a closing Markdown fence ```
    if not s.endswith("```"):
        # If the model appended a standalone Actor `/* note */` or `storeScratch(...)!!` on the final line outside fences, trim it
        fence_count = s.count("```")
        if fence_count % 2 == 0 and "\n" in s:
            lines = s.splitlines()
            while lines:
                tail = lines[-1].strip()
                if tail.startswith("/*") and tail.endswith("*/"):
                    lines.pop()
                    continue
                if tail.endswith("!!") and _is_likely_actor_code(tail):
                    lines.pop()
                    continue
                break
            s = "\n".join(lines).strip()
        if s.endswith("!!"):
            s = s[:-2].strip()
    return s


def _split_code_and_complete(raw_text: str) -> Tuple[str, Optional[str]]:
    """Separate executable Actor 4.1 code chunks from any trailing `COMPLETE:` or conversational summary."""
    text = _unwrap_xml_tool_call((raw_text or "").strip())
    if not text:
        return "", None

    # Strip outer single inline backticks if the LLM wrapped the entire line/response in `...`
    if len(text) >= 2 and text.startswith("`") and not text.startswith("```"):
        if text.endswith("`") and "`" not in text[1:-1]:
            text = text[1:-1].strip()
        elif text.endswith("`!!") and "`" not in text[1:-3]:
            text = (text[1:-3].strip().rstrip("!") + "!!")

    # 1. Check leading `COMPLETE:` FIRST so conversational Markdown code blocks (` ```bash `, ` ```python `)
    # inside human-facing answers are NEVER hijacked and executed as Actor 4.1 VM bytecode!
    lead_comp = re.match(r"^\s*`?\s*COMPLETE\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if lead_comp:
        return "", _clean_complete_summary(lead_comp.group(1))

    # 2. Check fenced blocks ONLY when tagged as Actor/untagged AND genuinely valid Actor 4.1 code
    m = _CODE_FENCE_RE.search(text)
    if m:
        fence_lang = (m.group(1) or "").strip().lower()
        fenced = (m.group(2) or "").strip()
        outside = (text[: m.start()] + "\n" + text[m.end() :]).strip()
        comp_match = re.search(r"^\s*`?\s*COMPLETE\s*:\s*(.+?)\s*`?\s*$", outside, flags=re.IGNORECASE | re.MULTILINE)
        is_actor_fence = (
            fence_lang not in _NON_ACTOR_FENCE_LANGS
            and _is_likely_actor_code(fenced)
            and (
                fence_lang in ("actor", "act", "smalltalk")
                or "!!" in fenced
                or "Actor[" in fenced
                or ":=" in fenced
                or bool(re.search(r"\b(?:inherit|now|Def|storeScratch|readScratch)\b", fenced))
            )
        )
        if comp_match and is_actor_fence:
            code_part = fenced if fenced.endswith("!!") else f"{fenced.rstrip('!')}!!"
            return code_part, _clean_complete_summary(comp_match.group(1))
        if is_actor_fence and not outside:
            text = fenced
        elif fence_lang in _NON_ACTOR_FENCE_LANGS:
            # Conversational answer containing a bash/python/json tutorial block without leading `COMPLETE:`
            if comp_match:
                return "", _clean_complete_summary(text)
            if "!!" not in outside or not _is_likely_actor_code(outside):
                return "", text.strip()

    # 3. Check if there is a trailing COMPLETE: directive after Actor code (either on a new line or after `!!`)
    comp_idx = re.search(r"(?:^|!!)\s*`?\s*COMPLETE\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if comp_idx:
        before = _strip_leading_prose_before_code(text[: comp_idx.start()].strip())
        summary = _clean_complete_summary(comp_idx.group(1))
        if before and _is_likely_actor_code(before):
            if not before.endswith("!!"):
                before = before.rstrip("!") + "!!"
            return before, summary
        return "", summary

    text = _strip_leading_prose_before_code(text)

    # Conversational prose detection: if the model replied in natural language without Actor code constructs,
    # treat it directly as a conversational completion unless it is mid-step planning prose ("Let me check...").
    if not _is_likely_actor_code(text):
        if re.search(
            r"\b(?:let me|let's check|i'll check|i will check|let me read|let me search|let's inspect|next step|will run|now let's|now i will)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return (
                '__INCOMPLETE_REASONING_STEP__("Output executable Actor 4.1 code ending in !! instead of reasoning prose")!!',
                None,
            )
        return "", text.strip()

    if not text.endswith("!!"):
        text = text.rstrip("!") + "!!"
    return text, None


def extract_actor_action(raw_text: str) -> str:
    """Extract clean 1993 Actor code or a `COMPLETE:` directive from an LLM response."""
    code_part, comp_summary = _split_code_and_complete(raw_text)
    if code_part and comp_summary:
        return f"{code_part}\nCOMPLETE: {comp_summary}"
    if comp_summary is not None and not code_part:
        return f"COMPLETE: {comp_summary}"
    return code_part


@actor_bridge(name="LLMBridge")
class LLMBridge:
    """First-class raw LLM completion primitive exposed in Actor bytecode as `Actor[#LLM]`."""

    def __init__(self, vm: ActorVM) -> None:
        self.vm = vm

    @actor_query
    def countTokens(self, text_or_messages: Any) -> int:
        """Estimate token count for a prompt string or message collection (Tier 0 `@actor_query`)."""
        if isinstance(text_or_messages, (list, tuple, ActorOrderedCollection)):
            total_chars = 0
            msg_count = 0
            for item in text_or_messages:
                msg_count += 1
                if isinstance(item, dict):
                    total_chars += len(str(item.get("content", ""))) + len(str(item.get("role", "")))
                elif isinstance(item, Dictionary):
                    c = item._entries.get(Symbol("content")) or item._entries.get("content") or ""
                    r = item._entries.get(Symbol("role")) or item._entries.get("role") or ""
                    total_chars += len(str(c)) + len(str(r))
                else:
                    total_chars += len(str(item))
            return max(1, (total_chars + 3) // 4 + msg_count * 4)
        s = str(text_or_messages or "")
        return max(1, (len(s) + 3) // 4) if s else 0

    def _build_append_only_messages(
        self,
        harness_obj: Agent,
        sandbox: "VMSandboxBridge",
    ) -> List[Dict[str, str]]:
        """Build a strictly append-only multi-message chat array so hybrid Gated-DeltaNet / KV-cache slots hit `f_keep=1.0` across steps."""
        goal = str(harness_obj._fields.get("goal", "") or "")
        step_num = int(harness_obj._fields.get("stepCount", 1) or 1)
        raw_max_steps = harness_obj._fields.get("maxSteps")
        max_steps = 0 if raw_max_steps is None else int(raw_max_steps)
        max_steps_lbl = f"{max_steps}" if max_steps > 0 else "(unlimited)"
        step_log_obj = harness_obj._fields.get("stepLog")
        chat_hist = sandbox.getChatHistory(harness_obj)
        turn_key = (id(step_log_obj), goal, len(chat_hist))

        cache: Optional[Dict[str, Any]] = getattr(harness_obj, "_kv_turn_cache", None)
        if not isinstance(cache, dict) or cache.get("key") != turn_key:
            # 1. Static system message (100% goal-invariant across ALL turns & steps so KV prefix stays locked!)
            grammar_obj = getattr(sandbox, "grammar_graph", None)
            core_grammar = (
                grammar_obj.compactCoreReference()
                if grammar_obj is not None and hasattr(grammar_obj, "compactCoreReference")
                else ""
            )
            sys_parts = [
                harness_obj.buildContextPrompt(),
                sandbox.defaultInstructions(),
            ]
            if core_grammar:
                sys_parts.append(core_grammar)
            sys_msg = {"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}

            # 2. Append-Only Epoch History Window:
            # Instead of a sliding `chat_hist[-6:]` window (which shifts message #1 right after `sys_msg` on EVERY turn
            # and invalidates cross-turn KV cache), anchor `anchor_idx` and append turns monotonically up to 14 turns
            # before rolling the epoch forward to `len(chat_hist) - 4`!
            epoch_turns: List[Any] = []
            if chat_hist:
                anchor = getattr(harness_obj, "_kv_history_anchor", None)
                if (
                    isinstance(anchor, tuple)
                    and len(anchor) == 2
                    and 0 <= anchor[1] < len(chat_hist)
                    and id(chat_hist[anchor[1]]) == anchor[0]
                    and (len(chat_hist) - anchor[1]) <= 14
                ):
                    anchor_idx = anchor[1]
                else:
                    anchor_idx = max(0, len(chat_hist) - 4)
                    setattr(harness_obj, "_kv_history_anchor", (id(chat_hist[anchor_idx]), anchor_idx))
                epoch_turns = chat_hist[anchor_idx:]

            prior_msgs: List[Dict[str, str]] = []
            for t in epoch_turns:
                if not isinstance(t, dict):
                    continue
                u_txt = str(t.get("user") or "").strip()
                r_txt = str(t.get("reply") or "").strip()
                if len(r_txt) > 1600:
                    r_txt = r_txt[:1597] + "..."
                acts = t.get("actions") or []
                act_note = ""
                if acts:
                    briefs = [summarize_actor_action_brief(str(a), max_len=64) for a in acts[:4]]
                    extra = f" (+{len(acts) - 4} more)" if len(acts) > 4 else ""
                    act_note = f"/* Executed {len(acts)} step(s): {'; '.join(briefs)}{extra} */\n"
                if u_txt:
                    prior_msgs.append({"role": "user", "content": u_txt})
                    prior_msgs.append({"role": "assistant", "content": f"{act_note}COMPLETE: {r_txt}"})

            # 3. Initial user message for this turn (built ONCE at Step 1 and NEVER mutated on Steps 2..N!)
            mem_obj = getattr(sandbox, "memory_graph", None)
            hydrated = mem_obj.hydrateContext(goal) if mem_obj is not None else ""
            task_grammar = (
                grammar_obj.queryTaskSpecificConstructs(goal)
                if grammar_obj is not None and hasattr(grammar_obj, "queryTaskSpecificConstructs")
                else ""
            )
            policy_txt = ""
            try:
                p_val = self.vm.eval_chunk("policyNote(Actor[#Harness])!!")
                policy_txt = str(p_val or "").strip()
            except Exception:
                policy_txt = ""

            init_lines: List[str] = []
            if policy_txt:
                init_lines.append(f"[HARNESS RUNTIME POLICY OVERRIDE]:\n{policy_txt}")
            if hydrated:
                init_lines.append(hydrated)
            if task_grammar:
                init_lines.append(task_grammar)
            init_lines.append(f"LIVE POOL SUMMARY: {sandbox.poolSummary()}")
            init_lines.append(f"SCRATCHPAD (`Actor[#Harness].scratchpad`): {sandbox.formatScratchpad(harness_obj)}")
            init_lines.append(f"GOAL: {goal}")
            init_lines.append(
                f"STEP: 1 of {max_steps_lbl}\n"
                "Output ONLY a single raw 1993 Actor 4.1 code chunk ending in `!!` or `COMPLETE: <your answer>`."
            )
            turn_user_msg = {"role": "user", "content": "\n\n".join(init_lines)}

            cache = {
                "key": turn_key,
                "base_messages": [sys_msg, *prior_msgs, turn_user_msg],
                "step_pairs": [],  # List[Tuple[int, Dict[str, str], Dict[str, str]]]
            }
            setattr(harness_obj, "_kv_turn_cache", cache)

        messages: List[Dict[str, str]] = list(cache["base_messages"])
        step_entries = list(step_log_obj) if isinstance(step_log_obj, (ActorOrderedCollection, list)) else []
        cached_pairs: List[Tuple[int, Dict[str, str], Dict[str, str]]] = cache["step_pairs"]

        # Append any newly executed steps without ever mutating earlier cached step pairs!
        start_idx = len(cached_pairs)
        for idx in range(start_idx, len(step_entries)):
            entry = step_entries[idx]
            if not isinstance(entry, dict):
                continue
            st_n = int(entry.get("step", idx + 1) or (idx + 1))
            st_status = str(entry.get("status", "ok"))
            raw_act = str(entry.get("action") or "").strip()
            if "__HALLUCINATED_XML_TOOL_CALL__" in raw_act:
                asst_act = "<invalid XML tool call>"
            elif len(raw_act) <= 12000:
                asst_act = raw_act
            else:
                asst_act = sandbox._compact_step_action_for_history(raw_act)
            asst_msg = {"role": "assistant", "content": asst_act or "nil!!"}

            cur_scratch = sandbox.formatScratchpad(harness_obj)
            next_st = st_n + 1
            if st_status == "error":
                err_txt = str(entry.get("error") or "")
                diag_txt = sandbox.formatErrorDiagnostic(harness_obj) if idx == len(step_entries) - 1 else f"Error: {err_txt}"
                obs_body = (
                    f"[Step {st_n} ROLLED BACK]\n{diag_txt}\n"
                    f"SCRATCHPAD (`Actor[#Harness].scratchpad`): {cur_scratch}\n"
                    f"STEP: {next_st} of {max_steps_lbl}\n"
                    "Self-correct with valid 1993 Actor 4.1 code ending in `!!` or reply `COMPLETE: <clear explanation>`."
                )
            else:
                res_repr = repr(entry.get("result"))
                if len(res_repr) > 3500:
                    res_repr = res_repr[:3500] + "...'"
                unhandled_hint = entry.get("unhandled_selector")
                hint_line = (
                    f"\n[Warning: `{unhandled_hint}` is not a defined method on that receiver — call `methodsOf(Actor[#VM], #ClassName)!!` to inspect real selectors!]"
                    if unhandled_hint
                    else ""
                )
                nudge_line = sandbox._step_loop_nudge(step_entries[: idx + 1])
                obs_body = (
                    f"[Step {st_n} OK | gas={entry.get('gas_used', 0)} ops]\n"
                    f"LAST RESULT: {res_repr}{hint_line}{nudge_line}\n"
                    f"SCRATCHPAD (`Actor[#Harness].scratchpad`): {cur_scratch}\n"
                    f"STEP: {next_st} of {max_steps_lbl}\n"
                    "Execute the next 1993 Actor 4.1 expression ending in `!!`, or if you have enough information to answer the user, reply `COMPLETE: <detailed answer>`."
                )
            obs_msg = {"role": "user", "content": obs_body}
            cached_pairs.append((st_n, asst_msg, obs_msg))

        for _, a_m, u_m in cached_pairs[-20:]:
            messages.append(a_m)
            messages.append(u_m)

        return messages

    @actor_query
    def complete(self, prompt_or_messages: Any, options_or_harness: Any = None) -> str:
        """Invoke the active LLM directly from Actor bytecode (`complete(Actor[#LLM], promptText, self)`)."""
        allow_shadow = (
            options_or_harness == Symbol("allowShadow")
            or options_or_harness == "allowShadow"
        )
        if getattr(self.vm, "_in_shadow_run", False) and not allow_shadow:
            return "COMPLETE: Shadow dry-run stub"

        harness_obj = (
            options_or_harness
            if isinstance(options_or_harness, Agent)
            else self.vm.global_vars.get("Actor", {}).get("Harness")
        )

        if isinstance(prompt_or_messages, (list, tuple, ActorOrderedCollection)):
            parts = []
            for item in prompt_or_messages:
                if isinstance(item, dict):
                    parts.append(f"{item.get('role', 'user')}: {item.get('content', '')}")
                elif isinstance(item, Dictionary):
                    role = item._entries.get(Symbol("role")) or item._entries.get("role") or "user"
                    content = item._entries.get(Symbol("content")) or item._entries.get("content") or ""
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(item))
            prompt_str = "\n".join(parts)
        else:
            prompt_str = str(prompt_or_messages or "")

        if isinstance(harness_obj, Agent):
            custom_prov = getattr(harness_obj, "custom_provider", None) or getattr(self.vm, "ai_provider", None)
            sandbox_ref = self.vm.global_vars.get("Actor", {}).get("Sandbox")
            sandbox_py = getattr(sandbox_ref, "_target", sandbox_ref)
            use_append_only_http = (
                custom_prov is None
                and isinstance(options_or_harness, Agent)
                and getattr(harness_obj, "class_name", "") == "AgentHarness"
                and isinstance(sandbox_py, VMSandboxBridge)
                and "GOAL: " in prompt_str
            )

            if use_append_only_http:
                step_num = int(harness_obj._fields.get("stepCount", 1) or 1)
                turn_id = str(getattr(harness_obj, "_active_turn_id", "") or "live-turn")
                messages = self._build_append_only_messages(harness_obj, sandbox_py)
                client = ValkyrieLLMClient(
                    endpoint=harness_obj.endpoint,
                    model=harness_obj.model,
                )
                try:
                    raw_reply = client.complete_messages(messages, temperature=0.2)
                    cleaned_check = _strip_leading_prose_before_code(
                        _unwrap_xml_tool_call(str(raw_reply or "").strip())
                    )
                    is_incomplete_thought = (
                        bool(cleaned_check)
                        and len(cleaned_check) < 350
                        and not re.search(r"\bCOMPLETE\s*:", cleaned_check, flags=re.IGNORECASE)
                        and not _is_likely_actor_code(cleaned_check)
                        and bool(
                            re.search(
                                r"\b(?:let me|let's|i will|i'll|i need to)\b",
                                cleaned_check,
                                flags=re.IGNORECASE,
                            )
                        )
                    )
                    if is_incomplete_thought:
                        cont_msgs = [
                            *messages,
                            {"role": "assistant", "content": str(raw_reply).strip()},
                            {
                                "role": "user",
                                "content": (
                                    "Now output ONLY the single executable 1993 Actor 4.1 expression ending in `!!` "
                                    "or `COMPLETE: <final summary>` if the task is already finished. Do NOT output prose:"
                                ),
                            },
                        ]
                        raw_reply = client.complete_messages(cont_msgs, temperature=0.2)
                    stats = client.last_call_stats or {}
                    GLOBAL_TELEMETRY.log_llm_call(
                        turn_id=turn_id,
                        step=step_num,
                        endpoint=str(stats.get("endpoint") or client.endpoint),
                        model=str(harness_obj.model),
                        message_count=int(stats.get("message_count") or len(messages)),
                        prompt_tokens=int(stats.get("prompt_tokens") or 0),
                        completion_tokens=int(stats.get("completion_tokens") or 0),
                        latency_ms=float(stats.get("latency_ms") or 0.0),
                        raw_reply=str(raw_reply or ""),
                        extracted_action=extract_actor_action(str(raw_reply or "")),
                    )
                except Exception as exc:
                    mem_keys = sorted(str(k).lstrip("#") for k in harness_obj.memory()._entries.keys())
                    raw_reply = (
                        f"[{harness_obj.name} ({harness_obj.class_name})] Processed: {prompt_str} "
                        f"(memory_keys={mem_keys}, error={exc})"
                    )
                harness_obj.last_response = str(raw_reply)
            else:
                saved_hist = list(harness_obj._history) if hasattr(harness_obj, "_history") else []
                harness_obj.clearHistory()
                try:
                    raw_reply = harness_obj.ask(prompt_str)
                    cleaned_check = _strip_leading_prose_before_code(
                        _unwrap_xml_tool_call(str(raw_reply or "").strip())
                    )
                    is_incomplete_thought = (
                        bool(cleaned_check)
                        and len(cleaned_check) < 350
                        and not str(raw_reply).startswith(f"[{harness_obj.name} ({harness_obj.class_name})] Processed:")
                        and not re.search(r"\bCOMPLETE\s*:", cleaned_check, flags=re.IGNORECASE)
                        and not _is_likely_actor_code(cleaned_check)
                        and bool(
                            re.search(
                                r"\b(?:let me|let's|i will|i'll|i need to)\b",
                                cleaned_check,
                                flags=re.IGNORECASE,
                            )
                        )
                    )
                    if is_incomplete_thought:
                        harness_obj.clearHistory()
                        followup_prompt = (
                            f"{prompt_str}\n\n"
                            f"[Your Reasoning]: {str(raw_reply).strip()}\n\n"
                            "Now output ONLY the single executable 1993 Actor 4.1 expression ending in `!!` "
                            "(e.g. `readFile(Actor[#Host], \"/workspace/...\", 1, 100)!!` or `editFile(...)!!`) "
                            "or `COMPLETE: <final summary>` if the task is already finished. Do NOT output prose:"
                        )
                        raw_reply = harness_obj.ask(followup_prompt)
                finally:
                    if hasattr(harness_obj, "_history") and isinstance(harness_obj._history, (ActorOrderedCollection, list)):
                        harness_obj._history.clear()
                        harness_obj._history.extend(saved_hist)

            if raw_reply.startswith(f"[{harness_obj.name} ({harness_obj.class_name})] Processed:"):
                goal = str(harness_obj._fields.get("goal", "") or "")
                step_count = int(harness_obj._fields.get("stepCount", 1) or 1)
                scratch = harness_obj._fields.get("scratchpad")
                has_res = False
                if isinstance(scratch, Dictionary):
                    has_res = Symbol("result") in scratch._entries or "result" in scratch._entries
                if step_count == 1 and not has_res:
                    return f"storeScratch(Actor[#Harness], #goalAck, {goal!r})!!"
                return f"COMPLETE: Goal processed ({goal})"
            return str(raw_reply)

        if self.vm.ai_provider is not None:
            return str(self.vm.ai_provider(prompt_str, HARNESS_SYSTEM_PROMPT, {"model": self.vm.ai_model}))
        return "COMPLETE: No active LLM provider"


@actor_bridge(name="VMSandbox")
class VMSandboxBridge:
    """Capability-gated reflective VM controller exposed to Actor code as `Actor[#Sandbox]`."""

    def __init__(
        self,
        vm: ActorVM,
        grammar_graph: Optional[ActorGrammarGraph] = None,
        memory_graph: Optional[ActorMemoryGraph] = None,
    ) -> None:
        self.vm = vm
        self.grammar_graph = grammar_graph
        self.memory_graph = memory_graph
        self.step_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    # =========================================================================
    # Tier 0: Read-Only Queries & Speculative Dry-Runs (`@actor_query`)
    # =========================================================================

    @actor_query
    def poolSummary(self) -> str:
        """Return a structured summary of live objects and user classes in `Actor[...]`."""
        actor_dict = self.vm.global_vars.get("Actor", {})
        builtin_classes = set(PROTECTED_SYSTEM_CLASSES) | {
            "Agent",
            "AgentHarness",
            "Float",
            "Int",
            "KeyedCollection",
            "String",
            "ThermoProTwin",
            "ThermometerUI",
        }
        builtin_singletons = {
            "Sandbox",
            "Grammar",
            "Memory",
            "LLM",
            "VM",
            "Host",
            "Bench",
            "Hue",
            "Pantry",
            "System",
            "Valkyrie",
            "Harness",
            "Bilby",
            "PDF",
            "Net",
            "Web",
            "ProductLearner",
            "Jetson",
            "Proxy",
            "Telegram",
            "ReverseEngineer",
            "REA",
        }
        objs: List[str] = []
        bridges: List[str] = []
        classes: List[str] = []
        if isinstance(actor_dict, dict):
            seen_keys = set()
            for k, v in actor_dict.items():
                norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                if norm_k in seen_keys:
                    continue
                seen_keys.add(norm_k)
                if isinstance(v, ActorClass):
                    if norm_k not in builtin_classes and not getattr(v, "is_bridge_class", False):
                        methods = sorted(v._methods.keys())
                        ivars = getattr(v, "ivars", [])
                        classes.append(f"{norm_k}(ivars={ivars}, methods={methods})")
                elif not isinstance(v, type):
                    cls_name = str(getattr(v, "class_name", type(v).__name__))
                    if norm_k in builtin_singletons:
                        bridges.append(f"#{norm_k}")
                    else:
                        passive_fn = getattr(v, "_passive_summary", None)
                        val_str = passive_fn() if callable(passive_fn) else repr(v)
                        objs.append(f"Actor[#{norm_k}]: {cls_name} = {val_str[:96]}")
        summary_parts = []
        if bridges:
            summary_parts.append("Bridges: " + ", ".join(sorted(bridges)))
        if objs:
            summary_parts.append("Objects: " + "; ".join(sorted(objs)))
        if classes:
            summary_parts.append("Classes: " + "; ".join(sorted(classes)))
        ws_dir = Path("/workspace")
        if ws_dir.is_dir():
            try:
                entries = sorted(p for p in ws_dir.iterdir() if not p.name.startswith("."))
                if 0 < len(entries) <= 10:
                    ws_items: List[str] = []
                    for p in entries:
                        if p.is_dir():
                            ws_items.append(f"{p.name}/")
                        elif p.is_file() and p.stat().st_size <= 250_000:
                            try:
                                lc = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
                                ws_items.append(f"{p.name} ({lc}L)")
                            except Exception:
                                ws_items.append(p.name)
                        else:
                            ws_items.append(p.name)
                    summary_parts.append("Workspace(/workspace): " + ", ".join(ws_items))
            except Exception:
                pass
        return " | ".join(summary_parts) if summary_parts else "Pool: empty"

    @actor_query
    def capabilities(self) -> str:
        """Return a structured 4-Tier Capability vs. Containment Matrix and live configuration status."""
        import os as _os
        import shutil as _shutil

        actor_dict = self.vm.global_vars.get("Actor", {})
        harness_obj = actor_dict.get("Harness")
        model = getattr(harness_obj, "model", "qwen3.8:latest")
        endpoint = getattr(harness_obj, "endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        provider = _os.environ.get("ACTORII_LLM_PROVIDER", "").strip() or (
            "ollama" if ":11434" in str(endpoint) else ("llamacpp" if ":8080" in str(endpoint) else "openai-compatible")
        )
        has_api_key = bool(
            _os.environ.get("ACTORII_LLM_API_KEY")
            or _os.environ.get("OPENAI_API_KEY")
            or _os.environ.get("OPENROUTER_API_KEY")
            or _os.environ.get("GEMINI_API_KEY")
        )
        in_docker = _os.environ.get("ACTORII_IN_DOCKER") == "1" or Path("/.dockerenv").exists()
        lockdown = _os.environ.get("ACTORII_HOST_LOCKDOWN") == "1"
        no_evolve = _os.environ.get("ACTORII_DISABLE_SELF_EVOLVE") == "1"
        ns = self.memory_graph.namespace if self.memory_graph is not None else "disabled"
        neo_url = self.memory_graph.tx_url if self.memory_graph is not None else _os.environ.get("ACTORII_NEO4J_HTTP_URL", "offline")

        pdf_obj = actor_dict.get("PDF")
        pdf_target = getattr(pdf_obj, "_target", pdf_obj)
        pdf_stat = pdf_target.summary() if pdf_target and hasattr(pdf_target, "summary") else "unknown"
        net_obj = actor_dict.get("Net")
        net_target = getattr(net_obj, "_target", net_obj)
        net_stat = net_target.summary() if net_target and hasattr(net_target, "summary") else "unknown"
        has_nvidia = bool(_shutil.which("nvidia-smi"))
        ws_exists = Path("/workspace").is_dir()

        lines = [
            "=== Bilby (ActorII v4.1) Capability vs. Containment Matrix ===",
            f"[Runtime Mode]: {'Docker Container' if in_docker else 'Host Native'} | ACTORII_HOST_LOCKDOWN={'1 (LOCKED — Default Defense-Up)' if lockdown else '0 (UNLOCKED)'}",
            "",
            "Tier 0 — Pure In-VM & Memory Graph (Safe in Lockdown, 0 Host Access):",
            f"  • LLM Provider (`Actor[#Harness]`) : READY (provider={provider}, model={model}, endpoint={endpoint}, api_key={'SET' if has_api_key else 'none'})",
            "    ↳ Configure: `ACTORII_LLM_PROVIDER`, `ACTORII_LLM_ENDPOINT`, `ACTORII_LLM_MODEL`, `ACTORII_LLM_API_KEY` or `/model <name> <url>`",
            f"  • Neo4j Memory (`Actor[#Memory]`)  : {'ENABLED' if self.memory_graph else 'DISABLED'} (namespace={ns}, url={neo_url})",
            "    ↳ Configure: `ACTORII_NEO4J_HTTP_URL`, `ACTORII_NEO4J_USER`, `ACTORII_NEO4J_PASSWORD`, `ACTORII_MEMORY_NAMESPACE`",
            f"  • Vector Recall (`:8084`/Ollama)   : model={_os.environ.get('ACTORII_EMBED_MODEL', 'bge-small-en-v1.5')} @ {_os.environ.get('ACTORII_EMBED_ENDPOINT', 'http://127.0.0.1:8084/v1/embeddings')}",
            "    ↳ Configure: `ACTORII_EMBED_ENDPOINT`, `ACTORII_EMBED_MODEL`, `ACTORII_EMBED_API_KEY`",
            "",
            "Tier 1 — Read-Only Document & Web Research (Safe in Lockdown, Container-Baked):",
            f"  • PDF Reader (`Actor[#PDF]`)       : {pdf_stat}",
            "    ↳ Host/Container Req: `pypdf` (pre-installed in container) + outbound HTTPS or local `/workspace`/`/tmp` PDF file",
            f"  • Web & Search (`Actor[#Web]`)     : {net_stat}",
            "    ↳ Host/Container Req: Outbound HTTPS (Keenable/Exa/DDGS + `trafilatura` work keyless; set `FIRECRAWL_API_KEY` for Firecrawl API)",
            "",
            "Tier 2 — Scoped Workspace File I/O & Ring-1 Evolution:",
            f"  • Workspace I/O (`Actor[#Host]`)   : {'SCOPED (/tmp & /workspace only)' if lockdown else 'UNLOCKED (project & workspace roots)'} (/workspace mounted={ws_exists})",
            "    ↳ To enable host folder access: mount `-v ./workspace:/workspace` in Docker; `src/actorii` & `~/.ssh` remain hard-blocked",
            f"  • Ring-1 Self-Evolve (`evolve`)    : {'LOCKED (ACTORII_DISABLE_SELF_EVOLVE=1)' if no_evolve else 'ENABLED (Cognitive hooks only; Kernel & Tier-2 calls blocked)'}",
            "    ↳ Configure: `ACTORII_DISABLE_SELF_EVOLVE=1` to freeze cognitive hook evolution",
            "",
            "Tier 3 — Privileged Execution & Hardware Telemetry (Blocked on Host in Lockdown):",
            f"  • Shell Exec (`execShell`)         : {'CONTAINER-SANDBOXED (Allowed inside isolated Docker)' if in_docker else ('BLOCKED (ACTORII_HOST_LOCKDOWN=1)' if lockdown else 'ENABLED ON HOST (48 destructive patterns blocked)')}",
            "    ↳ To enable safely: run inside Docker (`ACTORII_IN_DOCKER=1`) OR set `ACTORII_HOST_LOCKDOWN=0` on a dedicated sandbox host",
            f"  • GPU Telemetry (`Actor[#Valkyrie]`): {'READY (nvidia-smi detected)' if has_nvidia else 'OFFLINE (No GPU passthrough)'}",
            "    ↳ To enable in Docker: pass `--gpus all` or `deploy.resources.reservations.devices` in `docker-compose.yml`",
        ]
        return "\n".join(lines)

    @actor_query
    def lintChunk(self, source: str) -> str:
        """Statistically analyze an Actor code chunk and return its RiskTier or lint errors."""
        chunk_src = str(source or "").strip()
        if not chunk_src:
            return "ERROR:Empty source chunk"
        if "__HALLUCINATED_XML_TOOL_CALL__" in chunk_src:
            return (
                "ERROR:[XML_TOOL_FORBIDDEN] External XML tool calls are forbidden in ActorII! "
                "Output raw 1993 Actor 4.1 code ending in `!!`, e.g. "
                "`searchFile(Actor[#Host], \"/workspace\", \"pattern\", 15)!!`, "
                "`readFile(Actor[#Host], \"/workspace/file.py\", 1, 100)!!`, or "
                "`editFile(Actor[#Host], \"/workspace/file.py\", \"old\", \"new\")!!`."
            )
        if "__INCOMPLETE_REASONING_STEP__" in chunk_src:
            return (
                "ERROR:[REASONING_ONLY] You output reasoning prose ('Let me check...') without an executable "
                "1993 Actor 4.1 expression ending in `!!`. Output ONLY the Actor 4.1 code call ending in `!!`."
            )
        if "__PREMATURE_COMPLETE_ON_FAIL__" in chunk_src:
            return (
                "ERROR:[TESTS_STILL_FAILING] Cannot mark task COMPLETE while unit tests or syntax checks are failing! "
                "Read `/workspace/INSTRUCTIONS.md` or the failing test with `readFile(Actor[#Host], \"/workspace/...\", 1, 100)!!` "
                "and fix the code with `writeFile` or `editFile` until `[AUTO-TEST PASS]`."
            )
        if not chunk_src.endswith("!!"):
            chunk_src = chunk_src.rstrip("!") + "!!"

        linter = ActorLinter(
            self.vm,
            strict_ivars=self.vm.strict_ivars,
            protect_system_classes=self.vm.protect_system_classes,
        )
        max_tier = RiskTier.TIER_0_READ_ONLY
        errors: List[str] = []
        active_cls = self.vm._current_compilation_class or "Object"
        pending_ivars: Dict[str, List[str]] = {}

        try:
            for raw_piece in self.vm._split_chunks_safely(chunk_src):
                nodes = parse_actor(raw_piece)
                stmt_nodes = []
                for node in nodes:
                    if isinstance(node, ClassDefNode):
                        active_cls = node.name
                        pending_ivars[node.name] = [
                            str(iv).lstrip("#") for iv in getattr(node, "ivars", [])
                        ]
                        stub = ActorObject()
                        stub._fields = {iv: None for iv in pending_ivars[node.name]}
                        for m in getattr(node, "methods", []):
                            cm = self.vm.compile_method_node(m, class_name=node.name)
                            rep = linter.lint_method(cm, class_name=node.name, receiver=stub)
                            if rep.risk_tier > max_tier:
                                max_tier = rep.risk_tier
                            errors.extend(f"[{e.code}] {e.message}" for e in rep.errors)
                    elif hasattr(node, "target_class"):
                        active_cls = node.target_class
                    elif isinstance(node, MethodDefNode):
                        t_cls = getattr(node, "class_name", "")
                        if (not t_cls or t_cls == "Object") and active_cls:
                            t_cls = active_cls
                        if not t_cls:
                            t_cls = "Object"
                        stub = None
                        if t_cls in pending_ivars:
                            stub = ActorObject()
                            stub._fields = {iv: None for iv in pending_ivars[t_cls]}
                        cm = self.vm.compile_method_node(node, class_name=t_cls, raw_chunk=raw_piece)
                        rep = linter.lint_method(cm, class_name=t_cls, receiver=stub)
                        if rep.risk_tier > max_tier:
                            max_tier = rep.risk_tier
                        errors.extend(f"[{e.code}] {e.message}" for e in rep.errors)
                    else:
                        stmt_nodes.append(node)

                if stmt_nodes:
                    insts, consts, strings = compile_to_bytecode(stmt_nodes)
                    temp_cm = CompiledMethod(
                        name="<chunk>",
                        class_name="<Chunk>",
                        params=[],
                        locals=[],
                        instructions=list(insts),
                        constants=list(consts),
                        string_table=list(strings),
                    )
                    rep = linter.lint_method(temp_cm, class_name="<Chunk>", is_overwrite_check=False)
                    if rep.risk_tier > max_tier:
                        max_tier = rep.risk_tier
                    errors.extend(f"[{e.code}] {e.message}" for e in rep.errors)
        except Exception as exc:
            return f"ERROR:ParseError: {exc}"

        if errors:
            return f"ERROR:{'; '.join(errors)}"
        return f"OK:{max_tier.name}"

    @actor_query
    def evalShadow(self, source: str, max_opcodes: int = 25000) -> str:
        """Execute `source` in copy-on-write speculative shadow mode (`shadow_only=True`)."""
        chunk_src = str(source or "").strip()
        if not chunk_src.endswith("!!"):
            chunk_src = chunk_src.rstrip("!") + "!!"
        try:
            res = self.vm.eval_transactional(
                chunk_src,
                max_opcodes=int(max_opcodes or 25000),
                shadow_only=True,
            )
            gas = getattr(self.vm, "last_eval_opcodes", 0)
            return f"SHADOW_OK(gas={gas}): {res}"
        except Exception as exc:
            gas = getattr(self.vm, "last_eval_opcodes", 0)
            return f"SHADOW_ERR(gas={gas}): {type(exc).__name__}: {exc}"

    @actor_query
    def getChatHistory(self, harness_obj: Any = None) -> List[Dict[str, Any]]:
        """Return the multi-turn episodic conversation log for `Actor[#Harness]`."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        if target is None or not hasattr(target, "_fields"):
            return []
        chat_log = target._fields.get("chatLog")
        if isinstance(chat_log, ActorOrderedCollection):
            return list(chat_log)
        if isinstance(chat_log, list):
            return list(chat_log)
        return []

    @actor_command
    def clearChatHistory(self, harness_obj: Any = None) -> str:
        """Clear multi-turn conversation history on `Actor[#Harness]` while preserving VM pool state."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        if target is not None and hasattr(target, "_fields"):
            target._fields["chatLog"] = ActorOrderedCollection()
            if hasattr(target, "clearHistory"):
                target.clearHistory()
        return "CLEARED"

    @actor_command
    def compactChatHistory(self, harness_obj: Any = None, keep_turns: int = 3) -> int:
        """Compact older conversation turns in `chatLog` into a single summary turn, retaining `keep_turns` verbatim."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        if target is None or not hasattr(target, "_fields"):
            return 0
        chat_log = target._fields.get("chatLog")
        if not isinstance(chat_log, (ActorOrderedCollection, list)):
            return 0
        keep_n = max(1, int(keep_turns or 3))
        if len(chat_log) <= keep_n:
            return len(chat_log)

        all_items = list(chat_log)
        older = all_items[:-keep_n]
        recent = all_items[-keep_n:]

        user_ledger_lines: List[str] = []
        episode_records: List[Tuple[str, str, str, str, str]] = []
        for idx_o, t in enumerate(older):
            if isinstance(t, dict):
                u_full = str(t.get("user", "")).strip()
                r_full = str(t.get("reply", "")).strip()
                if u_full == "[COMPACTED PRIOR TURNS]":
                    if r_full:
                        user_ledger_lines.append(r_full)
                    continue
                t_num = t.get("turn", idx_o + 1)
                t_date = str(t.get("session_date") or t.get("date") or "").strip()
                t_sid = str(t.get("session_id") or f"turn_{t_num}").strip()
                date_tag = f"[{t_date}] " if t_date else ""
                u_clean = " ".join(u_full.split())
                r_clean = " ".join(r_full.split())
                user_ledger_lines.append(f"{date_tag}T{t_num} ({t_sid}): {u_clean[:220]}")
                ep_content = f"{date_tag}Turn {t_num} — User: {u_full[:1100]} | Assistant: {r_full[:550]}"
                ep_embed_text = f"User: {u_full[:750]} | Assistant: {r_full[:200]}"
                episode_records.append((t_sid, t_date, ep_content, f"T{t_num}", ep_embed_text))

        mem_obj = self.memory_graph
        if mem_obj is not None and getattr(mem_obj, "enable_engrams", False) and hasattr(mem_obj, "recordEpisode"):
            for ep_sid, ep_date, ep_body, ep_trng, ep_emb in episode_records:
                try:
                    mem_obj.recordEpisode(ep_sid, ep_date, ep_body, ep_trng, ep_emb)
                except Exception:
                    pass

        ledger_text = " | ".join(user_ledger_lines)
        if len(ledger_text) > 3600:
            ledger_text = "... " + ledger_text[-3600:]
        summary_entry = {
            "turn": 1,
            "user": "[COMPACTED PRIOR TURNS]",
            "reply": ledger_text,
            "actions": [],
            "steps": len(older),
            "gas_used": 0,
            "last_result": "",
            "timestamp": time.time(),
        }
        new_log = ActorOrderedCollection([summary_entry] + recent)
        target._fields["chatLog"] = new_log
        return len(new_log)

    def _build_error_context_dict(
        self,
        error_kind: str,
        message: str,
        frame_info: Optional[Dict[str, Any]] = None,
    ) -> Dictionary:
        """Construct a structured Actor `Dictionary` for `harness_obj.lastErrorContext`."""
        info = frame_info or {}
        d = Dictionary(16)
        d.atPut(Symbol("errorKind"), Symbol(error_kind))
        d.atPut("errorKind", Symbol(error_kind))
        d.atPut(Symbol("message"), str(message))
        d.atPut("message", str(message))
        f_cls = str(info.get("failedClass") or "<Chunk>")
        f_sel = str(info.get("failedSelector") or "<chunk>")
        f_off = int(info.get("bytecodeOffset") or 0)
        d.atPut(Symbol("failedClass"), Symbol(f_cls.lstrip("#")))
        d.atPut("failedClass", Symbol(f_cls.lstrip("#")))
        d.atPut(Symbol("failedSelector"), Symbol(f_sel.lstrip("#")))
        d.atPut("failedSelector", Symbol(f_sel.lstrip("#")))
        d.atPut(Symbol("bytecodeOffset"), f_off)
        d.atPut("bytecodeOffset", f_off)

        locals_dict = Dictionary(16)
        for lk, lv in (info.get("localBindings") or {}).items():
            locals_dict.atPut(Symbol(str(lk)), str(lv))
            locals_dict.atPut(str(lk), str(lv))
        d.atPut(Symbol("localBindings"), locals_dict)
        d.atPut("localBindings", locals_dict)
        return d

    def _record_completed_turn(
        self,
        harness_obj: Any,
        summary: str,
        last_exec_result: Any = None,
    ) -> Dict[str, Any]:
        """Record a completed multi-turn exchange into `chatLog` and `Agent._history`."""
        chat_log = harness_obj._fields.get("chatLog")
        if not isinstance(chat_log, ActorOrderedCollection):
            chat_log = ActorOrderedCollection()
            harness_obj._fields["chatLog"] = chat_log

        goal = str(harness_obj._fields.get("goal", "") or "")
        step_log = harness_obj._fields.get("stepLog")
        steps_list = list(step_log) if isinstance(step_log, ActorOrderedCollection) else []
        executed_actions = [
            str(e.get("action", "")).strip()
            for e in steps_list
            if isinstance(e, dict)
            and e.get("status") == "ok"
            and not str(e.get("action", "")).upper().startswith("COMPLETE:")
        ]
        turn_gas = sum(
            int(e.get("gas_used", 0) or 0)
            for e in steps_list
            if isinstance(e, dict)
        )
        turn_entry = {
            "turn": len(chat_log) + 1,
            "user": goal,
            "reply": str(summary),
            "actions": executed_actions,
            "steps": int(harness_obj._fields.get("stepCount", 1) or 1),
            "gas_used": turn_gas,
            "last_result": repr(last_exec_result) if last_exec_result is not None else "",
            "timestamp": time.time(),
        }
        chat_log.add(turn_entry)

        # Persist completed turn to Neo4j `:ActorEpisode` so multi-session recall & audit stay current
        import os as _os

        is_live_session = (
            _os.environ.get("ACTORII_BENCH_STRICT") != "1"
            and "PYTEST_CURRENT_TEST" not in _os.environ
        )
        if (
            is_live_session
            and self.memory_graph is not None
            and self.memory_graph.enable_engrams
            and getattr(self.memory_graph, "namespace", "default") == "default"
        ):
            try:
                import datetime as _dt

                s_date = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                t_num = turn_entry["turn"]
                acts_str = f" [Actions: {'; '.join(executed_actions)}]" if executed_actions else ""
                ep_content = f"User: {goal}\nBilby{acts_str}: {summary}"
                self.memory_graph.recordEpisode(
                    session_id=f"live_{getattr(harness_obj, 'name', 'Bilby').lower()}",
                    session_date=s_date,
                    content=ep_content[:2000],
                    turn_range=f"Turn {t_num}",
                )
            except Exception:
                pass

        # Sync clean compact turn pairs into `Agent._history` so `historyCount(Actor[#Harness])` stays accurate
        if hasattr(harness_obj, "_history") and isinstance(harness_obj._history, (ActorOrderedCollection, list)):
            harness_obj._history.clear()
            for t in list(chat_log)[-10:]:
                if isinstance(t, dict):
                    u_txt = str(t.get("user", ""))
                    a_txt = str(t.get("reply", ""))
                    acts = t.get("actions") or []
                    if acts:
                        a_txt = f"[Executed: {'; '.join(acts)}] {a_txt}"
                    harness_obj._history.add({"role": "user", "content": u_txt})
                    harness_obj._history.add({"role": "assistant", "content": a_txt})
        return turn_entry

    @actor_query
    def formatScratchpad(self, harness_obj: Any = None) -> str:
        """Format `harness_obj.scratchpad` as a compact string for `AgentHarness>>buildStepPrompt`."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        scratch = getattr(target, "_fields", {}).get("scratchpad") if target is not None else None
        scratch_dict: Dict[str, Any] = {}
        if isinstance(scratch, Dictionary):
            for k, v in scratch._entries.items():
                norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                scratch_dict[norm_k] = v
        return repr(scratch_dict)

    @actor_query
    def formatLastResult(self, harness_obj: Any = None) -> str:
        """Format `harness_obj.lastResult` as a Python repr string for `AgentHarness>>buildStepPrompt`."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        last_result = getattr(target, "_fields", {}).get("lastResult") if target is not None else None
        s = repr(last_result)
        return (s[:3500] + "...'") if len(s) > 3500 else s

    @actor_query
    def formatRecentTurns(self, harness_obj: Any = None) -> str:
        """Format recent multi-turn conversation history compactly so multi-line scripts never bloat prompts."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        recent_turns = self.getChatHistory(target)[-6:]
        if not recent_turns:
            return ""
        turn_lines = ["CONVERSATION HISTORY (RECENT TURNS):"]
        for t in recent_turns:
            if isinstance(t, dict):
                acts = t.get("actions") or []
                compact_acts: List[str] = []
                for a in acts[:4]:
                    a_s = " ".join(str(a).splitlines()).strip()
                    if len(a_s) > 220:
                        a_s = a_s[:190] + "... !!"
                    compact_acts.append(a_s)
                if len(acts) > 4:
                    compact_acts.append(f"(+{len(acts) - 4} more steps)")
                act_str = f" | Executed: `{' ; '.join(compact_acts)}`" if compact_acts else ""
                rep = str(t.get("reply") or "")
                if len(rep) > 650:
                    rep = rep[:647] + "..."
                turn_lines.append(
                    f"- Turn {t.get('turn')}: User={t.get('user')!r}{act_str} -> Reply={rep!r}"
                )
        return "\n".join(turn_lines)

    @staticmethod
    def _compact_step_action_for_history(act: str) -> str:
        """Compact multi-KB `writeFile` / `editFile` string literals in step history so old steps don't bloat prompts."""
        if "__HALLUCINATED_XML_TOOL_CALL__" in act:
            return "<invalid XML tool call>"
        if len(act) > 260 and ("writeFile" in act or "editFile" in act or "writeAndRun" in act):
            return f"{act[:140]} ... <{len(act)} chars total> ... {act[-35:]}"
        if len(act) > 600:
            return f"{act[:300]} ... <{len(act)} chars total> ... {act[-60:]}"
        return act

    @staticmethod
    def _step_loop_nudge(recent_steps: List[Any]) -> str:
        """Return targeted anti-thrashing guidance if recent steps are stuck in read-only or edge SSH loops."""
        is_lockdown = is_host_lockdown()
        consec_readonly = 0
        consec_edge_probes = 0
        has_mutated = False
        for s_entry in recent_steps[-12:]:
            if isinstance(s_entry, dict):
                st = s_entry.get("status", "ok")
                raw_act = str(s_entry.get("action", ""))
                if any(
                    m in raw_act
                    for m in (
                        "editFile",
                        "writeFile",
                        "execShell",
                        "storeScratch",
                        "Def ",
                        "inherit(",
                        "learnProduct",
                        "ingestManual",
                        "writeAndRun",
                        "pullFile",
                        "fetchFile",
                        "sendPhoto",
                        "sendDocument",
                        "sendFile",
                    )
                ):
                    has_mutated = True
                    consec_readonly = 0
                    consec_edge_probes = 0
                elif any(r in raw_act for r in ("searchFile", "readFile")):
                    consec_readonly += 1
                elif "Actor[#Jetson]" in raw_act and ("exec(" in raw_act or "runCommand(" in raw_act):
                    res_obj = s_entry.get("result")
                    failed_ssh = (st == "error") or (isinstance(res_obj, Dictionary) and not bool(res_obj.at("ok")))
                    is_probe = any(p in raw_act for p in ("ls ", "find ", "which ", "grep ", "cat ", "dpkg ", "apt-cache"))
                    if failed_ssh or is_probe:
                        consec_edge_probes += 1
        out: List[str] = []
        if consec_readonly >= 4 and not has_mutated and not is_lockdown:
            out.append(
                f"\n[ACTION REQUIRED — {consec_readonly} CONSECUTIVE READ-ONLY STEPS WITHOUT EDITING]: "
                "Stop searching/reading! Apply your fix NOW using `editFile(Actor[#Host], \"/workspace/...\", \"\"\"old\"\"\", \"\"\"new\"\"\")!!` "
                "(or `writeFile` / `execShell`), then verify it with `execShell`."
            )
        if consec_edge_probes >= 3:
            out.append(
                f"\n[READ THE MANUAL BEFORE GUESSING — {consec_edge_probes} CONSECUTIVE EDGE PROBES/FAILURES]: "
                "Stop blind trial-and-error shell probing! Call `recallProduct(Actor[#ProductLearner], \"NVIDIA Jetson TX1\", \"<topic>\")!!` "
                "or `learnProduct(Actor[#ProductLearner], \"NVIDIA Jetson TX1 L4T R32.7.6\", \"<subsystem/driver>\")!!` to research and store the official manual in Neo4j first, or reply `COMPLETE: <findings>` if you already have the answer!"
            )
        return "".join(out)

    @actor_query
    def formatStepHistory(self, harness_obj: Any = None, page_old_results: bool = False) -> str:
        """Format steps executed so far in the current goal turn (append-only uniform truncation for KV-cache reuse)."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        step_log = getattr(target, "_fields", {}).get("stepLog") if target is not None else None
        if not isinstance(step_log, (ActorOrderedCollection, list)) or len(step_log) == 0:
            return ""
        cur_step_lines = ["STEPS EXECUTED IN CURRENT TURN SO FAR:"]
        recent_steps = list(step_log)[-12:]
        n_recent = len(recent_steps)
        for idx_s, s_entry in enumerate(recent_steps):
            if isinstance(s_entry, dict):
                st = s_entry.get("status", "ok")
                raw_act = str(s_entry.get("action", ""))
                act = self._compact_step_action_for_history(raw_act)
                if st == "error":
                    cur_step_lines.append(
                        f"- Step {s_entry.get('step')} [ERROR]: `{act}` -> {s_entry.get('error')}"
                    )
                else:
                    res_repr = repr(s_entry.get("result"))
                    max_cap = 2500 if idx_s == n_recent - 1 else 1200
                    if page_old_results and idx_s < n_recent - 2 and len(res_repr) > 200:
                        res_repr = res_repr[:200] + "... [paged out]'"
                    elif len(res_repr) > max_cap:
                        res_repr = res_repr[:max_cap] + "...'"
                    unhandled = s_entry.get("unhandled_selector")
                    uh_note = (
                        f" [UnknownSelectorWarning: `{unhandled}` is not defined — check `methodsOf(Actor[#VM], #ClassName)!!`]"
                        if unhandled
                        else ""
                    )
                    cur_step_lines.append(
                        f"- Step {s_entry.get('step')} [OK | gas={s_entry.get('gas_used', 0)}]: `{act}` -> {res_repr}{uh_note}"
                    )
        nudge = self._step_loop_nudge(recent_steps)
        if nudge:
            cur_step_lines.append(nudge)
        return "\n".join(cur_step_lines)

    @actor_query
    def formatErrorDiagnostic(self, harness_obj: Any = None) -> str:
        """Format Neo4j error diagnostics when `self.lastError` is non-empty."""
        target = harness_obj or self.vm.global_vars.get("Actor", {}).get("Harness")
        if target is None:
            return ""
        last_error = str(getattr(target, "_fields", {}).get("lastError", "") or "")
        if not last_error:
            return ""
        step_log = getattr(target, "_fields", {}).get("stepLog")
        prev_bad = ""
        repeated_runtime_guard = ""
        if isinstance(step_log, (ActorOrderedCollection, list)) and len(step_log) > 0:
            entries = list(step_log)
            last_entry = entries[-1]
            if isinstance(last_entry, dict) and last_entry.get("status") == "error":
                raw_bad = str(last_entry.get("action", ""))
                if "__HALLUCINATED_XML_TOOL_CALL__" in raw_bad:
                    prev_bad = "<invalid XML tool call>"
                else:
                    prev_bad = self._compact_step_action_for_history(raw_bad)
                    if len(raw_bad) > 2000:
                        prev_bad += " [TRUNCATED AT MAX_TOKENS — write a compact Python script (< 70 lines, no comments) so it fits cleanly!]"
            if len(entries) >= 4 and "ParseError" not in last_error:
                tail4 = entries[-4:]
                if all(
                    isinstance(e, dict)
                    and e.get("status") == "error"
                    and str(e.get("error", "") or "") == last_error
                    for e in tail4
                ):
                    repeated_runtime_guard = (
                        "[CRITICAL — REPEATED RUNTIME/BRIDGE ERROR (4+ IDENTICAL FAILURES)]:\n"
                        "Stop retrying this exact broken call with minor argument tweaks in an infinite loop! "
                        "Either switch to an alternative pool object/method to achieve the goal, or immediately "
                        "reply with `COMPLETE: <clear explanation of the exact error blocking execution>` so the user is not left waiting.\n"
                    )
        diag = (
            self.grammar_graph.diagnoseError(last_error)
            if self.grammar_graph is not None
            else ""
        )
        err_ctx = getattr(target, "_fields", {}).get("lastErrorContext")
        ctx_line = ""
        method_src_hint = ""
        if isinstance(err_ctx, Dictionary):
            ek = err_ctx.at(Symbol("errorKind")) or ""
            fc = err_ctx.at(Symbol("failedClass")) or ""
            fs = err_ctx.at(Symbol("failedSelector")) or ""
            bo = err_ctx.at(Symbol("bytecodeOffset")) or 0
            ctx_line = f"Structured ErrorContext: kind={ek} at {fc}>>{fs} (offset={bo})\n"
            if fc and fs and fc not in ("<Chunk>", "Object", "AgentHarness"):
                cls_obj = self.vm.hierarchy.get(str(fc))
                m_obj = getattr(cls_obj, "_class_methods", {}).get(str(fs)) if cls_obj is not None else None
                m_src = getattr(m_obj, "source", None)
                if m_src:
                    method_src_hint = (
                        f"Failing Method Source (`{fc}>>{fs}`):\n  `{m_src.strip()}`\n"
                        f" Auto-Repair Hint: Redefine `{fc}>>{fs}` via `now({fc})!! Def {fs}(...) {{ ... }}!!` "
                        f"and re-save the Engram (`persistEngram(Actor[#Memory], #{fc}, \"...\")!!`).\n"
                    )
        return (
            "[PREVIOUS STEP FAILED — SELF-CORRECT THIS ERROR]:\n"
            + (f"Previous failed code: `{prev_bad}`\n" if prev_bad else "")
            + f"Error: {last_error}\n"
            + ctx_line
            + method_src_hint
            + repeated_runtime_guard
            + (f"{diag}\n" if diag else "")
            + "Fix the Actor 4.1 syntax, ivar declaration, or logic error in your next chunk."
        )

    @actor_query
    def defaultInstructions(self) -> str:
        """Return environment-aware Actor 4.1 generation instructions for `AgentHarness>>instructionBlock`."""
        lockdown = is_host_lockdown()
        lines = [
            "INSTRUCTIONS:",
            "1. Output ONLY a single raw 1993 Actor 4.1 code chunk ending in `!!` OR `COMPLETE: <answer>`. Never emit `<tool_call>` XML tags, markdown fences around Actor code, or prose before/after Actor code!",
            "2. For pure conversational greetings/follow-ups requiring NO math, NO file I/O, and NO fresh telemetry (or once `LAST RESULT` already has verified output), reply immediately with:",
            "   COMPLETE: <clear, well-structured human answer>",
            "   (Inside `COMPLETE: ...`, clean Markdown and ```bash / ```python blocks for the user are welcome. Never guess arithmetic in prose—compute in Actor bytecode first!)",
            "3. For live system/sensor/edge checks, run ONE concise expression ending in `!!` (e.g. `summary(Actor[#Valkyrie])!!`, `stackHealth(Actor[#Valkyrie])!!`, `summary(Actor[#Thermometer])!!`, `summary(Actor[#Jetson])!!`, `scoreboard(Actor[#Bench])!!`) and reply `COMPLETE: ...` on the next step.",
            "4. When writing Actor 4.1 code (`!!`):",
            "   - Save values via `storeScratch(Actor[#Harness], #key, value)!!` or `Actor[#Key] := value!!` (use `setMaxSteps`/`setMaxFailures` for step limits).",
        ]
        if lockdown:
            lines.extend(
                [
                    "   - [HOST LOCKDOWN ACTIVE (`ACTORII_HOST_LOCKDOWN=1`)]: `execShell(Actor[#Host], ...)` and host writes outside `/tmp` are locked, and `/workspace` is NOT mounted. NEVER call `execShell(Actor[#Host], ...)` or search `/workspace`! Use Digital Twins (`Actor[#Valkyrie]`, `Actor[#Jetson]`, `Actor[#Thermometer]`, `Actor[#Pantry]`, `Actor[#Web]`, `Actor[#PDF]`, `Actor[#ProductLearner]`, `Actor[#Memory]`, `Actor[#Bench]`, `Actor[#Threads]`). `Actor[#Threads]` (`spawn`, `spawnChunk`, `parallelAll`, `parallelMap`, `Future`, `Channel`) allows concurrent Actor 4.1 / Ring-1 twin calls while keeping raw host shell locked down.",
                    "   - For remote commands/scripts/CUDA/files on Jetson (`Actor[#Jetson]`): use `telemetry(Actor[#Jetson])!!`, `exec(Actor[#Jetson], \"cmd\", 20)!!`, `writeFile(Actor[#Jetson], \"/tmp/prog.cu\", \"\"\"...\"\"\", 20)!!`, `writeAndRun(Actor[#Jetson], \"/tmp/prog.cu\", \"\"\"...\"\"\", 45)!!` (natively compiles & runs `.cu` via `nvcc -arch=sm_53 -ccbin /usr/bin/gcc-7`, `.cpp`/`.c`, `.py`, and `.sh` — ALWAYS wrap multi-line code in triple quotes `\"\"\"...\"\"\"` and NEVER generate raw base64 in Actor code!), or `pullFile(Actor[#Jetson], \"/tmp/snap.jpg\", \"snap.jpg\", 30)!!` (stages into `/tmp/actorii_media/`; upload via `sendPhoto(Actor[#Telegram], at(res, #localPath), \"caption\")!!` or `sendDocument(...)!!`).",
                ]
            )
        else:
            lines.extend(
                [
                    "   - To search, read, write, patch, or test files in `/workspace` (`Actor[#Host]`): call `searchFile(Actor[#Host], \"/workspace\", \"pattern\", 15)!!`, `readFile(Actor[#Host], \"/workspace/file.py\", 1, 120)!!`, `editFile(Actor[#Host], \"/workspace/file.py\", \"\"\"old\"\"\", \"\"\"new\"\"\")!!`, `writeFile(Actor[#Host], \"/workspace/file.py\", \"\"\"full text\"\"\")!!`, or `execShell(Actor[#Host], \"python3 -m pytest -q\", 15)!!`.",
                    "   - When parsing log/CSV/JSON/Parquet files, inspect sample lines/schemas first (`head -n 5`) and match exact bracketed tags (`[ERROR]`, `[WARNING]`) or schema columns.",
                ]
            )
        lines.extend(
            [
                "   - Subclass `GraphService`, `MetricEvaluator`, `DomainEntity`, or an existing `:ActorEngram`; instantiate via `Actor[#Inst] := init(new(ClassName), arg1)!!` with prefix calls `compute(Actor[#Inst])`.",
                "5. Once verified in `LAST RESULT` or scratchpad, reply: `COMPLETE: <concise final answer>`.",
            ]
        )
        return "\n".join(lines)

    @actor_query
    def extractAction(self, raw_reply: str) -> str:
        """Extract clean Actor bytecode or a `COMPLETE:` directive from raw LLM output."""
        return extract_actor_action(str(raw_reply or ""))

    @actor_query
    def planNextStep(self, harness_obj: Any) -> str:
        """Ask the `AgentHarness` LLM brain to plan the next Actor 4.1 step toward `self.goal`."""
        return self.planStepWithContext(harness_obj, None, None, "")

    @actor_query
    def planStepWithContext(
        self,
        harness_obj: Any,
        engram_ctx: Optional[str] = None,
        grammar_ctx: Optional[str] = None,
        policy_note: str = "",
    ) -> str:
        """Metacircular planner invoked by `AgentHarness>>step` with Actor-selected contexts and policy note."""
        if not isinstance(harness_obj, Agent):
            return "COMPLETE: Invalid harness receiver"

        goal = str(harness_obj._fields.get("goal", "") or "")
        step_count = int(harness_obj._fields.get("stepCount", 1) or 1)
        raw_max_steps = harness_obj._fields.get("maxSteps")
        max_steps = 20 if raw_max_steps is None else int(raw_max_steps)
        last_error = str(harness_obj._fields.get("lastError", "") or "")
        last_result = harness_obj._fields.get("lastResult")
        scratch = harness_obj._fields.get("scratchpad")

        scratch_dict: Dict[str, Any] = {}
        if isinstance(scratch, Dictionary):
            for k, v in scratch._entries.items():
                norm_k = k.name if isinstance(k, Symbol) else str(k).lstrip("#")
                scratch_dict[norm_k] = v

        engram_context = (
            str(engram_ctx)
            if engram_ctx is not None
            else (
                self.memory_graph.hydrateRelevantEngrams(goal)
                if self.memory_graph is not None and self.memory_graph.enable_engrams
                else ""
            )
        )
        pool_info = self.poolSummary()
        grammar_context = (
            str(grammar_ctx)
            if grammar_ctx is not None
            else (
                self.grammar_graph.queryConstructsForTask(goal)
                if self.grammar_graph is not None
                else ""
            )
        )
        policy_str = str(policy_note or "").strip()

        prompt_lines = [
            "You are executing inside `ActorII` (the modern Linux-ported, Actor-inspired bytecode virtual machine & metacircular harness).",
            f"GOAL: {goal}",
        ]
        if policy_str:
            prompt_lines.append(f"\n[HARNESS RUNTIME POLICY OVERRIDE]:\n{policy_str}")
        if engram_context:
            prompt_lines.append(f"\n{engram_context}")
        if grammar_context:
            prompt_lines.append(f"\n{grammar_context}")

        inst_block = self.defaultInstructions()
        if inst_block:
            prompt_lines.append(f"\n{inst_block}")

        # 1. Include Multi-Turn Conversation History (last 6 turns) so follow-up questions work naturally
        recent_turns = self.getChatHistory(harness_obj)[-6:]
        if recent_turns:
            turn_lines = ["\nCONVERSATION HISTORY (RECENT TURNS):"]
            for t in recent_turns:
                if isinstance(t, dict):
                    acts = t.get("actions") or []
                    act_str = f" | Executed: `{' ; '.join(acts)}`" if acts else ""
                    turn_lines.append(
                        f"- Turn {t.get('turn')}: User={t.get('user')!r}{act_str} -> Reply={t.get('reply')!r}"
                    )
            prompt_lines.extend(turn_lines)

        # 2. Include steps executed so far in the current goal turn (append-only for prefix KV-cache reuse)
        steps_txt = self.formatStepHistory(harness_obj)
        if steps_txt:
            prompt_lines.append(f"\n{steps_txt}")

        prompt_lines.extend(
            [
                f"\nLIVE POOL SUMMARY: {pool_info}",
                (
                    f"STEP: {step_count} of {max_steps}"
                    if max_steps > 0
                    else f"STEP: {step_count} (unlimited)"
                ),
                f"SCRATCHPAD (`Actor[#Harness].scratchpad`): {scratch_dict}",
                f"LAST RESULT: {self.formatLastResult(harness_obj)}",
            ]
        )
        if last_error:
            err_txt = self.formatErrorDiagnostic(harness_obj)
            if err_txt:
                prompt_lines.append(f"\n{err_txt}")
        full_prompt = "\n".join(prompt_lines)

        # Temporarily clear `_history` during `ask(full_prompt)` so `full_prompt` boilerplate isn't duplicated,
        # since `CONVERSATION HISTORY (RECENT TURNS)` is already injected cleanly above.
        saved_hist = list(harness_obj._history) if hasattr(harness_obj, "_history") else []
        harness_obj.clearHistory()
        try:
            raw_reply = harness_obj.ask(full_prompt)
        finally:
            if hasattr(harness_obj, "_history") and isinstance(harness_obj._history, ActorOrderedCollection):
                harness_obj._history._items = list(saved_hist)

        # If offline deterministic fallback string was returned by `Agent.ask`, synthesize a deterministic action
        if raw_reply.startswith(f"[{harness_obj.name} ({harness_obj.class_name})] Processed:"):
            if step_count == 1 and "result" not in scratch_dict:
                return f'storeScratch(Actor[#Harness], #goalAck, {goal!r})!!'
            return f"COMPLETE: Goal processed ({goal})"

        return extract_actor_action(raw_reply)

    # =========================================================================
    # Tier 2: Audited State-Mutating Harness Commands (`@actor_command`)
    # =========================================================================

    @actor_command
    def executeStepAction(
        self,
        harness_obj: Any,
        action_text: str,
        max_opcodes: int = 25000,
    ) -> Any:
        """Execute a single planned step inside a guarded transaction with gas metering."""
        raw = str(action_text or "").strip()
        step_num = int(harness_obj._fields.get("stepCount", 1) or 1)
        step_log = harness_obj._fields.get("stepLog")
        prev_exec_result = harness_obj._fields.get("lastResult")

        chunk_src, comp_summary = _split_code_and_complete(raw)

        # 0a. Guard against LLM echoing history summary truncation placeholders (`... <N chars total> ...`) into code:
        if chunk_src and re.search(r"<\d+\s+chars\s+total>", chunk_src):
            trunc_msg = (
                "ActorGuardrailError: TruncatedPlaceholderError: Your code contains a literal history summary placeholder "
                "(`... <N chars total> ...`). Never copy abbreviated summary markers—always emit complete, compilable source code!"
            )
            failures = int(harness_obj._fields.get("failures", 0) or 0) + 1
            harness_obj._fields["failures"] = failures
            harness_obj._fields["lastError"] = trunc_msg
            trunc_entry = {
                "step": step_num,
                "action": chunk_src,
                "status": "error",
                "tier": "TIER_0_READ_ONLY",
                "gas_used": 0,
                "error": trunc_msg,
                "rolled_back": True,
                "failures": failures,
                "timestamp": time.time(),
            }
            if isinstance(step_log, ActorOrderedCollection):
                step_log.add(trunc_entry)
            self._notify_step(trunc_entry)
            return f"ERROR: {trunc_msg}"

        # 0. Guard against identical recent step repetition (including A-B-A-B ping-pong across warnings):
        # inspect the last 4 steps in `step_log` so a warning step never resets duplicate detection.
        if chunk_src and comp_summary is None and isinstance(step_log, (ActorOrderedCollection, list)) and len(step_log) > 0:
            norm_chunk = chunk_src.strip()
            matched_prev = None
            for cand_entry in reversed(list(step_log)[-4:]):
                if (
                    isinstance(cand_entry, dict)
                    and cand_entry.get("status") == "ok"
                    and str(cand_entry.get("action", "")).strip() == norm_chunk
                ):
                    matched_prev = cand_entry
                    break
            if matched_prev is not None:
                prior_dups = sum(
                    1
                    for e in list(step_log)
                    if isinstance(e, dict) and str(e.get("error", "")).startswith("DuplicateStepWarning:")
                )
                dup_count = prior_dups + 1
                failures = int(harness_obj._fields.get("failures", 0) or 0)
                if dup_count >= 2:
                    failures += 1
                    harness_obj._fields["failures"] = failures
                raw_max_failures = harness_obj._fields.get("maxFailures")
                max_failures = 5 if raw_max_failures is None else int(raw_max_failures)
                if dup_count >= 3 or (max_failures > 0 and failures >= max_failures):
                    harness_obj._fields["status"] = Symbol("blocked")
                dup_msg = (
                    "DuplicateStepWarning: You just re-emitted the exact same Actor 4.1 code chunk that already "
                    f"executed in Step {matched_prev.get('step')} (result={matched_prev.get('result')!r}, "
                    f"scratchpad={self.formatScratchpad(harness_obj)}). "
                    "Do NOT repeat identical code! If the values in SCRATCHPAD or LAST RESULT already answer the goal, reply "
                    "`COMPLETE: <summary>`. If you are unsure of the exact driver/command for a device or library, call "
                    "`recallProduct(Actor[#ProductLearner], \"<product>\", \"<topic>\")!!` or `learnProduct(Actor[#ProductLearner], \"<product>\", \"<topic>\")!!` "
                    "instead of repeating the same probe! If a method value is wrong, inspect `sourceOf(Actor[#VM], #ClassName, #selector)!!`."
                )
                harness_obj._fields["lastError"] = dup_msg
                dup_entry = {
                    "step": step_num,
                    "action": chunk_src,
                    "status": "error",
                    "tier": "TIER_0_READ_ONLY",
                    "gas_used": 0,
                    "error": dup_msg,
                    "rolled_back": True,
                    "failures": failures,
                    "timestamp": time.time(),
                }
                if isinstance(step_log, ActorOrderedCollection):
                    step_log.add(dup_entry)
                self._notify_step(dup_entry)
                return f"ERROR: {dup_msg}"

        # 1. Pure `COMPLETE:` directive (or direct conversational reply) with no preceding code
        if not chunk_src and comp_summary is not None:
            summary = comp_summary or str(prev_exec_result if prev_exec_result is not None else "Done")
            # Guard against verbatim or near-verbatim reply parroting across ALL steps (not only step 1!):
            # reject stale copy-pasted speeches from the last 3 turns unless already rejected twice in this turn.
            chat_log = harness_obj._fields.get("chatLog")
            current_goal = str(harness_obj._fields.get("goal", "") or "").strip()
            prior_parrot_warnings = (
                sum(
                    1
                    for e in list(step_log)
                    if isinstance(e, dict) and str(e.get("error", "")).startswith("DuplicateReplyError:")
                )
                if isinstance(step_log, (ActorOrderedCollection, list))
                else 0
            )
            if (
                prior_parrot_warnings < 2
                and len(summary) > 120
                and isinstance(chat_log, (ActorOrderedCollection, list))
                and len(chat_log) > 0
            ):
                has_valid_tool_step = any(
                    isinstance(e, dict)
                    and e.get("status") == "ok"
                    and not e.get("unhandled_selector")
                    and not str(e.get("action", "")).strip().upper().startswith("COMPLETE:")
                    for e in (list(step_log) if isinstance(step_log, (ActorOrderedCollection, list)) else [])
                )
                norm_sum = " ".join(summary.lower().split())
                nums_sum = re.findall(r"-?\d+(?:\.\d+)?", norm_sum)
                stop_goal_w = {"what", "whats", "where", "when", "check", "give", "show", "tell", "current", "status", "the", "and", "for", "with", "how", "now"}
                cur_goal_words = {
                    w.lower()
                    for w in re.findall(r"[A-Za-z0-9]+", current_goal)
                    if len(w) >= 4 and w.lower() not in stop_goal_w
                }
                matched_stale_turn = None
                for prev_turn in reversed(list(chat_log)[-3:]):
                    if not isinstance(prev_turn, dict):
                        continue
                    prev_user = str(prev_turn.get("user", "")).strip()
                    if prev_user == current_goal:
                        continue
                    prev_rep = str(prev_turn.get("reply", "") or "").strip()
                    if len(prev_rep) <= 100:
                        continue
                    norm_prev = " ".join(prev_rep.lower().split())
                    nums_prev = re.findall(r"-?\d+(?:\.\d+)?", norm_prev)
                    # Never flag as parroted when numeric measurements/metrics changed!
                    if nums_sum != nums_prev and (nums_sum or nums_prev):
                        continue
                    # When a valid tool step ran in THIS turn, allow template-similar answers or repeated sensor queries
                    if has_valid_tool_step:
                        prev_goal_words = {
                            w.lower()
                            for w in re.findall(r"[A-Za-z0-9]+", prev_user)
                            if len(w) >= 4 and w.lower() not in stop_goal_w
                        }
                        if norm_prev != norm_sum or bool(cur_goal_words & prev_goal_words):
                            continue
                    if norm_prev == norm_sum or SequenceMatcher(None, norm_prev[:1200], norm_sum[:1200]).ratio() >= 0.86:
                        matched_stale_turn = prev_turn
                        break
                if matched_stale_turn is not None:
                    rep_msg = (
                        "DuplicateReplyError: You just repeated nearly the exact same prose reply from a previous turn "
                        f"(Turn {matched_stale_turn.get('turn')}) instead of directly answering the user's current request ({current_goal!r})! "
                        "Synthesize a fresh, concise response grounded specifically in the steps executed during THIS turn (or execute the needed Actor 4.1 tool call first)."
                    )
                    harness_obj._fields["lastError"] = rep_msg
                    rep_entry = {
                        "step": step_num,
                        "action": f"COMPLETE: {summary}",
                        "status": "error",
                        "tier": "TIER_0_READ_ONLY",
                        "gas_used": 0,
                        "error": rep_msg,
                        "rolled_back": True,
                        "failures": int(harness_obj._fields.get("failures", 0) or 0),
                        "timestamp": time.time(),
                    }
                    if isinstance(step_log, ActorOrderedCollection):
                        step_log.add(rep_entry)
                    self._notify_step(rep_entry)
                    return f"ERROR: {rep_msg}"

            harness_obj._fields["status"] = Symbol("complete")
            harness_obj._fields["lastResult"] = summary
            harness_obj._fields["lastError"] = ""
            entry = {
                "step": step_num,
                "action": f"COMPLETE: {summary}",
                "status": "complete",
                "tier": "TIER_0_READ_ONLY",
                "gas_used": 0,
                "result": summary,
                "timestamp": time.time(),
            }
            step_log = harness_obj._fields.get("stepLog")
            if isinstance(step_log, ActorOrderedCollection):
                step_log.add(entry)
            self._record_completed_turn(harness_obj, summary, last_exec_result=prev_exec_result)
            self._notify_step(entry)
            return summary

        lint_status = self.lintChunk(chunk_src)
        tier_label = lint_status.split(":", 1)[1] if lint_status.startswith("OK:") else "UNKNOWN"

        try:
            if lint_status.startswith("ERROR:"):
                from actorii.guardrails.linter import ActorGuardrailError

                raise ActorGuardrailError(lint_status[6:])

            result = self.vm.eval_transactional(
                chunk_src,
                max_opcodes=int(max_opcodes or 25000),
                shadow_only=False,
            )
            gas_used = int(getattr(self.vm, "last_eval_opcodes", 0))
            prev_total_gas = int(harness_obj._fields.get("totalGasUsed", 0) or 0)
            prev_error = str(harness_obj._fields.get("lastError", "") or "")
            step_log = harness_obj._fields.get("stepLog")
            if prev_error and self.grammar_graph is not None:
                prev_bad = ""
                if isinstance(step_log, ActorOrderedCollection) and len(step_log) > 0:
                    last_entry = list(step_log)[-1]
                    if isinstance(last_entry, dict) and last_entry.get("status") == "error":
                        prev_bad = str(last_entry.get("action", ""))
                self.grammar_graph.recordLearnedFix(
                    prev_error,
                    prev_bad,
                    chunk_src,
                    model=getattr(harness_obj, "model", "unknown"),
                )
            harness_obj._fields["totalGasUsed"] = prev_total_gas + gas_used
            harness_obj._fields["failures"] = 0
            harness_obj._fields["lastError"] = ""
            harness_obj._fields["lastErrorContext"] = None
            harness_obj._fields["lastResult"] = result

            auto_engrams: List[str] = []
            if self.memory_graph is not None and self.memory_graph.enable_engrams:
                auto_engrams = self.memory_graph.auto_crystallize_from_chunk(
                    chunk_src,
                    goal_description=str(harness_obj._fields.get("goal", "") or ""),
                )

            unhandled_sel = getattr(self.vm, "_last_unhandled_selector", None)
            entry = {
                "step": step_num,
                "action": chunk_src,
                "status": "ok",
                "tier": tier_label,
                "gas_used": gas_used,
                "result": result,
                "auto_engrams": auto_engrams,
                "timestamp": time.time(),
            }
            if result is None and unhandled_sel:
                entry["unhandled_selector"] = str(unhandled_sel)
            step_log = harness_obj._fields.get("stepLog")
            if isinstance(step_log, ActorOrderedCollection):
                step_log.add(entry)
            self._notify_step(entry)

            if comp_summary is not None:
                harness_obj._fields["status"] = Symbol("complete")
                harness_obj._fields["lastResult"] = comp_summary
                comp_entry = {
                    "step": step_num,
                    "action": f"COMPLETE: {comp_summary}",
                    "status": "complete",
                    "tier": tier_label,
                    "gas_used": 0,
                    "result": comp_summary,
                    "timestamp": time.time(),
                }
                if isinstance(step_log, ActorOrderedCollection):
                    step_log.add(comp_entry)
                self._record_completed_turn(harness_obj, comp_summary, last_exec_result=result)
                self._notify_step(comp_entry)
                return comp_summary

            return result

        except Exception as exc:
            gas_used = int(getattr(self.vm, "last_eval_opcodes", 0))
            prev_total_gas = int(harness_obj._fields.get("totalGasUsed", 0) or 0)
            harness_obj._fields["totalGasUsed"] = prev_total_gas + gas_used

            failures = int(harness_obj._fields.get("failures", 0) or 0) + 1
            raw_max_failures = harness_obj._fields.get("maxFailures")
            max_failures = 5 if raw_max_failures is None else int(raw_max_failures)
            err_msg = f"{type(exc).__name__}: {exc}"

            exc_name = type(exc).__name__
            if "Syntax" in exc_name or "Parse" in exc_name:
                err_kind = "SyntaxError"
            elif "Guardrail" in exc_name or "ProtectedClass" in exc_name:
                err_kind = "LintViolation"
            elif "OpcodeBudget" in exc_name:
                err_kind = "GasExhausted"
            elif "BridgeSecurity" in exc_name:
                err_kind = "BridgeSecurityViolation"
            elif "not understood" in str(exc).lower() or "AttributeError" in exc_name:
                err_kind = "MessageNotUnderstood"
            else:
                err_kind = "RuntimeError"

            frame_info = getattr(self.vm, "_last_failed_frame_info", None)
            if isinstance(frame_info, dict) and self.memory_graph is not None and self.memory_graph.enable_engrams:
                failed_cls = str(frame_info.get("failedClass") or "")
                if (
                    failed_cls
                    and failed_cls not in PROTECTED_SYSTEM_CLASSES
                    and failed_cls not in ("<Chunk>", "Object", "AgentHarness")
                ):
                    try:
                        self.memory_graph.recordEngramUsage(failed_cls, success=False)
                    except Exception:
                        pass
            harness_obj._fields["failures"] = failures
            harness_obj._fields["lastError"] = err_msg
            harness_obj._fields["lastErrorContext"] = self._build_error_context_dict(
                err_kind, err_msg, frame_info
            )
            step_log = harness_obj._fields.get("stepLog")
            identical_err_streak = 0
            if isinstance(step_log, (ActorOrderedCollection, list)):
                for prev_e in reversed(list(step_log)[-3:]):
                    if (
                        isinstance(prev_e, dict)
                        and prev_e.get("status") == "error"
                        and str(prev_e.get("action", "")).strip() == chunk_src.strip()
                        and str(prev_e.get("error", "")) == err_msg
                    ):
                        identical_err_streak += 1
                    else:
                        break
            if (max_failures > 0 and failures >= max_failures) or (max_failures <= 0 and identical_err_streak >= 3):
                harness_obj._fields["status"] = Symbol("blocked")

            entry = {
                "step": step_num,
                "action": chunk_src,
                "status": "error",
                "tier": tier_label,
                "gas_used": gas_used,
                "error": err_msg,
                "rolled_back": True,
                "failures": failures,
                "timestamp": time.time(),
            }
            if isinstance(step_log, ActorOrderedCollection):
                step_log.add(entry)
            self._notify_step(entry)
            return f"ERROR: {err_msg}"

    @actor_command
    def rejectStepOutcome(self, harness_obj: Any, reason: str) -> str:
        """Called by `AgentHarness>>step` when `verifyOutcome(self, outcome)` returns `false`."""
        was_complete = harness_obj._fields.get("status") == Symbol("complete")
        if was_complete:
            chat_log = harness_obj._fields.get("chatLog")
            if isinstance(chat_log, (ActorOrderedCollection, list)) and len(chat_log) > 0:
                chat_log.pop()

        failures = int(harness_obj._fields.get("failures", 0) or 0) + 1
        raw_max_failures = harness_obj._fields.get("maxFailures")
        max_failures = 5 if raw_max_failures is None else int(raw_max_failures)
        last_res = harness_obj._fields.get("lastResult")
        err_msg = f"OutcomeVerificationError: {reason} (lastResult={last_res!r})"

        harness_obj._fields["failures"] = failures
        harness_obj._fields["lastError"] = err_msg
        harness_obj._fields["lastErrorContext"] = self._build_error_context_dict(
            "OutcomeVerificationError",
            err_msg,
            {"failedClass": getattr(harness_obj, "class_name", "AgentHarness"), "failedSelector": "verifyOutcome", "bytecodeOffset": 0, "localBindings": {"outcome": repr(last_res)[:120]}},
        )
        if max_failures > 0 and failures >= max_failures:
            harness_obj._fields["status"] = Symbol("blocked")
        elif harness_obj._fields.get("status") != Symbol("halted"):
            harness_obj._fields["status"] = Symbol("running")

        step_log = harness_obj._fields.get("stepLog")
        if isinstance(step_log, (ActorOrderedCollection, list)) and len(step_log) > 0:
            last_entry = step_log[-1]
            if isinstance(last_entry, dict):
                last_entry["status"] = "error"
                last_entry["error"] = err_msg
                last_entry["failures"] = failures
                self._notify_step(last_entry)
        return f"ERROR: {err_msg}"

    @actor_command
    def evolveHarness(
        self,
        harness_obj: Any,
        patch_source: str,
        validation_chunk: str = "",
    ) -> str:
        """Guarded Ring 1 self-modification of `AgentHarness` (or subclass) with shadow validation & Neo4j Engram versioning."""
        import os as _os

        if _os.environ.get("ACTORII_DISABLE_SELF_EVOLVE") == "1":
            return (
                "EVOLVE_REJECTED(lockdown): Self-modification of AgentHarness is explicitly disabled "
                "(`ACTORII_DISABLE_SELF_EVOLVE=1`)."
            )
        patch_src = str(patch_source or "").strip()
        if not patch_src:
            return "EVOLVE_REJECTED(empty): Empty patch source"
        if not patch_src.endswith("!!"):
            patch_src = patch_src.rstrip("!") + "!!"

        target_cls = getattr(harness_obj, "class_name", "AgentHarness") or "AgentHarness"
        if not re.search(r"\b(now|inherit)\s*\(", patch_src):
            patch_src = f"now({target_cls})!!\n{patch_src}"

        # Block Tier 2 side-effecting bridge calls (`execShell`, `writeFile`, `editFile`, `runShell`, `sendMessage`, or any `@actor_command`) inside evolved harness hooks
        for forbidden_call in ("execShell", "writeFile", "editFile", "runShell", "detonateSandbox", "sendMessage", "sendToChat"):
            if re.search(rf"\b{forbidden_call}\s*\(", patch_src):
                return f"EVOLVE_REJECTED(forbidden_bridge_call): Evolved AgentHarness hook cannot invoke Tier 2 side-effecting method `{forbidden_call}`."

        lint_status = self.lintChunk(patch_src)
        if lint_status.startswith("ERROR:"):
            return f"EVOLVE_REJECTED(lint): {lint_status[6:]}"
        if "TIER_2_SYSTEM_OR_BRIDGE" in lint_status:
            return (
                "EVOLVE_REJECTED(forbidden_bridge_call): Evolved AgentHarness cognitive hook cannot invoke "
                f"Tier 2 side-effecting commands ({lint_status})."
            )

        val_src = str(validation_chunk or "").strip()
        if val_src and not val_src.endswith("!!"):
            val_src = val_src.rstrip("!") + "!!"

        # 1. Snapshot baseline `AgentHarness` into Neo4j `:ActorEngram` before applying patch so rollback is always available
        if self.memory_graph is not None and self.memory_graph.enable_engrams:
            try:
                self.memory_graph.persistEngram(
                    target_cls,
                    f"Pre-evolution snapshot of {target_cls}",
                )
            except Exception:
                pass

        # 2. Execute `patch_src` + `val_src` inside a speculative copy-on-write shadow transaction (`shadow_only=True`)
        shadow_script = f"{patch_src}\n{val_src}" if val_src else patch_src
        try:
            val_res = self.vm.eval_transactional(
                shadow_script,
                max_opcodes=25000,
                shadow_only=True,
            )
        except Exception as exc:
            return f"EVOLVE_REJECTED(shadow_error): {type(exc).__name__}: {exc}"

        if val_src and (val_res is False or val_res == 0 or str(val_res).startswith("ERROR")):
            return f"EVOLVE_REJECTED(validation_failed): validation returned {val_res!r}"

        # 3. Commit the verified `patch_src` to the live VM and refresh `harness_obj` method bindings
        try:
            self.vm.eval_transactional(
                patch_src,
                max_opcodes=25000,
                shadow_only=False,
            )
            cls_obj = self.vm.hierarchy.get(target_cls)
            if cls_obj is not None and isinstance(harness_obj, ActorObject):
                harness_obj._methods = {
                    k: v
                    for k, v in harness_obj._methods.items()
                    if k in getattr(harness_obj, "_instance_methods", set())
                }
                harness_obj.inherit_from(cls_obj)
        except Exception as exc:
            return f"EVOLVE_REJECTED(commit_error): {type(exc).__name__}: {exc}"

        engram_status = "live_vm_only"
        if self.memory_graph is not None and self.memory_graph.enable_engrams:
            try:
                engram_status = self.memory_graph.persistEngram(
                    target_cls,
                    f"Evolved {target_cls} via evolveHarness",
                )
            except Exception as exc:
                engram_status = f"engram_warn: {exc}"

        return f"EVOLVE_COMMITTED: {target_cls} ({engram_status})"

    @actor_command
    def forkHarness(
        self,
        harness_obj: Any,
        sub_goal: str,
        policy_override: str = "",
        max_steps: int = 4,
        gas_limit: int = 25000,
    ) -> Any:
        """Spawn an isolated Copy-on-Write child `AgentHarness` to execute `sub_goal` without polluting the parent's transcript."""
        parent_snap = self.vm.snapshot_to_dict()
        child_vm = ActorVM()
        child_vm.restore_from_dict(parent_snap, clear_existing=True)
        child_vm.ai_provider = self.vm.ai_provider
        child_vm.ai_model = self.vm.ai_model

        parent_model = getattr(harness_obj, "model", "qwen3.8:latest") if isinstance(harness_obj, Agent) else "qwen3.8:latest"
        parent_endpoint = getattr(harness_obj, "endpoint", None) if isinstance(harness_obj, Agent) else None
        ns = self.memory_graph.namespace if self.memory_graph is not None else "default"
        engrams_on = self.memory_graph.enable_engrams if self.memory_graph is not None else False

        sub_max_steps = 4 if max_steps is None else int(max_steps)
        child_vm, child_harness, _ = bootstrap_agent_harness(
            child_vm,
            name="SubHarness",
            model=parent_model,
            endpoint=parent_endpoint,
            max_steps=sub_max_steps,
            max_failures=0,
            gas_limit=int(gas_limit or 25000),
            enable_memory_graph=(self.memory_graph is not None),
            enable_engrams=engrams_on,
            memory_namespace=ns,
        )
        if isinstance(harness_obj, Agent) and getattr(harness_obj, "custom_provider", None) is not None:
            child_harness.custom_provider = harness_obj.custom_provider

        policy_txt = str(policy_override or "").strip()
        if policy_txt:
            child_vm.eval_chunk(
                f"now(AgentHarness)!!\nDef policyNote(self) {{ ^{policy_txt!r}; }}!!"
            )

        res = child_vm.eval_chunk(
            f"runGoal(Actor[#Harness], {str(sub_goal)!r}, {sub_max_steps})!!"
        )
        return res

    @actor_command
    def checkpointImage(self, path: str = "ACTOR_HARNESS.IMG") -> str:
        """Save an atomic whole-world `.img` snapshot of the VM."""
        out = self.vm.save_image(str(path))
        return str(out)

    @actor_command
    def fileOutClass(self, class_name: str) -> str:
        """Export a live class to `<fileout_dir>/<class_name>.act`."""
        norm = class_name.name if isinstance(class_name, Symbol) else str(class_name).lstrip("#")
        out_dir = self.vm.fileout_dir or str(Path(__file__).resolve().parents[3] / "examples" / "fileout")
        mgr = ActorFileOutManager(out_dir)
        out_path = mgr.file_out_class(self.vm, norm)
        return str(out_path)

    def _notify_step(self, entry: Dict[str, Any]) -> None:
        if not getattr(self.vm, "_in_shadow_run", False):
            try:
                actor_dict = self.vm.global_vars.get("Actor", {})
                h_obj = actor_dict.get("Harness") if isinstance(actor_dict, dict) else None
                turn_id = str(getattr(h_obj, "_active_turn_id", "") or "vm-turn")
                GLOBAL_TELEMETRY.log_step_exec(turn_id=turn_id, entry=entry)
            except Exception:
                pass
        for cb in list(self.step_callbacks):
            try:
                cb(entry)
            except Exception:
                pass


def bootstrap_agent_harness(
    vm: Optional[ActorVM] = None,
    *,
    name: str = "Bilby",
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    max_steps: int = 20,
    max_failures: Optional[int] = None,
    gas_limit: int = 100000,
    enable_guardrails: bool = True,
    fileout_dir: Optional[str] = None,
    enable_memory_graph: bool = True,
    enable_engrams: bool = True,
    memory_namespace: Optional[str] = None,
) -> Tuple[ActorVM, Agent, VMSandboxBridge]:
    """Load `agent_harness.act` into `vm` and instantiate `Actor[#Harness]` / `Actor[#Bilby]` and `Actor[#Sandbox]`."""
    model = model or os.environ.get("ACTORII_LLM_MODEL", "qwen3.8:latest").strip() or "qwen3.8:latest"
    disable_demo = os.environ.get("ACTORII_DISABLE_DEMO_TWINS") == "1"
    active_vm = vm or ActorVM()
    active_vm.lock_system_singletons = False
    if enable_guardrails:
        active_vm.enable_guardrails(
            True,
            protect_system_classes=True,
            strict_ivars=True,
            max_opcodes=max(100_000, gas_limit * 4),
            fileout_dir=fileout_dir,
        )

    grammar_py = ActorGrammarGraph(auto_seed=True)
    active_vm.expose_object(grammar_py, global_key="#Grammar")

    llm_py = LLMBridge(active_vm)
    active_vm.expose_object(llm_py, global_key="#LLM")

    host_py = HostWorkspaceBridge(active_vm)
    active_vm.expose_object(host_py, global_key="#Host")

    vm_ref_py = VMReflectionBridge(active_vm)
    active_vm.expose_object(vm_ref_py, global_key="#VM")

    memory_py: Optional[ActorMemoryGraph] = None
    if enable_memory_graph:
        memory_py = ActorMemoryGraph(
            active_vm,
            enable_engrams=enable_engrams,
            namespace=memory_namespace,
        )
        active_vm.expose_object(memory_py, global_key="#Memory")

    sandbox_py = VMSandboxBridge(
        active_vm,
        grammar_graph=grammar_py,
        memory_graph=memory_py,
    )
    active_vm.expose_object(sandbox_py, global_key="#Sandbox")

    bench_py = ActorBenchSuite(active_vm)
    active_vm.expose_object(bench_py, global_key="#Bench")

    threads_py = ActorThreadPoolTwin(active_vm, max_workers=8)
    active_vm.expose_object(threads_py, global_key="#Threads")

    valkyrie_py = ValkyrieSystemTwin()
    active_vm.expose_object(valkyrie_py, global_key="#Valkyrie")
    active_vm.expose_object(valkyrie_py, global_key="#System")

    if not disable_demo:
        hue_py = SmartLightTwin()
        active_vm.expose_object(hue_py, global_key="#Hue")

    pantry_py = PantryGraphTwin()
    active_vm.expose_object(pantry_py, global_key="#Pantry")

    re_py = ReverseEngineeringTwin()
    active_vm.expose_object(re_py, global_key="#ReverseEngineer")
    active_vm.expose_object(re_py, global_key="#REA")

    tg_py = TelegramBotTwin()
    active_vm.expose_object(tg_py, global_key="#Telegram")

    proxy_py = ReverseProxyTwin()
    active_vm.expose_object(proxy_py, global_key="#Proxy")

    jetson_py = JetsonEdgeTwin()
    active_vm.expose_object(jetson_py, global_key="#Jetson")

    net_py = NetTransportBridge()
    active_vm.expose_object(net_py, global_key="#Net")

    pdf_py = PDFDocumentTwin()
    active_vm.expose_object(pdf_py, global_key="#PDF")

    thermo_mac = os.environ.get("ACTORII_THERMO_MAC", "" if disable_demo else "00:11:22:33:44:55").strip()
    thermo_label = "TP358S" if disable_demo else "TP358S (Demo)"
    active_vm.eval_chunk(
        f"Actor[#Thermometer] := new(ThermoProTwin, {thermo_mac!r}, {thermo_label!r})!!\n"
        'Actor[#ThermoUI] := new(ThermometerUI, Actor[#Thermometer], "ThermoPro TP358W Digital Twin")!!'
    )

    oop_source = OOP_FOUNDATION_ACT_PATH.read_text(encoding="utf-8")
    active_vm.eval_chunk(oop_source)

    web_source = WEB_SCRAPER_ACT_PATH.read_text(encoding="utf-8")
    active_vm.eval_chunk(web_source)
    active_vm.eval_chunk(
        'Actor[#Web] := initWeb(new(WebScraper), Actor[#Net], "https://api.firecrawl.dev", 8000)!!'
    )

    learner_source = PRODUCT_LEARNER_ACT_PATH.read_text(encoding="utf-8")
    active_vm.eval_chunk(learner_source)
    active_vm.eval_chunk(
        "Actor[#ProductLearner] := initLearner(new(ProductLearner), Actor[#Web], Actor[#PDF], Actor[#Net], at(Actor, #Memory))!!"
    )

    act_source = HARNESS_ACT_PATH.read_text(encoding="utf-8")
    active_vm.eval_chunk(act_source)

    active_vm.eval_chunk(
        f'Actor[#Harness] := initHarness(new(AgentHarness, {name!r}, {model!r}), Actor[#Sandbox])!!'
    )
    active_vm.eval_chunk("Actor[#Bilby] := Actor[#Harness]!!")
    active_vm.eval_chunk(f"setMaxSteps(Actor[#Harness], {int(max_steps)})!!")
    if max_failures is not None:
        active_vm.eval_chunk(f"setMaxFailures(Actor[#Harness], {int(max_failures)})!!")
    active_vm.eval_chunk(f"setGasLimit(Actor[#Harness], {int(gas_limit)})!!")

    harness_obj = active_vm.global_vars["Actor"]["Harness"]
    if isinstance(harness_obj, Agent):
        harness_obj.system_prompt = load_harness_system_prompt()
        harness_obj._fields["chatLog"] = ActorOrderedCollection()
        if endpoint:
            harness_obj.endpoint = endpoint

    if enable_guardrails:
        active_vm.lock_system_singletons = True

    return active_vm, harness_obj, sandbox_py


__all__ = [
    "ActorBenchSuite",
    "ActorMemoryGraph",
    "BENCHMARK_TASKS",
    "DEFAULT_HARNESS_SYSTEM_PROMPT",
    "HARNESS_ACT_PATH",
    "HARNESS_SYSTEM_PROMPT",
    "VMSandboxBridge",
    "bootstrap_agent_harness",
    "extract_actor_action",
    "load_harness_system_prompt",
]
