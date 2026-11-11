"""Neo4j-Backed 1993 Actor Grammar & Self-Learning Remediation Graph (`Actor[#Grammar]`).

Connects the ActorII `AgentHarness` to the local Neo4j graph database on `valkyrie`
(`http://127.0.0.1:7474/db/neo4j/tx/commit` / `bolt://127.0.0.1:7687`) so that:
1. 1993 Whitewater Actor 4.1 syntax rules (`:ActorSyntaxConstruct`), built-in class
   primitives (`:ActorBuiltinClass` -> `:ActorPrimitive`), and compiler/runtime
   error patterns (`:ActorErrorPattern`) live as a queryable knowledge graph in Neo4j.
2. Before code generation, `queryConstructsForTask(task_description)` retrieves the
   relevant grammar constructs, primitive signatures, and any learned fixes from Neo4j.
3. When a `ParseError`, `ActorGuardrailError`, or unit-test failure occurs,
   `diagnoseError(error_text)` traverses `(:ActorErrorPattern)-[:REMEDIATED_BY]->(:ActorSyntaxConstruct)`
   and `(:ActorErrorPattern)-[:HAS_LEARNED_FIX]->(:ActorLearnedFix)` in Neo4j.
4. When the agent self-corrects a failed step on the subsequent turn, `recordLearnedFix`
   persists a new `(:ActorLearnedFix)` node in Neo4j so future turns benefit from episodic syntax memory.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List, Optional

from actorii.bridge import actor_bridge, actor_command, actor_query

NEO4J_HTTP_TX_URL = "http://127.0.0.1:7474/db/neo4j/tx/commit"


SEED_CONSTRUCTS: List[Dict[str, Any]] = [
    {
        "id": "method_def",
        "title": "Method Definition & Local Pipe Declarations",
        "keywords": ["def", "method", "function", "return", "local", "variable", "pipe"],
        "rule": (
            "Define methods on the active class (`now(ClassName)!!`) using "
            "`Def selector(self, param1, param2 | local1, local2, i, n, acc, res) { ... }!!`. "
            "CRITICAL: Declare ALL local variables after the pipe `|` in the `Def` header! "
            "Every statement ends with `;`. Return values use caret `^expr;`."
        ),
        "canonical_example": (
            "now(Solution)!!\n"
            "Def sum_to_n(self, n | i, acc) {\n"
            "  acc := 0;\n"
            "  i := 1;\n"
            "  loop while (i <= n) begin\n"
            "    acc := acc + i;\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^acc;\n"
            "}!!"
        ),
        "anti_pattern": "Def f(self, n) { acc := 0; return acc; }!! /* ERROR: undeclared local `acc` and invalid `return` */",
    },
    {
        "id": "conditionals",
        "title": "Conditionals (`if ... then ... else ... endif;`)",
        "keywords": ["if", "then", "else", "endif", "condition", "branch", "boolean", "true", "false"],
        "rule": (
            "Conditionals MUST include `then` after the condition and close with `endif;`: "
            "`if (cond) then stmt1; else stmt2; endif;`. "
            "Never use `elsif` or `elif` — nest `if ... then ... endif;` inside `else` if needed. "
            "Booleans are `true` (1) and `false` (0)."
        ),
        "canonical_example": (
            "if (n <= 1) then\n"
            "  ^false;\n"
            "else\n"
            "  if (n == 2) then ^true; endif;\n"
            "endif;"
        ),
        "anti_pattern": "if (n <= 1) ^n; endif; /* ERROR: missing `then` after condition */",
    },
    {
        "id": "loops",
        "title": "While Loops (`loop while (cond) begin ... endLoop;`)",
        "keywords": ["loop", "while", "begin", "endloop", "iterate", "for", "repeat"],
        "rule": (
            "Loops use `loop while (cond) begin stmt1; stmt2; endLoop;`. "
            "Every statement inside `begin ... endLoop;` MUST end with `;`, and `endLoop;` MUST end with `;`."
        ),
        "canonical_example": (
            "i := 0;\n"
            "loop while (i < size(arr)) begin\n"
            "  if (arr[i] < 0) then ^true; endif;\n"
            "  i := i + 1;\n"
            "endLoop;"
        ),
        "anti_pattern": "while (i < n) { i := i + 1 } /* ERROR: C-style while loop is invalid in 1993 Actor */",
    },
    {
        "id": "operators_and_math",
        "title": "Assignment, Comparisons, Modulo & Float vs Integer Division",
        "keywords": ["assign", "divide", "float", "real", "area", "modulo", "equal", "math", "triangle", "half"],
        "rule": (
            "Assignment uses `:=`. Comparisons: `==`, `<>` (not equal), `<`, `<=`, `>`, `>=`. "
            "Modulo is `(a % b)`. "
            "CRITICAL DIVISION RULE: Dividing two integers `(a * h) / 2` performs INTEGER FLOOR DIVISION (`15 / 2 == 7`). "
            "Whenever a floating-point result (like `7.5` or area/average) is required, you MUST divide by a float literal `2.0` "
            "(`(a * h) / 2.0` or `(a * h * 0.5)`)."
        ),
        "canonical_example": "Def triangle_area(self, a, h) { ^((a * h) / 2.0); }!!",
        "anti_pattern": "Def triangle_area(self, a, h) { ^((a * h) / 2); }!! /* BUG: returns 7 instead of 7.5 for (5, 3) */",
    },
    {
        "id": "collections_and_dispatch",
        "title": "Prefix Message Dispatch, Arrays & Strings (`size(x)`, `arr[i]`)",
        "keywords": ["array", "list", "string", "size", "length", "index", "call", "prefix"],
        "rule": (
            "1993 Actor uses PREFIX function-call syntax for all messages: `selector(receiver, arg1)` "
            "(NEVER dot syntax `receiver.selector(arg1)`). "
            "Length of an Array or String `x` is `size(x)`. Array and String indexing `x[i]` is 0-based (`0` to `size(x) - 1`). "
            "Array literals are written `#(10, 20, -5)`. To dynamically build and return a list/array, use `OrderedCollection`: "
            "`res := new(OrderedCollection); add(res, val); ^res;`. To convert an integer/number `n` to a String (or inspect digits), call `s := asString(n)` and index characters `s[j] == \"7\"` or use modulo/floor division (`d := num % 10; num := (num - d) / 10`). String concatenation is `s1 + s2`, absolute value is `abs(x)`, and null is `nil`."
        ),
        "canonical_example": (
            "Def max_element(self, l | i, n, best) {\n"
            "  n := size(l);\n"
            "  best := l[0];\n"
            "  i := 1;\n"
            "  loop while (i < n) begin\n"
            "    if (l[i] > best) then best := l[i]; endif;\n"
            "    i := i + 1;\n"
            "  endLoop;\n"
            "  ^best;\n"
            "}!!"
        ),
        "anti_pattern": "n := l.size(); /* ERROR: dot syntax is invalid; use size(l) */",
    },
    {
        "id": "class_and_pool",
        "title": "Classes, Multi-Level Inheritance (`inherit(Super, #Sub, ...)`) & Object Pool",
        "keywords": ["class", "inherit", "object", "pool", "scratch", "store", "ivar", "subclass", "super", "hierarchy"],
        "rule": (
            "Define classes with `inherit(SuperClass, #ClassName, #(#ivar1 #ivar2), 2, nil)!! now(ClassName)!!`. "
            "Prefer subclassing foundational base classes (`GraphService`, `MetricEvaluator`, `DomainEntity`, `CompositePipeline`) "
            "or existing `:ActorEngram` classes (`inherit(GraphService, #MyAuditor, #(#threshold), 1, nil)!!`) instead of flat `Object`! "
            "Subclasses automatically inherit all ancestor `ivars` and methods (`queryRows(self, ...)`, `sumColumn(self, rows, #col)`). "
            "Store global objects in `Actor[#Key] := val!!` and harness results in `storeScratch(Actor[#Harness], #key, val)!!`."
        ),
        "canonical_example": (
            "inherit(GraphService, #SlaAuditor, #(#minUptime), 1, nil)!!\n"
            "now(SlaAuditor)!!\n"
            "Def init(self, minUp) { initService(self, \"ServiceNode\"); self.minUptime := minUp; ^self; }!!\n"
            "Actor[#SlaAuditor] := init(new(SlaAuditor), 99)!!"
        ),
        "anti_pattern": "self.undeclared := 5; /* ERROR: [E_UNDECLARED_IVAR] if not listed in class or ancestor inherit(...) */",
    },
    {
        "id": "oop_hierarchy_and_polymorphism",
        "title": "OOP Subclassing, Template Method Polymorphism & Object Composition",
        "keywords": ["oop", "subclass", "polymorphism", "override", "template", "composition", "pipeline", "evaluator", "hierarchy", "extend"],
        "rule": (
            "Build multi-level 1993 Actor 4.1 hierarchies: "
            "(1) Subclass `MetricEvaluator` (which extends `GraphService`) via `inherit(MetricEvaluator, #SubEvaluator, #(#bonus), 1, nil)!!`, "
            "initialize parent state via `initEvaluator(self, name, thresh, weight)`, and override `Def evaluateRow(self, row | val) { ... }!!` "
            "while inheriting `evaluateAll(self, rows)` and `weightedTotal(self, rows)` from `MetricEvaluator` and `queryRows(self, cypher)` from `GraphService`. "
            "(2) Compose multiple polymorphic evaluator objects inside `CompositePipeline` (`addStage(pipe, evalObj)`) and dispatch `runEvaluators(pipe, rows)` across all stages."
        ),
        "canonical_example": (
            "inherit(MetricEvaluator, #BonusEvaluator, #(#bonus), 1, nil)!!\n"
            "now(BonusEvaluator)!!\n"
            "Def initBonus(self, nameStr, thresh, weight, bonusVal) {\n"
            "  initEvaluator(self, nameStr, thresh, weight);\n"
            "  self.bonus := bonusVal;\n"
            "  ^self;\n"
            "}!!\n"
            "Def evaluateRow(self, row | v) {\n"
            "  v := at(row, #score);\n"
            "  if (v >= self.threshold) then ^(v + self.bonus); endif;\n"
            "  ^0;\n"
            "}!!"
        ),
        "anti_pattern": "inherit(Object, #FlatClass, ...); /* Prefer subclassing GraphService, MetricEvaluator, or an existing Engram when extending domain behavior! */",
    },
    {
        "id": "neo4j_cypher_and_engrams",
        "title": "Neo4j Graph Queries (`GraphService` / `cypherQuery`) & Active Engrams (`persistEngram`)",
        "keywords": ["neo4j", "graph", "cypher", "match", "merge", "engram", "memory", "schema", "node"],
        "rule": (
            "Query Neo4j from Actor code via `rows := cypherQuery(Actor[#Memory], \"MATCH (n:Label) RETURN n.name AS name, n.val AS val\")` "
            "or by subclassing `GraphService` (`inherit(GraphService, #SkillName, #(#limit), 1, nil)!!`) and calling `rows := queryRows(self, \"MATCH ...\")` "
            "and `sumColumn(self, rows, #val)` / `avgColumn(self, rows, #val)`. "
            "CRITICAL SINGLE-QUOTE RULE: Always use single quotes `'...'` for Cypher string literals inside Actor `\"...\"` strings! "
            "`cypherQuery` returns a 0-based `Array` of `Dictionary` rows: access columns via `at(rows[i], #val)`. "
            "Write graph facts via `cypherWrite(Actor[#Memory], \"MERGE (n:Label {name: 'X'}) SET n.val = 10\")` (or `writeGraph(self, ...)`). "
            "Any compiled user class is automatically crystallized into Neo4j as a versioned `:ActorEngram` with `[:SUBCLASS_OF]` and `[:OPERATES_ON]` edges."
        ),
        "canonical_example": (
            "inherit(GraphService, #RecipeScorer, #(#maxCost), 1, nil)!!\n"
            "now(RecipeScorer)!!\n"
            "Def score(self, maxCost | rows) {\n"
            "  rows := queryRows(self, \"MATCH (r:BenchRecipe) WHERE r.cost <= \" + asString(maxCost) + \" RETURN r.protein AS protein\");\n"
            "  ^sumColumn(self, rows, #protein);\n"
            "}!!\n"
            "Actor[#RecipeScorer] := new(RecipeScorer)!!\n"
            "persistEngram(Actor[#Memory], #RecipeScorer, \"Score recipes by maxCost using GraphService\")!!"
        ),
        "anti_pattern": "cypherQuery(Actor[#Memory], \"MATCH (n {name: \"foo\"}) RETURN n\"); /* ERROR: unescaped double quotes inside Actor string; use 'foo' */",
    },
]

SEED_PRIMITIVES: List[Dict[str, str]] = [
    {"cls": "Collection", "selector": "size", "syntax": "size(colOrStr)", "desc": "Return number of elements in Array or characters in String"},
    {"cls": "Array", "selector": "[]", "syntax": "arr[i]", "desc": "0-based element lookup at index i"},
    {"cls": "OrderedCollection", "selector": "add", "syntax": "add(col, item)", "desc": "Append item to OrderedCollection"},
    {"cls": "Dictionary", "selector": "at", "syntax": "at(dict, #key)", "desc": "Lookup value by symbol/string key"},
    {"cls": "Dictionary", "selector": "atPut", "syntax": "atPut(dict, #key, val)", "desc": "Store key-value pair in Dictionary"},
    {"cls": "Number", "selector": "/", "syntax": "(a / 2.0)", "desc": "Float division when either operand is float; integer floor division if both are int"},
    {"cls": "Number", "selector": "%", "syntax": "(a % b)", "desc": "Modulo remainder"},
    {"cls": "Object", "selector": "asString", "syntax": "asString(val)", "desc": "Convert integer, float, symbol, or object to String"},
    {"cls": "AgentHarness", "selector": "storeScratch", "syntax": "storeScratch(Actor[#Harness], #key, val)", "desc": "Store key-value in AgentHarness scratchpad"},
    {"cls": "AgentHarness", "selector": "readScratch", "syntax": "readScratch(Actor[#Harness], #key)", "desc": "Read value from AgentHarness scratchpad"},
    {"cls": "AgentHarness", "selector": "setMaxSteps", "syntax": "setMaxSteps(Actor[#Harness], 0)", "desc": "Update per-turn ReAct step limit on live AgentHarness (0 = unlimited)"},
    {"cls": "AgentHarness", "selector": "setMaxFailures", "syntax": "setMaxFailures(Actor[#Harness], 0)", "desc": "Update consecutive rollback strike limit on live AgentHarness (0 = unlimited)"},
    {"cls": "AgentHarness", "selector": "evolve", "syntax": "evolve(Actor[#Harness], patchCode, validationCode)", "desc": "Safely hot-patch AgentHarness methods in a shadow transaction and persist versioned Engram"},
    {"cls": "AgentHarness", "selector": "rollbackHarness", "syntax": "rollbackHarness(Actor[#Harness])", "desc": "Roll back AgentHarness to its previous version in Neo4j"},
    {"cls": "ActorMemoryGraph", "selector": "graphSchema", "syntax": "graphSchema(Actor[#Memory])", "desc": "Inspect domain node labels, properties, and relationships in Neo4j"},
    {"cls": "ActorMemoryGraph", "selector": "cypherQuery", "syntax": "cypherQuery(Actor[#Memory], \"MATCH ... RETURN ...\")", "desc": "Run read Cypher and return 0-based Array of Dictionary rows"},
    {"cls": "ActorMemoryGraph", "selector": "cypherQueryWith", "syntax": "cypherQueryWith(Actor[#Memory], \"MATCH (n {name: $p1}) RETURN n\", arg1)", "desc": "Run parameterized read Cypher ($p1, $p2 or Dictionary) and return Array of Dictionary rows"},
    {"cls": "ActorMemoryGraph", "selector": "cypherWrite", "syntax": "cypherWrite(Actor[#Memory], \"MERGE ...\")", "desc": "Execute mutating Cypher (MERGE/CREATE/SET) in Neo4j"},
    {"cls": "ActorMemoryGraph", "selector": "cypherWriteWith", "syntax": "cypherWriteWith(Actor[#Memory], \"MERGE (n {name: $p1}) SET n.v = $p2\", arg1, arg2)", "desc": "Execute parameterized mutating Cypher in Neo4j"},
    {"cls": "ActorMemoryGraph", "selector": "persistEngram", "syntax": "persistEngram(Actor[#Memory], #ClassName, desc)", "desc": "File-out a live compiled ActorClass and its methods into Neo4j as a reusable `:ActorEngram`"},
    {"cls": "ActorMemoryGraph", "selector": "rollbackEngram", "syntax": "rollbackEngram(Actor[#Memory], #ClassName)", "desc": "Roll back an `:ActorEngram` to its previous version in Neo4j and re-compile"},
    {"cls": "ActorMemoryGraph", "selector": "saveEngram", "syntax": "saveEngram(Actor[#Memory], #Name, desc, actorCode)", "desc": "Compile and persist a reusable Actor 4.1 computational memory skill in Neo4j"},
    {"cls": "ActorMemoryGraph", "selector": "loadEngram", "syntax": "loadEngram(Actor[#Memory], #Name)", "desc": "Hydrate a saved `:ActorEngram` (and its dependencies) from Neo4j into `Actor[#Name]`"},
    {"cls": "HostWorkspaceBridge", "selector": "readFile", "syntax": "readFile(Actor[#Host], \"/workspace/file.py\", 1, 120)", "desc": "Read lines start..end from a workspace file with line numbers"},
    {"cls": "HostWorkspaceBridge", "selector": "searchFile", "syntax": "searchFile(Actor[#Host], \"/workspace/file.py\", \"class MyClass\", 15)", "desc": "Search a workspace file for a pattern/class/method and return matching line numbers and context"},
    {"cls": "HostWorkspaceBridge", "selector": "writeFile", "syntax": "writeFile(Actor[#Host], \"/workspace/file.py\", content)", "desc": "Write full string content to a workspace file"},
    {"cls": "HostWorkspaceBridge", "selector": "editFile", "syntax": "editFile(Actor[#Host], \"/workspace/file.py\", oldText, newText)", "desc": "Replace exact oldText with newText in a workspace file"},
    {"cls": "HostWorkspaceBridge", "selector": "execShell", "syntax": "execShell(Actor[#Host], \"pytest -q\", 15)", "desc": "Run shell command in workspace and return Dictionary (#exitCode, #stdout, #stderr)"},
    {"cls": "JetsonEdgeTwin", "selector": "summary", "syntax": "summary(Actor[#Jetson])", "desc": "Get live status, IP, RAM, disk, temperature, and RTSP camera state of the NVIDIA Jetson board"},
    {"cls": "JetsonEdgeTwin", "selector": "scanLan", "syntax": "scanLan(Actor[#Jetson])", "desc": "Scan 192.168.68.0/22 LAN for SSH (:22) and RTSP (:8554) edge devices without host shell"},
    {"cls": "JetsonEdgeTwin", "selector": "discover", "syntax": "discover(Actor[#Jetson])", "desc": "Auto-discover Jetson IP via mDNS / LAN SSH banner scan"},
    {"cls": "JetsonEdgeTwin", "selector": "exec", "syntax": "exec(Actor[#Jetson], \"uname -a\", 20)", "desc": "Execute a guarded command over SSH on the NVIDIA Jetson board and return Dictionary (#ok, #exitCode, #stdout, #stderr)"},
]

SEED_ERROR_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": "Expected 'then' after if condition",
        "construct_id": "conditionals",
        "hint": "Every `if` statement in 1993 Actor MUST have `then` immediately after the condition and end with `endif;`: `if (cond) then stmt; endif;`.",
    },
    {
        "pattern": "ENDLOOP",
        "construct_id": "loops",
        "hint": (
            "Check your `loop while (cond) begin ... endLoop;` syntax: `begin` is required after `while (cond)`, "
            "every statement inside the loop must end with `;`, and any `if (cond) then` inside the loop MUST close "
            "with `endif;` (NEVER `end;`)."
        ),
    },
    {
        "pattern": "Expected attribute name after '.'",
        "construct_id": "collections_and_dispatch",
        "hint": "Do not use `.` dot method calls or output plain English sentences. Use prefix calls `size(x)` and output ONLY valid 1993 Actor code ending in `!!`.",
    },
    {
        "pattern": "E_UNDECLARED_IVAR",
        "construct_id": "class_and_pool",
        "hint": "Instance variable accessed via `self.ivar` was not declared in `inherit(Super, #Class, #(#ivar), ...)`. Declare local variables after `|` in `Def m(self | loc)`.",
    },
    {
        "pattern": "ZeroDivisionError",
        "construct_id": "method_def",
        "hint": "Ensure all temporary/local variables (like `t` or `r` in GCD/loops) are declared after the pipe `|` in `Def method(self, a, b | t, r)` so they are not `nil`, and in Euclidean GCD `loop while (b != 0) begin r := (a % b); a := b; b := r; endLoop; ^a;` compute `a % b` (NEVER `b % a`).",
    },
    {
        "pattern": ".5",
        "construct_id": "operators_and_math",
        "hint": "Float test failed because `/` on two integers in Actor performs floor division (`15 / 2 == 7`). Divide by float `2.0` (`(a * h) / 2.0`) to produce a float!",
    },
    {
        "pattern": "Expected ')' closing argument list",
        "construct_id": "collections_and_dispatch",
        "hint": "Missing closing `)` before `!!` or `;` in a prefix function call (e.g., `storeScratch(Actor[#Harness], #answer, val)!!`). Ensure `)` comes before `!!`.",
    },
    {
        "pattern": "RBRACE",
        "construct_id": "conditionals",
        "hint": "Unexpected `}` before closing `endif;` or `endLoop;`. Check that every `if (cond) then` has a matching `endif;` and every `loop while (cond) begin` has a matching `endLoop;` before `}`.",
    },
    {
        "pattern": "('LT', '<')",
        "construct_id": "collections_and_dispatch",
        "hint": "CRITICAL: Do NOT wrap your output in `<tool_call>`, `<function=...>`, `<parameter=...>`, or XML/HTML tags! Output ONLY raw 1993 Actor 4.1 code ending in `!!`.",
    },
    {
        "pattern": "Neo4j Cypher error",
        "construct_id": "neo4j_cypher_and_engrams",
        "hint": "Check your Cypher query syntax: use single quotes `'...'` for string literals inside Actor `\"...\"` strings, alias returned fields (`RETURN n.name AS name`), and call `graphSchema(Actor[#Memory])!!` to verify exact node labels and property keys.",
    },
    {
        "pattern": "invalid literal for int() with base 10",
        "construct_id": "collections_and_dispatch",
        "hint": "CRITICAL: `atPut(dict, #key, val)` mutates `dict` in-place and returns `val` (NOT `dict`)! Never write `d := atPut(new(Dictionary), #k1, v1)!! d := atPut(d, #k2, v2)!!` because `d` becomes `v1` (e.g. an Array, which then fails with `invalid literal for int()` when indexed with `#k2`). Always initialize the Dictionary first: `d := new(Dictionary, 8)!! atPut(d, #k1, v1)!! atPut(d, #k2, v2)!!`.",
    },
]


@actor_bridge(name="ActorGrammarGraph")
class ActorGrammarGraph:
    """Neo4j-backed 1993 Actor syntax, primitive & error-remediation knowledge graph (`Actor[#Grammar]`)."""

    def __init__(self, tx_url: str = NEO4J_HTTP_TX_URL, auto_seed: bool = True) -> None:
        self.tx_url = tx_url
        self.online = False
        if auto_seed:
            self.online = self.seed_graph()

    def _run_cypher(self, statement: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        payload = {
            "statements": [
                {
                    "statement": statement,
                    "parameters": parameters or {},
                }
            ]
        }
        req = urllib.request.Request(
            self.tx_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        if body.get("errors"):
            raise RuntimeError(f"Neo4j Cypher error: {body['errors']}")
        results = body.get("results", [])
        if not results:
            return []
        cols = results[0].get("columns", [])
        rows = []
        for item in results[0].get("data", []):
            row_vals = item.get("row", [])
            rows.append(dict(zip(cols, row_vals)))
        return rows

    def seed_graph(self) -> bool:
        """Idempotently seed `:ActorSyntaxConstruct`, `:ActorPrimitive`, and `:ActorErrorPattern` nodes in Neo4j."""
        try:
            self._run_cypher(
                """
                UNWIND $constructs AS c
                MERGE (sc:ActorSyntaxConstruct {id: c.id})
                SET sc.title = c.title,
                    sc.keywords = c.keywords,
                    sc.rule = c.rule,
                    sc.canonical_example = c.canonical_example,
                    sc.anti_pattern = c.anti_pattern,
                    sc.updated_at = timestamp()
                """,
                {"constructs": SEED_CONSTRUCTS},
            )
            self._run_cypher(
                """
                UNWIND $primitives AS p
                MERGE (bc:ActorBuiltinClass {name: p.cls})
                MERGE (pm:ActorPrimitive {cls: p.cls, selector: p.selector})
                SET pm.syntax = p.syntax,
                    pm.description = p.desc
                MERGE (bc)-[:HAS_PRIMITIVE]->(pm)
                """,
                {"primitives": SEED_PRIMITIVES},
            )
            self._run_cypher(
                """
                UNWIND $errors AS e
                MERGE (ep:ActorErrorPattern {pattern: e.pattern})
                SET ep.hint = e.hint
                WITH ep, e
                MATCH (sc:ActorSyntaxConstruct {id: e.construct_id})
                MERGE (ep)-[:REMEDIATED_BY]->(sc)
                """,
                {"errors": SEED_ERROR_PATTERNS},
            )
            return True
        except Exception:
            return False

    # =========================================================================
    # Tier 0: Read-Only Graph Queries (`@actor_query`)
    # =========================================================================

    @actor_query
    def graphStats(self) -> str:
        """Return live counts of 1993 Actor grammar nodes & learned fixes in Neo4j."""
        try:
            rows = self._run_cypher(
                """
                MATCH (sc:ActorSyntaxConstruct) WITH count(sc) AS constructs
                MATCH (pm:ActorPrimitive) WITH constructs, count(pm) AS primitives
                MATCH (ep:ActorErrorPattern) WITH constructs, primitives, count(ep) AS error_patterns
                OPTIONAL MATCH (lf:ActorLearnedFix)
                RETURN constructs, primitives, error_patterns, count(lf) AS learned_fixes
                """
            )
            if rows:
                r = rows[0]
                return (
                    f"Neo4j ActorGrammarGraph ONLINE (:7687/:7474) — "
                    f"constructs={r['constructs']}, primitives={r['primitives']}, "
                    f"error_patterns={r['error_patterns']}, learned_fixes={r['learned_fixes']}"
                )
        except Exception as exc:
            return f"Neo4j ActorGrammarGraph OFFLINE ({exc})"
        return "Neo4j ActorGrammarGraph empty"

    @actor_query
    def lookupSyntax(self, topic: str) -> str:
        """Query Neo4j for a specific 1993 Actor syntax construct or keyword."""
        q = str(topic or "").strip().lower()
        try:
            rows = self._run_cypher(
                """
                MATCH (sc:ActorSyntaxConstruct)
                WHERE toLower(sc.id) CONTAINS $q
                   OR toLower(sc.title) CONTAINS $q
                   OR any(k IN sc.keywords WHERE toLower(k) CONTAINS $q)
                RETURN sc.id AS id, sc.title AS title, sc.rule AS rule,
                       sc.canonical_example AS example, sc.anti_pattern AS anti
                ORDER BY sc.id
                """,
                {"q": q},
            )
            if rows:
                lines = [f"[Neo4j `:ActorSyntaxConstruct` Match for {topic!r}]:"]
                for r in rows:
                    lines.append(f"- {r['title']}: {r['rule']}")
                    lines.append(f"  Example:\n  {r['example']}")
                    lines.append(f"  Avoid: {r['anti']}")
                return "\n".join(lines)
        except Exception:
            pass
        # Offline fallback from SEED_CONSTRUCTS
        for c in SEED_CONSTRUCTS:
            if q in c["id"] or any(q in k for k in c["keywords"]):
                return f"- {c['title']}: {c['rule']}\n  Example:\n  {c['canonical_example']}"
        return f"No specific construct matched {topic!r}"

    @actor_query
    def queryConstructsForTask(self, task_text: str, entry_point: str = "", params_str: str = "") -> str:
        """Query Neo4j for all 1993 Actor grammar rules and primitives relevant to a task."""
        task_lower = str(task_text or "").lower()
        if (
            "/workspace" in task_lower
            and not entry_point
            and not any(k in task_lower for k in ("inherit(", "actor class", "engram", "neo4j", "cypher", "thermo", "valkyrie"))
        ):
            return (
                "[NEO4J 1993 ACTOR 4.1 WORKSPACE BRIDGE REFERENCE (`Actor[#Host]`)]:\n"
                "- 1993 Actor uses PREFIX call syntax ending in `!!`: `selector(Actor[#Host], arg1, arg2)!!`.\n"
                "- Search directory or file: `searchFile(Actor[#Host], \"/workspace\", \"pattern\", 15)!!` or `searchFile(Actor[#Host], \"/workspace/path.py\", \"class Foo\", 20)!!`.\n"
                "- Read file lines: `readFile(Actor[#Host], \"/workspace/path.py\", 1, 120)!!`.\n"
                "- Surgical edit: `editFile(Actor[#Host], \"/workspace/path.py\", \"exact old lines\", \"new lines\")!!`.\n"
                "- Write full file: `writeFile(Actor[#Host], \"/workspace/path.py\", \"content\")!!`.\n"
                "- Run shell/tests: `execShell(Actor[#Host], \"pytest -q\", 20)!!`.\n"
                "- For math word problems (`answer.txt`): NEVER guess arithmetic in a `writeFile` string literal! Always compute and write the exact number using Python (`execShell(Actor[#Host], \"python3 -c '...; open(\\\"/workspace/answer.txt\\\", \\\"w\\\").write(str(ans))'\", 10)!!`) or Actor arithmetic.\n"
                "- When verified, reply: `COMPLETE: <summary>`."
            )
        needs_graph_constructs = (not entry_point) and any(
            kw in task_lower
            for kw in ("neo4j", "cypher", "graph", "engram", "memory", "schema", "node", "persist", "actor[#memory]")
        )
        try:
            constructs = self._run_cypher(
                """
                MATCH (sc:ActorSyntaxConstruct)
                RETURN sc.id AS id, sc.title AS title, sc.rule AS rule, sc.canonical_example AS example
                ORDER BY sc.id
                """
            )
            primitives = self._run_cypher(
                """
                MATCH (bc:ActorBuiltinClass)-[:HAS_PRIMITIVE]->(pm:ActorPrimitive)
                RETURN bc.name AS cls, pm.syntax AS syntax, pm.description AS desc
                ORDER BY bc.name, pm.selector
                """
            )
            if not needs_graph_constructs:
                constructs = [c for c in constructs if c.get("id") != "neo4j_cypher_and_engrams"]
                primitives = [p for p in primitives if p.get("cls") != "ActorMemoryGraph"]
            lines = [
                "[NEO4J 1993 ACTOR 4.1 GRAMMAR GRAPH (`Actor[#Grammar]` @ :7687)]:",
            ]
            if entry_point:
                param_set = {p.strip() for p in params_str.split(",") if p.strip()}
                sug_locals = [v for v in ("i", "n", "acc", "res", "a", "b") if v not in param_set][:4]
                loc_str = ", ".join(sug_locals)
                lines.append(
                    f"Target Signature Template: `Def {entry_point}(self, {params_str} | {loc_str}) {{ ... }}!!`"
                )
            for idx, c in enumerate(constructs, 1):
                lines.append(f"{idx}. {c['title']}: {c['rule']}")
            if primitives:
                prim_strs = [f"`{p['syntax']}` ({p['desc']})" for p in primitives]
                lines.append("Built-in Primitives: " + "; ".join(prim_strs))
            return "\n".join(lines)
        except Exception:
            # Fallback if Neo4j is unreachable
            lines = ["[1993 Actor Grammar Fallback]:"]
            if entry_point:
                lines.append(f"Signature: `Def {entry_point}(self, {params_str} | i, n, acc, res) {{ ... }}!!`")
            for idx, c in enumerate(SEED_CONSTRUCTS, 1):
                lines.append(f"{idx}. {c['title']}: {c['rule']}")
            return "\n".join(lines)

    @actor_query
    def diagnoseError(self, error_text: str) -> str:
        """Traverse `(:ActorErrorPattern)-[:REMEDIATED_BY]->(:ActorSyntaxConstruct)` in Neo4j for an error."""
        err_str = str(error_text or "").strip()
        if not err_str:
            return ""
        try:
            rows = self._run_cypher(
                """
                MATCH (ep:ActorErrorPattern)-[:REMEDIATED_BY]->(sc:ActorSyntaxConstruct)
                WHERE $err CONTAINS ep.pattern OR toLower($err) CONTAINS toLower(ep.pattern)
                OPTIONAL MATCH (ep)-[:HAS_LEARNED_FIX]->(lf:ActorLearnedFix)
                RETURN ep.pattern AS pattern, ep.hint AS hint,
                       sc.title AS construct_title, sc.rule AS rule,
                       sc.canonical_example AS example,
                       collect(lf.fixed_code)[0..2] AS learned_fixes
                """,
                {"err": err_str},
            )
            if rows:
                parts = ["[NEO4J ERROR-REMEDIATION GRAPH DIAGNOSIS]:"]
                for r in rows:
                    parts.append(f"• Matched Pattern `{r['pattern']}` -> {r['hint']}")
                    parts.append(f"  Grammar Rule ({r['construct_title']}): {r['rule']}")
                    parts.append(f"  Canonical 1993 Actor Pattern:\n{r['example']}")
                    fixes = [f for f in (r.get("learned_fixes") or []) if f]
                    if fixes:
                        parts.append(f"  Previously Verified Fix in Neo4j: `{fixes[0][:140]}`")
                return "\n".join(parts)
        except Exception:
            pass
        return (
            "[NEO4J GRAMMAR REMINDER]: Declare all local variables after `|` (`Def m(self, x | i, n, acc)`), "
            "use `if (cond) then ... endif;`, end every statement before `endLoop;` with `;`, "
            "and use `2.0` for float division."
        )

    # =========================================================================
    # Tier 2: Episodic Self-Learning Graph Mutation (`@actor_command`)
    # =========================================================================

    @actor_command
    def recordLearnedFix(
        self,
        error_text: str,
        bad_code: str,
        fixed_code: str,
        model: str = "unknown",
    ) -> str:
        """Persist a verified self-correction `(Error) -> (Fixed Actor Code)` into Neo4j."""
        err_str = str(error_text or "").strip()
        if err_str.startswith("ERROR:"):
            err_str = err_str[6:].strip()
        err_str = err_str[:240]
        bad_str = str(bad_code or "")[:400]
        fix_str = str(fixed_code or "")[:400]
        try:
            self._run_cypher(
                """
                MERGE (lf:ActorLearnedFix {
                    error_summary: $err,
                    fixed_code: $fix
                })
                ON CREATE SET
                    lf.bad_code = $bad,
                    lf.model = $model,
                    lf.created_at = timestamp()
                WITH lf
                MATCH (ep:ActorErrorPattern)
                WHERE $err CONTAINS ep.pattern OR toLower($err) CONTAINS toLower(ep.pattern)
                MERGE (ep)-[:HAS_LEARNED_FIX]->(lf)
                """,
                {"err": err_str, "bad": bad_str, "fix": fix_str, "model": str(model)},
            )
            return "RECORDED_IN_NEO4J"
        except Exception as exc:
            return f"NEO4J_RECORD_SKIPPED: {exc}"
