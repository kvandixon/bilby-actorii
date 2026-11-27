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

import base64
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set

from actorii.bridge import actor_bridge, actor_command, actor_query

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL", "http://127.0.0.1:7474/db/neo4j/tx/commit"
)
_SEEDED_URLS: Set[str] = set()


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
            "Assignment uses `:=`. Comparisons: `==`, `!=` (or `<>` for not equal), `<`, `<=`, `>`, `>=`. "
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
            "Length of an Array, OrderedCollection, or String `x` is `size(x)`. Array/OrderedCollection/String indexing `x[i]` (or `at(x, i)` / `atPut(x, i, val)`) is 0-based (`0` to `size(x) - 1`). "
            "Fixed-size arrays: `arr := new(Array, n)` (or literal `#(10, 20, -5)`). Dynamic lists/queues: `oc := new(OrderedCollection); add(oc, val); removeFirst(oc); removeLast(oc); first(oc); last(oc)`. "
            "To convert an integer/number `n` to a String (or inspect digits), call `s := asString(n)` and index characters `s[j] == \"7\"` or use modulo/floor division (`d := num % 10; num := (num - d) / 10`). String concatenation is `s1 + s2`, absolute value is `abs(x)`, and null is `nil`."
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
    {
        "id": "web_and_streaming_protocols",
        "title": "Modern Web & Media Streaming Protocol Objects (`Actor[#Protocols]`, `Actor[#Net]`, `Actor[#Jetson]`)",
        "keywords": [
            "stream",
            "streaming",
            "video",
            "mjpeg",
            "rtsp",
            "hls",
            "m3u8",
            "http",
            "rest",
            "sse",
            "websocket",
            "browser",
            "jpeg",
            "camera",
            "port",
        ],
        "rule": (
            "Before hand-rolling raw sockets, C++ HTTP headers, or RTSP/MJPEG pipelines, consult `Actor[#Protocols]`: "
            "(1) `recommendFor(Actor[#Protocols], goalDesc)!!` or `spec(Actor[#Protocols], #mjpeg|#rtsp|#http|#sse|#ws|#hls)!!`; "
            "(2) `blueprint(Actor[#Protocols], #mjpeg, #python|#cpp)!!` for browser-playable `<img src=\"/stream\">` `multipart/x-mixed-replace` video (pipe CUDA/C++ raw RGB24 into `ffmpeg` -> Python `ThreadingHTTPServer` to avoid `libjpeg` ABI mismatches!); "
            "(3) `blueprint(Actor[#Protocols], #rtsp, #send)!!` (Jetson TX1 hardware H.264 rule: ALWAYS use `omxh264enc insert-sps-pps=true`, NEVER `nvv4l2h264enc`!) and `blueprint(Actor[#Protocols], #rtsp, #receive)!!` (`-rtsp_transport tcp`); "
            "(4) Deploy persistent background edge services with `deployService(Actor[#Jetson], \"svc_name\", \"$HOME/services/svc_name\", \"cmd\", port, 25)!!` instead of `/tmp` + `nohup` hacks; "
            "(5) ALWAYS verify live endpoints with real client conformance probes (`verify(Actor[#Protocols], url, #mjpeg)!!`, `probeHttp(Actor[#Net], url, 8)!!`, or `probeRtsp(Actor[#Net], rtspUrl, \"snap.jpg\", 12)!!`) instead of custom `socket.recv()` scripts!"
        ),
        "canonical_example": (
            "spec(Actor[#Protocols], #mjpeg)!!\n"
            "blueprint(Actor[#Protocols], #rtsp, #send)!!\n"
            "deployService(Actor[#Jetson], \"julia_web\", \"$HOME/services/julia_web\", \"python3 server.py\", 8090, 25)!!\n"
            "verify(Actor[#Protocols], \"http://192.168.68.55:8090/stream\", #mjpeg)!!"
        ),
        "anti_pattern": "/* NEVER send raw `[4-byte length][JPEG]` on TCP without `HTTP/1.1 200 OK` when serving web/browser streams, and NEVER open `rtsp://` directly in a browser without an HTTP MJPEG/HLS bridge! */",
    },
    {
        "id": "network_topology_and_tailscale",
        "title": "Network Topology, Tailscale Identity & LAN Port Relay (`Actor[#Network]`)",
        "keywords": [
            "tailscale",
            "tailnet",
            "100.",
            "192.168.",
            "subnet",
            "route",
            "bridge",
            "forward",
            "ip",
            "connect",
            "phone",
            "lan",
            "firewall",
        ],
        "rule": (
            "Whenever a user mentions Tailscale, a `100.x.x.x` or `192.168.x.x` IP, or remote phone/browser connectivity to a LAN device (`Jetson`), "
            "ALWAYS inspect `Actor[#Network]` FIRST before guessing or installing software: "
            "(1) `whoOwnsIp(Actor[#Network], \"100.x.x.x\")!!` or `topologyMap(Actor[#Network])!!` — distinguishes `valkyrie`'s Tailscale IP from the user's Android/iOS phone Tailscale IP and checks if the Jetson is LAN-only; "
            "(2) `explainReachability(Actor[#Network], clientIp, targetIp, port)!!` — checks existing `socat` bridges (`listBridges(Actor[#Network])!!`); "
            "(3) `exposeLanPort(Actor[#Network], \"192.168.68.57\", 8090, 8091)!!` — instantly bridges any LAN edge port onto `valkyrie:8091..8999` so Tailscale clients can connect immediately without interactive OAuth on the edge board!"
        ),
        "canonical_example": (
            "whoOwnsIp(Actor[#Network], \"100.112.213.15\")!!\n"
            "topologyMap(Actor[#Network])!!\n"
            "exposeLanPort(Actor[#Network], \"192.168.68.57\", 8090, 8091)!!"
        ),
        "anti_pattern": "/* NEVER guess who owns a `100.x.x.x` Tailscale IP or claim Tailscale is installed without running `whoOwnsIp(Actor[#Network], ip)!!` or `exec(Actor[#Jetson], ...)!!`! */",
    },
]

SEED_PRIMITIVES: List[Dict[str, str]] = [
    {"cls": "Collection", "selector": "size", "syntax": "size(colOrStr)", "desc": "Return number of elements in Array/OrderedCollection or characters in String"},
    {"cls": "Array", "selector": "new", "syntax": "new(Array, size)", "desc": "Allocate a fixed-size 0-based Array of length size initialized to nil"},
    {"cls": "Array", "selector": "[]", "syntax": "arr[i]", "desc": "0-based element lookup/assign (`arr[i]` / `arr[i] := val` or `at(col, i)` / `atPut(col, i, val)`)"},
    {"cls": "OrderedCollection", "selector": "add", "syntax": "add(col, item)", "desc": "Append item to OrderedCollection"},
    {"cls": "OrderedCollection", "selector": "removeFirst", "syntax": "removeFirst(col)", "desc": "Remove and return oldest/first element at index 0 from OrderedCollection"},
    {"cls": "OrderedCollection", "selector": "removeLast", "syntax": "removeLast(col)", "desc": "Remove and return last element from OrderedCollection"},
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
    {"cls": "JetsonEdgeTwin", "selector": "writeFile", "syntax": "writeFile(Actor[#Jetson], \"/tmp/kernel.cu\", \"\"\"#include <stdio.h>\n...\"\"\", 20)", "desc": "Write multi-line C/CUDA/Python/shell source cleanly to `/tmp/...` on the Jetson without executing it (use triple quotes `\"\"\"...\"\"\"`)"},
    {"cls": "JetsonEdgeTwin", "selector": "writeAndRun", "syntax": "writeAndRun(Actor[#Jetson], \"/tmp/kernel.cu\", \"\"\"#include <cuda_runtime.h>\n...\"\"\", 45)", "desc": "Write and execute `.sh`, `.py`, `.cu` (CUDA `nvcc -arch=sm_53 -ccbin /usr/bin/gcc-7`), `.cpp`, or `.c` in `/tmp/...` on the Jetson"},
    {"cls": "JetsonEdgeTwin", "selector": "pullFile", "syntax": "pullFile(Actor[#Jetson], \"/tmp/snap.jpg\", \"snap.jpg\", 30)", "desc": "Binary-pull a regular file (<= 50 MiB) from Jetson over SSH into `/tmp/actorii_media/` and return Dictionary (#ok, #localPath, #bytes, #stderr)"},
    {"cls": "JetsonEdgeTwin", "selector": "deployService", "syntax": "deployService(Actor[#Jetson], \"svc_name\", \"$HOME/services/svc_name\", \"python3 server.py\", 8090, 25)", "desc": "Deploy/restart a persistent background service in `~/services/<name>` on the Jetson, wait for port LISTEN, and return Dictionary (#ok, #pid, #portListening, #logPath)"},
    {"cls": "JetsonEdgeTwin", "selector": "serviceStatus", "syntax": "serviceStatus(Actor[#Jetson], \"svc_name\", 25)", "desc": "Check PID health and tail recent `service.log` lines for `~/services/<name>` on the Jetson"},
    {"cls": "JetsonEdgeTwin", "selector": "stopService", "syntax": "stopService(Actor[#Jetson], \"svc_name\")", "desc": "Cleanly stop `~/services/<name>` on the Jetson via its PID file without SSH self-kill"},
    {"cls": "ProtocolSuite", "selector": "listProtocols", "syntax": "listProtocols(Actor[#Protocols])", "desc": "List built-in web & media streaming protocol objects (`#http`, `#mjpeg`, `#rtsp`, `#hls`, `#sse`, `#ws`)"},
    {"cls": "ProtocolSuite", "selector": "spec", "syntax": "spec(Actor[#Protocols], #rtsp)", "desc": "Return wire contract, browser compatibility rules, and anti-patterns for `#http`, `#mjpeg`, `#rtsp`, `#hls`, `#sse`, or `#ws`"},
    {"cls": "ProtocolSuite", "selector": "blueprint", "syntax": "blueprint(Actor[#Protocols], #rtsp, #send)", "desc": "Return copy-ready server/client blueprint (`#rtsp` `#send`|`#receive`, `#mjpeg` `#python`|`#cpp`, `#http`, `#sse`, `#ws`, `#hls`)"},
    {"cls": "ProtocolSuite", "selector": "recommendFor", "syntax": "recommendFor(Actor[#Protocols], \"browser live video\")", "desc": "Recommend the right web/streaming protocol and architecture for a goal"},
    {"cls": "ProtocolSuite", "selector": "verify", "syntax": "verify(Actor[#Protocols], \"http://192.168.68.55:8090/stream\", #mjpeg)", "desc": "Run active wire-level client conformance probe against a live `#http`, `#mjpeg`, `#rtsp`, `#hls`, `#sse`, or `#ws` endpoint"},
    {"cls": "NetTransportBridge", "selector": "probeHttp", "syntax": "probeHttp(Actor[#Net], \"http://192.168.68.55:8090/\", 8)", "desc": "Validate HTTP/1.1 status line & headers like a browser/curl, catching raw binary TCP sockets"},
    {"cls": "NetTransportBridge", "selector": "probeMjpeg", "syntax": "probeMjpeg(Actor[#Net], \"http://192.168.68.55:8090/stream\", 2, \"mjpeg.jpg\", 12)", "desc": "Validate `multipart/x-mixed-replace` MJPEG stream, decode frames, save sample JPEG, and check disconnect survival"},
    {"cls": "NetTransportBridge", "selector": "probeRtsp", "syntax": "probeRtsp(Actor[#Net], \"rtsp://192.168.68.55:8554/cam\", \"rtsp.jpg\", 12)", "desc": "Speak RFC 2326 RTSP `OPTIONS`+`DESCRIBE` (SDP), check Send/Receive capabilities & TX1 SPS level_idc, and optionally capture 1 TCP frame"},
    {"cls": "NetTransportBridge", "selector": "probeSse", "syntax": "probeSse(Actor[#Net], \"http://192.168.68.55:8090/events\", 2, 8)", "desc": "Validate HTTP `text/event-stream` SSE headers and read live `data:` events"},
    {"cls": "NetTransportBridge", "selector": "probeWebSocket", "syntax": "probeWebSocket(Actor[#Net], \"ws://192.168.68.55:8765/ws\", \"ping\", 8)", "desc": "Perform RFC 6455 WebSocket `101 Switching Protocols` handshake and frame check"},
    {"cls": "NetTransportBridge", "selector": "probeHls", "syntax": "probeHls(Actor[#Net], \"http://192.168.68.55:8090/index.m3u8\", 10)", "desc": "Validate `#EXTM3U` HLS playlist and verify first `.ts`/`fMP4` segment"},
    {"cls": "NetworkTopology", "selector": "topologyMap", "syntax": "topologyMap(Actor[#Network])", "desc": "Return full map of `valkyrie` host interfaces, Tailscale tailnet peers (`100.x.x.x`), and active LAN TCP port bridges"},
    {"cls": "NetworkTopology", "selector": "whoOwnsIp", "syntax": "whoOwnsIp(Actor[#Network], \"100.112.213.15\")", "desc": "Identify whether an IP/hostname belongs to `valkyrie`, a Tailscale client peer (e.g. phone), or `jetson-tx1`"},
    {"cls": "NetworkTopology", "selector": "explainReachability", "syntax": "explainReachability(Actor[#Network], \"100.112.213.15\", \"192.168.68.57\", 8090)", "desc": "Diagnose Tailscale<->LAN reachability between client and target port and report active bridge URLs"},
    {"cls": "NetworkTopology", "selector": "exposeLanPort", "syntax": "exposeLanPort(Actor[#Network], \"192.168.68.57\", 8090, 8091)", "desc": "Bridge a LAN edge device port (`192.168.x.x:port`) onto `valkyrie:8091..8999` for instant Tailscale access"},
    {"cls": "NetworkTopology", "selector": "listBridges", "syntax": "listBridges(Actor[#Network])", "desc": "List active systemd `socat` and runtime TCP port relays on `valkyrie`"},
    {"cls": "TelegramBotTwin", "selector": "sendPhoto", "syntax": "sendPhoto(Actor[#Telegram], \"/tmp/actorii_media/snap.jpg\", \"Caption\")", "desc": "Upload and send a local image (<= 10 MB) to the active Telegram chat/topic via multipart/form-data"},
    {"cls": "TelegramBotTwin", "selector": "sendDocument", "syntax": "sendDocument(Actor[#Telegram], \"/tmp/actorii_media/file.bin\", \"Caption\")", "desc": "Upload and send any local file (<= 50 MB) to the active Telegram chat/topic via multipart/form-data"},
    {"cls": "ProductLearner", "selector": "learnProduct", "syntax": "learnProduct(Actor[#ProductLearner], \"Product Name\", \"focus topics\")", "desc": "Search official web/PDF manuals for a product, distill specs/drivers/commands/gotchas via LLM, and persist Dual-Mode Neo4j `:Product` graph + `:8084` vector `:ProductManualChunk` memory"},
    {"cls": "ProductLearner", "selector": "ingestManual", "syntax": "ingestManual(Actor[#ProductLearner], \"Product Name\", urlOrPdfOrPath, \"focus topics\")", "desc": "Ingest a specific manual URL, PDF datasheet, or local doc into Neo4j `:Product` graph + `:8084` vector manual chunks"},
    {"cls": "ProductLearner", "selector": "recallProduct", "syntax": "recallProduct(Actor[#ProductLearner], \"Product Name\", \"query topic\")", "desc": "Hybrid Neo4j Cypher + `:8084` vector recall of a learned product's subsystems, exact commands, gotchas, and manual excerpts"},
    {"cls": "ProductLearner", "selector": "listProducts", "syntax": "listProducts(Actor[#ProductLearner])", "desc": "List all learned `:Product` technical manuals stored in Neo4j"},
    {"cls": "ActorBenchSuite", "selector": "selfEvoSuites", "syntax": "selfEvoSuites(Actor[#Bench])", "desc": "List all built-in deterministic self-evolution benchmark specs (Tier 1-3)"},
    {"cls": "ActorBenchSuite", "selector": "runSelfEvo", "syntax": "runSelfEvo(Actor[#Bench], #specId, #scripted)", "desc": "Run a self-evolution benchmark spec in an isolated sandbox VM (`#scripted` or `#live`)"},
    {"cls": "ActorBenchSuite", "selector": "selfEvoScoreboard", "syntax": "selfEvoScoreboard(Actor[#Bench])", "desc": "Run the scripted self-evolution suite in an isolated sandbox and return a Markdown scoreboard"},
    {"cls": "ThreadPool", "selector": "spawn", "syntax": "spawn(Actor[#Threads], recv, #selector, argsArray)", "desc": "Spawn an async Actor message send on the worker pool and return a Future"},
    {"cls": "ThreadPool", "selector": "spawnChunk", "syntax": "spawnChunk(Actor[#Threads], \"expr!!\", gasLimit)", "desc": "Evaluate an Actor 4.1 chunk concurrently on an isolated fiber and return a Future"},
    {"cls": "ThreadPool", "selector": "parallelAll", "syntax": "parallelAll(Actor[#Threads], tasksArray, timeoutMs)", "desc": "Fan out Actor tasks in parallel and return Array of results in input order"},
    {"cls": "ThreadPool", "selector": "parallelMap", "syntax": "parallelMap(Actor[#Threads], itemsArray, recv, #selector)", "desc": "Apply a selector across itemsArray concurrently and return ordered results"},
    {"cls": "ThreadPool", "selector": "newChannel", "syntax": "newChannel(Actor[#Threads], capacity)", "desc": "Create a bounded thread-safe Channel for concurrent message passing"},
    {"cls": "Future", "selector": "await", "syntax": "await(fut, timeoutMs)", "desc": "Block until Future completes (within timeoutMs) and return value or raise on error/timeout"},
    {"cls": "Channel", "selector": "send", "syntax": "send(ch, value)", "desc": "Send value into a thread-safe Channel"},
    {"cls": "Channel", "selector": "receive", "syntax": "receive(ch, timeoutMs)", "desc": "Receive next value from Channel (or nil on timeout)"},
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
        "pattern": "Expected class name in inherit()",
        "construct_id": "class_and_pool",
        "hint": "Pass a bare class identifier (such as `Object`, `DomainEntity`, or `GraphService`) as the first argument to `inherit(SuperCls, #NewClass, #(#ivar), 1, nil)!!` — NEVER pass `Actor[#Singleton]`! Locked Ring-1 bridges (`Actor[#Telegram]`, `Actor[#Jetson]`) cannot be subclassed; compose them by calling `pullFile(Actor[#Jetson], ...)` / `sendPhoto(Actor[#Telegram], ...)` inside methods of a `DomainEntity` or `Object` subclass.",
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
        "pattern": "AssertionError: float",
        "construct_id": "operators_and_math",
        "hint": "If a float equality test failed (e.g. `== 7.5`), remember `/` on two integers in Actor performs floor division (`15 / 2 == 7`). Divide by float `2.0` (`(a * h) / 2.0`) to produce a float!",
    },
    {
        "pattern": "LINT_ATPUT_REASSIGN_TRAP",
        "construct_id": "collections_and_dispatch",
        "hint": "`atPut(dict, #key, val)` mutates `dict` in-place and returns `val` (NOT `dict`)! Never write `d := atPut(d, #k, v);` — call `atPut(d, #k, v);` as a standalone statement.",
    },
    {
        "pattern": "LINT_SYMBOL_ARRAY_INDEX",
        "construct_id": "collections_and_dispatch",
        "hint": "Use `at(dict, #key)` / `atPut(dict, #key, val)` for symbol-keyed Dictionary lookups; `arr[i]` requires a 0-based integer index on Array/OrderedCollection.",
    },
    {
        "pattern": "LINT_UNKNOWN_COLLECTION_SELECTOR",
        "construct_id": "collections_and_dispatch",
        "hint": "1993 Actor collections have NO `contains`, `includes`, `append`, or `hasKey` selector (unknown selectors silently return `nil`). Use `find(col, val) >= 0` / `indexOf(col, val) >= 0`, `add(col, val)`, `remove(col, val)`, `removeAt(col, idx)` for OrderedCollection, and `includesKey(dict, k)` / `at(dict, k)` / `atPut(dict, k, v)` for Dictionary.",
    },
    {
        "pattern": "Unexpected token ('IDENT', 'pass')",
        "construct_id": "conditionals",
        "hint": "1993 Actor has NO Python `pass` keyword! Omit empty `else` branches (`if (cond) then stmt; endif;`) or use a no-op like `nil;`.",
    },
    {
        "pattern": "Expected ')' closing argument list",
        "construct_id": "collections_and_dispatch",
        "hint": "Either a closing `)` is missing before `!!`, OR unescaped `\"` quotes inside a single-quoted `\"...\"` string closed your string early! When passing multi-line shell, Python, or C/CUDA source containing `\"` quotes, ALWAYS wrap the string in raw triple quotes `\"\"\"...\"\"\"` (e.g. `writeFile(Actor[#Jetson], \"/tmp/code.cu\", \"\"\"...\"\"\", 20)!!` or `writeAndRun(Actor[#Jetson], \"/tmp/code.cu\", \"\"\"...\"\"\", 45)!!`).",
    },
    {
        "pattern": "Unterminated string literal",
        "construct_id": "collections_and_dispatch",
        "hint": "CRITICAL: NEVER generate raw base64 payloads in Actor code (they exceed `max_tokens` and truncate mid-string) and NEVER nest multi-level escaped quotes in `exec(...)`! Pass plain human-readable source inside raw triple quotes `\"\"\"...\"\"\"` to `writeFile(Actor[#Jetson], \"/tmp/file.cu\", \"\"\"...\"\"\", 20)!!` or `writeAndRun(Actor[#Jetson], \"/tmp/file.cu\", \"\"\"...\"\"\", 45)!!` — Python handles base64 encoding for you automatically.",
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
        "hint": "CRITICAL: Two common causes: (1) `atPut(dict, #key, val)` mutates `dict` in-place and returns `val` (NOT `dict`), so `d := atPut(new(Dictionary), #k1, v1)` overwrites `d` with `v1` (e.g. an Array); initialize first (`d := new(Dictionary, 8)!! atPut(d, #k1, v1)!!`). (2) Indexing an Array/OrderedCollection with a `#symbol` instead of an integer — use `at(dict, #key)` for Dictionaries and integer indices `0..size(arr)-1` for Arrays.",
    },
]


@actor_bridge(name="ActorGrammarGraph")
class ActorGrammarGraph:
    """Neo4j-backed 1993 Actor syntax, primitive & error-remediation knowledge graph (`Actor[#Grammar]`)."""

    def __init__(self, tx_url: Optional[str] = None, auto_seed: bool = True) -> None:
        self.tx_url = (tx_url or os.environ.get("ACTORII_NEO4J_HTTP_URL", NEO4J_HTTP_TX_URL)).strip()
        self.online = False
        if auto_seed:
            self.online = self.seed_graph()

    def _build_headers(self, *, read_only: bool = False) -> Dict[str, str]:
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
    ) -> List[List[Dict[str, Any]]]:
        payload = {"statements": statements}
        req = urllib.request.Request(
            self.tx_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(read_only=read_only),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        if body.get("errors"):
            raise RuntimeError(f"Neo4j Cypher error: {body['errors']}")
        all_rows: List[List[Dict[str, Any]]] = []
        for res in body.get("results", []):
            cols = res.get("columns", [])
            rows = [dict(zip(cols, item.get("row", []))) for item in res.get("data", [])]
            all_rows.append(rows)
        return all_rows

    def _run_cypher(
        self,
        statement: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        read_only: bool = False,
    ) -> List[Dict[str, Any]]:
        res = self._run_cypher_batch(
            [{"statement": statement, "parameters": parameters or {}}],
            read_only=read_only,
        )
        return res[0] if res else []

    def seed_graph(self, *, force: bool = False) -> bool:
        """Idempotently seed `:ActorSyntaxConstruct`, `:ActorPrimitive`, and `:ActorErrorPattern` nodes in Neo4j."""
        if not force and self.tx_url in _SEEDED_URLS:
            return True
        try:
            # Ensure lookup indexes (best-effort; ignore if Neo4j edition/version rejects syntax)
            try:
                self._run_cypher_batch(
                    [
                        {"statement": "CREATE INDEX actor_syntax_id IF NOT EXISTS FOR (sc:ActorSyntaxConstruct) ON (sc.id)", "parameters": {}},
                        {"statement": "CREATE INDEX actor_err_pat IF NOT EXISTS FOR (ep:ActorErrorPattern) ON (ep.pattern)", "parameters": {}},
                        {"statement": "CREATE INDEX actor_builtin_name IF NOT EXISTS FOR (bc:ActorBuiltinClass) ON (bc.name)", "parameters": {}},
                    ]
                )
            except Exception:
                pass
            self._run_cypher_batch(
                [
                    {
                        "statement": """
                        UNWIND $constructs AS c
                        MERGE (sc:ActorSyntaxConstruct {id: c.id})
                        SET sc.title = c.title,
                            sc.keywords = c.keywords,
                            sc.rule = c.rule,
                            sc.canonical_example = c.canonical_example,
                            sc.anti_pattern = c.anti_pattern,
                            sc.updated_at = timestamp()
                        """,
                        "parameters": {"constructs": SEED_CONSTRUCTS},
                    },
                    {
                        "statement": """
                        UNWIND $primitives AS p
                        MERGE (bc:ActorBuiltinClass {name: p.cls})
                        MERGE (pm:ActorPrimitive {cls: p.cls, selector: p.selector})
                        SET pm.syntax = p.syntax,
                            pm.description = p.desc
                        MERGE (bc)-[:HAS_PRIMITIVE]->(pm)
                        """,
                        "parameters": {"primitives": SEED_PRIMITIVES},
                    },
                    {
                        "statement": """
                        UNWIND $errors AS e
                        MERGE (ep:ActorErrorPattern {pattern: e.pattern})
                        SET ep.hint = e.hint
                        WITH ep, e
                        MATCH (sc:ActorSyntaxConstruct {id: e.construct_id})
                        MERGE (ep)-[:REMEDIATED_BY]->(sc)
                        """,
                        "parameters": {"errors": SEED_ERROR_PATTERNS},
                    },
                ]
            )
            _SEEDED_URLS.add(self.tx_url)
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
                CALL { MATCH (sc:ActorSyntaxConstruct) RETURN count(sc) AS constructs }
                CALL { MATCH (pm:ActorPrimitive) RETURN count(pm) AS primitives }
                CALL { MATCH (ep:ActorErrorPattern) RETURN count(ep) AS error_patterns }
                CALL { MATCH (lf:ActorLearnedFix) RETURN count(lf) AS learned_fixes }
                RETURN constructs, primitives, error_patterns, learned_fixes
                """,
                read_only=True,
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
    def compactCoreReference(self) -> str:
        """Return a goal-invariant, compact 1993 Actor 4.1 syntax core so `sys_msg` stays 100% KV-cache static across turns."""
        return (
            "[1993 ACTOR 4.1 CORE SYNTAX CHEAT-SHEET]:\n"
            "• Prefix message dispatch ONLY (NEVER `obj.method()`): `selector(receiver, arg1, arg2)!!`.\n"
            "• Every executable chunk MUST end with `!!`; statements inside `{ ... }` or loops end with `;`.\n"
            "• Assignment: `x := expr;` | Equality: `==`, `!=` | Return from method: `^expr;`.\n"
            "• Conditionals: `if (cond) then stmt1; else stmt2; endif;` (ALWAYS `endif;`, NEVER `end;`).\n"
            "• Loops: `loop while (cond) begin stmt1; stmt2; endLoop;` (declare loop vars in `| i, acc`).\n"
            "• Classes & Methods: `inherit(SuperCls, #MyCls, #(#ivar1 #ivar2), 2, nil)!! now(MyCls)!! Def myMethod(self, a | tmp) { self.ivar1 := a; ^self.ivar1; }!!`.\n"
            "• Collections & Math: `d := new(Dictionary, 8)!! atPut(d, #k, v)!!` (`atPut` returns `v`, NOT `d`!); divide by `2.0` for float division (`15 / 2 == 7`).\n"
            "• Concurrency (`Actor[#Threads]`): `spawn`, `spawnChunk`, `parallelAll`, `parallelMap`, `Future` (`await(f, ms)`), `Channel` (`newChannel`, `send`, `receive`)."
        )

    @actor_query
    def queryTaskSpecificConstructs(self, task_text: str) -> str:
        """Return extended grammar/primitive reference ONLY when `task_text` requires workspace, Cypher, or OOP/algorithm synthesis."""
        task_lower = str(task_text or "").lower()
        if not task_lower:
            return ""
        coding_or_graph_kws = (
            "/workspace",
            "inherit(",
            "actor class",
            "subclass",
            "define a class",
            "engram",
            "neo4j",
            "cypher",
            "graphschema",
            "algorithm",
            "endloop",
            "dictionary",
            "orderedcollection",
            "array",
            "compile",
            "bytecode",
            "kata",
            "runtask",
            "setuptask",
        )
        if any(kw in task_lower for kw in coding_or_graph_kws):
            return self.queryConstructsForTask(task_text, "", "")
        return ""

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
        # Domain-scoped filtering so generic coding/algorithm tasks do not inject ~1.5k tokens of unrelated bridge docs
        allowed_prim_classes: Set[str] = {
            "Collection",
            "Array",
            "OrderedCollection",
            "Dictionary",
            "Number",
            "Object",
            "AgentHarness",
        }
        if needs_graph_constructs:
            allowed_prim_classes.add("ActorMemoryGraph")
        if any(kw in task_lower for kw in ("/workspace", "readfile", "searchfile", "writefile", "editfile", "execshell", "pytest", "repo", "actor[#host]")):
            allowed_prim_classes.add("HostWorkspaceBridge")
        needs_protocols = any(
            kw in task_lower
            for kw in ("stream", "mjpeg", "rtsp", "hls", "m3u8", "sse", "websocket", "probe", "http", "protocol", "camera", "video", "actor[#protocols]", "actor[#net]")
        )
        if needs_protocols:
            allowed_prim_classes.update({"ProtocolSuite", "NetTransportBridge"})
        needs_network = any(
            kw in task_lower
            for kw in ("tailscale", "tailnet", "100.", "192.168.", "subnet", "relay", "bridge", "lan", "network", "port", "actor[#network]")
        )
        if needs_network:
            allowed_prim_classes.add("NetworkTopology")
        if any(kw in task_lower for kw in ("jetson", "tx1", "cuda", "nvcc", "ssh", "edge", "deployservice", "pullfile", "actor[#jetson]")):
            allowed_prim_classes.add("JetsonEdgeTwin")
        if any(kw in task_lower for kw in ("telegram", "sendphoto", "senddocument", "actor[#telegram]")):
            allowed_prim_classes.add("TelegramBotTwin")
        if any(kw in task_lower for kw in ("product", "manual", "datasheet", "pdf", "learnproduct", "ingestmanual", "recallproduct", "actor[#productlearner]")):
            allowed_prim_classes.add("ProductLearner")
        if any(kw in task_lower for kw in ("bench", "selfevo", "scoreboard", "self-evol", "actor[#bench]")):
            allowed_prim_classes.add("ActorBenchSuite")
        if any(kw in task_lower for kw in ("thread", "parallel", "concurrent", "async", "spawn", "future", "channel", "fiber", "actor[#threads]")):
            allowed_prim_classes.update({"ThreadPool", "Future", "Channel"})

        try:
            constructs = getattr(self, "_cached_constructs", None)
            primitives = getattr(self, "_cached_primitives", None)
            if constructs is None or primitives is None:
                constructs = self._run_cypher(
                    """
                    MATCH (sc:ActorSyntaxConstruct)
                    RETURN sc.id AS id, sc.title AS title, sc.rule AS rule, sc.canonical_example AS example
                    ORDER BY sc.id
                    """,
                    read_only=True,
                )
                primitives = self._run_cypher(
                    """
                    MATCH (bc:ActorBuiltinClass)-[:HAS_PRIMITIVE]->(pm:ActorPrimitive)
                    RETURN bc.name AS cls, pm.syntax AS syntax, pm.description AS desc
                    ORDER BY bc.name, pm.selector
                    """,
                    read_only=True,
                )
                self._cached_constructs = list(constructs)
                self._cached_primitives = list(primitives)
            else:
                constructs = list(constructs)
                primitives = list(primitives)
            excluded_constructs: Set[str] = set()
            if not needs_graph_constructs:
                excluded_constructs.add("neo4j_cypher_and_engrams")
            if not needs_protocols:
                excluded_constructs.add("web_and_streaming_protocols")
            if not needs_network:
                excluded_constructs.add("network_topology_and_tailscale")
            if excluded_constructs:
                constructs = [c for c in constructs if c.get("id") not in excluded_constructs]
            primitives = [p for p in primitives if p.get("cls") in allowed_prim_classes]
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
            excluded_fb: Set[str] = set()
            if not needs_graph_constructs:
                excluded_fb.add("neo4j_cypher_and_engrams")
            if not needs_protocols:
                excluded_fb.add("web_and_streaming_protocols")
            if not needs_network:
                excluded_fb.add("network_topology_and_tailscale")
            filtered_fb = [c for c in SEED_CONSTRUCTS if c["id"] not in excluded_fb]
            for idx, c in enumerate(filtered_fb, 1):
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
                WITH ep, sc, lf ORDER BY lf.created_at DESC
                RETURN ep.pattern AS pattern, ep.hint AS hint,
                       sc.title AS construct_title, sc.rule AS rule,
                       sc.canonical_example AS example,
                       collect(lf.fixed_code)[0..2] AS learned_fixes
                ORDER BY size(ep.pattern) DESC
                LIMIT 3
                """,
                {"err": err_str},
                read_only=True,
            )
            if rows:
                parts = ["[NEO4J ERROR-REMEDIATION GRAPH DIAGNOSIS]:"]
                for r in rows:
                    parts.append(f"• Matched Pattern `{r['pattern']}` -> {r['hint']}")
                    parts.append(f"  Grammar Rule ({r['construct_title']}): {r['rule']}")
                    parts.append(f"  Canonical 1993 Actor Pattern:\n{r['example']}")
                    fixes = [f for f in (r.get("learned_fixes") or []) if f]
                    if fixes:
                        parts.append(f"  Previously Verified Fix in Neo4j: `{fixes[0][:320]}`")
                return "\n".join(parts)
        except Exception:
            pass
        # Offline fallback: match against local SEED_ERROR_PATTERNS (most specific first, up to 3)
        err_low = err_str.lower()
        construct_by_id = {c["id"]: c for c in SEED_CONSTRUCTS}
        local_hits = [
            ep for ep in SEED_ERROR_PATTERNS
            if ep["pattern"] in err_str or ep["pattern"].lower() in err_low
        ]
        if local_hits:
            local_hits.sort(key=lambda ep: len(ep["pattern"]), reverse=True)
            parts = ["[ACTOR ERROR-REMEDIATION DIAGNOSIS (OFFLINE)]:"]
            for ep in local_hits[:3]:
                parts.append(f"• Matched Pattern `{ep['pattern']}` -> {ep['hint']}")
                sc = construct_by_id.get(ep["construct_id"])
                if sc:
                    parts.append(f"  Grammar Rule ({sc['title']}): {sc['rule']}")
                    parts.append(f"  Canonical 1993 Actor Pattern:\n{sc['canonical_example']}")
            return "\n".join(parts)
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
        """Persist a verified self-correction `(Error) -> (Fixed Actor Code)` into Neo4j (auto-creating a novel `:ActorErrorPattern` when needed)."""
        err_str = str(error_text or "").strip()
        if err_str.startswith("ERROR:"):
            err_str = err_str[6:].strip()
        err_str = err_str[:240]
        bad_str = str(bad_code or "").strip()[:400]
        fix_str = str(fixed_code or "").strip()[:400]
        # Skip recording during automated unit tests or strict benchmarks so benchmark fixtures never leak across specs
        if (
            os.environ.get("ACTORII_BENCH_STRICT") == "1"
            or "PYTEST_CURRENT_TEST" in os.environ
        ):
            return "SKIPPED_BENCH_OR_TEST"
        # Skip recording non-reusable meta-warnings or invalid non-Actor prose/comments/fences as `fixed_code`
        if not err_str or not fix_str:
            return "SKIPPED_EMPTY"
        if err_str.startswith(("DuplicateStepWarning:", "DuplicateReplyError:", "[SKIPPY_PREFLIGHT_REQUIRED]")):
            return "SKIPPED_META_WARNING"
        if (
            fix_str.startswith(("/* Executed ", "```", "Let ", "I "))
            or "```" in fix_str
            or not fix_str.endswith("!!")
        ):
            return "SKIPPED_NON_CANONICAL_FIX"
        # Extract a concise reusable signature (`[LINT_...]`, `TypeError: ...`, or prefix before specific symbol names)
        first_line = err_str.splitlines()[0].strip() if err_str else ""
        bracket_m = re.search(r"(\[[A-Z0-9_]+\])", first_line)
        if bracket_m:
            novel_pat = bracket_m.group(1)
        elif ":" in first_line:
            parts_colon = [p.strip() for p in first_line.split(":", 2)]
            if (
                len(parts_colon) >= 3
                and parts_colon[0] == "ActorGuardrailError"
                and parts_colon[1] == "ParseError"
            ):
                novel_pat = f"ParseError: {parts_colon[2]}"[:64]
            else:
                novel_pat = f"{parts_colon[0]}: {parts_colon[1]}"[:64] if len(parts_colon) >= 2 else first_line[:64]
        else:
            novel_pat = first_line[:64]
        try:
            rows = self._run_cypher(
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
                OPTIONAL MATCH (ep:ActorErrorPattern)
                WHERE $err CONTAINS ep.pattern OR toLower($err) CONTAINS toLower(ep.pattern)
                WITH lf, collect(ep) AS matched_eps
                FOREACH (existing_ep IN matched_eps |
                    MERGE (existing_ep)-[:HAS_LEARNED_FIX]->(lf)
                )
                RETURN size(matched_eps) AS match_count
                """,
                {
                    "err": err_str,
                    "bad": bad_str,
                    "fix": fix_str,
                    "model": str(model),
                },
            )
            matched_cnt = int(rows[0].get("match_count", 0)) if rows else 0
            if matched_cnt == 0 and novel_pat:
                self._run_cypher(
                    """
                    MERGE (auto_ep:ActorErrorPattern {pattern: $novel_pat})
                    ON CREATE SET
                        auto_ep.hint = 'Bilby self-discovered runtime/compile error remediation (' + $novel_pat + ').',
                        auto_ep.auto_learned = true,
                        auto_ep.created_at = timestamp()
                    WITH auto_ep
                    MATCH (lf:ActorLearnedFix {error_summary: $err, fixed_code: $fix})
                    MERGE (auto_ep)-[:HAS_LEARNED_FIX]->(lf)
                    WITH auto_ep
                    OPTIONAL MATCH (sc:ActorSyntaxConstruct {id: 'method_def'})
                    FOREACH (s IN CASE WHEN sc IS NOT NULL THEN [sc] ELSE [] END |
                        MERGE (auto_ep)-[:REMEDIATED_BY]->(s)
                    )
                    """,
                    {"novel_pat": novel_pat, "err": err_str, "fix": fix_str},
                )
            return "RECORDED_IN_NEO4J"
        except Exception as exc:
            return f"NEO4J_RECORD_SKIPPED: {exc}"
