# ActorII & Bilby Agent Engineering & DevOps Constitution

## 1. Mandatory Release & Benchmark Gate (`scripts/release_gate.py`)

Never push core changes or restart live `bilby-actor-telegram.service` without passing the automated DevOps gate:

1. **Minor / Test / Doc Changes (`--tier fast`, ~55s, 0 GPU)**:
   ```bash
   python3 scripts/release_gate.py --tier fast
   ```
   Runs isolated `pytest -q`, deterministic scripted `5/5` Self-Evolution state-machine checks, deterministic `6/6` Red-Team containment checks, and planning-doc `> [!WARNING]` header verification.
2. **Major Core / Harness Changes (`--tier live`, ~5–8m on `:8080` — MANDATORY)**:
   Required whenever `src/actorii/vm.py`, `src/actorii/parser/`, `src/actorii/ai.py`, `src/actorii/guardrails/`, `src/actorii/harness/__init__.py`, `src/actorii/harness/agent_harness.act`, `src/actorii/classes/concurrency.py`, or `src/actorii/harness/self_evolution_bench.py` are modified (or `>= 80` lines change under `src/actorii/`):
   ```bash
   python3 scripts/release_gate.py --tier live
   ```
   - **Canaries (`benchmarks/gate_baseline.json`)**: Previously-passing live Self-Evolution specs MUST stay `PASS` (`exit 1` hard block on regression).
   - **Frontier Specs**: Tracks `holdout_pass_rate` delta on currently-failing specs and **automatically ratchets** newly-passing frontier specs into the mandatory Canary list.
3. **Deploying to Live Telegram Bilby**:
   Always use `/home/skippy/bin/deploy-bilby` (or `deploy-bilby --run-gate`), never raw `systemctl --user restart bilby-actor-telegram.service`.
4. **Periodic Extensive Suite (`--tier full`)**:
   Managed automatically by `actorii-periodic-bench.timer` (`Wed,Sun 03:15` local time), running all 5 live Self-Evolution specs + live 6-scenario Red-Team Containment + stratified HumanEval/GSM8K and updating `benchmarks/scoreboard.json` / `scoreboard.html` / `BENCHMARK_RESULTS.md`.
5. **Test State Isolation**:
   Always isolate `ACTORII_STATE_DIR` and `ACTORII_MEMORY_NAMESPACE` when running manual `pytest` or benchmarks so `/home/skippy/.actorii` and Neo4j `'default'` namespace are never polluted.

## 2. Knowledge Graph (`graphify`)

- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists (`graphify path "<A>" "<B>"`, `graphify explain "<concept>"`).
- After modifying code, run `graphify update .` to keep the AST knowledge graph current.
