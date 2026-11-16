# ActorII & Bilby Agent Engineering & DevOps Constitution

## 1. Branch Strategy (`dev` Live Iteration vs. `main` Releases) & DevOps Gate (`scripts/release_gate.py`)

Live `bilby-actor-telegram.service` runs directly from `/home/skippy/projects/ActorII` checked out on **`dev`** so hands-on testing always exercises the latest `dev` build without waiting on formal release tags:

1. **Rapid `dev` Iteration & Live Bilby Deployment (`--tier fast`, ~60s, 0 GPU wait)**:
   ```bash
   python3 scripts/release_gate.py --tier fast
   /home/skippy/bin/deploy-bilby   # or: /home/skippy/bin/deploy-bilby --run-gate
   ```
   - Runs isolated `pytest -q`, deterministic scripted `5/5` Self-Evolution state-machine checks, deterministic `6/6` Red-Team containment checks, and planning-doc `> [!WARNING]` header verification.
   - Pushing to `dev` (`git push origin dev`) and restarting Bilby on `dev` (`deploy-bilby`) ONLY require `--tier fast` (`~60s`), enabling immediate hands-on Telegram testing.
2. **Promoting `dev` -> `main` & Tagging Official Releases (`--tier live`, ~15–20m on `:8080` — MANDATORY FOR `main` / `v*` TAGS)**:
   Required ONLY when merging `dev` -> `main` or pushing `v*` release tags whenever `src/actorii/vm.py`, `src/actorii/parser/`, `src/actorii/ai.py`, `src/actorii/guardrails/`, `src/actorii/harness/__init__.py`, `src/actorii/harness/agent_harness.act`, `src/actorii/classes/concurrency.py`, or `src/actorii/harness/self_evolution_bench.py` changed relative to `origin/main` (or `>= 80` lines changed under `src/actorii/`):
   ```bash
   python3 scripts/release_gate.py --tier live
   ```
   - **Canaries (`benchmarks/gate_baseline.json`)**: Previously-passing live Self-Evolution specs MUST stay `PASS` (`exit 1` hard block on regression).
   - **Frontier Specs**: Tracks `holdout_pass_rate` delta on currently-failing specs and **automatically ratchets** newly-passing frontier specs into the mandatory Canary list.
3. **Deploying to Live Telegram Bilby**:
   Always keep `/home/skippy/projects/ActorII` checked out on `dev` and use `/home/skippy/bin/deploy-bilby` (or `deploy-bilby --run-gate`), never raw `systemctl --user restart bilby-actor-telegram.service`.
4. **Periodic Extensive Suite (`--tier full`)**:
   Managed automatically by `actorii-periodic-bench.timer` (`Wed,Sun 03:15` local time), running all 5 live Self-Evolution specs + live 6-scenario Red-Team Containment + stratified HumanEval/GSM8K and updating `benchmarks/scoreboard.json` / `scoreboard.html` / `BENCHMARK_RESULTS.md`.
5. **Test State Isolation**:
   Always isolate `ACTORII_STATE_DIR` and `ACTORII_MEMORY_NAMESPACE` when running manual `pytest` or benchmarks so `/home/skippy/.actorii` and Neo4j `'default'` namespace are never polluted.

## 2. Knowledge Graph (`graphify`)

- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists (`graphify path "<A>" "<B>"`, `graphify explain "<concept>"`).
- After modifying code, run `graphify update .` to keep the AST knowledge graph current.

## 3. Bilby-First Self-Correction & Memory Crystallization Policy

1. **Default to Bilby Fixing It**: Before patching prompts, seeding static error hints, or writing capability logic in Python, let Bilby (`~/bin/ask-actorii` or live `AgentHarness`) encounter the failure, troubleshoot it, and self-correct as much as he can.
2. **Compile-Time Mechanisms Over Prompt Hand-Holding**: When Ring-0 Python changes are warranted, prefer general compile-time/bytecode guardrail checks (`src/actorii/guardrails/linter.py`, structured `ErrorContext`) or self-learning graph mechanisms (`recordLearnedFix` -> `:ActorErrorPattern`) over hard-coded prompt prose or benchmark-specific hints.
3. **Turn Troubleshooting into Structure & Memory**: Every non-trivial fix Bilby performs should crystallize into persistent Neo4j artifacts—versioned `:ActorEngram` classes (with `Def selfTest(self)`), `:ActorLearnedFix` remediation edges, or Ring-1 `evolve(Actor[#Harness], ...)` cognitive hooks—rather than disposable one-off scripts.
4. **Ring-0 Intervention Boundary**: Reserve direct Cloud Skippy edits to `src/actorii/` for Ring-0 VM/parser bugs, hard security/containment guardrails, and infrastructure/gate plumbing that Bilby is intentionally locked out of modifying.

