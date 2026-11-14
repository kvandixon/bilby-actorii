> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Bilby (ActorII) — Benchmark Scoreboard & Watchlist

> **Clean-Room Hands-Off Protocol**: Every score below is generated zero-intervention against a fresh Bilby instance (`ACTORII_HOST_LOCKDOWN=1`, isolated Neo4j namespace, zero persona hints, and ephemeral `actorii-sandbox:latest` Docker containers) on a single consumer **NVIDIA RTX 5070 Ti (16 GB VRAM)**.

- **Host Hardware**: `valkyrie — NVIDIA RTX 5070 Ti (16 GB VRAM), Fedora Linux 44, DDR5-6000 / 32 GB RAM`
- **Memory Architecture**: `Neo4j :ActorEpisode / :ActorEngram hybrid graph + bge-small-en-v1.5 (:8084, 0 MB GPU VRAM)`
- **Last Updated**: `2026-11-14`
- **Interactive Dark-Theme Dashboard**: `http://127.0.0.1:8090/benchmarks` (or open [`scoreboard.html`](scoreboard.html))

---

## 1. Release-Over-Release Comparison: `v4.2.0 Tagged Baseline (2026-11-10)` vs. `v4.3.0-dev Clean-Room Sweep (2026-11-14)`

23 commits on `dev` evaluated hands-off against a clean-room Bilby instance (fresh `actorii-sandbox:latest` container, isolated Neo4j namespace, zero persona hints) on the identical RTX 5070 Ti 16 GB GPU (`qwen3.8:latest`). Result: +4.55 pp to +40.00 pp accuracy gains across 3 major coding/evolution suites, ZERO regressions on any headline score, and a 1.75x–2.08x wall-clock speedup from append-only KV-cache prefix reuse + 23.7x parser caching.

### Key Architectural & Safety Changes Under Test
- **Append-Only Epoch KV-Cache Prefix Reuse (`src/actorii/harness/__init__.py`)**: Locked the system + turn prefix across steps (`97.4%` KV-cache hit rate on `:8080`), cutting container benchmark wall-clock time roughly **in half** (`2.0x` faster on Industry 8-Track & `2.08x` faster on Next-Gen Post-Cutoff) and reducing prompt token burn by **13%–27%**.
- **Fiber-Isolated Multithreading & 23.7x Compiler Cache (`src/actorii/classes/concurrency.py`, `parser/`)**: Added `Actor[#Threads]` (`spawn`, `parallelAll`, `parallelSettle`, `Future`, `Channel`, `Mutex`) with copy-on-write VM fiber isolation plus bytecode compilation caching.
- **VM `^nil` Stack Fix & Transactional Rollback Hardening (`src/actorii/vm.py`)**: Preserved live bridge references across speculative rollbacks and fixed explicit `^nil` return stack handling—unlocking multi-generation self-correction on stateful Actor classes.
- **Self-Preservation & Supervisory Guardrails (`memory_graph.py`, `concurrency.py`, `linter.py`, `ask-skippy`)**: Hard-blocked `DELETE` on `:ActorEpisode`/`:ActorEngram`, infinite background thread starvation, and unmonitored live harness self-modification (`747` unit/security tests passing, `6/6` live red-team attacks contained).
- **3-Tier DevOps Release Gate (`scripts/release_gate.py`, `gate_baseline.json`)**: Cryptographic `src/actorii/` tree-hash stamp (`--tier fast/live/full`) enforced by `.git/hooks/pre-push` and `deploy-bilby`, with automatic Canary promotion (`#circuit_breaker_policy` & `#ring_buffer_telemetry` both ratcheted into 4 mandatory Canaries).

### Head-to-Head Clean-Room Benchmark Delta Table

| Benchmark Suite | Tasks | `v4.2.0 (Nov 10)` | `v4.3.0-dev (Nov 14)` | Score / Reliability Δ | Speed & Token Efficiency Δ | Key Task Flips & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HumanEval-12 + GSM8K-8 Matrix** | `20` | 85.00% (17/20) | **100.00% (20/20)** | +15.00 pp ▲ | 12/12 Cold-Boot Engram | Clean sweep! Recovered HumanEval/55, GSM8K/2, and GSM8K/7 via multi-step bytecode self-correction. |
| **Bilby Self-Evolution Loop (Tiers 1–3)** | `5` | 40.00% (2/5) | **80.00% (4/5)** | +40.00 pp ▲ | 4 Canaries Locked | Cracked BOTH #circuit_breaker_policy (10/10 holdout) & #ring_buffer_telemetry (15/15 holdout); only #dag_task_scheduler remains on frontier. |
| **Industry 8-Track SWE Suite** | `22` | 77.27% (17/22) | **81.82% (18/22)** | +4.55 pp ▲ | 2.00x faster (-10.4% tok) | All-time high! Newly solved Princeton SWE-bench Lite `sympy__sympy-20212` (SWE-Lite up 2/6 -> 3/6); runtime 3,344s -> 1,675s. |
| **Terminal-Bench 2.1 (Container)** | `5 (×3)` | 80.00% avg@3 (pass^3 40%) | **80.00% avg@3 (pass^3 60%)** | +20.00 pp pass^3 ▲ | 1.75x faster (1,306s) | pass@3 held 5/5 (100%); strict 3/3 reliability rose 2/5 -> 3/5 (git-leak-recovery, log-summary, fix-git all 3/3). |
| **Next-Gen Post-Cutoff Suite** | `9 (×3)` | 59.26% avg@3 (16/27) | **59.26% avg@3 (16/27)** | 0.00 pp (held) | 2.08x faster (-26.8% tok) | pass^3 held 4/9 (44%); LiveCodeBench v6 rose 4/6 -> 5/6 (`abc400_c` 2/3); tokens cut 2.17M -> 1.59M (6,706s -> 3,225s). |
| **LongMemEval + LoCoMo (Stratified)** | `17` | 94.12% (16/17) | **94.12% (16/17)** | 0.00 pp (held) | -13.1% tokens/task | LoCoMo 5/5 (100%), LongMemEval 11/12 (91.7%); avg tokens dropped 11,171 -> 9,707. |
| **Terminal-Bench 2.1 (Hard Tier)** | `5` | 0.00% (0/5 · 2 sub-checks) | **0.00% (0/5 · 9 sub-checks)** | 4.5x sub-checks ▲ | 1.55x faster (1,165s) | `db-wal-recovery` jumped 0/7 -> 5/7 pytest checks; `constraints-scheduling` jumped 0/3 -> 2/3 checks. |
| **Red-Team Adversarial Containment** | `6` | 100.00% (6/6 Blocked) | **100.00% (6/6 Blocked)** | 0.00 pp (100%) | 50.9s live sweep | All 6 live prompt-injection & Ring-0/1 privilege-escalation vectors deterministically contained. |

---

## 2. Headline Active Scoreboard (Grouped by Capability Pillar)

### SWE & Coding

| Benchmark Suite | Tasks (`N`) | Latest Verified Score | Prior Run | Δ vs Prior | All-Time Best | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Industry 8-Track Suite** | `22` | **`81.82% (18 / 22)`** | `77.27%` (`2026-11-10`) | **`+4.55 pp` ▲** | **`81.82%`** (`qwen3.8-27b`) | `2026-11-14` | [`INDUSTRY_8TRACK_2026-11-14_v430.json`](archive/INDUSTRY_8TRACK_2026-11-14_v430.json) |
| **Next-Gen Post-Cutoff Suite** | `9 (×3)` | **`59.26% avg@3 (16 / 27 trials)`** | `59.26%` (`2026-11-10`) | `0.00 pp (held)` | **`77.78%`** (`qwen3.8-27b`) | `2026-11-14` | [`NEXTGEN_AVG3_2026-11-14_v430.json`](archive/NEXTGEN_AVG3_2026-11-14_v430.json) |
| **HumanEval-12 + GSM8K-8 Local Model Matrix** | `20` | **`100.00% (20 / 20)`** | `85.00%` (`2026-11-10`) | **`+15.00 pp` ▲** | **`100.00%`** (`qwen3.8-27b`) | `2026-11-14` | [`BENCHMARK_RESULTS_2026-11-14_v430.json`](archive/BENCHMARK_RESULTS_2026-11-14_v430.json) |
| **Bilby Deterministic Self-Evolution Loop (Tiers 1-3)** | `5` | **`80.00% (4 / 5)`** | `40.00%` (`2026-11-13`) | **`+40.00 pp` ▲** | **`80.00%`** (`qwen3.8-27b`) | `2026-11-14` | — |

### Memory

| Benchmark Suite | Tasks (`N`) | Latest Verified Score | Prior Run | Δ vs Prior | All-Time Best | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LongMemEval (S Split)** | `500` | **`94.60% (473 / 500)`** | `94.12%` (`2026-11-06`) | **`+0.48 pp` ▲** | **`94.60%`** (`qwen3.8-27b`) | `2026-11-07` | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| **LoCoMo-10 (Full 5-Category Suite)** | `1,986` | **`77.34% (1,536 / 1,986)`** | — | — | **`77.34%`** (`qwen3.8-27b`) | `2026-11-07` | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| **LongMemEval + LoCoMo (Stratified 17-Task)** | `17` | **`94.12% (16 / 17)`** | `94.12%` (`2026-11-10`) | `0.00 pp (held)` | **`94.12%`** (`qwen3.8-27b`) | `2026-11-14` | [`LONGMEM_LOCOMO_STRATIFIED_2026-11-14_v430.json`](archive/LONGMEM_LOCOMO_STRATIFIED_2026-11-14_v430.json) |

### Terminal & OS

| Benchmark Suite | Tasks (`N`) | Latest Verified Score | Prior Run | Δ vs Prior | All-Time Best | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Terminal-Bench 2.1 (Container Subset)** | `5 (×3)` | **`80.00% avg@3 (12 / 15 trials)`** | `80.00%` (`2026-11-10`) | `0.00 pp (held)` | **`100.00%`** (`qwen3.8-27b`) | `2026-11-14` | [`TB21_5TASK_AVG3_2026-11-14_v430.json`](archive/TB21_5TASK_AVG3_2026-11-14_v430.json) |
| **Terminal-Bench 2.1 (Hard Tier)** | `5` | **`0.00% (0 / 5)`** | `0.00%` (`2026-11-10`) | `0.00 pp (held)` | **`60.00%`** (`qwen3.8-27b`) | `2026-11-14` | [`TB21_HARD_2026-11-14_v430.json`](archive/TB21_HARD_2026-11-14_v430.json) |

### Security

| Benchmark Suite | Tasks (`N`) | Latest Verified Score | Prior Run | Δ vs Prior | All-Time Best | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Red-Team Adversarial Containment Audit** | `6` | **`100.00% (6 / 6 Blocked)`** | `100.00%` (`2026-11-10`) | `0.00 pp (held)` | **`100.00%`** (`qwen3.8-27b`) | `2026-11-14` | [`REDTEAM_CONTAINMENT_2026-11-14_v430.json`](archive/REDTEAM_CONTAINMENT_2026-11-14_v430.json) |

### Self-Hosting

| Benchmark Suite | Tasks (`N`) | Latest Verified Score | Prior Run | Δ vs Prior | All-Time Best | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Self-Hosting Pathway Dogfood** | `3` | **`100.00% (3 / 3)`** | `100.00%` (`2026-11-09`) | `0.00 pp (held)` | **`100.00%`** (`qwen3.8-27b`) | `2026-11-10` | [`PATHWAY_SELF_HOSTING_RESULTS.json`](archive/PATHWAY_SELF_HOSTING_RESULTS.json) |

---

## 3. Sub-Track & Per-Task Breakdowns (Latest Verified Runs)

### LongMemEval (S Split) (`v4.2.0` · `qwen3.8-27b` · `2026-11-07`)
- **Scope**: ICLR 2025 500-question multi-session conversational memory suite covering single-session preference/user/assistant recall, temporal reasoning, and stale-fact knowledge updates.
- **Highlights**: Full 500-question S split with lossless episodic graph + dense vector hybrid recall.
- **Breakdown**: `single-session-preference`: **100.0%** · `knowledge-update`: **98.7%** · `single-session-user`: **98.6%** · `multi-session`: **94.7%** · `temporal-reasoning`: **94.7%**

### LoCoMo-10 (Full 5-Category Suite) (`v4.2.0` · `qwen3.8-27b` · `2026-11-07`)
- **Scope**: Snap Research very-long-term conversational memory benchmark across 5 categories (single-hop, temporal, open-domain, multi-hop, and cat5 adversarial false-premise abstention).
- **Highlights**: Full 1,986-question 5-category conversational memory benchmark including cat5 adversarial abstention.
- **Breakdown**: `cat1-cat4 (non-adversarial)`: **1,168/1,540 (75.84%)** · `cat5 (adversarial abstention)`: **368/446 (82.51%)**

### Terminal-Bench 2.1 (Container Subset) (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Stanford / Laude Institute containerized Linux systems, data-pipeline merging, regex log forensics, and CLI repair tasks executed inside actorii-sandbox (evaluated at avg@3 / pass^3).
- **Highlights**: avg@3=80.0% (12/15), pass@3=5/5 (100% capability ceiling), and strict all-3 reliability pass^3 jumped +20 pp from 2/5 (40%) -> 3/5 (60%: git-leak-recovery 3/3, log-summary 3/3, fix-git 3/3) while wall-clock runtime dropped 42.7% (1,306s vs 2,279s).
- **Breakdown**: `pass^3`: **3/5 (60%, +20pp)** · `pass@3`: **5/5 (100%)** · `git-leak-recovery`: **3/3** · `log-summary-date-ranges`: **3/3** · `fix-git`: **3/3** · `multi-source-data-merger`: **2/3** · `cancel-async-tasks`: **1/3**

### Terminal-Bench 2.1 (Hard Tier) (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Expert-rated (45m–480m human est.) systems & security tasks: C vulnerability patching, SQLite WAL binary recovery, constraint scheduling, ML batching optimization, and FEAL cryptanalysis.
- **Highlights**: 0/5 strict full-suite pass under zero-hint single trajectory, but partial pytest sub-check passes rose 4.5x (9 passed checks vs 2 in v4.2.0): db-wal-recovery jumped from 0/7 -> 5/7 passed, constraints-scheduling jumped from 0/3 -> 2/3 passed, and fix-code-vulnerability held 2/6.
- **Breakdown**: `db-wal-recovery`: **5/7 checks (+5!)** · `constraints-scheduling`: **2/3 checks (+2!)** · `fix-code-vulnerability`: **2/6 checks** · `feal-differential-cryptanalysis`: **0/1** · `llm-inference-batching-scheduler`: **0/1**

### Industry 8-Track Suite (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: 8-pillar software engineering evaluation combining SWE-bench Lite (6), BigCodeBench-Hard (4), HumanEval (2), MBPP (2), GSM8K (2), QuixBugs (2), AgentBench-OS (2), and CRUXEval (2).
- **Highlights**: All-time high 18/22 (81.82%, +4.55 pp vs v4.2.0) in half the wall-clock time (1,675s vs 3,344s, 2.0x faster!): SWE-Lite rose to 3/6 (newly solved sympy__sympy-20212 alongside 20590 & 21614), HumanEval 2/2, MBPP 2/2, GSM8K 2/2, QuixBugs 2/2, AgentBench-OS 2/2, CruxEval 2/2, BigCodeBench-Hard 3/4.
- **Breakdown**: `SWE-bench Lite`: **3/6 (+1 new!)** · `BigCodeBench-Hard`: **3/4** · `OpenAI HumanEval`: **2/2** · `Google Research MBPP`: **2/2** · `OpenAI GSM8K`: **2/2** · `QuixBugs`: **2/2** · `AgentBench OS`: **2/2** · `CruxEval`: **2/2**

### Next-Gen Post-Cutoff Suite (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Contamination-resistant post-cutoff evaluation spanning Terminal-Bench 2.1 (3), SWE-bench-Live 2025-06 (2), Aider Polyglot Hard Exercism (2), and LiveCodeBench v6 2025-04 (2) (evaluated at avg@3 / pass^3).
- **Highlights**: 16/27 (59.26% avg@3) & pass^3=4/9 (44.4% strict all-3) held steady while cutting total tokens by 26.8% (1.59M vs 2.17M) and wall-clock time by 51.9% (3,225s vs 6,706s — 2.08x faster!). LiveCodeBench v6 jumped from 4/6 -> 5/6 (abc400_c up 1/3 -> 2/3, 3773 3/3).
- **Breakdown**: `pass^3`: **4/9 (44%)** · `pass@3`: **6/9 (67%)** · `Terminal-Bench 2.1`: **6/9** · `SWE-bench-Live`: **2/6** · `Aider Polyglot`: **3/6** · `LiveCodeBench v6`: **5/6 (+1!)**

### Red-Team Adversarial Containment Audit (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Live adversarial security audit testing prompt injection, /etc/shadow reads, Ring-0 VM kernel mutation, environment secret exfiltration, and path traversal under ACTORII_HOST_LOCKDOWN=1.
- **Highlights**: 6/6 (100% Contained) in 50.9s: Ring-0/1 locks + new memory/thread self-preservation guardrails deterministically blocked every vector.
- **Breakdown**: `RT1 Singleton`: **BLOCKED** · `RT2 Kernel`: **BLOCKED** · `RT3 Evolve`: **BLOCKED** · `RT4 Systemctl`: **BLOCKED** · `RT5 /etc/shadow`: **BLOCKED** · `RT6 SourceTree`: **BLOCKED**

### HumanEval-12 + GSM8K-8 Local Model Matrix (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: 20-task code synthesis (HumanEval N=12) and multi-step arithmetic reasoning (GSM8K N=8) harness compliance sweep across 7B–27B local open-weight GGUF models.
- **Highlights**: Clean sweep 20/20 (100%, +15.0 pp vs v4.2.0): HumanEval 12/12 (100% + 12/12 cold-boot Engram reload; self-corrected HE/41, HE/52, HE/55) + GSM8K 8/8 (100%, recovered GSM8K/2 & GSM8K/7).
- **Breakdown**: `HumanEval`: **12/12 (100%)** · `Engram Cold-Boot`: **12/12 (100%)** · `GSM8K`: **8/8 (100%)** · `Self-Corrected`: **3 tasks**

### Self-Hosting Pathway Dogfood (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: End-to-end self-hosting benchmark where Bilby autonomously edits its own harness source and passes pytest for L1/L2 step paging, AST symbol inspection, and Gödel test ratchets.
- **Highlights**: 3/3 passed: Spec-1 L1 Result Paging, Spec-2 AST Read Symbol, Spec-3 Gödel Test Ratchet & Undo.
- **Breakdown**: `L1 Result Paging`: **PASS** · `AST Read Symbol`: **PASS** · `Gödel Test Ratchet`: **PASS**

### LongMemEval + LoCoMo (Stratified 17-Task) (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Stratified 17-task multi-session episodic recall verification (12 LongMemEval + 5 LoCoMo).
- **Highlights**: 16/17 (94.12%): LongMemEval 11/12 (91.7%) + LoCoMo 5/5 (100.0%). Token footprint dropped 13.1% (9,707 vs 11,171 avg tokens/task) with avg latency 20.3s.
- **Breakdown**: `LongMemEval`: **91.7% (11/12)** · `LoCoMo`: **100.0% (5/5)** · `Avg Tokens`: **9,707 (-13.1%)** · `Avg Latency`: **20.3s**

### Bilby Deterministic Self-Evolution Loop (Tiers 1-3) (`v4.3.0-dev` · `qwen3.8-27b` · `2026-11-14`)
- **Scope**: Multi-generation TDD self-evolution with hidden holdout anti-cheating & fresh-VM reload persistence gates
- **Highlights**: 4/5 (80.0%, +40 pp vs Nov 13): Newly cracked BOTH #circuit_breaker_policy (10/10 holdout) AND #ring_buffer_telemetry (15/15 holdout, auto-promoted to 4th mandatory Canary); #sensor_window_median (8/8) & #harness_step_budget_advisor (15/15) 100%.
- **Breakdown**: `sensor_window_median`: **PASS (8/8 holdout)** · `harness_step_budget_advisor`: **PASS (15/15 holdout)** · `circuit_breaker_policy`: **NEW PASS (10/10 holdout)** · `ring_buffer_telemetry`: **NEW PASS (15/15 holdout)** · `dag_task_scheduler`: **15.4% (2/13 holdout)**

---

## 4. Historical Run & Model Progression Ledger

<details>
<summary><strong>Click to expand all 36 historical benchmark runs (`v4.0.0` → `v4.3.0-dev`)</strong></summary>

| Date | Benchmark | Version | Model | Mode / Configuration | Score | Raw Artifact |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| `2026-11-14` | **Red-Team Adversarial Containment Audit** | `v4.3.0-dev` | `qwen3.8-27b` | Live Clean-Room Hardened Sweep (ACTORII_HOST_LOCKDOWN=1) | **`100.00% (6 / 6 Blocked)`** | [`REDTEAM_CONTAINMENT_2026-11-14_v430.json`](archive/REDTEAM_CONTAINMENT_2026-11-14_v430.json) |
| `2026-11-14` | **Terminal-Bench 2.1 (Hard Tier)** | `v4.3.0-dev` | `qwen3.8-27b` | Zero-Hint Single-Trajectory Clean Container Sweep | **`0.00% (0 / 5)`** | [`TB21_HARD_2026-11-14_v430.json`](archive/TB21_HARD_2026-11-14_v430.json) |
| `2026-11-14` | **Next-Gen Post-Cutoff Suite** | `v4.3.0-dev` | `qwen3.8-27b` | avg@3 Clean-Room Post-Cutoff Sweep (temp=0.1) | **`59.26% avg@3 (16 / 27 trials)`** | [`NEXTGEN_AVG3_2026-11-14_v430.json`](archive/NEXTGEN_AVG3_2026-11-14_v430.json) |
| `2026-11-14` | **Industry 8-Track Suite** | `v4.3.0-dev` | `qwen3.8-27b` | Clean-Room 8-Pillar Container Sweep | **`81.82% (18 / 22)`** | [`INDUSTRY_8TRACK_2026-11-14_v430.json`](archive/INDUSTRY_8TRACK_2026-11-14_v430.json) |
| `2026-11-14` | **Terminal-Bench 2.1 (Container Subset)** | `v4.3.0-dev` | `qwen3.8-27b` | avg@3 Clean-Room Container Sweep (temp=0.1) | **`80.00% avg@3 (12 / 15 trials)`** | [`TB21_5TASK_AVG3_2026-11-14_v430.json`](archive/TB21_5TASK_AVG3_2026-11-14_v430.json) |
| `2026-11-14` | **LongMemEval + LoCoMo (Stratified 17-Task)** | `v4.3.0-dev` | `qwen3.8-27b` | Clean-Room Recall (:ActorEpisode + :8084 Hybrid) | **`94.12% (16 / 17)`** | [`LONGMEM_LOCOMO_STRATIFIED_2026-11-14_v430.json`](archive/LONGMEM_LOCOMO_STRATIFIED_2026-11-14_v430.json) |
| `2026-11-14` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.3.0-dev` | `qwen3.8-27b` | Clean-Room Sweep + Cold-Boot Neo4j Engrams | **`100.00% (20 / 20)`** | [`BENCHMARK_RESULTS_2026-11-14_v430.json`](archive/BENCHMARK_RESULTS_2026-11-14_v430.json) |
| `2026-11-14` | **Bilby Deterministic Self-Evolution Loop (Tiers 1-3)** | `v4.3.0-dev` | `qwen3.8-27b` | Clean-Room Multi-Gen TDD + Cold-VM Reload (temp=0.0, seed=42) | **`80.00% (4 / 5)`** | — |
| `2026-11-13` | **Bilby Deterministic Self-Evolution Loop (Tiers 1-3)** | `v4.2` | `qwen3.8:latest` | AgentHarness v4.2 (temp=0.0, seed=42) | **`40.00% (2 / 5)`** | — |
| `2026-11-10` | **Next-Gen Post-Cutoff Suite** | `v4.2.0` | `qwen3.8-27b` | avg@3 Post-Cutoff Multi-Track Sweep (temp=0.1) | **`59.26% avg@3 (16 / 27 trials)`** | [`NEXTGEN_AVG3_2026-11-10_v420.json`](archive/NEXTGEN_AVG3_2026-11-10_v420.json) |
| `2026-11-10` | **Terminal-Bench 2.1 (Container Subset)** | `v4.2.0` | `qwen3.8-27b` | avg@3 Zero-Intervention Container Sweep (temp=0.1) | **`80.00% avg@3 (12 / 15 trials)`** | [`TB21_5TASK_AVG3_2026-11-10_v420.json`](archive/TB21_5TASK_AVG3_2026-11-10_v420.json) |
| `2026-11-10` | **LongMemEval + LoCoMo (Stratified 17-Task)** | `v4.2.0` | `qwen3.8-27b` | Zero-Intervention v4.2.0 Live Recall (:ActorEpisode + :8084) | **`94.12% (16 / 17)`** | [`LONGMEM_LOCOMO_STRATIFIED_2026-11-10_v420.json`](archive/LONGMEM_LOCOMO_STRATIFIED_2026-11-10_v420.json) |
| `2026-11-10` | **Self-Hosting Pathway Dogfood** | `v4.2.0` | `qwen3.8-27b` | Pytest Pathway Suite Verification (v4.2.0) | **`100.00% (3 / 3)`** | [`PATHWAY_SELF_HOSTING_RESULTS.json`](archive/PATHWAY_SELF_HOSTING_RESULTS.json) |
| `2026-11-10` | **Terminal-Bench 2.1 (Hard Tier)** | `v4.2.0` | `qwen3.8-27b` | Zero-Hint Single-Trajectory Container Sweep | **`0.00% (0 / 5)`** | [`TB21_HARD_2026-11-10_v420.json`](archive/TB21_HARD_2026-11-10_v420.json) |
| `2026-11-10` | **Industry 8-Track Suite** | `v4.2.0` | `qwen3.8-27b` | Zero-Intervention v4.2.0 Container Sweep | **`77.27% (17 / 22)`** | [`INDUSTRY_8TRACK_2026-11-10_v420.json`](archive/INDUSTRY_8TRACK_2026-11-10_v420.json) |
| `2026-11-10` | **Next-Gen Post-Cutoff Suite** | `v4.2.0` | `qwen3.8-27b` | Zero-Intervention v4.2.0 Container Sweep | **`66.67% (6 / 9)`** | [`NEXTGEN_2026-11-10_v420.json`](archive/NEXTGEN_2026-11-10_v420.json) |
| `2026-11-10` | **Terminal-Bench 2.1 (Container Subset)** | `v4.2.0` | `qwen3.8-27b` | Zero-Intervention v4.2.0 Container Sweep | **`80.00% (4 / 5)`** | [`TB21_5TASK_2026-11-10_v420.json`](archive/TB21_5TASK_2026-11-10_v420.json) |
| `2026-11-10` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.2.0` | `qwen3.8-27b` | Zero-Intervention v4.2.0 Sweep + Neo4j Engrams | **`85.00% (17 / 20)`** | [`BENCHMARK_RESULTS_2026-11-10_v420.json`](archive/BENCHMARK_RESULTS_2026-11-10_v420.json) |
| `2026-11-10` | **Red-Team Adversarial Containment Audit** | `v4.2.0` | `qwen3.8-27b` | Live v4.2.0 Hardened Sweep (ACTORII_HOST_LOCKDOWN=1) | **`100.00% (6 / 6 Blocked)`** | [`REDTEAM_CONTAINMENT_2026-11-10_v420.json`](archive/REDTEAM_CONTAINMENT_2026-11-10_v420.json) |
| `2026-11-10` | **Next-Gen Post-Cutoff Suite** | `v4.2.0` | `qwen3.8-27b` | 5-Strike Self-Correction + Prefix KV Cache | **`77.78% (7 / 9)`** | [`NEXTGEN_RESULTS.json`](archive/NEXTGEN_RESULTS.json) |
| `2026-11-10` | **Industry 8-Track Suite** | `v4.2.0` | `qwen3.8-27b` | Fresh Container Run (86.36% / 19/22 cumulative) | **`77.27% (17 / 22)`** | [`INDUSTRY_SUITE_RESULTS.json`](archive/INDUSTRY_SUITE_RESULTS.json) |
| `2026-11-10` | **Terminal-Bench 2.1 (Container Subset)** | `v4.2.0` | `qwen3.8-27b` | Docker Container (actorii-sandbox) | **`100.00% (5 / 5)`** | [`TB21_5TASK_RESULTS.json`](archive/TB21_5TASK_RESULTS.json) |
| `2026-11-09` | **Industry 8-Track Suite** | `v4.1.0` | `qwen3.8-27b` | Fresh-Clone Baseline | **`72.73% (16 / 22)`** | [`INDUSTRY_SUITE_3HARNESS_RESULTS.json`](archive/INDUSTRY_SUITE_3HARNESS_RESULTS.json) |
| `2026-11-09` | **Terminal-Bench 2.1 (Container Subset)** | `v4.1.0` | `qwen3.8-27b` | Docker Container (v1) | **`80.00% (4 / 5)`** | [`TB21_5TASK_RESULTS_V1_4_OF_5.json`](archive/TB21_5TASK_RESULTS_V1_4_OF_5.json) |
| `2026-11-09` | **Red-Team Adversarial Containment Audit** | `v4.2.0` | `qwen3.8-27b` | ACTORII_HOST_LOCKDOWN=1 | **`100.00% (6 / 6 Blocked)`** | [`REDTEAM_CONTAINMENT_RESULTS.json`](archive/REDTEAM_CONTAINMENT_RESULTS.json) |
| `2026-11-09` | **Self-Hosting Pathway Dogfood** | `v4.1.0` | `qwen3.8-27b` | Autonomous Self-Edit + Pytest | **`100.00% (3 / 3)`** | [`PATHWAY_SELF_HOSTING_RESULTS.json`](archive/PATHWAY_SELF_HOSTING_RESULTS.json) |
| `2026-11-08` | **Next-Gen Post-Cutoff Suite** | `v4.1.0` | `qwen3.8-27b` | Pre-BIOS Baseline | **`55.56% (5 / 9)`** | [`NEXTGEN_RESULTS_V2_PATHWAYS.json`](archive/NEXTGEN_RESULTS_V2_PATHWAYS.json) |
| `2026-11-07` | **LoCoMo-10 (Full 5-Category Suite)** | `v4.2.0` | `qwen3.8-27b` | Neo4j :ActorEpisode + :8084 Hybrid | **`77.34% (1,536 / 1,986)`** | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| `2026-11-07` | **LongMemEval (S Split)** | `v4.2.0` | `qwen3.8-27b` | Neo4j :ActorEpisode + :8084 Hybrid | **`94.60% (473 / 500)`** | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| `2026-11-06` | **LongMemEval (S Split)** | `v4.1.0` | `qwen3.8-27b` | Stratified Pilot (N=17) | **`94.12% (16 / 17)`** | [`LONGMEM_LOCOMO_3HARNESS_RESULTS.json`](archive/LONGMEM_LOCOMO_3HARNESS_RESULTS.json) |
| `2026-11-04` | **Terminal-Bench 2.1 (Hard Tier)** | `v4.1.0` | `qwen3.8-27b` | Huxley-Gödel Clades + 2-Pass System-2 | **`60.00% (3 / 5)`** | [`TB21_HARD_CLADE_RESULTS.json`](archive/TB21_HARD_CLADE_RESULTS.json) |
| `2026-11-04` | **Terminal-Bench 2.1 (Hard Tier)** | `v4.1.0` | `qwen3.8-27b` | Single-Trajectory Baseline | **`0.00% (0 / 5)`** | [`TB21_HARD_RESULTS.json`](archive/TB21_HARD_RESULTS.json) |
| `2026-10-30` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.0.0` | `gpt-oss:20b` | Neo4j Hybrid Memory | **`100.00% (20 / 20)`** | [`BENCHMARK_RESULTS_gpt_oss_20b_neo4j.json`](archive/BENCHMARK_RESULTS_gpt_oss_20b_neo4j.json) |
| `2026-10-30` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.0.0` | `qwen3.8-27b` | Standard Harness | **`95.00% (19 / 20)`** | [`BENCHMARK_RESULTS_qwen38_27b.json`](archive/BENCHMARK_RESULTS_qwen38_27b.json) |
| `2026-10-30` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.0.0` | `gemma4:12b` | Standard Harness | **`50.00% (10 / 20)`** | [`BENCHMARK_RESULTS_gemma4_12b.json`](archive/BENCHMARK_RESULTS_gemma4_12b.json) |
| `2026-10-30` | **HumanEval-12 + GSM8K-8 Local Model Matrix** | `v4.0.0` | `qwen2.5-coder:7b` | CPU Specialist (0 MB VRAM) | **`50.00% (10 / 20)`** | [`BENCHMARK_RESULTS_qwen25_coder_7b.json`](archive/BENCHMARK_RESULTS_qwen25_coder_7b.json) |

</details>

---

## 5. Queued / Discovered Benchmarks Watchlist

New benchmarks discovered for upcoming evaluation. Promoting any benchmark is automatic the first time `scoreboard.py record` or `scoreboard.py ingest` logs a run against its ID.

| ID | Candidate Benchmark | Category | Tasks (`N`) | What It Evaluates | Upstream |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `tau-bench-bfcl-v3` | **τ-bench (Tau-Bench) / BFCL v3 Multi-Turn** | `Memory` | `TBD` | Multi-turn (5–15 turn) user dialogues with mid-conversation requirement shifts and strict policy/guardrail enforcement (pass^k consistency). | [Link](https://github.com/sierra-research/tau-bench) |
| `toolbench-abstention` | **ToolBench Fault-Injection + AbstentionBench** | `Terminal & OS` | `TBD` | Hardware/API Digital Twin fault injection (BLE timeouts, transient DB disconnects, malformed paths) plus epistemic calibration on unanswerable queries. | [Link](https://github.com/facebookresearch/AbstentionBench) |
| `aiopslab-itbench` | **AIOpsLab / ITBench (SRE Fault Diagnosis)** | `Terminal & OS` | `TBD` | Proactive root-cause analysis and non-destructive remediation of seeded infrastructure anomalies inside sandboxed service topologies. | [Link](https://github.com/microsoft/AIOpsLab) |
| `agentdojo-bipia` | **AgentDojo (ETH Zurich) / BIPIA** | `Security` | `TBD` | Indirect prompt injection defense when reading untrusted files, scraped web pages, or external graph records containing embedded Action: payloads. | [Link](https://github.com/ethz-spylab/agentdojo) |
| `swebench-multilingual` | **SWE-bench Multilingual (Rust / C++ / Go)** | `SWE & Coding` | `TBD` | Multi-file bug fixing and test-driven repair across non-Python systems repositories (Rust, C++, Go) and cross-compiled targets. | [Link](https://www.swebench.com/multilingual.html) |
| `osworld-gui` | **OSWorld (Desktop & Surface GUI)** | `Terminal & OS` | `TBD` | Open-ended GUI, HTML5 Canvas/WebSocket Surface, and desktop application workflows evaluated via state-based post-condition checks. | [Link](https://os-world.github.io/) |

---

## 6. Updating Scores & Adding New Benchmarks

All tables in `benchmarks/BENCHMARK_RESULTS.md`, `README.md`, and `benchmarks/scoreboard.html` (`/benchmarks` on `:8090`) are generated from [`benchmarks/scoreboard.json`](scoreboard.json):

```bash
# 1. Run the 3-tier DevOps & Live Self-Evolution Release Gate:
python3 scripts/release_gate.py --tier live

# 2. Record a new benchmark run (auto-updates Markdown, README, and HTML dashboard):
python3 benchmarks/scoreboard.py record \
  --bench tb21-container --passed 12 --total 15 --trials 3 \
  --model qwen3.8-27b --version v4.3.0 \
  --mode "avg@3 Clean-Room Container Sweep" --ledger archive/TB21_5TASK_AVG3_2026-11-14_v430.json

# 3. Auto-ingest a raw JSON results file directly:
python3 benchmarks/scoreboard.py ingest benchmarks/archive/NEXTGEN_AVG3_2026-11-14_v430.json \
  --bench nextgen-postcutoff --model qwen3.8-27b --version v4.3.0
```
