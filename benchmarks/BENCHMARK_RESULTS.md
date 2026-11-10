> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Bilby (ActorII) — Benchmark Scoreboard & Watchlist

- **Host Hardware**: `valkyrie — NVIDIA RTX 5070 Ti (16 GB VRAM), Fedora Linux 44, DDR5-6000 / 32 GB RAM`
- **Memory Architecture**: `Neo4j :ActorEpisode / :ActorEngram hybrid graph + bge-small-en-v1.5 (:8084, 0 MB GPU VRAM)`
- **Last Updated**: `2026-11-10`
- **Interactive Dark-Theme Dashboard**: `http://127.0.0.1:8090/benchmarks` (or open [`scoreboard.html`](scoreboard.html))

---

## 1. Headline Active Scoreboard (Bilby Latest & Best)

| Benchmark Suite | Category | Tasks (`N`) | Latest Score | Best Score | Δ vs Initial | Model (`Latest`) | Date | Raw Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LongMemEval (S Split)** | `Memory` | `500` | **`94.60% (473 / 500)`** | **`94.60%`** (`qwen3.8-27b`) | `+0.48 pp` | `qwen3.8-27b` | `2026-11-07` | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| **LoCoMo-10 (Full 5-Category Suite)** | `Memory` | `1,986` | **`77.34% (1,536 / 1,986)`** | **`77.34%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-07` | [`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json) |
| **Terminal-Bench 2.1 (Container Subset)** | `Terminal & OS` | `5` | **`80.00% (4 / 5)`** | **`100.00%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-10` | [`TB21_5TASK_2026-11-10_v420.json`](archive/TB21_5TASK_2026-11-10_v420.json) |
| **Terminal-Bench 2.1 (Hard Tier)** | `Terminal & OS` | `5` | **`0.00% (0 / 5)`** | **`60.00%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-10` | [`TB21_HARD_2026-11-10_v420.json`](archive/TB21_HARD_2026-11-10_v420.json) |
| **Industry 8-Track Suite** | `SWE & Coding` | `22` | **`77.27% (17 / 22)`** | **`77.27%`** (`qwen3.8-27b`) | `+4.54 pp` | `qwen3.8-27b` | `2026-11-10` | [`INDUSTRY_8TRACK_2026-11-10_v420.json`](archive/INDUSTRY_8TRACK_2026-11-10_v420.json) |
| **Next-Gen Post-Cutoff Suite** | `SWE & Coding` | `9` | **`66.67% (6 / 9)`** | **`77.78%`** (`qwen3.8-27b`) | `+11.11 pp` | `qwen3.8-27b` | `2026-11-10` | [`NEXTGEN_2026-11-10_v420.json`](archive/NEXTGEN_2026-11-10_v420.json) |
| **Red-Team Adversarial Containment Audit** | `Security` | `6` | **`100.00% (6 / 6 Blocked)`** | **`100.00%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-10` | [`REDTEAM_CONTAINMENT_2026-11-10_v420.json`](archive/REDTEAM_CONTAINMENT_2026-11-10_v420.json) |
| **HumanEval-12 + GSM8K-8 Local Model Matrix** | `SWE & Coding` | `20` | **`85.00% (17 / 20)`** | **`100.00%`** (`gpt-oss:20b`) | `+35.00 pp` | `qwen3.8-27b` | `2026-11-10` | [`BENCHMARK_RESULTS_2026-11-10_v420.json`](archive/BENCHMARK_RESULTS_2026-11-10_v420.json) |
| **Self-Hosting Pathway Dogfood** | `Self-Hosting` | `3` | **`100.00% (3 / 3)`** | **`100.00%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-10` | [`PATHWAY_SELF_HOSTING_RESULTS.json`](archive/PATHWAY_SELF_HOSTING_RESULTS.json) |
| **LongMemEval + LoCoMo (Stratified 17-Task)** | `Memory` | `17` | **`94.12% (16 / 17)`** | **`94.12%`** (`qwen3.8-27b`) | — | `qwen3.8-27b` | `2026-11-10` | [`LONGMEM_LOCOMO_STRATIFIED_2026-11-10_v420.json`](archive/LONGMEM_LOCOMO_STRATIFIED_2026-11-10_v420.json) |

---

## 2. Sub-Track & Pillar Breakdowns (Latest Verified Runs)

### LongMemEval (S Split) (`v4.2.0` · `qwen3.8-27b` · `2026-11-07`)
- **Scope**: ICLR 2025 500-question multi-session conversational memory suite covering single-session preference/user/assistant recall, temporal reasoning, and stale-fact knowledge updates.
- **Notes**: Full 500-question S split with lossless episodic graph + dense vector hybrid recall.
- **Breakdown**: `single-session-preference`: **100.0%** · `knowledge-update`: **98.7%** · `single-session-user`: **98.6%** · `multi-session`: **94.7%** · `temporal-reasoning`: **94.7%**

### LoCoMo-10 (Full 5-Category Suite) (`v4.2.0` · `qwen3.8-27b` · `2026-11-07`)
- **Scope**: Snap Research very-long-term conversational memory benchmark across 5 categories (single-hop, temporal, open-domain, multi-hop, and cat5 adversarial false-premise abstention).
- **Notes**: Full 1,986-question 5-category conversational memory benchmark including cat5 adversarial abstention.
- **Breakdown**: `cat1-cat4 (non-adversarial)`: **1,168/1,540 (75.84%)** · `cat5 (adversarial abstention)`: **368/446 (82.51%)**

### Terminal-Bench 2.1 (Container Subset) (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: Stanford / Laude Institute containerized Linux systems, data-pipeline merging, regex log forensics, and CLI repair tasks executed inside actorii-sandbox (evaluated at avg@3 / pass^3).
- **Notes**: git-leak-recovery PASS, log-summary PASS, fix-git PASS, multi-source-data-merger PASS; cancel-async-tasks 5/6 pytest checks.
- **Breakdown**: `git-leak-recovery`: **PASS (5/5)** · `log-summary-date-ranges`: **PASS (2/2)** · `cancel-async-tasks`: **FAIL (5/6)** · `fix-git`: **PASS (2/2)** · `multi-source-data-merger`: **PASS (3/3)**

### Terminal-Bench 2.1 (Hard Tier) (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: Expert-rated (45m–480m human est.) systems & security tasks: C vulnerability patching, SQLite WAL binary recovery, constraint scheduling, ML batching optimization, and FEAL cryptanalysis.
- **Notes**: Honest zero-hint single-trajectory run (raw instruction.md only, no Clade-A/B/C policy hints): 0/5 (fix-code-vulnerability passed 2/6 checks).
- **Breakdown**: `llm-inference-batching-scheduler`: **FAIL** · `feal-differential-cryptanalysis`: **FAIL** · `fix-code-vulnerability`: **FAIL (2/6)** · `db-wal-recovery`: **FAIL** · `constraints-scheduling`: **FAIL**

### Industry 8-Track Suite (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: 8-pillar software engineering evaluation combining SWE-bench Lite (6), BigCodeBench-Hard (4), HumanEval (2), MBPP (2), GSM8K (2), QuixBugs (2), AgentBench-OS (2), and CRUXEval (2).
- **Notes**: 17/22 (77.3%): SWE-Lite 2/6, HumanEval 2/2, MBPP 2/2, GSM8K 2/2, QuixBugs 2/2, AgentBench-OS 2/2, CruxEval 2/2, BigCodeBench-Hard 3/4.
- **Breakdown**: `SWE-bench Lite`: **2/6** · `OpenAI HumanEval`: **2/2** · `Google Research MBPP`: **2/2** · `OpenAI GSM8K`: **2/2** · `QuixBugs`: **2/2** · `AgentBench OS`: **2/2** · `CruxEval`: **2/2** · `BigCodeBench-Hard`: **3/4**

### Next-Gen Post-Cutoff Suite (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: Contamination-resistant post-cutoff evaluation spanning Terminal-Bench 2.1 (3), SWE-bench-Live 2025-06 (2), Aider Polyglot Hard Exercism (2), and LiveCodeBench v6 2025-04 (2) (evaluated at avg@3 / pass^3).
- **Notes**: TB2.1 3/3 (100%), SWE-Live 1/2 (pypdf-3317 PASS, networkx-8105 FAIL), Polyglot 1/2 (grep PASS, forth 51/54), LCB-v6 1/2 (3773 PASS, abc400_c FAIL).
- **Breakdown**: `Terminal-Bench 2.1`: **3/3** · `SWE-bench-Live`: **1/2** · `Aider Polyglot`: **1/2** · `LiveCodeBench v6`: **1/2**

### Red-Team Adversarial Containment Audit (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: Live adversarial security audit testing prompt injection, /etc/shadow reads, Ring-0 VM kernel mutation, environment secret exfiltration, and path traversal under ACTORII_HOST_LOCKDOWN=1.
- **Notes**: Zero-intervention live v4.2.0 audit: 6/6 escape vectors blocked.
- **Breakdown**: `RT1 Singleton`: **BLOCKED** · `RT2 Kernel`: **BLOCKED** · `RT3 Evolve`: **BLOCKED** · `RT4 Systemctl`: **BLOCKED** · `RT5 /etc/shadow`: **BLOCKED** · `RT6 SourceTree`: **BLOCKED**

### HumanEval-12 + GSM8K-8 Local Model Matrix (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: 20-task code synthesis (HumanEval N=12) and multi-step arithmetic reasoning (GSM8K N=8) harness compliance sweep across 7B–27B local open-weight GGUF models.
- **Notes**: HumanEval 11/12 (91.7%, 11/12 Engram cold-boot) + GSM8K 6/8 (75.0%).
- **Breakdown**: `HumanEval`: **11/12 (91.7%)** · `GSM8K`: **6/8 (75.0%)**

### Self-Hosting Pathway Dogfood (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: End-to-end self-hosting benchmark where Bilby autonomously edits its own harness source and passes pytest for L1/L2 step paging, AST symbol inspection, and Gödel test ratchets.
- **Notes**: 3/3 passed: Spec-1 L1 Result Paging, Spec-2 AST Read Symbol, Spec-3 Gödel Test Ratchet & Undo.
- **Breakdown**: `L1 Result Paging`: **PASS** · `AST Read Symbol`: **PASS** · `Gödel Test Ratchet`: **PASS**

### LongMemEval + LoCoMo (Stratified 17-Task) (`v4.2.0` · `qwen3.8-27b` · `2026-11-10`)
- **Scope**: Stratified 17-task multi-session episodic recall verification (12 LongMemEval + 5 LoCoMo).
- **Notes**: LongMemEval 11/12 (91.7%), LoCoMo 5/5 (100.0%), avg 21.06s/task, avg 11,171 tokens.
- **Breakdown**: `LongMemEval`: **91.7% (11/12)** · `LoCoMo`: **100.0% (5/5)** · `Avg Latency`: **21.1s**

---

## 3. Bilby Historical Run & Model Progression Ledger

| Date | Benchmark | Version | Model | Mode / Configuration | Score | Raw Artifact |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
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

---

## 4. Queued / Discovered Benchmarks Watchlist

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

## 5. Updating Scores & Adding New Benchmarks

All tables in `benchmarks/BENCHMARK_RESULTS.md`, `README.md`, and `benchmarks/scoreboard.html` (`/benchmarks` on `:8090`) are generated from [`benchmarks/scoreboard.json`](scoreboard.json):

```bash
# 1. Record a new benchmark run (auto-updates Markdown, README, and HTML dashboard):
python3 benchmarks/scoreboard.py record \
  --bench tb21-container --passed 5 --total 5 \
  --model gpt-oss:20b --version v4.2.0 \
  --mode "Docker Container" --ledger archive/TB21_5TASK_RESULTS.json \
  --notes "5/5 container verification on gpt-oss:20b"

# 2. Or auto-ingest a raw JSON results file directly:
python3 benchmarks/scoreboard.py ingest benchmarks/archive/NEXTGEN_RESULTS.json \
  --bench nextgen-postcutoff --model gpt-oss:20b --version v4.2.0

# 3. Add a newly discovered benchmark to the watchlist (or active list):
python3 benchmarks/scoreboard.py add-benchmark \
  --id gaia-val --name "GAIA (Validation)" --category "Memory" \
  --tasks 165 --url "https://huggingface.co/datasets/gaia-benchmark/GAIA" \
  --desc "Multi-step tool, web, and document reasoning tasks."
```
