# Bilby (`ActorII v4.1`) — Metacircular Bytecode AI Agent & 1993 Actor VM

**Bilby (`ActorII v4.1`)** is a self-evolving, metacircular AI agent harness executing inside a clean-room 1993 Whitewater Actor 4.1 Smalltalk/Actor bytecode virtual machine, backed by **lossless Neo4j `:ActorEpisode` + `:ActorEngram` hybrid graph memory**.

Running **100% locally on a single 16 GB consumer GPU** (NVIDIA RTX 5070 Ti) with a 20B–27B open-weight model (`0 MB` GPU VRAM overhead for embeddings and memory indexing)—or connected to any cloud LLM (`OpenAI`, `OpenRouter`, `Gemini`, `Ollama`, `vLLM`)—Bilby achieves **#1 Global SOTA (`94.60%`) on `LongMemEval`** and **`77.34%` on `LOCOMO`** (`80.81%` combined across all 2,486 multi-session tasks).

[![LongMemEval SOTA](https://img.shields.io/badge/LongMemEval%20(500%20Tasks)-94.60%25%20(%231%20SOTA)-brightgreen.svg)](benchmarks/LONGMEM_LOCOMO_FULL_REPORT.md)
[![LOCOMO](https://img.shields.io/badge/LOCOMO%20(1986%20Tasks)-77.34%25%20(Local%2016GB)-blue.svg)](benchmarks/LONGMEM_LOCOMO_FULL_REPORT.md)
[![Test Suite](https://img.shields.io/badge/tests-600%2B%20passed-brightgreen.svg)](tests/)
[![Containment](https://img.shields.io/badge/containment-Defense--Up%20Docker%20%2B%203--Ring%20VM-orange.svg)](src/actorii/harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Global Leaderboard & Benchmark Comparison

### 1. `LongMemEval` (ICLR 2025 — Full 500-Task Multi-Session Suite) & `LOCOMO` (`1,986` Tasks)

| Agent / Memory Harness | Backing LLM & Compute Tier | `LongMemEval` (`500` Tasks) | `LOCOMO` (`1,986` Tasks) | Combined (`2,486` Tasks) | Evaluation Split & Scorer Disclosure |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Bilby (`ActorII v4.1`)** | **Local 20B–27B (`16 GB Consumer GPU`)** | **94.60% (`473/500`)** | **77.34% (`1,536/1,986`)** | **80.81% (`2,009/2,486`)** | Full 2,486-task corpus (`473/500` deterministic exact/lexical on `LongMemEval`; `1,050/1,986` deterministic + `486` `:8082` judge on `LOCOMO`) |
| **Hindsight** *(Prior Published SOTA)* | **Gemini-3 Pro** *(Frontier Cloud)* | 91.40% | — | — | Published benchmark |
| **Hindsight** | **o4-mini / o3** *(Frontier Reasoning Cloud)* | 89.00% | — | — | Published benchmark |
| **Hermes Agent** *(SQLite + FTS5)* | **Local 20B–27B (`16 GB GPU`)** | 83.33% *(`10/12` Smoke)* | — | — | 12-task stratified smoke subset |
| **Pi Coding Agent** *(Rolling Summary)* | **Local 20B–27B (`16 GB GPU`)** | 75.00% *(`9/12` Smoke)* | — | — | 12-task stratified smoke subset |
| **Zep / Graphiti** *(Temporal KG)* | **GPT-4o** *(Frontier Cloud)* | 71.20% | — | — | Published benchmark |
| **Raw Compaction Baseline** | **Local 20B–27B (`16 GB GPU`)** | 41.67% *(`5/12` Smoke)* | — | — | 12-task stratified smoke subset |

> **Academic Transparency & Offline Re-Scoring (`benchmarks/rejudge_saved_results.py`)**:
> Every question, gold answer, and raw model response across all 2,486 tasks is committed in [`benchmarks/LONGMEM_LOCOMO_FULL_RESULTS.json`](benchmarks/LONGMEM_LOCOMO_FULL_RESULTS.json). You can audit and re-score the entire saved run offline with **zero LLM judge calls** in `<3 seconds`:
> ```bash
> python3 benchmarks/rejudge_saved_results.py
> ```

### 2. Controlled 3-Harness Head-to-Head (`ActorII v4.1` vs. `Hermes` vs. `Pi` on Identical Local GPU)

All three harnesses were evaluated on the exact same local GPU model (`enable_thinking=False` normalizing proxy) across 5 benchmark dimensions:

| Benchmark Suite | **Bilby (`ActorII v4.1`)** | **Hermes Agent** (`SQLite+FTS5`) | **Pi Coding Agent** (`LLM Summary`) | Key Architectural Differentiator |
| :--- | :---: | :---: | :---: | :--- |
| **1. `LongMemEval`** | **94.6% (`473/500` Full)** | 83.3% *(`10/12` Smoke)* | 75.0% *(`9/12` Smoke)* | Lossless Neo4j `:ActorEpisode` + `:8084` RRF + $\pm 1$ turn context expansion |
| **2. `LOCOMO`** *(Multi-Session)* | **77.3% (`1,536/1,986` Full)** | 75.0% *(Smoke)* | 50.0% *(Smoke)* | Active 2-step `recallEpisodes(Actor[#Memory], ...)` bytecode multi-hop recall |
| **3. `MemoryAgentBench`** *(ICLR 2026)* | **100.0% (`12/12`)** | 83.3% (`10/12`) | 75.0% (`9/12`) | Ring-1 evolved `policyNote` enforcing latest-timestamp state supersession |
| **4. `SWE-bench Verified`** *(5-Task Subset)* | **100.0% (`5/5`)** | 100.0% (`5/5`) | 100.0% (`5/5`) | AST symbol reader + transactional copy-on-write VFS + Gödel test ratchet |
| **5. `Ring-1 Self-Evolution`** | **100.0% (`4/4`)** | 50.0% (`2/4`) | 25.0% (`1/4`) | Live compilation of `AgentHarness` `.act` hooks with 3-strike auto-rollback |

### 3. Clean-Room Unattended Baseline (`v4.1.0` Fresh Git Clone — Zero Cached State, Zero Intervention)

All 6 benchmark suites below were executed consecutively in a single unattended run (`1h 47m` on a single 16 GB consumer GPU, `qwen3.8:latest`) from a brand-new `git clone` with all prior results deleted and Neo4j benchmark namespaces wiped:

| Suite | Runner Script | Clean-Room Score | Pass Rate | Breakdown & Saved Report |
| :--- | :--- | :---: | :---: | :--- |
| **1. Red-Team Adversarial Containment** | `benchmarks/run_redteam_containment_audit.py` | **`6 / 6`** | **`100.0%`** | All 6 live prompt-injection & privilege-escalation attacks blocked ([JSON](benchmarks/REDTEAM_CONTAINMENT_RESULTS.json)) |
| **2. Standard Coding & Math** | `benchmarks/run_standard_benchmarks.py` | **`18 / 20`** | **`90.0%`** | `HumanEval`: **`11/12` (`91.7%`)** + `100%` Engram Cold-Boot; `GSM8K`: **`7/8` (`87.5%`)** ([Report](benchmarks/BENCHMARK_REPORT.md)) |
| **3. `LongMemEval + LOCOMO` (17-Task Stratified)** | `benchmarks/run_longmem_locomo_suite.py` | **`15 / 17`** | **`88.2%`** | `LongMemEval`: **`10/12` (`83.3%`)**; `LOCOMO`: **`5/5` (`100.0%`)** vs `76.5%` Raw Compaction ([Report](benchmarks/LONGMEM_LOCOMO_REPORT.md)) |
| **4. `Terminal-Bench 2.1` (5-Task Suite)** | `benchmarks/run_tb21_5task_suite.py` | **`4 / 5`** | **`80.0%`** | Passed `git-leak-recovery`, `log-summary-date-ranges`, `cancel-async-tasks`, `fix-git` ([Report](benchmarks/TB21_5TASK_REPORT.md)) |
| **5. Next-Gen Post-Cutoff (9-Task Suite)** | `benchmarks/run_nextgen_suite.py` | **`5 / 9`** | **`55.6%`** | `TB2.1`: **`3/3` (`100%`)**, `Aider Polyglot`: **`1/2`** (`grep` 25/25), `LiveCodeBench v6`: **`1/2`** ([Report](benchmarks/NEXTGEN_REPORT.md)) |
| **6. Industry 8-Track Container Suite** | `benchmarks/run_industry_harness_suite.py` | **`16 / 22`** | **`72.7%`** | `HumanEval` `3/3`, `MBPP` `3/3`, `QuixBugs` `3/3`, `AgentBench-OS` `2/2`, `GSM8K` `2/3`, `BigCodeBench` `2/3`, `CRUXEval` `1/2` ([Report](benchmarks/INDUSTRY_SUITE_REPORT.md)) |
| **Total Clean-Room Baseline** | **All 6 Suites Combined** | **`64 / 79`** | **`81.01%`** | Unattended, single-pass containerized baseline on 16 GB consumer GPU |

---

## Container-First, Defense-Up Packaging & Capability vs. Containment Matrix

Bilby ships **with all containment guardrails up by default** (`ACTORII_HOST_LOCKDOWN=1`, non-root `USER bilby`, `cap_drop: [ALL]`, `no-new-privileges:true`, read-only `/app/src`, and locked Ring-0/1 kernel classes).

At any time, you can run `/capabilities` (or `/config`) in the TUI/Telegram or `capabilities(Actor[#Sandbox])!!` in Actor 4.1 bytecode to see which capabilities are active and what host/container settings enable each tier:

| Tier | Capability & Actor Singleton | Default State (`ACTORII_HOST_LOCKDOWN=1`) | What Bilby Needs from the Container / Host | How to Configure or Unlock |
| :---: | :--- | :---: | :--- | :--- |
| **Tier 0** | **Core Reasoning & Multi-Provider LLM** (`Actor[#Harness]`, `Actor[#LLM]`) | **ENABLED** | Outbound HTTP/HTTPS to your chosen LLM (`llama.cpp`, `Ollama`, `OpenAI`, `OpenRouter`, `Gemini`, `vLLM`) | Set `ACTORII_LLM_PROVIDER`, `ACTORII_LLM_ENDPOINT`, `ACTORII_LLM_MODEL`, `ACTORII_LLM_API_KEY` in `.env` or `/config` |
| **Tier 0** | **Lossless Graph Memory & Engrams** (`Actor[#Memory]`) | **ENABLED** | HTTP connection to Neo4j (`:7474`) + optional embedding endpoint (`:8084` / Ollama / OpenAI) | Included in `docker compose up -d`; configure via `ACTORII_NEO4J_HTTP_URL`, `ACTORII_EMBED_ENDPOINT` |
| **Tier 1** | **PDF Reader & Text Search** (`Actor[#PDF]`) | **ENABLED** *(Pre-installed in Docker)* | Python `pypdf` package (baked into `Dockerfile`) + outbound HTTPS or files in `/workspace` / `/tmp` | Works out-of-the-box in Docker; on bare host run `pip install pypdf` |
| **Tier 1** | **Web Scraping & Search** (`Actor[#Web]`, `Actor[#Net]`) | **ENABLED** *(Pre-installed in Docker)* | `trafilatura` + `duckduckgo-search` + Keyless MCP Ring (baked into `Dockerfile`); optional Firecrawl API | Works keyless out-of-the-box; optionally set `FIRECRAWL_API_KEY` in `.env` for Firecrawl `/v2/scrape` |
| **Tier 2** | **Scoped Workspace File I/O** (`Actor[#Host]>>readFile/writeFile/editFile`) | **SCOPED** (`/workspace` & `/tmp` only) | Bind-mount `./workspace:/workspace` in `docker-compose.yml` (host `~/.ssh`, `/etc`, and `src/actorii` are hard-blocked) | Mount any project folder to `/workspace`; set `ACTORII_HOST_LOCKDOWN=0` only if running native host dev workflows |
| **Tier 2** | **Guarded Ring-1 Self-Evolution** (`Actor[#Harness]>>evolve`) | **ENABLED** *(Cognitive hooks only)* | Transactional CoW VM + Neo4j `:ActorEngramVersion` rollback (Tier-2 shell/file calls forbidden inside hooks) | Set `ACTORII_DISABLE_SELF_EVOLVE=1` in `.env` to completely freeze all Ring-1 cognitive hooks |
| **Tier 3** | **Sandboxed Shell Execution** (`Actor[#Host]>>execShell`) | **DOCKER-ONLY** *(Blocked on bare host)* | Isolated container user (`USER bilby`, `ACTORII_IN_DOCKER=1`) + 48-pattern destructive command filter | Enabled inside the isolated Docker container; blocked on bare host unless `ACTORII_HOST_LOCKDOWN=0` |
| **Tier 3** | **Hardware & GPU Telemetry** (`Actor[#Valkyrie]`) | **OPTIONAL** | `nvidia-smi` / GPU passthrough (`--gpus all`) if monitoring local GPU VRAM | Uncomment GPU reservation in `docker-compose.yml` or run on a host with NVIDIA drivers |

---

## Install-Time & Runtime Configuration Reference

Bilby supports switching LLM providers, models, memory namespaces, and containment settings **both at install time (`.env`) and live at runtime (`/config` or `/model` in TUI)**:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ACTORII_LLM_PROVIDER` | `llamacpp` | Provider preset: `llamacpp`, `ollama`, `vllm`, `openai`, `openrouter`, `gemini` |
| `ACTORII_LLM_ENDPOINT` | `http://127.0.0.1:8080/v1/chat/completions` | OpenAI-compatible `/v1/chat/completions` or Ollama `/api/chat` URL |
| `ACTORII_LLM_MODEL` | `qwen3.8:latest` | Model identifier sent in chat completion requests |
| `ACTORII_LLM_API_KEY` | *(empty)* | Bearer API key for `OpenAI`, `OpenRouter`, `Gemini`, or authenticated endpoints (also reads `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`) |
| `ACTORII_NEO4J_HTTP_URL` | `http://127.0.0.1:7474/db/neo4j/tx/commit` | Neo4j HTTP transactional Cypher endpoint |
| `ACTORII_NEO4J_USER` / `PASSWORD` | *(empty — `NEO4J_AUTH=none`)* | Optional Basic Auth credentials for authenticated Neo4j instances |
| `ACTORII_MEMORY_NAMESPACE` | `default` | Isolated graph partition for `:ActorEpisode` and `:ActorEngram` nodes |
| `ACTORII_EMBED_ENDPOINT` | `http://127.0.0.1:8084/v1/embeddings` | Dense vector embedding endpoint (`none` for lexical-only mode) |
| `ACTORII_EMBED_MODEL` | `bge-small-en-v1.5` | Embedding model name (`bge-small-en-v1.5`, `nomic-embed-text`, `text-embedding-3-small`) |
| `FIRECRAWL_API_KEY` | *(empty)* | Optional Firecrawl API key for `Actor[#Web]>>scrape` |
| `ACTORII_HOST_LOCKDOWN` | `1` *(Defense-Up)* | `1` locks bare-host shell execution and restricts writes to `/workspace` and `/tmp` |
| `ACTORII_DISABLE_SELF_EVOLVE` | `0` | `1` disables `evolve(Actor[#Harness], ...)` Ring-1 cognitive hook hot-patching |

---

## 5-Minute Quickstart (Docker Container or Local Virtualenv)

### Option A: Container-First Quickstart (Recommended — Defenses Up by Default)
```bash
git clone https://github.com/kvandixon/bilby-actorii.git
cd bilby-actorii
cp .env.example .env
# Edit .env to set your preferred ACTORII_LLM_PROVIDER / ACTORII_LLM_ENDPOINT / ACTORII_LLM_API_KEY

# Build and start Neo4j + hardened Bilby container (with pypdf, trafilatura, DDGS pre-installed):
docker compose up -d --build

# Attach to the interactive Bilby TUI inside the hardened container:
docker exec -it bilby-actorii bilby
```

### Option B: Local Python Virtualenv Quickstart
```bash
pip install -e .
pip install pypdf cryptography trafilatura duckduckgo-search
docker compose up -d neo4j

# Connect to Ollama, llama.cpp, OpenAI, or OpenRouter:
export ACTORII_LLM_PROVIDER="ollama"
export ACTORII_LLM_MODEL="qwen2.5:14b"

# Launch interactive TUI (try `/capabilities`, `/config`, `/twins`, `/engrams`):
python3 actor_tui.py
```

---

## Security, Containment & Reproducibility Audits

```bash
# 1. Run the 49-test unit, harness, and 6-layer security hardening suite:
PYTHONPATH=src pytest tests/test_agent_harness_and_tui.py \
  tests/test_pi_class_harness_evolution.py \
  tests/test_vm_execution.py \
  tests/test_bilby_security_hardening.py -q

# 2. Run the live 6-scenario Red-Team Adversarial Containment Audit:
PYTHONPATH=src python3 benchmarks/run_redteam_containment_audit.py

# 3. Audit saved LongMemEval (500) + LOCOMO (1,986) results with 0 LLM judge calls:
python3 benchmarks/rejudge_saved_results.py
```

---

## Clean-Room Implementation & Legal Notice

`ActorII` is an independent, clean-room bytecode virtual machine, compiler, and neuro-symbolic AI agent runtime written from scratch in Python. It is inspired by the historical 1993 Actor 4.1 object-oriented language syntax for interoperability and research purposes, and is **not** affiliated with, endorsed by, or sponsored by The Whitewater Group, Symantec Corporation, or Broadcom Inc. No proprietary 1993 source code or binary images are distributed in this Git repository.
