# Bilby (`ActorII v4.3.0-dev`) — Metacircular Bytecode AI Agent & 1993 Actor VM

**Bilby (`ActorII v4.3.0-dev`)** is a self-evolving, metacircular AI agent harness executing inside a clean-room 1993 Whitewater Actor 4.1 Smalltalk/Actor bytecode virtual machine, backed by **lossless Neo4j `:ActorEpisode` + `:ActorEngram` hybrid graph memory**.

Running **100% locally on a single 16 GB consumer GPU** (NVIDIA RTX 5070 Ti) with a 20B–27B open-weight model (`0 MB` GPU VRAM overhead for embeddings and memory indexing)—or connected to any cloud LLM (`OpenAI`, `OpenRouter`, `Gemini`, `Ollama`, `vLLM`)—Bilby provides a hardened, container-first neuro-symbolic agent runtime.

[![Benchmarks](https://img.shields.io/badge/Benchmarks-Bilby%20Results-brightgreen.svg)](benchmarks/BENCHMARK_RESULTS.md)
[![Test Suite](https://img.shields.io/badge/tests-745%2B%20passed-brightgreen.svg)](tests/)
[![Containment](https://img.shields.io/badge/containment-Defense--Up%20Docker%20%2B%203--Ring%20VM-orange.svg)](src/actorii/harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Benchmark Summary (`Bilby ActorII v4.3.0-dev` — Local RTX 5070 Ti 16 GB)

> **Clean-Room Release Delta (`v4.2.0` → `v4.3.0-dev`, 2026-11-14)**: **`+15.00 pp`** on HumanEval+GSM8K (`100.0%`), **`+40.00 pp`** on Self-Evolution (`80.0%` — `4/5` Canaries locked), **`+4.55 pp`** on Industry 8-Track (`81.82%`), **`+20.00 pp`** strict `pass^3` on Terminal-Bench 2.1 (`60%`), **ZERO headline regressions**, and **`1.75x–2.08x` faster wall-clock execution** via append-only KV-cache prefix reuse (`97.4%` hit rate). See Section 1 of [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md) for the full release-over-release comparison.

<!-- BENCHMARK_SCOREBOARD:START -->
| Benchmark Suite | Pillar | Tasks | Bilby Score (`Latest`) | Δ vs Prior Release | Date |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LongMemEval (S Split)** | `Memory` | `500` | **`94.60% (473 / 500)`** | **`+0.48 pp` ▲** | `2026-11-07` |
| **LoCoMo-10 (Full 5-Category Suite)** | `Memory` | `1,986` | **`77.34% (1,536 / 1,986)`** | — | `2026-11-07` |
| **Terminal-Bench 2.1 (Container Subset)** | `Terminal & OS` | `5 (×3)` | **`80.00% avg@3 (12 / 15 trials)`** | `0.00 pp (held)` | `2026-11-14` |
| **Terminal-Bench 2.1 (Hard Tier)** | `Terminal & OS` | `5` | **`0.00% (0 / 5)`** | `0.00 pp (held)` | `2026-11-14` |
| **Industry 8-Track Suite** | `SWE & Coding` | `22` | **`81.82% (18 / 22)`** | **`+4.55 pp` ▲** | `2026-11-14` |
| **Next-Gen Post-Cutoff Suite** | `SWE & Coding` | `9 (×3)` | **`59.26% avg@3 (16 / 27 trials)`** | `0.00 pp (held)` | `2026-11-14` |
| **Red-Team Adversarial Containment Audit** | `Security` | `6` | **`100.00% (6 / 6 Blocked)`** | `0.00 pp (held)` | `2026-11-14` |
| **HumanEval-12 + GSM8K-8 Local Model Matrix** | `SWE & Coding` | `20` | **`100.00% (20 / 20)`** | **`+15.00 pp` ▲** | `2026-11-14` |
| **LongMemEval + LoCoMo (Stratified 17-Task)** | `Memory` | `17` | **`94.12% (16 / 17)`** | `0.00 pp (held)` | `2026-11-14` |
| **Bilby Deterministic Self-Evolution Loop (Tiers 1-3)** | `SWE & Coding` | `5` | **`80.00% (4 / 5)`** | **`+40.00 pp` ▲** | `2026-11-14` |
<!-- BENCHMARK_SCOREBOARD:END -->

👉 **[Full details, run history & live HTML scoreboard (`benchmarks/BENCHMARK_RESULTS.md`)](benchmarks/BENCHMARK_RESULTS.md)**

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
| `ACTOR_BIND_IP` | `127.0.0.1` | Host interface bound by Docker Studio ports (`:8090` / `:8765`) |
| `ACTORII_STUDIO_TOKEN` | *(empty)* | Optional Bearer / `?token=` secret required for Studio `/api/*` and WebSocket upgrades |
| `ACTORII_PERSONA_FILE` | `~/.actorii/persona.md` | Optional local host-environment & persona markdown overlay appended to the system prompt |

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
