> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Benchmark #3 Report: 3-Harness Head-to-Head on `LongMemEval` (ICLR 2025) + `LOCOMO` (Snap Research)

**Date**: 2026-11-06 15:44:05 EST  
**Harnesses Evaluated**: `ActorII v4.1` (`Bilby`) vs `Hermes Agent` (`hermes-bench:latest`) vs `Pi Coding Agent v0.86.1` (`pi-bench:latest`)  
**Model & Inference Parity**: `qwen3.8:latest` on `:8080` via normalizing proxy `:18088` (`enable_thinking=False` enforced identically across all 3 harnesses)  
**Dataset**: `17` stratified multi-session tasks (`12` LongMemEval across 6 categories + `5` LOCOMO across 5 categories) = `68` total multi-session evaluations

---

## 1. Executive Summary: 3-Harness Head-to-Head Comparison

| Harness | Long-Horizon Memory & Compaction Architecture | Overall Score | `LongMemEval` (12) | `LOCOMO` (5) | Avg Tokens / Task | Avg Latency / Task |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ActorII v4.1 (`:ActorEpisode` + `:8084` Hybrid)** | Neo4j `:ActorEpisode` micro-chunks + `:8084` `bge-small-en-v1.5` vector/lexical/chronological recall | **17/17 (100.0%)** | `12/12` (100.0%) | `5/5` (100.0%) | `10779.2` | `39.49s` |
| **Hermes Agent (`hermes-bench:latest`)** | SQLite `state.db` + `ContextCompressor` + FTS5 `session_search` lineage recovery | **14/17 (82.4%)** | `10/12` (83.3%) | `4/5` (80.0%) | `20894.6` | `32.01s` |
| **Pi Coding Agent (`pi-bench:latest`)** | `SessionManager` JSONL + `generateSummaryWithUsage` LLM compaction summary | **13/17 (76.5%)** | `8/12` (66.7%) | `5/5` (100.0%) | `8180.6` | `111.39s` |
| **ActorII (`Raw Compaction` Ablation)** | Truncated `chatLog` digest only (no Neo4j `:ActorEpisode` archival) | **8/17 (47.1%)** | `5/12` (41.7%) | `3/5` (60.0%) | `5999.5` | `26.72s` |

---

## 2. Category Breakdown Across All 3 Harnesses

| Suite | Category | ActorII v4.1 (`:ActorEpisode` Hybrid) | Hermes Agent (`SQLite + FTS5`) | Pi Coding Agent (`LLM Summary`) | ActorII (`Raw Compaction`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LongMemEval** | `knowledge-update` | **2/2** | `2/2` | `2/2` | `0/2` |
| **LongMemEval** | `multi-session` | **2/2** | `1/2` | `0/2` | `0/2` |
| **LongMemEval** | `single-session-assistant` | **2/2** | `2/2` | `2/2` | `1/2` |
| **LongMemEval** | `single-session-preference` | **2/2** | `1/2` | `1/2` | `2/2` |
| **LongMemEval** | `single-session-user` | **2/2** | `2/2` | `1/2` | `1/2` |
| **LongMemEval** | `temporal-reasoning` | **2/2** | `2/2` | `2/2` | `1/2` |
| **LOCOMO** | `locomo-cat2-temporal` | **1/1** | `1/1` | `1/1` | `1/1` |
| **LOCOMO** | `locomo-cat3-multi-hop` | **1/1** | `1/1` | `1/1` | `1/1` |
| **LOCOMO** | `locomo-cat1-single-hop` | **1/1** | `0/1` | `1/1` | `0/1` |
| **LOCOMO** | `locomo-cat4-open-domain` | **1/1** | `1/1` | `1/1` | `0/1` |
| **LOCOMO** | `locomo-cat5-adversarial` | **1/1** | `1/1` | `1/1` | `1/1` |

---

## 3. Task-by-Task Head-to-Head Verification Ledger (`17` Tasks)

| Task ID | Category | Turns | Gold Answer | ActorII v4.1 | Hermes Agent | Pi Agent |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `lme_knowledge-update_6a1eabeb` | `knowledge-update` | `14` | 25 minutes and 50 seconds (or 25:50) | **PASS** (`32.51s`, `11620` tok) | **PASS** (`24.66s`, `12912` tok) | **PASS** (`136.87s`, `12774` tok) |
| `lme_knowledge-update_6aeb4375` | `knowledge-update` | `14` | four | **PASS** (`33.48s`, `11687` tok) | **PASS** (`29.78s`, `13955` tok) | **PASS** (`123.04s`, `9702` tok) |
| `lme_multi-session_0a995998` | `multi-session` | `19` | 3 | **PASS** (`43.76s`, `12398` tok) | FAIL (`96.86s`) | FAIL (`150.33s`) |
| `lme_multi-session_6d550036` | `multi-session` | `24` | 2 | **PASS** (`40.96s`, `11919` tok) | **PASS** (`18.88s`, `13982` tok) | FAIL (`150.23s`) |
| `lme_single-session-assistant_7161e7e2` | `single-session-assistant` | `9` | Admon was assigned to the 8 am - 4 pm (Day Shift) on Su | **PASS** (`37.24s`, `9484` tok) | **PASS** (`15.26s`, `13918` tok) | **PASS** (`88.86s`, `6041` tok) |
| `lme_single-session-assistant_c4f10528` | `single-session-assistant` | `9` | Miss Bee Providore | **PASS** (`22.77s`, `9002` tok) | **PASS** (`15.77s`, `13379` tok) | **PASS** (`68.6s`, `5251` tok) |
| `lme_single-session-preference_8a2466db` | `single-session-preference` | `11` | The user would prefer responses that suggest resources  | **PASS** (`83.73s`, `10438` tok) | **PASS** (`43.68s`, `13549` tok) | FAIL (`150.23s`) |
| `lme_single-session-preference_06878be2` | `single-session-preference` | `10` | The user would prefer suggestions of Sony-compatible ac | **PASS** (`78.8s`, `10734` tok) | FAIL (`96.91s`) | **PASS** (`128.17s`, `8974` tok) |
| `lme_single-session-user_e47becba` | `single-session-user` | `8` | Business Administration | **PASS** (`15.69s`, `9012` tok) | **PASS** (`10.44s`, `12857` tok) | **PASS** (`96.05s`, `7806` tok) |
| `lme_single-session-user_118b2229` | `single-session-user` | `8` | 45 minutes each way | **PASS** (`17.38s`, `8815` tok) | **PASS** (`35.4s`, `43172` tok) | FAIL (`78.7s`) |
| `lme_temporal-reasoning_gpt4_2655b836` | `temporal-reasoning` | `20` | GPS system not functioning correctly | **PASS** (`27.68s`, `12112` tok) | **PASS** (`26.86s`, `14686` tok) | **PASS** (`124.48s`, `13534` tok) |
| `lme_temporal-reasoning_gpt4_2487a7cb` | `temporal-reasoning` | `14` | 'Data Analysis using Python' webinar | **PASS** (`40.0s`, `11499` tok) | **PASS** (`58.38s`, `13891` tok) | **PASS** (`102.77s`, `11273` tok) |
| `locomo_cat2` | `locomo-cat2-temporal` | `41` | 7 May 2023 | **PASS** (`36.21s`, `10719` tok) | **PASS** (`24.86s`, `14972` tok) | **PASS** (`96.95s`, `7835` tok) |
| `locomo_cat3` | `locomo-cat3-multi-hop` | `41` | Psychology, counseling certification | **PASS** (`53.18s`, `11250` tok) | **PASS** (`18.03s`, `14943` tok) | **PASS** (`106.26s`, `8072` tok) |
| `locomo_cat1` | `locomo-cat1-single-hop` | `41` | Adoption agencies | **PASS** (`48.26s`, `10823` tok) | FAIL (`9.29s`) | **PASS** (`94.19s`, `7907` tok) |
| `locomo_cat4` | `locomo-cat4-open-domain` | `41` | mental health | **PASS** (`28.97s`, `10824` tok) | **PASS** (`9.69s`, `14475` tok) | **PASS** (`99.29s`, `7952` tok) |
| `locomo_cat5` | `locomo-cat5-adversarial` | `41` | False premise: Melanie (not Caroline) realized 'self-ca | **PASS** (`30.63s`, `10911` tok) | **PASS** (`9.43s`, `14874` tok) | **PASS** (`98.58s`, `7940` tok) |
