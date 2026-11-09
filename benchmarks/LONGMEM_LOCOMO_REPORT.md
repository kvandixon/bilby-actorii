> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Benchmark #3 Report: 3-Harness Head-to-Head on `LongMemEval` (ICLR 2025) + `LOCOMO` (Snap Research)

**Date**: 2026-11-09 12:22:41 EST  
**Harnesses Evaluated**: `ActorII v4.1` (`Bilby`) vs `Hermes Agent` (`hermes-bench:latest`) vs `Pi Coding Agent v0.86.1` (`pi-bench:latest`)  
**Model & Inference Parity**: `qwen3.8:latest` on `:8080` via normalizing proxy `:18088` (`enable_thinking=False` enforced identically across all 3 harnesses)  
**Dataset**: `17` multi-session tasks per harness

---

## 1. Executive Summary: 3-Harness Head-to-Head Comparison

| Harness | Long-Horizon Memory & Compaction Architecture | Overall Score | `LongMemEval` | `LOCOMO` | Avg Tokens / Task | Avg Latency / Task |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ActorII v4.1 (`:ActorEpisode` + `:8084` Hybrid)** | Neo4j `:ActorEpisode` micro-chunks + `:8084` `bge-small-en-v1.5` vector/lexical/chronological recall | **15/17 (88.2%)** | `10/12` (83.3%) | `5/5` (100.0%) | `13308.0` | `33.24s` |
| **Hermes Agent (`hermes-bench:latest`)** | SQLite `state.db` + `ContextCompressor` + FTS5 `session_search` lineage recovery | **0/0 (0.0%)** | `0/0` (0.0%) | `0/0` (0.0%) | `0.0` | `0.0s` |
| **Pi Coding Agent (`pi-bench:latest`)** | `SessionManager` JSONL + `generateSummaryWithUsage` LLM compaction summary | **0/0 (0.0%)** | `0/0` (0.0%) | `0/0` (0.0%) | `0.0` | `0.0s` |
| **ActorII (`Raw Compaction` Ablation)** | Truncated `chatLog` digest only (no Neo4j `:ActorEpisode` archival) | **13/17 (76.5%)** | `11/12` (91.7%) | `2/5` (40.0%) | `19674.9` | `44.16s` |

---

## 2. Category Breakdown Across All 3 Harnesses

| Suite | Category | ActorII v4.1 (`:ActorEpisode` Hybrid) | Hermes Agent (`SQLite + FTS5`) | Pi Coding Agent (`LLM Summary`) | ActorII (`Raw Compaction`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LongMemEval** | `knowledge-update` | **2/2** | `0/0` | `0/0` | `2/2` |
| **LongMemEval** | `multi-session` | **0/2** | `0/0` | `0/0` | `1/2` |
| **LongMemEval** | `single-session-assistant` | **2/2** | `0/0` | `0/0` | `2/2` |
| **LongMemEval** | `single-session-preference` | **2/2** | `0/0` | `0/0` | `2/2` |
| **LongMemEval** | `single-session-user` | **2/2** | `0/0` | `0/0` | `2/2` |
| **LongMemEval** | `temporal-reasoning` | **2/2** | `0/0` | `0/0` | `2/2` |
| **LOCOMO** | `locomo-cat2-temporal` | **1/1** | `0/0` | `0/0` | `0/1` |
| **LOCOMO** | `locomo-cat3-multi-hop` | **1/1** | `0/0` | `0/0` | `1/1` |
| **LOCOMO** | `locomo-cat1-single-hop` | **1/1** | `0/0` | `0/0` | `0/1` |
| **LOCOMO** | `locomo-cat4-open-domain` | **1/1** | `0/0` | `0/0` | `0/1` |
| **LOCOMO** | `locomo-cat5-adversarial` | **1/1** | `0/0` | `0/0` | `1/1` |

---

## 3. Task-by-Task Head-to-Head Verification Ledger (`17` Completed Tasks)

| Task ID | Category | Turns | Gold Answer | ActorII v4.1 | Hermes Agent | Pi Agent |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `lme_knowledge-update_6a1eabeb` | `knowledge-update` | `14` | 25 minutes and 50 seconds (or 25:50) | **PASS** (`25.27s`, `12845` tok) | PENDING | PENDING |
| `lme_knowledge-update_6aeb4375` | `knowledge-update` | `14` | four | **PASS** (`25.62s`, `12908` tok) | PENDING | PENDING |
| `lme_multi-session_0a995998` | `multi-session` | `19` | 3 | FAIL (`23.87s`) | PENDING | PENDING |
| `lme_multi-session_6d550036` | `multi-session` | `24` | 2 | FAIL (`71.17s`) | PENDING | PENDING |
| `lme_single-session-assistant_7161e7e2` | `single-session-assistant` | `9` | Admon was assigned to the 8 am - 4 pm (Day Shift) on Su | **PASS** (`23.11s`, `10649` tok) | PENDING | PENDING |
| `lme_single-session-assistant_c4f10528` | `single-session-assistant` | `9` | Miss Bee Providore | **PASS** (`21.99s`, `10281` tok) | PENDING | PENDING |
| `lme_single-session-preference_8a2466db` | `single-session-preference` | `11` | The user would prefer responses that suggest resources  | **PASS** (`79.4s`, `11675` tok) | PENDING | PENDING |
| `lme_single-session-preference_06878be2` | `single-session-preference` | `10` | The user would prefer suggestions of Sony-compatible ac | **PASS** (`72.63s`, `11356` tok) | PENDING | PENDING |
| `lme_single-session-user_e47becba` | `single-session-user` | `8` | Business Administration | **PASS** (`17.93s`, `10319` tok) | PENDING | PENDING |
| `lme_single-session-user_118b2229` | `single-session-user` | `8` | 45 minutes each way | **PASS** (`17.54s`, `10106` tok) | PENDING | PENDING |
| `lme_temporal-reasoning_gpt4_2655b836` | `temporal-reasoning` | `20` | GPS system not functioning correctly | **PASS** (`25.11s`, `13379` tok) | PENDING | PENDING |
| `lme_temporal-reasoning_gpt4_2487a7cb` | `temporal-reasoning` | `14` | 'Data Analysis using Python' webinar | **PASS** (`27.29s`, `12685` tok) | PENDING | PENDING |
| `locomo_cat2` | `locomo-cat2-temporal` | `41` | 7 May 2023 | **PASS** (`28.58s`, `11992` tok) | PENDING | PENDING |
| `locomo_cat3` | `locomo-cat3-multi-hop` | `41` | Psychology, counseling certification | **PASS** (`30.51s`, `12092` tok) | PENDING | PENDING |
| `locomo_cat1` | `locomo-cat1-single-hop` | `41` | Adoption agencies | **PASS** (`23.11s`, `11901` tok) | PENDING | PENDING |
| `locomo_cat4` | `locomo-cat4-open-domain` | `41` | mental health | **PASS** (`23.96s`, `12065` tok) | PENDING | PENDING |
| `locomo_cat5` | `locomo-cat5-adversarial` | `41` | False premise: Melanie (not Caroline) realized 'self-ca | **PASS** (`28.0s`, `12127` tok) | PENDING | PENDING |
