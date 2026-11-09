> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Benchmark #3 Report: 3-Harness Head-to-Head on `LongMemEval` (ICLR 2025) + `LOCOMO` (Snap Research)

**Date**: 2026-11-07 20:52:03 EST  
**Harnesses Evaluated**: `ActorII v4.1` (`Bilby`) vs `Hermes Agent` (`hermes-bench:latest`) vs `Pi Coding Agent v0.86.1` (`pi-bench:latest`)  
**Model & Inference Parity**: `qwen3.8:latest` on `:8080` via normalizing proxy `:18088` (`enable_thinking=False` enforced identically across all 3 harnesses)  
**Dataset**: `2486` multi-session tasks per harness

---

## 1. Executive Summary: 3-Harness Head-to-Head Comparison

| Harness | Long-Horizon Memory & Compaction Architecture | Overall Score | `LongMemEval` | `LOCOMO` | Avg Tokens / Task | Avg Latency / Task |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ActorII v4.1 (`:ActorEpisode` + `:8084` Hybrid)** | Neo4j `:ActorEpisode` micro-chunks + `:8084` `bge-small-en-v1.5` vector/lexical/chronological recall | **2009/2486 (80.81%)** | `473/500` (94.6%) | `1536/1986` (77.3%) | `15406.9` | `36.61s` |
| **Hermes Agent (`hermes-bench:latest`)** | SQLite `state.db` + `ContextCompressor` + FTS5 `session_search` lineage recovery | **10/12 (83.3%)** | `10/12` (83.3%) | `0/0` (0.0%) | `23422.7` | `39.41s` |
| **Pi Coding Agent (`pi-bench:latest`)** | `SessionManager` JSONL + `generateSummaryWithUsage` LLM compaction summary | **8/12 (66.7%)** | `8/12` (66.7%) | `0/0` (0.0%) | `8280.4` | `116.53s` |
| **ActorII (`Raw Compaction` Ablation)** | Truncated `chatLog` digest only (no Neo4j `:ActorEpisode` archival) | **5/12 (41.7%)** | `5/12` (41.7%) | `0/0` (0.0%) | `6174.9` | `28.79s` |

---

## 2. Category Breakdown Across All 3 Harnesses

| Suite | Category | ActorII v4.1 (`:ActorEpisode` Hybrid) | Hermes Agent (`SQLite + FTS5`) | Pi Coding Agent (`LLM Summary`) | ActorII (`Raw Compaction`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LongMemEval** | `knowledge-update` | **77/78** | `2/2` | `2/2` | `0/2` |
| **LongMemEval** | `multi-session` | **126/133** | `1/2` | `0/2` | `0/2` |
| **LongMemEval** | `single-session-assistant` | **45/56** | `2/2` | `2/2` | `1/2` |
| **LongMemEval** | `single-session-preference` | **30/30** | `1/2` | `1/2` | `2/2` |
| **LongMemEval** | `single-session-user` | **69/70** | `2/2` | `1/2` | `1/2` |
| **LongMemEval** | `temporal-reasoning` | **126/133** | `2/2` | `2/2` | `1/2` |
| **LOCOMO** | `locomo-cat2-temporal` | **262/321** | `0/0` | `0/0` | `0/0` |
| **LOCOMO** | `locomo-cat3-multi-hop` | **46/96** | `0/0` | `0/0` | `0/0` |
| **LOCOMO** | `locomo-cat1-single-hop` | **177/282** | `0/0` | `0/0` | `0/0` |
| **LOCOMO** | `locomo-cat4-open-domain` | **683/841** | `0/0` | `0/0` | `0/0` |
| **LOCOMO** | `locomo-cat5-adversarial` | **368/446** | `0/0` | `0/0` | `0/0` |

---

## 3. Task-by-Task Head-to-Head Verification Ledger (`2486` Completed Tasks)

| Task ID | Category | Turns | Gold Answer | ActorII v4.1 | Hermes Agent | Pi Agent |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `lme_knowledge-update_6a1eabeb` | `knowledge-update` | `14` | 25 minutes and 50 seconds (or 25:50) | **PASS** (`33.17s`, `11323` tok) | **PASS** (`24.66s`, `12912` tok) | **PASS** (`136.87s`, `12774` tok) |
| `lme_knowledge-update_6aeb4375` | `knowledge-update` | `14` | four | **PASS** (`30.12s`, `11226` tok) | **PASS** (`29.78s`, `13955` tok) | **PASS** (`123.04s`, `9702` tok) |
| `lme_knowledge-update_830ce83f` | `knowledge-update` | `14` | the suburbs | **PASS** (`26.16s`, `11026` tok) | PENDING | PENDING |
| `lme_knowledge-update_852ce960` | `knowledge-update` | `14` | $400,000 | FAIL (`18.74s`) | PENDING | PENDING |
| `lme_knowledge-update_945e3d21` | `knowledge-update` | `14` | Three times a week. | **PASS** (`24.71s`, `11817` tok) | PENDING | PENDING |
| `lme_knowledge-update_d7c942c3` | `knowledge-update` | `14` | Yes. | **PASS** (`25.43s`, `11374` tok) | PENDING | PENDING |
| `lme_knowledge-update_71315a70` | `knowledge-update` | `14` | 10-12 hours | **PASS** (`25.99s`, `11559` tok) | PENDING | PENDING |
| `lme_knowledge-update_89941a93` | `knowledge-update` | `14` | 4 | **PASS** (`32.47s`, `11933` tok) | PENDING | PENDING |
| `lme_knowledge-update_ce6d2d27` | `knowledge-update` | `14` | Friday | **PASS** (`25.11s`, `11586` tok) | PENDING | PENDING |
| `lme_knowledge-update_9ea5eabc` | `knowledge-update` | `14` | Paris | **PASS** (`34.39s`, `11580` tok) | PENDING | PENDING |
| `lme_knowledge-update_07741c44` | `knowledge-update` | `14` | under my bed | **PASS** (`25.62s`, `11358` tok) | PENDING | PENDING |
| `lme_knowledge-update_a1eacc2a` | `knowledge-update` | `14` | seven | **PASS** (`31.07s`, `11585` tok) | PENDING | PENDING |
| `lme_knowledge-update_184da446` | `knowledge-update` | `14` | 220 | **PASS** (`37.91s`, `11866` tok) | PENDING | PENDING |
| `lme_knowledge-update_031748ae` | `knowledge-update` | `14` | When you just started your new role as Senior Software  | **PASS** (`33.25s`, `11682` tok) | PENDING | PENDING |
| `lme_knowledge-update_4d6b87c8` | `knowledge-update` | `14` | 25 | **PASS** (`34.0s`, `11772` tok) | PENDING | PENDING |
| `lme_knowledge-update_0f05491a` | `knowledge-update` | `14` | 120 | **PASS** (`25.03s`, `11179` tok) | PENDING | PENDING |
| `lme_knowledge-update_08e075c7` | `knowledge-update` | `14` | 9 months | **PASS** (`24.85s`, `11905` tok) | PENDING | PENDING |
| `lme_knowledge-update_f9e8c073` | `knowledge-update` | `14` | five | **PASS** (`18.47s`, `11748` tok) | PENDING | PENDING |
| `lme_knowledge-update_41698283` | `knowledge-update` | `14` | a 70-200mm zoom lens | **PASS** (`36.35s`, `12048` tok) | PENDING | PENDING |
| `lme_knowledge-update_2698e78f` | `knowledge-update` | `14` | every week | **PASS** (`30.12s`, `11788` tok) | PENDING | PENDING |
| `lme_knowledge-update_b6019101` | `knowledge-update` | `13` | 5 | **PASS** (`31.68s`, `11486` tok) | PENDING | PENDING |
| `lme_knowledge-update_45dc21b6` | `knowledge-update` | `14` | 3 | **PASS** (`31.98s`, `11601` tok) | PENDING | PENDING |
| `lme_knowledge-update_5a4f22c0` | `knowledge-update` | `14` | TechCorp | **PASS** (`34.26s`, `11875` tok) | PENDING | PENDING |
| `lme_knowledge-update_6071bd76` | `knowledge-update` | `14` | You switched to less water (5 ounces) per tablespoon of | **PASS** (`31.38s`, `11343` tok) | PENDING | PENDING |
| `lme_knowledge-update_e493bb7c` | `knowledge-update` | `14` | in my bedroom | **PASS** (`37.53s`, `11967` tok) | PENDING | PENDING |
| `lme_knowledge-update_618f13b2` | `knowledge-update` | `14` | six | **PASS** (`35.77s`, `11872` tok) | PENDING | PENDING |
| `lme_knowledge-update_72e3ee87` | `knowledge-update` | `14` | 50 | **PASS** (`24.03s`, `11543` tok) | PENDING | PENDING |
| `lme_knowledge-update_c4ea545c` | `knowledge-update` | `14` | Yes | **PASS** (`29.85s`, `11511` tok) | PENDING | PENDING |
| `lme_knowledge-update_01493427` | `knowledge-update` | `14` | 25 | **PASS** (`38.95s`, `11800` tok) | PENDING | PENDING |
| `lme_knowledge-update_6a27ffc2` | `knowledge-update` | `14` | 30 | **PASS** (`31.2s`, `11670` tok) | PENDING | PENDING |
| `lme_knowledge-update_2133c1b5` | `knowledge-update` | `14` | 3 months | **PASS** (`25.48s`, `11914` tok) | PENDING | PENDING |
| `lme_knowledge-update_18bc8abd` | `knowledge-update` | `14` | Kansas City Masterpiece | **PASS** (`27.45s`, `11788` tok) | PENDING | PENDING |
| `lme_knowledge-update_db467c8c` | `knowledge-update` | `14` | nine months | **PASS** (`24.76s`, `11252` tok) | PENDING | PENDING |
| `lme_knowledge-update_7a87bd0c` | `knowledge-update` | `13` | 4 weeks | **PASS** (`19.65s`, `11395` tok) | PENDING | PENDING |
| `lme_knowledge-update_e61a7584` | `knowledge-update` | `14` | 9 months | **PASS** (`23.89s`, `11253` tok) | PENDING | PENDING |
| `lme_knowledge-update_1cea1afa` | `knowledge-update` | `14` | 600 | **PASS** (`34.33s`, `12030` tok) | PENDING | PENDING |
| `lme_knowledge-update_ed4ddc30` | `knowledge-update` | `14` | 20 | **PASS** (`25.77s`, `11550` tok) | PENDING | PENDING |
| `lme_knowledge-update_8fb83627` | `knowledge-update` | `14` | Five | **PASS** (`31.76s`, `11815` tok) | PENDING | PENDING |
| `lme_knowledge-update_b01defab` | `knowledge-update` | `14` | Yes | **PASS** (`32.69s`, `11905` tok) | PENDING | PENDING |
| `lme_knowledge-update_22d2cb42` | `knowledge-update` | `14` | The music shop on Main St. | **PASS** (`25.45s`, `11473` tok) | PENDING | PENDING |
| `lme_knowledge-update_0e4e4c46` | `knowledge-update` | `14` | 132 points | **PASS** (`31.15s`, `11563` tok) | PENDING | PENDING |
| `lme_knowledge-update_4b24c848` | `knowledge-update` | `14` | five | **PASS** (`24.68s`, `11374` tok) | PENDING | PENDING |
| `lme_knowledge-update_7e974930` | `knowledge-update` | `14` | $420 | **PASS** (`24.54s`, `11448` tok) | PENDING | PENDING |
| `lme_knowledge-update_603deb26` | `knowledge-update` | `13` | 10 | **PASS** (`27.9s`, `11646` tok) | PENDING | PENDING |
| `lme_knowledge-update_59524333` | `knowledge-update` | `13` | 6:00 pm | **PASS** (`27.43s`, `10849` tok) | PENDING | PENDING |
| `lme_knowledge-update_5831f84d` | `knowledge-update` | `13` | 15 | **PASS** (`39.19s`, `11228` tok) | PENDING | PENDING |
| `lme_knowledge-update_eace081b` | `knowledge-update` | `14` | Oahu | **PASS** (`32.35s`, `11220` tok) | PENDING | PENDING |
| `lme_knowledge-update_affe2881` | `knowledge-update` | `14` | 32 | **PASS** (`36.0s`, `11624` tok) | PENDING | PENDING |
| `lme_knowledge-update_50635ada` | `knowledge-update` | `14` | Premier Silver | **PASS** (`27.95s`, `11622` tok) | PENDING | PENDING |
| `lme_knowledge-update_e66b632c` | `knowledge-update` | `14` | 27 minutes and 45 seconds | **PASS** (`44.11s`, `12283` tok) | PENDING | PENDING |
| `lme_knowledge-update_0ddfec37` | `knowledge-update` | `13` | 15 | **PASS** (`39.06s`, `11706` tok) | PENDING | PENDING |
| `lme_knowledge-update_f685340e` | `knowledge-update` | `14` | Previously, you play tennis with your friends at the lo | **PASS** (`34.57s`, `11539` tok) | PENDING | PENDING |
| `lme_knowledge-update_cc5ded98` | `knowledge-update` | `13` | about two hours | **PASS** (`35.0s`, `11262` tok) | PENDING | PENDING |
| `lme_knowledge-update_dfde3500` | `knowledge-update` | `13` | Wednesday | **PASS** (`17.3s`, `10730` tok) | PENDING | PENDING |
| `lme_knowledge-update_69fee5aa` | `knowledge-update` | `13` | 38 | **PASS** (`47.76s`, `11395` tok) | PENDING | PENDING |
| `lme_knowledge-update_7401057b` | `knowledge-update` | `14` | Two | **PASS** (`25.93s`, `11413` tok) | PENDING | PENDING |
| `lme_knowledge-update_cf22b7bf` | `knowledge-update` | `13` | 10 pounds | **PASS** (`26.44s`, `11496` tok) | PENDING | PENDING |
| `lme_knowledge-update_a2f3aa27` | `knowledge-update` | `12` | 1300 | **PASS** (`35.13s`, `10631` tok) | PENDING | PENDING |
| `lme_knowledge-update_c7dc5443` | `knowledge-update` | `14` | 5-2 | **PASS** (`23.84s`, `11541` tok) | PENDING | PENDING |
| `lme_knowledge-update_06db6396` | `knowledge-update` | `14` | 5 | **PASS** (`34.44s`, `11571` tok) | PENDING | PENDING |
| `lme_knowledge-update_3ba21379` | `knowledge-update` | `14` | Ford F-150 pickup truck | **PASS** (`30.95s`, `11628` tok) | PENDING | PENDING |
| `lme_knowledge-update_9bbe84a2` | `knowledge-update` | `13` | level 100 | **PASS** (`40.99s`, `11217` tok) | PENDING | PENDING |
| `lme_knowledge-update_10e09553` | `knowledge-update` | `12` | 7 | **PASS** (`19.14s`, `10553` tok) | PENDING | PENDING |
| `lme_knowledge-update_dad224aa` | `knowledge-update` | `14` | 7:30 am | **PASS** (`32.04s`, `11686` tok) | PENDING | PENDING |
| `lme_knowledge-update_ba61f0b9` | `knowledge-update` | `13` | 6 | **PASS** (`20.28s`, `10889` tok) | PENDING | PENDING |
| `lme_knowledge-update_42ec0761` | `knowledge-update` | `14` | Yes | **PASS** (`29.52s`, `11326` tok) | PENDING | PENDING |
| `lme_knowledge-update_5c40ec5b` | `knowledge-update` | `12` | We've met up twice. | **PASS** (`36.48s`, `10677` tok) | PENDING | PENDING |
| `lme_knowledge-update_c6853660` | `knowledge-update` | `12` | You increased the limit (from one cup to two cups) | **PASS** (`30.8s`, `10965` tok) | PENDING | PENDING |
| `lme_knowledge-update_26bdc477` | `knowledge-update` | `14` | five | **PASS** (`37.93s`, `11674` tok) | PENDING | PENDING |
| `lme_knowledge-update_0977f2af` | `knowledge-update` | `14` | Instant Pot | **PASS** (`34.74s`, `11475` tok) | PENDING | PENDING |
| `lme_knowledge-update_6aeb4375_abs` | `knowledge-update` | `14` | The information provided is not enough. You mentioned t | **PASS** (`27.71s`, `11673` tok) | PENDING | PENDING |
| `lme_knowledge-update_031748ae_abs` | `knowledge-update` | `14` | The information provided is not enough. You mentioned s | **PASS** (`37.8s`, `11746` tok) | PENDING | PENDING |
| `lme_knowledge-update_2698e78f_abs` | `knowledge-update` | `14` | The information provided is not enough. You mentioned s | **PASS** (`33.74s`, `11848` tok) | PENDING | PENDING |
| `lme_knowledge-update_2133c1b5_abs` | `knowledge-update` | `14` | The information provided is not enough. You mentioned l | **PASS** (`34.64s`, `12022` tok) | PENDING | PENDING |
| `lme_knowledge-update_0ddfec37_abs` | `knowledge-update` | `13` | The information provided is not enough. You mentioned c | **PASS** (`38.85s`, `11735` tok) | PENDING | PENDING |
| `lme_knowledge-update_f685340e_abs` | `knowledge-update` | `14` | The information provided is not enough. You mentioned p | **PASS** (`32.23s`, `11500` tok) | PENDING | PENDING |
| `lme_knowledge-update_89941a94` | `knowledge-update` | `14` | Yes. (You have a road bike too.) | **PASS** (`42.26s`, `12039` tok) | PENDING | PENDING |
| `lme_knowledge-update_07741c45` | `knowledge-update` | `14` | in a shoe rack in my closet | **PASS** (`31.4s`, `11407` tok) | PENDING | PENDING |
| `lme_multi-session_0a995998` | `multi-session` | `19` | 3 | **PASS** (`41.77s`, `11405` tok) | FAIL (`96.86s`) | FAIL (`150.33s`) |
| `lme_multi-session_6d550036` | `multi-session` | `24` | 2 | **PASS** (`18.88s`, `13982` tok) | **PASS** (`18.88s`, `13982` tok) | FAIL (`150.23s`) |
| `lme_multi-session_gpt4_59c863d7` | `multi-session` | `25` | I have worked on or bought five model kits. The scales  | **PASS** (`60.07s`, `12381` tok) | PENDING | PENDING |
| `lme_multi-session_b5ef892d` | `multi-session` | `20` | 8 days. | **PASS** (`36.94s`, `12102` tok) | PENDING | PENDING |
| `lme_multi-session_e831120c` | `multi-session` | `14` | 3.5 weeks | **PASS** (`22.85s`, `11913` tok) | PENDING | PENDING |
| `lme_multi-session_3a704032` | `multi-session` | `20` | 3 | FAIL (`35.23s`) | PENDING | PENDING |
| `lme_multi-session_gpt4_d84a3211` | `multi-session` | `25` | $185 | **PASS** (`31.45s`, `12169` tok) | PENDING | PENDING |
| `lme_multi-session_aae3761f` | `multi-session` | `20` | 15 hours for getting to the three destinations (or 30 h | **PASS** (`39.64s`, `12212` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_f2262a51` | `multi-session` | `20` | I visited three different doctors: a primary care physi | **PASS** (`33.51s`, `12058` tok) | PENDING | PENDING |
| `lme_multi-session_dd2973ad` | `multi-session` | `14` | 2 AM | **PASS** (`42.65s`, `11607` tok) | PENDING | PENDING |
| `lme_multi-session_c4a1ceb8` | `multi-session` | `25` | 3 | **PASS** (`37.59s`, `12048` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_a56e767c` | `multi-session` | `20` | I attended four movie festivals. | **PASS** (`46.06s`, `12578` tok) | PENDING | PENDING |
| `lme_multi-session_6cb6f249` | `multi-session` | `14` | 17 days | **PASS** (`43.94s`, `11920` tok) | PENDING | PENDING |
| `lme_multi-session_46a3abf7` | `multi-session` | `20` | 3 | FAIL (`27.69s`) | PENDING | PENDING |
| `lme_multi-session_36b9f61e` | `multi-session` | `20` | $2,500 | **PASS** (`37.94s`, `12139` tok) | PENDING | PENDING |
| `lme_multi-session_28dc39ac` | `multi-session` | `32` | 140 hours | **PASS** (`45.1s`, `12074` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_2f8be40d` | `multi-session` | `20` | I attended three weddings. The couples were Rachel and  | **PASS** (`64.39s`, `12594` tok) | PENDING | PENDING |
| `lme_multi-session_2e6d26dc` | `multi-session` | `26` | 5 | **PASS** (`70.66s`, `12455` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_15e38248` | `multi-session` | `23` | 4 | **PASS** (`68.07s`, `12623` tok) | PENDING | PENDING |
| `lme_multi-session_88432d0a` | `multi-session` | `26` | 4 | **PASS** (`54.27s`, `12434` tok) | PENDING | PENDING |
| `lme_multi-session_80ec1f4f` | `multi-session` | `20` | 2 | **PASS** (`30.57s`, `12070` tok) | PENDING | PENDING |
| `lme_multi-session_d23cf73b` | `multi-session` | `26` | 4 | **PASS** (`58.01s`, `12489` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_7fce9456` | `multi-session` | `32` | I viewed four properties before making an offer on the  | **PASS** (`61.94s`, `12416` tok) | PENDING | PENDING |
| `lme_multi-session_d682f1a2` | `multi-session` | `20` | 3 | **PASS** (`56.7s`, `12243` tok) | PENDING | PENDING |
| `lme_multi-session_7024f17c` | `multi-session` | `20` | 0.5 hours | **PASS** (`69.35s`, `12480` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_5501fe77` | `multi-session` | `19` | TikTok | **PASS** (`118.4s`, `12846` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_2ba83207` | `multi-session` | `26` | Thrive Market | **PASS** (`44.14s`, `12522` tok) | PENDING | PENDING |
| `lme_multi-session_2318644b` | `multi-session` | `14` | $270 | **PASS** (`23.99s`, `11901` tok) | PENDING | PENDING |
| `lme_multi-session_2ce6a0f2` | `multi-session` | `25` | 4 | **PASS** (`51.29s`, `12151` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_d12ceb0e` | `multi-session` | `18` | 59.6 | **PASS** (`30.44s`, `12251` tok) | PENDING | PENDING |
| `lme_multi-session_00ca467f` | `multi-session` | `20` | 2 | **PASS** (`26.75s`, `12050` tok) | PENDING | PENDING |
| `lme_multi-session_b3c15d39` | `multi-session` | `14` | 5 days. 6 days (including the last day) is also accepta | **PASS** (`22.94s`, `11845` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_31ff4165` | `multi-session` | `32` | 4 | **PASS** (`61.45s`, `12372` tok) | PENDING | PENDING |
| `lme_multi-session_eeda8a6d` | `multi-session` | `14` | 17 | **PASS** (`37.86s`, `11714` tok) | PENDING | PENDING |
| `lme_multi-session_2788b940` | `multi-session` | `23` | 5 | **PASS** (`43.76s`, `12280` tok) | PENDING | PENDING |
| `lme_multi-session_60bf93ed` | `multi-session` | `14` | 5 days. 6 days (including the last day) is also accepta | **PASS** (`28.86s`, `11753` tok) | PENDING | PENDING |
| `lme_multi-session_9d25d4e0` | `multi-session` | `20` | 3 | **PASS** (`46.35s`, `12073` tok) | PENDING | PENDING |
| `lme_multi-session_129d1232` | `multi-session` | `19` | $5,850 | **PASS** (`46.85s`, `12169` tok) | PENDING | PENDING |
| `lme_multi-session_60472f9c` | `multi-session` | `20` | 2 | **PASS** (`35.41s`, `11806` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_194be4b3` | `multi-session` | `29` | I currently own 4 musical instruments. I've had the Fen | **PASS** (`35.31s`, `12130` tok) | PENDING | PENDING |
| `lme_multi-session_a9f6b44c` | `multi-session` | `20` | 2 | **PASS** (`62.17s`, `12299` tok) | PENDING | PENDING |
| `lme_multi-session_d851d5ba` | `multi-session` | `24` | $3,750 | **PASS** (`35.42s`, `11863` tok) | PENDING | PENDING |
| `lme_multi-session_5a7937c8` | `multi-session` | `18` | 3 days. | **PASS** (`49.88s`, `12147` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_ab202e7f` | `multi-session` | `32` | I replaced or fixed five items: the kitchen faucet, the | **PASS** (`65.83s`, `12474` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_e05b82a6` | `multi-session` | `25` | 10 times | **PASS** (`41.9s`, `12275` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_731e37d7` | `multi-session` | `25` | $720 | **PASS** (`51.83s`, `12289` tok) | PENDING | PENDING |
| `lme_multi-session_edced276` | `multi-session` | `11` | 15 days | **PASS** (`27.88s`, `10600` tok) | PENDING | PENDING |
| `lme_multi-session_10d9b85a` | `multi-session` | `13` | 3 days | **PASS** (`34.13s`, `10878` tok) | PENDING | PENDING |
| `lme_multi-session_e3038f8c` | `multi-session` | `26` | 99 | FAIL (`44.87s`) | PENDING | PENDING |
| `lme_multi-session_2b8f3739` | `multi-session` | `20` | $495 | **PASS** (`40.37s`, `12301` tok) | PENDING | PENDING |
| `lme_multi-session_1a8a66a6` | `multi-session` | `24` | 2 | **PASS** (`53.5s`, `12187` tok) | PENDING | PENDING |
| `lme_multi-session_c2ac3c61` | `multi-session` | `14` | 5 | **PASS** (`40.35s`, `11475` tok) | PENDING | PENDING |
| `lme_multi-session_bf659f65` | `multi-session` | `19` | 3 | **PASS** (`51.19s`, `12427` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_372c3eed` | `multi-session` | `19` | 10 years | **PASS** (`61.18s`, `12183` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_2f91af09` | `multi-session` | `19` | 23 | **PASS** (`33.97s`, `12274` tok) | PENDING | PENDING |
| `lme_multi-session_81507db6` | `multi-session` | `31` | 3 | **PASS** (`34.62s`, `11885` tok) | PENDING | PENDING |
| `lme_multi-session_88432d0a_abs` | `multi-session` | `26` | The information provided is not enough. You did not men | **PASS** (`71.86s`, `29948` tok) | PENDING | PENDING |
| `lme_multi-session_80ec1f4f_abs` | `multi-session` | `20` | 0. You did not mention visitng any museum in December | **PASS** (`78.2s`, `31775` tok) | PENDING | PENDING |
| `lme_multi-session_eeda8a6d_abs` | `multi-session` | `14` | The information provided is not enough. You did not men | **PASS** (`32.27s`, `11700` tok) | PENDING | PENDING |
| `lme_multi-session_60bf93ed_abs` | `multi-session` | `14` | The information provided is not enough. You did not men | **PASS** (`60.74s`, `29339` tok) | PENDING | PENDING |
| `lme_multi-session_edced276_abs` | `multi-session` | `11` | The information provided is not enough. You mentioned t | **PASS** (`28.17s`, `10630` tok) | PENDING | PENDING |
| `lme_multi-session_gpt4_372c3eed_abs` | `multi-session` | `19` | The information provided is not enough. You mentioned 4 | **PASS** (`61.13s`, `12162` tok) | PENDING | PENDING |
| `lme_multi-session_d3ab962e` | `multi-session` | `14` | 8 miles | **PASS** (`36.54s`, `11668` tok) | PENDING | PENDING |
| `lme_multi-session_2311e44b` | `multi-session` | `11` | 190 | **PASS** (`19.9s`, `10610` tok) | PENDING | PENDING |
| `lme_multi-session_cc06de0d` | `multi-session` | `14` | $6 | **PASS** (`21.57s`, `11589` tok) | PENDING | PENDING |
| `lme_multi-session_a11281a2` | `multi-session` | `14` | 100 | **PASS** (`32.97s`, `11609` tok) | PENDING | PENDING |
| `lme_multi-session_4f54b7c9` | `multi-session` | `14` | 5 | **PASS** (`34.61s`, `11797` tok) | PENDING | PENDING |
| `lme_multi-session_85fa3a3f` | `multi-session` | `14` | $50 | **PASS** (`36.29s`, `11639` tok) | PENDING | PENDING |
| `lme_multi-session_9aaed6a3` | `multi-session` | `14` | $0.75 | **PASS** (`55.27s`, `11537` tok) | PENDING | PENDING |
| `lme_multi-session_1f2b8d4f` | `multi-session` | `14` | $750 | **PASS** (`32.6s`, `11477` tok) | PENDING | PENDING |
| `lme_multi-session_e6041065` | `multi-session` | `13` | 40% | **PASS** (`30.03s`, `11580` tok) | PENDING | PENDING |
| `lme_multi-session_51c32626` | `multi-session` | `12` | February 1st | FAIL (`25.38s`) | PENDING | PENDING |
| `lme_multi-session_d905b33f` | `multi-session` | `14` | 20% | **PASS** (`21.57s`, `11632` tok) | PENDING | PENDING |
| `lme_multi-session_7405e8b1` | `multi-session` | `14` | Yes. | **PASS** (`25.42s`, `11453` tok) | PENDING | PENDING |
| `lme_multi-session_f35224e0` | `multi-session` | `13` | 27 | **PASS** (`41.89s`, `11500` tok) | PENDING | PENDING |
| `lme_multi-session_6456829e` | `multi-session` | `14` | 8 | **PASS** (`22.97s`, `11506` tok) | PENDING | PENDING |
| `lme_multi-session_a4996e51` | `multi-session` | `14` | 50 | **PASS** (`37.6s`, `11869` tok) | PENDING | PENDING |
| `lme_multi-session_3c1045c8` | `multi-session` | `14` | 2.5 years | **PASS** (`29.54s`, `11574` tok) | PENDING | PENDING |
| `lme_multi-session_60036106` | `multi-session` | `14` | 12,000 | **PASS** (`32.76s`, `11669` tok) | PENDING | PENDING |
| `lme_multi-session_681a1674` | `multi-session` | `14` | 2 | **PASS** (`39.96s`, `11832` tok) | PENDING | PENDING |
| `lme_multi-session_e25c3b8d` | `multi-session` | `14` | $300 | **PASS** (`34.38s`, `11489` tok) | PENDING | PENDING |
| `lme_multi-session_4adc0475` | `multi-session` | `14` | 5 | **PASS** (`41.21s`, `11737` tok) | PENDING | PENDING |
| `lme_multi-session_4bc144e2` | `multi-session` | `14` | $65 | **PASS** (`28.45s`, `11626` tok) | PENDING | PENDING |
| `lme_multi-session_ef66a6e5` | `multi-session` | `14` | two | **PASS** (`27.41s`, `11572` tok) | PENDING | PENDING |
| `lme_multi-session_5025383b` | `multi-session` | `14` | photography and cooking | **PASS** (`37.58s`, `11572` tok) | PENDING | PENDING |
| `lme_multi-session_a1cc6108` | `multi-session` | `12` | 11 | FAIL (`29.66s`) | PENDING | PENDING |
| `lme_multi-session_9ee3ecd6` | `multi-session` | `13` | 100 | **PASS** (`35.94s`, `11080` tok) | PENDING | PENDING |
| `lme_multi-session_3fdac837` | `multi-session` | `12` | 11 days (or 12 days, if April 15th to 22nd is considere | **PASS** (`33.31s`, `10647` tok) | PENDING | PENDING |
| `lme_multi-session_91b15a6e` | `multi-session` | `14` | $5,150 | **PASS** (`33.21s`, `11582` tok) | PENDING | PENDING |
| `lme_multi-session_27016adc` | `multi-session` | `14` | 10% | **PASS** (`42.2s`, `22707` tok) | PENDING | PENDING |
| `lme_multi-session_720133ac` | `multi-session` | `14` | $75 | **PASS** (`38.02s`, `11803` tok) | PENDING | PENDING |
| `lme_multi-session_77eafa52` | `multi-session` | `14` | $300 | **PASS** (`27.79s`, `11641` tok) | PENDING | PENDING |
| `lme_multi-session_8979f9ec` | `multi-session` | `13` | 8 meals | **PASS** (`29.4s`, `11488` tok) | PENDING | PENDING |
| `lme_multi-session_0100672e` | `multi-session` | `14` | $12 | **PASS** (`24.82s`, `11899` tok) | PENDING | PENDING |
| `lme_multi-session_a96c20ee` | `multi-session` | `13` | Harvard University | **PASS** (`32.93s`, `11036` tok) | PENDING | PENDING |
| `lme_multi-session_92a0aa75` | `multi-session` | `14` | 1 year and 5 months | **PASS** (`39.73s`, `11967` tok) | PENDING | PENDING |
| `lme_multi-session_3fe836c9` | `multi-session` | `13` | $25,000 | **PASS** (`24.46s`, `10909` tok) | PENDING | PENDING |
| `lme_multi-session_1c549ce4` | `multi-session` | `14` | $140 | **PASS** (`34.73s`, `11738` tok) | PENDING | PENDING |
| `lme_multi-session_6c49646a` | `multi-session` | `14` | 3,000 miles | **PASS** (`43.0s`, `11574` tok) | PENDING | PENDING |
| `lme_multi-session_1192316e` | `multi-session` | `14` | an hour and a half | FAIL (`35.66s`) | PENDING | PENDING |
| `lme_multi-session_0ea62687` | `multi-session` | `14` | 2 | **PASS** (`26.0s`, `11620` tok) | PENDING | PENDING |
| `lme_multi-session_67e0d0f2` | `multi-session` | `14` | 20 | **PASS** (`36.05s`, `11712` tok) | PENDING | PENDING |
| `lme_multi-session_bb7c3b45` | `multi-session` | `14` | $300 | **PASS** (`24.74s`, `11557` tok) | PENDING | PENDING |
| `lme_multi-session_ba358f49` | `multi-session` | `14` | 33 | **PASS** (`37.03s`, `11710` tok) | PENDING | PENDING |
| `lme_multi-session_61f8c8f8` | `multi-session` | `14` | 10 minutes | **PASS** (`21.27s`, `11630` tok) | PENDING | PENDING |
| `lme_multi-session_60159905` | `multi-session` | `13` | three | **PASS** (`43.82s`, `11489` tok) | PENDING | PENDING |
| `lme_multi-session_ef9cf60a` | `multi-session` | `13` | $300 | **PASS** (`29.29s`, `11454` tok) | PENDING | PENDING |
| `lme_multi-session_73d42213` | `multi-session` | `14` | 9:00 AM | **PASS** (`43.24s`, `11811` tok) | PENDING | PENDING |
| `lme_multi-session_bc149d6b` | `multi-session` | `13` | 70 pounds | **PASS** (`28.54s`, `11345` tok) | PENDING | PENDING |
| `lme_multi-session_099778bb` | `multi-session` | `14` | 20% | **PASS** (`32.88s`, `11372` tok) | PENDING | PENDING |
| `lme_multi-session_09ba9854` | `multi-session` | `13` | $50 | **PASS** (`28.48s`, `11045` tok) | PENDING | PENDING |
| `lme_multi-session_d6062bb9` | `multi-session` | `13` | 1,998 | **PASS** (`30.98s`, `11333` tok) | PENDING | PENDING |
| `lme_multi-session_157a136e` | `multi-session` | `13` | 43 | **PASS** (`34.01s`, `11587` tok) | PENDING | PENDING |
| `lme_multi-session_c18a7dc8` | `multi-session` | `13` | 7 | **PASS** (`39.98s`, `11318` tok) | PENDING | PENDING |
| `lme_multi-session_a3332713` | `multi-session` | `14` | $200 | **PASS** (`34.19s`, `11520` tok) | PENDING | PENDING |
| `lme_multi-session_55241a1f` | `multi-session` | `12` | 33 | **PASS** (`25.78s`, `10753` tok) | PENDING | PENDING |
| `lme_multi-session_a08a253f` | `multi-session` | `14` | 4 days. | **PASS** (`26.15s`, `11583` tok) | PENDING | PENDING |
| `lme_multi-session_f0e564bc` | `multi-session` | `14` | $1,300 | **PASS** (`36.32s`, `11784` tok) | PENDING | PENDING |
| `lme_multi-session_078150f1` | `multi-session` | `14` | $50 | **PASS** (`23.27s`, `11568` tok) | PENDING | PENDING |
| `lme_multi-session_8cf4d046` | `multi-session` | `14` | 3.83 | **PASS** (`33.28s`, `11537` tok) | PENDING | PENDING |
| `lme_multi-session_a346bb18` | `multi-session` | `14` | 12 | **PASS** (`24.05s`, `11929` tok) | PENDING | PENDING |
| `lme_multi-session_37f165cf` | `multi-session` | `12` | 856 | FAIL (`38.99s`) | PENDING | PENDING |
| `lme_multi-session_8e91e7d9` | `multi-session` | `13` | 4 | **PASS** (`31.39s`, `11186` tok) | PENDING | PENDING |
| `lme_multi-session_87f22b4a` | `multi-session` | `14` | $120 | **PASS** (`25.91s`, `11850` tok) | PENDING | PENDING |
| `lme_multi-session_e56a43b9` | `multi-session` | `14` | $5 | **PASS** (`23.11s`, `11819` tok) | PENDING | PENDING |
| `lme_multi-session_efc3f7c2` | `multi-session` | `14` | 30 minutes | **PASS** (`22.14s`, `11655` tok) | PENDING | PENDING |
| `lme_multi-session_21d02d0d` | `multi-session` | `13` | 2 | **PASS** (`38.77s`, `11613` tok) | PENDING | PENDING |
| `lme_multi-session_2311e44b_abs` | `multi-session` | `11` | The information provided is not enough. You did not men | **PASS** (`28.38s`, `10725` tok) | PENDING | PENDING |
| `lme_multi-session_6456829e_abs` | `multi-session` | `14` | The information provided is not enough. You mentioned p | **PASS** (`30.5s`, `11620` tok) | PENDING | PENDING |
| `lme_multi-session_e5ba910e_abs` | `multi-session` | `13` | The information provided is not enough. You mentioned p | **PASS** (`34.6s`, `11604` tok) | PENDING | PENDING |
| `lme_multi-session_a96c20ee_abs` | `multi-session` | `13` | The information provided is not enough. You did not men | **PASS** (`39.38s`, `11123` tok) | PENDING | PENDING |
| `lme_multi-session_ba358f49_abs` | `multi-session` | `14` | The information provided is not enough. You did not men | **PASS** (`29.29s`, `11718` tok) | PENDING | PENDING |
| `lme_multi-session_09ba9854_abs` | `multi-session` | `13` | The information provided is not enough. You did not men | **PASS** (`34.32s`, `11122` tok) | PENDING | PENDING |
| `lme_single-session-assistant_7161e7e2` | `single-session-assistant` | `9` | Admon was assigned to the 8 am - 4 pm (Day Shift) on Su | **PASS** (`37.46s`, `9561` tok) | **PASS** (`15.26s`, `13918` tok) | **PASS** (`88.86s`, `6041` tok) |
| `lme_single-session-assistant_c4f10528` | `single-session-assistant` | `9` | Miss Bee Providore | **PASS** (`23.9s`, `8787` tok) | **PASS** (`15.77s`, `13379` tok) | **PASS** (`68.6s`, `5251` tok) |
| `lme_single-session-assistant_89527b6b` | `single-session-assistant` | `3` | The Plesiosaur had a blue scaly body. | **PASS** (`36.85s`, `15560` tok) | PENDING | PENDING |
| `lme_single-session-assistant_e9327a54` | `single-session-assistant` | `9` | The Sugar Factory at Icon Park. | **PASS** (`19.48s`, `9051` tok) | PENDING | PENDING |
| `lme_single-session-assistant_4c36ccef` | `single-session-assistant` | `6` | Roscioli | **PASS** (`17.0s`, `7743` tok) | PENDING | PENDING |
| `lme_single-session-assistant_6ae235be` | `single-session-assistant` | `6` | Atmospheric distillation, fluid catalytic cracking (FCC | **PASS** (`26.17s`, `7871` tok) | PENDING | PENDING |
| `lme_single-session-assistant_7e00a6cb` | `single-session-assistant` | `5` | International Budget Hostel | **PASS** (`33.02s`, `7657` tok) | PENDING | PENDING |
| `lme_single-session-assistant_1903aded` | `single-session-assistant` | `3` | Transcriptionist. | **PASS** (`14.19s`, `6887` tok) | PENDING | PENDING |
| `lme_single-session-assistant_ceb54acb` | `single-session-assistant` | `4` | I suggested 'sexual fixations', 'problematic sexual beh | **PASS** (`31.63s`, `7257` tok) | PENDING | PENDING |
| `lme_single-session-assistant_f523d9fe` | `single-session-assistant` | `13` | Doc Martin | **PASS** (`22.19s`, `10299` tok) | PENDING | PENDING |
| `lme_single-session-assistant_0e5e2d1a` | `single-session-assistant` | `5` | 38 subjects | **PASS** (`30.43s`, `7760` tok) | PENDING | PENDING |
| `lme_single-session-assistant_fea54f57` | `single-session-assistant` | `9` | Evolution | **PASS** (`21.84s`, `9366` tok) | PENDING | PENDING |
| `lme_single-session-assistant_cc539528` | `single-session-assistant` | `7` | I recommended learning Ruby, Python, or PHP as a back-e | **PASS** (`22.51s`, `8314` tok) | PENDING | PENDING |
| `lme_single-session-assistant_dc439ea3` | `single-session-assistant` | `6` | Hoop Dance | FAIL (`25.14s`) | PENDING | PENDING |
| `lme_single-session-assistant_18dcd5a5` | `single-session-assistant` | `3` | 4 | FAIL (`51.23s`) | PENDING | PENDING |
| `lme_single-session-assistant_488d3006` | `single-session-assistant` | `9` | The GR-90 trail. | **PASS** (`19.0s`, `8878` tok) | PENDING | PENDING |
| `lme_single-session-assistant_58470ed2` | `single-session-assistant` | `4` | According to Borges, 'The Library is a sphere whose exa | FAIL (`35.1s`) | PENDING | PENDING |
| `lme_single-session-assistant_8cf51dda` | `single-session-assistant` | `3` | The three objectives were: 1) to identify molecular sub | **PASS** (`54.71s`, `15196` tok) | PENDING | PENDING |
| `lme_single-session-assistant_1d4da289` | `single-session-assistant` | `5` | I mentioned biometric authentication or one-time passwo | **PASS** (`25.33s`, `7721` tok) | PENDING | PENDING |
| `lme_single-session-assistant_8464fc84` | `single-session-assistant` | `8` | Roscioli | **PASS** (`18.76s`, `8661` tok) | PENDING | PENDING |
| `lme_single-session-assistant_8aef76bc` | `single-session-assistant` | `6` | Mod Podge or another sealant | **PASS** (`15.95s`, `7913` tok) | PENDING | PENDING |
| `lme_single-session-assistant_71a3fd6b` | `single-session-assistant` | `5` | +49 (0) 62 32 / 14 23 - 0 | **PASS** (`15.78s`, `7557` tok) | PENDING | PENDING |
| `lme_single-session-assistant_2bf43736` | `single-session-assistant` | `8` | Chapter 4 of Book 1, titled 'Vocal Prayer and Meditatio | **PASS** (`30.55s`, `8635` tok) | PENDING | PENDING |
| `lme_single-session-assistant_70b3e69b` | `single-session-assistant` | `9` | Manolo García | **PASS** (`29.53s`, `9229` tok) | PENDING | PENDING |
| `lme_single-session-assistant_8752c811` | `single-session-assistant` | `5` | The 27th parameter was 'Sound effects (e.g., ambient, d | FAIL (`40.3s`) | PENDING | PENDING |
| `lme_single-session-assistant_3249768e` | `single-session-assistant` | `4` | Absinthe | FAIL (`33.95s`) | PENDING | PENDING |
| `lme_single-session-assistant_1b9b7252` | `single-session-assistant` | `6` | Mindful.org. | **PASS** (`18.65s`, `7995` tok) | PENDING | PENDING |
| `lme_single-session-assistant_1568498a` | `single-session-assistant` | `14` | 28. Kg3 | **PASS** (`19.76s`, `12035` tok) | PENDING | PENDING |
| `lme_single-session-assistant_6222b6eb` | `single-session-assistant` | `10` | The 6S algorithm is implemented in the SIAC_GEE tool. | **PASS** (`36.52s`, `9188` tok) | PENDING | PENDING |
| `lme_single-session-assistant_e8a79c70` | `single-session-assistant` | `6` | 2-3 eggs | **PASS** (`21.3s`, `8039` tok) | PENDING | PENDING |
| `lme_single-session-assistant_d596882b` | `single-session-assistant` | `6` | By Chloe | **PASS** (`19.02s`, `8140` tok) | PENDING | PENDING |
| `lme_single-session-assistant_e3fc4d6e` | `single-session-assistant` | `3` | Dr. Arati Prabhakar | FAIL (`40.23s`) | PENDING | PENDING |
| `lme_single-session-assistant_51b23612` | `single-session-assistant` | `8` | Nu, pogodi! | **PASS** (`50.68s`, `8878` tok) | PENDING | PENDING |
| `lme_single-session-assistant_3e321797` | `single-session-assistant` | `6` | 10 minutes | **PASS** (`19.52s`, `7824` tok) | PENDING | PENDING |
| `lme_single-session-assistant_e982271f` | `single-session-assistant` | `5` | Revolution Hall | **PASS** (`18.18s`, `7621` tok) | PENDING | PENDING |
| `lme_single-session-assistant_352ab8bd` | `single-session-assistant` | `10` | The average improvement in framerate was approximately  | **PASS** (`21.25s`, `9067` tok) | PENDING | PENDING |
| `lme_single-session-assistant_fca762bc` | `single-session-assistant` | `5` | Memrise | **PASS** (`35.04s`, `7847` tok) | PENDING | PENDING |
| `lme_single-session-assistant_7a8d0b71` | `single-session-assistant` | `5` | $2,000 | FAIL (`41.4s`) | PENDING | PENDING |
| `lme_single-session-assistant_a40e080f` | `single-session-assistant` | `6` | Patagonia and Southwest Airlines. | FAIL (`29.58s`) | PENDING | PENDING |
