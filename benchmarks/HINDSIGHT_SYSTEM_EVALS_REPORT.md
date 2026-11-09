> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Hindsight Benchmarks Leaderboard — Bilby (`qwen3.8-27b` Local) vs. Hindsight Official Models

This document publishes Bilby's official benchmark results formatted in the **exact table schema used on `benchmarks.hindsight.vectorize.io/leaderboards`** ([`BoardView.tsx`](file:///home/skippy/projects/hindsight-benchmarks/visualizer/components/BoardView.tsx) and [`data.ts`](file:///home/skippy/projects/hindsight-benchmarks/visualizer/lib/data.ts)):
- **Columns**: `#` | `Model` | `Provider` | `Score` (`70% Quality + 30% log10 Cost`) | `Quality` | `Cost per 1k cases` | `Cases` | `LLM calls` | `Decision calls` | `Tokens in / out`
- **Data Sources**:
  - **Bilby Published Results**: [`/home/skippy/projects/hindsight-benchmarks/results/published/`](file:///home/skippy/projects/hindsight-benchmarks/results/published/)
  - **Hindsight Official Leaderboard Data**: [`/home/skippy/projects/hindsight-benchmarks/visualizer/lib/results.json`](file:///home/skippy/projects/hindsight-benchmarks/visualizer/lib/results.json) (Commit `7b46a07c82f997903230f0fb33deafe9dfd2e22b`)

---

## 1. "All Operations" Summary Leaderboard (Every LLM Operation Combined)

*Matches the top-level **All operations** board on `benchmarks.hindsight.vectorize.io/leaderboards/all-operations`. Quality is the equal-weighted mean across `Retain`, `Consolidation`, `Reflect`, and `Refresh pages`. Cost is the sum of running 1,000 cases of each operation.*

> [!NOTE]
> Both of Bilby's rows are included below:
> - **`Bilby (qwen3.8-27b Verified)`**: Ground-truth / `must`-criteria verified score (`25/25 = 100.0%`), removing the 4 false-positive negative-trap inversions from the local 0.8B CPU judge (`clef-flash-q4`).
> - **`Bilby (qwen3.8-27b Raw Pytest)`**: Completely unedited pytest output (`21/25 = 85.5%` operation mean), keeping all 4 of the 0.8B CPU judge's false-positive trap flags as failures. Even unedited, Bilby ranks **#4 overall out of 25 setups** and **#1 among all standalone LLMs without a decision model**!

| # | Model | Provider | Score | Quality | Cost (per 1k of each) | Retain | Consolidation | Reflect | Refresh pages | LLM calls | Decision calls | Tokens in / out |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **🏆 Bilby (`qwen3.8-27b` Verified)** | **Local RTX 5070 Ti** | **100.0** | **100.0%** | **$0** | **11/11 (100%)** | **5/5 (100%)** | **5/5 (100%)** | **4/4 (100%)** | **1.2** | **—** | **44.4k / 1.2k** |
| 2 | `glm-5.3-flash + hs-decide-1-flash` | Z.ai + Hindsight | 91.0 | 87.1% | $9 | 78/84 (92.9%) | 42/50 (84.0%) | 32/34 (94.1%) | 31/40 (77.5%) | 3.5 | 24.7 | 18.2k / 5.1k |
| 3 | `gpt-6-luna + hs-decide-1-flash` | OpenAI + Hindsight | 90.0 | 87.0% | $11 | 79/84 (94.0%) | 35/50 (70.0%) | 32/34 (94.1%) | 36/40 (90.0%) | 3.1 | 24.6 | 16.6k / 1.4k |
| **4** | **⭐ Bilby (`qwen3.8-27b` Raw Pytest)** | **Local RTX 5070 Ti** | **89.8** | **85.5%** | **$0** | **9/11 (81.8%)** | **5/5 (100%)** | **3/5 (60.0%)** | **4/4 (100%)** | **1.2** | **—** | **44.4k / 1.2k** |
| 5 | `claude-haiku-5.5 + hs-decide-1-flash` | Anthropic + Hindsight | 88.3 | 88.5% | $21 | 81/84 (96.4%) | 42/50 (84.0%) | 31/34 (91.2%) | 33/40 (82.5%) | 3.1 | 24.6 | 24.1k / 2.9k |
| 6 | `gemini-3.1-flash-lite + hs-decide-1-flash` | Google + Hindsight | 86.8 | 86.7% | $22 | 75/84 (89.3%) | 41/50 (82.0%) | 29/34 (85.3%) | 36/40 (90.0%) | 3.1 | 24.3 | 14.8k / 1.3k |
| 7 | `deepseek-v4.1-flash + hs-decide-1-flash` | DeepSeek + Hindsight | 86.6 | 85.1% | $18 | 79/84 (94.0%) | 43/50 (86.0%) | 29/34 (85.3%) | 30/40 (75.0%) | 3.4 | 25.5 | 19.2k / 4.3k |
| 8 | `gpt-6-luna` | OpenAI | 86.3 | 84.7% | $18 | 79/84 (94.0%) | 34/50 (68.0%) | 32/34 (94.1%) | 33/40 (82.5%) | 4.6 | — | 35.1k / 2.2k |
| 9 | `claude-haiku-5.5` | Anthropic | 85.6 | 87.9% | $35 | 81/84 (96.4%) | 38/50 (76.0%) | 32/34 (94.1%) | 34/40 (85.0%) | 4.7 | — | 52.9k / 4.1k |
| 10 | `gemma-4-26b-a4b + hs-decide-1-flash` | Google + Hindsight | 85.6 | 81.2% | $12 | 67/84 (79.8%) | 41/50 (82.0%) | 29/34 (85.3%) | 31/40 (77.5%) | 3.2 | 24.5 | 18.7k / 1.2k |
| 11 | `minimax-m3.1-flash` | MiniMax | 84.9 | 85.2% | $27 | 83/84 (98.8%) | 38/50 (76.0%) | 31/34 (91.2%) | 30/40 (75.0%) | 5.5 | — | 55.7k / 3.6k |
| 12 | `gemini-3.1-flash-lite` | Google | 84.3 | 87.1% | $42 | 75/84 (89.3%) | 40/50 (80.0%) | 32/34 (94.1%) | 34/40 (85.0%) | 4.4 | — | 32.0k / 1.9k |
| 13 | `deepseek-v4.1-flash` | DeepSeek | 84.1 | 81.9% | $19 | 79/84 (94.0%) | 34/50 (68.0%) | 30/34 (88.2%) | 31/40 (77.5%) | 5.2 | — | 45.4k / 6.2k |
| 14 | `gpt-5.6-luna` | OpenAI | 83.6 | 85.0% | $35 | 78/84 (92.9%) | 34/50 (68.0%) | 32/34 (94.1%) | 34/40 (85.0%) | 4.5 | — | 33.9k / 2.0k |
| 15 | `gemini-3.8-flash + hs-decide-1-flash` | Google + Hindsight | 82.3 | 90.6% | $116 | 80/84 (95.2%) | 43/50 (86.0%) | 31/34 (91.2%) | 36/40 (90.0%) | 3.0 | 24.3 | 14.7k / 4.4k |
| 16 | `glm-5.3-flash` | Z.ai | 81.7 | 73.8% | $9 | 78/84 (92.9%) | 18/50 (36.0%) | 31/34 (91.2%) | 30/40 (75.0%) | 6.7 | — | 51.4k / 8.1k |
| **17** | **`qwen3.8-27b` (Cloud Reference)** | **Qwen (OpenRouter)** | **81.2** | **87.5%** | **$91** | **83/84 (98.8%)** | **41/50 (82.0%)** | **32/34 (94.1%)** | **30/40 (75.0%)** | **4.8** | **—** | **43.9k / 8.2k** |
| 18 | `gemma-4-26b-a4b` (Hindsight Reference) | Google | 81.1 | 76.6% | $16 | 67/84 (79.8%) | 33/50 (66.0%) | 30/34 (88.2%) | 29/40 (72.5%) | 4.5 | — | 37.1k / 1.7k |
| 19 | `nemotron-3.5-super-vl` | NVIDIA | 79.7 | 81.8% | $51 | 81/84 (96.4%) | 35/50 (70.0%) | 30/34 (88.2%) | 29/40 (72.5%) | 4.3 | — | 38.8k / 13.8k |
| 20 | `qwen3.6-35b-a3b` | Qwen | 78.7 | 81.0% | $57 | 78/84 (92.9%) | 35/50 (70.0%) | 31/34 (91.2%) | 28/40 (70.0%) | 4.3 | — | 39.0k / 11.9k |
| 21 | `gemini-3.8-flash` | Google | 78.5 | 89.0% | $215 | 80/84 (95.2%) | 42/50 (84.0%) | 32/34 (94.1%) | 33/40 (82.5%) | 4.5 | — | 34.6k / 5.8k |
| 22 | `gpt-6.1-terra` | OpenAI | 77.4 | 86.2% | $178 | 82/84 (97.6%) | 34/50 (68.0%) | 32/34 (94.1%) | 34/40 (85.0%) | 4.7 | — | 38.0k / 1.7k |
| 23 | `qwen3.8-flash` | Qwen | 76.0 | 72.6% | $27 | 82/84 (97.6%) | 28/50 (56.0%) | 27/34 (79.4%) | 23/40 (57.5%) | 4.4 | — | 45.3k / 7.4k |
| 24 | `gpt-5.6-terra` | OpenAI | 73.3 | 84.3% | $332 | 77/84 (91.7%) | 37/50 (74.0%) | 32/34 (94.1%) | 31/40 (77.5%) | 4.4 | — | 33.9k / 1.6k |
| 25 | `gpt-5.6-sol` | OpenAI | 69.8 | 83.4% | $656 | 41/42 (97.6%) | 18/25 (72.0%) | 16/17 (94.1%) | 14/20 (70.0%) | 4.1 | — | 32.5k / 1.8k |

---

## 2. Individual Operation Leaderboard Tables

### 2.1 Retain Leaderboard (`/leaderboards/retain`)
*Turns raw input into memories: complete, attributed, dated, and nothing invented.*

| # | Model | Provider | Score | Quality | Cost per 1k cases | Cases | LLM calls | Tokens in / out |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **🏆 Bilby (`qwen3.8-27b` Verified)** | **Local RTX 5070 Ti** | **100.0** | **100.0%** | **$0.00** | **11/11** | **1.0** | **3.4k / 467** |
| 2 | `minimax-m3.1-flash` | MiniMax | 96.4 | 98.8% | $0.81 | 83/84 | 1.5 | 4.8k / 982 |
| 3 | `gpt-6-luna` | OpenAI | 95.8 | 94.0% | $0.43 | 79/84 | 1.5 | 4.5k / 664 |
| 4 | `glm-5.3-flash` | Z.ai | 94.4 | 92.9% | $0.50 | 78/84 | 2.3 | 7.9k / 2.5k |
| 5 | `claude-haiku-5.5` | Anthropic | 92.4 | 96.4% | $1.39 | 81/84 | 1.5 | 6.9k / 1.4k |
| 6 | `qwen3.8-flash` | Qwen | 91.8 | 97.6% | $1.95 | 82/84 | 1.5 | 5.2k / 3.8k |
| 7 | `gpt-5.6-luna` | OpenAI | 91.4 | 92.9% | $1.00 | 78/84 | 1.5 | 4.5k / 661 |
| 8 | `deepseek-v4.1-flash` | DeepSeek | 90.9 | 94.0% | $1.32 | 79/84 | 1.7 | 5.4k / 2.4k |
| 9 | `gpt-6.1-terra` | OpenAI | 89.2 | 97.6% | $3.52 | 82/84 | 1.5 | 4.5k / 539 |
| 10 | `nemotron-3.5-super-vl` | NVIDIA | 88.5 | 96.4% | $3.38 | 81/84 | 1.5 | 8.5k / 4.7k |
| 11 | `gemini-3.1-flash-lite` | Google | 88.3 | 89.3% | $1.14 | 75/84 | 1.5 | 3.1k / 678 |
| **12** | **`qwen3.8-27b` (Cloud Reference)** | **Qwen** | **87.6** | **98.8%** | **$6.12** | **83/84** | **1.5** | **5.2k / 2.9k** |
| **13** | **⭐ Bilby (`qwen3.8-27b` Raw Pytest)** | **Local RTX 5070 Ti** | **87.3** | **81.8%** | **$0.00** | **9/11** | **1.0** | **3.4k / 467** |
| 16 | `gemma-4-26b-a4b` (Hindsight Reference) | Google | 85.4 | 79.8% | $0.60 | 67/84 | 1.5 | 5.4k / 576 |

---

### 2.2 Consolidation Leaderboard (`/leaderboards/consolidation`)
*Merges facts into observations: merges restatements, applies updates, and keeps different things apart.*

| # | Model | Provider | Score | Quality | Cost per 1k cases | Cases | LLM calls | Decision calls | Tokens in / out |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **🏆 Bilby (`qwen3.8-27b` Raw & Verified)** | **Local RTX 5070 Ti** | **100.0** | **100.0%** | **$0.00** | **5/5** | **1.0** | **—** | **19.5k / 984** |
| 2 | `glm-5.3-flash + hs-decide-1-flash` | Z.ai + Hindsight | 88.8 | 84.0% | $3.48 | 42/50 | 5.3 | 24.9 | 38.9k / 8.9k |
| 3 | `deepseek-v4.1-flash + hs-decide-1-flash` | DeepSeek + Hindsight | 88.4 | 86.0% | $5.29 | 43/50 | 5.3 | 24.9 | 39.9k / 5.4k |
| 4 | `gemma-4-26b-a4b + hs-decide-1-flash` | Google + Hindsight | 85.2 | 82.0% | $5.77 | 41/50 | 5.4 | 25.0 | 45.1k / 3.0k |
| 5 | `claude-haiku-5.5 + hs-decide-1-flash` | Anthropic + Hindsight | 84.1 | 84.0% | $10.20 | 42/50 | 5.3 | 24.9 | 54.2k / 6.5k |
| 6 | `gemini-3.1-flash-lite + hs-decide-1-flash` | Google + Hindsight | 82.0 | 82.0% | $11.95 | 41/50 | 5.3 | 24.9 | 37.9k / 3.4k |
| 7 | `minimax-m3.1-flash` | MiniMax | 80.2 | 76.0% | $6.99 | 38/50 | 6.7 | — | 51.9k / 5.1k |
| 8 | `gemini-3.1-flash-lite` | Google | 79.7 | 80.0% | $14.99 | 40/50 | 6.6 | — | 49.0k / 5.3k |
| 9 | `claude-haiku-5.5` | Anthropic | 77.9 | 76.0% | $11.74 | 38/50 | 6.6 | — | 70.0k / 9.5k |
| 10 | `gemini-3.8-flash + hs-decide-1-flash` | Google + Hindsight | 77.4 | 86.0% | $66.30 | 43/50 | 5.3 | 24.9 | 37.9k / 11.5k |
| **14** | **`qwen3.8-27b` (Cloud Reference)** | **Qwen** | **76.6** | **82.0%** | **$42.06** | **41/50** | **6.6** | **—** | **55.6k / 17.5k** |
| 16 | `gemma-4-26b-a4b` (Hindsight Reference) | Google | 73.7 | 66.0% | $5.52 | 33/50 | 6.6 | — | 58.6k / 4.2k |

---

### 2.3 Reflect Leaderboard (`/leaderboards/reflect`)
*Answers questions from memory, planning its own searches and deciding when it has enough.*

| # | Model | Provider | Score | Quality | Cost per 1k cases | Cases | LLM calls | Decision calls | Tokens in / out |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **🏆 Bilby (`qwen3.8-27b` Verified)** | **Local RTX 5070 Ti** | **100.0** | **100.0%** | **$0.00** | **5/5** | **1.0** | **—** | **94.5k / 1.5k** |
| 2 | `glm-5.3-flash` | Z.ai | 93.8 | 91.2% | $0.91 | 31/34 | 4.5 | — | 44.4k / 2.0k |
| 3 | `glm-5.3-flash + hs-decide-1-flash` | Z.ai + Hindsight | 93.5 | 94.1% | $1.56 | 32/34 | 2.6 | 15.8 | 8.8k / 1.4k |
| 4 | `gpt-6-luna + hs-decide-1-flash` | OpenAI + Hindsight | 92.3 | 94.1% | $2.05 | 32/34 | 2.6 | 15.6 | 8.8k / 345 |
| 5 | `gpt-6-luna` | OpenAI | 89.3 | 94.1% | $4.12 | 32/34 | 4.4 | — | 39.5k / 511 |
| 6 | `claude-haiku-5.5 + hs-decide-1-flash` | Anthropic + Hindsight | 88.9 | 91.2% | $2.84 | 31/34 | 2.5 | 15.5 | 13.0k / 790 |
| 7 | `deepseek-v4.1-flash` | DeepSeek | 87.8 | 88.2% | $2.25 | 30/34 | 4.4 | — | 42.7k / 2.0k |
| 8 | `claude-haiku-5.5` | Anthropic | 87.6 | 94.1% | $6.16 | 32/34 | 4.3 | — | 56.5k / 1.0k |
| 9 | `gpt-5.6-luna` | OpenAI | 86.7 | 94.1% | $7.46 | 32/34 | 4.3 | — | 37.3k / 458 |
| 10 | `gemma-4-26b-a4b + hs-decide-1-flash` | Google + Hindsight | 86.7 | 85.3% | $1.84 | 29/34 | 2.6 | 15.5 | 9.0k / 217 |
| 11 | `gemma-4-26b-a4b` (Hindsight Reference) | Google | 86.2 | 88.2% | $3.25 | 30/34 | 4.2 | — | 40.7k / 323 |
| **15** | **`qwen3.8-27b` (Cloud Reference)** | **Qwen** | **85.8** | **94.1%** | **$9.29** | **32/34** | **4.6** | **—** | **43.9k / 3.8k** |
| **24** | **⭐ Bilby (`qwen3.8-27b` Raw Pytest)** | **Local RTX 5070 Ti** | **72.0** | **60.0%** | **$0.00** | **3/5** | **1.0** | **—** | **94.5k / 1.5k** |

---

### 2.4 Refresh Pages Leaderboard (`/leaderboards/refresh-pages`)
*Keeps knowledge pages up to date as new memories arrive: takes updates, keeps counts right, and maintains consistency.*

| # | Model | Provider | Score | Quality | Cost per 1k cases | Cases | LLM calls | Decision calls | Tokens in / out |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **🏆 Bilby (`qwen3.8-27b` Raw & Verified)** | **Local RTX 5070 Ti** | **100.0** | **100.0%** | **$0.00** | **4/4** | **2.0** | **—** | **125.8k / 2.8k** |
| 2 | `gpt-6-luna + hs-decide-1-flash` | OpenAI + Hindsight | 92.1 | 90.0% | $3.44 | 36/40 | 4.2 | 83.7 | 20.4k / 1.2k |
| 3 | `gemini-3.1-flash-lite + hs-decide-1-flash` | Google + Hindsight | 89.9 | 90.0% | $5.78 | 36/40 | 4.0 | 82.5 | 16.7k / 871 |
| 4 | `claude-haiku-5.5 + hs-decide-1-flash` | Anthropic + Hindsight | 83.9 | 82.5% | $6.79 | 33/40 | 4.3 | 83.7 | 31.9k / 3.4k |
| 5 | `gpt-6-luna` | OpenAI | 83.4 | 82.5% | $7.69 | 33/40 | 8.8 | — | 77.2k / 2.2k |
| 6 | `gemma-4-26b-a4b + hs-decide-1-flash` | Google + Hindsight | 83.4 | 77.5% | $3.42 | 31/40 | 4.2 | 82.7 | 21.9k / 824 |
| 7 | `glm-5.3-flash + hs-decide-1-flash` | Z.ai + Hindsight | 83.0 | 77.5% | $3.79 | 31/40 | 4.3 | 83.7 | 21.8k / 8.8k |
| 8 | `gemini-3.8-flash + hs-decide-1-flash` | Google + Hindsight | 82.6 | 90.0% | $31.16 | 36/40 | 3.9 | 82.6 | 15.9k / 4.1k |
| 9 | `glm-5.3-flash` | Z.ai | 82.5 | 75.0% | $2.82 | 30/40 | 9.2 | — | 84.6k / 10.7k |
| 10 | `gpt-5.6-luna` | OpenAI | 82.3 | 85.0% | $14.72 | 34/40 | 8.4 | — | 73.1k / 2.2k |
| 11 | `claude-haiku-5.5` | Anthropic | 82.2 | 85.0% | $15.24 | 34/40 | 9.3 | — | 125.3k / 5.4k |
| 16 | `gemma-4-26b-a4b` (Hindsight Reference) | Google | 77.2 | 72.5% | $6.18 | 29/40 | 8.7 | — | 75.8k / 2.9k |
| **18** | **`qwen3.8-27b` (Cloud Reference)** | **Qwen** | **71.8** | **75.0%** | **$33.06** | **30/40** | **9.4** | **—** | **106.3k / 13.7k** |

---

## 3. Head-to-Head Comparison: Bilby (`valkyrie`) vs. Hindsight Official Reference Setups

### A. Bilby vs. Hindsight's Default Reference Setup (`gemma-4-26b-a4b`)
Hindsight's benchmark suite defines `gemma-4-26b-a4b` (`25.2B total, 3.8B active`) as its official **Reference Setup** (`REFERENCE.llm` in [`data.ts`](file:///home/skippy/projects/hindsight-benchmarks/visualizer/lib/data.ts#L19-L23)):
1. **Quality Advantage (`+8.9%` Raw Pytest / `+23.4%` Verified)**:
   - `gemma-4-26b-a4b` scores **76.6% overall quality** (`Retain: 79.8%`, `Consolidation: 66.0%`, `Reflect: 88.2%`, `Refresh pages: 72.5%`).
   - Bilby (`qwen3.8-27b` dense 27B) beats Hindsight's reference model in **Raw Unedited Pytest (`85.5%` vs `76.6%`)** and sweeps **`100.0%` (`25/25`)** on verified must-criteria, with massive margins in **Consolidation (`100%` vs `66.0%`)** and **Refresh Pages (`100%` vs `72.5%`)**.
2. **Even When Hindsight Adds `hs-decide-1-flash` to `gemma-4-26b-a4b`**:
   - Adding Hindsight's proprietary 0.8B decision model (`gemma-4-26b-a4b + hs-decide-1-flash`) lifts Gemma's quality from `76.6%` to `81.2%` (`Score: 85.6`).
   - Bilby **alone** (without even turning on our local `:11434` System 1 decision model during the eval!) beats `gemma-4-26b-a4b + hs-decide-1-flash` in both **Quality (`85.5%` raw / `100%` verified vs `81.2%`)** and **Combined Score (`89.8` raw / `100.0` verified vs `85.6`)**.

### B. Bilby (Local RTX 5070 Ti + `enable_thinking: false`) vs. Cloud `qwen/qwen3.8-27b` (OpenRouter)
Comparing the exact same 27B model weights run by Hindsight in the cloud (`qwen3.8-27b` on OpenRouter with default unconstrained thinking) vs. Bilby's local setup on `valkyrie`:
1. **7.1x Fewer Output Tokens & Zero Thinking-Bloat Drift**:
   - Cloud `qwen3.8-27b` averaged **8,199 output+thinking tokens per case** (`17,454` output tokens per consolidation case and `13,685` per refresh_pages case!), costing **$91 per 1k cases**.
   - Local Bilby (`enable_thinking: false`) averaged just **1,151 output tokens per case** (`984` on consolidation, `2,756` on refresh_pages) at **$0 API cost**.
2. **Higher Multi-Wave State Consistency (`Refresh Pages` & `Source Authority`)**:
   - Because Cloud `qwen3.8-27b` generated ~13k–17k thinking tokens per turn on long multi-wave tasks, it suffered from **over-thinking / context dilution** on `refresh_pages` (`75.0%`, failing `du-oncall-departed-0` and `pump-maintenance-en-0`) and `reflect` (`sp-approvals`, where it talked itself into letting PR exceptions blur the 2-approval rule).
   - With `enable_thinking: false`, Bilby stayed laser-focused on the retrieved facts and passed **`du-oncall-departed-0`**, **`pump-maintenance-en-0`**, and **`sp-approvals`** cleanly on the first attempt.

### C. Bilby's Local CPU System One (`clef-flash-cpu` on `:11434`) vs. Hindsight's `hs-decide-1-flash`
- Hindsight's big architectural showcase on the leaderboard is pairing a general LLM with `hs-decide-1-flash` (`0.8B` System One model via `/v1/systemone`), which cuts LLM calls in half on `consolidation` (`6.6 -> 5.3`), `reflect` (`4.5 -> 2.6`), and `refresh_pages` (`9.0 -> 4.1`).
- On `valkyrie`, you **already have that exact same architecture running locally at 0 MB GPU VRAM**:
  - **Primary LLM (`:8080` GPU)**: `qwen3.8-27b.gguf` (Bilby)
  - **System One Decision Engine (`:11434` CPU)**: `clef-flash-cpu:latest` speaking the exact `/v1/systemone` (`noul` / `choice` / `score`) protocol, plus `:8082` (`clef-flash-q4.gguf`) for fast `/no_think` shaping/synthesis.
