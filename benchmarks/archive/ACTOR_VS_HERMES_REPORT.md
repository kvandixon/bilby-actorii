> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Head-to-Head Benchmark Report: `ActorII AgentHarness` vs. `Hermes Agent` (Option C — 30 Tasks)

- **Shared Local Model (`:8080`)**: `qwen3.8:latest` (`qwen3.8-27b.gguf` on RTX 5070 Ti)
- **Completed Tasks**: `30/30`
- **Timestamp**: `2026-11-01T14:58:01-0500`

## 1. Executive Summary (`ActorII AgentHarness` vs. `Hermes Agent`)

| Track | Tasks ($N$) | **`ActorII AgentHarness` Accuracy** | **`Hermes Agent` Accuracy** | `ActorII` Total Tokens | `Hermes` Total Tokens | `ActorII` Avg Latency | `Hermes` Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HumanEval** | 15 | **14/15 (93.3%)** | 8/15 (53.3%) | `40,414` | `156,730` | `33.15s` | `74.07s` |
| **GSM8K** | 10 | **9/10 (90.0%)** | 6/10 (60.0%) | `74,902` | `212,282` | `40.3s` | `71.06s` |
| **MemGraph** | 5 | **5/5 (100.0%)** | 0/5 (0.0%) | `17,129` | `16,501` | `9.55s` | `110.05s` |
| **OVERALL (30 Tasks)** | **30** | **28/30 (93.3%)** | **14/30 (46.7%)** | **`132,445`** | **`385,513`** | `31.6s` | `79.06s` |

## 2. Per-Task Breakdown (All 30 Tasks)

| Track | Task ID | Title / Entry Point | **`ActorII` Result** | `ActorII` Steps / Tokens / Time | **`Hermes` Result** | `Hermes` Calls / Tokens / Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| HumanEval | `HumanEval/0` | `has_close_elements` | **PASS** | 1st / 1803tok / 19.67s | FAIL | 1c / 3327tok / 90.1s |
| HumanEval | `HumanEval/2` | `truncate_number` | **PASS** | 2st / 3764tok / 52.29s | **PASS** | 1c / 16738tok / 71.39s |
| HumanEval | `HumanEval/4` | `mean_absolute_deviation` | **PASS** | 1st / 1818tok / 19.29s | FAIL | 1c / 3348tok / 90.1s |
| HumanEval | `HumanEval/5` | `intersperse` | **PASS** | 1st / 1770tok / 18.54s | **PASS** | 1c / 16695tok / 64.2s |
| HumanEval | `HumanEval/9` | `rolling_max` | **PASS** | 2st / 3932tok / 44.39s | FAIL | 1c / 3310tok / 90.1s |
| HumanEval | `HumanEval/12` | `longest` | **PASS** | 1st / 1786tok / 17.66s | FAIL | 1c / 3331tok / 90.1s |
| HumanEval | `HumanEval/18` | `how_many_times` | **PASS** | 2st / 4049tok / 85.46s | FAIL | 1c / 3313tok / 90.1s |
| HumanEval | `HumanEval/28` | `concatenate` | **PASS** | 1st / 1725tok / 14.07s | **PASS** | 1c / 16682tok / 67.1s |
| HumanEval | `HumanEval/30` | `get_positive` | **PASS** | 1st / 1732tok / 15.99s | FAIL | 1c / 3296tok / 90.1s |
| HumanEval | `HumanEval/36` | `fizz_buzz` | FAIL | 4st / 9123tok / 119.44s | FAIL | 1c / 3295tok / 90.1s |
| HumanEval | `HumanEval/40` | `triples_sum_to_zero` | **PASS** | 1st / 1841tok / 22.98s | **PASS** | 1c / 16893tok / 81.62s |
| HumanEval | `HumanEval/42` | `incr_list` | **PASS** | 1st / 1711tok / 13.11s | **PASS** | 1c / 16465tok / 28.88s |
| HumanEval | `HumanEval/43` | `pairs_sum_to_zero` | **PASS** | 1st / 1822tok / 19.23s | **PASS** | 1c / 16740tok / 60.79s |
| HumanEval | `HumanEval/46` | `fib4` | **PASS** | 1st / 1802tok / 21.18s | **PASS** | 1c / 16899tok / 81.37s |
| HumanEval | `HumanEval/48` | `is_palindrome` | **PASS** | 1st / 1736tok / 13.97s | **PASS** | 1c / 16398tok / 25.02s |
| GSM8K | `GSM8K/8` | `John drives for 3 hours at a speed of 60 mph and...` | **PASS** | 2st / 7676tok / 36.41s | **PASS** | 2c / 33413tok / 73.3s |
| GSM8K | `GSM8K/9` | `Eliza's rate per hour for the first 40 hours she...` | **PASS** | 2st / 7240tok / 60.57s | **PASS** | 2c / 33069tok / 49.13s |
| GSM8K | `GSM8K/10` | `A new program had 60 downloads in the first mont...` | **PASS** | 1st / 3471tok / 19.81s | FAIL | 1c / 3292tok / 90.11s |
| GSM8K | `GSM8K/11` | `Toula went to the bakery and bought various type...` | **PASS** | 2st / 6946tok / 18.64s | **PASS** | 2c / 33141tok / 63.03s |
| GSM8K | `GSM8K/12` | `Carlos is planting a lemon tree. The tree will c...` | FAIL | 3st / 10934tok / 89.14s | FAIL | 1c / 3289tok / 90.01s |
| GSM8K | `GSM8K/13` | `Melanie is a door-to-door saleswoman. She sold a...` | **PASS** | 4st / 16993tok / 84.36s | FAIL | 1c / 3284tok / 90.1s |
| GSM8K | `GSM8K/14` | `In a dance class of 20 students, 20% enrolled in...` | **PASS** | 2st / 7088tok / 23.32s | **PASS** | 2c / 33176tok / 57.11s |
| GSM8K | `GSM8K/15` | `A merchant wants to make a choice of purchase be...` | **PASS** | 1st / 3560tok / 21.02s | **PASS** | 2c / 33228tok / 53.51s |
| GSM8K | `GSM8K/16` | `Two trains leave San Rafael at the same time. Th...` | **PASS** | 1st / 3385tok / 10.63s | FAIL | 1c / 3280tok / 90.07s |
| GSM8K | `GSM8K/17` | `Jill gets paid $20 per hour to teach and $30 to ...` | **PASS** | 2st / 7609tok / 39.1s | **PASS** | 2c / 33110tok / 54.21s |
| MemGraph | `MemGraph/1` | `Contractor Overtime Payroll (hours=52)` | **PASS** | 1st / 3427tok / 9.15s | FAIL | 1c / 3299tok / 110.02s |
| MemGraph | `MemGraph/2` | `Supply Chain Landed Tariff Cost (surcharge=15%)` | **PASS** | 1st / 3415tok / 9.26s | FAIL | 1c / 3303tok / 110.11s |
| MemGraph | `MemGraph/3` | `Microservice SLA Latency Breach Score (budget=65ms)` | **PASS** | 1st / 3426tok / 9.39s | FAIL | 1c / 3300tok / 110.01s |
| MemGraph | `MemGraph/4` | `Smart-Grid Time-of-Use Bill Cents (days=10)` | **PASS** | 1st / 3425tok / 9.25s | FAIL | 1c / 3301tok / 110.02s |
| MemGraph | `MemGraph/5` | `EV Fleet Thermal Charging Energy Wh (lossPct=12)` | **PASS** | 1st / 3436tok / 10.68s | FAIL | 1c / 3298tok / 110.11s |
