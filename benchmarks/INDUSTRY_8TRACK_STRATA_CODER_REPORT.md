> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **12/22** | **54.5%** | 706,214 | 32,100 | 12.00s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,414 | 5.44s | test_immutable: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,310 | 2.88s | test_Derivative_kind: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,672 | 2.77s | test_issue_22684: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 22 | 326,770 | 67.21s | test_issue_14941: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,329 | 2.79s | test_zero: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,696 | 3.03s | test_solve_poly_system: FAIL (no files modified) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,478 | 6.73s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,400 | 5.72s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,354 | 3.47s | remove_Occ (3 asserts): NameError: name 'remove_Occ' is not defined |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 18,666 | 9.83s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 1 | 4,213 | 1.49s | expected=18, found_in_output=True (file='') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 8,929 | 6.49s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 34,096 | 12.18s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 36,821 | 18.49s | pytest test_find_in_sorted: PASS (7 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,215 | 2.01s | expected='.plain.log', file='' (FAIL) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,206 | 1.41s | expected='99', file='' (FAIL) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,715 | 10.24s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,374 | 8.32s | output: PASS (ans="[('74', 31)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 37,336 | 18.83s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 37,561 | 19.35s | unittest (task_func): FAIL (FAILED (failures=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 33,494 | 23.50s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/800` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 54,165 | 31.78s | unittest (task_func): PASS (OK) |
