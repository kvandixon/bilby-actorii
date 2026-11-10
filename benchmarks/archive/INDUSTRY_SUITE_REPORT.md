> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **17/22** | **77.3%** | 1,123,948 | 51,088 | 98.27s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 132,253 | 248.87s | test_immutable: PASS ( 1 file changed, 1 insertion(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **PASS** | 11 | 146,719 | 336.06s | test_Derivative_kind: PASS ( 1 file changed, 4 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 9 | 95,094 | 187.42s | test_issue_22684: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 16 | 140,561 | 249.63s | test_issue_14941: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 185,744 | 300.62s | test_zero: PASS ( 1 file changed, 2 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 7 | 74,375 | 275.71s | test_solve_poly_system: FAIL ( 2 files changed, 5 insertions(+), 1 deletion(-)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,509 | 18.36s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,332 | 16.11s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,332 | 22.32s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,316 | 15.99s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,267 | 14.87s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 10,010 | 10.99s | expected=3, found_in_output=True (file='3.0') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 42,928 | 51.93s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 35,197 | 45.94s | pytest test_find_in_sorted: PASS (7 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 29,737 | 45.74s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 21,323 | 18.38s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,027 | 41.20s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 15,537 | 22.79s | output: PASS (ans="[('74', 31)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 10,898 | 23.40s | unittest (task_func): FAIL (FAILED (failures=1, errors=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 43,707 | 106.80s | unittest (task_func): FAIL (FAILED (failures=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,428 | 55.16s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/800` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,654 | 53.64s | unittest (task_func): PASS (OK) |
