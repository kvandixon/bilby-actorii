> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **19/22** | **86.4%** | 2,167,925 | 98,542 | 42.38s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 24 | 260,397 | 92.21s | test_immutable: PASS ( 1 file changed, 1 insertion(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **PASS** | 22 | 237,258 | 73.52s | test_Derivative_kind: PASS ( 1 file changed, 4 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **PASS** | 11 | 101,283 | 54.74s | test_issue_22684: PASS ( 1 file changed, 1 insertion(+), 1 deletion(-)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **PASS** | 27 | 390,191 | 133.61s | test_issue_14941: PASS ( 1 file changed, 4 insertions(+), 1 deletion(-)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 21 | 324,758 | 85.17s | test_zero: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **PASS** | 21 | 425,546 | 159.60s | test_solve_poly_system: PASS ( 2 files changed, 12 insertions(+), 1 deletion(-)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,856 | 10.40s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,222 | 7.25s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 19,447 | 18.00s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 14,092 | 13.83s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 9,426 | 10.35s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,719 | 10.79s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 28,314 | 20.75s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 36,497 | 21.00s | pytest test_find_in_sorted: PASS (7 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 43,635 | 35.40s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 33,199 | 23.32s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 19,633 | 13.96s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 36,686 | 30.86s | output: PASS (ans="[('74', 31)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 33,377 | 30.85s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 38,020 | 23.64s | unittest (task_func): FAIL (FAILED (failures=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 28,349 | 27.15s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/800` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 4 | 37,020 | 36.03s | unittest (task_func): FAIL (FAILED (failures=1)) |
