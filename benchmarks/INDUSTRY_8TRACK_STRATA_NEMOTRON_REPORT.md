> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **13/22** | **59.1%** | 1,602,481 | 72,840 | 471.05s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 21 | 228,708 | 1544.76s | test_immutable: PASS ( 1 file changed, 3 insertions(+), 2 deletions(-)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,468 | 56.19s | test_Derivative_kind: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 15 | 104,715 | 994.50s | test_issue_22684: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 28 | 543,817 | 1604.30s | test_issue_14941: FAIL ( 1 file changed, 6 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **PASS** | 12 | 136,199 | 842.73s | test_zero: PASS ( 1 file changed, 2 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 18 | 179,646 | 981.87s | test_solve_poly_system: FAIL ( 1 file changed, 1 insertion(+), 1 deletion(-)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 24,090 | 200.18s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,959 | 183.67s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,635 | 138.58s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,714 | 131.59s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 1 | 4,317 | 26.49s | expected=18, found_in_output=True (file='') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 1 | 4,275 | 25.14s | expected=3, found_in_output=True (file='') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 35,093 | 272.08s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 4 | 27,503 | 362.96s | pytest test_find_in_sorted: FAIL (2 failed, 5 passed in 0.02s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 45,770 | 269.68s | expected='.plain.log', file='' (FAIL) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 55,261 | 287.08s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 29,370 | 295.68s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 24,691 | 363.88s | output: FAIL (ans='') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 31,906 | 453.14s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 27,133 | 454.26s | unittest (task_func): FAIL (FAILED (failures=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 33,183 | 421.08s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/800` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 16,028 | 453.29s | unittest (task_func): FAIL (FAILED (failures=5)) |
