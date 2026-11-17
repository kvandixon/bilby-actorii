> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **15/22** | **68.2%** | 1,863,644 | 84,711 | 224.22s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 13 | 109,932 | 281.35s | test_immutable: PASS ( 1 file changed, 2 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 28 | 249,613 | 264.04s | test_Derivative_kind: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 9 | 105,276 | 333.84s | test_issue_22684: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 28 | 386,760 | 786.24s | test_issue_14941: FAIL (no files modified) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **PASS** | 24 | 361,526 | 767.27s | test_zero: PASS ( 1 file changed, 1 insertion(+), 1 deletion(-)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 17 | 291,717 | 902.57s | test_solve_poly_system: FAIL ( 2 files changed, 5 insertions(+), 1 deletion(-)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 34,189 | 58.26s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,361 | 32.46s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 4,963 | 120.36s | remove_Occ (3 asserts): NameError: name 'remove_Occ' is not defined |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 20,085 | 83.04s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,957 | 58.34s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 13,597 | 42.02s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 22,591 | 113.79s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 25,662 | 107.25s | pytest test_find_in_sorted: PASS (7 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 15,060 | 96.19s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 21,172 | 90.28s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 24,889 | 100.06s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 14,961 | 120.18s | output: PASS (ans="[('74', 31)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 26,832 | 150.33s | unittest (task_func): FAIL (SyntaxError: unexpected character after line continuation ch) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 27,220 | 150.32s | unittest (task_func): FAIL (FAILED (failures=4)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 33,971 | 150.25s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/800` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 41,310 | 124.30s | unittest (task_func): PASS (OK) |
