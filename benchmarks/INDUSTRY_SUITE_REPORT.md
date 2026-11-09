> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **16/22** | **72.7%** | 630,569 | 28,662 | 87.08s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 12 | 140,596 | 231.63s | test_immutable: FAIL ( 71 files changed, 6 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 50,163 | 86.99s | test_Derivative_kind: FAIL ( 73 files changed, 6 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 4 | 35,321 | 102.42s | test_issue_22684: FAIL ( 76 files changed, 7 insertions(+)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,873 | 42.73s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,457 | 37.99s | check(strlen): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/45` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,395 | 38.74s | check(triangle_area): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,312 | 120.11s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,497 | 46.70s | sort_matrix (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/14` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 16,046 | 38.31s | find_Volume (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 10,506 | 23.33s | expected=18, found_in_output=False (file='26') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 10,335 | 21.08s | expected=3, found_in_output=True (file='3') |
| Track 4: OpenAI GSM8K | `GSM8K/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 10,549 | 29.62s | expected=70000, found_in_output=True (file='120000') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 32,245 | 120.38s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.02s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 33,050 | 85.82s | pytest test_find_in_sorted: PASS (7 passed in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/levenshtein` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 36,575 | 120.70s | pytest test_levenshtein: PASS (6 passed, 1 skipped in 0.47s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 38,914 | 93.15s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,772 | 46.89s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 18,133 | 78.75s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 16,404 | 120.62s | output: FAIL (ans="[('a', 12)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 34,557 | 150.69s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 5,069 | 150.29s | unittest (task_func): FAIL (FAILED (failures=3, errors=1)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 28,800 | 128.78s | unittest (task_func): PASS (OK) |
