> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **19/22** | **86.4%** | 705,456 | 32,066 | 84.40s |
| **`pi`** | `pi-bench:latest` | **10/22** | **45.5%** | 368,213 | 16,736 | 118.10s |
| **`hermes`** | `hermes-bench:latest` | **10/22** | **45.5%** | 289,817 | 13,173 | 117.16s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 41,393 | 88.74s | test_immutable: PASS ( 2 files changed, 7 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 86,942 | 179.88s | test_Derivative_kind: PASS ( 2 files changed, 10 insertions(+), 1 deletion(-)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 24,883 | 59.51s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 11,524 | 30.93s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,978 | 69.85s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 23,864 | 66.03s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 1 | 5,730 | 28.93s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `actorii` | `actorii-sandbox:latest` | **PASS** | 1 | 5,482 | 10.60s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 42,251 | 81.93s | pytest test_knapsack: PASS (9 passed, 1 skipped, 1 warning in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 29,342 | 90.74s | pytest test_find_in_sorted: PASS (7 passed, 1 warning in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 28,667 | 70.25s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 26,182 | 45.31s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 12,767 | 47.75s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `actorii` | `actorii-sandbox:latest` | **PASS** | 2 | 11,918 | 28.41s | output: PASS (ans="[('74', 31)]") |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/15` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 22,504 | 127.85s | unittest (task_func): FAIL (FAILED (failures=1)) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 20,694 | 93.74s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 19,926 | 74.59s | unittest (task_func): PASS (OK) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 20,137 | 78.10s | unittest (task_func): PASS (OK) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 8 | 76,015 | 180.77s | test_issue_22684: FAIL ( 1 file changed, 7 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 49,851 | 115.68s | test_issue_14941: PASS ( 2 files changed, 4 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 53,814 | 105.74s | test_zero: PASS ( 2 files changed, 5 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 7 | 67,592 | 181.46s | test_solve_poly_system: FAIL ( 2 files changed, 5 insertions(+), 1 deletion(-)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `pi` | `pi-bench:latest` | **FAIL** | 4 | 58,487 | 151.22s | test_immutable: FAIL ( 1758 files changed, 6 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `pi` | `pi-bench:latest` | **FAIL** | 3 | 37,935 | 151.26s | test_Derivative_kind: FAIL ( 1818 files changed, 6 insertions(+)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `pi` | `pi-bench:latest` | **PASS** | 4 | 8,976 | 120.13s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `pi` | `pi-bench:latest` | **PASS** | 5 | 10,452 | 57.64s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `pi` | `pi-bench:latest` | **PASS** | 4 | 9,716 | 90.34s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `pi` | `pi-bench:latest` | **PASS** | 3 | 6,306 | 51.28s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `pi` | `pi-bench:latest` | **PASS** | 2 | 3,796 | 22.73s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `pi` | `pi-bench:latest` | **PASS** | 2 | 3,672 | 19.74s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `pi` | `pi-bench:latest` | **PASS** | 5 | 17,399 | 111.43s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `pi` | `pi-bench:latest` | **FAIL** | 3 | 7,284 | 120.29s | pytest test_find_in_sorted: FAIL (2 failed, 5 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `pi` | `pi-bench:latest` | **PASS** | 6 | 16,708 | 101.52s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `pi` | `pi-bench:latest` | **PASS** | 4 | 12,561 | 58.85s | expected='99', file='99' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `pi` | `pi-bench:latest` | **PASS** | 3 | 7,812 | 94.91s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 2,610 | 120.22s | output: FAIL (ans='') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/15` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 0 | 150.21s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 0 | 150.23s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 0 | 150.22s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 0 | 150.23s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 2,318 | 181.27s | test_issue_22684: FAIL ( 1885 files changed, 7 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `pi` | `pi-bench:latest` | **FAIL** | 8 | 131,272 | 181.77s | test_issue_14941: FAIL ( 1891 files changed, 2 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `pi` | `pi-bench:latest` | **FAIL** | 6 | 28,626 | 181.28s | test_zero: FAIL ( 1752 files changed, 3 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `pi` | `pi-bench:latest` | **FAIL** | 1 | 2,283 | 181.44s | test_solve_poly_system: FAIL ( 1847 files changed, 5 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `hermes` | `hermes-bench:latest` | **PASS** | 1 | 0 | 151.33s | test_immutable: PASS ( 2 files changed, 12 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-21614` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 151.34s | test_Derivative_kind: FAIL ( 1 file changed, 6 insertions(+)) |
| Track 2: OpenAI HumanEval | `HumanEval/2` | `hermes` | `hermes-bench:latest` | **PASS** | 4 | 40,831 | 83.76s | check(truncate_number): PASS |
| Track 2: OpenAI HumanEval | `HumanEval/23` | `hermes` | `hermes-bench:latest` | **PASS** | 4 | 40,076 | 37.51s | check(strlen): PASS |
| Track 3: Google Research MBPP | `MBPP/11` | `hermes` | `hermes-bench:latest` | **PASS** | 3 | 30,602 | 110.92s | remove_Occ (3 asserts): PASS |
| Track 3: Google Research MBPP | `MBPP/12` | `hermes` | `hermes-bench:latest` | **PASS** | 3 | 30,296 | 64.03s | sort_matrix (3 asserts): PASS |
| Track 4: OpenAI GSM8K | `GSM8K/0` | `hermes` | `hermes-bench:latest` | **PASS** | 2 | 19,779 | 27.55s | expected=18, found_in_output=True (file='18') |
| Track 4: OpenAI GSM8K | `GSM8K/1` | `hermes` | `hermes-bench:latest` | **PASS** | 2 | 19,695 | 26.77s | expected=3, found_in_output=True (file='3') |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `hermes` | `hermes-bench:latest` | **PASS** | 4 | 45,871 | 88.71s | pytest test_knapsack: PASS (9 passed, 1 skipped, 1 warning in 0.01s) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/find_in_sorted` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 120.46s | pytest test_find_in_sorted: FAIL (2 failed, 5 passed in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `hermes` | `hermes-bench:latest` | **PASS** | 4 | 42,282 | 72.76s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/25` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 120.30s | expected='99', file='' (FAIL) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `hermes` | `hermes-bench:latest` | **PASS** | 2 | 20,385 | 74.25s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 7: CruxEval (Meta Research) | `CruxEval-O/6` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 120.35s | output: FAIL (ans='') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/15` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 150.65s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 150.29s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/777` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 150.35s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/854` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 150.37s | unittest (task_func): FAIL (IndentationError: expected an indented block after function ) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22714` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 181.32s | test_issue_22684: FAIL ( 1 file changed, 7 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-23262` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 181.80s | test_issue_14941: FAIL ( 1 file changed, 2 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20212` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 181.35s | test_zero: FAIL ( 1 file changed, 3 insertions(+)) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-22005` | `hermes` | `hermes-bench:latest` | **FAIL** | 1 | 0 | 181.39s | test_solve_poly_system: FAIL ( 2 files changed, 5 insertions(+), 1 deletion(-)) |
