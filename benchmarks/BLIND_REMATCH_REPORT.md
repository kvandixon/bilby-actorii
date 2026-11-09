> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **5/5** | **100.0%** | 126,680 | 25,336 | 71.66s |
| **`pi`** | `pi-bench:latest` | **4/5** | **80.0%** | 103,732 | 20,746 | 92.18s |
| **`hermes`** | `hermes-bench:latest` | **3/5** | **60.0%** | 409,014 | 81,802 | 114.78s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `actorii` | `actorii-sandbox:latest` | **PASS** | 7 | 57,458 | 118.80s | test_immutable: PASS ( 2 files changed, 7 insertions(+)) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 18,852 | 45.82s | pytest test_knapsack: PASS (9 passed, 1 skipped, 1 warning in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 26,902 | 83.50s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 10,872 | 47.51s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 12,596 | 62.67s | unittest (task_func): PASS (OK) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `pi` | `pi-bench:latest` | **PASS** | 11 | 29,521 | 126.15s | test_immutable: PASS ( 1758 files changed, 9 insertions(+)) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `pi` | `pi-bench:latest` | **PASS** | 6 | 17,882 | 58.06s | pytest test_knapsack: PASS (9 passed, 1 skipped in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `pi` | `pi-bench:latest` | **PASS** | 7 | 22,912 | 66.29s | expected='.plain.log', file='.plain.log' (PASS) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `pi` | `pi-bench:latest` | **FAIL** | 9 | 21,255 | 103.61s | input: FAIL (ans='(2, 2, 4, 4, 4, 4)') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `pi` | `pi-bench:latest` | **PASS** | 5 | 12,162 | 106.81s | unittest (task_func): PASS (OK) |
| Track 1: SWE-bench Lite (Princeton) | `sympy__sympy-20590` | `hermes` | `hermes-bench:latest` | **FAIL** | 10 | 147,912 | 182.01s | test_immutable: FAIL ( 1 file changed, 6 insertions(+)) |
| Track 5: QuixBugs (Fault Localization) | `QuixBugs/knapsack` | `hermes` | `hermes-bench:latest` | **PASS** | 10 | 108,991 | 97.06s | pytest test_knapsack: PASS (9 passed, 1 skipped, 1 warning in 0.01s) |
| Track 6: AgentBench OS (THUDM) | `AgentBench-OS/2` | `hermes` | `hermes-bench:latest` | **FAIL** | 7 | 68,553 | 120.87s | expected='.plain.log', file='' (FAIL) |
| Track 7: CruxEval (Meta Research) | `CruxEval-I/0` | `hermes` | `hermes-bench:latest` | **PASS** | 5 | 41,239 | 63.90s | input: PASS (ans='[1, 1, 1, 1, 3, 3]') |
| Track 8: BigCodeBench-Hard (ICLR 2025) | `BigCodeBench/19` | `hermes` | `hermes-bench:latest` | **PASS** | 5 | 42,319 | 110.06s | unittest (task_func): PASS (OK) |
