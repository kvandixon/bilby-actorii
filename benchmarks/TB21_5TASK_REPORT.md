> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **4/5** | **80.0%** | 139,464 | 27,892 | 140.18s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 38,535 | 123.58s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 34,986 | 132.86s | pytest test_outputs.py: PASS (2 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 18,899 | 114.34s | pytest test_outputs.py: PASS (6 passed in 13.63s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 30,198 | 89.06s | pytest test_outputs.py: PASS (2 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 16,846 | 241.08s | pytest test_outputs.py: FAIL (3 failed in 0.49s) |
