> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **12/15** | **80.0%** | 1,348,310 | 89,887 | 67.41s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 16 | 193,570 | 106.69s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 17,040 | 22.15s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 59,741 | 74.46s | pytest test_outputs.py: PASS (6 passed in 13.64s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 29,393 | 28.31s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `actorii` | `actorii-sandbox:latest` | **PASS** | 16 | 195,869 | 94.40s | pytest test_outputs.py: PASS (3 passed in 0.45s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 9 | 73,617 | 41.60s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 48,705 | 46.99s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 14 | 231,186 | 203.51s | pytest test_outputs.py: PASS (6 passed in 13.63s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 35,043 | 31.48s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 70,551 | 67.79s | pytest test_outputs.py: PASS (3 passed in 0.37s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 16 | 157,784 | 77.55s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 16,819 | 20.46s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 9 | 97,729 | 97.31s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.64s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 75,516 | 47.42s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 45,747 | 50.99s | pytest test_outputs.py: PASS (3 passed in 0.22s) |
