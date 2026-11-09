> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **4/5** | **80.0%** | 131,282 | 26,256 | 122.18s |
| **`pi`** | `pi-bench:latest` | **4/5** | **80.0%** | 165,348 | 33,069 | 182.27s |
| **`hermes`** | `hermes-bench:latest` | **3/5** | **60.0%** | 401,998 | 80,399 | 203.09s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 51,047 | 137.37s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `pi` | `pi-bench:latest` | **PASS** | 17 | 69,721 | 190.19s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `hermes` | `hermes-bench:latest` | **PASS** | 12 | 133,781 | 240.52s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.03s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 23,542 | 95.42s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `pi` | `pi-bench:latest` | **PASS** | 5 | 19,395 | 112.72s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `hermes` | `hermes-bench:latest` | **PASS** | 8 | 92,082 | 205.97s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 12,810 | 84.70s | pytest test_outputs.py: PASS (6 passed in 13.64s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `pi` | `pi-bench:latest` | **FAIL** | 3 | 13,540 | 271.91s | pytest test_outputs.py: FAIL (5 failed, 1 passed in 31.67s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `hermes` | `hermes-bench:latest` | **FAIL** | 6 | 56,751 | 253.13s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.66s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 15,899 | 52.87s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `pi` | `pi-bench:latest` | **PASS** | 10 | 39,428 | 122.83s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/fix-git` | `hermes` | `hermes-bench:latest` | **PASS** | 8 | 72,937 | 75.02s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 27,984 | 240.55s | pytest test_outputs.py: PASS (3 passed in 0.24s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `pi` | `pi-bench:latest` | **PASS** | 6 | 23,264 | 213.71s | pytest test_outputs.py: PASS (3 passed in 0.22s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/multi-source-data-merger` | `hermes` | `hermes-bench:latest` | **FAIL** | 5 | 46,447 | 240.81s | pytest test_outputs.py: FAIL (3 failed in 0.29s) |
