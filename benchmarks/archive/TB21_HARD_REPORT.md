> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **0/5** | **0.0%** | 271,896 | 54,379 | 226.52s |
| **`pi`** | `pi-bench:latest` | **0/5** | **0.0%** | 288,605 | 57,721 | 221.09s |
| **`hermes`** | `hermes-bench:latest` | **0/5** | **0.0%** | 455,754 | 91,150 | 240.58s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 63,639 | 240.54s | pytest test_outputs.py: FAIL (1 error in 0.09s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `pi` | `pi-bench:latest` | **FAIL** | 6 | 40,749 | 240.42s | pytest test_outputs.py: FAIL (1 error in 0.07s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `hermes` | `hermes-bench:latest` | **FAIL** | 6 | 81,200 | 240.56s | pytest test_outputs.py: FAIL (1 error in 0.08s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 3,338 | 189.14s | pytest test_outputs.py: FAIL (1 failed in 0.42s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `pi` | `pi-bench:latest` | **FAIL** | 8 | 27,097 | 240.49s | pytest test_outputs.py: FAIL (1 failed in 0.15s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `hermes` | `hermes-bench:latest` | **FAIL** | 6 | 59,478 | 240.65s | pytest test_outputs.py: FAIL (1 failed in 0.17s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/fix-code-vulnerability` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 9 | 96,346 | 243.04s | pytest test_outputs.py: FAIL (4 failed, 2 passed in 0.06s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/fix-code-vulnerability` | `pi` | `pi-bench:latest` | **FAIL** | 23 | 171,935 | 240.44s | pytest test_outputs.py: FAIL (4 failed, 2 passed in 0.06s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/fix-code-vulnerability` | `hermes` | `hermes-bench:latest` | **FAIL** | 14 | 214,548 | 240.56s | pytest test_outputs.py: FAIL (4 failed, 2 passed in 0.06s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/db-wal-recovery` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 9 | 86,427 | 240.27s | pytest test_outputs.py: FAIL (7 failed in 0.02s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/db-wal-recovery` | `pi` | `pi-bench:latest` | **FAIL** | 9 | 31,483 | 240.35s | pytest test_outputs.py: FAIL (7 failed in 0.03s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/db-wal-recovery` | `hermes` | `hermes-bench:latest` | **FAIL** | 7 | 75,830 | 240.55s | pytest test_outputs.py: FAIL (7 failed in 0.03s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 4 | 22,146 | 219.59s | pytest test_outputs.py: FAIL (3 failed in 0.06s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `pi` | `pi-bench:latest` | **FAIL** | 4 | 17,341 | 143.74s | pytest test_outputs.py: FAIL (2 failed, 1 passed in 0.15s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `hermes` | `hermes-bench:latest` | **FAIL** | 3 | 24,698 | 240.57s | pytest test_outputs.py: FAIL (3 failed in 0.07s) |
