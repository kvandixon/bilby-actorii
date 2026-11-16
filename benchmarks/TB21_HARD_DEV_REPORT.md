> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **0/5** | **0.0%** | 471,604 | 94,320 | 290.92s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 8 | 99,681 | 600.65s | pytest test_outputs.py: FAIL (1 error in 0.05s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 7 | 96,725 | 387.74s | pytest test_outputs.py: FAIL (1 failed in 0.33s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/fix-code-vulnerability` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 16 | 147,608 | 136.53s | pytest test_outputs.py: FAIL (3 failed, 3 passed in 0.04s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/db-wal-recovery` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 11 | 91,736 | 174.81s | pytest test_outputs.py: FAIL (7 failed in 0.02s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 5 | 35,854 | 154.86s | pytest test_outputs.py: FAIL (1 failed, 2 passed in 0.14s) |
