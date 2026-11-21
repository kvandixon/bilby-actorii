> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **1/3** | **33.3%** | 698,068 | 232,689 | 941.87s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 166,541 | 723.64s | pytest test_outputs.py: PASS (3 passed in 0.28s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 11 | 197,017 | 901.36s | pytest test_outputs.py: FAIL (1 failed in 0.24s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 16 | 334,510 | 1200.62s | pytest test_outputs.py: FAIL (1 error in 0.12s) |
