> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **2/5** | **40.0%** | 649,527 | 129,905 | 595.64s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/llm-inference-batching-scheduler` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 11 | 183,610 | 1200.64s | pytest test_outputs.py: FAIL (1 error in 0.10s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/feal-differential-cryptanalysis` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 119,572 | 1047.53s | pytest test_outputs.py: FAIL (1 failed in 0.31s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/fix-code-vulnerability` | `actorii` | `actorii-sandbox:latest` | **PASS** | 17 | 238,011 | 352.63s | pytest test_outputs.py: PASS (6 passed in 0.03s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/db-wal-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 38,139 | 176.31s | pytest test_outputs.py: PASS (7 passed in 0.01s) |
| Track 1B: Terminal-Bench 2.1 Hard Tier (Stanford/Laude) | `TB2.1-Hard/constraints-scheduling` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 70,195 | 201.08s | pytest test_outputs.py: FAIL (1 failed, 2 passed in 0.15s) |
