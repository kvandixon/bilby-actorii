> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `harborframework/terminal-bench-2.1` (Stanford/Laude), `SWE-bench-Live/SWE-bench-Live` (2025-06 post-cutoff split), `Aider-AI/polyglot-benchmark`, and `drproduck/livecodebench-v6` (2025-04 post-cutoff contest window).

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **5/9** | **55.6%** | 408,296 | 45,366 | 168.94s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 41,811 | 134.99s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 35,073 | 146.58s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 17,889 | 87.84s | pytest test_outputs.py: PASS (6 passed in 13.64s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 8 | 95,215 | 241.40s | test_get_encoding__encoding_value_is_none: FAIL ( 1 file changed, 3 insertions(+)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 5 | 38,240 | 105.47s | test_random_tree: FAIL ( 2 files changed, 10 insertions(+)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 5 | 52,580 | 186.79s | pytest forth_test.py: FAIL (4 failed, 50 passed in 0.03s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 51,379 | 164.95s | pytest grep_test.py: PASS (25 passed in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 61,547 | 212.20s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 14,562 | 240.20s | solution.py not created |
