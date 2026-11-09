> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `harborframework/terminal-bench-2.1` (Stanford/Laude), `SWE-bench-Live/SWE-bench-Live` (2025-06 post-cutoff split), `Aider-AI/polyglot-benchmark`, and `drproduck/livecodebench-v6` (2025-04 post-cutoff contest window).

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **7/9** | **77.8%** | 361,057 | 40,117 | 159.49s |
| **`pi`** | `pi-bench:latest` | **5/9** | **55.6%** | 444,172 | 49,352 | 164.03s |
| **`hermes`** | `hermes-bench:latest` | **5/9** | **55.6%** | 864,198 | 96,022 | 159.04s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 38,029 | 118.08s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `pi` | `pi-bench:latest` | **PASS** | 8 | 22,409 | 87.95s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `hermes` | `hermes-bench:latest` | **PASS** | 10 | 104,458 | 161.60s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **PASS** | 3 | 25,786 | 102.90s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `pi` | `pi-bench:latest` | **PASS** | 4 | 13,226 | 116.19s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `hermes` | `hermes-bench:latest` | **PASS** | 7 | 77,144 | 147.71s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 30,412 | 193.93s | pytest test_outputs.py: PASS (6 passed in 13.65s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `pi` | `pi-bench:latest` | **PASS** | 1 | 2,549 | 193.97s | pytest test_outputs.py: PASS (6 passed in 13.67s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `hermes` | `hermes-bench:latest` | **FAIL** | 3 | 21,230 | 193.07s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.66s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 111,459 | 240.58s | test_get_encoding__encoding_value_is_none: PASS ( 1 file changed, 2 insertions(+)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `pi` | `pi-bench:latest` | **FAIL** | 22 | 143,777 | 240.62s | test_get_encoding__encoding_value_is_none: FAIL ( 284 files changed, 0 insertions(+), 0 deletions(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `hermes` | `hermes-bench:latest` | **FAIL** | 17 | 371,123 | 240.86s | test_get_encoding__encoding_value_is_none: FAIL (no files modified) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 5 | 26,923 | 74.06s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `pi` | `pi-bench:latest` | **FAIL** | 9 | 26,601 | 116.33s | test_random_tree: FAIL ( 950 files changed, 12 insertions(+)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `hermes` | `hermes-bench:latest` | **FAIL** | 6 | 59,029 | 91.55s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 5 | 36,063 | 240.58s | pytest forth_test.py: FAIL (7 failed, 47 passed, 2 warnings in 0.09s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `pi` | `pi-bench:latest` | **FAIL** | 18 | 114,560 | 180.50s | pytest forth_test.py: FAIL (54 failed in 0.13s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `hermes` | `hermes-bench:latest` | **FAIL** | 3 | 25,724 | 180.41s | pytest forth_test.py: FAIL (2 failed, 52 passed in 0.03s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 35,903 | 137.22s | pytest grep_test.py: PASS (25 passed, 1 warning in 0.02s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `pi` | `pi-bench:latest` | **PASS** | 8 | 91,008 | 180.40s | pytest grep_test.py: PASS (25 passed in 0.03s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `hermes` | `hermes-bench:latest` | **PASS** | 5 | 57,605 | 95.84s | pytest grep_test.py: PASS (25 passed, 1 warning in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 28,349 | 106.17s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `pi` | `pi-bench:latest` | **PASS** | 6 | 14,949 | 180.20s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `hermes` | `hermes-bench:latest` | **PASS** | 7 | 65,770 | 139.97s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **PASS** | 5 | 28,133 | 221.88s | tests: 3/3 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `pi` | `pi-bench:latest` | **FAIL** | 5 | 15,093 | 180.14s | tests: 0/3 (FAIL) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `hermes` | `hermes-bench:latest` | **PASS** | 7 | 82,115 | 180.32s | tests: 3/3 (PASS) |
