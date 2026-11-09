> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `nemotron-cascade-2:30b` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `harborframework/terminal-bench-2.1` (Stanford/Laude), `SWE-bench-Live/SWE-bench-Live` (2025-06 post-cutoff split), `Aider-AI/polyglot-benchmark`, and `drproduck/livecodebench-v6` (2025-04 post-cutoff contest window).

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **3/9** | **33.3%** | 623,390 | 69,265 | 111.69s |
| **`pi`** | `pi-bench:latest` | **3/9** | **33.3%** | 8,698,152 | 966,461 | 133.10s |
| **`hermes`** | `hermes-bench:latest` | **3/9** | **33.3%** | 5,396,876 | 599,652 | 101.55s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 10 | 61,378 | 144.33s | pytest test_outputs.py: FAIL (1 failed, 4 passed, 1 warning in 0.03s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `pi` | `pi-bench:latest` | **PASS** | 29 | 94,900 | 21.67s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `hermes` | `hermes-bench:latest` | **PASS** | 21 | 215,432 | 33.40s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 11,584 | 26.62s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `pi` | `pi-bench:latest` | **FAIL** | 70 | 901,820 | 89.48s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `hermes` | `hermes-bench:latest` | **FAIL** | 9 | 151,973 | 20.93s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 10 | 69,042 | 132.53s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.64s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `pi` | `pi-bench:latest` | **FAIL** | 52 | 329,445 | 252.89s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.64s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `hermes` | `hermes-bench:latest` | **FAIL** | 3 | 19,585 | 19.48s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.64s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **PASS** | 7 | 75,739 | 81.47s | test_get_encoding__encoding_value_is_none: PASS ( 1 file changed, 2 insertions(+), 2 deletions(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `pi` | `pi-bench:latest` | **PASS** | 112 | 2,682,346 | 240.66s | test_get_encoding__encoding_value_is_none: PASS ( 284 files changed, 2 insertions(+), 2 deletions(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `hermes` | `hermes-bench:latest` | **FAIL** | 97 | 3,436,508 | 240.70s | test_get_encoding__encoding_value_is_none: FAIL ( 54 files changed, 36176 deletions(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 12 | 91,155 | 142.57s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+), 2 deletions(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `pi` | `pi-bench:latest` | **FAIL** | 25 | 151,059 | 33.63s | test_random_tree: FAIL ( 950 files changed, 10 insertions(+)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `hermes` | `hermes-bench:latest` | **FAIL** | 24 | 381,085 | 37.84s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+), 4 deletions(-)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 10 | 126,722 | 160.70s | pytest forth_test.py: FAIL (54 failed, 1 warning in 0.14s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `pi` | `pi-bench:latest` | **FAIL** | 48 | 1,135,691 | 240.40s | pytest forth_test.py: FAIL (17 failed, 37 passed in 0.12s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `hermes` | `hermes-bench:latest` | **FAIL** | 29 | 803,403 | 240.45s | pytest forth_test.py: FAIL (14 failed, 40 passed, 2 warnings in 0.08s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 10 | 110,155 | 174.74s | pytest grep_test.py: FAIL (2 warnings, 1 error in 0.06s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `pi` | `pi-bench:latest` | **FAIL** | 93 | 3,091,662 | 240.35s | pytest grep_test.py: FAIL (1 error in 0.04s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `hermes` | `hermes-bench:latest` | **PASS** | 14 | 258,366 | 61.44s | pytest grep_test.py: PASS (25 passed, 1 warning in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 51,662 | 96.50s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `pi` | `pi-bench:latest` | **PASS** | 10 | 31,716 | 25.23s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `hermes` | `hermes-bench:latest` | **PASS** | 11 | 110,422 | 19.55s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 25,953 | 45.71s | tests: 3/3 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `pi` | `pi-bench:latest` | **FAIL** | 60 | 279,513 | 53.62s | solution.py not created |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `hermes` | `hermes-bench:latest` | **FAIL** | 2 | 20,102 | 240.20s | solution.py not created |
