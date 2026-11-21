> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Containerized Official Multi-Harness Benchmark Report

- **Model Backend**: `qwen3.8:latest` (local `:8080` GPU inference on `valkyrie`)
- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.
- **Datasets**: Official `harborframework/terminal-bench-2.1` (Stanford/Laude), `SWE-bench-Live/SWE-bench-Live` (2025-06 post-cutoff split), `Aider-AI/polyglot-benchmark`, and `drproduck/livecodebench-v6` (2025-04 post-cutoff contest window).

## 1. Overall Containerized Leaderboard

| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`actorii`** | `actorii-sandbox:latest` | **17/27** | **63.0%** | 2,227,741 | 82,508 | 47.48s |

## 2. Per-Task Breakdown Across Official Upstream Benchmarks

| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 137,734 | 75.39s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 2 | 17,064 | 22.65s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 66,844 | 75.53s | pytest test_outputs.py: FAIL (1 failed, 5 passed in 12.64s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 16 | 143,905 | 54.67s | test_get_encoding__encoding_value_is_none: FAIL ( 1 file changed, 10 insertions(+), 1 deletion(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 6 | 39,802 | 20.77s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 16 | 224,628 | 163.99s | Exception: Command '['python3', '-m', 'pytest', '-q', '/tmp/bench_poly_forth_actorii_8uj2wx8c/forth_test.py']' timed out after 30 seconds |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 45,073 | 13.94s | pytest grep_test.py: PASS (25 passed in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 8 | 67,408 | 44.72s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 26,914 | 17.74s | tests: 3/3 (PASS) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 11 | 101,993 | 69.10s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.03s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 3 | 22,205 | 22.25s | pytest test_outputs.py: FAIL (1 failed, 1 passed in 0.01s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 10,155 | 13.75s | pytest test_outputs.py: FAIL (6 failed in 2.00s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **PASS** | 12 | 116,041 | 53.88s | test_get_encoding__encoding_value_is_none: PASS ( 1 file changed, 1 insertion(+), 1 deletion(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 4 | 28,240 | 23.46s | test_random_tree: FAIL ( 1 file changed, 9 insertions(+)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **PASS** | 10 | 124,353 | 54.73s | pytest forth_test.py: PASS (54 passed in 0.02s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **PASS** | 7 | 63,580 | 24.18s | pytest grep_test.py: PASS (25 passed in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 11 | 100,328 | 57.12s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **PASS** | 12 | 90,174 | 44.56s | tests: 3/3 (PASS) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/git-leak-recovery` | `actorii` | `actorii-sandbox:latest` | **PASS** | 15 | 233,150 | 104.17s | pytest test_outputs.py: PASS (5 passed, 1 warning in 0.02s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/log-summary-date-ranges` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 40,213 | 31.28s | pytest test_outputs.py: PASS (2 passed in 0.00s) |
| Track 1: Terminal-Bench 2.1 (Stanford/Laude) | `TB2.1/cancel-async-tasks` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 1 | 10,162 | 13.88s | pytest test_outputs.py: FAIL (6 failed in 1.66s) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/py-pdf__pypdf-3317` | `actorii` | `actorii-sandbox:latest` | **PASS** | 16 | 186,540 | 74.35s | test_get_encoding__encoding_value_is_none: PASS ( 1 file changed, 1 insertion(+), 1 deletion(-)) |
| Track 2: SWE-bench-Live (2025-06 Post-Cutoff) | `SWE-Live/networkx__networkx-8105` | `actorii` | `actorii-sandbox:latest` | **FAIL** | 13 | 95,598 | 47.86s | test_random_tree: FAIL ( 1 file changed, 10 insertions(+)) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/forth` | `actorii` | `actorii-sandbox:latest` | **PASS** | 7 | 69,462 | 36.10s | pytest forth_test.py: PASS (54 passed in 0.02s) |
| Track 3: Aider Polyglot (Hard Exercism) | `Polyglot/grep` | `actorii` | `actorii-sandbox:latest` | **PASS** | 4 | 29,558 | 25.42s | pytest grep_test.py: PASS (25 passed in 0.02s) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/3773` | `actorii` | `actorii-sandbox:latest` | **PASS** | 9 | 84,238 | 60.27s | tests: 2/2 (PASS) |
| Track 4: LiveCodeBench v6 (2025-04 Post-Cutoff) | `LCB-v6/abc400_c` | `actorii` | `actorii-sandbox:latest` | **PASS** | 6 | 52,379 | 36.24s | tests: 3/3 (PASS) |
