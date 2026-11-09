> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Bounded System-2 Reasoning (`ACTORII_ENABLE_THINKING=1`) on the 2 Unsolved `Terminal-Bench 2.1` Hard Tasks

- **Model Backend**: `qwen3.8:latest` (`27B` dense GGUF on `:8080`, `valkyrie` RTX 5070 Ti 16 GB VRAM)
- **Reasoning Architecture**: 2-Pass Adaptive System-1 / System-2 Pipeline in [`ValkyrieLLMClient._call_http_endpoint`](file:///workspace/ActorII/src/actorii/ai.py#L180-L280):
  1. **Step 1 (System-1 Fast Inspection)**: `enable_thinking=False` (`~3s`) to inspect workspace files, test suites, and baseline scripts before spending reasoning tokens.
  2. **Step 2+ & Error Recovery (System-2 Bounded Reasoning -> Code Continuation)**:
     - **Pass 1**: `enable_thinking=True` (`max_tokens=600`, `temperature=0.6`, `top_p=0.95`) to derive the mathematical/algorithmic plan in `reasoning_content`.
     - **Pass 2**: Automatic continuation (`enable_thinking=False`, `force_no_think=True`, `max_tokens=2048`, `temperature=0.2`) conditioned on `[Draft Analysis from Thinking Phase]` so long solver scripts are emitted completely without mid-script token truncation.
  3. **Socket & Verifier Fixes**:
     - Increased HTTP socket timeout from `180s` to `360s` (`qwen3.8-27b` decodes at `~8.3 tok/s` with `-fit on` CPU layer offload, so 1,500-token solver scripts require `~190s`).
     - Added automatic `gcc -shared -fPIC -O3 feal.c -o libfeal.so` + `ctypes` binding in [`_verify_task_workspace`](file:///workspace/ActorII/benchmarks/run_tb21_hard_clade_tournament.py#L194-L220) so host verification does not fail on missing `<Python.h>`.

---

## Results Comparison: No-Think Clade vs Thinking Clade (`ACTORII_ENABLE_THINKING=1`)

### 1. `TB2.1-Hard/llm-inference-batching-scheduler` (ML Systems / Non-Linear Cost Optimization, 45m Expert Est.)
- **Single-Trajectory Baseline (`ActorII` / `Pi` / `Hermes`, No-Think)**: `1/6` tests (`fitness=0.1667`)
- **Huxley-Gödel Clade Tournament (No-Think)**:
  - Gen-1 `Clade-A`: `1/6` tests (`fitness=0.1667`)
  - Gen-1 `Clade-B`: `1/6` tests (`fitness=0.1667`)
  - Gen-2 `Clade-A.1`: `4/6` tests (`fitness=0.6667`, `2 failed, 4 passed`)
- **Huxley-Gödel Clade Tournament (`ACTORII_ENABLE_THINKING=1`)**:
  - Gen-1 `Clade-B` (Run 2): **`4/6` tests (`fitness=0.6667`) in Generation 1** (`7` steps, `132,463` tok)
  - Gen-2 `Clade-A.1` (Run 1): **`5/6` tests (`fitness=0.8333`, `1 failed, 5 passed in 0.05s`)** (`6` steps, `93,026` tok)
  - **What Thinking Solved**: Cleared the joint cross-bucket unique-shape constraint ($\le 8$ unique `(seq_align, heads_align, hidden_align)` tuples combined across both `plan_b1.jsonl` and `plan_b2.jsonl`), passing `test_output_files_exist`, `test_input_data_integrity`, `test_generate_and_schema`, `test_solution_shape_feasibility_and_batch_consistency`, and `test_solution_coverage_no_duplicates`. Only the simultaneous 8-bound non-linear `test_performance_thresholds` check remained.

### 2. `TB2.1-Hard/feal-differential-cryptanalysis` (4-Round FEAL Chosen-Plaintext Key Recovery, 480m Expert Est.)
- **Single-Trajectory Baseline (`ActorII` / `Pi` / `Hermes`, No-Think)**: `0/1` (`fitness=0.0`, timed out before creating `attack.py`)
- **Huxley-Gödel Clade Tournament (No-Think)**: `0/1` (`fitness=0.0`, `2` steps, aborted at `180s` socket timeout before writing `attack.py`)
- **Huxley-Gödel Clade Tournament (`ACTORII_ENABLE_THINKING=1`)**:
  - Gen-1 `Clade-B`: `fitness=0.25` (`4` steps, `57,858` tok — synthesized and executed `/workspace/attack.py`)
  - Gen-1 `Clade-A`: `fitness=0.25` (`4` steps, `57,664` tok — synthesized and executed `/workspace/attack.py`)
  - Gen-2 `Clade-B.1`: `fitness=0.25` (`10` steps, `126,584` tok, `521.72s` — iteratively tested differential pairs against `feal.py`)
  - **Bottleneck**: `feal-differential-cryptanalysis` is rated `480 minutes` (8 hours) for a human cryptographer in `Terminal-Bench 2.1`. While 2-pass thinking enabled the 27B model to write and iterate on a chosen-plaintext differential solver (`/workspace/attack.py`) across 10 steps without syntax or timeout failures, deriving the exact 3-round differential characteristic (`0x80800000 -> 0x02000000`) and inverting the post-whitening `left ^= right` swap to isolate `key[5]` from the `65,536` candidate seeds (`(x * 1234567) & 0xFFFFFFFF`) exceeded the 27B model's cryptanalytic derivation depth.
