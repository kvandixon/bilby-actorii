> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# ActorII AgentHarness — Official OpenAI HumanEval & GSM8K Benchmark Report

- **Model**: `qwen3.8:latest` (`valkyrie` local `:8080`)
- **Timestamp**: `2026-11-09T12:00:18-0500`

## 1. Executive Summary (Head-to-Head Harness Comparison)

| Benchmark Suite | 1-Shot Actor 4.1 (`Pass@1`, No Harness) | **ActorII `AgentHarness` (Multi-Step + Linter/Test Recovery)** | Python `Pass@1` Reference |
| :--- | :---: | :---: | :---: |
| **OpenAI HumanEval** (MultiPL-E Track, $N=12$) | 10/12 (83.3%) | **11/12 (91.7%)** (+1 self-corrected) | 12/12 (100.0%) |
| **OpenAI GSM8K** (PAL Math Track, $N=8$) | 7/8 (87.5%) | **7/8 (87.5%)** | — |

## 2. Per-Task OpenAI HumanEval Results

| Task ID | Entry Point | 1-Shot Actor 4.1 | ActorII `AgentHarness` | Steps | Gas (Opcodes) | Python `Pass@1` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `HumanEval/3` | `below_zero` | PASS | PASS | 1 | 641 | PASS |
| `HumanEval/13` | `greatest_common_divisor` | PASS | PASS | 1 | 225 | PASS |
| `HumanEval/23` | `strlen` | PASS | PASS | 1 | 39 | PASS |
| `HumanEval/24` | `largest_divisor` | PASS | PASS | 1 | 441 | PASS |
| `HumanEval/31` | `is_prime` | PASS | PASS | 1 | 1706 | PASS |
| `HumanEval/35` | `max_element` | PASS | PASS | 1 | 266 | PASS |
| `HumanEval/41` | `car_race_collision` | PASS | PASS | 1 | 56 | PASS |
| `HumanEval/45` | `triangle_area` | PASS | PASS | 1 | 51 | PASS |
| `HumanEval/52` | `below_threshold` | FAIL | **PASS (Recovered)** | 2 | 522 | PASS |
| `HumanEval/53` | `add` | PASS | PASS | 1 | 1371 | PASS |
| `HumanEval/55` | `fib` | FAIL | FAIL | 4 | 954 | PASS |
| `HumanEval/60` | `sum_to_n` | PASS | PASS | 1 | 2010 | PASS |

## 3. Per-Task OpenAI GSM8K Results

| Task ID | Gold Answer | Predicted | ActorII `AgentHarness` | Steps | Gas (Opcodes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `GSM8K/0` | `18` | `18` | PASS | 2 | 89 |
| `GSM8K/1` | `3` | `3` | PASS | 1 | 34 |
| `GSM8K/2` | `70000` | `195000.0` | FAIL | 2 | 79 |
| `GSM8K/3` | `540` | `540` | PASS | 2 | 108 |
| `GSM8K/4` | `20` | `20` | PASS | 2 | 92 |
| `GSM8K/5` | `64` | `64.0` | PASS | 2 | 452 |
| `GSM8K/6` | `260` | `260` | PASS | 2 | 173 |
| `GSM8K/7` | `160` | `160` | PASS | 2 | 105 |
