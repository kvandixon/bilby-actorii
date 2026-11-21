> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# ActorII AgentHarness — Official OpenAI HumanEval & GSM8K Benchmark Report

- **Model**: `qwen3.8:latest` (`valkyrie` local `:8080`)
- **Timestamp**: `2026-11-17T16:28:14-0500`

## 1. Executive Summary (Head-to-Head Harness Comparison)

| Benchmark Suite | 1-Shot Actor 4.1 (`Pass@1`, No Harness) | **ActorII `AgentHarness` (Multi-Step + Linter/Test Recovery)** | Python `Pass@1` Reference |
| :--- | :---: | :---: | :---: |
| **OpenAI HumanEval** (MultiPL-E Track, $N=12$) | 12/12 (100.0%) | **12/12 (100.0%)** (+0 self-corrected) | 12/12 (100.0%) |
| **OpenAI GSM8K** (PAL Math Track, $N=8$) | 8/8 (100.0%) | **8/8 (100.0%)** | — |

## 2. Per-Task OpenAI HumanEval Results

| Task ID | Entry Point | 1-Shot Actor 4.1 | ActorII `AgentHarness` | Steps | Gas (Opcodes) | Python `Pass@1` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `HumanEval/3` | `below_zero` | PASS | PASS | 1 | 692 | PASS |
| `HumanEval/13` | `greatest_common_divisor` | PASS | PASS | 1 | 225 | PASS |
| `HumanEval/23` | `strlen` | PASS | PASS | 1 | 39 | PASS |
| `HumanEval/24` | `largest_divisor` | PASS | PASS | 1 | 1604 | PASS |
| `HumanEval/31` | `is_prime` | PASS | PASS | 1 | 1792 | PASS |
| `HumanEval/35` | `max_element` | PASS | PASS | 1 | 282 | PASS |
| `HumanEval/41` | `car_race_collision` | PASS | PASS | 1 | 56 | PASS |
| `HumanEval/45` | `triangle_area` | PASS | PASS | 1 | 51 | PASS |
| `HumanEval/52` | `below_threshold` | PASS | PASS | 1 | 482 | PASS |
| `HumanEval/53` | `add` | PASS | PASS | 1 | 1161 | PASS |
| `HumanEval/55` | `fib` | PASS | PASS | 1 | 810 | PASS |
| `HumanEval/60` | `sum_to_n` | PASS | PASS | 1 | 96 | PASS |

## 3. Per-Task OpenAI GSM8K Results

| Task ID | Gold Answer | Predicted | ActorII `AgentHarness` | Steps | Gas (Opcodes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `GSM8K/0` | `18` | `18` | PASS | 1 | 25 |
| `GSM8K/1` | `3` | `3` | PASS | 1 | 34 |
| `GSM8K/2` | `70000` | `70000` | PASS | 1 | 37 |
| `GSM8K/3` | `540` | `540` | PASS | 2 | 32 |
| `GSM8K/4` | `20` | `20` | PASS | 2 | 33 |
| `GSM8K/5` | `64` | `64` | PASS | 2 | 344 |
| `GSM8K/6` | `260` | `260` | PASS | 2 | 36 |
| `GSM8K/7` | `160` | `160` | PASS | 3 | 25 |
