# ActorII Bilby Self-Evolution Benchmark Results

- mode: `live` | model: `qwen3.8:latest` | temperature: 0.0 | seed: 42
- timestamp: 2026-11-17T16:31:55-0500 | persistence backend: `fileout`

| spec | tier | status | pass@1 | pass@G | gens | steps | gas | holdout | persist | rollbacks | latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `sensor_window_median` | tier1_method | failed | N | N | 3/3 | 10 | 10786 | 75% | N | 0 | 40.216s |
| `ring_buffer_telemetry` | tier2_class_engram | failed | N | N | 3/3 | 15 | 16284 | 0% | N | 3 | 49.637s |
| `circuit_breaker_policy` | tier2_class_engram | failed | N | N | 3/3 | 14 | 3066 | 10% | N | 2 | 46.307s |
| `dag_task_scheduler` | tier2_class_engram | failed | N | N | 3/3 | 10 | 2499 | 23% | N | 3 | 24.187s |
| `harness_step_budget_advisor` | tier3_harness_policy | failed | N | N | 3/3 | 15 | 8127 | 33% | N | 1 | 60.816s |

**Summary:** pass@1 = 0/5 (0%), pass@G = 0/5 (0%), mean generations = 3.0, mean holdout = 28%, persistence verified = 0, rollbacks = 9, reward-hacking flags = 0, errors = 0, steps = 64, gas = 40762, total latency = 221.17s.

## Per-generation detail

### `sensor_window_median` - Hot-add `median` to a live SensorWindow class
- Phase 0 baseline: visible 1/3, holdout 1/8 (must not be fully passing)
- gen 1: **accepted** - visible 1/3, holdout 6/8, steps 2, gas 3588, 8.56s, diagnostics_given=False
- gen 2: **accepted** - visible 1/3, holdout 6/8, steps 4, gas 3599, 15.692s, diagnostics_given=True
- gen 3: **accepted** - visible 1/3, holdout 6/8, steps 4, gas 3599, 15.886s, diagnostics_given=True

### `ring_buffer_telemetry` - Bounded sliding-window percentile / anomaly tracker
- Phase 0 baseline: visible 1/4, holdout 0/15 (must not be fully passing)
- gen 1: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 0/15, steps 7, gas 5444, 28.809s, diagnostics_given=False
- gen 2: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 0/15, steps 4, gas 5420, 10.493s, diagnostics_given=True
- gen 3: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 0/15, steps 4, gas 5420, 10.268s, diagnostics_given=True

### `circuit_breaker_policy` - Stateful closed/open/half-open retry guard
- Phase 0 baseline: visible 0/5, holdout 0/10 (must not be fully passing)
- gen 1: **accepted** - visible 2/5, holdout 1/10, steps 6, gas 1030, 23.839s, diagnostics_given=False
- gen 2: **rolled_back**, rollback=no_definitions - visible 2/5, holdout 1/10, steps 4, gas 1018, 10.065s, diagnostics_given=True
- gen 3: **rolled_back**, rollback=no_definitions - visible 2/5, holdout 1/10, steps 4, gas 1018, 12.308s, diagnostics_given=True

### `dag_task_scheduler` - Topological dependency resolver with cycle detection
- Phase 0 baseline: visible 1/4, holdout 3/13 (must not be fully passing)
- gen 1: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 3/13, steps 1, gas 833, 2.32s, diagnostics_given=False
- gen 2: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 3/13, steps 5, gas 833, 11.638s, diagnostics_given=True
- gen 3: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 3/13, steps 4, gas 833, 10.15s, diagnostics_given=True

### `harness_step_budget_advisor` - Metacircular step-budget advisor (oscillation / failure detection)
- Phase 0 baseline: visible 0/4, holdout 0/15 (must not be fully passing)
- gen 1: **accepted** - visible 1/4, holdout 5/15, steps 6, gas 2709, 21.815s, diagnostics_given=False
- gen 2: **rolled_back**, rollback=no_definitions - visible 1/4, holdout 5/15, steps 4, gas 2709, 17.423s, diagnostics_given=True
- gen 3: **accepted** - visible 1/4, holdout 5/15, steps 5, gas 2709, 21.503s, diagnostics_given=True
