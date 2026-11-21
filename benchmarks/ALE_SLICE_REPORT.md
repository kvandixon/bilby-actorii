# Agents' Last Exam (ALE v1.1) Stratified Slice Report

- **Model:** `qwen3.8:latest`
- **Tasks:** 2 official UC Berkeley RDI ALE v1.1 tasks (deterministic official Python graders)

| Task | Domain | Harness | Score | Pass | Steps | Tokens | Latency | Details |
|---|---|---|---:|:---:|---:|---:|---:|---|
| `ALE/cfr_game_theory_equilibrium` | Algorithms / Game Theory (CFR + MCCFR) | `actorii` | **0.0000** | FAIL | 23 | 194,468 | 900.34s | ALE score=0.0000 (FAIL) | Missing output/results.json |
| `ALE/power_10kv_feeder_reliability_001` | Electrical Engineering / CIM-RDF + SVG Reliability | `actorii` | **0.0000** | FAIL | 32 | 648,970 | 548.25s | ALE score=0.0000 (FAIL) | Missing output/reliability_indices.json |

**Aggregate:** `0/2` full pass | Mean ALE Score: **`0.0000`**
