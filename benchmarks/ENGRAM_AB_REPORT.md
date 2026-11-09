> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Empirical A/B/C Benchmark Report: Active Engrams (Memory+Skill) vs Passive Graph-RAG vs RAM-Only OOP

## 1. Executive Summary Comparison Table

| Condition | Cold-Start Accuracy | Avg Steps / Query | Avg Tokens / Query | Avg Latency / Query | Upfront Ingest Tokens | Total Amortized Tokens (Setup + 12 Queries) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: RAM-Only OOP (No Neo4j Bridge)** | **0/12 (0.0%)** | `3.0` | `11087.7` | `35.69s` | `30687` | `163739` (625.13s) |
| **Condition B: Passive Graph-RAG (Cypher-in-Actor Only, No Stored Engrams)** | **8/12 (66.7%)** | `2.67` | `10212.8` | `40.13s` | `18541` | `141095` (549.04s) |
| **Condition C: Active Engrams (Neo4j Data + Stored & Hydrated Actor 4.1 Bytecode Skills)** | **12/12 (100.0%)** | `1.08` | `4002.6` | `10.16s` | `51204` | `99235` (371.59s) |

## 2. Per-Task Breakdown Across All 3 Conditions

| Task | Workflow | Condition A (RAM-Only) | Condition B (Passive Graph-RAG) | Condition C (Active Engrams) |
| :--- | :--- | :---: | :---: | :---: |
| `recipe_efficiency` (Multi-Constraint Recipe Efficiency Scorer) | Workflow 1: Existing Graph | **0/3** (9 steps, 32973 tok) | **2/3** (9 steps, 34406 tok) | **3/3** (4 steps, 14833 tok) |
| `service_cascade` (Service Cluster Spike Risk Calculator) | Workflow 1: Existing Graph | **0/3** (9 steps, 34886 tok) | **3/3** (7 steps, 27730 tok) | **3/3** (3 steps, 11061 tok) |
| `thermal_cooling` (Server Zone Iterative Cooling Curve Memory) | Workflow 2: Chat-Ingested Memory+Skill | **0/3** (9 steps, 33053 tok) | **3/3** (7 steps, 25897 tok) | **3/3** (3 steps, 11076 tok) |
| `contractor_ledger` (Contractor Tiered Overtime Payroll Memory) | Workflow 2: Chat-Ingested Memory+Skill | **0/3** (9 steps, 32140 tok) | **0/3** (9 steps, 34521 tok) | **3/3** (3 steps, 11061 tok) |
