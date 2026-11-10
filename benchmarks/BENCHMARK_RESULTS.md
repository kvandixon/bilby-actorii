# Bilby (`ActorII v4.1`) — Benchmark Results Summary

All results below were executed locally on `valkyrie` (NVIDIA RTX 5070 Ti 16 GB VRAM) using **`Bilby (ActorII v4.1)`** backed by Neo4j `:ActorEpisode`/`:ActorEngram` hybrid memory and `qwen3.8-27b` (`:8080`, `Q3_K_M`) with zero mid-run intervention.

| Benchmark Suite | Questions / Tasks (`N`) | Bilby (`ActorII v4.1`) Score | Completed | Notes & Raw Ledger |
| :--- | :---: | :---: | :---: | :--- |
| **`LongMemEval` (`S` Split)** | `500` | **`94.60%` (`473 / 500`)** | `2026-11-07` | Multi-session conversational memory (`single-session-preference` `100%`, `knowledge-update` `98.7%`, `single-session-user` `98.6%`, `multi-session` `94.7%`, `temporal-reasoning` `94.7%`). ([`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json)) |
| **`LoCoMo-10` (Full 5-Category Suite)** | `1,986` | **`77.34%` (`1,536 / 1,986`)** | `2026-11-07` | Full Snap Research 5-category conversational memory benchmark (`cat1`–`cat4` non-adversarial: `1,168 / 1,540 = 75.84%`; `cat5` adversarial abstention: `368 / 446 = 82.51%`). ([`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json)) |
| **`Terminal-Bench 2.1` (Container Subset)** | `5` | **`80.00%` (`4 / 5`)** | `2026-11-09` | Isolated Docker container evaluation (`actorii-sandbox:latest`). ([`TB21_5TASK_RESULTS.json`](archive/TB21_5TASK_RESULTS.json)) |
| **`Industry 8-Track Suite`** | `22` | **`86.36%` (`19 / 22`)** | `2026-11-09` | `HumanEval`, `MBPP`, `GSM8K`, `MMLU-Stem`, `SWE-Lite`, `QuixBugs`, `AgentBench-OS`, `CRUXEval` (`72.73%` fresh-clone baseline). ([`INDUSTRY_SUITE_3HARNESS_RESULTS.json`](archive/INDUSTRY_SUITE_3HARNESS_RESULTS.json)) |
| **`Next-Gen Post-Cutoff Suite`** | `9` | **`77.78%` (`7 / 9`)** | `2026-11-03` | `TB2.1`, `SWE-Bench-Live`, `Aider Polyglot`, `LiveCodeBench v6` (`55.56%` fresh-clone baseline). ([`NEXTGEN_RESULTS_QWEN38_BASELINE_7_OF_9.json`](archive/NEXTGEN_RESULTS_QWEN38_BASELINE_7_OF_9.json)) |
| **`Red-Team Adversarial Containment Audit`** | `6` | **`100.00%` (`6 / 6` Blocked)** | `2026-11-09` | Prompt injection, `/etc/shadow` read, Ring-0 mutation, env exfiltration, and path traversal under `ACTORII_HOST_LOCKDOWN=1`. ([`REDTEAM_CONTAINMENT_RESULTS.json`](archive/REDTEAM_CONTAINMENT_RESULTS.json)) |
