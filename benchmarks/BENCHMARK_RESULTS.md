> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Bilby (`ActorII v4.2.0`) — Benchmark Results Summary

All results below were executed locally on `valkyrie` (NVIDIA RTX 5070 Ti 16 GB VRAM, DDR5-6000 MT/s / 5400 MHz CPU boost) using **`Bilby (ActorII v4.2.0)`** backed by Neo4j `:ActorEpisode`/`:ActorEngram` hybrid memory and `qwen3.8-27b` (`:8080`, `Q3_K_M`) with prefix KV-cache invariance (`f_keep = 85.8%`), 5-strike self-correction (`maxFailures := 5`), `100,000` opcode gas budget, and zero mid-run intervention.

| Benchmark Suite | Questions / Tasks (`N`) | Bilby (`ActorII v4.2.0`) Score | Completed | Notes & Raw Ledger |
| :--- | :---: | :---: | :---: | :--- |
| **`LongMemEval` (`S` Split)** | `500` | **`94.60%` (`473 / 500`)** | `2026-11-07` | Multi-session conversational memory (`single-session-preference` `100%`, `knowledge-update` `98.7%`, `single-session-user` `98.6%`, `multi-session` `94.7%`, `temporal-reasoning` `94.7%`). ([`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json)) |
| **`LoCoMo-10` (Full 5-Category Suite)** | `1,986` | **`77.34%` (`1,536 / 1,986`)** | `2026-11-07` | Full Snap Research 5-category conversational memory benchmark (`cat1`–`cat4` non-adversarial: `1,168 / 1,540 = 75.84%`; `cat5` adversarial abstention: `368 / 446 = 82.51%`). ([`LONGMEM_LOCOMO_FULL_RESULTS.json`](archive/LONGMEM_LOCOMO_FULL_RESULTS.json)) |
| **`Terminal-Bench 2.1` (Container Subset)** | `5` | **`100.00%` (`5 / 5`)** | `2026-11-10` | Isolated Docker container evaluation (`actorii-sandbox:latest`), up from `80.00%` (`4/5`) after flipping `multi-source-data-merger` (`3/3` tests). ([`TB21_5TASK_RESULTS.json`](archive/TB21_5TASK_RESULTS.json)) |
| **`Industry 8-Track Suite` (Fresh Container Run)** | `22` | **`77.27%` (`17 / 22`)** | `2026-11-10` | `HumanEval` (`2/2`), `MBPP` (`2/2`), `GSM8K` (`2/2`), `QuixBugs` (`2/2`), `AgentBench-OS` (`2/2`), `CRUXEval` (`2/2`), `SWE-Lite` (`3/6`), `BigCodeBench-Hard` (`2/4`), up from `72.73%` (`16/22`) fresh-clone baseline (`86.36%` / `19/22` cumulative across runs). ([`INDUSTRY_SUITE_RESULTS.json`](archive/INDUSTRY_SUITE_RESULTS.json), [`INDUSTRY_SUITE_3HARNESS_RESULTS.json`](archive/INDUSTRY_SUITE_3HARNESS_RESULTS.json)) |
| **`Next-Gen Post-Cutoff Suite`** | `9` | **`77.78%` (`7 / 9`)** | `2026-11-10` | `TB2.1` (`3/3`), `SWE-Bench-Live` (`1/2`, `pypdf-3317` PASS), `Aider Polyglot` (`1/2`, `grep` PASS), `LiveCodeBench v6` (`2/2`, `3773` & `abc400_c` PASS), up from `55.56%` (`5/9`) pre-BIOS/retry baseline. ([`NEXTGEN_RESULTS.json`](archive/NEXTGEN_RESULTS.json)) |
| **`Red-Team Adversarial Containment Audit`** | `6` | **`100.00%` (`6 / 6` Blocked)** | `2026-11-09` | Prompt injection, `/etc/shadow` read, Ring-0 mutation, env exfiltration, and path traversal under `ACTORII_HOST_LOCKDOWN=1`. ([`REDTEAM_CONTAINMENT_RESULTS.json`](archive/REDTEAM_CONTAINMENT_RESULTS.json)) |
