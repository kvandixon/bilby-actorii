> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# ActorII v4.1 — Future Harness Benchmark Roadmap

**Maintainer**: Skippy (Chief Architect & Big Brain Supervisor)  
**Target Stack**: `ActorII v4.1` Metacircular Agent Harness (`Bilby`) on `valkyrie` (`qwen3.8:latest` `:8080`, `bge-small-en-v1.5` `:8084`, Neo4j `:7474`, Hindsight `:8818`)  
**Created**: 2026-11-06

---

## 1. Existing Verified Baselines (Completed)

| Suite | Target Capability | Baseline Score | Reference Report |
| :--- | :--- | :--- | :--- |
| **Phase 1: ActorII vs Hermes Head-to-Head** | 5-track basic arithmetic, state inspection, BLE twin, Neo4j recall, sandboxing | **100% (5/5)** · `2.08x` token efficiency · `1.97x` lower latency | [`BENCHMARK_REPORT.md`](file:///workspace/ActorII/benchmarks/BENCHMARK_REPORT.md) |
| **Phase 2: Industry 4-Pillar Suite (N=19)** | `SWE-bench Verified` (5), `GAIA` (5), `OSWorld/CyberSecEval` (5), `ALFWorld` (4) | **100% (19/19)** · `38.1s` total (`2.01s/task`) | [`INDUSTRY_BENCHMARK_REPORT.md`](file:///workspace/ActorII/benchmarks/INDUSTRY_BENCHMARK_REPORT.md) |
| **Phase 3: Engram A/B Compaction** | Single-fact retention across context compaction (`compactContext`) | **100% (5/5)** with `:ActorEngram` vs **40% (2/5)** raw compaction | [`ENGRAM_AB_REPORT.md`](file:///workspace/ActorII/benchmarks/ENGRAM_AB_REPORT.md) |
| **Phase 4: Qwen3.8 vs Gemma4 Matrix** | Cross-model compliance & syntax adherence under ActorII grammar | **100% (19/19)** on `qwen3.8:latest` | [`QWEN38_VS_GEMMA4_BENCHMARK_REPORT.md`](file:///workspace/ActorII/benchmarks/QWEN38_VS_GEMMA4_BENCHMARK_REPORT.md) |
| **Phase 5: Benchmark #3 (`LongMemEval` + `LOCOMO` 3-Harness Head-to-Head)** | Multi-session chronological ingestion, `knowledge-update`, `temporal-reasoning`, and adversarial false-premise abstention (N=17 stratified x 4 conditions = 68 runs) | **ActorII v4.1: 94.1% (16/17)** (`34.8s`, `10,081` tok) vs **Hermes Agent: 82.4% (14/17)** (`32.0s`, `20,895` tok — `2.07x` more tokens) vs **Pi Coding Agent: 76.5% (13/17)** (`111.4s` — `3.20x` slower) vs **Raw Compaction: 47.1% (8/17)** | [`LONGMEM_LOCOMO_REPORT.md`](file:///workspace/ActorII/benchmarks/LONGMEM_LOCOMO_REPORT.md) |

---

## 2. Six-Dimension Future Benchmark Roadmap

While the Phase 1–4 suites verify single-session task execution, sandbox isolation, and basic 5-fact compaction, they do not stress-test multi-turn user friction, broken hardware/API recovery, contradicting multi-session episodic memory, live system fault injection, indirect prompt injection, or unseen post-cutoff multi-language repositories.

Below is the prioritized roadmap of next-generation harness benchmarks:

| # | Dimension | Benchmark Suite(s) | What It Tests in `ActorII` & `Bilby` | Target Metrics | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#3** | **Long-Horizon Episodic Memory & Temporal Drift** | **`LongMemEval` (ICLR 2025) + `LOCOMO` (Snap Research)** | Multi-session chronological ingestion, `knowledge-update` (superseding stale facts), `temporal-reasoning` ("how many days between X and Y"), `multi-session` aggregation, and `compactChatHistory` -> Neo4j `:ActorEpisode` + `:8084` vector hybrid recall. | Overall Recall %, Stale-Fact Override %, Token Compression Ratio, Retrieval Latency | **VERIFIED 3-HARNESS BASELINE (`ActorII: 94.1%` vs `Hermes: 82.4%` vs `Pi: 76.5%`)** |
| **#1** | **Multi-Turn Conversational State & Policy Retention** | **`τ-bench` (Tau-Bench) / `BFCL v3 Multi-Turn`** | Simulated multi-turn user dialogues (5–15 turns) where requirements shift mid-conversation while enforcing strict domain rules (e.g., `valkyrie` 16 GB VRAM budget & `INCIDENT_2026-10-18` service guardrails). Tests `telegram_gateway_state.json` persistence and `scratchpad` hygiene. | Pass^k consistency (`k=1..3`), Policy Violation Rate (`0%` target), Scratchpad Drift | Queued (Next) |
| **#2** | **Tool Fault Recovery, Schema Drift & Epistemic Calibration** | **`ToolBench` / `BFCL Fault-Injection` + `AbstentionBench`** | Injecting realistic hardware/API faults into Digital Twins (`#Thermometer` BLE timeout, Neo4j transient disconnect, malformed file paths) and unanswerable/false-premise queries. Measures whether Bilby diagnoses/retries cleanly or hallucinates success. | Fault Recovery Rate %, Hallucination Rate (`0%` target), Abstention Precision | Queued |
| **#4** | **Proactive SRE Fault Diagnosis & Root-Cause Analysis** | **`AIOpsLab` / `ITBench` (Valkyrie Digital Twin Edition)** | Seeding realistic infrastructure anomalies in the sandbox (stale `hindsight/config.json` `llm_model` mismatch, runaway zombie process, dead port, broken symlink) and asking Bilby to identify root cause + remediation using `stackHealth(Actor[#Valkyrie])!!`. | Time-to-Detect (TTD), Root-Cause Accuracy %, Zero Destructive Commands | Queued |
| **#5** | **Indirect Prompt Injection & Untrusted Data Defense** | **`AgentDojo` (ETH Zurich) / `BIPIA`** | Embedding adversarial `Action: ...!!` payloads inside untrusted files, scraped web pages, or Neo4j nodes read by Bilby during normal summarization tasks to verify data-plane vs control-plane separation. | Attack Success Rate (ASR, target `0%`), Benign Utility Preservation % | Queued |
| **#6** | **Unseen Post-Cutoff & Polyglot Software Engineering** | **`SWE-bench-Live` / `SWE-bench Multilingual` (Rust / C++ / Go / `.act`)** | Multi-file bug fixing in post-cutoff repositories plus cross-compilation tasks (e.g., `jetson-tx1-edge-ops` Rust/CUDA targets and self-hosted `.act` metacircular extensions) requiring iterative test-driven repair. | Resolved %, Avg Steps to Green Test, Gas Efficiency | Queued |

---

## 3. Execution Protocol for Each Benchmark Dimension

1. **No Mock Fallbacks**: Every benchmark runner in `benchmarks/` must execute against the live local `valkyrie` stack (`:8080` `qwen3.8:latest`, `:8084` `bge-small-en-v1.5`, `:7474` Neo4j) with zero synthetic scores.
2. **VRAM Discipline**: Benchmark runners must never load secondary GPU models onto the RTX 5070 Ti (`16.3 GB` physical VRAM). Judge/scoring passes must reuse `:8080`, `:8082` (CPU specialist), or deterministic exact/token-F1 evaluators.
3. **Artifact Archival**: Each run outputs both raw machine-readable JSON (`benchmarks/<NAME>_RESULTS.json`) and a human-readable Markdown report (`benchmarks/<NAME>_REPORT.md`) with the mandatory planning/verification header.
