#!/usr/bin/env bash
set -euo pipefail

cd /home/skippy/projects/ActorII
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

MODEL="qwen3.8:latest"
VERSION="v4.4.0"
DATE_STR="2026-11-21"
NOTES="v4.4.0 clean-room sweep (Step-1 prompt trim, UnaryOp fix, Leaps #1-#3 cognitive bridges + autonomous hooks)"
ARCHIVE="benchmarks/archive"
mkdir -p "${ARCHIVE}"

ingest_and_render() {
    local bench_id="$1"
    local json_file="$2"
    local extra="${3:-}"
    if [[ -f "${json_file}" ]]; then
        echo ">>> [SCOREBOARD] Ingesting ${bench_id} from ${json_file} ..."
        python3 benchmarks/scoreboard.py ingest \
            --bench "${bench_id}" \
            --model "${MODEL}" \
            --version "${VERSION}" \
            --date "${DATE_STR}" \
            --notes "${NOTES}" \
            ${extra} \
            "${json_file}" || echo "WARNING: ingest failed for ${bench_id}"
        python3 benchmarks/scoreboard.py render || true
    fi
}

echo "========================================================================"
echo "  ACTORII v4.4.0 FULL GITHUB CLEAN-ROOM BENCHMARK SWEEP (${DATE_STR})"
echo "========================================================================"

echo ""
echo "[1/8] Live Red-Team Adversarial Containment Audit (6 tasks)..."
python3 benchmarks/run_redteam_containment_audit.py --model "${MODEL}"
cp -f benchmarks/REDTEAM_CONTAINMENT_RESULTS.json "${ARCHIVE}/REDTEAM_CONTAINMENT_${DATE_STR}_v440.json"
ingest_and_render "redteam-containment" "${ARCHIVE}/REDTEAM_CONTAINMENT_${DATE_STR}_v440.json"

echo ""
echo "[2/8] HumanEval-12 + GSM8K-8 Standard Matrix (20 tasks)..."
python3 benchmarks/run_standard_benchmarks.py --humaneval-n 12 --gsm8k-n 8 --model "${MODEL}"
cp -f benchmarks/BENCHMARK_RESULTS.json "${ARCHIVE}/BENCHMARK_RESULTS_${DATE_STR}_v440.json"
ingest_and_render "humaneval-gsm8k-matrix" "${ARCHIVE}/BENCHMARK_RESULTS_${DATE_STR}_v440.json"

echo ""
echo "[3/8] LongMemEval + LoCoMo Stratified 17-Task Suite..."
python3 benchmarks/run_longmem_locomo_suite.py --only-actorii --skip-raw
cp -f benchmarks/LONGMEM_LOCOMO_RESULTS.json "${ARCHIVE}/LONGMEM_LOCOMO_STRATIFIED_${DATE_STR}_v440.json"
ingest_and_render "longmem-locomo-stratified" "${ARCHIVE}/LONGMEM_LOCOMO_STRATIFIED_${DATE_STR}_v440.json"

echo ""
echo "[4/8] Bilby Deterministic Self-Evolution Loop (5 specs live on :8080)..."
python3 scripts/run_self_evolution_benchmark.py \
    --mode live --specs all --model "${MODEL}" \
    --output-json "${ARCHIVE}/SELF_EVO_${DATE_STR}_v440.json" || true
ingest_and_render "self-evolution-loop" "${ARCHIVE}/SELF_EVO_${DATE_STR}_v440.json"

echo ""
echo "[5/8] Industry 8-Track SWE Suite (22 tasks in actorii-sandbox:latest)..."
python3 benchmarks/run_industry_harness_suite.py \
    --harnesses actorii --preset 8track --model "${MODEL}" \
    --output-json "${ARCHIVE}/INDUSTRY_8TRACK_${DATE_STR}_v440.json" \
    --output-md "${ARCHIVE}/INDUSTRY_8TRACK_${DATE_STR}_v440.md"
ingest_and_render "industry-8track" "${ARCHIVE}/INDUSTRY_8TRACK_${DATE_STR}_v440.json"

echo ""
echo "[6/8] Terminal-Bench 2.1 Container Subset (5 tasks x 3 trials)..."
python3 benchmarks/run_tb21_5task_suite.py \
    --harness actorii --trials 3 --model "${MODEL}" \
    --results-json "${ARCHIVE}/TB21_5TASK_AVG3_${DATE_STR}_v440.json" \
    --report-md "${ARCHIVE}/TB21_5TASK_AVG3_${DATE_STR}_v440.md"
ingest_and_render "tb21-container" "${ARCHIVE}/TB21_5TASK_AVG3_${DATE_STR}_v440.json" "--trials 3"

echo ""
echo "[7/8] Terminal-Bench 2.1 Hard Tier (5 tasks)..."
python3 benchmarks/run_tb21_hard_suite.py \
    --harness actorii --trials 1 --model "${MODEL}" \
    --results-json "${ARCHIVE}/TB21_HARD_${DATE_STR}_v440.json" \
    --report-md "${ARCHIVE}/TB21_HARD_${DATE_STR}_v440.md"
ingest_and_render "tb21-hard" "${ARCHIVE}/TB21_HARD_${DATE_STR}_v440.json" "--trials 1"

echo ""
echo "[8/8] Next-Gen Post-Cutoff Suite (9 tasks x 3 trials)..."
python3 benchmarks/run_nextgen_suite.py \
    --harness actorii --trials 3 --reset --model "${MODEL}" \
    --results-json "${ARCHIVE}/NEXTGEN_AVG3_${DATE_STR}_v440.json" \
    --report-md "${ARCHIVE}/NEXTGEN_AVG3_${DATE_STR}_v440.md"
ingest_and_render "nextgen-postcutoff" "${ARCHIVE}/NEXTGEN_AVG3_${DATE_STR}_v440.json" "--trials 3"

echo ""
echo "========================================================================"
echo "  ALL 8 SUITES COMPLETE — SCOREBOARD RENDERED!"
echo "========================================================================"
