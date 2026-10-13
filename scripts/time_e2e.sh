#!/usr/bin/env bash
# End-to-end timing script for Tier 1 and Tier 2 benchmarks (Milestone 6.5 Part F)
# Times 5 replications with every metric ON:
# - Nested CV (5-fold x 2 repeats)
# - Bootstrap stability selection (50 resamples)
# - TreeSHAP interaction values (M4, M5)
# - Residual-resampling null interaction test (B=50 refits)

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TIMING_DIR="timing_shards"
rm -rf "${TIMING_DIR}"
mkdir -p "${TIMING_DIR}"

echo "========================================================================"
echo "Starting End-to-End Timing Benchmark (5 replications each at n=150)"
echo "Output Directory: ${TIMING_DIR} (writes outside results/)"
echo "========================================================================"

T0=$(date +%s.%N)

echo ""
echo "[1/4] Tier 1: base_n150_none (5 reps, n_jobs=6)..."
T_START=$(date +%s.%N)
uv run python -m src.run --tier tier1 --scenario base_n150_none --max-reps 5 --n-jobs 6 --out-dir "${TIMING_DIR}" --force
T_END=$(date +%s.%N)
DUR_T1_NONE=$(echo "$T_END - $T_START" | bc)
echo "  -> Completed in ${DUR_T1_NONE}s"

echo ""
echo "[2/4] Tier 1: base_n150_inter_med (5 reps, n_jobs=6)..."
T_START=$(date +%s.%N)
uv run python -m src.run --tier tier1 --scenario base_n150_inter_med --max-reps 5 --n-jobs 6 --out-dir "${TIMING_DIR}" --force
T_END=$(date +%s.%N)
DUR_T1_INTER=$(echo "$T_END - $T_START" | bc)
echo "  -> Completed in ${DUR_T1_INTER}s"

echo ""
echo "[3/4] Tier 2: base_n150_none (5 reps, n_jobs=6)..."
T_START=$(date +%s.%N)
uv run python -m src.run --tier tier2 --scenario base_n150_none --max-reps 5 --n-jobs 6 --out-dir "${TIMING_DIR}" --force
T_END=$(date +%s.%N)
DUR_T2_NONE=$(echo "$T_END - $T_START" | bc)
echo "  -> Completed in ${DUR_T2_NONE}s"

echo ""
echo "[4/4] Tier 2: base_n150_inter_med (5 reps, n_jobs=6)..."
T_START=$(date +%s.%N)
uv run python -m src.run --tier tier2 --scenario base_n150_inter_med --max-reps 5 --n-jobs 6 --out-dir "${TIMING_DIR}" --force
T_END=$(date +%s.%N)
DUR_T2_INTER=$(echo "$T_END - $T_START" | bc)
echo "  -> Completed in ${DUR_T2_INTER}s"

T_TOTAL=$(echo "$(date +%s.%N) - $T0" | bc)

echo ""
echo "========================================================================"
echo "Timing Benchmark Summary"
echo "========================================================================"
echo "Tier 1 base_n150_none:      ${DUR_T1_NONE}s (for 5 reps)"
echo "Tier 1 base_n150_inter_med:  ${DUR_T1_INTER}s (for 5 reps)"
echo "Tier 2 base_n150_none:      ${DUR_T2_NONE}s (for 5 reps)"
echo "Tier 2 base_n150_inter_med:  ${DUR_T2_INTER}s (for 5 reps)"
echo "Total Wall-Clock Time:      ${T_TOTAL}s"
echo "========================================================================"
