#!/usr/bin/env bash
# End-to-end timing script for Tier 1 and Tier 2 benchmarks (Milestone 6.5 Part F / 6.6 Part E)
# Times 5 replications with every metric ON:
# - Nested CV (5-fold x 2 repeats)
# - Bootstrap stability selection (50 resamples)
# - TreeSHAP interaction values (M4, M5)
# - Residual-resampling null interaction test (B=50 refits)
# Output Directory: timing_shards/ (writes outside results/)

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
echo "Expected Total Time: ~3 to 5 minutes"
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

# Calculate measured seconds per replication per tier
uv run python -c "
import json
dur_t1_none = float('$DUR_T1_NONE')
dur_t1_inter = float('$DUR_T1_INTER')
dur_t2_none = float('$DUR_T2_NONE')
dur_t2_inter = float('$DUR_T2_INTER')
total_time = float('$T_TOTAL')

# In parallel runs with 5 reps on 6 jobs, wall-clock duration reflects per-rep time
sec_per_rep_t1 = (dur_t1_none + dur_t1_inter) / 2.0
sec_per_rep_t2 = (dur_t2_none + dur_t2_inter) / 2.0

data = {
    'tier1_sec_per_rep': round(sec_per_rep_t1, 3),
    'tier2_sec_per_rep': round(sec_per_rep_t2, 3),
    'dur_t1_none': dur_t1_none,
    'dur_t1_inter': dur_t1_inter,
    'dur_t2_none': dur_t2_none,
    'dur_t2_inter': dur_t2_inter,
    'total_wall_clock': total_time,
}

with open('${TIMING_DIR}/measured_timing.json', 'w') as f:
    json.dump(data, f, indent=2)

print('\n' + '=' * 72)
print('TIMING BENCHMARK RESULTS (PER REPLICATION)')
print('=' * 72)
print(f'Tier 1 Measured Rate: {sec_per_rep_t1:.2f}s per replication (n=150)')
print(f'Tier 2 Measured Rate: {sec_per_rep_t2:.2f}s per replication (n=150)')
print(f'Total Benchmark Wall-Clock Duration: {total_time:.1f}s ({total_time / 60:.2f} minutes)')
print('=' * 72)
print(f'Saved benchmark data to ${TIMING_DIR}/measured_timing.json')
"
