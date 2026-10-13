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

# Calculate measured seconds per replication per tier and print component timing breakdown
uv run python -c "
import json
from pathlib import Path
import pandas as pd
import numpy as np

dur_t1_none = float('$DUR_T1_NONE')
dur_t1_inter = float('$DUR_T1_INTER')
dur_t2_none = float('$DUR_T2_NONE')
dur_t2_inter = float('$DUR_T2_INTER')
total_time = float('$T_TOTAL')

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

print('\n' + '=' * 80)
print('TIMING BENCHMARK RESULTS (PER REPLICATION)')
print('=' * 80)
print(f'Tier 1 Measured Rate: {sec_per_rep_t1:.2f}s per replication (n=150)')
print(f'Tier 2 Measured Rate: {sec_per_rep_t2:.2f}s per replication (n=150)')
print(f'Total Benchmark Wall-Clock Duration: {total_time:.1f}s ({total_time / 60:.2f} minutes)')
print(f'Saved benchmark data to ${TIMING_DIR}/measured_timing.json')

# Component timing breakdown from shards
shard_dir = Path('${TIMING_DIR}')
parquet_files = sorted(list(shard_dir.glob('*.parquet')))
if parquet_files:
    print('\n' + '-' * 80)
    print('COMPONENT TIMING BREAKDOWN PER METHOD (MEAN SECONDS PER REPLICATION)')
    print('-' * 80)
    for p_file in parquet_files:
        df = pd.read_parquet(p_file)
        sc_name = p_file.stem
        print(f'\nScenario: {sc_name} (R = {len(df)} reps)')
        print(f'  Overall Mean Replication Time: {df[\"t_total\"].mean():.2f}s')
        
        methods = ['m3', 'm4', 'm5', 'm6']
        header = f'  {\"Method\":<8s} | {\"Nested CV\":<12s} | {\"Full Fit\":<12s} | {\"Null Test\":<12s} | {\"Bootstrap\":<12s} | {\"Method Total\":<12s}'
        print(header)
        print('  ' + '-' * 75)
        for m in methods:
            tot_col = f'{m}_t_total'
            if tot_col in df.columns and df[tot_col].sum() > 0:
                ncv = df.get(f'{m}_time_nested_cv', pd.Series([0])).mean()
                ff = df.get(f'{m}_time_full_fit', pd.Series([0])).mean()
                null_t = df.get(f'{m}_time_null_test', pd.Series([0])).mean()
                boot = df.get(f'{m}_time_bootstrap', pd.Series([0])).mean()
                tot = df.get(tot_col, pd.Series([0])).mean()
                print(f'  {m.upper():<8s} | {ncv:<12.2f} | {ff:<12.2f} | {null_t:<12.2f} | {boot:<12.2f} | {tot:<12.2f}')
print('=' * 80)
"
