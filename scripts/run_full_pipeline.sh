#!/usr/bin/env bash
# Master Pipeline Runner: Full Monte Carlo Simulation & Publication Reporting
# SPEC.md Sections 3-10, 12 | Milestone 6.8 & Milestone 7
#
# Executes the entire study end-to-end at full scale:
#   1. Real Hardware Timing Calibration (scripts/time_e2e.sh, scripts/compare_timing.py)
#   2. Tier 1 Core Grid (28 scenarios x 1,000 replications = 28,000 tasks)
#   3. Tier 1 Sensitivity Grid (40 scenarios x 1,000 replications = 40,000 tasks)
#   4. ML Residual-Resampling Null Calibration (scripts/null_calibration.py)
#   5. Tier 2 Machine Learning Grid (14 scenarios x 300 replications = 4,200 tasks)
#   6. Parquet Results Archive & Backup (scripts/backup_results.sh)
#   7. Publication Figures & Appendix Tables (scripts/generate_paper_plots.py)
#
# Features:
#   - Pinned single-threading to eliminate CPU thread contention
#   - High priority execution via nice -n 10
#   - System sleep/idle inhibition via systemd-inhibit
#   - Full replication-level resume (safe to interrupt and restart at any time)
#   - Master execution logging with UTC timestamps to logs/pipeline_master.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure directories exist
mkdir -p logs results results_backup report/figures report/tables calibration

LOG_FILE="logs/pipeline_master.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "STARTING FULL MONTE CARLO SIMULATION PIPELINE"
echo "UTC Start Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Repository:     $REPO_ROOT"
echo "Git Commit:     $(git rev-parse HEAD) (dirty: $(git status --porcelain | wc -l))"
echo "Hardware:       $(lscpu | grep 'Model name' | sed 's/Model name:[ ]*//')"
echo "================================================================================"

# Pinned threadpools to avoid oversubscription (Decision D10)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    echo ""
    echo "================================================================================"
    echo "[STEP ${step_num}] ${step_name}"
    echo "Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "================================================================================"
    
    local t_start
    t_start=$(date +%s)
    
    # Run command under systemd-inhibit if available
    if command -v systemd-inhibit >/dev/null 2>&1; then
        systemd-inhibit --what=sleep:idle \
            --who="Monte Carlo Simulation" \
            --why="Running ${step_name}" \
            --mode=block \
            nice -n 10 "$@"
    else
        nice -n 10 "$@"
    fi

    local t_end
    t_end=$(date +%s)
    local duration=$((t_end - t_start))
    echo "[STEP ${step_num} COMPLETED] Duration: ${duration}s ($((duration / 60)) min)"
}

# ------------------------------------------------------------------------------
# STEP 1: Hardware Timing Calibration (~3-5 min)
# ------------------------------------------------------------------------------
run_step "1/7" "Hardware Timing Calibration" \
    bash -c "./scripts/time_e2e.sh && uv run python scripts/compare_timing.py"

# ------------------------------------------------------------------------------
# STEP 2: Tier 1 Core Scenarios (28,000 tasks, ~2.5 hrs)
# ------------------------------------------------------------------------------
run_step "2/7" "Tier 1 Core Scenarios (28 Scenarios, R=1,000)" \
    ./scripts/run_tier.sh tier1 core

echo ""
echo "[Audit] Running sanity check on finished Tier 1 Core shards..."
uv run python scripts/sanity_report.py --group core

# ------------------------------------------------------------------------------
# STEP 3: Tier 1 Sensitivity Scenarios (40,000 tasks, ~3.4 hrs)
# ------------------------------------------------------------------------------
run_step "3/7" "Tier 1 Sensitivity Scenarios (40 Scenarios, R=1,000)" \
    ./scripts/run_tier.sh tier1 sens

echo ""
echo "[Audit] Running sanity check on finished Tier 1 Sensitivity shards..."
uv run python scripts/sanity_report.py --group sens

# ------------------------------------------------------------------------------
# STEP 4: Tier 2 Null Calibration Check (~2 min)
# ------------------------------------------------------------------------------
run_step "4/7" "Tier 2 Residual-Resampling Null Calibration" \
    uv run python scripts/null_calibration.py --reps 100 --reps-m6 50 --n-jobs 6 --out calibration/

# ------------------------------------------------------------------------------
# STEP 5: Tier 2 Machine Learning Grid (4,200 tasks, ~11.7 hrs)
# Staged in 100-replication intervals (Decision D16) for checkpointing
# ------------------------------------------------------------------------------
run_step "5/7 (Stage A)" "Tier 2 ML Grid: R = 100 Replications" \
    uv run python -m src.run --tier tier2 --max-reps 100

run_step "5/7 (Stage B)" "Tier 2 ML Grid: R = 200 Replications" \
    uv run python -m src.run --tier tier2 --max-reps 200

run_step "5/7 (Stage C)" "Tier 2 ML Grid: R = 300 Replications (Final)" \
    uv run python -m src.run --tier tier2 --max-reps 300

# ------------------------------------------------------------------------------
# STEP 6: Backup Raw Parquet Shards
# ------------------------------------------------------------------------------
run_step "6/7" "Results Archiving & Backup" \
    ./scripts/backup_results.sh results_backup/

# ------------------------------------------------------------------------------
# STEP 7: Generate Publication Figures & Appendix Tables
# ------------------------------------------------------------------------------
run_step "7/7" "Generate Publication Figures & Tables" \
    uv run python scripts/generate_paper_plots.py --results-dir results/ --out-dir report/

echo ""
echo "================================================================================"
echo "FULL SIMULATION PIPELINE COMPLETED SUCCESSFULLY!"
echo "UTC Completion Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "================================================================================"
echo ""
echo "Summary of Generated Artifacts:"
echo "  - Raw Parquet Shards: results/*.parquet"
echo "  - Archived Shards:    results_backup/"
echo "  - Master Log:         $LOG_FILE"
echo "  - Publication Plots:  report/figures/ (Figures 1 through 7)"
echo "  - Appendix Tables:    report/tables/ (LaTeX & CSV)"
echo "================================================================================"
