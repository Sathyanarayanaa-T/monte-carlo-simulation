#!/usr/bin/env bash
# Production Tier Runner (Milestone 6.5 Part G)
# Usage: ./scripts/run_tier.sh <tier> (e.g. ./scripts/run_tier.sh tier1)

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tier> (e.g. tier1 or tier2)"
    exit 1
fi

TIER="$1"

if [ "$TIER" != "tier1" ] && [ "$TIER" != "tier2" ]; then
    echo "Error: Tier must be either 'tier1' or 'tier2', got '$TIER'"
    exit 1
fi

# Pin threadpool environment to avoid CPU oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs results

LOG_FILE="logs/${TIER}.log"
echo "========================================================================" >> "$LOG_FILE"
echo "Starting Run: Tier '${TIER}' at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$LOG_FILE"
echo "System: $(uname -a)" >> "$LOG_FILE"
echo "Threads: OMP=$OMP_NUM_THREADS, OPENBLAS=$OPENBLAS_NUM_THREADS, MKL=$MKL_NUM_THREADS" >> "$LOG_FILE"
echo "Nice: -n 10 | Systemd Inhibit: sleep:idle | Auto-resume: ENABLED" >> "$LOG_FILE"
echo "========================================================================" >> "$LOG_FILE"

echo "Running Tier '${TIER}' under nice -n 10 and systemd-inhibit. Logging to ${LOG_FILE}..."

# Execute with systemd-inhibit, nice, and append to log file while teeing to terminal
systemd-inhibit --what=sleep:idle \
    --who="Monte Carlo Simulation" \
    --why="Running ${TIER} simulation batch" \
    --mode=block \
    nice -n 10 \
    uv run python -m src.run --tier "$TIER" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE="${PIPESTATUS[0]}"

echo "========================================================================" >> "$LOG_FILE"
echo "Finished Run: Tier '${TIER}' at $(date -u +"%Y-%m-%dT%H:%M:%SZ") with exit code ${EXIT_CODE}" >> "$LOG_FILE"
echo "========================================================================" >> "$LOG_FILE"

exit "$EXIT_CODE"
