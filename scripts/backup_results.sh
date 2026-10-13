#!/usr/bin/env bash
# scripts/backup_results.sh
# Milestone 6.6 Part G:
# rsync results/ and logs/ to <destination>/<YYYY-MM-DD-HHMM>/, then print sizes.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <destination_directory>"
    exit 1
fi

DEST_ROOT="$1"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M")
TARGET_DIR="${DEST_ROOT}/${TIMESTAMP}"

echo "Creating backup directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

if [ -d "results" ]; then
    echo "Syncing results/ ..."
    mkdir -p "${TARGET_DIR}/results"
    rsync -av results/ "${TARGET_DIR}/results/"
else
    echo "Directory 'results/' does not exist; skipping."
fi

if [ -d "logs" ]; then
    echo "Syncing logs/ ..."
    mkdir -p "${TARGET_DIR}/logs"
    rsync -av logs/ "${TARGET_DIR}/logs/"
else
    echo "Directory 'logs/' does not exist; skipping."
fi

echo ""
echo "Backup complete. Disk usage summary:"
du -sh "${TARGET_DIR}"
if [ -d "${TARGET_DIR}/results" ]; then
    du -sh "${TARGET_DIR}/results"
fi
if [ -d "${TARGET_DIR}/logs" ]; then
    du -sh "${TARGET_DIR}/logs"
fi
