#!/bin/bash
#SBATCH --job-name=download_echo
#SBATCH --partition=mit_preemptable
#SBATCH --array=0-99
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/sebasmos/download_echo_%A_%a.log

# ============================================================
# MIMIC-IV-Echo Parallel Download (100 tasks, direct per-file)
# ============================================================
# Uses a pre-generated manifest of 525,422 file paths.
# Each array task downloads every 100th file (round-robin).
# wget -N skips already-downloaded files; -c resumes partials.
#
# Usage:
#   1. Edit scripts/.env with your PhysioNet credentials
#   2. Submit:  sbatch scripts/download.sh
#   3. Resume:  sbatch scripts/download.sh  (auto-resumes)
#
# Monitor:
#   squeue -u $USER --name=download_echo | wc -l
#   find /orcd/pool/006/lceli_shared/mimic-iv-echo/files/ -name "*.dcm" | wc -l
# ============================================================

DEST="/orcd/pool/006/lceli_shared/mimic-iv-echo"
LOGDIR="${DEST}/.download_metadata"
MANIFEST="${LOGDIR}/manifest.txt"
BASE_URL="https://physionet.org/files/mimic-iv-echo/0.1"

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_TASKS=100
PROGRESS_LOG="${LOGDIR}/progress_chunk_${TASK_ID}.log"

# Load credentials from .env
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ENV_FILE="${SLURM_SUBMIT_DIR}/scripts/.env"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ENV_FILE="${SCRIPT_DIR}/.env"
fi

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .env file not found at ${ENV_FILE}"
    exit 1
fi

PN_USER="${PHYSIONET_USER:?Error: PHYSIONET_USER not set in .env}"
PN_PASS="${PHYSIONET_PASS:?Error: PHYSIONET_PASS not set in .env}"

# Check manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at ${MANIFEST}"
    echo "Generate it: tail -n +2 echo-record-list.csv | cut -d',' -f4 > manifest.txt"
    exit 1
fi

mkdir -p "$LOGDIR"

# Extract this task's chunk (every NUM_TASKS-th line starting at TASK_ID)
# awk is 0-indexed via (NR-1), manifest lines are 1-indexed
CHUNK_FILE="/tmp/download_chunk_${SLURM_JOB_ID}_${TASK_ID}.txt"
awk -v id="$TASK_ID" -v n="$NUM_TASKS" '(NR - 1) % n == id' "$MANIFEST" > "$CHUNK_FILE"

TOTAL=$(wc -l < "$CHUNK_FILE")
echo "$(date) | Task ${TASK_ID}: ${TOTAL} files to download" | tee -a "$PROGRESS_LOG"

# Download loop
DOWNLOADED=0
SKIPPED=0
ERRORED=0
COUNT=0

while IFS= read -r filepath; do
    COUNT=$((COUNT + 1))

    # Full local path
    LOCAL_FILE="${DEST}/${filepath}"

    # Skip if already exists (fast check before wget)
    if [ -f "$LOCAL_FILE" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Create parent directory
    mkdir -p "$(dirname "$LOCAL_FILE")"

    # Download directly (no recursive crawl)
    wget -q -N -c \
        --user "$PN_USER" --password "$PN_PASS" \
        -O "$LOCAL_FILE" \
        "${BASE_URL}/${filepath}" 2>/dev/null

    if [ $? -eq 0 ]; then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        ERRORED=$((ERRORED + 1))
    fi

    # Progress update every 100 files
    if [ $((COUNT % 100)) -eq 0 ]; then
        echo "$(date) | Task ${TASK_ID}: ${COUNT}/${TOTAL} (dl=${DOWNLOADED} skip=${SKIPPED} err=${ERRORED})" >> "$PROGRESS_LOG"
    fi

done < "$CHUNK_FILE"

# Cleanup temp file
rm -f "$CHUNK_FILE"

# Final summary
cat << SUMMARY | tee -a "$PROGRESS_LOG"
============================================================
Task ${TASK_ID} COMPLETE  $(date)
Total files:  ${TOTAL}
Downloaded:   ${DOWNLOADED}
Skipped:      ${SKIPPED}
Errored:      ${ERRORED}
============================================================
SUMMARY
