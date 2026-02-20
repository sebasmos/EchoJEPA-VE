#!/bin/bash
#SBATCH --job-name=download_echo
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/sebasmos/download_echo_%j.log

# ============================================================
# MIMIC-IV-Echo Download Script (wget, resumable)
# ============================================================
# Usage:
#   1. Edit scripts/.env with your PhysioNet credentials:
#
#        PHYSIONET_USER="your_physionet_username"
#        PHYSIONET_PASS="your_physionet_password"
#
#   2. Submit the SLURM job:
#
#        sbatch scripts/download.sh
#
#   3. After a preemption or timeout, simply resubmit (auto-resumes):
#
#        sbatch scripts/download.sh
#
# Example (interactive, for testing on a single node):
#
#   bash scripts/download.sh
#
# wget -c resumes partial files automatically
# wget -N skips files already downloaded with matching timestamp
# ============================================================

DEST="/orcd/pool/006/lceli_shared/mimic-iv-echo"
LOGDIR="${DEST}/.download_metadata"
PROGRESS_LOG="${LOGDIR}/progress.log"

# Load credentials from .env file
# When run via sbatch, BASH_SOURCE points to SLURM's spool copy,
# so we use SLURM_SUBMIT_DIR to find the original scripts/ directory.
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
    echo "Create scripts/.env with:"
    echo '  PHYSIONET_USER="your_username"'
    echo '  PHYSIONET_PASS="your_password"'
    exit 1
fi

USER="${PHYSIONET_USER:?Error: PHYSIONET_USER not set in .env}"
PASS="${PHYSIONET_PASS:?Error: PHYSIONET_PASS not set in .env}"
URL="https://physionet.org/files/mimic-iv-echo/0.1/"

mkdir -p "$LOGDIR"

# Log start
echo "$(date) | Download started (or resumed)" >> "$PROGRESS_LOG"

# Snapshot before
FILES_BEFORE=$(find "${DEST}" -type f -name "*.dcm" 2>/dev/null | wc -l)
SIZE_BEFORE=$(du -sh "${DEST}" 2>/dev/null | cut -f1)
echo "$(date) | Before: ${FILES_BEFORE} DCM files, ${SIZE_BEFORE} total" >> "$PROGRESS_LOG"

# Download with resume
# -r     recursive
# -N     only download newer files (skip already downloaded)
# -c     continue partial downloads
# -np    don't go to parent directory
# -nH    don't create physionet.org/ directory
# --cut-dirs=3  skip files/mimic-iv-echo/0.1/ in path
# -P     save to destination directory
wget -r -N -c -np -nH --cut-dirs=3 \
    --user "$USER" --password "$PASS" \
    -P "$DEST" \
    --progress=dot:mega \
    "$URL" 2>&1

EXIT_CODE=$?

# Snapshot after
FILES_AFTER=$(find "${DEST}" -type f -name "*.dcm" 2>/dev/null | wc -l)
SIZE_AFTER=$(du -sh "${DEST}" 2>/dev/null | cut -f1)
NEW_FILES=$((FILES_AFTER - FILES_BEFORE))

# Log result
cat << SUMMARY | tee -a "$PROGRESS_LOG"
============================================================
RUN COMPLETE  $(date)
Exit code:    ${EXIT_CODE}
Files before: ${FILES_BEFORE} DCM
Files after:  ${FILES_AFTER} DCM
New files:    ${NEW_FILES}
Total size:   ${SIZE_AFTER}
============================================================
SUMMARY

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "$(date) | wget exited with code ${EXIT_CODE}. Resubmit: sbatch download_echo.sh" | tee -a "$PROGRESS_LOG"
fi