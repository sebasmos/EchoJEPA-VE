#!/bin/bash
#SBATCH --job-name=gcs_download_echo
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/sebasmos/gcs_download_echo_%j.log

# ============================================================
# MIMIC-IV-Echo GCS Download (single job, gsutil parallelism)
# ============================================================
# Uses gsutil -m for built-in multi-threaded parallel download.
# Much faster than wget since GCS doesn't rate-limit like the
# PhysioNet web server.
#
# Prerequisites:
#   gcloud auth login --no-launch-browser --project=mit-mri
#   (credentials stored in ~/.config/gcloud/, shared filesystem)
#
# Usage:
#   sbatch scripts/download_gcs.sh
#
# Monitor:
#   tail -f ~/gcs_download_echo_<jobid>.log
#   find /orcd/pool/006/lceli_shared/mimic-iv-echo/files/ -name "*.dcm" | wc -l
# ============================================================

export PATH="$HOME/.local/google-cloud-sdk/bin:$PATH"

DEST="/orcd/pool/006/lceli_shared/mimic-iv-echo/files"
BUCKET="gs://mimic-iv-echo-0.1.physionet.org"
PROJECT="mit-mri"

echo "$(date) | Starting GCS download"
echo "Bucket: ${BUCKET}"
echo "Destination: ${DEST}"
echo "Project: ${PROJECT}"
echo "============================================================"

# Verify auth
gcloud auth list 2>&1
echo "============================================================"

mkdir -p "$DEST"

# gsutil flags:
#   -u PROJECT    : requester-pays billing project
#   -m            : parallel multi-threaded transfer
#   -n            : no-clobber (skip existing files)
#   -r            : recursive
# The parallel_composite_upload_threshold is for uploads, not needed here.
# gsutil automatically handles retries.

gsutil -u "$PROJECT" -m cp -n -r "${BUCKET}/" "$DEST/"

EXIT_CODE=$?

echo "============================================================"
echo "$(date) | GCS download finished with exit code: ${EXIT_CODE}"

# Count files
TOTAL_FILES=$(find "$DEST" -type f | wc -l)
echo "Total files in destination: ${TOTAL_FILES}"
