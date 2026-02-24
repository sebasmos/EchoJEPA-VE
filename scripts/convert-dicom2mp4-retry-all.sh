#!/bin/bash
#SBATCH --job-name=dicom2mp4-fix
#SBATCH --partition=mit_preemptable
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=11,19
#SBATCH --output=/home/sebasmos/dicom2mp4_fix_%A_%a.log

# ============================================================
# DICOM → MP4 conversion (all folders)
# ============================================================
# To run ALL folders:  change --array to 10-19
# To run specific ones: --array=10,11,14  (comma-separated)
#
# Safe to re-run: skip-existing skips already-converted files.
# Uses --workers=1 for reliable signal-based timeout
# (signal.alarm doesn't work in Pool workers for C-level code).
# ============================================================

module load miniforge/24.3.0-0
conda activate vjepa2-312

cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE

BASE_DIR="/orcd/pool/006/lceli_shared/mimic-iv-echo/files/mimic-iv-echo-0.1.physionet.org"
OUTPUT_DIR="/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4"
FOLDER="p${SLURM_ARRAY_TASK_ID}"

echo "$(date) | Starting DICOM→MP4 retry for ${FOLDER}"
echo "Input:   ${BASE_DIR}/${FOLDER}"
echo "Output:  ${OUTPUT_DIR}/${FOLDER}"
echo "Workers: 1 (sequential, reliable timeout)"
echo "Timeout: 600s per file"
echo "Memory:  64G"
echo "============================================================"

PYTHONUNBUFFERED=1 python data/convert_dicom.py \
    --input_dir "${BASE_DIR}/${FOLDER}" \
    --output_dir "${OUTPUT_DIR}/${FOLDER}" \
    --workers 1 \
    --timeout 600

EXIT_CODE=$?

echo "============================================================"
echo "$(date) | ${FOLDER} finished with exit code: ${EXIT_CODE}"
DCM_COUNT=$(find "${BASE_DIR}/${FOLDER}" -name "*.dcm" 2>/dev/null | wc -l)
MP4_COUNT=$(find "${OUTPUT_DIR}/${FOLDER}" -name "*.mp4" 2>/dev/null | wc -l)
echo "DCM files: ${DCM_COUNT} | MP4 files: ${MP4_COUNT}"
if [ "$DCM_COUNT" -eq "$MP4_COUNT" ]; then
    echo "STATUS: COMPLETE"
else
    echo "STATUS: INCOMPLETE ($(( DCM_COUNT - MP4_COUNT )) missing)"
fi
