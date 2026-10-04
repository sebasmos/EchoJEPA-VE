#!/usr/bin/env bash
# Download EchoJEPA checkpoints from Google Drive (Alif Munim's shared folder)
# Run inside tmux: tmux new -s echojepa-dl
# Then: bash scripts/download_echojeepa_weights.sh

set -euo pipefail

OUTDIR="/orcd/pool/006/lceli_shared/weights"
LOG="$OUTDIR/download_echojeepa.log"

# File ID → filename
declare -A FILES=(
  ["1y899SZlVL10kGfEPPNXXH4M3aWaP5ujc"]="vitl-scratch-pt-210-c25.pt"
  ["1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB"]="vitl-vmix22m-pt220-c55.pt"
  ["15jVvmlMmaRLI5nspoCRPX3P-KKfI4pLH"]="vjepa21_vitl_mimic_pt100.pt"
  ["1doZsI4tIPHKF7s4pAdE0tnDwdVngIcfU"]="vjepa21_vitl_mimic_pt117.pt"
  ["102kUdwIAv9Sw0QlSyThHQwyrPtVMe9lh"]="vjepa2_1_vitb_mimic_pt169_c60.pt"
)

mkdir -p "$OUTDIR"
echo "[$(date)] Starting EchoJEPA weight downloads" | tee -a "$LOG"

download_one() {
  local fid="$1"
  local fname="$2"
  local dest="$OUTDIR/$fname"

  if [[ -f "$dest" ]]; then
    local size
    size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    echo "[$(date)] SKIP $fname (already exists, ${size} bytes)" | tee -a "$LOG"
    return 0
  fi

  echo "[$(date)] START $fname (id=$fid)" | tee -a "$LOG"
  if python3 -c "import gdown" 2>/dev/null; then
    python3 - <<PYEOF
import gdown, sys
url = "https://drive.google.com/uc?id=${fid}"
out = "${dest}"
try:
    gdown.download(url, out, quiet=False)
    import os
    size = os.path.getsize(out)
    print(f"[OK] ${fname}: {size:,} bytes")
except Exception as e:
    print(f"[FAIL] ${fname}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
  else
    echo "[ERROR] gdown not found. Run: pip install gdown" | tee -a "$LOG"
    return 1
  fi

  echo "[$(date)] DONE $fname" | tee -a "$LOG"
}

export -f download_one
export OUTDIR LOG

# Launch all 5 downloads in parallel
pids=()
for fid in "${!FILES[@]}"; do
  fname="${FILES[$fid]}"
  download_one "$fid" "$fname" &
  pids+=($!)
done

# Wait and report
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    ((failed++)) || true
  fi
done

echo "" | tee -a "$LOG"
echo "[$(date)] === Summary ===" | tee -a "$LOG"
for fid in "${!FILES[@]}"; do
  fname="${FILES[$fid]}"
  dest="$OUTDIR/$fname"
  if [[ -f "$dest" ]]; then
    size=$(stat -c%s "$dest")
    echo "  OK  $fname  ($(numfmt --to=iec $size))" | tee -a "$LOG"
  else
    echo "  FAIL $fname" | tee -a "$LOG"
  fi
done

if [[ $failed -gt 0 ]]; then
  echo "[$(date)] $failed download(s) failed. Check $LOG" | tee -a "$LOG"
  exit 1
fi
echo "[$(date)] All downloads complete." | tee -a "$LOG"
