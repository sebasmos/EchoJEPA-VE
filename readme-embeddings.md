# Embedding Extraction Guide

Extract frozen V-JEPA2 / EchoJEPA embeddings from MIMIC-IV-Echo (or any DICOM echo dataset).

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Download Weights](#2-download-weights)
3. [Available Models](#3-available-models)
4. [Stage 1: Convert DICOM to MP4](#4-stage-1-convert-dicom-to-mp4)
5. [Stage 2: Extract Embeddings](#5-stage-2-extract-embeddings)
6. [Stage 3: Inspect & Visualize](#6-stage-3-inspect--visualize)
7. [Quick Test (5 files)](#7-quick-test-5-files)
8. [Full Run](#8-full-run)
9. [Output Format](#9-output-format)

---

## 1. Prerequisites

```bash
module load miniforge/24.3.0-0
conda activate vjepa2-312
pip install pydicom   # if not already installed
```

**Credentials:** If downloading MIMIC-IV-Echo from PhysioNet, copy the credentials template and fill in your details:

```bash
cp scripts/.env.example scripts/.env
# Edit scripts/.env with your PhysioNet username and password
```

## 2. Download Weights

Download V-JEPA2 pretrained checkpoints to the shared weights directory:

```bash
cd /orcd/pool/006/lceli_shared/weights/

wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt       # ViT-L   (4.8 GB)
wget https://dl.fbaipublicfiles.com/vjepa2/vith.pt       # ViT-H   (9.7 GB)
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt       # ViT-g   (16 GB)
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt   # ViT-g/384 (16 GB)
```

## 3. Available Models

| Model | Constructor | Parameters | embed_dim | Resolution | Output Shape |
|---|---|---|---|---|---|
| `vitl` | `vit_large` | 300M | 1024 | 256x256 | `[1024]` |
| `vith` | `vit_huge` | 600M | 1280 | 256x256 | `[1280]` |
| `vitg` | `vit_giant_xformers` | 1B | 1408 | 256x256 | `[1408]` |
| `vitg-384` | `vit_giant_xformers` | 1B | 1408 | 384x384 | `[1408]` |

All models use: `patch_size=16, tubelet_size=2, num_frames=16, uniform_power=True, use_rope=True`

## 4. Stage 1: Convert DICOM to MP4

The conversion script recursively walks the input directory, converts multi-frame DICOMs to MP4, and skips single-frame static images.

```bash
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE

# Test with 5 files (256x256, default for vitl/vith/vitg)
python data/convert_dicom.py --limit 5

# Convert all files at 256x256
python data/convert_dicom.py

# Convert at 384x384 (for vitg-384)
python data/convert_dicom.py --target_size 384 \
    --output_dir /orcd/pool/006/lceli_shared/data_delete/mimic-iv-echo-mp4-384
```

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | MIMIC-IV-Echo DICOM path | Input directory with .dcm files |
| `--output_dir` | mimic-iv-echo-mp4 | Output directory for MP4 files |
| `--target_size` | 256 | Resize resolution (256 or 384) |
| `--fps` | 30 | Output video frame rate |
| `--limit` | 0 (all) | Max files to convert |

## 5. Stage 2: Extract Embeddings

The extraction script loads a frozen V-JEPA2 encoder, processes each MP4 video, and saves mean-pooled embeddings. Uses parallel DataLoader for fast GPU utilization.

All extraction scripts live in [`scripts/extract-embeddings/`](scripts/extract-embeddings/). See [`scripts/extract-embeddings/README.md`](scripts/extract-embeddings/README.md) for detailed SLURM configuration.

```bash
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE

# Single folder (interactive)
python scripts/extract-embeddings/extract_embeddings.py --model vitl --folder p10 --num_workers 4

# Full run via SLURM (10-folder array, 1 L40S GPU per folder)
sbatch scripts/extract-embeddings/extract_slurm.sh vitl

# Single folder via SLURM
sbatch --array=0 scripts/extract-embeddings/extract_slurm.sh vitl   # p10 only

# Merge per-folder files into one
python scripts/extract-embeddings/merge_embeddings.py --model vitl
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `vitl` | Model variant: vitl, vith, vitg, vitg-384 |
| `--input_dir` | `/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4` | Directory with MP4 files |
| `--folder` | None | Single subfolder (e.g. `p10`) for array jobs |
| `--checkpoint` | auto from `--model` | Path to .pt weights |
| `--output_path` | auto from `--model` | Output .pt path |
| `--batch_size` | auto per model | Override GPU batch size |
| `--num_workers` | 8 | DataLoader workers (0 = sequential) |
| `--num_frames` | 16 | Frames to sample per video |
| `--save_every` | 10000 | Checkpoint interval |
| `--limit` | 0 (all) | Max files to process |

## 6. Stage 3: Inspect & Visualize

After extraction, open the inspection notebook to verify results:

**[`notebooks/inspect_embeddings.ipynb`](notebooks/inspect_embeddings.ipynb)**

The notebook provides:
- Metadata: file sizes, num videos, embed_dim, dtypes
- Health checks: NaN, Inf, all-zero, dead dimensions
- Statistics: mean, std, min, max, L2 norms
- Cosine similarity heatmap
- Embedding value distributions and per-dimension variance
- PCA projection (2D scatter + explained variance)
- Per-video embedding profile plots

Sample embeddings (5 videos, ViT-L) are included in `data/sample/` so the notebook runs out of the box.

## 7. Quick Test (5 files)

```bash
module load miniforge/24.3.0-0
conda activate vjepa2-312
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE

# Stage 1: Convert 5 DICOMs
python data/convert_dicom.py --limit 5

# Stage 2: Extract ViT-L embeddings
python scripts/extract-embeddings/extract_embeddings.py --model vitl --limit 5 --num_workers 0

# Stage 3: Open notebook to inspect
jupyter notebook notebooks/inspect_embeddings.ipynb
```

## 8. Full Run

```bash
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE

# Stage 1: Convert all DICOMs (run once)
python data/convert_dicom.py

# Stage 2: Extract embeddings (10-folder SLURM array)
sbatch scripts/extract-embeddings/extract_slurm.sh vitl

# Stage 2b: Merge per-folder outputs
python scripts/extract-embeddings/merge_embeddings.py --model vitl

# For other models:
sbatch scripts/extract-embeddings/extract_slurm.sh vith
sbatch scripts/extract-embeddings/extract_slurm.sh vitg

# For vitg-384, need 384px MP4s first
python data/convert_dicom.py --target_size 384 \
    --output_dir /orcd/pool/006/lceli_shared/data_delete/mimic-iv-echo-mp4-384
sbatch scripts/extract-embeddings/extract_slurm.sh vitg-384
```

## 9. Output Format

Embeddings are saved per folder as Python dicts: `{relative_path: tensor}`.

```
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/
├── vitl_embeddings_p10.pt   (~210 MB, ~51K embeddings)
├── vitl_embeddings_p11.pt
├── ...
├── vitl_embeddings_p19.pt
└── vitl_embeddings_all.pt   (merged, ~2.1 GB)
```

```python
import torch

# Load single folder
embeddings = torch.load("/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/vitl_embeddings_p10.pt")

# Load merged file (all folders)
embeddings = torch.load("/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/vitl_embeddings_all.pt")

# Iterate
for filename, emb in embeddings.items():
    print(filename, emb.shape)  # e.g. "p10036337/s91664836/file.mp4", torch.Size([1024])

# Stack into a matrix
all_embs = torch.stack(list(embeddings.values()))  # [N, embed_dim]
```

| Model | Per-folder file | Merged file | Shape per video |
|---|---|---|---|
| vitl | `vitl_embeddings_p{10..19}.pt` | `vitl_embeddings_all.pt` | `[1024]` |
| vith | `vith_embeddings_p{10..19}.pt` | `vith_embeddings_all.pt` | `[1280]` |
| vitg | `vitg_embeddings_p{10..19}.pt` | `vitg_embeddings_all.pt` | `[1408]` |
| vitg-384 | `vitg-384_embeddings_p{10..19}.pt` | `vitg-384_embeddings_all.pt` | `[1408]` |
