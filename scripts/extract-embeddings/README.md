# Extract Embeddings

## Table of Contents

1. [Setup](#1-setup)
2. [Extract embeddings (SLURM array)](#2-extract-embeddings)
3. [Merge per-folder .pt files](#3-merge-per-folder-pt-files)
4. [Convert to Parquet (for Hugging Face)](#4-convert-to-parquet)
5. [Monitor & logs](#5-monitor--logs)
6. [Reference tables](#6-reference-tables)

---

## 1. Setup

```bash
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate vjepa2-312
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE
```

## 2. Extract embeddings

Full run (10 folders in parallel via SLURM array):

```bash
sbatch scripts/extract-embeddings/extract_slurm.sh vitl
```

Single folder:

```bash
sbatch --array=0 scripts/extract-embeddings/extract_slurm.sh vitl   # p10 only
```

Interactive test (no GPU allocation needed if already on a GPU node):

```bash
python scripts/extract-embeddings/extract_embeddings.py --model vitl --folder p10 --limit 50 --num_workers 4
```

## 3. Merge per-folder .pt files

Each SLURM array task saves one `.pt` per folder (parallel GPUs can't write to the same file). Merge combines the 10 dicts into one `_all.pt` so you can load all 525K embeddings at once.

```bash
python scripts/extract-embeddings/merge_embeddings.py --model vitl
```

## 4. Convert to Parquet

Converts the 10 `.pt` files + both MIMIC-IV-Echo metadata CSVs into 10 sharded Parquet files. Each row is self-contained with all metadata — no external CSV joins needed.

```bash
python scripts/extract-embeddings/to_parquet.py --model vitl
```

The resulting Parquet dataset is hosted on Hugging Face: [MITCriticalData/mimic-iv-echo-jepa-embeddings](https://huggingface.co/datasets/MITCriticalData/mimic-iv-echo-jepa-embeddings).

To validate the full dataset (525K rows, embedding health checks, PCA, cosine similarity), see [`notebooks/testing_full_data.ipynb`](../../notebooks/testing_full_data.ipynb).

**Parquet columns:**

| Column | Type | Source |
|--------|------|--------|
| `subject_id` | int64 | echo-record-list.csv |
| `study_id` | int64 | echo-record-list.csv |
| `dicom_id` | str | filename w/o extension |
| `file_path` | str | relative MP4 path |
| `acquisition_datetime` | str | echo-record-list.csv (per-video) |
| `study_datetime` | str | echo-study-list.csv (per-study) |
| `note_id` | str (nullable) | echo-study-list.csv |
| `note_seq` | str (nullable) | echo-study-list.csv |
| `note_charttime` | str (nullable) | echo-study-list.csv |
| `embedding` | list\[float32\] (1024) | V-JEPA2 ViT-L embedding |

## 5. Monitor & logs

```bash
tail -f ~/jepa-l_<jobid>_*.log
```

## 6. Reference tables

### Output structure

```
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/
├── vitl_embeddings_p10.pt
├── vitl_embeddings_p11.pt
├── ...
├── vitl_embeddings_p19.pt
├── vitl_embeddings_all.pt                              (after merge)
└── mimic-iv-echo-jepa-embeddings/
    └── jepa-l-embeddings/
        ├── train-00000-of-00010.parquet                (p10)
        ├── train-00001-of-00010.parquet                (p11)
        ├── ...
        └── train-00009-of-00010.parquet                (p19)
```

### Embedding counts (vitl)

| Folder | Videos | Embedding file | Embeddings | Shape | File size | Duration | Status |
|--------|--------|----------------|------------|-------|-----------|----------|--------|
| p10 | 51,176 | `vitl_embeddings_p10.pt` | 51,176 | `[1024]` | 206 MB | ~32 min | DONE |
| p11 | 51,816 | `vitl_embeddings_p11.pt` | 51,816 | `[1024]` | 209 MB | ~33 min | DONE |
| p12 | 49,299 | `vitl_embeddings_p12.pt` | 49,299 | `[1024]` | 199 MB | ~28 min | DONE |
| p13 | 59,253 | `vitl_embeddings_p13.pt` | 59,253 | `[1024]` | 239 MB | ~33 min | DONE |
| p14 | 49,431 | `vitl_embeddings_p14.pt` | 49,431 | `[1024]` | 199 MB | ~29 min | DONE |
| p15 | 56,970 | `vitl_embeddings_p15.pt` | 56,970 | `[1024]` | 229 MB | ~32 min | DONE |
| p16 | 49,635 | `vitl_embeddings_p16.pt` | 49,635 | `[1024]` | 200 MB | ~30 min | DONE |
| p17 | 54,047 | `vitl_embeddings_p17.pt` | 54,047 | `[1024]` | 218 MB | ~32 min | DONE |
| p18 | 50,965 | `vitl_embeddings_p18.pt` | 50,965 | `[1024]` | 205 MB | ~30 min | DONE |
| p19 | 52,736 | `vitl_embeddings_p19.pt` | 52,736 | `[1024]` | 212 MB | ~30 min | DONE |
| **Total** | **525,328** | `vitl_embeddings_all.pt` | **525,328** | `[1024]` | **2.1 GB** | — | **ALL DONE** |

Rate: ~1,600-1,800 videos/min per L40S GPU (batch=256, 8 workers). 0 errors across all folders.

### SLURM configuration

| Parameter | Value |
|-----------|-------|
| Job name | `jepa-l` |
| Partition | `mit_preemptable` |
| GPU | `l40s:1` (48 GB VRAM) |
| CPUs | 16 |
| Memory | 250G |
| Time limit | 2 days |
| Array | 0-9 (p10-p19) |
| Log | `~/jepa-l_%A_%a.log` |

### Extraction parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `vitl` | `vitl`, `vith`, `vitg`, `vitg-384` |
| `--input_dir` | `/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4` | MP4 source |
| `--folder` | None | Single subfolder (e.g. `p10`) |
| `--batch_size` | auto per model | Override GPU batch size |
| `--num_workers` | 8 | DataLoader workers (0 = sequential) |
| `--num_frames` | 16 | Frames sampled per video |
| `--save_every` | 10000 | Checkpoint interval |
| `--limit` | 0 (all) | Max files to process |

### Model defaults

| Model | Batch size | Embed dim | Resolution | GPU VRAM |
|-------|-----------|-----------|------------|----------|
| `vitl` | 256 | 1024 | 256x256 | ~20 GB |
| `vith` | 128 | 1280 | 256x256 | ~25 GB |
| `vitg` | 64 | 1408 | 256x256 | ~30 GB |
| `vitg-384` | 16 | 1408 | 384x384 | ~40 GB |

### Merge parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `vitl` | Model to merge |
| `--embeddings_dir` | `/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo` | Folder with per-folder .pt files |

### Parquet parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `vitl` | Model to convert |
| `--embeddings_dir` | `/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo` | Folder with .pt files |
| `--metadata_dir` | `/orcd/pool/006/lceli_shared/mimic-iv-echo` | Folder with CSV metadata |
