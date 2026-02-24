# Extract Embeddings

## Setup

```bash
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate vjepa2-312
cd /home/sebasmos/orcd/pool/code/EchoJEPA-VE
```

## Extract (full run — 10-folder array)

```bash
sbatch scripts/extract-embeddings/extract_slurm.sh vitl
```

## Extract (single folder)

```bash
sbatch --array=0 scripts/extract-embeddings/extract_slurm.sh vitl   # p10 only
```

## Extract (interactive test)

```bash
python scripts/extract-embeddings/extract_embeddings.py --model vitl --folder p10 --limit 50 --num_workers 4
```

## Merge per-folder files

```bash
python scripts/extract-embeddings/merge_embeddings.py --model vitl
```

## Monitor

```bash
tail -f ~/jepa-l_<jobid>_*.log
```

## Output

```
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/
├── vitl_embeddings_p10.pt
├── vitl_embeddings_p11.pt
├── ...
├── vitl_embeddings_p19.pt
└── vitl_embeddings_all.pt   (after merge)
```

## Expected Embedding Counts (vitl)

| Folder | MP4 count | Embedding file | Embeddings | Shape |
|--------|-----------|----------------|------------|-------|
| p10 | 51,176 | `vitl_embeddings_p10.pt` | 51,176 | `[1024]` |
| p11 | 51,816 | `vitl_embeddings_p11.pt` | 51,816 | `[1024]` |
| p12 | 49,299 | `vitl_embeddings_p12.pt` | 49,299 | `[1024]` |
| p13 | 59,253 | `vitl_embeddings_p13.pt` | 59,253 | `[1024]` |
| p14 | 49,431 | `vitl_embeddings_p14.pt` | 49,431 | `[1024]` |
| p15 | 56,970 | `vitl_embeddings_p15.pt` | 56,970 | `[1024]` |
| p16 | 49,635 | `vitl_embeddings_p16.pt` | 49,635 | `[1024]` |
| p17 | 54,047 | `vitl_embeddings_p17.pt` | 54,047 | `[1024]` |
| p18 | 50,965 | `vitl_embeddings_p18.pt` | 50,965 | `[1024]` |
| p19 | 52,736 | `vitl_embeddings_p19.pt` | 52,736 | `[1024]` |
| **Total** | **525,328** | `vitl_embeddings_all.pt` | **525,328** | `[1024]` |

## SLURM Configuration

| Parameter | Value |
|-----------|-------|
| Job name | `jepa-l` |
| Partition | `mit_preemptable` |
| GPU | `l40s:1` (48 GB VRAM) |
| CPUs | 16 |
| Memory | 96G |
| Time limit | 2 days |
| Array | 0-9 (p10-p19) |
| Log | `~/jepa-l_%A_%a.log` |

## Extraction Parameters

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

## Model Defaults

| Model | Batch size | Embed dim | Resolution | GPU VRAM |
|-------|-----------|-----------|------------|----------|
| `vitl` | 256 | 1024 | 256x256 | ~20 GB |
| `vith` | 128 | 1280 | 256x256 | ~25 GB |
| `vitg` | 64 | 1408 | 256x256 | ~30 GB |
| `vitg-384` | 16 | 1408 | 384x384 | ~40 GB |

## Merge Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `vitl` | Model to merge |
| `--embeddings_dir` | `/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo` | Folder with per-folder .pt files |
