"""
Extract frozen V-JEPA2 embeddings from echo MP4 videos.

Supports all V-JEPA2 model variants: ViT-L, ViT-H, ViT-g, ViT-g/384.
Loads the encoder with pretrained weights, runs each video through it,
mean-pools the patch tokens to a single embedding vector, and saves all embeddings.

Usage:
    python scripts/extract_embeddings.py --model vitl --limit 5
    python scripts/extract_embeddings.py --model vith
    python scripts/extract_embeddings.py --model vitg
    python scripts/extract_embeddings.py --model vitg-384
"""

import argparse
import os
import sys

import numpy as np
import torch
from decord import VideoReader, cpu

# Add project root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers, vit_huge, vit_large

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEIGHTS_DIR = "/orcd/pool/006/lceli_shared/weights"
EMBEDDINGS_DIR = "/orcd/pool/006/lceli_shared/embeddings"

MODEL_REGISTRY = {
    "vitl": {"constructor": vit_large, "embed_dim": 1024, "img_size": 256},
    "vith": {"constructor": vit_huge, "embed_dim": 1280, "img_size": 256},
    "vitg": {"constructor": vit_giant_xformers, "embed_dim": 1408, "img_size": 256},
    "vitg-384": {"constructor": vit_giant_xformers, "embed_dim": 1408, "img_size": 384},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract V-JEPA2 embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="vitl",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant (default: vitl)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/data_delete/mimic-iv-echo-mp4",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt weights (auto-resolved from --model)")
    parser.add_argument("--output_path", type=str, default=None, help="Output .pt path (auto-resolved from --model)")
    parser.add_argument("--img_size", type=int, default=None, help="Input resolution (auto-resolved from --model)")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    args = parser.parse_args()

    # Auto-resolve defaults from model registry
    cfg = MODEL_REGISTRY[args.model]
    if args.checkpoint is None:
        args.checkpoint = os.path.join(WEIGHTS_DIR, f"{args.model}.pt")
    if args.output_path is None:
        args.output_path = os.path.join(EMBEDDINGS_DIR, f"{args.model}_embeddings.pt")
    if args.img_size is None:
        args.img_size = cfg["img_size"]

    return args


def load_model(model_name, checkpoint_path, img_size, num_frames):
    """Load a V-JEPA2 encoder with pretrained weights."""
    cfg = MODEL_REGISTRY[model_name]
    model = cfg["constructor"](
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=True,
        use_rope=True,
    )

    # Load weights — same pattern as notebooks/vjepa2_demo.py:load_pretrained_vjepa_pt_weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["encoder"]
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"Loaded {model_name} weights: {msg}")

    model.cuda().eval()
    return model


def build_transform(img_size):
    """Eval transform: resize, center crop, normalize (from notebooks/vjepa2_demo.py)."""
    short_side = int(256.0 / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_video(path, num_frames):
    """Load video and uniformly sample num_frames frames."""
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return None
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    video = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    return video


def main():
    args = parse_args()
    cfg = MODEL_REGISTRY[args.model]

    print(f"Model: {args.model} (embed_dim={cfg['embed_dim']}, img_size={args.img_size})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_path}")

    # Collect MP4 files
    mp4_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".mp4"):
                mp4_files.append(os.path.join(root, f))
    mp4_files.sort()

    if args.limit > 0:
        mp4_files = mp4_files[: args.limit]

    if not mp4_files:
        print(f"No MP4 files found in {args.input_dir}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s)")

    # Load model and transform
    model = load_model(args.model, args.checkpoint, args.img_size, args.num_frames)
    transform = build_transform(args.img_size)

    # Extract embeddings
    embeddings = {}
    processed = 0
    errored = 0

    for i, mp4_path in enumerate(mp4_files):
        rel_path = os.path.relpath(mp4_path, args.input_dir)
        try:
            video = load_video(mp4_path, args.num_frames)
            if video is None:
                print(f"  [{i+1}/{len(mp4_files)}] SKIP: {rel_path} (empty video)")
                continue

            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [T, C, H, W]
            video_tensor = transform(video_tensor)  # [C, T, H, W]
            video_tensor = video_tensor.unsqueeze(0).cuda()  # [1, C, T, H, W]

            with torch.inference_mode():
                features = model(video_tensor)  # [1, num_patches, embed_dim]
                embedding = features.mean(dim=1).squeeze(0).cpu()  # [embed_dim]

            embeddings[rel_path] = embedding
            processed += 1
            print(f"  [{i+1}/{len(mp4_files)}] OK: {rel_path} -> {embedding.shape}")

        except Exception as e:
            errored += 1
            print(f"  [{i+1}/{len(mp4_files)}] ERROR: {rel_path} ({e})")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(embeddings, args.output_path)

    # Summary
    print(f"\nDone! processed={processed}, errored={errored}")
    print(f"Saved {len(embeddings)} embeddings to {args.output_path}")
    if embeddings:
        all_embs = torch.stack(list(embeddings.values()))
        print(f"Shape per video: {list(embeddings.values())[0].shape}")
        print(f"Stats: mean={all_embs.mean():.4f}, std={all_embs.std():.4f}")
        print(f"All finite: {torch.isfinite(all_embs).all()}")


if __name__ == "__main__":
    main()
