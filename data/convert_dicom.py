"""
Convert MIMIC-IV-Echo DICOM files to MP4 videos.

Recursively walks the input directory, converts multi-frame DICOMs to MP4 videos,
and skips single-frame (static) images.

Usage:
    python data/convert_dicom.py --limit 5                    # test with 5 files
    python data/convert_dicom.py --target_size 384            # for ViT-g/16_384
    python data/convert_dicom.py                              # convert all (256x256)
"""

import argparse
import os

import cv2
import numpy as np
import pydicom


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DICOM to MP4")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/data_delete/mimic-iv-echo/physionet.org/files/mimic-iv-echo/0.1/files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/data_delete/mimic-iv-echo-mp4",
    )
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Max files to convert (0 = all)")
    return parser.parse_args()


def process_dicom(file_path, save_path, target_size, fps):
    """Convert a single DICOM to MP4. Returns (status, reason)."""
    ds = pydicom.dcmread(file_path)

    if "PixelData" not in ds:
        return "skipped", "no pixel data"

    pixel_array = ds.pixel_array

    # Single-frame (2D) → skip
    if pixel_array.ndim == 2:
        return "skipped", "single frame"

    num_frames = pixel_array.shape[0]
    if num_frames < 2:
        return "skipped", f"only {num_frames} frame(s)"

    # Normalize to 0-255
    pixel_array = pixel_array.astype(np.float32)
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max - p_min > 0:
        pixel_array = ((pixel_array - p_min) / (p_max - p_min)) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)
    pixel_array = pixel_array.astype(np.uint8)

    # Write MP4
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    size = (target_size, target_size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, size, isColor=True)

    for frame in pixel_array:
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    return "converted", f"{num_frames} frames"


def main():
    args = parse_args()

    # Recursively collect all .dcm files
    dcm_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))
    dcm_files.sort()

    if args.limit > 0:
        dcm_files = dcm_files[: args.limit]

    print(f"Found {len(dcm_files)} DICOM file(s) to process")
    print(f"Target size: {args.target_size}x{args.target_size}")

    converted = 0
    skipped = 0
    errored = 0

    for i, dcm_path in enumerate(dcm_files):
        # Preserve relative path structure: p10/p10036337/s91664836/file.mp4
        rel_path = os.path.relpath(dcm_path, args.input_dir)
        save_path = os.path.join(args.output_dir, rel_path.replace(".dcm", ".mp4"))

        try:
            status, reason = process_dicom(dcm_path, save_path, args.target_size, args.fps)
            if status == "converted":
                converted += 1
                print(f"  [{i+1}/{len(dcm_files)}] OK: {rel_path} ({reason})")
            else:
                skipped += 1
                print(f"  [{i+1}/{len(dcm_files)}] SKIP: {rel_path} ({reason})")
        except Exception as e:
            errored += 1
            print(f"  [{i+1}/{len(dcm_files)}] ERROR: {rel_path} ({e})")

    print(f"\nDone! converted={converted}, skipped={skipped}, errored={errored}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
