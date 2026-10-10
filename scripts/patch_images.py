"""Patching script: Slices 1024x1024 remote sensing images into 256x256 patches.

Splits: train, val, test
Inputs:
    data/raw/{split}/A/
    data/raw/{split}/B/
    data/raw/{split}/label/
Outputs:
    data/processed/{split}/A/
    data/processed/{split}/B/
    data/processed/{split}/label/
"""
import os
import glob
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm


def patch_single_image(args):
    """
    Slices a single triplet (A, B, label) into non-overlapping patches.
    """
    a_path, b_path, lbl_path, out_dir_a, out_dir_b, out_dir_lbl, patch_size, stride, binarize_to_one = args

    stem = Path(a_path).stem
    ext = Path(a_path).suffix

    # Read images
    img_a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    img_b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    img_lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE) if lbl_path and os.path.exists(lbl_path) else None

    if img_a is None or img_b is None:
        return 0, 0, 0

    h, w = img_a.shape[:2]
    patch_count = 0
    total_change_pixels = 0
    total_pixels = 0

    patch_idx = 0
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            bottom = top + patch_size
            right = left + patch_size

            patch_a = img_a[top:bottom, left:right]
            patch_b = img_b[top:bottom, left:right]

            patch_name = f"{stem}_p{patch_idx:02d}{ext}"
            cv2.imwrite(os.path.join(out_dir_a, patch_name), patch_a)
            cv2.imwrite(os.path.join(out_dir_b, patch_name), patch_b)

            if img_lbl is not None:
                patch_lbl = img_lbl[top:bottom, left:right]
                # Binarize: 0 (no change), 1 (change)
                if binarize_to_one:
                    patch_lbl_binary = (patch_lbl > 128).astype(np.uint8)
                else:
                    patch_lbl_binary = (patch_lbl > 128).astype(np.uint8) * 255
                    
                total_change_pixels += int(np.sum(patch_lbl > 128))
                total_pixels += patch_size * patch_size

                cv2.imwrite(os.path.join(out_dir_lbl, patch_name), patch_lbl_binary)

            patch_count += 1
            patch_idx += 1

    return patch_count, total_change_pixels, total_pixels


def process_split(
    raw_dir: str,
    processed_dir: str,
    split: str,
    patch_size: int = 256,
    stride: int = 256,
    binarize_to_one: bool = True,
    num_workers: int = 4,
):
    """Processes a single dataset split (train, val, or test)."""
    split_raw = os.path.join(raw_dir, split)
    a_dir = os.path.join(split_raw, "A")
    b_dir = os.path.join(split_raw, "B")
    lbl_dir = os.path.join(split_raw, "label")

    if not os.path.exists(a_dir):
        print(f"[{split}] Skipping split: Directory '{a_dir}' does not exist.")
        return

    # Target directories
    out_a = os.path.join(processed_dir, split, "A")
    out_b = os.path.join(processed_dir, split, "B")
    out_lbl = os.path.join(processed_dir, split, "label")
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # Collect images
    image_paths = sorted(glob.glob(os.path.join(a_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]

    if not image_paths:
        print(f"[{split}] No images found in {a_dir}.")
        return

    print(f"\n[{split}] Processing {len(image_paths)} images with patch size {patch_size}x{patch_size}...")

    tasks = []
    for a_path in image_paths:
        filename = os.path.basename(a_path)
        b_path = os.path.join(b_dir, filename)
        lbl_path = os.path.join(lbl_dir, filename)

        if not os.path.exists(b_path):
            print(f"[{split} Warning] Matching B image missing for {filename}, skipping.")
            continue

        tasks.append((
            a_path,
            b_path,
            lbl_path if os.path.exists(lbl_path) else None,
            out_a,
            out_b,
            out_lbl,
            patch_size,
            stride,
            binarize_to_one,
        ))

    total_patches = 0
    total_change_px = 0
    total_px = 0

    start_time = time.time()
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(patch_single_image, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Patching {split}"):
                p_cnt, chg_px, px = f.result()
                total_patches += p_cnt
                total_change_px += chg_px
                total_px += px
    else:
        for t in tqdm(tasks, desc=f"Patching {split}"):
            p_cnt, chg_px, px = patch_single_image(t)
            total_patches += p_cnt
            total_change_px += chg_px
            total_px += px

    elapsed = time.time() - start_time
    change_ratio = (total_change_px / total_px * 100) if total_px > 0 else 0.0

    print(f"[{split} Done] Created {total_patches} patches in {elapsed:.2f}s.")
    print(f"[{split} Stats] Changed pixel ratio: {change_ratio:.2f}% ({total_change_px:,} / {total_px:,} px)")


def main():
    parser = argparse.ArgumentParser(description="Slice 1024x1024 remote sensing images into 256x256 patches.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed output directory")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size (default: 256)")
    parser.add_argument("--stride", type=int, default=256, help="Stride size (default: 256, non-overlapping)")
    parser.add_argument("--binarize_to_one", action="store_true", default=True, help="Save labels as 0/1 binary instead of 0/255")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process")

    args = parser.parse_args()

    print("=" * 60)
    print("LEVIR-CD Image Patching Pipeline")
    print(f"Raw dir:       {args.raw_dir}")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Patch size:    {args.patch_size}x{args.patch_size} (stride: {args.stride})")
    print(f"Binarize:      {'0/1' if args.binarize_to_one else '0/255'}")
    print(f"Workers:       {args.workers}")
    print("=" * 60)

    for split in args.splits:
        process_split(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            split=split,
            patch_size=args.patch_size,
            stride=args.stride,
            binarize_to_one=args.binarize_to_one,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
