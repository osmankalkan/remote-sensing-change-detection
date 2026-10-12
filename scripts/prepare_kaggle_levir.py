"""Organize a downloaded LEVIR-CD(-256) dataset into the layout expected by the project:

    data/processed/{train,val,test}/{A,B,label}/<same_filename>.png

Supported sources:
  1. Direct Hugging Face download (--download_hf) from 'satellite-image-deep-learning/LEVIR-CD'
     (completely public, NO TOKEN or API key required).
  2. Any local directory or Kaggle dataset extraction:
     - <root>/{train,val,test}/{A,B,label}
     - <root>/{A,B,label} + <root>/list/{train,val,test}.txt
     - <root>/{A,B,label} only               -> random 80/10/10 split
     - <root>/{train,test}/... (no val split) -> val carved out of train

Auto-Slicing:
  If input images are 1024x1024 (standard LEVIR-CD), they are automatically sliced
  into 256x256 non-overlapping patches (16 patches per image).
  If input images are already 256x256, they are copied/moved directly without re-slicing.

Usage:
    python scripts/prepare_kaggle_levir.py --download_hf --dst data/processed
    python scripts/prepare_kaggle_levir.py --src /content/data_temp --dst data/processed
    python scripts/prepare_kaggle_levir.py --dst data/processed --verify_only
"""
import os
import sys
import random
import shutil
import argparse
import urllib.request
import zipfile
from collections import deque
import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SPLITS = ("train", "val", "test")

SPLIT_ALIASES = {
    "train": ["train", "training", "trn"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test", "testing", "tst"],
}
SUB_ALIASES = {
    "A": ["a", "t1", "im1", "time1", "pre", "image1", "img1"],
    "B": ["b", "t2", "im2", "time2", "post", "image2", "img2"],
    "label": ["label", "labels", "mask", "masks", "gt", "out", "cd", "change"],
}

HF_URLS = {
    "train": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/train.zip",
    "val": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/val.zip",
    "test": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/test.zip",
}


def _subdirs(path):
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith((".", "__"))]
    except FileNotFoundError:
        return []


def _find_child(parent, aliases):
    """Case-insensitive lookup of a child directory matching one of the aliases."""
    lookup = {d.lower(): d for d in _subdirs(parent)}
    for alias in aliases:
        if alias in lookup:
            return os.path.join(parent, lookup[alias])
    return None


def _images(path):
    return sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXT) and not f.startswith("."))


def _has_ab(path):
    return _find_child(path, SUB_ALIASES["A"]) and _find_child(path, SUB_ALIASES["B"])


def _bfs(root, max_depth=6):
    queue = deque([(root, 0)])
    while queue:
        path, depth = queue.popleft()
        yield path
        if depth < max_depth:
            for d in sorted(_subdirs(path)):
                queue.append((os.path.join(path, d), depth + 1))


def detect_layout(src):
    """Returns ('splits', {split: dir}) or ('pool', dir)."""
    for path in _bfs(src):
        found = {}
        for split, aliases in SPLIT_ALIASES.items():
            cand = _find_child(path, aliases)
            if cand and _has_ab(cand):
                found[split] = cand
        if "train" in found:
            return "splits", found
    for path in _bfs(src):
        if _has_ab(path):
            return "pool", path
    return None, None


def _transfer(src_file, dst_file, copy):
    if os.path.exists(dst_file):
        return
    if copy:
        shutil.copy2(src_file, dst_file)
    else:
        shutil.move(src_file, dst_file)


def _slice_or_transfer(a_path, b_path, lbl_path, dst_root, split, name, copy, patch_size=256):
    """If image is 256x256, moves/copies directly. If >256x256 (e.g. 1024x1024), slices into patches."""
    out_a = os.path.join(dst_root, split, "A")
    out_b = os.path.join(dst_root, split, "B")
    out_lbl = os.path.join(dst_root, split, "label")

    img_a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    if img_a is None:
        return 0
    h, w = img_a.shape[:2]

    if h <= patch_size and w <= patch_size:
        _transfer(a_path, os.path.join(out_a, name), copy)
        _transfer(b_path, os.path.join(out_b, name), copy)
        if lbl_path and os.path.isfile(lbl_path):
            _transfer(lbl_path, os.path.join(out_lbl, name), copy)
        return 1

    img_b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    img_lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE) if (lbl_path and os.path.isfile(lbl_path)) else None
    stem = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1] or ".png"

    count = 0
    patch_idx = 0
    for top in range(0, h - patch_size + 1, patch_size):
        for left in range(0, w - patch_size + 1, patch_size):
            p_name = f"{stem}_p{patch_idx:02d}{ext}"
            cv2.imwrite(os.path.join(out_a, p_name), img_a[top:top+patch_size, left:left+patch_size])
            if img_b is not None:
                cv2.imwrite(os.path.join(out_b, p_name), img_b[top:top+patch_size, left:left+patch_size])
            if img_lbl is not None:
                p_lbl = img_lbl[top:top+patch_size, left:left+patch_size]
                p_lbl_bin = (p_lbl > 127).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(out_lbl, p_name), p_lbl_bin)
            count += 1
            patch_idx += 1
    return count


def _place(names, src_dirs, dst_root, split, copy, patch_size=256):
    """Move/copy every name in `names` from src_dirs[A|B|label] into dst_root/split/{A,B,label}."""
    for sub in ("A", "B", "label"):
        os.makedirs(os.path.join(dst_root, split, sub), exist_ok=True)
    placed = 0
    for name in names:
        a = os.path.join(src_dirs["A"], name)
        b = os.path.join(src_dirs["B"], name)
        if not (os.path.isfile(a) and os.path.isfile(b)):
            continue
        lbl = None
        if src_dirs.get("label"):
            lbl_candidate = os.path.join(src_dirs["label"], name)
            if not os.path.isfile(lbl_candidate):
                stem = os.path.splitext(name)[0]
                alt = [f for f in os.listdir(src_dirs["label"]) if os.path.splitext(f)[0] == stem]
                lbl_candidate = os.path.join(src_dirs["label"], alt[0]) if alt else None
            if lbl_candidate and os.path.isfile(lbl_candidate):
                lbl = lbl_candidate
        placed += _slice_or_transfer(a, b, lbl, dst_root, split, name, copy, patch_size=patch_size)
    return placed


def _sub_dirs_of(path):
    return {
        "A": _find_child(path, SUB_ALIASES["A"]),
        "B": _find_child(path, SUB_ALIASES["B"]),
        "label": _find_child(path, SUB_ALIASES["label"]),
    }


def _read_split_lists(pool_dir):
    """Looks for list/{train,val,test}.txt next to (or above) the A/B folders."""
    for base in (pool_dir, os.path.dirname(pool_dir)):
        list_dir = _find_child(base, ["list", "lists", "splits", "split"])
        candidates = [list_dir] if list_dir else []
        candidates.append(base)
        for d in candidates:
            lists = {}
            for split, aliases in SPLIT_ALIASES.items():
                for alias in aliases:
                    p = os.path.join(d, alias + ".txt")
                    if os.path.isfile(p):
                        with open(p) as f:
                            lists[split] = [os.path.basename(l.strip()) for l in f if l.strip()]
                        break
            if "train" in lists:
                return lists
    return None


def organize(src, dst, val_ratio=0.1, test_ratio=0.1, seed=42, copy=False, patch_size=256):
    kind, info = detect_layout(src)
    if kind is None:
        raise RuntimeError(
            f"Could not find A/B image folders anywhere under '{src}'.\n"
            f"Top-level content: {os.listdir(src) if os.path.isdir(src) else 'MISSING'}"
        )

    rng = random.Random(seed)
    os.makedirs(dst, exist_ok=True)

    if kind == "splits":
        print(f"[Layout] Split folders detected: { {k: os.path.relpath(v, src) for k, v in info.items()} }")
        split_names = {}
        for split, path in info.items():
            dirs = _sub_dirs_of(path)
            split_names[split] = (_images(dirs["A"]), dirs)

        if "val" not in split_names:
            names, dirs = split_names["train"]
            names = names[:]
            rng.shuffle(names)
            n_val = max(1, int(len(names) * val_ratio))
            print(f"[Layout] No 'val' split found -> moving {n_val} random train samples to val.")
            _place(sorted(names[:n_val]), dirs, dst, "val", copy, patch_size=patch_size)
            split_names["train"] = (sorted(names[n_val:]), dirs)

        for split, (names, dirs) in split_names.items():
            n = _place(names, dirs, dst, split, copy, patch_size=patch_size)
            print(f"[Organize] {split}: {n} patches/samples placed")
    else:
        dirs = _sub_dirs_of(info)
        print(f"[Layout] Single pool detected at '{os.path.relpath(info, src)}'")
        lists = _read_split_lists(info)
        if lists:
            print(f"[Layout] Using split lists: { {k: len(v) for k, v in lists.items()} }")
            if "val" not in lists:
                train = lists["train"][:]
                rng.shuffle(train)
                n_val = max(1, int(len(train) * val_ratio))
                lists["val"], lists["train"] = sorted(train[:n_val]), sorted(train[n_val:])
        else:
            names = _images(dirs["A"])
            rng.shuffle(names)
            n = len(names)
            n_test, n_val = int(n * test_ratio), int(n * val_ratio)
            lists = {
                "test": sorted(names[:n_test]),
                "val": sorted(names[n_test:n_test + n_val]),
                "train": sorted(names[n_test + n_val:]),
            }
            print(f"[Layout] No split lists -> random split {len(lists['train'])}/{len(lists['val'])}/{len(lists['test'])}")
        for split, names in lists.items():
            n = _place(names, dirs, dst, split, copy, patch_size=patch_size)
            print(f"[Organize] {split}: {n} patches/samples placed")


def download_from_huggingface(dest_dir):
    """Directly downloads LEVIR-CD train, val, and test zips from Hugging Face with no tokens needed."""
    os.makedirs(dest_dir, exist_ok=True)
    print("=" * 65)
    print("🚀 HUGGING FACE DIRECT DOWNLOAD (NO TOKEN / LOGIN REQUIRED)")
    print("   Source: satellite-image-deep-learning/LEVIR-CD")
    print("=" * 65)

    for split, url in HF_URLS.items():
        zpath = os.path.join(dest_dir, f"{split}.zip")
        print(f"\n📥 Downloading {split}.zip ...")
        
        # Simple progress reporter
        last_pct = [-1]
        def _report(block_num, block_size, total_size):
            if total_size > 0:
                pct = int(block_num * block_size * 100 / total_size)
                if pct % 20 == 0 and pct != last_pct[0]:
                    print(f"   [{split}.zip] {pct}% ({block_num * block_size / 1024**2:.1f} / {total_size / 1024**2:.1f} MB)", flush=True)
                    last_pct[0] = pct

        urllib.request.urlretrieve(url, zpath, reporthook=_report)
        print(f"📦 Extracting {split}.zip ...", flush=True)
        split_dir = os.path.join(dest_dir, split)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(split_dir)
        os.remove(zpath)
        print(f"✅ Extracted {split} successfully.", flush=True)

    print("\n✅ All splits downloaded and extracted from Hugging Face!")
    return dest_dir


def verify(dst):
    """Prints a report; returns True if train & val are usable."""
    print("=" * 65)
    print(f"DATASET REPORT ({dst})")
    print("=" * 65)
    ok = True
    for split in SPLITS:
        counts = {}
        for sub in ("A", "B", "label"):
            p = os.path.join(dst, split, sub)
            counts[sub] = set(_images(p)) if os.path.isdir(p) else set()
        n_a, n_b, n_l = (len(counts[s]) for s in ("A", "B", "label"))
        paired = len(counts["A"] & counts["B"] & counts["label"])
        status = "OK" if n_a > 0 and n_a == n_b == n_l == paired else ("EMPTY" if n_a == 0 else "MISMATCH")
        print(f"[{split.upper():5s}] A: {n_a:6d} | B: {n_b:6d} | label: {n_l:6d} | complete triplets: {paired:6d}  -> {status}")
        if split in ("train", "val") and paired == 0:
            ok = False
    print("=" * 65)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Organize LEVIR-CD into data/processed/{split}/{A,B,label}")
    parser.add_argument("--src", type=str, default=None, help="Extracted download directory")
    parser.add_argument("--dst", type=str, default="data/processed", help="Target processed directory")
    parser.add_argument("--download_hf", action="store_true", help="Download directly from Hugging Face (no token required)")
    parser.add_argument("--patch_size", type=int, default=256, help="Target patch size (default 256)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        src = args.src
        if args.download_hf:
            download_dir = src or "/content/data_temp"
            download_from_huggingface(download_dir)
            src = download_dir

        if not src or not os.path.isdir(src):
            print(f"[Error] Source directory not found: {src}")
            sys.exit(1)
        organize(src, args.dst, args.val_ratio, args.test_ratio, args.seed, args.copy, patch_size=args.patch_size)

    if not verify(args.dst):
        print("[Error] train/val splits are empty - dataset is NOT ready for training.")
        sys.exit(2)
    print("[OK] Dataset is ready for training.")


if __name__ == "__main__":
    main()
