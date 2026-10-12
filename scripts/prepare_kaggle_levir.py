"""Organize a downloaded LEVIR-CD(-256) dataset into the layout expected by the project:

    data/processed/{train,val,test}/{A,B,label}/<same_filename>.png

Supported source layouts (any nesting depth, case-insensitive folder names):
  1. <root>/{train,val,test}/{A,B,label}
  2. <root>/{A,B,label} + <root>/list/{train,val,test}.txt
  3. <root>/{A,B,label} only               -> random 80/10/10 split
  4. <root>/{train,test}/... (no val split) -> val carved out of train

Usage:
    python scripts/prepare_kaggle_levir.py --src /content/data_temp --dst data/processed
    python scripts/prepare_kaggle_levir.py --dst data/processed --verify_only
"""
import os
import sys
import random
import shutil
import argparse
from collections import deque

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


def _place(names, src_dirs, dst_root, split, copy):
    """Move/copy every name in `names` from src_dirs[A|B|label] into dst_root/split/{A,B,label}."""
    for sub in ("A", "B", "label"):
        os.makedirs(os.path.join(dst_root, split, sub), exist_ok=True)
    placed = 0
    for name in names:
        a = os.path.join(src_dirs["A"], name)
        b = os.path.join(src_dirs["B"], name)
        if not (os.path.isfile(a) and os.path.isfile(b)):
            continue
        _transfer(a, os.path.join(dst_root, split, "A", name), copy)
        _transfer(b, os.path.join(dst_root, split, "B", name), copy)
        if src_dirs.get("label"):
            lbl = os.path.join(src_dirs["label"], name)
            if not os.path.isfile(lbl):
                # Some releases store labels with a different extension
                stem = os.path.splitext(name)[0]
                alt = [f for f in os.listdir(src_dirs["label"]) if os.path.splitext(f)[0] == stem]
                lbl = os.path.join(src_dirs["label"], alt[0]) if alt else None
            if lbl and os.path.isfile(lbl):
                _transfer(lbl, os.path.join(dst_root, split, "label", name), copy)
        placed += 1
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


def organize(src, dst, val_ratio=0.1, test_ratio=0.1, seed=42, copy=False):
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
            _place(sorted(names[:n_val]), dirs, dst, "val", copy)
            split_names["train"] = (sorted(names[n_val:]), dirs)

        for split, (names, dirs) in split_names.items():
            n = _place(names, dirs, dst, split, copy)
            print(f"[Organize] {split}: {n} samples")
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
            n = _place(names, dirs, dst, split, copy)
            print(f"[Organize] {split}: {n} samples")


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
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of move (keeps source intact)")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        if not args.src or not os.path.isdir(args.src):
            print(f"[Error] Source directory not found: {args.src}")
            sys.exit(1)
        organize(args.src, args.dst, args.val_ratio, args.test_ratio, args.seed, args.copy)

    if not verify(args.dst):
        print("[Error] train/val splits are empty - dataset is NOT ready for training.")
        sys.exit(2)
    print("[OK] Dataset is ready for training.")


if __name__ == "__main__":
    main()
