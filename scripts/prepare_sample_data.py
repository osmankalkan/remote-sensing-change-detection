"""Data preparation and synthetic sample generator for smoke testing."""
import os
import argparse
import numpy as np
import cv2


def generate_synthetic_remote_sensing_pair(size=(1024, 1024)):
    """
    Generates a realistic synthetic bitemporal satellite image pair (T1, T2)
    and binary change mask (0: no change, 255: change).
    Simulates:
    - Landscape background (fields, terrain)
    - Roads and trees
    - Buildings present in both T1 and T2
    - Newly constructed buildings in T2 (positive change)
    - Light / seasonal illumination differences
    """
    h, w = size

    # Background terrain (vegetation / soil)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # Add natural texture noise
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
    terrain_color = np.array([70, 110, 60], dtype=np.float32) # Olive green
    base = np.clip(terrain_color + noise, 0, 255).astype(np.uint8)

    # Road network
    cv2.line(base, (0, h // 3), (w, h // 3), (120, 120, 120), 16)
    cv2.line(base, (w // 2, 0), (w // 2, h), (130, 130, 130), 14)
    cv2.line(base, (0, int(h * 0.75)), (w, int(h * 0.75)), (115, 115, 115), 12)

    # Pre-event image T1
    t1 = base.copy()
    # Post-event image T2 (seasonal shift: slightly brighter / warmer)
    t2 = np.clip(base.astype(np.float32) * 1.08 + 10, 0, 255).astype(np.uint8)

    # Ground truth change mask (0: no change, 255: change)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Existing buildings (present in both T1 and T2 -> NO change)
    for _ in range(12):
        bx = np.random.randint(50, w - 120)
        by = np.random.randint(50, h - 120)
        bw = np.random.randint(40, 80)
        bh = np.random.randint(40, 80)
        color = (np.random.randint(180, 220), np.random.randint(160, 200), np.random.randint(150, 190))
        cv2.rectangle(t1, (bx, by), (bx + bw, by + bh), color, -1)
        # Slight shadow or weathering in T2
        color_t2 = (int(color[0] * 0.95), int(color[1] * 0.95), int(color[2] * 0.95))
        cv2.rectangle(t2, (bx, by), (bx + bw, by + bh), color_t2, -1)

    # Newly constructed buildings in T2 (CHANGE: only present in T2)
    num_new_buildings = np.random.randint(5, 10)
    for _ in range(num_new_buildings):
        bx = np.random.randint(50, w - 100)
        by = np.random.randint(50, h - 100)
        bw = np.random.randint(35, 75)
        bh = np.random.randint(35, 75)

        # Construction in T2: bright concrete / roof colors
        new_color = (np.random.randint(210, 245), np.random.randint(210, 240), np.random.randint(220, 250))
        cv2.rectangle(t2, (bx, by), (bx + bw, by + bh), new_color, -1)
        # Add to change mask
        cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

    return t1, t2, mask


def prepare_sample_dataset(raw_dir="data/raw", num_samples={"train": 4, "val": 2, "test": 2}):
    """Generates synthetic LEVIR-CD format samples for local verification and smoke tests."""
    print(f"[Dataset Generator] Generating sample 1024x1024 dataset into {raw_dir}...")
    for split, count in num_samples.items():
        a_dir = os.path.join(raw_dir, split, "A")
        b_dir = os.path.join(raw_dir, split, "B")
        lbl_dir = os.path.join(raw_dir, split, "label")

        os.makedirs(a_dir, exist_ok=True)
        os.makedirs(b_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(1, count + 1):
            filename = f"sample_{split}_{i:03d}.png"
            t1, t2, mask = generate_synthetic_remote_sensing_pair((1024, 1024))
            cv2.imwrite(os.path.join(a_dir, filename), cv2.cvtColor(t1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(b_dir, filename), cv2.cvtColor(t2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(lbl_dir, filename), mask)
            print(f"  [{split}] Created pair: {filename}")

    print("[Dataset Generator] Sample dataset created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate sample LEVIR-CD pairs for rapid testing.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Target raw directory")
    parser.add_argument("--train_count", type=int, default=4, help="Number of train samples")
    parser.add_argument("--val_count", type=int, default=2, help="Number of val samples")
    parser.add_argument("--test_count", type=int, default=2, help="Number of test samples")
    args = parser.parse_args()

    prepare_sample_dataset(
        raw_dir=args.raw_dir,
        num_samples={"train": args.train_count, "val": args.val_count, "test": args.test_count},
    )


if __name__ == "__main__":
    main()
