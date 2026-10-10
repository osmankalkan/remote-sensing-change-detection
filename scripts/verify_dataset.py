"""Visual verification of DualAugmentationDataset transforms.

Verifies:
1. Geometrical transformations (Flips, Rotations) are 100% synchronized between T1, T2, and Mask.
2. Radiometric transformations (Color jitter, brightness, contrast) vary independently between T1 and T2.
3. Plots 5 random samples and saves to visualizations/augmentation_verification.png.
"""
import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets import DualAugmentationDataset, get_transforms


def verify_dataset(data_dir="data/processed", split="train", num_samples=5, save_path="visualizations/augmentation_verification.png"):
    print("=" * 65)
    print("Verifying Dual Augmentation Pipeline Synchronization")
    print(f"Data Dir: {data_dir}/{split}")
    print(f"Samples:  {num_samples}")
    print("=" * 65)

    dataset = DualAugmentationDataset(
        root_dir=data_dir,
        split=split,
        transforms=get_transforms(split),
    )

    if len(dataset) == 0:
        print("[Error] Dataset is empty! Please run patch_images.py first.")
        return

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "T1 (Pre-event + Radiometric 1)",
        "T2 (Post-event + Radiometric 2)",
        "Ground Truth Mask",
        "T2 + Change Overlay",
    ]

    for row_idx, data_idx in enumerate(indices):
        item = dataset[data_idx]
        t1 = item["raw_t1"]
        t2 = item["raw_t2"]
        mask = item["raw_mask"]

        # Create overlay: Red highlight for changed pixels
        overlay = t2.copy()
        change_idx = (mask > 0)
        overlay[change_idx] = [255, 30, 30]  # Highlight red
        blended = cv2.addWeighted(t2, 0.6, overlay, 0.4, 0)

        # Plot T1
        axes[row_idx, 0].imshow(t1)
        axes[row_idx, 0].axis("off")
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=13, weight="bold")

        # Plot T2
        axes[row_idx, 1].imshow(t2)
        axes[row_idx, 1].axis("off")
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=13, weight="bold")

        # Plot Mask
        axes[row_idx, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].axis("off")
        if row_idx == 0:
            axes[row_idx, 2].set_title(col_titles[2], fontsize=13, weight="bold")

        # Plot Overlay
        axes[row_idx, 3].imshow(blended)
        axes[row_idx, 3].axis("off")
        if row_idx == 0:
            axes[row_idx, 3].set_title(col_titles[3], fontsize=13, weight="bold")

        # Row label with filename
        axes[row_idx, 0].text(
            -0.1, 0.5, f"Sample #{row_idx + 1}\n{item['filename']}",
            transform=axes[row_idx, 0].transAxes,
            va="center", ha="right", fontsize=10, weight="semibold",
        )

    plt.suptitle("Albumentations Dual-Target Augmentation Verification", fontsize=16, weight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n[Success] Verification image successfully saved to:\n  -> {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    verify_dataset()
