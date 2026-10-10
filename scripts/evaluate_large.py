import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.baseline_unet import BaselineEarlyFusionUNet
from src.models.siamese_unet import SiameseUNet
from src.utils import get_device, load_config


def create_gaussian_weight(size=256, sigma=64):
    """Generates a 2D Gaussian window to blend overlapping patch borders smoothly."""
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weight / np.max(weight)


def sliding_window_inference(
    model,
    t1_img,
    t2_img,
    device,
    patch_size=256,
    stride=128,
    is_early_fusion=True,
):
    """
    Runs sliding window inference with Gaussian blending to prevent seam edge artifacts.
    """
    model.eval()
    h, w, c = t1_img.shape

    # Pad image if smaller than patch size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h > 0 or pad_w > 0:
        t1_img = cv2.copyMakeBorder(t1_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        t2_img = cv2.copyMakeBorder(t2_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h, w = t1_img.shape[:2]

    # Normalize images
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_t1 = (t1_img / 255.0 - mean) / std
    norm_t2 = (t2_img / 255.0 - mean) / std

    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    gaussian_kernel = create_gaussian_weight(size=patch_size)

    y_steps = list(range(0, h - patch_size + 1, stride))
    if (h - patch_size) % stride != 0 and (h - patch_size) not in y_steps:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, w - patch_size + 1, stride))
    if (w - patch_size) % stride != 0 and (w - patch_size) not in x_steps:
        x_steps.append(w - patch_size)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                patch_t1 = norm_t1[y:y+patch_size, x:x+patch_size]
                patch_t2 = norm_t2[y:y+patch_size, x:x+patch_size]

                t1_tensor = torch.from_numpy(patch_t1.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                t2_tensor = torch.from_numpy(patch_t2.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                logits = model(t1_tensor, t2_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_map[y:y+patch_size, x:x+patch_size] += probs * gaussian_kernel
                weight_map[y:y+patch_size, x:x+patch_size] += gaussian_kernel

    # Safe division
    weight_map[weight_map == 0] = 1.0
    seamless_probs = prob_map / weight_map

    # Remove padding if added
    if pad_h > 0 or pad_w > 0:
        seamless_probs = seamless_probs[:h - pad_h, :w - pad_w]

    return seamless_probs


def main():
    parser = argparse.ArgumentParser(description="Seamless inference on full-sized satellite images.")
    parser.add_argument("--t1", type=str, required=False, default=None, help="Path to T1 image")
    parser.add_argument("--t2", type=str, required=False, default=None, help="Path to T2 image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/large_prediction.png", help="Output path")
    args = parser.parse_args()

    print("Seamless Sliding Window Inference Engine ready.")


if __name__ == "__main__":
    main()
