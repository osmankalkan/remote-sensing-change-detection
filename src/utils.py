"""Utility functions for Remote Sensing Change Detection."""
import os
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_device(preferred="auto"):
    """
    Select computing device (MPS for Apple Silicon, CUDA for Nvidia GPU, or CPU).
    
    Args:
        preferred (str): 'auto', 'mps', 'cuda', or 'cpu'.
    
    Returns:
        torch.device: The selected PyTorch device.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Using Apple Silicon MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            print("[Device] Using CPU")
    elif preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Using Apple Silicon MPS")
        else:
            print("[Device Warning] MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Device Warning] CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def seed_everything(seed=42):
    """Seed random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path="configs/config.yaml"):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def visualize_change_detection(
    t1_img,
    t2_img,
    gt_mask,
    pred_mask=None,
    title="Change Detection Sample",
    save_path=None,
    show=False,
):
    """
    Visualizes T1, T2, Ground Truth mask, and optional Model Prediction.
    
    Args:
        t1_img: np.ndarray or torch.Tensor (H, W, 3) or (3, H, W) normalized [0, 1] or [0, 255]
        t2_img: np.ndarray or torch.Tensor (H, W, 3) or (3, H, W)
        gt_mask: np.ndarray or torch.Tensor (H, W) or (1, H, W) binary (0 or 1)
        pred_mask: Optional predicted binary or probability mask
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to call plt.show()
    """
    def _to_numpy_img(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in [1, 3]:  # Channels-first
            x = np.transpose(x, (1, 2, 0))
        if x.max() <= 1.0:
            x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def _to_numpy_mask(m):
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze()
        return (m > 0.5).astype(np.uint8)

    t1_np = _to_numpy_img(t1_img)
    t2_np = _to_numpy_img(t2_img)
    gt_np = _to_numpy_mask(gt_mask)

    ncols = 4 if pred_mask is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(t1_np)
    axes[0].set_title("T1 (Pre-event)")
    axes[0].axis("off")

    axes[1].imshow(t2_np)
    axes[1].set_title("T2 (Post-event)")
    axes[1].axis("off")

    axes[2].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    if pred_mask is not None:
        p_np = _to_numpy_mask(pred_mask)
        axes[3].imshow(p_np, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Predicted Mask")
        axes[3].axis("off")

    plt.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] Saved plot to {save_path}")

    if show:
        plt.show()
    plt.close(fig)
