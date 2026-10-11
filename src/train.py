"""Training and validation pipeline for Remote Sensing Change Detection."""
import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.models.baseline_unet import BaselineEarlyFusionUNet
from src.models.siamese_unet import SiameseUNet
from src.losses import CompoundLoss
from src.metrics import ChangeDetectionMetrics
from src.utils import get_device, seed_everything, load_config, visualize_change_detection


def train_one_epoch(model, loader, criterion, optimizer, device, is_early_fusion=True):
    model.train()
    running_loss = 0.0
    metrics = ChangeDetectionMetrics()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad()

        if is_early_fusion:
            logits = model(t1, t2)
        else:
            logits = model(t1, t2)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        metrics.update(logits, targets)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(len(loader), 1)
    scores = metrics.compute()
    return epoch_loss, scores


@torch.no_grad()
def validate(model, loader, criterion, device, is_early_fusion=True):
    model.eval()
    running_loss = 0.0
    metrics = ChangeDetectionMetrics()

    sample_to_visualize = None

    pbar = tqdm(loader, desc="Validation", leave=False)
    for i, batch in enumerate(pbar):
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        targets = batch["mask"].to(device)

        if is_early_fusion:
            logits = model(t1, t2)
        else:
            logits = model(t1, t2)

        loss = criterion(logits, targets)
        running_loss += loss.item()
        metrics.update(logits, targets)

        if i == 0 and sample_to_visualize is None:
            # Store first sample for visual snapshot
            probs = torch.sigmoid(logits[0:1])
            sample_to_visualize = (
                batch["raw_t1"][0],
                batch["raw_t2"][0],
                batch["raw_mask"][0],
                probs[0].squeeze().cpu().numpy(),
            )

    epoch_loss = running_loss / max(len(loader), 1)
    scores = metrics.compute()
    return epoch_loss, scores, sample_to_visualize


def run_training(
    config_path="configs/config.yaml",
    save_dir=None,
    epochs=None,
    batch_size=None,
    model_type=None,
    device=None,
    lr=None,
):
    config = load_config(config_path)

    # Apply command-line overrides if provided
    if save_dir:
        config["training"]["save_dir"] = save_dir
    if epochs:
        config["training"]["epochs"] = int(epochs)
    if batch_size:
        config["training"]["batch_size"] = int(batch_size)
    if model_type:
        config["model"]["type"] = model_type
    if device:
        config["training"]["device"] = device
    if lr:
        config["training"]["lr"] = float(lr)

    seed_everything(config["project"]["seed"])

    device = get_device(config["training"]["device"])
    print(f"[Training] Using compute device: {device}")

    # Build Dataloaders
    train_loader, val_loader = get_dataloaders(config)
    print(f"[Dataset] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # Model instantiation
    m_cfg = config["model"]
    model_type = m_cfg.get("type", "baseline_early_fusion")
    is_early_fusion = "baseline" in model_type or "early" in model_type

    if is_early_fusion:
        print(f"[Model] Building Baseline Early Fusion U-Net ({m_cfg['encoder_name']})...")
        model = BaselineEarlyFusionUNet(
            encoder_name=m_cfg["encoder_name"],
            encoder_weights=m_cfg.get("encoder_weights", "imagenet"),
            in_channels=m_cfg.get("in_channels", 6),
            classes=m_cfg.get("classes", 1),
        ).to(device)
    else:
        print(f"[Model] Building Siamese U-Net ({m_cfg['encoder_name']})...")
        model = SiameseUNet(
            encoder_name=m_cfg["encoder_name"],
            encoder_weights=m_cfg.get("encoder_weights", "imagenet"),
            classes=m_cfg.get("classes", 1),
        ).to(device)

    # Loss & Optimizer
    t_cfg = config["training"]
    criterion = CompoundLoss(
        loss_type=t_cfg.get("loss", "bce_dice"),
        bce_weight=t_cfg.get("bce_weight", 0.5),
        dice_weight=t_cfg.get("dice_weight", 0.5),
        focal_gamma=t_cfg.get("focal_gamma", 2.0),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"], eta_min=1e-6)

    # Output directory
    save_dir = t_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    best_iou = 0.0

    epochs = t_cfg["epochs"]
    print(f"\n[Training] Commencing training for {epochs} epochs...")
    print(f"[Output] Checkpoints and logs will be saved to: {os.path.abspath(save_dir)}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_scores = train_one_epoch(model, train_loader, criterion, optimizer, device, is_early_fusion)
        val_loss, val_scores, sample_vis = validate(model, val_loader, criterion, device, is_early_fusion)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_scores['f1']:.4f} | Val IoU: {val_scores['iou']:.4f}"
        )

        # Save latest checkpoint
        latest_path = os.path.join(save_dir, "latest_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_scores": val_scores,
                "config": config,
            },
            latest_path,
        )

        # Save best model
        if val_scores["iou"] > best_iou:
            best_iou = val_scores["iou"]
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_scores": val_scores,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (IoU: {best_iou:.4f})")

        # Save snapshot visualization
        if sample_vis is not None and epoch % 5 == 0:
            raw_t1, raw_t2, raw_gt, pred_p = sample_vis
            vis_path = os.path.join(vis_dir, f"epoch_{epoch:02d}_val_prediction.png")
            visualize_change_detection(
                raw_t1, raw_t2, raw_gt, pred_p,
                title=f"Epoch {epoch} Prediction (IoU: {val_scores['iou']:.3f})",
                save_path=vis_path,
            )

    print(f"\n[Finished] Training complete! Best Validation IoU: {best_iou:.4f}")
    print(f"Final weights stored in: {os.path.join(save_dir, 'best_model.pth')}")


def main():
    parser = argparse.ArgumentParser(description="Train Remote Sensing Change Detection Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--model_type", type=str, default=None, help="Model architecture (baseline_early_fusion or siamese_unet)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        device=args.device,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
