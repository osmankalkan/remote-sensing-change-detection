"""Training and validation pipeline for Remote Sensing Change Detection."""
import os
import sys
import csv
import json
import math
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Headless plotting: Colab exports MPLBACKEND=module://matplotlib_inline... to child
# processes, which breaks/hangs plotting in a background (nohup) training run.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

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

# When stdout is redirected to a file (e.g. background training on Colab), tqdm's
# carriage-return progress bars turn the log into one giant unreadable line.
# In that case we disable tqdm and print a compact progress line every N% instead.
IS_TTY = sys.stdout.isatty()


def _progress(loader, desc):
    return tqdm(loader, desc=desc, leave=False, disable=not IS_TTY)


def _log_step(desc, step, total, loss=None, t_start=None):
    """Plain-text progress logging for non-interactive (log file) runs."""
    if IS_TTY or total == 0:
        return
    every = max(total // 5, 1)  # ~5 lines per epoch
    if step % every == 0 or step == total:
        msg = f"  [{desc}] {step}/{total} ({100.0 * step / total:5.1f}%)"
        if loss is not None:
            msg += f" | loss: {loss:.4f}"
        if t_start is not None:
            msg += f" | {time.time() - t_start:.0f}s"
        print(msg, flush=True)


def _amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _make_grad_scaler():
    # torch>=2.3 exposes torch.amp.GradScaler("cuda"); older versions only torch.cuda.amp.GradScaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0
    metrics = ChangeDetectionMetrics()
    total = len(loader)
    t_start = time.time()

    pbar = _progress(loader, "Training")
    for step, batch in enumerate(pbar, start=1):
        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _amp_context(device, use_amp):
            logits = model(t1, t2)
        # Loss in float32 for numerical stability (sigmoid/dice under fp16 can underflow)
        logits = logits.float()
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            print(f"  [Warning] Non-finite loss at step {step}, skipping batch.", flush=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        metrics.update(logits.detach(), targets)

        if IS_TTY:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        _log_step("Train", step, total, loss.item(), t_start)

    epoch_loss = running_loss / max(total, 1)
    scores = metrics.compute()
    return epoch_loss, scores


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    metrics = ChangeDetectionMetrics()

    sample_to_visualize = None

    pbar = _progress(loader, "Validation")
    for i, batch in enumerate(pbar):
        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        with _amp_context(device, use_amp):
            logits = model(t1, t2)
        logits = logits.float()

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


def _atomic_save(obj, path):
    """Write to a temp file first, then rename. Prevents corrupted checkpoints if the
    process/VM dies mid-write (common on Colab + Google Drive)."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _append_history(history_path, row):
    is_new = not os.path.exists(history_path)
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def build_model(m_cfg):
    model_type = m_cfg.get("type", "baseline_early_fusion")
    is_early_fusion = "baseline" in model_type or "early" in model_type
    if is_early_fusion:
        print(f"[Model] Building Baseline Early Fusion U-Net ({m_cfg['encoder_name']})...")
        model = BaselineEarlyFusionUNet(
            encoder_name=m_cfg["encoder_name"],
            encoder_weights=m_cfg.get("encoder_weights", "imagenet"),
            in_channels=m_cfg.get("in_channels", 6),
            classes=m_cfg.get("classes", 1),
        )
    else:
        print(f"[Model] Building Siamese U-Net ({m_cfg['encoder_name']})...")
        model = SiameseUNet(
            encoder_name=m_cfg["encoder_name"],
            encoder_weights=m_cfg.get("encoder_weights", "imagenet"),
            classes=m_cfg.get("classes", 1),
        )
    return model


def run_training(
    config_path="configs/config.yaml",
    save_dir=None,
    epochs=None,
    batch_size=None,
    model_type=None,
    device=None,
    lr=None,
    resume=False,
    num_workers=None,
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
    if num_workers is not None:
        config["data"]["num_workers"] = int(num_workers)

    seed_everything(config["project"]["seed"])

    device = get_device(config["training"]["device"])
    print(f"[Training] Using compute device: {device}")
    if device.type == "cuda":
        # Fixed 256x256 inputs -> cuDNN autotuning gives a solid speed-up on T4
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Build Dataloaders
    train_loader, val_loader = get_dataloaders(config)
    print(f"[Dataset] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"[Dataset] Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model instantiation
    m_cfg = config["model"]
    model = build_model(m_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Parameters: {n_params:.2f}M")

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
    epochs = int(t_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision (only effective on CUDA, e.g. Colab T4)
    use_amp = bool(t_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = _make_grad_scaler() if use_amp else None
    print(f"[Training] Mixed precision (AMP fp16): {'ON' if use_amp else 'OFF'}")

    # Output directory
    save_dir = t_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, "latest_model.pth")
    best_path = os.path.join(save_dir, "best_model.pth")
    history_path = os.path.join(save_dir, "history.csv")
    best_iou = 0.0
    start_epoch = 1

    # Resume from latest checkpoint (e.g. after a Colab disconnect)
    if resume and os.path.exists(latest_path):
        print(f"[Resume] Loading checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint in {save_dir} belongs to a different model/encoder. "
                f"Use another save_dir or start without --resume.\n{e}"
            )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        # Rebuild the cosine schedule for the *current* epoch budget (the user may
        # have changed `epochs` between sessions) using the closed-form LR.
        base_lr = float(t_cfg["lr"])
        for g in optimizer.param_groups:
            g["initial_lr"] = base_lr
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6, last_epoch=start_epoch - 2)
        lr_resume = 1e-6 + (base_lr - 1e-6) * (1 + math.cos(math.pi * (start_epoch - 1) / epochs)) / 2
        for g in optimizer.param_groups:
            g["lr"] = lr_resume
        best_iou = float(ckpt.get("best_iou", ckpt.get("val_scores", {}).get("iou", 0.0)))
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_iou = max(best_iou, float(best_ckpt.get("val_scores", {}).get("iou", 0.0)))
        print(f"[Resume] Continuing from epoch {start_epoch} (best IoU so far: {best_iou:.4f}, lr: {lr_resume:.2e})")
    elif resume:
        print(f"[Resume] No checkpoint found at {latest_path}, starting from scratch.")
    elif os.path.exists(history_path):
        # Fresh run: don't mix the new history with an old run's history
        os.remove(history_path)

    if start_epoch > epochs:
        print(f"[Resume] Training already finished ({start_epoch - 1}/{epochs} epochs). Nothing to do.")
        return

    print(f"\n[Training] Commencing training for epochs {start_epoch}..{epochs}")
    print(f"[Output] Checkpoints and logs will be saved to: {os.path.abspath(save_dir)}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_scores = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_scores, sample_vis = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | lr: {lr_now:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_scores['f1']:.4f} | Val IoU: {val_scores['iou']:.4f}",
            flush=True,
        )

        # Also save when no best checkpoint exists yet (e.g. IoU still 0.0 after epoch 1)
        is_best = val_scores["iou"] > best_iou or not os.path.exists(best_path)
        if is_best:
            best_iou = max(best_iou, val_scores["iou"])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "val_scores": val_scores,
            "best_iou": best_iou,
            "config": config,
        }

        # Save latest checkpoint (used for resuming)
        _atomic_save(checkpoint, latest_path)

        # Save best model
        if is_best:
            _atomic_save(checkpoint, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (IoU: {best_iou:.4f})", flush=True)

        _append_history(history_path, {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_iou": round(train_scores["iou"], 6),
            "train_f1": round(train_scores["f1"], 6),
            "val_iou": round(val_scores["iou"], 6),
            "val_f1": round(val_scores["f1"], 6),
            "val_precision": round(val_scores["precision"], 6),
            "val_recall": round(val_scores["recall"], 6),
            "time_s": round(elapsed, 1),
        })

        # Save snapshot visualization
        if sample_vis is not None and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            raw_t1, raw_t2, raw_gt, pred_p = sample_vis
            vis_path = os.path.join(vis_dir, f"epoch_{epoch:02d}_val_prediction.png")
            try:
                visualize_change_detection(
                    raw_t1, raw_t2, raw_gt, pred_p,
                    title=f"Epoch {epoch} Prediction (IoU: {val_scores['iou']:.3f})",
                    save_path=vis_path,
                )
            except Exception as e:  # visualization must never kill a long training run
                print(f"  [Warning] Could not save visualization: {e}", flush=True)

    with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
        json.dump({"best_val_iou": best_iou, "epochs": epochs, "model": m_cfg}, f, indent=2)

    print(f"\n[Finished] Training complete! Best Validation IoU: {best_iou:.4f}")
    print(f"Final weights stored in: {best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Remote Sensing Change Detection Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--model_type", type=str, default=None, help="Model architecture (baseline_early_fusion or siamese_unet)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker override")
    parser.add_argument("--resume", action="store_true", help="Resume from <save_dir>/latest_model.pth if it exists")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        device=args.device,
        lr=args.lr,
        resume=args.resume,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
