"""Day 4 Smoke Test: Forward & Backward pass on Apple Silicon MPS device.

Verifies:
1. Baseline Early Fusion U-Net (ResNet18 backbone, 6-channel input).
2. PyTorch MPS device tensor transfer.
3. Forward pass [B, 6, 256, 256] -> [B, 1, 256, 256].
4. Compound Loss calculation (BCE + Dice).
5. Backward pass (autograd gradients on MPS).
6. Optimizer step update.
7. Siamese U-Net late-fusion forward/backward pass.
"""
import os
import sys
import time
import torch
import torch.nn as nn

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.baseline_unet import BaselineEarlyFusionUNet
from src.models.siamese_unet import SiameseUNet
from src.losses import CompoundLoss
from src.metrics import ChangeDetectionMetrics
from src.utils import get_device


def run_smoke_test(batch_size=2, height=256, width=256):
    print("=" * 65)
    print("Remote Sensing Change Detection: MPS Smoke Test (Day 4)")
    print("=" * 65)

    device = get_device("mps")
    print(f"Target Device: {device}")

    # 1. Instantiate Early Fusion Baseline
    print("\n[Step 1] Initializing Baseline Early Fusion U-Net (ResNet18)...")
    baseline_model = BaselineEarlyFusionUNet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1,
    ).to(device)
    baseline_model.train()
    print("  -> Baseline model successfully loaded on", device)

    # 2. Create Dummy Batch
    print(f"\n[Step 2] Generating Batch on {device}: Batch Size={batch_size}, Shape=({height}, {width})...")
    t1 = torch.randn(batch_size, 3, height, width, device=device)
    t2 = torch.randn(batch_size, 3, height, width, device=device)
    # Binary mask: 0 or 1
    masks = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32, device=device)

    # 3. Setup Loss, Optimizer, Metrics
    criterion = CompoundLoss(loss_type="bce_dice", bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4)
    metrics = ChangeDetectionMetrics()

    # 4. Baseline Forward Pass
    print("\n[Step 3] Running Baseline Forward Pass...")
    t_start = time.time()
    optimizer.zero_grad()
    logits = baseline_model(t1, t2)
    fwd_time = time.time() - t_start

    print(f"  -> Output shape: {list(logits.shape)} (Expected: [{batch_size}, 1, {height}, {width}])")
    assert logits.shape == (batch_size, 1, height, width), f"Shape mismatch: {logits.shape}"
    print(f"  -> Forward pass time: {fwd_time * 1000:.2f} ms")

    # 5. Loss Calculation
    print("\n[Step 4] Computing Compound Loss (BCE + Dice)...")
    loss = criterion(logits, masks)
    print(f"  -> Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss computed as NaN!"

    # 6. Backward Pass & Step
    print("\n[Step 5] Running Backward Pass (Autograd) & Optimizer Step...")
    t_bwd_start = time.time()
    loss.backward()
    optimizer.step()
    bwd_time = time.time() - t_bwd_start
    print(f"  -> Backward pass time: {bwd_time * 1000:.2f} ms")
    print("  -> Gradients calculated and model parameters updated successfully!")

    # 7. Evaluate Metrics
    metrics.update(logits, masks)
    scores = metrics.compute()
    print("\n[Step 6] Batch Metrics:")
    print(f"  -> Precision: {scores['precision']:.4f}")
    print(f"  -> Recall:    {scores['recall']:.4f}")
    print(f"  -> F1-Score:  {scores['f1']:.4f}")
    print(f"  -> IoU:       {scores['iou']:.4f}")

    # 8. Siamese U-Net Smoke Test
    print("\n[Step 7] Testing Siamese U-Net with CBAM Attention...")
    siamese_model = SiameseUNet(
        encoder_name="resnet34",
        encoder_weights=None,  # fast smoke test without downloading weights
        classes=1,
        use_attention=True,
    ).to(device)
    siamese_model.train()

    opt_siamese = torch.optim.AdamW(siamese_model.parameters(), lr=1e-4)
    opt_siamese.zero_grad()
    s_logits = siamese_model(t1, t2)
    s_loss = criterion(s_logits, masks)
    s_loss.backward()
    opt_siamese.step()
    print(f"  -> Siamese U-Net Forward/Backward successful! Loss: {s_loss.item():.4f}")

    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED CLEANLY ON APPLE SILICON MPS!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    run_smoke_test()
