"""Loss functions for Remote Sensing Change Detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Dice = 2 * |X n Y| / (|X| + |Y|)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Model raw output (before sigmoid) [B, 1, H, W] or [B, H, W]
            targets: Binary ground truth (0 or 1) [B, 1, H, W] or [B, H, W]
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance (e.g. 3-5% changed pixels).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CompoundLoss(nn.Module):
    """
    Composite loss combining BCE/Focal and Dice Loss.
    """
    def __init__(self, loss_type="bce_dice", bce_weight=0.5, dice_weight=0.5, focal_gamma=2.0):
        super().__init__()
        self.loss_type = loss_type
        self.w1 = bce_weight
        self.w2 = dice_weight
        self.dice = DiceLoss()
        if "focal" in loss_type:
            self.base_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        l_base = self.base_loss(logits, targets)
        l_dice = self.dice(logits, targets)
        return self.w1 * l_base + self.w2 * l_dice
