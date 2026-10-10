"""Evaluation metrics for Remote Sensing Change Detection."""
import torch


class ChangeDetectionMetrics:
    """
    Accumulates confusion matrix statistics across batches and computes:
    - Precision
    - Recall
    - F1-Score
    - Change-IoU (Intersection over Union for the change class)
    - Overall Accuracy (OA)
    """
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits_or_probs, targets):
        """
        Update running metrics with a batch.
        
        Args:
            logits_or_probs: Tensor of raw logits or probabilities [B, 1, H, W]
            targets: Tensor of binary targets [B, 1, H, W]
        """
        if logits_or_probs.min() < 0.0 or logits_or_probs.max() > 1.0:
            probs = torch.sigmoid(logits_or_probs)
        else:
            probs = logits_or_probs

        preds = (probs >= self.threshold).long().view(-1)
        labels = (targets >= 0.5).long().view(-1)

        self.tp += ((preds == 1) & (labels == 1)).sum().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().item()

    def compute(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "oa": float(oa),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
        }
