"""
metrics.py - Evaluation metrics for the human matting model

Metrics
-------
- IoU (Intersection over Union): binarise at 0.5 threshold.
  Target: >= 0.85 on the AISegment validation split.
- MAD (Mean Absolute Difference): average |pred - gt| over all pixels,
  useful as a continuous measure of matte quality.
"""

import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Binary IoU after thresholding at `threshold`.

    Parameters
    ----------
    pred   : (B, 1, H, W) float tensor in [0, 1]
    target : (B, 1, H, W) float tensor in [0, 1]

    Returns
    -------
    mean IoU across the batch (Python float)
    """
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union        = ((pred_bin + target_bin) >= 1).float().sum(dim=(1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def mad_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Absolute Difference between predicted and ground-truth alpha.

    Returns
    -------
    Python float
    """
    return (pred - target).abs().mean().item()


class RunningMetrics:
    """Accumulate IoU and MAD over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._iou_sum = 0.0
        self._mad_sum = 0.0
        self._n       = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        b = pred.size(0)
        self._iou_sum += iou_score(pred, target) * b
        self._mad_sum += mad_score(pred, target) * b
        self._n       += b

    @property
    def iou(self) -> float:
        return self._iou_sum / max(self._n, 1)

    @property
    def mad(self) -> float:
        return self._mad_sum / max(self._n, 1)
