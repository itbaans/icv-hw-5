"""
losses.py - Combined matting loss

L_total = lambda_l1 * L1(alpha_pred, alpha_gt)
        + lambda_dice * DiceLoss(alpha_pred, alpha_gt)

DiceLoss is equivalent to (1 - Dice coefficient) and doubles as a soft
binary-cross-entropy surrogate that is robust to class imbalance (most
pixels are either fully foreground or fully background).

Weighting choice (documented in config.yaml):
  lambda_l1   = 0.5   — penalises absolute pixel error
  lambda_dice = 0.5   — penalises mis-shape of the matte boundary

The assignment says "L1 on the alpha matte and a binary cross-entropy or
Dice term"; we choose Dice because it is invariant to the background /
foreground imbalance typical in portrait images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss: 1 - (2 * intersection) / (union + eps)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred and target: (B, 1, H, W) in [0, 1]
        pred   = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()


class MattingLoss(nn.Module):
    """
    Combined L1 + Dice loss.

    Parameters
    ----------
    lambda_l1  : weight for the L1 term   (default 0.5)
    lambda_dice: weight for the Dice term  (default 0.5)
    """

    def __init__(self, lambda_l1: float = 0.5, lambda_dice: float = 0.5):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_dice = lambda_dice
        self.l1_loss     = nn.L1Loss()
        self.dice_loss   = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1   = self.l1_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total = self.lambda_l1 * l1 + self.lambda_dice * dice
        return total, l1, dice
