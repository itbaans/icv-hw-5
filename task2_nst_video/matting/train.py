import torch
import torch.nn as nn
import torch.optim as optim
from model import UNetMatting

# TODO: Add AISegment dataset loader
# TODO: Implement mixed loss: L_total = L1(alpha, alpha_gt) + BCE(alpha, alpha_gt) or Dice Loss
# TODO: Train loop targeting IoU >= 0.85
