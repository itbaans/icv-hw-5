"""
gradcam.py  --  GradCAM visualisation for Task 1 CNN models.

Usage (from task1_cnn/):
    python gradcam.py --checkpoint cnn_outputs/model_a_best.pth --n 6
    python gradcam.py --checkpoint cnn_outputs/model_b_best.pth --n 6

Outputs:
    cnn_outputs/gradcam_model_a.png
    cnn_outputs/gradcam_model_b.png

Each output is a grid of rows [original | heatmap | overlay | pred vs gt].
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import yaml

from models import AblationModel
from data   import get_dataloaders


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for a regression output."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Returns
        -------
        cam   : (H, W) numpy array in [0, 1], resized to input spatial dims
        pred  : scalar prediction
        """
        self.model.zero_grad()
        out = self.model(x)           # (1, 1) or (1,)
        pred = out.squeeze().item()
        out.squeeze().backward()

        # weight activations by global-average-pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # normalise to [0,1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # resize to input size
        h, w = x.shape[2], x.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam, pred


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) normalised tensor → (H,W,3) uint8 RGB."""
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def heatmap_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend GradCAM heatmap onto RGB image."""
    heat = cm.jet(cam)[:, :, :3]   # (H,W,3) float
    heat = (heat * 255).astype(np.uint8)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--n",          type=int, default=6,
                        help="Number of images to visualise (half success, half failure)")
    parser.add_argument("--out",        default="cnn_outputs/gradcam.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt   = torch.load(args.checkpoint, map_location=device)
    params = ckpt["params"]
    model  = AblationModel(
        filters      = params["filters"],
        kernel_size  = params["kernel_size"],
        dropout_rate = params["dropout_rate"],
        use_residual = params["use_residual"],
        activation   = params["activation"],
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded: {args.checkpoint}  |  config: {params}")

    # ── attach GradCAM to last conv block's activation (before GAP) ─────────
    # AblationModel stores blocks in self.features (nn.Sequential)
    target_layer = model.features[-1]   # last conv block
    gcam = GradCAM(model, target_layer)

    # ── get validation data ──────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])
    _, val_loader = get_dataloaders(cfg)

    # collect all predictions and errors
    all_imgs, all_preds, all_gts = [], [], []
    model.eval()
    with torch.enable_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            for i in range(imgs.size(0)):
                x   = imgs[i:i+1]
                cam, pred = gcam(x)
                gt  = labels[i].item()
                all_imgs.append((x, cam))
                all_preds.append(pred)
                all_gts.append(gt)
                if len(all_imgs) >= 200:  # sample pool
                    break
            if len(all_imgs) >= 200:
                break

    errors = [abs(p - g) for p, g in zip(all_preds, all_gts)]
    sorted_idx = sorted(range(len(errors)), key=lambda i: errors[i])

    n_each  = args.n // 2
    best_idx  = sorted_idx[:n_each]           # lowest error
    worst_idx = sorted_idx[-n_each:]          # highest error

    selected = best_idx + worst_idx
    labels_type = ["success"] * n_each + ["failure"] * n_each

    # ── build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(selected), 3,
                             figsize=(12, 3.2 * len(selected)))
    fig.suptitle("GradCAM Visualisations — Task 1 CNN\n"
                 "(Top rows: successes, Bottom rows: failures)", fontsize=13)

    col_titles = ["Original", "GradCAM heatmap", "Overlay + prediction"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=11)

    for row, (idx, ltype) in enumerate(zip(selected, labels_type)):
        x, cam      = all_imgs[idx]
        pred        = all_preds[idx]
        gt          = all_gts[idx]
        rgb         = tensor_to_rgb(x)
        overlay     = heatmap_overlay(rgb, cam)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(f"[{ltype.upper()}]", fontsize=9)

        axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xlabel(f"pred={pred:.1f}  gt={gt:.0f}  "
                                f"|err|={abs(pred-gt):.1f}", fontsize=9)

        for c in range(3):
            axes[row, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
