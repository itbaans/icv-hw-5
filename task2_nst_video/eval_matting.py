"""
eval_matting.py  --  Evaluate the saved matting model on the test split.

Produces (in outputs/):
  matting_eval_metrics.txt     IoU, MAD, loss on 500 test images (print + save)
  matting_overlay.png          5-row overlay grid: frame | alpha | cutout
  matting_error_grid.png       6-row error analysis: best 3 + worst 3 IoU images
  matting_training_curves.png  re-saved from CSV (if matting_train_log.csv exists)

Usage (from task2_nst_video/):
    python eval_matting.py --config config.yaml

On Kaggle the checkpoint is read from config:
    matting.checkpoint: /kaggle/input/models/.../matting_best.pth
"""

import os, sys, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import yaml

# make sure matting/ is on path when called from task2_nst_video/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "matting"))

from model   import UNetMatting
from dataset import AISegmentDataset
from metrics import iou_score, mad_score
from losses  import CombinedLoss
from torch.utils.data import DataLoader, Subset


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_rgb(img_tensor):
    """(1,3,H,W) tensor → (H,W,3) uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img  = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def _alpha_img(alpha_tensor):
    """(1,1,H,W) tensor → (H,W) uint8 grayscale."""
    return (alpha_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _cutout(rgb, alpha_u8):
    """RGBA cutout: person on white background."""
    canvas = np.ones_like(rgb) * 255
    a = alpha_u8.astype(np.float32) / 255.0
    a3 = a[:, :, np.newaxis]
    return (rgb * a3 + canvas * (1 - a3)).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = total_mad = n = 0.0
    with torch.no_grad():
        for imgs, mattes in loader:
            imgs, mattes = imgs.to(device), mattes.to(device)
            preds = model(imgs)
            loss  = criterion(preds, mattes)
            b     = imgs.size(0)
            total_loss += loss.item() * b
            total_iou  += iou_score(preds, mattes) * b
            total_mad  += mad_score(preds, mattes) * b
            n += b
    return total_loss / n, total_iou / n, total_mad / n


# ──────────────────────────────────────────────────────────────────────────────
# figure 1: overlay grid (5 rows)
# ──────────────────────────────────────────────────────────────────────────────

def save_overlay_grid(model, dataset, device, out_path, n=5):
    """5 evenly spaced samples: [image | GT alpha | pred alpha | cutout]."""
    model.eval()
    indices = [int(len(dataset) * f) for f in np.linspace(0.05, 0.95, n)]

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    col_titles = ["Input frame", "GT alpha matte", "Predicted alpha", "Foreground cutout"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=11, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, matte = dataset[idx]
            img_t   = img.unsqueeze(0).to(device)
            pred    = torch.sigmoid(model(img_t)).cpu()

            rgb      = _to_rgb(img_t.cpu())
            gt_u8    = _alpha_img(matte.unsqueeze(0))
            pred_u8  = _alpha_img(pred)
            cut      = _cutout(rgb, pred_u8)

            iou = iou_score(pred.unsqueeze(0), matte.unsqueeze(0).unsqueeze(0))
            mad = mad_score(pred.unsqueeze(0), matte.unsqueeze(0).unsqueeze(0))

            axes[row, 0].imshow(rgb)
            axes[row, 0].set_ylabel(f"sample {idx}", fontsize=9)
            axes[row, 1].imshow(gt_u8,   cmap="gray", vmin=0, vmax=255)
            axes[row, 2].imshow(pred_u8, cmap="gray", vmin=0, vmax=255)
            axes[row, 2].set_xlabel(f"IoU={iou:.3f}  MAD={mad:.4f}", fontsize=8)
            axes[row, 3].imshow(cut)

            for c in range(4):
                axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    fig.suptitle("Matting Model — Qualitative Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# figure 2: error analysis grid (best 3 + worst 3 by IoU)
# ──────────────────────────────────────────────────────────────────────────────

def save_error_grid(model, dataset, device, out_path, pool=100, n_each=3):
    """Randomly sample pool images, pick best/worst n_each by IoU."""
    model.eval()
    indices = list(range(min(pool, len(dataset))))
    results = []

    with torch.no_grad():
        for idx in indices:
            img, matte = dataset[idx]
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device))).cpu()
            iou  = iou_score(pred.unsqueeze(0), matte.unsqueeze(0).unsqueeze(0))
            mad  = mad_score(pred.unsqueeze(0), matte.unsqueeze(0).unsqueeze(0))
            results.append((idx, iou, mad, img, matte, pred.squeeze(0)))

    results.sort(key=lambda x: x[1])
    worst = results[:n_each]
    best  = results[-n_each:]
    selected = best + worst
    tags     = ["Best"] * n_each + ["Worst"] * n_each

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.2 * n_rows))
    col_titles = ["Input frame", "Predicted alpha", "GT alpha"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=11, fontweight="bold")

    for row, ((idx, iou, mad, img, matte, pred), tag) in enumerate(zip(selected, tags)):
        rgb     = _to_rgb(img.unsqueeze(0))
        pred_u8 = _alpha_img(pred.unsqueeze(0).unsqueeze(0))
        gt_u8   = _alpha_img(matte.unsqueeze(0))

        color = "green" if tag == "Best" else "red"
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(f"[{tag}] #{idx}", fontsize=9, color=color)
        axes[row, 1].imshow(pred_u8, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_xlabel(f"IoU={iou:.3f}  MAD={mad:.4f}", fontsize=9)
        axes[row, 2].imshow(gt_u8,  cmap="gray", vmin=0, vmax=255)

        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    fig.suptitle("Matting Error Analysis — Best and Worst Predictions by IoU",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# figure 3: re-render training curves from CSV (if available)
# ──────────────────────────────────────────────────────────────────────────────

def save_training_curves(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        print(f"  Skipping training curves — log not found: {log_path}")
        return

    epochs = []; tr_loss=[]; vl_loss=[]; tr_iou=[]; vl_iou=[]; tr_mad=[]; vl_mad=[]; lrs=[]
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"])); vl_loss.append(float(row["val_loss"]))
            tr_iou.append(float(row["train_iou"]));   vl_iou.append(float(row["val_iou"]))
            tr_mad.append(float(row["train_mad"]));   vl_mad.append(float(row["val_mad"]))
            lrs.append(float(row["lr"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Matting U-Net — Training Curves", fontsize=14, fontweight="bold")

    def _subplot(ax, train, val, ylabel, title, target=None):
        ax.plot(epochs, train, label="Train", linewidth=1.8)
        ax.plot(epochs, val,   label="Val",   linewidth=1.8, linestyle="--")
        if target is not None:
            ax.axhline(target, color="red", linestyle=":", linewidth=1.2,
                       label=f"Target ({target})")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    _subplot(axes[0,0], tr_loss, vl_loss, "Loss",      "Combined Loss (L1 + Dice)")
    _subplot(axes[0,1], tr_iou,  vl_iou,  "IoU",       "IoU",       target=0.85)
    _subplot(axes[1,0], tr_mad,  vl_mad,  "MAD",       "Mean Absolute Difference")
    axes[1,1].semilogy(epochs, lrs, linewidth=1.8, color="purple")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Learning rate (log)")
    axes[1,1].set_title("Learning Rate Schedule"); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t2      = cfg["task2"]
    mat_cfg = t2["matting"]
    out_dir = t2.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt_path = mat_cfg["checkpoint"]
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = UNetMatting(n_channels=3, n_classes=1).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Loaded.  Best val IoU from training: {ckpt.get('best_iou', 'N/A')}")

    # ── build test split (same seed as training = reproducible) ───────────────
    data_cfg = mat_cfg["data"]
    torch.manual_seed(42)

    full_dataset = AISegmentDataset(
        clip_root   = data_cfg["clip_root"],
        matte_root  = data_cfg["matte_root"],
        img_size    = tuple(data_cfg["img_size"]),
        max_pairs   = data_cfg["n_train"] + data_cfg["n_val"] + data_cfg["n_test"],
        augment     = False,
        seed        = 42,
    )

    n_train = data_cfg["n_train"]
    n_val   = data_cfg["n_val"]
    n_test  = data_cfg["n_test"]
    test_dataset = Subset(full_dataset, range(n_train + n_val, n_train + n_val + n_test))
    print(f"Test split: {len(test_dataset)} images")

    test_loader = DataLoader(test_dataset, batch_size=16,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── evaluation ────────────────────────────────────────────────────────────
    loss_fn = CombinedLoss(
        lambda_l1   = float(mat_cfg["loss"]["lambda_l1"]),
        lambda_dice = float(mat_cfg["loss"]["lambda_dice"]),
    )
    test_loss, test_iou, test_mad = evaluate(model, test_loader, loss_fn, device)

    print("\n── Test Metrics ──────────────────────────────────────")
    print(f"  Loss  : {test_loss:.4f}")
    print(f"  IoU   : {test_iou:.4f}   (target >= 0.85)")
    print(f"  MAD   : {test_mad:.6f}")
    print(f"  n     : {len(test_dataset)}")
    print("──────────────────────────────────────────────────────")

    # save metrics text
    metrics_path = os.path.join(out_dir, "matting_eval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test set evaluation ({len(test_dataset)} images)\n")
        f.write(f"Loss : {test_loss:.4f}\n")
        f.write(f"IoU  : {test_iou:.4f}\n")
        f.write(f"MAD  : {test_mad:.6f}\n")
    print(f"  Saved: {metrics_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    save_overlay_grid(
        model, test_dataset, device,
        os.path.join(out_dir, "matting_overlay.png"), n=5
    )
    save_error_grid(
        model, test_dataset, device,
        os.path.join(out_dir, "matting_error_grid.png"), pool=100, n_each=3
    )
    save_training_curves(
        os.path.join(out_dir, "matting_train_log.csv"),
        os.path.join(out_dir, "matting_training_curves.png")
    )

    print("\nDone. Outputs:")
    for fname in ["matting_eval_metrics.txt", "matting_overlay.png",
                  "matting_error_grid.png", "matting_training_curves.png"]:
        p = os.path.join(out_dir, fname)
        status = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{status}] {fname}")


if __name__ == "__main__":
    main()
