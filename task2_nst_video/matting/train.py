"""
train.py - Training script for the UNet Human Matting model

Usage
-----
    python matting/train.py --config config.yaml

All hyperparameters are read from config.yaml (task2.matting section).
Training logs are written as CSV to outputs/matting_train_log.csv and
the best checkpoint (highest val IoU) is saved to
outputs/matting_best.pth.

Loss
----
L_total = 0.5 * L1(alpha_pred, alpha_gt) + 0.5 * DiceLoss(alpha_pred, alpha_gt)
See matting/losses.py for rationale.

Target: IoU >= 0.85 on AISegment validation split.
"""

import os
import sys
import csv
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

# ── make sure the task2 root is importable ───────────────────────────────────
TASK2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK2_ROOT))

from matting.model   import UNetMatting
from matting.dataset import make_dataloaders
from matting.losses  import MattingLoss
from matting.metrics import RunningMetrics


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    metrics = RunningMetrics()
    total_loss = 0.0

    for imgs, mattes in loader:
        imgs   = imgs.to(device, non_blocking=True)
        mattes = mattes.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                preds = model(imgs)
                loss, l1, dice = criterion(preds, mattes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss, l1, dice = criterion(preds, mattes)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            metrics.update(preds.detach(), mattes)

    n = len(loader.dataset)
    return total_loss / n, metrics.iou, metrics.mad


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    metrics = RunningMetrics()
    total_loss = 0.0

    for imgs, mattes in loader:
        imgs   = imgs.to(device, non_blocking=True)
        mattes = mattes.to(device, non_blocking=True)
        preds  = model(imgs)
        loss, _, _ = criterion(preds, mattes)
        total_loss += loss.item() * imgs.size(0)
        metrics.update(preds, mattes)

    n = len(loader.dataset)
    return total_loss / n, metrics.iou, metrics.mad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UNet matting model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg  = cfg["task2"]["matting"]
    tcfg  = mcfg["training"]
    dcfg  = mcfg["data"]
    ocfg  = cfg["task2"]["output_dir"]

    os.makedirs(ocfg, exist_ok=True)

    set_seed(tcfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[matting/train.py] device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    # Build a cfg dict in the shape make_dataloaders expects
    loader_cfg = {
        "data": {
            "clip_root":  dcfg["clip_root"],
            "matte_root": dcfg["matte_root"],
            "img_size":   dcfg["img_size"],
            "n_train":    dcfg["n_train"],
            "n_val":      dcfg["n_val"],
            "n_test":     dcfg["n_test"],
        },
        "training": {
            "seed":        tcfg["seed"],
            "batch_size":  tcfg["batch_size"],
            "num_workers": tcfg.get("num_workers", 4),
        },
    }
    loaders = make_dataloaders(loader_cfg)
    print(f"  train: {len(loaders['train'].dataset)} samples")
    print(f"  val  : {len(loaders['val'].dataset)}   samples")
    print(f"  test : {len(loaders['test'].dataset)}  samples")

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNetMatting(
        n_channels=3,
        n_classes=1,
        bilinear=mcfg.get("bilinear", False),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNetMatting parameters: {n_params:,}")

    # ── loss / optimiser / scheduler ──────────────────────────────────────────
    criterion = MattingLoss(
        lambda_l1   = mcfg["loss"]["lambda_l1"],
        lambda_dice = mcfg["loss"]["lambda_dice"],
    )
    optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=tcfg.get("lr_patience", 3)
    )

    use_amp = tcfg.get("amp", False) and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── training loop ─────────────────────────────────────────────────────────
    best_iou     = 0.0
    patience_cnt = 0
    early_stop   = tcfg.get("early_stop_patience", 7)
    log_path     = os.path.join(ocfg, "matting_train_log.csv")
    ckpt_path    = os.path.join(ocfg, "matting_best.pth")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou", "train_mad",
                                  "val_loss",   "val_iou",   "val_mad",   "lr"])

        for epoch in range(1, tcfg["epochs"] + 1):
            tr_loss, tr_iou, tr_mad = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, scaler
            )
            vl_loss, vl_iou, vl_mad = evaluate(
                model, loaders["val"], criterion, device
            )
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(vl_iou)

            print(
                f"Epoch {epoch:03d}/{tcfg['epochs']} | "
                f"train loss {tr_loss:.4f} iou {tr_iou:.4f} mad {tr_mad:.4f} | "
                f"val   loss {vl_loss:.4f} iou {vl_iou:.4f} mad {vl_mad:.4f} | "
                f"lr {lr:.2e}"
            )
            writer.writerow([epoch, tr_loss, tr_iou, tr_mad, vl_loss, vl_iou, vl_mad, lr])

            # ── checkpoint ────────────────────────────────────────────────────
            if vl_iou > best_iou:
                best_iou = vl_iou
                patience_cnt = 0
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_iou":    vl_iou,
                    "val_mad":    vl_mad,
                    "cfg":        mcfg,
                }, ckpt_path)
                print(f"  ✓ Saved best checkpoint (IoU={best_iou:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= early_stop:
                    print(f"  Early stopping triggered after {epoch} epochs.")
                    break

    # ── final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    ts_loss, ts_iou, ts_mad = evaluate(model, loaders["test"], criterion, device)
    print(f"TEST  | loss {ts_loss:.4f}  IoU {ts_iou:.4f}  MAD {ts_mad:.4f}")
    print(f"Best val IoU: {best_iou:.4f}  (target >= 0.85)")


if __name__ == "__main__":
    main()
