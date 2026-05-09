"""
train_final.py  --  Final training run for Model A and Model B.

Produces (in cnn_outputs/):
  model_a_best.pth           checkpoint
  model_b_best.pth           checkpoint
  model_a_curves.png         training curves (loss + MAE, train vs val)
  model_b_curves.png         training curves
  model_a_gradcam.png        6-row success/failure GradCAM grid
  model_b_gradcam.png        6-row success/failure GradCAM grid
  final_results.csv          val MAE/RMSE/best-epoch for both models

Usage:
    python train_final.py               # reads config.yaml
    python train_final.py --kaggle      # uses Kaggle paths from config
    python train_final.py --model a     # train only Model A
    python train_final.py --model b     # train only Model B
"""

import os, csv, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import yaml

from data   import get_dataloaders, SeedDataset
from models import AblationModel

# ──────────────────────────────────────────────────────────────────────────────
# Model configurations  (from ablation study best_ablation_config.json)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "a": {
        "label":        "Model A (3-block)",
        "filters":      [32, 64, 128],
        "kernel_size":  3,
        "dropout_rate": 0.3,
        "use_residual": True,
        "activation":   "leaky_relu",
        "optimizer":    "sgd",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "momentum":     0.9,
    },
    "b": {
        "label":        "Model B (4-block)",
        "filters":      [64, 128, 256, 512],
        "kernel_size":  3,
        "dropout_rate": 0.3,
        "use_residual": True,
        "activation":   "leaky_relu",   # leaky_relu from Group 6; Group 8 used relu
        "optimizer":    "sgd",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "momentum":     0.9,
    },
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_mse = total_mae = n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_mse += loss.item() * labels.size(0)
            total_mae += torch.abs(preds.detach() - labels).sum().item()
            n += labels.size(0)
    return total_mse / n, total_mae / n   # mse, mae


def train_model(key: str, cfg: dict, train_loader, val_loader,
                device: torch.device, out_dir: str, epochs: int,
                patience: int) -> dict:
    """Train one model; return result dict."""
    params = MODEL_CONFIGS[key]
    print(f"\n{'='*60}")
    print(f"  {params['label']}")
    print(f"  filters={params['filters']}  dropout={params['dropout_rate']}"
          f"  residual={params['use_residual']}  activation={params['activation']}")
    print(f"  wd={params['weight_decay']}  lr={params['learning_rate']}")
    print(f"{'='*60}")

    model = AblationModel(
        filters      = params["filters"],
        kernel_size  = params["kernel_size"],
        dropout_rate = params["dropout_rate"],
        use_residual = params["use_residual"],
        activation   = params["activation"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=params["learning_rate"],
                          momentum=params["momentum"],
                          weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=5)

    history = {"epoch": [], "train_mse": [], "train_mae": [],
               "val_mse": [], "val_mae": [], "lr": []}

    best_val_mae = float("inf")
    best_epoch   = 0
    no_improve   = 0
    ckpt_path    = os.path.join(out_dir, f"model_{key}_best.pth")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_mse, tr_mae = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_mse, vl_mae = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(vl_mae)

        history["epoch"].append(epoch)
        history["train_mse"].append(tr_mse)
        history["train_mae"].append(tr_mae)
        history["val_mse"].append(vl_mse)
        history["val_mae"].append(vl_mae)
        history["lr"].append(lr)

        if vl_mae < best_val_mae:
            best_val_mae = vl_mae
            best_epoch   = epoch
            no_improve   = 0
            torch.save({"state_dict": model.state_dict(), "params": params,
                        "best_val_mae": best_val_mae, "best_epoch": best_epoch,
                        "n_params": n_params},
                       ckpt_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d}/{epochs} | "
                  f"train MAE={tr_mae:.3f} | val MAE={vl_mae:.3f} | "
                  f"lr={lr:.6f} | best={best_val_mae:.3f}@{best_epoch}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - t0
    val_rmse = float(np.sqrt(best_val_mae**2))   # approx — reload for exact
    print(f"  Finished in {elapsed:.0f}s.  Best val MAE={best_val_mae:.4f} @ ep {best_epoch}")

    return {
        "key": key, "label": params["label"],
        "best_val_mae": best_val_mae, "best_epoch": best_epoch,
        "n_params": n_params, "history": history,
        "ckpt_path": ckpt_path, "params": params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training curves plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(result: dict, out_path: str):
    h = result["history"]
    epochs = h["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['label']} — Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, h["train_mse"], label="Train MSE", linewidth=1.8)
    ax.plot(epochs, h["val_mse"],   label="Val MSE",   linewidth=1.8, linestyle="--")
    ax.axvline(result["best_epoch"], color="red", linestyle=":", linewidth=1.2,
               label=f"Best epoch ({result['best_epoch']})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Loss (MSE)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, h["train_mae"], label="Train MAE", linewidth=1.8)
    ax.plot(epochs, h["val_mae"],   label="Val MAE",   linewidth=1.8, linestyle="--")
    ax.axvline(result["best_epoch"], color="red", linestyle=":", linewidth=1.2,
               label=f"Best val MAE={result['best_val_mae']:.3f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (seeds)")
    ax.set_title("Mean Absolute Error"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# GradCAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def compute(self, x: torch.Tensor, model: nn.Module):
        model.zero_grad()
        out  = model(x)
        pred = out.squeeze().item()
        out.squeeze().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam, pred


def tensor_to_rgb(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return (img * 255).astype(np.uint8)


def heatmap_overlay(rgb, cam, alpha=0.45):
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def plot_gradcam(result: dict, val_loader, device: torch.device,
                 out_path: str, n_rows: int = 6):
    """
    n_rows: total rows in the figure (n_rows//2 successes, n_rows//2 failures).
    Each row: [Original | GradCAM heat | Overlay | text label]
    """
    params = result["params"]
    model  = AblationModel(
        filters      = params["filters"],
        kernel_size  = params["kernel_size"],
        dropout_rate = params["dropout_rate"],
        use_residual = params["use_residual"],
        activation   = params["activation"],
    ).to(device)
    ckpt = torch.load(result["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # attach GradCAM to last conv block (self.features is the nn.Sequential of conv blocks)
    target_layer = model.features[-1]
    gcam = GradCAM(model, target_layer)

    # collect predictions on up to 300 val images
    pool_imgs, pool_preds, pool_gts = [], [], []
    with torch.enable_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            for i in range(imgs.size(0)):
                x   = imgs[i:i+1]
                cam, pred = gcam.compute(x, model)
                gt = labels[i].item()
                pool_imgs.append((x.cpu(), cam))
                pool_preds.append(pred)
                pool_gts.append(gt)
                if len(pool_imgs) >= 300:
                    break
            if len(pool_imgs) >= 300:
                break

    errors = [abs(p - g) for p, g in zip(pool_preds, pool_gts)]
    srt    = sorted(range(len(errors)), key=lambda i: errors[i])
    n_each = n_rows // 2
    selected = [(i, "Success") for i in srt[:n_each]] + \
               [(i, "Failure") for i in srt[-n_each:]]

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows))
    col_titles = ["Original image", "GradCAM heatmap", "Overlay + prediction"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=11, fontweight="bold")

    for row, (idx, tag) in enumerate(selected):
        x_cpu, cam = pool_imgs[idx]
        pred, gt   = pool_preds[idx], pool_gts[idx]
        rgb        = tensor_to_rgb(x_cpu)
        overlay    = heatmap_overlay(rgb, cam)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(f"[{tag}]\nGT={gt:.0f}", fontsize=9,
                                 color="green" if tag == "Success" else "red")
        axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xlabel(f"pred={pred:.1f}  gt={gt:.0f}  "
                                 f"|err|={abs(pred-gt):.1f} seeds",
                                 fontsize=9)
        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    fig.suptitle(f"{result['label']} — GradCAM: Successes (top) & Failures (bottom)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--kaggle",  action="store_true",
                        help="Print Kaggle path note (no behavioural change)")
    parser.add_argument("--model",   default="both", choices=["a", "b", "both"])
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--patience",type=int, default=15)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.get("output_dir", "cnn_outputs")
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}   out_dir: {out_dir}")

    train_loader, val_loader = get_dataloaders(cfg)
    print(f"Train size: {len(train_loader.dataset)}  "
          f"Val size: {len(val_loader.dataset)}")

    models_to_run = ["a", "b"] if args.model == "both" else [args.model]
    results = {}

    # ── training ──────────────────────────────────────────────────────────────
    for key in models_to_run:
        result = train_model(key, cfg, train_loader, val_loader, device,
                             out_dir, args.epochs, args.patience)
        results[key] = result

    # ── plots ─────────────────────────────────────────────────────────────────
    for key, result in results.items():
        plot_curves(result, os.path.join(out_dir, f"model_{key}_curves.png"))
        plot_gradcam(result, val_loader, device,
                     os.path.join(out_dir, f"model_{key}_gradcam.png"),
                     n_rows=6)

    # ── summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "final_results.csv")
    fieldnames   = ["model", "label", "filters", "n_params",
                    "best_val_mae", "best_epoch"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, r in results.items():
            w.writerow({
                "model":        key.upper(),
                "label":        r["label"],
                "filters":      str(r["params"]["filters"]),
                "n_params":     r["n_params"],
                "best_val_mae": round(r["best_val_mae"], 4),
                "best_epoch":   r["best_epoch"],
            })
    print(f"\nSummary saved: {summary_path}")

    print("\n── Final results ─────────────────────────────────")
    for key, r in results.items():
        print(f"  {r['label']:25s}  val MAE={r['best_val_mae']:.4f}  "
              f"epoch={r['best_epoch']}  params={r['n_params']:,}")

    print("\nOutputs in", out_dir)
    for key in models_to_run:
        for suffix in ["_best.pth", "_curves.png", "_gradcam.png"]:
            p = os.path.join(out_dir, f"model_{key}{suffix}")
            status = "OK" if os.path.exists(p) else "MISSING"
            print(f"  [{status}] model_{key}{suffix}")


if __name__ == "__main__":
    main()
