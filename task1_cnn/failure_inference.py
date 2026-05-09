"""
failure_inference.py  --  Run Model A/B inference on the 20 selected failure cases
                          from the Assignment 2 evaluator and compare with prior methods.

Outputs:
  cnn_outputs/failure_comparison.csv    per-image table (GT, A2-pred, ModelA, ModelB, errors)
  cnn_outputs/failure_comparison.png    publication-ready figure (3 density groups)
  cnn_outputs/failure_summary.txt       aggregate MAE/RMSE per method for the LaTeX table

Usage (on Kaggle):
    %cd /kaggle/working/icv-hw-5/task1_cnn
    !python failure_inference.py

Usage (local):
    python failure_inference.py --img-dir /path/to/filtered \\
                                --ckpt-a cnn_outputs/model_a_best.pth \\
                                --ckpt-b cnn_outputs/model_b_best.pth
"""

import os, csv, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import AblationModel

# ──────────────────────────────────────────────────────────────────────────────
# 20 selected failure cases  (GT value = filename stem)
# Groups: 8 in 50-70, 8 in 70-100, 4 in 100+
# ──────────────────────────────────────────────────────────────────────────────
SELECTED = [
    # --- 50-70 (moderate density, A2 under-counts by ~15-28%) ---
    {"gt": 50,  "a2_pred": 43},
    {"gt": 53,  "a2_pred": 43},
    {"gt": 56,  "a2_pred": 42},
    {"gt": 59,  "a2_pred": 42},
    {"gt": 62,  "a2_pred": 45},
    {"gt": 65,  "a2_pred": 58},
    {"gt": 68,  "a2_pred": 58},
    {"gt": 70,  "a2_pred": 57},
    # --- 70-100 (high density, A2 under-counts by ~13-20%) ---
    {"gt": 71,  "a2_pred": 58},
    {"gt": 73,  "a2_pred": 59},
    {"gt": 75,  "a2_pred": 65},
    {"gt": 77,  "a2_pred": 66},
    {"gt": 80,  "a2_pred": 7},
    {"gt": 85,  "a2_pred": 7},
    {"gt": 90,  "a2_pred": 7},
    {"gt": 96,  "a2_pred": 7},
    # --- 100+ (very high density, A2 almost always predicts 7) ---
    {"gt": 101, "a2_pred": 7},
    {"gt": 110, "a2_pred": 7},
    {"gt": 120, "a2_pred": 7},
    {"gt": 135, "a2_pred": 7},
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> AblationModel:
    ckpt   = torch.load(ckpt_path, map_location=device)
    params = ckpt["params"]
    model  = AblationModel(
        filters      = params["filters"],
        kernel_size  = params["kernel_size"],
        dropout_rate = params["dropout_rate"],
        use_residual = params["use_residual"],
        activation   = params["activation"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def infer(model: AblationModel, img_path: str, device: torch.device) -> float:
    img = Image.open(img_path).convert("RGB")
    t   = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(t).squeeze().item()
    return pred


def find_image(img_dir: str, gt: int) -> str:
    """Try .png then .jpg (assignment dataset uses .png)."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(img_dir, f"{gt}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No image for GT={gt} in {img_dir}")


def rmse(errors):
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir",  default="/kaggle/input/datasets/abdullahahmedani/seeds-data/filtered")
    parser.add_argument("--ckpt-a",   default="cnn_outputs/model_a_best.pth")
    parser.add_argument("--ckpt-b",   default="cnn_outputs/model_b_best.pth")
    parser.add_argument("--out-dir",  default="cnn_outputs")
    args = parser.parse_args()

    # local fallback path
    if not os.path.isdir(args.img_dir):
        alt = "/home/itbaan/Assignment5/Assignment2/output/preprocessed_images/filtered"
        if os.path.isdir(alt):
            args.img_dir = alt
            print(f"Using local image dir: {alt}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading Model A …")
    model_a = load_model(args.ckpt_a, device)
    print("Loading Model B …")
    model_b = load_model(args.ckpt_b, device)

    # ── inference ─────────────────────────────────────────────────────────────
    rows = []
    for entry in SELECTED:
        gt  = entry["gt"]
        a2  = entry["a2_pred"]
        try:
            img_path = find_image(args.img_dir, gt)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        pred_a = infer(model_a, img_path, device)
        pred_b = infer(model_b, img_path, device)

        group = "50-70" if gt <= 70 else ("70-100" if gt <= 100 else "100+")
        rows.append({
            "gt": gt, "group": group,
            "a2_pred": a2,
            "model_a_pred": round(pred_a, 1),
            "model_b_pred": round(pred_b, 1),
            "err_a2":     abs(a2 - gt),
            "err_model_a": abs(pred_a - gt),
            "err_model_b": abs(pred_b - gt),
        })
        print(f"  GT={gt:3d}  A2={a2:3d}  ModelA={pred_a:6.1f}  ModelB={pred_b:6.1f}"
              f"  |err_A|={abs(pred_a-gt):.1f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "failure_comparison.csv")
    fields   = ["gt", "group", "a2_pred", "model_a_pred", "model_b_pred",
                "err_a2", "err_model_a", "err_model_b"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # ── aggregate summary ─────────────────────────────────────────────────────
    def agg(key):
        errs = [r[key] for r in rows]
        return np.mean(errs), rmse(errs)

    mae_a2,  rmse_a2  = agg("err_a2")
    mae_ma,  rmse_ma  = agg("err_model_a")
    mae_mb,  rmse_mb  = agg("err_model_b")

    # "fixed" = CNN error < A2 error on the same image
    fixed_a = sum(1 for r in rows if r["err_model_a"] < r["err_a2"])
    fixed_b = sum(1 for r in rows if r["err_model_b"] < r["err_a2"])
    n       = len(rows)

    summary_path = os.path.join(args.out_dir, "failure_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Failure case comparison ({n} images)\n")
        f.write(f"{'Method':<15} {'MAE':>6} {'RMSE':>6}  {'Failures fixed':>15}\n")
        f.write(f"{'A2 (Edge Det)':<15} {mae_a2:6.2f} {rmse_a2:6.2f}  {'---':>15}\n")
        f.write(f"{'CNN Model A':<15} {mae_ma:6.2f} {rmse_ma:6.2f}  {fixed_a}/{n}\n")
        f.write(f"{'CNN Model B':<15} {mae_mb:6.2f} {rmse_mb:6.2f}  {fixed_b}/{n}\n")

    print(open(summary_path).read())
    print(f"Saved: {summary_path}")

    # ── figure ────────────────────────────────────────────────────────────────
    groups = ["50-70", "70-100", "100+"]
    colors = {"A2 (Edge Det.)": "#e07b54",
              "CNN Model A":    "#4c8bb5",
              "CNN Model B":    "#5aaa78"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("CNN vs Edge Detection on Failure Cases (by Seed Density Group)",
                 fontsize=13, fontweight="bold")

    for ax, grp in zip(axes, groups):
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            ax.set_visible(False); continue
        gts     = [r["gt"] for r in grp_rows]
        err_a2  = [r["err_a2"] for r in grp_rows]
        err_ma  = [r["err_model_a"] for r in grp_rows]
        err_mb  = [r["err_model_b"] for r in grp_rows]

        x  = np.arange(len(gts))
        w  = 0.25
        ax.bar(x - w,   err_a2, w, label="A2 (Edge Det.)", color=colors["A2 (Edge Det.)"], alpha=0.85)
        ax.bar(x,       err_ma, w, label="CNN Model A",    color=colors["CNN Model A"],    alpha=0.85)
        ax.bar(x + w,   err_mb, w, label="CNN Model B",    color=colors["CNN Model B"],    alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels([str(g) for g in gts], fontsize=8, rotation=45)
        ax.set_xlabel("Ground-truth seed count")
        ax.set_ylabel("Absolute error (seeds)")
        ax.set_title(f"Density group: {grp}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "failure_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # ── print LaTeX snippet ───────────────────────────────────────────────────
    print("\n── LaTeX table numbers ──────────────────────────────────────────")
    print(f"  CNN Model A:  MAE={mae_ma:.2f}  RMSE={rmse_ma:.2f}  fixed={fixed_a}/{n}")
    print(f"  CNN Model B:  MAE={mae_mb:.2f}  RMSE={rmse_mb:.2f}  fixed={fixed_b}/{n}")


if __name__ == "__main__":
    main()
