"""
visualize.py - Generate all required report figures for Task 2

Usage
-----
    python visualize.py --config config.yaml

Outputs (all in outputs/)
-------------------------
grid.png              : 5 content × 3 style = 15 NST results
beta_alpha_ablation.png : same content+style at 3 different style weights
layer_ablation.png    : shallow-only vs deep-only style layers
matting_overlay.png   : 5 sample frames + alpha matte + cutout overlay
feature_maps.png      : 8 channels from shallow & deep VGG19 layers,
                        applied to one video frame and one seed image
branded_poster.png    : 1024×1024 cherry-picked stylised still
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TASK2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK2_ROOT))

from nst import (
    VGG19Features,
    run_nst,
    load_img_as_tensor,
    tensor_to_uint8,
    IMAGENET_MEAN_255,
)
from matting.model import UNetMatting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rgb(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) ImageNet-normalised → uint8 RGB (H,W,3)."""
    return tensor_to_uint8(t)


def _load(path, height, device):
    return load_img_as_tensor(path, height, device)


def _matte_frame(frame_bgr, model, img_size, device):
    """Returns float32 alpha (H,W) in [0,1] at original resolution."""
    h_orig, w_orig = frame_bgr.shape[:2]
    mh, mw = img_size
    resized = cv2.resize(frame_bgr, (mw, mh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        a_small = model(t).squeeze().cpu().numpy()
    return cv2.resize(a_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# 1. NST Grid: 5 content × 3 style
# ---------------------------------------------------------------------------

def make_grid(content_paths, style_paths, nst_cfg, height, device, out_path):
    nrows = len(content_paths)
    ncols = len(style_paths)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for r, cp in enumerate(content_paths):
        for c, sp in enumerate(style_paths):
            print(f"  grid [{r},{c}]: {Path(cp).name} + {Path(sp).name}")
            ct = _load(cp, height, device)
            st = _load(sp, height, device)
            res = run_nst(ct, st, nst_cfg, verbose=False)
            img = _to_rgb(res)
            axes[r][c].imshow(img)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(Path(sp).stem, fontsize=10)
            if c == 0:
                axes[r][c].set_ylabel(Path(cp).stem, fontsize=10, rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# 2. β/α ablation: 3 style weights for 1 content + 1 style
# ---------------------------------------------------------------------------

def make_beta_alpha_ablation(content_path, style_path, style_weights,
                              base_cfg, height, device, out_path):
    n = len(style_weights)
    fig, axes = plt.subplots(1, n + 2, figsize=(5 * (n + 2), 5))

    ct = _load(content_path, height, device)
    st = _load(style_path,   height, device)

    axes[0].imshow(_to_rgb(ct)); axes[0].set_title("Content"); axes[0].axis("off")
    axes[1].imshow(_to_rgb(st)); axes[1].set_title("Style");   axes[1].axis("off")

    for i, sw in enumerate(style_weights):
        cfg = dict(base_cfg)
        cfg["style_weight"] = sw
        print(f"  beta/alpha sw={sw}")
        res = run_nst(ct, st, cfg, verbose=False)
        axes[i + 2].imshow(_to_rgb(res))
        axes[i + 2].set_title(f"sw={float(sw):.0e}")
        axes[i + 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# 3. Layer ablation: shallow-only vs deep-only
# ---------------------------------------------------------------------------

def make_layer_ablation(content_path, style_path, base_cfg,
                         height, device, out_path):
    ct = _load(content_path, height, device)
    st = _load(style_path,   height, device)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(_to_rgb(ct)); axes[0].set_title("Content"); axes[0].axis("off")
    axes[1].imshow(_to_rgb(st)); axes[1].set_title("Style");   axes[1].axis("off")

    # shallow only: relu1_1 (index 0)
    print("  layer ablation: shallow only")
    shallow_cfg = dict(base_cfg, _style_idxs_override=[0])
    res_shallow = _run_nst_with_custom_style_idxs(ct, st, shallow_cfg)
    axes[2].imshow(_to_rgb(res_shallow))
    axes[2].set_title("Shallow only (relu1_1)")
    axes[2].axis("off")

    # deep only: relu5_1 (index 5)
    print("  layer ablation: deep only")
    deep_cfg = dict(base_cfg, _style_idxs_override=[5])
    res_deep = _run_nst_with_custom_style_idxs(ct, st, deep_cfg)
    axes[3].imshow(_to_rgb(res_deep))
    axes[3].set_title("Deep only (relu5_1)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


def _run_nst_with_custom_style_idxs(content_t, style_t, cfg):
    """
    Thin wrapper that temporarily patches VGG19Features.STYLE_IDXS so
    we can ablate which style layers are active.
    """
    from nst import gram_matrix, total_variation
    import torch.nn.functional as F
    from torch.optim import Adam, LBFGS

    device   = content_t.device
    vgg      = VGG19Features().to(device).eval()
    override = cfg.get("_style_idxs_override", VGG19Features.STYLE_IDXS)

    with torch.no_grad():
        cf = vgg(content_t)
        sf = vgg(style_t)

    target_content = cf[VGG19Features.CONTENT_IDX].squeeze(0).detach()
    target_style   = [gram_matrix(sf[i]).detach() for i in override]

    opt_img = content_t.clone().detach().requires_grad_(True)

    cw = float(cfg.get("content_weight", 1e5))
    sw = float(cfg.get("style_weight",   3e4))
    tv = float(cfg.get("tv_weight",      1e0))
    ni = int(cfg.get("iterations",       300))

    opt = Adam([opt_img], lr=float(cfg.get("adam_lr", 1e1)))
    for _ in range(ni):
        with torch.enable_grad():
            opt.zero_grad()
            feats   = vgg(opt_img)
            c_loss  = F.mse_loss(feats[VGG19Features.CONTENT_IDX].squeeze(0), target_content)
            s_loss  = sum(F.mse_loss(gram_matrix(feats[i]), target_style[k])
                          for k, i in enumerate(override)) / float(max(len(override), 1))
            tv_loss = total_variation(opt_img)
            (cw * c_loss + sw * s_loss + tv * tv_loss).backward()
            opt.step()

    return opt_img.detach()


# ---------------------------------------------------------------------------
# 4. Matting overlay: 5 frames + alpha + cutout
# ---------------------------------------------------------------------------

def make_matting_overlay(frame_paths, model, img_size, device, out_path):
    n    = len(frame_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    cols = ["Original", "Alpha Matte", "Cutout (α·F)"]

    for r, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        alpha = _matte_frame(frame, model, img_size, device)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        alpha_rgb = (alpha * 255).astype(np.uint8)
        cutout    = (frame_rgb * alpha[:, :, np.newaxis]).astype(np.uint8)

        for c, img in enumerate([frame_rgb, alpha_rgb, cutout]):
            ax = axes[r][c] if n > 1 else axes[c]
            ax.imshow(img, cmap="gray" if c == 1 else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11, fontweight="bold")

    plt.suptitle("Matting Model: Sample Frames", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# 5. Feature maps: 8 channels from shallow + deep VGG19
# ---------------------------------------------------------------------------

def make_feature_maps(video_frame_path, seed_img_path, height, device, out_path):
    vgg = VGG19Features().to(device).eval()

    def _feats(path):
        t = _load(path, height, device)
        with torch.no_grad():
            return vgg(t)

    video_feats = _feats(video_frame_path)
    seed_feats  = _feats(seed_img_path)

    # shallow = relu1_1 (index 0), deep = relu5_1 (index 5)
    layers = [("relu1_1 (shallow)", 0), ("relu5_1 (deep)", 5)]
    images = [("Video frame", video_feats), ("Seed image", seed_feats)]

    n_channels = 8
    n_rows     = len(layers) * len(images)  # 4
    fig, axes  = plt.subplots(n_rows, n_channels, figsize=(2 * n_channels, 2.5 * n_rows))

    row = 0
    for layer_name, layer_idx in layers:
        for img_name, feats in images:
            fm = feats[layer_idx].squeeze(0).cpu().numpy()  # (C, H, W)
            for ch in range(n_channels):
                ax  = axes[row][ch]
                act = fm[ch % fm.shape[0]]
                ax.imshow(act, cmap="viridis")
                ax.axis("off")
                if ch == 0:
                    ax.set_ylabel(f"{img_name}\n{layer_name}", fontsize=7, rotation=90, labelpad=50)
            row += 1

    plt.suptitle("VGG19 Feature Maps", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# 6. Branded poster (1024×1024 crop from a stylised frame)
# ---------------------------------------------------------------------------

def make_branded_poster(stylised_frame_path, out_path):
    img = cv2.imread(stylised_frame_path)
    if img is None:
        print(f"  ✗ Cannot read {stylised_frame_path}; skipping poster.")
        return
    h, w = img.shape[:2]
    # centre-crop to square, then resize to 1024
    s    = min(h, w)
    top  = (h - s) // 2
    left = (w - s) // 2
    crop = img[top:top+s, left:left+s]
    poster = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(out_path, poster)
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t2      = cfg["task2"]
    nst_cfg = t2["nst"]
    mat_cfg = t2["matting"]
    out_dir = t2["output_dir"]
    height  = nst_cfg.get("height", 360)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize] device: {device}")

    # ── load matting model ────────────────────────────────────────────────────
    ckpt  = torch.load(mat_cfg["checkpoint"], map_location=device)
    model = UNetMatting(n_channels=3, n_classes=1).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    img_size = tuple(mat_cfg["data"]["img_size"])

    # ── paths from config ─────────────────────────────────────────────────────
    content_dir  = t2["content_dir"]   # e.g. "content/"  (5 frames)
    style_dir    = t2["style_dir"]     # e.g. "style/"    (3 paintings)
    seed_img_path = t2.get("seed_img_for_feature_maps", "")

    content_paths = sorted(Path(content_dir).glob("*.jpg")) + \
                    sorted(Path(content_dir).glob("*.png"))
    style_paths   = sorted(Path(style_dir).glob("*.jpg")) + \
                    sorted(Path(style_dir).glob("*.png"))

    content_paths = [str(p) for p in content_paths[:5]]
    style_paths   = [str(p) for p in style_paths[:3]]

    print(f"Content frames : {len(content_paths)}")
    print(f"Style images   : {len(style_paths)}")

    # ── 1. Grid ───────────────────────────────────────────────────────────────
    print("\n[1/6] Generating NST grid …")
    make_grid(
        content_paths, style_paths, nst_cfg, height, device,
        os.path.join(out_dir, "grid.png")
    )

    # ── 2. β/α ablation ──────────────────────────────────────────────────────
    print("\n[2/6] β/α ablation …")
    sweep = t2.get("style_weight_sweep", [1e3, 1e5, 1e7])
    make_beta_alpha_ablation(
        content_paths[0], style_paths[0], sweep, nst_cfg, height, device,
        os.path.join(out_dir, "beta_alpha_ablation.png")
    )

    # ── 3. Layer ablation ────────────────────────────────────────────────────
    print("\n[3/6] Layer ablation …")
    make_layer_ablation(
        content_paths[0], style_paths[0], nst_cfg, height, device,
        os.path.join(out_dir, "layer_ablation.png")
    )

    # ── 4. Matting overlay ───────────────────────────────────────────────────
    print("\n[4/6] Matting overlay …")
    make_matting_overlay(
        content_paths[:5], model, img_size, device,
        os.path.join(out_dir, "matting_overlay.png")
    )

    # ── 5. Feature maps ──────────────────────────────────────────────────────
    print("\n[5/6] Feature maps …")
    if content_paths and seed_img_path and os.path.exists(seed_img_path):
        make_feature_maps(
            content_paths[0], seed_img_path, height, device,
            os.path.join(out_dir, "feature_maps.png")
        )
    else:
        print("  Skipping feature maps (seed_img_for_feature_maps not set or missing).")

    # ── 6. Branded poster ────────────────────────────────────────────────────
    print("\n[6/6] Branded poster …")
    # Try to use the first stylised_background frame if it exists, else content
    poster_src = t2.get("branded_poster_source", content_paths[0] if content_paths else "")
    if poster_src:
        make_branded_poster(poster_src, os.path.join(out_dir, "branded_poster.png"))

    print("\nAll visualisations complete.")


if __name__ == "__main__":
    main()
