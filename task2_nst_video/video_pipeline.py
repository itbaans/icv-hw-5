"""
video_pipeline.py - End-to-end video stylisation pipeline

Usage
-----
    python video_pipeline.py --config config.yaml

Workflow
--------
1. Decode input_video.mp4 into individual frames (OpenCV).
2. For each frame:
   a. Run the matting model → alpha matte α_t  (float [0,1])
   b. Run NST              → stylised version S_t
      - First frame:  init from content frame
      - Subsequent frames: init from previous stylised frame
        (temporal consistency trick → reduces flicker at zero extra cost)
3. Composite three output variants per frame:
      bg_stylised : O = α * F + (1−α) * S   (subject natural)
      fg_stylised : O = α * S + (1−α) * F   (background natural)
      full        : O = S                    (full-frame stylised)
4. Re-encode each variant back to MP4 at the original frame rate.

Output files (written to outputs/ directory):
    stylized_background.mp4
    stylized_subject.mp4
    stylized_full.mp4
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ── importable from task2 root ───────────────────────────────────────────────
TASK2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK2_ROOT))

from matting.model import UNetMatting
from nst import (
    run_nst,
    load_img_as_tensor,
    tensor_to_uint8,
    IMAGENET_MEAN_255,
)


# ---------------------------------------------------------------------------
# Frame I/O helpers
# ---------------------------------------------------------------------------

def frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an OpenCV BGR uint8 frame to a (1, 3, H, W) ImageNet-normalised
    float32 tensor ready for VGG19 / NST.
    """
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    t    = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = IMAGENET_MEAN_255.to(device)
    return t - mean


def tensor_to_frame(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) ImageNet-normalised tensor to BGR uint8 frame.
    """
    mean = IMAGENET_MEAN_255.to(t.device)
    rgb  = (t + mean).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    rgb  = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def matte_frame(frame_bgr: np.ndarray, model: UNetMatting,
                img_size: tuple, device: torch.device) -> np.ndarray:
    """
    Run the matting model on one frame.

    Parameters
    ----------
    frame_bgr : (H, W, 3) uint8 BGR frame
    model     : loaded UNetMatting (eval mode)
    img_size  : (H, W) the model was trained on
    device    : torch device

    Returns
    -------
    alpha : (H_orig, W_orig) float32 array in [0, 1]
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    mh, mw = img_size

    # resize → model input
    resized = cv2.resize(frame_bgr, (mw, mh), interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t       = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        alpha_small = model(t)                          # (1, 1, mh, mw)

    alpha_small = alpha_small.squeeze().cpu().numpy()  # (mh, mw)
    # resize back to original frame dimensions
    alpha = cv2.resize(alpha_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return alpha.astype(np.float32)


def composite(frame_bgr: np.ndarray, stylised_bgr: np.ndarray,
              alpha: np.ndarray, mode: str) -> np.ndarray:
    """
    Composite original frame F and stylised frame S using alpha matte.

    mode
    ----
    'background' : O = α·F + (1−α)·S   (background stylised, subject natural)
    'subject'    : O = α·S + (1−α)·F   (subject stylised, background natural)
    'full'       : O = S
    """
    a = alpha[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
    F = frame_bgr.astype(np.float32)
    S = stylised_bgr.astype(np.float32)

    if mode == "background":
        out = a * F + (1 - a) * S
    elif mode == "subject":
        out = a * S + (1 - a) * F
    elif mode == "full":
        out = S
    else:
        raise ValueError(f"Unknown composite mode: {mode}")

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ffmpeg encode helper
# ---------------------------------------------------------------------------

def encode_video(frame_dir: str, out_path: str, fps: float):
    """
    Encode frames stored as %06d.png in frame_dir to out_path using ffmpeg.
    Falls back to OpenCV VideoWriter if ffmpeg is not found.
    """
    pattern = os.path.join(frame_dir, "%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✓ Saved {out_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ffmpeg failed ({e}), falling back to OpenCV writer.")
        _encode_opencv(frame_dir, out_path, fps)


def _encode_opencv(frame_dir: str, out_path: str, fps: float):
    frames = sorted(Path(frame_dir).glob("*.png"))
    if not frames:
        print("  No frames to encode.")
        return
    sample = cv2.imread(str(frames[0]))
    h, w   = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fp in frames:
        writer.write(cv2.imread(str(fp)))
    writer.release()
    print(f"  ✓ Saved {out_path} (OpenCV fallback)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict):
    task2_cfg = cfg["task2"]
    nst_cfg   = task2_cfg["nst"]
    mat_cfg   = task2_cfg["matting"]
    out_dir   = task2_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[video_pipeline] device: {device}")

    # ── load matting model ────────────────────────────────────────────────────
    mat_ckpt = mat_cfg["checkpoint"]
    print(f"Loading matting model from {mat_ckpt} …")
    ckpt  = torch.load(mat_ckpt, map_location=device)
    model = UNetMatting(n_channels=3, n_classes=1).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    img_size = tuple(mat_cfg["data"]["img_size"])  # (H, W)

    # ── load style image once ─────────────────────────────────────────────────
    style_path = task2_cfg["style_image"]
    height     = nst_cfg.get("height", 360)
    style_t    = load_img_as_tensor(style_path, height, device)
    print(f"Style image: {style_path}")

    # ── open input video ──────────────────────────────────────────────────────
    video_path = task2_cfg["input_video"]
    cap        = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  |  FPS={fps:.2f}  |  frames={n_frames}")

    # ── temp directories for each variant ─────────────────────────────────────
    tmp_bg  = tempfile.mkdtemp(prefix="nst_bg_")
    tmp_fg  = tempfile.mkdtemp(prefix="nst_fg_")
    tmp_full = tempfile.mkdtemp(prefix="nst_full_")

    prev_stylised: torch.Tensor | None = None
    frame_idx = 0
    max_frames = int(nst_cfg.get("max_frames", 0))  # 0 = process all
    iter_default = int(nst_cfg.get("iterations",       200))
    iter_first   = int(nst_cfg.get("iterations_first", iter_default))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_idx >= max_frames:
            print(f"  max_frames={max_frames} reached — stopping early.")
            break

        print(f"  Frame {frame_idx + 1}/{min(n_frames, max_frames) if max_frames else n_frames}", end="\r")

        # ── resize frame to NST height ────────────────────────────────────────
        h_orig, w_orig = frame.shape[:2]
        new_w = int(w_orig * height / h_orig)
        frame_r = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_LINEAR)

        # ── alpha matte ───────────────────────────────────────────────────────
        alpha = matte_frame(frame_r, model, img_size, device)

        # ── NST ───────────────────────────────────────────────────────────────
        content_t = frame_to_tensor(frame_r, device)
        # use more iterations for frame 0 (no temporal init yet)
        frame_cfg = dict(nst_cfg)
        frame_cfg["iterations"] = iter_first if frame_idx == 0 else iter_default
        stylised  = run_nst(
            content_tensor = content_t,
            style_tensor   = style_t,
            cfg            = frame_cfg,
            init_tensor    = prev_stylised,
            verbose        = False,
        )
        prev_stylised = stylised

        stylised_frame = tensor_to_frame(stylised)

        # ── composite ─────────────────────────────────────────────────────────
        bg_frame   = composite(frame_r, stylised_frame, alpha, "background")
        fg_frame   = composite(frame_r, stylised_frame, alpha, "subject")
        full_frame = composite(frame_r, stylised_frame, alpha, "full")

        name = f"{frame_idx:06d}.png"
        cv2.imwrite(os.path.join(tmp_bg,   name), bg_frame)
        cv2.imwrite(os.path.join(tmp_fg,   name), fg_frame)
        cv2.imwrite(os.path.join(tmp_full, name), full_frame)

        frame_idx += 1

    cap.release()
    print(f"\nProcessed {frame_idx} frames. Encoding videos …")

    # ── encode ────────────────────────────────────────────────────────────────
    encode_video(tmp_bg,   os.path.join(out_dir, "stylized_background.mp4"), fps)
    encode_video(tmp_fg,   os.path.join(out_dir, "stylized_subject.mp4"),    fps)
    encode_video(tmp_full, os.path.join(out_dir, "stylized_full.mp4"),       fps)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 2 video stylisation pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
