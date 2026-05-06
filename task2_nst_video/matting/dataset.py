"""
dataset.py - AISegment Human Matting Dataset Loader

Loads image/alpha-matte pairs from the AISegment dataset.
Expected directory layout (after Kaggle download):

  data/
    clip_img/
      XXXXX/
        clip_00000000/
          *.jpg
    matting/
      XXXXX/
        matting_00000000/
          *.png

This loader scans both trees, pairs them by matching relative paths, and
applies augmentation (flip, colour jitter, random crop) as required.
The subsampling strategy (5000 train / 500 val / 500 test) is controlled
by config.yaml and logged there.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_pairs(clip_root: Path, matte_root: Path):
    """Walk clip_root and find the matching matte for every image."""
    pairs = []
    for img_path in sorted(clip_root.rglob("*.jpg")):
        # Build the expected matte path: replace clip_root prefix with
        # matte_root and change extension to .png
        rel = img_path.relative_to(clip_root)
        # AISegment naming: clip_img/XXXX/clip_YYYY/image.jpg
        #                   matting/XXXX/matting_YYYY/image.png
        parts = list(rel.parts)
        if len(parts) >= 2:
            parts[1] = parts[1].replace("clip_", "matting_")
        matte_path = matte_root / Path(*parts).with_suffix(".png")
        if matte_path.exists():
            pairs.append((str(img_path), str(matte_path)))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AISegmentDataset(Dataset):
    """
    Parameters
    ----------
    clip_root  : path to  data/clip_img/
    matte_root : path to  data/matting/
    split      : 'train' | 'val' | 'test'
    img_size   : (H, W) to resize every sample to
    n_train    : maximum training samples   (default 5000)
    n_val      : maximum validation samples (default 500)
    n_test     : maximum test samples       (default 500)
    seed       : RNG seed for reproducible subsampling
    """

    def __init__(
        self,
        clip_root: str,
        matte_root: str,
        split: str = "train",
        img_size: tuple = (256, 256),
        n_train: int = 5000,
        n_val: int = 500,
        n_test: int = 500,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.img_size = img_size  # (H, W)

        # ---- collect & shuffle all pairs ----
        all_pairs = _collect_pairs(Path(clip_root), Path(matte_root))
        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        # ---- deterministic train / val / test split ----
        total = n_train + n_val + n_test
        all_pairs = all_pairs[:total]
        train_pairs = all_pairs[:n_train]
        val_pairs   = all_pairs[n_train : n_train + n_val]
        test_pairs  = all_pairs[n_train + n_val : n_train + n_val + n_test]

        self.pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[split]

        # ---- augmentation ----
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path, matte_path = self.pairs[idx]

        # --- load ---
        img       = cv2.imread(img_path,   cv2.IMREAD_COLOR)      # BGR  uint8  H×W×3
        matte_raw = cv2.imread(matte_path, cv2.IMREAD_UNCHANGED)  # BGRA uint8  H×W×4

        if img is None or matte_raw is None:
            raise FileNotFoundError(f"Could not read:\n  {img_path}\n  {matte_path}")

        # AISegment mattes are RGBA .png files.
        # The actual alpha matte is channel index 3 (A channel).
        # DO NOT use IMREAD_GRAYSCALE — that gives the luminance blend of RGB,
        # not the matte. Per the dataset author's README:
        #   in_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #   alpha    = in_image[:, :, 3]
        if matte_raw.ndim == 2:
            # Fallback: already grayscale (some files in the dataset are plain masks)
            matte = matte_raw
        elif matte_raw.shape[2] == 4:
            matte = matte_raw[:, :, 3]   # extract alpha channel
        else:
            # 3-channel PNG: treat luminance as matte (shouldn't happen normally)
            matte = cv2.cvtColor(matte_raw, cv2.COLOR_BGR2GRAY)

        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- resize ---
        H, W = self.img_size
        img   = cv2.resize(img,   (W, H), interpolation=cv2.INTER_LINEAR)
        matte = cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)

        # --- to tensors (float32, [0,1]) ---
        img   = torch.from_numpy(img.astype(np.float32)   / 255.0).permute(2, 0, 1)  # C,H,W
        matte = torch.from_numpy(matte.astype(np.float32) / 255.0).unsqueeze(0)       # 1,H,W

        # --- training augmentation ---
        if self.split == "train":
            # Random horizontal flip
            if random.random() > 0.5:
                img   = TF.hflip(img)
                matte = TF.hflip(matte)

            # Colour jitter (image only)
            img = self.color_jitter(img)

            # Random crop (crop to 90% area, then resize back)
            crop_frac = random.uniform(0.85, 1.0)
            ch = int(H * crop_frac)
            cw = int(W * crop_frac)
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            img   = TF.resized_crop(img,   top, left, ch, cw, (H, W))
            matte = TF.resized_crop(matte, top, left, ch, cw, (H, W))

        return img, matte


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a config dict.

    Expected cfg keys
    -----------------
    data.clip_root, data.matte_root,
    data.img_size (list [H, W]),
    data.n_train, data.n_val, data.n_test,
    training.seed,
    training.batch_size, training.num_workers
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = AISegmentDataset(
            clip_root  = cfg["data"]["clip_root"],
            matte_root = cfg["data"]["matte_root"],
            split      = split,
            img_size   = tuple(cfg["data"]["img_size"]),
            n_train    = cfg["data"]["n_train"],
            n_val      = cfg["data"]["n_val"],
            n_test     = cfg["data"]["n_test"],
            seed       = cfg["training"]["seed"],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = cfg["training"]["batch_size"],
            shuffle     = (split == "train"),
            num_workers = cfg["training"]["num_workers"],
            pin_memory  = True,
            drop_last   = (split == "train"),
        )
    return loaders
