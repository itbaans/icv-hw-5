"""
nst.py - Neural Style Transfer (Gatys et al., 2015) using pretrained VGG19

Public API
----------
  run_nst(content_img, style_img, cfg) -> stylized_tensor

  content_img : (1, 3, H, W) float32 tensor in [0, 255] ImageNet-normalised
  style_img   : same format
  cfg         : dict (see config.yaml, task2.nst section)
  returns     : (1, 3, H, W) same format as input

Internal design
---------------
- VGG19 is kept frozen in eval() mode.
- Content layer  : relu4_2  (index 4 in VGG19 output, i.e. 'conv4_2')
- Style layers   : relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
- Gram matrix is normalised by (C * H * W) to be scale-invariant.
- Total-variation regularisation is added to suppress checkerboard artefacts.
- Loss: L_total = alpha * L_content + beta * L_style + tv_weight * L_TV
- Optimizer: Adam (fast) or L-BFGS (higher quality, slower).
- Temporal consistency: caller passes `init_tensor` (= previous stylised
  frame); when None, init from the content image.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
import torchvision.models as tv_models
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# ImageNet statistics (same as reference implementation)
# ---------------------------------------------------------------------------
IMAGENET_MEAN_255 = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGENET_STD_1    = torch.tensor([1.0,     1.0,    1.0   ]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Image I/O utilities
# ---------------------------------------------------------------------------

def load_img_as_tensor(img_path: str, height: int, device: torch.device) -> torch.Tensor:
    """
    Load an image from disk, resize to `height` (preserving aspect ratio),
    and return a (1, 3, H, W) float32 tensor in [0, 255], ImageNet-normalised.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize preserving aspect ratio
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    img = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)            # [0, 255]
    t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    mean = IMAGENET_MEAN_255.to(device)
    t    = (t.to(device) - mean)
    return t


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) ImageNet-normalised tensor back to a uint8 RGB
    numpy array suitable for cv2.imwrite (which expects BGR).
    """
    mean = IMAGENET_MEAN_255.to(t.device)
    img  = (t + mean).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img  = np.clip(img, 0, 255).astype(np.uint8)
    return img  # RGB


def save_tensor_as_img(t: torch.Tensor, path: str):
    img = tensor_to_uint8(t)           # RGB
    cv2.imwrite(path, img[:, :, ::-1]) # save as BGR


# ---------------------------------------------------------------------------
# VGG19 feature extractor
# ---------------------------------------------------------------------------

class VGG19Features(nn.Module):
    """
    Pretrained VGG19 with six intermediate outputs:
      relu1_1, relu2_1, relu3_1, relu4_1, conv4_2 (content), relu5_1

    Indices used in the assignment:
      Content : conv4_2  -> index 4
      Style   : 0,1,2,3,5 (relu1_1 … relu5_1)
    """

    CONTENT_IDX = 4
    STYLE_IDXS  = [0, 1, 2, 3, 5]
    LAYER_NAMES = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "conv4_2", "relu5_1"]

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        # slice the feature stack at the required checkpoints
        # VGG19 layer indices (0-based):
        #   relu1_1: 1,  relu2_1: 6,  relu3_1: 11,
        #   relu4_1: 20, conv4_2: 21, relu5_1: 29   (offset +1 for relu vs conv)
        # We use relu4_1=20, then conv4_2=21 (no relu applied), then relu5_1=29
        cuts = [1, 6, 11, 20, 21, 29]  # inclusive end of each slice
        self.slices = nn.ModuleList()
        prev = 0
        for cut in cuts:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, cut + 1)]))
            prev = cut + 1

        # Freeze weights
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for s in self.slices:
            x = s(x)
            outs.append(x)
        return outs  # list of 6 tensors


# ---------------------------------------------------------------------------
# Gram matrix & TV loss
# ---------------------------------------------------------------------------

def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, C), normalised by C*H*W."""
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    gram = feat.bmm(feat.transpose(1, 2))
    return gram / (c * h * w)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    )


# ---------------------------------------------------------------------------
# Core NST routine
# ---------------------------------------------------------------------------

def run_nst(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    cfg:            dict,
    init_tensor:    torch.Tensor | None = None,
    verbose:        bool = False,
) -> torch.Tensor:
    """
    Run the Gatys et al. style transfer optimisation.

    Parameters
    ----------
    content_tensor : (1, 3, H, W) ImageNet-normalised float32
    style_tensor   : (1, 3, H, W) same
    cfg            : task2.nst section of config.yaml
    init_tensor    : optional init (temporal consistency); if None, init
                     from content_tensor
    verbose        : print loss every 50 iterations

    Returns
    -------
    (1, 3, H, W) ImageNet-normalised float32, detached from graph
    """
    device = content_tensor.device

    vgg = VGG19Features().to(device).eval()

    # ---- target representations (computed once, no grad) ----
    with torch.no_grad():
        content_feats = vgg(content_tensor)
        style_feats   = vgg(style_tensor)

    target_content = content_feats[VGG19Features.CONTENT_IDX].squeeze(0).detach()
    target_style   = [gram_matrix(style_feats[i]).detach() for i in VGG19Features.STYLE_IDXS]

    # ---- initialise the optimising image ----
    # .contiguous() is required for L-BFGS: it calls p.grad.view(-1) internally,
    # which fails if the tensor (or its gradient) is non-contiguous (e.g. when
    # init_tensor came from AMP, a permute, or a previous detach chain).
    if init_tensor is not None:
        opt_img = init_tensor.clone().detach().contiguous().requires_grad_(True)
    else:
        opt_img = content_tensor.clone().detach().contiguous().requires_grad_(True)

    # Explicit float() cast: YAML parses `tv_weight: 1` as Python int, not float.
    # In PyTorch 2.x, `int * tensor` can fall through to tensor.__index__() and
    # raise "only integer tensors of a single element can be converted to an index".
    content_w = float(cfg.get("content_weight", 1e5))
    style_w   = float(cfg.get("style_weight",   3e4))
    tv_w      = float(cfg.get("tv_weight",      1e0))
    optimizer_name = cfg.get("optimizer", "adam").lower()
    max_iter       = int(cfg.get("iterations", 200))

    # ---- loss closure ----
    def compute_loss():
        feats = vgg(opt_img)

        # content loss
        cur_content = feats[VGG19Features.CONTENT_IDX].squeeze(0)
        c_loss = F.mse_loss(cur_content, target_content)

        # style loss — accumulate in a list then sum to avoid seeding with
        # a torch.tensor(0.0) which can break gradient tracking in some paths
        style_terms = [
            F.mse_loss(gram_matrix(feats[i]), target_style[k])
            for k, i in enumerate(VGG19Features.STYLE_IDXS)
        ]
        s_loss = sum(style_terms) / float(len(style_terms))

        tv_loss = total_variation(opt_img)

        total = content_w * c_loss + style_w * s_loss + tv_w * tv_loss
        return total, c_loss, s_loss, tv_loss


    # ---- optimise ----
    if optimizer_name == "adam":
        lr  = cfg.get("adam_lr", 1e1)
        opt = Adam([opt_img], lr=lr)
        # Adam loop — wrap each step in enable_grad so the graph is built
        # even if called from a no_grad context (e.g. video pipeline loop).
        for it in range(max_iter):
            with torch.enable_grad():
                opt.zero_grad()
                total, c, s, tv = compute_loss()
                total.backward()
                opt.step()
            if verbose and it % 50 == 0:
                print(f"  Adam iter {it:04d} | total={total.item():.2f} "
                      f"content={content_w*c.item():.2f} "
                      f"style={style_w*s.item():.2f} "
                      f"tv={tv_w*tv.item():.2f}")

    elif optimizer_name == "lbfgs":
        # NOTE: In PyTorch 2.x, optimizer.step() is decorated with @no_grad.
        # L-BFGS calls the closure from inside that no_grad context, so without
        # torch.enable_grad() the computation graph is never built and
        # total.backward() silently fails (surfaces as a cryptic index TypeError).
        opt = LBFGS([opt_img], max_iter=max_iter, line_search_fn="strong_wolfe")
        _cnt = [0]

        def closure():
            with torch.enable_grad():
                opt.zero_grad()
                total, c, s, tv = compute_loss()
                total.backward()
            if verbose and _cnt[0] % 50 == 0:
                print(f"  L-BFGS iter {_cnt[0]:04d} | total={total.item():.2f} "
                      f"content={content_w*c.item():.2f} "
                      f"style={style_w*s.item():.2f} "
                      f"tv={tv_w*tv.item():.2f}")
            _cnt[0] += 1
            return total

        opt.step(closure)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return opt_img.detach()


# ---------------------------------------------------------------------------
# Convenience: run NST from file paths
# ---------------------------------------------------------------------------

def run_nst_from_paths(
    content_path: str,
    style_path:   str,
    out_path:     str,
    cfg:          dict,
    height:       int | None = None,
    device:       torch.device | None = None,
    init_tensor:  torch.Tensor | None = None,
    verbose:      bool = False,
) -> torch.Tensor:
    """Load images from disk, run NST, save result, return tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = height or cfg.get("height", 400)
    content_t = load_img_as_tensor(content_path, h, device)
    style_t   = load_img_as_tensor(style_path,   h, device)
    result    = run_nst(content_t, style_t, cfg, init_tensor=init_tensor, verbose=verbose)
    save_tensor_as_img(result, out_path)
    return result
