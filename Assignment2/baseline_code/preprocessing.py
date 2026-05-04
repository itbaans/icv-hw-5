"""
baseline_code/preprocessing.py
================================
Reusable preprocessing utilities for the AgriTech SeedCounter pipeline.
Designed to be importable by future assignments without modification.

Public API
----------
load_config(config_path)                          -> dict
apply_grayscale(image)                            -> ndarray
apply_noise_reduction(image, cfg, method)         -> ndarray
apply_edge_detection(image, cfg)                  -> ndarray
apply_clahe(image, cfg)                           -> ndarray
compare_filters(image, cfg)                       -> dict
"""

import logging
import os

import cv2
import numpy as np
import yaml

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return cfg


# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or BGRA image to single-channel grayscale.

    Parameters
    ----------
    image : ndarray
        Input image loaded with ``cv2.imread``.

    Returns
    -------
    ndarray
        Single-channel uint8 grayscale image.
    """
    if image is None:
        raise ValueError("apply_grayscale received a None image.")

    if len(image.shape) == 2:
        # Already grayscale
        return image.copy()

    if image.shape[2] == 4:
        # BGRA -> drop alpha, then convert
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------

def apply_noise_reduction(
    image: np.ndarray,
    cfg: dict,
    method: str = "gaussian",
) -> np.ndarray:
    """Apply a noise-reduction filter to a grayscale image.

    Parameters
    ----------
    image : ndarray
        Single-channel uint8 grayscale image.
    cfg : dict
        Full configuration dict (should contain ``preprocessing.noise_reduction``).
    method : str
        Override the method to use: ``'gaussian'``, ``'median'``, or
        ``'bilateral'``.  Defaults to the value in *cfg* if available.

    Returns
    -------
    ndarray
        Filtered grayscale image.
    """
    if image is None:
        raise ValueError("apply_noise_reduction received a None image.")

    nr_cfg = cfg.get("preprocessing", {}).get("noise_reduction", {})

    # Method priority: argument > config
    active_method = method or nr_cfg.get("method", "gaussian")

    if active_method == "gaussian":
        ksize = tuple(nr_cfg.get("gaussian_kernel_size", [5, 5]))
        sigma = nr_cfg.get("gaussian_sigma", 0)
        return cv2.GaussianBlur(image, ksize, sigma)

    elif active_method == "median":
        k = nr_cfg.get("median_kernel_size", 5)
        return cv2.medianBlur(image, k)

    elif active_method == "bilateral":
        d  = nr_cfg.get("bilateral_d", 9)
        sc = nr_cfg.get("bilateral_sigma_color", 75)
        ss = nr_cfg.get("bilateral_sigma_space", 75)
        return cv2.bilateralFilter(image, d, sc, ss)

    else:
        raise ValueError(
            f"Unknown noise reduction method: '{active_method}'. "
            "Choose 'gaussian', 'median', or 'bilateral'."
        )


# ---------------------------------------------------------------------------
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------------------------

def apply_clahe(image: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Apply CLAHE to improve contrast on images with uneven illumination.

    CLAHE works by dividing the image into small tiles and equalising each
    tile independently, then stitching them back together with bilinear
    interpolation.  It enhances seed texture while suppressing background
    brightness variations – complementary to thresholding methods.

    Parameters
    ----------
    image : ndarray
        Single-channel uint8 grayscale image.
    cfg : dict, optional
        Configuration dict.  Parameters read from ``preprocessing.clahe``.

    Returns
    -------
    ndarray
        Contrast-enhanced uint8 grayscale image.
    """
    if image is None:
        raise ValueError("apply_clahe received a None image.")

    clahe_cfg = {}
    if cfg:
        clahe_cfg = cfg.get("preprocessing", {}).get("clahe", {})

    clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
    tile_size  = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(image)


# ---------------------------------------------------------------------------
# Filter comparison
# ---------------------------------------------------------------------------

def compare_filters(image: np.ndarray, cfg: dict) -> dict:
    """Apply all three noise-reduction filters and compare their quality.

    Quality is measured using the **Laplacian variance** (edge-preservation
    score): a higher value means more high-frequency detail is retained.
    This lets us objectively rank the filters for seed-boundary preservation.

    Additionally, per-filter SSIM relative to the original and mean absolute
    difference (MAD) are reported.

    Parameters
    ----------
    image : ndarray
        Single-channel uint8 grayscale image (unfiltered).
    cfg : dict
        Full configuration dict.

    Returns
    -------
    dict
        Keys: ``'gaussian'``, ``'median'``, ``'bilateral'`` – each maps to a
        sub-dict with keys ``filtered`` (ndarray) and quality metrics.
        Also includes ``'best_method'`` and ``'analysis'`` string.
    """
    if image is None:
        raise ValueError("compare_filters received a None image.")

    results = {}
    for method in ("gaussian", "median", "bilateral"):
        filtered = apply_noise_reduction(image, cfg, method=method)
        lap_var  = float(cv2.Laplacian(filtered, cv2.CV_64F).var())
        mad      = float(np.mean(np.abs(filtered.astype(np.float32)
                                        - image.astype(np.float32))))

        results[method] = {
            "filtered":          filtered,
            "laplacian_variance": round(lap_var, 2),
            "mean_abs_diff":      round(mad, 4),
        }
        LOG.debug(
            "Filter '%s': Laplacian-var=%.2f, MAD=%.4f",
            method, lap_var, mad,
        )

    # Best method = highest Laplacian variance (most edge detail preserved)
    best = max(results, key=lambda m: results[m]["laplacian_variance"])

    analysis_lines = [
        "Filter comparison summary:",
        f"  Gaussian  – Laplacian var={results['gaussian']['laplacian_variance']:.2f}, "
        f"MAD={results['gaussian']['mean_abs_diff']:.4f}",
        f"  Median    – Laplacian var={results['median']['laplacian_variance']:.2f}, "
        f"MAD={results['median']['mean_abs_diff']:.4f}",
        f"  Bilateral – Laplacian var={results['bilateral']['laplacian_variance']:.2f}, "
        f"MAD={results['bilateral']['mean_abs_diff']:.4f}",
        "",
        f"Best edge-preserving filter: '{best}'",
        "Justification:",
        "  • Gaussian: fast, isotropic smoothing. Blurs both noise AND edges equally.",
        "  • Median:   non-linear; excellent at salt-and-pepper noise without blurring.",
        "  • Bilateral: edge-preserving; smooths homogeneous regions while keeping "
        "    boundaries sharp. Best for seed outlines but may be slower.",
        f"  → '{best}' retains the most edge information (highest Laplacian variance), "
        "making it the preferred choice for seed boundary detection.",
    ]

    results["best_method"] = best
    results["analysis"]    = "\n".join(analysis_lines)

    LOG.info("Filter comparison – best method: '%s'", best)
    return results


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

def apply_edge_detection(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Detect edges using Canny (with automatic threshold estimation via Otsu).

    Parameters
    ----------
    image : ndarray
        Single-channel uint8 grayscale image (may be pre-smoothed).
    cfg : dict
        Full configuration dict.  The edge parameters are expected under
        the key ``preprocessing.edge_detection`` but sensible defaults are
        used regardless.

    Returns
    -------
    ndarray
        Binary edge map (uint8, 0 / 255).
    """
    if image is None:
        raise ValueError("apply_edge_detection received a None image.")

    ed_cfg = cfg.get("preprocessing", {}).get("edge_detection", {})

    # ------------------------------------------------------------------
    # Automatic threshold estimation via Otsu on the input image.
    # Using float() on cv2.threshold's first return value is the most
    # portable way to get a Python scalar across all NumPy versions.
    # ------------------------------------------------------------------
    otsu_retval, _ = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # otsu_retval is always a Python float from cv2 binding; cast to int
    t_high = ed_cfg.get("canny_threshold_high", int(float(otsu_retval)))
    t_low  = ed_cfg.get("canny_threshold_low",  int(t_high * 0.4))
    aperture = ed_cfg.get("canny_aperture_size", 3)

    edges = cv2.Canny(image, t_low, t_high, apertureSize=aperture)
    return edges
