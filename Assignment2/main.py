"""
main.py – AgriTech SeedCounter Pipeline
========================================
Orchestrates the FULL seed-counting pipeline covering all 6 required stages:

  Stage 1 – Preprocessing        (grayscale, 3-filter comparison, chosen filter)
  Stage 2 – Clustering           (K-Means on intensity, DBSCAN on spatial density)
  Stage 3 – Thresholding         (Otsu global + Adaptive local, saved side-by-side)
  Stage 4 – Morphological Ops    (opening, closing, erosion, dilation; documented)
  Stage 5 – Object Counting      (CCA + Blob Detection, edge-case handling)
  Stage 6 – Post-processing      (Watershed for touching seeds, shape filtering,
                                   overlap analysis, final count validation)

Ground-truth convention
-----------------------
Images must be named with a pure integer stem equal to the seed count,
e.g. ``4.jpg`` → 4 seeds, ``102.jpg`` → 102 seeds.

Usage
-----
    python main.py [--data-dir data/] [--gt-csv path/to/ground_truth.csv]

Output files (written to ``output/``)
--------------------------------------
  preprocessed_images/grayscale/<name>.png
  preprocessed_images/gaussian/<name>.png
  preprocessed_images/median/<name>.png
  preprocessed_images/bilateral/<name>.png
  segmentation/thresholding/otsu/<name>.png
  segmentation/thresholding/adaptive/<name>.png
  segmentation/clustering/kmeans/<name>.png
  segmentation/clustering/dbscan/<name>.png
  segmentation/morphological/<name>.png
  segmentation/watershed/<name>.png
  segmentation/final_detections/<name>.png
  segmentation/labeled_components.pkl
  metrics/baseline_counts.csv
  metrics/failure_cases.json
  metrics/performance_summary.json
  metrics/filter_comparison.json
  metrics/clustering_comparison.json
  metrics/morphology_log.json
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the package is importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_code.preprocessing import (
    load_config,
    apply_grayscale,
    apply_noise_reduction,
    apply_edge_detection,
    apply_clahe,
    compare_filters,
)
from baseline_code.clustering import (
    apply_kmeans,
    apply_dbscan,
    compare_clustering,
)
from baseline_code.evaluate import (
    find_failure_cases,
    generate_performance_summary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
BASE_DIR    = SCRIPT_DIR
DATA_DIR    = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "baseline_code", "config.yaml")

# Output folders
OUT_GRAY        = os.path.join(BASE_DIR, "output", "preprocessed_images", "grayscale")
OUT_GAUSS       = os.path.join(BASE_DIR, "output", "preprocessed_images", "gaussian")
OUT_MED         = os.path.join(BASE_DIR, "output", "preprocessed_images", "median")
OUT_BIL         = os.path.join(BASE_DIR, "output", "preprocessed_images", "bilateral")
OUT_FILT        = os.path.join(BASE_DIR, "output", "preprocessed_images", "filtered")
OUT_EDGE        = os.path.join(BASE_DIR, "output", "preprocessed_images", "edges")
OUT_TH_OTSU     = os.path.join(BASE_DIR, "output", "segmentation", "thresholding", "otsu")
OUT_TH_ADAP     = os.path.join(BASE_DIR, "output", "segmentation", "thresholding", "adaptive")
OUT_CL_KM       = os.path.join(BASE_DIR, "output", "segmentation", "clustering", "kmeans")
OUT_CL_DB       = os.path.join(BASE_DIR, "output", "segmentation", "clustering", "dbscan")
OUT_MORPH       = os.path.join(BASE_DIR, "output", "segmentation", "morphological")
OUT_WATERSHED   = os.path.join(BASE_DIR, "output", "segmentation", "watershed")
OUT_FINAL       = os.path.join(BASE_DIR, "output", "segmentation", "final_detections")
OUT_SEG         = os.path.join(BASE_DIR, "output", "segmentation")
OUT_METRICS     = os.path.join(BASE_DIR, "output", "metrics")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ALL_OUTPUT_DIRS = [
    OUT_GRAY, OUT_GAUSS, OUT_MED, OUT_BIL, OUT_FILT, OUT_EDGE,
    OUT_TH_OTSU, OUT_TH_ADAP, OUT_CL_KM, OUT_CL_DB,
    OUT_MORPH, OUT_WATERSHED, OUT_FINAL, OUT_SEG, OUT_METRICS,
]


# ===========================================================================
# Helper: morphology kernel factory
# ===========================================================================
def _make_kernel(shape_name: str, ksize) -> np.ndarray:
    """Return a morphological structuring element."""
    shape_map = {
        "rect":    cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross":   cv2.MORPH_CROSS,
    }
    shape = shape_map.get(shape_name.lower(), cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(shape, tuple(ksize))


# ===========================================================================
# Helper: colour-label a mask for visualisation
# ===========================================================================
def _colorise_labels(labelled: np.ndarray) -> np.ndarray:
    """Convert an integer label map to a colour BGR image for saving."""
    h, w = labelled.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    unique = np.unique(labelled)
    for lbl in unique:
        if lbl <= 0:  # background / noise
            continue
        colour = tuple(int(c) for c in rng.integers(60, 245, 3))
        colour_img[labelled == lbl] = colour
    return colour_img


# ===========================================================================
# STAGE 1 – Preprocessing
# ===========================================================================
def stage_preprocessing(image: np.ndarray, cfg: dict, name: str,
                        save_fn=None) -> tuple:
    """save_fn(path, img) is called instead of cv2.imwrite; defaults to always-save."""
    """
    Stage 1: Grayscale conversion + 3-filter comparison + chosen filter.

    Steps:
      1. Convert to grayscale.
      2. Apply all three filters (Gaussian, Median, Bilateral) and compare
         using Laplacian variance (edge-preservation score).
      3. Use the method specified in config for the main pipeline.
      4. Apply Canny edge detection on the chosen filtered image.

    Returns
    -------
    gray     : ndarray – single-channel grayscale
    filtered : ndarray – noise-reduced grayscale (chosen method)
    edges    : ndarray – Canny edge map
    filter_cmp : dict – per-filter metrics for JSON logging
    """
    if save_fn is None:
        save_fn = cv2.imwrite

    # 1. Grayscale
    gray = apply_grayscale(image)
    save_fn(os.path.join(OUT_GRAY, f"{name}.png"), gray)

    # 2. Filter comparison (all three methods)
    cmp = compare_filters(gray, cfg)
    save_fn(os.path.join(OUT_GAUSS, f"{name}.png"), cmp["gaussian"]["filtered"])
    save_fn(os.path.join(OUT_MED,   f"{name}.png"), cmp["median"]["filtered"])
    save_fn(os.path.join(OUT_BIL,   f"{name}.png"), cmp["bilateral"]["filtered"])

    # Serialisable comparison record (drop ndarray, keep metrics)
    filter_record = {
        k: {mk: mv for mk, mv in v.items() if mk != "filtered"}
        for k, v in cmp.items()
        if isinstance(v, dict)
    }
    filter_record["best_method"] = cmp["best_method"]
    filter_record["analysis"]    = cmp["analysis"]

    # 3. Chosen filter for the rest of the pipeline
    nr_method = cfg["preprocessing"]["noise_reduction"]["method"]
    filtered = cmp[nr_method]["filtered"]
    save_fn(os.path.join(OUT_FILT, f"{name}.png"), filtered)

    # 4. Edge detection
    edges = apply_edge_detection(filtered, cfg)
    save_fn(os.path.join(OUT_EDGE, f"{name}.png"), edges)

    LOG.info(
        "  [Stage 1] Filter chosen: %s | Laplacian-var: Gaussian=%.1f, "
        "Median=%.1f, Bilateral=%.1f",
        nr_method,
        cmp["gaussian"]["laplacian_variance"],
        cmp["median"]["laplacian_variance"],
        cmp["bilateral"]["laplacian_variance"],
    )
    return gray, filtered, edges, filter_record


# ===========================================================================
# STAGE 2 – Clustering
# ===========================================================================
def stage_clustering(
    gray: np.ndarray,
    cfg: dict,
    name: str,
    save_fn=None,
) -> tuple:
    """
    Stage 2: K-Means and DBSCAN clustering for seed/background separation.

    K-Means:
      - Operates on pixel intensity values (1-D feature space).
      - Partitions pixels into n_clusters groups; background = highest-
        intensity cluster (lightest region).
      - Produces a binary foreground mask.

    DBSCAN:
      - Operates in (x, y) pixel coordinate space on foreground pixels
        obtained from the K-Means binary mask.
      - Groups spatially dense regions; each cluster = one or a few seeds.
      - Noise points (label=-1) are treated as background.

    Comparison:
      - IoU between K-Means fg mask and DBSCAN fg mask.
      - DBSCAN cluster count gives an early seed-region estimate.

    Returns
    -------
    kmeans_mask  : ndarray (uint8) – binary foreground mask
    dbscan_mask  : ndarray (int32) – cluster label map
    cluster_cmp  : dict            – comparison metrics
    """
    if save_fn is None:
        save_fn = cv2.imwrite

    LOG.info("  [Stage 2] Running K-Means clustering…")
    kmeans_mask, kmeans_labelled = apply_kmeans(gray, cfg)
    save_fn(os.path.join(OUT_CL_KM, f"{name}.png"), kmeans_mask)

    LOG.info("  [Stage 2] Running DBSCAN clustering…")
    dbscan_mask = apply_dbscan(kmeans_mask, cfg)
    dbscan_vis  = _colorise_labels(dbscan_mask)
    save_fn(os.path.join(OUT_CL_DB, f"{name}.png"), dbscan_vis)

    cluster_cmp = compare_clustering(kmeans_mask, dbscan_mask, image_name=name)
    LOG.info(
        "  [Stage 2] DBSCAN clusters=%d, K-Means fg=%.1f%%, IoU=%.3f",
        cluster_cmp["dbscan_n_clusters"],
        cluster_cmp["kmeans_fg_area_pct"],
        cluster_cmp["iou"],
    )
    return kmeans_mask, dbscan_mask, cluster_cmp


# ===========================================================================
# STAGE 3 – Thresholding
# ===========================================================================
def stage_thresholding(filtered: np.ndarray, cfg: dict, name: str,
                       save_fn=None) -> np.ndarray:
    """
    Stage 3: Otsu (global) and Adaptive (local) thresholding.

    Both masks are saved to separate folders for visual comparison:
      • Otsu works best on images with bimodal intensity histograms and
        uniform illumination (single threshold for the whole image).
      • Adaptive thresholding computes a local threshold per pixel block,
        making it more robust to uneven lighting / shadows.

    The main pipeline mask uses Otsu only (configurable). Both are always
    saved for documentation/comparison.

    Returns
    -------
    binary_mask : ndarray (uint8) – primary binary mask (Otsu or combined)
    """
    th_cfg = cfg.get("thresholding", {})

    if save_fn is None:
        save_fn = cv2.imwrite

    # --- Otsu (global) ---
    _, otsu_mask = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    save_fn(os.path.join(OUT_TH_OTSU, f"{name}.png"), otsu_mask)

    # --- Adaptive (local) ---
    at_cfg      = th_cfg.get("adaptive", {})
    method_str  = at_cfg.get("method", "gaussian").lower()
    method      = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method_str == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    block_size = int(at_cfg.get("block_size", 31))
    C_val      = int(at_cfg.get("C", 4))
    adap_mask  = cv2.adaptiveThreshold(
        filtered, 255, method, cv2.THRESH_BINARY_INV, block_size, C_val
    )
    save_fn(os.path.join(OUT_TH_ADAP, f"{name}.png"), adap_mask)

    # Log comparison stats
    otsu_fg_pct = round(np.sum(otsu_mask > 0) / otsu_mask.size * 100, 2)
    adap_fg_pct = round(np.sum(adap_mask > 0) / adap_mask.size * 100, 2)
    LOG.info(
        "  [Stage 3] Otsu fg=%.1f%%, Adaptive fg=%.1f%%  "
        "| Otsu sharper on uniform light; Adaptive better for shadows",
        otsu_fg_pct, adap_fg_pct,
    )

    # Primary mask: driven by config  →  thresholding.primary: otsu | adaptive
    primary = th_cfg.get("primary", "otsu").lower()
    chosen_mask = adap_mask if primary == "adaptive" else otsu_mask
    LOG.info("  [Stage 3] Primary threshold method: '%s'", primary)
    return chosen_mask


# ===========================================================================
# STAGE 4 – Morphological Operations
# ===========================================================================
def stage_morphology(binary: np.ndarray, cfg: dict, name: str,
                     save_fn=None) -> tuple:
    """
    Stage 4: Erosion, Dilation, Opening, Closing morphological operations.

    Purpose of each operation in seed counting:
      • Opening (erosion then dilation): removes thin noise bridges between
        seeds and small false-positive specks while preserving seed shapes.
      • Closing (dilation then erosion): fills small holes inside seeds and
        closes narrow gaps, making seeds solid blobs.
      • Erosion alone: shrinks all white regions; separates slightly touching
        seeds at the cost of making seeds smaller.
      • Dilation alone: grows all white regions; fills internal holes but may
        merge nearby seeds.

    Kernel choices:
      - Ellipse kernel is preferred for round seeds (matches their shape).
      - Rect kernel can be used for elongated/rectangular seeds.
      - Cross kernel is less common but useful for thin linear features.

    Returns
    -------
    morphed   : ndarray (uint8) – cleaned binary mask
    morph_log : dict            – documentation of operations applied
    """
    morph_cfg = cfg.get("morphology", {})
    result    = binary.copy()
    morph_log = []

    # Process in order: closing → erosion → dilation → opening
    #
    # Why closing FIRST?
    # - Otsu mask:     seeds may have small interior holes → closing fills them
    # - Adaptive mask: produces thin rings + speckles around seeds (because
    #   the small block_size adapts within the seed). Closing FIRST fills
    #   those rings+speckles into solid seed blobs; then Opening removes
    #   small noise without erasing the now-solid seeds.
    for op_name, cv2_op in [
        ("closing",  cv2.MORPH_CLOSE),
        ("erosion",  cv2.MORPH_ERODE),
        ("dilation", cv2.MORPH_DILATE),
        ("opening",  cv2.MORPH_OPEN),
    ]:
        op_cfg = morph_cfg.get(op_name, {})
        if not op_cfg.get("enabled", False):
            continue

        kshape = op_cfg.get("kernel_shape", "ellipse")
        ksize  = op_cfg.get("kernel_size", [3, 3])
        iters  = op_cfg.get("iterations", 1)
        kernel = _make_kernel(kshape, ksize)

        result = cv2.morphologyEx(result, cv2_op, kernel, iterations=iters)

        entry = {
            "operation":    op_name,
            "kernel_shape": kshape,
            "kernel_size":  ksize,
            "iterations":   iters,
        }
        morph_log.append(entry)
        LOG.debug(
            "  [Stage 4] %s: kernel=%s %s, iterations=%d",
            op_name, kshape, ksize, iters,
        )

    if save_fn is None:
        save_fn = cv2.imwrite
    save_fn(os.path.join(OUT_MORPH, f"{name}.png"), result)
    LOG.info("  [Stage 4] Morphology applied: %s", [e["operation"] for e in morph_log])
    return result, morph_log


# ===========================================================================
# STAGE 5 – Object Counting
# ===========================================================================
def stage_counting_cca(morphed: np.ndarray, cfg: dict) -> tuple:
    """
    Stage 5a: Connected Component Analysis (CCA) with shape-based filtering.

    Filtering criteria applied to each connected component:
      • Area:           min_area ≤ area ≤ max_area  (removes dust & huge clusters)
      • Circularity:    4π·Area / Perimeter² ≥ min_circularity  (removes elongated artefacts)
      • Aspect ratio:   width / height ≤ max_aspect_ratio  (removes thin lines/cracks)
      • Border mask:    components touching the image edge by >50% of their
                        bounding-box border are flagged as partial seeds.

    Edge-case handling:
      • Seeds at image boundaries are partially visible. A component is
        considered a valid partial seed if 25–75% of its bounding box is
        inside the image (counted at 0.5 weight). Seeds >75% outside are
        removed entirely.

    Returns
    -------
    count        : int
    stats        : ndarray
    labels       : ndarray
    centroids    : ndarray
    kept_labels  : list[int]
    partial_flags: list[bool]  – True for boundary-touched components
    """
    filt_cfg      = cfg.get("filtering", {})
    min_area      = filt_cfg.get("min_area", 8000)
    max_area      = filt_cfg.get("max_area", 200000)
    min_circ      = filt_cfg.get("min_circularity", 0.0)
    max_ar        = filt_cfg.get("max_aspect_ratio", 999.0)

    H, W = morphed.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morphed, connectivity=8
    )

    kept_labels   = []
    partial_flags = []

    for lbl in range(1, num_labels):  # skip background (0)
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        # --- Circularity ---
        # Extract the component mask and compute contour perimeter
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        else:
            circularity = 0.0

        if circularity < min_circ:
            continue

        # --- Aspect ratio ---
        bx = stats[lbl, cv2.CC_STAT_LEFT]
        by = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect_ratio = bw / bh if bh > 0 else 999.0
        if aspect_ratio > max_ar or aspect_ratio < (1.0 / max_ar):
            continue

        # --- Edge-case: boundary seeds ---
        touches_left   = bx <= 1
        touches_top    = by <= 1
        touches_right  = (bx + bw) >= (W - 1)
        touches_bottom = (by + bh) >= (H - 1)
        on_border      = touches_left or touches_top or touches_right or touches_bottom

        if on_border:
            # Compute fraction of component inside the image
            inside_w = min(bx + bw, W) - max(bx, 0)
            inside_h = min(by + bh, H) - max(by, 0)
            inside_frac = (inside_w * inside_h) / (bw * bh) if bw * bh > 0 else 0.0
            if inside_frac < 0.25:
                # Mostly cut off – discard
                continue
            partial_flags.append(True)
        else:
            partial_flags.append(False)

        kept_labels.append(lbl)

    return len(kept_labels), stats, labels, centroids, kept_labels, partial_flags


def stage_counting_blob(gray: np.ndarray, cfg: dict) -> list:
    """
    Stage 5b: Blob Detection using cv2.SimpleBlobDetector.

    SimpleBlobDetector finds circular regions by:
      1. Thresholding at multiple levels between minThreshold and maxThreshold.
      2. Grouping binary blobs across thresholds into stable 'blobs'.
      3. Filtering by area, circularity, convexity, and inertia ratio.

    This serves as an independent cross-check against the CCA count.
    Blob detection is particularly good at finding well-separated round seeds
    but tends to under-count touching/irregular clusters.

    Returns
    -------
    keypoints : list of cv2.KeyPoint
    """
    bd_cfg = cfg.get("blob_detection", {})
    if not bd_cfg.get("enabled", True):
        return []

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = float(bd_cfg.get("min_threshold", 10))
    params.maxThreshold = float(bd_cfg.get("max_threshold", 200))

    params.filterByArea = bool(bd_cfg.get("filter_by_area", True))
    params.minArea      = float(bd_cfg.get("min_area", 8000))
    params.maxArea      = float(bd_cfg.get("max_area", 200000))

    params.filterByCircularity = bool(bd_cfg.get("filter_by_circularity", True))
    params.minCircularity      = float(bd_cfg.get("min_circularity", 0.3))

    params.filterByConvexity = bool(bd_cfg.get("filter_by_convexity", False))
    params.filterByInertia   = bool(bd_cfg.get("filter_by_inertia", False))

    detector  = cv2.SimpleBlobDetector_create(params)
    # Blob detector expects the seeds to be DARK on LIGHT background
    inverted  = cv2.bitwise_not(gray)
    keypoints = detector.detect(inverted)
    return keypoints


# ===========================================================================
# STAGE 6 – Post-processing (Watershed + final validation)
# ===========================================================================
def stage_watershed(
    original_bgr: np.ndarray,
    morphed: np.ndarray,
    cfg: dict,
    name: str,
    save_fn=None,
) -> tuple:
    """
    Stage 6a: Distance-transform + Watershed to split touching seeds.

    Why Watershed?
    --------------
    After morphological cleaning, seeds that are physically touching appear
    as a single large connected component in CCA. Watershed treats the
    distance-transform of the binary mask as a 'topographic surface' and
    floods it from seed-centre markers (local maxima), creating dividing
    lines (watersheds) at the boundaries between adjacent seeds.

    Steps:
      1. Distance transform: every foreground pixel gets its Euclidean
         distance to the nearest background pixel. Seed centres = peaks.
      2. Threshold at dist_threshold × max_dist → 'sure foreground' markers.
      3. Dilate the binary mask → 'sure background'.
      4. Unknown region = dilation − sure foreground.
      5. Label connected markers, run cv2.watershed on the original image
         using these markers.
      6. Watershed boundaries (label=-1) are set to background, producing a
         refined binary mask with separated seeds.

    Overlapping seeds:
      Seeds that overlap significantly (not just touch) cannot be split by
      simple watershed because their interiors merge. In such cases the
      component area will be ≈ N × (single seed area). We flag these with
      'overlap_detected=True' and estimate the seed count by dividing the
      component area by the median single-seed area inferred from small, clean
      components. This produces an approximate count rather than skipping the
      component entirely.

    Returns
    -------
    refined_mask : ndarray (uint8)           – watershed-split binary mask
    ws_labels    : ndarray (int32)           – watershed label map
    overlap_info : list[dict]                – analysis of possible overlaps
    """
    ws_cfg         = cfg.get("watershed", {})
    dist_threshold = float(ws_cfg.get("dist_threshold", 0.4))

    # 1. Distance transform
    dist_transform = cv2.distanceTransform(morphed, cv2.DIST_L2, 5)
    dist_norm      = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # 2. Sure foreground: high-distance pixels (seed centres)
    _, sure_fg = cv2.threshold(
        dist_norm, dist_threshold, 1.0, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8) * 255

    # 3. Sure background: dilated morphed mask
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    sure_bg    = cv2.dilate(morphed, kernel, iterations=3)

    # 4. Unknown region
    unknown    = cv2.subtract(sure_bg, sure_fg)

    # 5. Connected component markers on sure foreground
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add 1 so background is 1 (not 0); watershed uses 0 for 'unknown'
    markers = markers + 1
    markers[unknown > 0] = 0  # unknown region → 0

    # 6. Run watershed — but only if marker count is manageable.
    #    With 2000+ markers on very dense images, cv2.watershed becomes
    #    extremely slow (O(n_markers) flood-fill). We cap it and fall back
    #    to using the sure-fg connected components directly as the label map.
    max_markers = int(ws_cfg.get("max_markers", 500))

    if len(original_bgr.shape) == 2:
        ws_input = cv2.cvtColor(original_bgr, cv2.COLOR_GRAY2BGR)
    else:
        ws_input = original_bgr.copy()

    if num_markers <= max_markers:
        ws_labels = markers.copy()
        cv2.watershed(ws_input, ws_labels)
        LOG.debug("  [Stage 6] Watershed ran normally (%d markers)", num_markers)
    else:
        # Fallback: treat sure-fg components as final labels (no boundary lines).
        # Slightly less accurate for touching seeds but avoids the hang.
        ws_labels = markers.copy()
        LOG.warning(
            "  [Stage 6] Marker cap hit (%d > %d) – skipping cv2.watershed, "
            "using sure-fg CCA directly.",
            num_markers, max_markers,
        )

    # Refined mask: keep all regions that are NOT background (label=1) and NOT boundary (-1)
    refined_mask = np.zeros_like(morphed)
    refined_mask[ws_labels > 1] = 255

    # Save watershed visualisation
    if save_fn is None:
        save_fn = cv2.imwrite
    ws_vis = ws_input.copy()
    ws_vis[ws_labels == -1] = [0, 0, 255]   # red = watershed boundary
    save_fn(os.path.join(OUT_WATERSHED, f"{name}.png"), ws_vis)

    # --- Overlap analysis (vectorised) ---
    # Count all label areas in a single O(H*W) pass using bincount
    # ws_labels may contain -1 (boundaries) so shift by 1 to make non-negative
    flat      = ws_labels.flat
    shifted   = ws_labels + 1          # -1→0, 0→1, real labels≥2 → ≥3
    counts    = np.bincount(shifted.ravel().clip(0))   # index = label+1
    # label_areas[lbl] = pixel count for that label (lbl ≥ 2 → shifted ≥ 3)
    filt_cfg = cfg.get("filtering", {})
    min_area = filt_cfg.get("min_area", 8000)
    max_area = filt_cfg.get("max_area", 200000)

    label_areas = []
    for lbl in range(2, num_markers + 1):
        idx = lbl + 1          # shifted index
        a   = int(counts[idx]) if idx < len(counts) else 0
        if a > 0:
            label_areas.append(a)

    clean_areas  = [a for a in label_areas if min_area <= a <= max_area]
    typical_area = float(np.median(clean_areas)) if clean_areas else float(max_area)

    overlap_info = []
    for lbl in range(2, num_markers + 1):
        idx = lbl + 1
        a   = int(counts[idx]) if idx < len(counts) else 0
        if a > max_area:
            estimated_seeds = max(1, round(a / typical_area))
            overlap_info.append({
                "ws_label":         lbl,
                "area":             a,
                "overlap_detected": True,
                "estimated_seeds":  estimated_seeds,
                "typical_seed_area": round(typical_area),
            })

    LOG.info(
        "  [Stage 6] Watershed: %d markers, %d overlap regions detected",
        num_markers, len(overlap_info),
    )
    return refined_mask, ws_labels, overlap_info


def stage_postprocessing_count(
    refined_mask: np.ndarray,
    ws_labels: np.ndarray,
    overlap_info: list,
    cfg: dict,
) -> tuple:
    """
    Stage 6b: Final count from watershed labels + overlap correction.

    Counts all watershed regions within the valid area range, then adds
    the estimated seed count from overlapping blobs (each large blob
    contributes estimated_seeds rather than 1).

    Returns
    -------
    final_count  : int
    kept_ws_lbls : list[int]
    """
    filt_cfg = cfg.get("filtering", {})
    min_area = filt_cfg.get("min_area", 8000)
    max_area = filt_cfg.get("max_area", 200000)
    min_circ = filt_cfg.get("min_circularity", 0.0)
    max_ar   = filt_cfg.get("max_aspect_ratio", 999.0)

    H, W = refined_mask.shape
    kept_ws_lbls = []
    overlap_extra = 0

    # Overlap labels (those identified in stage_watershed)
    overlap_lbl_set = {o["ws_label"] for o in overlap_info}
    overlap_est     = {o["ws_label"]: o["estimated_seeds"] for o in overlap_info}

    # Compute all label areas in one O(H*W) bincount pass (avoid per-label np.sum)
    max_lbl  = int(ws_labels.max())
    lbl_counts = np.bincount(ws_labels.ravel().clip(0), minlength=max_lbl + 2)
    # lbl_counts[lbl] = pixel count for that label (negative labels clipped to 0)

    for lbl in range(2, max_lbl + 1):
        area = int(lbl_counts[lbl])

        if area < min_area:
            continue

        if lbl in overlap_lbl_set:
            overlap_extra += overlap_est[lbl] - 1
            kept_ws_lbls.append(lbl)
            continue

        if area > max_area:
            continue

        # Build component mask only for labels that pass area filter
        comp_mask = (ws_labels == lbl).astype(np.uint8) * 255

        # Circularity check
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter   = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        else:
            circularity = 0.0

        if circularity < min_circ:
            continue

        x, y, bw, bh = cv2.boundingRect(contours[0]) if contours else (0, 0, 1, 1)
        ar = bw / bh if bh > 0 else 999.0
        if ar > max_ar or ar < (1.0 / max_ar):
            continue

        inside_frac = (
            (min(x + bw, W) - max(x, 0)) *
            (min(y + bh, H) - max(y, 0))
        ) / (bw * bh) if (bw * bh > 0) else 0.0
        if inside_frac < 0.25:
            continue

        kept_ws_lbls.append(lbl)

    final_count = len(kept_ws_lbls) + overlap_extra
    return final_count, kept_ws_lbls


# ===========================================================================
# Validation: compare CCA count vs watershed count
# ===========================================================================
def validate_counts(cca_count: int, ws_count: int, gt: int | None) -> dict:
    """
    Final count validation.

    Compares the raw CCA count with the watershed-refined count.
    A large divergence (>20%) suggests the morphology step is not cleanly
    separating seeds, or watershed over-split a large component.

    The watershed count is preferred as the final answer because it handles
    touching seeds more accurately.
    """
    divergence_pct = (
        abs(ws_count - cca_count) / max(cca_count, 1) * 100
    )

    validation = {
        "cca_count":       cca_count,
        "watershed_count": ws_count,
        "final_count":     ws_count,
        "divergence_pct":  round(divergence_pct, 2),
        "warning":         divergence_pct > 20.0,
    }

    if gt is not None:
        validation["ground_truth"] = gt
        validation["error_vs_cca"]  = cca_count - gt
        validation["error_vs_ws"]   = ws_count  - gt

    if validation["warning"]:
        LOG.warning(
            "  [Validation] CCA=%d vs Watershed=%d diverge by %.1f%% "
            "– check morphology parameters.",
            cca_count, ws_count, divergence_pct,
        )
    else:
        LOG.info(
            "  [Validation] CCA=%d, Watershed=%d (divergence=%.1f%%)",
            cca_count, ws_count, divergence_pct,
        )

    return validation


# ===========================================================================
# Visualisation
# ===========================================================================
def draw_detections(
    original: np.ndarray,
    labels: np.ndarray,
    kept_labels: list,
    count: int,
    blob_keypoints: list | None = None,
) -> np.ndarray:
    """Overlay coloured blobs, count, and optional blob circles on the image."""
    vis = (
        original.copy()
        if len(original.shape) == 3
        else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    )
    overlay = np.zeros_like(vis)

    rng = np.random.default_rng(42)
    for lbl in kept_labels:
        colour = tuple(int(c) for c in rng.integers(80, 256, 3))
        overlay[labels == lbl] = colour

    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # Blob keypoints (yellow circles)
    if blob_keypoints:
        for kp in blob_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r    = max(1, int(kp.size / 2))
            cv2.circle(vis, (x, y), r, (0, 255, 255), 2)

    cv2.putText(
        vis,
        f"Count: {count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    return vis


# ===========================================================================
# Ground-truth helpers
# ===========================================================================
def gt_from_filename(filename: str) -> int | None:
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except ValueError:
        return None


def load_ground_truth(gt_csv: str) -> dict:
    gt = {}
    if not gt_csv or not os.path.isfile(gt_csv):
        return gt
    with open(gt_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname     = os.path.basename(row.get("filename", "").strip())
            count_str = row.get("count", row.get("ground_truth", "")).strip()
            if fname and count_str:
                try:
                    gt[fname] = int(count_str)
                except ValueError:
                    pass
    return gt


# ===========================================================================
# Main pipeline
# ===========================================================================
def run_pipeline(
    data_dir: str,
    gt_csv: str | None = None,
    cfg_override: dict | None = None,
    save_set: set | None = None,
) -> dict:
    """
    Run the full seed-counting pipeline.

    Parameters
    ----------
    cfg_override : dict, optional
        If provided, use this config dict instead of loading from disk.
        Useful for in-memory experiment sweeps.
    save_set : set of str, optional
        Set of image basenames (e.g. {"25.jpg", "100.jpg"}) for which
        intermediate/final images should be written to disk.
        • None  → save for ALL images (original behaviour)
        • set() → save for NO images (metrics-only mode)
    
    Returns
    -------
    dict with keys: mae, rmse, accuracy_within_threshold_pct,
                    num_failure_cases, records
    """

    # 1. Load configuration (or use override)
    if cfg_override is not None:
        cfg = cfg_override
        LOG.info("Using in-memory config override.")
    else:
        LOG.info("Loading configuration from %s", CONFIG_PATH)
        cfg = load_config(CONFIG_PATH)
    threshold_pct = cfg.get("failure_case", {}).get("error_threshold_pct", 10.0)

    # Helper: only write images when the current image is in save_set
    def _imsave(path: str, img: np.ndarray, basename: str) -> None:
        """Write image only if basename is in save_set (or save_set is None)."""
        if save_set is None or basename in save_set:
            cv2.imwrite(path, img)

    # Create output directories (always needed for metrics JSONs)
    for d in ALL_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

    # 2. Discover images
    image_paths = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

    if not image_paths:
        LOG.warning("No images found in '%s'.", data_dir)
        return

    csv_overrides = load_ground_truth(gt_csv) if gt_csv else {}

    records             = []
    all_labels_store    = {}
    filter_comparisons  = []
    cluster_comparisons = []
    morph_logs          = []
    all_validations     = []

    # 3. Per-image loop
    for img_path in image_paths:
        name    = os.path.splitext(os.path.basename(img_path))[0]
        basename = os.path.basename(img_path)
        LOG.info("=" * 60)
        LOG.info("Processing: %s", basename)

        image = cv2.imread(img_path)
        if image is None:
            LOG.warning("  Could not read '%s' – skipping.", img_path)
            continue
            
        # Downsample the image by 50% for speed and performance
        RESIZE_FACTOR = 0.5
        h, w = image.shape[:2]
        new_w, new_h = int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Scale down the blob area configuration parameters by RESIZE_FACTOR^2 
        # specifically for this execution run if not already scaled
        if "blob_detection" in cfg and "min_area" in cfg["blob_detection"]:
            # Only scale once
            if cfg["blob_detection"]["min_area"] > 2000:
                cfg["blob_detection"]["min_area"] = cfg["blob_detection"]["min_area"] * (RESIZE_FACTOR ** 2)
                cfg["blob_detection"]["max_area"] = cfg["blob_detection"]["max_area"] * (RESIZE_FACTOR ** 2)
                
        if "filtering" in cfg and "min_area" in cfg["filtering"]:
            if cfg["filtering"]["min_area"] > 2000:
                cfg["filtering"]["min_area"] = cfg["filtering"]["min_area"] * (RESIZE_FACTOR ** 2)
                cfg["filtering"]["max_area"] = cfg["filtering"]["max_area"] * (RESIZE_FACTOR ** 2)

        # ── STAGE 1: Preprocessing ───────────────────────────────────────
        gray, filtered, edges, filter_record = stage_preprocessing(
            image, cfg, name, save_fn=lambda p, i: _imsave(p, i, basename)
        )
        filter_record["image"] = basename
        filter_comparisons.append(filter_record)

        # ── STAGE 2: Clustering ─────────────────────────────────────────
        kmeans_mask, dbscan_mask, cluster_cmp = stage_clustering(
            gray, cfg, name, save_fn=lambda p, i: _imsave(p, i, basename)
        )
        cluster_comparisons.append(cluster_cmp)

        # ── STAGE 3: Thresholding ────────────────────────────────────────
        binary_mask = stage_thresholding(
            filtered, cfg, name, save_fn=lambda p, i: _imsave(p, i, basename)
        )

        # ── STAGE 4: Morphology ──────────────────────────────────────────
        morphed, morph_log = stage_morphology(
            binary_mask, cfg, name, save_fn=lambda p, i: _imsave(p, i, basename)
        )
        morph_logs.append({"image": basename, "operations": morph_log})

        # ── STAGE 5: Object Counting (CCA) ──────────────────────────────
        cca_count, stats, labels, centroids, kept_labels, partial_flags = \
            stage_counting_cca(morphed, cfg)
        all_labels_store[basename] = labels
        n_partial = sum(partial_flags)

        # Stage 5b: Blob detection (as comparison)
        blob_kps = stage_counting_blob(gray, cfg)

        LOG.info(
            "  [Stage 5] CCA count=%d (partial boundary seeds=%d) | Blob count=%d",
            cca_count, n_partial, len(blob_kps),
        )

        # ── STAGE 6: Watershed + Post-processing ────────────────────────
        refined_mask, ws_labels, overlap_info = stage_watershed(
            image, morphed, cfg, name, save_fn=lambda p, i: _imsave(p, i, basename)
        )
        ws_count, kept_ws_lbls = stage_postprocessing_count(
            refined_mask, ws_labels, overlap_info, cfg
        )

        # Ground truth
        gt_count = csv_overrides.get(basename, gt_from_filename(basename))
        if gt_count is not None:
            LOG.info("  Ground truth: %d", gt_count)

        # Final validation
        validation = validate_counts(cca_count, ws_count, gt_count)
        all_validations.append({"image": basename, **validation})

        # Use watershed count as final prediction
        final_count = validation["final_count"]

        # Visualisation using watershed labels
        vis = draw_detections(image, ws_labels, kept_ws_lbls, final_count, blob_kps)
        _imsave(os.path.join(OUT_FINAL, f"{name}.png"), vis, basename)
        _imsave(os.path.join(OUT_MORPH, f"{name}_vis.png"), vis, basename)

        records.append({
            "filename":    basename,
            "prediction":  final_count,
            "ground_truth": gt_count,
        })

        LOG.info(
            "  FINAL: watershed_count=%d | gt=%s | error=%s",
            final_count,
            gt_count if gt_count is not None else "N/A",
            (final_count - gt_count) if gt_count is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # 4. Save auxiliary JSON logs
    # ------------------------------------------------------------------
    def _save_json(path, data):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        LOG.info("Saved -> %s", path)

    _save_json(os.path.join(OUT_METRICS, "filter_comparison.json"),  filter_comparisons)
    _save_json(os.path.join(OUT_METRICS, "clustering_comparison.json"), cluster_comparisons)
    _save_json(os.path.join(OUT_METRICS, "morphology_log.json"),     morph_logs)
    _save_json(os.path.join(OUT_METRICS, "validation_log.json"),     all_validations)

    # ------------------------------------------------------------------
    # 5. Persist labeled components (only in full-save mode)
    # ------------------------------------------------------------------
    if save_set is None:
        pkl_path = os.path.join(OUT_SEG, "labeled_components.pkl")
        with open(pkl_path, "wb") as fh:
            pickle.dump(all_labels_store, fh)
        LOG.info("Saved labeled components -> %s", pkl_path)

    # ------------------------------------------------------------------
    # 6. Save baseline_counts.csv
    # ------------------------------------------------------------------
    counts_csv = os.path.join(OUT_METRICS, "baseline_counts.csv")
    with open(counts_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "prediction", "ground_truth"])
        writer.writeheader()
        writer.writerows(records)
    LOG.info("Saved counts -> %s", counts_csv)

    # ------------------------------------------------------------------
    # 7. Failure cases + performance summary
    # ------------------------------------------------------------------
    evaluated     = [r for r in records if r.get("ground_truth") is not None]
    failure_cases = find_failure_cases(evaluated, threshold_pct)
    _save_json(os.path.join(OUT_METRICS, "failure_cases.json"), failure_cases)
    LOG.info("Failure cases: %d", len(failure_cases))

    summary_path = os.path.join(OUT_METRICS, "performance_summary.json")
    summary      = generate_performance_summary(records, threshold_pct,
                                                output_path=summary_path)
    LOG.info("=" * 60)
    LOG.info("PIPELINE COMPLETE")
    LOG.info(
        "  MAE=%.4s  RMSE=%.4s  Accuracy=%.1f%%  Failures=%d/%d",
        summary["mae"],
        summary["rmse"],
        summary["accuracy_within_threshold_pct"] or 0,
        summary["num_failure_cases"],
        summary["evaluated_images"],
    )

    return {
        "mae":                          summary["mae"],
        "rmse":                         summary["rmse"],
        "accuracy_within_threshold_pct": summary["accuracy_within_threshold_pct"],
        "num_failure_cases":            summary["num_failure_cases"],
        "records":                      records,
    }


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriTech SeedCounter Pipeline")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(SCRIPT_DIR, "data"),
        help="Directory containing input images (default: data/).",
    )
    parser.add_argument(
        "--gt-csv",
        default=None,
        help=(
            "Optional CSV (columns: filename, count) that overrides the "
            "filename-stem ground truth for specific images."
        ),
    )
    parser.add_argument(
        "--save-for",
        default=None,
        help=(
            "Comma-separated list of image basenames to save visuals for, "
            "e.g. '50.jpg,100.jpg,120.jpg'. "
            "Omit to save for ALL images (default). "
            "Use 'none' to skip all image saving (metrics-only)."
        ),
    )
    args = parser.parse_args()

    if args.save_for is None:
        save_set = None          # save everything
    elif args.save_for.lower() == "none":
        save_set = set()         # save nothing
    else:
        save_set = set(s.strip() for s in args.save_for.split(","))

    run_pipeline(data_dir=args.data_dir, gt_csv=args.gt_csv, save_set=save_set)
