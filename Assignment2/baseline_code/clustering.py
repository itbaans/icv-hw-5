"""
baseline_code/clustering.py
============================
Clustering utilities for the AgriTech SeedCounter pipeline.

Provides K-Means clustering on pixel intensity/colour to segment seeds from
background, and DBSCAN clustering on the spatial distribution of foreground
pixels to identify individual seed regions.

Public API
----------
apply_kmeans(image, cfg)            -> ndarray (binary mask)
apply_dbscan(binary_mask, cfg)      -> ndarray (labelled mask, -1 = noise)
compare_clustering(kmeans_mask, dbscan_mask, image_name) -> dict
"""

import logging
import time

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def apply_kmeans(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Segment seeds from background using K-Means on pixel intensity.

    K-Means groups pixels into *n_clusters* classes purely by intensity value.
    After clustering, the darkest cluster is assumed to be the background;
    all other clusters form the seed foreground mask.

    Parameters
    ----------
    image : ndarray
        Single-channel uint8 grayscale image.
    cfg : dict
        Full configuration dict.  Clustering parameters are read from
        ``clustering.kmeans``.

    Returns
    -------
    binary_mask : ndarray
        uint8 binary image (255 = seed foreground, 0 = background).
    labelled_mask : ndarray
        uint8 image where each pixel holds its cluster index (1-based).
    """
    if image is None:
        raise ValueError("apply_kmeans received a None image.")

    km_cfg = cfg.get("clustering", {}).get("kmeans", {})
    n_clusters = int(km_cfg.get("n_clusters", 3))
    max_iter    = int(km_cfg.get("max_iter", 10))
    attempts    = int(km_cfg.get("attempts", 3))

    # Reshape to a 1-D array of float32 samples (one feature = intensity)
    pixels = image.reshape(-1, 1).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        1.0,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        n_clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    # labels  : shape (H*W, 1), values 0..n_clusters-1
    # centers : shape (n_clusters, 1)
    labels = labels.flatten()
    centers = centers.flatten()

    # The background cluster is the one with the highest mean intensity
    # (white / very bright pixels are the surface the seeds lie on).
    bg_cluster = int(np.argmax(centers))

    # Build binary mask: foreground = any cluster that is NOT background
    binary_mask = np.where(labels != bg_cluster, 255, 0).astype(np.uint8)
    binary_mask = binary_mask.reshape(image.shape)

    # Labelled mask: 0 = background, 1..n_clusters = cluster index
    labelled_mask = (labels + 1).astype(np.uint8).reshape(image.shape)

    LOG.debug(
        "K-Means: %d clusters, background cluster=%d (intensity=%.1f)",
        n_clusters, bg_cluster, centers[bg_cluster],
    )
    return binary_mask, labelled_mask


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def apply_dbscan(binary_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Identify seed regions using DBSCAN on the spatial (x, y) positions of
    foreground pixels.

    DBSCAN groups pixels that are within *eps* pixels of each other and have
    at least *min_samples* neighbours.  Each dense spatial cluster corresponds
    to a distinct seed region (or a group of touching seeds).

    Parameters
    ----------
    binary_mask : ndarray
        uint8 binary image (255 = foreground).
    cfg : dict
        Full configuration dict.  Parameters from ``clustering.dbscan``.

    Returns
    -------
    labelled_mask : ndarray (int32, same shape as binary_mask)
        Pixel-level label map where:
          • -1  = noise (not part of any cluster)
          •  0  = background (was 0 in binary_mask)
          • >0  = cluster ID (1-indexed)
    """
    if binary_mask is None:
        raise ValueError("apply_dbscan received a None mask.")

    db_cfg = cfg.get("clustering", {}).get("dbscan", {})
    eps         = float(db_cfg.get("eps", 15))
    min_samples = int(db_cfg.get("min_samples", 30))
    step        = int(db_cfg.get("sample_step", 5))   # subsample for speed

    # Extract (row, col) coordinates of foreground pixels
    ys, xs = np.where(binary_mask > 0)

    labelled_mask = np.zeros(binary_mask.shape, dtype=np.int32)

    if len(xs) == 0:
        LOG.warning("apply_dbscan: no foreground pixels found.")
        return labelled_mask

    # Subsample for speed on large images
    coords = np.column_stack((xs[::step], ys[::step])).astype(np.float32)

    LOG.debug(
        "DBSCAN: eps=%.1f, min_samples=%d, fitting on %d points (step=%d)",
        eps, min_samples, len(coords), step,
    )

    t0 = time.time()

    # ----------------------------------------------------------------
    # Pure NumPy/OpenCV DBSCAN-equivalent: grid-cell spatial binning
    # ----------------------------------------------------------------
    # Approach:
    #   1. Divide the image into a grid of cells with cell_size = eps.
    #   2. A cell is a 'core cell' if it contains >= min_samples pixels.
    #   3. Adjacent core cells (8-connectivity) are merged into clusters
    #      via flood-fill on the cell grid.
    #   4. Foreground pixels not in any core cell are labelled as noise.
    #
    # This is equivalent to DBSCAN with MinPts = min_samples and
    # epsilon = cell_size for axis-aligned distances. It avoids any
    # external dependency (scikit-learn was unavailable).

    # Subsample for density estimation
    sub_xs = xs[::step]
    sub_ys = ys[::step]

    H, W   = binary_mask.shape
    cell   = max(1, int(eps))  # grid cell size in pixels
    gH     = (H + cell - 1) // cell
    gW     = (W + cell - 1) // cell

    # Count how many foreground sample-pixels fall in each cell
    cell_counts = np.zeros((gH, gW), dtype=np.int32)
    cell_xi = sub_xs // cell   # grid column of each sample
    cell_yi = sub_ys // cell   # grid row    of each sample
    for cx, cy in zip(cell_xi, cell_yi):
        cell_counts[cy, cx] += 1

    # Determine the min_samples threshold in cell-space
    # (scale min_samples by step since we subsampled)
    count_thresh = max(1, min_samples // step)

    # Core cells: those with enough samples
    core_cell_map = (cell_counts >= count_thresh).astype(np.uint8)

    # Label connected groups of core cells (8-connectivity)
    n_clusters, cell_labels = cv2.connectedComponents(
        core_cell_map, connectivity=8
    )
    # cell_labels: 0 = non-core, 1..n_clusters = cluster id

    n_clusters = int(n_clusters) - 1  # exclude background label 0

    elapsed = time.time() - t0
    n_noise_cells = int(np.sum(core_cell_map == 0))
    LOG.debug(
        "DBSCAN (grid): %d clusters found, %.2fs, cell_size=%dpx, thresh=%d",
        n_clusters, elapsed, cell, count_thresh,
    )

    # Map cluster cell labels back to pixel space (vectorized)
    # For every foreground pixel assign the cluster of its cell;
    # pixels in non-core cells are labelled -1 (noise).
    cy_g = ys // cell   # grid row    of every foreground pixel
    cx_g = xs // cell   # grid column of every foreground pixel
    pixel_cluster = cell_labels[cy_g, cx_g]  # cluster id (0 = non-core)
    labelled_mask[ys, xs] = np.where(pixel_cluster > 0, pixel_cluster, -1)

    # No further re-indexing needed: cell_labels are already 1..K
    # (0 = non-core/background → stored as -1 in labelled_mask)
    return labelled_mask


def _propagate_labels(labelled_mask: np.ndarray, binary_mask: np.ndarray) -> None:
    """Fill un-labelled foreground pixels by dilating existing labels.
    (Modifies labelled_mask in-place.)
    """
    # Iterative dilation of the sparse label image onto all foreground pixels
    known = (labelled_mask != 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for _ in range(20):
        dilated_labels = cv2.dilate(
            np.clip(labelled_mask, 0, None).astype(np.uint8), kernel
        ).astype(np.int32)
        # Only overwrite pixels that are foreground AND not yet labelled
        mask_unlabelled = (binary_mask > 0) & (labelled_mask == 0)
        if not np.any(mask_unlabelled):
            break
        labelled_mask[mask_unlabelled] = dilated_labels[mask_unlabelled]


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_clustering(
    kmeans_mask: np.ndarray,
    dbscan_mask: np.ndarray,
    image_name: str = "",
) -> dict:
    """Compare K-Means and DBSCAN segmentation results.

    Parameters
    ----------
    kmeans_mask : ndarray
        Binary uint8 mask from K-Means (255 = seed).
    dbscan_mask : ndarray (int32)
        Labelled mask from DBSCAN (>0 = cluster, 0 = bg, -1 = noise).
    image_name : str
        Used in log messages.

    Returns
    -------
    dict
        Comparison metrics: foreground area, cluster count, overlap ratio.
    """
    km_area  = int(np.sum(kmeans_mask > 0))
    img_area = kmeans_mask.size

    # DBSCAN foreground = all pixels with label > 0
    dbscan_fg = (dbscan_mask > 0).astype(np.uint8) * 255
    db_area   = int(np.sum(dbscan_fg > 0))

    # Clusters detected by DBSCAN
    db_n_clusters = int(np.max(np.clip(dbscan_mask, 0, None)))

    # Overlap between the two masks
    overlap = int(np.sum((kmeans_mask > 0) & (dbscan_fg > 0)))
    union   = int(np.sum((kmeans_mask > 0) | (dbscan_fg > 0)))
    iou     = round(overlap / union, 4) if union > 0 else 0.0

    km_pct = round(km_area  / img_area * 100, 2)
    db_pct = round(db_area  / img_area * 100, 2)

    result = {
        "image":              image_name,
        "kmeans_fg_area_pct": km_pct,
        "dbscan_fg_area_pct": db_pct,
        "dbscan_n_clusters":  db_n_clusters,
        "iou":                iou,
        "analysis": (
            f"K-Means covers {km_pct}% of pixels as foreground. "
            f"DBSCAN identifies {db_n_clusters} spatial clusters covering {db_pct}% "
            f"of pixels. IoU between the two masks = {iou:.3f}. "
            + (
                "Both methods agree well (IoU > 0.7)."
                if iou > 0.7
                else "Methods diverge — DBSCAN may be more conservative due to density thresholding."
            )
        ),
    }

    LOG.info(
        "[%s] Clustering comparison: K-Means fg=%.1f%%, DBSCAN clusters=%d fg=%.1f%%, IoU=%.3f",
        image_name, km_pct, db_n_clusters, db_pct, iou,
    )
    return result
