"""
run_experiments.py – AgriTech SeedCounter Comparison Runner (Optimised)
========================================================================
Runs 7 essential pipeline variants efficiently.

Key optimisations vs naive approach
------------------------------------
1. Stratified 20-image sample (not all 141) – ~7× faster, still representative.
2. DBSCAN skipped unless the experiment's seg_source requires it – biggest
   speed win (DBSCAN on high-res dense images is very slow).
3. Experiments trimmed to 7 essential comparisons for the report.

Output
------
  output/metrics/experiment_results.csv   – one row per experiment
  output/experiments/<exp_id>/            – visuals saved in Phase 2

Usage
-----
    python run_experiments.py             # Phase 1: metrics sweep
    python run_experiments.py --visuals   # Phase 2: save report images
    python run_experiments.py --exp D2    # run one specific experiment
"""

import argparse
import copy
import csv
import logging
import math
import os
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CONFIG_PATH = os.path.join(SCRIPT_DIR, "baseline_code", "config.yaml")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
OUT_METRICS = os.path.join(SCRIPT_DIR, "output", "metrics")
OUT_EXP     = os.path.join(SCRIPT_DIR, "output", "experiments")
RESULTS_CSV = os.path.join(OUT_METRICS, "experiment_results.csv")

# ---------------------------------------------------------------------------
# Stratified image sample (20 images across density range)
# Covers sparse (1-10), medium (11-50), dense (51-140+)
# ---------------------------------------------------------------------------
SAMPLE_IMAGES = {
    "1.jpg", "5.jpg", "10.jpg",                            # sparse
    "15.jpg", "20.jpg", "25.jpg", "30.jpg",                # low-medium
    "40.jpg", "50.jpg", "60.jpg",                          # medium
    "70.jpg", "80.jpg", "90.jpg", "100.jpg",               # medium-high
    "110.jpg", "115.jpg", "120.jpg",                       # high
    "125.jpg", "130.jpg", "140.jpg",                       # very dense
}

# High-density images to save report visuals for (Phase 2)
VISUAL_IMAGES = {"50.jpg", "100.jpg", "120.jpg", "130.jpg", "140.jpg"}

# ---------------------------------------------------------------------------
# Experiment definitions (7 essential comparisons)
# ---------------------------------------------------------------------------
# seg_source: 'threshold'(default) | 'kmeans' | 'dbscan' | 'thresh_and_kmeans'
# needs_dbscan: True only when seg_source is dbscan (skip otherwise for speed)
# blob_only: True → use blob count, skip morphology/watershed
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # ── Group A: Filter comparison ────────────────────────────────────────
    {
        "id": "A2", "group": "A-Filter",
        "description": "Median filter",
        "config_patch": {"preprocessing.noise_reduction.method": "median"},
    },
    # ── Group B: Threshold comparison ─────────────────────────────────────
    {
        "id": "B2", "group": "B-Threshold",
        "description": "Otsu global threshold",
        "config_patch": {"thresholding.primary": "otsu"},
    },
    # ── Group C: Morphology comparison ────────────────────────────────────
    {
        "id": "C2", "group": "C-Morphology",
        "description": "No closing (opening only)",
        "config_patch": {"morphology.closing.enabled": False},
    },
    # ── Group D: Segmentation source (the novel comparison) ───────────────
    {
        "id": "D2", "group": "D-SegSource",
        "description": "K-Means mask → CCA + Watershed",
        "config_patch": {},
        "seg_source": "kmeans",
    },
    {
        "id": "D3", "group": "D-SegSource",
        "description": "DBSCAN mask → CCA + Watershed",
        "config_patch": {},
        "seg_source": "dbscan",
        "needs_dbscan": True,   # only experiment that actually needs DBSCAN
    },
    # ── Group E: Counting method ──────────────────────────────────────────
    {
        "id": "E2", "group": "E-Counter",
        "description": "CCA only – no watershed",
        "config_patch": {"watershed.enabled": False},
    },
    {
        "id": "E3", "group": "E-Counter",
        "description": "Blob detection only",
        "config_patch": {},
        "blob_only": True,
    },
]

# Which variants get visuals saved in Phase 2
VISUAL_VARIANTS = {"D2", "D3", "B2", "E2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_patch(cfg: dict, patch: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotpath, value in patch.items():
        keys = dotpath.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def _gt_from_name(basename: str):
    try:
        return int(os.path.splitext(basename)[0])
    except ValueError:
        return None


def _compute_metrics(records: list, threshold_pct: float) -> dict:
    evaluated = [r for r in records if r["ground_truth"] is not None]
    if not evaluated:
        return {}
    errors  = [abs(r["prediction"] - r["ground_truth"]) for r in evaluated]
    mae     = round(sum(errors) / len(errors), 4)
    rmse    = round(math.sqrt(sum(e ** 2 for e in errors) / len(errors)), 4)
    within  = sum(
        1 for r in evaluated
        if r["ground_truth"] > 0
        and abs(r["prediction"] - r["ground_truth"]) / r["ground_truth"] * 100
            <= threshold_pct
    )
    acc_pct  = round(within / len(evaluated) * 100, 2)
    failures = len(evaluated) - within
    return {
        "mae": mae, "rmse": rmse,
        "accuracy_pct": acc_pct, "num_failures": failures,
        "n_images": len(evaluated),
    }


def _no_op_save(path, img):
    """Drop-in replacement for cv2.imwrite that does nothing."""
    pass


def _make_save_fn(exp_out_dir: str, basename: str):
    """
    Returns a save_fn that redirects any cv2.imwrite call into:
        exp_out_dir/<image_stem>/<stage>_<filename>

    Stage functions call save_fn with full paths like:
        output/inspection/gray/50.png
        output/inspection/watershed/50.png
    We derive a unique name by prefixing the parent directory (=stage name):
        gray_50.png, watershed_50.png, morphology_50.png …
    If the path has no meaningful parent (e.g. 'kmeans_50.png' with no dir),
    the filename is used as-is.
    """
    stem = os.path.splitext(basename)[0]
    dest = os.path.join(exp_out_dir, stem)
    os.makedirs(dest, exist_ok=True)

    def _save(path: str, img) -> bool:
        parent = os.path.basename(os.path.dirname(path))  # e.g. "gray", "watershed"
        fname  = os.path.basename(path)                    # e.g. "50.png"
        # Prefix stage name to make filenames unique across stages
        unique = f"{parent}_{fname}" if parent and parent not in (".", "") else fname
        return cv2.imwrite(os.path.join(dest, unique), img)

    return _save


# ---------------------------------------------------------------------------
# Core: run one experiment over a list of images
# ---------------------------------------------------------------------------
def run_one(exp: dict, base_cfg: dict, image_paths: list,
            save_set: set | None = None,
            exp_out_dir: str | None = None) -> dict:
    """
    Run experiment over image_paths.

    save_set    : set of basenames to save images for; None=all, set()=none
    exp_out_dir : when set, images for save_set go here (per-experiment folder)
    """
    from main import (
        stage_preprocessing,
        stage_thresholding, stage_morphology,
        stage_watershed, stage_postprocessing_count,
        stage_counting_blob, stage_counting_cca,
    )
    from baseline_code.clustering import apply_kmeans

    cfg           = _apply_patch(base_cfg, exp.get("config_patch", {}))
    seg_source    = exp.get("seg_source", "threshold")
    blob_only     = exp.get("blob_only", False)
    needs_dbscan  = exp.get("needs_dbscan", False)
    threshold_pct = cfg.get("failure_case", {}).get("error_threshold_pct", 10.0)

    records = []

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        name     = os.path.splitext(basename)[0]

        img = cv2.imread(img_path)
        if img is None:
            LOG.warning("Could not read %s – skipping", basename)
            continue

        # Decide save function for this image
        want_save = (save_set is None) or (basename in save_set)
        if want_save and exp_out_dir:
            # Use per-experiment redirecting save function
            _sfn = _make_save_fn(exp_out_dir, basename)
        elif want_save and exp_out_dir is None:
            _sfn = None          # stage defaults to cv2.imwrite (normal mode)
        else:
            _sfn = _no_op_save   # skip saving entirely

        # ── Stage 1: Preprocessing ───────────────────────────────────────
        gray, filtered, edges, _ = stage_preprocessing(
            img, cfg, name, save_fn=_sfn)

        # ── Stage 2: K-Means only (DBSCAN skipped unless seg_source=dbscan) ─
        kmeans_mask, _ = apply_kmeans(gray, cfg)
        if want_save and _sfn is not None:
            _sfn(f"kmeans_{name}.png", kmeans_mask)

        if needs_dbscan:
            from baseline_code.clustering import apply_dbscan
            dbscan_labels = apply_dbscan(kmeans_mask, cfg)
        else:
            dbscan_labels = None

        # ── Stage 3: Thresholding ────────────────────────────────────────
        binary_mask = stage_thresholding(filtered, cfg, name, save_fn=_sfn)

        # ── Segmentation source selection ────────────────────────────────
        if seg_source == "kmeans":
            seg_mask = kmeans_mask
        elif seg_source == "dbscan" and dbscan_labels is not None:
            seg_mask = (dbscan_labels > 0).astype(np.uint8) * 255
        elif seg_source == "thresh_and_kmeans":
            seg_mask = cv2.bitwise_and(binary_mask, kmeans_mask)
        else:
            seg_mask = binary_mask

        if blob_only:
            kps         = stage_counting_blob(gray, cfg)
            final_count = len(kps)
        else:
            # ── Stage 4: Morphology ──────────────────────────────────────
            morphed, _ = stage_morphology(seg_mask, cfg, name, save_fn=_sfn)

            # ── Stage 5+6: Watershed or CCA only ────────────────────────
            ws_enabled = cfg.get("watershed", {}).get("enabled", True)
            if ws_enabled:
                refined, ws_labels, overlap = stage_watershed(
                    img, morphed, cfg, name, save_fn=_sfn)
                final_count, _ = stage_postprocessing_count(
                    refined, ws_labels, overlap, cfg)
            else:
                cca_count, _, _, _, _, _ = stage_counting_cca(morphed, cfg)
                final_count = cca_count

        gt = _gt_from_name(basename)
        records.append({"filename": basename,
                         "prediction": final_count,
                         "ground_truth": gt})
        LOG.info("  %-12s predicted=%-4d  gt=%s",
                 basename, final_count, gt if gt is not None else "?")

    return _compute_metrics(records, threshold_pct)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
CSV_FIELDS = ["exp_id", "group", "description",
              "mae", "rmse", "accuracy_pct", "num_failures", "n_images"]


def _append_csv(path: str, row_id: str, group: str,
                description: str, metrics: dict) -> None:
    is_new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "exp_id":      row_id,
            "group":       group,
            "description": description,
            **{k: metrics.get(k) for k in
               ["mae", "rmse", "accuracy_pct", "num_failures", "n_images"]},
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visuals", action="store_true",
                        help="Phase 2: save images for VISUAL_IMAGES")
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Run only these IDs, e.g. --exp A2 D2")
    args = parser.parse_args()

    from baseline_code.preprocessing import load_config
    base_cfg = load_config(CONFIG_PATH)

    # Build stratified sample (only images that actually exist in data/)
    all_files = {f for f in os.listdir(DATA_DIR)
                 if os.path.splitext(f)[1].lower() in
                    {".jpg", ".jpeg", ".png", ".bmp"}}
    sample = sorted(SAMPLE_IMAGES & all_files)
    full   = sorted(all_files)

    LOG.info("Using %d-image stratified sample (out of %d total)",
             len(sample), len(full))

    # Filter experiments
    experiments = EXPERIMENTS
    if args.exp:
        ids = set(args.exp)
        experiments = [e for e in EXPERIMENTS if e["id"] in ids]
        LOG.info("Running %d experiment(s): %s", len(experiments), ids)

    os.makedirs(OUT_METRICS, exist_ok=True)

    # ── Phase 2: save report visuals ──────────────────────────────────────
    if args.visuals:
        vis_exps = [e for e in EXPERIMENTS if e["id"] in VISUAL_VARIANTS]
        if args.exp:
            vis_exps = [e for e in vis_exps if e["id"] in set(args.exp)]
        visual_paths = [os.path.join(DATA_DIR, f) for f in SAMPLE_IMAGES
                        if f in VISUAL_IMAGES
                        and os.path.isfile(os.path.join(DATA_DIR, f))]
        for exp in vis_exps:
            exp_out = os.path.join(OUT_EXP, exp["id"])
            os.makedirs(exp_out, exist_ok=True)
            LOG.info("=" * 55)
            LOG.info("Phase 2 – %s: %s  →  %s", exp["id"],
                     exp["description"], exp_out)
            run_one(exp, base_cfg, visual_paths,
                    save_set=VISUAL_IMAGES,
                    exp_out_dir=exp_out)   # ← redirects saves into per-exp folder
            LOG.info("  Saved to: %s", exp_out)
        LOG.info("Phase 2 done.")
        LOG.info("Folder structure:  output/experiments/<exp_id>/<image_stem>/<stage>.png")
        return

    # ── Phase 1: metrics sweep on stratified sample ───────────────────────
    # Baseline row (only if CSV doesn't exist yet)
    if not os.path.isfile(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0:
        LOG.info("=" * 55)
        LOG.info("Running BASELINE on %d-image sample…", len(sample))
        paths   = [os.path.join(DATA_DIR, f) for f in sample
                   if os.path.isfile(os.path.join(DATA_DIR, f))]
        metrics = run_one(
            {"id": "★", "group": "Baseline",
             "description": "Current best config",
             "config_patch": {}, "seg_source": "threshold"},
            base_cfg, paths, save_set=set(),  # no images
        )
        LOG.info("BASELINE → MAE=%.2f  RMSE=%.2f  Acc=%.1f%%",
                 metrics.get("mae", 0), metrics.get("rmse", 0),
                 metrics.get("accuracy_pct", 0))
        _append_csv(RESULTS_CSV, "★-Baseline", "Baseline",
                    "Gaussian + Adaptive-G + Close→Open + Threshold + WS(0.4/25)",
                    metrics)

    for exp in experiments:
        LOG.info("=" * 55)
        LOG.info("Running %s: %s", exp["id"], exp["description"])
        paths   = [os.path.join(DATA_DIR, f) for f in sample
                   if os.path.isfile(os.path.join(DATA_DIR, f))]
        metrics = run_one(exp, base_cfg, paths, save_set=set())
        LOG.info("  → MAE=%.2f  RMSE=%.2f  Acc=%.1f%%  Failures=%s",
                 metrics.get("mae", 0), metrics.get("rmse", 0),
                 metrics.get("accuracy_pct", 0),
                 metrics.get("num_failures"))
        _append_csv(RESULTS_CSV, exp["id"], exp["group"],
                    exp["description"], metrics)

    LOG.info("=" * 55)
    LOG.info("Done. Results: %s", RESULTS_CSV)
    LOG.info("Run with --visuals to save report images for key variants.")


if __name__ == "__main__":
    main()
