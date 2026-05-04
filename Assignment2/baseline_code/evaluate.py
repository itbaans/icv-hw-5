"""
baseline_code/evaluate.py
==========================
Evaluation utilities for the AgriTech SeedCounter pipeline.
Importable by future assignments without modification.

Public API
----------
compute_mae(predictions, ground_truths)               -> float
compute_rmse(predictions, ground_truths)              -> float
find_failure_cases(records, threshold_pct)            -> list[dict]
generate_performance_summary(records, threshold_pct)  -> dict
"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_mae(
    predictions: Sequence[float],
    ground_truths: Sequence[float],
) -> float:
    """Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    predictions : sequence of float
        Predicted seed counts.
    ground_truths : sequence of float
        Ground-truth seed counts (same length as predictions).

    Returns
    -------
    float
        MAE value, or NaN if sequences are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths."
        )
    if not predictions:
        return float("nan")

    return float(
        sum(abs(p - g) for p, g in zip(predictions, ground_truths))
        / len(predictions)
    )


def compute_rmse(
    predictions: Sequence[float],
    ground_truths: Sequence[float],
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    predictions : sequence of float
        Predicted seed counts.
    ground_truths : sequence of float
        Ground-truth seed counts (same length as predictions).

    Returns
    -------
    float
        RMSE value, or NaN if sequences are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths."
        )
    if not predictions:
        return float("nan")

    mse = sum((p - g) ** 2 for p, g in zip(predictions, ground_truths)) / len(
        predictions
    )
    return float(math.sqrt(mse))


# ---------------------------------------------------------------------------
# Failure-case detection
# ---------------------------------------------------------------------------

def find_failure_cases(
    records: List[Dict],
    threshold_pct: float = 10.0,
) -> List[Dict]:
    """Identify images where the percentage prediction error exceeds *threshold_pct*.

    Parameters
    ----------
    records : list of dict
        Each dict must have the keys ``filename``, ``prediction``, and
        ``ground_truth``.  ``ground_truth`` may be ``None`` / missing, in
        which case the record is skipped.
    threshold_pct : float
        Error percentage above which a record is considered a failure case.
        Default is 10 %.

    Returns
    -------
    list of dict
        Subset of *records* that qualify as failure cases, each enriched
        with an ``error_pct`` key.
    """
    failures = []
    for rec in records:
        gt = rec.get("ground_truth")
        if gt is None:
            continue
        gt = float(gt)
        pred = float(rec.get("prediction", 0))
        if gt == 0:
            # Avoid division-by-zero; flag only if prediction is non-zero
            err_pct = 100.0 if pred != 0 else 0.0
        else:
            err_pct = abs(pred - gt) / gt * 100.0

        if err_pct > threshold_pct:
            failures.append({**rec, "error_pct": round(err_pct, 2)})

    return failures


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_performance_summary(
    records: List[Dict],
    threshold_pct: float = 10.0,
    output_path: Optional[str] = None,
) -> dict:
    """Compute and (optionally) persist overall performance statistics.

    Parameters
    ----------
    records : list of dict
        Each dict with keys ``filename``, ``prediction``, ``ground_truth``.
    threshold_pct : float
        Error % threshold used to flag failure cases.
    output_path : str, optional
        If provided, the summary JSON is written to this path.

    Returns
    -------
    dict
        Summary dictionary with keys: ``total_images``, ``evaluated_images``,
        ``mae``, ``rmse``, ``failure_cases``, ``accuracy_within_threshold``.
    """
    # Split records into those with ground truth and those without
    evaluated = [r for r in records if r.get("ground_truth") is not None]
    preds = [float(r["prediction"]) for r in evaluated]
    gts = [float(r["ground_truth"]) for r in evaluated]

    mae = compute_mae(preds, gts) if evaluated else float("nan")
    rmse = compute_rmse(preds, gts) if evaluated else float("nan")
    failure_cases = find_failure_cases(evaluated, threshold_pct)

    accuracy_within = (
        round((1 - len(failure_cases) / len(evaluated)) * 100, 2)
        if evaluated
        else float("nan")
    )

    summary = {
        "total_images": len(records),
        "evaluated_images": len(evaluated),
        "mae": round(mae, 4) if not math.isnan(mae) else None,
        "rmse": round(rmse, 4) if not math.isnan(rmse) else None,
        "num_failure_cases": len(failure_cases),
        "accuracy_within_threshold_pct": (
            accuracy_within if not math.isnan(accuracy_within) else None
        ),
        "threshold_pct_used": threshold_pct,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    return summary
