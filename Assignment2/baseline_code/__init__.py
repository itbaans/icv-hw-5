# baseline_code/__init__.py
# Makes baseline_code a proper Python package importable by future assignments.

from .preprocessing import (
    load_config,
    apply_grayscale,
    apply_noise_reduction,
    apply_edge_detection,
    apply_clahe,
    compare_filters,
)
from .evaluate import (
    compute_mae,
    compute_rmse,
    generate_performance_summary,
    find_failure_cases,
)
from .clustering import (
    apply_kmeans,
    apply_dbscan,
    compare_clustering,
)

__all__ = [
    # preprocessing
    "load_config",
    "apply_grayscale",
    "apply_noise_reduction",
    "apply_edge_detection",
    "apply_clahe",
    "compare_filters",
    # evaluate
    "compute_mae",
    "compute_rmse",
    "generate_performance_summary",
    "find_failure_cases",
    # clustering
    "apply_kmeans",
    "apply_dbscan",
    "compare_clustering",
]
