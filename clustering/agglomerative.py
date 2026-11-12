"""Agglomerative clustering on node embeddings.

The hierarchy is built once per event with scipy's ``linkage``; a given
distance threshold is then a cheap cut of that tree (``fcluster``). This is
what makes a threshold sweep efficient — the expensive O(n^2) step is paid
once per event rather than once per (event, threshold).

InfoNCE embeddings are trained under cosine similarity, so the default
metric is ``cosine`` with ``average`` linkage (UPGMA), which does not assume
Euclidean geometry.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def build_linkage(
    embeddings: np.ndarray,
    method: str = "average",
    metric: str = "cosine",
) -> np.ndarray | None:
    """Build the agglomerative hierarchy for one event.

    Returns the scipy linkage matrix, or None if there are fewer than two
    points (nothing to merge).
    """

    if len(embeddings) < 2:
        return None

    return linkage(
        np.ascontiguousarray(embeddings, dtype=np.float64),
        method=method,
        metric=metric,
    )


def cluster_at(
    link: np.ndarray | None,
    n_points: int,
    distance_threshold: float,
) -> np.ndarray:
    """Cut a linkage tree at ``distance_threshold`` into flat cluster ids.

    Every point receives a cluster id (>= 0); there is no outlier label.
    Small clusters are filtered downstream via ``min_lc`` in the metrics,
    mirroring how CLUE outliers are treated.
    """

    if link is None:
        return np.zeros(n_points, dtype=np.int32)

    labels = fcluster(link, t=distance_threshold, criterion="distance")

    return labels.astype(np.int32)


def run_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float,
    method: str = "average",
    metric: str = "cosine",
) -> np.ndarray:
    """Cluster one event's embeddings at a single distance threshold."""

    link = build_linkage(embeddings, method=method, metric=metric)

    return cluster_at(link, len(embeddings), distance_threshold)
