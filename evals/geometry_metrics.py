from typing import NamedTuple

import torch
from torch import Tensor


class EventGeometry(NamedTuple):
    intra_distances: Tensor
    separation: float | None
    axis_separation: Tensor | None


def cluster_centroids(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    _, inverse = torch.unique(labels, return_inverse=True)
    n_clusters = int(inverse.max().item()) + 1
    counts = torch.bincount(inverse, minlength=n_clusters).to(embeddings.dtype)

    centroids = torch.zeros(
        n_clusters,
        embeddings.shape[1],
        dtype=embeddings.dtype,
        device=embeddings.device,
    ).index_add_(0, inverse, embeddings)

    return centroids / counts.unsqueeze(1), inverse


def event_geometry(embeddings: Tensor, labels: Tensor) -> EventGeometry:
    centroids, inverse = cluster_centroids(embeddings, labels)
    intra_distances = embeddings - centroids[inverse]
    n_clusters = len(centroids)

    if n_clusters < 2:
        return EventGeometry(intra_distances, None, None)

    off_diagonal = ~torch.eye(n_clusters, dtype=torch.bool, device=embeddings.device)
    axis_pairwise = (centroids.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    pairwise = axis_pairwise.norm(dim=2)

    return EventGeometry(
        intra_distances,
        float(pairwise[off_diagonal].mean().item()),
        axis_pairwise[off_diagonal].mean(dim=0),
    )


def geometry_metrics(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, float | None]:
    geometry = event_geometry(embeddings, labels)

    return geometry.intra_distances, geometry.separation
