import torch
from torch import Tensor


def geometry_metrics(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, float | None]:
    _, inverse = torch.unique(labels, return_inverse=True)
    n_clusters = int(inverse.max().item()) + 1
    counts = torch.bincount(inverse, minlength=n_clusters).to(embeddings.dtype)

    centroids = torch.zeros(
        n_clusters,
        embeddings.shape[1],
        dtype=embeddings.dtype,
        device=embeddings.device,
    ).index_add_(0, inverse, embeddings)

    centroids = centroids / counts.unsqueeze(1)
    intra_distances = embeddings - centroids[inverse]

    if n_clusters < 2:
        return intra_distances, None

    pairwise = torch.cdist(centroids, centroids)
    off_diagonal = ~torch.eye(n_clusters, dtype=torch.bool, device=embeddings.device)

    return intra_distances, float(pairwise[off_diagonal].mean().item())
