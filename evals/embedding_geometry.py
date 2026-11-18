import torch
from torch import Tensor


def event_geometry(embeddings: Tensor, labels: Tensor) -> tuple[float, float | None]:
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
    intra_distance = float((embeddings - centroids[inverse]).norm(dim=1).mean().item())

    if n_clusters < 2:
        return intra_distance, None

    pairwise = torch.cdist(centroids, centroids)
    off_diagonal = ~torch.eye(n_clusters, dtype=torch.bool, device=embeddings.device)

    return intra_distance, float(pairwise[off_diagonal].mean().item())
