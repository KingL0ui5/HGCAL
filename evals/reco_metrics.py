from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from clustering.agglomerative import run_agglomerative

MATCH_ENERGY_FRACTION = 0.7
PURE_SCORE_THRESHOLD = 0.2
NOISE_LABEL = -1


@dataclass
class ClusteringParameters:
    distance_threshold: float = 0.05
    method: str = "average"
    metric: str = "cosine"
    min_cluster_size: int = 1


@dataclass
class ReconstructionCounts:
    n_sim: int = 0
    n_matched: int = 0
    n_reco: int = 0
    n_pure: int = 0

    def add(self, other: "ReconstructionCounts") -> None:
        self.n_sim += other.n_sim
        self.n_matched += other.n_matched
        self.n_reco += other.n_reco
        self.n_pure += other.n_pure

    @property
    def efficiency(self) -> float:
        return self.n_matched / max(self.n_sim, 1)

    @property
    def purity(self) -> float:
        return self.n_pure / max(self.n_reco, 1)

    @property
    def multiplicity_ratio(self) -> float:
        return self.n_reco / max(self.n_sim, 1)


def drop_small_clusters(labels: Tensor, min_cluster_size: int) -> Tensor:
    if min_cluster_size <= 1:
        return labels

    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    is_small = (counts < min_cluster_size)[inverse]

    return labels.masked_fill(is_small, NOISE_LABEL)


def cluster_labels(embeddings: Tensor, parameters: ClusteringParameters) -> Tensor:
    labels = run_agglomerative(
        embeddings.detach().cpu().numpy(),
        parameters.distance_threshold,
        method=parameters.method,
        metric=parameters.metric,
    )

    return drop_small_clusters(
        torch.from_numpy(labels.astype(np.int64)), parameters.min_cluster_size
    )


def event_counts(
    reco_labels: Tensor,
    sim_labels: Tensor,
    energy: Tensor,
) -> ReconstructionCounts:

    energy = energy.to(torch.float64)
    _, sim_index = torch.unique(sim_labels, return_inverse=True)
    n_sim = int(sim_index.max().item()) + 1
    sim_energy = torch.zeros(n_sim, dtype=torch.float64).index_add_(
        0, sim_index, energy
    )

    clustered = reco_labels != NOISE_LABEL

    if not bool(clustered.any()):
        return ReconstructionCounts(n_sim=n_sim)

    _, reco_index = torch.unique(reco_labels[clustered], return_inverse=True)
    n_reco = int(reco_index.max().item()) + 1
    clustered_sim = sim_index[clustered]
    clustered_energy = energy[clustered]

    shared_energy = torch.zeros(n_reco, n_sim, dtype=torch.float64).index_put_(
        (reco_index, clustered_sim), clustered_energy, accumulate=True
    )
    shared_squared = torch.zeros(n_reco, n_sim, dtype=torch.float64).index_put_(
        (reco_index, clustered_sim), clustered_energy.square(), accumulate=True
    )

    best_fraction = shared_energy.max(dim=0).values / sim_energy.clamp_min(1e-12)
    reco_to_sim_score = 1.0 - shared_squared.max(dim=1).values / shared_squared.sum(
        dim=1
    ).clamp_min(1e-12)

    return ReconstructionCounts(
        n_sim=n_sim,
        n_matched=int((best_fraction >= MATCH_ENERGY_FRACTION).sum().item()),
        n_reco=n_reco,
        n_pure=int((reco_to_sim_score < PURE_SCORE_THRESHOLD).sum().item()),
    )


def print_reconstruction_report(name: str, counts: ReconstructionCounts) -> None:
    print(f"\n== {name}: reconstruction ==")
    print(f"N_sim:                {counts.n_sim}")
    print(f"N_reco:               {counts.n_reco}")
    print(f"efficiency:           {counts.efficiency:.4f}")
    print(f"purity:               {counts.purity:.4f}")
    print(f"multiplicity ratio:   {counts.multiplicity_ratio:.4f}")
