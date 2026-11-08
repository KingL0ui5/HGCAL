"""Energy and purity-weighted discriminative embedding loss, per event.

See docs/learned-coords-for-clue.md §4-§6. With node weights w_i and truth
objects c = 1..C:

    mu_c   = sum_{i in c} w_i z_i / sum_{i in c} w_i
    L_var  = mean_c [ sum_{i in c} w_i relu(||z_i - mu_c|| - dv)^2 / sum_{i in c} w_i ]
    L_dist = sum_{c != c'} relu(2 dd - ||mu_c - mu_c'||)^2 / (C (C-1))
    L_reg  = mean_c ||mu_c||
    L      = L_var + L_dist + reg_weight * L_reg

Both hinges are one-sided: a blob of radius dv costs nothing, and centers
further apart than 2*dd cost nothing. The fixed margins are what standardize
every truth object — tiny EM shower or sprawling hadronic one — into a blob of
the same size and spacing, which is what lets a single CLUE parameter set work.

Node weight w_i = E_i * phi(p_i), where p_i is the argmax truth fraction:

    phi(p) = floor + (1 - floor) * clamp((p - p0) / (1 - p0), 0, 1)

The floor matters. In this dataset the stored fraction is 1/multiplicity, so a
hit shared between two showers has p = 0.5 exactly; the un-floored ramp would
zero it out and let it drift into the density valley between the two blobs,
re-merging them. A small nonzero weight commits it to its argmax side without
letting a near-tie dominate the centroid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

EPS = 1e-12


def _norm(v: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 norm, safe to differentiate at zero (a singleton cluster sits
    exactly on its own centroid, where the plain norm's gradient is NaN)."""
    return (v.pow(2).sum(dim=-1) + EPS).sqrt()


def _event_loss(z, y, w, delta_v, delta_d, reg_weight):
    """Margin loss for one event. z (n, d), y (n,) labels, w (n,) node weights."""
    _, inv = torch.unique(y, return_inverse=True)
    n_clusters = int(inv.max()) + 1

    w_sum = z.new_zeros(n_clusters).index_add_(0, inv, w).clamp_min(EPS)
    mu = z.new_zeros(n_clusters, z.shape[1]).index_add_(0, inv, w.unsqueeze(1) * z)
    mu = mu / w_sum.unsqueeze(1)

    dist = _norm(z - mu[inv])
    pull = w * F.relu(dist - delta_v).pow(2)
    l_var = (z.new_zeros(n_clusters).index_add_(0, inv, pull) / w_sum).mean()

    if n_clusters > 1:
        centers = _norm(mu.unsqueeze(1) - mu.unsqueeze(0))
        push = F.relu(2.0 * delta_d - centers).pow(2)
        off_diag = ~torch.eye(n_clusters, dtype=torch.bool, device=z.device)
        l_dist = push[off_diag].sum() / (n_clusters * (n_clusters - 1))
    else:
        l_dist = z.new_tensor(0.0)

    return l_var + l_dist + reg_weight * _norm(mu).mean()


@register_loss("discriminative")
class Discriminative(nn.Module):
    def __init__(
        self,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        reg_weight: float = 1e-3,
        purity_p0: float = 0.5,
        purity_floor: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= purity_p0 < 1.0:
            raise ValueError(f"purity_p0 must be in [0, 1), got {purity_p0}")
        if not 0.0 <= purity_floor <= 1.0:
            raise ValueError(f"purity_floor must be in [0, 1], got {purity_floor}")
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.reg_weight = reg_weight
        self.purity_p0 = purity_p0
        self.purity_floor = purity_floor

    def purity_weight(self, frac: torch.Tensor) -> torch.Tensor:
        ramp = ((frac - self.purity_p0) / (1.0 - self.purity_p0)).clamp(0.0, 1.0)
        return self.purity_floor + (1.0 - self.purity_floor) * ramp

    def forward(self, embeddings: torch.Tensor, data) -> torch.Tensor:
        if not hasattr(data, "energy"):
            raise AttributeError(
                "discriminative loss needs raw LC energy on the batch (data.energy); "
                "reload the dataset with src.data.HDF5Events."
            )
        # fp32 throughout: the margins are absolute distances, and bf16 rounding
        # on a squared hinge is a meaningful fraction of delta_v.
        z = embeddings.float()
        w = data.energy.float() * self.purity_weight(data.frac.float())

        counts = torch.bincount(data.batch).tolist()
        losses = [
            _event_loss(ze, ye, we, self.delta_v, self.delta_d, self.reg_weight)
            for ze, ye, we in zip(
                torch.split(z, counts),
                torch.split(data.y, counts),
                torch.split(w, counts),
            )
        ]
        return torch.stack(losses).mean()
