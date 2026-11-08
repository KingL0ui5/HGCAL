"""Supervised InfoNCE over node embeddings, computed per event. - taken from https://github.com/mmarriot/CaloEmbed"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss


def _event_infonce(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE within one event: positives = nodes with the same shower id.

    Node-equal reduction (each anchor with ≥1 positive contributes equally).
    Returns 0 if no anchor has a positive.
    """
    n = z.size(0)
    if n <= 1:
        return z.new_tensor(0.0)

    logits = (z @ z.t()) / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    valid = pos_mask.any(dim=1)
    if not bool(valid.any()):
        return z.new_tensor(0.0)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (
        log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    )
    return -mean_log_prob_pos[valid].mean()


@register_loss("infonce")
class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, data) -> torch.Tensor:
        # fp32 + L2-normalize for numerical stability under AMP.
        z = F.normalize(embeddings.float(), p=2, dim=1)
        counts = torch.bincount(data.batch).tolist()
        losses = [
            _event_infonce(ze, ye, self.temperature)
            for ze, ye in zip(torch.split(z, counts), torch.split(data.y, counts))
        ]
        return torch.stack(losses).mean()
