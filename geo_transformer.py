"""Geometric transformer producing a low-dimensional coordinate space for CLUE.

Global attention over the layer clusters of one event, biased by a learned
function of pairwise coordinate differences (Point-Transformer / ParT style
relative-position attention). Modern encoder block: pre-RMSNorm, QK-norm,
SwiGLU FFN, DropPath. See docs/learned-coords-for-clue.md §3.

Output is raw Euclidean R^d with NO L2-normalization — the margin loss sets the
absolute scale, and CLUE needs that scale to be meaningful.

Memory note: the geometry bias is (B, heads, N, N), the dominant cost at
N ~ 1500-3300 layer clusters per event. It is computed once and shared by every
block, and its MLP is evaluated in checkpointed chunks along the query axis so
the (B, N, N, hidden) intermediate is never held for the backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import to_dense_batch

from . import register_model


def mask_value(dtype: torch.dtype) -> float:
    """Additive bias that zeroes a padded key under softmax.

    Must be derived from the dtype, not a constant: under fp16 autocast anything
    beyond ~-65504 overflows, and half the finite minimum leaves headroom for
    the attention logit that gets added to it.
    """
    return torch.finfo(dtype).min / 2


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch, per event in the batch."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class GeometryBias(nn.Module):
    """MLP( pairwise coordinate differences ) -> one scalar per attention head."""

    def __init__(self, n_geom: int, n_heads: int, hidden: int = 32, chunk: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.chunk = chunk
        self.mlp = nn.Sequential(
            nn.Linear(n_geom, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )

    def _block(self, geom_q, geom_k):
        # (B, Q, G) x (B, S, G) -> (B, heads, Q, S)
        delta = geom_q.unsqueeze(2) - geom_k.unsqueeze(1)
        return self.mlp(delta).permute(0, 3, 1, 2)

    def forward(self, geom):
        blocks = []
        for start in range(0, geom.shape[1], self.chunk):
            geom_q = geom[:, start : start + self.chunk]
            if self.training and torch.is_grad_enabled():
                blocks.append(
                    checkpoint(self._block, geom_q, geom, use_reentrant=False)
                )
            else:
                blocks.append(self._block(geom_q, geom))
        return torch.cat(blocks, dim=2)


class Block(nn.Module):
    """Pre-RMSNorm block: biased MHSA with QK-norm, then a SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: int, drop_path: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"hidden_dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_mult * dim)

        self.drop_path = DropPath(drop_path)

    def forward(self, x, bias):
        b, s, _ = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, S, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        attn = attn.transpose(1, 2).reshape(b, s, -1)

        x = x + self.drop_path(self.proj(attn))
        return x + self.drop_path(self.ffn(self.norm2(x)))


@register_model("geo_transformer")
class GeoTransformer(nn.Module):
    """Encode layer clusters -> geometry-biased global attention -> R^out_dim.

    geom_idx selects which input-feature columns form the pairwise differences
    that drive the attention bias; None uses every input feature. With
    features = [x, y, z, energy, eta, sin_phi, cos_phi] the geometric subset is
    [0, 1, 2, 4, 5, 6] — the sin/cos pair encodes the phi difference without a
    wrap discontinuity.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        n_heads: int = 8,
        out_dim: int = 3,
        geom_idx=None,
        bias_hidden: int = 32,
        bias_chunk: int = 512,
        ffn_mult: int = 2,
        drop_path: float = 0.1,
    ):
        super().__init__()
        geom_idx = list(range(in_dim)) if geom_idx is None else list(geom_idx)
        if not geom_idx or max(geom_idx) >= in_dim or min(geom_idx) < 0:
            raise ValueError(f"geom_idx {geom_idx} out of range for in_dim {in_dim}")
        self.register_buffer(
            "geom_idx", torch.tensor(geom_idx, dtype=torch.long), persistent=False
        )

        self.input = nn.Linear(in_dim, hidden_dim)
        self.geom_bias = GeometryBias(len(geom_idx), n_heads, bias_hidden, bias_chunk)
        self.blocks = nn.ModuleList(
            [Block(hidden_dim, n_heads, ffn_mult, drop_path) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        # (N, F) -> dense (B, S, F) + validity mask; padded rows attend to real
        # keys only, so no attention row is fully masked and softmax stays finite.
        x, mask = to_dense_batch(data.x, getattr(data, "batch", None))

        bias = self.geom_bias(x[..., self.geom_idx])
        bias = bias.masked_fill(~mask[:, None, None, :], mask_value(bias.dtype))

        h = self.input(x)
        for block in self.blocks:
            h = block(h, bias)

        return self.output(self.norm(h))[mask]
