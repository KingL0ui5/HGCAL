from pydantic import BaseModel
from torch import Tensor, nn


class SpatialConfig(BaseModel):
    in_features: int = 4
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    out_dim: int = 3


class SpatialTransformer(nn.Module):
    def __init__(self, config: SpatialConfig) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(config.in_features, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        self.output = nn.Linear(config.d_model, config.out_dim)

    def forward(self, features: Tensor, padding_mask: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        hidden = self.input_projection(features)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)

        return self.output(encoded)
