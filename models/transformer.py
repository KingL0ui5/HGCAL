from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class ModelConfig:
    num_classes: int = 5
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(4, config.d_model),
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

        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, features: Tensor, padding_mask: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        hidden = self.input_projection(features)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

        return self.classifier(hidden).transpose(1, 2)
