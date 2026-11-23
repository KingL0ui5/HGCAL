import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import IGNORE_LABEL, XYZ_SCALE
from evals.base_eval import BaseEvaluator
from evals.eval_utils import EventOutputs
from models.spatial_transformer import SpatialConfig, SpatialTransformer
from train.train_utils import TrainConfig, batch_to_device, build_loader

ENERGY_FEATURE_INDEX = 3


class SpatialTransformerEvaluator(BaseEvaluator[SpatialTransformer]):
    display_name = "SpatialTransformer"
    output_prefix = "spatial_transformer"
    checkpoint_pattern = re.compile(r"^spatial_(?P<loss>[a-z]+)_\d{8}_\d{6}")
    batch_size = 8

    def build_model(self) -> SpatialTransformer:
        return SpatialTransformer(SpatialConfig(**self.run_config.get("model", {})))

    def build_loader(self, file_paths: list[Path]) -> DataLoader:
        config = TrainConfig(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )

        return build_loader(file_paths, config, self.pin_memory)

    def embed_batch(
        self,
        model: SpatialTransformer,
        cpu_batch: Any,
    ) -> list[EventOutputs]:

        batch = batch_to_device(cpu_batch, self.device)
        embeddings = model(batch["features"], batch["padding_mask"]).float().cpu()
        energy = torch.expm1(batch["features"][..., ENERGY_FEATURE_INDEX]).float().cpu()
        positions = (batch["features"][..., :3] * XYZ_SCALE).float().cpu()
        labels = batch["labels"].cpu()
        is_valid = ((~batch["padding_mask"]) & (batch["labels"] != IGNORE_LABEL)).cpu()

        return [
            EventOutputs(
                F.normalize(embeddings[index][keep], dim=1),
                labels[index][keep],
                energy[index][keep],
                positions[index][keep],
            )
            for index, keep in enumerate(is_valid)
        ]


if __name__ == "__main__":
    SpatialTransformerEvaluator.main()
