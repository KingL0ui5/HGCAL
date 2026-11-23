import argparse
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader

from evals.base_eval import BaseEvaluator
from evals.eval_utils import (
    EmbeddingStats,
    EventOutputs,
    accumulate_event,
    print_report,
    report_by_sample,
)
from losses import build_loss
from models.geo_transformer import GeoTransformer
from train.geo_transformer.geo_train import (
    FEATURE_NAMES,
    GEOMETRIC_FEATURE_INDICES,
    ModelConfig,
    TrainConfig,
    build_loader,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "trained_models"
    / "geo_transformer"
    / "geo_discriminative_20261109_143754_best.pt"
)


class GeoTransformerEvaluator(BaseEvaluator[GeoTransformer]):
    display_name = "GeoTransformer"
    output_prefix = "geo_transformer"
    checkpoint_pattern = re.compile(r"^geo_(?P<loss>.+?)_\d{8}_\d{6}")
    default_checkpoint = MODEL_PATH
    batch_size = 1

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--loss", default=None)
        parser.add_argument("--by-sample", action="store_true")

    def build_model(self) -> GeoTransformer:
        model_config = ModelConfig(**self.run_config.get("model", {}))

        return GeoTransformer(
            in_dim=len(FEATURE_NAMES),
            geom_idx=GEOMETRIC_FEATURE_INDICES,
            **model_config.model_dump(),
        )

    def build_loader(self, file_paths: list[Path]) -> DataLoader:
        config = TrainConfig(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )

        return build_loader(file_paths, config, self.pin_memory)

    def embed_batch(self, model: GeoTransformer, cpu_batch: Any) -> list[EventOutputs]:
        batch = cpu_batch.to(self.device)
        embeddings = model(batch).float().cpu()
        event_sizes = torch.bincount(batch.batch).tolist()

        return [
            EventOutputs(*event)
            for event in zip(
                torch.split(embeddings, event_sizes),
                torch.split(batch.y.cpu(), event_sizes),
                torch.split(batch.energy.cpu(), event_sizes),
            )
        ]

    def build_loss_fn(self) -> nn.Module:
        saved_loss = self.run_config.get("loss", {})
        loss_override = self.args.loss

        if loss_override:
            return build_loss(str(loss_override)).to(self.device)

        return build_loss(self.loss_name, **dict(saved_loss.get("kwargs", {}))).to(
            self.device
        )

    def loss_stats(self, file_paths: list[Path], loss_fn: nn.Module) -> EmbeddingStats:
        stats = EmbeddingStats()

        with no_grad():
            for cpu_batch in self.build_loader(file_paths):
                batch = cpu_batch.to(self.device)
                embeddings = self.model(batch).float()
                stats.loss += float(loss_fn(embeddings, batch).item())
                stats.n_batches += 1
                event_sizes = torch.bincount(batch.batch).tolist()

                for event_embeddings, event_labels in zip(
                    torch.split(embeddings, event_sizes),
                    torch.split(batch.y, event_sizes),
                ):
                    accumulate_event(stats, event_embeddings, event_labels)

        return stats

    def evaluate(self) -> None:
        loss_fn = self.build_loss_fn()

        if self.args.by_sample:
            report_by_sample(
                self.split,
                self.split_files,
                lambda file_paths: self.loss_stats(file_paths, loss_fn),
            )
        else:
            print_report(
                f"{self.split} (all samples)",
                self.loss_stats(self.split_files, loss_fn),
            )

        super().evaluate()


if __name__ == "__main__":
    GeoTransformerEvaluator.main()
