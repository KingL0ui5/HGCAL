import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from evals.basic_transformer.transformer_eval import (
    iter_events,
    select_embeddings,
)
from evals.basic_transformer.transformer_eval import (
    load_model as load_basic_transformer,
)
from evals.embedding_geometry import event_geometry
from losses import build_loss
from models.geo_transformer import GeoTransformer
from models.transformer import TransformerModel
from train.basic_transformer.train import TrainConfig as BasicTrainConfig
from train.basic_transformer.train import build_loader as build_basic_loader
from train.basic_transformer.train import select_device
from train.geo_transformer.geo_train import (
    FEATURE_NAMES,
    GEOMETRIC_FEATURE_INDICES,
    LossConfig,
    ModelConfig,
    TrainConfig,
    build_loader,
)

MODEL_CHOICES = ("geo", "transformer")
BASIC_EMBEDDING_SOURCE = "hidden"
CHECKPOINT_LOSS_PATTERN = re.compile(r"^geo_(?P<loss>.+?)_\d{8}_\d{6}")


@dataclass
class EmbeddingStats:
    loss: float = 0.0
    intra_distance: float = 0.0
    centroid_separation: float = 0.0
    n_batches: int = 0
    n_events: int = 0
    n_points: int = 0
    n_separation_events: int = 0


def group_by_sample(file_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}

    for path in file_paths:
        sample_name = path.relative_to(DATA_PATH).parts[0]
        groups.setdefault(sample_name, []).append(path)

    return dict(sorted(groups.items()))


def evaluate_loader(
    model: GeoTransformer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EmbeddingStats:

    model.eval()
    stats = EmbeddingStats()

    with no_grad():
        for cpu_batch in loader:
            batch: Any = cpu_batch.to(device)
            embeddings = model(batch).float()
            stats.loss += float(loss_fn(embeddings, batch).item())
            stats.n_batches += 1
            event_sizes = torch.bincount(batch.batch).tolist()

            for event_embeddings, event_labels in zip(
                torch.split(embeddings, event_sizes),
                torch.split(batch.y, event_sizes),
            ):
                accumulate_event(stats, event_embeddings, event_labels)

    return stats


def accumulate_event(
    stats: EmbeddingStats,
    embeddings: Tensor,
    labels: Tensor,
) -> None:

    intra_distance, separation = event_geometry(embeddings, labels)
    stats.intra_distance += intra_distance
    stats.n_events += 1
    stats.n_points += int(embeddings.shape[0])

    if separation is not None:
        stats.centroid_separation += separation
        stats.n_separation_events += 1


def evaluate_basic_transformer_loader(
    model: TransformerModel,
    loader: DataLoader,
    device: torch.device,
) -> EmbeddingStats:

    stats = EmbeddingStats()

    for event in iter_events(model, loader, device, max_events=None):
        embeddings = select_embeddings(event, BASIC_EMBEDDING_SOURCE)
        stats.loss += float(F.cross_entropy(event.logits, event.labels).item())
        stats.n_batches += 1
        accumulate_event(stats, embeddings, event.labels)

    return stats


def print_report(name: str, stats: EmbeddingStats) -> None:
    print(f"\n== {name} ==")
    print(f"events:                   {stats.n_events}")
    print(f"points:                   {stats.n_points}")
    print(f"loss:                     {stats.loss / max(stats.n_batches, 1):.4f}")
    print(
        f"mean intra-cluster dist:  {stats.intra_distance / max(stats.n_events, 1):.4f}"
    )
    print(
        f"mean centroid separation: "
        f"{stats.centroid_separation / max(stats.n_separation_events, 1):.4f}"
    )


def load_run_config(checkpoint: Path) -> dict[str, Any]:
    run_name = checkpoint.stem.removesuffix("_best").removesuffix("_last")
    config_path = checkpoint.with_name(f"{run_name}_config.json")

    if not config_path.exists():
        return {}

    return json.loads(config_path.read_text())


def checkpoint_loss_name(checkpoint: Path) -> str:
    configured_loss = load_run_config(checkpoint).get("loss", {}).get("name")

    if configured_loss:
        return str(configured_loss)

    match = CHECKPOINT_LOSS_PATTERN.match(checkpoint.stem)

    return match.group("loss") if match else "unknown"


def load_model(
    checkpoint: Path,
    model_config: ModelConfig,
    device: torch.device,
) -> GeoTransformer:

    model = GeoTransformer(
        in_dim=len(FEATURE_NAMES),
        geom_idx=GEOMETRIC_FEATURE_INDICES,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        n_heads=model_config.n_heads,
        out_dim=model_config.out_dim,
        bias_hidden=model_config.bias_hidden,
        bias_chunk=model_config.bias_chunk,
        ffn_mult=model_config.ffn_mult,
        drop_path=model_config.drop_path,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="geo")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--loss", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser


def report_by_sample(
    split: str,
    split_files: list[Path],
    evaluate_files: Callable[[list[Path]], EmbeddingStats],
) -> None:

    print_report(f"{split} (all samples)", evaluate_files(split_files))

    for sample_name, sample_files in group_by_sample(split_files).items():
        print_report(f"{split} / {sample_name}", evaluate_files(sample_files))


def main_basic_transformer(args: argparse.Namespace) -> None:
    train_config = BasicTrainConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    device = select_device()
    print(
        f"using device: {device} | basic transformer | "
        f"embedding: {BASIC_EMBEDDING_SOURCE} (L2-normalised) | loss: crossentropy"
    )

    model = load_basic_transformer(Path(args.checkpoint), device)
    pin_memory = device.type == "cuda"
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[args.split]

    def evaluate_files(file_paths: list[Path]) -> EmbeddingStats:
        loader = build_basic_loader(file_paths, train_config, pin_memory)

        return evaluate_basic_transformer_loader(model, loader, device)

    report_by_sample(str(args.split), split_files, evaluate_files)


def main_geo_transformer(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    run_config = load_run_config(checkpoint)
    model_config = ModelConfig(**run_config.get("model", {}))
    saved_loss = run_config.get("loss", {})
    loss_config = LossConfig(
        name=str(args.loss or saved_loss.get("name", LossConfig().name)),
        kwargs={} if args.loss else dict(saved_loss.get("kwargs", {})),
    )
    train_config = TrainConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(run_config.get("train", {}).get("seed", TrainConfig().seed)),
    )

    device = select_device()
    print(f"using device: {device} | loss: {loss_config.name}")

    model = load_model(checkpoint, model_config, device)
    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)
    pin_memory = device.type == "cuda"
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[args.split]

    def evaluate_files(file_paths: list[Path]) -> EmbeddingStats:
        loader = build_loader(file_paths, train_config, pin_memory)

        return evaluate_loader(model, loader, loss_fn, device)

    report_by_sample(str(args.split), split_files, evaluate_files)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.model == "transformer":
        main_basic_transformer(args)
    else:
        main_geo_transformer(args)


if __name__ == "__main__":
    main()
