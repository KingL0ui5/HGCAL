import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from losses import build_loss
from models.geo_transformer import GeoTransformer
from train.geo_train import (
    FEATURE_NAMES,
    GEOMETRIC_FEATURE_INDICES,
    LossConfig,
    ModelConfig,
    TrainConfig,
    build_loader,
)
from train.train import select_device


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
            stats.n_points += int(embeddings.shape[0])
            event_sizes = torch.bincount(batch.batch).tolist()

            for event_embeddings, event_labels in zip(
                torch.split(embeddings, event_sizes),
                torch.split(batch.y, event_sizes),
            ):
                intra_distance, separation = event_geometry(
                    event_embeddings, event_labels
                )
                stats.intra_distance += intra_distance
                stats.n_events += 1

                if separation is not None:
                    stats.centroid_separation += separation
                    stats.n_separation_events += 1

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GeoTransformer")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--loss", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

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

    overall_loader = build_loader(split_files, train_config, pin_memory)
    print_report(
        f"{args.split} (all samples)",
        evaluate_loader(model, overall_loader, loss_fn, device),
    )

    for sample_name, sample_files in group_by_sample(split_files).items():
        sample_loader = build_loader(sample_files, train_config, pin_memory)
        print_report(
            f"{args.split} / {sample_name}",
            evaluate_loader(model, sample_loader, loss_fn, device),
        )


if __name__ == "__main__":
    main()
