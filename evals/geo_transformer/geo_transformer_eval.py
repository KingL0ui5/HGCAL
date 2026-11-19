import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from evals.embedding_geometry import event_geometry
from evals.evaluate import (
    checkpoint_loss_name,
    evaluate_loader,
    load_model,
    load_run_config,
    print_report,
)
from evals.reco_metrics import (
    ClusteringParameters,
    ReconstructionCounts,
    cluster_labels,
    event_counts,
    print_reconstruction_report,
)
from evals.stratified import (
    StratifiedReconstruction,
    event_multiplicity,
    plot_by_multiplicity,
    print_geometry_report,
    print_stratified_table,
    save_metrics,
)
from losses import build_loss
from train.basic_transformer.train import select_device
from train.geo_transformer.geo_train import (
    LossConfig,
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
EVAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "geo_transformer_runs"
SPLIT = "test"
BATCH_SIZE = 1
NUM_WORKERS = 2


def output_path(loss_name: str, kind: str, suffix: str = "png") -> Path:
    return EVAL_DATA_DIR / f"geo_transformer_{loss_name}_{kind}.{suffix}"


class EventEmbedding(NamedTuple):
    embeddings: Tensor
    labels: Tensor
    energy: Tensor


def iter_event_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Iterator[EventEmbedding]:

    model.eval()

    with no_grad():
        for cpu_batch in loader:
            batch: Any = cpu_batch.to(device)
            embeddings = model(batch).float().cpu()
            event_sizes = torch.bincount(batch.batch).tolist()

            yield from (
                EventEmbedding(*event)
                for event in zip(
                    torch.split(embeddings, event_sizes),
                    torch.split(batch.y.cpu(), event_sizes),
                    torch.split(batch.energy.cpu(), event_sizes),
                )
            )


def evaluate(
    device: torch.device,
    checkpoint: Path,
    stratified_path: Path,
    metrics_path: Path,
    clustering_parameters: ClusteringParameters,
) -> None:

    run_config = load_run_config(checkpoint)
    train_config = TrainConfig(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=int(run_config.get("train", {}).get("seed", TrainConfig().seed)),
    )
    saved_loss = run_config.get("loss", {})
    loss_config = LossConfig(
        name=checkpoint_loss_name(checkpoint),
        kwargs=dict(saved_loss.get("kwargs", {})),
    )

    model = load_model(checkpoint, ModelConfig(**run_config.get("model", {})), device)
    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)

    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")
    print_report(
        f"{SPLIT} (all samples)", evaluate_loader(model, loader, loss_fn, device)
    )

    event_loader = build_loader(split_files, train_config, device.type == "cuda")
    counts = ReconstructionCounts()
    stratified = StratifiedReconstruction()

    for event in iter_event_embeddings(model, event_loader, device):
        reco_labels = cluster_labels(event.embeddings, clustering_parameters)
        event_reconstruction = event_counts(reco_labels, event.labels, event.energy)
        counts.add(event_reconstruction)
        multiplicity = event_multiplicity(event.labels)
        intra_distance, separation = event_geometry(event.embeddings, event.labels)
        stratified.add(multiplicity, event_reconstruction, intra_distance, separation)

    print_reconstruction_report(SPLIT, counts)
    print_geometry_report(SPLIT, stratified)

    stratified_metrics = stratified.metrics()
    print_stratified_table(SPLIT, stratified_metrics, stratified.n_events)
    plot_by_multiplicity(
        stratified_metrics,
        stratified.n_events,
        "GeoTransformer",
        stratified_path,
    )
    print(f"stratified plot saved to {stratified_path}")
    save_metrics(stratified_metrics, stratified.n_events, metrics_path)
    print(f"metrics saved to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GeoTransformer")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default=str(select_device()))
    parser.add_argument("--stratified-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=ClusteringParameters().distance_threshold,
    )
    parser.add_argument("--linkage-method", default=ClusteringParameters().method)
    parser.add_argument("--metric", default=ClusteringParameters().metric)
    parser.add_argument(
        "--min-cluster-size", type=int, default=ClusteringParameters().min_cluster_size
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    loss_name = checkpoint_loss_name(checkpoint)
    stratified_output = (
        Path(args.stratified_output)
        if args.stratified_output
        else output_path(loss_name, "by_multiplicity")
    )
    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path(loss_name, "metrics", "json")
    )
    evaluate(
        torch.device(str(args.device)),
        checkpoint,
        stratified_output,
        metrics_output,
        ClusteringParameters(
            distance_threshold=float(args.distance_threshold),
            method=str(args.linkage_method),
            metric=str(args.metric),
            min_cluster_size=int(args.min_cluster_size),
        ),
    )
