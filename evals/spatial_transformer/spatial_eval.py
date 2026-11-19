import argparse
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, IGNORE_LABEL, get_splits_by_file
from evals.embedding_geometry import event_geometry
from evals.evaluate import load_run_config
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
from models.spatial_transformer import SpatialConfig, SpatialTransformer
from train.basic_transformer.train import (
    TrainConfig,
    batch_to_device,
    build_loader,
    select_device,
)

EVAL_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "spatial_transformer_runs"
)
CHECKPOINT_LOSS_PATTERN = re.compile(r"^spatial_(?P<loss>[a-z]+)_\d{8}_\d{6}")
SPLIT = "test"
BATCH_SIZE = 8
ENERGY_FEATURE_INDEX = 3


class EventOutputs(NamedTuple):
    embeddings: Tensor
    labels: Tensor
    energy: Tensor


def checkpoint_loss_name(checkpoint: Path) -> str:
    match = CHECKPOINT_LOSS_PATTERN.match(checkpoint.stem)

    return match.group("loss") if match else "unknown"


def output_path(loss_name: str, kind: str, suffix: str = "png") -> Path:
    return EVAL_DATA_DIR / f"spatial_transformer_{loss_name}_{kind}.{suffix}"


def load_model(checkpoint: Path, device: torch.device) -> SpatialTransformer:
    model_config = SpatialConfig(**load_run_config(checkpoint).get("model", {}))
    model = SpatialTransformer(model_config).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model


def iter_events(
    model: SpatialTransformer,
    loader: DataLoader,
    device: torch.device,
    max_events: int | None,
) -> Iterator[EventOutputs]:

    model.eval()
    n_events = 0

    with no_grad():
        for cpu_batch in loader:
            batch = batch_to_device(cpu_batch, device)
            embeddings = model(batch["features"], batch["padding_mask"])
            energy = torch.expm1(batch["features"][..., ENERGY_FEATURE_INDEX])
            is_valid = (~batch["padding_mask"]) & (batch["labels"] != IGNORE_LABEL)

            embeddings, energy, labels, is_valid = (
                embeddings.float().cpu(),
                energy.float().cpu(),
                batch["labels"].cpu(),
                is_valid.cpu(),
            )

            for index in range(labels.shape[0]):
                if max_events is not None and n_events >= max_events:
                    return

                keep = is_valid[index]
                yield EventOutputs(
                    F.normalize(embeddings[index][keep], dim=1),
                    labels[index][keep],
                    energy[index][keep],
                )
                n_events += 1


def evaluate(
    device: torch.device,
    checkpoint: Path,
    loss_name: str,
    stratified_path: Path,
    metrics_path: Path,
    num_workers: int,
    max_events: int | None,
    clustering_parameters: ClusteringParameters,
) -> None:

    train_config = TrainConfig(batch_size=BATCH_SIZE, num_workers=num_workers)
    model = load_model(checkpoint, device)
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")

    stratified = StratifiedReconstruction()
    counts = ReconstructionCounts()
    n_events = 0

    for event in iter_events(model, loader, device, max_events):
        multiplicity = event_multiplicity(event.labels)
        event_reconstruction = event_counts(
            cluster_labels(event.embeddings, clustering_parameters),
            event.labels,
            event.energy,
        )
        intra_distance, separation = event_geometry(event.embeddings, event.labels)
        counts.add(event_reconstruction)
        n_events += 1
        stratified.add(multiplicity, event_reconstruction, intra_distance, separation)

    print(f"events: {n_events} | latent dimensions: {model.output.out_features}")
    print_reconstruction_report(SPLIT, counts)
    print_geometry_report(SPLIT, stratified)

    stratified_metrics = stratified.metrics()

    print_stratified_table(SPLIT, stratified_metrics, stratified.n_events)
    plot_by_multiplicity(
        stratified_metrics,
        stratified.n_events,
        f"SpatialTransformer {loss_name}",
        stratified_path,
    )
    print(f"stratified plot saved to {stratified_path}")
    save_metrics(stratified_metrics, stratified.n_events, metrics_path)
    print(f"metrics saved to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the SpatialTransformer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=str(select_device()))
    parser.add_argument("--stratified-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=None)
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
        loss_name,
        stratified_output,
        metrics_output,
        int(args.num_workers),
        int(args.max_events) if args.max_events is not None else None,
        ClusteringParameters(
            distance_threshold=float(args.distance_threshold),
            method=str(args.linkage_method),
            metric=str(args.metric),
            min_cluster_size=int(args.min_cluster_size),
        ),
    )
