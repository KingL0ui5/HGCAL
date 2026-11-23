import argparse
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, IGNORE_LABEL, get_splits_by_file
from evals.eval_utils import checkpoint_loss_name, load_run_config, load_weights
from evals.geometry_metrics import geometry_metrics
from evals.pileup_stratified import (
    PileupStratifiedReco,
    event_multiplicity,
    plot_by_multiplicity,
    save_metrics,
)
from evals.reco_metrics import (
    ClusteringParameters,
    ReconstructionCounts,
    cluster_labels,
    event_counts,
)
from models.spatial_transformer import SpatialConfig, SpatialTransformer
from train.train_utils import (
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


class Timings(BaseModel):
    inference: float = 0.0
    clustering: float = 0.0


def output_path(loss_name: str, kind: str, suffix: str = "png") -> Path:
    return EVAL_DATA_DIR / f"spatial_transformer_{loss_name}_{kind}.{suffix}"


def iter_events(
    model: SpatialTransformer,
    loader: DataLoader,
    device: torch.device,
    max_events: int | None,
    timings: Timings,
) -> Iterator[EventOutputs]:

    model.eval()
    n_events = 0

    with no_grad():
        for cpu_batch in loader:
            start = time.perf_counter()
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
            timings.inference += time.perf_counter() - start

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
    model_config = SpatialConfig(**load_run_config(checkpoint).get("model", {}))

    model = load_weights(
        SpatialTransformer(model_config).to(device), checkpoint, device
    )
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")

    stratified = PileupStratifiedReco()
    counts = ReconstructionCounts()

    timings = Timings()
    n_events = 0

    for event in iter_events(model, loader, device, max_events, timings):
        multiplicity = event_multiplicity(event.labels)
        start = time.perf_counter()
        predicted = cluster_labels(event.embeddings, clustering_parameters)
        timings.clustering += time.perf_counter() - start
        event_reconstruction = event_counts(predicted, event.labels, event.energy)
        intra_distances, separation = geometry_metrics(event.embeddings, event.labels)
        counts.add(event_reconstruction)
        n_events += 1
        stratified.add(multiplicity, event_reconstruction, intra_distances, separation)

    print(f"events: {n_events} | latent dimensions: {model.output.out_features}")
    print(
        f"inference time: {timings.inference:.2f}s "
        f"({1e3 * timings.inference / max(n_events, 1):.2f} ms/event) | "
        f"agglomerative clustering time: {timings.clustering:.2f}s "
        f"({1e3 * timings.clustering / max(n_events, 1):.2f} ms/event)"
    )

    stratified_metrics = stratified.metrics()

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
    loss_name = checkpoint_loss_name(checkpoint, CHECKPOINT_LOSS_PATTERN)
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
