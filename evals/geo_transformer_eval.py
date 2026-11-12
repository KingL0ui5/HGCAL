import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from evals.evaluate import (
    evaluate_loader,
    event_geometry,
    load_model,
    print_report,
)
from evals.reco_metrics import (
    ClueParameters,
    ReconstructionCounts,
    clue_labels,
    event_counts,
    print_reconstruction_report,
)
from evals.stratified import (
    StratifiedReconstruction,
    event_multiplicity,
    plot_by_multiplicity,
    print_stratified_table,
)
from losses import build_loss
from train.geo_train import (
    LossConfig,
    ModelConfig,
    TrainConfig,
    build_loader,
)
from train.train import select_device

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "trained_models"
    / "geo_discriminative_20261109_143754_best.pt"
)
EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
ROC_PATH = EVAL_DATA_DIR / "geo_transformer_roc.png"
STRATIFIED_PATH = EVAL_DATA_DIR / "geo_transformer_by_multiplicity.png"
SPLIT = "test"
BATCH_SIZE = 1
NUM_WORKERS = 2
PAIRS_PER_EVENT = 20_000


def sample_pair_scores(
    embeddings: Tensor,
    labels: Tensor,
    n_pairs: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:

    n_points = embeddings.shape[0]
    first = torch.randint(n_points, (n_pairs,), generator=generator)
    second = torch.randint(n_points, (n_pairs,), generator=generator)
    distinct = first != second
    first, second = first[distinct], second[distinct]

    distances = (embeddings[first] - embeddings[second]).norm(dim=1)
    same_shower = labels[first] == labels[second]

    return -distances, same_shower


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


def plot_roc(scores: np.ndarray, labels: np.ndarray, output_path: Path) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"GeoTransformer (AUC = {auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    axis.set_xlabel("False positive rate (different showers paired)")
    axis.set_ylabel("True positive rate (same-shower pairs recovered)")
    axis.set_title(f"Same-shower pair ROC ({SPLIT} split)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return auc


def evaluate(
    device: torch.device,
    checkpoint: Path,
    roc_path: Path | None,
    stratified_path: Path,
    pairs_per_event: int,
    clue_parameters: ClueParameters,
) -> None:

    train_config = TrainConfig(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    loss_config = LossConfig()

    model = load_model(checkpoint, ModelConfig(), device)
    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)

    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")
    print_report(
        f"{SPLIT} (all samples)", evaluate_loader(model, loader, loss_fn, device)
    )

    event_loader = build_loader(split_files, train_config, device.type == "cuda")
    generator = torch.Generator().manual_seed(train_config.seed)
    counts = ReconstructionCounts()
    stratified = StratifiedReconstruction()
    intra_by_multiplicity: dict[int, list[float]] = {}
    separation_by_multiplicity: dict[int, list[float]] = {}
    pair_scores: list[Tensor] = []
    pair_labels: list[Tensor] = []

    for event in iter_event_embeddings(model, event_loader, device):
        reco_labels = clue_labels(event.embeddings, clue_parameters)
        event_reconstruction = event_counts(reco_labels, event.labels, event.energy)
        counts.add(event_reconstruction)
        multiplicity = event_multiplicity(event.labels)
        intra_distance, separation = event_geometry(event.embeddings, event.labels)
        intra_by_multiplicity.setdefault(multiplicity, []).append(intra_distance)

        if separation is not None:
            separation_by_multiplicity.setdefault(multiplicity, []).append(separation)

        scores: Tensor | None = None
        same_shower: Tensor | None = None

        if roc_path is not None:
            scores, same_shower = sample_pair_scores(
                event.embeddings, event.labels, pairs_per_event, generator
            )
            pair_scores.append(scores)
            pair_labels.append(same_shower)

        stratified.add(multiplicity, event_reconstruction, scores, same_shower)

    print_reconstruction_report(SPLIT, counts)

    stratified_metrics = stratified.metrics()
    stratified_metrics["intra-cluster distance"] = {
        key: float(np.mean(values))
        for key, values in sorted(intra_by_multiplicity.items())
    }
    stratified_metrics["centroid separation"] = {
        key: float(np.mean(values))
        for key, values in sorted(separation_by_multiplicity.items())
    }
    print_stratified_table(SPLIT, stratified_metrics, stratified.n_events)
    plot_by_multiplicity(
        stratified_metrics,
        stratified.n_events,
        f"GeoTransformer ({SPLIT})",
        stratified_path,
    )
    print(f"stratified plot saved to {stratified_path}")

    if roc_path is None:
        return

    scores_array = torch.cat(pair_scores).numpy().astype(np.float32)
    labels_array = torch.cat(pair_labels).numpy().astype(np.int8)
    auc = plot_roc(scores_array, labels_array, roc_path)
    print(
        f"\nROC pairs: {len(labels_array)} | "
        f"same-shower fraction: {labels_array.mean():.4f}"
    )
    print(f"ROC AUC: {auc:.4f} | saved to {roc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GeoTransformer")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default=str(select_device()))
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--roc-output", type=Path, default=ROC_PATH)
    parser.add_argument("--stratified-output", type=Path, default=STRATIFIED_PATH)
    parser.add_argument("--pairs-per-event", type=int, default=PAIRS_PER_EVENT)
    parser.add_argument(
        "--density-radius", type=float, default=ClueParameters().density_radius
    )
    parser.add_argument(
        "--min-density", type=float, default=ClueParameters().min_density
    )
    parser.add_argument(
        "--seeding-distance", type=float, default=ClueParameters().seeding_distance
    )
    parser.add_argument(
        "--outlier-distance", type=float, default=ClueParameters().outlier_distance
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        torch.device(str(args.device)),
        Path(args.checkpoint),
        Path(args.roc_output) if args.roc else None,
        Path(args.stratified_output),
        int(args.pairs_per_event),
        ClueParameters(
            density_radius=float(args.density_radius),
            min_density=float(args.min_density),
            seeding_distance=float(args.seeding_distance),
            outlier_distance=float(args.outlier_distance),
        ),
    )
