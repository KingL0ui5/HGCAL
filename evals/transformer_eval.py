import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, IGNORE_LABEL, get_splits_by_file
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
from models.transformer import ModelConfig, TransformerModel
from train.train import TrainConfig, batch_to_device, build_loader, select_device

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "trained_models"
    / "model_20261108_184730.pt"
)
EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
LOSS_NAME = "crossentropy"
SPLIT = "test"
BATCH_SIZE = 8
PAIRS_PER_EVENT = 20_000
ENERGY_FEATURE_INDEX = 3
EMBEDDING_SOURCES = ("hidden", "logits")


class EventOutputs(NamedTuple):
    hidden: Tensor
    logits: Tensor
    labels: Tensor
    energy: Tensor


@dataclass
class ClassificationStats:
    n_correct: int = 0
    n_points: int = 0
    slot_counts: Tensor = field(
        default_factory=lambda: torch.zeros(ModelConfig().num_classes, dtype=torch.long)
    )

    def add(self, logits: Tensor, labels: Tensor) -> None:
        self.n_correct += int((logits.argmax(dim=1) == labels).sum().item())
        self.n_points += int(labels.numel())
        self.slot_counts += torch.bincount(labels, minlength=ModelConfig().num_classes)

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(self.n_points, 1)

    @property
    def majority_baseline(self) -> float:
        return int(self.slot_counts.max().item()) / max(self.n_points, 1)


def output_path(source: str, kind: str) -> Path:
    return EVAL_DATA_DIR / f"transformer_{LOSS_NAME}_{source}_{kind}.png"


def load_model(checkpoint: Path, device: torch.device) -> TransformerModel:
    model = TransformerModel(ModelConfig()).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model


def iter_events(
    model: TransformerModel,
    loader: DataLoader,
    device: torch.device,
    max_events: int | None,
) -> Iterator[EventOutputs]:

    model.eval()
    n_events = 0

    with no_grad():
        for cpu_batch in loader:
            batch = batch_to_device(cpu_batch, device)
            hidden = model.encoder(
                model.input_projection(batch["features"]),
                src_key_padding_mask=batch["padding_mask"],
            )
            logits = model.classifier(hidden)
            energy = torch.expm1(batch["features"][..., ENERGY_FEATURE_INDEX])
            is_valid = (~batch["padding_mask"]) & (batch["labels"] != IGNORE_LABEL)

            hidden, logits, energy, labels, is_valid = (
                hidden.float().cpu(),
                logits.float().cpu(),
                energy.float().cpu(),
                batch["labels"].cpu(),
                is_valid.cpu(),
            )

            for index in range(labels.shape[0]):
                if max_events is not None and n_events >= max_events:
                    return

                keep = is_valid[index]
                yield EventOutputs(
                    hidden[index][keep],
                    logits[index][keep],
                    labels[index][keep],
                    energy[index][keep],
                )
                n_events += 1


def select_embeddings(event: EventOutputs, source: str) -> Tensor:
    raw = event.hidden if source == "hidden" else event.logits

    return F.normalize(raw, dim=1)


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


def plot_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    source: str,
    output_path: Path,
) -> float:

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Transformer {source} (AUC = {auc:.4f})",
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


def print_classification_report(stats: ClassificationStats) -> None:
    print(f"\n== {SPLIT}: slot classification ==")
    print(f"points:               {stats.n_points}")
    print(f"accuracy:             {stats.accuracy:.4f}")
    print(f"majority-slot rate:   {stats.majority_baseline:.4f}")
    print(f"slot counts:          {stats.slot_counts.tolist()}")


def evaluate(
    device: torch.device,
    checkpoint: Path,
    source: str,
    roc_path: Path | None,
    stratified_path: Path,
    pairs_per_event: int,
    num_workers: int,
    max_events: int | None,
    clue_parameters: ClueParameters,
) -> None:

    train_config = TrainConfig(batch_size=BATCH_SIZE, num_workers=num_workers)
    model = load_model(checkpoint, device)
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")

    generator = torch.Generator().manual_seed(train_config.seed)
    classification = ClassificationStats()
    classification_by_multiplicity: dict[int, ClassificationStats] = {}
    stratified = StratifiedReconstruction()
    counts = ReconstructionCounts()
    pair_scores: list[Tensor] = []
    pair_labels: list[Tensor] = []
    n_events = 0

    for event in iter_events(model, loader, device, max_events):
        embeddings = select_embeddings(event, source)
        multiplicity = event_multiplicity(event.labels)
        event_reconstruction = event_counts(
            clue_labels(embeddings, clue_parameters), event.labels, event.energy
        )
        classification.add(event.logits, event.labels)
        classification_by_multiplicity.setdefault(
            multiplicity, ClassificationStats()
        ).add(event.logits, event.labels)
        counts.add(event_reconstruction)
        n_events += 1
        scores: Tensor | None = None
        same_shower: Tensor | None = None

        if roc_path is not None:
            scores, same_shower = sample_pair_scores(
                embeddings, event.labels, pairs_per_event, generator
            )
            pair_scores.append(scores)
            pair_labels.append(same_shower)

        stratified.add(multiplicity, event_reconstruction, scores, same_shower)

    print(f"events: {n_events} | embedding source: {source} (L2-normalised)")
    print_classification_report(classification)
    print_reconstruction_report(SPLIT, counts)

    stratified_metrics = stratified.metrics()
    stratified_metrics["slot accuracy"] = {
        key: classification_by_multiplicity[key].accuracy
        for key in sorted(classification_by_multiplicity)
    }
    print_stratified_table(SPLIT, stratified_metrics, stratified.n_events)
    plot_by_multiplicity(
        stratified_metrics,
        stratified.n_events,
        f"Transformer {source} ({SPLIT})",
        stratified_path,
    )
    print(f"stratified plot saved to {stratified_path}")

    if roc_path is None:
        return

    scores_array = torch.cat(pair_scores).numpy().astype(np.float32)
    labels_array = torch.cat(pair_labels).numpy().astype(np.int8)
    auc = plot_roc(scores_array, labels_array, source, roc_path)
    print(
        f"\nROC pairs: {len(labels_array)} | "
        f"same-shower fraction: {labels_array.mean():.4f}"
    )
    print(f"ROC AUC: {auc:.4f} | saved to {roc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the raw Transformer")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default=str(select_device()))
    parser.add_argument("--embedding", choices=EMBEDDING_SOURCES, default="hidden")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--roc-output", type=Path, default=None)
    parser.add_argument("--stratified-output", type=Path, default=None)
    parser.add_argument("--pairs-per-event", type=int, default=PAIRS_PER_EVENT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=None)
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
    source = str(args.embedding)
    roc_output = (
        Path(args.roc_output) if args.roc_output else output_path(source, "roc")
    )
    stratified_output = (
        Path(args.stratified_output)
        if args.stratified_output
        else output_path(source, "by_multiplicity")
    )
    evaluate(
        torch.device(str(args.device)),
        Path(args.checkpoint),
        source,
        roc_output if args.roc else None,
        stratified_output,
        int(args.pairs_per_event),
        int(args.num_workers),
        int(args.max_events) if args.max_events is not None else None,
        ClueParameters(
            density_radius=float(args.density_radius),
            min_density=float(args.min_density),
            seeding_distance=float(args.seeding_distance),
            outlier_distance=float(args.outlier_distance),
        ),
    )
