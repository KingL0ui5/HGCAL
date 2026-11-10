import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from evaluate import evaluate_loader, load_model, print_report
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
ROC_PATH = Path(__file__).resolve().parent / "geo_transformer_roc.png"
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


def collect_pair_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pairs_per_event: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    all_scores: list[Tensor] = []
    all_labels: list[Tensor] = []

    with no_grad():
        for cpu_batch in loader:
            batch: Any = cpu_batch.to(device)
            embeddings = model(batch).float().cpu()
            labels = batch.y.cpu()
            event_sizes = torch.bincount(batch.batch).tolist()

            for event_embeddings, event_labels in zip(
                torch.split(embeddings, event_sizes),
                torch.split(labels, event_sizes),
            ):
                scores, same_shower = sample_pair_scores(
                    event_embeddings, event_labels, pairs_per_event, generator
                )
                all_scores.append(scores)
                all_labels.append(same_shower)

    return (
        torch.cat(all_scores).numpy().astype(np.float32),
        torch.cat(all_labels).numpy().astype(np.int8),
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
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return auc


def evaluate(
    device: torch.device,
    checkpoint: Path,
    roc_path: Path | None,
    pairs_per_event: int,
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

    if roc_path is None:
        return

    roc_loader = build_loader(split_files, train_config, device.type == "cuda")
    scores, labels = collect_pair_scores(
        model, roc_loader, device, pairs_per_event, train_config.seed
    )
    auc = plot_roc(scores, labels, roc_path)
    print(f"\nROC pairs: {len(labels)} | same-shower fraction: {labels.mean():.4f}")
    print(f"ROC AUC: {auc:.4f} | saved to {roc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GeoTransformer")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default=str(select_device()))
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--roc-output", type=Path, default=ROC_PATH)
    parser.add_argument("--pairs-per-event", type=int, default=PAIRS_PER_EVENT)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        torch.device(str(args.device)),
        Path(args.checkpoint),
        Path(args.roc_output) if args.roc else None,
        int(args.pairs_per_event),
    )
