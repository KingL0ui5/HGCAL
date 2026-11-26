from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

from data import (
    DATA_PATH,
    EventDataset,
    collate_events,
    get_splits_by_file,
)

TRAINING_OUTPUT_DIR = Path("trained_models/training_output")


class TrainConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    n_epochs: int = 50
    log_every: int = 100
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def batch_to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_loader(
    file_paths: list[Path],
    config: TrainConfig,
    pin_memory: bool,
) -> DataLoader:

    has_workers = config.num_workers > 0

    return DataLoader(
        EventDataset(file_paths, seed=config.seed),
        batch_size=config.batch_size,
        collate_fn=collate_events,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=has_workers,
        prefetch_factor=2 if has_workers else None,
    )


def build_loaders(
    config: TrainConfig,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:

    splits = get_splits_by_file(DATA_PATH, seed=config.seed)
    train_loader = build_loader(splits["train"], config, pin_memory)
    validation_loader = build_loader(splits["val"], config, pin_memory)

    return train_loader, validation_loader


def plot_losses(
    train_losses: list[float],
    validation_losses: list[float],
    path: Path,
    learning_rates: list[float] | None = None,
) -> None:

    epochs = range(1, len(train_losses) + 1)

    if not learning_rates:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig, (ax, lr_ax) = plt.subplots(
            2, 1, figsize=(6, 6), sharex=True, height_ratios=(2, 1)
        )
        lr_ax.plot(range(1, len(learning_rates) + 1), learning_rates, color="tab:green")
        lr_ax.set_ylabel("learning rate")
        lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        lr_ax.set_xlabel("epoch")

    ax.plot(epochs, train_losses, marker="o", label="train")
    ax.plot(epochs, validation_losses, marker="o", label="validation")
    ax.set_ylabel("loss")
    ax.legend()

    if not learning_rates:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
