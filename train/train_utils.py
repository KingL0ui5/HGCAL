from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data import (
    DATA_PATH,
    EventDataset,
    collate_events,
    get_splits_by_file,
)


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    n_epochs: int = 20
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
