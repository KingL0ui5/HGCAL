import argparse
from pathlib import Path

import torch

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
SPLIT = "test"
BATCH_SIZE = 1
NUM_WORKERS = 2


def evaluate(device: torch.device, checkpoint: Path) -> None:
    train_config = TrainConfig(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    loss_config = LossConfig()

    model = load_model(checkpoint, ModelConfig(), device)
    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)

    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[SPLIT]
    loader = build_loader(split_files, train_config, device.type == "cuda")
    print_report(
        f"{SPLIT} (all samples)", evaluate_loader(model, loader, loss_fn, device)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GeoTransformer")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default=str(select_device()))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(torch.device(str(args.device)), Path(args.checkpoint))
