import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from train.train_utils import plot_losses

EPOCH_LOSS_TAG = "Loss/epoch"
LEARNING_RATE_TAG = "LearningRate"
DEFAULT_OUTPUT_DIR = Path("trained_models/training_outputs")


def epoch_scalars(series_dir: Path, tag: str = EPOCH_LOSS_TAG) -> list[float]:
    events = EventAccumulator(str(series_dir))
    events.Reload()

    if tag not in events.Tags()["scalars"]:
        return []

    by_epoch = {int(event.step): float(event.value) for event in events.Scalars(tag)}

    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def plot_run(run_dir: Path, output_dir: Path) -> Path | None:
    train_losses = epoch_scalars(run_dir / "Loss_epoch_train")
    validation_losses = epoch_scalars(run_dir / "Loss_epoch_validation")
    learning_rates = epoch_scalars(run_dir, LEARNING_RATE_TAG)
    n_epochs = min(len(train_losses), len(validation_losses))

    if n_epochs == 0:
        return None

    output_path = output_dir / f"{run_dir.name}_loss.png"
    plot_losses(
        train_losses[:n_epochs],
        validation_losses[:n_epochs],
        output_path,
        learning_rates[:n_epochs],
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-epoch train/validation loss from TensorBoard run directories"
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for run_dir in args.run_dirs:
        output_path = plot_run(run_dir, args.output_dir)

        if output_path is None:
            print(f"no epoch losses found in {run_dir}")
            continue

        print(f"loss plot saved to {output_path}")
