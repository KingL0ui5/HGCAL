import argparse
from pathlib import Path

import torch
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, IGNORE_LABEL, get_splits_by_file
from train import (
    ModelConfig,
    TrainConfig,
    TransformerModel,
    batch_to_device,
    build_loader,
    select_device,
)


def group_by_sample(file_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}

    for path in file_paths:
        sample_name = path.relative_to(DATA_PATH).parts[0]
        groups.setdefault(sample_name, []).append(path)

    return dict(sorted(groups.items()))


def evaluate_loader(
    model: TransformerModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    num_classes: int,
    device: torch.device,
) -> dict[str, float | Tensor]:

    model.eval()
    total_loss = 0.0
    n_batches = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with no_grad():
        for cpu_batch in loader:
            batch = batch_to_device(cpu_batch, device)
            outputs = model(batch["features"], batch["padding_mask"])
            total_loss += float(loss_fn(outputs, batch["labels"]).item())
            n_batches += 1

            valid = batch["labels"] != IGNORE_LABEL
            labels = batch["labels"][valid]
            predictions = outputs.argmax(dim=1)[valid]
            pair_index = labels * num_classes + predictions
            confusion += (
                torch.bincount(pair_index, minlength=num_classes * num_classes)
                .reshape(num_classes, num_classes)
                .cpu()
            )

    n_points = int(confusion.sum().item())
    n_correct = int(confusion.diagonal().sum().item())
    majority_count = int(confusion.sum(dim=1).max().item())

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": n_correct / max(n_points, 1),
        "majority_baseline": majority_count / max(n_points, 1),
        "n_points": float(n_points),
        "confusion": confusion,
    }


def print_report(name: str, metrics: dict[str, float | Tensor]) -> None:
    confusion = metrics["confusion"]
    assert isinstance(confusion, Tensor)

    print(f"\n== {name} ==")
    print(f"points:            {int(float(metrics['n_points']))}")
    print(f"loss:              {float(metrics['loss']):.4f}")
    print(f"accuracy:          {float(metrics['accuracy']):.4f}")
    print(f"majority baseline: {float(metrics['majority_baseline']):.4f}")

    row_totals = confusion.sum(dim=1)
    print("per-class recall:")

    for class_index in range(confusion.shape[0]):
        total = int(row_totals[class_index].item())

        if total == 0:
            continue

        recall = float(confusion[class_index, class_index].item()) / total
        print(f"  class {class_index}: {recall:.4f} ({total} points)")

    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion)


def load_model(
    checkpoint: Path,
    model_config: ModelConfig,
    device: torch.device,
) -> TransformerModel:

    model = TransformerModel(model_config).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TransformerModel")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train_config = TrainConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    model_config = ModelConfig()
    device = select_device()
    print(f"using device: {device}")

    model = load_model(Path(args.checkpoint), model_config, device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    pin_memory = device.type == "cuda"
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[args.split]

    overall_loader = build_loader(split_files, train_config, pin_memory)
    overall = evaluate_loader(
        model, overall_loader, loss_fn, model_config.num_classes, device
    )
    print_report(f"{args.split} (all samples)", overall)

    for sample_name, sample_files in group_by_sample(split_files).items():
        sample_loader = build_loader(sample_files, train_config, pin_memory)
        sample_metrics = evaluate_loader(
            model, sample_loader, loss_fn, model_config.num_classes, device
        )
        print_report(f"{args.split} / {sample_name}", sample_metrics)


if __name__ == "__main__":
    main()
