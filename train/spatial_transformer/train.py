import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from data import IGNORE_LABEL
from losses import build_loss
from models.spatial_transformer import SpatialConfig, SpatialTransformer
from train.train_utils import (
    TrainConfig,
    batch_to_device,
    build_loaders,
    select_device,
)

ENERGY_FEATURE_INDEX = 3
LOSS_CHOICES = ("infonce", "discriminative")


@dataclass
class FlatBatch:
    y: Tensor
    batch: Tensor
    energy: Tensor
    frac: Tensor


def flatten_batch(
    embeddings: Tensor,
    batch: dict[str, Tensor],
) -> tuple[Tensor, FlatBatch]:

    is_valid = (~batch["padding_mask"]) & (batch["labels"] != IGNORE_LABEL)
    event_index = torch.arange(embeddings.shape[0], device=embeddings.device)
    event_index = event_index.unsqueeze(1).expand_as(is_valid)
    energy = torch.expm1(batch["features"][..., ENERGY_FEATURE_INDEX])[is_valid].float()

    return embeddings[is_valid], FlatBatch(
        y=batch["labels"][is_valid],
        batch=event_index[is_valid],
        energy=energy,
        frac=torch.ones_like(energy),
    )


def run_epoch(
    model: SpatialTransformer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainConfig,
    optimizer: optim.Optimizer | None = None,
    writer: tensorboard.SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[float, int]:

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    window_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_training):
        for cpu_batch in loader:
            batch = batch_to_device(cpu_batch, device)
            embeddings = model(batch["features"], batch["padding_mask"])
            flat_embeddings, flat_batch = flatten_batch(embeddings, batch)
            loss = loss_fn(flat_embeddings, flat_batch)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                global_step += 1

            batch_loss = float(loss.item())
            total_loss += batch_loss
            window_loss += batch_loss
            n_batches += 1

            if is_training and n_batches % config.log_every == 0:
                mean_window_loss = window_loss / config.log_every
                print(f"  batch {n_batches} loss: {mean_window_loss:.4f}")

                if writer is not None:
                    writer.add_scalar("Loss/train", mean_window_loss, global_step)

                window_loss = 0.0

    return total_loss / max(n_batches, 1), global_step


def train(
    model: SpatialTransformer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: nn.Module,
    config: TrainConfig,
    device: torch.device,
    run_name: str,
    output_dir: Path,
) -> None:

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    writer = tensorboard.SummaryWriter(f"runs/{run_name}")
    best_validation_loss = float("inf")
    global_step = 0

    for epoch in range(config.n_epochs):
        start_time = time.perf_counter()
        print(f"epoch {epoch + 1}/{config.n_epochs}")

        train_loss, global_step = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            config,
            optimizer=optimizer,
            writer=writer,
            global_step=global_step,
        )
        validation_loss, _ = run_epoch(
            model, validation_loader, loss_fn, device, config
        )

        seconds = time.perf_counter() - start_time
        is_best = validation_loss < best_validation_loss
        print(
            f"  train loss {train_loss:.4f} | val loss {validation_loss:.4f} "
            f"| {seconds:.0f}s{' *best*' if is_best else ''}"
        )

        writer.add_scalars(
            "Loss/epoch",
            {"train": train_loss, "validation": validation_loss},
            epoch + 1,
        )
        writer.flush()

        if is_best:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output_dir / f"{run_name}_best.pt")

        torch.save(model.state_dict(), output_dir / f"{run_name}_last.pt")

    writer.close()


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the SpatialTransformer with a metric-learning loss"
    )
    parser.add_argument("--loss", choices=LOSS_CHOICES, required=True)
    parser.add_argument("--epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--out-dim", type=int, default=SpatialConfig().out_dim)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("trained_models/spatial_transformer")
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        n_epochs=int(args.epochs),
        num_workers=int(args.num_workers),
    )
    model_config = SpatialConfig(out_dim=int(args.out_dim))
    loss_name = str(args.loss)
    output_dir = Path(args.output_dir)
    torch.manual_seed(train_config.seed)

    device = select_device()
    train_loader, validation_loader = build_loaders(
        train_config, pin_memory=device.type == "cuda"
    )
    model = SpatialTransformer(model_config).to(device)
    loss_fn = build_loss(loss_name).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpatialTransformer ({n_parameters:,} params) | loss: {loss_name}")

    run_name = f"spatial_{loss_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run_name}_config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "loss": {"name": loss_name},
                "train": asdict(train_config),
            },
            indent=2,
        )
    )

    train(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        train_config,
        device,
        run_name,
        output_dir,
    )


if __name__ == "__main__":
    main()
