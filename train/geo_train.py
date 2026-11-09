import argparse
import gzip
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from data import (
    DATA_PATH,
    IGNORE_LABEL,
    XYZ_SCALE,
    EventDataset,
    get_splits_by_file,
)
from losses import build_loss
from models.geo_transformer import GeoTransformer
from train.train import select_device

FEATURE_NAMES = ["x", "y", "z", "energy", "eta", "sin_phi", "cos_phi"]
GEOMETRIC_FEATURE_INDICES = [0, 1, 2, 4, 5, 6]
RHO_MIN = 1e-6


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 6
    n_heads: int = 8
    out_dim: int = 3
    bias_hidden: int = 32
    bias_chunk: int = 512
    ffn_mult: int = 2
    drop_path: float = 0.1


@dataclass
class LossConfig:
    name: str = "discriminative"
    kwargs: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    n_epochs: int = 20
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 42
    num_workers: int = 4
    use_bf16: bool = True
    output_dir: str = "models"


def event_to_graph(event: dict[str, Tensor]) -> Data:
    keep = event["labels"] != IGNORE_LABEL
    xyz = event["xyz"][keep]
    energy = event["energy"][keep]
    rho = xyz[:, :2].norm(dim=1).clamp_min(RHO_MIN)
    eta = torch.asinh(xyz[:, 2] / rho)
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])

    features = torch.cat(
        [
            xyz / XYZ_SCALE,
            torch.log1p(energy).unsqueeze(1),
            eta.unsqueeze(1),
            torch.sin(phi).unsqueeze(1),
            torch.cos(phi).unsqueeze(1),
        ],
        dim=1,
    )

    return Data(
        x=features,
        y=event["labels"][keep],
        energy=energy,
        frac=torch.ones_like(energy),
    )


def collate_graphs(events: list[dict[str, Tensor]]) -> Batch:
    return Batch.from_data_list([event_to_graph(event) for event in events])


def build_loader(
    file_paths: list[Path],
    config: TrainConfig,
    pin_memory: bool,
) -> DataLoader:

    has_workers = config.num_workers > 0

    return DataLoader(
        EventDataset(file_paths, seed=config.seed),
        batch_size=config.batch_size,
        collate_fn=collate_graphs,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=has_workers,
        prefetch_factor=2 if has_workers else None,
    )


def build_optimizer(model: nn.Module, config: TrainConfig) -> optim.Optimizer:
    parameters = [p for p in model.parameters() if p.requires_grad]
    decayed = [p for p in parameters if p.ndim > 1]
    undecayed = [p for p in parameters if p.ndim <= 1]

    return optim.AdamW(
        [
            {"params": decayed, "weight_decay": config.weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    total_steps: int,
) -> LambdaLR:

    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_scale)


def count_steps(file_paths: list[Path], batch_size: int) -> int:
    n_events = 0

    for path in file_paths:
        with gzip.open(path, "rt") as stream:
            n_events += sum(1 for _ in stream)

    return math.ceil(n_events / batch_size)


def run_epoch(
    model: GeoTransformer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainConfig,
    optimizer: optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    writer: tensorboard.SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[float, int]:

    is_training = optimizer is not None
    model.train(is_training)
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if config.use_bf16 and device.type == "cuda"
        else nullcontext()
    )

    total_loss = 0.0
    window_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            with autocast:
                embeddings = model(batch)

            loss = loss_fn(embeddings, batch)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

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
    model: GeoTransformer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: nn.Module,
    config: TrainConfig,
    device: torch.device,
    run_name: str,
    steps_per_epoch: int,
) -> None:

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, config.n_epochs * steps_per_epoch)
    writer = tensorboard.SummaryWriter(f"runs/{run_name}")

    best_validation_loss = float("inf")
    global_step = 0

    for epoch in range(config.n_epochs):
        start_time = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{config.n_epochs}")

        train_loss, global_step = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            config,
            optimizer=optimizer,
            scheduler=scheduler,
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
            f"| lr {learning_rate:.2e} | {seconds:.0f}s{' *best*' if is_best else ''}"
        )

        writer.add_scalars(
            "Loss/epoch",
            {"train": train_loss, "validation": validation_loss},
            epoch + 1,
        )

        writer.add_scalar("LearningRate", learning_rate, epoch + 1)
        writer.flush()

        if is_best:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output_dir / f"{run_name}_best.pt")

        torch.save(model.state_dict(), output_dir / f"{run_name}_last.pt")

    writer.close()


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the GeoTransformer")
    parser.add_argument("--loss", default=LossConfig().name)
    parser.add_argument("--epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--no-bf16", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        n_epochs=int(args.epochs),
        num_workers=int(args.num_workers),
        output_dir=str(args.output_dir),
        use_bf16=not args.no_bf16,
    )

    model_config = ModelConfig()
    loss_config = LossConfig(name=str(args.loss))
    torch.manual_seed(train_config.seed)

    device = select_device()
    pin_memory = device.type == "cuda"
    splits = get_splits_by_file(DATA_PATH, seed=train_config.seed)
    train_loader = build_loader(splits["train"], train_config, pin_memory)
    validation_loader = build_loader(splits["val"], train_config, pin_memory)

    model = GeoTransformer(
        in_dim=len(FEATURE_NAMES),
        geom_idx=GEOMETRIC_FEATURE_INDICES,
        **asdict(model_config),
    ).to(device)

    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GeoTransformer ({n_parameters:,} params) | loss: {loss_config.name}")

    run_name = f"geo_{loss_config.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(train_config.output_dir) / f"{run_name}_config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "loss": asdict(loss_config),
                "train": asdict(train_config),
                "features": FEATURE_NAMES,
                "geom_idx": GEOMETRIC_FEATURE_INDICES,
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
        count_steps(splits["train"], train_config.batch_size),
    )


if __name__ == "__main__":
    main()
