from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor, nn, no_grad, optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from data import (
    DATA_PATH,
    PADDING_LABEL,
    EventDataset,
    collate_events,
    get_splits_by_file,
)


@dataclass
class ModelConfig:
    num_classes: int = 2
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    n_epochs: int = 20
    log_every: int = 100
    grad_clip: float = 1.0
    seed: int = 42


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(4, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, features: Tensor, padding_mask: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        hidden = self.input_projection(features)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

        return self.classifier(hidden).transpose(1, 2)


def build_loaders(
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:

    splits = get_splits_by_file(DATA_PATH, seed=seed)

    train_loader = DataLoader(
        EventDataset(splits["train"], seed=seed),
        batch_size=batch_size,
        collate_fn=collate_events,
    )

    validation_loader = DataLoader(
        EventDataset(splits["val"], seed=seed),
        batch_size=batch_size,
        collate_fn=collate_events,
    )

    return train_loader, validation_loader


def train_epoch(
    model: TransformerModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    config: TrainConfig,
    tb_writer: tensorboard.SummaryWriter,
    global_step: int,
) -> tuple[float, int]:

    model.train()
    running_loss = 0.0
    last_loss = 0.0

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        outputs = model(batch["features"], batch["padding_mask"])
        loss = loss_fn(outputs, batch["labels"])
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        running_loss += float(loss.item())
        global_step += 1

        if (i + 1) % config.log_every == 0:
            last_loss = running_loss / config.log_every
            print(f"  batch {i + 1} loss: {last_loss:.4f}")
            tb_writer.add_scalar("Loss/train", last_loss, global_step)
            running_loss = 0.0

    return last_loss, global_step


def validate(
    model: TransformerModel,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float]:

    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_points = 0

    with no_grad():
        for batch in loader:
            outputs = model(batch["features"], batch["padding_mask"])
            total_loss += float(loss_fn(outputs, batch["labels"]).item())
            n_batches += 1

            valid = ~batch["padding_mask"]
            predictions = outputs.argmax(dim=1)

            n_correct += int(
                (predictions[valid] == batch["labels"][valid]).sum().item()
            )

            n_points += int(valid.sum().item())

    return total_loss / max(n_batches, 1), n_correct / max(n_points, 1)


def train(
    model: TransformerModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig,
) -> None:

    loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_LABEL)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = tensorboard.SummaryWriter(f"runs/hgcal_transformer_{timestamp}")
    best_validation_loss = float("inf")
    global_step = 0

    for epoch in range(config.n_epochs):
        print(f"epoch {epoch + 1}/{config.n_epochs}")
        train_loss, global_step = train_epoch(
            model, train_loader, loss_fn, optimizer, config, writer, global_step
        )
        validation_loss, validation_accuracy = validate(
            model, validation_loader, loss_fn
        )
        print(
            f"  train loss {train_loss:.4f} | val loss {validation_loss:.4f} "
            f"| val acc {validation_accuracy:.4f}"
        )

        writer.add_scalars(
            "Loss/epoch",
            {"validation": validation_loss},
            epoch + 1,
        )

        writer.add_scalar("Accuracy/validation", validation_accuracy, epoch + 1)
        writer.flush()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), f"model_{timestamp}.pt")

    writer.close()


def main() -> None:
    train_config = TrainConfig()
    torch.manual_seed(train_config.seed)

    train_loader, validation_loader = build_loaders(
        train_config.batch_size, train_config.seed
    )

    model = TransformerModel(ModelConfig())
    train(model, train_loader, validation_loader, train_config)


if __name__ == "__main__":
    main()
