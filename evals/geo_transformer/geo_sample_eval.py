import argparse
from pathlib import Path

from data import DATA_PATH, get_splits_by_file
from evals.eval_utils import (
    EmbeddingStats,
    load_run_config,
    report_by_sample,
)
from evals.geo_transformer.geo_transformer_eval import evaluate_loader, load_model
from losses import build_loss
from train.geo_transformer.geo_train import (
    LossConfig,
    ModelConfig,
    TrainConfig,
    build_loader,
)
from train.train_utils import select_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained GeoTransformer")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--loss", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser


def main_geo_transformer(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    run_config = load_run_config(checkpoint)
    model_config = ModelConfig(**run_config.get("model", {}))
    saved_loss = run_config.get("loss", {})
    loss_config = LossConfig(
        name=str(args.loss or saved_loss.get("name", LossConfig().name)),
        kwargs={} if args.loss else dict(saved_loss.get("kwargs", {})),
    )
    train_config = TrainConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(run_config.get("train", {}).get("seed", TrainConfig().seed)),
    )

    device = select_device()
    print(f"using device: {device} | loss: {loss_config.name}")

    model = load_model(checkpoint, model_config, device)
    loss_fn = build_loss(loss_config.name, **loss_config.kwargs).to(device)
    pin_memory = device.type == "cuda"
    split_files = get_splits_by_file(DATA_PATH, seed=train_config.seed)[args.split]

    def evaluate_files(file_paths: list[Path]) -> EmbeddingStats:
        loader = build_loader(file_paths, train_config, pin_memory)

        return evaluate_loader(model, loader, loss_fn, device)

    report_by_sample(str(args.split), split_files, evaluate_files)


def main() -> None:
    args = build_arg_parser().parse_args()

    main_geo_transformer(args)


if __name__ == "__main__":
    main()
