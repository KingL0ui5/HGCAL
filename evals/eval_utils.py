import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from data import DATA_PATH
from evals.geometry_metrics import geometry_metrics

ModelT = TypeVar("ModelT", bound=nn.Module)
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}")


class EventOutputs(NamedTuple):
    embeddings: Tensor
    labels: Tensor
    energy: Tensor
    positions: Tensor


class Timings(BaseModel):
    inference: float = 0.0
    clustering: float = 0.0


class EmbeddingStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: float = 0.0
    intra_distance: Tensor | None = None
    centroid_separation: float = 0.0
    n_batches: int = 0
    n_events: int = 0
    n_points: int = 0
    n_separation_events: int = 0


def group_by_sample(file_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}

    for path in file_paths:
        sample_name = path.relative_to(DATA_PATH).parts[0]
        groups.setdefault(sample_name, []).append(path)

    return dict(sorted(groups.items()))


def accumulate_event(
    stats: EmbeddingStats,
    embeddings: Tensor,
    labels: Tensor,
) -> None:

    intra_distances, separation = geometry_metrics(embeddings, labels)
    event_spread = intra_distances.abs().mean(dim=0)
    stats.intra_distance = (
        event_spread
        if stats.intra_distance is None
        else stats.intra_distance + event_spread
    )
    stats.n_events += 1
    stats.n_points += int(embeddings.shape[0])

    if separation is not None:
        stats.centroid_separation += separation
        stats.n_separation_events += 1


def print_report(name: str, stats: EmbeddingStats) -> None:
    print(f"\n== {name} ==")
    print(f"events:                   {stats.n_events}")
    print(f"points:                   {stats.n_points}")
    print(f"loss:                     {stats.loss / max(stats.n_batches, 1):.4f}")

    if stats.intra_distance is not None:
        for dim, spread in enumerate((stats.intra_distance / stats.n_events).tolist()):
            print(f"mean intra-cluster spread dim {dim}: {spread:.4f}")

    print(
        f"mean centroid separation: "
        f"{stats.centroid_separation / max(stats.n_separation_events, 1):.4f}"
    )


def report_by_sample(
    split: str,
    split_files: list[Path],
    evaluate_files: Callable[[list[Path]], EmbeddingStats],
) -> None:

    print_report(f"{split} (all samples)", evaluate_files(split_files))

    for sample_name, sample_files in group_by_sample(split_files).items():
        print_report(f"{split} / {sample_name}", evaluate_files(sample_files))


def load_run_config(checkpoint: Path) -> dict[str, Any]:
    run_name = checkpoint.stem.removesuffix("_best").removesuffix("_last")
    config_path = checkpoint.with_name(f"{run_name}_config.json")

    if not config_path.exists():
        return {}

    return json.loads(config_path.read_text())


def checkpoint_loss_name(checkpoint: Path, filename_pattern: re.Pattern[str]) -> str:
    configured_loss = load_run_config(checkpoint).get("loss", {}).get("name")

    if configured_loss:
        return str(configured_loss)

    match = filename_pattern.match(checkpoint.stem)

    return match.group("loss") if match else "unknown"


def checkpoint_run_id(checkpoint: Path) -> str | None:
    match = RUN_ID_PATTERN.search(checkpoint.stem)

    return match.group(0) if match else None


def load_weights(
    model: ModelT,
    checkpoint: Path,
    device: torch.device,
) -> ModelT:

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model
