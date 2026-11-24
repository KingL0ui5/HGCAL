import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from evals.reco_metrics import ReconstructionCounts

LINE_COLOR = "#2a6fdb"
INPUT_LINE_COLOR = "#8a8a8a"
INPUT_PREFIX = "input "
IDEAL_VALUES = {"multiplicity ratio": 1.0}
UNIT_INTERVAL_METRICS = {"efficiency", "purity"}
PANEL_COLUMNS = 3
SPREAD_RATIO_NAME = "mean intra-cluster spread/mean centroid separation"
TITLE_WIDTH = 34


def event_multiplicity(labels: Tensor) -> int:
    return int(torch.unique(labels).numel())


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


class MultiplicityBin(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    counts: ReconstructionCounts = Field(default_factory=ReconstructionCounts)
    spread_ratios: list[Tensor] = Field(default_factory=list)
    input_spread_ratios: list[Tensor] = Field(default_factory=list)
    n_events: int = 0


def spread_ratio(intra_distances: Tensor, separation: float) -> Tensor:
    return intra_distances.abs().mean(dim=0) / separation


def spread_ratio_by_dimension(
    ratios: dict[int, list[Tensor]],
    prefix: str = "",
) -> dict[str, dict[int, float]]:

    n_dimensions = next((len(r) for rs in ratios.values() for r in rs), 0)

    return {
        f"{prefix}{SPREAD_RATIO_NAME} dim {dim}": {
            k: mean_or_nan([float(r[dim]) for r in rs]) for k, rs in ratios.items()
        }
        for dim in range(n_dimensions)
    }


class MultiplicityStratifiedReco:
    def __init__(self) -> None:
        self.bins: dict[int, MultiplicityBin] = {}

    def add(
        self,
        multiplicity: int,
        counts: ReconstructionCounts,
        intra_distances: Tensor,
        separation: float | None,
    ) -> None:

        bins = self.bins.setdefault(multiplicity, MultiplicityBin())
        bins.counts.add(counts)
        bins.n_events += 1

        if separation is not None:
            bins.spread_ratios.append(spread_ratio(intra_distances, separation))

    def add_input_geometry(
        self,
        multiplicity: int,
        intra_distances: Tensor,
        separation: float | None,
    ) -> None:

        bins = self.bins.setdefault(multiplicity, MultiplicityBin())

        if separation is not None:
            bins.input_spread_ratios.append(spread_ratio(intra_distances, separation))

    @property
    def n_events(self) -> dict[int, int]:
        return {key: self.bins[key].n_events for key in sorted(self.bins)}

    def metrics(self) -> dict[str, dict[int, float]]:
        keys = sorted(self.bins)

        return {
            "efficiency": {k: self.bins[k].counts.efficiency for k in keys},
            "purity": {k: self.bins[k].counts.purity for k in keys},
            "multiplicity ratio": {
                k: self.bins[k].counts.multiplicity_ratio for k in keys
            },
            **spread_ratio_by_dimension({k: self.bins[k].spread_ratios for k in keys}),
            **spread_ratio_by_dimension(
                {k: self.bins[k].input_spread_ratios for k in keys}, INPUT_PREFIX
            ),
        }


def save_metrics(
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metrics": metrics, "n_events": n_events}, indent=2)
    )


def load_metrics(
    input_path: Path,
) -> tuple[dict[str, dict[int, float]], dict[int, int]]:

    payload = json.loads(input_path.read_text())

    metrics = {
        name: {int(key): float(value) for key, value in values.items()}
        for name, values in payload["metrics"].items()
    }

    n_events = {int(key): int(value) for key, value in payload["n_events"].items()}

    return metrics, n_events


def y_axis_top(name: str, values: list[float]) -> float:
    if name in UNIT_INTERVAL_METRICS:
        return 1.05

    finite_values = [value for value in values if math.isfinite(value)]
    peak = max([*finite_values, IDEAL_VALUES.get(name, 0.0), 1e-6])

    return peak * 1.15


def panel_names(metrics: dict[str, dict[int, float]]) -> list[str]:
    return [name for name in metrics if not name.startswith(INPUT_PREFIX)]


def spread_ratio_names(metrics: dict[str, dict[int, float]]) -> list[str]:
    return [name for name in panel_names(metrics) if SPREAD_RATIO_NAME in name]


def reco_metric_names(metrics: dict[str, dict[int, float]]) -> list[str]:
    return [name for name in panel_names(metrics) if SPREAD_RATIO_NAME not in name]


def panel_title(name: str) -> str:
    return textwrap.fill(name, TITLE_WIDTH)


def plot_input_line(
    axis: plt.Axes,
    metrics: dict[str, dict[int, float]],
    name: str,
    keys: list[int],
) -> list[float]:

    if INPUT_PREFIX + name not in metrics:
        return []

    values = [metrics[INPUT_PREFIX + name].get(k, math.nan) for k in keys]
    sns.lineplot(
        x=keys,
        y=values,
        ax=axis,
        color=INPUT_LINE_COLOR,
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=6,
        label="input data",
        legend=False,
    )

    return values


def plot_by_multiplicity(
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
    title: str,
    output_path: Path,
) -> None:

    names = reco_metric_names(metrics)
    n_rows = math.ceil(len(names) / PANEL_COLUMNS)
    sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})
    figure, axes = plt.subplots(
        n_rows,
        PANEL_COLUMNS,
        figsize=(4.2 * PANEL_COLUMNS, 3.4 * n_rows),
        squeeze=False,
    )
    keys = sorted(n_events)

    for axis, name in zip(axes.flat, names):
        values = [metrics[name].get(k, math.nan) for k in keys]
        sns.lineplot(
            x=keys,
            y=values,
            ax=axis,
            color=LINE_COLOR,
            linewidth=2,
            marker="o",
            markersize=8,
        )

        if name in IDEAL_VALUES:
            axis.axhline(IDEAL_VALUES[name], color="grey", linestyle="--", linewidth=1)

        axis.set_xticks(keys)
        axis.set_xlabel("multiplicity")
        axis.set_title(panel_title(name))
        sns.despine(ax=axis)
        axis.set_ylim(0.0, y_axis_top(name, values))

    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"{title} | events per bin: " + ", ".join(f"{k}: {n_events[k]}" for k in keys)
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_spread_ratios(
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
    title: str,
    output_path: Path,
) -> None:

    names = spread_ratio_names(metrics)
    keys = sorted(n_events)
    sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(4.2 * len(names), 3.8),
        squeeze=False,
    )

    for dim, (axis, name) in enumerate(zip(axes.flat, names)):
        values = [metrics[name].get(k, math.nan) for k in keys]
        sns.lineplot(
            x=keys,
            y=values,
            ax=axis,
            color=LINE_COLOR,
            linewidth=2,
            marker="o",
            markersize=8,
            label="latent space",
            legend=False,
        )
        values += plot_input_line(axis, metrics, name, keys)

        axis.legend(loc="best")
        axis.set_xticks(keys)
        axis.set_xlabel("multiplicity")
        axis.set_title(f"dim {dim}")
        sns.despine(ax=axis)
        axis.set_ylim(0.0, y_axis_top(name, values))

    axes.flat[0].set_ylabel(panel_title(SPREAD_RATIO_NAME))
    figure.suptitle(f"{title} | {SPREAD_RATIO_NAME}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
