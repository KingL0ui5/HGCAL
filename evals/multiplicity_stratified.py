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

from evals.geometry_metrics import EventGeometry
from evals.reco_metrics import ReconstructionCounts

LINE_COLOR = "#2a6fdb"
INPUT_LINE_COLOR = "#8a8a8a"
INPUT_PREFIX = "input "
IDEAL_VALUES = {"multiplicity ratio": 1.0}
UNIT_INTERVAL_METRICS = {"efficiency", "purity"}
PANEL_COLUMNS = 3
SPREAD_RATIO_NAME = "mean intra-cluster spread/mean centroid separation"
OVERLAP_LINE_COLOR = "#c0392b"
# a 3D gaussian of width s has mean distance to centre s * sqrt(8 / pi), and mean
# |offset| s * sqrt(2 / pi) along one axis; two equal gaussians merge into one peak
# once their centres are closer than 2s
GAUSSIAN_OVERLAP_RATIO = math.sqrt(8.0 / math.pi) / 2.0
AXIS_GAUSSIAN_OVERLAP_RATIO = math.sqrt(2.0 / math.pi) / 2.0
TITLE_WIDTH = 34


def event_multiplicity(labels: Tensor) -> int:
    return int(torch.unique(labels).numel())


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def finite_mean_or_nan(values: list[float]) -> float:
    return mean_or_nan([value for value in values if math.isfinite(value)])


class MultiplicityBin(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    counts: ReconstructionCounts = Field(default_factory=ReconstructionCounts)
    spread_ratios: list[float] = Field(default_factory=list)
    axis_spread_ratios: list[Tensor] = Field(default_factory=list)
    input_spread_ratios: list[float] = Field(default_factory=list)
    input_axis_spread_ratios: list[Tensor] = Field(default_factory=list)
    n_events: int = 0


def spread_ratio(geometry: EventGeometry) -> float | None:
    if geometry.separation is None:
        return None

    spread = float(geometry.intra_distances.norm(dim=1).mean().item())

    return spread / geometry.separation


def axis_spread_ratio(geometry: EventGeometry) -> Tensor | None:
    if geometry.axis_separation is None:
        return None

    return geometry.intra_distances.abs().mean(dim=0) / geometry.axis_separation


def spread_ratio_metric(
    ratios: dict[int, list[float]],
    prefix: str = "",
) -> dict[str, dict[int, float]]:

    return {
        f"{prefix}{SPREAD_RATIO_NAME}": {k: mean_or_nan(rs) for k, rs in ratios.items()}
    }


def spread_ratio_by_dimension(
    ratios: dict[int, list[Tensor]],
    prefix: str = "",
) -> dict[str, dict[int, float]]:

    n_dimensions = next((len(r) for rs in ratios.values() for r in rs), 0)

    return {
        f"{prefix}{SPREAD_RATIO_NAME} dim {dim}": {
            k: finite_mean_or_nan([float(r[dim]) for r in rs])
            for k, rs in ratios.items()
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
        geometry: EventGeometry,
    ) -> None:

        bins = self.bins.setdefault(multiplicity, MultiplicityBin())
        bins.counts.add(counts)
        bins.n_events += 1
        ratio, axis_ratio = spread_ratio(geometry), axis_spread_ratio(geometry)

        if ratio is not None and axis_ratio is not None:
            bins.spread_ratios.append(ratio)
            bins.axis_spread_ratios.append(axis_ratio)

    def add_input_geometry(
        self,
        multiplicity: int,
        geometry: EventGeometry,
    ) -> None:

        bins = self.bins.setdefault(multiplicity, MultiplicityBin())
        ratio, axis_ratio = spread_ratio(geometry), axis_spread_ratio(geometry)

        if ratio is not None and axis_ratio is not None:
            bins.input_spread_ratios.append(ratio)
            bins.input_axis_spread_ratios.append(axis_ratio)

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
            **spread_ratio_metric({k: self.bins[k].spread_ratios for k in keys}),
            **spread_ratio_by_dimension(
                {k: self.bins[k].axis_spread_ratios for k in keys}
            ),
            **spread_ratio_metric(
                {k: self.bins[k].input_spread_ratios for k in keys}, INPUT_PREFIX
            ),
            **spread_ratio_by_dimension(
                {k: self.bins[k].input_axis_spread_ratios for k in keys}, INPUT_PREFIX
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


def overlap_ratio(name: str) -> float | None:
    if name == SPREAD_RATIO_NAME:
        return GAUSSIAN_OVERLAP_RATIO

    if SPREAD_RATIO_NAME in name:
        return AXIS_GAUSSIAN_OVERLAP_RATIO

    return None


def spread_ratio_label(name: str) -> str:
    return (
        "3D"
        if name == SPREAD_RATIO_NAME
        else name.removeprefix(SPREAD_RATIO_NAME).strip()
    )


def plot_overlap_line(axis: plt.Axes, name: str) -> list[float]:
    ratio = overlap_ratio(name)

    if ratio is None:
        return []

    axis.axhline(
        ratio,
        color=OVERLAP_LINE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"gaussian overlap ({ratio:.2f})",
    )

    return [ratio]


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
            label="latent space",
            legend=False,
        )
        values += plot_input_line(axis, metrics, name, keys)
        values += plot_overlap_line(axis, name)

        axis.legend(loc="best")
        axis.set_xticks(keys)
        axis.set_xlabel("multiplicity")
        axis.set_title(spread_ratio_label(name))
        sns.despine(ax=axis)
        axis.set_ylim(0.0, y_axis_top(name, values))

    axes.flat[0].set_ylabel(panel_title(SPREAD_RATIO_NAME))
    figure.suptitle(f"{title} | {SPREAD_RATIO_NAME}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
