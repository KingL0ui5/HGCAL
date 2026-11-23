import json
import math
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
IDEAL_VALUES = {"multiplicity ratio": 1.0}
UNIT_INTERVAL_METRICS = {"efficiency", "purity"}
PANEL_COLUMNS = 3


def event_multiplicity(labels: Tensor) -> int:
    return int(torch.unique(labels).numel())


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


class PileupBin(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    counts: ReconstructionCounts = Field(default_factory=ReconstructionCounts)
    intra_distances: list[Tensor] = Field(default_factory=list)
    separations: list[float] = Field(default_factory=list)
    n_events: int = 0


class PileupStratifiedReco:
    def __init__(self) -> None:
        self.bins: dict[int, PileupBin] = {}

    def add(
        self,
        multiplicity: int,
        counts: ReconstructionCounts,
        intra_distances: Tensor,
        separation: float | None,
    ) -> None:

        bins = self.bins.setdefault(multiplicity, PileupBin())
        bins.counts.add(counts)
        bins.intra_distances.append(intra_distances.abs().mean(dim=0))
        bins.n_events += 1

        if separation is not None:
            bins.separations.append(separation)

    @property
    def n_events(self) -> dict[int, int]:
        return {key: self.bins[key].n_events for key in sorted(self.bins)}

    @property
    def mean_centroid_separation(self) -> float:
        return mean_or_nan([s for b in self.bins.values() for s in b.separations])

    @property
    def mean_intra_distances(self) -> list[float]:
        per_event = [d for b in self.bins.values() for d in b.intra_distances]

        return torch.stack(per_event).mean(dim=0).tolist() if per_event else []

    def metrics(self) -> dict[str, dict[int, float]]:
        keys = sorted(self.bins)

        mean_intra_distances_by_dimension = {
            f"cluster spread dim {dim}": {
                k: mean_or_nan([float(d[dim]) for d in self.bins[k].intra_distances])
                for k in keys
            }
            for dim in range(len(self.mean_intra_distances))
        }

        return {
            "efficiency": {k: self.bins[k].counts.efficiency for k in keys},
            "purity": {k: self.bins[k].counts.purity for k in keys},
            "multiplicity ratio": {
                k: self.bins[k].counts.multiplicity_ratio for k in keys
            },
            "centroid separation": {
                k: mean_or_nan(self.bins[k].separations) for k in keys
            },
            **mean_intra_distances_by_dimension,
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


def plot_by_multiplicity(
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
    title: str,
    output_path: Path,
) -> None:

    names = list(metrics)
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
        axis.set_xlabel("pileup")
        axis.set_title(name)
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
