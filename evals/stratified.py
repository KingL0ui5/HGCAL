import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from evals.reco_metrics import ReconstructionCounts

LINE_COLOR = "#2a6fdb"
IDEAL_VALUES = {"multiplicity ratio": 1.0}
UNIT_INTERVAL_METRICS = {"efficiency", "purity", "pair ROC AUC", "slot accuracy"}
PANEL_COLUMNS = 3


def event_multiplicity(labels: Tensor) -> int:
    return int(torch.unique(labels).numel())


@dataclass
class MultiplicityBin:
    counts: ReconstructionCounts = field(default_factory=ReconstructionCounts)
    pair_scores: list[Tensor] = field(default_factory=list)
    pair_labels: list[Tensor] = field(default_factory=list)
    n_events: int = 0

    @property
    def auc(self) -> float:
        if not self.pair_scores:
            return math.nan

        scores = torch.cat(self.pair_scores).numpy().astype(np.float32)
        labels = torch.cat(self.pair_labels).numpy().astype(np.int8)

        if len(np.unique(labels)) < 2:
            return math.nan

        return float(roc_auc_score(labels, scores))


class StratifiedReconstruction:
    def __init__(self) -> None:
        self.bins: dict[int, MultiplicityBin] = {}

    def add(
        self,
        multiplicity: int,
        counts: ReconstructionCounts,
        pair_scores: Tensor | None = None,
        pair_labels: Tensor | None = None,
    ) -> None:

        stratum = self.bins.setdefault(multiplicity, MultiplicityBin())
        stratum.counts.add(counts)
        stratum.n_events += 1

        if pair_scores is not None and pair_labels is not None:
            stratum.pair_scores.append(pair_scores)
            stratum.pair_labels.append(pair_labels)

    @property
    def n_events(self) -> dict[int, int]:
        return {key: self.bins[key].n_events for key in sorted(self.bins)}

    def metrics(self) -> dict[str, dict[int, float]]:
        keys = sorted(self.bins)
        metrics: dict[str, dict[int, float]] = {
            "efficiency": {k: self.bins[k].counts.efficiency for k in keys},
            "purity": {k: self.bins[k].counts.purity for k in keys},
            "multiplicity ratio": {
                k: self.bins[k].counts.multiplicity_ratio for k in keys
            },
        }

        if any(self.bins[k].pair_scores for k in keys):
            metrics["pair ROC AUC"] = {k: self.bins[k].auc for k in keys}

        return metrics


def print_stratified_table(
    title: str,
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
) -> None:

    keys = sorted(n_events)
    print(f"\n== {title}: by particle multiplicity ==")
    print(f"{'metric':<22}" + "".join(f"{f'n={k}':>10}" for k in keys))
    print(f"{'events':<22}" + "".join(f"{n_events[k]:>10}" for k in keys))

    for name, values in metrics.items():
        print(
            f"{name:<22}" + "".join(f"{values.get(k, math.nan):>10.4f}" for k in keys)
        )


def plot_by_multiplicity(
    metrics: dict[str, dict[int, float]],
    n_events: dict[int, int],
    title: str,
    output_path: Path,
) -> None:

    names = list(metrics)
    n_rows = math.ceil(len(names) / PANEL_COLUMNS)
    figure, axes = plt.subplots(
        n_rows,
        PANEL_COLUMNS,
        figsize=(4.2 * PANEL_COLUMNS, 3.4 * n_rows),
        squeeze=False,
    )
    keys = sorted(n_events)

    for axis, name in zip(axes.flat, names):
        values = [metrics[name].get(k, math.nan) for k in keys]
        axis.plot(keys, values, color=LINE_COLOR, linewidth=2, marker="o", markersize=8)

        if name in IDEAL_VALUES:
            axis.axhline(IDEAL_VALUES[name], color="grey", linestyle="--", linewidth=1)

        axis.set_xticks(keys)
        axis.set_xlabel("particles per event")
        axis.set_title(name)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

        peak = max(float(np.nanmax(values)), IDEAL_VALUES.get(name, 0.0))
        top = 1.05 if name in UNIT_INTERVAL_METRICS else peak * 1.15
        axis.set_ylim(0.0, top)

    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"{title} | events per bin: " + ", ".join(f"{k}: {n_events[k]}" for k in keys)
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
