import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from evals.pileup_stratified import (
    IDEAL_VALUES,
    PANEL_COLUMNS,
    load_metrics,
    y_axis_top,
)

EVAL_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "spatial_transformer_runs"
)
INFONCE_COLOR = "#2a6fdb"
DISCRIMINATIVE_COLOR = "#e0662b"


def default_metrics_path(loss_name: str) -> Path:
    return EVAL_DATA_DIR / f"spatial_transformer_{loss_name}_metrics.json"


def plot_loss_comparison(
    infonce_path: Path,
    discriminative_path: Path,
    output_path: Path,
) -> None:

    series = {
        "infonce": (load_metrics(infonce_path), INFONCE_COLOR),
        "discriminative": (load_metrics(discriminative_path), DISCRIMINATIVE_COLOR),
    }

    names = [
        name
        for name in series["infonce"][0][0]
        if name in series["discriminative"][0][0]
    ]

    keys = sorted({key for (_, n_events), _ in series.values() for key in n_events})
    n_rows = math.ceil(len(names) / PANEL_COLUMNS)
    sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})

    figure, axes = plt.subplots(
        n_rows,
        PANEL_COLUMNS,
        figsize=(4.2 * PANEL_COLUMNS, 3.4 * n_rows),
        squeeze=False,
    )

    for axis, name in zip(axes.flat, names):
        all_values: list[float] = []

        for label, ((metrics, _), color) in series.items():
            values = [metrics[name].get(key, math.nan) for key in keys]
            all_values.extend(values)
            sns.lineplot(
                x=keys,
                y=values,
                ax=axis,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                label=label,
                legend=False,
            )

        if name in IDEAL_VALUES:
            axis.axhline(IDEAL_VALUES[name], color="grey", linestyle="--", linewidth=1)

        axis.set_xticks(keys)
        axis.set_xlabel("multiplicity")
        axis.set_title(name)
        sns.despine(ax=axis)
        axis.set_ylim(0.0, y_axis_top(name, all_values))

    axes.flat[0].legend(loc="best")

    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)

    figure.suptitle("SpatialTransformer: infonce vs discriminative loss")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay infonce and discriminative SpatialTransformer metrics"
    )

    parser.add_argument("--infonce", type=Path, default=default_metrics_path("infonce"))
    parser.add_argument(
        "--discriminative", type=Path, default=default_metrics_path("discriminative")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DATA_DIR / "spatial_transformer_loss_comparison.png",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_loss_comparison(
        Path(args.infonce), Path(args.discriminative), Path(args.output)
    )
    print(f"comparison plot saved to {args.output}")
