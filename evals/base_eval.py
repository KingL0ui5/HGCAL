import argparse
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Generic

import torch
from torch import no_grad
from torch.utils.data import DataLoader

from data import DATA_PATH, get_splits_by_file
from evals.eval_utils import (
    EventOutputs,
    ModelT,
    Timings,
    checkpoint_loss_name,
    load_run_config,
    load_weights,
)
from evals.geometry_metrics import geometry_metrics
from evals.multiplicity_stratified import (
    MultiplicityStratifiedReco,
    event_multiplicity,
    plot_by_multiplicity,
    save_metrics,
)
from evals.reco_metrics import (
    ClusteringParameters,
    ReconstructionCounts,
    cluster_labels,
    event_counts,
    print_reconstruction_report,
)
from train.train_utils import TrainConfig, select_device

EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"


class BaseEvaluator(ABC, Generic[ModelT]):
    display_name: ClassVar[str]
    output_prefix: ClassVar[str]
    checkpoint_pattern: ClassVar[re.Pattern[str]]
    default_checkpoint: ClassVar[Path | None] = None
    batch_size: ClassVar[int] = 8

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checkpoint = Path(args.checkpoint)
        self.device = torch.device(str(args.device))
        self.split = str(args.split)
        self.num_workers = int(args.num_workers)
        self.max_events = int(args.max_events) if args.max_events is not None else None
        self.clustering_parameters = ClusteringParameters(
            distance_threshold=float(args.distance_threshold),
            method=str(args.linkage_method),
            metric=str(args.metric),
            min_cluster_size=int(args.min_cluster_size),
        )
        self.run_config = load_run_config(self.checkpoint)
        self.seed = int(
            self.run_config.get("train", {}).get("seed", TrainConfig().seed)
        )
        self.loss_name = checkpoint_loss_name(self.checkpoint, self.checkpoint_pattern)
        self.stratified_path = (
            Path(args.stratified_output)
            if args.stratified_output
            else self.output_path("by_multiplicity")
        )
        self.metrics_path = (
            Path(args.metrics_output)
            if args.metrics_output
            else self.output_path("metrics", "json")
        )
        self.timings = Timings()

    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        defaults = ClusteringParameters()
        parser = argparse.ArgumentParser(description=f"Evaluate the {cls.display_name}")
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=cls.default_checkpoint is None,
            default=cls.default_checkpoint,
        )
        parser.add_argument("--device", default=str(select_device()))
        parser.add_argument("--split", choices=("val", "test"), default="test")
        parser.add_argument("--stratified-output", type=Path, default=None)
        parser.add_argument("--metrics-output", type=Path, default=None)
        parser.add_argument("--num-workers", type=int, default=2)
        parser.add_argument("--max-events", type=int, default=None)
        parser.add_argument(
            "--distance-threshold", type=float, default=defaults.distance_threshold
        )
        parser.add_argument("--linkage-method", default=defaults.method)
        parser.add_argument("--metric", default=defaults.metric)
        parser.add_argument(
            "--min-cluster-size", type=int, default=defaults.min_cluster_size
        )
        cls.add_arguments(parser)

        return parser

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def main(cls) -> None:
        cls(cls.build_arg_parser().parse_args()).evaluate()

    @abstractmethod
    def build_model(self) -> ModelT: ...

    @abstractmethod
    def build_loader(self, file_paths: list[Path]) -> DataLoader: ...

    @abstractmethod
    def embed_batch(self, model: ModelT, cpu_batch: Any) -> list[EventOutputs]: ...

    @cached_property
    def model(self) -> ModelT:
        model = load_weights(
            self.build_model().to(self.device), self.checkpoint, self.device
        )
        model.eval()

        return model

    @cached_property
    def split_files(self) -> list[Path]:
        return get_splits_by_file(DATA_PATH, seed=self.seed)[self.split]

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda"

    def output_path(self, kind: str, suffix: str = "png") -> Path:
        return (
            EVAL_DATA_DIR
            / f"{self.output_prefix}_runs"
            / f"{self.output_prefix}_{self.loss_name}_{kind}.{suffix}"
        )

    def iter_events(self, loader: DataLoader) -> Iterator[EventOutputs]:
        n_events = 0

        with no_grad():
            for cpu_batch in loader:
                start = time.perf_counter()
                events = self.embed_batch(self.model, cpu_batch)
                self.timings.inference += time.perf_counter() - start

                for event in events:
                    if self.max_events is not None and n_events >= self.max_events:
                        return

                    yield event
                    n_events += 1

    def evaluate(self) -> None:
        print(f"{self.display_name} | loss: {self.loss_name} | device: {self.device}")
        counts = ReconstructionCounts()
        stratified = MultiplicityStratifiedReco()
        latent_dimensions = 0

        for event in self.iter_events(self.build_loader(self.split_files)):
            start = time.perf_counter()
            reco_labels = cluster_labels(event.embeddings, self.clustering_parameters)
            self.timings.clustering += time.perf_counter() - start

            reconstruction = event_counts(reco_labels, event.labels, event.energy)
            intra_distances, separation = geometry_metrics(
                event.embeddings, event.labels
            )
            input_intra_distances, input_separation = geometry_metrics(
                event.positions, event.labels
            )

            counts.add(reconstruction)
            stratified.add(
                event_multiplicity(event.labels),
                reconstruction,
                intra_distances,
                separation,
            )
            stratified.add_input_geometry(
                event_multiplicity(event.labels),
                input_intra_distances,
                input_separation,
            )

            latent_dimensions = int(event.embeddings.shape[1])

        self.report(counts, stratified, latent_dimensions)

    def report(
        self,
        counts: ReconstructionCounts,
        stratified: MultiplicityStratifiedReco,
        latent_dimensions: int,
    ) -> None:

        n_events = sum(stratified.n_events.values())
        print(f"events: {n_events} | latent dimensions: {latent_dimensions}")
        print(
            f"inference time: {self.timings.inference:.2f}s "
            f"({1e3 * self.timings.inference / max(n_events, 1):.2f} ms/event) | "
            f"agglomerative clustering time: {self.timings.clustering:.2f}s "
            f"({1e3 * self.timings.clustering / max(n_events, 1):.2f} ms/event)"
        )
        print_reconstruction_report(self.split, counts)

        stratified_metrics = stratified.metrics()
        plot_by_multiplicity(
            stratified_metrics,
            stratified.n_events,
            f"{self.display_name} {self.loss_name}",
            self.stratified_path,
        )
        print(f"stratified plot saved to {self.stratified_path}")

        save_metrics(stratified_metrics, stratified.n_events, self.metrics_path)
        print(f"metrics saved to {self.metrics_path}")
