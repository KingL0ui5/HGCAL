import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

DATA_PATH = "CLUEEEE/closeby_layer_clusters/"


def extract_file(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_frames: list[pd.DataFrame] = []
    gen_frames: list[pd.DataFrame] = []

    with gzip.open(path, "rt") as stream:
        for event_idx, line in enumerate(stream):
            record: dict = json.loads(line)
            clusters: dict[str, list] = record["layer_clusters"]
            gen: dict[str, list] = record["genparticles"]
            event_id: tuple[int, int, int] = (
                int(record["run"]),
                int(record["luminosityBlock"]),
                int(record["event"]),
            )

            cluster_df = pd.DataFrame(clusters)
            cluster_df["event_idx"] = int(event_idx)
            cluster_df["event_id"] = [event_id] * len(cluster_df)

            cluster_frames.append(cluster_df)

            gen_df = pd.DataFrame(gen)
            gen_df["event_idx"] = int(event_idx)
            gen_df["event_id"] = [event_id] * len(gen_df)
            gen_frames.append(gen_df)

    clusters_df = pd.concat(cluster_frames, ignore_index=True)
    genparticles_df = pd.concat(gen_frames, ignore_index=True)

    return clusters_df, genparticles_df


def get_splits_by_file(
    data_dir: str | Path,
    seed: int = 42,
) -> dict[str, list[Path]]:

    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.jsonl.gz"))

    if not files:
        raise FileNotFoundError(f"No extracted data under {data_dir}")

    groups: dict[str, list[Path]] = {}

    for path in files:
        groups.setdefault(path.relative_to(data_dir).parts[0], []).append(path)

    rng = np.random.default_rng(seed)

    splits_by_file: dict[str, list[Path]] = {
        key: [] for key in ("train", "val", "test")
    }

    for sample, paths in sorted(groups.items()):
        if len(paths) < 3:
            raise ValueError(f"{sample}: need at least three files for disjoint splits")

        paths = [paths[i] for i in rng.permutation(len(paths))]
        nv, nt = max(1, round(0.12 * len(paths))), max(1, round(0.08 * len(paths)))
        splits_by_file["val"].extend(paths[:nv])
        splits_by_file["test"].extend(paths[nv : nv + nt])
        splits_by_file["train"].extend(paths[nv + nt :])

    assert not (
        set(splits_by_file["train"]) & set(splits_by_file["val"])
        or set(splits_by_file["train"]) & set(splits_by_file["test"])
        or set(splits_by_file["val"]) & set(splits_by_file["test"])
    )

    return splits_by_file


def split_data_in_directory(
    data_dir: str | Path = DATA_PATH,
    seed: int = 42,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:

    splits_by_file = get_splits_by_file(data_dir, seed=seed)
    splits_by_file_with_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for split_name, file_paths in splits_by_file.items():
        cluster_frames: list[pd.DataFrame] = []
        gen_frames: list[pd.DataFrame] = []

        for path in file_paths:
            clusters_df, genparticles_df = extract_file(path)
            clusters_df["source_file"] = str(path)
            genparticles_df["source_file"] = str(path)

            cluster_frames.append(clusters_df)
            gen_frames.append(genparticles_df)

        splits_by_file_with_data[split_name] = (
            pd.concat(cluster_frames, ignore_index=True),
            pd.concat(gen_frames, ignore_index=True),
        )

    return splits_by_file_with_data


def frame_to_tensors(
    clusters_df: pd.DataFrame,
) -> dict[str, torch.Tensor]:

    xyz = torch.from_numpy(
        clusters_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    )
    energy = torch.from_numpy(clusters_df["E"].to_numpy(dtype=np.float32, copy=True))
    labels = torch.from_numpy(
        clusters_df["gen_idx"].to_numpy(dtype=np.int64, copy=True)
    )

    return {"xyz": xyz, "energy": energy, "labels": labels}


class EventDataset(IterableDataset):
    def __init__(self, file_paths: list[Path], seed: int | None = None) -> None:
        self.file_paths = file_paths
        self.rng = np.random.default_rng(seed)

    def _load_data_from_paths(self) -> Iterator[dict[str, torch.Tensor]]:
        file_paths = list(self.file_paths)
        self.rng.shuffle(file_paths)

        for path in file_paths:
            clusters_df, _ = extract_file(path)
            events = [
                event_df for _, event_df in clusters_df.groupby("event_idx", sort=False)
            ]
            self.rng.shuffle(events)

            for event_df in events:
                yield frame_to_tensors(event_df)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from self._load_data_from_paths()
