import gzip
import json
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


def stratify_file_paths(data_dir: str | Path, seed: int = 42) -> dict[str, list[Path]]:

    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.jsonl.gz"))

    if not files:
        raise FileNotFoundError(f"No extracted data under {data_dir}")

    groups: dict[str, list[Path]] = {}

    for path in files:
        groups.setdefault(path.relative_to(data_dir).parts[0], []).append(path)

    rng = np.random.default_rng(seed)
    splits: dict[str, list[Path]] = {key: [] for key in ("train", "val", "test")}

    for sample, paths in sorted(groups.items()):
        if len(paths) < 3:
            raise ValueError(f"{sample}: need at least three files for disjoint splits")

        paths = [paths[i] for i in rng.permutation(len(paths))]
        nv, nt = max(1, round(0.12 * len(paths))), max(1, round(0.08 * len(paths)))
        splits["val"].extend(paths[:nv])
        splits["test"].extend(paths[nv : nv + nt])
        splits["train"].extend(paths[nv + nt :])

    assert not (
        set(splits["train"]) & set(splits["val"])
        or set(splits["train"]) & set(splits["test"])
        or set(splits["val"]) & set(splits["test"])
    )

    return splits


def load_stratified_dataframes(
    data_dir: str | Path = DATA_PATH, seed: int = 42
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:

    splits = stratify_file_paths(data_dir, seed=seed)
    all_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for split_name, paths in splits.items():
        cluster_frames: list[pd.DataFrame] = []
        gen_frames: list[pd.DataFrame] = []

        for path in paths:
            clusters_df, genparticles_df = extract_file(path)
            clusters_df["source_file"] = str(path)
            genparticles_df["source_file"] = str(path)

            cluster_frames.append(clusters_df)
            gen_frames.append(genparticles_df)

        all_splits[split_name] = (
            pd.concat(cluster_frames, ignore_index=True),
            pd.concat(gen_frames, ignore_index=True),
        )

    return all_splits


def frame_to_tensors(
    clusters_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    xyz = torch.from_numpy(clusters_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True))
    energy = torch.from_numpy(clusters_df["E"].to_numpy(dtype=np.float32, copy=True))
    labels = torch.from_numpy(clusters_df["gen_idx"].to_numpy(dtype=np.int64, copy=True))

    return xyz, energy, labels


class EventDataset(IterableDataset):

    def __init__(self, file_paths: list[Path]) -> None:
        self.file_paths = file_paths
        self.clusters_df, self.genparticles_df = self._load_data_from_paths()

    def _load_data_from_paths(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cluster_frames: list[pd.DataFrame] = []
        gen_frames: list[pd.DataFrame] = []

        for path in self.file_paths:
            clusters_df, genparticles_df = extract_file(path)
            clusters_df["source_file"] = str(path)
            genparticles_df["source_file"] = str(path)

            cluster_frames.append(clusters_df)
            gen_frames.append(genparticles_df)

        return (
            pd.concat(cluster_frames, ignore_index=True),
            pd.concat(gen_frames, ignore_index=True),
        )

