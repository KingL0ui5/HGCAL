import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def split_files(data_dir, seed=42):
    """Stratify by sample directory, hold out full files (20/3/2 per 25)."""
    files = sorted(Path(data_dir).rglob("*.jsonl.gz"))

    if not files:
        raise FileNotFoundError(f"No extracted data under {data_dir}")

    groups = {}

    for path in files:
        groups.setdefault(path.relative_to(data_dir).parts[0], []).append(path)

    rng = np.random.default_rng(seed)
    splits = {key: [] for key in ("train", "val", "test")}

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


def cache_file(source, data_dir, cache_dir):
    """Compact per-file ragged arrays; only one input file in RAM at a time."""
    source = Path(source)
    target = Path(cache_dir) / source.relative_to(data_dir).with_suffix("").with_suffix(
        ".npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = np.array([source.stat().st_size, source.stat().st_mtime_ns], dtype=np.int64)

    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if np.array_equal(saved["source_stamp"], stamp):
                return target

    xyz, energies, labels, offsets, event_ids = [], [], [], [0], []

    with gzip.open(source, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            clusters = record["layer_clusters"]
            position = np.column_stack([clusters[a] for a in "xyz"]).astype(np.float32)
            energy = np.asarray(clusters["E"], dtype=np.float32)
            group = np.asarray(clusters["gen_idx"], dtype=np.int64)

            if not (len(position) == len(energy) == len(group)):
                raise ValueError(f"Invalid array lengths in {source}")

            if (
                not np.isfinite(position).all()
                or not np.isfinite(energy).all()
                or (energy < 0).any()
            ):
                raise ValueError(f"Invalid coordinates/energy in {source}")
            ngen = len(record["genparticles"]["energy"])

            if ((group < -1) | (group >= ngen)).any():
                raise ValueError(f"Invalid generator index in {source}")

            xyz.append(position)
            energies.append(energy)
            labels.append(group)
            offsets.append(offsets[-1] + len(energy))
            event_ids.append(
                [record["run"], record["luminosityBlock"], record["event"]]
            )

    if not xyz:
        raise ValueError(f"Empty extracted file: {source}")

    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        xyz=np.concatenate(xyz),
        E=np.concatenate(energies),
        labels=np.concatenate(labels),
        offsets=np.asarray(offsets, dtype=np.int64),
        event_ids=np.asarray(event_ids, dtype=np.uint64),
        source_stamp=stamp,
    )
    temporary.replace(target)

    return target


def prepare_cache(splits, data_dir, cache_dir):
    cached = {key: [] for key in splits}
    total = sum(map(len, splits.values()))
    done = 0

    for split, sources in splits.items():
        for source in sources:
            cached[split].append(cache_file(source, data_dir, cache_dir))
            done += 1

            if done % 10 == 0 or done == total:
                print(f"Cache: {done}/{total} files", flush=True)

    return cached


def make_event(xyz, energy, labels, indices, key, cfg):
    """Features and graph coordinates use no labels or generator properties."""
    xyz = torch.as_tensor(np.array(xyz, copy=True), dtype=torch.float32)
    energy = torch.as_tensor(np.array(energy, copy=True), dtype=torch.float32)
    weights = energy.clamp_min(0)
    center = (xyz * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8)

    if weights.sum() == 0:
        center = xyz.mean(0)

    relative = xyz - center
    features = torch.cat(
        (relative / torch.tensor([20.0, 20.0, 50.0]), torch.log1p(energy)[:, None] / 2),
        dim=1,
    )
    graph_xyz = relative / torch.tensor([10.0, 10.0, 30.0])
    clue_weights = (energy.clamp_min(1e-6) / energy.mean().clamp_min(1e-6)).pow(
        cfg.energy_weight_power
    )
    clue_weights = clue_weights.clamp(0.1, 10.0)
    clue_weights = clue_weights / clue_weights.mean()

    return dict(
        features=features,
        graph_xyz=graph_xyz,
        xyz=xyz,
        energy=energy,
        labels=torch.as_tensor(np.array(labels, copy=True), dtype=torch.long),
        weights=clue_weights,
        indices=torch.as_tensor(np.array(indices, copy=True), dtype=torch.long),
        key=key,
    )


class EventDataset(IterableDataset):
    """Uniform label-blind sampling; reproducible fixed subsets for evaluation."""

    def __init__(self, files, cfg, training=False, max_events=None):
        self.files = list(files)
        self.cfg = cfg
        self.training = training
        self.max_events = max_events
        self.epoch = 0

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()

        if worker is not None and self.max_events is not None:
            raise ValueError(
                "Use num_workers=0 with max_events to enforce a global event cap"
            )

        rng = np.random.default_rng(
            self.cfg.seed + (self.epoch if self.training else 0)
        )
        files = self.files.copy()

        if self.training:
            rng.shuffle(files)

        if worker:
            files = files[worker.id :: worker.num_workers]

        emitted = 0

        for path in files:
            with np.load(path, allow_pickle=False) as saved:
                # Load each compressed array once, not once per event.
                data = {
                    k: saved[k] for k in ("xyz", "E", "labels", "offsets", "event_ids")
                }
                order = np.arange(len(data["offsets"]) - 1)

                if self.training:
                    rng.shuffle(order)

                for i in order:
                    start, stop = data["offsets"][i : i + 2]
                    size = int(stop - start)

                    if size < 2:
                        continue

                    # Stable event key includes sample/chunk: event numbers repeat across files.
                    key = f"{path.parent.parent.name}/{path.parent.name}/{path.name}:{int(i)}"
                    digest = int.from_bytes(
                        hashlib.sha256(key.encode()).digest()[:8], "little"
                    )
                    sample_rng = np.random.default_rng(
                        (digest + self.cfg.seed + (self.epoch if self.training else 0))
                        % (2**64)
                    )
                    indices = np.arange(size)

                    if self.cfg.max_nodes is not None and size > self.cfg.max_nodes:
                        indices = np.sort(
                            sample_rng.choice(size, self.cfg.max_nodes, replace=False)
                        )

                    selected = start + indices
                    event = make_event(
                        data["xyz"][selected],
                        data["E"][selected],
                        data["labels"][selected],
                        indices,
                        key,
                        self.cfg,
                    )
                    event["event_id"] = tuple(int(v) for v in data["event_ids"][i])
                    event["original_nodes"] = size

                    yield event
                    emitted += 1

                    if self.max_events is not None and emitted >= self.max_events:
                        return
