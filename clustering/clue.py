"""CLUEstering wrapper with automatic GPU/CPU backend selection."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

try:
    import CLUEstering as clue

    _CLUE_AVAILABLE = True
except ImportError:
    _CLUE_AVAILABLE = False
    clue = cast("Any", None)


@dataclass(frozen=True)
class ClueResult:
    cluster_ids: np.ndarray
    n_clusters: int
    n_outliers: int
    elapsed_ms: float
    backend: str


def available_backends() -> list[str]:
    if not _CLUE_AVAILABLE:
        return []

    return list(clue.backends)


_BACKEND_PRIORITY = ["gpu cuda", "gpu hip", "cpu openmp", "cpu serial"]


def probe_backend(requested: str = "auto", device_id: int = 0) -> str:
    """Resolve and verify a backend once by running a tiny dummy job.

    Raises RuntimeError if no working backend is found.
    """

    if not _CLUE_AVAILABLE:
        raise RuntimeError("CLUEstering is not installed.")

    chain = (
        [b for b in _BACKEND_PRIORITY if b in available_backends()]
        if requested == "auto"
        else [requested]
    )

    if not chain:
        raise RuntimeError("No compiled backends found.")

    if requested != "auto" and requested not in available_backends():
        raise ValueError(
            f"Backend '{requested}' not available. Available: {available_backends()}"
        )

    # 5-point dummy dataset, initialises backend once
    dummy = [
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.ones(5, dtype=np.float32),
    ]

    for backend in chain:
        try:
            c = clue.clusterer(1.0, 1.0, None, None, 10)
            c.choose_kernel("flat", [0.5])
            c.read_data(dummy)
            c.run_clue(backend=backend, block_size=1024, device_id=device_id)

            return backend
        except RuntimeError:
            if requested != "auto":
                raise

            print(f"  [{backend}] not available at runtime, trying next backend...")

    raise RuntimeError(f"No working backend found among: {chain}")


def run_clue(
    coords: np.ndarray,
    weights: np.ndarray,
    dc: float,
    rhoc: float,
    do: float | None = None,
    dm: float | None = None,
    ppbin: int = 128,
    metric: str = "euclidean",
    metric_params: list[float] | None = None,
    backend: str = "auto",
    block_size: int = 1024,
    device_id: int = 0,
) -> ClueResult:
    """Run CLUE clustering on one event.

    Args:
        coords:        (N, n_dim) float32
        weights:       (N,) float32
        dc:            critical distance for local density calculation
        rhoc:          density threshold separating seeds from outliers
        do:            nearest-higher search radius; defaults to dc
        dm:            follower connection radius; defaults to dc
        ppbin:         average points per spatial tile
        metric:        'euclidean', 'manhattan', 'chebyshev',
                       'weighted_euclidean', 'weighted_chebyshev', 'periodic_euclidean'
        metric_params: per-dimension weights/periods for parameterised metrics
        backend:       resolved backend string from probe_backend(), or 'auto'
                       to probe on every call (not recommended for loops)
        block_size:    CUDA thread block size
        device_id:     GPU device index
    """

    if not _CLUE_AVAILABLE:
        raise RuntimeError(
            "CLUEstering is not installed. Run: pip install -e CLUEstering/"
        )

    selected = probe_backend(backend, device_id) if backend == "auto" else backend

    n_dim = coords.shape[1]
    data = [np.ascontiguousarray(coords[:, i], dtype=np.float32) for i in range(n_dim)]
    data.append(np.ascontiguousarray(weights, dtype=np.float32))

    c = clue.clusterer(dc, rhoc, do, dm, ppbin)
    c.choose_kernel("flat", [0.5])

    if metric != "euclidean":
        c.choose_metric(metric, metric_params)

    c.read_data(data)
    c.run_clue(backend=selected, block_size=block_size, device_id=device_id)

    return ClueResult(
        cluster_ids=c.cluster_ids.copy(),
        n_clusters=c.n_clusters,
        n_outliers=int(np.sum(c.cluster_ids == -1)),
        elapsed_ms=c._elapsed_time,
        backend=selected,
    )
