"""Small deterministic clustering and representative-selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RepresentativeSelection:
    labels: IntArray
    medoid_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    reasons: tuple[tuple[str, ...], ...]


def deterministic_k_medoids(
    features: FloatArray, cluster_count: int
) -> tuple[IntArray, tuple[int, ...]]:
    """Cluster standardized features with deterministic farthest-first PAM updates."""
    if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("clustering features must be a non-empty finite matrix")
    count = min(cluster_count, len(features))
    scale = np.std(features, axis=0)
    normalized = (features - np.mean(features, axis=0)) / np.where(scale > 0, scale, 1.0)
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    global_distance = np.linalg.norm(normalized - np.mean(normalized, axis=0), axis=1)
    medoids = [int(np.argmin(global_distance))]
    while len(medoids) < count:
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[medoids] = -1
        medoids.append(int(np.argmax(nearest)))
    for _ in range(100):
        labels = np.argmin(distances[:, medoids], axis=1).astype(np.int64)
        for cluster, medoid in enumerate(medoids):
            labels[medoid] = cluster
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            costs = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(costs))]))
        if updated == medoids:
            return labels, tuple(medoids)
        medoids = updated
    raise RuntimeError("deterministic k-medoids did not converge")


def select_representatives(
    features: FloatArray,
    displacement: FloatArray,
    spread: FloatArray,
    coastal_contact: FloatArray,
    *,
    cluster_count: int,
    displacement_extremes: int,
    spread_extremes: int,
    coastal_extremes: int,
) -> RepresentativeSelection:
    labels, medoids = deterministic_k_medoids(features, cluster_count)
    reasons: list[list[str]] = [[] for _ in range(len(features))]
    for index in medoids:
        reasons[index].append("cluster_medoid")
    _add_extremes(reasons, displacement, displacement_extremes, "largest_displacement", None)
    _add_extremes(reasons, spread, spread_extremes, "largest_spread", None)
    _add_extremes(
        reasons,
        coastal_contact,
        coastal_extremes,
        "coastal_contact",
        coastal_contact > 0,
    )
    selected = tuple(index for index, value in enumerate(reasons) if value)
    return RepresentativeSelection(
        labels=labels,
        medoid_indices=medoids,
        selected_indices=selected,
        reasons=tuple(tuple(value) for value in reasons),
    )


def _add_extremes(
    reasons: list[list[str]],
    values: FloatArray,
    count: int,
    reason: str,
    eligible: NDArray[np.bool_] | None,
) -> None:
    candidates = [
        index for index in range(len(values)) if eligible is None or bool(eligible[index])
    ]
    candidates.sort(key=lambda index: (-float(values[index]), index))
    for index in candidates[:count]:
        reasons[index].append(reason)
