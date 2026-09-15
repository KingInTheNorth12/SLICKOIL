"""Deterministic support-weighted local refinement for hybrid source states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import rowcol

from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.geospatial import Coordinate, PointGeometry, SpatialGeometry
from oilspill.domain.hybrid import HybridSourceState
from oilspill.eulerian.initialization import affine, read_land_mask
from oilspill.hybrid.config import RefinementPerturbationScales, SelectiveRefinementConfig
from oilspill.hybrid.high_fidelity import RawHighFidelityEvaluation
from oilspill.requests import HybridHindcastRequest


@dataclass(frozen=True)
class ChildGeneration:
    states: tuple[HybridSourceState, ...]
    scales: RefinementPerturbationScales


FloatArray = NDArray[np.float64]


def normalized_support_weights(scores: FloatArray, temperature: float) -> FloatArray:
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("support scores must be a finite non-empty vector")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("support temperature must be finite and positive")
    shifted = scores - float(scores.min())
    support = np.exp(-shifted / temperature)
    return np.asarray(support / support.sum(), dtype=np.float64)


def scales_for_generation(
    base: RefinementPerturbationScales, rho: float, generation: int
) -> RefinementPerturbationScales:
    if generation < 1:
        raise ValueError("child generation numbering starts at one")
    factor = rho ** (generation - 1)
    return RefinementPerturbationScales(
        x_metres=base.x_metres * factor,
        y_metres=base.y_metres * factor,
        release_time_seconds=base.release_time_seconds * factor,
        windage_coefficient=base.windage_coefficient * factor,
        diffusivity_m2_s=base.diffusivity_m2_s * factor,
    )


def generate_children(
    evaluations: tuple[RawHighFidelityEvaluation, ...],
    generation: int,
    count: int,
    request: HybridHindcastRequest,
    grid: EulerianGrid,
    config: SelectiveRefinementConfig,
    *,
    random_seed: int,
) -> ChildGeneration:
    if not evaluations:
        raise ValueError("refinement requires evaluated parent states")
    scales = scales_for_generation(
        config.perturbation_scales, config.perturbation_shrink_rho, generation
    )
    ranked = sorted(
        evaluations,
        key=lambda item: (item.mismatch_score, item.source_state.source_state_id),
    )
    parents = ranked[: config.parent_pool_size]
    temperature = config.initial_temperature * config.temperature_decay ** (generation - 1)
    probabilities = normalized_support_weights(
        np.asarray([item.mismatch_score for item in parents]), temperature
    )
    rng = np.random.default_rng(random_seed)
    left = grid.transform.c + grid.dx_metres / 2
    right = grid.transform.c + grid.transform.a * grid.width - grid.dx_metres / 2
    top = grid.transform.f - grid.dy_metres / 2
    bottom = grid.transform.f + grid.transform.e * grid.height + grid.dy_metres / 2
    earliest = request.observation.discharge_time.interval.start
    latest = min(
        request.observation.discharge_time.interval.end,
        request.observation.observed_at
        - timedelta(seconds=config.minimum_forward_duration_seconds),
    )
    if latest < earliest:
        raise ValueError("configured minimum forward duration excludes release-time interval")
    interval_seconds = (latest - earliest).total_seconds()
    land = read_land_mask(grid)
    children: list[HybridSourceState] = []
    attempts = 0
    maximum_attempts = count * config.maximum_child_sampling_attempts
    while len(children) < count and attempts < maximum_attempts:
        attempts += 1
        parent_index = int(rng.choice(len(parents), p=probabilities))
        parent = parents[parent_index].source_state
        point = parent.geometry.geometry
        assert point.type == "Point"
        exploratory = bool(rng.random() < config.exploratory_fraction)
        if exploratory:
            x = float(rng.uniform(left, right))
            y = float(rng.uniform(bottom, top))
            seconds = float(rng.uniform(0.0, interval_seconds)) if interval_seconds else 0.0
            release_time = earliest + timedelta(seconds=seconds)
            windage = float(
                rng.uniform(config.windage_range.minimum, config.windage_range.maximum)
            )
            diffusivity = float(
                rng.uniform(
                    config.diffusivity_range_m2_s.minimum,
                    config.diffusivity_range_m2_s.maximum,
                )
            )
        else:
            x = float(
                np.clip(
                    point.coordinate.x + rng.normal(0.0, scales.x_metres), left, right
                )
            )
            y = float(
                np.clip(
                    point.coordinate.y + rng.normal(0.0, scales.y_metres), bottom, top
                )
            )
            release_time = parent.release_time + timedelta(
                seconds=float(rng.normal(0.0, scales.release_time_seconds))
            )
            release_time = min(max(release_time, earliest), latest)
            windage = float(
                np.clip(
                    parent.windage_coefficient
                    + rng.normal(0.0, scales.windage_coefficient),
                    config.windage_range.minimum,
                    config.windage_range.maximum,
                )
            )
            diffusivity = float(
                np.clip(
                    parent.horizontal_diffusivity_m2_s
                    + rng.normal(0.0, scales.diffusivity_m2_s),
                    config.diffusivity_range_m2_s.minimum,
                    config.diffusivity_range_m2_s.maximum,
                )
            )
        row, column = rowcol(affine(grid), x, y)
        if land[row, column]:
            continue
        identity = sha256(
            (
                f"{random_seed}:{generation}:{len(children)}:{parent.source_state_id}:"
                f"{x:.17g}:{y:.17g}:{release_time.isoformat()}:"
                f"{windage:.17g}:{diffusivity:.17g}"
            ).encode()
        ).hexdigest()
        children.append(
            HybridSourceState(
                source_state_id=f"refined-source:{identity[:24]}",
                geometry=SpatialGeometry(
                    geometry=PointGeometry(coordinate=Coordinate(x=x, y=y)),
                    crs=grid.crs,
                ),
                release_time=release_time,
                windage_coefficient=windage,
                horizontal_diffusivity_m2_s=diffusivity,
                first_order_loss_rate_s=parent.first_order_loss_rate_s,
                forcing_member_id=parent.forcing_member_id,
                assumptions=(
                    f"Generation {generation} support-weighted source perturbation.",
                    config.validation_note,
                ),
            )
        )
    return ChildGeneration(states=tuple(children), scales=scales)
