"""AIS-independent quasi-random source-state generation."""

from __future__ import annotations

import math
from datetime import timedelta
from hashlib import sha256

from scipy.stats import qmc  # type: ignore[import-untyped]

from oilspill.domain.geospatial import Coordinate, PointGeometry, SpatialGeometry
from oilspill.domain.hybrid import HybridSourceState
from oilspill.hybrid.config import HybridHindcastConfig
from oilspill.hybrid.reachability import ReachabilityEnvelope


def sample_source_states(
    envelope: ReachabilityEnvelope,
    config: HybridHindcastConfig,
    *,
    seed: int | None = None,
) -> tuple[HybridSourceState, ...]:
    """Transform a five/six-dimensional Sobol design into physical ranges."""
    count = config.initial_sobol_candidates
    dimensions = 6 if config.loss_enabled else 5
    effective_seed = config.sobol_seed if seed is None else seed
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=effective_seed)
    points = sampler.random_base2(m=math.ceil(math.log2(count)))[:count]
    min_x, min_y, max_x, max_y = envelope.sampling_bounds
    interval_seconds = (
        envelope.release_interval.end - envelope.release_interval.start
    ).total_seconds()
    states: list[HybridSourceState] = []
    for index, sample in enumerate(points):
        x = min_x + sample[0] * (max_x - min_x)
        y = min_y + sample[1] * (max_y - min_y)
        release_time = envelope.release_interval.start + timedelta(
            seconds=float(sample[2] * interval_seconds)
        )
        windage = config.windage_range.minimum + sample[3] * (
            config.windage_range.maximum - config.windage_range.minimum
        )
        diffusivity = config.diffusivity_range_m2_s.minimum + sample[4] * (
            config.diffusivity_range_m2_s.maximum
            - config.diffusivity_range_m2_s.minimum
        )
        loss_rate = None
        if config.loss_rate_range_s is not None:
            loss_rate = config.loss_rate_range_s.minimum + sample[5] * (
                config.loss_rate_range_s.maximum - config.loss_rate_range_s.minimum
            )
        identity = sha256(
            (
                f"{effective_seed}:{index}:{x:.17g}:{y:.17g}:"
                f"{release_time.isoformat()}:{windage:.17g}:{diffusivity:.17g}:"
                f"{loss_rate!r}"
            ).encode()
        ).hexdigest()
        states.append(
            HybridSourceState(
                source_state_id=f"sobol-source:{identity[:24]}",
                geometry=SpatialGeometry(
                    geometry=PointGeometry(coordinate=Coordinate(x=x, y=y)),
                    crs=envelope.grid_crs,
                ),
                release_time=release_time,
                windage_coefficient=float(windage),
                horizontal_diffusivity_m2_s=float(diffusivity),
                first_order_loss_rate_s=(
                    float(loss_rate) if loss_rate is not None else None
                ),
                assumptions=(
                    "AIS-independent Sobol source-state design.",
                    config.validation_note,
                ),
            )
        )
    return tuple(states)
