"""Conservative, deliberately permissive pre-transport reachability checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from rasterio.transform import rowcol
from rasterio.warp import transform_geom
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

from oilspill.domain.common import TimeRange, UTCDateTime
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.geospatial import CRS
from oilspill.domain.hybrid import HybridSourceState
from oilspill.domain.spill import SpillObservation
from oilspill.eulerian.grid import _mapping
from oilspill.eulerian.initialization import affine, crs_text, read_land_mask
from oilspill.hybrid.config import HybridHindcastConfig


@dataclass(frozen=True)
class ReachabilityEnvelope:
    grid_crs: CRS
    observation_time: UTCDateTime
    release_interval: TimeRange
    source_region: BaseGeometry
    observation_region: BaseGeometry
    maximum_advective_speed_m_s: float
    diffusion_allowance_metres: float
    source_padding_metres: float

    @property
    def sampling_bounds(self) -> tuple[float, float, float, float]:
        min_x, min_y, max_x, max_y = self.source_region.bounds
        return float(min_x), float(min_y), float(max_x), float(max_y)


@dataclass(frozen=True)
class ReachabilityAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    physics_penalty: float
    required_speed_m_s: float


def build_reachability_envelope(
    observation: SpillObservation,
    grid: EulerianGrid,
    config: HybridHindcastConfig,
) -> ReachabilityEnvelope:
    horizon_start = observation.observed_at - timedelta(
        seconds=config.backward_horizon_seconds
    )
    start, end = horizon_start, observation.observed_at
    if config.source_time_range_policy == "discharge_interval_intersection":
        start = max(start, observation.discharge_time.interval.start)
        end = min(end, observation.discharge_time.interval.end)
        if end < start:
            raise ValueError("discharge interval does not overlap with hindcast horizon")
    if start >= observation.observed_at:
        raise ValueError("hindcast source-time interval must precede the observation")
    interval = TimeRange(start=start, end=end)
    observed = shape(
        transform_geom(
            crs_text(observation.geometry.polygon.crs),
            crs_text(grid.crs),
            _mapping(observation.geometry.polygon),
            precision=-1,
        )
    )
    maximum_speed = config.conservative_maximum_current_speed_m_s + (
        config.conservative_maximum_windage
        * config.conservative_maximum_wind_speed_m_s
    )
    diffusion = config.diffusion_reach_sigma_multiplier * math.sqrt(
        2.0
        * config.conservative_maximum_diffusivity_m2_s
        * config.backward_horizon_seconds
    )
    allowance = (
        maximum_speed * config.backward_horizon_seconds
        + diffusion
        + config.source_roi_padding_metres
    )
    grid_bounds = box(
        grid.transform.c,
        grid.transform.f + grid.transform.e * grid.height,
        grid.transform.c + grid.transform.a * grid.width,
        grid.transform.f,
    )
    source_region = observed.buffer(allowance).intersection(grid_bounds)
    if source_region.is_empty:
        raise ValueError("conservative source region does not overlap the analysis grid")
    return ReachabilityEnvelope(
        grid_crs=grid.crs,
        observation_time=observation.observed_at,
        release_interval=interval,
        source_region=source_region,
        observation_region=observed,
        maximum_advective_speed_m_s=maximum_speed,
        diffusion_allowance_metres=diffusion,
        source_padding_metres=config.source_roi_padding_metres,
    )


def assess_reachability(
    state: HybridSourceState,
    grid: EulerianGrid,
    envelope: ReachabilityEnvelope,
) -> ReachabilityAssessment:
    """Reject only violations of configured conservative bounds."""
    point_value = state.geometry.geometry
    assert point_value.type == "Point"
    point = Point(point_value.coordinate.x, point_value.coordinate.y)
    reasons: list[str] = []
    if state.geometry.crs != grid.crs:
        reasons.append("SOURCE_CRS_GRID_MISMATCH")
    if not envelope.release_interval.start <= state.release_time <= envelope.release_interval.end:
        reasons.append("RELEASE_TIME_OUTSIDE_ALLOWED_INTERVAL")
    if not envelope.source_region.covers(point):
        reasons.append("SOURCE_OUTSIDE_CONSERVATIVE_REACHABLE_ROI")

    transform = affine(grid)
    row, column = rowcol(transform, point.x, point.y)
    if not (0 <= row < grid.height and 0 <= column < grid.width):
        reasons.append("SOURCE_OUTSIDE_ANALYSIS_GRID")
    elif read_land_mask(grid)[row, column]:
        reasons.append("SOURCE_ON_LAND")

    age = (envelope.observation_time - state.release_time).total_seconds()
    uncovered_distance = max(
        0.0,
        point.distance(envelope.observation_region)
        - envelope.diffusion_allowance_metres
        - envelope.source_padding_metres,
    )
    required_speed = math.inf if age <= 0.0 and uncovered_distance > 0.0 else (
        uncovered_distance / age if age > 0.0 else 0.0
    )
    if required_speed > envelope.maximum_advective_speed_m_s:
        reasons.append("REQUIRED_SPEED_EXCEEDS_CONSERVATIVE_BOUND")
    if envelope.maximum_advective_speed_m_s > 0:
        penalty = min(required_speed / envelope.maximum_advective_speed_m_s, 1.0)
    else:
        penalty = 0.0 if required_speed == 0.0 else 1.0
    return ReachabilityAssessment(
        accepted=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        physics_penalty=penalty,
        required_speed_m_s=required_speed,
    )
