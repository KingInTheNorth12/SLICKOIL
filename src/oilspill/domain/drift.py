"""Generic drift release and simulation contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    QualityFlag,
    TimeRange,
    UTCDateTime,
)
from oilspill.domain.geospatial import CRS, SpatialGeometry, SpatialUnit


class DriftMode(StrEnum):
    FORWARD_TRACE = "forward_trace"
    BACKWARD_TRACE = "backward_trace"
    FORECAST = "forecast"


class SimulationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class ReleaseHypothesis(FrozenModel):
    """Configured release hypothesis; unknown oil properties may remain absent."""

    release_id: Identifier
    geometry: SpatialGeometry
    interval: TimeRange
    source_candidate_id: str | None = None
    oil_type: str | None = None
    assumptions: tuple[str, ...] = ()


class SimulationDiagnostics(FrozenModel):
    particle_count_requested: int = Field(ge=0)
    particle_count_completed: int = Field(ge=0)
    attributes: tuple[MetadataEntry, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _completed_not_greater(self) -> SimulationDiagnostics:
        if self.particle_count_completed > self.particle_count_requested:
            raise ValueError("completed particle count cannot exceed requested count")
        return self


class ParticleDistribution(FrozenModel):
    """Canonical particle state at one UTC time, independent of a drift engine.

    ``positions`` references the complete particle output. ``comparison_geometry`` is a
    CRS-explicit envelope derived by the adapter for spatial comparison; it does not imply a
    scientifically calibrated slick boundary or concentration contour.
    """

    timestamp: UTCDateTime
    positions: ArtifactRef
    particle_count: int = Field(ge=0)
    comparison_geometry: SpatialGeometry
    attributes: tuple[MetadataEntry, ...] = ()


class ParticlePosition(FrozenModel):
    """One simulated particle position in its declared CRS at a UTC timestamp."""

    timestamp: UTCDateTime
    position: SpatialGeometry

    @model_validator(mode="after")
    def _point(self) -> ParticlePosition:
        if self.position.geometry.type != "Point":
            raise ValueError("particle position must contain Point geometry")
        return self


class ParticleTrajectory(FrozenModel):
    """Chronological canonical trajectory suitable for later AIS-path comparison."""

    particle_id: Identifier
    positions: tuple[ParticlePosition, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _consistent(self) -> ParticleTrajectory:
        timestamps = tuple(item.timestamp for item in self.positions)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("particle trajectory positions must be chronological")
        crs = self.positions[0].position.crs
        if any(item.position.crs != crs for item in self.positions):
            raise ValueError("particle trajectory positions must use one CRS")
        return self


class DriftSimulation(FrozenModel):
    """Engine-neutral drift output; native engine objects never cross this boundary."""

    simulation_id: Identifier
    mode: DriftMode
    release: ReleaseHypothesis
    release_coordinates: SpatialGeometry | None = None
    release_timestamp: UTCDateTime | None = None
    target_timestamp: UTCDateTime | None = None
    forcing_ids: tuple[str, ...]
    forcing_artifacts: tuple[ArtifactRef, ...] = ()
    interval: TimeRange
    trajectories: ArtifactRef
    output_crs: CRS
    spatial_unit: SpatialUnit
    model: ComponentMetadata
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    random_seed: int
    configured_uncertainty: tuple[MetadataEntry, ...] = ()
    model_parameters: tuple[MetadataEntry, ...] = ()
    target_distribution: ParticleDistribution | None = None
    particle_trajectories: tuple[ParticleTrajectory, ...] = ()
    status: SimulationStatus
    diagnostics: SimulationDiagnostics

    @model_validator(mode="after")
    def _unit_matches_crs(self) -> DriftSimulation:
        if self.spatial_unit not in self.output_crs.axis_units:
            raise ValueError("simulation spatial unit must match an output CRS axis unit")
        if (
            self.release_coordinates is not None
            and self.release_coordinates.geometry.type != "Point"
        ):
            raise ValueError("simulation release_coordinates must contain Point geometry")
        if self.release_timestamp is not None and not (
            self.release.interval.start <= self.release_timestamp <= self.release.interval.end
        ):
            raise ValueError("simulation release_timestamp must lie in release hypothesis interval")
        if self.target_distribution is not None:
            if self.target_timestamp != self.target_distribution.timestamp:
                raise ValueError("target distribution timestamp must equal target_timestamp")
            if self.target_distribution.comparison_geometry.crs != self.output_crs:
                raise ValueError("target distribution geometry CRS must equal output_crs")
        if any(
            position.position.crs != self.output_crs
            for trajectory in self.particle_trajectories
            for position in trajectory.positions
        ):
            raise ValueError("particle trajectory CRS must equal output_crs")
        return self
