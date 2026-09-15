"""Engine-neutral forward and backward source-tracing results."""

from __future__ import annotations

from pydantic import Field, model_validator

from oilspill.domain.ais import CandidateVessel
from oilspill.domain.common import FrozenModel, Identifier, MetadataEntry, TimeRange, UTCDateTime
from oilspill.domain.drift import (
    DriftMode,
    DriftSimulation,
    ParticleDistribution,
    ParticleTrajectory,
)
from oilspill.domain.geospatial import SpatialGeometry
from oilspill.domain.spill import SpillObservation


class ForwardSourceTraceResult(FrozenModel):
    """One candidate release propagated to the SAR observation time.

    This object carries comparison inputs, not an attribution score. ``simulation`` is the
    complete provenance-bearing engine-neutral simulation record.
    """

    trace_id: Identifier
    candidate: CandidateVessel
    release_position: SpatialGeometry
    release_time: UTCDateTime
    simulated_particle_distribution: ParticleDistribution
    comparison_ready_geometry: SpatialGeometry
    simulation: DriftSimulation

    @model_validator(mode="after")
    def _consistent(self) -> ForwardSourceTraceResult:
        if self.release_position.geometry.type != "Point":
            raise ValueError("forward trace release_position must contain Point geometry")
        if self.simulation.mode != DriftMode.FORWARD_TRACE:
            raise ValueError("forward trace result requires a forward simulation")
        if self.simulation.release_coordinates != self.release_position:
            raise ValueError("release_position must match simulation provenance")
        if self.simulation.release_timestamp != self.release_time:
            raise ValueError("release_time must match simulation provenance")
        if self.simulation.target_distribution != self.simulated_particle_distribution:
            raise ValueError("particle distribution must match simulation output")
        if (
            self.simulated_particle_distribution.comparison_geometry
            != self.comparison_ready_geometry
        ):
            raise ValueError("comparison geometry must match the particle distribution")
        return self


class BackwardSourceTraceResult(FrozenModel):
    """Backward-model hypotheses prepared for later AIS trajectory comparison.

    Paths and source locations are model-dependent possibilities, not physically exact
    reconstructions. No vessel score or rank is represented here.
    """

    trace_id: Identifier
    observation: SpillObservation
    plausible_source_time_range: TimeRange
    plausible_source_distribution: ParticleDistribution
    plausible_source_geometry: SpatialGeometry
    particle_trajectories: tuple[ParticleTrajectory, ...] = Field(min_length=1)
    plausible_source_paths: tuple[SpatialGeometry, ...] = Field(min_length=1)
    configured_uncertainty: tuple[MetadataEntry, ...]
    model_parameters: tuple[MetadataEntry, ...]
    assumptions: tuple[str, ...] = Field(min_length=1)
    simulation: DriftSimulation

    @model_validator(mode="after")
    def _consistent(self) -> BackwardSourceTraceResult:
        if self.simulation.mode != DriftMode.BACKWARD_TRACE:
            raise ValueError("backward trace result requires a backward simulation")
        if self.simulation.target_distribution != self.plausible_source_distribution:
            raise ValueError("source distribution must match simulation output")
        if self.plausible_source_distribution.comparison_geometry != self.plausible_source_geometry:
            raise ValueError("source geometry must match the particle distribution")
        if self.simulation.particle_trajectories != self.particle_trajectories:
            raise ValueError("particle trajectories must match simulation output")
        if self.simulation.configured_uncertainty != self.configured_uncertainty:
            raise ValueError("configured uncertainty must match simulation provenance")
        if self.simulation.model_parameters != self.model_parameters:
            raise ValueError("model parameters must match simulation provenance")
        if any(path.geometry.type != "LineString" for path in self.plausible_source_paths):
            raise ValueError("plausible source paths must contain LineString geometry")
        return self
