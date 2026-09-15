"""Typed inputs to replaceable component interfaces.

Scientific parameters are represented explicitly by component-specific validated settings at
composition time. These request objects contain only shared cross-component data contracts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from oilspill.domain.ais import CandidateVessel, VesselTrack
from oilspill.domain.attribution import CandidateMetricEvidence, EvidenceValue
from oilspill.domain.common import (
    ArtifactRef,
    DatasetMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    TimeRange,
    UTCDateTime,
)
from oilspill.domain.drift import DriftSimulation, ReleaseHypothesis
from oilspill.domain.environment import EnvironmentalField, EnvironmentalVariable
from oilspill.domain.eulerian import (
    EulerianGrid,
    EulerianScenario,
    ObservedSlickInitialCondition,
    PointSourceInitialCondition,
)
from oilspill.domain.forecast import ForecastResult
from oilspill.domain.geospatial import SpatialGeometry
from oilspill.domain.hybrid import HybridHindcastResult, SourcePosterior
from oilspill.domain.sar import SARScene
from oilspill.domain.source_inference import SourceInferenceResult
from oilspill.domain.source_trace import BackwardSourceTraceResult, ForwardSourceTraceResult
from oilspill.domain.spill import SpillDetection, SpillGeometry, SpillObservation


class SARReadRequest(FrozenModel):
    source: ArtifactRef
    scene_id: str | None = None


class AgeEstimationRequest(FrozenModel):
    scene: SARScene
    detection: SpillDetection
    geometry: SpillGeometry
    vessel_tracks: tuple[VesselTrack, ...] = ()


class AISQuery(FrozenModel):
    area: SpatialGeometry
    interval: TimeRange
    vessel_ids: tuple[str, ...] = ()


class CandidateGenerationRequest(FrozenModel):
    observation: SpillObservation
    tracks: tuple[VesselTrack, ...]
    search_area: SpatialGeometry | None = None
    search_interval: TimeRange | None = None


class ReleaseGenerationRequest(FrozenModel):
    observation: SpillObservation
    candidate: CandidateVessel


class AttributionEvidenceRequest(FrozenModel):
    observation: SpillObservation
    candidate: CandidateVessel
    forward_trace: ForwardSourceTraceResult | None = None
    backward_trace: BackwardSourceTraceResult | None = None
    source_inference: SourceInferenceResult | None = None
    source_posterior: SourcePosterior | None = None
    hybrid_hindcast: HybridHindcastResult | None = None


class EnvironmentalQuery(FrozenModel):
    area: SpatialGeometry
    interval: TimeRange
    variables: tuple[EnvironmentalVariable, ...]
    requested_vertical_reference: str | None = None


class ForwardTraceRequest(FrozenModel):
    observation: SpillObservation
    candidate: CandidateVessel | None = None
    release: ReleaseHypothesis
    forcing: tuple[EnvironmentalField, ...]
    random_seed: int
    model_parameters: tuple[MetadataEntry, ...] = ()
    configured_uncertainty: tuple[MetadataEntry, ...] = ()


class BackwardTraceRequest(FrozenModel):
    observation: SpillObservation
    release: ReleaseHypothesis
    forcing: tuple[EnvironmentalField, ...]
    candidate: CandidateVessel | None = None
    random_seed: int


class DriftForecastRequest(FrozenModel):
    observation: SpillObservation
    release: ReleaseHypothesis
    forcing: tuple[EnvironmentalField, ...]
    valid_until: UTCDateTime
    random_seed: int
    initial_position: SpatialGeometry | None = None
    initial_timestamp: UTCDateTime | None = None
    model_parameters: tuple[MetadataEntry, ...] = ()
    configured_uncertainty: tuple[MetadataEntry, ...] = ()


class AttributionRequest(FrozenModel):
    observation: SpillObservation
    candidates: tuple[CandidateVessel, ...]
    simulations: tuple[DriftSimulation, ...]
    candidate_metrics: tuple[CandidateMetricEvidence, ...] = ()
    forward_traces: tuple[ForwardSourceTraceResult, ...] = ()
    backward_traces: tuple[BackwardSourceTraceResult, ...] = ()
    supplemental_evidence: tuple[EvidenceValue, ...] = ()
    interpretation_caveat: str | None = None


class EnsembleForecastRequest(FrozenModel):
    observation: SpillObservation
    release: ReleaseHypothesis
    forcing: tuple[EnvironmentalField, ...]
    horizons: tuple[UTCDateTime, ...] = ()
    random_seed: int
    coastline_geometries: tuple[SpatialGeometry, ...] = ()
    coastline_source: ArtifactRef | None = None


class CoastlineDataset(FrozenModel):
    dataset_id: Identifier
    segments: tuple[SpatialGeometry, ...]
    source: ArtifactRef
    metadata: DatasetMetadata


class CoastalImpactRequest(FrozenModel):
    forecast: ForecastResult
    coastline: CoastlineDataset


class ObservationComparisonRequest(FrozenModel):
    observation: SpillObservation
    simulation: DriftSimulation


class SourceSearchRequest(FrozenModel):
    observation: SpillObservation
    candidates: tuple[CandidateVessel, ...]
    forcing: tuple[EnvironmentalField, ...]
    random_seed: int
    model_parameters: tuple[MetadataEntry, ...] = ()
    configured_uncertainty: tuple[MetadataEntry, ...] = ()
    backward_evidence: tuple[BackwardSourceTraceResult, ...] = ()


EulerianInitialCondition = Annotated[
    PointSourceInitialCondition | ObservedSlickInitialCondition,
    Field(discriminator="kind"),
]


class EulerianTransportRequest(FrozenModel):
    """Complete, typed input to a replaceable coarse transport engine."""

    grid: EulerianGrid
    scenario: EulerianScenario
    initial_condition: EulerianInitialCondition
    forcing: tuple[EnvironmentalField, ...]
    snapshot_times: tuple[UTCDateTime, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> EulerianTransportRequest:
        if self.snapshot_times != tuple(sorted(self.snapshot_times)) or len(
            set(self.snapshot_times)
        ) != len(self.snapshot_times):
            raise ValueError("Eulerian snapshot times must be unique and chronological")
        start = (
            self.initial_condition.release_time
            if self.initial_condition.kind == "point_source"
            else self.initial_condition.observed_at
        )
        if any(timestamp <= start for timestamp in self.snapshot_times):
            raise ValueError("Eulerian snapshots must follow the initial-condition time")
        field_ids = tuple(field.field_id for field in self.forcing)
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("Eulerian request forcing IDs must be unique")
        if self.scenario.forcing_field_ids and set(self.scenario.forcing_field_ids) != set(
            field_ids
        ):
            raise ValueError("Eulerian request forcing does not match scenario selection")
        if (
            self.initial_condition.kind == "observed_slick_field"
            and (
                self.initial_condition.concentration.grid.crs != self.grid.crs
                or self.initial_condition.concentration.grid.width != self.grid.width
                or self.initial_condition.concentration.grid.height != self.grid.height
                or self.initial_condition.concentration.grid.transform != self.grid.transform
            )
        ):
            raise ValueError("observed slick field must use the configured Eulerian grid")
        return self


class HybridHindcastRequest(FrozenModel):
    observation: SpillObservation
    forcing: tuple[EnvironmentalField, ...]
    random_seed: int = Field(ge=0, lt=2**32)
    land_geometries: tuple[SpatialGeometry, ...] = ()
    land_source: ArtifactRef | None = None
