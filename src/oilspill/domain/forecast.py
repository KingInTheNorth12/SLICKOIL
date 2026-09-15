"""Ensemble forecast and coastal-impact contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    DatasetMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    Probability,
    QualityFlag,
    UTCDateTime,
)
from oilspill.domain.drift import DriftSimulation, ParticleDistribution
from oilspill.domain.ensemble import EnsembleMemberOutcome, OccupancySummary
from oilspill.domain.geospatial import Coordinate, RasterAsset, SpatialGeometry, SpatialUnit
from oilspill.domain.hybrid import HybridForecastDiagnostics


class UncertaintySummary(FrozenModel):
    """Describes uncertainty semantics without assuming a particular ensemble method."""

    method: Identifier
    ensemble_member_count: int = Field(ge=0)
    calibrated: bool = False
    random_seed: int | None = None
    configured_perturbations: tuple[MetadataEntry, ...] = ()
    notes: tuple[str, ...] = ()


class ForecastDecision(StrEnum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"


class SpatialProbabilityRepresentation(FrozenModel):
    """Uncalibrated empirical mixture of member support geometries."""

    method: Identifier
    member_geometries: tuple[SpatialGeometry, ...] = Field(min_length=1)
    probability_per_member: Probability | None = None
    member_weights: tuple[float, ...] | None = None
    probability_raster: RasterAsset | None = None
    calibrated: bool = False
    explanation: Identifier

    @model_validator(mode="after")
    def _shared_crs_and_mass(self) -> SpatialProbabilityRepresentation:
        crs = self.member_geometries[0].crs
        if any(geometry.crs != crs for geometry in self.member_geometries):
            raise ValueError("spatial probability member geometries must share one CRS")
        if self.member_weights is None:
            if self.probability_per_member is None:
                raise ValueError("spatial probability requires equal or explicit member weights")
            expected = 1.0 / len(self.member_geometries)
            if abs(self.probability_per_member - expected) > 1e-12:
                raise ValueError("equal-weight probability must be reciprocal of member count")
        else:
            if len(self.member_weights) != len(self.member_geometries):
                raise ValueError("member weights must match support geometries")
            if any(not math.isfinite(weight) or weight < 0.0 for weight in self.member_weights):
                raise ValueError("member weights must be finite and non-negative")
            if abs(sum(self.member_weights) - 1.0) > 1e-9:
                raise ValueError("member weights must sum to one")
            if self.probability_per_member is not None and any(
                abs(weight - self.probability_per_member) > 1e-12
                for weight in self.member_weights
            ):
                raise ValueError("equal and explicit member weights disagree")
        return self


class ForecastRunProvenance(FrozenModel):
    member_id: Identifier
    horizon: UTCDateTime
    simulation_id: Identifier
    random_seed: int
    forcing_ids: tuple[str, ...]
    position_offset: Coordinate
    position_offset_unit: SpatialUnit
    time_offset_seconds: float
    model_parameters: tuple[MetadataEntry, ...] = ()
    configured_uncertainty: tuple[MetadataEntry, ...] = ()
    validation_note: Identifier
    trajectory_artifact: ArtifactRef
    scenario_id: Identifier | None = None
    cluster_id: Identifier | None = None
    scenario_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    design_weight: float | None = Field(default=None, ge=0.0)


class ForecastHorizon(FrozenModel):
    valid_at: UTCDateTime
    occupancy_or_probability: RasterAsset | None = None
    particle_ensemble: tuple[ParticleDistribution, ...] = ()
    spatial_probability: SpatialProbabilityRepresentation | None = None
    member_simulation_ids: tuple[str, ...]
    occupancy_summary: OccupancySummary | None = None
    occupancy_artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def _distribution_count(self) -> ForecastHorizon:
        if self.particle_ensemble and len(self.particle_ensemble) != len(
            self.member_simulation_ids
        ):
            raise ValueError("particle ensemble count must match horizon simulation count")
        return self


class ForecastResult(FrozenModel):
    forecast_id: Identifier
    observation_id: Identifier
    issued_at: UTCDateTime
    horizons: tuple[ForecastHorizon, ...] = Field(min_length=1)
    members: tuple[DriftSimulation, ...]
    ensemble_plan: ArtifactRef | None = None
    member_outcomes: tuple[EnsembleMemberOutcome, ...] = ()
    uncertainty: UncertaintySummary
    run_provenance: tuple[ForecastRunProvenance, ...] = ()
    warnings: tuple[QualityFlag, ...] = ()
    hybrid_diagnostics: HybridForecastDiagnostics | None = None
    decision: ForecastDecision = ForecastDecision.COMPLETED
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _valid_members(self) -> ForecastResult:
        if self.decision == ForecastDecision.ABSTAINED:
            if not self.abstention_reason:
                raise ValueError("abstained forecast requires a reason")
            if self.members:
                raise ValueError("abstained forecast cannot expose aggregated members")
        elif self.abstention_reason is not None:
            raise ValueError("completed forecast cannot include an abstention reason")
        if not self.members and not self.member_outcomes:
            raise ValueError("empty forecasts require explicit member failure outcomes")
        if self.member_outcomes:
            successful = {
                o.simulation_id
                for o in self.member_outcomes
                if o.status == "succeeded" and o.included_in_aggregation is not False
            }
            if self.decision != ForecastDecision.ABSTAINED and successful != {
                s.simulation_id for s in self.members
            }:
                raise ValueError("successful outcomes must match forecast simulations")
            if len({o.member_id for o in self.member_outcomes}) != len(self.member_outcomes):
                raise ValueError("duplicate member outcomes")
        member_ids = {item.simulation_id for item in self.members}
        referenced_ids = {
            simulation_id
            for horizon in self.horizons
            for simulation_id in horizon.member_simulation_ids
        }
        if not referenced_ids.issubset(member_ids):
            raise ValueError("forecast horizons reference unknown simulations")
        if self.uncertainty.ensemble_member_count != len(self.members):
            raise ValueError("uncertainty member count must match forecast members")
        provenance_ids = {item.simulation_id for item in self.run_provenance}
        if provenance_ids and provenance_ids != member_ids:
            raise ValueError("forecast run provenance must cover every simulation exactly once")
        return self


class ScenarioContactArrival(FrozenModel):
    scenario_id: Identifier
    first_contact: UTCDateTime
    scenario_weight: Probability
    weight_semantics: Literal["explicit_design_weight", "implicit_equal_member_weight"]


class CoastlineSegmentImpact(FrozenModel):
    segment_id: Identifier
    geometry: SpatialGeometry
    impact_probability: Probability | None
    estimated_arrival: UTCDateTime | None = None
    earliest_arrival: UTCDateTime | None = None
    latest_arrival: UTCDateTime | None = None
    arrival_p10: UTCDateTime | None = None
    arrival_p50: UTCDateTime | None = None
    arrival_p90: UTCDateTime | None = None
    arrival_estimate_method: str | None = None
    contacting_member_ids: tuple[str, ...] = ()
    contact_arrivals: tuple[ScenarioContactArrival, ...] = ()
    ensemble_hit_count: int = Field(ge=0)
    ensemble_member_count: int = Field(gt=0)
    provenance: tuple[MetadataEntry, ...] = ()

    @model_validator(mode="after")
    def _valid_counts_and_times(self) -> CoastlineSegmentImpact:
        if self.ensemble_hit_count > self.ensemble_member_count:
            raise ValueError("ensemble hits cannot exceed member count")
        if self.ensemble_hit_count != len(self.contacting_member_ids):
            raise ValueError("ensemble hit count must match contacting member IDs")
        if self.contact_arrivals:
            arrival_ids = tuple(item.scenario_id for item in self.contact_arrivals)
            if len(arrival_ids) != len(set(arrival_ids)):
                raise ValueError("scenario contact arrivals must have unique IDs")
            if set(arrival_ids) != set(self.contacting_member_ids):
                raise ValueError("scenario contact arrivals must match contacting member IDs")
        if (
            self.earliest_arrival is not None
            and self.latest_arrival is not None
            and self.latest_arrival < self.earliest_arrival
        ):
            raise ValueError("latest arrival cannot precede earliest arrival")
        quantiles = (self.arrival_p10, self.arrival_p50, self.arrival_p90)
        if any(value is not None for value in quantiles):
            if any(value is None for value in quantiles):
                raise ValueError("arrival P10/P50/P90 must be provided together")
            p10, p50, p90 = quantiles
            assert p10 is not None and p50 is not None and p90 is not None
            if not p10 <= p50 <= p90:
                raise ValueError("arrival quantiles must satisfy P10 <= P50 <= P90")
            if self.earliest_arrival is not None and p10 < self.earliest_arrival:
                raise ValueError("arrival P10 cannot precede earliest arrival")
            if self.latest_arrival is not None and p90 > self.latest_arrival:
                raise ValueError("arrival P90 cannot follow latest arrival")
        return self


class CoastalImpactResult(FrozenModel):
    impact_id: Identifier
    forecast_id: Identifier
    coastline_dataset: DatasetMetadata
    segments: tuple[CoastlineSegmentImpact, ...]
    uncertainty: UncertaintySummary
    analysis: ComponentMetadata | None = None
    coastline_source: ArtifactRef | None = None
    contact_method: str | None = None
    created_at: UTCDateTime | None = None
    warnings: tuple[QualityFlag, ...] = ()
