"""Framework-neutral, JSON-safe view models for dashboards and other output clients."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from oilspill.domain.ais import CandidateDecision
from oilspill.domain.attribution import EvidenceDirection, MetricStatus
from oilspill.domain.common import (
    ArtifactRef,
    CalibrationStatus,
    DatasetMetadata,
    Decision,
    FrozenModel,
    Identifier,
    MetadataEntry,
    Probability,
    QualityFlag,
    UTCDateTime,
)
from oilspill.domain.geospatial import CRS, AffineTransform


class PointLayerGeometry(FrozenModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]
    crs: CRS


class LineStringLayerGeometry(FrozenModel):
    type: Literal["LineString"] = "LineString"
    coordinates: tuple[tuple[float, float], ...]
    crs: CRS


class PolygonLayerGeometry(FrozenModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: tuple[tuple[tuple[float, float], ...], ...]
    crs: CRS


class MultiPolygonLayerGeometry(FrozenModel):
    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: tuple[tuple[tuple[tuple[float, float], ...], ...], ...]
    crs: CRS


LayerGeometry = Annotated[
    PointLayerGeometry | LineStringLayerGeometry | PolygonLayerGeometry | MultiPolygonLayerGeometry,
    Field(discriminator="type"),
]
PolygonalLayerGeometry = Annotated[
    PolygonLayerGeometry | MultiPolygonLayerGeometry,
    Field(discriminator="type"),
]


class RasterBandView(FrozenModel):
    name: Identifier
    unit: Identifier
    nodata: float | int | None


class RasterLayerView(FrozenModel):
    """Client-ready raster reference with complete geospatial interpretation."""

    layer_id: Identifier
    layer_kind: Identifier
    artifact: ArtifactRef
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crs: CRS
    transform: AffineTransform
    bands: tuple[RasterBandView, ...] = Field(min_length=1)
    semantics: Identifier
    calibrated_probability: bool | None = None


class ComponentView(FrozenModel):
    name: Identifier
    version: Identifier
    implementation: Identifier
    framework: str | None = None
    configuration_sha256: str | None = None
    attributes: tuple[MetadataEntry, ...] = ()


class SpillDetectionView(FrozenModel):
    observation_id: Identifier
    detection_id: Identifier
    scene_id: Identifier
    observed_at: UTCDateTime
    class_label: Identifier
    class_id: int | None
    confidence: Probability | None
    footprint: LayerGeometry
    mask: RasterLayerView
    probability: RasterLayerView | None
    detector: ComponentView
    quality_flags: tuple[QualityFlag, ...] = ()


class SlickPolygonView(FrozenModel):
    observation_id: Identifier
    geometry_id: Identifier
    detection_id: Identifier
    polygon: PolygonalLayerGeometry
    centroid: PointLayerGeometry
    centerline: LineStringLayerGeometry | None
    measurement_crs: CRS | None
    quality_flags: tuple[QualityFlag, ...] = ()


class MeasurementView(FrozenModel):
    value: float
    unit: Identifier


class SpillGeometryStatisticsView(FrozenModel):
    observation_id: Identifier
    geometry_id: Identifier
    area: MeasurementView | None
    length: MeasurementView | None
    width: MeasurementView | None
    dominant_orientation: MeasurementView | None
    orientation_convention: str | None
    measurement_crs: CRS | None
    method: ComponentView


class DischargeTimeWindowView(FrozenModel):
    observation_id: Identifier
    earliest: UTCDateTime
    latest: UTCDateTime
    central_estimate: UTCDateTime | None
    confidence: Probability | None
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]
    estimator: ComponentView
    quality_flags: tuple[QualityFlag, ...] = ()


class AttributionEvidenceView(FrozenModel):
    attribution_id: Identifier
    observation_id: Identifier
    candidate_id: Identifier
    vessel_id: str | None
    name: Identifier
    status: MetricStatus
    raw_value: float | None
    normalized_value: float | None
    unit: str | None
    direction: EvidenceDirection
    configured_weight: float | None
    normalized_weight: float | None
    weighted_contribution: float | None
    explanation: str | None
    provenance: tuple[MetadataEntry, ...] = ()
    source_simulation_ids: tuple[str, ...] = ()


class RankedVesselView(FrozenModel):
    decision: Decision | None = None
    abstention_reason: str | None = None
    attribution_id: Identifier
    observation_id: Identifier
    candidate_id: Identifier
    vessel_id: str | None
    rank: int = Field(gt=0)
    aggregate_score: float | None
    calibrated_probability: Probability | None
    calibration_status: CalibrationStatus
    trajectory: LineStringLayerGeometry | None
    screening_decision: CandidateDecision | None
    screening_reasons: tuple[str, ...] = ()
    evidence: tuple[AttributionEvidenceView, ...] = ()
    disclaimer: str | None = None


class AttributionSummaryView(FrozenModel):
    attribution_id: Identifier
    observation_id: Identifier
    decision: Decision
    ranked_candidate_ids: tuple[Identifier, ...]
    ranked_vessel_ids: tuple[str, ...]
    abstention_reason: str | None = None
    calibrated: bool
    disclaimer: Identifier


class ReleaseTimeQuantilesView(FrozenModel):
    p10: UTCDateTime
    p50: UTCDateTime
    p90: UTCDateTime


class HybridHindcastView(FrozenModel):
    hindcast_id: Identifier
    observation_id: Identifier
    source_posterior_raster_uri: str | None
    credible_region_50: PolygonalLayerGeometry | None
    credible_region_90: PolygonalLayerGeometry | None
    credible_region_95: PolygonalLayerGeometry | None
    release_time: ReleaseTimeQuantilesView | None
    convergence_status: Identifier
    initial_candidate_count: int = Field(ge=0)
    retained_eulerian_count: int = Field(ge=0)
    high_fidelity_count: int = Field(ge=0)
    calibrated: bool
    semantics: Identifier


class HybridScenarioWeightView(FrozenModel):
    scenario_id: Identifier
    cluster_id: Identifier | None
    weight: Probability
    design_weight: float | None
    selection_reason: Identifier


class HybridForecastSummaryView(FrozenModel):
    forecast_id: Identifier
    observation_id: Identifier
    horizons: tuple[UTCDateTime, ...]
    scenario_weights: tuple[HybridScenarioWeightView, ...]
    coarse_scenario_count: int = Field(ge=0)
    high_fidelity_scenario_count: int = Field(ge=0)
    convergence_status: Identifier
    calibrated: bool
    semantics: Identifier
    warnings: tuple[QualityFlag, ...] = ()


class PipelineProvenanceView(FrozenModel):
    random_seed: int | None
    configuration_sha256: str
    configuration_artifact: ArtifactRef
    forcing_ids: tuple[Identifier, ...]
    forcing_artifacts: tuple[ArtifactRef, ...]
    forcing_datasets: tuple[DatasetMetadata, ...]
    components: tuple[ComponentView, ...]


class SourceLocationView(FrozenModel):
    """A model-derived plausible source region, not a physically exact reconstruction."""

    simulation_id: Identifier
    observation_id: str | None
    candidate_id: str | None
    plausible_at: UTCDateTime
    discharge_window_start: UTCDateTime
    discharge_window_end: UTCDateTime
    geometry: LayerGeometry
    particle_positions: ArtifactRef
    particle_count: int = Field(ge=0)
    model: ComponentView
    assumptions: tuple[str, ...] = ()
    caveat: Identifier


class ParticleDistributionView(FrozenModel):
    timestamp: UTCDateTime
    positions: ArtifactRef
    particle_count: int = Field(ge=0)
    support_geometry: LayerGeometry


class SpatialProbabilityView(FrozenModel):
    method: Identifier
    member_support_geometries: tuple[LayerGeometry, ...] = Field(min_length=1)
    probability_per_member: Probability | None = None
    member_weights: tuple[float, ...] | None = None
    probability_raster: RasterLayerView | None
    calibrated: bool
    explanation: Identifier


class ForecastLayerView(FrozenModel):
    forecast_id: Identifier
    observation_id: Identifier
    issued_at: UTCDateTime
    valid_at: UTCDateTime
    member_simulation_ids: tuple[str, ...]
    particle_ensemble: tuple[ParticleDistributionView, ...]
    probability: SpatialProbabilityView | None
    occupancy_or_probability: RasterLayerView | None
    uncertainty_method: Identifier
    calibrated: bool
    configured_perturbations: tuple[MetadataEntry, ...] = ()


class CoastalImpactSegmentView(FrozenModel):
    segment_id: Identifier
    geometry: LayerGeometry
    contact_probability: Probability | None
    estimated_arrival: UTCDateTime | None
    earliest_plausible_arrival: UTCDateTime | None
    latest_plausible_arrival: UTCDateTime | None
    arrival_p10: UTCDateTime | None = None
    arrival_p50: UTCDateTime | None = None
    arrival_p90: UTCDateTime | None = None
    contacting_member_ids: tuple[str, ...]
    ensemble_hit_count: int = Field(ge=0)
    ensemble_member_count: int = Field(gt=0)
    provenance: tuple[MetadataEntry, ...] = ()


class CoastalImpactLayerView(FrozenModel):
    impact_id: Identifier
    forecast_id: Identifier
    coastline_dataset_name: Identifier
    coastline_dataset_version: Identifier
    contact_method: str | None
    created_at: UTCDateTime | None
    segments: tuple[CoastalImpactSegmentView, ...]
    uncertainty_method: Identifier
    calibrated: bool
    quality_flags: tuple[QualityFlag, ...] = ()


class DashboardViewModel(FrozenModel):
    """Complete dashboard contract derived only from ``PipelineResult``."""

    schema_version: int = 1
    run_id: Identifier
    configuration_sha256: str
    provenance: ArtifactRef
    spill_detections: tuple[SpillDetectionView, ...]
    slick_polygons: tuple[SlickPolygonView, ...]
    spill_geometry_statistics: tuple[SpillGeometryStatisticsView, ...]
    discharge_time_windows: tuple[DischargeTimeWindowView, ...]
    ranked_vessels: tuple[RankedVesselView, ...]
    attribution_evidence: tuple[AttributionEvidenceView, ...]
    estimated_source_locations: tuple[SourceLocationView, ...]
    forecast_layers: tuple[ForecastLayerView, ...]
    coastal_impact_layers: tuple[CoastalImpactLayerView, ...]
    pipeline_warnings: tuple[QualityFlag, ...] = ()
    hybrid_hindcasts: tuple[HybridHindcastView, ...] = ()
    attribution_summaries: tuple[AttributionSummaryView, ...] = ()
    hybrid_forecast_summaries: tuple[HybridForecastSummaryView, ...] = ()
    execution_provenance: PipelineProvenanceView | None = None
