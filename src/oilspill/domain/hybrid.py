"""Typed physics-first hybrid hindcast, posterior, and forecast diagnostics."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    Identifier,
    Probability,
    QualityFlag,
    UTCDateTime,
)
from oilspill.domain.drift import DriftSimulation
from oilspill.domain.eulerian import EulerianGrid, EulerianScenario, EulerianTransportResult
from oilspill.domain.geospatial import RasterAsset, SpatialGeometry


class HybridConvergenceStatus(StrEnum):
    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PARTIAL = "partial"
    NOT_ASSESSED = "not_assessed"


class HybridSourceState(FrozenModel):
    source_state_id: Identifier
    geometry: SpatialGeometry
    release_time: UTCDateTime
    windage_coefficient: float = Field(ge=0.0, allow_inf_nan=False)
    horizontal_diffusivity_m2_s: float = Field(ge=0.0, allow_inf_nan=False)
    first_order_loss_rate_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    forcing_member_id: Identifier | None = None
    assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _point(self) -> HybridSourceState:
        if self.geometry.geometry.type != "Point":
            raise ValueError("hybrid source state geometry must be a Point")
        if not any("DATASET_VALIDATION_REQUIRED" in item for item in self.assumptions):
            raise ValueError("hybrid source assumptions must mark DATASET_VALIDATION_REQUIRED")
        return self


class CoarseSourceEvaluation(FrozenModel):
    source_state: HybridSourceState
    eulerian_result_id: Identifier
    soft_iou: Probability
    centroid_error_metres: float = Field(ge=0.0, allow_inf_nan=False)
    centroid_error: float = Field(ge=0.0, allow_inf_nan=False)
    area_error_square_metres: float = Field(ge=0.0, allow_inf_nan=False)
    area_error: float = Field(ge=0.0, allow_inf_nan=False)
    covariance_shape_error: float = Field(ge=0.0, allow_inf_nan=False)
    physics_penalty: float = Field(ge=0.0, allow_inf_nan=False)
    total_mismatch_score: float = Field(ge=0.0, allow_inf_nan=False)
    retained: bool


class ReachabilityRejection(FrozenModel):
    source_state: HybridSourceState
    reasons: tuple[Identifier, ...] = Field(min_length=1)


class CoarseHindcastDiagnostics(FrozenModel):
    valid_candidate_count: int = Field(ge=0)
    rejected_candidate_count: int = Field(ge=0)
    absolute_threshold_pass_count: int = Field(ge=0)
    low_confidence: bool
    converged: bool
    status: Literal["screening_supported", "low_confidence_no_absolute_match"]
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent_confidence(self) -> CoarseHindcastDiagnostics:
        expected_low = self.absolute_threshold_pass_count == 0
        if self.low_confidence != expected_low or self.converged == expected_low:
            raise ValueError("coarse confidence must reflect absolute-threshold support")
        expected_status = (
            "low_confidence_no_absolute_match" if expected_low else "screening_supported"
        )
        if self.status != expected_status:
            raise ValueError("coarse confidence status is inconsistent")
        return self


class CoarseHybridHindcastResult(FrozenModel):
    hindcast_id: Identifier
    observation_id: Identifier
    initial_candidate_count: int = Field(ge=0)
    reachable_candidate_count: int = Field(ge=0)
    retained_candidate_count: int = Field(ge=0)
    sampled_source_states: tuple[HybridSourceState, ...]
    reachability_rejections: tuple[ReachabilityRejection, ...] = ()
    coarse_evaluations: tuple[CoarseSourceEvaluation, ...]
    eulerian_results: tuple[EulerianTransportResult, ...]
    analysis_grid: EulerianGrid | None = None
    diagnostics: CoarseHindcastDiagnostics
    component: ComponentMetadata
    provenance: ArtifactRef
    warnings: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _consistent_counts(self) -> CoarseHybridHindcastResult:
        if self.initial_candidate_count != len(self.sampled_source_states):
            raise ValueError("initial candidate count must match sampled source states")
        if self.reachable_candidate_count != len(self.coarse_evaluations):
            raise ValueError("reachable candidate count must match coarse evaluations")
        if self.retained_candidate_count != sum(e.retained for e in self.coarse_evaluations):
            raise ValueError("retained candidate count must match retained evaluations")
        if len(self.eulerian_results) != len(self.coarse_evaluations):
            raise ValueError("every coarse evaluation requires one Eulerian result")
        if {e.eulerian_result_id for e in self.coarse_evaluations} != {
            result.result_id for result in self.eulerian_results
        }:
            raise ValueError("coarse evaluation Eulerian result provenance mismatch")
        if self.diagnostics.valid_candidate_count != self.reachable_candidate_count:
            raise ValueError("coarse diagnostic valid count mismatch")
        if self.diagnostics.rejected_candidate_count != len(self.reachability_rejections):
            raise ValueError("coarse diagnostic rejected count mismatch")
        return self


class HighFidelitySourceEvaluation(FrozenModel):
    source_state: HybridSourceState
    simulation: DriftSimulation
    mismatch_score: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_posterior_weight: Probability
    generation_number: int = Field(ge=0)
    soft_iou: Probability = 0.0
    centroid_error_metres: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    centroid_error: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    area_error_square_metres: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    area_error: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    covariance_shape_error: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class WeightedSourceState(FrozenModel):
    source_state: HybridSourceState
    normalized_weight: Probability


class ReleaseTimeQuantiles(FrozenModel):
    p10: UTCDateTime
    p50: UTCDateTime
    p90: UTCDateTime

    @model_validator(mode="after")
    def _ordered(self) -> ReleaseTimeQuantiles:
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("release-time quantiles must satisfy P10 <= P50 <= P90")
        return self


class SourcePosterior(FrozenModel):
    """Normalized model-support weights, explicitly uncalibrated for the MVP."""

    posterior_id: Identifier
    observation_id: Identifier
    weighted_source_states: tuple[WeightedSourceState, ...] = Field(min_length=1)
    weighted_source_centroid: SpatialGeometry | None = None
    source_probability_raster: RasterAsset | None = None
    credible_region_50: SpatialGeometry | None = None
    credible_region_90: SpatialGeometry | None = None
    credible_region_95: SpatialGeometry | None = None
    release_time_quantiles: ReleaseTimeQuantiles | None = None
    calibrated: Literal[False] = False
    semantics: Literal["weighted model support; not proof of origin"] = (
        "weighted model support; not proof of origin"
    )
    convergence_status: HybridConvergenceStatus
    warnings: tuple[QualityFlag, ...] = ()
    provenance: ArtifactRef

    @model_validator(mode="after")
    def _consistent(self) -> SourcePosterior:
        ids = tuple(item.source_state.source_state_id for item in self.weighted_source_states)
        if len(ids) != len(set(ids)):
            raise ValueError("source posterior state IDs must be unique")
        total = sum(item.normalized_weight for item in self.weighted_source_states)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("source posterior weights must sum to one")
        source_crs = self.weighted_source_states[0].source_state.geometry.crs
        if any(
            item.source_state.geometry.crs != source_crs
            for item in self.weighted_source_states
        ):
            raise ValueError("source posterior states must use one CRS")
        if self.weighted_source_centroid is not None and (
            self.weighted_source_centroid.crs != source_crs
            or self.weighted_source_centroid.geometry.type != "Point"
        ):
            raise ValueError("weighted source centroid must be a Point in the source CRS")
        regions = tuple(
            region
            for region in (
                self.credible_region_50,
                self.credible_region_90,
                self.credible_region_95,
            )
            if region is not None
        )
        if any(
            region.crs != source_crs
            or region.geometry.type not in {"Polygon", "MultiPolygon"}
            for region in regions
        ):
            raise ValueError("credible regions must be polygonal and use the source-state CRS")
        return self


class HybridConvergenceDiagnostics(FrozenModel):
    status: HybridConvergenceStatus
    converged: bool | None = None
    generation_count: int = Field(ge=0)
    evaluated_state_count: int = Field(ge=0)
    retained_state_count: int = Field(ge=0)
    termination_reason: Identifier
    best_mismatch_by_generation: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _valid_counts(self) -> HybridConvergenceDiagnostics:
        if self.retained_state_count > self.evaluated_state_count:
            raise ValueError("retained source states cannot exceed evaluated source states")
        if any(value < 0.0 for value in self.best_mismatch_by_generation):
            raise ValueError("generation mismatch diagnostics cannot be negative")
        if self.converged is not None and self.converged != (
            self.status == HybridConvergenceStatus.CONVERGED
        ):
            raise ValueError("convergence boolean must agree with convergence status")
        return self


class HybridHindcastResult(FrozenModel):
    hindcast_id: Identifier
    observation_id: Identifier
    initial_candidate_count: int = Field(ge=0)
    retained_eulerian_count: int = Field(ge=0)
    high_fidelity_count: int = Field(ge=0)
    coarse_evaluations: tuple[CoarseSourceEvaluation, ...]
    high_fidelity_evaluations: tuple[HighFidelitySourceEvaluation, ...]
    posterior: SourcePosterior
    convergence_diagnostics: HybridConvergenceDiagnostics
    component: ComponentMetadata
    warnings: tuple[QualityFlag, ...] = ()

    @property
    def source_posterior(self) -> SourcePosterior:
        """Explicit stage-ordering name for the posterior produced before AIS access."""
        return self.posterior

    @model_validator(mode="after")
    def _consistent(self) -> HybridHindcastResult:
        if self.posterior.observation_id != self.observation_id:
            raise ValueError("hybrid hindcast posterior observation mismatch")
        if self.initial_candidate_count < len(self.coarse_evaluations):
            raise ValueError("initial candidate count cannot be below coarse evaluations")
        if self.retained_eulerian_count != sum(
            evaluation.retained for evaluation in self.coarse_evaluations
        ):
            raise ValueError("retained Eulerian count must match coarse evaluations")
        if self.high_fidelity_count != len(self.high_fidelity_evaluations):
            raise ValueError("high-fidelity count must match high-fidelity evaluations")
        if self.high_fidelity_count:
            total = sum(
                evaluation.normalized_posterior_weight
                for evaluation in self.high_fidelity_evaluations
            )
            if abs(total - 1.0) > 1e-9:
                raise ValueError("high-fidelity posterior weights must sum to one")
        return self


class HybridHindcastTiming(FrozenModel):
    """Non-scientific execution timing kept outside deterministic hindcast results."""

    hindcast_id: Identifier
    eulerian_runtime_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    high_fidelity_runtime_seconds: float = Field(ge=0.0, allow_inf_nan=False)


class HybridScenarioWeight(FrozenModel):
    scenario_id: Identifier
    normalized_weight: Probability
    design_weight: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    represented_scenario_ids: tuple[Identifier, ...] = Field(min_length=1)
    selection_reason: Literal["medoid", "high_risk_extreme", "divergent_extreme"]
    cluster_id: Identifier | None = None


class HybridForecastScenarioEvaluation(FrozenModel):
    scenario: EulerianScenario
    eulerian_result_id: Identifier
    feature_names: tuple[Identifier, ...] = Field(min_length=1)
    feature_vector: tuple[float, ...] = Field(min_length=1)
    cluster_id: Identifier
    selected: bool
    selection_reasons: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _feature_alignment(self) -> HybridForecastScenarioEvaluation:
        if len(self.feature_names) != len(self.feature_vector):
            raise ValueError("hybrid forecast feature names and values must align")
        return self


class HybridForecastBatchDiagnostics(FrozenModel):
    batch_number: int = Field(ge=0)
    scenario_count: int = Field(gt=0)
    cumulative_scenario_count: int = Field(gt=0)
    raster_l1_change: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    centroid_change_metres: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    support_area_change_m2: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    coastal_contact_weight_change: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    convergence_thresholds_met: bool = False


class HybridForecastDiagnostics(FrozenModel):
    initial_scenario_count: int = Field(ge=0)
    eulerian_screened_count: int = Field(ge=0)
    high_fidelity_scenario_count: int = Field(ge=0)
    scenario_weights: tuple[HybridScenarioWeight, ...]
    eulerian_result_ids: tuple[Identifier, ...] = ()
    convergence_status: HybridConvergenceStatus
    component: ComponentMetadata
    warnings: tuple[QualityFlag, ...] = ()
    scenario_evaluations: tuple[HybridForecastScenarioEvaluation, ...] = ()
    batches: tuple[HybridForecastBatchDiagnostics, ...] = ()
    budget_exhausted: bool = False
    semantics: Literal["ensemble frequency / design weight, not calibrated probability"] = (
        "ensemble frequency / design weight, not calibrated probability"
    )

    @model_validator(mode="after")
    def _consistent(self) -> HybridForecastDiagnostics:
        if self.eulerian_screened_count > self.initial_scenario_count:
            raise ValueError("screened forecast scenarios cannot exceed initial scenarios")
        if self.high_fidelity_scenario_count != len(self.scenario_weights):
            raise ValueError("high-fidelity scenario count must match scenario weights")
        if self.high_fidelity_scenario_count > self.eulerian_screened_count:
            raise ValueError("high-fidelity scenarios cannot exceed screened scenarios")
        if self.eulerian_result_ids and len(self.eulerian_result_ids) != (
            self.eulerian_screened_count
        ):
            raise ValueError("Eulerian result IDs must cover every screened scenario")
        if self.scenario_evaluations and len(self.scenario_evaluations) != (
            self.eulerian_screened_count
        ):
            raise ValueError("scenario evaluations must cover every screened scenario")
        if self.scenario_weights:
            total = sum(item.normalized_weight for item in self.scenario_weights)
            if abs(total - 1.0) > 1e-9:
                raise ValueError("hybrid forecast scenario weights must sum to one")
        ids = tuple(item.scenario_id for item in self.scenario_weights)
        if len(ids) != len(set(ids)):
            raise ValueError("hybrid forecast scenario IDs must be unique")
        if self.scenario_evaluations:
            selected_ids = {
                item.scenario.scenario_id
                for item in self.scenario_evaluations
                if item.selected
            }
            if selected_ids != set(ids):
                raise ValueError("scenario weights must match selected scenario evaluations")
        return self
