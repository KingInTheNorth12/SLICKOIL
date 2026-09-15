"""Sequential physics-first coarse hindcast composed through the Eulerian port."""

from __future__ import annotations

import json
import math
from hashlib import sha256

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import Field

from oilspill.artifacts import local_path_from_artifact, write_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
)
from oilspill.domain.eulerian import (
    EulerianGrid,
    EulerianScenario,
    EulerianTransportResult,
    PointSourceInitialCondition,
)
from oilspill.domain.hybrid import (
    CoarseHindcastDiagnostics,
    CoarseHybridHindcastResult,
    CoarseSourceEvaluation,
    HybridSourceState,
    ReachabilityRejection,
)
from oilspill.eulerian.grid import EulerianGridBuilder, EulerianGridBuildResult
from oilspill.eulerian.initialization import SpillRasterInitializer, validate_raster
from oilspill.hybrid.config import HybridHindcastConfig
from oilspill.hybrid.reachability import assess_reachability, build_reachability_envelope
from oilspill.hybrid.sampling import sample_source_states
from oilspill.hybrid.scoring import evaluate_source, field_mismatch, select_evaluations
from oilspill.ports import EulerianTransportEngine
from oilspill.requests import EulerianTransportRequest, HybridHindcastRequest

FloatArray = NDArray[np.float64]


class CoarseHindcastManifest(FrozenModel):
    observation_id: str
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: HybridHindcastConfig
    forcing_field_ids: tuple[str, ...]
    forcing_artifacts: tuple[ArtifactRef, ...]
    observation_raster: ArtifactRef
    land_source: ArtifactRef | None
    sampled_source_state_ids: tuple[str, ...]
    rejected_source_state_ids: tuple[str, ...]
    eulerian_result_ids: tuple[str, ...]
    retained_source_state_ids: tuple[str, ...]
    metric_evaluations: tuple[CoarseSourceEvaluation, ...]
    semantics: str


class PhysicsFirstCoarseHindcast:
    """AIS-independent coarse screening; high-fidelity refinement is a later stage."""

    def __init__(
        self,
        config: HybridHindcastConfig,
        eulerian: EulerianTransportEngine,
    ) -> None:
        self._config = config
        self._eulerian = eulerian

    def run(self, request: HybridHindcastRequest) -> CoarseHybridHindcastResult:
        grid_result = self._build_grid(request)
        grid = grid_result.grid
        observed_initial = SpillRasterInitializer(
            self._config.initialization
        ).from_observation(request.observation, grid)
        observed = _read_field(observed_initial.concentration.artifact, grid)
        envelope = build_reachability_envelope(request.observation, grid, self._config)
        effective_seed = self._config.sobol_seed ^ request.random_seed
        sampled = sample_source_states(envelope, self._config, seed=effective_seed)

        rejections: list[ReachabilityRejection] = []
        valid: list[tuple[HybridSourceState, float]] = []
        for state in sampled:
            assessment = assess_reachability(state, grid, envelope)
            if assessment.accepted:
                valid.append((state, assessment.physics_penalty))
            else:
                rejections.append(
                    ReachabilityRejection(source_state=state, reasons=assessment.reasons)
                )

        evaluations: list[CoarseSourceEvaluation] = []
        transport_results: list[EulerianTransportResult] = []
        warnings = list(grid_result.warnings)
        forcing_ids = tuple(field.field_id for field in request.forcing)
        for state, physics_penalty in valid:
            scenario = EulerianScenario(
                scenario_id=f"coarse:{state.source_state_id}",
                windage_coefficient=state.windage_coefficient,
                horizontal_diffusivity_m2_s=state.horizontal_diffusivity_m2_s,
                first_order_loss_rate_s=state.first_order_loss_rate_s,
                forcing_field_ids=forcing_ids,
                metadata=(MetadataEntry(key="stage", value="coarse_hindcast"),),
                validation_assumptions=(
                    "Relative surface transport used only for coarse source screening.",
                    self._config.validation_note,
                ),
            )
            transport = self._eulerian.run(
                EulerianTransportRequest(
                    grid=grid,
                    scenario=scenario,
                    initial_condition=PointSourceInitialCondition(
                        geometry=state.geometry,
                        release_time=state.release_time,
                    ),
                    forcing=request.forcing,
                    snapshot_times=(request.observation.observed_at,),
                )
            )
            predicted = _read_field(transport.snapshots[-1].concentration.artifact, grid)
            mismatch = field_mismatch(
                predicted,
                observed,
                grid,
                support_threshold=self._config.observation_support_threshold,
            )
            evaluations.append(
                evaluate_source(
                    state,
                    transport.result_id,
                    mismatch,
                    physics_penalty,
                    self._config.mismatch_weights,
                )
            )
            transport_results.append(transport)
            warnings.extend(transport.warnings)

        selected, threshold_count = select_evaluations(
            tuple(evaluations),
            top_k=self._config.top_k,
            absolute_score_threshold=self._config.absolute_score_threshold,
        )
        low_confidence = threshold_count == 0
        if low_confidence:
            warnings.append(
                QualityFlag(
                    code="NO_ABSOLUTE_COARSE_MATCH",
                    severity=QualitySeverity.WARNING,
                    message=(
                        "No source state met the configured absolute mismatch threshold; "
                        "top-K is retained only for diagnostic/high-fidelity investigation"
                    ),
                )
            )
        manifest = self._write_manifest(
            request, sampled, tuple(rejections), selected, tuple(transport_results)
        )
        identity = sha256(manifest.model_dump_json().encode()).hexdigest()
        return CoarseHybridHindcastResult(
            hindcast_id=f"coarse-hindcast:{identity[:24]}",
            observation_id=request.observation.observation_id,
            initial_candidate_count=len(sampled),
            reachable_candidate_count=len(selected),
            retained_candidate_count=sum(item.retained for item in selected),
            sampled_source_states=sampled,
            reachability_rejections=tuple(rejections),
            coarse_evaluations=selected,
            eulerian_results=tuple(transport_results),
            analysis_grid=grid,
            diagnostics=CoarseHindcastDiagnostics(
                valid_candidate_count=len(selected),
                rejected_candidate_count=len(rejections),
                absolute_threshold_pass_count=threshold_count,
                low_confidence=low_confidence,
                converged=not low_confidence,
                status=(
                    "low_confidence_no_absolute_match"
                    if low_confidence
                    else "screening_supported"
                ),
                notes=(
                    "Coarse screening support is not a calibrated source probability.",
                    self._config.validation_note,
                ),
            ),
            component=self.component_metadata(),
            provenance=manifest,
            warnings=_unique_warnings(warnings),
        )

    def _build_grid(self, request: HybridHindcastRequest) -> EulerianGridBuildResult:
        maximum_speed = self._config.conservative_maximum_current_speed_m_s + (
            self._config.conservative_maximum_windage
            * self._config.conservative_maximum_wind_speed_m_s
        )
        diffusion = self._config.diffusion_reach_sigma_multiplier * math.sqrt(
            2.0
            * self._config.conservative_maximum_diffusivity_m2_s
            * self._config.backward_horizon_seconds
        )
        required_padding = (
            maximum_speed * self._config.backward_horizon_seconds
            + diffusion
            + self._config.source_roi_padding_metres
        )
        grid_config = self._config.grid.model_copy(
            update={
                "roi_padding_metres": max(
                    self._config.grid.roi_padding_metres, required_padding
                )
            }
        )
        return EulerianGridBuilder(grid_config).build(
            request.observation.geometry.polygon,
            land_geometries=request.land_geometries,
            land_source=request.land_source,
        )

    def _write_manifest(
        self,
        request: HybridHindcastRequest,
        sampled: tuple[HybridSourceState, ...],
        rejections: tuple[ReachabilityRejection, ...],
        evaluations: tuple[CoarseSourceEvaluation, ...],
        results: tuple[EulerianTransportResult, ...],
    ) -> ArtifactRef:
        configuration_sha = sha256(self._config.model_dump_json().encode()).hexdigest()
        manifest = CoarseHindcastManifest(
            observation_id=request.observation.observation_id,
            configuration_sha256=configuration_sha,
            configuration=self._config,
            forcing_field_ids=tuple(field.field_id for field in request.forcing),
            forcing_artifacts=tuple(field.values for field in request.forcing),
            observation_raster=(
                request.observation.detection.probability_raster
                or request.observation.detection.mask
            ).artifact,
            land_source=request.land_source,
            sampled_source_state_ids=tuple(item.source_state_id for item in sampled),
            rejected_source_state_ids=tuple(
                item.source_state.source_state_id for item in rejections
            ),
            eulerian_result_ids=tuple(item.result_id for item in results),
            retained_source_state_ids=tuple(
                item.source_state.source_state_id for item in evaluations if item.retained
            ),
            metric_evaluations=evaluations,
            semantics="weighted screening mismatch support; not proof of origin",
        )
        content = json.dumps(
            manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        identity = sha256(content).hexdigest()
        path = self._config.output_root / identity[:16] / "coarse-hindcast.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("deterministic hindcast provenance path contains different data")
            from oilspill.eulerian.initialization import file_artifact

            return file_artifact(path, "application/json")
        return write_artifact(path, content)

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="physics_first",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="scipy-sobol-eulerian-port",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="ais_dependency", value=False),
                MetadataEntry(key="high_fidelity_drift", value=False),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
        )


def _read_field(artifact: ArtifactRef, grid: EulerianGrid) -> FloatArray:
    path = local_path_from_artifact(artifact)
    with rasterio.open(path) as dataset:
        validate_raster(dataset, grid, "coarse hindcast concentration")
        values = np.asarray(dataset.read(1), dtype=np.float64)
        if dataset.nodata is not None:
            values[values == dataset.nodata] = 0.0
    return np.maximum(np.nan_to_num(values, nan=0.0), 0.0)


def _unique_warnings(warnings: list[QualityFlag]) -> tuple[QualityFlag, ...]:
    unique = {warning.model_dump_json(): warning for warning in warnings}
    return tuple(unique[key] for key in sorted(unique))


def create_physics_first_coarse(
    config: ComponentConfig,
    eulerian: EulerianTransportEngine,
) -> PhysicsFirstCoarseHindcast:
    return PhysicsFirstCoarseHindcast(
        HybridHindcastConfig.model_validate(config.settings), eulerian
    )
