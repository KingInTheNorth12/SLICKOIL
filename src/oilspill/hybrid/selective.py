"""Selective high-fidelity hybrid hindcast using only stable component ports."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np
import rasterio
from numpy.typing import NDArray

from oilspill.artifacts import local_path_from_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
)
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.hybrid import (
    HybridConvergenceDiagnostics,
    HybridConvergenceStatus,
    HybridHindcastResult,
    HybridHindcastTiming,
)
from oilspill.eulerian.initialization import SpillRasterInitializer, validate_raster
from oilspill.hybrid.config import SelectiveRefinementConfig
from oilspill.hybrid.high_fidelity import RawHighFidelityEvaluation, evaluate_high_fidelity_state
from oilspill.hybrid.posterior import (
    build_source_posterior,
    weighted_release_quantiles,
    weighted_source_centroid,
)
from oilspill.hybrid.refinement import generate_children, normalized_support_weights
from oilspill.ports import CoarseHindcastEngine, DriftModel
from oilspill.registry import HybridHindcastDependencies
from oilspill.requests import HybridHindcastRequest

FloatArray = NDArray[np.float64]


class SelectiveHybridHindcast:
    def __init__(
        self,
        config: SelectiveRefinementConfig,
        coarse: CoarseHindcastEngine,
        drift: DriftModel,
    ) -> None:
        self._config = config
        self._coarse = coarse
        self._drift = drift
        self._last_timing: HybridHindcastTiming | None = None

    def run(self, request: HybridHindcastRequest) -> HybridHindcastResult:
        coarse_started = perf_counter()
        coarse = self._coarse.run(request)
        eulerian_runtime = perf_counter() - coarse_started
        retained = sorted(
            (item for item in coarse.coarse_evaluations if item.retained),
            key=lambda item: (item.total_mismatch_score, item.source_state.source_state_id),
        )
        if not retained:
            raise ValueError("high-fidelity refinement requires retained Eulerian source states")
        domain_grid = coarse.analysis_grid
        if domain_grid is None:
            raise ValueError("coarse hindcast result must preserve its Eulerian analysis grid")
        initial = SpillRasterInitializer(self._config.initialization).from_observation(
            request.observation, domain_grid
        )
        observed = _read_field(initial.concentration.artifact, domain_grid)
        evaluations: list[RawHighFidelityEvaluation] = []
        high_fidelity_runtime = 0.0
        warnings = list(coarse.warnings)
        initial_count = min(len(retained), self._config.maximum_high_fidelity_evaluations)
        for index, item in enumerate(retained[:initial_count]):
            evaluation_started = perf_counter()
            evaluation = evaluate_high_fidelity_state(
                item.source_state,
                0,
                request,
                domain_grid,
                observed,
                self._config,
                self._drift,
                random_seed=_engine_seed(
                    request.random_seed,
                    0,
                    index,
                    item.source_state.source_state_id,
                ),
            )
            high_fidelity_runtime += perf_counter() - evaluation_started
            evaluations.append(evaluation)
        best_scores = [min(item.mismatch_score for item in evaluations)]
        previous_summary = _summary(tuple(evaluations), self._config.initial_temperature)
        converged = False
        executed_refinements = 0
        for generation in range(1, self._config.maximum_generations + 1):
            remaining = self._config.maximum_high_fidelity_evaluations - len(evaluations)
            if remaining <= 0:
                break
            child_count = min(self._config.children_per_generation, remaining)
            child_generation = generate_children(
                tuple(evaluations),
                generation,
                child_count,
                request,
                domain_grid,
                self._config,
                random_seed=_engine_seed(request.random_seed, generation, 0, "children"),
            )
            if not child_generation.states:
                break
            for index, state in enumerate(child_generation.states):
                evaluation_started = perf_counter()
                evaluation = evaluate_high_fidelity_state(
                    state,
                    generation,
                    request,
                    domain_grid,
                    observed,
                    self._config,
                    self._drift,
                    random_seed=_engine_seed(
                        request.random_seed, generation, index, state.source_state_id
                    ),
                )
                high_fidelity_runtime += perf_counter() - evaluation_started
                evaluations.append(evaluation)
            executed_refinements = generation
            best_scores.append(min(item.mismatch_score for item in evaluations))
            temperature = self._config.initial_temperature * (
                self._config.temperature_decay**generation
            )
            current_summary = _summary(tuple(evaluations), temperature)
            improvement = previous_summary.best_score - current_summary.best_score
            centroid_movement = float(
                np.linalg.norm(np.subtract(current_summary.centroid, previous_summary.centroid))
            )
            quantile_change = max(
                abs(current - previous)
                for current, previous in zip(
                    current_summary.quantile_seconds,
                    previous_summary.quantile_seconds,
                    strict=True,
                )
            )
            converged = (
                improvement <= self._config.best_score_improvement_tolerance
                and centroid_movement <= self._config.source_centroid_movement_tolerance_metres
                and quantile_change <= self._config.release_time_quantile_change_tolerance_seconds
            )
            previous_summary = current_summary
            if converged:
                break

        if converged:
            status = HybridConvergenceStatus.CONVERGED
            termination = "configured_convergence_tolerances_met"
        elif len(evaluations) >= self._config.maximum_high_fidelity_evaluations:
            status = HybridConvergenceStatus.BUDGET_EXHAUSTED
            termination = "maximum_high_fidelity_evaluations_reached"
            warnings.append(
                QualityFlag(
                    code="HIGH_FIDELITY_BUDGET_EXHAUSTED",
                    severity=QualitySeverity.WARNING,
                    message=(
                        "Posterior returned before convergence because the high-fidelity "
                        "evaluation budget was exhausted"
                    ),
                )
            )
        else:
            status = HybridConvergenceStatus.PARTIAL
            termination = "maximum_refinement_generations_reached"
            warnings.append(
                QualityFlag(
                    code="HIGH_FIDELITY_GENERATIONS_EXHAUSTED",
                    severity=QualitySeverity.WARNING,
                    message=(
                        "Posterior returned without satisfying configured convergence tolerances"
                    ),
                )
            )
        warnings_tuple = _unique_warnings(warnings)
        final_temperature = self._config.initial_temperature * (
            self._config.temperature_decay**executed_refinements
        )
        posterior, weighted_evaluations = build_source_posterior(
            request.observation.observation_id,
            tuple(evaluations),
            domain_grid,
            self._config,
            status,
            warnings_tuple,
            temperature=final_temperature,
        )
        identity = sha256(
            (coarse.hindcast_id + posterior.posterior_id + self._config.model_dump_json()).encode()
        ).hexdigest()
        result = HybridHindcastResult(
            hindcast_id=f"hybrid-hindcast:{identity[:24]}",
            observation_id=request.observation.observation_id,
            initial_candidate_count=coarse.initial_candidate_count,
            retained_eulerian_count=len(retained),
            high_fidelity_count=len(weighted_evaluations),
            coarse_evaluations=coarse.coarse_evaluations,
            high_fidelity_evaluations=weighted_evaluations,
            posterior=posterior,
            convergence_diagnostics=HybridConvergenceDiagnostics(
                status=status,
                converged=converged,
                generation_count=executed_refinements + 1,
                evaluated_state_count=len(weighted_evaluations),
                retained_state_count=len(weighted_evaluations),
                termination_reason=termination,
                best_mismatch_by_generation=tuple(best_scores),
            ),
            component=self.component_metadata(),
            warnings=warnings_tuple,
        )
        self._last_timing = HybridHindcastTiming(
            hindcast_id=result.hindcast_id,
            eulerian_runtime_seconds=eulerian_runtime,
            high_fidelity_runtime_seconds=high_fidelity_runtime,
        )
        return result

    def last_run_timing(self) -> HybridHindcastTiming | None:
        return self._last_timing

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="selective_refinement",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="drift-model-port",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="forward_only", value=True),
                MetadataEntry(key="calibrated_posterior", value=False),
                MetadataEntry(key="drift_component", value=self._drift.component_metadata().name),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
        )


class _Summary:
    def __init__(
        self,
        best_score: float,
        centroid: tuple[float, float],
        quantile_seconds: tuple[float, float, float],
    ) -> None:
        self.best_score = best_score
        self.centroid = centroid
        self.quantile_seconds = quantile_seconds


def _summary(evaluations: tuple[RawHighFidelityEvaluation, ...], temperature: float) -> _Summary:
    weights = normalized_support_weights(
        np.asarray([item.mismatch_score for item in evaluations]), temperature
    )
    centroid = weighted_source_centroid(evaluations, weights).geometry
    assert centroid.type == "Point"
    quantiles = weighted_release_quantiles(evaluations, weights)
    return _Summary(
        best_score=min(item.mismatch_score for item in evaluations),
        centroid=(centroid.coordinate.x, centroid.coordinate.y),
        quantile_seconds=(
            quantiles.p10.timestamp(),
            quantiles.p50.timestamp(),
            quantiles.p90.timestamp(),
        ),
    )


def _read_field(artifact: ArtifactRef, grid: EulerianGrid) -> FloatArray:
    path = local_path_from_artifact(artifact)
    with rasterio.open(path) as dataset:
        validate_raster(dataset, grid, "high-fidelity observed concentration")
        values = np.asarray(dataset.read(1), dtype=np.float64)
        if dataset.nodata is not None:
            values[values == dataset.nodata] = 0.0
    return np.maximum(np.nan_to_num(values, nan=0.0), 0.0)


def _engine_seed(root: int, generation: int, index: int, state_id: str) -> int:
    digest = sha256(f"{root}:{generation}:{index}:{state_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _unique_warnings(warnings: list[QualityFlag]) -> tuple[QualityFlag, ...]:
    unique = {item.model_dump_json(): item for item in warnings}
    return tuple(unique[key] for key in sorted(unique))


def create_selective_hybrid(
    config: ComponentConfig,
    dependencies: HybridHindcastDependencies,
) -> SelectiveHybridHindcast:
    if dependencies.coarse is None:
        raise ValueError("selective hybrid hindcast requires a coarse hindcast dependency")
    return SelectiveHybridHindcast(
        SelectiveRefinementConfig.model_validate(config.settings),
        dependencies.coarse,
        dependencies.drift,
    )
