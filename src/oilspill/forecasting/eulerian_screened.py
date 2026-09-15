"""Sobol Eulerian screening followed by weighted high-fidelity forecasts."""

from __future__ import annotations

import warnings
from datetime import timedelta
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc  # type: ignore[import-untyped]

from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ComponentMetadata,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
)
from oilspill.domain.ensemble import EnsembleMemberOutcome
from oilspill.domain.eulerian import EulerianScenario
from oilspill.domain.forecast import (
    ForecastDecision,
    ForecastHorizon,
    ForecastResult,
    ForecastRunProvenance,
    SpatialProbabilityRepresentation,
    UncertaintySummary,
)
from oilspill.domain.geospatial import Coordinate, SpatialUnit
from oilspill.domain.hybrid import (
    HybridConvergenceStatus,
    HybridForecastBatchDiagnostics,
    HybridForecastDiagnostics,
    HybridForecastScenarioEvaluation,
    HybridScenarioWeight,
)
from oilspill.eulerian.grid import EulerianGridBuilder
from oilspill.eulerian.initialization import SpillRasterInitializer
from oilspill.forecasting.eulerian_screened_config import (
    EulerianScreenedForecastConfig,
    ForecastParameterRange,
    _seconds,
)
from oilspill.forecasting.features import CoarseOutcomeFeatures, extract_outcome_features
from oilspill.forecasting.selection import RepresentativeSelection, select_representatives
from oilspill.ports import DriftModel, EulerianTransportEngine
from oilspill.registry import HybridForecastDependencies
from oilspill.requests import (
    DriftForecastRequest,
    EnsembleForecastRequest,
    EulerianTransportRequest,
)

FloatArray = NDArray[np.float64]


class EulerianScreenedForecaster:
    """Screen uncertainty cheaply, then run only weighted representatives at high fidelity."""

    def __init__(
        self,
        config: EulerianScreenedForecastConfig,
        eulerian: EulerianTransportEngine,
        drift: DriftModel,
    ) -> None:
        self._config = config
        self._eulerian = eulerian
        self._drift = drift

    def forecast(self, request: EnsembleForecastRequest) -> ForecastResult:
        horizons = request.horizons or tuple(
            request.observation.observed_at + timedelta(seconds=_seconds(offset))
            for offset in self._config.horizons
        )
        if horizons != tuple(sorted(horizons)) or len(horizons) != len(set(horizons)):
            raise ValueError("hybrid forecast horizons must be unique and chronological")
        if any(item <= request.observation.observed_at for item in horizons):
            raise ValueError("hybrid forecast horizons must follow the observation")

        grid_result = EulerianGridBuilder(self._config.grid).build(
            request.observation.geometry.polygon,
            land_geometries=request.coastline_geometries,
            land_source=request.coastline_source,
        )
        grid = grid_result.grid
        initial = SpillRasterInitializer(self._config.initialization).from_observation(
            request.observation, grid
        )
        sampler = qmc.Sobol(d=6, scramble=True, seed=request.random_seed)
        scenarios: list[EulerianScenario] = []
        results = []
        features: list[CoarseOutcomeFeatures] = []
        batches: list[HybridForecastBatchDiagnostics] = []
        previous: tuple[FloatArray, tuple[float, float], float, float] | None = None
        consecutive = 0
        converged = False
        budget_exhausted = False
        for batch in range(self._config.maximum_batches):
            requested_count = (
                self._config.initial_scenario_count
                if batch == 0
                else self._config.additional_scenarios_per_batch
            )
            remaining = self._config.maximum_coarse_scenarios - len(scenarios)
            count = min(requested_count, remaining)
            if count <= 0:
                budget_exhausted = True
                break
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The balance properties of Sobol' points"
                )
                points = sampler.random(count)
            start_index = len(scenarios)
            for offset, point in enumerate(points):
                scenario = self._scenario(point, start_index + offset, request)
                result = self._eulerian.run(
                    EulerianTransportRequest(
                        grid=grid,
                        scenario=scenario,
                        initial_condition=initial,
                        forcing=request.forcing,
                        snapshot_times=horizons,
                    )
                )
                outcome = extract_outcome_features(
                    result,
                    grid,
                    initial.concentration,
                    request.coastline_geometries,
                    support_threshold=self._config.observation_support_threshold,
                    anisotropy_threshold=(
                        self._config.dominant_direction_anisotropy_threshold
                    ),
                )
                scenarios.append(scenario)
                results.append(result)
                features.append(outcome)
            current = _aggregate(features)
            changes = _changes(previous, current)
            threshold_met = previous is not None and self._within_thresholds(changes)
            consecutive = consecutive + 1 if threshold_met else 0
            batches.append(
                HybridForecastBatchDiagnostics(
                    batch_number=batch,
                    scenario_count=count,
                    cumulative_scenario_count=len(scenarios),
                    raster_l1_change=changes[0],
                    centroid_change_metres=changes[1],
                    support_area_change_m2=changes[2],
                    coastal_contact_weight_change=changes[3],
                    convergence_thresholds_met=threshold_met,
                )
            )
            previous = current
            if consecutive >= self._config.consecutive_converged_batches:
                converged = True
                break
            if count < requested_count:
                budget_exhausted = True
                break
        if self._config.maximum_batches > 1 and not converged:
            budget_exhausted = True

        matrix = np.asarray([item.vector for item in features], dtype=np.float64)
        selection = select_representatives(
            matrix,
            np.asarray([item.displacement_metres for item in features]),
            np.asarray([item.maximum_spread_metres for item in features]),
            np.asarray([item.coastal_contact for item in features]),
            cluster_count=self._config.cluster_count,
            displacement_extremes=self._config.displacement_extreme_count,
            spread_extremes=self._config.spread_extreme_count,
            coastal_extremes=self._config.coastal_contact_extreme_count,
        )
        weights = self._weights(selection, scenarios)
        if len(weights) * len(horizons) > self._config.maximum_high_fidelity_evaluations:
            raise ValueError(
                "configured high-fidelity budget cannot cover selected scenarios and horizons"
            )
        attempts = {}
        outcomes = []
        failed_scenario_ids: set[str] = set()
        member_index = 0
        for weight in weights:
            scenario = scenarios[_scenario_index(weight.scenario_id)]
            parameters = self._parameters(scenario)
            for horizon in horizons:
                seed = _seed(request.random_seed, scenario.scenario_id, horizon.isoformat())
                outcome_id = f"{scenario.scenario_id}:{horizon.isoformat()}"
                try:
                    simulation = self._drift.forecast(
                        DriftForecastRequest(
                            observation=request.observation,
                            release=request.release,
                            forcing=request.forcing,
                            valid_until=horizon,
                            random_seed=seed,
                            initial_position=request.observation.geometry.centroid,
                            initial_timestamp=request.observation.observed_at,
                            model_parameters=parameters,
                        )
                    )
                    distribution = simulation.target_distribution
                    if distribution is None or simulation.target_timestamp != horizon:
                        raise ValueError(
                            "drift model must return the requested target distribution"
                        )
                except Exception as error:
                    failed_scenario_ids.add(scenario.scenario_id)
                    outcomes.append(
                        EnsembleMemberOutcome(
                            member_id=outcome_id,
                            member_index=member_index,
                            engine_seed=seed,
                            status="failed",
                            error_type=type(error).__name__,
                            error_message=str(error),
                            included_in_aggregation=False,
                        )
                    )
                else:
                    attempts[(scenario.scenario_id, horizon)] = (
                        simulation,
                        distribution,
                        seed,
                        parameters,
                    )
                    outcomes.append(
                        EnsembleMemberOutcome(
                            member_id=outcome_id,
                            member_index=member_index,
                            engine_seed=seed,
                            status="succeeded",
                            simulation_id=simulation.simulation_id,
                            simulation_record=simulation.trajectories,
                        )
                    )
                member_index += 1

        quality_warnings = list(grid_result.warnings)
        eligible = tuple(
            weight for weight in weights if weight.scenario_id not in failed_scenario_ids
        )
        eligible_mass = sum(weight.normalized_weight for weight in eligible)
        abstained = False
        if failed_scenario_ids:
            if (
                self._config.partial_high_fidelity_failure_policy == "renormalize_successful"
                and eligible
                and eligible_mass > 0
            ):
                weights = tuple(
                    weight.model_copy(
                        update={
                            "normalized_weight": weight.normalized_weight / eligible_mass
                        }
                    )
                    for weight in eligible
                )
                quality_warnings.append(
                    QualityFlag(
                        code="HYBRID_FORECAST_PARTIAL_RENORMALIZED",
                        severity=QualitySeverity.WARNING,
                        message=(
                            "One or more selected high-fidelity scenarios failed; only scenarios "
                            "successful at every horizon were aggregated and their design mass "
                            "was renormalized"
                        ),
                    )
                )
            else:
                weights = ()
                abstained = True
                quality_warnings.append(
                    QualityFlag(
                        code="HYBRID_FORECAST_PARTIAL_FAILURE_ABSTAINED",
                        severity=QualitySeverity.WARNING,
                        message=(
                            "Selected high-fidelity scenario failure caused forecast "
                            "aggregation to abstain under configured policy"
                        ),
                    )
                )

        included_simulation_ids = {
            attempts[(weight.scenario_id, horizon)][0].simulation_id
            for weight in weights
            for horizon in horizons
        }
        outcomes = [
            outcome.model_copy(
                update={
                    "included_in_aggregation": (
                        outcome.status == "succeeded"
                        and outcome.simulation_id in included_simulation_ids
                    )
                }
            )
            for outcome in outcomes
        ]
        simulations = []
        provenance = []
        horizon_results = []
        for horizon in horizons:
            horizon_simulations = []
            distributions = []
            horizon_weights = []
            for weight in weights:
                scenario = scenarios[_scenario_index(weight.scenario_id)]
                simulation, distribution, seed, parameters = attempts[
                    (scenario.scenario_id, horizon)
                ]
                simulations.append(simulation)
                horizon_simulations.append(simulation)
                distributions.append(distribution)
                horizon_weights.append(weight.normalized_weight)
                provenance.append(
                    ForecastRunProvenance(
                        member_id=scenario.scenario_id,
                        horizon=horizon,
                        simulation_id=simulation.simulation_id,
                        random_seed=seed,
                        forcing_ids=simulation.forcing_ids,
                        position_offset=Coordinate(x=0, y=0),
                        position_offset_unit=SpatialUnit.METRE,
                        time_offset_seconds=0,
                        model_parameters=parameters,
                        configured_uncertainty=simulation.configured_uncertainty,
                        validation_note=self._config.validation_note,
                        trajectory_artifact=simulation.trajectories,
                        scenario_id=scenario.scenario_id,
                        cluster_id=weight.cluster_id,
                        scenario_weight=weight.normalized_weight,
                        design_weight=weight.design_weight,
                    )
                )
            horizon_results.append(
                ForecastHorizon(
                    valid_at=horizon,
                    particle_ensemble=tuple(distributions),
                    spatial_probability=(
                        SpatialProbabilityRepresentation(
                            method="weighted_selected_scenario_support",
                            member_geometries=tuple(
                                item.comparison_geometry for item in distributions
                            ),
                            member_weights=tuple(horizon_weights),
                            calibrated=False,
                            explanation=(
                                "Ensemble frequency / design weight, not calibrated probability; "
                                "zero-weight extremes are retained for risk inspection without "
                                "double-counting cluster mass."
                            ),
                        )
                        if distributions
                        else None
                    ),
                    member_simulation_ids=tuple(
                        item.simulation_id for item in horizon_simulations
                    ),
                )
            )

        if self._config.maximum_batches == 1:
            status = HybridConvergenceStatus.NOT_ASSESSED
            quality_warnings.append(
                QualityFlag(
                    code="SINGLE_BATCH_CONVERGENCE_NOT_ASSESSED",
                    severity=QualitySeverity.INFO,
                    message=(
                        "One Sobol batch was configured; progressive convergence was not "
                        "assessed"
                    ),
                )
            )
        elif converged:
            status = HybridConvergenceStatus.CONVERGED
        else:
            status = HybridConvergenceStatus.BUDGET_EXHAUSTED
            quality_warnings.append(
                QualityFlag(
                    code="HYBRID_FORECAST_BUDGET_EXHAUSTED",
                    severity=QualitySeverity.WARNING,
                    message="Configured coarse batch/compute budget was reached before convergence",
                )
            )
        final_scenario_ids = {item.scenario_id for item in weights}
        evaluations = tuple(
            HybridForecastScenarioEvaluation(
                scenario=scenario,
                eulerian_result_id=result.result_id,
                feature_names=_feature_names(len(horizons)),
                feature_vector=feature.vector,
                cluster_id=f"cluster-{int(selection.labels[index]):04d}",
                selected=scenario.scenario_id in final_scenario_ids,
                selection_reasons=selection.reasons[index],
            )
            for index, (scenario, result, feature) in enumerate(
                zip(scenarios, results, features, strict=True)
            )
        )
        diagnostics = HybridForecastDiagnostics(
            initial_scenario_count=len(scenarios),
            eulerian_screened_count=len(results),
            high_fidelity_scenario_count=len(weights),
            scenario_weights=weights,
            eulerian_result_ids=tuple(item.result_id for item in results),
            convergence_status=status,
            component=self.component_metadata(),
            warnings=tuple(quality_warnings),
            scenario_evaluations=evaluations,
            batches=tuple(batches),
            budget_exhausted=budget_exhausted,
        )
        digest = sha256(
            (
                request.observation.observation_id
                + str(request.random_seed)
                + self._config.model_dump_json()
            ).encode()
        ).hexdigest()[:20]
        return ForecastResult(
            forecast_id=f"eulerian-screened:{digest}",
            observation_id=request.observation.observation_id,
            issued_at=request.observation.observed_at,
            horizons=tuple(horizon_results),
            members=tuple(simulations),
            member_outcomes=tuple(outcomes),
            uncertainty=UncertaintySummary(
                method="sobol_eulerian_screened_high_fidelity",
                ensemble_member_count=len(simulations),
                calibrated=False,
                random_seed=request.random_seed,
                configured_perturbations=(
                    MetadataEntry(key="coarse_scenario_count", value=len(scenarios)),
                    MetadataEntry(key="selected_scenario_count", value=len(weights)),
                    MetadataEntry(key="validation_note", value=self._config.validation_note),
                ),
                notes=(
                    "Ensemble frequency / design weight, not calibrated probability.",
                    "Extreme representatives have zero aggregation weight to avoid duplicate mass.",
                ),
            ),
            run_provenance=tuple(provenance),
            warnings=tuple(quality_warnings),
            hybrid_diagnostics=diagnostics,
            decision=(
                ForecastDecision.ABSTAINED
                if abstained
                else ForecastDecision.COMPLETED
            ),
            abstention_reason=(
                "High-fidelity scenario failures prevented aggregation under configured policy."
                if abstained
                else None
            ),
        )

    def _scenario(
        self, point: FloatArray, index: int, request: EnsembleForecastRequest
    ) -> EulerianScenario:
        values = tuple(
            _scale(float(value), bounds)
            for value, bounds in zip(
                point,
                (
                    self._config.wind_bias_u_m_s,
                    self._config.wind_bias_v_m_s,
                    self._config.current_bias_u_m_s,
                    self._config.current_bias_v_m_s,
                    self._config.windage,
                    self._config.diffusivity_m2_s,
                ),
                strict=True,
            )
        )
        identifier = sha256(
            f"{request.random_seed}:{index}:".encode() + np.asarray(values).tobytes()
        ).hexdigest()[:20]
        return EulerianScenario(
            scenario_id=f"sobol-{index:06d}-{identifier}",
            wind_bias_u_m_s=values[0],
            wind_bias_v_m_s=values[1],
            current_bias_u_m_s=values[2],
            current_bias_v_m_s=values[3],
            windage_coefficient=values[4],
            horizontal_diffusivity_m2_s=values[5],
            forcing_field_ids=tuple(item.field_id for item in request.forcing),
            weight=1.0,
            weight_kind="design",
            validation_assumptions=(self._config.validation_note,),
        )

    def _weights(
        self,
        selection: RepresentativeSelection,
        scenarios: list[EulerianScenario],
    ) -> tuple[HybridScenarioWeight, ...]:
        total = len(scenarios)
        medoid_set = set(selection.medoid_indices)
        output = []
        for index in selection.selected_indices:
            cluster = int(selection.labels[index])
            is_medoid = index in medoid_set
            cluster_members = tuple(
                scenario.scenario_id
                for scenario_index, scenario in enumerate(scenarios)
                if int(selection.labels[scenario_index]) == cluster
            )
            reasons = selection.reasons[index]
            output.append(
                HybridScenarioWeight(
                    scenario_id=scenarios[index].scenario_id,
                    normalized_weight=(len(cluster_members) / total if is_medoid else 0.0),
                    design_weight=1.0 / total,
                    represented_scenario_ids=(
                        cluster_members if is_medoid else (scenarios[index].scenario_id,)
                    ),
                    selection_reason=(
                        "medoid"
                        if is_medoid
                        else (
                            "divergent_extreme"
                            if "largest_displacement" in reasons
                            else "high_risk_extreme"
                        )
                    ),
                    cluster_id=f"cluster-{cluster:04d}",
                )
            )
        return tuple(output)

    def _parameters(self, scenario: EulerianScenario) -> tuple[MetadataEntry, ...]:
        return tuple(
            MetadataEntry(key=key, value=value)
            for key, value in (
                (self._config.wind_bias_u_parameter_key, scenario.wind_bias_u_m_s),
                (self._config.wind_bias_v_parameter_key, scenario.wind_bias_v_m_s),
                (self._config.current_bias_u_parameter_key, scenario.current_bias_u_m_s),
                (self._config.current_bias_v_parameter_key, scenario.current_bias_v_m_s),
                (self._config.windage_parameter_key, scenario.windage_coefficient),
                (
                    self._config.diffusivity_parameter_key,
                    scenario.horizontal_diffusivity_m2_s,
                ),
            )
        )

    def _within_thresholds(
        self, changes: tuple[float | None, float | None, float | None, float | None]
    ) -> bool:
        if any(value is None for value in changes):
            return False
        return (
            changes[0] <= self._config.raster_l1_tolerance  # type: ignore[operator]
            and changes[1] <= self._config.centroid_tolerance_metres  # type: ignore[operator]
            and changes[2] <= self._config.support_area_tolerance_m2  # type: ignore[operator]
            and changes[3] <= self._config.coastal_contact_weight_tolerance  # type: ignore[operator]
        )

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="eulerian_screened",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="numpy-scipy-generic-drift",
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="sampling", value="scipy.stats.qmc.Sobol"),
                MetadataEntry(key="calibrated_probability", value=False),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
        )


def _scale(value: float, bounds: ForecastParameterRange) -> float:
    return bounds.minimum + value * (bounds.maximum - bounds.minimum)


def _aggregate(
    features: list[CoarseOutcomeFeatures],
) -> tuple[FloatArray, tuple[float, float], float, float]:
    return (
        np.mean(np.stack([item.final_cell_mass for item in features]), axis=0),
        (
            float(np.mean([item.final_centroid[0] for item in features])),
            float(np.mean([item.final_centroid[1] for item in features])),
        ),
        float(np.mean([item.final_support_area_m2 for item in features])),
        float(np.mean([item.coastal_contact for item in features])),
    )


def _changes(
    previous: tuple[FloatArray, tuple[float, float], float, float] | None,
    current: tuple[FloatArray, tuple[float, float], float, float],
) -> tuple[float | None, float | None, float | None, float | None]:
    if previous is None:
        return None, None, None, None
    return (
        float(np.abs(current[0] - previous[0]).sum()),
        float(np.linalg.norm(np.subtract(current[1], previous[1]))),
        abs(current[2] - previous[2]),
        abs(current[3] - previous[3]),
    )


def _scenario_index(identifier: str) -> int:
    return int(identifier.split("-", maxsplit=2)[1])


def _seed(base: int, scenario_id: str, horizon: str) -> int:
    return int.from_bytes(
        sha256(f"{base}:{scenario_id}:{horizon}".encode()).digest()[:4], "big"
    )


def _feature_names(horizon_count: int) -> tuple[str, ...]:
    fields = (
        "centroid_x_metres",
        "centroid_y_metres",
        "covariance_xx_m2",
        "covariance_xy_m2",
        "covariance_yy_m2",
        "support_area_m2",
        "dominant_direction_cosine",
        "dominant_direction_sine",
        "dominant_direction_stable",
        "coastal_contact",
    )
    return tuple(
        f"horizon_{index:02d}_{field}"
        for index in range(horizon_count)
        for field in fields
    )


def create_eulerian_screened(
    config: ComponentConfig, dependencies: HybridForecastDependencies
) -> EulerianScreenedForecaster:
    return EulerianScreenedForecaster(
        EulerianScreenedForecastConfig.model_validate(config.settings),
        dependencies.eulerian,
        dependencies.drift,
    )
