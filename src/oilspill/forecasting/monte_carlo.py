"""Protocol-driven execution/replay with append-only plans, outcomes and occupancy."""

from hashlib import sha256
from uuid import uuid4

from oilspill.artifacts import read_artifact, write_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.common import ArtifactRef, ComponentMetadata, QualityFlag, QualitySeverity
from oilspill.domain.drift import DriftSimulation, SimulationStatus
from oilspill.domain.ensemble import EnsembleMemberOutcome
from oilspill.domain.forecast import (
    ForecastHorizon,
    ForecastResult,
    ForecastRunProvenance,
    UncertaintySummary,
)
from oilspill.domain.geospatial import SpatialUnit
from oilspill.forecasting.monte_carlo_config import MonteCarloConfig
from oilspill.forecasting.occupancy import aggregate_occupancy
from oilspill.forecasting.planning import EnsemblePlan, plan_ensemble
from oilspill.ports import DriftModel
from oilspill.requests import EnsembleForecastRequest


class EnsembleExecutionError(RuntimeError):
    def __init__(self, result: ForecastResult, artifact: ArtifactRef) -> None:
        super().__init__(f"Ensemble member failures recorded at {artifact.uri}")
        self.result = result
        self.artifact = artifact


class MonteCarloForecaster:
    def __init__(self, config: MonteCarloConfig, drift_model: DriftModel) -> None:
        self.config = config
        self.drift_model = drift_model

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="monte_carlo",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="generic-drift-ensemble",
            configuration_sha256=sha256(self.config.model_dump_json().encode()).hexdigest(),
        )

    def plan(self, request: EnsembleForecastRequest) -> EnsemblePlan:
        return plan_ensemble(self.config, request, self.drift_model.component_metadata())

    def forecast(self, request: EnsembleForecastRequest) -> ForecastResult:
        return self.execute(self.plan(request))

    def replay(self, artifact: ArtifactRef) -> ForecastResult:
        return self.execute(EnsemblePlan.model_validate_json(read_artifact(artifact)))

    def execute(self, plan: EnsemblePlan) -> ForecastResult:
        if plan.drift != self.drift_model.component_metadata():
            raise ValueError("replay drift version/configuration differs from saved plan")
        directory = self.config.output_root / plan.plan_id / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        saved_plan = write_artifact(directory / "plan.json", plan.model_dump_json().encode())
        simulations: list[DriftSimulation] = []
        outcomes = []
        provenance = []
        for member in plan.members:
            try:
                for field in member.request.forcing:
                    if field.values.uri.startswith("file:"):
                        read_artifact(field.values)
                simulation = self.drift_model.forecast(member.request)
                distribution = simulation.target_distribution
                if (
                    simulation.status != SimulationStatus.SUCCEEDED
                    or distribution is None
                    or distribution.particle_count == 0
                    or simulation.target_timestamp != member.request.valid_until
                    or simulation.random_seed != member.engine_seed
                    or simulation.release != member.request.release
                    or simulation.forcing_ids != tuple(f.field_id for f in member.request.forcing)
                    or simulation.model != plan.drift
                    or simulation.configuration_sha256 != plan.drift.configuration_sha256
                ):
                    raise ValueError("drift result is not a valid matching ensemble member")
                if any(s.simulation_id == simulation.simulation_id for s in simulations):
                    raise ValueError("duplicate member simulation identifier")
                # Validate support before admitting it into the occupancy denominator.
                aggregate_occupancy(plan, (simulation,))
                if simulation.trajectories.uri.startswith("file:"):
                    original = simulation.trajectories
                    snapshot = write_artifact(
                        directory / f"member-{member.member_index}.trajectories",
                        read_artifact(original),
                        original.media_type,
                    )
                    distribution = (
                        distribution.model_copy(update={"positions": snapshot})
                        if distribution.positions == original
                        else distribution
                    )
                    simulation = simulation.model_copy(
                        update={"trajectories": snapshot, "target_distribution": distribution}
                    )
                record = write_artifact(
                    directory / f"member-{member.member_index}.json",
                    simulation.model_dump_json().encode(),
                )
                outcome = EnsembleMemberOutcome(
                    member_id=member.member_id,
                    member_index=member.member_index,
                    engine_seed=member.engine_seed,
                    status="succeeded",
                    simulation_id=simulation.simulation_id,
                    simulation_record=record,
                )
                provenance.append(
                    ForecastRunProvenance(
                        member_id=member.member_id,
                        horizon=member.request.valid_until,
                        simulation_id=simulation.simulation_id,
                        random_seed=member.engine_seed,
                        forcing_ids=simulation.forcing_ids,
                        position_offset=member.position_offset_metres,
                        position_offset_unit=SpatialUnit.METRE,
                        time_offset_seconds=member.time_offset_seconds,
                        model_parameters=member.request.model_parameters,
                        configured_uncertainty=member.request.configured_uncertainty,
                        validation_note=plan.config.validation_note,
                        trajectory_artifact=simulation.trajectories,
                    )
                )
                simulations.append(simulation)
            except Exception as error:
                outcome = EnsembleMemberOutcome(
                    member_id=member.member_id,
                    member_index=member.member_index,
                    engine_seed=member.engine_seed,
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            outcomes.append(outcome)
            write_artifact(
                directory / f"outcome-{member.member_index}.json",
                outcome.model_dump_json().encode(),
            )
        summary, content = aggregate_occupancy(plan, tuple(simulations))
        occupancy = write_artifact(directory / "occupancy.json", content)
        result = ForecastResult(
            forecast_id=f"ensemble:{plan.plan_id}:{directory.name}",
            observation_id=plan.root.observation.observation_id,
            issued_at=plan.root.observation.observed_at,
            ensemble_plan=saved_plan,
            member_outcomes=tuple(outcomes),
            members=tuple(simulations),
            horizons=(
                ForecastHorizon(
                    valid_at=plan.members[0].request.valid_until,
                    particle_ensemble=tuple(
                        s.target_distribution
                        for s in simulations
                        if s.target_distribution is not None
                    ),
                    member_simulation_ids=tuple(s.simulation_id for s in simulations),
                    occupancy_summary=summary,
                    occupancy_artifact=occupancy,
                ),
            ),
            uncertainty=UncertaintySummary(
                method="ensemble occupancy frequency",
                ensemble_member_count=len(simulations),
                calibrated=False,
                random_seed=plan.root.random_seed,
                notes=summary.warnings,
            ),
            run_provenance=tuple(provenance),
            warnings=(
                QualityFlag(
                    code="ENSEMBLE_MEMBER_FAILURES",
                    severity=QualitySeverity.WARNING,
                    message=(
                        f"{summary.failed} of {summary.requested} members failed; "
                        "see member outcomes"
                    ),
                ),
            )
            if summary.failed
            else (),
        )
        result_artifact = write_artifact(
            directory / "forecast.json", result.model_dump_json().encode()
        )
        if summary.failed and plan.config.failure_policy == "raise_after_recording":
            raise EnsembleExecutionError(result, result_artifact)
        return result


def create_monte_carlo(config: ComponentConfig, drift_model: DriftModel) -> MonteCarloForecaster:
    return MonteCarloForecaster(MonteCarloConfig.model_validate(config.settings), drift_model)
