"""Backward source tracing through the generic drift-model boundary."""

from __future__ import annotations

from hashlib import sha256

from oilspill.domain.drift import DriftMode, ParticleTrajectory, SimulationStatus
from oilspill.domain.geospatial import LineStringGeometry, SpatialGeometry
from oilspill.domain.source_trace import BackwardSourceTraceResult
from oilspill.ports import DriftModel
from oilspill.requests import BackwardTraceRequest

BACKWARD_MODEL_DISCLAIMER = (
    "Backward trajectories are model-dependent hypotheses and are not physically exact "
    "reconstructions; this evidence is not source probability or confirmed origin."
)


class BackwardSourceTracer:
    """Produce source hypotheses without coupling tracing to ranking or a drift engine."""

    def __init__(self, drift_model: DriftModel) -> None:
        self._drift_model = drift_model

    def trace(self, request: BackwardTraceRequest) -> BackwardSourceTraceResult:
        simulation = self._drift_model.backward_source_trace(request)
        if simulation.mode != DriftMode.BACKWARD_TRACE:
            raise ValueError("drift model returned a non-backward simulation")
        if simulation.status != SimulationStatus.SUCCEEDED:
            raise ValueError("backward source trace requires a successful simulation")
        if simulation.target_timestamp is None or not (
            request.release.interval.start
            <= simulation.target_timestamp
            <= request.release.interval.end
        ):
            raise ValueError("backward simulation target must lie in the source-time window")
        distribution = simulation.target_distribution
        if distribution is None:
            raise ValueError("drift model omitted the plausible source distribution")
        if not simulation.particle_trajectories:
            raise ValueError("drift model omitted canonical backward particle trajectories")
        paths = tuple(self._path(trajectory) for trajectory in simulation.particle_trajectories)
        trace_key = sha256(
            f"{request.observation.observation_id}:{request.release.release_id}:"
            f"{simulation.simulation_id}".encode()
        ).hexdigest()[:16]
        return BackwardSourceTraceResult(
            trace_id=f"backward-trace:{trace_key}",
            observation=request.observation,
            plausible_source_time_range=request.release.interval,
            plausible_source_distribution=distribution,
            plausible_source_geometry=distribution.comparison_geometry,
            particle_trajectories=simulation.particle_trajectories,
            plausible_source_paths=paths,
            configured_uncertainty=simulation.configured_uncertainty,
            model_parameters=simulation.model_parameters,
            assumptions=(*request.release.assumptions, BACKWARD_MODEL_DISCLAIMER),
            simulation=simulation,
        )

    @staticmethod
    def _path(trajectory: ParticleTrajectory) -> SpatialGeometry:
        coordinates = []
        for item in trajectory.positions:
            geometry = item.position.geometry
            if geometry.type != "Point":  # Enforced by ParticlePosition; retains type narrowing.
                raise ValueError("particle trajectory contains a non-point position")
            coordinates.append(geometry.coordinate)
        return SpatialGeometry(
            geometry=LineStringGeometry(coordinates=tuple(coordinates)),
            crs=trajectory.positions[0].position.crs,
        )
