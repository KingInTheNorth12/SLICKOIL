"""Forward candidate-release tracing without engine-specific calls or ranking."""

from __future__ import annotations

from hashlib import sha256

from oilspill.domain.drift import DriftMode, SimulationStatus
from oilspill.domain.source_trace import ForwardSourceTraceResult
from oilspill.ports import DriftModel
from oilspill.requests import ForwardTraceRequest


class ForwardSourceTracer:
    """Run a supplied ``DriftModel`` and expose comparison-ready canonical output."""

    def __init__(self, drift_model: DriftModel) -> None:
        self._drift_model = drift_model

    def trace(self, request: ForwardTraceRequest) -> ForwardSourceTraceResult:
        self._validate_request(request)
        candidate = request.candidate
        assert candidate is not None
        simulation = self._drift_model.forward_source_trace(request)
        if simulation.mode != DriftMode.FORWARD_TRACE:
            raise ValueError("drift model returned a non-forward simulation")
        if simulation.status != SimulationStatus.SUCCEEDED:
            raise ValueError("forward source trace requires a successful simulation")
        if simulation.target_timestamp != request.observation.observed_at:
            raise ValueError("forward simulation target must equal the SAR observation time")
        if simulation.release_coordinates is None or simulation.release_timestamp is None:
            raise ValueError("drift model omitted canonical release provenance")
        distribution = simulation.target_distribution
        if distribution is None:
            raise ValueError("drift model omitted the target-time particle distribution")
        trace_key = sha256(
            f"{candidate.candidate_id}:{request.release.release_id}:"
            f"{simulation.simulation_id}".encode()
        ).hexdigest()[:16]
        return ForwardSourceTraceResult(
            trace_id=f"forward-trace:{trace_key}",
            candidate=candidate,
            release_position=simulation.release_coordinates,
            release_time=simulation.release_timestamp,
            simulated_particle_distribution=distribution,
            comparison_ready_geometry=distribution.comparison_geometry,
            simulation=simulation,
        )

    @staticmethod
    def _validate_request(request: ForwardTraceRequest) -> None:
        if request.candidate is None:
            raise ValueError("legacy forward source tracing requires a vessel candidate")
        if request.candidate.observation_id != request.observation.observation_id:
            raise ValueError("candidate does not belong to the supplied spill observation")
        source_candidate = request.release.source_candidate_id
        if source_candidate is not None and source_candidate != request.candidate.candidate_id:
            raise ValueError("release hypothesis identifies a different candidate")
        if request.release.geometry.geometry.type != "Point":
            raise ValueError("candidate release hypothesis must contain Point geometry")
