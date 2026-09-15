"""Shared canonical forward execution and immutable snapshots for source services."""

from pathlib import Path

from oilspill.artifacts import read_artifact, write_artifact
from oilspill.domain.common import ComponentMetadata
from oilspill.domain.drift import DriftSimulation
from oilspill.domain.source_trace import ForwardSourceTraceResult
from oilspill.ports import DriftModel
from oilspill.requests import ForwardTraceRequest
from oilspill.source_tracing.forward import ForwardSourceTracer


def execute_forward(
    drift: DriftModel,
    request: ForwardTraceRequest,
    metadata: ComponentMetadata,
    directory: Path,
    index: int,
) -> ForwardSourceTraceResult:
    for field in request.forcing:
        read_artifact(field.values)
    trace = ForwardSourceTracer(drift).trace(request)
    simulation = trace.simulation
    distribution = trace.simulated_particle_distribution
    if (
        simulation.random_seed != request.random_seed
        or simulation.release != request.release
        or simulation.release_timestamp != request.release.interval.start
        or simulation.model != metadata
        or simulation.configuration_sha256 != metadata.configuration_sha256
        or simulation.forcing_ids != tuple(f.field_id for f in request.forcing)
        or simulation.forcing_artifacts != tuple(f.values for f in request.forcing)
        or distribution.timestamp != request.observation.observed_at
        or distribution.particle_count == 0
    ):
        raise ValueError("simulation disagrees with saved member request or lacks target support")
    for supplied, actual in (
        (request.model_parameters, simulation.model_parameters),
        (request.configured_uncertainty, simulation.configured_uncertainty),
    ):
        if any(entry not in actual for entry in supplied):
            raise ValueError("drift adapter did not preserve requested uncertainty/parameters")
    simulation = snapshot_simulation(simulation, directory, index)
    return trace.model_copy(
        update={
            "simulation": simulation,
            "simulated_particle_distribution": simulation.target_distribution,
        }
    )


def snapshot_simulation(
    simulation: DriftSimulation, directory: Path, index: int
) -> DriftSimulation:
    artifact = simulation.trajectories
    distribution = simulation.target_distribution
    snapshot = artifact
    if not artifact.uri.startswith("memory://"):
        snapshot = write_artifact(
            directory / f"trajectories-{index}.bin", read_artifact(artifact), artifact.media_type
        )
    if distribution:
        positions = distribution.positions
        if positions == artifact:
            positions = snapshot
        elif not positions.uri.startswith("memory://"):
            positions = write_artifact(
                directory / f"positions-{index}.bin", read_artifact(positions), positions.media_type
            )
        distribution = distribution.model_copy(update={"positions": positions})
    # Memory references remain explicitly synthetic; canonical trajectories persist inline.
    return simulation.model_copy(
        update={"trajectories": snapshot, "target_distribution": distribution}
    )
