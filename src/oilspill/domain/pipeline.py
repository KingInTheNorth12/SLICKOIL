"""Top-level pipeline result aggregate."""

from __future__ import annotations

from oilspill.domain.ais import CandidateVessel
from oilspill.domain.attribution import AttributionResult
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    Identifier,
    QualityFlag,
    StageOutcome,
)
from oilspill.domain.drift import DriftSimulation
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.forecast import CoastalImpactResult, ForecastResult
from oilspill.domain.hybrid import HybridHindcastResult, HybridHindcastTiming
from oilspill.domain.source_inference import SourceInferenceResult
from oilspill.domain.spill import SpillObservation


class PipelineResult(FrozenModel):
    """Serializable index of all completed, skipped, abstained, or failed work."""

    run_id: Identifier
    observations: tuple[SpillObservation, ...]
    candidates: tuple[CandidateVessel, ...] = ()
    simulations: tuple[DriftSimulation, ...] = ()
    attributions: tuple[AttributionResult, ...] = ()
    forecasts: tuple[ForecastResult, ...] = ()
    coastal_impacts: tuple[CoastalImpactResult, ...] = ()
    stages: tuple[StageOutcome, ...]
    configuration_sha256: str
    provenance: ArtifactRef
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[QualityFlag, ...] = ()
    source_inferences: tuple[SourceInferenceResult, ...] = ()
    hybrid_hindcasts: tuple[HybridHindcastResult, ...] = ()
    random_seed: int | None = None
    component_metadata: tuple[ComponentMetadata, ...] = ()
    hybrid_hindcast_timings: tuple[HybridHindcastTiming, ...] = ()
    environmental_forcing: tuple[EnvironmentalField, ...] = ()
