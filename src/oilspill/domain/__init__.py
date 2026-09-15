"""Public domain model API."""

# ruff: noqa: F401 - this module intentionally re-exports the public domain API.

from oilspill.domain.ais import (
    AISPoint,
    CandidateDecision,
    CandidateEvidence,
    CandidateVessel,
    TrackCoverage,
    VesselTrack,
)
from oilspill.domain.attribution import (
    AttributionResult,
    CandidateMetricEvidence,
    EvidenceDirection,
    EvidenceValue,
    MetricResult,
    MetricStatus,
    RankedCandidate,
)
from oilspill.domain.common import (
    ArtifactRef,
    CalibrationStatus,
    ComponentMetadata,
    DataKind,
    DatasetMetadata,
    Decision,
    MetadataEntry,
    ProcessingRecord,
    QualityFlag,
    QualitySeverity,
    StageOutcome,
    StageStatus,
    TimeRange,
    UTCDateTime,
)
from oilspill.domain.drift import (
    DriftMode,
    DriftSimulation,
    ParticleDistribution,
    ParticlePosition,
    ParticleTrajectory,
    ReleaseHypothesis,
    SimulationDiagnostics,
    SimulationStatus,
)
from oilspill.domain.environment import (
    EnvironmentalField,
    EnvironmentalVariable,
    FieldComponent,
    GridDefinition,
    GridKind,
)
from oilspill.domain.eulerian import (
    EulerianDiagnostics,
    EulerianGrid,
    EulerianInputProvenance,
    EulerianScenario,
    EulerianSnapshot,
    EulerianTransportResult,
    ObservedSlickInitialCondition,
    PointSourceInitialCondition,
)
from oilspill.domain.forecast import (
    CoastalImpactResult,
    CoastlineSegmentImpact,
    ForecastDecision,
    ForecastHorizon,
    ForecastResult,
    ForecastRunProvenance,
    ScenarioContactArrival,
    SpatialProbabilityRepresentation,
    UncertaintySummary,
)
from oilspill.domain.geospatial import (
    CRS,
    AffineTransform,
    Angle,
    AngularUnit,
    Area,
    AreaUnit,
    Coordinate,
    Duration,
    Length,
    LengthUnit,
    LineStringGeometry,
    MultiPolygonGeometry,
    PointGeometry,
    PolygonGeometry,
    RasterAsset,
    RasterBand,
    RasterGrid,
    SpatialGeometry,
    SpatialUnit,
    Speed,
)
from oilspill.domain.hybrid import (
    CoarseHindcastDiagnostics,
    CoarseHybridHindcastResult,
    CoarseSourceEvaluation,
    HighFidelitySourceEvaluation,
    HybridConvergenceDiagnostics,
    HybridConvergenceStatus,
    HybridForecastBatchDiagnostics,
    HybridForecastDiagnostics,
    HybridForecastScenarioEvaluation,
    HybridHindcastResult,
    HybridHindcastTiming,
    HybridScenarioWeight,
    HybridSourceState,
    ReachabilityRejection,
    ReleaseTimeQuantiles,
    SourcePosterior,
    WeightedSourceState,
)
from oilspill.domain.pipeline import PipelineResult
from oilspill.domain.sar import SARScene
from oilspill.domain.source_inference import (
    HypothesisEvaluation,
    ObservationSimilarity,
    PlausibleSourceRegion,
    ReleaseHypothesisSet,
    SourceInferenceResult,
)
from oilspill.domain.source_trace import BackwardSourceTraceResult, ForwardSourceTraceResult
from oilspill.domain.spill import (
    DischargeTimeEstimate,
    DischargeTimeEvidence,
    ReviewStatus,
    SpillDetection,
    SpillGeometry,
    SpillObservation,
)

__all__ = [name for name in globals() if not name.startswith("_")]
