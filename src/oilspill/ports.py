"""Stable ports implemented by replaceable scientific and provider adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from oilspill.domain.ais import AISPoint, CandidateVessel, VesselTrack
from oilspill.domain.attribution import AttributionResult, CandidateMetricEvidence
from oilspill.domain.common import ComponentMetadata, DatasetMetadata
from oilspill.domain.drift import DriftSimulation, ReleaseHypothesis
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.eulerian import EulerianTransportResult
from oilspill.domain.forecast import CoastalImpactResult, ForecastResult
from oilspill.domain.hybrid import (
    CoarseHybridHindcastResult,
    HybridHindcastResult,
    HybridHindcastTiming,
)
from oilspill.domain.sar import SARScene
from oilspill.domain.source_inference import (
    ObservationSimilarity,
    ReleaseHypothesisSet,
    SourceInferenceResult,
)
from oilspill.domain.spill import DischargeTimeEstimate, SpillDetection, SpillGeometry
from oilspill.requests import (
    AgeEstimationRequest,
    AISQuery,
    AttributionEvidenceRequest,
    AttributionRequest,
    BackwardTraceRequest,
    CandidateGenerationRequest,
    CoastalImpactRequest,
    CoastlineDataset,
    DriftForecastRequest,
    EnsembleForecastRequest,
    EnvironmentalQuery,
    EulerianTransportRequest,
    ForwardTraceRequest,
    HybridHindcastRequest,
    ObservationComparisonRequest,
    ReleaseGenerationRequest,
    SARReadRequest,
    SourceSearchRequest,
)


@runtime_checkable
class SARReader(Protocol):
    def read(self, request: SARReadRequest) -> SARScene: ...
    def dataset_metadata(self) -> DatasetMetadata: ...


@runtime_checkable
class SARPreprocessor(Protocol):
    def preprocess(self, scene: SARScene) -> SARScene: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class DetectorModel(Protocol):
    def load(self) -> None: ...
    def predict(self, scene: SARScene) -> tuple[SpillDetection, ...]: ...
    def predict_batch(self, scenes: Sequence[SARScene]) -> tuple[SpillDetection, ...]: ...
    def model_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class SpillCharacterizer(Protocol):
    def characterize(self, detection: SpillDetection) -> SpillGeometry: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class SpillAgeEstimator(Protocol):
    def estimate(self, request: AgeEstimationRequest) -> DischargeTimeEstimate: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class AISProvider(Protocol):
    def fetch_points(self, query: AISQuery) -> tuple[AISPoint, ...]: ...
    def dataset_metadata(self) -> DatasetMetadata: ...


@runtime_checkable
class VesselTrackReconstructor(Protocol):
    def build(self, points: Sequence[AISPoint]) -> tuple[VesselTrack, ...]: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateVessel, ...]: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class ReleaseHypothesisGenerator(Protocol):
    def generate(self, request: ReleaseGenerationRequest) -> ReleaseHypothesis: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class AttributionEvidenceBuilder(Protocol):
    def build(self, request: AttributionEvidenceRequest) -> CandidateMetricEvidence: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class EnvironmentalProvider(Protocol):
    def fetch(self, query: EnvironmentalQuery) -> tuple[EnvironmentalField, ...]: ...
    def dataset_metadata(self) -> DatasetMetadata: ...


@runtime_checkable
class EulerianTransportEngine(Protocol):
    def run(self, request: EulerianTransportRequest) -> EulerianTransportResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class HybridHindcastEngine(Protocol):
    def run(self, request: HybridHindcastRequest) -> HybridHindcastResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class HybridHindcastTimingProvider(Protocol):
    def last_run_timing(self) -> HybridHindcastTiming | None: ...


@runtime_checkable
class CoarseHindcastEngine(Protocol):
    """AIS-independent physics-first screening stage."""

    def run(self, request: HybridHindcastRequest) -> CoarseHybridHindcastResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class DriftModel(Protocol):
    def forward_source_trace(self, request: ForwardTraceRequest) -> DriftSimulation: ...
    def backward_source_trace(self, request: BackwardTraceRequest) -> DriftSimulation: ...
    def forecast(self, request: DriftForecastRequest) -> DriftSimulation: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class AttributionModel(Protocol):
    def rank(self, request: AttributionRequest) -> AttributionResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class EnsembleForecaster(Protocol):
    def forecast(self, request: EnsembleForecastRequest) -> ForecastResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class CoastalImpactAnalyzer(Protocol):
    def analyze(self, request: CoastalImpactRequest) -> CoastalImpactResult: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class CoastlineProvider(Protocol):
    def load(self) -> CoastlineDataset: ...
    def dataset_metadata(self) -> DatasetMetadata: ...


@runtime_checkable
class ReleaseHypothesisEnumerator(Protocol):
    def enumerate(self, request: ReleaseGenerationRequest) -> ReleaseHypothesisSet: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class ObservationLikelihood(Protocol):
    """Replaceable comparison; initial implementations return heuristic similarity only."""

    def compare(self, request: ObservationComparisonRequest) -> ObservationSimilarity: ...
    def component_metadata(self) -> ComponentMetadata: ...


@runtime_checkable
class SourceInference(Protocol):
    def search(self, request: SourceSearchRequest) -> SourceInferenceResult: ...
    def component_metadata(self) -> ComponentMetadata: ...
