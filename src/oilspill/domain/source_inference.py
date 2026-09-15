"""Forward-search evidence, explicitly separate from vessel responsibility."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.attribution import EvidenceDirection, MetricResult, MetricStatus
from oilspill.domain.common import ArtifactRef, ComponentMetadata, Decision, FrozenModel, Identifier
from oilspill.domain.drift import ReleaseHypothesis
from oilspill.domain.source_trace import BackwardSourceTraceResult, ForwardSourceTraceResult

# Preserve the existing public class and JSON representation.
PlausibleSourceRegion = BackwardSourceTraceResult


class ReleaseHypothesisSet(FrozenModel):
    candidate_id: Identifier
    hypotheses: tuple[ReleaseHypothesis, ...]
    coverage_issues: tuple[str, ...]
    generator: ComponentMetadata

    @model_validator(mode="after")
    def consistent(self) -> ReleaseHypothesisSet:
        ids = [h.release_id for h in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate release IDs")
        if any(h.source_candidate_id != self.candidate_id for h in self.hypotheses):
            raise ValueError("hypothesis candidate mismatch")
        return self


class ObservationSimilarity(FrozenModel):
    observation_id: Identifier
    release_id: Identifier
    simulation_id: Identifier
    metrics: tuple[MetricResult, ...] = Field(min_length=1)
    primary_metric: Identifier
    component: ComponentMetadata
    semantics: Literal["heuristic similarity; not a probability"] = (
        "heuristic similarity; not a probability"
    )

    @model_validator(mode="after")
    def consistent(self) -> ObservationSimilarity:
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)) or self.primary_metric not in names:
            raise ValueError("metrics must be unique and contain primary metric")
        if self.primary.direction == EvidenceDirection.DESCRIPTIVE:
            raise ValueError("primary metric must declare an ordering direction")
        for metric in self.metrics:
            if any(
                value is not None and not math.isfinite(value)
                for value in (metric.raw_value, metric.normalized_value)
            ):
                raise ValueError("similarity metrics must be finite or explicitly missing")
            if metric.source_simulation_ids != (self.simulation_id,):
                raise ValueError("metric simulation provenance mismatch")
        return self

    @property
    def primary(self) -> MetricResult:
        return next(m for m in self.metrics if m.name == self.primary_metric)

    @property
    def score(self) -> float | None:
        """Raw primary metric, with its original unit and direction."""
        return self.primary.raw_value if self.primary.status == MetricStatus.VALID else None

    @property
    def ordering_value(self) -> float | None:
        value = self.score
        if value is None or self.primary.direction == EvidenceDirection.DESCRIPTIVE:
            return None
        return -value if self.primary.direction == EvidenceDirection.HIGHER_IS_BETTER else value


class HypothesisEvaluation(FrozenModel):
    hypothesis: ReleaseHypothesis
    iteration: int = Field(ge=0)
    parent_ids: tuple[str, ...] = ()
    random_seed: int = Field(ge=0, lt=2**32)
    trace: ForwardSourceTraceResult | None = None
    evidence: ObservationSimilarity | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> HypothesisEvaluation:
        if self.error is None and (self.trace is None or self.evidence is None):
            raise ValueError("evaluation requires trace/evidence or an error")
        if self.error is not None and self.evidence is not None:
            raise ValueError("failed evaluation cannot carry successful evidence")
        if self.trace and (
            self.trace.simulation.release != self.hypothesis
            or self.trace.simulation.random_seed != self.random_seed
        ):
            raise ValueError("evaluation trace provenance mismatch")
        if self.evidence and (
            self.evidence.release_id != self.hypothesis.release_id
            or self.trace is None
            or self.evidence.simulation_id != self.trace.simulation.simulation_id
        ):
            raise ValueError("evaluation evidence provenance mismatch")
        return self


class SourceInferenceResult(FrozenModel):
    inference_id: Identifier
    observation_id: Identifier
    evaluated_hypotheses: tuple[HypothesisEvaluation, ...]
    hypothesis_sets: tuple[ReleaseHypothesisSet, ...]
    best_supported_hypothesis_ids: tuple[str, ...]
    decision: Decision
    abstention_reasons: tuple[str, ...]
    termination_reason: Identifier
    unevaluated_hypothesis_ids: tuple[str, ...]
    backward_evidence: tuple[PlausibleSourceRegion, ...] = ()
    provenance: ArtifactRef
    iterations: tuple[ArtifactRef, ...]
    component: ComponentMetadata
    warnings: tuple[str, ...]
    semantics: Literal["conditional forward-search support; not confirmed origin"] = (
        "conditional forward-search support; not confirmed origin"
    )

    @model_validator(mode="after")
    def consistent(self) -> SourceInferenceResult:
        ids = [e.hypothesis.release_id for e in self.evaluated_hypotheses]
        universe = [h.release_id for s in self.hypothesis_sets for h in s.hypotheses]
        remaining = self.unevaluated_hypothesis_ids
        if len(remaining) != len(set(remaining)):
            raise ValueError("duplicate unevaluated hypotheses")
        if any(
            e.evidence and e.evidence.observation_id != self.observation_id
            for e in self.evaluated_hypotheses
        ):
            raise ValueError("evidence observation mismatch")
        if len(ids) != len(set(ids)) or len(universe) != len(set(universe)):
            raise ValueError("duplicate evaluated or enumerated hypotheses")
        if set(ids) & set(remaining) or set(ids) | set(remaining) != set(universe):
            raise ValueError("evaluated/unevaluated hypotheses must partition the search space")
        supported = set(self.best_supported_hypothesis_ids)
        usable = {
            e.hypothesis.release_id
            for e in self.evaluated_hypotheses
            if e.evidence and e.evidence.ordering_value is not None and not e.error
        }
        if not supported <= usable:
            raise ValueError("supported hypotheses require valid forward evidence")
        if self.decision == Decision.ABSTAINED:
            if not self.abstention_reasons or supported:
                raise ValueError("abstention requires reasons and no operational selection")
        elif not supported or self.abstention_reasons:
            raise ValueError("ranked inference requires supported hypotheses and no abstention")
        if any(b.observation.observation_id != self.observation_id for b in self.backward_evidence):
            raise ValueError("backward evidence observation mismatch")
        return self
