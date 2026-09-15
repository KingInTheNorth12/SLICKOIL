"""Transparent vessel-attribution evidence and ranking contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from oilspill.domain.common import (
    CalibrationStatus,
    ComponentMetadata,
    Decision,
    FrozenModel,
    Identifier,
    MetadataEntry,
    Probability,
    QualityFlag,
    UTCDateTime,
)


class EvidenceDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DESCRIPTIVE = "descriptive"


class MetricStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


class MetricResult(FrozenModel):
    """One reusable metric result before attribution-model aggregation."""

    name: Identifier
    status: MetricStatus
    raw_value: float | None
    normalized_value: float | None = None
    unit: str | None = None
    direction: EvidenceDirection
    explanation: Identifier
    provenance: tuple[MetadataEntry, ...] = ()
    source_simulation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _status_matches_values(self) -> MetricResult:
        if self.status == MetricStatus.VALID and self.raw_value is None:
            raise ValueError("valid metric result requires raw_value")
        if self.status != MetricStatus.VALID and (
            self.raw_value is not None or self.normalized_value is not None
        ):
            raise ValueError("missing or invalid metric results cannot carry numeric values")
        return self


class CandidateMetricEvidence(FrozenModel):
    """Metrics associated with exactly one screened vessel candidate."""

    candidate_id: Identifier
    metrics: tuple[MetricResult, ...]
    eligible_for_ranking: bool = True
    quality_issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_metric_names(self) -> CandidateMetricEvidence:
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("candidate metric names must be unique")
        return self


class EvidenceValue(FrozenModel):
    """One named, interpretable feature contributing to a ranking."""

    name: Identifier
    value: float | None
    normalized_value: float | None = None
    unit: str | None
    direction: EvidenceDirection
    status: MetricStatus = MetricStatus.VALID
    configured_weight: float | None = Field(default=None, ge=0.0)
    normalized_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    weighted_contribution: float | None = None
    explanation: str | None = None
    provenance: tuple[MetadataEntry, ...] = ()
    source_simulation_ids: tuple[str, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()


class RankedCandidate(FrozenModel):
    candidate_id: Identifier
    vessel_id: Identifier | None = None
    rank: int = Field(gt=0)
    score: float | None = None
    calibrated_probability: Probability | None = None
    evidence: tuple[EvidenceValue, ...]
    configured_weight_sum: float | None = Field(default=None, ge=0.0)
    effective_weight_sum: float | None = Field(default=None, ge=0.0)


class AttributionResult(FrozenModel):
    """Evidence ranking or explicit abstention, never legal proof of responsibility."""

    attribution_id: Identifier
    observation_id: Identifier
    created_at: UTCDateTime
    ranked_candidates: tuple[RankedCandidate, ...]
    model: ComponentMetadata
    decision: Decision
    abstention_reason: str | None = None
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    warnings: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _valid_decision(self) -> AttributionResult:
        if self.decision == Decision.ABSTAINED and not self.abstention_reason:
            raise ValueError("abstained attribution requires a reason")
        if self.decision == Decision.RANKED and not self.ranked_candidates:
            raise ValueError("ranked attribution requires candidates")
        return self
