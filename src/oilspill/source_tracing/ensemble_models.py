"""Auditable forward-source ensembles, not calibrated responsibility estimates."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.attribution.models import DistanceMetricConfig, GeometryMetricConfig
from oilspill.domain.ais import CandidateVessel
from oilspill.domain.attribution import AttributionResult, CandidateMetricEvidence
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    UTCDateTime,
)
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.geospatial import SpatialGeometry
from oilspill.domain.source_trace import ForwardSourceTraceResult
from oilspill.domain.spill import SpillObservation
from oilspill.forecasting.monte_carlo_config import (
    BaselineUncertainty,
    ForcingSelection,
    NumericUncertainty,
)
from oilspill.requests import ForwardTraceRequest


class ReleaseTimeSampling(FrozenModel):
    """Discrete uniform sampling is over available AIS points, NOT uniform elapsed time."""

    kind: Literal["fixed", "uniform_existing_track_points"]
    timestamp: UTCDateTime | None = None
    representation: Literal["external_member_sampling"]
    validation_note: Identifier

    @model_validator(mode="after")
    def binding(self) -> "ReleaseTimeSampling":
        if (self.kind == "fixed") != (self.timestamp is not None):
            raise ValueError("only fixed release sampling requires a timestamp")
        return self


class SourceEnsembleConfig(FrozenModel):
    output_root: Path
    member_count: int = Field(gt=0, le=2**32)
    release_time: ReleaseTimeSampling
    uncertainties: tuple[NumericUncertainty, ...]
    baseline_uncertainties: tuple[BaselineUncertainty, ...]
    baseline_inventory_complete: Literal[True]
    forcing: ForcingSelection
    require_forcing: bool
    sampling_dependence: Literal["independent"]
    failure_policy: Literal["record_and_continue", "raise_after_recording"]
    minimum_successful_members: int = Field(gt=0)
    coverage_policy: Literal["warn", "abstain"]
    overlapping_score_ranges_policy: Literal["warn", "abstain"]
    quantiles: tuple[float, ...]
    quantile_method: Literal["linear"]
    metrics: tuple[Literal["hausdorff_distance", "forward_fit_similarity"], ...] = Field(
        min_length=1
    )
    distance: DistanceMetricConfig
    geometry: GeometryMetricConfig
    validation_note: Identifier

    @model_validator(mode="after")
    def choices(self) -> "SourceEnsembleConfig":
        if self.minimum_successful_members > self.member_count:
            raise ValueError("minimum successful members exceeds requested count")
        if any(not 0 <= q <= 1 for q in self.quantiles):
            raise ValueError("quantiles must lie in [0, 1]")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and sorted")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("duplicate metrics")
        ids = (
            [u.source_id for u in self.uncertainties]
            + [u.source_id for u in self.baseline_uncertainties]
            + [self.forcing.source_id, "release_time"]
        )
        keys = [k for u in self.uncertainties for k in u.exclusive_parameter_keys] + [
            k for u in self.baseline_uncertainties for k in u.parameter_keys
        ]
        targets = [u.target for u in self.uncertainties if u.target != "model_parameter"]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("duplicate outer/internal uncertainty declaration")
        if "offset_seconds" in targets or len(set(targets)) != len(targets):
            raise ValueError("release time uses exact AIS lookup; duplicate/time offsets forbidden")
        return self


class SourceEnsembleRequest(FrozenModel):
    observation: SpillObservation
    candidates: tuple[CandidateVessel, ...] = Field(min_length=1)
    forcing: tuple[EnvironmentalField, ...]
    random_seed: int
    # Explicit non-ensemble evidence, e.g. deterministic backward evidence. Never overwritten.
    supplemental_metrics: tuple[CandidateMetricEvidence, ...] = ()


class SourceMemberPlan(FrozenModel):
    candidate_id: Identifier
    member_id: Identifier
    member_index: int = Field(ge=0)
    engine_seed: int = Field(ge=0, lt=2**32)
    sampled_time: UTCDateTime | None
    sampled_position: SpatialGeometry | None
    forcing_member_id: Identifier
    sampled_values: tuple[MetadataEntry, ...]
    request: ForwardTraceRequest | None
    planning_error: str | None = None


class CandidateCoverageIssue(FrozenModel):
    candidate_id: Identifier
    issues: tuple[str, ...]


class SourceEnsemblePlan(FrozenModel):
    schema_version: Literal["1"] = "1"
    planner_version: Literal["canonical-ais-python-random-v1"] = "canonical-ais-python-random-v1"
    plan_id: Identifier
    root: SourceEnsembleRequest
    config: SourceEnsembleConfig
    drift: ComponentMetadata
    members: tuple[SourceMemberPlan, ...]
    coverage_issues: tuple[CandidateCoverageIssue, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> "SourceEnsemblePlan":
        ids = {c.candidate_id for c in self.root.candidates}
        expected = {(c, i) for c in ids for i in range(self.config.member_count)}
        if (
            len(self.members) != len(expected)
            or {(m.candidate_id, m.member_index) for m in self.members} != expected
        ):
            raise ValueError("candidate/member plan count mismatch")
        if len({m.engine_seed for m in self.members}) != len(self.members):
            raise ValueError("duplicate member seeds")
        if len({m.member_id for m in self.members}) != len(self.members):
            raise ValueError("duplicate member IDs")
        if {c.candidate_id for c in self.coverage_issues} != ids:
            raise ValueError("coverage record mismatch")
        for m in self.members:
            if (m.request is None) != (m.planning_error is not None):
                raise ValueError("member must contain either a request or a planning failure")
            if m.request is not None:
                candidate = m.request.candidate
                if (
                    candidate is None
                    or m.request.random_seed != m.engine_seed
                    or m.request.release.interval.start != m.sampled_time
                    or m.request.release.interval.end != m.sampled_time
                    or m.request.release.release_id != m.member_id
                    or m.request.release.geometry != m.sampled_position
                    or candidate.candidate_id != m.candidate_id
                    or m.request.observation != self.root.observation
                ):
                    raise ValueError("member request identity/seed/time mismatch")
        return self


class SourceMemberResult(FrozenModel):
    member_id: Identifier
    candidate_id: Identifier
    member_index: int
    status: Literal["succeeded", "failed"]
    trace: ForwardSourceTraceResult | None = None
    evidence: CandidateMetricEvidence | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> "SourceMemberResult":
        if self.status == "succeeded":
            if self.trace is None or self.evidence is None or self.error is not None:
                raise ValueError("successful member requires trace and evidence, no error")
        elif self.error is None or self.trace is not None or self.evidence is not None:
            raise ValueError("failed member requires error, no successful products")
        return self


class QuantileValue(FrozenModel):
    quantile: float
    raw_value: float
    normalized_value: float | None


class MetricDistribution(FrozenModel):
    name: Identifier
    valid_count: int
    missing_or_invalid_count: int
    median_raw_value: float | None
    median_normalized_value: float | None
    unit: str | None
    quantiles: tuple[QuantileValue, ...]


class CandidateEnsembleSummary(FrozenModel):
    candidate_id: Identifier
    requested_member_count: int
    valid_member_count: int
    failed_member_count: int
    metrics: tuple[MetricDistribution, ...]
    evidence: CandidateMetricEvidence
    quality_issues: tuple[str, ...]
    comparable_member_count: int
    first_rank_member_frequency: float | None = Field(default=None, ge=0, le=1)


class MemberRanking(FrozenModel):
    member_index: int
    attribution: AttributionResult
    comparable: bool


class SourceEnsembleResult(FrozenModel):
    plan: ArtifactRef
    members: tuple[SourceMemberResult, ...]
    summaries: tuple[CandidateEnsembleSummary, ...]
    attribution: AttributionResult
    ranker: ComponentMetadata
    member_rankings: tuple[MemberRanking, ...]
    semantics: Literal["first-rank member frequency; shared first ranks split unit credit"] = (
        "first-rank member frequency; shared first ranks split unit credit"
    )
    comparable_member_count: int
    warnings: tuple[str, ...]
