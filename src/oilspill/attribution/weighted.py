"""Transparent configurable weighted attribution over precomputed metric evidence."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from oilspill.config import ComponentConfig
from oilspill.domain.ais import CandidateVessel
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
    CalibrationStatus,
    ComponentMetadata,
    Decision,
    FrozenModel,
    Identifier,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
)
from oilspill.requests import AttributionRequest


class FeatureWeight(FrozenModel):
    name: Identifier
    weight: float = Field(gt=0.0)


class WeightedTrajectoryAttributionConfig(FrozenModel):
    """All aggregation choices are explicit and make no validation claim."""

    weights: tuple[FeatureWeight, ...] = Field(min_length=1)
    missing_metric_policy: Literal["renormalize_available", "require_all"]
    tie_strategy: Literal["shared_rank", "deterministic_order"]
    tie_tolerance: float = Field(ge=0.0)
    scientific_validation_note: Identifier

    @model_validator(mode="after")
    def _unique_weights(self) -> WeightedTrajectoryAttributionConfig:
        names = tuple(item.name for item in self.weights)
        if len(names) != len(set(names)):
            raise ValueError("feature weight names must be unique")
        return self


class WeightedTrajectoryAttribution:
    """Rank candidate-scoped normalized metrics without hiding feature contributions."""

    def __init__(
        self,
        config: WeightedTrajectoryAttributionConfig,
        *,
        component_name: str = "weighted_trajectory",
    ) -> None:
        self._config = config
        self._component_name = component_name

    def rank(self, request: AttributionRequest) -> AttributionResult:
        candidates = {candidate.candidate_id: candidate for candidate in request.candidates}
        metric_sets = {item.candidate_id: item for item in request.candidate_metrics}
        if len(candidates) != len(request.candidates):
            raise ValueError("attribution candidate IDs must be unique")
        if len(metric_sets) != len(request.candidate_metrics):
            raise ValueError("candidate metric sets must have unique candidate IDs")
        unknown = set(metric_sets).difference(candidates)
        if unknown:
            raise ValueError(f"metrics reference unknown candidates: {sorted(unknown)}")

        evaluated = [
            self._evaluate(candidate, metric_sets.get(candidate_id))
            for candidate_id, candidate in candidates.items()
        ]
        evaluated.sort(
            key=lambda item: (
                item.score is None,
                -(item.score or 0.0),
                item.vessel_id or "",
                item.candidate_id,
            )
        )
        ranked = self._assign_ranks(evaluated)
        scored_count = sum(item.score is not None for item in ranked)
        decision = Decision.RANKED if scored_count else Decision.ABSTAINED
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        result_key = sha256(
            f"{request.observation.observation_id}:{config_hash}:"
            f"{','.join(item.candidate_id for item in ranked)}".encode()
        ).hexdigest()[:16]
        return AttributionResult(
            attribution_id=f"weighted-attribution:{result_key}",
            observation_id=request.observation.observation_id,
            created_at=request.observation.observed_at,
            ranked_candidates=ranked,
            model=self.component_metadata(),
            decision=decision,
            abstention_reason=(
                "No candidate had all evidence required by the configured missing-metric policy."
                if decision == Decision.ABSTAINED
                else None
            ),
            calibration_status=CalibrationStatus.UNCALIBRATED,
            warnings=(
                (
                    QualityFlag(
                        code="INVESTIGATIVE_PRIORITIZATION_ONLY",
                        severity=QualitySeverity.WARNING,
                        message=request.interpretation_caveat,
                    ),
                )
                if request.interpretation_caveat
                else ()
            ),
        )

    def component_metadata(self) -> ComponentMetadata:
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name=self._component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="transparent-weighted-metric-aggregation",
            configuration_sha256=config_hash,
            attributes=(
                MetadataEntry(
                    key="scientific_validation", value=self._config.scientific_validation_note
                ),
                MetadataEntry(
                    key="missing_metric_policy", value=self._config.missing_metric_policy
                ),
                MetadataEntry(
                    key="weight_normalization",
                    value="sum_to_one_over_available_configured_weights",
                ),
                MetadataEntry(key="tie_strategy", value=self._config.tie_strategy),
                MetadataEntry(key="tie_tolerance", value=self._config.tie_tolerance),
            ),
        )

    def _evaluate(
        self,
        candidate: CandidateVessel,
        metric_set: CandidateMetricEvidence | None,
    ) -> RankedCandidate:
        metrics = {metric.name: metric for metric in metric_set.metrics} if metric_set else {}
        configured_sum = sum(item.weight for item in self._config.weights)
        available = {
            item.name: metrics[item.name]
            for item in self._config.weights
            if self._usable(metrics.get(item.name))
        }
        all_available = len(available) == len(self._config.weights)
        can_score = (
            bool(available)
            and (metric_set is None or metric_set.eligible_for_ranking)
            and (self._config.missing_metric_policy == "renormalize_available" or all_available)
        )
        effective_sum = (
            sum(item.weight for item in self._config.weights if item.name in available)
            if can_score
            else 0.0
        )
        evidence = tuple(
            self._evidence(
                metrics.get(item.name),
                item,
                effective_sum=effective_sum,
                can_score=can_score,
            )
            for item in self._config.weights
        )
        score = sum(item.weighted_contribution or 0.0 for item in evidence) if can_score else None
        if metric_set and metric_set.quality_issues:
            evidence = tuple(
                item.model_copy(
                    update={
                        "explanation": (item.explanation or "")
                        + " Quality: "
                        + "; ".join(metric_set.quality_issues)
                    }
                )
                for item in evidence
            )
        return RankedCandidate(
            candidate_id=candidate.candidate_id,
            vessel_id=candidate.track.vessel_id,
            rank=1,
            score=score,
            evidence=evidence,
            configured_weight_sum=configured_sum,
            effective_weight_sum=effective_sum,
        )

    @staticmethod
    def _usable(metric: MetricResult | None) -> bool:
        return (
            metric is not None
            and metric.status == MetricStatus.VALID
            and metric.normalized_value is not None
        )

    def _evidence(
        self,
        metric: MetricResult | None,
        feature: FeatureWeight,
        *,
        effective_sum: float,
        can_score: bool,
    ) -> EvidenceValue:
        usable = self._usable(metric)
        normalized_weight = feature.weight / effective_sum if usable and can_score else None
        contribution = (
            metric.normalized_value * normalized_weight
            if metric is not None
            and metric.normalized_value is not None
            and normalized_weight is not None
            else None
        )
        return EvidenceValue(
            name=feature.name,
            value=metric.raw_value if metric else None,
            normalized_value=metric.normalized_value if metric else None,
            unit=metric.unit if metric else None,
            direction=metric.direction if metric else EvidenceDirection.DESCRIPTIVE,
            status=metric.status if metric else MetricStatus.MISSING,
            configured_weight=feature.weight,
            normalized_weight=normalized_weight,
            weighted_contribution=contribution,
            explanation=(
                metric.explanation
                if metric
                else "Configured feature was not supplied; no numeric value was substituted."
            ),
            provenance=metric.provenance if metric else (),
            source_simulation_ids=metric.source_simulation_ids if metric else (),
        )

    def _assign_ranks(self, candidates: list[RankedCandidate]) -> tuple[RankedCandidate, ...]:
        ranked: list[RankedCandidate] = []
        previous_score: float | None = None
        previous_rank = 0
        for index, candidate in enumerate(candidates, start=1):
            if (
                ranked
                and self._config.tie_strategy == "shared_rank"
                and self._tied(candidate.score, previous_score)
            ):
                rank = previous_rank
            else:
                rank = index
            ranked.append(candidate.model_copy(update={"rank": rank}))
            previous_score = candidate.score
            previous_rank = rank
        return tuple(ranked)

    def _tied(self, score: float | None, previous: float | None) -> bool:
        if score is None or previous is None:
            return score is previous
        return abs(score - previous) <= self._config.tie_tolerance


def create_weighted_trajectory(config: ComponentConfig) -> WeightedTrajectoryAttribution:
    settings = WeightedTrajectoryAttributionConfig.model_validate(config.settings)
    return WeightedTrajectoryAttribution(settings, component_name=config.name)
