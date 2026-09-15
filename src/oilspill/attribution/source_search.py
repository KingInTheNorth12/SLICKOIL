"""Explicit bridge from source-fit evidence to the existing attribution port."""

from hashlib import sha256
from typing import Literal

from oilspill.config import ComponentConfig
from oilspill.domain.attribution import CandidateMetricEvidence, MetricResult
from oilspill.domain.common import (
    ComponentMetadata,
    Decision,
    FrozenModel,
    Identifier,
    MetadataEntry,
)
from oilspill.requests import AttributionEvidenceRequest


class SourceSearchEvidenceConfig(FrozenModel):
    selection: Literal["best_valid_forward_per_candidate"]
    source_decision_policy: Literal["require_ranked_source_inference"]
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]


class SourceSearchEvidenceBuilder:
    """Preserve raw metrics and selection lineage; never assign vessel responsibility."""

    def __init__(self, config: SourceSearchEvidenceConfig) -> None:
        self.config = config

    def build(self, request: AttributionEvidenceRequest) -> CandidateMetricEvidence:
        source = request.source_inference
        if source is None:
            raise ValueError("source-search evidence requires a SourceInferenceResult")
        if (
            source.observation_id != request.observation.observation_id
            or request.candidate.observation_id != source.observation_id
        ):
            raise ValueError("source evidence observation mismatch")
        group = next(
            (s for s in source.hypothesis_sets if s.candidate_id == request.candidate.candidate_id),
            None,
        )
        if group is None:
            raise ValueError("candidate absent from source inference")
        usable = [
            e
            for e in source.evaluated_hypotheses
            if e.hypothesis.source_candidate_id == request.candidate.candidate_id
            and not e.error
            and e.evidence
            and e.evidence.ordering_value is not None
        ]
        best = min(
            usable,
            key=lambda e: (e.evidence.ordering_value if e.evidence else 0, e.hypothesis.release_id),
            default=None,
        )
        metrics: tuple[MetricResult, ...] = ()
        if best and best.evidence:
            metrics = tuple(
                m.model_copy(
                    update={
                        "provenance": (
                            *m.provenance,
                            MetadataEntry(key="source_inference_id", value=source.inference_id),
                            MetadataEntry(key="release_id", value=best.hypothesis.release_id),
                            MetadataEntry(key="selection", value=self.config.selection),
                            MetadataEntry(key="validation_note", value=self.config.validation_note),
                            MetadataEntry(
                                key="validation_status", value=self.config.validation_status
                            ),
                        )
                    }
                )
                for m in best.evidence.metrics
            )
        return CandidateMetricEvidence(
            candidate_id=request.candidate.candidate_id,
            metrics=metrics,
            eligible_for_ranking=bool(best) and source.decision == Decision.RANKED,
            quality_issues=(
                *group.coverage_issues,
                *source.abstention_reasons,
                *(("no usable forward comparisons",) if best is None else ()),
            ),
        )

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="source_search_metrics",
            version="1",
            implementation=f"{__name__}.SourceSearchEvidenceBuilder",
            configuration_sha256=sha256(self.config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="validation_status", value=self.config.validation_status),
            ),
        )


def create_source_search_evidence(config: ComponentConfig) -> SourceSearchEvidenceBuilder:
    return SourceSearchEvidenceBuilder(SourceSearchEvidenceConfig.model_validate(config.settings))
