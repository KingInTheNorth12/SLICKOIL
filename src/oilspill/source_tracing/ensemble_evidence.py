"""Existing metric definitions summarized across valid forward hypotheses."""

from statistics import median

import numpy as np

from oilspill.attribution.metrics import ForwardFitSimilarityMetric, HausdorffDistanceMetric
from oilspill.domain.attribution import CandidateMetricEvidence, MetricResult, MetricStatus
from oilspill.domain.common import MetadataEntry
from oilspill.domain.source_trace import ForwardSourceTraceResult
from oilspill.source_tracing.ensemble_models import (
    CandidateEnsembleSummary,
    MetricDistribution,
    QuantileValue,
    SourceEnsemblePlan,
    SourceMemberResult,
)


def member_evidence(
    plan: SourceEnsemblePlan, trace: ForwardSourceTraceResult
) -> CandidateMetricEvidence:
    config = plan.config
    metrics = []
    for name in config.metrics:
        # Metric dispatch, not drift-model dispatch; definitions remain in attribution.metrics.
        metric = (
            HausdorffDistanceMetric(config.distance)
            if name == "hausdorff_distance"
            else ForwardFitSimilarityMetric(config.geometry)
        )
        value = metric.compute(
            trace.comparison_ready_geometry,
            plan.root.observation.geometry.polygon,
            source_simulation_ids=(trace.simulation.simulation_id,),
        )
        metrics.append(
            value.model_copy(
                update={
                    "name": "ensemble_" + value.name,
                    "provenance": (
                        *value.provenance,
                        MetadataEntry(key="base_metric", value=value.name),
                        MetadataEntry(key="summary_role", value="individual_member"),
                    ),
                }
            )
        )
    supplemental = next(
        (
            e
            for e in plan.root.supplemental_metrics
            if e.candidate_id == trace.candidate.candidate_id
        ),
        None,
    )
    return CandidateMetricEvidence(
        candidate_id=trace.candidate.candidate_id,
        metrics=tuple(metrics) + (supplemental.metrics if supplemental else ()),
        eligible_for_ranking=supplemental.eligible_for_ranking if supplemental else True,
        quality_issues=supplemental.quality_issues if supplemental else (),
    )


def summarize_candidate(
    plan: SourceEnsemblePlan, candidate_id: str, members: tuple[SourceMemberResult, ...]
) -> CandidateEnsembleSummary:
    selected = tuple(m for m in members if m.candidate_id == candidate_id)
    succeeded = tuple(m for m in selected if m.status == "succeeded")
    issues = list(next(c.issues for c in plan.coverage_issues if c.candidate_id == candidate_id))
    enough = len(succeeded) >= plan.config.minimum_successful_members
    if not enough:
        issues.append(
            "too few successful ensemble members under configured policy; "
            "DATASET_VALIDATION_REQUIRED"
        )
    metrics = []
    distributions = []
    for name in plan.config.metrics:
        full_name = "ensemble_" + name
        all_values = [
            v for m in succeeded if m.evidence for v in m.evidence.metrics if v.name == full_name
        ]
        valid = [v for v in all_values if v.status == MetricStatus.VALID]
        raw = [v.raw_value for v in valid if v.raw_value is not None]
        normalized = [v.normalized_value for v in valid if v.normalized_value is not None]
        # Never summarize normalization over a different subset than raw values.
        has_normalized = bool(raw) and len(normalized) == len(raw)
        raw_median = float(median(raw)) if raw else None
        norm_median = float(median(normalized)) if has_normalized else None
        template = next(iter(all_values), None)
        distributions.append(
            MetricDistribution(
                name=full_name,
                valid_count=len(raw),
                missing_or_invalid_count=len(selected) - len(raw),
                median_raw_value=raw_median,
                median_normalized_value=norm_median,
                unit=template.unit if template else None,
                quantiles=tuple(
                    QuantileValue(
                        quantile=q,
                        raw_value=float(np.quantile(raw, q, method=plan.config.quantile_method)),
                        normalized_value=float(
                            np.quantile(normalized, q, method=plan.config.quantile_method)
                        )
                        if has_normalized
                        else None,
                    )
                    for q in plan.config.quantiles
                )
                if raw
                else (),
            )
        )
        if template:
            metrics.append(
                MetricResult(
                    name=full_name,
                    status=MetricStatus.VALID if raw else MetricStatus.MISSING,
                    raw_value=raw_median,
                    normalized_value=norm_median,
                    unit=template.unit,
                    direction=template.direction,
                    explanation=(
                        "Median of valid member metric values; "
                        "not a calibrated responsibility estimate."
                    ),
                    provenance=(
                        *(entry for entry in template.provenance if entry.key != "summary_role"),
                        MetadataEntry(key="aggregation", value="median_valid_members"),
                        MetadataEntry(key="valid_count", value=len(raw)),
                        MetadataEntry(key="plan_id", value=plan.plan_id),
                    ),
                    source_simulation_ids=tuple(s for v in valid for s in v.source_simulation_ids),
                )
            )
    supplemental = next(
        (e for e in plan.root.supplemental_metrics if e.candidate_id == candidate_id), None
    )
    coverage_ok = not issues or plan.config.coverage_policy == "warn"
    if supplemental:
        issues.extend(supplemental.quality_issues)
    evidence = CandidateMetricEvidence(
        candidate_id=candidate_id,
        metrics=tuple(metrics) + (supplemental.metrics if supplemental else ()),
        eligible_for_ranking=enough
        and coverage_ok
        and (supplemental.eligible_for_ranking if supplemental else True),
        quality_issues=tuple(issues),
    )
    return CandidateEnsembleSummary(
        candidate_id=candidate_id,
        requested_member_count=len(selected),
        valid_member_count=len(succeeded),
        failed_member_count=len(selected) - len(succeeded),
        metrics=tuple(distributions),
        evidence=evidence,
        quality_issues=tuple(issues),
        comparable_member_count=0,
    )
