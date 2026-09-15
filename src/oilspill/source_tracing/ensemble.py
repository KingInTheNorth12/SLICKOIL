"""Generic serial forward-source ensemble service; attribution stays behind its port."""

from dataclasses import dataclass
from uuid import uuid4

from oilspill.artifacts import read_artifact, write_artifact
from oilspill.domain.attribution import AttributionResult, CandidateMetricEvidence
from oilspill.domain.common import ArtifactRef, Decision
from oilspill.domain.drift import DriftSimulation
from oilspill.ports import AttributionModel, DriftModel
from oilspill.requests import AttributionRequest
from oilspill.source_tracing.ensemble_evidence import member_evidence, summarize_candidate
from oilspill.source_tracing.ensemble_models import (
    MemberRanking,
    SourceEnsembleConfig,
    SourceEnsemblePlan,
    SourceEnsembleRequest,
    SourceEnsembleResult,
    SourceMemberResult,
)
from oilspill.source_tracing.ensemble_planning import plan_source_ensemble
from oilspill.source_tracing.execution import execute_forward, snapshot_simulation


@dataclass(frozen=True)
class SavedSourceEnsemble:
    result: SourceEnsembleResult
    artifact: ArtifactRef


class SourceEnsembleExecutionError(RuntimeError):
    def __init__(self, saved: SavedSourceEnsemble) -> None:
        super().__init__(
            "source ensemble member failures; result persisted at " + saved.artifact.uri
        )
        self.saved = saved


class ForwardSourceEnsemble:
    def __init__(
        self,
        config: SourceEnsembleConfig,
        drift_model: DriftModel,
        attribution_model: AttributionModel,
    ) -> None:
        self.config = config
        self.drift = drift_model
        self.attribution = attribution_model

    def plan(self, request: SourceEnsembleRequest) -> SourceEnsemblePlan:
        return plan_source_ensemble(self.config, request, self.drift.component_metadata())

    def run(self, request: SourceEnsembleRequest) -> SavedSourceEnsemble:
        return self.execute(self.plan(request))

    def replay(self, artifact: ArtifactRef) -> SavedSourceEnsemble:
        return self.execute(SourceEnsemblePlan.model_validate_json(read_artifact(artifact)))

    def execute(self, plan: SourceEnsemblePlan) -> SavedSourceEnsemble:
        if plan.drift != self.drift.component_metadata():
            raise ValueError("saved drift version/configuration differs from current adapter")
        directory = plan.config.output_root / plan.plan_id / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        plan_ref = write_artifact(directory / "plan.json", plan.model_dump_json().encode())
        results = []
        for index, member in enumerate(plan.members):
            try:
                if member.request is None:
                    raise ValueError(member.planning_error)
                trace = execute_forward(self.drift, member.request, plan.drift, directory, index)
                result = SourceMemberResult(
                    member_id=member.member_id,
                    candidate_id=member.candidate_id,
                    member_index=member.member_index,
                    status="succeeded",
                    trace=trace,
                    evidence=member_evidence(plan, trace),
                )
            except Exception as error:
                result = SourceMemberResult(
                    member_id=member.member_id,
                    candidate_id=member.candidate_id,
                    member_index=member.member_index,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            write_artifact(directory / f"member-{index}.json", result.model_dump_json().encode())
            results.append(result)
        result_tuple = tuple(results)
        summaries = tuple(
            summarize_candidate(plan, c.candidate_id, result_tuple) for c in plan.root.candidates
        )
        rankings = []
        credits = dict.fromkeys((c.candidate_id for c in plan.root.candidates), 0.0)
        score_samples: dict[str, list[float]] = {key: [] for key in credits}
        comparable = 0
        for i in range(plan.config.member_count):
            group = tuple(r for r in result_tuple if r.member_index == i)
            if any(r.status != "succeeded" or r.evidence is None for r in group):
                continue
            ranking = self._rank(
                plan,
                tuple(r.evidence for r in group if r.evidence),
                tuple(r.trace.simulation for r in group if r.trace),
            )
            scored = tuple(c for c in ranking.ranked_candidates if c.score is not None)
            is_comparable = ranking.decision == Decision.RANKED and {
                c.candidate_id for c in scored
            } == set(credits)
            rankings.append(
                MemberRanking(member_index=i, attribution=ranking, comparable=is_comparable)
            )
            if ranking.decision != Decision.RANKED or {c.candidate_id for c in scored} != set(
                credits
            ):
                continue
            comparable += 1
            best_rank = min(c.rank for c in scored)
            winners = tuple(c for c in scored if c.rank == best_rank)
            for candidate in winners:
                credits[candidate.candidate_id] += 1 / len(winners)
            for candidate in scored:
                assert candidate.score is not None
                score_samples[candidate.candidate_id].append(candidate.score)
        summaries = tuple(
            s.model_copy(
                update={
                    "comparable_member_count": comparable,
                    "first_rank_member_frequency": credits[s.candidate_id] / comparable
                    if comparable
                    else None,
                }
            )
            for s in summaries
        )
        warnings = [
            "Member frequencies describe rank stability, not calibrated responsibility estimates.",
            "Discrete release sampling weights AIS points equally, not elapsed time; "
            "DATASET_VALIDATION_REQUIRED",
        ]
        if not comparable:
            warnings.append(
                "No comparable successful member indices; rank frequencies are missing."
            )
        overlapping = []
        keys = sorted(score_samples)
        for i, first in enumerate(keys):
            for second in keys[i + 1 :]:
                a, b = score_samples[first], score_samples[second]
                if a and b and max(min(a), min(b)) <= min(max(a), max(b)):
                    overlapping.append(
                        f"overlapping observed member-score ranges: {first}, {second}"
                    )
        warnings.extend(overlapping)
        ranking = self._rank(
            plan,
            tuple(s.evidence for s in summaries),
            tuple(r.trace.simulation for r in result_tuple if r.trace),
        )
        reasons = [
            f"Evidence quality blocks candidate {s.candidate_id}: "
            + ("; ".join(s.quality_issues) or "supplemental evidence ineligible")
            for s in summaries
            if not s.evidence.eligible_for_ranking
        ]
        if overlapping and plan.config.overlapping_score_ranges_policy == "abstain":
            reasons.extend(overlapping)
        if reasons:
            ranking = ranking.model_copy(
                update={"decision": Decision.ABSTAINED, "abstention_reason": "; ".join(reasons)}
            )
        ensemble_result = SourceEnsembleResult(
            plan=plan_ref,
            members=result_tuple,
            summaries=summaries,
            attribution=ranking,
            ranker=self.attribution.component_metadata(),
            member_rankings=tuple(rankings),
            comparable_member_count=comparable,
            warnings=tuple(warnings),
        )
        saved = SavedSourceEnsemble(
            ensemble_result,
            write_artifact(directory / "result.json", ensemble_result.model_dump_json().encode()),
        )
        if (
            any(r.status == "failed" for r in results)
            and plan.config.failure_policy == "raise_after_recording"
        ):
            raise SourceEnsembleExecutionError(saved)
        return saved

    def _rank(
        self,
        plan: SourceEnsemblePlan,
        evidence: tuple[CandidateMetricEvidence, ...],
        simulations: tuple[DriftSimulation, ...],
    ) -> AttributionResult:
        return self.attribution.rank(
            AttributionRequest(
                observation=plan.root.observation,
                candidates=plan.root.candidates,
                simulations=simulations,
                candidate_metrics=evidence,
            )
        )


# Keep the previous helper import available for callers; implementation is now shared.
_snapshot = snapshot_simulation
