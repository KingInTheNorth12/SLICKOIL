"""Serial, bounded forward search over explicit recorded AIS releases."""

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from oilspill.artifacts import read_artifact, write_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.ais import CandidateDecision
from oilspill.domain.attribution import EvidenceDirection
from oilspill.domain.common import ArtifactRef, ComponentMetadata, Decision
from oilspill.domain.source_inference import (
    HypothesisEvaluation,
    SourceInferenceResult,
)
from oilspill.registry import SourceSearchDependencies
from oilspill.requests import (
    ForwardTraceRequest,
    ObservationComparisonRequest,
    ReleaseGenerationRequest,
    SourceSearchRequest,
)
from oilspill.source_tracing.execution import execute_forward
from oilspill.source_tracing.hypotheses import component_metadata
from oilspill.source_tracing.search_models import (
    PlannedHypothesis,
    SearchIteration,
    SearchReplayPlan,
    SearchSelection,
    SourceSearchConfig,
    SourceSearchPlan,
)


class SourceSearchExecutionError(RuntimeError):
    def __init__(self, result: SourceInferenceResult) -> None:
        super().__init__("source-search failures recorded; see " + result.provenance.uri)
        self.result = result


class IterativeForwardSourceSearch:
    def __init__(self, config: SourceSearchConfig, dependencies: SourceSearchDependencies) -> None:
        self.config = config
        self.dependencies = dependencies

    def component_metadata(self) -> ComponentMetadata:
        return component_metadata(
            "iterative_forward", self.config, f"{__name__}.IterativeForwardSourceSearch"
        )

    def plan(self, request: SourceSearchRequest) -> SourceSearchPlan:
        ids = [c.candidate_id for c in request.candidates]
        if len(ids) != len(set(ids)) or any(
            c.decision != CandidateDecision.RETAINED for c in request.candidates
        ):
            raise ValueError("search requires unique retained candidates")
        if any(c.observation_id != request.observation.observation_id for c in request.candidates):
            raise ValueError("candidate observation mismatch")
        if any(b.observation != request.observation for b in request.backward_evidence):
            raise ValueError("backward evidence observation mismatch")
        if len({f.field_id for f in request.forcing}) != len(request.forcing):
            raise ValueError("duplicate forcing IDs")
        request = request.model_copy(
            update={"candidates": tuple(sorted(request.candidates, key=lambda c: c.candidate_id))}
        )
        deps = self.dependencies
        drift, generator, likelihood = (
            deps.drift.component_metadata(),
            deps.releases.component_metadata(),
            deps.likelihood.component_metadata(),
        )
        if any(m.configuration_sha256 is None for m in (drift, generator, likelihood)):
            raise ValueError("resolved component configuration checksums required")
        sets = tuple(
            deps.releases.enumerate(
                ReleaseGenerationRequest(observation=request.observation, candidate=c)
            )
            for c in request.candidates
        )
        identity = sha256(
            (
                request.model_dump_json()
                + self.config.model_dump_json()
                + drift.model_dump_json()
                + generator.model_dump_json()
                + likelihood.model_dump_json()
                + "".join(s.model_dump_json() for s in sets)
            ).encode()
        ).hexdigest()
        # Context-only backward evidence must not alter even a stochastic forward run.
        engine_identity = sha256(
            (
                request.model_copy(update={"backward_evidence": ()}).model_dump_json()
                + self.config.model_dump_json()
                + drift.model_dump_json()
            ).encode()
        ).hexdigest()
        candidates = {c.candidate_id: c for c in request.candidates}
        hypotheses = []
        for group in sets:
            for release in group.hypotheses:
                seed = int(
                    sha256(f"{engine_identity}:engine:{release.release_id}".encode()).hexdigest()[
                        :8
                    ],
                    16,
                )
                hypotheses.append(
                    PlannedHypothesis(
                        request=ForwardTraceRequest(
                            observation=request.observation,
                            candidate=candidates[group.candidate_id],
                            release=release,
                            forcing=request.forcing,
                            random_seed=seed,
                            model_parameters=request.model_parameters,
                            configured_uncertainty=request.configured_uncertainty,
                        )
                    )
                )
        return SourceSearchPlan(
            plan_id=identity,
            root=request,
            config=self.config,
            hypothesis_sets=sets,
            hypotheses=tuple(hypotheses),
            drift=drift,
            generator=generator,
            likelihood=likelihood,
            resolved_components=deps.resolved_components,
        )

    def search(self, request: SourceSearchRequest) -> SourceInferenceResult:
        plan = self.plan(request)
        return self._execute(plan, None)

    def replay(self, artifact: ArtifactRef) -> SourceInferenceResult:
        """Re-execute the recorded requests/schedule; never enumerate or refine again."""
        saved = SearchReplayPlan.model_validate_json(read_artifact(artifact))
        plan = SourceSearchPlan.model_validate_json(read_artifact(saved.plan))
        return self._execute(plan, saved)

    def _execute(
        self, plan: SourceSearchPlan, replay: SearchReplayPlan | None
    ) -> SourceInferenceResult:
        deps = self.dependencies
        if (
            plan.config != self.config
            or plan.drift != deps.drift.component_metadata()
            or plan.generator != deps.releases.component_metadata()
            or plan.likelihood != deps.likelihood.component_metadata()
            or plan.resolved_components != deps.resolved_components
        ):
            raise ValueError("saved search component/configuration mismatch")
        directory = Path(plan.config.output_root) / plan.plan_id / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        plan_ref = write_artifact(directory / "plan.json", plan.model_dump_json().encode())
        by_id = {h.request.release.release_id: h.request for h in plan.hypotheses}
        evaluations: list[HypothesisEvaluation] = []
        rounds: list[SearchIteration] = []
        artifacts: list[ArtifactRef] = []
        seen: set[str] = set()
        termination = "refinement_levels_exhausted"
        count = len(replay.iterations) if replay else len(plan.config.refinement_levels) + 1
        for iteration in range(count):
            selections = (
                replay.iterations[iteration].selections
                if replay
                else self._select(plan, evaluations, iteration)
            )
            if not selections:
                termination = "no_new_hypotheses"
                break
            if any(
                s.release_id not in by_id or s.release_id in seen or not set(s.parent_ids) <= seen
                for s in selections
            ):
                raise ValueError("invalid replay selection or parent identity")
            if len({s.release_id for s in selections}) != len(selections):
                raise ValueError("duplicate iteration selection")
            remaining = plan.config.maximum_evaluations - len(evaluations)
            if replay and len(selections) > remaining:
                raise ValueError("replay exceeds saved evaluation budget")
            selections = selections[:remaining]
            if not selections:
                termination = "evaluation_budget_exhausted"
                break
            # Record the exact work before executing it, including parent lineage.
            write_artifact(
                directory / f"iteration-{iteration}-requests.json",
                SearchIteration(iteration=iteration, selections=selections, retained_ids=())
                .model_dump_json()
                .encode(),
            )
            for selection in selections:
                request = by_id[selection.release_id]
                trace = None
                evidence = None
                error = None
                try:
                    if plan.config.require_forcing and not request.forcing:
                        raise ValueError("required forcing absent")
                    if any(
                        f.interval.start > request.release.interval.start
                        or f.interval.end < request.observation.observed_at
                        for f in request.forcing
                    ):
                        raise ValueError("forcing does not cover forward integration interval")
                    trace = execute_forward(
                        deps.drift, request, plan.drift, directory, len(evaluations)
                    )
                    evidence = deps.likelihood.compare(
                        ObservationComparisonRequest(
                            observation=request.observation, simulation=trace.simulation
                        )
                    )
                    if (
                        evidence.observation_id != plan.root.observation.observation_id
                        or evidence.component != plan.likelihood
                        or evidence.release_id != request.release.release_id
                        or evidence.simulation_id != trace.simulation.simulation_id
                    ):
                        raise ValueError("similarity provenance mismatch")
                    previous = next((e.evidence for e in evaluations if e.evidence), None)
                    if previous and (
                        evidence.primary_metric != previous.primary_metric
                        or evidence.primary.unit != previous.primary.unit
                        or evidence.primary.direction != previous.primary.direction
                    ):
                        raise ValueError(
                            "incomparable primary metric definitions across hypotheses"
                        )
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"
                    evidence = None
                evaluation = HypothesisEvaluation(
                    hypothesis=request.release,
                    iteration=iteration,
                    parent_ids=selection.parent_ids,
                    random_seed=request.random_seed,
                    trace=trace,
                    evidence=evidence,
                    error=error,
                )
                write_artifact(
                    directory / f"evaluation-{len(evaluations)}.json",
                    evaluation.model_dump_json().encode(),
                )
                evaluations.append(evaluation)
                seen.add(selection.release_id)
            retained = self._retained(plan, evaluations)
            record = SearchIteration(
                iteration=iteration,
                selections=selections,
                retained_ids=tuple(e.hypothesis.release_id for e in retained),
            )
            rounds.append(record)
            artifacts.append(
                write_artifact(
                    directory / f"iteration-{iteration}.json", record.model_dump_json().encode()
                )
            )
            if len(evaluations) == plan.config.maximum_evaluations and len(seen) < len(by_id):
                termination = "evaluation_budget_exhausted"
                break
        if replay:
            termination = replay.termination_reason
        manifest = SearchReplayPlan(
            plan=plan_ref, iterations=tuple(rounds), termination_reason=termination
        )
        provenance = write_artifact(directory / "replay.json", manifest.model_dump_json().encode())
        result = self._result(plan, evaluations, termination, provenance, tuple(artifacts))
        write_artifact(directory / "result.json", result.model_dump_json().encode())
        if plan.config.failure_policy == "raise_after_recording" and any(
            e.error for e in evaluations
        ):
            raise SourceSearchExecutionError(result)
        return result

    @staticmethod
    def _ordered(evaluations: list[HypothesisEvaluation]) -> list[HypothesisEvaluation]:
        usable = [
            e
            for e in evaluations
            if not e.error and e.evidence and e.evidence.ordering_value is not None
        ]
        return sorted(
            usable,
            key=lambda e: (e.evidence.ordering_value if e.evidence else 0, e.hypothesis.release_id),
        )

    def _retained(
        self, plan: SourceSearchPlan, evaluations: list[HypothesisEvaluation]
    ) -> list[HypothesisEvaluation]:
        return [
            e
            for c in plan.root.candidates
            for e in self._ordered(
                [e for e in evaluations if e.hypothesis.source_candidate_id == c.candidate_id]
            )[: plan.config.retain_per_candidate]
        ]

    def _select(
        self, plan: SourceSearchPlan, evaluations: list[HypothesisEvaluation], iteration: int
    ) -> tuple[SearchSelection, ...]:
        seen = {e.hypothesis.release_id for e in evaluations}
        retained = self._retained(plan, evaluations)
        groups = []
        for group in plan.hypothesis_sets:
            available = [h for h in group.hypotheses if h.release_id not in seen]
            parents = [
                e.hypothesis
                for e in retained
                if e.hypothesis.source_candidate_id == group.candidate_id
            ]
            level = plan.config.refinement_levels[iteration - 1] if iteration else None
            width = level.bin_width_seconds if level else plan.config.initial_bin_width_seconds
            origin = plan.root.observation.discharge_time.interval.start
            bins: dict[int, list[SearchSelection]] = {}
            by_id = {h.release_id: h for h in available}
            for hypothesis in available:
                parent_ids = tuple(
                    p.release_id
                    for p in parents
                    if level
                    and abs((p.interval.start - hypothesis.interval.start).total_seconds())
                    <= level.radius_seconds
                )
                if level and not parent_ids:
                    continue
                bucket = int((hypothesis.interval.start - origin).total_seconds() // width)
                bins.setdefault(bucket, []).append(
                    SearchSelection(release_id=hypothesis.release_id, parent_ids=parent_ids)
                )
            selected = []
            for bucket, choices in sorted(bins.items()):
                midpoint = origin + timedelta(seconds=(bucket + 0.5) * width)
                selected.append(
                    min(
                        choices,
                        key=lambda s: (
                            abs((by_id[s.release_id].interval.start - midpoint).total_seconds()),
                            by_id[s.release_id].interval.start,
                            s.release_id,
                        ),
                    )
                )
            groups.append(selected)
        # Round-robin candidate allocation prevents one dense track consuming the broad budget.
        return tuple(
            group[i]
            for i in range(max((len(g) for g in groups), default=0))
            for group in groups
            if i < len(group)
        )

    def _result(
        self,
        plan: SourceSearchPlan,
        evaluations: list[HypothesisEvaluation],
        termination: str,
        provenance: ArtifactRef,
        iterations: tuple[ArtifactRef, ...],
    ) -> SourceInferenceResult:
        ordered = self._ordered(evaluations)
        reasons = []
        warnings = [
            "DATASET_VALIDATION_REQUIRED: conditional search over available AIS; "
            "no confirmed origin or responsibility probability.",
            "Backward evidence is complementary context only and does not change forward scores.",
        ]
        if not plan.root.candidates:
            reasons.append("no retained candidates")
        for group in plan.hypothesis_sets:
            if group.coverage_issues:
                message = group.candidate_id + ": " + "; ".join(group.coverage_issues)
                warnings.append(message)
                if plan.config.coverage_policy == "abstain":
                    reasons.append(message)
            count = sum(e.hypothesis.source_candidate_id == group.candidate_id for e in ordered)
            if count < plan.config.minimum_successful_per_candidate:
                reasons.append(group.candidate_id + ": insufficient valid forward comparisons")
        if termination == "evaluation_budget_exhausted":
            warnings.append("evaluation budget exhausted; search incomplete")
            if plan.config.budget_policy == "abstain":
                reasons.append("evaluation budget exhausted")
        if not ordered:
            reasons.append("no usable forward evidence")
        else:
            best = ordered[0].evidence
            assert best and best.score is not None
            passes = (
                best.score >= plan.config.acceptance_bound
                if best.primary.direction == EvidenceDirection.HIGHER_IS_BETTER
                else best.score <= plan.config.acceptance_bound
            )
            if not passes:
                reasons.append("best fit fails configured acceptance bound")
            if len(ordered) > 1:
                second = ordered[1].evidence
                assert (
                    second and second.ordering_value is not None and best.ordering_value is not None
                )
                if second.ordering_value - best.ordering_value <= plan.config.ambiguity_tolerance:
                    reasons.append(
                        "ambiguous release hypotheses under configured raw-metric tolerance"
                    )
        evaluated_ids = {e.hypothesis.release_id for e in evaluations}
        remaining_count = len(plan.hypotheses) - len(evaluated_ids)
        if remaining_count:
            warnings.append(
                f"{remaining_count} enumerated hypotheses not evaluated; no global optimum claim."
            )
        return SourceInferenceResult(
            inference_id=f"source-search:{plan.plan_id}",
            observation_id=plan.root.observation.observation_id,
            evaluated_hypotheses=tuple(evaluations),
            hypothesis_sets=plan.hypothesis_sets,
            best_supported_hypothesis_ids=() if reasons else (ordered[0].hypothesis.release_id,),
            decision=Decision.ABSTAINED if reasons else Decision.RANKED,
            abstention_reasons=tuple(reasons),
            termination_reason=termination,
            unevaluated_hypothesis_ids=tuple(
                h.request.release.release_id
                for h in plan.hypotheses
                if h.request.release.release_id not in evaluated_ids
            ),
            backward_evidence=plan.root.backward_evidence,
            provenance=provenance,
            iterations=iterations,
            component=self.component_metadata(),
            warnings=tuple(warnings),
        )


def create_iterative_forward(
    config: ComponentConfig, dependencies: SourceSearchDependencies
) -> IterativeForwardSourceSearch:
    return IterativeForwardSourceSearch(
        SourceSearchConfig.model_validate(config.settings), dependencies
    )
