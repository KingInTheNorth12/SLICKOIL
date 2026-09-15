"""Typed configuration and replay records for discrete forward search."""

from typing import Literal

from pydantic import Field, model_validator

from oilspill.config import ComponentConfig
from oilspill.domain.common import ArtifactRef, ComponentMetadata, FrozenModel, Identifier
from oilspill.domain.source_inference import ReleaseHypothesisSet
from oilspill.requests import ForwardTraceRequest, SourceSearchRequest


class RefinementLevel(FrozenModel):
    radius_seconds: float = Field(gt=0, allow_inf_nan=False)
    bin_width_seconds: float = Field(gt=0, allow_inf_nan=False)


class SourceSearchConfig(FrozenModel):
    output_root: Identifier
    initial_bin_width_seconds: float = Field(gt=0, allow_inf_nan=False)
    refinement_levels: tuple[RefinementLevel, ...]
    retain_per_candidate: int = Field(gt=0)
    maximum_evaluations: int = Field(gt=0, le=2**32)
    minimum_successful_per_candidate: int = Field(gt=0)
    acceptance_bound: float = Field(allow_inf_nan=False)
    ambiguity_tolerance: float = Field(ge=0, allow_inf_nan=False)
    coverage_policy: Literal["warn", "abstain"]
    budget_policy: Literal["warn", "abstain"]
    failure_policy: Literal["record_and_continue", "raise_after_recording"]
    require_forcing: bool
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def consistent(self) -> "SourceSearchConfig":
        width = self.initial_bin_width_seconds
        radius = float("inf")
        for level in self.refinement_levels:
            if level.bin_width_seconds >= width or level.radius_seconds >= radius:
                raise ValueError("refinement widths and radii must strictly decrease")
            width, radius = level.bin_width_seconds, level.radius_seconds
        if self.minimum_successful_per_candidate > self.maximum_evaluations:
            raise ValueError("minimum successes exceed total evaluation budget")
        return self


class PlannedHypothesis(FrozenModel):
    request: ForwardTraceRequest


class SourceSearchPlan(FrozenModel):
    schema_version: Literal["1"] = "1"
    planner_version: Literal["observed-ais-time-bins-v1"] = "observed-ais-time-bins-v1"
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    root: SourceSearchRequest
    config: SourceSearchConfig
    hypothesis_sets: tuple[ReleaseHypothesisSet, ...]
    hypotheses: tuple[PlannedHypothesis, ...]
    drift: ComponentMetadata
    generator: ComponentMetadata
    likelihood: ComponentMetadata
    resolved_components: tuple[ComponentConfig, ...]

    @model_validator(mode="after")
    def consistent(self) -> "SourceSearchPlan":
        candidates = {c.candidate_id: c for c in self.root.candidates}
        sets = {s.candidate_id: s for s in self.hypothesis_sets}
        if (
            len(candidates) != len(self.root.candidates)
            or len(sets) != len(self.hypothesis_sets)
            or set(sets) != set(candidates)
        ):
            raise ValueError("candidate/coverage identity mismatch")
        expected = {h.release_id: h for s in self.hypothesis_sets for h in s.hypotheses}
        if len(expected) != sum(len(s.hypotheses) for s in self.hypothesis_sets):
            raise ValueError("duplicate enumerated hypothesis IDs")
        if any(s.generator != self.generator for s in self.hypothesis_sets):
            raise ValueError("generator provenance mismatch")
        ids = [h.request.release.release_id for h in self.hypotheses]
        seeds = [h.request.random_seed for h in self.hypotheses]
        if any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("engine seed outside uint32 range")
        if len(ids) != len(set(ids)) or set(ids) != set(expected) or len(set(seeds)) != len(seeds):
            raise ValueError("hypothesis identity/seed mismatch")
        for item in self.hypotheses:
            r = item.request
            candidate = r.candidate
            if (
                candidate is None
                or r.observation != self.root.observation
                or candidate != candidates.get(candidate.candidate_id)
                or r.release != expected[r.release.release_id]
                or r.release.source_candidate_id != candidate.candidate_id
                or r.release.geometry.geometry.type != "Point"
                or r.release.interval.start != r.release.interval.end
                or not self.root.observation.discharge_time.interval.start
                <= r.release.interval.start
                < self.root.observation.observed_at
                or r.release.interval.end > self.root.observation.discharge_time.interval.end
                or r.forcing != self.root.forcing
                or r.model_parameters != self.root.model_parameters
                or r.configured_uncertainty != self.root.configured_uncertainty
            ):
                raise ValueError("planned request disagrees with canonical search inputs")
        return self


class SearchSelection(FrozenModel):
    release_id: Identifier
    parent_ids: tuple[str, ...]


class SearchIteration(FrozenModel):
    iteration: int = Field(ge=0)
    selections: tuple[SearchSelection, ...]
    retained_ids: tuple[str, ...]


class SearchReplayPlan(FrozenModel):
    schema_version: Literal["1"] = "1"
    plan: ArtifactRef
    iterations: tuple[SearchIteration, ...]
    termination_reason: Identifier

    @model_validator(mode="after")
    def consistent(self) -> "SearchReplayPlan":
        if tuple(r.iteration for r in self.iterations) != tuple(range(len(self.iterations))):
            raise ValueError("replay iterations must be contiguous and ordered")
        return self
