"""Enumerate observed AIS releases; no inferred vessel positions."""

from collections import Counter
from hashlib import sha256
from typing import Literal

from oilspill.config import ComponentConfig
from oilspill.domain.ais import AISPoint
from oilspill.domain.common import (
    ComponentMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    TimeRange,
)
from oilspill.domain.drift import ReleaseHypothesis
from oilspill.domain.source_inference import ReleaseHypothesisSet
from oilspill.requests import ReleaseGenerationRequest


class AISReleaseConfig(FrozenModel):
    position_policy: Literal["unique_observed_points_only"]
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]


def component_metadata(name: str, config: FrozenModel, implementation: str) -> ComponentMetadata:
    return ComponentMetadata(
        name=name,
        version="1",
        implementation=implementation,
        configuration_sha256=sha256(config.model_dump_json().encode()).hexdigest(),
        attributes=(MetadataEntry(key="validation_status", value="DATASET_VALIDATION_REQUIRED"),),
    )


def eligible_points(request: ReleaseGenerationRequest) -> tuple[AISPoint, ...]:
    """Legacy ensemble eligibility, before unique-time and gap checks."""
    window = request.observation.discharge_time.interval
    return tuple(
        p
        for p in request.candidate.track.points
        if window.start <= p.observed_at <= window.end
        and p.observed_at < request.observation.observed_at
    )


class AISReleaseHypothesisGenerator:
    """Plural generator, leaving the existing singular generator contract intact."""

    def __init__(self, config: AISReleaseConfig) -> None:
        self.config = config

    def enumerate(self, request: ReleaseGenerationRequest) -> ReleaseHypothesisSet:
        candidate = request.candidate
        if candidate.observation_id != request.observation.observation_id:
            raise ValueError("candidate observation mismatch")
        interval = request.observation.discharge_time.interval
        track = candidate.track
        issues = []
        if interval.start < track.interval.start or interval.end > track.interval.end:
            issues.append("discharge interval outside observed AIS coverage")
        if any(
            g.start < interval.end and g.end > interval.start for g in track.coverage.detected_gaps
        ):
            issues.append("declared AIS gaps overlap discharge interval")
        points = eligible_points(request)
        counts = Counter(p.observed_at for p in points)
        if any(n != 1 for n in counts.values()):
            issues.append("ambiguous duplicate AIS timestamps excluded")
        # Existing builder supplies observed points only. Do not trust synthetic positions from
        # another reconstructor without an explicit validated capability in this version.
        if track.coverage.interpolated_point_count or track.coverage.interpolated_intervals:
            issues.append(
                "interpolated track unsupported; observed/interpolated points not identifiable"
            )
            points = ()
        hypotheses = []
        for point in points:
            if counts[point.observed_at] != 1 or any(
                g.start < point.observed_at < g.end for g in track.coverage.detected_gaps
            ):
                continue
            identity = sha256(
                (candidate.candidate_id + point.model_dump_json()).encode()
            ).hexdigest()
            hypotheses.append(
                ReleaseHypothesis(
                    release_id=f"observed-release:{identity}",
                    geometry=point.position,
                    interval=TimeRange(start=point.observed_at, end=point.observed_at),
                    source_candidate_id=candidate.candidate_id,
                    assumptions=(
                        self.config.validation_note,
                        "Observed AIS point; hypothetical instantaneous discharge; "
                        "DATASET_VALIDATION_REQUIRED",
                    ),
                )
            )
        if not hypotheses:
            issues.append("no eligible unique observed AIS positions before observation time")
        return ReleaseHypothesisSet(
            candidate_id=candidate.candidate_id,
            hypotheses=tuple(hypotheses),
            coverage_issues=tuple(issues),
            generator=self.component_metadata(),
        )

    def component_metadata(self) -> ComponentMetadata:
        return component_metadata(
            "observed_ais_releases", self.config, f"{__name__}.AISReleaseHypothesisGenerator"
        )


def create_observed_ais_releases(config: ComponentConfig) -> AISReleaseHypothesisGenerator:
    return AISReleaseHypothesisGenerator(AISReleaseConfig.model_validate(config.settings))
