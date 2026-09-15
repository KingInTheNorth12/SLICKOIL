"""Configuration-driven release hypotheses derived from canonical AIS points."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from oilspill.config import ComponentConfig
from oilspill.domain.common import ComponentMetadata, FrozenModel, MetadataEntry, TimeRange
from oilspill.domain.drift import ReleaseHypothesis
from oilspill.requests import ReleaseGenerationRequest


class AISPointReleaseGeneratorConfig(FrozenModel):
    point_policy: Literal["earliest_in_window", "latest_in_window", "closest_to_central_estimate"]
    validation_note: str


class AISPointReleaseHypothesisGenerator:
    """Select an observed AIS point using an explicit policy; no interpolation is implied."""

    def __init__(self, config: AISPointReleaseGeneratorConfig) -> None:
        self._config = config

    def generate(self, request: ReleaseGenerationRequest) -> ReleaseHypothesis:
        interval = request.observation.discharge_time.interval
        eligible = tuple(
            point
            for point in request.candidate.track.points
            if interval.start <= point.observed_at <= interval.end
        )
        if not eligible:
            raise ValueError("candidate has no observed AIS point in the discharge-time window")
        if self._config.point_policy == "earliest_in_window":
            selected = eligible[0]
        elif self._config.point_policy == "latest_in_window":
            selected = eligible[-1]
        else:
            central = request.observation.discharge_time.central_estimate
            if central is None:
                raise ValueError("closest_to_central_estimate requires a central estimate")
            selected = min(
                eligible,
                key=lambda point: (
                    abs((point.observed_at - central).total_seconds()),
                    point.observed_at,
                ),
            )
        identity = sha256(
            f"{request.candidate.candidate_id}:{selected.observed_at.isoformat()}:"
            f"{selected.position.model_dump_json()}".encode()
        ).hexdigest()[:16]
        return ReleaseHypothesis(
            release_id=f"ais-release:{identity}",
            geometry=selected.position,
            interval=TimeRange(start=selected.observed_at, end=selected.observed_at),
            source_candidate_id=request.candidate.candidate_id,
            assumptions=(
                "Release location/time is an explicitly selected observed AIS point; it is not "
                "a validated discharge estimate.",
                self._config.validation_note,
            ),
        )

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="ais_point_release",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="canonical-ais",
            configuration_sha256=digest,
            attributes=(MetadataEntry(key="validation_note", value=self._config.validation_note),),
        )


def create_ais_point_release(
    config: ComponentConfig,
) -> AISPointReleaseHypothesisGenerator:
    return AISPointReleaseHypothesisGenerator(
        AISPointReleaseGeneratorConfig.model_validate(config.settings)
    )
