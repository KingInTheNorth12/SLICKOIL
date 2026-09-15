"""Build transparent candidate evidence from canonical trace results."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from oilspill.attribution.metrics import (
    AISTrajectoryConsistencyMetric,
    BackwardFitSimilarityMetric,
    ForwardFitSimilarityMetric,
    HausdorffDistanceMetric,
    SourceDistanceMetric,
    TemporalConsistencyMetric,
)
from oilspill.attribution.models import (
    DistanceMetricConfig,
    GeometryMetricConfig,
    TemporalMetricConfig,
)
from oilspill.config import ComponentConfig
from oilspill.domain.attribution import CandidateMetricEvidence, MetricResult
from oilspill.domain.common import ComponentMetadata, FrozenModel, MetadataEntry
from oilspill.requests import AttributionEvidenceRequest

MetricName = Literal[
    "hausdorff_distance",
    "source_distance",
    "temporal_consistency",
    "forward_fit_similarity",
    "backward_fit_similarity",
    "ais_trajectory_consistency",
]


class TraceMetricEvidenceConfig(FrozenModel):
    metrics: tuple[MetricName, ...] = Field(min_length=1)
    distance: DistanceMetricConfig
    geometry: GeometryMetricConfig
    temporal: TemporalMetricConfig

    @model_validator(mode="after")
    def _unique_metrics(self) -> TraceMetricEvidenceConfig:
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("evidence metric names must be unique")
        return self


class TraceMetricEvidenceBuilder:
    """Compute only explicitly configured metrics; missing evidence remains missing."""

    def __init__(self, config: TraceMetricEvidenceConfig) -> None:
        self._config = config

    def build(self, request: AttributionEvidenceRequest) -> CandidateMetricEvidence:
        forward = request.forward_trace
        backward = request.backward_trace
        forward_ids = (forward.simulation.simulation_id,) if forward else ()
        backward_ids = (backward.simulation.simulation_id,) if backward else ()
        simulation_ids = (*forward_ids, *backward_ids)
        results: list[MetricResult] = []
        for name in self._config.metrics:
            if name == "hausdorff_distance":
                result = HausdorffDistanceMetric(self._config.distance).compute(
                    forward.comparison_ready_geometry if forward else None,
                    request.observation.geometry.polygon,
                    source_simulation_ids=forward_ids,
                )
            elif name == "source_distance":
                result = SourceDistanceMetric(self._config.distance).compute(
                    backward.plausible_source_geometry if backward else None,
                    request.candidate.track.path,
                    source_simulation_ids=backward_ids,
                )
            elif name == "temporal_consistency":
                result = TemporalConsistencyMetric(self._config.temporal).compute(
                    request.observation.discharge_time.interval,
                    request.candidate.track.interval,
                )
            elif name == "forward_fit_similarity":
                result = ForwardFitSimilarityMetric(self._config.geometry).compute(
                    forward.comparison_ready_geometry if forward else None,
                    request.observation.geometry.polygon,
                    source_simulation_ids=forward_ids,
                )
            elif name == "backward_fit_similarity":
                result = BackwardFitSimilarityMetric(self._config.distance).compute(
                    backward.plausible_source_paths[0]
                    if backward and backward.plausible_source_paths
                    else None,
                    request.candidate.track.path,
                    source_simulation_ids=backward_ids,
                )
            else:
                trajectory = (
                    backward.particle_trajectories[0]
                    if backward and backward.particle_trajectories
                    else None
                )
                result = AISTrajectoryConsistencyMetric(self._config.distance).compute(
                    trajectory,
                    request.candidate.track,
                    source_simulation_ids=simulation_ids,
                )
            results.append(result)
        return CandidateMetricEvidence(
            candidate_id=request.candidate.candidate_id,
            metrics=tuple(results),
        )

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="trace_metrics",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="canonical-attribution-metrics",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="configured_metrics", value=",".join(self._config.metrics)),
            ),
        )


def create_trace_metric_evidence(config: ComponentConfig) -> TraceMetricEvidenceBuilder:
    return TraceMetricEvidenceBuilder(TraceMetricEvidenceConfig.model_validate(config.settings))
