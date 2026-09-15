"""Named geometry similarity using existing metric definitions without new probabilities."""

import math
from typing import Literal

from pydantic import Field, model_validator

from oilspill.attribution.metrics import ForwardFitSimilarityMetric, HausdorffDistanceMetric
from oilspill.attribution.models import DistanceMetricConfig, GeometryMetricConfig
from oilspill.config import ComponentConfig
from oilspill.domain.attribution import MetricResult, MetricStatus
from oilspill.domain.common import ComponentMetadata, FrozenModel, Identifier
from oilspill.domain.drift import DriftMode, SimulationStatus
from oilspill.domain.source_inference import ObservationSimilarity
from oilspill.requests import ObservationComparisonRequest
from oilspill.source_tracing.hypotheses import component_metadata

MetricName = Literal["hausdorff_distance", "forward_fit_similarity"]


class GeometrySimilarityConfig(FrozenModel):
    metrics: tuple[MetricName, ...] = Field(min_length=1)
    primary_metric: MetricName
    distance: DistanceMetricConfig
    geometry: GeometryMetricConfig
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def consistent(self) -> "GeometrySimilarityConfig":
        if len(set(self.metrics)) != len(self.metrics) or self.primary_metric not in self.metrics:
            raise ValueError("unique metrics must include primary metric")
        return self


class GeometryObservationLikelihood:
    def __init__(self, config: GeometrySimilarityConfig) -> None:
        self.config = config
        self._distance = HausdorffDistanceMetric(config.distance)
        self._overlap = ForwardFitSimilarityMetric(config.geometry)

    def component_metadata(self) -> ComponentMetadata:
        return component_metadata(
            "geometry_similarity", self.config, f"{__name__}.GeometryObservationLikelihood"
        )

    def compare(self, request: ObservationComparisonRequest) -> ObservationSimilarity:
        simulation, observation = request.simulation, request.observation
        distribution = simulation.target_distribution
        if (
            simulation.mode != DriftMode.FORWARD_TRACE
            or simulation.status != SimulationStatus.SUCCEEDED
            or simulation.target_timestamp != observation.observed_at
            or distribution is None
            or distribution.timestamp != observation.observed_at
            or distribution.particle_count <= 0
        ):
            raise ValueError("comparison requires successful forward support at observation time")
        results = []
        for name in self.config.metrics:
            metric = self._distance if name == "hausdorff_distance" else self._overlap
            value = metric.compute(
                distribution.comparison_geometry,
                observation.geometry.polygon,
                source_simulation_ids=(simulation.simulation_id,),
            )
            if value.raw_value is not None and not math.isfinite(value.raw_value):
                value = MetricResult(
                    name=value.name,
                    status=MetricStatus.INVALID,
                    raw_value=None,
                    direction=value.direction,
                    unit=value.unit,
                    explanation="Non-finite metric; no value substituted.",
                    provenance=value.provenance,
                )
            # Existing missing/invalid metrics do not always attach simulation IDs.
            results.append(
                value.model_copy(update={"source_simulation_ids": (simulation.simulation_id,)})
            )
        return ObservationSimilarity(
            observation_id=observation.observation_id,
            release_id=simulation.release.release_id,
            simulation_id=simulation.simulation_id,
            metrics=tuple(results),
            primary_metric=self.config.primary_metric,
            component=self.component_metadata(),
        )


def create_geometry_similarity(config: ComponentConfig) -> GeometryObservationLikelihood:
    return GeometryObservationLikelihood(GeometrySimilarityConfig.model_validate(config.settings))
