"""Reusable attribution evidence metrics; ranking remains a separate subsystem."""

from oilspill.attribution.metrics import (
    AISTrajectoryConsistencyMetric,
    BackwardFitSimilarityMetric,
    DirectionalConsistencyMetric,
    ForwardFitSimilarityMetric,
    HausdorffDistanceMetric,
    SourceDistanceMetric,
    SpatialOverlapMetric,
    TemporalConsistencyMetric,
)
from oilspill.attribution.models import (
    DirectionMetricConfig,
    DistanceMetricConfig,
    GeometryMetricConfig,
    MinMaxNormalization,
    TemporalMetricConfig,
)
from oilspill.attribution.posterior import (
    PosteriorSourceSupportConfig,
    PosteriorSourceSupportEvidenceBuilder,
)
from oilspill.attribution.weighted import (
    FeatureWeight,
    WeightedTrajectoryAttribution,
    WeightedTrajectoryAttributionConfig,
)
from oilspill.domain.attribution import MetricResult, MetricStatus

__all__ = [
    "AISTrajectoryConsistencyMetric",
    "BackwardFitSimilarityMetric",
    "DirectionMetricConfig",
    "DirectionalConsistencyMetric",
    "DistanceMetricConfig",
    "ForwardFitSimilarityMetric",
    "FeatureWeight",
    "GeometryMetricConfig",
    "HausdorffDistanceMetric",
    "MetricResult",
    "MetricStatus",
    "MinMaxNormalization",
    "PosteriorSourceSupportConfig",
    "PosteriorSourceSupportEvidenceBuilder",
    "SourceDistanceMetric",
    "SpatialOverlapMetric",
    "TemporalConsistencyMetric",
    "TemporalMetricConfig",
    "WeightedTrajectoryAttribution",
    "WeightedTrajectoryAttributionConfig",
]
