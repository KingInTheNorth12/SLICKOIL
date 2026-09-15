"""Auditable vessel evidence against an uncalibrated hybrid source posterior."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Literal

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import Field, model_validator
from rasterio.transform import xy
from rasterio.warp import transform
from shapely.geometry import MultiPoint, Point

from oilspill.artifacts import local_path_from_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.attribution import (
    CandidateMetricEvidence,
    EvidenceDirection,
    MetricResult,
    MetricStatus,
)
from oilspill.domain.common import ComponentMetadata, FrozenModel, MetadataEntry, TimeRange
from oilspill.domain.geospatial import Duration, SpatialGeometry
from oilspill.eulerian.initialization import file_artifact
from oilspill.requests import AttributionEvidenceRequest

PosteriorMetricName = Literal[
    "posterior_source_support",
    "posterior_source_distance",
    "posterior_temporal_overlap",
    "track_coverage_quality",
    "ais_gap_seconds",
]
FloatArray = NDArray[np.float64]


class PosteriorSourceSupportConfig(FrozenModel):
    metrics: tuple[PosteriorMetricName, ...] = Field(min_length=1)
    support_aggregation: Literal["mean", "maximum"]
    raster_sampling: Literal["nearest"]
    posterior_cell_relative_threshold: float = Field(
        gt=0.0, le=1.0, allow_inf_nan=False
    )
    distance_kernel_sigma_metres: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    gap_reference_duration: Duration
    validation_note: str = Field(min_length=1)
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def _validated(self) -> PosteriorSourceSupportConfig:
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("posterior evidence metrics must be unique")
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("posterior evidence must mark DATASET_VALIDATION_REQUIRED")
        if _seconds(self.gap_reference_duration) <= 0:
            raise ValueError("AIS gap reference duration must be positive")
        return self


class PosteriorSourceSupportEvidenceBuilder:
    """Sample source support without interpreting the result as responsibility."""

    def __init__(self, config: PosteriorSourceSupportConfig) -> None:
        self._config = config

    def build(self, request: AttributionEvidenceRequest) -> CandidateMetricEvidence:
        posterior = request.source_posterior
        if posterior is None:
            raise ValueError("posterior source-support evidence requires a source posterior")
        quantiles = posterior.release_time_quantiles
        raster = posterior.source_probability_raster
        if quantiles is None or raster is None:
            raise ValueError("source posterior requires raster and release-time quantiles")
        interval = TimeRange(start=quantiles.p10, end=quantiles.p90)
        path = local_path_from_artifact(raster.artifact)
        if not path.is_file() or file_artifact(path).sha256 != raster.artifact.sha256:
            raise ValueError("posterior raster is missing or fails checksum validation")
        with rasterio.open(path) as dataset:
            values = np.asarray(dataset.read(1), dtype=np.float64)
            if dataset.nodata is not None:
                values[values == dataset.nodata] = 0.0
            values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
            maximum = float(values.max(initial=0.0))
            normalized = values / maximum if maximum > 0 else values
            interval_points = tuple(
                point
                for point in request.candidate.track.points
                if interval.start <= point.observed_at <= interval.end
            )
            samples = tuple(
                _sample_nearest(dataset, normalized, point.position)
                for point in interval_points
            )
            all_points = tuple(
                _projected_point(dataset, point.position)
                for point in request.candidate.track.points
            )
            threshold = self._config.posterior_cell_relative_threshold
            rows, columns = np.nonzero(normalized >= threshold)
            centers = tuple(
                Point(float(x_value), float(y_value))
                for x_value, y_value in zip(
                    *xy(dataset.transform, rows, columns, offset="center"), strict=True
                )
            )
        support = _aggregate(samples, self._config.support_aggregation)
        distance = (
            min(point.distance(MultiPoint(centers)) for point in all_points)
            if centers and all_points
            else None
        )
        overlap = _temporal_overlap(request.candidate.track.interval, interval)
        coverage = request.candidate.track.coverage
        coverage_quality = coverage.source_point_count / (
            coverage.source_point_count + coverage.interpolated_point_count
        )
        maximum_gap = _maximum_gap_seconds(request)
        metrics = {
            "posterior_source_support": MetricResult(
                name="posterior_source_support",
                status=MetricStatus.VALID,
                raw_value=support,
                normalized_value=support,
                unit="relative_posterior_raster_support",
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation=(
                    f"{self._config.support_aggregation} nearest-cell posterior support at "
                    "observed AIS positions in the P10-P90 release interval; uncalibrated"
                ),
                provenance=self._provenance(posterior.posterior_id, raster.artifact.sha256),
            ),
            "posterior_source_distance": _distance_metric(
                distance,
                self._config.distance_kernel_sigma_metres,
                self._provenance(posterior.posterior_id, raster.artifact.sha256),
            ),
            "posterior_temporal_overlap": MetricResult(
                name="posterior_temporal_overlap",
                status=MetricStatus.VALID,
                raw_value=overlap,
                normalized_value=overlap,
                unit="fraction",
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Track-time overlap with posterior P10-P90 release interval.",
                provenance=self._provenance(posterior.posterior_id, raster.artifact.sha256),
            ),
            "track_coverage_quality": MetricResult(
                name="track_coverage_quality",
                status=MetricStatus.VALID,
                raw_value=coverage_quality,
                normalized_value=coverage_quality,
                unit="observed_point_fraction",
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Observed AIS points divided by observed plus interpolated points.",
            ),
            "ais_gap_seconds": MetricResult(
                name="ais_gap_seconds",
                status=MetricStatus.VALID,
                raw_value=maximum_gap,
                normalized_value=None,
                unit="second",
                direction=EvidenceDirection.DESCRIPTIVE,
                explanation=(
                    "Maximum AIS gap is a track-confidence limitation and investigative "
                    "context; it is not evidence or proof of discharge."
                ),
                provenance=(
                    MetadataEntry(
                        key="confidence_reference_seconds",
                        value=_seconds(self._config.gap_reference_duration),
                    ),
                ),
            ),
        }
        quality_issues = (
            (
                "AIS gaps reduce confidence in reconstructed-track evidence; missing AIS is "
                "not evidence of discharge."
            ),
        ) if maximum_gap > _seconds(self._config.gap_reference_duration) else ()
        return CandidateMetricEvidence(
            candidate_id=request.candidate.candidate_id,
            metrics=tuple(metrics[name] for name in self._config.metrics),
            quality_issues=quality_issues,
        )

    def _provenance(self, posterior_id: str, raster_sha: str) -> tuple[MetadataEntry, ...]:
        return (
            MetadataEntry(key="source_posterior_id", value=posterior_id),
            MetadataEntry(key="posterior_raster_sha256", value=raster_sha),
            MetadataEntry(key="support_aggregation", value=self._config.support_aggregation),
            MetadataEntry(key="calibrated_responsibility_probability", value=False),
        )

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="posterior_source_support",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="rasterio-shapely",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="calibrated_responsibility_probability", value=False),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
        )


def _seconds(value: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return value.value * factors[value.unit]


def _crs_text(geometry: SpatialGeometry) -> str:
    crs = geometry.crs
    if crs.authority and crs.code:
        return f"{crs.authority}:{crs.code}"
    if crs.wkt:
        return str(crs.wkt)
    raise ValueError("AIS point CRS lacks an identifier")


def _projected_point(
    dataset: rasterio.io.DatasetReader, geometry: SpatialGeometry
) -> Point:
    point = geometry.geometry
    if point.type != "Point":
        raise ValueError("AIS support sampling requires Point geometry")
    xs, ys = transform(
        _crs_text(geometry),
        str(dataset.crs),
        [point.coordinate.x],
        [point.coordinate.y],
    )
    return Point(xs[0], ys[0])


def _sample_nearest(
    dataset: rasterio.io.DatasetReader, values: FloatArray, geometry: SpatialGeometry
) -> float:
    point = _projected_point(dataset, geometry)
    row, column = dataset.index(point.x, point.y)
    if not (0 <= row < dataset.height and 0 <= column < dataset.width):
        return 0.0
    return float(values[row, column])


def _aggregate(values: tuple[float, ...], method: Literal["mean", "maximum"]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values) if method == "mean" else max(values))


def _temporal_overlap(track: TimeRange, posterior: TimeRange) -> float:
    start, end = max(track.start, posterior.start), min(track.end, posterior.end)
    if posterior.start == posterior.end:
        return 1.0 if track.start <= posterior.start <= track.end else 0.0
    return max(0.0, (end - start).total_seconds()) / (
        posterior.end - posterior.start
    ).total_seconds()


def _maximum_gap_seconds(request: AttributionEvidenceRequest) -> float:
    explicit = request.candidate.track.coverage.maximum_observed_gap
    if explicit is not None:
        return _seconds(explicit)
    timestamps = tuple(point.observed_at for point in request.candidate.track.points)
    return max(
        (
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        ),
        default=0.0,
    )


def _distance_metric(
    distance: float | None,
    sigma: float | None,
    provenance: tuple[MetadataEntry, ...],
) -> MetricResult:
    if distance is None:
        return MetricResult(
            name="posterior_source_distance",
            status=MetricStatus.MISSING,
            raw_value=None,
            direction=EvidenceDirection.LOWER_IS_BETTER,
            explanation="No posterior-support cell or AIS position was available.",
            provenance=provenance,
        )
    normalized = math.exp(-0.5 * (distance / sigma) ** 2) if sigma is not None else None
    return MetricResult(
        name="posterior_source_distance",
        status=MetricStatus.VALID,
        raw_value=distance,
        normalized_value=normalized,
        unit="metre",
        direction=EvidenceDirection.LOWER_IS_BETTER,
        explanation=(
            "Minimum observed AIS-position distance to configured posterior-support cells; "
            "Gaussian normalization is uncalibrated."
        ),
        provenance=provenance,
    )


def create_posterior_source_support(
    config: ComponentConfig,
) -> PosteriorSourceSupportEvidenceBuilder:
    return PosteriorSourceSupportEvidenceBuilder(
        PosteriorSourceSupportConfig.model_validate(config.settings)
    )
