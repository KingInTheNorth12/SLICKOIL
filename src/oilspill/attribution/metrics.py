"""Independent, reusable, threshold-free attribution evidence metrics."""

from __future__ import annotations

import math

from oilspill.attribution._geometry import MetricProjector
from oilspill.attribution.models import (
    DirectionMetricConfig,
    DistanceMetricConfig,
    GeometryMetricConfig,
    MinMaxNormalization,
    TemporalMetricConfig,
)
from oilspill.domain.ais import VesselTrack
from oilspill.domain.attribution import EvidenceDirection, MetricResult, MetricStatus
from oilspill.domain.common import MetadataEntry, TimeRange
from oilspill.domain.drift import ParticleTrajectory
from oilspill.domain.geospatial import Angle, AngularUnit, SpatialGeometry


def _normalization_provenance(
    normalization: MinMaxNormalization | None,
) -> tuple[MetadataEntry, ...]:
    if normalization is None:
        return (MetadataEntry(key="normalization", value="none"),)
    return (
        MetadataEntry(key="normalization", value="explicit_min_max"),
        MetadataEntry(key="normalization_lower_bound", value=normalization.lower_bound),
        MetadataEntry(key="normalization_upper_bound", value=normalization.upper_bound),
        MetadataEntry(key="normalization_invert", value=normalization.invert),
        MetadataEntry(key="normalization_clamp", value=normalization.clamp),
    )


def _valid(
    *,
    name: str,
    value: float,
    unit: str | None,
    direction: EvidenceDirection,
    explanation: str,
    provenance: tuple[MetadataEntry, ...],
    normalization: MinMaxNormalization | None = None,
    source_simulation_ids: tuple[str, ...] = (),
) -> MetricResult:
    return MetricResult(
        name=name,
        status=MetricStatus.VALID,
        raw_value=value,
        normalized_value=normalization.apply(value) if normalization else None,
        unit=unit,
        direction=direction,
        explanation=explanation,
        provenance=(*provenance, *_normalization_provenance(normalization)),
        source_simulation_ids=source_simulation_ids,
    )


def _unavailable(
    *,
    name: str,
    status: MetricStatus,
    direction: EvidenceDirection,
    explanation: str,
    unit: str | None,
    provenance: tuple[MetadataEntry, ...] = (),
) -> MetricResult:
    return MetricResult(
        name=name,
        status=status,
        raw_value=None,
        normalized_value=None,
        unit=unit,
        direction=direction,
        explanation=explanation,
        provenance=(MetadataEntry(key="calculation", value="not_performed"), *provenance),
    )


class _GeometryDistanceMetric:
    name: str
    definition: str

    def __init__(self, config: DistanceMetricConfig) -> None:
        self._config = config
        self._projector = MetricProjector(config.measurement_crs)

    def _compute(
        self,
        first: SpatialGeometry | None,
        second: SpatialGeometry | None,
        *,
        hausdorff: bool,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        if first is None or second is None:
            return _unavailable(
                name=self.name,
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation="Required geometry is missing; no numeric value was substituted.",
                unit="metre",
                provenance=(
                    MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                ),
            )
        try:
            first_projected = self._projector.project(first)
            second_projected = self._projector.project(second)
        except (TypeError, ValueError) as error:
            return _unavailable(
                name=self.name,
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation=f"Geometry could not be projected safely: {error}",
                unit="metre",
                provenance=(
                    MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                ),
            )
        if first_projected.is_empty or second_projected.is_empty:
            return _unavailable(
                name=self.name,
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation="Distance is undefined for an empty geometry.",
                unit="metre",
                provenance=(
                    MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                ),
            )
        value = float(
            first_projected.hausdorff_distance(second_projected)
            if hausdorff
            else first_projected.distance(second_projected)
        )
        return _valid(
            name=self.name,
            value=value,
            unit="metre",
            direction=EvidenceDirection.LOWER_IS_BETTER,
            explanation=self.definition,
            provenance=(
                MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
            normalization=self._config.normalization,
            source_simulation_ids=source_simulation_ids,
        )


class HausdorffDistanceMetric(_GeometryDistanceMetric):
    name = "hausdorff_distance"
    definition = "Symmetric Shapely Hausdorff distance after projection to the configured CRS."

    def compute(
        self,
        first: SpatialGeometry | None,
        second: SpatialGeometry | None,
        *,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        return self._compute(
            first, second, hausdorff=True, source_simulation_ids=source_simulation_ids
        )


class SourceDistanceMetric(_GeometryDistanceMetric):
    name = "source_distance"
    definition = "Minimum geometry distance after projection to the configured metric CRS."

    def compute(
        self,
        source_geometry: SpatialGeometry | None,
        vessel_path: SpatialGeometry | None,
        *,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        return self._compute(
            source_geometry,
            vessel_path,
            hausdorff=False,
            source_simulation_ids=source_simulation_ids,
        )


class SpatialOverlapMetric:
    """Intersection-over-union for valid areal geometries."""

    def __init__(self, config: GeometryMetricConfig) -> None:
        self._config = config
        self._projector = MetricProjector(config.measurement_crs)

    def compute(
        self,
        first: SpatialGeometry | None,
        second: SpatialGeometry | None,
        *,
        name: str = "spatial_overlap_iou",
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        if first is None or second is None:
            return _unavailable(
                name=name,
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Required areal geometry is missing; no zero overlap was assumed.",
                unit="dimensionless",
            )
        if first.geometry.type not in {"Polygon", "MultiPolygon"} or second.geometry.type not in {
            "Polygon",
            "MultiPolygon",
        }:
            return _unavailable(
                name=name,
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Intersection-over-union requires polygon or multipolygon inputs.",
                unit="dimensionless",
            )
        first_projected = self._projector.project(first)
        second_projected = self._projector.project(second)
        if not first_projected.is_valid or not second_projected.is_valid:
            return _unavailable(
                name=name,
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Intersection-over-union is not computed for invalid polygons.",
                unit="dimensionless",
            )
        union_area = float(first_projected.union(second_projected).area)
        if union_area <= 0:
            return _unavailable(
                name=name,
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.HIGHER_IS_BETTER,
                explanation="Intersection-over-union is undefined for zero-area union.",
                unit="dimensionless",
            )
        value = float(first_projected.intersection(second_projected).area) / union_area
        return MetricResult(
            name=name,
            status=MetricStatus.VALID,
            raw_value=value,
            normalized_value=value,
            unit="dimensionless",
            direction=EvidenceDirection.HIGHER_IS_BETTER,
            explanation="Polygon intersection area divided by polygon union area (Jaccard IoU).",
            provenance=(
                MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                MetadataEntry(key="normalization", value="identity; IoU is bounded [0,1]"),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
            source_simulation_ids=source_simulation_ids,
        )


class DirectionalConsistencyMetric:
    def __init__(self, config: DirectionMetricConfig) -> None:
        self._config = config

    def compute(self, first: Angle | None, second: Angle | None) -> MetricResult:
        if first is None or second is None:
            return _unavailable(
                name="directional_consistency",
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation=(
                    "One or both directions are missing; no angular difference was assumed."
                ),
                unit="degree",
            )
        if first.convention != second.convention:
            return _unavailable(
                name="directional_consistency",
                status=MetricStatus.INVALID,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation="Direction conventions differ and no implicit conversion is defined.",
                unit="degree",
            )
        first_degrees = (
            math.degrees(first.value) if first.unit == AngularUnit.RADIAN else first.value
        )
        second_degrees = (
            math.degrees(second.value) if second.unit == AngularUnit.RADIAN else second.value
        )
        period = self._config.period_degrees
        difference = abs((first_degrees - second_degrees) % period)
        value = min(difference, period - difference)
        return _valid(
            name="directional_consistency",
            value=value,
            unit="degree",
            direction=EvidenceDirection.LOWER_IS_BETTER,
            explanation="Minimum absolute angular difference modulo the configured period.",
            provenance=(
                MetadataEntry(key="period_degrees", value=period),
                MetadataEntry(key="direction_convention", value=first.convention),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
            normalization=self._config.normalization,
        )


class TemporalConsistencyMetric:
    def __init__(self, config: TemporalMetricConfig) -> None:
        self._config = config

    def compute(self, first: TimeRange | None, second: TimeRange | None) -> MetricResult:
        if first is None or second is None:
            return _unavailable(
                name="temporal_consistency",
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation="One or both UTC intervals are missing; no temporal gap was assumed.",
                unit="second",
            )
        if first.end < second.start:
            gap = (second.start - first.end).total_seconds()
        elif second.end < first.start:
            gap = (first.start - second.end).total_seconds()
        else:
            gap = 0.0
        return _valid(
            name="temporal_consistency",
            value=gap,
            unit="second",
            direction=EvidenceDirection.LOWER_IS_BETTER,
            explanation=(
                "Minimum UTC interval separation; overlapping closed intervals have gap zero."
            ),
            provenance=(MetadataEntry(key="validation_note", value=self._config.validation_note),),
            normalization=self._config.normalization,
        )


class ForwardFitSimilarityMetric:
    """Uncalibrated forward fit defined strictly as simulated/observed polygon IoU."""

    def __init__(self, config: GeometryMetricConfig) -> None:
        self._overlap = SpatialOverlapMetric(config)

    def compute(
        self,
        simulated_slick: SpatialGeometry | None,
        observed_slick: SpatialGeometry | None,
        *,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        return self._overlap.compute(
            simulated_slick,
            observed_slick,
            name="forward_fit_similarity",
            source_simulation_ids=source_simulation_ids,
        )


class BackwardFitSimilarityMetric(_GeometryDistanceMetric):
    """Uncalibrated backward fit represented by path Hausdorff distance."""

    name = "backward_fit_similarity"
    definition = (
        "Symmetric Hausdorff distance between backward particle path and AIS path after projection."
    )

    def compute(
        self,
        backward_path: SpatialGeometry | None,
        vessel_path: SpatialGeometry | None,
        *,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        return self._compute(
            backward_path,
            vessel_path,
            hausdorff=True,
            source_simulation_ids=source_simulation_ids,
        )


class AISTrajectoryConsistencyMetric:
    """Mean projected separation at timestamps present in both canonical trajectories."""

    def __init__(self, config: DistanceMetricConfig) -> None:
        self._config = config
        self._projector = MetricProjector(config.measurement_crs)

    def compute(
        self,
        particle_trajectory: ParticleTrajectory | None,
        vessel_track: VesselTrack | None,
        *,
        source_simulation_ids: tuple[str, ...] = (),
    ) -> MetricResult:
        if particle_trajectory is None or vessel_track is None:
            return _unavailable(
                name="ais_trajectory_consistency",
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation=(
                    "Particle trajectory or AIS track is missing; no zero distance was assumed."
                ),
                unit="metre",
            )
        ais_by_time = {point.observed_at: point.position for point in vessel_track.points}
        matched = [
            (position.position, ais_by_time[position.timestamp])
            for position in particle_trajectory.positions
            if position.timestamp in ais_by_time
        ]
        if not matched:
            return _unavailable(
                name="ais_trajectory_consistency",
                status=MetricStatus.MISSING,
                direction=EvidenceDirection.LOWER_IS_BETTER,
                explanation=(
                    "No exact UTC timestamps are shared; interpolation is intentionally not "
                    "assumed."
                ),
                unit="metre",
            )
        distances = [
            float(self._projector.project(first).distance(self._projector.project(second)))
            for first, second in matched
        ]
        value = sum(distances) / len(distances)
        return _valid(
            name="ais_trajectory_consistency",
            value=value,
            unit="metre",
            direction=EvidenceDirection.LOWER_IS_BETTER,
            explanation="Arithmetic mean separation over exact shared UTC timestamps only.",
            provenance=(
                MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                MetadataEntry(key="matched_timestamp_count", value=len(matched)),
                MetadataEntry(key="interpolation", value="none"),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
            normalization=self._config.normalization,
            source_simulation_ids=source_simulation_ids,
        )
