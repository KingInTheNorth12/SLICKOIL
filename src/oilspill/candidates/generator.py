"""Metric, evidence-preserving vessel candidate screening."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from hashlib import sha256
from typing import Any

import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape

from oilspill.candidates.config import CandidateGeneratorConfig
from oilspill.config import ComponentConfig
from oilspill.domain.ais import (
    CandidateDecision,
    CandidateEvidence,
    CandidateVessel,
    VesselTrack,
)
from oilspill.domain.common import ComponentMetadata, MetadataEntry, TimeRange
from oilspill.domain.geospatial import (
    Angle,
    AngularUnit,
    Duration,
    Length,
    LengthUnit,
    PolygonGeometry,
    SpatialGeometry,
)
from oilspill.requests import CandidateGenerationRequest


def _seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]


def _metres(length: Length) -> float:
    factors = {LengthUnit.METRE: 1.0, LengthUnit.KILOMETRE: 1_000.0}
    if length.unit not in factors:
        raise ValueError("spatial_window must use metre or kilometre units")
    return length.value * factors[length.unit]


def _crs_text(geometry: SpatialGeometry) -> str:
    crs = geometry.crs
    if crs.authority and crs.code:
        return f"{crs.authority}:{crs.code}"
    if crs.wkt:
        return crs.wkt
    raise ValueError("geometry CRS has no usable identifier")


def _polygon(value: PolygonGeometry) -> Polygon:
    return Polygon(
        [(point.x, point.y) for point in value.exterior],
        [[(point.x, point.y) for point in ring] for ring in value.holes],
    )


def _shape(value: SpatialGeometry) -> Point | LineString | Polygon | MultiPolygon:
    geometry = value.geometry
    if geometry.type == "Point":
        return Point(geometry.coordinate.x, geometry.coordinate.y)
    if geometry.type == "LineString":
        return LineString([(point.x, point.y) for point in geometry.coordinates])
    if geometry.type == "Polygon":
        return _polygon(geometry)
    return MultiPolygon([_polygon(polygon) for polygon in geometry.polygons])


class ExplainableCandidateGenerator:
    """Apply independently configured screening rules without producing attribution scores."""

    def __init__(self, config: CandidateGeneratorConfig) -> None:
        self._config = config
        target = rasterio.crs.CRS.from_user_input(config.measurement_crs)
        if not target.is_projected:
            raise ValueError("candidate measurement_crs must be projected")
        units = (target.linear_units or "").lower()
        if units not in {"metre", "meter", "metres", "meters"}:
            raise ValueError("candidate measurement_crs must use metre units")
        self._target_crs = target

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateVessel, ...]:
        source = request.search_area or request.observation.geometry.polygon
        source_metric = self._project(_shape(source), _crs_text(source))
        if request.search_interval is None:
            discharge = request.observation.discharge_time.interval
            eligible_interval = TimeRange(
                start=discharge.start
                - timedelta(seconds=_seconds(self._config.temporal_margin_before)),
                end=discharge.end
                + timedelta(seconds=_seconds(self._config.temporal_margin_after)),
            )
        else:
            eligible_interval = request.search_interval
        results = tuple(
            self._evaluate(
                request.observation.observation_id, track, source_metric, eligible_interval, request
            )
            for track in request.tracks
        )
        if self._config.return_excluded:
            return results
        return tuple(item for item in results if item.decision == CandidateDecision.RETAINED)

    def component_metadata(self) -> ComponentMetadata:
        payload = self._config.model_dump(mode="json")
        digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return ComponentMetadata(
            name="explainable_candidate_generator",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="shapely-rasterio",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                MetadataEntry(key="final_attribution_score_computed", value=False),
                MetadataEntry(key="filter_validation", value=self._config.filter_validation_note),
                MetadataEntry(
                    key="direction_validation", value=self._config.direction.validation_note
                ),
            ),
        )

    def _evaluate(
        self,
        observation_id: str,
        track: VesselTrack,
        source_metric: Any,
        eligible_interval: TimeRange,
        request: CandidateGenerationRequest,
    ) -> CandidateVessel:
        path_metric = self._project(_shape(track.path), _crs_text(track.path))
        points_metric = [
            self._project(_shape(point.position), _crs_text(point.position))
            for point in track.points
        ]
        closest = float(path_metric.distance(source_metric))
        nearest_position = min(float(point.distance(source_metric)) for point in points_metric)
        intersects = bool(path_metric.intersects(source_metric))
        temporal = not (
            track.interval.end < eligible_interval.start
            or track.interval.start > eligible_interval.end
        )
        temporal_offset = self._temporal_offset(track.interval, eligible_interval)
        difference, direction_consistent = self._direction(track, path_metric, request)
        within_window = closest <= _metres(self._config.spatial_window)
        position_near = nearest_position <= _metres(self._config.spatial_window)

        failures: list[str] = []
        if self._config.temporal_filter_enabled and not temporal:
            failures.append("track does not overlap the configured plausible discharge window")
        if self._config.spatial_filter_enabled and not within_window:
            failures.append("trajectory exceeds the configured spatial proximity window")
        if self._config.trajectory_intersection_filter_enabled and not intersects:
            failures.append("trajectory does not intersect the plausible source geometry")
        if self._config.observed_position_filter_enabled and not position_near:
            failures.append("no observed vessel position is within the configured spatial window")
        if self._config.direction.enabled and direction_consistent is False:
            failures.append("trajectory direction exceeds the configured angular difference")
        decision = CandidateDecision.EXCLUDED if failures else CandidateDecision.RETAINED
        reasons = (
            tuple(failures) if failures else ("passed every enabled candidate-generation filter",)
        )
        return CandidateVessel(
            candidate_id=f"candidate:{observation_id}:{track.track_id}",
            observation_id=observation_id,
            track=track,
            screening_evidence=CandidateEvidence(
                closest_approach=Length(value=closest, unit=LengthUnit.METRE),
                closest_approach_time=None,
                temporal_offset=Duration(value=temporal_offset, unit="second"),
                direction_difference=(
                    Angle(
                        value=difference,
                        unit=AngularUnit.DEGREE,
                        convention=f"minimum_difference_modulo_{self._config.direction.period_degrees}",
                    )
                    if difference is not None
                    else None
                ),
                intersects_search_area=intersects,
                temporal_eligible=temporal,
                spatially_eligible=within_window,
                direction_consistent=direction_consistent,
                nearest_observed_position=Length(value=nearest_position, unit=LengthUnit.METRE),
                track_coverage=track.coverage,
            ),
            decision=decision,
            reasons=reasons,
        )

    def _project(self, geometry: Any, source_crs: str) -> Any:
        return shape(transform_geom(source_crs, self._target_crs, mapping(geometry), precision=-1))

    @staticmethod
    def _temporal_offset(track: TimeRange, eligible: TimeRange) -> float:
        if track.end < eligible.start:
            return (eligible.start - track.end).total_seconds()
        if track.start > eligible.end:
            return (track.start - eligible.end).total_seconds()
        return 0.0

    def _direction(
        self, track: VesselTrack, path_metric: Any, request: CandidateGenerationRequest
    ) -> tuple[float | None, bool | None]:
        config = self._config.direction
        if not config.enabled:
            return None, None
        coordinates = list(path_metric.coords)
        first, last = coordinates[0], coordinates[-1]
        if first == last:
            return self._unavailable_direction()
        track_angle = math.degrees(math.atan2(last[0] - first[0], last[1] - first[1])) % 360.0
        if config.reference == "slick_orientation":
            orientation = request.observation.geometry.orientation
            if orientation is None or orientation.unit != AngularUnit.DEGREE:
                return self._unavailable_direction()
            reference = orientation.value
        else:
            if config.configured_angle_degrees is None:
                return self._unavailable_direction()
            reference = config.configured_angle_degrees
        period = float(config.period_degrees)
        raw = abs((track_angle - reference) % period)
        difference = min(raw, period - raw)
        return difference, difference <= config.maximum_difference_degrees

    def _unavailable_direction(self) -> tuple[None, bool | None]:
        if self._config.direction.unavailable_policy == "ignore":
            return None, None
        return None, False


def create_explainable_candidate_generator(
    config: ComponentConfig,
) -> ExplainableCandidateGenerator:
    return ExplainableCandidateGenerator(CandidateGeneratorConfig.model_validate(config.settings))
