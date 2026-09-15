"""AIS observations, reconstructed tracks, and candidate-vessel contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier, QualityFlag, TimeRange, UTCDateTime
from oilspill.domain.geospatial import Angle, Duration, Length, SpatialGeometry, Speed


class AISPoint(FrozenModel):
    """Provider-neutral AIS position in an explicit CRS."""

    vessel_id: Identifier
    observed_at: UTCDateTime
    position: SpatialGeometry
    speed_over_ground: Speed | None = None
    course_over_ground: Angle | None = None
    heading: Angle | None = None
    navigation_status: str | None = None
    provider_record_id: str | None = None
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _point_geometry(self) -> AISPoint:
        if self.position.geometry.type != "Point":
            raise ValueError("AISPoint.position must contain Point geometry")
        if self.position.crs.is_geographic:
            coordinate = self.position.geometry.coordinate
            if not -180.0 <= coordinate.x <= 180.0:
                raise ValueError("AIS longitude must lie within [-180, 180] degrees")
            if not -90.0 <= coordinate.y <= 90.0:
                raise ValueError("AIS latitude must lie within [-90, 90] degrees")
        return self

    @property
    def longitude(self) -> float | None:
        """Longitude in degrees when ``position`` uses a geographic CRS."""

        geometry = self.position.geometry
        if not self.position.crs.is_geographic or geometry.type != "Point":
            return None
        return geometry.coordinate.x

    @property
    def latitude(self) -> float | None:
        """Latitude in degrees when ``position`` uses a geographic CRS."""

        geometry = self.position.geometry
        if not self.position.crs.is_geographic or geometry.type != "Point":
            return None
        return geometry.coordinate.y


class TrackCoverage(FrozenModel):
    """Completeness and interpolation metadata for a reconstructed track."""

    source_point_count: int = Field(ge=2)
    interpolated_point_count: int = Field(ge=0)
    maximum_observed_gap: Duration | None = None
    detected_gaps: tuple[TimeRange, ...] = ()
    interpolated_intervals: tuple[TimeRange, ...] = ()


class VesselTrack(FrozenModel):
    """Chronologically ordered AIS track and CRS-explicit path."""

    track_id: Identifier
    vessel_id: Identifier
    interval: TimeRange
    points: tuple[AISPoint, ...] = Field(min_length=2)
    path: SpatialGeometry
    coverage: TrackCoverage
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _consistent_track(self) -> VesselTrack:
        if self.path.geometry.type != "LineString":
            raise ValueError("VesselTrack.path must contain LineString geometry")
        if any(point.vessel_id != self.vessel_id for point in self.points):
            raise ValueError("all AIS points must belong to the track vessel")
        timestamps = tuple(point.observed_at for point in self.points)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("AIS points must be chronologically ordered")
        if self.interval.start != timestamps[0] or self.interval.end != timestamps[-1]:
            raise ValueError("track interval must equal its first and last point timestamps")
        if any(point.position.crs != self.path.crs for point in self.points):
            raise ValueError("track points and path must use the same CRS")
        if self.coverage.source_point_count > len(self.points):
            raise ValueError("source point count cannot exceed total track points")
        return self


class CandidateDecision(StrEnum):
    RETAINED = "retained"
    EXCLUDED = "excluded"


class CandidateEvidence(FrozenModel):
    """Screening evidence only; final attribution belongs to AttributionResult."""

    closest_approach: Length | None = None
    closest_approach_time: UTCDateTime | None = None
    temporal_offset: Duration | None = None
    direction_difference: Angle | None = None
    intersects_search_area: bool
    temporal_eligible: bool | None = None
    spatially_eligible: bool | None = None
    direction_consistent: bool | None = None
    nearest_observed_position: Length | None = None
    track_coverage: TrackCoverage
    quality_flags: tuple[QualityFlag, ...] = ()


class CandidateVessel(FrozenModel):
    candidate_id: Identifier
    observation_id: Identifier
    track: VesselTrack
    screening_evidence: CandidateEvidence
    decision: CandidateDecision
    reasons: tuple[str, ...] = ()
