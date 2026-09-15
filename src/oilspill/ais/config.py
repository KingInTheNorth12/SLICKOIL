"""Explicit future-dataset column mapping and track-building configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.domain.geospatial import Duration


class AISColumnMapping(FrozenModel):
    """Map an arbitrary source schema to canonical AIS fields.

    No source column name is inferred. The mapping for any real dataset remains
    DATASET_VALIDATION_REQUIRED.
    """

    vessel_id: Identifier
    timestamp: Identifier
    longitude: Identifier
    latitude: Identifier
    speed_over_ground: str | None = None
    course_over_ground: str | None = None
    heading: str | None = None
    navigation_status: str | None = None
    record_id: str | None = None


class CSVAISProviderConfig(FrozenModel):
    path: Path
    columns: AISColumnMapping
    coordinate_crs: Literal["EPSG:4326"]
    coordinate_crs_validation_note: Identifier
    delimiter: str = Field(min_length=1, max_length=1)
    encoding: Identifier
    timestamp_format: Identifier
    naive_timestamp_timezone: str | None
    speed_unit: Literal["knot", "metre_per_second", "kilometre_per_hour"] | None
    course_unit: Literal["degree", "radian"] | None
    heading_unit: Literal["degree", "radian"] | None
    angle_convention: str | None
    invalid_row_policy: Literal["reject", "skip"]
    dataset_name: Identifier
    dataset_version: Identifier
    provider_name: Identifier
    license: str | None = None
    schema_validation_note: Identifier

    @model_validator(mode="after")
    def _units_match_optional_columns(self) -> CSVAISProviderConfig:
        if (self.columns.speed_over_ground is None) != (self.speed_unit is None):
            raise ValueError("speed_unit and speed_over_ground column must be configured together")
        if (self.columns.course_over_ground is None) != (self.course_unit is None):
            raise ValueError(
                "course_unit and course_over_ground column must be configured together"
            )
        if (self.columns.heading is None) != (self.heading_unit is None):
            raise ValueError("heading_unit and heading column must be configured together")
        if (self.course_unit or self.heading_unit) and self.angle_convention is None:
            raise ValueError("angle_convention is required for configured angular fields")
        return self


class VesselTrackBuilderConfig(FrozenModel):
    duplicate_policy: Literal["keep_first", "keep_last", "reject"]
    gap_threshold: Duration
    interpolation_method: Literal["none", "linear_in_declared_crs"]
    interpolation_interval: Duration | None
    maximum_interpolation_gap: Duration | None
    allow_geographic_linear_interpolation: bool
    minimum_points_per_track: int = Field(ge=2)
    insufficient_points_policy: Literal["skip", "reject"]
    track_id_prefix: Identifier
    interpolation_validation_note: Identifier

    @model_validator(mode="after")
    def _valid_interpolation(self) -> VesselTrackBuilderConfig:
        configured = self.interpolation_method != "none"
        if configured != (self.interpolation_interval is not None):
            raise ValueError(
                "interpolation_interval must be set exactly when interpolation is enabled"
            )
        if configured != (self.maximum_interpolation_gap is not None):
            raise ValueError(
                "maximum_interpolation_gap must be set exactly when interpolation is enabled"
            )
        if self.interpolation_interval and _duration_seconds(self.interpolation_interval) <= 0:
            raise ValueError("interpolation_interval must be positive")
        return self


def _duration_seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]
