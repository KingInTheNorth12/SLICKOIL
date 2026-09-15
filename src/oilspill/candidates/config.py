"""Configuration for permissive, explainable vessel candidate generation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.domain.geospatial import Duration, Length


class DirectionFilterConfig(FrozenModel):
    enabled: bool
    reference: Literal["slick_orientation", "configured_angle"]
    configured_angle_degrees: float | None = None
    maximum_difference_degrees: float = Field(ge=0.0, le=180.0)
    period_degrees: Literal[180, 360]
    unavailable_policy: Literal["ignore", "exclude"]
    validation_note: Identifier

    @model_validator(mode="after")
    def _reference_available(self) -> DirectionFilterConfig:
        if self.reference == "configured_angle" and self.configured_angle_degrees is None:
            raise ValueError("configured_angle reference requires configured_angle_degrees")
        return self


class CandidateGeneratorConfig(FrozenModel):
    measurement_crs: Identifier
    spatial_window: Length
    temporal_margin_before: Duration
    temporal_margin_after: Duration
    temporal_filter_enabled: bool
    spatial_filter_enabled: bool
    trajectory_intersection_filter_enabled: bool
    observed_position_filter_enabled: bool
    direction: DirectionFilterConfig
    return_excluded: bool
    measurement_crs_validation_note: Identifier
    filter_validation_note: Identifier
