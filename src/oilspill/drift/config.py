"""Typed OpenOil configuration without hidden physical defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier, MetadataEntry
from oilspill.domain.environment import EnvironmentalVariable
from oilspill.domain.geospatial import Duration


class OpenOilDriftConfig(FrozenModel):
    output_root: Path
    particle_count: int = Field(gt=0)
    time_step: Duration
    output_interval: Duration
    release_time_policy: Literal["start", "midpoint", "end"]
    release_position_policy: Literal["point_only", "geometry_centroid"]
    backward_seed_position: Literal["release_geometry", "observation_centroid"]
    forecast_seed_position: Literal["release_geometry", "observation_centroid"]
    required_forcing: tuple[EnvironmentalVariable, ...]
    oil_type: str | None = None
    configured_uncertainty: tuple[MetadataEntry, ...]
    model_parameters: tuple[MetadataEntry, ...]
    forcing_reader: Literal["netcdf_cf_generic"]
    output_trajectory_dimension: Identifier
    output_time_dimension: Identifier
    output_longitude_variable: Identifier
    output_latitude_variable: Identifier
    particle_support_geometry: Literal["convex_hull"]
    particle_support_geometry_validation_note: Identifier
    forcing_compatibility_validation_note: Identifier
    parameter_validation_note: Identifier

    @model_validator(mode="after")
    def _positive_steps(self) -> OpenOilDriftConfig:
        if _seconds(self.time_step) <= 0 or _seconds(self.output_interval) <= 0:
            raise ValueError("OpenOil time_step and output_interval must be positive")
        return self


def _seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]
