"""Explicit source-to-canonical metocean mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator

from oilspill.domain.common import DataKind, FrozenModel, Identifier


class EnvironmentalVariableMapping(FrozenModel):
    """One source variable and its explicit affine conversion to canonical units."""

    source_name: Identifier
    source_unit: Identifier
    canonical_unit: Identifier
    scale_factor: float
    add_offset: float
    vertical_reference: str | None = None


class XarrayEnvironmentalProviderConfig(FrozenModel):
    """Initial rectilinear xarray adapter configuration.

    Real variable names, unit conversions, calendar/timezone interpretation, coordinate order,
    and longitude convention are DATASET_VALIDATION_REQUIRED.
    """

    source: Path
    output_root: Path
    engine: str | None = None
    wind_u: EnvironmentalVariableMapping
    wind_v: EnvironmentalVariableMapping
    current_u: EnvironmentalVariableMapping
    current_v: EnvironmentalVariableMapping
    latitude: Identifier
    longitude: Identifier
    time: Identifier
    source_dimension_order: tuple[Identifier, Identifier, Identifier]
    source_time_zone: Literal["UTC"]
    longitude_convention: Literal["-180_180", "0_360"]
    coordinate_crs: Literal["EPSG:4326"]
    boundary_policy: Literal["strict", "clip"]
    data_kind: DataKind
    dataset_name: Identifier
    dataset_version: Identifier
    provider_name: Identifier
    license: str | None = None
    mapping_validation_note: Identifier

    @model_validator(mode="after")
    def _unique_coordinates(self) -> XarrayEnvironmentalProviderConfig:
        coordinates = (self.time, self.latitude, self.longitude)
        if len(set(coordinates)) != 3:
            raise ValueError("time, latitude, and longitude source names must be distinct")
        if tuple(coordinates) != self.source_dimension_order:
            raise ValueError(
                "source_dimension_order must explicitly match time, latitude, longitude mapping"
            )
        return self
