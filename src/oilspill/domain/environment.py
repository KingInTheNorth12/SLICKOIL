"""Provider-neutral environmental forcing contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    DataKind,
    DatasetMetadata,
    FrozenModel,
    Identifier,
    QualityFlag,
    TimeRange,
    UTCDateTime,
)
from oilspill.domain.geospatial import CRS, AffineTransform, SpatialGeometry


class GridKind(StrEnum):
    REGULAR_RASTER = "regular_raster"
    RECTILINEAR = "rectilinear"
    CURVILINEAR = "curvilinear"
    UNSTRUCTURED = "unstructured"


class GridDefinition(FrozenModel):
    """Spatial grid; regular rasters must include a pixel-to-CRS transform."""

    kind: GridKind
    shape: Annotated[tuple[int, ...], Field(min_length=2)]
    crs: CRS
    extent: SpatialGeometry
    transform: AffineTransform | None = None

    @model_validator(mode="after")
    def _raster_has_transform(self) -> GridDefinition:
        if self.kind == GridKind.REGULAR_RASTER and self.transform is None:
            raise ValueError("regular raster environmental grids require an affine transform")
        if any(size <= 0 for size in self.shape):
            raise ValueError("grid dimensions must be positive")
        if self.extent.crs != self.crs:
            raise ValueError("grid extent and grid must use the same CRS")
        return self


class EnvironmentalVariable(StrEnum):
    WIND = "wind"
    OCEAN_CURRENT = "ocean_current"
    WAVE = "wave"
    SEA_SURFACE_TEMPERATURE = "sea_surface_temperature"


class FieldComponent(FrozenModel):
    name: Identifier
    unit: Identifier
    artifact_variable: str | None = None
    vertical_reference: str | None = None


class EnvironmentalField(FrozenModel):
    """Environmental values with explicit grid, units, times, and provenance."""

    field_id: Identifier
    variable: EnvironmentalVariable
    components: tuple[FieldComponent, ...] = Field(min_length=1)
    dimension_order: tuple[Identifier, ...] = ("time", "latitude", "longitude")
    interval: TimeRange
    timestamps: tuple[UTCDateTime, ...] = Field(min_length=1)
    grid: GridDefinition
    values: ArtifactRef
    dataset: DatasetMetadata
    data_kind: DataKind
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _times_in_range(self) -> EnvironmentalField:
        if tuple(sorted(self.timestamps)) != self.timestamps:
            raise ValueError("environmental timestamps must be ordered")
        if any(not (self.interval.start <= item <= self.interval.end) for item in self.timestamps):
            raise ValueError("environmental timestamps must fall inside interval")
        return self
