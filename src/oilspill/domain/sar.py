"""Synthetic-aperture-radar scene contracts."""

from __future__ import annotations

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    FrozenModel,
    Identifier,
    MetadataEntry,
    ProcessingRecord,
    TimeRange,
)
from oilspill.domain.geospatial import RasterAsset, SpatialGeometry


class SARScene(FrozenModel):
    """A SAR product whose footprint and raster are fully georeferenced."""

    scene_id: Identifier
    platform: Identifier
    product_type: Identifier
    acquisition: TimeRange
    polarizations: tuple[str, ...] = Field(min_length=1)
    footprint: SpatialGeometry
    raster: RasterAsset
    source: ArtifactRef
    processing_level: str | None = None
    processing: tuple[ProcessingRecord, ...] = ()
    metadata: tuple[MetadataEntry, ...] = ()

    @model_validator(mode="after")
    def _consistent_crs(self) -> SARScene:
        if self.footprint.crs != self.raster.grid.crs:
            raise ValueError("SAR footprint and raster must use the same CRS")
        return self
