"""CRS-explicit geometry, raster, and physical measurement values.

Coordinates are expressed in the declared CRS axis units. Physical lengths and areas use
separate types that intentionally do not accept angular units; callers must project or use a
geodesic algorithm before constructing physical measurements.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from oilspill.domain.common import ArtifactRef, FrozenModel, Identifier


class SpatialUnit(StrEnum):
    DEGREE = "degree"
    METRE = "metre"
    KILOMETRE = "kilometre"


class LengthUnit(StrEnum):
    METRE = "metre"
    KILOMETRE = "kilometre"
    NAUTICAL_MILE = "nautical_mile"


class AreaUnit(StrEnum):
    SQUARE_METRE = "square_metre"
    SQUARE_KILOMETRE = "square_kilometre"


class AngularUnit(StrEnum):
    DEGREE = "degree"
    RADIAN = "radian"


class CRS(FrozenModel):
    """Coordinate reference system and its horizontal axis units."""

    authority: str | None = None
    code: str | None = None
    wkt: str | None = None
    is_geographic: bool
    axis_units: tuple[SpatialUnit, SpatialUnit]

    @model_validator(mode="after")
    def _identified(self) -> CRS:
        if not self.wkt and not (self.authority and self.code):
            raise ValueError("CRS requires authority/code or WKT")
        if self.is_geographic and SpatialUnit.DEGREE not in self.axis_units:
            raise ValueError("a geographic CRS must declare angular axis units")
        if not self.is_geographic and SpatialUnit.DEGREE in self.axis_units:
            raise ValueError("a projected CRS cannot declare degree axis units")
        return self


class Coordinate(FrozenModel):
    """Two-dimensional coordinate in its containing geometry's CRS units."""

    x: float
    y: float


class PointGeometry(FrozenModel):
    type: Literal["Point"] = "Point"
    coordinate: Coordinate


class LineStringGeometry(FrozenModel):
    type: Literal["LineString"] = "LineString"
    coordinates: Annotated[tuple[Coordinate, ...], Field(min_length=2)]


class PolygonGeometry(FrozenModel):
    type: Literal["Polygon"] = "Polygon"
    exterior: Annotated[tuple[Coordinate, ...], Field(min_length=4)]
    holes: tuple[tuple[Coordinate, ...], ...] = ()

    @model_validator(mode="after")
    def _closed_rings(self) -> PolygonGeometry:
        rings = (self.exterior, *self.holes)
        if any(len(ring) < 4 for ring in rings):
            raise ValueError("polygon rings require at least four coordinates")
        if any(ring[0] != ring[-1] for ring in rings):
            raise ValueError("polygon rings must be closed")
        return self


class MultiPolygonGeometry(FrozenModel):
    """One or more disconnected polygon components in a shared containing CRS."""

    type: Literal["MultiPolygon"] = "MultiPolygon"
    polygons: Annotated[tuple[PolygonGeometry, ...], Field(min_length=1)]


Geometry = PointGeometry | LineStringGeometry | PolygonGeometry | MultiPolygonGeometry


class SpatialGeometry(FrozenModel):
    """Geometry together with the CRS and spatial units of its coordinates."""

    geometry: Geometry = Field(discriminator="type")
    crs: CRS


class AffineTransform(FrozenModel):
    """Raster pixel-to-CRS transform: x=a*col+b*row+c, y=d*col+e*row+f."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


class RasterGrid(FrozenModel):
    """Regular raster grid with an explicit pixel-to-CRS affine transform."""

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    crs: CRS
    transform: AffineTransform


class RasterBand(FrozenModel):
    name: Identifier
    unit: Identifier
    nodata: float | int | None = None
    source_index: int | None = Field(default=None, gt=0)


class RasterAsset(FrozenModel):
    """Immutable raster artifact and complete spatial interpretation."""

    artifact: ArtifactRef
    grid: RasterGrid
    bands: Annotated[tuple[RasterBand, ...], Field(min_length=1)]


class Length(FrozenModel):
    """Physical length; degree units are deliberately not representable."""

    value: Annotated[float, Field(ge=0.0)]
    unit: LengthUnit


class Area(FrozenModel):
    """Physical area; square-degree units are deliberately not representable."""

    value: Annotated[float, Field(ge=0.0)]
    unit: AreaUnit


class Speed(FrozenModel):
    value: Annotated[float, Field(ge=0.0)]
    unit: Literal["metre_per_second", "knot", "kilometre_per_hour"]


class Angle(FrozenModel):
    value: float
    unit: AngularUnit
    convention: Identifier


class Duration(FrozenModel):
    value: Annotated[float, Field(ge=0.0)]
    unit: Literal["second", "minute", "hour", "day"]
