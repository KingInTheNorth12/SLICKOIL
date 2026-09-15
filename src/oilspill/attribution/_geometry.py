"""CRS-safe geometry conversion used only by attribution metrics."""

from __future__ import annotations

from typing import Any

import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape

from oilspill.domain.geospatial import PolygonGeometry, SpatialGeometry


def _crs_text(geometry: SpatialGeometry) -> str:
    if geometry.crs.authority and geometry.crs.code:
        return f"{geometry.crs.authority}:{geometry.crs.code}"
    if geometry.crs.wkt:
        return geometry.crs.wkt
    raise ValueError("geometry CRS has no usable identifier")


def _polygon(value: PolygonGeometry) -> Polygon:
    return Polygon(
        [(point.x, point.y) for point in value.exterior],
        [[(point.x, point.y) for point in ring] for ring in value.holes],
    )


def as_shapely(value: SpatialGeometry) -> Any:
    geometry = value.geometry
    if geometry.type == "Point":
        return Point(geometry.coordinate.x, geometry.coordinate.y)
    if geometry.type == "LineString":
        return LineString([(point.x, point.y) for point in geometry.coordinates])
    if geometry.type == "Polygon":
        return _polygon(geometry)
    return MultiPolygon([_polygon(polygon) for polygon in geometry.polygons])


class MetricProjector:
    def __init__(self, target_crs: str) -> None:
        target = rasterio.crs.CRS.from_user_input(target_crs)
        if not target.is_projected:
            raise ValueError("metric measurement_crs must be projected")
        units = (target.linear_units or "").lower()
        if units not in {"metre", "meter", "metres", "meters"}:
            raise ValueError("metric measurement_crs must use metre units")
        self.target_crs = target

    def project(self, geometry: SpatialGeometry) -> Any:
        return shape(
            transform_geom(
                _crs_text(geometry),
                self.target_crs,
                mapping(as_shapely(geometry)),
                precision=-1,
            )
        )
