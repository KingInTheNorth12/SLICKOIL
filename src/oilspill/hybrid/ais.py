"""Posterior-derived AIS query windows for the hybrid pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Literal

from shapely.geometry import MultiPolygon, Polygon, shape

from oilspill.config import HybridAISConfig
from oilspill.domain.common import TimeRange
from oilspill.domain.geospatial import (
    CRS,
    Coordinate,
    Duration,
    Length,
    LengthUnit,
    MultiPolygonGeometry,
    PolygonGeometry,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.domain.hybrid import SourcePosterior
from oilspill.eulerian.grid import _mapping
from oilspill.requests import AISQuery


def _seconds(value: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return value.value * factors[value.unit]


def _metres(value: Length) -> float:
    factors = {LengthUnit.METRE: 1.0, LengthUnit.KILOMETRE: 1_000.0}
    if value.unit not in factors:
        raise ValueError("hybrid posterior buffer must use metre or kilometre units")
    return value.value * factors[value.unit]


def _ring(coordinates: Iterable[Sequence[float]]) -> tuple[Coordinate, ...]:
    return tuple(Coordinate(x=float(item[0]), y=float(item[1])) for item in coordinates)


def _domain_polygon(value: Polygon | MultiPolygon, crs: CRS) -> SpatialGeometry:
    polygons = (value,) if isinstance(value, Polygon) else tuple(value.geoms)
    converted = tuple(
        PolygonGeometry(
            exterior=_ring(polygon.exterior.coords),
            holes=tuple(_ring(interior.coords) for interior in polygon.interiors),
        )
        for polygon in polygons
    )
    geometry = converted[0] if len(converted) == 1 else MultiPolygonGeometry(polygons=converted)
    return SpatialGeometry(geometry=geometry, crs=crs)


def build_posterior_ais_query(
    posterior: SourcePosterior, config: HybridAISConfig
) -> AISQuery:
    """Build a query only from posterior spatial and temporal support."""
    region = posterior_credible_region(posterior, config.credible_region_percent)
    interval = posterior_release_interval(posterior)
    buffered = shape(_mapping(region)).buffer(_metres(config.spatial_buffer))
    if not isinstance(buffered, (Polygon, MultiPolygon)) or buffered.is_empty:
        raise ValueError("posterior AIS query buffer did not produce a polygon")
    return AISQuery(
        area=_domain_polygon(buffered, region.crs),
        interval=TimeRange(
            start=interval.start
            - timedelta(seconds=_seconds(config.temporal_margin_before)),
            end=interval.end
            + timedelta(seconds=_seconds(config.temporal_margin_after)),
        ),
    )


def posterior_credible_region(
    posterior: SourcePosterior, percent: Literal[90, 95]
) -> SpatialGeometry:
    region = (
        posterior.credible_region_90
        if percent == 90
        else posterior.credible_region_95
    )
    if region is None:
        raise ValueError(f"posterior lacks its {percent}% credible region")
    if region.crs.is_geographic or region.crs.axis_units != (
        SpatialUnit.METRE,
        SpatialUnit.METRE,
    ):
        raise ValueError("posterior AIS buffering requires a projected metre CRS")
    return region


def posterior_release_interval(posterior: SourcePosterior) -> TimeRange:
    quantiles = posterior.release_time_quantiles
    if quantiles is None:
        raise ValueError("posterior lacks release-time quantiles")
    return TimeRange(start=quantiles.p10, end=quantiles.p90)
