"""Coastal-impact analysis independent of drift simulation."""

from oilspill.coastal.analyzer import (
    GeometryCoastalImpactAnalyzer,
    GeometryCoastalImpactConfig,
    create_geometry_coastal_impact,
)
from oilspill.coastal.provider import GeoJSONCoastlineProvider, create_geojson_coastline_provider

__all__ = [
    "GeometryCoastalImpactAnalyzer",
    "GeometryCoastalImpactConfig",
    "GeoJSONCoastlineProvider",
    "create_geometry_coastal_impact",
    "create_geojson_coastline_provider",
]
