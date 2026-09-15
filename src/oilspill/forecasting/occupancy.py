"""Cell intersections count members, never particles; empty denominators remain missing."""

import json
import math

from rasterio.warp import transform_geom
from shapely.geometry import Polygon, shape

from oilspill.domain.drift import DriftSimulation
from oilspill.domain.ensemble import OccupancySummary
from oilspill.domain.geospatial import SpatialGeometry
from oilspill.forecasting.planning import EnsemblePlan


def geometry_mapping(spatial: SpatialGeometry) -> dict[str, object]:
    value = spatial.geometry
    if value.type == "Point":
        return {"type": "Point", "coordinates": [value.coordinate.x, value.coordinate.y]}
    if value.type == "LineString":
        return {"type": "LineString", "coordinates": [[p.x, p.y] for p in value.coordinates]}
    polygons = [value] if value.type == "Polygon" else value.polygons
    coordinates = [
        [[[p.x, p.y] for p in ring] for ring in (p.exterior, *p.holes)] for p in polygons
    ]
    return {"type": "MultiPolygon", "coordinates": coordinates}


def aggregate_occupancy(
    plan: EnsemblePlan, simulations: tuple[DriftSimulation, ...]
) -> tuple[OccupancySummary, bytes]:
    grid = plan.config.grid
    target = grid.crs.wkt or f"{grid.crs.authority}:{grid.crs.code}"
    supports = []
    for simulation in simulations:
        distribution = simulation.target_distribution
        if distribution is None:
            raise ValueError("valid member needs support geometry")
        spatial = distribution.comparison_geometry
        source = spatial.crs.wkt or f"{spatial.crs.authority}:{spatial.crs.code}"
        support = shape(transform_geom(source, target, geometry_mapping(spatial)))
        if (
            support.is_empty
            or not support.is_valid
            or not all(math.isfinite(v) for v in support.bounds)
        ):
            raise ValueError("member support must be nonempty, finite and valid")
        supports.append(support)
    denominator = len(supports)
    summary = OccupancySummary(
        requested=len(plan.members),
        succeeded=denominator,
        failed=len(plan.members) - denominator,
        denominator=denominator,
        denominator_policy=plan.config.denominator_policy,
        grid=grid,
        horizon=plan.members[0].request.valid_until,
        warnings=(
            "Support intersections include cell boundaries; convex hulls may bridge empty areas.",
            "Failed members are excluded; frequency is conditional on valid members.",
            "Zero valid members produces null cells, not zero frequency.",
            "DATASET_VALIDATION_REQUIRED",
        ),
    )
    counts = []
    for row in range(grid.height):
        values = []
        for column in range(grid.width):
            t = grid.transform
            x, y = t.c + column * t.a, t.f + row * t.e
            cell = Polygon([(x, y), (x + t.a, y), (x + t.a, y + t.e), (x, y + t.e)])
            values.append(sum(s.intersects(cell) for s in supports))
        counts.append(values)
    frequencies = [[n / denominator if denominator else None for n in row] for row in counts]
    payload = {
        "summary": summary.model_dump(mode="json"),
        "member_counts": counts,
        "occupancy_frequency": frequencies,
    }
    return summary, json.dumps(payload, sort_keys=True, allow_nan=False).encode()
