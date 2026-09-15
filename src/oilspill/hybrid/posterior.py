"""Deterministic empirical posterior products for high-fidelity source support."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from rasterio.features import shapes
from rasterio.transform import rowcol
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from oilspill.artifacts import write_artifact
from oilspill.domain.common import FrozenModel, QualityFlag
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.geospatial import (
    Coordinate,
    MultiPolygonGeometry,
    PointGeometry,
    PolygonGeometry,
    SpatialGeometry,
)
from oilspill.domain.hybrid import (
    HighFidelitySourceEvaluation,
    HybridConvergenceStatus,
    ReleaseTimeQuantiles,
    SourcePosterior,
    WeightedSourceState,
)
from oilspill.eulerian.initialization import affine, file_artifact, read_land_mask, write_raster
from oilspill.hybrid.config import SelectiveRefinementConfig
from oilspill.hybrid.high_fidelity import RawHighFidelityEvaluation
from oilspill.hybrid.refinement import normalized_support_weights


class PosteriorManifest(FrozenModel):
    observation_id: str
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_ids: tuple[str, ...]
    source_state_ids: tuple[str, ...]
    normalized_weights: tuple[float, ...]
    semantics: str


FloatArray = NDArray[np.float64]


def weighted_release_quantiles(
    evaluations: tuple[RawHighFidelityEvaluation, ...], weights: FloatArray
) -> ReleaseTimeQuantiles:
    ordered = sorted(
        zip(evaluations, weights, strict=True),
        key=lambda item: (item[0].source_state.release_time, item[0].source_state.source_state_id),
    )
    cumulative = np.cumsum([float(weight) for _, weight in ordered])

    def at(probability: float) -> datetime:
        index = int(np.searchsorted(cumulative, probability, side="left"))
        return ordered[min(index, len(ordered) - 1)][0].source_state.release_time

    return ReleaseTimeQuantiles(p10=at(0.10), p50=at(0.50), p90=at(0.90))


def weighted_source_centroid(
    evaluations: tuple[RawHighFidelityEvaluation, ...], weights: FloatArray
) -> SpatialGeometry:
    points = [item.source_state.geometry.geometry for item in evaluations]
    if not all(isinstance(point, PointGeometry) for point in points):
        raise ValueError("posterior source states must contain Point geometry")
    point_values = [point for point in points if isinstance(point, PointGeometry)]
    x = sum(
        point.coordinate.x * float(weight)
        for point, weight in zip(point_values, weights, strict=True)
    )
    y = sum(
        point.coordinate.y * float(weight)
        for point, weight in zip(point_values, weights, strict=True)
    )
    return SpatialGeometry(
        geometry=PointGeometry(coordinate=Coordinate(x=x, y=y)),
        crs=evaluations[0].source_state.geometry.crs,
    )


def _ring(coordinates: Iterable[Sequence[float]]) -> tuple[Coordinate, ...]:
    return tuple(Coordinate(x=float(item[0]), y=float(item[1])) for item in coordinates)


def _spatial_polygon(value: Polygon | MultiPolygon, grid: EulerianGrid) -> SpatialGeometry:
    polygons = (value,) if isinstance(value, Polygon) else tuple(value.geoms)
    converted = tuple(
        PolygonGeometry(
            exterior=_ring(polygon.exterior.coords),
            holes=tuple(_ring(interior.coords) for interior in polygon.interiors),
        )
        for polygon in polygons
    )
    geometry = converted[0] if len(converted) == 1 else MultiPolygonGeometry(polygons=converted)
    return SpatialGeometry(geometry=geometry, crs=grid.crs)


def _credible_region(
    cell_mass: FloatArray, target_mass: float, grid: EulerianGrid
) -> SpatialGeometry:
    flat = cell_mass.ravel()
    valid = np.flatnonzero(flat > 0.0)
    if not valid.size:
        raise ValueError("posterior raster contains no positive source mass")
    ordered = sorted(valid.tolist(), key=lambda index: (-float(flat[index]), index))
    selected = np.zeros(flat.shape, dtype=np.uint8)
    cumulative = 0.0
    for index in ordered:
        selected[index] = 1
        cumulative += float(flat[index])
        if cumulative >= target_mass:
            break
    mask = selected.reshape(cell_mass.shape)
    polygons = [
        shape(geometry)
        for geometry, value in shapes(mask, mask=mask.astype(bool), transform=affine(grid))
        if value == 1
    ]
    merged = unary_union(polygons)
    if not isinstance(merged, (Polygon, MultiPolygon)):
        raise ValueError("credible cells did not polygonize to polygonal geometry")
    return _spatial_polygon(merged, grid)


def build_source_posterior(
    observation_id: str,
    evaluations: tuple[RawHighFidelityEvaluation, ...],
    grid: EulerianGrid,
    config: SelectiveRefinementConfig,
    convergence_status: HybridConvergenceStatus,
    warnings: tuple[QualityFlag, ...],
    *,
    temperature: float,
) -> tuple[SourcePosterior, tuple[HighFidelitySourceEvaluation, ...]]:
    scores = np.asarray([item.mismatch_score for item in evaluations], dtype=np.float64)
    weights = normalized_support_weights(scores, temperature)
    weighted_states = tuple(
        WeightedSourceState(source_state=item.source_state, normalized_weight=float(weight))
        for item, weight in zip(evaluations, weights, strict=True)
    )
    high_fidelity = tuple(
        HighFidelitySourceEvaluation(
            source_state=item.source_state,
            simulation=item.simulation,
            mismatch_score=item.mismatch_score,
            normalized_posterior_weight=float(weight),
            generation_number=item.generation_number,
            soft_iou=item.mismatch.soft_iou,
            centroid_error_metres=item.mismatch.centroid_error_metres,
            centroid_error=item.mismatch.centroid_error,
            area_error_square_metres=item.mismatch.area_error_square_metres,
            area_error=item.mismatch.area_error,
            covariance_shape_error=item.mismatch.shape_error,
        )
        for item, weight in zip(evaluations, weights, strict=True)
    )
    land = read_land_mask(grid)
    cell_mass = np.zeros((grid.height, grid.width), dtype=np.float64)
    for item, weight in zip(evaluations, weights, strict=True):
        point = item.source_state.geometry.geometry
        assert point.type == "Point"
        row, column = rowcol(affine(grid), point.coordinate.x, point.coordinate.y)
        if 0 <= row < grid.height and 0 <= column < grid.width and not land[row, column]:
            cell_mass[row, column] += float(weight)
    total = float(cell_mass.sum())
    if total <= 0:
        raise ValueError("no high-fidelity source state maps to a water grid cell")
    cell_mass /= total
    digest_input = ":".join(
        f"{item.source_state.source_state_id}:{weight:.17g}"
        for item, weight in zip(evaluations, weights, strict=True)
    )
    identity = sha256((digest_input + config.model_dump_json()).encode()).hexdigest()
    raster = write_raster(
        config.output_root / identity[:16] / "source-posterior.tif",
        cell_mass / (grid.dx_metres * grid.dy_metres),
        grid,
        land,
        config.posterior_output_nodata,
    )
    manifest = PosteriorManifest(
        observation_id=observation_id,
        configuration_sha256=sha256(config.model_dump_json().encode()).hexdigest(),
        simulation_ids=tuple(item.simulation.simulation_id for item in evaluations),
        source_state_ids=tuple(item.source_state.source_state_id for item in evaluations),
        normalized_weights=tuple(float(weight) for weight in weights),
        semantics="weighted model support; not proof of origin",
    )
    content = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_path = config.output_root / identity[:16] / "source-posterior.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_bytes() != content:
            raise ValueError("deterministic posterior path contains different content")
        provenance = file_artifact(manifest_path, "application/json")
    else:
        provenance = write_artifact(manifest_path, content)
    posterior = SourcePosterior(
        posterior_id=f"source-posterior:{identity[:24]}",
        observation_id=observation_id,
        weighted_source_states=weighted_states,
        weighted_source_centroid=weighted_source_centroid(evaluations, weights),
        source_probability_raster=raster,
        credible_region_50=_credible_region(cell_mass, 0.50, grid),
        credible_region_90=_credible_region(cell_mass, 0.90, grid),
        credible_region_95=_credible_region(cell_mass, 0.95, grid),
        release_time_quantiles=weighted_release_quantiles(evaluations, weights),
        calibrated=False,
        convergence_status=convergence_status,
        warnings=warnings,
        provenance=provenance,
    )
    return posterior, high_fidelity
