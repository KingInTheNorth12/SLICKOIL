"""Deterministic low-dimensional features from coarse Eulerian forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from oilspill.artifacts import local_path_from_artifact
from oilspill.domain.eulerian import EulerianGrid, EulerianTransportResult
from oilspill.domain.geospatial import RasterAsset, SpatialGeometry
from oilspill.eulerian.grid import _mapping
from oilspill.eulerian.initialization import affine, crs_text, file_artifact, validate_raster

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoarseOutcomeFeatures:
    vector: tuple[float, ...]
    displacement_metres: float
    maximum_spread_metres: float
    coastal_contact: float
    final_cell_mass: FloatArray
    final_centroid: tuple[float, float]
    final_support_area_m2: float


def extract_outcome_features(
    result: EulerianTransportResult,
    grid: EulerianGrid,
    initial: RasterAsset,
    coastline: tuple[SpatialGeometry, ...],
    *,
    support_threshold: float,
    anisotropy_threshold: float,
) -> CoarseOutcomeFeatures:
    initial_mass, initial_centroid, _, _, _, _ = _raster_moments(
        initial, grid, support_threshold
    )
    if initial_mass <= 0:
        raise ValueError("Eulerian forecast initial raster has no mass")
    coast_mask = _coast_mask(coastline, grid)
    vector: list[float] = []
    final_mass = np.zeros((grid.height, grid.width), dtype=np.float64)
    final_centroid = initial_centroid
    final_area = 0.0
    maximum_spread = 0.0
    contact_any = 0.0
    maximum_displacement = 0.0
    for snapshot in result.snapshots:
        mass, centroid, covariance, area, support, cell_mass = _raster_moments(
            snapshot.concentration, grid, support_threshold
        )
        if mass <= 0:
            centroid = initial_centroid
            covariance = (0.0, 0.0, 0.0)
        xx, xy, yy = covariance
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray([[xx, xy], [xy, yy]]))
        largest = max(float(eigenvalues[-1]), 0.0)
        smallest = max(float(eigenvalues[0]), 0.0)
        stable = largest > 0 and (
            smallest == 0 or largest / smallest >= anisotropy_threshold
        )
        direction = float(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1])) if stable else 0.0
        contact = float(bool(coast_mask is not None and np.any(support & coast_mask)))
        displacement = float(np.linalg.norm(np.subtract(centroid, initial_centroid)))
        spread = float(np.sqrt(max(xx + yy, 0.0)))
        vector.extend(
            (
                centroid[0],
                centroid[1],
                xx,
                xy,
                yy,
                area,
                float(np.cos(direction)) if stable else 0.0,
                float(np.sin(direction)) if stable else 0.0,
                float(stable),
                contact,
            )
        )
        maximum_displacement = max(maximum_displacement, displacement)
        maximum_spread = max(maximum_spread, spread)
        contact_any = max(contact_any, contact)
        final_mass = np.asarray(cell_mass, dtype=np.float64).reshape(grid.height, grid.width)
        final_centroid, final_area = centroid, area
    return CoarseOutcomeFeatures(
        vector=tuple(vector),
        displacement_metres=maximum_displacement,
        maximum_spread_metres=maximum_spread,
        coastal_contact=contact_any,
        final_cell_mass=final_mass,
        final_centroid=final_centroid,
        final_support_area_m2=final_area,
    )


def _raster_moments(
    asset: RasterAsset, grid: EulerianGrid, threshold: float
) -> tuple[
    float,
    tuple[float, float],
    tuple[float, float, float],
    float,
    NDArray[np.bool_],
    FloatArray,
]:
    path = local_path_from_artifact(asset.artifact)
    if not path.is_file() or file_artifact(path).sha256 != asset.artifact.sha256:
        raise ValueError("forecast feature raster is missing or fails checksum validation")
    with rasterio.open(path) as dataset:
        validate_raster(dataset, grid, "forecast feature raster")
        values = np.asarray(dataset.read(1), dtype=np.float64)
        if dataset.nodata is not None:
            values[values == dataset.nodata] = 0.0
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    cell_area = grid.dx_metres * grid.dy_metres
    cell_mass = values * cell_area
    total = float(cell_mass.sum())
    maximum = float(values.max(initial=0.0))
    support = values >= maximum * threshold if maximum > 0 else np.zeros_like(values, dtype=bool)
    if total <= 0:
        return 0.0, (0.0, 0.0), (0.0, 0.0, 0.0), 0.0, support, cell_mass
    rows, columns = np.indices(values.shape, dtype=np.float64)
    xs = grid.transform.c + (columns + 0.5) * grid.transform.a
    ys = grid.transform.f + (rows + 0.5) * grid.transform.e
    normalized = cell_mass / total
    mean_x, mean_y = float((normalized * xs).sum()), float((normalized * ys).sum())
    dx, dy = xs - mean_x, ys - mean_y
    covariance = (
        float((normalized * dx * dx).sum()),
        float((normalized * dx * dy).sum()),
        float((normalized * dy * dy).sum()),
    )
    return (
        total,
        (mean_x, mean_y),
        covariance,
        float(np.count_nonzero(support) * cell_area),
        support,
        normalized,
    )


def _coast_mask(
    coastline: tuple[SpatialGeometry, ...], grid: EulerianGrid
) -> NDArray[np.bool_] | None:
    if not coastline:
        return None
    geometries = tuple(
        transform_geom(
            crs_text(item.crs),
            crs_text(grid.crs),
            _mapping(item),
            precision=-1,
        )
        for item in coastline
    )
    values = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=(grid.height, grid.width),
        transform=affine(grid),
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return np.asarray(values != 0, dtype=np.bool_)
