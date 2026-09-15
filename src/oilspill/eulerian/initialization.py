"""Raster/point initialization adapters for the Eulerian kernel."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import model_validator
from rasterio.enums import Resampling
from rasterio.transform import Affine, rowcol
from rasterio.warp import reproject, transform

from oilspill.artifacts import local_path_from_artifact
from oilspill.domain.common import ArtifactRef, FrozenModel, Identifier
from oilspill.domain.eulerian import (
    EulerianGrid,
    ObservedSlickInitialCondition,
    PointSourceInitialCondition,
)
from oilspill.domain.geospatial import PointGeometry, RasterAsset, RasterBand, RasterGrid
from oilspill.domain.spill import SpillObservation
from oilspill.eulerian.errors import EulerianTransportError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class EulerianInitializationConfig(FrozenModel):
    output_root: Path
    raster_resampling: Literal["bilinear", "nearest"]
    output_nodata: float = -9999.0
    validation_note: Identifier

    @model_validator(mode="after")
    def _valid_settings(self) -> EulerianInitializationConfig:
        if not np.isfinite(self.output_nodata) or self.output_nodata >= 0:
            raise ValueError("initialization nodata must be finite and negative")
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("initialization settings must mark DATASET_VALIDATION_REQUIRED")
        return self


def crs_text(crs: object) -> str:
    authority, code, wkt = crs.authority, crs.code, crs.wkt  # type: ignore[attr-defined]
    if authority and code:
        return f"{authority}:{code}"
    if wkt:
        return str(wkt)
    raise EulerianTransportError("CRS lacks an identifier")


def affine(grid: EulerianGrid) -> Affine:
    t = grid.transform
    return Affine(t.a, t.b, t.c, t.d, t.e, t.f)


def file_artifact(path: Path, media_type: str = "image/tiff; application=geotiff") -> ArtifactRef:
    content = path.read_bytes()
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type=media_type,
        sha256=sha256(content).hexdigest(),
        byte_size=len(content),
    )


def validate_raster(dataset: rasterio.io.DatasetReader, grid: EulerianGrid, label: str) -> None:
    expected_crs = rasterio.crs.CRS.from_user_input(crs_text(grid.crs))
    if dataset.width != grid.width or dataset.height != grid.height:
        raise EulerianTransportError(f"{label} dimensions do not match Eulerian grid")
    if dataset.transform != affine(grid) or dataset.crs != expected_crs:
        raise EulerianTransportError(f"{label} georeferencing does not match Eulerian grid")


def read_land_mask(grid: EulerianGrid) -> BoolArray:
    if grid.land_mask is None:
        return np.zeros((grid.height, grid.width), dtype=np.bool_)
    path = local_path_from_artifact(grid.land_mask)
    if not path.is_file() or file_artifact(path).sha256 != grid.land_mask.sha256:
        raise EulerianTransportError("land mask is missing or fails checksum validation")
    with rasterio.open(path) as dataset:
        validate_raster(dataset, grid, "land mask")
        return np.asarray(dataset.read(1) != 0, dtype=np.bool_)


def normalized_point_field(
    initial: PointSourceInitialCondition,
    grid: EulerianGrid,
    land: BoolArray,
    *,
    method: Literal["containing_cell", "gaussian"],
    gaussian_sigma_metres: float | None,
) -> FloatArray:
    point = initial.geometry.geometry
    if not isinstance(point, PointGeometry):
        raise EulerianTransportError("point release requires Point geometry")
    xs, ys = transform(
        crs_text(initial.geometry.crs),
        crs_text(grid.crs),
        [point.coordinate.x],
        [point.coordinate.y],
    )
    row, column = rowcol(affine(grid), xs[0], ys[0])
    if not (0 <= row < grid.height and 0 <= column < grid.width):
        raise EulerianTransportError("point source falls outside analysis grid")
    if land[row, column]:
        raise EulerianTransportError("point source falls on land")
    if method == "containing_cell":
        values = np.zeros((grid.height, grid.width), dtype=np.float64)
        values[row, column] = 1.0
    else:
        if gaussian_sigma_metres is None:
            raise EulerianTransportError("Gaussian initialization requires sigma")
        columns = np.arange(grid.width, dtype=np.float64) + 0.5
        rows = np.arange(grid.height, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(
            grid.transform.a * columns + grid.transform.c,
            grid.transform.e * rows + grid.transform.f,
        )
        values = np.exp(-((xx - xs[0]) ** 2 + (yy - ys[0]) ** 2) / (2 * gaussian_sigma_metres**2))
    return normalize(values, land, grid)


def normalize(values: FloatArray, land: BoolArray, grid: EulerianGrid) -> FloatArray:
    clean = np.where(land, 0.0, np.maximum(np.nan_to_num(values, nan=0.0), 0.0))
    mass = float(clean.sum() * grid.dx_metres * grid.dy_metres)
    if mass <= 0:
        raise EulerianTransportError("initial condition has no positive water-cell mass")
    return np.asarray(clean / mass, dtype=np.float64)


class SpillRasterInitializer:
    def __init__(self, config: EulerianInitializationConfig) -> None:
        self.config = config

    def from_observation(
        self, observation: SpillObservation, grid: EulerianGrid
    ) -> ObservedSlickInitialCondition:
        detection = observation.detection
        source = detection.probability_raster or detection.mask
        source_kind: Literal["probability_raster", "binary_mask"] = (
            "probability_raster" if detection.probability_raster is not None else "binary_mask"
        )
        path = local_path_from_artifact(source.artifact)
        if not path.is_file() or file_artifact(path).sha256 != source.artifact.sha256:
            raise EulerianTransportError("slick raster is missing or fails checksum validation")
        destination = np.zeros((grid.height, grid.width), dtype=np.float64)
        with rasterio.open(path) as dataset:
            values = np.asarray(dataset.read(1), dtype=np.float64)
            if dataset.nodata is not None:
                values[values == dataset.nodata] = 0.0
            reproject(
                values,
                destination,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=affine(grid),
                dst_crs=crs_text(grid.crs),
                src_nodata=dataset.nodata,
                dst_nodata=0.0,
                resampling=(
                    Resampling.bilinear
                    if self.config.raster_resampling == "bilinear"
                    else Resampling.nearest
                ),
            )
        land = read_land_mask(grid)
        normalized = normalize(destination, land, grid)
        identity = sha256(
            (
                source.artifact.sha256 + grid.model_dump_json() + self.config.model_dump_json()
            ).encode()
        ).hexdigest()
        output = self.config.output_root / identity[:16] / "slick-initial-condition.tif"
        asset = write_raster(output, normalized, grid, land, self.config.output_nodata)
        return ObservedSlickInitialCondition(
            observation_id=observation.observation_id,
            observed_at=observation.observed_at,
            concentration=asset,
            source_raster=source.artifact,
            source_kind=source_kind,
            assumptions=(
                "Raster-derived normalized relative surface mass; not absolute oil mass.",
                self.config.validation_note,
            ),
        )


def write_raster(
    target: Path,
    values: FloatArray,
    grid: EulerianGrid,
    land: BoolArray,
    nodata: float,
) -> RasterAsset:
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.asarray(values, dtype=np.float64).copy()
    output[land] = nodata
    with NamedTemporaryFile(dir=target.parent, suffix=".tmp.tif", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="float64",
            crs=crs_text(grid.crs),
            transform=affine(grid),
            nodata=nodata,
        ) as dataset:
            dataset.write(output, 1)
            dataset.set_band_description(1, "relative_surface_mass_density")
            dataset.update_tags(
                units="relative_mass_per_square_metre",
                semantics="relative surface mass; not tonnes or calibrated probability",
            )
        if target.exists():
            if target.read_bytes() != temporary.read_bytes():
                raise EulerianTransportError("deterministic raster path contains different content")
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != temporary.read_bytes():
                    raise EulerianTransportError("concurrent raster output differs") from None
    finally:
        temporary.unlink(missing_ok=True)
    return RasterAsset(
        artifact=file_artifact(target),
        grid=RasterGrid(
            width=grid.width,
            height=grid.height,
            crs=grid.crs,
            transform=grid.transform,
        ),
        bands=(
            RasterBand(
                name="relative_surface_mass_density",
                unit="relative_mass_per_square_metre",
                nodata=nodata,
                source_index=1,
            ),
        ),
    )
