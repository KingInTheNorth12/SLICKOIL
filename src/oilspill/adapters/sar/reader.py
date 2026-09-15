"""Canonical georeferenced raster ingestion without Sentinel product-layout assumptions."""

from __future__ import annotations

from pathlib import Path

import rasterio

from oilspill.adapters.sar.config import RasterioSARReaderConfig
from oilspill.adapters.sar.errors import DatasetValidationRequiredError, SARInputError
from oilspill.artifacts import UnsupportedArtifactURIError
from oilspill.artifacts import local_path_from_artifact as _local_path_from_artifact
from oilspill.domain.common import ArtifactRef, DatasetMetadata
from oilspill.domain.geospatial import (
    CRS,
    AffineTransform,
    Coordinate,
    PolygonGeometry,
    RasterAsset,
    RasterBand,
    RasterGrid,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.domain.sar import SARScene
from oilspill.requests import SARReadRequest


def local_path_from_artifact(artifact: ArtifactRef) -> Path:
    """Backward-compatible SAR-specific wrapper around the shared artifact utility."""

    try:
        return _local_path_from_artifact(artifact)
    except UnsupportedArtifactURIError as error:
        raise SARInputError(f"SAR raster adapter {error}") from error


def _domain_crs(source: rasterio.crs.CRS) -> CRS:
    authority = source.to_authority()
    is_geographic = bool(source.is_geographic)
    if is_geographic:
        units = (SpatialUnit.DEGREE, SpatialUnit.DEGREE)
    else:
        linear_units = (source.linear_units or "").lower()
        if linear_units in {"metre", "meter", "metres", "meters"}:
            units = (SpatialUnit.METRE, SpatialUnit.METRE)
        elif linear_units in {"kilometre", "kilometer", "kilometres", "kilometers"}:
            units = (SpatialUnit.KILOMETRE, SpatialUnit.KILOMETRE)
        else:
            raise DatasetValidationRequiredError(
                "unsupported projected CRS axis units; DATASET_VALIDATION_REQUIRED"
            )
    return CRS(
        authority=authority[0] if authority else None,
        code=authority[1] if authority else None,
        wkt=source.to_wkt() if authority is None else None,
        is_geographic=is_geographic,
        axis_units=units,
    )


class RasterioSARReader:
    """Read a configured GeoTIFF-like raster into the canonical ``SARScene`` contract."""

    def __init__(self, config: RasterioSARReaderConfig) -> None:
        self._config = config

    def read(self, request: SARReadRequest) -> SARScene:
        path = local_path_from_artifact(request.source)
        if not path.is_file():
            raise SARInputError(f"SAR raster does not exist: {path}")
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise DatasetValidationRequiredError(
                    "SAR raster has no CRS; DATASET_VALIDATION_REQUIRED"
                )
            mapping = self._config.bands.ordered()
            missing = tuple(index for _, index, _ in mapping if index > dataset.count)
            if missing:
                raise SARInputError(
                    f"configured polarization band indexes exceed raster count: {missing}"
                )
            crs = _domain_crs(dataset.crs)
            transform = dataset.transform
            grid = RasterGrid(
                width=dataset.width,
                height=dataset.height,
                crs=crs,
                transform=AffineTransform(
                    a=transform.a,
                    b=transform.b,
                    c=transform.c,
                    d=transform.d,
                    e=transform.e,
                    f=transform.f,
                ),
            )
            bands = tuple(
                RasterBand(
                    name=name,
                    unit=unit,
                    nodata=dataset.nodatavals[index - 1],
                    source_index=index,
                )
                for name, index, unit in mapping
            )
            bounds = dataset.bounds
            footprint = SpatialGeometry(
                geometry=PolygonGeometry(
                    exterior=(
                        Coordinate(x=bounds.left, y=bounds.bottom),
                        Coordinate(x=bounds.right, y=bounds.bottom),
                        Coordinate(x=bounds.right, y=bounds.top),
                        Coordinate(x=bounds.left, y=bounds.top),
                        Coordinate(x=bounds.left, y=bounds.bottom),
                    )
                ),
                crs=crs,
            )

        scene_id = request.scene_id or path.stem
        return SARScene(
            scene_id=scene_id,
            platform=self._config.platform,
            product_type=self._config.product_type,
            acquisition=self._config.acquisition,
            polarizations=tuple(name for name, _, _ in mapping),
            footprint=footprint,
            raster=RasterAsset(artifact=request.source, grid=grid, bands=bands),
            source=request.source,
            processing_level=self._config.processing_level,
            metadata=self._config.metadata,
        )

    def dataset_metadata(self) -> DatasetMetadata:
        """A raster file is an artifact, not a versioned source dataset."""

        return DatasetMetadata(
            name="configured-sar-raster",
            version=requested_version(self._config),
            provider="rasterio",
        )


def requested_version(config: RasterioSARReaderConfig) -> str:
    """Stable reader-config identity used where a dataset version is required by the port."""

    from hashlib import sha256

    return sha256(config.model_dump_json().encode()).hexdigest()
