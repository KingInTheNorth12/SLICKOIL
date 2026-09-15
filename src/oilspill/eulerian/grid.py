"""Incident-local projected grid construction and optional polygon land rasterization."""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from pydantic import Field, model_validator
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from oilspill.domain.common import (
    ArtifactRef,
    FrozenModel,
    Identifier,
    QualityFlag,
    QualitySeverity,
)
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.geospatial import CRS, AffineTransform, SpatialGeometry, SpatialUnit
from oilspill.eulerian.errors import EulerianTransportError
from oilspill.eulerian.initialization import crs_text, file_artifact


class EulerianGridBuilderConfig(FrozenModel):
    analysis_crs: Identifier
    cell_size_metres: float = Field(gt=0.0, allow_inf_nan=False)
    roi_padding_metres: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_cell_count: int = Field(gt=0)
    resampling_method: Literal["bilinear", "nearest"]
    land_mask_output_root: Path
    validation_note: Identifier

    @model_validator(mode="after")
    def _validation_required(self) -> EulerianGridBuilderConfig:
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("Eulerian grid settings must mark DATASET_VALIDATION_REQUIRED")
        return self


class EulerianGridBuildResult(FrozenModel):
    grid: EulerianGrid
    warnings: tuple[QualityFlag, ...] = ()
    land_source: ArtifactRef | None = None


def _mapping(value: SpatialGeometry) -> dict[str, Any]:
    geometry = value.geometry
    if geometry.type == "Point":
        return {"type": "Point", "coordinates": (geometry.coordinate.x, geometry.coordinate.y)}
    if geometry.type == "LineString":
        return {"type": "LineString", "coordinates": [(p.x, p.y) for p in geometry.coordinates]}
    if geometry.type == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [
                [(p.x, p.y) for p in ring] for ring in (geometry.exterior, *geometry.holes)
            ],
        }
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [[(p.x, p.y) for p in ring] for ring in (polygon.exterior, *polygon.holes)]
            for polygon in geometry.polygons
        ],
    }


class EulerianGridBuilder:
    def __init__(self, config: EulerianGridBuilderConfig) -> None:
        self.config = config
        target = rasterio.crs.CRS.from_user_input(config.analysis_crs)
        if not target.is_projected or (target.linear_units or "").lower() not in {
            "metre",
            "meter",
            "metres",
            "meters",
        }:
            raise EulerianTransportError("analysis CRS must be projected with metre units")
        authority = target.to_authority()
        self._crs = CRS(
            authority=authority[0] if authority else None,
            code=authority[1] if authority else None,
            wkt=target.to_wkt() if authority is None else None,
            is_geographic=False,
            axis_units=(SpatialUnit.METRE, SpatialUnit.METRE),
        )

    def build(
        self,
        roi: SpatialGeometry,
        *,
        land_geometries: tuple[SpatialGeometry, ...] = (),
        land_source: ArtifactRef | None = None,
    ) -> EulerianGridBuildResult:
        projected_roi = shape(
            transform_geom(crs_text(roi.crs), crs_text(self._crs), _mapping(roi), precision=-1)
        )
        if projected_roi.is_empty:
            raise EulerianTransportError("ROI is empty after projection")
        padding, cell = self.config.roi_padding_metres, self.config.cell_size_metres
        min_x, min_y, max_x, max_y = projected_roi.bounds
        left = math.floor((min_x - padding) / cell) * cell
        bottom = math.floor((min_y - padding) / cell) * cell
        right = math.ceil((max_x + padding) / cell) * cell
        top = math.ceil((max_y + padding) / cell) * cell
        width, height = int(round((right - left) / cell)), int(round((top - bottom) / cell))
        if width <= 0 or height <= 0 or width * height > self.config.maximum_cell_count:
            raise EulerianTransportError("analysis grid exceeds configured cell-count safety limit")
        transform = Affine(cell, 0, left, 0, -cell, top)
        base = EulerianGrid(
            crs=self._crs,
            width=width,
            height=height,
            transform=AffineTransform(a=cell, b=0, c=left, d=0, e=-cell, f=top),
            dx_metres=cell,
            dy_metres=cell,
        )
        polygons = []
        has_non_polygon = False
        for geometry in land_geometries:
            if geometry.geometry.type not in {"Polygon", "MultiPolygon"}:
                has_non_polygon = True
                continue
            polygons.append(
                shape(
                    transform_geom(
                        crs_text(geometry.crs),
                        crs_text(self._crs),
                        _mapping(geometry),
                        precision=-1,
                    )
                )
            )
        warnings: list[QualityFlag] = []
        mask_artifact = None
        if polygons:
            mask = rasterize(
                [(mapping(polygon), 1) for polygon in polygons],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            identity = sha256(
                (base.model_dump_json() + "".join(p.wkb_hex for p in polygons)).encode()
            ).hexdigest()
            path = self.config.land_mask_output_root / identity[:16] / "land-mask.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=width,
                    height=height,
                    count=1,
                    dtype="uint8",
                    crs=crs_text(self._crs),
                    transform=transform,
                    nodata=255,
                ) as dataset:
                    dataset.write(np.asarray(mask, dtype=np.uint8), 1)
            mask_artifact = file_artifact(path)
        else:
            warnings.append(
                QualityFlag(
                    code="LAND_MASK_UNAVAILABLE",
                    severity=QualitySeverity.WARNING,
                    message=(
                        "No polygonal land geometry was available; coastline lines do not "
                        "unambiguously define land/water"
                    ),
                )
            )
        if has_non_polygon and polygons:
            warnings.append(
                QualityFlag(
                    code="NON_POLYGON_LAND_GEOMETRY_IGNORED",
                    severity=QualitySeverity.INFO,
                    message="Line/point geometries were not used to infer land polygons",
                )
            )
        grid = base.model_copy(update={"land_mask": mask_artifact})
        return EulerianGridBuildResult(
            grid=grid,
            warnings=tuple(warnings),
            land_source=land_source if polygons else None,
        )
