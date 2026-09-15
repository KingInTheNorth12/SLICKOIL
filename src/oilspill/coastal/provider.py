"""Configured GeoJSON coastline adapter producing canonical geometries."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import rasterio
from pydantic import Field
from shapely.geometry import shape

from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    DatasetMetadata,
    FrozenModel,
    Identifier,
    UTCDateTime,
)
from oilspill.domain.geospatial import (
    CRS,
    Coordinate,
    LineStringGeometry,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.requests import CoastlineDataset


class GeoJSONCoastlineConfig(FrozenModel):
    path: Path
    dataset_id: Identifier
    dataset_name: Identifier
    dataset_version: Identifier
    provider_name: Identifier
    coordinate_crs: Identifier
    retrieved_at: UTCDateTime | None = None
    license: str | None = None
    geometry_types: tuple[str, ...] = Field(min_length=1)
    schema_validation_note: Identifier


class GeoJSONCoastlineProvider:
    def __init__(self, config: GeoJSONCoastlineConfig) -> None:
        self._config = config

    def load(self) -> CoastlineDataset:
        path = self._config.path
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        source_crs = rasterio.crs.CRS.from_user_input(self._config.coordinate_crs)
        authority = source_crs.to_authority()
        if source_crs.is_geographic:
            axis_units = (SpatialUnit.DEGREE, SpatialUnit.DEGREE)
        else:
            linear_units = (source_crs.linear_units or "").lower()
            if linear_units in {"metre", "meter", "metres", "meters"}:
                axis_units = (SpatialUnit.METRE, SpatialUnit.METRE)
            elif linear_units in {"kilometre", "kilometer", "kilometres", "kilometers"}:
                axis_units = (SpatialUnit.KILOMETRE, SpatialUnit.KILOMETRE)
            else:
                raise ValueError(
                    "unsupported coastline CRS axis units; DATASET_VALIDATION_REQUIRED"
                )
        crs = CRS(
            authority=authority[0] if authority else None,
            code=authority[1] if authority else None,
            wkt=source_crs.to_wkt() if authority is None else None,
            is_geographic=bool(source_crs.is_geographic),
            axis_units=axis_units,
        )
        segments: list[SpatialGeometry] = []
        for feature in payload.get("features", ()):
            geometry = shape(feature["geometry"])
            if geometry.geom_type not in self._config.geometry_types:
                continue
            lines = geometry.geoms if geometry.geom_type == "MultiLineString" else (geometry,)
            segments.extend(
                SpatialGeometry(
                    geometry=LineStringGeometry(
                        coordinates=tuple(
                            Coordinate(x=float(x), y=float(y)) for x, y in line.coords
                        )
                    ),
                    crs=crs,
                )
                for line in lines
            )
        if not segments:
            raise ValueError("configured coastline contains no supported line geometry")
        return CoastlineDataset(
            dataset_id=self._config.dataset_id,
            segments=tuple(segments),
            source=self._artifact(),
            metadata=self.dataset_metadata(),
        )

    def dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=self._config.dataset_name,
            version=self._config.dataset_version,
            provider=self._config.provider_name,
            retrieved_at=self._config.retrieved_at,
            license=self._config.license,
            manifest=self._artifact(),
        )

    def _artifact(self) -> ArtifactRef:
        content = self._config.path.read_bytes()
        return ArtifactRef(
            uri=self._config.path.resolve().as_uri(),
            media_type="application/geo+json",
            sha256=sha256(content).hexdigest(),
            byte_size=len(content),
        )


def create_geojson_coastline_provider(config: ComponentConfig) -> GeoJSONCoastlineProvider:
    return GeoJSONCoastlineProvider(GeoJSONCoastlineConfig.model_validate(config.settings))
