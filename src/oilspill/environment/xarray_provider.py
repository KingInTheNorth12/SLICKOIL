"""Map explicitly configured xarray variables into canonical environmental fields."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from rasterio.warp import transform_geom
from shapely.geometry import shape

from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    DatasetMetadata,
    QualityFlag,
    QualitySeverity,
    TimeRange,
)
from oilspill.domain.environment import (
    EnvironmentalField,
    EnvironmentalVariable,
    FieldComponent,
    GridDefinition,
    GridKind,
)
from oilspill.domain.geospatial import (
    CRS,
    Coordinate,
    PolygonGeometry,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.environment.config import (
    EnvironmentalVariableMapping,
    XarrayEnvironmentalProviderConfig,
)
from oilspill.environment.errors import EnvironmentalDataError
from oilspill.requests import EnvironmentalQuery

WGS84 = CRS(
    authority="EPSG",
    code="4326",
    is_geographic=True,
    axis_units=(SpatialUnit.DEGREE, SpatialUnit.DEGREE),
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type=media_type,
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _crs_text(geometry: SpatialGeometry) -> str:
    crs = geometry.crs
    if crs.authority and crs.code:
        return f"{crs.authority}:{crs.code}"
    if crs.wkt:
        return crs.wkt
    raise EnvironmentalDataError("query geometry CRS has no usable identifier")


def _geometry_mapping(value: SpatialGeometry) -> dict[str, Any]:
    geometry = value.geometry
    if geometry.type == "Point":
        return {"type": "Point", "coordinates": (geometry.coordinate.x, geometry.coordinate.y)}
    if geometry.type == "LineString":
        return {
            "type": "LineString",
            "coordinates": [(item.x, item.y) for item in geometry.coordinates],
        }
    if geometry.type == "Polygon":
        rings = [geometry.exterior, *geometry.holes]
        return {
            "type": "Polygon",
            "coordinates": [[(item.x, item.y) for item in ring] for ring in rings],
        }
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [[(item.x, item.y) for item in ring] for ring in (polygon.exterior, *polygon.holes)]
            for polygon in geometry.polygons
        ],
    }


def _utc_datetime(value: object) -> datetime:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise EnvironmentalDataError("environmental time coordinate contains NaT")
        nanoseconds = value.astype("datetime64[ns]").astype("int64")
        return datetime.fromtimestamp(float(nanoseconds) / 1_000_000_000.0, tz=UTC)
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    raise EnvironmentalDataError(f"unsupported decoded environmental time value {type(value)}")


class XarrayEnvironmentalProvider:
    """Subset and canonicalize a configured local rectilinear xarray dataset."""

    def __init__(self, config: XarrayEnvironmentalProviderConfig) -> None:
        self._config = config

    def fetch(self, query: EnvironmentalQuery) -> tuple[EnvironmentalField, ...]:
        if not query.variables:
            return ()
        if not self._config.source.is_file():
            raise EnvironmentalDataError(
                f"environmental source does not exist: {self._config.source}"
            )
        with xr.open_dataset(self._config.source, engine=self._config.engine) as source:
            self._validate_source(source)
            timestamps = tuple(_utc_datetime(value) for value in source[self._config.time].values)
            if timestamps != tuple(sorted(timestamps)) or len(set(timestamps)) != len(timestamps):
                raise EnvironmentalDataError(
                    "environmental time coordinates must be unique and chronologically ordered"
                )
            latitudes = np.asarray(source[self._config.latitude].values, dtype="float64")
            longitudes = np.asarray(source[self._config.longitude].values, dtype="float64")
            self._validate_spatial_coordinates(latitudes, longitudes)
            self._validate_vertical_reference(query)
            bounds = self._query_bounds(query.area)
            time_indexes, latitude_indexes, longitude_indexes, clipped = self._indexes(
                query, timestamps, latitudes, longitudes, bounds
            )
            subset = source.isel(
                {
                    self._config.time: time_indexes,
                    self._config.latitude: latitude_indexes,
                    self._config.longitude: longitude_indexes,
                }
            ).load()
        canonical = self._canonical_dataset(subset, query.variables)
        output = self._write_output(canonical, query)
        artifact = _artifact(output, "application/x-netcdf")
        selected_times = tuple(timestamps[index] for index in time_indexes)
        selected_latitudes = latitudes[latitude_indexes]
        selected_longitudes = longitudes[longitude_indexes]
        grid = self._grid(selected_latitudes, selected_longitudes)
        flags = (
            (
                QualityFlag(
                    code="QUERY_CLIPPED_TO_AVAILABLE_FORCING",
                    severity=QualitySeverity.WARNING,
                    message="query exceeded source bounds and was clipped by configured policy",
                ),
            )
            if clipped
            else ()
        )
        fields = tuple(
            self._field(variable, selected_times, grid, artifact, flags)
            for variable in query.variables
        )
        return fields

    def dataset_metadata(self) -> DatasetMetadata:
        if not self._config.source.is_file():
            raise EnvironmentalDataError(
                f"environmental source does not exist: {self._config.source}"
            )
        return DatasetMetadata(
            name=self._config.dataset_name,
            version=self._config.dataset_version,
            provider=self._config.provider_name,
            license=self._config.license,
            manifest=_artifact(self._config.source, "application/x-netcdf"),
        )

    def _validate_source(self, source: xr.Dataset) -> None:
        mappings = self._mappings().values()
        required = {
            self._config.time,
            self._config.latitude,
            self._config.longitude,
            *(mapping.source_name for mapping in mappings),
        }
        missing = tuple(sorted(required.difference(source.variables)))
        if missing:
            raise EnvironmentalDataError(
                f"configured environmental variables are missing: {missing}"
            )
        for mapping in self._mappings().values():
            variable = source[mapping.source_name]
            if tuple(variable.dims) != self._config.source_dimension_order:
                raise EnvironmentalDataError(
                    f"{mapping.source_name} dimensions {variable.dims} do not match configured "
                    f"order {self._config.source_dimension_order}"
                )
            actual_unit = variable.attrs.get("units")
            if actual_unit != mapping.source_unit:
                raise EnvironmentalDataError(
                    f"{mapping.source_name} unit {actual_unit!r} does not match configured "
                    f"source unit {mapping.source_unit!r}"
                )
        for coordinate in (self._config.time, self._config.latitude, self._config.longitude):
            if source[coordinate].ndim != 1:
                raise EnvironmentalDataError(
                    "initial xarray adapter requires one-dimensional rectilinear coordinates"
                )

    def _mappings(self) -> dict[str, EnvironmentalVariableMapping]:
        return {
            "wind_u": self._config.wind_u,
            "wind_v": self._config.wind_v,
            "current_u": self._config.current_u,
            "current_v": self._config.current_v,
        }

    def _validate_spatial_coordinates(
        self,
        latitudes: np.ndarray[Any, Any],
        longitudes: np.ndarray[Any, Any],
    ) -> None:
        for name, values in (("latitude", latitudes), ("longitude", longitudes)):
            if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
                raise EnvironmentalDataError(f"{name} coordinates must be finite and unique")
            differences = np.diff(values)
            if not ((differences > 0).all() or (differences < 0).all()):
                raise EnvironmentalDataError(f"{name} coordinates must be strictly monotonic")
        if (latitudes < -90.0).any() or (latitudes > 90.0).any():
            raise EnvironmentalDataError("latitude coordinates fall outside degree bounds")
        if self._config.longitude_convention == "-180_180" and (
            (longitudes < -180.0).any() or (longitudes > 180.0).any()
        ):
            raise EnvironmentalDataError("longitude coordinates violate -180_180 convention")
        if self._config.longitude_convention == "0_360" and (
            (longitudes < 0.0).any() or (longitudes >= 360.0).any()
        ):
            raise EnvironmentalDataError("longitude coordinates violate 0_360 convention")

    def _validate_vertical_reference(self, query: EnvironmentalQuery) -> None:
        requested = query.requested_vertical_reference
        if requested is None:
            return
        requested_mappings: list[EnvironmentalVariableMapping] = []
        if EnvironmentalVariable.WIND in query.variables:
            requested_mappings.extend((self._config.wind_u, self._config.wind_v))
        if EnvironmentalVariable.OCEAN_CURRENT in query.variables:
            requested_mappings.extend((self._config.current_u, self._config.current_v))
        mismatched = tuple(
            mapping.source_name
            for mapping in requested_mappings
            if mapping.vertical_reference != requested
        )
        if mismatched:
            raise EnvironmentalDataError(
                f"requested vertical reference is unavailable for mapped variables: {mismatched}"
            )

    def _query_bounds(self, area: SpatialGeometry) -> tuple[float, float, float, float]:
        geometry = _geometry_mapping(area)
        if _crs_text(area) != "EPSG:4326":
            geometry = transform_geom(_crs_text(area), "EPSG:4326", geometry, precision=-1)
        return tuple(float(value) for value in shape(geometry).bounds)  # type: ignore[return-value]

    def _indexes(
        self,
        query: EnvironmentalQuery,
        timestamps: tuple[datetime, ...],
        latitudes: np.ndarray[Any, Any],
        longitudes: np.ndarray[Any, Any],
        bounds: tuple[float, float, float, float],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
        min_x, min_y, max_x, max_y = bounds
        if self._config.longitude_convention == "0_360":
            min_x %= 360.0
            max_x %= 360.0
            if min_x > max_x:
                raise EnvironmentalDataError(
                    "antimeridian-crossing 0_360 queries require a future dataset adapter; "
                    "DATASET_VALIDATION_REQUIRED"
                )
        time_array = np.asarray(timestamps, dtype=object)
        time_mask = np.asarray(
            [query.interval.start <= item <= query.interval.end for item in timestamps]
        )
        latitude_mask = (latitudes >= min_y) & (latitudes <= max_y)
        longitude_mask = (longitudes >= min_x) & (longitudes <= max_x)
        outside = (
            query.interval.start < min(timestamps)
            or query.interval.end > max(timestamps)
            or min_y < float(latitudes.min())
            or max_y > float(latitudes.max())
            or min_x < float(longitudes.min())
            or max_x > float(longitudes.max())
        )
        if outside and self._config.boundary_policy == "strict":
            raise EnvironmentalDataError("query exceeds environmental source boundaries")
        indexes = tuple(np.flatnonzero(mask) for mask in (time_mask, latitude_mask, longitude_mask))
        if any(len(index) == 0 for index in indexes):
            raise EnvironmentalDataError("query has no samples on the configured source grid")
        del time_array
        return indexes[0], indexes[1], indexes[2], outside

    def _canonical_dataset(
        self,
        subset: xr.Dataset,
        variables: tuple[EnvironmentalVariable, ...],
    ) -> xr.Dataset:
        result = xr.Dataset(
            coords={
                "time": subset[self._config.time].values,
                "latitude": subset[self._config.latitude].values,
                "longitude": subset[self._config.longitude].values,
            },
            attrs={
                "canonical_schema": "oilspill.environment.v1",
                "source_dataset": self._config.dataset_name,
                "mapping_validation": self._config.mapping_validation_note,
                "DATASET_VALIDATION_REQUIRED": "true",
            },
        )
        requested_names: set[str] = set()
        if EnvironmentalVariable.WIND in variables:
            requested_names.update(("wind_u", "wind_v"))
        if EnvironmentalVariable.OCEAN_CURRENT in variables:
            requested_names.update(("current_u", "current_v"))
        unsupported = set(variables).difference(
            {EnvironmentalVariable.WIND, EnvironmentalVariable.OCEAN_CURRENT}
        )
        if unsupported:
            raise EnvironmentalDataError(f"initial adapter does not map variables: {unsupported}")
        for canonical_name in sorted(requested_names):
            mapping = self._mappings()[canonical_name]
            values = subset[mapping.source_name] * mapping.scale_factor + mapping.add_offset
            result[canonical_name] = values.rename(
                {
                    self._config.time: "time",
                    self._config.latitude: "latitude",
                    self._config.longitude: "longitude",
                }
            ).transpose("time", "latitude", "longitude")
            result[canonical_name].attrs = {
                "units": mapping.canonical_unit,
                "source_variable": mapping.source_name,
                "source_units": mapping.source_unit,
                "vertical_reference": mapping.vertical_reference or "unspecified",
            }
        result["time"].attrs["timezone"] = "UTC"
        result["latitude"].attrs["units"] = "degrees_north"
        result["longitude"].attrs["units"] = "degrees_east"
        return result

    def _write_output(self, dataset: xr.Dataset, query: EnvironmentalQuery) -> Path:
        identity = sha256(
            (
                self._config.model_dump_json()
                + query.model_dump_json()
                + _sha256_file(self._config.source)
            ).encode()
        ).hexdigest()
        output = self._config.output_root / identity[:16] / "environmental_forcing.nc"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.nc")
        try:
            dataset.to_netcdf(temporary, engine="scipy")
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return output

    @staticmethod
    def _grid(latitudes: np.ndarray[Any, Any], longitudes: np.ndarray[Any, Any]) -> GridDefinition:
        min_lat, max_lat = float(latitudes.min()), float(latitudes.max())
        min_lon, max_lon = float(longitudes.min()), float(longitudes.max())
        extent = SpatialGeometry(
            geometry=PolygonGeometry(
                exterior=(
                    Coordinate(x=min_lon, y=min_lat),
                    Coordinate(x=max_lon, y=min_lat),
                    Coordinate(x=max_lon, y=max_lat),
                    Coordinate(x=min_lon, y=max_lat),
                    Coordinate(x=min_lon, y=min_lat),
                )
            ),
            crs=WGS84,
        )
        return GridDefinition(
            kind=GridKind.RECTILINEAR,
            shape=(len(latitudes), len(longitudes)),
            crs=WGS84,
            extent=extent,
        )

    def _field(
        self,
        variable: EnvironmentalVariable,
        timestamps: tuple[datetime, ...],
        grid: GridDefinition,
        artifact: ArtifactRef,
        flags: tuple[QualityFlag, ...],
    ) -> EnvironmentalField:
        names = (
            ("wind_u", "wind_v")
            if variable == EnvironmentalVariable.WIND
            else ("current_u", "current_v")
        )
        components = tuple(
            FieldComponent(
                name=name.removeprefix("wind_").removeprefix("current_"),
                unit=self._mappings()[name].canonical_unit,
                artifact_variable=name,
                vertical_reference=self._mappings()[name].vertical_reference,
            )
            for name in names
        )
        if self._config.coordinate_crs != "EPSG:4326":
            raise EnvironmentalDataError("unsupported configured coordinate CRS")
        return EnvironmentalField(
            field_id=f"{self._config.dataset_name}:{variable.value}:{artifact.sha256[:12]}",
            variable=variable,
            components=components,
            dimension_order=("time", "latitude", "longitude"),
            interval=TimeRange(start=timestamps[0], end=timestamps[-1]),
            timestamps=timestamps,
            grid=grid,
            values=artifact,
            dataset=self.dataset_metadata(),
            data_kind=self._config.data_kind,
            quality_flags=flags,
        )


def create_xarray_environmental_provider(config: ComponentConfig) -> XarrayEnvironmentalProvider:
    settings = XarrayEnvironmentalProviderConfig.model_validate(config.settings)
    return XarrayEnvironmentalProvider(settings)
