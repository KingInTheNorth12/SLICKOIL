"""Explicit-column CSV adapter producing only canonical AIS domain objects."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError
from rasterio.warp import transform
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from oilspill.ais.config import CSVAISProviderConfig
from oilspill.ais.errors import AISIngestionError
from oilspill.config import ComponentConfig
from oilspill.domain.ais import AISPoint
from oilspill.domain.common import ArtifactRef, DatasetMetadata
from oilspill.domain.geospatial import (
    CRS,
    Angle,
    AngularUnit,
    Coordinate,
    PointGeometry,
    PolygonGeometry,
    SpatialGeometry,
    SpatialUnit,
    Speed,
)
from oilspill.requests import AISQuery

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


def _source_artifact(path: Path) -> ArtifactRef:
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type="text/csv",
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _crs_text(crs: CRS) -> str:
    if crs.authority and crs.code:
        return f"{crs.authority}:{crs.code}"
    if crs.wkt:
        return crs.wkt
    raise AISIngestionError("query geometry CRS has no usable identifier")


def _polygon(polygon: PolygonGeometry) -> Polygon:
    exterior = [(item.x, item.y) for item in polygon.exterior]
    holes = [[(item.x, item.y) for item in ring] for ring in polygon.holes]
    return Polygon(exterior, holes)


def _query_shape(area: SpatialGeometry) -> Point | LineString | Polygon | MultiPolygon:
    geometry = area.geometry
    if geometry.type == "Point":
        return Point(geometry.coordinate.x, geometry.coordinate.y)
    if geometry.type == "LineString":
        return LineString([(item.x, item.y) for item in geometry.coordinates])
    if geometry.type == "Polygon":
        return _polygon(geometry)
    return MultiPolygon([_polygon(item) for item in geometry.polygons])


class CSVAISProvider:
    """Map an explicitly described local CSV schema into canonical AIS points.

    The initial adapter supports configured EPSG:4326 coordinates only. The eventual AIS
    dataset's coordinate reference remains explicit and DATASET_VALIDATION_REQUIRED.
    """

    def __init__(self, config: CSVAISProviderConfig) -> None:
        self._config = config

    def fetch_points(self, query: AISQuery) -> tuple[AISPoint, ...]:
        path = self._config.path
        if not path.is_file():
            raise AISIngestionError(f"AIS CSV does not exist: {path}")
        area = _query_shape(query.area)
        destination_crs = _crs_text(query.area.crs)
        points: list[AISPoint] = []
        with path.open(encoding=self._config.encoding, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=self._config.delimiter)
            self._validate_columns(reader.fieldnames)
            for row_number, row in enumerate(reader, start=2):
                try:
                    point = self._map_row(row)
                    x_values, y_values = transform(
                        self._config.coordinate_crs,
                        destination_crs,
                        [point.longitude],
                        [point.latitude],
                    )
                    if (
                        query.interval.start <= point.observed_at <= query.interval.end
                        and (not query.vessel_ids or point.vessel_id in query.vessel_ids)
                        and area.covers(Point(x_values[0], y_values[0]))
                    ):
                        points.append(point)
                except (KeyError, TypeError, ValueError, ValidationError) as error:
                    if self._config.invalid_row_policy == "skip":
                        continue
                    raise AISIngestionError(
                        f"invalid AIS record at CSV row {row_number}"
                    ) from error
        return tuple(points)

    def dataset_metadata(self) -> DatasetMetadata:
        path = self._config.path
        if not path.is_file():
            raise AISIngestionError(f"AIS CSV does not exist: {path}")
        return DatasetMetadata(
            name=self._config.dataset_name,
            version=self._config.dataset_version,
            provider=self._config.provider_name,
            license=self._config.license,
            manifest=_source_artifact(path),
        )

    def _validate_columns(self, fieldnames: Sequence[str] | None) -> None:
        if fieldnames is None:
            raise AISIngestionError("AIS CSV has no header")
        columns = self._config.columns
        configured = (
            columns.vessel_id,
            columns.timestamp,
            columns.longitude,
            columns.latitude,
            columns.speed_over_ground,
            columns.course_over_ground,
            columns.heading,
            columns.navigation_status,
            columns.record_id,
        )
        missing = tuple(item for item in configured if item is not None and item not in fieldnames)
        if missing:
            raise AISIngestionError(f"configured AIS columns are absent from CSV: {missing}")

    def _map_row(self, row: dict[str, str | None]) -> AISPoint:
        columns = self._config.columns
        longitude = float(self._required(row, columns.longitude))
        latitude = float(self._required(row, columns.latitude))
        return AISPoint(
            vessel_id=self._required(row, columns.vessel_id),
            observed_at=self._timestamp(self._required(row, columns.timestamp)),
            position=SpatialGeometry(
                geometry=PointGeometry(coordinate=Coordinate(x=longitude, y=latitude)),
                crs=WGS84,
            ),
            speed_over_ground=self._speed(row),
            course_over_ground=self._angle(
                row, columns.course_over_ground, self._config.course_unit
            ),
            heading=self._angle(row, columns.heading, self._config.heading_unit),
            navigation_status=self._optional(row, columns.navigation_status),
            provider_record_id=self._optional(row, columns.record_id),
        )

    @staticmethod
    def _required(row: dict[str, str | None], column: str) -> str:
        value = row[column]
        if value is None or not value.strip():
            raise ValueError(f"required AIS column {column!r} is empty")
        return value.strip()

    @staticmethod
    def _optional(row: dict[str, str | None], column: str | None) -> str | None:
        if column is None:
            return None
        value = row[column]
        return value.strip() if value and value.strip() else None

    def _timestamp(self, value: str) -> datetime:
        if self._config.timestamp_format == "ISO8601":
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.strptime(value, self._config.timestamp_format)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            timezone_name = self._config.naive_timestamp_timezone
            if timezone_name is None:
                raise ValueError("naive AIS timestamp has no configured source timezone")
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
            except ZoneInfoNotFoundError as error:
                raise ValueError(f"unknown AIS source timezone {timezone_name!r}") from error
        return parsed.astimezone(UTC)

    def _speed(self, row: dict[str, str | None]) -> Speed | None:
        column = self._config.columns.speed_over_ground
        value = self._optional(row, column)
        if value is None:
            return None
        if self._config.speed_unit is None:
            raise ValueError("speed unit is not configured")
        return Speed(value=float(value), unit=self._config.speed_unit)

    def _angle(
        self,
        row: dict[str, str | None],
        column: str | None,
        unit: str | None,
    ) -> Angle | None:
        value = self._optional(row, column)
        if value is None:
            return None
        if unit is None or self._config.angle_convention is None:
            raise ValueError("angle unit or convention is not configured")
        return Angle(
            value=float(value),
            unit=AngularUnit(unit),
            convention=self._config.angle_convention,
        )


def create_csv_ais_provider(config: ComponentConfig) -> CSVAISProvider:
    return CSVAISProvider(CSVAISProviderConfig.model_validate(config.settings))
