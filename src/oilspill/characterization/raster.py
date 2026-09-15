"""Characterize a georeferenced detection mask independently from model inference."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from scipy.ndimage import (  # type: ignore[import-untyped]
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    label,
)
from shapely import make_valid
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import unary_union

from oilspill.artifacts import UnsupportedArtifactURIError, local_path_from_artifact
from oilspill.characterization.config import SpillCharacterizationConfig
from oilspill.characterization.errors import (
    EmptySpillMaskError,
    InvalidSpillGeometryError,
    SpillCharacterizationError,
)
from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
)
from oilspill.domain.geospatial import (
    CRS,
    Angle,
    AngularUnit,
    Area,
    AreaUnit,
    Coordinate,
    Length,
    LengthUnit,
    LineStringGeometry,
    MultiPolygonGeometry,
    PointGeometry,
    PolygonGeometry,
    RasterAsset,
    RasterBand,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.domain.spill import SpillDetection, SpillGeometry


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _config_digest(config: SpillCharacterizationConfig) -> str:
    payload = config.model_dump(mode="json", exclude={"output_root"})
    return sha256(_canonical_json(payload).encode()).hexdigest()


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


def _safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise SpillCharacterizationError("detection_id is unsafe for an artifact path")
    return result


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for item in geometry.geoms for part in _polygon_parts(item)]
    return []


def _line_parts(geometry: Any) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString | GeometryCollection):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _polygon_to_domain(polygon: Polygon) -> PolygonGeometry:
    exterior = tuple(Coordinate(x=float(x), y=float(y)) for x, y in polygon.exterior.coords)
    holes = tuple(
        tuple(Coordinate(x=float(x), y=float(y)) for x, y in ring.coords)
        for ring in polygon.interiors
    )
    return PolygonGeometry(exterior=exterior, holes=holes)


def _area_geometry_to_domain(geometry: Any, crs: CRS) -> SpatialGeometry:
    parts = _polygon_parts(geometry)
    if not parts:
        raise InvalidSpillGeometryError("characterized geometry contains no polygonal area")
    if len(parts) == 1:
        value: PolygonGeometry | MultiPolygonGeometry = _polygon_to_domain(parts[0])
    else:
        value = MultiPolygonGeometry(polygons=tuple(_polygon_to_domain(item) for item in parts))
    return SpatialGeometry(geometry=value, crs=crs)


def _point_to_domain(point: Point, crs: CRS) -> SpatialGeometry:
    return SpatialGeometry(
        geometry=PointGeometry(coordinate=Coordinate(x=float(point.x), y=float(point.y))),
        crs=crs,
    )


def _line_to_domain(line: LineString, crs: CRS) -> SpatialGeometry:
    return SpatialGeometry(
        geometry=LineStringGeometry(
            coordinates=tuple(Coordinate(x=float(x), y=float(y)) for x, y in line.coords)
        ),
        crs=crs,
    )


class RasterSpillCharacterizer:
    """Polygonize, filter, skeletonize, and measure one canonical detection mask."""

    def __init__(self, config: SpillCharacterizationConfig) -> None:
        self._config = config
        self._measurement_raster_crs = rasterio.crs.CRS.from_user_input(config.measurement_crs)
        if not self._measurement_raster_crs.is_projected:
            raise ValueError(
                "measurement_crs must be projected; physical measurements in degrees are forbidden"
            )
        linear_units = (self._measurement_raster_crs.linear_units or "").lower()
        if linear_units not in {"metre", "meter", "metres", "meters"}:
            raise ValueError("measurement_crs must use metre axis units")
        authority = self._measurement_raster_crs.to_authority()
        self._measurement_domain_crs = CRS(
            authority=authority[0] if authority else None,
            code=authority[1] if authority else None,
            wkt=self._measurement_raster_crs.to_wkt() if authority is None else None,
            is_geographic=False,
            axis_units=(SpatialUnit.METRE, SpatialUnit.METRE),
        )

    def characterize(self, detection: SpillDetection) -> SpillGeometry:
        try:
            mask_path = local_path_from_artifact(detection.mask.artifact)
        except UnsupportedArtifactURIError as error:
            raise SpillCharacterizationError(str(error)) from error
        if not mask_path.is_file():
            raise SpillCharacterizationError(f"detection mask does not exist: {mask_path}")
        with rasterio.open(mask_path) as dataset:
            if dataset.crs is None:
                raise SpillCharacterizationError("detection mask has no CRS")
            if (
                dataset.width != detection.mask.grid.width
                or dataset.height != detection.mask.grid.height
            ):
                raise SpillCharacterizationError(
                    "detection mask dimensions do not match SpillDetection metadata"
                )
            band_index = detection.mask.bands[0].source_index or 1
            raw = dataset.read(band_index, masked=True)
            invalid_pixels = np.ma.getmaskarray(raw)  # type: ignore[no-untyped-call]
            foreground = np.isin(np.asarray(raw.data), self._config.foreground_values)
            foreground &= ~invalid_pixels
            filtered, flags = self._filter_components(foreground)
            source_geometry = self._polygonize(filtered, dataset.transform)
            source_raster_crs = dataset.crs
            profile = dataset.profile.copy()

        metric_geometry = self._transform(
            source_geometry, source_raster_crs, self._measurement_raster_crs
        )
        metric_geometry = self._validated_polygonal(metric_geometry)
        area = float(metric_geometry.area)
        if not math.isfinite(area) or area <= 0.0:
            raise InvalidSpillGeometryError("projected spill geometry has no positive finite area")
        centroid_metric = metric_geometry.centroid
        length, width, orientation, direction = self._oriented_dimensions(metric_geometry)
        centerline_metric = (
            self._centerline(metric_geometry, centroid_metric, direction)
            if self._config.create_centerline
            else None
        )
        centroid_source = self._transform(
            centroid_metric, self._measurement_raster_crs, source_raster_crs
        )
        centerline_source = (
            self._transform(centerline_metric, self._measurement_raster_crs, source_raster_crs)
            if centerline_metric is not None
            else None
        )
        skeleton = (
            self._write_skeleton(detection, filtered, invalid_pixels, profile)
            if self._config.create_skeleton
            else None
        )
        if len(_polygon_parts(source_geometry)) > 1 and centerline_source is not None:
            flags.append(
                QualityFlag(
                    code="DOMINANT_COMPONENT_CENTERLINE",
                    severity=QualitySeverity.INFO,
                    message=(
                        "centerline represents the longest intersection across disconnected regions"
                    ),
                )
            )
        return SpillGeometry(
            geometry_id=f"geometry:{detection.detection_id}",
            detection_id=detection.detection_id,
            polygon=_area_geometry_to_domain(source_geometry, detection.mask.grid.crs),
            centroid=_point_to_domain(cast(Point, centroid_source), detection.mask.grid.crs),
            centerline=(
                _line_to_domain(cast(LineString, centerline_source), detection.mask.grid.crs)
                if centerline_source is not None
                else None
            ),
            skeleton=skeleton,
            area=Area(value=area, unit=AreaUnit.SQUARE_METRE),
            length=Length(value=length, unit=LengthUnit.METRE),
            width=Length(value=width, unit=LengthUnit.METRE),
            orientation=Angle(
                value=orientation,
                unit=AngularUnit.DEGREE,
                convention="clockwise_from_projected_grid_north_modulo_180",
            ),
            measurement_crs=self._measurement_domain_crs,
            method=self.component_metadata(),
            quality_flags=tuple(flags),
        )

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="raster_mask_characterizer",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="rasterio-shapely-scipy",
            configuration_sha256=_config_digest(self._config),
            attributes=(
                MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                MetadataEntry(
                    key="measurement_crs_validation",
                    value=self._config.measurement_crs_validation_note,
                ),
                MetadataEntry(
                    key="component_threshold_validation",
                    value=self._config.threshold_validation_note,
                ),
                MetadataEntry(key="dimension_method", value="minimum_rotated_rectangle"),
                MetadataEntry(key="skeleton_method", value="morphological_skeleton"),
            ),
        )

    def _filter_components(
        self, foreground: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], list[QualityFlag]]:
        structure = generate_binary_structure(2, 1 if self._config.connectivity == 4 else 2)
        labels, count = label(foreground, structure=structure)
        if count == 0:
            raise EmptySpillMaskError("detection mask contains no configured foreground values")
        sizes = np.bincount(labels.reshape(-1))[1:]
        retained = np.flatnonzero(sizes >= self._config.minimum_component_pixels) + 1
        if len(retained) == 0:
            raise EmptySpillMaskError("no foreground component meets minimum_component_pixels")
        flags: list[QualityFlag] = []
        removed = int(count - len(retained))
        if removed:
            flags.append(
                QualityFlag(
                    code="TINY_COMPONENTS_REMOVED",
                    severity=QualitySeverity.INFO,
                    message=f"removed {removed} components below configured pixel threshold",
                )
            )
        if self._config.component_policy == "largest" and len(retained) > 1:
            selected = retained[int(np.argmax(sizes[retained - 1]))]
            discarded = len(retained) - 1
            retained = np.asarray([selected])
            flags.append(
                QualityFlag(
                    code="DISCONNECTED_COMPONENTS_DISCARDED",
                    severity=QualitySeverity.WARNING,
                    message=f"largest-component policy discarded {discarded} valid components",
                )
            )
        elif len(retained) > 1:
            flags.append(
                QualityFlag(
                    code="DISCONNECTED_COMPONENTS_RETAINED",
                    severity=QualitySeverity.INFO,
                    message=f"retained {len(retained)} disconnected components",
                )
            )
        return np.isin(labels, retained), flags

    def _polygonize(self, mask: np.ndarray[Any, Any], transform: Any) -> Any:
        polygons = [
            shape(geometry)
            for geometry, value in shapes(
                mask.astype("uint8"),
                mask=mask,
                transform=transform,
                connectivity=self._config.connectivity,
            )
            if int(value) == 1
        ]
        if not polygons:
            raise EmptySpillMaskError("polygonization produced no foreground geometry")
        geometry = unary_union(polygons)
        return self._validated_polygonal(geometry)

    def _validated_polygonal(self, geometry: Any) -> Any:
        if not geometry.is_valid:
            if self._config.invalid_polygon_policy == "reject":
                raise InvalidSpillGeometryError("polygonized spill geometry is invalid")
            geometry = make_valid(geometry)
        parts = [part for part in _polygon_parts(geometry) if not part.is_empty and part.area > 0]
        if not parts:
            raise InvalidSpillGeometryError("geometry repair produced no polygonal area")
        repaired = unary_union(parts)
        if repaired.is_empty or not repaired.is_valid:
            raise InvalidSpillGeometryError(
                "spill geometry remains invalid after configured repair"
            )
        return repaired

    @staticmethod
    def _transform(geometry: Any, source_crs: Any, destination_crs: Any) -> Any:
        transformed = transform_geom(
            source_crs,
            destination_crs,
            mapping(geometry),
            precision=-1,
        )
        return shape(transformed)

    @staticmethod
    def _oriented_dimensions(geometry: Any) -> tuple[float, float, float, tuple[float, float]]:
        hull = geometry.convex_hull
        if not isinstance(hull, Polygon):
            raise InvalidSpillGeometryError("spill convex hull is not polygonal")
        points = np.asarray(hull.exterior.coords[:-1], dtype="float64")
        reference = points.mean(axis=0)
        points -= reference
        candidates: list[tuple[float, float, float, tuple[float, float]]] = []
        for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
            delta = end - start
            angle = math.atan2(float(delta[1]), float(delta[0]))
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated_x = points[:, 0] * cosine + points[:, 1] * sine
            rotated_y = -points[:, 0] * sine + points[:, 1] * cosine
            x_span = float(np.ptp(rotated_x))
            y_span = float(np.ptp(rotated_y))
            if x_span >= y_span:
                direction = (cosine, sine)
            else:
                direction = (-sine, cosine)
            candidates.append(
                (x_span * y_span, max(x_span, y_span), min(x_span, y_span), direction)
            )
        _, longest, shortest, unit = min(candidates, key=lambda item: item[0])
        if shortest <= 0.0 or longest <= 0.0:
            raise InvalidSpillGeometryError("minimum rotated rectangle is degenerate")
        orientation = math.degrees(math.atan2(unit[0], unit[1])) % 180.0
        return longest, shortest, orientation, unit

    @staticmethod
    def _centerline(geometry: Any, centroid: Point, direction: tuple[float, float]) -> LineString:
        min_x, min_y, max_x, max_y = geometry.bounds
        reach = math.hypot(max_x - min_x, max_y - min_y) * 2.0
        candidate = LineString(
            (
                (centroid.x - direction[0] * reach, centroid.y - direction[1] * reach),
                (centroid.x + direction[0] * reach, centroid.y + direction[1] * reach),
            )
        )
        segments = [
            item for item in _line_parts(geometry.intersection(candidate)) if item.length > 0
        ]
        if not segments:
            raise InvalidSpillGeometryError("could not derive a non-empty dominant centerline")
        return max(segments, key=lambda item: item.length)

    def _write_skeleton(
        self,
        detection: SpillDetection,
        foreground: np.ndarray[Any, Any],
        invalid_pixels: np.ndarray[Any, Any],
        profile: dict[str, Any],
    ) -> RasterAsset:
        structure = generate_binary_structure(2, 1 if self._config.connectivity == 4 else 2)
        remaining = foreground.copy()
        skeleton = np.zeros_like(foreground, dtype=bool)
        while remaining.any():
            eroded = binary_erosion(remaining, structure=structure)
            opened = binary_dilation(eroded, structure=structure)
            skeleton |= remaining & ~opened
            remaining = eroded
        values = skeleton.astype("uint8")
        values[invalid_pixels] = 255
        output_dir = (
            self._config.output_root
            / _safe_identifier(detection.detection_id)
            / sha256(
                f"{detection.mask.artifact.sha256}:{_config_digest(self._config)}".encode()
            ).hexdigest()[:16]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "slick-skeleton.tif"
        temporary = output.with_suffix(".tmp.tif")
        for inherited_option in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(inherited_option, None)
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        try:
            with rasterio.open(temporary, "w", **profile) as dataset:
                dataset.write(values, 1)
                dataset.set_band_description(1, "morphological slick skeleton")
                dataset.update_tags(
                    SOURCE_DETECTION_ID=detection.detection_id,
                    SCIENTIFIC_THRESHOLDS=self._config.threshold_validation_note,
                )
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return RasterAsset(
            artifact=_artifact(output, "image/tiff; application=geotiff"),
            grid=detection.mask.grid,
            bands=(RasterBand(name="slick-skeleton", unit="binary", nodata=255),),
        )


def create_raster_characterizer(config: ComponentConfig) -> RasterSpillCharacterizer:
    return RasterSpillCharacterizer(SpillCharacterizationConfig.model_validate(config.settings))
