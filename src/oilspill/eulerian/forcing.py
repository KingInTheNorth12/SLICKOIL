"""Adapter from canonical environmental NetCDF fields to kernel-ready velocities."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pydantic import model_validator
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from oilspill.artifacts import local_path_from_artifact
from oilspill.domain.common import FrozenModel, Identifier, QualityFlag, QualitySeverity
from oilspill.domain.environment import EnvironmentalField, EnvironmentalVariable, GridKind
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.geospatial import CRS
from oilspill.eulerian.errors import EulerianForcingError

FloatArray = NDArray[np.float64]


class EulerianForcingConfig(FrozenModel):
    spatial_resampling: Literal["bilinear", "nearest"]
    allow_missing_wind: bool
    approximate_east_north_as_projected_xy: bool
    validation_note: Identifier

    @model_validator(mode="after")
    def _validation_required(self) -> EulerianForcingConfig:
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("Eulerian forcing settings must mark DATASET_VALIDATION_REQUIRED")
        return self


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def _crs(crs: CRS) -> str:
    authority, code, wkt = crs.authority, crs.code, crs.wkt
    if authority and code:
        return f"{authority}:{code}"
    if wkt:
        return str(wkt)
    raise EulerianForcingError("forcing/grid CRS lacks an identifier")


def _affine(grid: EulerianGrid) -> Affine:
    t = grid.transform
    return Affine(t.a, t.b, t.c, t.d, t.e, t.f)


class EnvironmentalForcingAdapter:
    """Lazily interpolate incident-local canonical fields onto solver times."""

    def __init__(
        self,
        fields: tuple[EnvironmentalField, ...],
        grid: EulerianGrid,
        start_time: datetime,
        end_time: datetime,
        windage: float,
        config: EulerianForcingConfig,
        *,
        wind_bias_u_m_s: float = 0.0,
        wind_bias_v_m_s: float = 0.0,
        current_bias_u_m_s: float = 0.0,
        current_bias_v_m_s: float = 0.0,
    ) -> None:
        self._start = start_time
        self._windage = windage
        self._wind_bias_u_m_s = wind_bias_u_m_s
        self._wind_bias_v_m_s = wind_bias_v_m_s
        self._current_bias_u_m_s = current_bias_u_m_s
        self._current_bias_v_m_s = current_bias_v_m_s
        self._zeros = np.zeros((grid.height, grid.width), dtype=np.float64)
        self._series: dict[EnvironmentalVariable, tuple[FloatArray, FloatArray, FloatArray]] = {}
        self.warnings: tuple[QualityFlag, ...] = ()
        by_variable = {field.variable: field for field in fields}
        if len(by_variable) != len(fields):
            raise EulerianForcingError("only one field per forcing variable is supported")
        if EnvironmentalVariable.OCEAN_CURRENT not in by_variable:
            raise EulerianForcingError("ocean current forcing is required")
        if EnvironmentalVariable.WIND not in by_variable and not config.allow_missing_wind:
            raise EulerianForcingError("wind forcing is required unless explicitly optional")
        unsupported = set(by_variable) - {
            EnvironmentalVariable.OCEAN_CURRENT,
            EnvironmentalVariable.WIND,
        }
        if unsupported:
            raise EulerianForcingError(f"unsupported forcing variables: {sorted(unsupported)}")
        cross_crs = any(field.grid.crs != grid.crs for field in fields)
        if cross_crs:
            if not config.approximate_east_north_as_projected_xy:
                raise EulerianForcingError(
                    "cross-CRS forcing requires explicit east/north-as-x/y approximation"
                )
            if "DATASET_VALIDATION_REQUIRED" not in config.validation_note:
                raise EulerianForcingError(
                    "forcing approximation requires DATASET_VALIDATION_REQUIRED"
                )
            self.warnings = (
                QualityFlag(
                    code="FORCING_VECTOR_BASIS_ROTATION_OMITTED",
                    severity=QualitySeverity.WARNING,
                    message=(
                        "Local east/north forcing is approximated as projected x/y after scalar "
                        "resampling; vector rotation is omitted; DATASET_VALIDATION_REQUIRED"
                    ),
                ),
            )
        for variable, field in by_variable.items():
            if (
                field.interval.start > start_time
                or field.interval.end < end_time
                or field.timestamps[0] > start_time
                or field.timestamps[-1] < end_time
                or len(set(field.timestamps)) != len(field.timestamps)
            ):
                raise EulerianForcingError("forcing interval does not cover solver times")
            self._series[variable] = self._load(field, grid, config.spatial_resampling)

    def velocity_at_seconds(self, elapsed: float) -> tuple[FloatArray, FloatArray]:
        timestamp = self._start.timestamp() + elapsed
        current_u, current_v = self._at(EnvironmentalVariable.OCEAN_CURRENT, timestamp)
        wind_u, wind_v = self._at(EnvironmentalVariable.WIND, timestamp)
        return (
            current_u
            + self._current_bias_u_m_s
            + self._windage * (wind_u + self._wind_bias_u_m_s),
            current_v
            + self._current_bias_v_m_s
            + self._windage * (wind_v + self._wind_bias_v_m_s),
        )

    def _at(
        self, variable: EnvironmentalVariable, timestamp: float
    ) -> tuple[FloatArray, FloatArray]:
        series = self._series.get(variable)
        if series is None:
            return self._zeros, self._zeros
        times, u, v = series
        upper = int(np.searchsorted(times, timestamp, side="right"))
        if upper == 0:
            return u[0], v[0]
        if upper >= len(times):
            return u[-1], v[-1]
        lower = upper - 1
        fraction = (timestamp - times[lower]) / (times[upper] - times[lower])
        return (
            u[lower] * (1 - fraction) + u[upper] * fraction,
            v[lower] * (1 - fraction) + v[upper] * fraction,
        )

    @staticmethod
    def _load(
        field: EnvironmentalField,
        grid: EulerianGrid,
        resampling: Literal["bilinear", "nearest"],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        expected = (
            ("current_u", "current_v")
            if field.variable == EnvironmentalVariable.OCEAN_CURRENT
            else ("wind_u", "wind_v")
        )
        variables = tuple(
            component.artifact_variable or component.name for component in field.components
        )
        if set(variables) != set(expected) or len(variables) != 2:
            raise EulerianForcingError(f"missing required canonical components {expected}")
        if any(component.unit != "metre_per_second" for component in field.components):
            raise EulerianForcingError("forcing components must use metre_per_second")
        path = local_path_from_artifact(field.values)
        if not path.is_file() or _hash(path) != field.values.sha256:
            raise EulerianForcingError("forcing artifact is missing or fails checksum validation")
        with xr.open_dataset(path) as dataset:
            for name in expected:
                if name not in dataset:
                    raise EulerianForcingError(f"canonical forcing variable {name!r} is missing")
                if dataset[name].attrs.get("units") != "metre_per_second":
                    raise EulerianForcingError(
                        f"canonical forcing variable {name!r} has invalid units"
                    )
            spatial_dims = field.dimension_order[1:]
            coordinates = [
                np.asarray(dataset[name].values, dtype=np.float64) for name in spatial_dims
            ]
            arrays = [
                np.asarray(dataset[name].transpose(*field.dimension_order).values, dtype=np.float64)
                for name in expected
            ]
        mapped = tuple(_resample(array, field, coordinates, grid, resampling) for array in arrays)
        times = np.asarray([time.timestamp() for time in field.timestamps], dtype=np.float64)
        return times, mapped[0], mapped[1]


def _resample(
    values: FloatArray,
    field: EnvironmentalField,
    coordinates: list[FloatArray],
    grid: EulerianGrid,
    method: Literal["bilinear", "nearest"],
) -> FloatArray:
    if values.ndim != 3 or values.shape[0] != len(field.timestamps):
        raise EulerianForcingError("forcing must have time plus two spatial dimensions")
    if field.grid.kind == GridKind.RECTILINEAR:
        y, x = coordinates
        if len(x) < 2 or len(y) < 2:
            raise EulerianForcingError("rectilinear forcing requires two points per axis")
        oriented = values[:, :: -1 if y[0] < y[-1] else 1, :: -1 if x[0] > x[-1] else 1]
        xs, ys = np.sort(x), np.sort(y)
        dx, dy = float(np.median(np.diff(xs))), float(np.median(np.diff(ys)))
        if not np.allclose(np.diff(xs), dx) or not np.allclose(np.diff(ys), dy):
            raise EulerianForcingError("forcing coordinates must be regularly spaced")
        source_transform = Affine(dx, 0, xs[0] - dx / 2, 0, -dy, ys[-1] + dy / 2)
    elif field.grid.kind == GridKind.REGULAR_RASTER and field.grid.transform is not None:
        t = field.grid.transform
        source_transform = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
        oriented = values
    else:
        raise EulerianForcingError("unsupported forcing grid for MVP resampling")
    output = np.empty((len(field.timestamps), grid.height, grid.width), dtype=np.float64)
    algorithm = Resampling.bilinear if method == "bilinear" else Resampling.nearest
    for index, source in enumerate(oriented):
        destination = np.full((grid.height, grid.width), np.nan, dtype=np.float64)
        reproject(
            source,
            destination,
            src_transform=source_transform,
            src_crs=_crs(field.grid.crs),
            dst_transform=_affine(grid),
            dst_crs=_crs(grid.crs),
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=algorithm,
        )
        if not np.isfinite(destination).all():
            raise EulerianForcingError("forcing does not cover the analysis grid")
        output[index] = destination
    return output
