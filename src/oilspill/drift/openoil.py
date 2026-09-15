"""OpenOil/OpenDrift isolated behind the generic DriftModel contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import xarray as xr
from rasterio.warp import transform_geom
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon, mapping, shape

from oilspill.config import ComponentConfig
from oilspill.domain.common import ArtifactRef, ComponentMetadata, MetadataEntry, TimeRange
from oilspill.domain.drift import (
    DriftMode,
    DriftSimulation,
    ParticleDistribution,
    ParticlePosition,
    ParticleTrajectory,
    ReleaseHypothesis,
    SimulationDiagnostics,
    SimulationStatus,
)
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.geospatial import (
    CRS,
    Coordinate,
    LineStringGeometry,
    PointGeometry,
    PolygonGeometry,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.drift.config import OpenOilDriftConfig, _seconds
from oilspill.drift.errors import DriftDependencyError, DriftExecutionError
from oilspill.requests import BackwardTraceRequest, DriftForecastRequest, ForwardTraceRequest

WGS84 = CRS(
    authority="EPSG",
    code="4326",
    is_geographic=True,
    axis_units=(SpatialUnit.DEGREE, SpatialUnit.DEGREE),
)


def _local_path(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme not in ("", "file"):
        raise DriftExecutionError(
            f"OpenOil requires local forcing artifacts, got {parsed.scheme!r}"
        )
    return Path(unquote(parsed.path if parsed.scheme else artifact.uri))


def _safe(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise DriftExecutionError("simulation identifier cannot form a safe output path")
    return result


def _artifact(path: Path) -> ArtifactRef:
    digest = sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type="application/x-netcdf",
        sha256=digest,
        byte_size=path.stat().st_size,
    )


class OpenOilDriftModel:
    """Execute OpenOil while returning only canonical ``DriftSimulation`` values."""

    def __init__(self, config: OpenOilDriftConfig, *, component_name: str = "openoil") -> None:
        self._config = config
        self._component_name = component_name

    def forward_source_trace(self, request: ForwardTraceRequest) -> DriftSimulation:
        release_time = self._release_time(request.release)
        return self._run(
            mode=DriftMode.FORWARD_TRACE,
            observation_id=request.observation.observation_id,
            release=request.release,
            release_point=self._release_point(request.release),
            release_time=release_time,
            engine_seed_time=release_time,
            target_time=request.observation.observed_at,
            forcing=request.forcing,
            random_seed=request.random_seed,
            request_model_parameters=request.model_parameters,
            request_uncertainty=request.configured_uncertainty,
        )

    def backward_source_trace(self, request: BackwardTraceRequest) -> DriftSimulation:
        release_time = self._release_time(request.release)
        seed = (
            self._release_point(request.release)
            if self._config.backward_seed_position == "release_geometry"
            else self._centroid(request.observation.geometry.polygon)
        )
        return self._run(
            mode=DriftMode.BACKWARD_TRACE,
            observation_id=request.observation.observation_id,
            release=request.release,
            release_point=self._release_point(request.release),
            release_time=release_time,
            engine_seed_point=seed,
            engine_seed_time=request.observation.observed_at,
            target_time=release_time,
            forcing=request.forcing,
            random_seed=request.random_seed,
        )

    def forecast(self, request: DriftForecastRequest) -> DriftSimulation:
        release_time = self._release_time(request.release)
        seed = request.initial_position or (
            self._release_point(request.release)
            if self._config.forecast_seed_position == "release_geometry"
            else self._centroid(request.observation.geometry.polygon)
        )
        return self._run(
            mode=DriftMode.FORECAST,
            observation_id=request.observation.observation_id,
            release=request.release,
            release_point=self._release_point(request.release),
            release_time=release_time,
            engine_seed_point=seed,
            engine_seed_time=request.initial_timestamp or request.observation.observed_at,
            target_time=request.valid_until,
            forcing=request.forcing,
            random_seed=request.random_seed,
            request_model_parameters=request.model_parameters,
            request_uncertainty=request.configured_uncertainty,
        )

    def component_metadata(self) -> ComponentMetadata:
        try:
            version = importlib.metadata.version("opendrift")
        except importlib.metadata.PackageNotFoundError:
            version = "dependency-unavailable"
        return ComponentMetadata(
            name=self._component_name,
            version=version,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="OpenDrift OpenOil",
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(
                    key="forcing_compatibility_validation",
                    value=self._config.forcing_compatibility_validation_note,
                ),
                MetadataEntry(
                    key="physical_parameter_validation",
                    value=self._config.parameter_validation_note,
                ),
            ),
        )

    def _run(
        self,
        *,
        mode: DriftMode,
        observation_id: str,
        release: ReleaseHypothesis,
        release_point: SpatialGeometry,
        release_time: datetime,
        target_time: datetime,
        forcing: tuple[EnvironmentalField, ...],
        random_seed: int,
        engine_seed_point: SpatialGeometry | None = None,
        engine_seed_time: datetime | None = None,
        request_model_parameters: tuple[MetadataEntry, ...] = (),
        request_uncertainty: tuple[MetadataEntry, ...] = (),
    ) -> DriftSimulation:
        openoil_class, reader_class = self._runtime()
        self._validate_forcing(forcing)
        seed_point = self._to_wgs84(engine_seed_point) if engine_seed_point else release_point
        seed_time = engine_seed_time or release_time
        duration_seconds = abs((target_time - seed_time).total_seconds())
        if duration_seconds <= 0:
            raise DriftExecutionError("OpenOil simulation duration must be positive")
        simulation_key = sha256(
            (
                self._config.model_dump_json() + observation_id + mode.value + str(random_seed)
            ).encode()
        ).hexdigest()[:16]
        output = (
            self._config.output_root / _safe(observation_id) / simulation_key / "trajectories.nc"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.nc")
        seed_geometry = seed_point.geometry
        if seed_geometry.type != "Point":
            raise DriftExecutionError("internal OpenOil seed conversion did not produce a point")
        longitude = seed_geometry.coordinate.x
        latitude = seed_geometry.coordinate.y
        try:
            # Apply the same request seed recorded below, before particle seeding or processes.
            model = openoil_class(seed=random_seed, loglevel=20)
            for field in forcing:
                model.add_reader(reader_class(str(_local_path(field.values))))
            for parameter in (
                *self._config.model_parameters,
                *self._config.configured_uncertainty,
                *request_model_parameters,
                *request_uncertainty,
            ):
                model.set_config(parameter.key, parameter.value)
            seed_arguments: dict[str, Any] = {
                "lon": longitude,
                "lat": latitude,
                "number": self._config.particle_count,
                "time": seed_time,
            }
            if self._config.oil_type is not None:
                seed_arguments["oiltype"] = self._config.oil_type
            model.seed_elements(**seed_arguments)
            signed_step = (
                -_seconds(self._config.time_step)
                if mode == DriftMode.BACKWARD_TRACE
                else _seconds(self._config.time_step)
            )
            model.run(
                duration=timedelta(seconds=duration_seconds),
                time_step=signed_step,
                time_step_output=_seconds(self._config.output_interval),
                outfile=str(temporary),
            )
            if not temporary.is_file():
                raise DriftExecutionError(
                    "OpenOil returned without producing a trajectory artifact"
                )
            try:
                with xr.open_dataset(temporary) as trajectory_dataset:
                    completed_particles = trajectory_dataset.sizes[
                        self._config.output_trajectory_dimension
                    ]
                    time_dimension = self._config.output_time_dimension
                    if (
                        time_dimension not in trajectory_dataset.coords
                        or self._utc_datetime(trajectory_dataset[time_dimension].values[-1])
                        != target_time
                    ):
                        raise DriftExecutionError(
                            "OpenOil final output time must equal requested target; "
                            "no implicit temporal interpolation"
                        )
                    comparison_geometry, target_particle_count = self._target_geometry(
                        trajectory_dataset
                    )
                    particle_trajectories = self._particle_trajectories(trajectory_dataset)
            except (OSError, KeyError) as error:
                raise DriftExecutionError(
                    "OpenOil output lacks the configured particle trajectory dimension"
                ) from error
            temporary.replace(output)
        except DriftExecutionError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise DriftExecutionError(f"OpenOil {mode.value} execution failed") from error
        interval = TimeRange(start=min(seed_time, target_time), end=max(seed_time, target_time))
        trajectory_artifact = _artifact(output)
        return DriftSimulation(
            simulation_id=f"openoil:{mode.value}:{observation_id}:{simulation_key}",
            mode=mode,
            release=release,
            release_coordinates=release_point,
            release_timestamp=release_time,
            target_timestamp=target_time,
            forcing_ids=tuple(field.field_id for field in forcing),
            forcing_artifacts=tuple(field.values for field in forcing),
            interval=interval,
            trajectories=trajectory_artifact,
            output_crs=WGS84,
            spatial_unit=SpatialUnit.DEGREE,
            model=self.component_metadata(),
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            random_seed=random_seed,
            configured_uncertainty=(
                *self._config.configured_uncertainty,
                *request_uncertainty,
            ),
            model_parameters=(*self._config.model_parameters, *request_model_parameters),
            target_distribution=ParticleDistribution(
                timestamp=target_time,
                positions=trajectory_artifact,
                particle_count=target_particle_count,
                comparison_geometry=comparison_geometry,
                attributes=(
                    MetadataEntry(
                        key="geometry_method",
                        value=f"particle_{self._config.particle_support_geometry}",
                    ),
                    MetadataEntry(
                        key="scientific_boundary_interpretation",
                        value=self._config.particle_support_geometry_validation_note,
                    ),
                ),
            ),
            particle_trajectories=particle_trajectories,
            status=SimulationStatus.SUCCEEDED,
            diagnostics=SimulationDiagnostics(
                particle_count_requested=self._config.particle_count,
                particle_count_completed=completed_particles,
                attributes=(
                    MetadataEntry(key="engine_seed_timestamp", value=seed_time.isoformat()),
                    MetadataEntry(key="random_seed_recorded", value=random_seed),
                    MetadataEntry(key="random_seed", value=random_seed),
                    MetadataEntry(key="random_seed_applied_to_engine", value=True),
                ),
            ),
        )

    @staticmethod
    def _runtime() -> tuple[Any, Any]:
        try:
            openoil = importlib.import_module("opendrift.models.openoil")
            reader = importlib.import_module("opendrift.readers.reader_netCDF_CF_generic")
            return openoil.OpenOil, reader.Reader
        except (ImportError, AttributeError) as error:
            raise DriftDependencyError(
                "OpenOil requires the optional 'opendrift' runtime; install it before selecting "
                "drift.name=openoil"
            ) from error

    def _validate_forcing(self, forcing: tuple[EnvironmentalField, ...]) -> None:
        available = {field.variable for field in forcing}
        missing = set(self._config.required_forcing).difference(available)
        if missing:
            raise DriftExecutionError(
                f"required environmental forcing is missing: {sorted(missing)}"
            )
        for field in forcing:
            path = _local_path(field.values)
            if not path.is_file():
                raise DriftExecutionError(f"environmental forcing artifact does not exist: {path}")

    def _release_time(self, release: ReleaseHypothesis) -> datetime:
        if self._config.release_time_policy == "start":
            return release.interval.start
        if self._config.release_time_policy == "end":
            return release.interval.end
        return release.interval.start + (release.interval.end - release.interval.start) / 2

    def _target_geometry(self, dataset: xr.Dataset) -> tuple[SpatialGeometry, int]:
        try:
            longitude = dataset[self._config.output_longitude_variable]
            latitude = dataset[self._config.output_latitude_variable]
        except KeyError as error:
            raise DriftExecutionError(
                "OpenOil output lacks the configured longitude/latitude variables"
            ) from error
        time_dimension = self._config.output_time_dimension
        if time_dimension in longitude.dims and time_dimension in latitude.dims:
            longitude = longitude.isel({time_dimension: -1})
            latitude = latitude.isel({time_dimension: -1})
        elif time_dimension in longitude.dims or time_dimension in latitude.dims:
            raise DriftExecutionError(
                "configured OpenOil time dimension is inconsistent across position variables"
            )
        longitudes = np.asarray(longitude.values, dtype=float).reshape(-1)
        latitudes = np.asarray(latitude.values, dtype=float).reshape(-1)
        if longitudes.shape != latitudes.shape:
            raise DriftExecutionError("OpenOil longitude/latitude outputs have different shapes")
        valid = np.isfinite(longitudes) & np.isfinite(latitudes)
        points = sorted(
            set(zip(longitudes[valid].tolist(), latitudes[valid].tolist(), strict=True))
        )
        if not points:
            raise DriftExecutionError("OpenOil output has no finite particles at target time")
        hull = MultiPoint(points).convex_hull
        domain_geometry: PointGeometry | LineStringGeometry | PolygonGeometry
        if isinstance(hull, Point):
            domain_geometry = PointGeometry(coordinate=Coordinate(x=float(hull.x), y=float(hull.y)))
        elif isinstance(hull, LineString):
            domain_geometry = LineStringGeometry(
                coordinates=tuple(Coordinate(x=float(x), y=float(y)) for x, y in hull.coords)
            )
        elif isinstance(hull, Polygon):
            domain_geometry = PolygonGeometry(
                exterior=tuple(Coordinate(x=float(x), y=float(y)) for x, y in hull.exterior.coords),
                holes=tuple(
                    tuple(Coordinate(x=float(x), y=float(y)) for x, y in ring.coords)
                    for ring in hull.interiors
                ),
            )
        else:  # pragma: no cover - a convex hull of finite points has only these three forms.
            raise DriftExecutionError("unsupported OpenOil target particle geometry")
        return SpatialGeometry(geometry=domain_geometry, crs=WGS84), int(valid.sum())

    def _particle_trajectories(self, dataset: xr.Dataset) -> tuple[ParticleTrajectory, ...]:
        longitude = dataset[self._config.output_longitude_variable]
        latitude = dataset[self._config.output_latitude_variable]
        trajectory_dimension = self._config.output_trajectory_dimension
        time_dimension = self._config.output_time_dimension
        expected_dimensions = {trajectory_dimension, time_dimension}
        if set(longitude.dims) != expected_dimensions or set(latitude.dims) != expected_dimensions:
            raise DriftExecutionError(
                "OpenOil position variables must use exactly the configured trajectory and "
                "time dimensions; DATASET_VALIDATION_REQUIRED"
            )
        if time_dimension not in dataset.coords:
            raise DriftExecutionError("OpenOil output lacks the configured time coordinate")
        longitudes = np.asarray(
            longitude.transpose(trajectory_dimension, time_dimension).values, dtype=float
        )
        latitudes = np.asarray(
            latitude.transpose(trajectory_dimension, time_dimension).values, dtype=float
        )
        if longitudes.shape != latitudes.shape:
            raise DriftExecutionError("OpenOil longitude/latitude outputs have different shapes")
        timestamps = tuple(self._utc_datetime(value) for value in dataset[time_dimension].values)
        trajectories: list[ParticleTrajectory] = []
        for particle_index in range(longitudes.shape[0]):
            positions = [
                ParticlePosition(
                    timestamp=timestamp,
                    position=SpatialGeometry(
                        geometry=PointGeometry(
                            coordinate=Coordinate(
                                x=float(longitudes[particle_index, time_index]),
                                y=float(latitudes[particle_index, time_index]),
                            )
                        ),
                        crs=WGS84,
                    ),
                )
                for time_index, timestamp in enumerate(timestamps)
                if np.isfinite(longitudes[particle_index, time_index])
                and np.isfinite(latitudes[particle_index, time_index])
            ]
            positions.sort(key=lambda item: item.timestamp)
            if len(positions) >= 2:
                trajectories.append(
                    ParticleTrajectory(
                        particle_id=f"openoil-particle-{particle_index}",
                        positions=tuple(positions),
                    )
                )
        if not trajectories:
            raise DriftExecutionError(
                "OpenOil output has no particle trajectory with at least two finite positions"
            )
        return tuple(trajectories)

    @staticmethod
    def _utc_datetime(value: Any) -> datetime:
        try:
            parsed = np.datetime64(value)
            if np.isnat(parsed):
                raise ValueError("NaT")
            text = np.datetime_as_string(parsed, unit="us", timezone="UTC")
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        except (TypeError, ValueError) as error:
            raise DriftExecutionError(
                "OpenOil time coordinates must decode to UTC-compatible datetime values"
            ) from error

    def _release_point(self, release: ReleaseHypothesis) -> SpatialGeometry:
        if release.geometry.geometry.type == "Point":
            return self._to_wgs84(release.geometry)
        if self._config.release_position_policy == "point_only":
            raise DriftExecutionError("configured OpenOil release requires a point geometry")
        return self._centroid(release.geometry)

    def _centroid(self, geometry: SpatialGeometry) -> SpatialGeometry:
        source = (
            f"{geometry.crs.authority}:{geometry.crs.code}"
            if geometry.crs.authority and geometry.crs.code
            else geometry.crs.wkt
        )
        if source is None:
            raise DriftExecutionError("release geometry CRS has no usable identifier")
        projected = shape(
            transform_geom(source, "EPSG:4326", mapping(self._shapely(geometry)), precision=-1)
        )
        centroid = projected.centroid
        return SpatialGeometry(
            geometry=PointGeometry(coordinate=Coordinate(x=centroid.x, y=centroid.y)), crs=WGS84
        )

    def _to_wgs84(self, geometry: SpatialGeometry) -> SpatialGeometry:
        return self._centroid(geometry)

    @staticmethod
    def _shapely(geometry: SpatialGeometry) -> Any:
        value = geometry.geometry
        if value.type == "Point":
            return Point(value.coordinate.x, value.coordinate.y)
        if value.type == "LineString":
            return LineString([(point.x, point.y) for point in value.coordinates])

        def polygon(item: Any) -> Polygon:
            return Polygon(
                [(point.x, point.y) for point in item.exterior],
                [[(point.x, point.y) for point in ring] for ring in item.holes],
            )

        if value.type == "Polygon":
            return polygon(value)
        return MultiPolygon([polygon(item) for item in value.polygons])


def create_openoil(config: ComponentConfig) -> OpenOilDriftModel:
    settings = OpenOilDriftConfig.model_validate(config.settings)
    return OpenOilDriftModel(settings, component_name=config.name)
