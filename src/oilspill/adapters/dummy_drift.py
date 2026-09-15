"""Deterministic drift adapters used for registry and orchestration tests only."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256

from pydantic import Field
from rasterio.warp import transform

from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    MetadataEntry,
    TimeRange,
)
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
from oilspill.domain.geospatial import Coordinate, PointGeometry, SpatialGeometry
from oilspill.requests import BackwardTraceRequest, DriftForecastRequest, ForwardTraceRequest


class DummyDriftSettings(FrozenModel):
    output_uri: str = Field(min_length=1)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    particle_count: int = Field(default=0, ge=0)
    uncertainty: tuple[MetadataEntry, ...] = ()
    model_parameters: tuple[MetadataEntry, ...] = ()


class FakeDriftModel:
    """Deterministic engine-neutral test double; performs no physical simulation."""

    component_name = "dummy_drift"

    def __init__(self, config: ComponentConfig) -> None:
        self._config = config
        self.component_name = config.name
        self._settings = DummyDriftSettings.model_validate(config.settings)

    def forward_source_trace(self, request: ForwardTraceRequest) -> DriftSimulation:
        return self._result(
            mode=DriftMode.FORWARD_TRACE,
            observation_id=request.observation.observation_id,
            release=request.release,
            forcing_ids=tuple(field.field_id for field in request.forcing),
            forcing_artifacts=tuple(field.values for field in request.forcing),
            interval=TimeRange(
                start=request.release.interval.start,
                end=request.observation.observed_at,
            ),
            target_timestamp=request.observation.observed_at,
            observed_position=request.observation.geometry.centroid,
            random_seed=request.random_seed,
            model_parameters=request.model_parameters,
            configured_uncertainty=request.configured_uncertainty,
        )

    def backward_source_trace(self, request: BackwardTraceRequest) -> DriftSimulation:
        return self._result(
            mode=DriftMode.BACKWARD_TRACE,
            observation_id=request.observation.observation_id,
            release=request.release,
            forcing_ids=tuple(field.field_id for field in request.forcing),
            forcing_artifacts=tuple(field.values for field in request.forcing),
            interval=TimeRange(
                start=request.release.interval.start,
                end=request.observation.observed_at,
            ),
            target_timestamp=request.release.interval.start,
            observed_position=request.observation.geometry.centroid,
            random_seed=request.random_seed,
        )

    def forecast(self, request: DriftForecastRequest) -> DriftSimulation:
        initial_timestamp = request.initial_timestamp or request.observation.observed_at
        return self._result(
            mode=DriftMode.FORECAST,
            observation_id=request.observation.observation_id,
            release=request.release,
            forcing_ids=tuple(field.field_id for field in request.forcing),
            forcing_artifacts=tuple(field.values for field in request.forcing),
            interval=TimeRange(start=initial_timestamp, end=request.valid_until),
            target_timestamp=request.valid_until,
            observed_position=request.initial_position or request.observation.geometry.centroid,
            random_seed=request.random_seed,
            model_parameters=request.model_parameters,
            configured_uncertainty=request.configured_uncertainty,
        )

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name=self.component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="deterministic-test-double",
            configuration_sha256=digest,
        )

    def _result(
        self,
        *,
        mode: DriftMode,
        observation_id: str,
        release: ReleaseHypothesis,
        forcing_ids: tuple[str, ...],
        forcing_artifacts: tuple[ArtifactRef, ...],
        interval: TimeRange,
        target_timestamp: datetime,
        observed_position: SpatialGeometry,
        random_seed: int,
        model_parameters: tuple[MetadataEntry, ...] = (),
        configured_uncertainty: tuple[MetadataEntry, ...] = (),
    ) -> DriftSimulation:
        metadata = self.component_metadata()
        trajectories = ArtifactRef(
            uri=self._settings.output_uri,
            media_type="application/x-parquet",
            sha256=self._settings.output_sha256,
            byte_size=0,
        )
        release_point = release.geometry if release.geometry.geometry.type == "Point" else None
        if release_point is not None and release_point.crs != observed_position.crs:
            observed_position = _transform_point(observed_position, release_point)
        target_geometry = release_point if mode == DriftMode.BACKWARD_TRACE else observed_position
        target_distribution = (
            ParticleDistribution(
                timestamp=target_timestamp,
                positions=trajectories,
                particle_count=self._settings.particle_count,
                comparison_geometry=target_geometry,
                attributes=(
                    MetadataEntry(key="synthetic", value=True),
                    MetadataEntry(key="scientifically_representative", value=False),
                ),
            )
            if release_point is not None
            else None
        )
        particle_trajectories = (
            tuple(
                ParticleTrajectory(
                    particle_id=f"synthetic-particle-{index}",
                    positions=(
                        ParticlePosition(
                            timestamp=release.interval.start,
                            position=release_point,
                        ),
                        ParticlePosition(timestamp=interval.end, position=observed_position),
                    ),
                )
                for index in range(self._settings.particle_count)
            )
            if release_point is not None and mode != DriftMode.FORECAST
            else ()
        )
        simulation_key = sha256(
            (
                f"{observation_id}:{mode.value}:{target_timestamp.isoformat()}:"
                f"{random_seed}:{observed_position.model_dump_json()}:"
                f"{model_parameters}:{configured_uncertainty}"
            ).encode()
        ).hexdigest()[:16]
        return DriftSimulation(
            simulation_id=(f"{self.component_name}:{mode.value}:{observation_id}:{simulation_key}"),
            mode=mode,
            release=release,
            release_coordinates=release_point,
            release_timestamp=release.interval.start,
            target_timestamp=target_timestamp,
            forcing_ids=forcing_ids,
            forcing_artifacts=forcing_artifacts,
            interval=interval,
            trajectories=trajectories,
            output_crs=release.geometry.crs,
            spatial_unit=release.geometry.crs.axis_units[0],
            model=metadata,
            configuration_sha256=metadata.configuration_sha256 or "0" * 64,
            random_seed=random_seed,
            configured_uncertainty=(*self._settings.uncertainty, *configured_uncertainty),
            model_parameters=(*self._settings.model_parameters, *model_parameters),
            target_distribution=target_distribution,
            particle_trajectories=particle_trajectories,
            status=SimulationStatus.SUCCEEDED,
            diagnostics=SimulationDiagnostics(
                particle_count_requested=self._settings.particle_count,
                particle_count_completed=self._settings.particle_count,
                attributes=(MetadataEntry(key="synthetic", value=True),),
            ),
        )


class DummyDriftA(FakeDriftModel):
    component_name = "dummy_drift_a"


class DummyDriftB(FakeDriftModel):
    component_name = "dummy_drift_b"


def create_dummy_drift_a(config: ComponentConfig) -> DummyDriftA:
    return DummyDriftA(config)


def create_dummy_drift_b(config: ComponentConfig) -> DummyDriftB:
    return DummyDriftB(config)


def create_fake_drift(config: ComponentConfig) -> FakeDriftModel:
    return FakeDriftModel(config)


def _transform_point(
    value: SpatialGeometry, target: SpatialGeometry
) -> SpatialGeometry:
    point = value.geometry
    if point.type != "Point":
        raise ValueError("FakeDriftModel observation position must be a Point")

    def crs_text(geometry: SpatialGeometry) -> str:
        crs = geometry.crs
        if crs.authority and crs.code:
            return f"{crs.authority}:{crs.code}"
        if crs.wkt:
            return crs.wkt
        raise ValueError("FakeDriftModel geometry CRS lacks an identifier")

    xs, ys = transform(
        crs_text(value),
        crs_text(target),
        [point.coordinate.x],
        [point.coordinate.y],
    )
    return SpatialGeometry(
        geometry=PointGeometry(coordinate=Coordinate(x=xs[0], y=ys[0])),
        crs=target.crs,
    )
