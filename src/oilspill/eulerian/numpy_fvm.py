"""File-backed Eulerian adapter composed around the pure NumPy kernel."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import rasterio
from numpy.typing import NDArray

from oilspill.artifacts import local_path_from_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.common import ComponentMetadata, MetadataEntry
from oilspill.domain.eulerian import (
    EulerianDiagnostics,
    EulerianInputProvenance,
    EulerianSnapshot,
    EulerianTransportResult,
)
from oilspill.eulerian.config import NumpyFVMConfig
from oilspill.eulerian.errors import EulerianTransportError
from oilspill.eulerian.forcing import EnvironmentalForcingAdapter, EulerianForcingConfig
from oilspill.eulerian.initialization import (
    normalize,
    normalized_point_field,
    read_land_mask,
    validate_raster,
    write_raster,
)
from oilspill.eulerian.kernel import (
    KernelResult,
    finite_volume_step,
    integrate_transport,
    stable_time_step,
)
from oilspill.requests import EulerianTransportRequest

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "KernelResult",
    "NumpyFVMTransportEngine",
    "finite_volume_step",
    "integrate_transport",
    "stable_time_step",
]


class NumpyFVMTransportEngine:
    def __init__(self, config: NumpyFVMConfig) -> None:
        self._config = config

    def run(self, request: EulerianTransportRequest) -> EulerianTransportResult:
        start = (
            request.initial_condition.release_time
            if request.initial_condition.kind == "point_source"
            else request.initial_condition.observed_at
        )
        land = read_land_mask(request.grid)
        initial, initial_clipped = self._initial(request, land)
        forcing = EnvironmentalForcingAdapter(
            request.forcing,
            request.grid,
            start,
            request.snapshot_times[-1],
            request.scenario.windage_coefficient,
            EulerianForcingConfig(
                spatial_resampling=self._config.forcing_spatial_resampling,
                allow_missing_wind=self._config.allow_missing_wind,
                approximate_east_north_as_projected_xy=(
                    self._config.approximate_east_north_as_projected_xy
                ),
                validation_note=self._config.validation_note,
            ),
            wind_bias_u_m_s=request.scenario.wind_bias_u_m_s,
            wind_bias_v_m_s=request.scenario.wind_bias_v_m_s,
            current_bias_u_m_s=request.scenario.current_bias_u_m_s,
            current_bias_v_m_s=request.scenario.current_bias_v_m_s,
        )
        kernel = integrate_transport(
            initial,
            land,
            forcing.velocity_at_seconds,
            tuple((time - start).total_seconds() for time in request.snapshot_times),
            dx_metres=request.grid.dx_metres,
            dy_metres=request.grid.dy_metres,
            diffusivity_m2_s=request.scenario.horizontal_diffusivity_m2_s,
            loss_rate_s=request.scenario.first_order_loss_rate_s,
            maximum_dt_seconds=self._config.maximum_time_step_seconds,
            maximum_advective_cfl=self._config.maximum_advective_cfl,
            maximum_diffusive_cfl=self._config.maximum_diffusive_cfl,
            boundary=self._config.ocean_edge_boundary,
            loss_integration=self._config.loss_integration,
        )
        identity = sha256(
            (request.model_dump_json() + self._config.model_dump_json()).encode()
        ).hexdigest()
        directory = self._config.output_root / identity[:16]
        assets = tuple(
            write_raster(
                directory / f"concentration-{index:04d}.tif",
                values,
                request.grid,
                land,
                self._config.output_nodata,
            )
            for index, values in enumerate(kernel.snapshots)
        )
        source_artifact = (
            request.initial_condition.source_raster
            if request.initial_condition.kind == "observed_slick_field"
            else None
        )
        return EulerianTransportResult(
            result_id=f"numpy-fvm:{identity[:24]}",
            scenario_id=request.scenario.scenario_id,
            start_time=start,
            end_time=request.snapshot_times[-1],
            snapshots=tuple(
                EulerianSnapshot(timestamp=time, concentration=asset)
                for time, asset in zip(request.snapshot_times, assets, strict=True)
            ),
            diagnostics=EulerianDiagnostics(
                initial_mass=kernel.initial_mass,
                final_mass=kernel.final_mass,
                min_concentration=kernel.minimum_concentration,
                max_concentration=kernel.maximum_concentration,
                maximum_advective_cfl=kernel.maximum_advective_cfl,
                maximum_diffusive_cfl=kernel.maximum_diffusive_cfl,
                number_of_time_steps=kernel.number_of_time_steps,
                clipped_negative_cell_count=initial_clipped + kernel.clipped_negative_cell_count,
            ),
            component=self.component_metadata(),
            input_provenance=EulerianInputProvenance(
                forcing_field_ids=tuple(field.field_id for field in request.forcing),
                forcing_artifacts=tuple(field.values for field in request.forcing),
                initial_condition_source_artifact=source_artifact,
            ),
            warnings=forcing.warnings,
        )

    def _initial(
        self, request: EulerianTransportRequest, land: BoolArray
    ) -> tuple[FloatArray, int]:
        initial = request.initial_condition
        if initial.kind == "point_source":
            return (
                normalized_point_field(
                    initial,
                    request.grid,
                    land,
                    method=self._config.point_initialization,
                    gaussian_sigma_metres=self._config.gaussian_sigma_metres,
                ),
                0,
            )
        path = local_path_from_artifact(initial.concentration.artifact)
        if not path.is_file():
            raise EulerianTransportError("initial concentration raster is missing")
        with rasterio.open(path) as dataset:
            validate_raster(dataset, request.grid, "initial concentration")
            values = np.asarray(dataset.read(1), dtype=np.float64)
            if dataset.nodata is not None:
                values[values == dataset.nodata] = 0
        if not np.isfinite(values).all():
            raise EulerianTransportError("initial concentration contains non-finite values")
        clipped = int(np.count_nonzero(values < 0))
        return normalize(values, land, request.grid), clipped

    def component_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name="numpy_fvm",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="numpy-rasterio-xarray",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="transported_quantity", value="relative_surface_mass"),
                MetadataEntry(key="advection", value="first_order_upwind_finite_volume"),
                MetadataEntry(key="diffusion", value="centered_explicit_isotropic"),
                MetadataEntry(key="stokes_drift", value="omitted_in_mvp"),
                MetadataEntry(key="vertical_processes", value="omitted_in_mvp"),
                MetadataEntry(key="oil_weathering", value="omitted_in_mvp"),
                MetadataEntry(key="validation_note", value=self._config.validation_note),
            ),
        )


def create_numpy_fvm(config: ComponentConfig) -> NumpyFVMTransportEngine:
    return NumpyFVMTransportEngine(NumpyFVMConfig.model_validate(config.settings))
