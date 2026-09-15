"""Engine-neutral contracts for coarse Eulerian surface transport."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    QualityFlag,
    UTCDateTime,
)
from oilspill.domain.geospatial import (
    CRS,
    AffineTransform,
    RasterAsset,
    SpatialGeometry,
    SpatialUnit,
)


class EulerianGrid(FrozenModel):
    """Axis-aligned, fixed regular grid in a projected metre CRS."""

    crs: CRS
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    transform: AffineTransform
    dx_metres: float = Field(gt=0.0, allow_inf_nan=False)
    dy_metres: float = Field(gt=0.0, allow_inf_nan=False)
    land_mask: ArtifactRef | None = None

    @model_validator(mode="after")
    def _projected_regular_metres(self) -> EulerianGrid:
        if self.crs.is_geographic or self.crs.axis_units != (
            SpatialUnit.METRE,
            SpatialUnit.METRE,
        ):
            raise ValueError("Eulerian grid CRS must be projected with metre axis units")
        if self.transform.b != 0.0 or self.transform.d != 0.0:
            raise ValueError("Eulerian MVP grid must be axis-aligned")
        if not math.isclose(abs(self.transform.a), self.dx_metres) or not math.isclose(
            abs(self.transform.e), self.dy_metres
        ):
            raise ValueError("Eulerian grid spacing must match its affine transform")
        return self


class EulerianScenario(FrozenModel):
    """One explicitly configured transport-physics and forcing scenario."""

    scenario_id: Identifier
    windage_coefficient: float = Field(ge=0.0, allow_inf_nan=False)
    horizontal_diffusivity_m2_s: float = Field(ge=0.0, allow_inf_nan=False)
    wind_bias_u_m_s: float = Field(default=0.0, allow_inf_nan=False)
    wind_bias_v_m_s: float = Field(default=0.0, allow_inf_nan=False)
    current_bias_u_m_s: float = Field(default=0.0, allow_inf_nan=False)
    current_bias_v_m_s: float = Field(default=0.0, allow_inf_nan=False)
    first_order_loss_rate_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    forcing_field_ids: tuple[Identifier, ...] = ()
    forcing_member_id: Identifier | None = None
    weight: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    weight_kind: Literal["probability", "design"] | None = None
    metadata: tuple[MetadataEntry, ...] = ()
    validation_assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_weight(self) -> EulerianScenario:
        if (self.weight is None) != (self.weight_kind is None):
            raise ValueError("Eulerian scenario weight and weight_kind must be provided together")
        if self.weight_kind == "probability" and self.weight is not None and self.weight > 1.0:
            raise ValueError("Eulerian scenario probability weight cannot exceed one")
        if len(set(self.forcing_field_ids)) != len(self.forcing_field_ids):
            raise ValueError("Eulerian scenario forcing field IDs must be unique")
        if not any(
            "DATASET_VALIDATION_REQUIRED" in assumption
            for assumption in self.validation_assumptions
        ):
            raise ValueError("Eulerian scenario assumptions must mark DATASET_VALIDATION_REQUIRED")
        return self


class PointSourceInitialCondition(FrozenModel):
    """Point/source release used for candidate-independent hindcast screening."""

    kind: Literal["point_source"] = "point_source"
    geometry: SpatialGeometry
    release_time: UTCDateTime

    @model_validator(mode="after")
    def _point(self) -> PointSourceInitialCondition:
        if self.geometry.geometry.type != "Point":
            raise ValueError("point-source initial condition requires Point geometry")
        return self


class ObservedSlickInitialCondition(FrozenModel):
    """Observed concentration/support field used to initialize a forecast."""

    kind: Literal["observed_slick_field"] = "observed_slick_field"
    observation_id: Identifier
    observed_at: UTCDateTime
    concentration: RasterAsset
    source_raster: ArtifactRef | None = None
    source_kind: Literal["probability_raster", "binary_mask"] | None = None
    assumptions: tuple[str, ...] = Field(min_length=1)


class EulerianInputProvenance(FrozenModel):
    forcing_field_ids: tuple[Identifier, ...] = ()
    forcing_artifacts: tuple[ArtifactRef, ...] = ()
    initial_condition_source_artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def _forcing_alignment(self) -> EulerianInputProvenance:
        if len(self.forcing_field_ids) != len(self.forcing_artifacts):
            raise ValueError("forcing IDs and artifacts must have matching provenance entries")
        return self


class EulerianSnapshot(FrozenModel):
    timestamp: UTCDateTime
    concentration: RasterAsset


class EulerianDiagnostics(FrozenModel):
    initial_mass: float = Field(ge=0.0, allow_inf_nan=False)
    final_mass: float = Field(ge=0.0, allow_inf_nan=False)
    min_concentration: float = Field(allow_inf_nan=False)
    max_concentration: float = Field(allow_inf_nan=False)
    maximum_advective_cfl: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_diffusive_cfl: float = Field(ge=0.0, allow_inf_nan=False)
    number_of_time_steps: int = Field(ge=0)
    clipped_negative_cell_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered_concentration(self) -> EulerianDiagnostics:
        if self.max_concentration < self.min_concentration:
            raise ValueError("maximum concentration cannot be below minimum concentration")
        return self


class EulerianTransportResult(FrozenModel):
    """Coarse transport output; concentration values are not calibrated probabilities."""

    result_id: Identifier
    scenario_id: Identifier
    start_time: UTCDateTime
    end_time: UTCDateTime
    snapshots: tuple[EulerianSnapshot, ...] = Field(min_length=1)
    diagnostics: EulerianDiagnostics
    component: ComponentMetadata
    input_provenance: EulerianInputProvenance = EulerianInputProvenance()
    warnings: tuple[QualityFlag, ...] = ()
    semantics: Literal["surface concentration support; not a calibrated probability"] = (
        "surface concentration support; not a calibrated probability"
    )

    @model_validator(mode="after")
    def _consistent(self) -> EulerianTransportResult:
        if self.end_time <= self.start_time:
            raise ValueError("Eulerian transport end time must follow start time")
        timestamps = tuple(snapshot.timestamp for snapshot in self.snapshots)
        if timestamps != tuple(sorted(timestamps)) or len(set(timestamps)) != len(timestamps):
            raise ValueError("Eulerian result snapshots must be unique and chronological")
        if timestamps[-1] != self.end_time or any(
            timestamp <= self.start_time or timestamp > self.end_time for timestamp in timestamps
        ):
            raise ValueError("Eulerian snapshots must lie after start and end at end_time")
        grid = self.snapshots[0].concentration.grid
        if any(snapshot.concentration.grid != grid for snapshot in self.snapshots):
            raise ValueError("Eulerian result snapshots must share one raster grid")
        return self
