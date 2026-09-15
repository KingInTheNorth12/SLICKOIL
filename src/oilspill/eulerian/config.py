"""Validated settings for the NumPy finite-volume Eulerian engine."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier


class NumpyFVMConfig(FrozenModel):
    output_root: Path
    maximum_time_step_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_advective_cfl: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_diffusive_cfl: float = Field(gt=0.0, le=0.5, allow_inf_nan=False)
    ocean_edge_boundary: Literal["closed", "advective_outflow"]
    point_initialization: Literal["containing_cell", "gaussian"]
    gaussian_sigma_metres: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    loss_integration: Literal["exponential", "explicit"]
    forcing_spatial_resampling: Literal["bilinear", "nearest"]
    allow_missing_wind: bool
    approximate_east_north_as_projected_xy: bool
    output_nodata: float = -9999.0
    validation_note: Identifier

    @model_validator(mode="after")
    def _consistent(self) -> NumpyFVMConfig:
        if self.point_initialization == "gaussian" and self.gaussian_sigma_metres is None:
            raise ValueError("Gaussian point initialization requires gaussian_sigma_metres")
        if (
            self.point_initialization == "containing_cell"
            and self.gaussian_sigma_metres is not None
        ):
            raise ValueError("gaussian_sigma_metres is only valid for Gaussian initialization")
        if not math.isfinite(self.output_nodata) or self.output_nodata >= 0.0:
            raise ValueError("Eulerian output nodata must be a finite negative value")
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("Eulerian numerical settings must mark DATASET_VALIDATION_REQUIRED")
        return self
