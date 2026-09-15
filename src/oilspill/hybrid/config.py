"""Validated configuration for physics-first coarse hindcast screening."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.eulerian.grid import EulerianGridBuilderConfig
from oilspill.eulerian.initialization import EulerianInitializationConfig


class ParameterRange(FrozenModel):
    minimum: float = Field(ge=0.0, allow_inf_nan=False)
    maximum: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> ParameterRange:
        if self.maximum < self.minimum:
            raise ValueError("parameter range maximum must not be below minimum")
        return self


class HybridMismatchWeights(FrozenModel):
    soft_iou: float = Field(ge=0.0, allow_inf_nan=False)
    centroid: float = Field(ge=0.0, allow_inf_nan=False)
    area: float = Field(ge=0.0, allow_inf_nan=False)
    shape: float = Field(ge=0.0, allow_inf_nan=False)
    physics_penalty: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _nonzero(self) -> HybridMismatchWeights:
        if not any(
            (self.soft_iou, self.centroid, self.area, self.shape, self.physics_penalty)
        ):
            raise ValueError("at least one mismatch weight must be positive")
        return self


class HybridHindcastConfig(FrozenModel):
    """No values here are asserted to be scientifically authoritative."""

    output_root: Path
    grid: EulerianGridBuilderConfig
    initialization: EulerianInitializationConfig
    backward_horizon_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    source_spatial_roi_policy: Literal["observation_backward_reachability"]
    source_roi_padding_metres: float = Field(ge=0.0, allow_inf_nan=False)
    diffusion_reach_sigma_multiplier: float = Field(gt=0.0, allow_inf_nan=False)
    initial_sobol_candidates: int = Field(gt=0)
    source_time_range_policy: Literal[
        "backward_horizon", "discharge_interval_intersection"
    ]
    windage_range: ParameterRange
    diffusivity_range_m2_s: ParameterRange
    loss_enabled: bool
    loss_rate_range_s: ParameterRange | None = None
    conservative_maximum_current_speed_m_s: float = Field(
        ge=0.0, allow_inf_nan=False
    )
    conservative_maximum_wind_speed_m_s: float = Field(ge=0.0, allow_inf_nan=False)
    conservative_maximum_windage: float = Field(ge=0.0, allow_inf_nan=False)
    conservative_maximum_diffusivity_m2_s: float = Field(
        ge=0.0, allow_inf_nan=False
    )
    top_k: int = Field(gt=0)
    absolute_score_threshold: float = Field(ge=0.0, allow_inf_nan=False)
    mismatch_weights: HybridMismatchWeights
    observation_support_threshold: float = Field(
        gt=0.0, le=1.0, allow_inf_nan=False
    )
    sobol_seed: int = Field(ge=0, lt=2**32)
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def _consistent(self) -> HybridHindcastConfig:
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("hybrid hindcast settings must mark DATASET_VALIDATION_REQUIRED")
        if self.loss_enabled != (self.loss_rate_range_s is not None):
            raise ValueError("loss-rate range must be present exactly when loss is enabled")
        if self.windage_range.maximum > self.conservative_maximum_windage:
            raise ValueError("sampled windage exceeds its conservative maximum")
        if (
            self.diffusivity_range_m2_s.maximum
            > self.conservative_maximum_diffusivity_m2_s
        ):
            raise ValueError("sampled diffusivity exceeds its conservative maximum")
        return self


class RefinementPerturbationScales(FrozenModel):
    x_metres: float = Field(gt=0.0, allow_inf_nan=False)
    y_metres: float = Field(gt=0.0, allow_inf_nan=False)
    release_time_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    windage_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    diffusivity_m2_s: float = Field(gt=0.0, allow_inf_nan=False)


class SelectiveRefinementConfig(FrozenModel):
    """High-fidelity settings requiring dataset-specific scientific validation."""

    output_root: Path
    initialization: EulerianInitializationConfig
    maximum_generations: int = Field(ge=0)
    maximum_high_fidelity_evaluations: int = Field(gt=0)
    children_per_generation: int = Field(gt=0)
    parent_pool_size: int = Field(gt=0)
    exploratory_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    maximum_child_sampling_attempts: int = Field(gt=0)
    perturbation_scales: RefinementPerturbationScales
    perturbation_shrink_rho: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    initial_temperature: float = Field(gt=0.0, allow_inf_nan=False)
    temperature_decay: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    windage_range: ParameterRange
    diffusivity_range_m2_s: ParameterRange
    child_bounds_policy: Literal["analysis_grid_and_discharge_interval"]
    minimum_forward_duration_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    best_score_improvement_tolerance: float = Field(ge=0.0, allow_inf_nan=False)
    source_centroid_movement_tolerance_metres: float = Field(
        ge=0.0, allow_inf_nan=False
    )
    release_time_quantile_change_tolerance_seconds: float = Field(
        ge=0.0, allow_inf_nan=False
    )
    mismatch_weights: HybridMismatchWeights
    observation_support_threshold: float = Field(
        gt=0.0, le=1.0, allow_inf_nan=False
    )
    posterior_output_nodata: float = Field(lt=0.0, allow_inf_nan=False)
    rasterization_all_touched: bool
    windage_parameter_key: Identifier
    diffusivity_parameter_key: Identifier
    loss_rate_parameter_key: Identifier | None = None
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def _validated(self) -> SelectiveRefinementConfig:
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("refinement settings must mark DATASET_VALIDATION_REQUIRED")
        return self
