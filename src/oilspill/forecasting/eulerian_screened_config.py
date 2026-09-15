"""Configuration for Eulerian-screened hybrid forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.domain.geospatial import Duration
from oilspill.eulerian.grid import EulerianGridBuilderConfig
from oilspill.eulerian.initialization import EulerianInitializationConfig


class ForecastParameterRange(FrozenModel):
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> ForecastParameterRange:
        if self.maximum < self.minimum:
            raise ValueError("forecast parameter maximum must be >= minimum")
        return self


class EulerianScreenedForecastConfig(FrozenModel):
    output_root: Path
    grid: EulerianGridBuilderConfig
    initialization: EulerianInitializationConfig
    horizons: tuple[Duration, ...] = Field(min_length=1)
    initial_scenario_count: int = Field(gt=0)
    additional_scenarios_per_batch: int = Field(gt=0)
    maximum_batches: int = Field(gt=0)
    maximum_coarse_scenarios: int = Field(gt=0)
    maximum_high_fidelity_evaluations: int = Field(gt=0)
    partial_high_fidelity_failure_policy: Literal[
        "renormalize_successful", "abstain"
    ] = "abstain"
    cluster_count: int = Field(gt=0)
    displacement_extreme_count: int = Field(ge=0)
    spread_extreme_count: int = Field(ge=0)
    coastal_contact_extreme_count: int = Field(ge=0)
    wind_bias_u_m_s: ForecastParameterRange
    wind_bias_v_m_s: ForecastParameterRange
    current_bias_u_m_s: ForecastParameterRange
    current_bias_v_m_s: ForecastParameterRange
    windage: ForecastParameterRange
    diffusivity_m2_s: ForecastParameterRange
    observation_support_threshold: float = Field(gt=0.0, le=1.0)
    dominant_direction_anisotropy_threshold: float = Field(ge=1.0)
    raster_l1_tolerance: float = Field(ge=0.0)
    centroid_tolerance_metres: float = Field(ge=0.0)
    support_area_tolerance_m2: float = Field(ge=0.0)
    coastal_contact_weight_tolerance: float = Field(ge=0.0, le=1.0)
    consecutive_converged_batches: int = Field(gt=0)
    windage_parameter_key: Identifier
    diffusivity_parameter_key: Identifier
    wind_bias_u_parameter_key: Identifier
    wind_bias_v_parameter_key: Identifier
    current_bias_u_parameter_key: Identifier
    current_bias_v_parameter_key: Identifier
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def _validated(self) -> EulerianScreenedForecastConfig:
        seconds = tuple(_seconds(item) for item in self.horizons)
        if seconds != tuple(sorted(seconds)) or len(seconds) != len(set(seconds)):
            raise ValueError("hybrid forecast horizons must be unique and chronological")
        if any(value <= 0 for value in seconds):
            raise ValueError("hybrid forecast horizons must be positive")
        if self.initial_scenario_count > self.maximum_coarse_scenarios:
            raise ValueError("initial scenario count exceeds coarse compute budget")
        if self.cluster_count > self.maximum_coarse_scenarios:
            raise ValueError("cluster count exceeds coarse scenario budget")
        maximum_selected = (
            self.cluster_count
            + self.displacement_extreme_count
            + self.spread_extreme_count
            + self.coastal_contact_extreme_count
        )
        if self.maximum_high_fidelity_evaluations < maximum_selected * len(seconds):
            raise ValueError(
                "high-fidelity budget must cover configured representatives at every horizon"
            )
        if self.windage.minimum < 0 or self.diffusivity_m2_s.minimum < 0:
            raise ValueError("windage and diffusivity ranges cannot be negative")
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("hybrid forecast settings must mark DATASET_VALIDATION_REQUIRED")
        return self


def _seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]
