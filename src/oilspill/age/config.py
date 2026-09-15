"""Typed configuration for conservative discharge-time windows."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.domain.geospatial import Duration


class WindowAnchor(StrEnum):
    ACQUISITION_START = "acquisition_start"
    ACQUISITION_MIDPOINT = "acquisition_midpoint"
    ACQUISITION_END = "acquisition_end"
    DETECTION_OBSERVED_AT = "detection_observed_at"


class CentralEstimatePolicy(StrEnum):
    NONE = "none"
    INTERVAL_MIDPOINT = "interval_midpoint"


class ConfigurableWindowAgeEstimatorConfig(FrozenModel):
    """All unsourced heuristic choices for the initial window estimator.

    ``scientifically_validated`` is deliberately fixed to false. A future sourced estimator
    should use a separate implementation and configuration schema.
    """

    anchor: WindowAnchor
    earliest_before_anchor: Duration
    latest_before_anchor: Duration
    central_estimate_policy: CentralEstimatePolicy
    required_morphology_fields: tuple[Literal["area", "length", "width", "orientation"], ...]
    require_ais_history: bool
    assumptions: tuple[Identifier, ...]
    heuristic_validation_note: Identifier
    scientifically_validated: Literal[False] = False

    @model_validator(mode="after")
    def _ordered_offsets(self) -> ConfigurableWindowAgeEstimatorConfig:
        if _duration_seconds(self.earliest_before_anchor) < _duration_seconds(
            self.latest_before_anchor
        ):
            raise ValueError(
                "earliest_before_anchor must be at least as long as latest_before_anchor"
            )
        return self


def _duration_seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]
