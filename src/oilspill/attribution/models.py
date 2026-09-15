"""Typed configuration and result schemas for independent evidence metrics."""

from __future__ import annotations

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier


class MinMaxNormalization(FrozenModel):
    """Explicit affine normalization to [0, 1], optionally reversed and clamped.

    The base formula is ``(value - lower_bound) / (upper_bound - lower_bound)``. ``invert``
    changes it to ``1 - base``. Bounds are configuration, never hidden scientific thresholds.
    """

    lower_bound: float
    upper_bound: float
    invert: bool = False
    clamp: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> MinMaxNormalization:
        if self.upper_bound <= self.lower_bound:
            raise ValueError("normalization upper_bound must exceed lower_bound")
        return self

    def apply(self, value: float) -> float:
        normalized = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        if self.invert:
            normalized = 1.0 - normalized
        if self.clamp:
            normalized = min(1.0, max(0.0, normalized))
        return normalized


class DistanceMetricConfig(FrozenModel):
    """Metric CRS and optional explicit normalization for physical distances."""

    measurement_crs: Identifier
    normalization: MinMaxNormalization | None = None
    validation_note: Identifier


class GeometryMetricConfig(FrozenModel):
    """Projected CRS selection for dimensionless geometry comparisons."""

    measurement_crs: Identifier
    validation_note: Identifier


class TemporalMetricConfig(FrozenModel):
    normalization: MinMaxNormalization | None = None
    validation_note: Identifier


class DirectionMetricConfig(FrozenModel):
    """Angular equivalence period and optional normalization.

    A period of 360 treats direction as oriented; 180 treats an unoriented axis as equivalent
    after reversal. Choosing the period is domain configuration.
    """

    period_degrees: float = Field(gt=0.0, le=360.0)
    normalization: MinMaxNormalization | None = None
    validation_note: Identifier
