"""Explicit, unvalidated uncertainty declarations for independent member sampling."""

import math
from pathlib import Path
from typing import Annotated, Literal

import rasterio
from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier
from oilspill.domain.geospatial import Duration, RasterGrid, SpatialUnit


class Fixed(FrozenModel):
    kind: Literal["fixed"]
    value: float = Field(allow_inf_nan=False)


class Uniform(FrozenModel):
    kind: Literal["uniform"]
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> "Uniform":
        if self.upper <= self.lower:
            raise ValueError("uniform upper must exceed lower")
        return self


Distribution = Annotated[Fixed | Uniform, Field(discriminator="kind")]


class NumericUncertainty(FrozenModel):
    source_id: Identifier
    representation: Literal["external_member_sampling", "engine_internal"]
    target: Literal["offset_x_metres", "offset_y_metres", "offset_seconds", "model_parameter"]
    parameter_key: str | None = None
    # All bindings representing this physical uncertainty, including internal counterparts.
    exclusive_parameter_keys: tuple[Identifier, ...]
    distribution: Distribution
    units: Identifier
    validation_note: Identifier

    @model_validator(mode="after")
    def valid_binding(self) -> "NumericUncertainty":
        if (self.target == "model_parameter") != (self.parameter_key is not None):
            raise ValueError("only model_parameter requires a parameter_key")
        if self.parameter_key and self.parameter_key not in self.exclusive_parameter_keys:
            raise ValueError("parameter_key must be included in exclusive_parameter_keys")
        if self.representation == "engine_internal" and (
            self.target != "model_parameter" or self.distribution.kind != "fixed"
        ):
            raise ValueError("engine_internal needs a fixed engine parameter, not outer sampling")
        expected = {
            "offset_x_metres": "metre",
            "offset_y_metres": "metre",
            "offset_seconds": "second",
        }
        if self.target in expected and self.units != expected[self.target]:
            raise ValueError("offset units do not match target")
        return self


class ForcingChoice(FrozenModel):
    member_id: Identifier
    field_ids: tuple[Identifier, ...]


class ForcingSelection(FrozenModel):
    source_id: Identifier
    representation: Literal["external_member_sampling"]
    kind: Literal["fixed", "uniform"]
    choices: tuple[ForcingChoice, ...] = Field(min_length=1)
    validation_note: Identifier

    @model_validator(mode="after")
    def valid_choices(self) -> "ForcingSelection":
        if self.kind == "fixed" and len(self.choices) != 1:
            raise ValueError("fixed forcing requires exactly one choice")
        if len({c.member_id for c in self.choices}) != len(self.choices):
            raise ValueError("forcing member identifiers must be unique")
        if any(len(set(c.field_ids)) != len(c.field_ids) for c in self.choices):
            raise ValueError("forcing field identifiers must be unique within a choice")
        return self


class BaselineUncertainty(FrozenModel):
    """Explicit inventory of uncertainty already enabled in the drift configuration."""

    source_id: Identifier
    representation: Literal["engine_internal"]
    parameter_keys: tuple[Identifier, ...]
    validation_note: Identifier


class MonteCarloConfig(FrozenModel):
    output_root: Path
    member_count: int = Field(gt=0, le=2**32)
    horizon: Duration
    uncertainties: tuple[NumericUncertainty, ...]
    baseline_uncertainties: tuple[BaselineUncertainty, ...]
    baseline_inventory_complete: Literal[True]
    forcing: ForcingSelection
    sampling_dependence: Literal["independent"]
    failure_policy: Literal["record_and_continue", "raise_after_recording"]
    denominator_policy: Literal["valid_members"]
    grid: RasterGrid
    validation_note: Identifier

    @model_validator(mode="after")
    def validate_choices(self) -> "MonteCarloConfig":
        if self.horizon.value <= 0:
            raise ValueError("horizon must be positive")
        ids = (
            [u.source_id for u in self.uncertainties]
            + [u.source_id for u in self.baseline_uncertainties]
            + [self.forcing.source_id]
        )
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate uncertainty source: outer/internal double counting")
        keys = [k for u in self.uncertainties for k in u.exclusive_parameter_keys] + [
            k for u in self.baseline_uncertainties for k in u.parameter_keys
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate uncertainty parameter binding")
        targets = [u.target for u in self.uncertainties if u.target != "model_parameter"]
        if len(targets) != len(set(targets)):
            raise ValueError("duplicate offset target")
        crs = rasterio.crs.CRS.from_user_input(
            self.grid.crs.wkt or f"{self.grid.crs.authority}:{self.grid.crs.code}"
        )
        if (
            not crs.is_projected
            or crs.linear_units not in {"metre", "meter"}
            or self.grid.crs.axis_units != (SpatialUnit.METRE, SpatialUnit.METRE)
        ):
            raise ValueError("occupancy grid must use a projected metre CRS")
        t = self.grid.transform
        if not all(math.isfinite(v) for v in (t.a, t.b, t.c, t.d, t.e, t.f)):
            raise ValueError("occupancy transform must be finite")
        if t.a <= 0 or t.e >= 0 or t.b != 0 or t.d != 0:
            raise ValueError("initial occupancy grid requires north-up positive resolution")
        return self
