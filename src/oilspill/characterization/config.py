"""Configuration for raster-mask slick characterization."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from oilspill.domain.common import FrozenModel, Identifier


class SpillCharacterizationConfig(FrozenModel):
    """Explicit geometric decisions; contains no dataset-derived hidden thresholds."""

    output_root: Path
    foreground_values: Annotated[tuple[float, ...], Field(min_length=1)]
    minimum_component_pixels: int = Field(gt=0)
    connectivity: Literal[4, 8]
    component_policy: Literal["retain_all", "largest"]
    invalid_polygon_policy: Literal["repair", "reject"]
    measurement_crs: Identifier
    create_skeleton: bool
    create_centerline: bool
    threshold_validation_note: Identifier
    measurement_crs_validation_note: Identifier
