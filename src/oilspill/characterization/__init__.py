"""Independent raster-mask slick characterization."""

from oilspill.characterization.config import SpillCharacterizationConfig
from oilspill.characterization.errors import (
    EmptySpillMaskError,
    InvalidSpillGeometryError,
    SpillCharacterizationError,
)
from oilspill.characterization.raster import RasterSpillCharacterizer

__all__ = [
    "EmptySpillMaskError",
    "InvalidSpillGeometryError",
    "RasterSpillCharacterizer",
    "SpillCharacterizationConfig",
    "SpillCharacterizationError",
]
