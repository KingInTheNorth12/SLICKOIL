"""Characterization boundary errors."""


class SpillCharacterizationError(RuntimeError):
    """Base characterization failure."""


class EmptySpillMaskError(SpillCharacterizationError):
    """No foreground remains after configured component filtering."""


class InvalidSpillGeometryError(SpillCharacterizationError):
    """Polygonization or metric geometry validation failed."""
