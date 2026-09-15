"""Errors raised at the segmentation adapter boundary."""

from oilspill.errors import (
    ConfiguredArtifactError,
    MissingConfiguredArtifactError,
    OptionalDependencyError,
)


class DetectorError(RuntimeError):
    """Base segmentation adapter error."""


class DetectorDependencyError(DetectorError, OptionalDependencyError):
    """A selected optional model framework is unavailable."""


class DetectorArtifactError(DetectorError, ConfiguredArtifactError):
    """A checkpoint or generated detection artifact is unavailable or invalid."""


class MissingModelCheckpointError(DetectorArtifactError, MissingConfiguredArtifactError):
    """A configured local model checkpoint does not exist.

    This is distinct from a corrupt or checksum-mismatched artifact so user-facing
    composition layers can provide a precise, actionable diagnostic.
    """


class DetectorInputError(DetectorError):
    """A SAR scene cannot be interpreted using the explicit detector configuration."""


class DetectorOutputError(DetectorError):
    """A model returned data that cannot be converted to the stable domain contract."""
