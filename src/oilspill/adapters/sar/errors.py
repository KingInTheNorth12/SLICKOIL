"""Explicit SAR adapter failure types."""

from oilspill.errors import OptionalDependencyError


class SARError(RuntimeError):
    pass


class SARDependencyError(SARError, OptionalDependencyError):
    """Required heavyweight backend executable or library is unavailable."""


class SARInputError(SARError):
    """Input cannot be interpreted without unsupported assumptions."""


class SARBackendExecutionError(SARError):
    """The selected preprocessing backend failed or did not produce valid output."""


class DatasetValidationRequiredError(SARInputError):
    """Real-data metadata or conventions must be validated before ingestion."""
