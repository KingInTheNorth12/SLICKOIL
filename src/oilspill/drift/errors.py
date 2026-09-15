"""Drift adapter dependency and runtime errors."""

from oilspill.errors import OptionalDependencyError


class DriftDependencyError(OptionalDependencyError):
    """The configured drift runtime is unavailable."""


class DriftExecutionError(RuntimeError):
    """The drift runtime failed before producing a validated output artifact."""
