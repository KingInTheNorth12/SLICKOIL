"""Errors raised by the NumPy finite-volume Eulerian adapter."""


class EulerianTransportError(RuntimeError):
    """Base error for invalid inputs, artifacts, or numerical execution."""


class EulerianStabilityError(EulerianTransportError):
    """Raised when a stable positive time step cannot be constructed."""


class EulerianForcingError(EulerianTransportError):
    """Raised when forcing cannot be interpreted on the Eulerian grid."""
