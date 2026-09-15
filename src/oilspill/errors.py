"""Application-facing error categories independent of concrete adapters."""


class OptionalDependencyError(RuntimeError):
    """A selected optional implementation cannot run because its dependency is absent."""


class ConfiguredArtifactError(RuntimeError):
    """A required configured artifact is unavailable or invalid."""


class MissingConfiguredArtifactError(ConfiguredArtifactError):
    """A required, explicitly configured artifact does not exist."""
