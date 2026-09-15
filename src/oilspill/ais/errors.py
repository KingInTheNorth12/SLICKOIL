"""AIS provider and track-reconstruction errors."""


class AISIngestionError(RuntimeError):
    """A source record cannot be mapped into a canonical AIS point."""


class VesselTrackBuildError(RuntimeError):
    """Canonical AIS points cannot form a track under the configured policy."""
