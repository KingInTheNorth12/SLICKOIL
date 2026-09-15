"""Environmental provider boundary errors."""


class EnvironmentalDataError(RuntimeError):
    """Environmental data cannot be mapped without violating configured assumptions."""
