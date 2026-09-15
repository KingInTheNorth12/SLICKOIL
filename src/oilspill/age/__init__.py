"""Conservative, replaceable spill-age estimation."""

from oilspill.age.config import ConfigurableWindowAgeEstimatorConfig
from oilspill.age.errors import SpillAgeEstimationError
from oilspill.age.window import ConfigurableWindowAgeEstimator, create_configurable_window

__all__ = [
    "ConfigurableWindowAgeEstimator",
    "ConfigurableWindowAgeEstimatorConfig",
    "SpillAgeEstimationError",
    "create_configurable_window",
]
