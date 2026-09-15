"""Generic drift adapters and OpenOil implementation."""

from oilspill.adapters.dummy_drift import FakeDriftModel
from oilspill.drift.config import OpenOilDriftConfig
from oilspill.drift.errors import DriftDependencyError, DriftExecutionError
from oilspill.drift.openoil import OpenOilDriftModel, create_openoil
from oilspill.ports import DriftModel

__all__ = [
    "DriftModel",
    "DriftDependencyError",
    "DriftExecutionError",
    "FakeDriftModel",
    "OpenOilDriftConfig",
    "OpenOilDriftModel",
    "create_openoil",
]
