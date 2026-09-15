"""SAR ingestion and backend-neutral preprocessing adapters."""

from oilspill.adapters.sar.config import (
    PolarizationBandMapping,
    RasterioSARReaderConfig,
    SARPreprocessingConfig,
    SARPreprocessingStage,
    SnapGPTConfig,
    SnapSARPreprocessorSettings,
)
from oilspill.adapters.sar.factory import create_snap_preprocessor
from oilspill.adapters.sar.preprocessor import BackendSARPreprocessor, SARPreprocessingBackend
from oilspill.adapters.sar.reader import RasterioSARReader
from oilspill.adapters.sar.snap import SnapGPTBackend

__all__ = [
    "BackendSARPreprocessor",
    "PolarizationBandMapping",
    "RasterioSARReader",
    "RasterioSARReaderConfig",
    "SARPreprocessingBackend",
    "SARPreprocessingConfig",
    "SARPreprocessingStage",
    "SnapGPTBackend",
    "SnapGPTConfig",
    "SnapSARPreprocessorSettings",
    "create_snap_preprocessor",
]
