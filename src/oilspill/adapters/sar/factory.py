"""Registry factories for SAR preprocessing adapters."""

from oilspill.adapters.sar.config import (
    RasterioSARReaderConfig,
    SnapSARPreprocessorSettings,
)
from oilspill.adapters.sar.preprocessor import BackendSARPreprocessor
from oilspill.adapters.sar.reader import RasterioSARReader
from oilspill.adapters.sar.snap import SnapGPTBackend
from oilspill.adapters.sar.synthetic import SyntheticRasterCopyBackend
from oilspill.config import ComponentConfig


def create_rasterio_sar_reader(config: ComponentConfig) -> RasterioSARReader:
    return RasterioSARReader(RasterioSARReaderConfig.model_validate(config.settings))


def create_snap_preprocessor(config: ComponentConfig) -> BackendSARPreprocessor:
    settings = SnapSARPreprocessorSettings.model_validate(config.settings)
    return BackendSARPreprocessor(settings, SnapGPTBackend(settings.snap))


def create_synthetic_copy_preprocessor(config: ComponentConfig) -> BackendSARPreprocessor:
    settings = SnapSARPreprocessorSettings.model_validate(config.settings)
    return BackendSARPreprocessor(settings, SyntheticRasterCopyBackend())
