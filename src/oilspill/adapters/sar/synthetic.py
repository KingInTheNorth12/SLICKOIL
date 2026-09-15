"""Synthetic-only SAR backend for deterministic pipeline tests and demos."""

from __future__ import annotations

import shutil
from pathlib import Path

from oilspill.adapters.sar.config import SARPreprocessingConfig
from oilspill.adapters.sar.errors import SARBackendExecutionError
from oilspill.adapters.sar.preprocessor import BackendExecutionResult
from oilspill.domain.common import ComponentMetadata, MetadataEntry


class SyntheticRasterCopyBackend:
    """Copy a fixture raster while explicitly refusing scientific preprocessing claims."""

    def execute(
        self,
        source: Path,
        destination: Path,
        config: SARPreprocessingConfig,
    ) -> BackendExecutionResult:
        enabled = tuple(name for name, stage in config.stages() if stage.enabled)
        if enabled:
            raise SARBackendExecutionError(
                f"synthetic copy backend cannot execute scientific preprocessing stages: {enabled}"
            )
        shutil.copyfile(source, destination)
        return BackendExecutionResult(
            output_path=destination,
            executed_stages=("synthetic_copy_only",),
        )

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="synthetic_raster_copy",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="test-only",
            attributes=(
                MetadataEntry(key="synthetic", value=True),
                MetadataEntry(key="scientifically_representative", value=False),
                MetadataEntry(key="scientific_preprocessing_performed", value=False),
            ),
        )
