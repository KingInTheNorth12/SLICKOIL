"""Typed configuration for SAR ingestion and preprocessing."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier, MetadataEntry, TimeRange


class PolarizationBandMapping(FrozenModel):
    """Explicit one-based raster band mapping; no band name is inferred."""

    vv_band: int | None = Field(default=None, gt=0)
    vh_band: int | None = Field(default=None, gt=0)
    vv_unit: str | None = None
    vh_unit: str | None = None

    @model_validator(mode="after")
    def _valid_mapping(self) -> PolarizationBandMapping:
        selected = tuple(item for item in (self.vv_band, self.vh_band) if item is not None)
        if not selected:
            raise ValueError("at least one explicit VV/VH band mapping is required")
        if len(selected) != len(set(selected)):
            raise ValueError("VV and VH cannot map to the same raster band")
        if self.vv_band is not None and not self.vv_unit:
            raise ValueError("vv_unit is required when vv_band is configured")
        if self.vh_band is not None and not self.vh_unit:
            raise ValueError("vh_unit is required when vh_band is configured")
        return self

    def ordered(self) -> tuple[tuple[str, int, str], ...]:
        result: list[tuple[str, int, str]] = []
        if self.vv_band is not None and self.vv_unit is not None:
            result.append(("VV", self.vv_band, self.vv_unit))
        if self.vh_band is not None and self.vh_unit is not None:
            result.append(("VH", self.vh_band, self.vh_unit))
        return tuple(result)


class RasterioSARReaderConfig(FrozenModel):
    """Metadata required to interpret a canonical georeferenced raster.

    Acquisition time and polarization layout are explicit because the future real Sentinel-1
    product metadata/layout has not been supplied: DATASET_VALIDATION_REQUIRED.
    """

    platform: Identifier
    product_type: Identifier
    acquisition: TimeRange
    bands: PolarizationBandMapping
    processing_level: str | None = None
    metadata: tuple[MetadataEntry, ...] = ()


class SARPreprocessingStage(FrozenModel):
    """One requested stage and its fully recorded, backend-neutral parameters."""

    enabled: bool
    parameters: tuple[MetadataEntry, ...] = ()
    validation_note: Identifier


class SARPreprocessingConfig(FrozenModel):
    """Canonical preprocessing plan; backend APIs remain outside this model."""

    output_root: Path
    calibration: SARPreprocessingStage
    speckle_filtering: SARPreprocessingStage
    terrain_correction: SARPreprocessingStage
    land_sea_masking: SARPreprocessingStage
    output_bands: PolarizationBandMapping

    def stages(self) -> tuple[tuple[str, SARPreprocessingStage], ...]:
        return (
            ("calibration", self.calibration),
            ("speckle_filtering", self.speckle_filtering),
            ("terrain_correction", self.terrain_correction),
            ("land_sea_masking", self.land_sea_masking),
        )


class SnapGPTConfig(FrozenModel):
    """SNAP command configuration; graph semantics are externally supplied and checksummed."""

    executable: Path
    graph: Path
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: tuple[MetadataEntry, ...] = ()


class SnapSARPreprocessorSettings(SARPreprocessingConfig):
    """Fully typed settings accepted by the registry's SNAP factory."""

    snap: SnapGPTConfig
