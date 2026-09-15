"""Typed, explicit configuration for segmentation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from oilspill.domain.common import FrozenModel, Identifier


class DetectorInputChannel(FrozenModel):
    """Map one source raster band into one uint8 model-input channel.

    Values are clipped to ``input_min``/``input_max`` and linearly mapped to [0, 255]. The
    source band, range, and nodata fill are deliberately mandatory because appropriate
    Sentinel-1 channel normalization is DATASET_VALIDATION_REQUIRED.
    """

    source_band: int = Field(gt=0)
    polarization: Identifier
    input_min: float
    input_max: float
    nodata_fill: float
    validation_note: Identifier

    @model_validator(mode="after")
    def _ordered_range(self) -> DetectorInputChannel:
        if self.input_max <= self.input_min:
            raise ValueError("input_max must be greater than input_min")
        if not self.input_min <= self.nodata_fill <= self.input_max:
            raise ValueError("nodata_fill must lie within the configured input range")
        return self


class YOLOv8SegSettings(FrozenModel):
    """Complete settings for the Ultralytics adapter; no inference defaults are hidden."""

    checkpoint: Path
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: Identifier
    output_root: Path
    input_channels: tuple[DetectorInputChannel, DetectorInputChannel, DetectorInputChannel]
    device: Identifier
    image_size: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    mask_threshold: float = Field(ge=0.0, le=1.0)
    channel_assumption: Identifier


class DenseInputChannel(FrozenModel):
    """Explicit SAR-band transformation for a dense tensor channel.

    The adapter clips to ``input_min``/``input_max``, maps to [0, 1], and then applies
    ``(value - normalization_mean) / normalization_std``. All values are mandatory because
    training-compatible SAR normalization remains DATASET_VALIDATION_REQUIRED.
    """

    source_band: int = Field(gt=0)
    polarization: Identifier
    input_min: float
    input_max: float
    nodata_fill: float
    normalization_mean: float
    normalization_std: float = Field(gt=0.0)
    validation_note: Identifier

    @model_validator(mode="after")
    def _ordered_range(self) -> DenseInputChannel:
        if self.input_max <= self.input_min:
            raise ValueError("input_max must be greater than input_min")
        if not self.input_min <= self.nodata_fill <= self.input_max:
            raise ValueError("nodata_fill must lie within the configured input range")
        return self


class DenseSpatialTransform(FrozenModel):
    """Configured model-grid conversion performed only inside a dense-model adapter."""

    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    input_resampling: Literal["nearest", "bilinear"]
    output_resampling: Literal["nearest", "bilinear"]
    validation_note: Identifier


class DenseSegmentationSettings(FrozenModel):
    """Shared explicit contract for checkpoint-backed semantic segmentation adapters."""

    checkpoint: Path
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: Identifier
    output_root: Path
    input_channels: Annotated[tuple[DenseInputChannel, ...], Field(min_length=1)]
    spatial_transform: DenseSpatialTransform
    device: Identifier
    output_activation: Literal["sigmoid", "softmax"]
    foreground_class_index: int = Field(ge=0)
    class_id: int = Field(ge=0)
    class_label: Identifier
    mask_threshold: float = Field(ge=0.0, le=1.0)
    encoder_name: Identifier
    # Non-null pretrained encoder weights can trigger network downloads, so they are forbidden.
    encoder_weights: None = None
    output_class_count: int = Field(gt=0)
    state_dict_key: str | None = None
    strict_checkpoint_loading: bool
    preprocessing_assumption: Identifier
    # Opt in to the versioned training checkpoint with exact saved model/preprocessing.
    training_checkpoint: bool = False

    @model_validator(mode="after")
    def _valid_foreground_channel(self) -> DenseSegmentationSettings:
        if self.foreground_class_index >= self.output_class_count:
            raise ValueError("foreground_class_index must be below output_class_count")
        return self


class UNetSettings(DenseSegmentationSettings):
    """U-Net settings for the segmentation-models-pytorch backend."""


class DeepLabV3PlusSettings(DenseSegmentationSettings):
    """DeepLabV3+ settings for the segmentation-models-pytorch backend."""
