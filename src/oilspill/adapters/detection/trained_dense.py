"""Load versioned dense training artifacts inside the inference adapter boundary."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import rasterio

from oilspill.adapters.detection.config import DenseSegmentationSettings
from oilspill.adapters.detection.errors import DetectorArtifactError, DetectorInputError
from oilspill.artifacts import local_path_from_artifact
from oilspill.domain.sar import SARScene
from oilspill.training.checkpoints import load_checkpoint
from oilspill.training.config import ModelConfig, RasterConfig

ARCHITECTURES = {
    "Unet": "unet",
    "UnetPlusPlus": "unetplusplus",
    "DeepLabV3Plus": "deeplabv3plus",
    "Segformer": "segformer",
}


def construct_training_backend(settings: DenseSegmentationSettings, architecture: str) -> Any:
    from oilspill.adapters.detection.dense import _TorchDenseBackend
    from oilspill.training.segmentation.models import create_segmentation_model

    torch = importlib.import_module("torch")
    payload = load_checkpoint(settings.checkpoint)
    config = ModelConfig.model_validate(payload["model_config"])
    allowed = {ARCHITECTURES[architecture]}
    if architecture == "DeepLabV3Plus":
        allowed.add("deeplabv3plus_scse_boundary")
    if (
        config.architecture not in allowed
        or config.in_channels != len(settings.input_channels)
        or settings.output_class_count != 1
        or settings.output_activation != "sigmoid"
        or settings.encoder_name != (config.backbone or config.encoder)
    ):
        raise DetectorArtifactError("detector configuration differs from training model")
    if payload.get("threshold") != settings.mask_threshold:
        raise DetectorArtifactError("detector threshold differs from training evaluation threshold")
    raster = RasterConfig.model_validate(payload["raster_config"])
    if raster.size is not None and raster.size != (
        settings.spatial_transform.input_height,
        settings.spatial_transform.input_width,
    ):
        raise DetectorArtifactError("detector input size differs from training raster size")
    # Never fetch initialization weights when reconstructing an already trained model.
    model = create_segmentation_model(config, initialize_pretrained=False)
    model.load_state_dict(
        payload["model_state_dict"],
        strict=(
            config.architecture == "deeplabv3plus_scse_boundary"
            or settings.strict_checkpoint_loading
        ),
    )
    model.to(settings.device).eval()
    return _TorchDenseBackend(torch, model, settings)


def prepare_training_input(
    settings: DenseSegmentationSettings, scene: SARScene, raster: RasterConfig
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Use checkpoint normalization, retaining every band in its audited source order."""
    with rasterio.open(local_path_from_artifact(scene.raster.artifact)) as src:
        if (src.width, src.height) != (scene.raster.grid.width, scene.raster.grid.height):
            raise DetectorInputError("SAR artifact dimensions do not match scene")
        if src.count != len(settings.input_channels) or tuple(
            c.source_band for c in settings.input_channels
        ) != tuple(src.indexes):
            raise DetectorInputError("training checkpoint requires every band in source order")
        target = (settings.spatial_transform.input_height, settings.spatial_transform.input_width)
        if raster.size is None and target != (src.height, src.width):
            raise DetectorInputError(
                "training preserved native dimensions; implicit resize forbidden"
            )
        raw = src.read()
        invalid = np.any(src.read_masks() == 0, axis=0) | ~np.isfinite(raw).all(axis=0)
        if invalid.any():
            raise DetectorInputError("training preprocessing rejects nodata/nonfinite pixels")
        # PyTorch bilinear uses different sampling from GDAL; reuse precisely when resizing.
        data = raw.astype("float32")
        norm = raster.normalization
        if norm.mode == "standardize":
            data = (data - np.asarray(norm.mean, dtype="float32")[:, None, None]) / np.asarray(
                norm.std, dtype="float32"
            )[:, None, None]
        if raster.size is not None and target != (src.height, src.width):
            torch = importlib.import_module("torch")
            data = torch.nn.functional.interpolate(
                torch.from_numpy(data)[None], size=target, mode="bilinear", align_corners=False
            )[0].numpy()
        return data[None], invalid
