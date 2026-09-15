"""Shared dense-segmentation adapter machinery for U-Net and DeepLabV3+."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import zoom  # type: ignore[import-untyped]

from oilspill.adapters.detection.config import (
    DeepLabV3PlusSettings,
    DenseSegmentationSettings,
    UNetSettings,
)
from oilspill.adapters.detection.errors import (
    DetectorArtifactError,
    DetectorDependencyError,
    DetectorInputError,
    DetectorOutputError,
    MissingModelCheckpointError,
)
from oilspill.artifacts import UnsupportedArtifactURIError, local_path_from_artifact
from oilspill.config import ComponentConfig
from oilspill.domain.common import ArtifactRef, ComponentMetadata, MetadataEntry
from oilspill.domain.geospatial import RasterAsset, RasterBand
from oilspill.domain.sar import SARScene
from oilspill.domain.spill import SpillDetection


class DenseInferenceBackend(Protocol):
    """Adapter-internal tensor boundary; never exposed to pipeline code."""

    def predict_probabilities(self, batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...


DenseBackendFactory = Callable[[DenseSegmentationSettings, str], DenseInferenceBackend]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise DetectorInputError("scene_id has no characters safe for an artifact path")
    return result


def _artifact(path: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type=media_type,
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _resampling(name: str) -> Resampling:
    if name == "nearest":
        return Resampling.nearest
    if name == "bilinear":
        return Resampling.bilinear
    raise DetectorInputError(f"unsupported configured resampling method: {name}")


class _TorchDenseBackend:
    """Keep all PyTorch tensors and calls behind the internal NumPy boundary."""

    def __init__(self, torch_module: Any, model: Any, settings: DenseSegmentationSettings) -> None:
        self._torch = torch_module
        self._model = model
        self._settings = settings

    def predict_probabilities(self, batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        tensor = self._torch.from_numpy(batch).to(self._settings.device)
        with self._torch.no_grad():
            logits = self._model(tensor)
            if self._settings.output_activation == "sigmoid":
                probabilities = self._torch.sigmoid(logits)
            else:
                probabilities = self._torch.softmax(logits, dim=1)
        return np.asarray(probabilities.detach().cpu().numpy())


def _construct_smp_backend(
    settings: DenseSegmentationSettings,
    architecture: str,
) -> DenseInferenceBackend:
    """Construct an SMP model lazily without downloading encoder or model weights."""

    if settings.training_checkpoint:
        from oilspill.adapters.detection.trained_dense import construct_training_backend

        return cast(DenseInferenceBackend, construct_training_backend(settings, architecture))
    try:
        torch_module = importlib.import_module("torch")
        smp_module = importlib.import_module("segmentation_models_pytorch")
        model_class = getattr(smp_module, architecture)
    except (ImportError, AttributeError) as error:
        raise DetectorDependencyError(
            f"{architecture} requires optional 'torch' and 'segmentation_models_pytorch' "
            "packages; install them before selecting this detector"
        ) from error

    model = model_class(
        encoder_name=settings.encoder_name,
        encoder_weights=None,
        in_channels=len(settings.input_channels),
        classes=settings.output_class_count,
    )
    try:
        payload = torch_module.load(
            settings.checkpoint,
            map_location=settings.device,
            weights_only=True,
        )
        if settings.state_dict_key is not None:
            payload = payload[settings.state_dict_key]
        model.load_state_dict(payload, strict=settings.strict_checkpoint_loading)
        model.to(settings.device)
        model.eval()
    except Exception as error:
        raise DetectorArtifactError(
            f"could not load configured {architecture} state dictionary from {settings.checkpoint}"
        ) from error
    return _TorchDenseBackend(torch_module, model, settings)


class _DenseSegmentationAdapter:
    settings_type: type[DenseSegmentationSettings] = DenseSegmentationSettings
    architecture: str = "dense-segmentation"
    framework_name: str = "segmentation-models-pytorch"

    def __init__(
        self,
        config: ComponentConfig,
        *,
        backend_factory: DenseBackendFactory = _construct_smp_backend,
    ) -> None:
        self._component_config = config
        self._settings = self.settings_type.model_validate(config.settings)
        self._backend_factory = backend_factory
        self._backend: DenseInferenceBackend | None = None
        self._training_raster: Any = None

    def load(self) -> None:
        checkpoint = self._settings.checkpoint
        if not checkpoint.is_file():
            raise MissingModelCheckpointError(
                f"configured {self.architecture} checkpoint does not exist: {checkpoint}; "
                "automatic download is disabled"
            )
        if _sha256_file(checkpoint) != self._settings.checkpoint_sha256:
            raise DetectorArtifactError(
                f"{self.architecture} checkpoint checksum mismatch for {checkpoint}; "
                "refusing to load"
            )
        self._backend = self._backend_factory(self._settings, self.architecture)
        if self._settings.training_checkpoint:
            from oilspill.training.checkpoints import load_checkpoint
            from oilspill.training.config import RasterConfig

            self._training_raster = RasterConfig.model_validate(
                load_checkpoint(checkpoint)["raster_config"]
            )

    def predict(self, scene: SARScene) -> tuple[SpillDetection, ...]:
        if self._backend is None:
            raise RuntimeError("detector must be loaded before prediction")
        batch, invalid_pixels = self._prepare_input(scene)
        output = self._backend.predict_probabilities(batch)
        probability = self._select_probability(output)
        probability = self._restore_scene_shape(probability, scene)
        if (
            not np.isfinite(probability).all()
            or (probability < 0.0).any()
            or (probability > 1.0).any()
        ):
            raise DetectorOutputError(
                f"{self.architecture} output is not a finite probability raster in [0, 1]"
            )
        mask = np.where(probability >= self._settings.mask_threshold, 1, 0).astype("uint8")
        mask[invalid_pixels] = 255
        probability = probability.astype("float32")
        probability[invalid_pixels] = np.float32(-9999.0)
        mask_raster = self._write_output(scene, mask, "mask", "uint8", 255)
        probability_raster = self._write_output(
            scene, probability, "probability", "float32", -9999.0
        )
        return (
            SpillDetection(
                detection_id=f"{self._component_config.name}:{scene.scene_id}:semantic",
                scene_id=scene.scene_id,
                observed_at=scene.acquisition.end,
                mask=mask_raster,
                probability_raster=probability_raster,
                class_label=self._settings.class_label,
                class_id=self._settings.class_id,
                confidence=None,
                footprint=scene.footprint,
                model=self.model_metadata(),
            ),
        )

    def predict_batch(self, scenes: Sequence[SARScene]) -> tuple[SpillDetection, ...]:
        return tuple(detection for scene in scenes for detection in self.predict(scene))

    def model_metadata(self) -> ComponentMetadata:
        checkpoint = self._settings.checkpoint
        checkpoint_ref = ArtifactRef(
            uri=checkpoint.resolve().as_uri(),
            media_type="application/x-pytorch-state-dict",
            sha256=self._settings.checkpoint_sha256,
            byte_size=checkpoint.stat().st_size if checkpoint.is_file() else None,
        )
        return ComponentMetadata(
            name=self._component_config.name,
            version=self._settings.model_version,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework=self.framework_name,
            checkpoint=checkpoint_ref,
            configuration_sha256=sha256(self._settings.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="architecture", value=self.architecture),
                MetadataEntry(
                    key="input_preprocessing_validation",
                    value=self._settings.preprocessing_assumption,
                ),
                MetadataEntry(key="real_dataset_status", value="DATASET_VALIDATION_REQUIRED"),
            ),
        )

    def _prepare_input(self, scene: SARScene) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self._settings.training_checkpoint:
            from oilspill.adapters.detection.trained_dense import prepare_training_input

            return prepare_training_input(self._settings, scene, self._training_raster)
        try:
            source = local_path_from_artifact(scene.raster.artifact)
        except UnsupportedArtifactURIError as error:
            raise DetectorInputError(str(error)) from error
        transform = self._settings.spatial_transform
        channels: list[np.ndarray[Any, Any]] = []
        with rasterio.open(source) as dataset:
            if (
                dataset.width != scene.raster.grid.width
                or dataset.height != scene.raster.grid.height
            ):
                raise DetectorInputError("SAR artifact dimensions do not match SARScene metadata")
            invalid_pixels = dataset.dataset_mask() == 0
            for channel in self._settings.input_channels:
                if channel.source_band > dataset.count:
                    raise DetectorInputError(
                        f"configured source band {channel.source_band} exceeds raster band count"
                    )
                if channel.polarization not in scene.polarizations:
                    raise DetectorInputError(
                        f"configured polarization {channel.polarization!r} is absent from SARScene"
                    )
                values = dataset.read(
                    channel.source_band,
                    out_shape=(transform.input_height, transform.input_width),
                    masked=True,
                    resampling=_resampling(transform.input_resampling),
                ).astype("float32")
                filled = np.asarray(values.filled(channel.nodata_fill), dtype=np.float32)
                unit_interval = np.clip(
                    (filled - channel.input_min) / (channel.input_max - channel.input_min),
                    0.0,
                    1.0,
                )
                channels.append(
                    np.asarray(
                        (unit_interval - channel.normalization_mean) / channel.normalization_std,
                        dtype=np.float32,
                    )
                )
        return np.stack(channels, axis=0)[np.newaxis, ...], invalid_pixels

    def _select_probability(self, output: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if output.ndim != 4 or output.shape[0] != 1:
            raise DetectorOutputError(
                f"{self.architecture} backend returned shape {output.shape}; expected NCHW batch"
            )
        index = self._settings.foreground_class_index
        if index >= output.shape[1]:
            raise DetectorOutputError(
                f"foreground_class_index {index} exceeds output channel count {output.shape[1]}"
            )
        return np.asarray(output[0, index], dtype=np.float32)

    def _restore_scene_shape(
        self, probability: np.ndarray[Any, Any], scene: SARScene
    ) -> np.ndarray[Any, Any]:
        expected_model_shape = (
            self._settings.spatial_transform.input_height,
            self._settings.spatial_transform.input_width,
        )
        if probability.shape != expected_model_shape:
            raise DetectorOutputError(
                f"{self.architecture} output shape {probability.shape} does not match configured "
                f"model grid {expected_model_shape}"
            )
        scene_shape = (scene.raster.grid.height, scene.raster.grid.width)
        if probability.shape == scene_shape:
            return probability
        order = 0 if self._settings.spatial_transform.output_resampling == "nearest" else 1
        restored = zoom(
            probability,
            (scene_shape[0] / probability.shape[0], scene_shape[1] / probability.shape[1]),
            order=order,
            prefilter=False,
        )
        if restored.shape != scene_shape:
            raise DetectorOutputError(
                f"configured output resampling produced {restored.shape}, expected {scene_shape}"
            )
        return cast(np.ndarray[Any, Any], restored)

    def _output_directory(self, scene: SARScene) -> Path:
        identity = f"{scene.raster.artifact.sha256}:{self.model_metadata().configuration_sha256}"
        digest = sha256(identity.encode()).hexdigest()[:16]
        return self._settings.output_root / _safe_identifier(scene.scene_id) / digest

    def _write_output(
        self,
        scene: SARScene,
        values: np.ndarray[Any, Any],
        kind: str,
        dtype: str,
        nodata: int | float,
    ) -> RasterAsset:
        output_dir = self._output_directory(scene)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"semantic-{kind}.tif"
        temporary = output.with_suffix(".tmp.tif")
        try:
            source = local_path_from_artifact(scene.raster.artifact)
        except UnsupportedArtifactURIError as error:
            raise DetectorInputError(str(error)) from error
        try:
            with rasterio.open(source) as dataset:
                profile = dataset.profile.copy()
            profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
            with rasterio.open(temporary, "w", **profile) as dataset:
                dataset.write(values, 1)
                dataset.set_band_description(1, f"oil-spill semantic {kind}")
                dataset.update_tags(
                    SOURCE_SCENE_ID=scene.scene_id,
                    MODEL_NAME=self._component_config.name,
                    GEOMETRY_SEMANTICS="raster_coverage_not_characterized_slick_geometry",
                )
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        unit = "binary" if kind == "mask" else "probability"
        return RasterAsset(
            artifact=_artifact(output, "image/tiff; application=geotiff"),
            grid=scene.raster.grid,
            bands=(RasterBand(name=f"oil-spill-{kind}", unit=unit, nodata=nodata),),
        )


class UNetAdapter(_DenseSegmentationAdapter):
    """U-Net adapter; model-specific tensor handling remains internal."""

    settings_type = UNetSettings
    architecture = "Unet"
    framework_name = "segmentation-models-pytorch-unet"


class DeepLabV3PlusAdapter(_DenseSegmentationAdapter):
    """DeepLabV3+ adapter; model-specific tensor handling remains internal."""

    settings_type = DeepLabV3PlusSettings
    architecture = "DeepLabV3Plus"
    framework_name = "segmentation-models-pytorch-deeplabv3plus"


def create_unet(config: ComponentConfig) -> UNetAdapter:
    return UNetAdapter(config)


def create_deeplabv3plus(config: ComponentConfig) -> DeepLabV3PlusAdapter:
    return DeepLabV3PlusAdapter(config)


class UNetPlusPlusAdapter(_DenseSegmentationAdapter):
    """U-Net++ uses the same canonical raster inference contract."""

    architecture = "UnetPlusPlus"


class SegFormerAdapter(_DenseSegmentationAdapter):
    """SMP SegFormer; training checkpoint mode also aligns dense logits."""

    architecture = "Segformer"


def create_unetplusplus(config: ComponentConfig) -> UNetPlusPlusAdapter:
    return UNetPlusPlusAdapter(config)


def create_segformer(config: ComponentConfig) -> SegFormerAdapter:
    return SegFormerAdapter(config)
