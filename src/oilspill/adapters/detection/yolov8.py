"""Ultralytics YOLOv8-Seg adapter with a framework-neutral public boundary."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio

from oilspill.adapters.detection.config import YOLOv8SegSettings
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

ModelFactory = Callable[[Path], Any]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type=media_type,
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise DetectorInputError("scene_id has no characters safe for an artifact path")
    return result


def _construct_ultralytics_model(checkpoint: Path) -> Any:
    """Import Ultralytics lazily so selecting another detector needs no YOLO dependency."""

    try:
        module = importlib.import_module("ultralytics")
        yolo_class = module.YOLO
    except (ImportError, AttributeError) as error:
        raise DetectorDependencyError(
            "YOLOv8-Seg requires the optional 'ultralytics' package; install it in the runtime "
            "environment before selecting detector.name=yolov8_seg"
        ) from error
    return yolo_class(str(checkpoint), task="segment")


def _as_numpy(value: object) -> np.ndarray[Any, Any]:
    """Convert a backend array/tensor entirely inside the adapter boundary."""

    current: Any = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


class YOLOv8SegAdapter:
    """Adapt local YOLOv8-Seg inference into canonical ``SpillDetection`` objects.

    The adapter never downloads checkpoints. It validates the configured local checkpoint and
    converts every Ultralytics result/tensor to NumPy and then to georeferenced raster artifacts
    before returning. Slick polygon characterization is intentionally absent.
    """

    def __init__(
        self,
        config: ComponentConfig,
        *,
        model_factory: ModelFactory = _construct_ultralytics_model,
    ) -> None:
        self._component_config = config
        self._settings = YOLOv8SegSettings.model_validate(config.settings)
        self._model_factory = model_factory
        self._model: Any | None = None

    def load(self) -> None:
        checkpoint = self._settings.checkpoint
        if not checkpoint.is_file():
            raise MissingModelCheckpointError(
                f"configured YOLO checkpoint does not exist: {checkpoint}; automatic download "
                "is disabled"
            )
        actual_checksum = _sha256_file(checkpoint)
        if actual_checksum != self._settings.checkpoint_sha256:
            raise DetectorArtifactError(
                f"YOLO checkpoint checksum mismatch for {checkpoint}; refusing to load"
            )
        self._model = self._model_factory(checkpoint)

    def predict(self, scene: SARScene) -> tuple[SpillDetection, ...]:
        if self._model is None:
            raise RuntimeError("detector must be loaded before prediction")
        image, invalid_pixels = self._prepare_input(scene)
        raw = cast(Any, self._model).predict(
            source=image,
            imgsz=self._settings.image_size,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            device=self._settings.device,
            verbose=False,
        )
        results = tuple(raw)
        if len(results) != 1:
            raise DetectorOutputError(
                f"single-scene inference returned {len(results)} result objects; expected one"
            )
        return self._convert_result(scene, results[0], invalid_pixels)

    def predict_batch(self, scenes: Sequence[SARScene]) -> tuple[SpillDetection, ...]:
        return tuple(detection for scene in scenes for detection in self.predict(scene))

    def model_metadata(self) -> ComponentMetadata:
        checkpoint = self._settings.checkpoint
        checkpoint_ref = ArtifactRef(
            uri=checkpoint.resolve().as_uri(),
            media_type="application/x-ultralytics-checkpoint",
            sha256=self._settings.checkpoint_sha256,
            byte_size=checkpoint.stat().st_size if checkpoint.is_file() else None,
        )
        return ComponentMetadata(
            name=self._component_config.name,
            version=self._settings.model_version,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="ultralytics-yolov8-seg",
            checkpoint=checkpoint_ref,
            configuration_sha256=sha256(self._settings.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(
                    key="channel_normalization_validation",
                    value=self._settings.channel_assumption,
                ),
                MetadataEntry(key="real_dataset_status", value="DATASET_VALIDATION_REQUIRED"),
            ),
        )

    def _prepare_input(self, scene: SARScene) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        try:
            source = local_path_from_artifact(scene.raster.artifact)
        except UnsupportedArtifactURIError as error:
            raise DetectorInputError(str(error)) from error
        if not source.is_file():
            raise DetectorInputError(f"SAR raster does not exist: {source}")
        channels: list[np.ndarray[Any, Any]] = []
        with rasterio.open(source) as dataset:
            if (
                dataset.width != scene.raster.grid.width
                or dataset.height != scene.raster.grid.height
            ):
                raise DetectorInputError("SAR artifact dimensions do not match SARScene metadata")
            if dataset.crs is None:
                raise DetectorInputError("SAR artifact has no CRS")
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
                values = dataset.read(channel.source_band, masked=True).astype("float32")
                filled = np.asarray(values.filled(channel.nodata_fill), dtype=np.float32)
                normalized = np.clip(
                    (filled - channel.input_min) / (channel.input_max - channel.input_min),
                    0.0,
                    1.0,
                )
                channels.append(np.asarray(np.rint(normalized * 255.0), dtype=np.uint8))
        return np.stack(channels, axis=-1), invalid_pixels

    def _convert_result(
        self,
        scene: SARScene,
        result: object,
        invalid_pixels: np.ndarray[Any, Any],
    ) -> tuple[SpillDetection, ...]:
        masks_container = getattr(result, "masks", None)
        if masks_container is None:
            return ()
        mask_data = getattr(masks_container, "data", None)
        if mask_data is None:
            raise DetectorOutputError("YOLO result masks contain no data")
        masks = _as_numpy(mask_data)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        expected_shape = (scene.raster.grid.height, scene.raster.grid.width)
        if masks.ndim != 3 or tuple(masks.shape[1:]) != expected_shape:
            raise DetectorOutputError(
                f"YOLO mask shape {masks.shape} does not match scene grid {expected_shape}; "
                "implicit resizing is disabled"
            )

        boxes = getattr(result, "boxes", None)
        confidences = self._box_values(boxes, "conf", len(masks))
        class_ids = self._box_values(boxes, "cls", len(masks))
        names = getattr(result, "names", None)
        detections: list[SpillDetection] = []
        for index, raw_mask in enumerate(masks):
            raw_class_id = float(class_ids[index])
            confidence = float(confidences[index])
            if not np.isfinite(raw_class_id) or raw_class_id < 0 or not raw_class_id.is_integer():
                raise DetectorOutputError(f"YOLO returned invalid class ID {raw_class_id!r}")
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise DetectorOutputError(f"YOLO returned invalid confidence {confidence!r}")
            class_id = int(raw_class_id)
            class_label = self._class_label(names, class_id)
            mask = np.where(raw_mask >= self._settings.mask_threshold, 1, 0).astype("uint8")
            mask[invalid_pixels] = 255
            raster = self._write_mask(scene, index, mask)
            detections.append(
                SpillDetection(
                    detection_id=self._detection_id(scene, index),
                    scene_id=scene.scene_id,
                    observed_at=scene.acquisition.end,
                    mask=raster,
                    probability_raster=None,
                    class_label=class_label,
                    class_id=class_id,
                    confidence=confidence,
                    footprint=scene.footprint,
                    model=self.model_metadata(),
                )
            )
        return tuple(detections)

    @staticmethod
    def _box_values(boxes: object, name: str, expected: int) -> np.ndarray[Any, Any]:
        value = getattr(boxes, name, None)
        if value is None:
            raise DetectorOutputError(f"YOLO result boxes contain no {name} values")
        array = _as_numpy(value).reshape(-1)
        if len(array) != expected:
            raise DetectorOutputError(
                f"YOLO returned {len(array)} {name} values for {expected} masks"
            )
        return array

    @staticmethod
    def _class_label(names: object, class_id: int) -> str:
        if isinstance(names, dict):
            value = names.get(class_id)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            value = names[class_id]
        else:
            value = None
        if not isinstance(value, str) or not value:
            raise DetectorOutputError(f"YOLO returned no class label for class ID {class_id}")
        return value

    def _output_directory(self, scene: SARScene) -> Path:
        identity = f"{scene.raster.artifact.sha256}:{self.model_metadata().configuration_sha256}"
        digest = sha256(identity.encode()).hexdigest()[:16]
        return self._settings.output_root / _safe_identifier(scene.scene_id) / digest

    def _detection_id(self, scene: SARScene, index: int) -> str:
        return f"{self._component_config.name}:{scene.scene_id}:{index:04d}"

    def _write_mask(self, scene: SARScene, index: int, mask: np.ndarray[Any, Any]) -> RasterAsset:
        output_dir = self._output_directory(scene)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"detection-{index:04d}-mask.tif"
        temporary = output.with_suffix(".tmp.tif")
        try:
            source = local_path_from_artifact(scene.raster.artifact)
        except UnsupportedArtifactURIError as error:
            raise DetectorInputError(str(error)) from error
        try:
            with rasterio.open(source) as dataset:
                profile = dataset.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
            with rasterio.open(temporary, "w", **profile) as dataset:
                dataset.write(mask, 1)
                dataset.set_band_description(1, "oil-spill binary segmentation mask")
                dataset.update_tags(
                    SOURCE_SCENE_ID=scene.scene_id,
                    MODEL_NAME=self._component_config.name,
                    GEOMETRY_SEMANTICS="raster_coverage_not_characterized_slick_geometry",
                )
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return RasterAsset(
            artifact=_artifact(output, "image/tiff; application=geotiff"),
            grid=scene.raster.grid,
            bands=(RasterBand(name="oil-spill-mask", unit="binary", nodata=255),),
        )


def create_yolov8_seg(config: ComponentConfig) -> YOLOv8SegAdapter:
    return YOLOv8SegAdapter(config)
