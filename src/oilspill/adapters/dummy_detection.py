"""Deterministic detector adapters used for registry and orchestration tests only."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from pydantic import Field

from oilspill.config import ComponentConfig
from oilspill.domain.common import ArtifactRef, ComponentMetadata, FrozenModel, MetadataEntry
from oilspill.domain.geospatial import RasterAsset, RasterBand
from oilspill.domain.sar import SARScene
from oilspill.domain.spill import SpillDetection


class DummyDetectorSettings(FrozenModel):
    class_label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    mask_artifact: ArtifactRef | None = None
    mask_nodata: float | int | None = 255


class DummyDetector:
    """Deterministic test double returning a typed synthetic mask reference.

    It performs no image analysis and is not scientifically representative.
    """

    component_name = "dummy"

    def __init__(self, config: ComponentConfig) -> None:
        self._config = config
        self._settings = DummyDetectorSettings.model_validate(config.settings)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def predict(self, scene: SARScene) -> tuple[SpillDetection, ...]:
        if not self._loaded:
            raise RuntimeError("detector must be loaded before prediction")
        mask_identity = f"{self.component_name}:{scene.scene_id}:synthetic-mask"
        mask_artifact = self._settings.mask_artifact or ArtifactRef(
            uri=f"memory://{mask_identity}",
            media_type="application/x.synthetic-binary-mask",
            sha256=sha256(mask_identity.encode()).hexdigest(),
        )
        mask = RasterAsset(
            artifact=mask_artifact,
            grid=scene.raster.grid,
            bands=(
                RasterBand(
                    name="synthetic-dummy-mask",
                    unit="binary",
                    nodata=self._settings.mask_nodata,
                    source_index=1,
                ),
            ),
        )
        return (
            SpillDetection(
                detection_id=f"{self.component_name}:{scene.scene_id}",
                scene_id=scene.scene_id,
                observed_at=scene.acquisition.end,
                mask=mask,
                class_label=self._settings.class_label,
                class_id=0,
                confidence=self._settings.confidence,
                footprint=scene.footprint,
                model=self.model_metadata(),
            ),
        )

    def predict_batch(self, scenes: Sequence[SARScene]) -> tuple[SpillDetection, ...]:
        return tuple(detection for scene in scenes for detection in self.predict(scene))

    def model_metadata(self) -> ComponentMetadata:
        digest = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name=self.component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="deterministic-test-double",
            configuration_sha256=digest,
            attributes=(
                MetadataEntry(key="synthetic", value=True),
                MetadataEntry(key="scientifically_representative", value=False),
            ),
        )


class DummyDetectorA(DummyDetector):
    component_name = "dummy_a"


class DummyDetectorB(DummyDetector):
    component_name = "dummy_b"


def create_dummy_a(config: ComponentConfig) -> DummyDetectorA:
    return DummyDetectorA(config)


def create_dummy_b(config: ComponentConfig) -> DummyDetectorB:
    return DummyDetectorB(config)
