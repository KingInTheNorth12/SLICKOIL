"""Backend-neutral SAR preprocessing coordinator and canonical output conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol, runtime_checkable

from oilspill.adapters.sar.config import RasterioSARReaderConfig, SARPreprocessingConfig
from oilspill.adapters.sar.errors import SARBackendExecutionError, SARInputError
from oilspill.adapters.sar.reader import RasterioSARReader, local_path_from_artifact
from oilspill.domain.common import (
    ArtifactRef,
    ComponentMetadata,
    MetadataEntry,
    ProcessingRecord,
)
from oilspill.domain.sar import SARScene
from oilspill.requests import SARReadRequest


@dataclass(frozen=True)
class BackendExecutionResult:
    """Backend-internal output; only the canonical SARScene leaves the adapter layer."""

    output_path: Path
    executed_stages: tuple[str, ...]


@runtime_checkable
class SARPreprocessingBackend(Protocol):
    """Internal adapter port implemented by SNAP, pyroSAR, or test backends."""

    def execute(
        self,
        source: Path,
        destination: Path,
        config: SARPreprocessingConfig,
    ) -> BackendExecutionResult: ...

    def component_metadata(self) -> ComponentMetadata: ...


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _processing_payload(config: SARPreprocessingConfig) -> dict[str, object]:
    """Exclude deployment location from the content-addressed scientific plan identity."""

    return config.model_dump(mode="json", exclude={"output_root"})


def _processing_digest(config: SARPreprocessingConfig) -> str:
    return sha256(_canonical_json(_processing_payload(config)).encode()).hexdigest()


def _artifact(path: Path, media_type: str) -> ArtifactRef:
    digest = sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        media_type=media_type,
        sha256=digest,
        byte_size=path.stat().st_size,
    )


def _safe_scene_id(scene_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", scene_id).strip("-.")
    if not normalized:
        raise SARInputError("scene_id has no characters safe for a deterministic output path")
    return normalized


class BackendSARPreprocessor:
    """Run a selected backend and convert its raster into a canonical ``SARScene``.

    The coordinator validates that every enabled generic stage is explicitly reported by the
    backend. It does not claim that an unreported stage succeeded.
    """

    def __init__(
        self,
        config: SARPreprocessingConfig,
        backend: SARPreprocessingBackend,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._clock = clock or (lambda: datetime.now(UTC))

    def deterministic_output_path(self, scene: SARScene) -> Path:
        digest = _processing_digest(self._config)
        return (
            self._config.output_root
            / _safe_scene_id(scene.scene_id)
            / digest[:16]
            / "sar_preprocessed.tif"
        )

    def preprocess(self, scene: SARScene) -> SARScene:
        source_path = local_path_from_artifact(scene.raster.artifact)
        output_path = self.deterministic_output_path(scene)
        config_path = output_path.with_name("preprocessing_config.json")
        report_path = output_path.with_name("preprocessing_execution.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started_at = self._clock()
        if output_path.exists() and config_path.exists() and report_path.exists():
            executed_stages = self._read_execution_report(report_path)
        else:
            temporary_path = output_path.with_name(".sar_preprocessed.tmp.tif")
            try:
                result = self._backend.execute(source_path, temporary_path, self._config)
                if result.output_path != temporary_path:
                    raise SARBackendExecutionError(
                        "backend returned an unexpected output path; no result was promoted"
                    )
                if not temporary_path.is_file():
                    raise SARBackendExecutionError(
                        "backend reported success but produced no output raster"
                    )
                executed_stages = result.executed_stages
                self._validate_executed_stages(executed_stages)
                temporary_path.replace(output_path)
                config_path.write_text(self._config.model_dump_json(indent=2), encoding="utf-8")
                report_path.write_text(
                    _canonical_json(
                        {
                            "backend": self._backend.component_metadata().model_dump(mode="json"),
                            "executed_stages": executed_stages,
                        }
                    ),
                    encoding="utf-8",
                )
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        self._validate_executed_stages(executed_stages)
        output_artifact = _artifact(output_path, "image/tiff; application=geotiff")
        config_artifact = _artifact(config_path, "application/json")
        report_artifact = _artifact(report_path, "application/json")
        output_reader = RasterioSARReader(
            RasterioSARReaderConfig(
                platform=scene.platform,
                product_type=scene.product_type,
                acquisition=scene.acquisition,
                bands=self._config.output_bands,
                processing_level="backend-preprocessed",
                metadata=scene.metadata,
            )
        )
        canonical = output_reader.read(
            SARReadRequest(source=output_artifact, scene_id=scene.scene_id)
        )
        ended_at = self._clock()
        processing_record = ProcessingRecord(
            component=self.component_metadata(),
            started_at=started_at,
            ended_at=ended_at,
            input_artifacts=(scene.raster.artifact, config_artifact),
            output_artifacts=(output_artifact, report_artifact),
        )
        return canonical.model_copy(
            update={
                "processing": (*scene.processing, processing_record),
                "metadata": (
                    *canonical.metadata,
                    MetadataEntry(key="preprocessing_config_uri", value=config_artifact.uri),
                    MetadataEntry(key="preprocessing_execution_uri", value=report_artifact.uri),
                    MetadataEntry(key="executed_stages", value=",".join(executed_stages)),
                ),
            }
        )

    def component_metadata(self) -> ComponentMetadata:
        backend_metadata = self._backend.component_metadata()
        return ComponentMetadata(
            name="backend_sar_preprocessor",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework=backend_metadata.name,
            configuration_sha256=_processing_digest(self._config),
            attributes=(
                MetadataEntry(key="backend", value=backend_metadata.implementation),
                MetadataEntry(key="dataset_validation_status", value="DATASET_VALIDATION_REQUIRED"),
            ),
        )

    def _validate_executed_stages(self, executed_stages: tuple[str, ...]) -> None:
        requested = {name for name, stage in self._config.stages() if stage.enabled}
        missing = requested.difference(executed_stages)
        if missing:
            raise SARBackendExecutionError(
                f"backend did not report required preprocessing stages: {sorted(missing)}"
            )

    @staticmethod
    def _read_execution_report(path: Path) -> tuple[str, ...]:
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            stages = content["executed_stages"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise SARBackendExecutionError(
                "invalid cached preprocessing execution report"
            ) from error
        if not isinstance(stages, list) or not all(isinstance(item, str) for item in stages):
            raise SARBackendExecutionError("cached executed_stages must be a list of strings")
        return tuple(stages)
