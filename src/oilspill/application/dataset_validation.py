"""Non-scientific validation of canonical local dataset manifests and artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import Field

from oilspill.benchmarking.models import DetectionBenchmarkDataset, load_detection_dataset
from oilspill.config import load_config
from oilspill.domain.common import ArtifactRef, FrozenModel, Identifier


class DatasetValidationConfig(FrozenModel):
    """Validation controls for an already-adapted canonical detection dataset.

    Source-specific SAR, AIS, metocean, and coastline validation belongs in those
    providers. This service deliberately validates only the canonical manifest boundary.
    """

    schema_version: int = Field(default=1, ge=1)
    manifest: Path
    verify_checksums: bool = True
    verify_byte_sizes: bool = True


class DatasetValidationIssue(FrozenModel):
    code: Identifier
    message: Identifier
    case_id: str | None = None
    artifact_uri: str | None = None


class DatasetValidationResult(FrozenModel):
    schema_version: int = 1
    valid: bool
    dataset_identifier: Identifier
    dataset_version: Identifier
    case_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    synthetic: bool
    scientifically_representative: bool
    issues: tuple[DatasetValidationIssue, ...] = ()


def load_dataset_validation_config(path: str | Path) -> DatasetValidationConfig:
    owner = Path(path).resolve()
    config = load_config(owner, DatasetValidationConfig)
    manifest = config.manifest
    if not manifest.is_absolute():
        manifest = (owner.parent / manifest).resolve()
    return config.model_copy(update={"manifest": manifest})


def _local_path(artifact: ArtifactRef) -> Path | None:
    parsed = urlparse(artifact.uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "":
        return Path(artifact.uri)
    return None


def _artifacts(dataset: DetectionBenchmarkDataset) -> tuple[tuple[str, ArtifactRef], ...]:
    found: list[tuple[str, ArtifactRef]] = []
    for case in dataset.cases:
        found.extend(
            (
                (case.case_id, case.scene.source),
                (case.case_id, case.scene.raster.artifact),
            )
        )
        for detection in case.ground_truth:
            found.append((case.case_id, detection.mask.artifact))
            if detection.probability_raster is not None:
                found.append((case.case_id, detection.probability_raster.artifact))
    unique: dict[tuple[str, str], tuple[str, ArtifactRef]] = {}
    for case_id, artifact in found:
        unique.setdefault((artifact.uri, artifact.sha256), (case_id, artifact))
    return tuple(unique.values())


def validate_canonical_detection_dataset(
    config: DatasetValidationConfig,
) -> DatasetValidationResult:
    """Validate manifest structure and referenced local files without interpreting science."""

    dataset = load_detection_dataset(config.manifest)
    issues: list[DatasetValidationIssue] = []
    artifacts = _artifacts(dataset)
    for case_id, artifact in artifacts:
        local_path = _local_path(artifact)
        if local_path is None:
            issues.append(
                DatasetValidationIssue(
                    code="unsupported_artifact_uri",
                    message="only local paths and file URIs can be validated by this command",
                    case_id=case_id,
                    artifact_uri=artifact.uri,
                )
            )
            continue
        if not local_path.is_file():
            issues.append(
                DatasetValidationIssue(
                    code="missing_artifact",
                    message=f"referenced artifact does not exist: {local_path}",
                    case_id=case_id,
                    artifact_uri=artifact.uri,
                )
            )
            continue
        content = local_path.read_bytes()
        if config.verify_checksums and sha256(content).hexdigest() != artifact.sha256:
            issues.append(
                DatasetValidationIssue(
                    code="checksum_mismatch",
                    message=f"SHA-256 does not match manifest: {local_path}",
                    case_id=case_id,
                    artifact_uri=artifact.uri,
                )
            )
        if (
            config.verify_byte_sizes
            and artifact.byte_size is not None
            and len(content) != artifact.byte_size
        ):
            issues.append(
                DatasetValidationIssue(
                    code="byte_size_mismatch",
                    message=f"byte size does not match manifest: {local_path}",
                    case_id=case_id,
                    artifact_uri=artifact.uri,
                )
            )
    return DatasetValidationResult(
        valid=not issues,
        dataset_identifier=dataset.identifier,
        dataset_version=dataset.version,
        case_count=len(dataset.cases),
        artifact_count=len(artifacts),
        synthetic=dataset.synthetic,
        scientifically_representative=dataset.scientifically_representative,
        issues=tuple(issues),
    )
