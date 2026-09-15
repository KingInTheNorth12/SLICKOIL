"""Shared immutable values, timestamps, metadata, and quality records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _to_utc(value: object) -> object:
    """Reject naive datetimes and normalize aware datetimes to UTC."""
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _parsed_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


UTCDateTime = Annotated[datetime, BeforeValidator(_to_utc), AfterValidator(_parsed_utc)]
Identifier = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
ScalarMetadataValue: TypeAlias = str | int | float | bool


class FrozenModel(BaseModel):
    """Base for immutable, strict-boundary domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class TimeRange(FrozenModel):
    """Closed UTC time range."""

    start: UTCDateTime
    end: UTCDateTime

    @model_validator(mode="after")
    def _ordered(self) -> TimeRange:
        if self.end < self.start:
            raise ValueError("time range end must not precede start")
        return self


class ArtifactRef(FrozenModel):
    """Immutable external data artifact referenced by checksum."""

    uri: Identifier
    media_type: Identifier
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    byte_size: Annotated[int, Field(ge=0)] | None = None


class MetadataEntry(FrozenModel):
    """One serializable metadata entry; used instead of free-form dictionaries."""

    key: Identifier
    value: ScalarMetadataValue


class ComponentMetadata(FrozenModel):
    """Version identity for a replaceable implementation."""

    name: Identifier
    version: Identifier
    implementation: Identifier
    framework: str | None = None
    checkpoint: ArtifactRef | None = None
    configuration_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    attributes: tuple[MetadataEntry, ...] = ()


class DatasetMetadata(FrozenModel):
    """Version and provenance identity for an input dataset."""

    name: Identifier
    version: Identifier
    provider: Identifier
    retrieved_at: UTCDateTime | None = None
    license: str | None = None
    manifest: ArtifactRef | None = None


class QualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityFlag(FrozenModel):
    """Auditable quality condition attached to a domain object."""

    code: Identifier
    severity: QualitySeverity
    message: Identifier


class ProcessingRecord(FrozenModel):
    """One provenance-preserving processing step."""

    component: ComponentMetadata
    started_at: UTCDateTime
    ended_at: UTCDateTime
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> ProcessingRecord:
        if self.ended_at < self.started_at:
            raise ValueError("processing ended_at must not precede started_at")
        return self


class StageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABSTAINED = "abstained"


class StageOutcome(FrozenModel):
    """Typed execution record for a pipeline stage."""

    stage: Identifier
    status: StageStatus
    started_at: UTCDateTime
    ended_at: UTCDateTime
    input_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    warnings: tuple[QualityFlag, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _valid_state(self) -> StageOutcome:
        if self.ended_at < self.started_at:
            raise ValueError("stage ended_at must not precede started_at")
        if self.status == StageStatus.FAILED and self.error_code is None:
            raise ValueError("failed stage outcomes require an error_code")
        return self


class Decision(StrEnum):
    RANKED = "ranked"
    ABSTAINED = "abstained"


class CalibrationStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"
    NOT_APPLICABLE = "not_applicable"


class DataKind(StrEnum):
    ANALYSIS = "analysis"
    FORECAST = "forecast"
    OBSERVATION = "observation"


AttributeName = Literal["source", "derived", "synthetic"]
