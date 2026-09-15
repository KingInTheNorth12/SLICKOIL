"""Engine-neutral ensemble outcomes and occupancy provenance."""

from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import ArtifactRef, FrozenModel, Identifier, UTCDateTime
from oilspill.domain.geospatial import RasterGrid


class EnsembleMemberOutcome(FrozenModel):
    member_id: Identifier
    member_index: int = Field(ge=0)
    engine_seed: int
    status: Literal["succeeded", "failed"]
    simulation_id: str | None = None
    simulation_record: ArtifactRef | None = None
    error_type: str | None = None
    error_message: str | None = None
    included_in_aggregation: bool | None = None

    @model_validator(mode="after")
    def status_matches_record(self) -> "EnsembleMemberOutcome":
        if self.status == "succeeded":
            if (
                not self.simulation_id
                or self.simulation_record is None
                or self.error_type is not None
            ):
                raise ValueError("successful outcome needs a simulation record and no error")
        elif self.error_type is None or self.error_message is None:
            raise ValueError("failed outcome requires error details")
        if self.status == "failed" and self.included_in_aggregation is True:
            raise ValueError("failed outcome cannot be included in aggregation")
        return self


class OccupancySummary(FrozenModel):
    requested: int = Field(gt=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    denominator: int = Field(ge=0)
    denominator_policy: Literal["valid_members"]
    grid: RasterGrid
    horizon: UTCDateTime
    implementation: Literal["member_cell_intersection"] = "member_cell_intersection"
    version: Literal["1"] = "1"
    semantics: Literal["ensemble occupancy frequency"] = "ensemble occupancy frequency"
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def consistent_counts(self) -> "OccupancySummary":
        if self.requested != self.succeeded + self.failed or self.denominator != self.succeeded:
            raise ValueError("occupancy counts/valid-member denominator are inconsistent")
        return self
