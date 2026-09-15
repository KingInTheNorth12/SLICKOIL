"""Oil-slick detection, geometry, and observation contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from oilspill.domain.common import (
    ComponentMetadata,
    FrozenModel,
    Identifier,
    MetadataEntry,
    Probability,
    QualityFlag,
    TimeRange,
    UTCDateTime,
)
from oilspill.domain.geospatial import CRS, Angle, Area, Length, RasterAsset, SpatialGeometry
from oilspill.domain.sar import SARScene


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class SpillDetection(FrozenModel):
    """Framework-neutral segmentation result for one candidate slick.

    ``mask`` and the optional ``probability_raster`` carry their CRS, affine transform,
    dimensions, spatial units, and nodata semantics through ``RasterAsset``. ``footprint`` is
    the spatial coverage of the detection raster; deriving a slick polygon belongs to a
    separate ``SpillCharacterizer`` implementation.
    """

    detection_id: Identifier
    scene_id: Identifier
    observed_at: UTCDateTime
    mask: RasterAsset
    probability_raster: RasterAsset | None = None
    class_label: Identifier
    class_id: int | None = Field(default=None, ge=0)
    confidence: Probability | None = None
    footprint: SpatialGeometry
    model: ComponentMetadata
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _consistent_crs(self) -> SpillDetection:
        if self.footprint.crs != self.mask.grid.crs:
            raise ValueError("detection footprint and mask must use the same CRS")
        if self.probability_raster and self.probability_raster.grid != self.mask.grid:
            raise ValueError("probability raster and mask must use the same grid")
        return self


class SpillGeometry(FrozenModel):
    """CRS-aware slick geometry with physical (not angular) measurements.

    Vector geometries remain in the detection raster CRS. Physical measurements are computed
    in ``measurement_crs``, which must be projected; ``skeleton`` remains on the detection grid.
    """

    geometry_id: Identifier
    detection_id: Identifier
    polygon: SpatialGeometry
    centroid: SpatialGeometry
    centerline: SpatialGeometry | None = None
    skeleton: RasterAsset | None = None
    area: Area | None = None
    length: Length | None = None
    width: Length | None = None
    orientation: Angle | None = None
    measurement_crs: CRS | None = None
    method: ComponentMetadata
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _consistent_geometry(self) -> SpillGeometry:
        if self.polygon.geometry.type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("SpillGeometry.polygon must contain Polygon or MultiPolygon geometry")
        if self.centroid.geometry.type != "Point":
            raise ValueError("SpillGeometry.centroid must contain Point geometry")
        if self.centerline is not None and self.centerline.geometry.type != "LineString":
            raise ValueError("SpillGeometry.centerline must contain LineString geometry")
        geometries = (
            (self.centroid,) if self.centerline is None else (self.centroid, self.centerline)
        )
        if any(item.crs != self.polygon.crs for item in geometries):
            raise ValueError("all slick geometries must use the same CRS")
        if self.skeleton is not None and self.skeleton.grid.crs != self.polygon.crs:
            raise ValueError("slick skeleton and vector geometries must use the same CRS")
        if self.measurement_crs is not None and self.measurement_crs.is_geographic:
            raise ValueError("measurement_crs must be projected for physical measurements")
        return self


class DischargeTimeEvidence(FrozenModel):
    """Auditable input evidence considered by a spill-age estimator."""

    kind: Literal["sar_acquisition", "slick_morphology", "ais_history", "configuration"]
    description: Identifier
    source_ids: tuple[str, ...] = ()
    attributes: tuple[MetadataEntry, ...] = ()


class DischargeTimeEstimate(FrozenModel):
    """Plausible interval, not an assertion of exact discharge time."""

    interval: TimeRange
    central_estimate: UTCDateTime | None = None
    method: ComponentMetadata
    confidence: Probability | None = None
    assumptions: tuple[str, ...] = ()
    evidence: tuple[DischargeTimeEvidence, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _central_within_interval(self) -> DischargeTimeEstimate:
        if self.central_estimate is not None and not (
            self.interval.start <= self.central_estimate <= self.interval.end
        ):
            raise ValueError("central_estimate must lie within the plausible interval")
        return self


class SpillObservation(FrozenModel):
    """Validated aggregate passed from image analysis into tracing stages."""

    observation_id: Identifier
    scene: SARScene
    observed_at: UTCDateTime
    detection: SpillDetection
    geometry: SpillGeometry
    discharge_time: DischargeTimeEstimate
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _consistent_links(self) -> SpillObservation:
        if self.detection.scene_id != self.scene.scene_id:
            raise ValueError("detection scene_id must match observation scene")
        if self.geometry.detection_id != self.detection.detection_id:
            raise ValueError("geometry detection_id must match observation detection")
        if self.observed_at != self.detection.observed_at:
            raise ValueError("observation and detection timestamps must match")
        if not (self.scene.acquisition.start <= self.observed_at <= self.scene.acquisition.end):
            raise ValueError("observed_at must fall within the SAR acquisition interval")
        if self.discharge_time.interval.end > self.observed_at:
            raise ValueError("plausible discharge interval cannot end after observation")
        return self
