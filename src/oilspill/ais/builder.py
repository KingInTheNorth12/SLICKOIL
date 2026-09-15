"""Canonical AIS-point to vessel-track reconstruction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256

from oilspill.ais.config import VesselTrackBuilderConfig, _duration_seconds
from oilspill.ais.errors import VesselTrackBuildError
from oilspill.config import ComponentConfig
from oilspill.domain.ais import AISPoint, TrackCoverage, VesselTrack
from oilspill.domain.common import ComponentMetadata, MetadataEntry, TimeRange
from oilspill.domain.geospatial import Coordinate, LineStringGeometry, SpatialGeometry


class VesselTrackBuilder:
    """Deterministically sort, deduplicate, and reconstruct configured canonical tracks."""

    def __init__(self, config: VesselTrackBuilderConfig) -> None:
        if config.interpolation_method != "none":
            raise VesselTrackBuildError(
                "configured AIS interpolation is not implemented in this builder; "
                "DATASET_VALIDATION_REQUIRED"
            )
        self._config = config

    def build(self, points: Sequence[AISPoint]) -> tuple[VesselTrack, ...]:
        grouped: dict[str, list[AISPoint]] = defaultdict(list)
        for point in points:
            grouped[point.vessel_id].append(point)
        tracks = []
        for vessel_id in sorted(grouped):
            ordered = sorted(
                grouped[vessel_id],
                key=lambda item: (item.observed_at, item.provider_record_id or ""),
            )
            deduplicated = self._deduplicate(ordered)
            if len(deduplicated) < self._config.minimum_points_per_track:
                if self._config.insufficient_points_policy == "skip":
                    continue
                raise VesselTrackBuildError(f"insufficient AIS points for vessel {vessel_id!r}")
            crs = deduplicated[0].position.crs
            if any(point.position.crs != crs for point in deduplicated):
                raise VesselTrackBuildError("one vessel track cannot mix coordinate systems")
            gaps = tuple(
                TimeRange(start=first.observed_at, end=second.observed_at)
                for first, second in zip(deduplicated, deduplicated[1:], strict=False)
                if (second.observed_at - first.observed_at).total_seconds()
                > _duration_seconds(self._config.gap_threshold)
            )
            coordinates = []
            for point in deduplicated:
                geometry = point.position.geometry
                if geometry.type != "Point":
                    raise VesselTrackBuildError("canonical AIS position is not a point")
                coordinates.append(Coordinate(x=geometry.coordinate.x, y=geometry.coordinate.y))
            tracks.append(
                VesselTrack(
                    track_id=f"{self._config.track_id_prefix}:{vessel_id}",
                    vessel_id=vessel_id,
                    interval=TimeRange(
                        start=deduplicated[0].observed_at,
                        end=deduplicated[-1].observed_at,
                    ),
                    points=tuple(deduplicated),
                    path=SpatialGeometry(
                        geometry=LineStringGeometry(coordinates=tuple(coordinates)), crs=crs
                    ),
                    coverage=TrackCoverage(
                        source_point_count=len(deduplicated),
                        interpolated_point_count=0,
                        maximum_observed_gap=self._config.gap_threshold if gaps else None,
                        detected_gaps=gaps,
                    ),
                )
            )
        return tuple(tracks)

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="vessel_track_builder",
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="canonical-ais-reconstruction",
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="interpolation_method", value="none"),
                MetadataEntry(
                    key="interpolation_validation",
                    value=self._config.interpolation_validation_note,
                ),
            ),
        )

    def _deduplicate(self, points: list[AISPoint]) -> list[AISPoint]:
        by_time: dict[datetime, AISPoint] = {}
        for point in points:
            if point.observed_at in by_time and self._config.duplicate_policy == "reject":
                raise VesselTrackBuildError("duplicate AIS vessel timestamp")
            if self._config.duplicate_policy == "keep_first":
                by_time.setdefault(point.observed_at, point)
            else:
                by_time[point.observed_at] = point
        return [by_time[key] for key in sorted(by_time)]


def create_vessel_track_builder(config: ComponentConfig) -> VesselTrackBuilder:
    return VesselTrackBuilder(VesselTrackBuilderConfig.model_validate(config.settings))
