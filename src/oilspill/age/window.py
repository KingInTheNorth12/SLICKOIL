"""Configured plausible discharge window without an unsupported physical age equation."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256

from oilspill.age.config import (
    CentralEstimatePolicy,
    ConfigurableWindowAgeEstimatorConfig,
    WindowAnchor,
    _duration_seconds,
)
from oilspill.age.errors import SpillAgeEstimationError
from oilspill.config import ComponentConfig
from oilspill.domain.common import (
    ComponentMetadata,
    MetadataEntry,
    QualityFlag,
    QualitySeverity,
    TimeRange,
)
from oilspill.domain.spill import DischargeTimeEstimate, DischargeTimeEvidence
from oilspill.requests import AgeEstimationRequest

HEURISTIC_WARNING = (
    "Configurable time window only; not a scientifically validated spill-age estimate. "
    "DATASET_VALIDATION_REQUIRED"
)


class ConfigurableWindowAgeEstimator:
    """Create a conservative configured interval and preserve all considered evidence.

    Morphology and AIS inputs can be required and are recorded, but do not change the interval.
    No relationship between shape, vessel activity, and slick age is asserted without a sourced
    scientific model.
    """

    def __init__(
        self,
        config: ConfigurableWindowAgeEstimatorConfig,
        *,
        component_name: str = "configurable_window",
    ) -> None:
        self._config = config
        self._component_name = component_name

    def estimate(self, request: AgeEstimationRequest) -> DischargeTimeEstimate:
        self._validate_request(request)
        anchor = self._anchor(request)
        earliest = anchor - timedelta(
            seconds=_duration_seconds(self._config.earliest_before_anchor)
        )
        latest = anchor - timedelta(seconds=_duration_seconds(self._config.latest_before_anchor))
        interval = TimeRange(start=earliest, end=latest)
        central = (
            interval.start + (interval.end - interval.start) / 2
            if self._config.central_estimate_policy == CentralEstimatePolicy.INTERVAL_MIDPOINT
            else None
        )
        evidence = (
            self._sar_evidence(request, anchor),
            self._morphology_evidence(request),
            self._ais_evidence(request),
            self._configuration_evidence(),
        )
        return DischargeTimeEstimate(
            interval=interval,
            central_estimate=central,
            method=self.component_metadata(),
            confidence=None,
            assumptions=(*self._config.assumptions, HEURISTIC_WARNING),
            evidence=evidence,
            quality_flags=(
                QualityFlag(
                    code="UNVALIDATED_CONFIGURABLE_AGE_WINDOW",
                    severity=QualitySeverity.WARNING,
                    message=HEURISTIC_WARNING,
                ),
            ),
        )

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name=self._component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="configured-time-window",
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="scientifically_validated", value=False),
                MetadataEntry(
                    key="heuristic_validation", value=self._config.heuristic_validation_note
                ),
                MetadataEntry(key="confidence_supported", value=False),
            ),
        )

    def _validate_request(self, request: AgeEstimationRequest) -> None:
        if request.detection.scene_id != request.scene.scene_id:
            raise SpillAgeEstimationError("detection does not belong to the supplied SAR scene")
        if request.geometry.detection_id != request.detection.detection_id:
            raise SpillAgeEstimationError("geometry does not belong to the supplied detection")
        if not (
            request.scene.acquisition.start
            <= request.detection.observed_at
            <= request.scene.acquisition.end
        ):
            raise SpillAgeEstimationError(
                "detection timestamp falls outside the SAR acquisition interval"
            )
        missing = tuple(
            name
            for name in self._config.required_morphology_fields
            if getattr(request.geometry, name) is None
        )
        if missing:
            raise SpillAgeEstimationError(
                f"configured required morphology fields are unavailable: {missing}"
            )
        if self._config.require_ais_history and not request.vessel_tracks:
            raise SpillAgeEstimationError("configured estimator requires AIS history")

    def _anchor(self, request: AgeEstimationRequest) -> datetime:
        if self._config.anchor == WindowAnchor.ACQUISITION_START:
            return request.scene.acquisition.start
        if self._config.anchor == WindowAnchor.ACQUISITION_END:
            return request.scene.acquisition.end
        if self._config.anchor == WindowAnchor.DETECTION_OBSERVED_AT:
            return request.detection.observed_at
        start = request.scene.acquisition.start
        return start + (request.scene.acquisition.end - start) / 2

    def _sar_evidence(
        self, request: AgeEstimationRequest, anchor: datetime
    ) -> DischargeTimeEvidence:
        return DischargeTimeEvidence(
            kind="sar_acquisition",
            description="configured timestamp anchor from the SAR observation",
            source_ids=(request.scene.scene_id, request.detection.detection_id),
            attributes=(
                MetadataEntry(key="anchor_policy", value=self._config.anchor.value),
                MetadataEntry(key="anchor_utc", value=anchor.isoformat()),
                MetadataEntry(key="used_to_construct_window", value=True),
            ),
        )

    @staticmethod
    def _morphology_evidence(request: AgeEstimationRequest) -> DischargeTimeEvidence:
        geometry = request.geometry
        available = tuple(
            name
            for name in ("area", "length", "width", "orientation")
            if getattr(geometry, name) is not None
        )
        return DischargeTimeEvidence(
            kind="slick_morphology",
            description="available morphology recorded without an unsourced age relationship",
            source_ids=(geometry.geometry_id,),
            attributes=(
                MetadataEntry(key="available_fields", value=",".join(available) or "none"),
                MetadataEntry(key="used_to_adjust_window", value=False),
            ),
        )

    @staticmethod
    def _ais_evidence(request: AgeEstimationRequest) -> DischargeTimeEvidence:
        return DischargeTimeEvidence(
            kind="ais_history",
            description="AIS availability recorded without an unsourced age relationship",
            source_ids=tuple(track.track_id for track in request.vessel_tracks),
            attributes=(
                MetadataEntry(key="track_count", value=len(request.vessel_tracks)),
                MetadataEntry(key="used_to_adjust_window", value=False),
            ),
        )

    def _configuration_evidence(self) -> DischargeTimeEvidence:
        return DischargeTimeEvidence(
            kind="configuration",
            description="unvalidated configured offsets defining the plausible interval",
            attributes=(
                MetadataEntry(
                    key="earliest_before_anchor_seconds",
                    value=_duration_seconds(self._config.earliest_before_anchor),
                ),
                MetadataEntry(
                    key="latest_before_anchor_seconds",
                    value=_duration_seconds(self._config.latest_before_anchor),
                ),
                MetadataEntry(
                    key="central_estimate_policy",
                    value=self._config.central_estimate_policy.value,
                ),
                MetadataEntry(key="scientifically_validated", value=False),
            ),
        )


def create_configurable_window(config: ComponentConfig) -> ConfigurableWindowAgeEstimator:
    settings = ConfigurableWindowAgeEstimatorConfig.model_validate(config.settings)
    return ConfigurableWindowAgeEstimator(settings, component_name=config.name)
