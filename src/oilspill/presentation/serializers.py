"""Pure adapters from stable domain results to dashboard-facing view models."""

from __future__ import annotations

from oilspill.domain.ais import CandidateVessel
from oilspill.domain.attribution import AttributionResult, EvidenceValue
from oilspill.domain.common import ComponentMetadata
from oilspill.domain.drift import DriftMode, ParticleDistribution
from oilspill.domain.forecast import CoastalImpactResult, ForecastResult
from oilspill.domain.geospatial import RasterAsset, SpatialGeometry
from oilspill.domain.pipeline import PipelineResult
from oilspill.domain.spill import SpillObservation
from oilspill.presentation.models import (
    AttributionEvidenceView,
    AttributionSummaryView,
    CoastalImpactLayerView,
    CoastalImpactSegmentView,
    ComponentView,
    DashboardViewModel,
    DischargeTimeWindowView,
    ForecastLayerView,
    HybridForecastSummaryView,
    HybridHindcastView,
    HybridScenarioWeightView,
    LayerGeometry,
    LineStringLayerGeometry,
    MeasurementView,
    MultiPolygonLayerGeometry,
    ParticleDistributionView,
    PipelineProvenanceView,
    PointLayerGeometry,
    PolygonLayerGeometry,
    RankedVesselView,
    RasterBandView,
    RasterLayerView,
    ReleaseTimeQuantilesView,
    SlickPolygonView,
    SourceLocationView,
    SpatialProbabilityView,
    SpillDetectionView,
    SpillGeometryStatisticsView,
)


def geometry_view(value: SpatialGeometry) -> LayerGeometry:
    """Convert canonical geometry to coordinate arrays while preserving its CRS."""

    geometry = value.geometry
    if geometry.type == "Point":
        return PointLayerGeometry(
            coordinates=(geometry.coordinate.x, geometry.coordinate.y), crs=value.crs
        )
    if geometry.type == "LineString":
        return LineStringLayerGeometry(
            coordinates=tuple((item.x, item.y) for item in geometry.coordinates),
            crs=value.crs,
        )
    if geometry.type == "Polygon":
        rings = (geometry.exterior, *geometry.holes)
        return PolygonLayerGeometry(
            coordinates=tuple(tuple((item.x, item.y) for item in ring) for ring in rings),
            crs=value.crs,
        )
    return MultiPolygonLayerGeometry(
        coordinates=tuple(
            tuple(
                tuple((item.x, item.y) for item in ring)
                for ring in (polygon.exterior, *polygon.holes)
            )
            for polygon in geometry.polygons
        ),
        crs=value.crs,
    )


def raster_layer_view(
    value: RasterAsset,
    *,
    layer_id: str,
    layer_kind: str,
    semantics: str,
    calibrated_probability: bool | None = None,
) -> RasterLayerView:
    return RasterLayerView(
        layer_id=layer_id,
        layer_kind=layer_kind,
        artifact=value.artifact,
        width=value.grid.width,
        height=value.grid.height,
        crs=value.grid.crs,
        transform=value.grid.transform,
        bands=tuple(
            RasterBandView(name=band.name, unit=band.unit, nodata=band.nodata)
            for band in value.bands
        ),
        semantics=semantics,
        calibrated_probability=calibrated_probability,
    )


def _component(value: ComponentMetadata) -> ComponentView:
    return ComponentView(
        name=value.name,
        version=value.version,
        implementation=value.implementation,
        framework=value.framework,
        configuration_sha256=value.configuration_sha256,
        attributes=value.attributes,
    )


def spill_detection_view(observation: SpillObservation) -> SpillDetectionView:
    detection = observation.detection
    probability = detection.probability_raster
    return SpillDetectionView(
        observation_id=observation.observation_id,
        detection_id=detection.detection_id,
        scene_id=detection.scene_id,
        observed_at=detection.observed_at,
        class_label=detection.class_label,
        class_id=detection.class_id,
        confidence=detection.confidence,
        footprint=geometry_view(detection.footprint),
        mask=raster_layer_view(
            detection.mask,
            layer_id=f"{detection.detection_id}:mask",
            layer_kind="spill_segmentation_mask",
            semantics="detector class mask; slick geometry is represented separately",
        ),
        probability=(
            raster_layer_view(
                probability,
                layer_id=f"{detection.detection_id}:probability",
                layer_kind="spill_probability",
                semantics="detector probability output",
                calibrated_probability=None,
            )
            if probability is not None
            else None
        ),
        detector=_component(detection.model),
        quality_flags=detection.quality_flags,
    )


def slick_polygon_view(observation: SpillObservation) -> SlickPolygonView:
    geometry = observation.geometry
    polygon = geometry_view(geometry.polygon)
    centroid = geometry_view(geometry.centroid)
    centerline = geometry_view(geometry.centerline) if geometry.centerline is not None else None
    if not isinstance(centroid, PointLayerGeometry):
        raise ValueError("canonical slick centroid must be a point")
    if not isinstance(polygon, (PolygonLayerGeometry, MultiPolygonLayerGeometry)):
        raise ValueError("canonical slick polygon must be polygonal")
    if centerline is not None and not isinstance(centerline, LineStringLayerGeometry):
        raise ValueError("canonical slick centerline must be a line string")
    return SlickPolygonView(
        observation_id=observation.observation_id,
        geometry_id=geometry.geometry_id,
        detection_id=geometry.detection_id,
        polygon=polygon,
        centroid=centroid,
        centerline=centerline,
        measurement_crs=geometry.measurement_crs,
        quality_flags=geometry.quality_flags,
    )


def spill_geometry_statistics_view(
    observation: SpillObservation,
) -> SpillGeometryStatisticsView:
    geometry = observation.geometry
    return SpillGeometryStatisticsView(
        observation_id=observation.observation_id,
        geometry_id=geometry.geometry_id,
        area=(
            MeasurementView(value=geometry.area.value, unit=geometry.area.unit.value)
            if geometry.area is not None
            else None
        ),
        length=(
            MeasurementView(value=geometry.length.value, unit=geometry.length.unit.value)
            if geometry.length is not None
            else None
        ),
        width=(
            MeasurementView(value=geometry.width.value, unit=geometry.width.unit.value)
            if geometry.width is not None
            else None
        ),
        dominant_orientation=(
            MeasurementView(value=geometry.orientation.value, unit=geometry.orientation.unit.value)
            if geometry.orientation is not None
            else None
        ),
        orientation_convention=(
            geometry.orientation.convention if geometry.orientation is not None else None
        ),
        measurement_crs=geometry.measurement_crs,
        method=_component(geometry.method),
    )


def discharge_time_window_view(observation: SpillObservation) -> DischargeTimeWindowView:
    estimate = observation.discharge_time
    return DischargeTimeWindowView(
        observation_id=observation.observation_id,
        earliest=estimate.interval.start,
        latest=estimate.interval.end,
        central_estimate=estimate.central_estimate,
        confidence=estimate.confidence,
        assumptions=estimate.assumptions,
        evidence=tuple(item.description for item in estimate.evidence),
        estimator=_component(estimate.method),
        quality_flags=estimate.quality_flags,
    )


def _evidence_view(
    attribution: AttributionResult,
    candidate_id: str,
    vessel_id: str | None,
    evidence: EvidenceValue,
) -> AttributionEvidenceView:
    return AttributionEvidenceView(
        attribution_id=attribution.attribution_id,
        observation_id=attribution.observation_id,
        candidate_id=candidate_id,
        vessel_id=vessel_id,
        name=evidence.name,
        status=evidence.status,
        raw_value=evidence.value,
        normalized_value=evidence.normalized_value,
        unit=evidence.unit,
        direction=evidence.direction,
        configured_weight=evidence.configured_weight,
        normalized_weight=evidence.normalized_weight,
        weighted_contribution=evidence.weighted_contribution,
        explanation=evidence.explanation,
        provenance=evidence.provenance,
        source_simulation_ids=evidence.source_simulation_ids,
    )


def attribution_evidence_views(result: PipelineResult) -> tuple[AttributionEvidenceView, ...]:
    return tuple(
        _evidence_view(attribution, ranked.candidate_id, ranked.vessel_id, evidence)
        for attribution in result.attributions
        for ranked in sorted(attribution.ranked_candidates, key=lambda item: item.rank)
        for evidence in ranked.evidence
    )


def ranked_vessel_views(result: PipelineResult) -> tuple[RankedVesselView, ...]:
    candidates: dict[str, CandidateVessel] = {
        candidate.candidate_id: candidate for candidate in result.candidates
    }
    views: list[RankedVesselView] = []
    for attribution in result.attributions:
        for ranked in sorted(attribution.ranked_candidates, key=lambda item: item.rank):
            candidate = candidates.get(ranked.candidate_id)
            trajectory = geometry_view(candidate.track.path) if candidate is not None else None
            if trajectory is not None and not isinstance(trajectory, LineStringLayerGeometry):
                raise ValueError("canonical vessel trajectory must be a line string")
            evidence = tuple(
                _evidence_view(attribution, ranked.candidate_id, ranked.vessel_id, evidence_value)
                for evidence_value in ranked.evidence
            )
            views.append(
                RankedVesselView(
                    decision=attribution.decision,
                    abstention_reason=attribution.abstention_reason,
                    attribution_id=attribution.attribution_id,
                    observation_id=attribution.observation_id,
                    candidate_id=ranked.candidate_id,
                    vessel_id=ranked.vessel_id,
                    rank=ranked.rank,
                    aggregate_score=ranked.score,
                    calibrated_probability=ranked.calibrated_probability,
                    calibration_status=attribution.calibration_status,
                    trajectory=trajectory,
                    screening_decision=candidate.decision if candidate is not None else None,
                    screening_reasons=candidate.reasons if candidate is not None else (),
                    evidence=evidence,
                    disclaimer=_attribution_disclaimer(attribution),
                )
            )
    return tuple(views)


def _attribution_disclaimer(attribution: AttributionResult) -> str:
    warning = next(
        (
            item.message
            for item in attribution.warnings
            if "not proof of discharge" in item.message.lower()
        ),
        None,
    )
    return warning or (
        "Ranked vessel candidate is investigative prioritization, not proof of discharge."
    )


def attribution_summary_views(
    result: PipelineResult,
) -> tuple[AttributionSummaryView, ...]:
    return tuple(
        AttributionSummaryView(
            attribution_id=attribution.attribution_id,
            observation_id=attribution.observation_id,
            decision=attribution.decision,
            ranked_candidate_ids=tuple(
                item.candidate_id
                for item in sorted(attribution.ranked_candidates, key=lambda item: item.rank)
            ),
            ranked_vessel_ids=tuple(
                item.vessel_id
                for item in sorted(attribution.ranked_candidates, key=lambda item: item.rank)
                if item.vessel_id is not None
            ),
            abstention_reason=attribution.abstention_reason,
            calibrated=attribution.calibration_status == "calibrated",
            disclaimer=_attribution_disclaimer(attribution),
        )
        for attribution in result.attributions
    )


def hybrid_hindcast_views(result: PipelineResult) -> tuple[HybridHindcastView, ...]:
    views = []
    for hindcast in result.hybrid_hindcasts:
        posterior = hindcast.source_posterior
        regions = tuple(
            geometry_view(region) if region is not None else None
            for region in (
                posterior.credible_region_50,
                posterior.credible_region_90,
                posterior.credible_region_95,
            )
        )
        if any(
            region is not None
            and not isinstance(region, (PolygonLayerGeometry, MultiPolygonLayerGeometry))
            for region in regions
        ):
            raise ValueError("source posterior credible regions must be polygonal")
        quantiles = posterior.release_time_quantiles
        views.append(
            HybridHindcastView(
                hindcast_id=hindcast.hindcast_id,
                observation_id=hindcast.observation_id,
                source_posterior_raster_uri=(
                    posterior.source_probability_raster.artifact.uri
                    if posterior.source_probability_raster is not None
                    else None
                ),
                credible_region_50=regions[0],
                credible_region_90=regions[1],
                credible_region_95=regions[2],
                release_time=(
                    ReleaseTimeQuantilesView(
                        p10=quantiles.p10,
                        p50=quantiles.p50,
                        p90=quantiles.p90,
                    )
                    if quantiles is not None
                    else None
                ),
                convergence_status=hindcast.convergence_diagnostics.status,
                initial_candidate_count=hindcast.initial_candidate_count,
                retained_eulerian_count=hindcast.retained_eulerian_count,
                high_fidelity_count=hindcast.high_fidelity_count,
                calibrated=posterior.calibrated,
                semantics=posterior.semantics,
            )
        )
    return tuple(views)


def hybrid_forecast_summary_views(
    result: PipelineResult,
) -> tuple[HybridForecastSummaryView, ...]:
    return tuple(
        HybridForecastSummaryView(
            forecast_id=forecast.forecast_id,
            observation_id=forecast.observation_id,
            horizons=tuple(item.valid_at for item in forecast.horizons),
            scenario_weights=tuple(
                HybridScenarioWeightView(
                    scenario_id=item.scenario_id,
                    cluster_id=item.cluster_id,
                    weight=item.normalized_weight,
                    design_weight=item.design_weight,
                    selection_reason=item.selection_reason,
                )
                for item in diagnostics.scenario_weights
            ),
            coarse_scenario_count=diagnostics.eulerian_screened_count,
            high_fidelity_scenario_count=diagnostics.high_fidelity_scenario_count,
            convergence_status=diagnostics.convergence_status,
            calibrated=forecast.uncertainty.calibrated,
            semantics=diagnostics.semantics,
            warnings=forecast.warnings,
        )
        for forecast in result.forecasts
        if (diagnostics := forecast.hybrid_diagnostics) is not None
    )


def pipeline_provenance_view(result: PipelineResult) -> PipelineProvenanceView:
    forcing_artifacts = {
        artifact.sha256: artifact
        for simulation in result.simulations
        for artifact in simulation.forcing_artifacts
    }
    return PipelineProvenanceView(
        random_seed=result.random_seed,
        configuration_sha256=result.configuration_sha256,
        configuration_artifact=result.provenance,
        forcing_ids=tuple(
            sorted(
                {
                    forcing_id
                    for simulation in result.simulations
                    for forcing_id in simulation.forcing_ids
                }
            )
        ),
        forcing_artifacts=tuple(
            forcing_artifacts[key] for key in sorted(forcing_artifacts)
        ),
        forcing_datasets=tuple(
            {
                field.dataset.model_dump_json(): field.dataset
                for field in result.environmental_forcing
            }[key]
            for key in sorted(
                {
                    field.dataset.model_dump_json()
                    for field in result.environmental_forcing
                }
            )
        ),
        components=tuple(_component(item) for item in result.component_metadata),
    )


def estimated_source_location_views(result: PipelineResult) -> tuple[SourceLocationView, ...]:
    """Expose only explicit backward target distributions as plausible source regions."""

    candidate_observations = {
        candidate.candidate_id: candidate.observation_id for candidate in result.candidates
    }
    views: list[SourceLocationView] = []
    for simulation in result.simulations:
        distribution = simulation.target_distribution
        if simulation.mode != DriftMode.BACKWARD_TRACE or distribution is None:
            continue
        candidate_id = simulation.release.source_candidate_id
        views.append(
            SourceLocationView(
                simulation_id=simulation.simulation_id,
                observation_id=candidate_observations.get(candidate_id) if candidate_id else None,
                candidate_id=candidate_id,
                plausible_at=distribution.timestamp,
                discharge_window_start=simulation.release.interval.start,
                discharge_window_end=simulation.release.interval.end,
                geometry=geometry_view(distribution.comparison_geometry),
                particle_positions=distribution.positions,
                particle_count=distribution.particle_count,
                model=_component(simulation.model),
                assumptions=simulation.release.assumptions,
                caveat=(
                    "Model-derived plausible source region; backward simulation is not a "
                    "physically exact reconstruction."
                ),
            )
        )
    return tuple(views)


def _particle_view(value: ParticleDistribution) -> ParticleDistributionView:
    return ParticleDistributionView(
        timestamp=value.timestamp,
        positions=value.positions,
        particle_count=value.particle_count,
        support_geometry=geometry_view(value.comparison_geometry),
    )


def forecast_layer_views(forecast: ForecastResult) -> tuple[ForecastLayerView, ...]:
    views: list[ForecastLayerView] = []
    for horizon in forecast.horizons:
        probability = horizon.spatial_probability
        probability_view = (
            SpatialProbabilityView(
                method=probability.method,
                member_support_geometries=tuple(
                    geometry_view(item) for item in probability.member_geometries
                ),
                probability_per_member=probability.probability_per_member,
                member_weights=probability.member_weights,
                probability_raster=(
                    raster_layer_view(
                        probability.probability_raster,
                        layer_id=f"{forecast.forecast_id}:{horizon.valid_at.isoformat()}:probability",
                        layer_kind="forecast_probability",
                        semantics=probability.explanation,
                        calibrated_probability=probability.calibrated,
                    )
                    if probability.probability_raster is not None
                    else None
                ),
                calibrated=probability.calibrated,
                explanation=probability.explanation,
            )
            if probability is not None
            else None
        )
        occupancy = horizon.occupancy_or_probability
        views.append(
            ForecastLayerView(
                forecast_id=forecast.forecast_id,
                observation_id=forecast.observation_id,
                issued_at=forecast.issued_at,
                valid_at=horizon.valid_at,
                member_simulation_ids=horizon.member_simulation_ids,
                particle_ensemble=tuple(_particle_view(item) for item in horizon.particle_ensemble),
                probability=probability_view,
                occupancy_or_probability=(
                    raster_layer_view(
                        occupancy,
                        layer_id=f"{forecast.forecast_id}:{horizon.valid_at.isoformat()}:occupancy",
                        layer_kind="forecast_occupancy_or_probability",
                        semantics=(
                            "forecast occupancy or probability raster; interpretation is "
                            "defined by its producing component provenance"
                        ),
                        calibrated_probability=forecast.uncertainty.calibrated,
                    )
                    if occupancy is not None
                    else None
                ),
                uncertainty_method=forecast.uncertainty.method,
                calibrated=forecast.uncertainty.calibrated,
                configured_perturbations=forecast.uncertainty.configured_perturbations,
            )
        )
    return tuple(views)


def coastal_impact_layer_view(impact: CoastalImpactResult) -> CoastalImpactLayerView:
    return CoastalImpactLayerView(
        impact_id=impact.impact_id,
        forecast_id=impact.forecast_id,
        coastline_dataset_name=impact.coastline_dataset.name,
        coastline_dataset_version=impact.coastline_dataset.version,
        contact_method=impact.contact_method,
        created_at=impact.created_at,
        segments=tuple(
            CoastalImpactSegmentView(
                segment_id=segment.segment_id,
                geometry=geometry_view(segment.geometry),
                contact_probability=segment.impact_probability,
                estimated_arrival=segment.estimated_arrival,
                earliest_plausible_arrival=segment.earliest_arrival,
                latest_plausible_arrival=segment.latest_arrival,
                arrival_p10=segment.arrival_p10,
                arrival_p50=segment.arrival_p50,
                arrival_p90=segment.arrival_p90,
                contacting_member_ids=segment.contacting_member_ids,
                ensemble_hit_count=segment.ensemble_hit_count,
                ensemble_member_count=segment.ensemble_member_count,
                provenance=segment.provenance,
            )
            for segment in impact.segments
        ),
        uncertainty_method=impact.uncertainty.method,
        calibrated=impact.uncertainty.calibrated,
        quality_flags=impact.warnings,
    )


def dashboard_view_model(result: PipelineResult) -> DashboardViewModel:
    """Build the complete presentation contract using only canonical pipeline output."""

    evidence = attribution_evidence_views(result)
    return DashboardViewModel(
        run_id=result.run_id,
        configuration_sha256=result.configuration_sha256,
        provenance=result.provenance,
        spill_detections=tuple(spill_detection_view(item) for item in result.observations),
        slick_polygons=tuple(slick_polygon_view(item) for item in result.observations),
        spill_geometry_statistics=tuple(
            spill_geometry_statistics_view(item) for item in result.observations
        ),
        discharge_time_windows=tuple(
            discharge_time_window_view(item) for item in result.observations
        ),
        ranked_vessels=ranked_vessel_views(result),
        attribution_evidence=evidence,
        estimated_source_locations=estimated_source_location_views(result),
        forecast_layers=tuple(
            layer for forecast in result.forecasts for layer in forecast_layer_views(forecast)
        ),
        coastal_impact_layers=tuple(
            coastal_impact_layer_view(impact) for impact in result.coastal_impacts
        ),
        pipeline_warnings=result.warnings,
        hybrid_hindcasts=hybrid_hindcast_views(result),
        attribution_summaries=attribution_summary_views(result),
        hybrid_forecast_summaries=hybrid_forecast_summary_views(result),
        execution_provenance=pipeline_provenance_view(result),
    )


def serialize_dashboard(result: PipelineResult, *, indent: int | None = 2) -> str:
    """Serialize a pipeline result to the stable dashboard JSON contract."""

    return dashboard_view_model(result).model_dump_json(indent=indent)
