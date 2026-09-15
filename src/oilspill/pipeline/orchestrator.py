"""Thin end-to-end orchestration over stable domain ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from oilspill.bootstrap import create_registries_with_builtins, create_source_inference
from oilspill.config import ConfigurationError, EndToEndPipelineConfig, SourceInferenceConfig
from oilspill.domain.ais import CandidateDecision, CandidateVessel
from oilspill.domain.attribution import AttributionResult
from oilspill.domain.common import (
    ArtifactRef,
    Decision,
    QualityFlag,
    QualitySeverity,
    StageOutcome,
    StageStatus,
    TimeRange,
)
from oilspill.domain.drift import DriftSimulation, ReleaseHypothesis
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.forecast import CoastalImpactResult, ForecastDecision, ForecastResult
from oilspill.domain.hybrid import HybridHindcastResult, HybridHindcastTiming
from oilspill.domain.pipeline import PipelineResult
from oilspill.domain.sar import SARScene
from oilspill.domain.source_trace import BackwardSourceTraceResult
from oilspill.domain.spill import SpillDetection, SpillObservation
from oilspill.hybrid.ais import (
    build_posterior_ais_query,
    posterior_credible_region,
    posterior_release_interval,
)
from oilspill.ports import HybridHindcastTimingProvider
from oilspill.registry import (
    ComponentRegistries,
    HybridForecastDependencies,
    HybridHindcastDependencies,
)
from oilspill.requests import (
    AgeEstimationRequest,
    AISQuery,
    AttributionEvidenceRequest,
    AttributionRequest,
    BackwardTraceRequest,
    CandidateGenerationRequest,
    CoastalImpactRequest,
    CoastlineDataset,
    EnsembleForecastRequest,
    EnvironmentalQuery,
    HybridHindcastRequest,
    SARReadRequest,
    SourceSearchRequest,
)
from oilspill.source_tracing.backward import BackwardSourceTracer


@dataclass(frozen=True)
class _HybridOutputs:
    candidates: tuple[CandidateVessel, ...]
    simulations: tuple[DriftSimulation, ...]
    attribution: AttributionResult
    forecast: ForecastResult
    impact: CoastalImpactResult | None
    hindcast: HybridHindcastResult
    artifacts: tuple[ArtifactRef, ...]
    warnings: tuple[QualityFlag, ...]
    hindcast_timing: HybridHindcastTiming | None


@dataclass(frozen=True)
class _HybridAttributionOutputs:
    candidates: tuple[CandidateVessel, ...]
    attribution: AttributionResult
    warnings: tuple[QualityFlag, ...]


class PipelineOrchestrator:
    """Coordinate configured components without implementing their scientific methods."""

    def __init__(
        self,
        config: EndToEndPipelineConfig,
        registries: ComponentRegistries | None = None,
    ) -> None:
        self._config = config
        self._hybrid_mode = config.execution.pipeline_mode == "hybrid_mvp"
        if not self._hybrid_mode and config.source_inference is None:
            raise ConfigurationError(
                "pipeline requires explicit source_inference configuration; "
                "legacy single-AIS-point fallback is disabled"
            )
        if config.execution.forecast_source_policy != "observed_slick":
            raise ConfigurationError(
                "pipeline forecasting requires forecast_source_policy=observed_slick"
            )
        note = config.execution.forecast_initialization_validation_note
        if not note or "DATASET_VALIDATION_REQUIRED" not in note:
            raise ConfigurationError(
                "observed-slick initialization requires a DATASET_VALIDATION_REQUIRED note"
            )
        self._source_config = config.source_inference
        self._registries = registries or create_registries_with_builtins()
        self._sar_reader = self._registries.sar_reader.create(
            config.sar_reader.name, config.sar_reader
        )
        self._sar_preprocessor = self._registries.sar_preprocessor.create(
            config.sar_preprocessor.name, config.sar_preprocessor
        )
        self._detector = self._registries.detector.create(config.detector.name, config.detector)
        self._characterizer = self._registries.spill_characterizer.create(
            config.spill_characterizer.name, config.spill_characterizer
        )
        self._age_estimator = self._registries.spill_age_estimator.create(
            config.spill_age_estimator.name, config.spill_age_estimator
        )
        self._ais_provider = self._registries.ais_provider.create(
            config.ais_provider.name, config.ais_provider
        )
        self._track_reconstructor = self._registries.vessel_track_reconstructor.create(
            config.vessel_track_reconstructor.name, config.vessel_track_reconstructor
        )
        self._candidate_generator = self._registries.candidate_generator.create(
            config.candidate_generator.name, config.candidate_generator
        )
        self._environmental_provider = self._registries.environmental_provider.create(
            config.environmental_provider.name, config.environmental_provider
        )
        self._drift_model = self._registries.drift.create(config.drift.name, config.drift)
        self._source_inference = None
        self._hybrid_hindcast = None
        self._hybrid_eulerian = None
        if self._hybrid_mode:
            assert config.eulerian_transport is not None
            assert config.hybrid_coarse_hindcast is not None
            assert config.hybrid_hindcast is not None
            eulerian = self._registries.eulerian_transport.create(
                config.eulerian_transport.name, config.eulerian_transport
            )
            self._hybrid_eulerian = eulerian
            coarse = self._registries.coarse_hindcast.create(
                config.hybrid_coarse_hindcast.name,
                config.hybrid_coarse_hindcast,
                eulerian,
            )
            dependencies = HybridHindcastDependencies(
                eulerian=eulerian,
                drift=self._drift_model,
                coarse=coarse,
                resolved_components=(
                    config.eulerian_transport,
                    config.hybrid_coarse_hindcast,
                    config.drift,
                ),
            )
            self._hybrid_hindcast = self._registries.hybrid_hindcast.create(
                config.hybrid_hindcast.name,
                config.hybrid_hindcast,
                dependencies,
            )
        else:
            assert self._source_config is not None
            self._source_inference = create_source_inference(
                SourceInferenceConfig(
                    drift=config.drift,
                    release_enumerator=self._source_config.release_enumerator,
                    observation_likelihood=self._source_config.observation_likelihood,
                    search=self._source_config.search,
                ),
                self._registries,
                drift_model=self._drift_model,
            )
        self._evidence_builder = self._registries.attribution_evidence.create(
            config.attribution_evidence.name, config.attribution_evidence
        )
        self._attribution_model = self._registries.attribution.create(
            config.attribution.name, config.attribution
        )
        if self._hybrid_mode and config.hybrid_forecast is not None:
            assert self._hybrid_eulerian is not None
            self._forecaster = self._registries.hybrid_ensemble_forecaster.create(
                config.hybrid_forecast.name,
                config.hybrid_forecast,
                HybridForecastDependencies(
                    eulerian=self._hybrid_eulerian,
                    drift=self._drift_model,
                    resolved_components=(config.eulerian_transport, config.drift),  # type: ignore[arg-type]
                ),
            )
        else:
            self._forecaster = self._registries.ensemble_forecaster.create(
                config.ensemble_forecaster.name,
                config.ensemble_forecaster,
                self._drift_model,
            )
        self._coastline_provider = self._registries.coastline_provider.create(
            config.coastline_provider.name, config.coastline_provider
        )
        self._coastal_analyzer = self._registries.coastal_impact.create(
            config.coastal_impact.name, config.coastal_impact
        )
        self._backward_tracer = BackwardSourceTracer(self._drift_model)

    def run(self) -> PipelineResult:
        """Dispatch to the configured execution mode without embedding scientific work."""
        return self._run_hybrid_pipeline() if self._hybrid_mode else self._run_legacy_pipeline()

    def _run_legacy_pipeline(self) -> PipelineResult:
        """Execute and serialize one complete configured run."""

        stages: list[StageOutcome] = []
        source = self._config.execution.sar_source

        started = datetime.now(UTC)
        raw_scene = self._sar_reader.read(
            SARReadRequest(source=source, scene_id=self._config.execution.scene_id)
        )
        self._record(stages, "load_sar", started, (source.uri,), (raw_scene.scene_id,))

        started = datetime.now(UTC)
        scene = self._sar_preprocessor.preprocess(raw_scene)
        self._record(stages, "preprocess_sar", started, (raw_scene.scene_id,), (scene.scene_id,))

        started = datetime.now(UTC)
        self._detector.load()
        detections = self._detector.predict(scene)
        self._record(
            stages,
            "detect_spill",
            started,
            (scene.scene_id,),
            tuple(item.detection_id for item in detections),
        )

        started = datetime.now(UTC)
        coastline = self._coastline_provider.load()
        self._record(
            stages,
            "load_coastline",
            started,
            (),
            (coastline.dataset_id,),
        )
        observations: list[SpillObservation] = []
        candidates_all: list[CandidateVessel] = []
        simulations: list[DriftSimulation] = []
        attributions = []
        forecasts = []
        impacts = []
        source_inferences = []
        hybrid_hindcasts: list[HybridHindcastResult] = []
        source_artifacts: list[ArtifactRef] = []
        warnings: list[QualityFlag] = []
        environmental_forcing: list[EnvironmentalField] = []

        for detection in detections:
            started = datetime.now(UTC)
            geometry = self._characterizer.characterize(detection)
            self._record(
                stages,
                "characterize_slick",
                started,
                (detection.detection_id,),
                (geometry.geometry_id,),
            )

            started = datetime.now(UTC)
            discharge = self._age_estimator.estimate(
                AgeEstimationRequest(
                    scene=scene,
                    detection=detection,
                    geometry=geometry,
                    vessel_tracks=(),
                )
            )
            observation = SpillObservation(
                observation_id=f"observation:{detection.detection_id}",
                scene=scene,
                observed_at=detection.observed_at,
                detection=detection,
                geometry=geometry,
                discharge_time=discharge,
            )
            observations.append(observation)
            self._record(
                stages,
                "estimate_discharge_time",
                started,
                (geometry.geometry_id,),
                (observation.observation_id,),
            )

            started = datetime.now(UTC)
            points = self._ais_provider.fetch_points(
                AISQuery(area=scene.footprint, interval=discharge.interval)
            )
            tracks = self._track_reconstructor.build(points)
            self._record(
                stages,
                "load_reconstruct_ais",
                started,
                (observation.observation_id,),
                tuple(item.track_id for item in tracks),
            )

            started = datetime.now(UTC)
            candidates = self._candidate_generator.generate(
                CandidateGenerationRequest(observation=observation, tracks=tracks)
            )
            candidates_all.extend(candidates)
            retained = tuple(
                item for item in candidates if item.decision == CandidateDecision.RETAINED
            )
            self._record(
                stages,
                "generate_candidates",
                started,
                tuple(item.track_id for item in tracks),
                tuple(item.candidate_id for item in candidates),
            )

            started = datetime.now(UTC)
            forcing = self._environmental_provider.fetch(
                EnvironmentalQuery(
                    area=scene.footprint,
                    interval=self._config.execution.environmental_query_interval,
                    variables=self._config.execution.environmental_variables,
                )
            )
            environmental_forcing.extend(forcing)
            self._record(
                stages,
                "load_environmental_forcing",
                started,
                (observation.observation_id,),
                tuple(item.field_id for item in forcing),
            )

            backward_traces = self._optional_backward(observation, forcing, stages, warnings)
            simulations.extend(trace.simulation for trace in backward_traces)
            started = datetime.now(UTC)
            assert self._source_inference is not None
            source_inference = self._source_inference.search(
                SourceSearchRequest(
                    observation=observation,
                    candidates=retained,
                    forcing=forcing,
                    random_seed=self._config.execution.random_seed,
                    backward_evidence=backward_traces,
                )
            )
            source_inferences.append(source_inference)
            source_artifacts.extend((source_inference.provenance, *source_inference.iterations))
            forward_traces = tuple(
                e.trace for e in source_inference.evaluated_hypotheses if e.trace is not None
            )
            simulations.extend(trace.simulation for trace in forward_traces)
            self._record(
                stages,
                "source_inference",
                started,
                (observation.observation_id, *(c.candidate_id for c in retained)),
                (source_inference.inference_id,),
                status=StageStatus.ABSTAINED
                if source_inference.decision == Decision.ABSTAINED
                else StageStatus.SUCCEEDED,
            )
            metric_evidence = []
            for candidate in retained:
                started = datetime.now(UTC)
                evidence = self._evidence_builder.build(
                    AttributionEvidenceRequest(
                        observation=observation,
                        candidate=candidate,
                        source_inference=source_inference,
                    )
                )
                metric_evidence.append(evidence)
                self._record(
                    stages,
                    "build_attribution_evidence",
                    started,
                    (source_inference.inference_id, candidate.candidate_id),
                    (evidence.candidate_id,),
                )

            started = datetime.now(UTC)
            attribution = self._attribution_model.rank(
                AttributionRequest(
                    observation=observation,
                    candidates=retained,
                    simulations=tuple(item.simulation for item in forward_traces)
                    + tuple(item.simulation for item in backward_traces),
                    candidate_metrics=tuple(metric_evidence),
                    forward_traces=tuple(forward_traces),
                    backward_traces=tuple(backward_traces),
                )
            )
            attributions.append(attribution)
            self._record(
                stages,
                "vessel_attribution",
                started,
                tuple(item.candidate_id for item in retained),
                (attribution.attribution_id,),
                status=StageStatus.ABSTAINED
                if attribution.decision == Decision.ABSTAINED
                else StageStatus.SUCCEEDED,
            )

            # Forecast initial condition comes from the observed slick, independently of
            # source-inference and attribution decisions. This is not a historical release.
            release = ReleaseHypothesis(
                release_id=f"forecast-initial-condition:{observation.observation_id}",
                geometry=observation.geometry.centroid,
                interval=TimeRange(start=observation.observed_at, end=observation.observed_at),
                assumptions=(
                    "Observed-slick forecast initialization, not inferred discharge.",
                    self._config.execution.forecast_initialization_validation_note or "",
                ),
            )
            started = datetime.now(UTC)
            forecast = self._forecaster.forecast(
                EnsembleForecastRequest(
                    observation=observation,
                    release=release,
                    forcing=forcing,
                    random_seed=self._config.execution.random_seed,
                )
            )
            forecasts.append(forecast)
            simulations.extend(forecast.members)
            self._record(
                stages,
                "ensemble_forecast",
                started,
                (observation.observation_id,),
                (forecast.forecast_id,),
            )

            started = datetime.now(UTC)
            impact = self._coastal_analyzer.analyze(
                CoastalImpactRequest(forecast=forecast, coastline=coastline)
            )
            impacts.append(impact)
            self._record(
                stages,
                "coastal_impact",
                started,
                (forecast.forecast_id,),
                (impact.impact_id,),
            )

        provenance = self._write_configuration_provenance()
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        run_key = sha256(f"{source.sha256}:{config_hash}".encode()).hexdigest()[:16]
        result = PipelineResult(
            run_id=f"pipeline:{run_key}",
            observations=tuple(observations),
            candidates=tuple(candidates_all),
            simulations=tuple(simulations),
            attributions=tuple(attributions),
            forecasts=tuple(forecasts),
            coastal_impacts=tuple(impacts),
            stages=tuple(stages),
            configuration_sha256=config_hash,
            provenance=provenance,
            artifacts=(source, coastline.source, *source_artifacts),
            source_inferences=tuple(source_inferences),
            hybrid_hindcasts=tuple(hybrid_hindcasts),
            warnings=tuple(warnings),
            random_seed=self._config.execution.random_seed,
            component_metadata=(
                self._drift_model.component_metadata(),
                self._forecaster.component_metadata(),
                self._coastal_analyzer.component_metadata(),
            ),
            environmental_forcing=_unique_environmental_fields(environmental_forcing),
        )
        self.serialize(result, self._config.execution.output_path)
        return result

    def _run_hybrid_pipeline(self) -> PipelineResult:
        stages: list[StageOutcome] = []
        source = self._config.execution.sar_source
        started = datetime.now(UTC)
        raw_scene = self._sar_reader.read(
            SARReadRequest(source=source, scene_id=self._config.execution.scene_id)
        )
        self._record(stages, "load_sar", started, (source.uri,), (raw_scene.scene_id,))
        started = datetime.now(UTC)
        scene = self._sar_preprocessor.preprocess(raw_scene)
        self._record(
            stages, "preprocess_sar", started, (raw_scene.scene_id,), (scene.scene_id,)
        )
        started = datetime.now(UTC)
        self._detector.load()
        detections = self._detector.predict(scene)
        self._record(
            stages,
            "detect_spill",
            started,
            (scene.scene_id,),
            tuple(item.detection_id for item in detections),
        )
        observations = [
            self._build_observation(scene, detection, stages) for detection in detections
        ]
        started = datetime.now(UTC)
        coastline = self._coastline_provider.load()
        self._record(stages, "load_coastline", started, (), (coastline.dataset_id,))

        candidates: list[CandidateVessel] = []
        simulations: list[DriftSimulation] = []
        attributions: list[AttributionResult] = []
        forecasts: list[ForecastResult] = []
        impacts: list[CoastalImpactResult] = []
        hindcasts: list[HybridHindcastResult] = []
        artifacts: list[ArtifactRef] = []
        warnings: list[QualityFlag] = []
        hindcast_timings: list[HybridHindcastTiming] = []
        environmental_forcing: list[EnvironmentalField] = []
        for observation in observations:
            forcing = self._load_environment(observation, stages)
            environmental_forcing.extend(forcing)
            output = self._run_hybrid_observation(
                observation, forcing, coastline, stages
            )
            candidates.extend(output.candidates)
            simulations.extend(output.simulations)
            attributions.append(output.attribution)
            forecasts.append(output.forecast)
            if output.impact is not None:
                impacts.append(output.impact)
            hindcasts.append(output.hindcast)
            artifacts.extend(output.artifacts)
            warnings.extend(output.warnings)
            if output.hindcast_timing is not None:
                hindcast_timings.append(output.hindcast_timing)

        provenance = self._write_configuration_provenance()
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        run_key = sha256(f"{source.sha256}:{config_hash}".encode()).hexdigest()[:16]
        assert self._hybrid_eulerian is not None
        assert self._hybrid_hindcast is not None
        result = PipelineResult(
            run_id=f"pipeline:{run_key}",
            observations=tuple(observations),
            candidates=tuple(candidates),
            simulations=tuple(simulations),
            attributions=tuple(attributions),
            forecasts=tuple(forecasts),
            coastal_impacts=tuple(impacts),
            stages=tuple(stages),
            configuration_sha256=config_hash,
            provenance=provenance,
            artifacts=(source, coastline.source, *artifacts),
            source_inferences=(),
            hybrid_hindcasts=tuple(hindcasts),
            warnings=tuple(warnings),
            random_seed=self._config.execution.random_seed,
            component_metadata=(
                self._hybrid_eulerian.component_metadata(),
                self._hybrid_hindcast.component_metadata(),
                self._drift_model.component_metadata(),
                self._forecaster.component_metadata(),
                self._coastal_analyzer.component_metadata(),
            ),
            hybrid_hindcast_timings=tuple(hindcast_timings),
            environmental_forcing=_unique_environmental_fields(environmental_forcing),
        )
        self.serialize(result, self._config.execution.output_path)
        return result

    def _build_observation(
        self,
        scene: SARScene,
        detection: SpillDetection,
        stages: list[StageOutcome],
    ) -> SpillObservation:
        started = datetime.now(UTC)
        geometry = self._characterizer.characterize(detection)
        self._record(
            stages,
            "characterize_slick",
            started,
            (detection.detection_id,),
            (geometry.geometry_id,),
        )
        started = datetime.now(UTC)
        discharge = self._age_estimator.estimate(
            AgeEstimationRequest(
                scene=scene,
                detection=detection,
                geometry=geometry,
                vessel_tracks=(),
            )
        )
        observation = SpillObservation(
            observation_id=f"observation:{detection.detection_id}",
            scene=scene,
            observed_at=detection.observed_at,
            detection=detection,
            geometry=geometry,
            discharge_time=discharge,
        )
        self._record(
            stages,
            "build_spill_observation",
            started,
            (geometry.geometry_id,),
            (observation.observation_id,),
        )
        return observation

    def _load_environment(
        self, observation: SpillObservation, stages: list[StageOutcome]
    ) -> tuple[EnvironmentalField, ...]:
        started = datetime.now(UTC)
        forcing = self._environmental_provider.fetch(
            EnvironmentalQuery(
                area=observation.scene.footprint,
                interval=self._config.execution.environmental_query_interval,
                variables=self._config.execution.environmental_variables,
            )
        )
        self._record(
            stages,
            "load_environmental_forcing",
            started,
            (observation.observation_id,),
            tuple(item.field_id for item in forcing),
        )
        return forcing

    def _run_hybrid_observation(
        self,
        observation: SpillObservation,
        forcing: tuple[EnvironmentalField, ...],
        coastline: CoastlineDataset,
        stages: list[StageOutcome],
    ) -> _HybridOutputs:
        """Run posterior-first attribution; AIS access occurs only after hindcast returns."""
        hindcast = self._run_hybrid_hindcast(observation, forcing, coastline, stages)
        timing = (
            self._hybrid_hindcast.last_run_timing()
            if isinstance(self._hybrid_hindcast, HybridHindcastTimingProvider)
            else None
        )
        attribution = self._run_hybrid_attribution(observation, hindcast, stages)
        forecast = self._run_hybrid_forecast(observation, forcing, coastline, stages)
        if forecast.decision == ForecastDecision.ABSTAINED:
            self._record(
                stages,
                "hybrid_coastal_impact",
                datetime.now(UTC),
                (forecast.forecast_id,),
                (),
                status=StageStatus.SKIPPED,
                warnings=forecast.warnings,
            )
            impact = None
        else:
            impact = self._run_coastal_impact(forecast, coastline, stages, hybrid=True)
        posterior = hindcast.source_posterior

        posterior_artifacts = [posterior.provenance]
        if posterior.source_probability_raster is not None:
            posterior_artifacts.append(posterior.source_probability_raster.artifact)
        return _HybridOutputs(
            candidates=attribution.candidates,
            simulations=(
                *(item.simulation for item in hindcast.high_fidelity_evaluations),
                *forecast.members,
            ),
            attribution=attribution.attribution,
            forecast=forecast,
            impact=impact,
            hindcast=hindcast,
            artifacts=tuple(posterior_artifacts),
            warnings=(
                *hindcast.warnings,
                *attribution.warnings,
                *forecast.warnings,
                *(impact.warnings if impact is not None else ()),
            ),
            hindcast_timing=timing,
        )

    def _run_hybrid_hindcast(
        self,
        observation: SpillObservation,
        forcing: tuple[EnvironmentalField, ...],
        coastline: CoastlineDataset,
        stages: list[StageOutcome],
    ) -> HybridHindcastResult:
        assert self._hybrid_hindcast is not None
        started = datetime.now(UTC)
        try:
            result = self._hybrid_hindcast.run(
                HybridHindcastRequest(
                    observation=observation,
                    forcing=forcing,
                    random_seed=self._config.execution.random_seed,
                    land_geometries=coastline.segments,
                    land_source=coastline.source,
                )
            )
        except Exception as error:
            self._record_failure(
                stages,
                "hybrid_eulerian_hindcast_screen",
                started,
                (observation.observation_id,),
                error,
            )
            raise
        ended = datetime.now(UTC)
        self._record(
            stages,
            "hybrid_eulerian_hindcast_screen",
            started,
            (observation.observation_id, *(item.field_id for item in forcing)),
            tuple(item.eulerian_result_id for item in result.coarse_evaluations),
            ended_at=ended,
        )
        self._record(
            stages,
            "hybrid_high_fidelity_source_refinement",
            started,
            tuple(
                item.source_state.source_state_id
                for item in result.coarse_evaluations
                if item.retained
            ),
            tuple(
                item.simulation.simulation_id
                for item in result.high_fidelity_evaluations
            ),
            status=(
                StageStatus.SUCCEEDED
                if result.high_fidelity_evaluations
                else StageStatus.ABSTAINED
            ),
            ended_at=ended,
        )
        self._record(
            stages,
            "hybrid_source_posterior",
            ended,
            (result.hindcast_id,),
            (result.source_posterior.posterior_id,),
        )
        return result

    def _run_hybrid_attribution(
        self,
        observation: SpillObservation,
        hindcast: HybridHindcastResult,
        stages: list[StageOutcome],
    ) -> _HybridAttributionOutputs:
        assert self._config.hybrid_ais is not None
        posterior = hindcast.source_posterior
        query = build_posterior_ais_query(posterior, self._config.hybrid_ais)
        started = datetime.now(UTC)
        try:
            points = self._ais_provider.fetch_points(query)
            tracks = self._track_reconstructor.build(points)
        except Exception as error:
            warning = QualityFlag(
                code="HYBRID_AIS_UNAVAILABLE",
                severity=QualitySeverity.WARNING,
                message=(
                    f"{type(error).__name__}: {error}; source posterior and forecast remain valid"
                ),
            )
            self._record_failure(
                stages,
                "hybrid_ais_query",
                started,
                (posterior.posterior_id,),
                error,
                warnings=(warning,),
            )
            attribution = self._abstained_hybrid_attribution(
                observation, hindcast, warning
            )
            self._record(
                stages,
                "hybrid_vessel_attribution",
                datetime.now(UTC),
                (posterior.posterior_id,),
                (attribution.attribution_id,),
                status=StageStatus.ABSTAINED,
                warnings=(warning,),
            )
            return _HybridAttributionOutputs(
                candidates=(), attribution=attribution, warnings=(warning,)
            )
        self._record(
            stages,
            "hybrid_ais_query",
            started,
            (posterior.posterior_id,),
            tuple(item.track_id for item in tracks),
        )
        candidate_area = posterior_credible_region(
            posterior, self._config.hybrid_ais.credible_region_percent
        )
        candidate_interval = posterior_release_interval(posterior)
        started = datetime.now(UTC)
        candidates = self._candidate_generator.generate(
            CandidateGenerationRequest(
                observation=observation,
                tracks=tracks,
                search_area=candidate_area,
                search_interval=candidate_interval,
            )
        )
        retained = tuple(
            item for item in candidates if item.decision == CandidateDecision.RETAINED
        )
        self._record(
            stages,
            "hybrid_candidate_generation",
            started,
            tuple(item.track_id for item in tracks),
            tuple(item.candidate_id for item in candidates),
        )
        evidence_sets = tuple(
            self._evidence_builder.build(
                AttributionEvidenceRequest(
                    observation=observation,
                    candidate=candidate,
                    source_posterior=posterior,
                    hybrid_hindcast=hindcast,
                )
            )
            for candidate in retained
        )
        caveat = (
            "Ranked vessel candidate is investigative prioritization, not proof of discharge."
        )
        started = datetime.now(UTC)
        attribution = self._attribution_model.rank(
            AttributionRequest(
                observation=observation,
                candidates=retained,
                simulations=tuple(
                    item.simulation for item in hindcast.high_fidelity_evaluations
                ),
                candidate_metrics=evidence_sets,
                interpretation_caveat=caveat,
            )
        )
        self._record(
            stages,
            "hybrid_vessel_attribution",
            started,
            tuple(item.candidate_id for item in retained),
            (attribution.attribution_id,),
            status=(
                StageStatus.ABSTAINED
                if attribution.decision == Decision.ABSTAINED
                else StageStatus.SUCCEEDED
            ),
        )
        return _HybridAttributionOutputs(
            candidates=candidates,
            attribution=attribution,
            warnings=attribution.warnings,
        )

    def _abstained_hybrid_attribution(
        self,
        observation: SpillObservation,
        hindcast: HybridHindcastResult,
        warning: QualityFlag,
    ) -> AttributionResult:
        result = self._attribution_model.rank(
            AttributionRequest(
                observation=observation,
                candidates=(),
                simulations=tuple(
                    item.simulation for item in hindcast.high_fidelity_evaluations
                ),
                interpretation_caveat=(
                    "Ranked vessel candidate is investigative prioritization, not proof of "
                    "discharge."
                ),
            )
        )
        return result.model_copy(update={"warnings": (*result.warnings, warning)})

    def _run_hybrid_forecast(
        self,
        observation: SpillObservation,
        forcing: tuple[EnvironmentalField, ...],
        coastline: CoastlineDataset,
        stages: list[StageOutcome],
    ) -> ForecastResult:
        release = ReleaseHypothesis(
            release_id=f"forecast-initial-condition:{observation.observation_id}",
            geometry=observation.geometry.centroid,
            interval=TimeRange(start=observation.observed_at, end=observation.observed_at),
            assumptions=(
                "Observed-slick forecast initialization, independent of source and vessel results.",
                self._config.execution.forecast_initialization_validation_note or "",
            ),
        )
        started = datetime.now(UTC)
        try:
            forecast = self._forecaster.forecast(
                EnsembleForecastRequest(
                    observation=observation,
                    release=release,
                    forcing=forcing,
                    random_seed=self._config.execution.random_seed,
                    coastline_geometries=coastline.segments,
                    coastline_source=coastline.source,
                )
            )
        except Exception as error:
            self._record_failure(
                stages,
                "hybrid_eulerian_forecast_screen",
                started,
                (observation.observation_id,),
                error,
            )
            raise
        diagnostics = forecast.hybrid_diagnostics
        if diagnostics is None:
            raise ValueError("hybrid forecaster must return HybridForecastDiagnostics")
        ended = datetime.now(UTC)
        self._record(
            stages,
            "hybrid_eulerian_forecast_screen",
            started,
            (observation.observation_id, *(item.field_id for item in forcing)),
            diagnostics.eulerian_result_ids,
            ended_at=ended,
        )
        self._record(
            stages,
            "hybrid_high_fidelity_forecast",
            started,
            tuple(item.scenario_id for item in diagnostics.scenario_weights),
            tuple(item.simulation_id for item in forecast.members),
            status=(StageStatus.SUCCEEDED if forecast.members else StageStatus.ABSTAINED),
            ended_at=ended,
            warnings=forecast.warnings,
        )
        return forecast

    def _run_coastal_impact(
        self,
        forecast: ForecastResult,
        coastline: CoastlineDataset,
        stages: list[StageOutcome],
        *,
        hybrid: bool,
    ) -> CoastalImpactResult:
        started = datetime.now(UTC)
        impact = self._coastal_analyzer.analyze(
            CoastalImpactRequest(forecast=forecast, coastline=coastline)
        )
        self._record(
            stages,
            "hybrid_coastal_impact" if hybrid else "coastal_impact",
            started,
            (forecast.forecast_id,),
            (impact.impact_id,),
            warnings=impact.warnings,
        )
        return impact

    def _optional_backward(
        self,
        observation: SpillObservation,
        forcing: tuple[EnvironmentalField, ...],
        stages: list[StageOutcome],
        warnings: list[QualityFlag],
    ) -> tuple[BackwardSourceTraceResult, ...]:
        started = datetime.now(UTC)
        assert self._source_config is not None
        config = self._source_config.backward_evidence
        if not config.enabled:
            self._record(
                stages,
                "backward_source_evidence",
                started,
                (observation.observation_id,),
                (),
                status=StageStatus.SKIPPED,
            )
            return ()
        window = ReleaseHypothesis(
            release_id=f"backward-search-window:{observation.observation_id}",
            geometry=observation.geometry.centroid,
            interval=observation.discharge_time.interval,
            assumptions=(
                config.validation_note,
                "Plausible backward-search window; not confirmed origin or source probability.",
            ),
        )
        try:
            evidence = self._backward_tracer.trace(
                BackwardTraceRequest(
                    observation=observation,
                    release=window,
                    forcing=forcing,
                    random_seed=self._config.execution.random_seed,
                )
            )
        except Exception as error:
            warning = QualityFlag(
                code="optional_backward_evidence_unavailable",
                severity=QualitySeverity.WARNING,
                message=f"{type(error).__name__}: {error}",
            )
            warnings.append(warning)
            stages.append(
                StageOutcome(
                    stage="backward_source_evidence",
                    status=StageStatus.FAILED,
                    started_at=started,
                    ended_at=datetime.now(UTC),
                    input_ids=(observation.observation_id,),
                    warnings=(warning,),
                    error_code=warning.code,
                    error_message=warning.message,
                )
            )
            return ()
        self._record(
            stages,
            "backward_source_evidence",
            started,
            (observation.observation_id,),
            (evidence.trace_id,),
        )
        return (evidence,)

    @staticmethod
    def serialize(result: PipelineResult, path: str | Path) -> ArtifactRef:
        """Serialize a result as canonical JSON and return its immutable artifact reference."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        content = target.read_bytes()
        return ArtifactRef(
            uri=target.resolve().as_uri(),
            media_type="application/json",
            sha256=sha256(content).hexdigest(),
            byte_size=len(content),
        )

    def _write_configuration_provenance(self) -> ArtifactRef:
        target = self._config.execution.output_path.with_name("pipeline_configuration.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self._config.model_dump_json(indent=2).encode()
        target.write_bytes(content)
        return ArtifactRef(
            uri=target.resolve().as_uri(),
            media_type="application/json",
            sha256=sha256(content).hexdigest(),
            byte_size=len(content),
        )

    @staticmethod
    def _record(
        stages: list[StageOutcome],
        stage: str,
        started_at: datetime,
        input_ids: tuple[str, ...],
        output_ids: tuple[str, ...],
        *,
        status: StageStatus = StageStatus.SUCCEEDED,
        ended_at: datetime | None = None,
        warnings: tuple[QualityFlag, ...] = (),
    ) -> None:
        stages.append(
            StageOutcome(
                stage=stage,
                status=status,
                started_at=started_at,
                ended_at=ended_at or datetime.now(UTC),
                input_ids=input_ids,
                output_ids=output_ids,
                warnings=warnings,
            )
        )

    @staticmethod
    def _record_failure(
        stages: list[StageOutcome],
        stage: str,
        started_at: datetime,
        input_ids: tuple[str, ...],
        error: Exception,
        *,
        warnings: tuple[QualityFlag, ...] = (),
    ) -> None:
        stages.append(
            StageOutcome(
                stage=stage,
                status=StageStatus.FAILED,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                input_ids=input_ids,
                warnings=warnings,
                error_code=type(error).__name__,
                error_message=str(error),
            )
        )


def _unique_environmental_fields(
    fields: list[EnvironmentalField],
) -> tuple[EnvironmentalField, ...]:
    unique = {field.field_id: field for field in fields}
    return tuple(unique[key] for key in sorted(unique))
