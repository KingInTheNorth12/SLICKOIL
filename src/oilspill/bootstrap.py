"""Registration of built-in adapters at the application composition boundary."""

from oilspill.adapters.detection.dense import (
    create_deeplabv3plus,
    create_segformer,
    create_unet,
    create_unetplusplus,
)
from oilspill.adapters.detection.yolov8 import create_yolov8_seg
from oilspill.adapters.dummy_detection import create_dummy_a, create_dummy_b
from oilspill.adapters.dummy_drift import (
    create_dummy_drift_a,
    create_dummy_drift_b,
    create_fake_drift,
)
from oilspill.adapters.sar.factory import (
    create_rasterio_sar_reader,
    create_snap_preprocessor,
    create_synthetic_copy_preprocessor,
)
from oilspill.age.window import create_configurable_window
from oilspill.ais.builder import create_vessel_track_builder
from oilspill.ais.csv_provider import create_csv_ais_provider
from oilspill.attribution.evidence import create_trace_metric_evidence
from oilspill.attribution.posterior import create_posterior_source_support
from oilspill.attribution.source_search import create_source_search_evidence
from oilspill.attribution.weighted import create_weighted_trajectory
from oilspill.candidates.generator import create_explainable_candidate_generator
from oilspill.characterization.raster import create_raster_characterizer
from oilspill.coastal.analyzer import create_geometry_coastal_impact
from oilspill.coastal.provider import create_geojson_coastline_provider
from oilspill.config import SourceInferenceConfig
from oilspill.drift.openoil import create_openoil
from oilspill.environment.xarray_provider import create_xarray_environmental_provider
from oilspill.eulerian.numpy_fvm import create_numpy_fvm
from oilspill.forecasting.ensemble import create_configured_ensemble
from oilspill.forecasting.eulerian_screened import create_eulerian_screened
from oilspill.forecasting.monte_carlo import create_monte_carlo
from oilspill.hybrid.hindcast import create_physics_first_coarse
from oilspill.hybrid.selective import create_selective_hybrid
from oilspill.ports import (
    AttributionModel,
    CoastalImpactAnalyzer,
    DetectorModel,
    DriftModel,
    EnvironmentalProvider,
    EulerianTransportEngine,
    SourceInference,
    SpillAgeEstimator,
)
from oilspill.registry import (
    ComponentFactory,
    ComponentRegistries,
    SourceSearchDependencies,
    default_registries,
)
from oilspill.source_tracing.hypotheses import create_observed_ais_releases
from oilspill.source_tracing.release import create_ais_point_release
from oilspill.source_tracing.search import create_iterative_forward
from oilspill.source_tracing.similarity import create_geometry_similarity


def register_builtin_components(registries: ComponentRegistries = default_registries) -> None:
    """Register deterministic development adapters without selecting any of them."""

    # The high-level hybrid hindcast registry remains empty until that orchestration is built.

    if "observed_ais_releases" not in registries.release_enumerator.available():
        registries.release_enumerator.register(
            "observed_ais_releases", create_observed_ais_releases
        )
    if "geometry_similarity" not in registries.observation_likelihood.available():
        registries.observation_likelihood.register(
            "geometry_similarity", create_geometry_similarity
        )
    if "iterative_forward" not in registries.source_inference.available():
        registries.source_inference.register("iterative_forward", create_iterative_forward)
    detector_factories: dict[str, ComponentFactory[DetectorModel]] = {
        "dummy_a": create_dummy_a,
        "dummy_b": create_dummy_b,
        "deeplabv3plus": create_deeplabv3plus,
        "unet": create_unet,
        "unetplusplus": create_unetplusplus,
        "segformer": create_segformer,
        "yolov8_seg": create_yolov8_seg,
    }
    drift_factories: dict[str, ComponentFactory[DriftModel]] = {
        "dummy_drift_a": create_dummy_drift_a,
        "dummy_drift_b": create_dummy_drift_b,
        "fake_drift": create_fake_drift,
        "openoil": create_openoil,
    }
    age_factories: dict[str, ComponentFactory[SpillAgeEstimator]] = {
        "configurable_window": create_configurable_window,
    }
    environmental_factories: dict[str, ComponentFactory[EnvironmentalProvider]] = {
        "xarray_mapped": create_xarray_environmental_provider,
    }
    eulerian_factories: dict[str, ComponentFactory[EulerianTransportEngine]] = {
        "numpy_fvm": create_numpy_fvm,
    }
    attribution_factories: dict[str, ComponentFactory[AttributionModel]] = {
        "weighted_trajectory": create_weighted_trajectory,
    }
    coastal_impact_factories: dict[str, ComponentFactory[CoastalImpactAnalyzer]] = {
        "geometry_intersection": create_geometry_coastal_impact,
    }
    if "rasterio" not in registries.sar_reader.available():
        registries.sar_reader.register("rasterio", create_rasterio_sar_reader)
    if "synthetic_copy" not in registries.sar_preprocessor.available():
        registries.sar_preprocessor.register("synthetic_copy", create_synthetic_copy_preprocessor)
    if "raster_mask" not in registries.spill_characterizer.available():
        registries.spill_characterizer.register("raster_mask", create_raster_characterizer)
    if "csv" not in registries.ais_provider.available():
        registries.ais_provider.register("csv", create_csv_ais_provider)
    if "canonical_track_builder" not in registries.vessel_track_reconstructor.available():
        registries.vessel_track_reconstructor.register(
            "canonical_track_builder", create_vessel_track_builder
        )
    if "explainable" not in registries.candidate_generator.available():
        registries.candidate_generator.register(
            "explainable", create_explainable_candidate_generator
        )
    if "ais_point_release" not in registries.release_hypothesis_generator.available():
        registries.release_hypothesis_generator.register(
            "ais_point_release", create_ais_point_release
        )
    if "source_search_metrics" not in registries.attribution_evidence.available():
        registries.attribution_evidence.register(
            "source_search_metrics", create_source_search_evidence
        )
    if "trace_metrics" not in registries.attribution_evidence.available():
        registries.attribution_evidence.register("trace_metrics", create_trace_metric_evidence)
    if "posterior_source_support" not in registries.attribution_evidence.available():
        registries.attribution_evidence.register(
            "posterior_source_support", create_posterior_source_support
        )
    if "geojson" not in registries.coastline_provider.available():
        registries.coastline_provider.register("geojson", create_geojson_coastline_provider)
    if "configured_ensemble" not in registries.ensemble_forecaster.available():
        registries.ensemble_forecaster.register("configured_ensemble", create_configured_ensemble)
    if "monte_carlo" not in registries.ensemble_forecaster.available():
        registries.ensemble_forecaster.register("monte_carlo", create_monte_carlo)
    if "eulerian_screened" not in registries.hybrid_ensemble_forecaster.available():
        registries.hybrid_ensemble_forecaster.register(
            "eulerian_screened", create_eulerian_screened
        )
    if "sentinel1_snap" not in registries.sar_preprocessor.available():
        registries.sar_preprocessor.register("sentinel1_snap", create_snap_preprocessor)
    for name, detector_factory in detector_factories.items():
        if name not in registries.detector.available():
            registries.detector.register(name, detector_factory)
    for name, drift_factory in drift_factories.items():
        if name not in registries.drift.available():
            registries.drift.register(name, drift_factory)
    for name, age_factory in age_factories.items():
        if name not in registries.spill_age_estimator.available():
            registries.spill_age_estimator.register(name, age_factory)
    for name, environmental_factory in environmental_factories.items():
        if name not in registries.environmental_provider.available():
            registries.environmental_provider.register(name, environmental_factory)
    for name, eulerian_factory in eulerian_factories.items():
        if name not in registries.eulerian_transport.available():
            registries.eulerian_transport.register(name, eulerian_factory)
    if "physics_first" not in registries.coarse_hindcast.available():
        registries.coarse_hindcast.register("physics_first", create_physics_first_coarse)
    if "selective_refinement" not in registries.hybrid_hindcast.available():
        registries.hybrid_hindcast.register(
            "selective_refinement", create_selective_hybrid
        )
    for name, attribution_factory in attribution_factories.items():
        if name not in registries.attribution.available():
            registries.attribution.register(name, attribution_factory)
    for name, coastal_impact_factory in coastal_impact_factories.items():
        if name not in registries.coastal_impact.available():
            registries.coastal_impact.register(name, coastal_impact_factory)


def create_registries_with_builtins() -> ComponentRegistries:
    registries = ComponentRegistries()
    register_builtin_components(registries)
    return registries


def create_source_inference(
    config: SourceInferenceConfig,
    registries: ComponentRegistries | None = None,
    *,
    drift_model: DriftModel | None = None,
) -> SourceInference:
    """Resolve replaceable strategies at the composition boundary only."""
    registry = registries or create_registries_with_builtins()
    dependencies = SourceSearchDependencies(
        drift=drift_model
        if drift_model is not None
        else registry.drift.create(config.drift.name, config.drift),
        releases=registry.release_enumerator.create(
            config.release_enumerator.name, config.release_enumerator
        ),
        likelihood=registry.observation_likelihood.create(
            config.observation_likelihood.name, config.observation_likelihood
        ),
        resolved_components=(
            config.drift,
            config.release_enumerator,
            config.observation_likelihood,
        ),
    )
    return registry.source_inference.create(config.search.name, config.search, dependencies)
