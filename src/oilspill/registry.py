"""Type-safe component registries and project-wide registry instances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar, cast

from oilspill.config import ComponentConfig
from oilspill.ports import (
    AISProvider,
    AttributionEvidenceBuilder,
    AttributionModel,
    CandidateGenerator,
    CoarseHindcastEngine,
    CoastalImpactAnalyzer,
    CoastlineProvider,
    DetectorModel,
    DriftModel,
    EnsembleForecaster,
    EnvironmentalProvider,
    EulerianTransportEngine,
    HybridHindcastEngine,
    ObservationLikelihood,
    ReleaseHypothesisEnumerator,
    ReleaseHypothesisGenerator,
    SARPreprocessor,
    SARReader,
    SourceInference,
    SpillAgeEstimator,
    SpillCharacterizer,
    VesselTrackReconstructor,
)

ComponentT = TypeVar("ComponentT")
ComponentFactory = Callable[[ComponentConfig], ComponentT]
DependencyT = TypeVar("DependencyT")


class RegistryError(ValueError):
    """Base error for registry construction and lookup failures."""


class DuplicateComponentError(RegistryError):
    pass


class UnknownComponentError(RegistryError):
    pass


class ComponentConfigurationMismatchError(RegistryError):
    pass


class ComponentContractError(RegistryError):
    pass


class ComponentRegistry(Generic[ComponentT]):
    """Maps stable configuration names to adapter factories.

    Selection is data-driven. The registry performs exact-name lookup and runtime protocol
    validation; callers and orchestrators never branch on implementation names.
    """

    def __init__(self, kind: str, contract: type[Any]) -> None:
        self._kind = kind
        self._contract = contract
        self._factories: dict[str, ComponentFactory[ComponentT]] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(self, name: str, factory: ComponentFactory[ComponentT]) -> None:
        if name in self._factories:
            raise DuplicateComponentError(f"{self._kind} component {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, config: ComponentConfig) -> ComponentT:
        if config.name != name:
            raise ComponentConfigurationMismatchError(
                f"requested {self._kind} {name!r}, but configuration selects {config.name!r}"
            )
        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(self.available()) or "<none>"
            raise UnknownComponentError(
                f"unknown {self._kind} component {name!r}; available: {available}"
            ) from error

        component = factory(config)
        # Runtime-checkable Protocols are valid isinstance targets but typing cannot express that.
        if not isinstance(component, cast(Any, self._contract)):
            raise ComponentContractError(
                f"factory for {self._kind} {name!r} returned an incompatible component"
            )
        return component

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


class DependentComponentRegistry(Generic[ComponentT, DependencyT]):
    """Registry for components whose factory receives one already-resolved dependency."""

    def __init__(self, kind: str, contract: type[Any]) -> None:
        self._kind = kind
        self._contract = contract
        self._factories: dict[str, Callable[[ComponentConfig, DependencyT], ComponentT]] = {}

    def register(
        self, name: str, factory: Callable[[ComponentConfig, DependencyT], ComponentT]
    ) -> None:
        if name in self._factories:
            raise DuplicateComponentError(f"{self._kind} component {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, config: ComponentConfig, dependency: DependencyT) -> ComponentT:
        if config.name != name:
            raise ComponentConfigurationMismatchError(
                f"requested {self._kind} {name!r}, but configuration selects {config.name!r}"
            )
        try:
            component = self._factories[name](config, dependency)
        except KeyError as error:
            available = ", ".join(self.available()) or "<none>"
            raise UnknownComponentError(
                f"unknown {self._kind} component {name!r}; available: {available}"
            ) from error
        if not isinstance(component, cast(Any, self._contract)):
            raise ComponentContractError(
                f"factory for {self._kind} {name!r} returned an incompatible component"
            )
        return component

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


DetectorRegistry: TypeAlias = ComponentRegistry[DetectorModel]


@dataclass(frozen=True)
class SourceSearchDependencies:
    drift: DriftModel
    releases: ReleaseHypothesisEnumerator
    likelihood: ObservationLikelihood
    resolved_components: tuple[ComponentConfig, ...]


@dataclass(frozen=True)
class HybridHindcastDependencies:
    eulerian: EulerianTransportEngine
    drift: DriftModel
    resolved_components: tuple[ComponentConfig, ...]
    coarse: CoarseHindcastEngine | None = None


@dataclass(frozen=True)
class HybridForecastDependencies:
    eulerian: EulerianTransportEngine
    drift: DriftModel
    resolved_components: tuple[ComponentConfig, ...]


@dataclass(frozen=True)
class ComponentRegistries:
    sar_reader: ComponentRegistry[SARReader] = field(
        default_factory=lambda: ComponentRegistry("SAR reader", SARReader)
    )
    detector: ComponentRegistry[DetectorModel] = field(
        default_factory=lambda: ComponentRegistry("detector", DetectorModel)
    )
    sar_preprocessor: ComponentRegistry[SARPreprocessor] = field(
        default_factory=lambda: ComponentRegistry("SAR preprocessor", SARPreprocessor)
    )
    spill_characterizer: ComponentRegistry[SpillCharacterizer] = field(
        default_factory=lambda: ComponentRegistry("spill characterizer", SpillCharacterizer)
    )
    ais_provider: ComponentRegistry[AISProvider] = field(
        default_factory=lambda: ComponentRegistry("AIS provider", AISProvider)
    )
    vessel_track_reconstructor: ComponentRegistry[VesselTrackReconstructor] = field(
        default_factory=lambda: ComponentRegistry(
            "vessel-track reconstructor", VesselTrackReconstructor
        )
    )
    candidate_generator: ComponentRegistry[CandidateGenerator] = field(
        default_factory=lambda: ComponentRegistry("candidate generator", CandidateGenerator)
    )
    release_hypothesis_generator: ComponentRegistry[ReleaseHypothesisGenerator] = field(
        default_factory=lambda: ComponentRegistry(
            "release-hypothesis generator", ReleaseHypothesisGenerator
        )
    )
    environmental_provider: ComponentRegistry[EnvironmentalProvider] = field(
        default_factory=lambda: ComponentRegistry("environmental provider", EnvironmentalProvider)
    )
    eulerian_transport: ComponentRegistry[EulerianTransportEngine] = field(
        default_factory=lambda: ComponentRegistry(
            "Eulerian transport engine", EulerianTransportEngine
        )
    )
    hybrid_hindcast: DependentComponentRegistry[
        HybridHindcastEngine, HybridHindcastDependencies
    ] = field(
        default_factory=lambda: DependentComponentRegistry(
            "hybrid hindcast engine", HybridHindcastEngine
        )
    )
    coarse_hindcast: DependentComponentRegistry[
        CoarseHindcastEngine, EulerianTransportEngine
    ] = field(
        default_factory=lambda: DependentComponentRegistry(
            "coarse hindcast engine", CoarseHindcastEngine
        )
    )
    spill_age_estimator: ComponentRegistry[SpillAgeEstimator] = field(
        default_factory=lambda: ComponentRegistry("spill-age estimator", SpillAgeEstimator)
    )
    drift: ComponentRegistry[DriftModel] = field(
        default_factory=lambda: ComponentRegistry("drift model", DriftModel)
    )
    attribution: ComponentRegistry[AttributionModel] = field(
        default_factory=lambda: ComponentRegistry("attribution model", AttributionModel)
    )
    attribution_evidence: ComponentRegistry[AttributionEvidenceBuilder] = field(
        default_factory=lambda: ComponentRegistry(
            "attribution evidence builder", AttributionEvidenceBuilder
        )
    )
    ensemble_forecaster: DependentComponentRegistry[EnsembleForecaster, DriftModel] = field(
        default_factory=lambda: DependentComponentRegistry(
            "ensemble forecaster", EnsembleForecaster
        )
    )
    hybrid_ensemble_forecaster: DependentComponentRegistry[
        EnsembleForecaster, HybridForecastDependencies
    ] = field(
        default_factory=lambda: DependentComponentRegistry(
            "hybrid ensemble forecaster", EnsembleForecaster
        )
    )
    coastline_provider: ComponentRegistry[CoastlineProvider] = field(
        default_factory=lambda: ComponentRegistry("coastline provider", CoastlineProvider)
    )
    coastal_impact: ComponentRegistry[CoastalImpactAnalyzer] = field(
        default_factory=lambda: ComponentRegistry("coastal-impact analyzer", CoastalImpactAnalyzer)
    )

    release_enumerator: ComponentRegistry[ReleaseHypothesisEnumerator] = field(
        default_factory=lambda: ComponentRegistry("release enumerator", ReleaseHypothesisEnumerator)
    )
    observation_likelihood: ComponentRegistry[ObservationLikelihood] = field(
        default_factory=lambda: ComponentRegistry("observation similarity", ObservationLikelihood)
    )
    source_inference: DependentComponentRegistry[SourceInference, SourceSearchDependencies] = field(
        default_factory=lambda: DependentComponentRegistry("source inference", SourceInference)
    )


default_registries = ComponentRegistries()
sar_reader_registry = default_registries.sar_reader
detector_registry = default_registries.detector
sar_preprocessor_registry = default_registries.sar_preprocessor
spill_characterizer_registry = default_registries.spill_characterizer
ais_provider_registry = default_registries.ais_provider
vessel_track_reconstructor_registry = default_registries.vessel_track_reconstructor
candidate_generator_registry = default_registries.candidate_generator
release_hypothesis_generator_registry = default_registries.release_hypothesis_generator
environmental_provider_registry = default_registries.environmental_provider
eulerian_transport_registry = default_registries.eulerian_transport
hybrid_hindcast_registry = default_registries.hybrid_hindcast
coarse_hindcast_registry = default_registries.coarse_hindcast
spill_age_estimator_registry = default_registries.spill_age_estimator
drift_registry = default_registries.drift
attribution_registry = default_registries.attribution
attribution_evidence_registry = default_registries.attribution_evidence
ensemble_forecaster_registry = default_registries.ensemble_forecaster
hybrid_ensemble_forecaster_registry = default_registries.hybrid_ensemble_forecaster
coastline_provider_registry = default_registries.coastline_provider
coastal_impact_registry = default_registries.coastal_impact

release_enumerator_registry = default_registries.release_enumerator
observation_likelihood_registry = default_registries.observation_likelihood
source_inference_registry = default_registries.source_inference
