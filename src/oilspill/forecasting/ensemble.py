"""Deterministic ensemble orchestration over the generic ``DriftModel`` interface."""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from oilspill.config import ComponentConfig
from oilspill.domain.common import ComponentMetadata, FrozenModel, Identifier, MetadataEntry
from oilspill.domain.environment import EnvironmentalField
from oilspill.domain.forecast import (
    ForecastHorizon,
    ForecastResult,
    ForecastRunProvenance,
    SpatialProbabilityRepresentation,
    UncertaintySummary,
)
from oilspill.domain.geospatial import (
    Coordinate,
    Duration,
    PointGeometry,
    SpatialGeometry,
    SpatialUnit,
)
from oilspill.ports import DriftModel
from oilspill.requests import DriftForecastRequest, EnsembleForecastRequest


def _seconds(duration: Duration) -> float:
    factors = {"second": 1.0, "minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
    return duration.value * factors[duration.unit]


class EnsembleMemberPerturbation(FrozenModel):
    """One explicit ensemble member; values make no scientific-validity claim."""

    member_id: Identifier
    forcing_field_ids: tuple[Identifier, ...] | None = None
    position_offset_x_metres: float
    position_offset_y_metres: float
    time_offset_seconds: float
    model_parameters: tuple[MetadataEntry, ...] = ()
    configured_uncertainty: tuple[MetadataEntry, ...] = ()
    validation_note: Identifier

    @model_validator(mode="after")
    def _unique_entries(self) -> EnsembleMemberPerturbation:
        if self.forcing_field_ids is not None and len(self.forcing_field_ids) != len(
            set(self.forcing_field_ids)
        ):
            raise ValueError("ensemble forcing field IDs must be unique")
        for label, entries in (
            ("model parameter", self.model_parameters),
            ("uncertainty", self.configured_uncertainty),
        ):
            keys = tuple(item.key for item in entries)
            if len(keys) != len(set(keys)):
                raise ValueError(f"ensemble {label} keys must be unique")
        return self


class EnsembleForecastConfig(FrozenModel):
    horizons: tuple[Duration, ...] = Field(min_length=1)
    members: tuple[EnsembleMemberPerturbation, ...] = Field(min_length=1)
    initial_position_strategy: Literal["observed_centroid"]
    spatial_probability_method: Literal["equal_weight_member_support"]
    uncertainty_validation_note: Identifier

    @model_validator(mode="after")
    def _unique_positive_values(self) -> EnsembleForecastConfig:
        horizon_seconds = tuple(_seconds(item) for item in self.horizons)
        if any(value <= 0 for value in horizon_seconds):
            raise ValueError("forecast horizons must be positive")
        if len(horizon_seconds) != len(set(horizon_seconds)):
            raise ValueError("forecast horizons must be unique")
        if horizon_seconds != tuple(sorted(horizon_seconds)):
            raise ValueError("configured forecast horizons must be chronological")
        member_ids = tuple(item.member_id for item in self.members)
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("ensemble member IDs must be unique")
        return self


class ConfiguredEnsembleForecaster:
    """Run explicit perturbation members at each horizon using any ``DriftModel``."""

    def __init__(
        self,
        config: EnsembleForecastConfig,
        drift_model: DriftModel,
        *,
        component_name: str = "configured_ensemble",
    ) -> None:
        self._config = config
        self._drift_model = drift_model
        self._component_name = component_name

    def forecast(self, request: EnsembleForecastRequest) -> ForecastResult:
        horizons = request.horizons or tuple(
            request.observation.observed_at + timedelta(seconds=_seconds(offset))
            for offset in self._config.horizons
        )
        if tuple(sorted(horizons)) != horizons or len(horizons) != len(set(horizons)):
            raise ValueError("forecast horizons must be unique and chronologically ordered")
        if any(horizon <= request.observation.observed_at for horizon in horizons):
            raise ValueError("forecast horizons must follow the observation time")

        simulations = []
        provenance = []
        horizon_results = []
        for horizon in horizons:
            horizon_simulations = []
            distributions = []
            for member in self._config.members:
                initial_position = self._perturbed_position(
                    request.observation.geometry.centroid, member
                )
                initial_timestamp = request.observation.observed_at + timedelta(
                    seconds=member.time_offset_seconds
                )
                if initial_timestamp >= horizon:
                    raise ValueError(
                        f"ensemble member {member.member_id!r} starts at or after its horizon"
                    )
                forcing = self._forcing(request, member)
                seed = self._member_seed(request.random_seed, member.member_id, horizon.isoformat())
                simulation = self._drift_model.forecast(
                    DriftForecastRequest(
                        observation=request.observation,
                        release=request.release,
                        forcing=forcing,
                        valid_until=horizon,
                        random_seed=seed,
                        initial_position=initial_position,
                        initial_timestamp=initial_timestamp,
                        model_parameters=member.model_parameters,
                        configured_uncertainty=member.configured_uncertainty,
                    )
                )
                distribution = simulation.target_distribution
                if distribution is None or simulation.target_timestamp != horizon:
                    raise ValueError(
                        "drift model must return a target-time particle distribution for "
                        "forecasting"
                    )
                simulations.append(simulation)
                horizon_simulations.append(simulation)
                distributions.append(distribution)
                provenance.append(
                    ForecastRunProvenance(
                        member_id=member.member_id,
                        horizon=horizon,
                        simulation_id=simulation.simulation_id,
                        random_seed=seed,
                        forcing_ids=simulation.forcing_ids,
                        position_offset=Coordinate(
                            x=member.position_offset_x_metres,
                            y=member.position_offset_y_metres,
                        ),
                        position_offset_unit=SpatialUnit.METRE,
                        time_offset_seconds=member.time_offset_seconds,
                        model_parameters=member.model_parameters,
                        configured_uncertainty=member.configured_uncertainty,
                        validation_note=member.validation_note,
                        trajectory_artifact=simulation.trajectories,
                    )
                )
            probability = SpatialProbabilityRepresentation(
                method=self._config.spatial_probability_method,
                member_geometries=tuple(
                    distribution.comparison_geometry for distribution in distributions
                ),
                probability_per_member=1.0 / len(distributions),
                probability_raster=None,
                calibrated=False,
                explanation=(
                    "Equal-weight empirical member support; no calibrated gridded probability "
                    "is claimed."
                ),
            )
            horizon_results.append(
                ForecastHorizon(
                    valid_at=horizon,
                    particle_ensemble=tuple(distributions),
                    spatial_probability=probability,
                    member_simulation_ids=tuple(
                        simulation.simulation_id for simulation in horizon_simulations
                    ),
                )
            )
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        forecast_key = sha256(
            f"{request.observation.observation_id}:{request.random_seed}:{config_hash}:"
            f"{','.join(item.isoformat() for item in horizons)}".encode()
        ).hexdigest()[:16]
        return ForecastResult(
            forecast_id=f"ensemble-forecast:{forecast_key}",
            observation_id=request.observation.observation_id,
            issued_at=request.observation.observed_at,
            horizons=tuple(horizon_results),
            members=tuple(simulations),
            uncertainty=UncertaintySummary(
                method="explicit_configured_perturbation_ensemble",
                ensemble_member_count=len(simulations),
                calibrated=False,
                random_seed=request.random_seed,
                configured_perturbations=(
                    MetadataEntry(key="member_count_per_horizon", value=len(self._config.members)),
                    MetadataEntry(key="horizon_count", value=len(horizons)),
                    MetadataEntry(
                        key="validation_note", value=self._config.uncertainty_validation_note
                    ),
                ),
                notes=(
                    "Perturbation ranges are configuration, not scientifically validated defaults.",
                ),
            ),
            run_provenance=tuple(provenance),
        )

    def component_metadata(self) -> ComponentMetadata:
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name=self._component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="generic-drift-ensemble",
            configuration_sha256=config_hash,
            attributes=(
                MetadataEntry(
                    key="uncertainty_validation",
                    value=self._config.uncertainty_validation_note,
                ),
            ),
        )

    @staticmethod
    def _perturbed_position(
        position: SpatialGeometry, member: EnsembleMemberPerturbation
    ) -> SpatialGeometry:
        geometry = position.geometry
        if geometry.type != "Point":
            raise ValueError("forecast initial position must contain Point geometry")
        has_offset = member.position_offset_x_metres != 0 or member.position_offset_y_metres != 0
        if has_offset and (
            position.crs.is_geographic
            or position.crs.axis_units != (SpatialUnit.METRE, SpatialUnit.METRE)
        ):
            raise ValueError("non-zero forecast position offsets require a projected metre CRS")
        return SpatialGeometry(
            geometry=PointGeometry(
                coordinate=Coordinate(
                    x=geometry.coordinate.x + member.position_offset_x_metres,
                    y=geometry.coordinate.y + member.position_offset_y_metres,
                )
            ),
            crs=position.crs,
        )

    @staticmethod
    def _member_seed(base_seed: int, member_id: str, horizon: str) -> int:
        digest = sha256(f"{base_seed}:{member_id}:{horizon}".encode()).digest()
        return int.from_bytes(digest[:4], "big", signed=False)

    @staticmethod
    def _forcing(
        request: EnsembleForecastRequest, member: EnsembleMemberPerturbation
    ) -> tuple[EnvironmentalField, ...]:
        if member.forcing_field_ids is None:
            return request.forcing
        available = {field.field_id: field for field in request.forcing}
        missing = set(member.forcing_field_ids).difference(available)
        if missing:
            raise ValueError(
                f"ensemble member {member.member_id!r} references unknown forcing: "
                f"{sorted(missing)}"
            )
        return tuple(available[field_id] for field_id in member.forcing_field_ids)


def create_configured_ensemble(
    config: ComponentConfig, drift_model: DriftModel
) -> ConfiguredEnsembleForecaster:
    settings = EnsembleForecastConfig.model_validate(config.settings)
    return ConfiguredEnsembleForecaster(settings, drift_model, component_name=config.name)
