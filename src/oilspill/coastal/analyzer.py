"""CRS-safe ensemble contact analysis against canonical coastline geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal

import rasterio
from pydantic import model_validator
from rasterio.warp import transform_geom
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape

from oilspill.config import ComponentConfig
from oilspill.domain.common import ComponentMetadata, FrozenModel, Identifier, MetadataEntry
from oilspill.domain.forecast import (
    CoastalImpactResult,
    CoastlineSegmentImpact,
    ForecastRunProvenance,
    ScenarioContactArrival,
)
from oilspill.domain.geospatial import PolygonGeometry, SpatialGeometry
from oilspill.requests import CoastalImpactRequest


@dataclass(frozen=True)
class _ForecastMember:
    member_id: str
    weight: float | None


@dataclass(frozen=True)
class _ContactRecord:
    member: _ForecastMember
    first_contact: datetime


def _forecast_members(
    provenance: tuple[ForecastRunProvenance, ...],
) -> tuple[dict[str, _ForecastMember], bool]:
    if not provenance:
        raise ValueError("coastal-impact probability requires forecast run provenance")
    explicit = tuple(item.scenario_weight is not None for item in provenance)
    if any(explicit) and not all(explicit):
        raise ValueError("forecast provenance cannot mix weighted and unweighted runs")
    weighted = all(explicit)
    logical_weights: dict[str, float | None] = {}
    logical_by_simulation: dict[str, str] = {}
    for item in provenance:
        if item.simulation_id in logical_by_simulation:
            raise ValueError("forecast run provenance simulation IDs must be unique")
        logical_id = item.scenario_id or item.member_id
        weight = item.scenario_weight
        previous = logical_weights.get(logical_id)
        if logical_id in logical_weights and previous != weight:
            raise ValueError("scenario weight must be consistent across forecast horizons")
        if weight is not None and (not math.isfinite(weight) or weight < 0):
            raise ValueError("scenario weights must be finite and non-negative")
        logical_weights[logical_id] = weight
        logical_by_simulation[item.simulation_id] = logical_id
    if weighted:
        total = sum(weight or 0.0 for weight in logical_weights.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise ValueError("unique forecast scenario weights must sum to one")
        logical_weights = {
            logical_id: (weight or 0.0) / total
            for logical_id, weight in logical_weights.items()
        }
    return (
        {
            simulation_id: _ForecastMember(
                member_id=logical_id,
                weight=logical_weights[logical_id],
            )
            for simulation_id, logical_id in logical_by_simulation.items()
        },
        weighted,
    )


def weighted_datetime_quantile(
    values: tuple[datetime, ...], weights: tuple[float, ...], quantile: float
) -> datetime:
    """Return the inverse empirical-CDF quantile with deterministic timestamp tie handling."""
    if not values or len(values) != len(weights):
        raise ValueError("weighted arrival quantiles require aligned non-empty values")
    if not 0 <= quantile <= 1 or not math.isfinite(quantile):
        raise ValueError("arrival quantile must lie within [0, 1]")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("arrival weights must be finite and non-negative")
    combined: dict[datetime, float] = {}
    for value, weight in zip(values, weights, strict=True):
        combined[value] = combined.get(value, 0.0) + weight
    ordered = tuple(sorted(combined.items()))
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        raise ValueError("arrival weights must contain positive mass")
    target = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target and weight > 0:
            return value
    return ordered[-1][0]


class GeometryCoastalImpactConfig(FrozenModel):
    measurement_crs: Identifier
    contact_method: Literal["geometry_intersects"]
    arrival_estimate_policy: Literal["none", "median_contacting_members"]
    include_unimpacted_segments: bool
    validation_note: Identifier

    @model_validator(mode="after")
    def _projected_metres(self) -> GeometryCoastalImpactConfig:
        target = rasterio.crs.CRS.from_user_input(self.measurement_crs)
        if not target.is_projected:
            raise ValueError("coastal-impact measurement_crs must be projected")
        units = (target.linear_units or "").lower()
        if units not in {"metre", "meter", "metres", "meters"}:
            raise ValueError("coastal-impact measurement_crs must use metre units")
        return self


class GeometryCoastalImpactAnalyzer:
    """Estimate discrete ensemble coastline contact without invoking a drift engine."""

    def __init__(
        self,
        config: GeometryCoastalImpactConfig,
        *,
        component_name: str = "geometry_coastal_impact",
    ) -> None:
        self._config = config
        self._target_crs = rasterio.crs.CRS.from_user_input(config.measurement_crs)
        self._component_name = component_name

    def analyze(self, request: CoastalImpactRequest) -> CoastalImpactResult:
        forecast = request.forecast
        if not forecast.run_provenance:
            raise ValueError("coastal-impact probability requires forecast run provenance")
        member_by_simulation, explicit_weights = _forecast_members(forecast.run_provenance)
        member_ids = tuple(sorted({item.member_id for item in member_by_simulation.values()}))
        arrivals_by_segment: list[dict[str, _ContactRecord]] = [
            {} for _ in request.coastline.segments
        ]
        projected_segments = [self._project(segment) for segment in request.coastline.segments]

        for horizon in forecast.horizons:
            if len(horizon.particle_ensemble) != len(horizon.member_simulation_ids):
                raise ValueError("coastal analysis requires one particle distribution per run")
            for simulation_id, distribution in zip(
                horizon.member_simulation_ids, horizon.particle_ensemble, strict=True
            ):
                try:
                    member = member_by_simulation[simulation_id]
                except KeyError as error:
                    raise ValueError("forecast horizon lacks matching run provenance") from error
                support = self._project(distribution.comparison_geometry)
                for index, coastline in enumerate(projected_segments):
                    if support.intersects(coastline):
                        previous = arrivals_by_segment[index].get(member.member_id)
                        if previous is None or horizon.valid_at < previous.first_contact:
                            arrivals_by_segment[index][member.member_id] = _ContactRecord(
                                member=member, first_contact=horizon.valid_at
                            )

        segments = []
        for index, (geometry, member_arrivals) in enumerate(
            zip(request.coastline.segments, arrivals_by_segment, strict=True)
        ):
            if not member_arrivals and not self._config.include_unimpacted_segments:
                continue
            records = tuple(
                sorted(member_arrivals.values(), key=lambda item: item.first_contact)
            )
            arrivals = tuple(item.first_contact for item in records)
            arrival_weights = tuple(
                item.member.weight if item.member.weight is not None else 1.0
                for item in records
            )
            has_arrival_mass = sum(arrival_weights) > 0
            quantiles = (
                tuple(
                    weighted_datetime_quantile(arrivals, arrival_weights, quantile)
                    for quantile in (0.1, 0.5, 0.9)
                )
                if arrivals and has_arrival_mass
                else (None, None, None)
            )
            if explicit_weights:
                estimated = (
                    quantiles[1]
                    if self._config.arrival_estimate_policy != "none"
                    else None
                )
                arrival_method = (
                    "weighted_empirical_first_contact_quantiles"
                    if estimated is not None
                    else None
                )
                impact_probability = sum(arrival_weights)
            else:
                estimated = self._estimated_arrival(arrivals)
                arrival_method = (
                    self._config.arrival_estimate_policy if estimated is not None else None
                )
                impact_probability = len(member_arrivals) / len(member_ids)
            segments.append(
                CoastlineSegmentImpact(
                    segment_id=f"{request.coastline.dataset_id}:segment:{index:04d}",
                    geometry=geometry,
                    impact_probability=impact_probability,
                    estimated_arrival=estimated,
                    earliest_arrival=arrivals[0] if arrivals else None,
                    latest_arrival=arrivals[-1] if arrivals else None,
                    arrival_p10=quantiles[0],
                    arrival_p50=quantiles[1],
                    arrival_p90=quantiles[2],
                    arrival_estimate_method=arrival_method,
                    contacting_member_ids=tuple(sorted(member_arrivals)),
                    contact_arrivals=tuple(
                        ScenarioContactArrival(
                            scenario_id=item.member.member_id,
                            first_contact=item.first_contact,
                            scenario_weight=(
                                item.member.weight
                                if item.member.weight is not None
                                else 1.0 / len(member_ids)
                            ),
                            weight_semantics=(
                                "explicit_design_weight"
                                if explicit_weights
                                else "implicit_equal_member_weight"
                            ),
                        )
                        for item in sorted(
                            records, key=lambda value: value.member.member_id
                        )
                    ),
                    ensemble_hit_count=len(member_arrivals),
                    ensemble_member_count=len(member_ids),
                    provenance=(
                        MetadataEntry(key="measurement_crs", value=self._config.measurement_crs),
                        MetadataEntry(key="contact_method", value=self._config.contact_method),
                        MetadataEntry(key="temporal_interpolation", value="none"),
                        MetadataEntry(
                            key="temporal_resolution",
                            value="limited_to_available_forecast_snapshots",
                        ),
                        MetadataEntry(
                            key="impact_probability_aggregation",
                            value=(
                                "explicit_scenario_weights"
                                if explicit_weights
                                else "equal_member_frequency"
                            ),
                        ),
                        MetadataEntry(key="validation_note", value=self._config.validation_note),
                    ),
                )
            )
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        impact_key = sha256(
            f"{forecast.forecast_id}:{request.coastline.dataset_id}:{config_hash}".encode()
        ).hexdigest()[:16]
        return CoastalImpactResult(
            impact_id=f"coastal-impact:{impact_key}",
            forecast_id=forecast.forecast_id,
            coastline_dataset=request.coastline.metadata,
            segments=tuple(segments),
            uncertainty=forecast.uncertainty,
            analysis=self.component_metadata(),
            coastline_source=request.coastline.source,
            contact_method=self._config.contact_method,
            created_at=forecast.issued_at,
        )

    def component_metadata(self) -> ComponentMetadata:
        config_hash = sha256(self._config.model_dump_json().encode()).hexdigest()
        return ComponentMetadata(
            name=self._component_name,
            version="1",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="shapely-rasterio",
            configuration_sha256=config_hash,
            attributes=(
                MetadataEntry(key="contact_method", value=self._config.contact_method),
                MetadataEntry(
                    key="arrival_estimate_policy",
                    value=self._config.arrival_estimate_policy,
                ),
                MetadataEntry(key="scientific_validation", value=self._config.validation_note),
            ),
        )

    def _estimated_arrival(self, arrivals: tuple[datetime, ...]) -> datetime | None:
        if not arrivals or self._config.arrival_estimate_policy == "none":
            return None
        midpoint = len(arrivals) // 2
        if len(arrivals) % 2:
            return arrivals[midpoint]
        return arrivals[midpoint - 1] + (arrivals[midpoint] - arrivals[midpoint - 1]) / 2

    def _project(self, geometry: SpatialGeometry) -> Any:
        projected = shape(
            transform_geom(
                self._crs_text(geometry),
                self._target_crs,
                mapping(self._shapely(geometry)),
                precision=-1,
            )
        )
        if projected.is_empty or not projected.is_valid:
            raise ValueError("coastal-impact geometry must be non-empty and valid")
        return projected

    @staticmethod
    def _crs_text(geometry: SpatialGeometry) -> str:
        if geometry.crs.authority and geometry.crs.code:
            return f"{geometry.crs.authority}:{geometry.crs.code}"
        if geometry.crs.wkt:
            return geometry.crs.wkt
        raise ValueError("coastal-impact geometry CRS has no usable identifier")

    @staticmethod
    def _polygon(value: PolygonGeometry) -> Polygon:
        return Polygon(
            [(point.x, point.y) for point in value.exterior],
            [[(point.x, point.y) for point in ring] for ring in value.holes],
        )

    @classmethod
    def _shapely(cls, value: SpatialGeometry) -> Any:
        geometry = value.geometry
        if geometry.type == "Point":
            return Point(geometry.coordinate.x, geometry.coordinate.y)
        if geometry.type == "LineString":
            return LineString([(point.x, point.y) for point in geometry.coordinates])
        if geometry.type == "Polygon":
            return cls._polygon(geometry)
        return MultiPolygon([cls._polygon(polygon) for polygon in geometry.polygons])


def create_geometry_coastal_impact(
    config: ComponentConfig,
) -> GeometryCoastalImpactAnalyzer:
    settings = GeometryCoastalImpactConfig.model_validate(config.settings)
    return GeometryCoastalImpactAnalyzer(settings, component_name=config.name)
