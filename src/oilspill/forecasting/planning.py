"""Deterministic member planning; replay uses saved requests without consulting an RNG."""

import random
from datetime import timedelta
from hashlib import sha256

from pydantic import Field, model_validator
from rasterio.warp import transform

from oilspill.domain.common import ComponentMetadata, FrozenModel, MetadataEntry, TimeRange
from oilspill.domain.geospatial import Coordinate, PointGeometry, SpatialGeometry
from oilspill.forecasting.ensemble import _seconds
from oilspill.forecasting.monte_carlo_config import MonteCarloConfig
from oilspill.requests import DriftForecastRequest, EnsembleForecastRequest


class PlannedMember(FrozenModel):
    member_index: int = Field(ge=0)
    member_id: str
    root_seed: int
    engine_seed: int = Field(ge=0, lt=2**32)
    forcing_member_id: str
    sampled_values: tuple[MetadataEntry, ...]
    position_offset_metres: Coordinate
    time_offset_seconds: float
    request: DriftForecastRequest


class EnsemblePlan(FrozenModel):
    schema_version: str = "1"
    planner_version: str = "independent-python-random-v1"
    plan_id: str
    root: EnsembleForecastRequest
    config: MonteCarloConfig
    drift: ComponentMetadata
    members: tuple[PlannedMember, ...]

    @model_validator(mode="after")
    def member_invariants(self) -> "EnsemblePlan":
        if len(self.members) != self.config.member_count:
            raise ValueError("plan count mismatch")
        if len({m.engine_seed for m in self.members}) != len(self.members):
            raise ValueError("duplicate engine seed")
        for i, member in enumerate(self.members):
            if (
                member.member_index != i
                or member.root_seed != self.root.random_seed
                or member.engine_seed != member.request.random_seed
            ):
                raise ValueError("member identity/seed mismatch")
        if len({m.member_id for m in self.members}) != len(self.members):
            raise ValueError("duplicate member id")
        return self


def plan_ensemble(
    config: MonteCarloConfig, root: EnsembleForecastRequest, drift: ComponentMetadata
) -> EnsemblePlan:
    if drift.configuration_sha256 is None:
        raise ValueError("drift configuration checksum is required for replay")
    horizon = root.observation.observed_at + timedelta(seconds=_seconds(config.horizon))
    if root.horizons:
        if len(root.horizons) != 1:
            raise ValueError("initial Monte Carlo planner supports one horizon per plan")
        horizon = root.horizons[0]
    if horizon <= root.observation.observed_at:
        raise ValueError("forecast horizon must follow observation")
    identity = sha256(
        (root.model_dump_json() + config.model_dump_json() + drift.model_dump_json()).encode()
    ).hexdigest()
    # Separate seed streams: sampling order cannot consume or change engine-seed generation.
    rng = random.Random(int(sha256((identity + ":sampling").encode()).hexdigest(), 16))
    seed_base = int(sha256((identity + ":engine").encode()).hexdigest()[:8], 16)
    available = {f.field_id: f for f in root.forcing}
    if len(available) != len(root.forcing):
        raise ValueError("duplicate canonical forcing identifiers")
    for choice in config.forcing.choices:
        if set(choice.field_ids).difference(available):
            raise ValueError("unknown forcing field in configured member")
    position = root.observation.geometry.centroid
    if position.geometry.type != "Point":
        raise ValueError("forecast centroid must be a point")
    members = []
    for index in range(config.member_count):
        sampled = []
        offsets = {"offset_x_metres": 0.0, "offset_y_metres": 0.0, "offset_seconds": 0.0}
        parameters: list[MetadataEntry] = []
        internal: list[MetadataEntry] = []
        for uncertainty in config.uncertainties:
            distribution = uncertainty.distribution
            value = (
                distribution.value
                if distribution.kind == "fixed"
                else rng.uniform(distribution.lower, distribution.upper)
            )
            sampled.append(MetadataEntry(key=uncertainty.source_id, value=value))
            if uncertainty.parameter_key is not None:
                entry = MetadataEntry(key=uncertainty.parameter_key, value=value)
                (
                    internal if uncertainty.representation == "engine_internal" else parameters
                ).append(entry)
            else:
                offsets[uncertainty.target] = value
        x, y = position.geometry.coordinate.x, position.geometry.coordinate.y
        if offsets["offset_x_metres"] or offsets["offset_y_metres"]:
            source = position.crs.wkt or f"{position.crs.authority}:{position.crs.code}"
            metric = config.grid.crs.wkt or f"{config.grid.crs.authority}:{config.grid.crs.code}"
            xs, ys = transform(source, metric, [x], [y])
            xs, ys = transform(
                metric,
                source,
                [xs[0] + offsets["offset_x_metres"]],
                [ys[0] + offsets["offset_y_metres"]],
            )
            x, y = xs[0], ys[0]
        initial = SpatialGeometry(
            geometry=PointGeometry(
                coordinate=Coordinate(
                    x=x,
                    y=y,
                )
            ),
            crs=position.crs,
        )
        timestamp = root.observation.observed_at + timedelta(seconds=offsets["offset_seconds"])
        if timestamp >= horizon:
            raise ValueError(
                "sampled release must precede horizon; no clipping/resampling is applied"
            )
        forcing = (
            config.forcing.choices[0]
            if config.forcing.kind == "fixed"
            else rng.choice(config.forcing.choices)
        )
        engine_seed = (seed_base + index) % 2**32
        member_id = f"{identity}:member:{index}"
        # Forecast initialization is sampled around the observation, not historical discharge.
        release = root.release.model_copy(
            update={
                "release_id": member_id,
                "geometry": initial,
                "interval": TimeRange(start=timestamp, end=timestamp),
                "assumptions": (
                    *root.release.assumptions,
                    "Forecast initialization hypothesis around observed centroid/time; "
                    "DATASET_VALIDATION_REQUIRED",
                    config.validation_note,
                ),
            }
        )
        request = DriftForecastRequest(
            observation=root.observation,
            release=release,
            forcing=tuple(available[f] for f in forcing.field_ids),
            valid_until=horizon,
            random_seed=engine_seed,
            initial_position=initial,
            initial_timestamp=timestamp,
            model_parameters=tuple(parameters),
            configured_uncertainty=tuple(internal),
        )
        members.append(
            PlannedMember(
                member_index=index,
                member_id=member_id,
                root_seed=root.random_seed,
                engine_seed=engine_seed,
                forcing_member_id=forcing.member_id,
                sampled_values=tuple(sampled),
                position_offset_metres=Coordinate(
                    x=offsets["offset_x_metres"], y=offsets["offset_y_metres"]
                ),
                time_offset_seconds=offsets["offset_seconds"],
                request=request,
            )
        )
    return EnsemblePlan(
        plan_id=identity, root=root, config=config, drift=drift, members=tuple(members)
    )
