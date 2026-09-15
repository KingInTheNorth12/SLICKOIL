"""DriftModel-only high-fidelity evaluation of neutral hybrid source states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from oilspill.domain.common import MetadataEntry, TimeRange
from oilspill.domain.drift import DriftMode, DriftSimulation, ReleaseHypothesis, SimulationStatus
from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.hybrid import HybridSourceState
from oilspill.eulerian.grid import _mapping
from oilspill.eulerian.initialization import affine, crs_text, read_land_mask
from oilspill.hybrid.config import SelectiveRefinementConfig
from oilspill.hybrid.scoring import FieldMismatch, field_mismatch
from oilspill.ports import DriftModel
from oilspill.requests import ForwardTraceRequest, HybridHindcastRequest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RawHighFidelityEvaluation:
    source_state: HybridSourceState
    simulation: DriftSimulation
    mismatch: FieldMismatch
    mismatch_score: float
    generation_number: int


def rasterize_distribution(
    simulation: DriftSimulation,
    grid: EulerianGrid,
    *,
    all_touched: bool,
) -> FloatArray:
    distribution = simulation.target_distribution
    if distribution is None:
        raise ValueError("high-fidelity simulation omitted target particle distribution")
    geometry = transform_geom(
        crs_text(distribution.comparison_geometry.crs),
        crs_text(grid.crs),
        _mapping(distribution.comparison_geometry),
        precision=-1,
    )
    values = rasterize(
        [(geometry, 1.0)],
        out_shape=(grid.height, grid.width),
        transform=affine(grid),
        fill=0.0,
        dtype="float64",
        all_touched=all_touched,
    )
    values = np.asarray(values, dtype=np.float64)
    values[read_land_mask(grid)] = 0.0
    return values


def evaluate_high_fidelity_state(
    state: HybridSourceState,
    generation_number: int,
    request: HybridHindcastRequest,
    grid: EulerianGrid,
    observed: FloatArray,
    config: SelectiveRefinementConfig,
    drift: DriftModel,
    *,
    random_seed: int,
) -> RawHighFidelityEvaluation:
    release = ReleaseHypothesis(
        release_id=f"hybrid-release:{state.source_state_id}",
        geometry=state.geometry,
        interval=TimeRange(start=state.release_time, end=state.release_time),
        source_candidate_id=None,
        assumptions=(
            "Physics-only source hypothesis; no vessel attribution.",
            config.validation_note,
        ),
    )
    parameters = [
        MetadataEntry(key=config.windage_parameter_key, value=state.windage_coefficient),
        MetadataEntry(
            key=config.diffusivity_parameter_key,
            value=state.horizontal_diffusivity_m2_s,
        ),
    ]
    if state.first_order_loss_rate_s is not None and config.loss_rate_parameter_key:
        parameters.append(
            MetadataEntry(
                key=config.loss_rate_parameter_key,
                value=state.first_order_loss_rate_s,
            )
        )
    simulation = drift.forward_source_trace(
        ForwardTraceRequest(
            observation=request.observation,
            candidate=None,
            release=release,
            forcing=request.forcing,
            random_seed=random_seed,
            model_parameters=tuple(parameters),
            configured_uncertainty=(),
        )
    )
    if simulation.mode != DriftMode.FORWARD_TRACE:
        raise ValueError("DriftModel returned a non-forward high-fidelity simulation")
    if simulation.status != SimulationStatus.SUCCEEDED:
        raise ValueError("high-fidelity source refinement requires a successful simulation")
    if simulation.target_timestamp != request.observation.observed_at:
        raise ValueError("high-fidelity simulation did not end at the observation time")
    predicted = rasterize_distribution(
        simulation, grid, all_touched=config.rasterization_all_touched
    )
    mismatch = field_mismatch(
        predicted,
        observed,
        grid,
        support_threshold=config.observation_support_threshold,
    )
    weights = config.mismatch_weights
    total = (
        weights.soft_iou * (1.0 - mismatch.soft_iou)
        + weights.centroid * mismatch.centroid_error
        + weights.area * mismatch.area_error
        + weights.shape * mismatch.shape_error
    )
    return RawHighFidelityEvaluation(
        source_state=state,
        simulation=simulation,
        mismatch=mismatch,
        mismatch_score=total,
        generation_number=generation_number,
    )
