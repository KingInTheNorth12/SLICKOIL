"""Pure NumPy mismatch metrics and deterministic coarse-candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oilspill.domain.eulerian import EulerianGrid
from oilspill.domain.hybrid import CoarseSourceEvaluation, HybridSourceState
from oilspill.hybrid.config import HybridMismatchWeights

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FieldMismatch:
    soft_iou: float
    centroid_error_metres: float
    centroid_error: float
    area_error_square_metres: float
    area_error: float
    shape_error: float


def _clean_normalized(values: FloatArray) -> FloatArray:
    clean = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(clean.sum())
    return np.asarray(clean / total if total > 0.0 else clean, dtype=np.float64)


def soft_iou(predicted: FloatArray, observed: FloatArray) -> float:
    predicted_n, observed_n = _clean_normalized(predicted), _clean_normalized(observed)
    denominator = float(np.maximum(predicted_n, observed_n).sum())
    if denominator == 0.0:
        return 1.0
    return float(np.clip(np.minimum(predicted_n, observed_n).sum() / denominator, 0.0, 1.0))


def _centers(grid: EulerianGrid) -> tuple[FloatArray, FloatArray]:
    columns = np.arange(grid.width, dtype=np.float64) + 0.5
    rows = np.arange(grid.height, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(
        grid.transform.a * columns + grid.transform.c,
        grid.transform.e * rows + grid.transform.f,
    )
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


def _centroid(values: FloatArray, xx: FloatArray, yy: FloatArray) -> tuple[float, float] | None:
    total = float(values.sum())
    if total <= 0.0:
        return None
    return float((values * xx).sum() / total), float((values * yy).sum() / total)


def _support(values: FloatArray, threshold: float) -> NDArray[np.bool_]:
    maximum = float(values.max(initial=0.0))
    return np.asarray(values >= threshold * maximum, dtype=np.bool_) if maximum > 0 else (
        np.zeros(values.shape, dtype=np.bool_)
    )


def _covariance(
    values: FloatArray, support: NDArray[np.bool_], xx: FloatArray, yy: FloatArray
) -> FloatArray:
    weights = np.where(support, values, 0.0)
    center = _centroid(weights, xx, yy)
    if center is None:
        return np.zeros((2, 2), dtype=np.float64)
    total = float(weights.sum())
    dx, dy = xx - center[0], yy - center[1]
    return np.asarray(
        [
            [(weights * dx * dx).sum() / total, (weights * dx * dy).sum() / total],
            [(weights * dx * dy).sum() / total, (weights * dy * dy).sum() / total],
        ],
        dtype=np.float64,
    )


def field_mismatch(
    predicted: FloatArray,
    observed: FloatArray,
    grid: EulerianGrid,
    *,
    support_threshold: float,
) -> FieldMismatch:
    if predicted.shape != observed.shape or predicted.shape != (grid.height, grid.width):
        raise ValueError("predicted and observed fields must match the Eulerian grid")
    predicted_n, observed_n = _clean_normalized(predicted), _clean_normalized(observed)
    xx, yy = _centers(grid)
    observed_center = _centroid(observed_n, xx, yy)
    if observed_center is None:
        raise ValueError("observed field must contain positive support")
    predicted_center = _centroid(predicted_n, xx, yy)
    if predicted_center is None:
        grid_center = (
            (float(xx.min()) + float(xx.max())) / 2,
            (float(yy.min()) + float(yy.max())) / 2,
        )
        centroid_metres = float(np.linalg.norm(np.subtract(grid_center, observed_center)))
    else:
        centroid_metres = float(np.linalg.norm(np.subtract(predicted_center, observed_center)))
    observed_dx = xx - observed_center[0]
    observed_dy = yy - observed_center[1]
    rms_observed_extent = float(
        np.sqrt((observed_n * (observed_dx**2 + observed_dy**2)).sum())
    )
    length_reference = max(rms_observed_extent, grid.dx_metres, grid.dy_metres)
    observed_support = _support(observed_n, support_threshold)
    predicted_support = _support(predicted_n, support_threshold)
    cell_area = grid.dx_metres * grid.dy_metres
    observed_area = float(observed_support.sum() * cell_area)
    predicted_area = float(predicted_support.sum() * cell_area)
    area_metres = abs(predicted_area - observed_area)
    numerical_epsilon = float(np.finfo(np.float64).eps)
    area_error = area_metres / max(observed_area, numerical_epsilon)
    observed_covariance = _covariance(observed_n, observed_support, xx, yy)
    predicted_covariance = _covariance(predicted_n, predicted_support, xx, yy)
    shape_norm = float(np.linalg.norm(observed_covariance, ord="fro"))
    shape_error = float(
        np.linalg.norm(predicted_covariance - observed_covariance, ord="fro")
        / max(shape_norm, numerical_epsilon)
    )
    return FieldMismatch(
        soft_iou=soft_iou(predicted_n, observed_n),
        centroid_error_metres=centroid_metres,
        centroid_error=centroid_metres / length_reference,
        area_error_square_metres=area_metres,
        area_error=area_error,
        shape_error=shape_error,
    )


def evaluate_source(
    state: HybridSourceState,
    eulerian_result_id: str,
    mismatch: FieldMismatch,
    physics_penalty: float,
    weights: HybridMismatchWeights,
) -> CoarseSourceEvaluation:
    total = (
        weights.soft_iou * (1.0 - mismatch.soft_iou)
        + weights.centroid * mismatch.centroid_error
        + weights.area * mismatch.area_error
        + weights.shape * mismatch.shape_error
        + weights.physics_penalty * physics_penalty
    )
    return CoarseSourceEvaluation(
        source_state=state,
        eulerian_result_id=eulerian_result_id,
        soft_iou=mismatch.soft_iou,
        centroid_error_metres=mismatch.centroid_error_metres,
        centroid_error=mismatch.centroid_error,
        area_error_square_metres=mismatch.area_error_square_metres,
        area_error=mismatch.area_error,
        covariance_shape_error=mismatch.shape_error,
        physics_penalty=physics_penalty,
        total_mismatch_score=total,
        retained=False,
    )


def select_evaluations(
    evaluations: tuple[CoarseSourceEvaluation, ...],
    *,
    top_k: int,
    absolute_score_threshold: float,
) -> tuple[tuple[CoarseSourceEvaluation, ...], int]:
    absolute_ids = {
        item.source_state.source_state_id
        for item in evaluations
        if item.total_mismatch_score <= absolute_score_threshold
    }
    ranked = sorted(
        evaluations,
        key=lambda item: (item.total_mismatch_score, item.source_state.source_state_id),
    )
    retained_ids = absolute_ids | {
        item.source_state.source_state_id for item in ranked[:top_k]
    }
    selected = tuple(
        item.model_copy(
            update={"retained": item.source_state.source_state_id in retained_ids}
        )
        for item in evaluations
    )
    return selected, len(absolute_ids)
