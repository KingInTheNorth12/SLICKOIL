"""Pure NumPy finite-volume kernel with no file, CRS, or provider dependencies."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from oilspill.eulerian.errors import EulerianStabilityError, EulerianTransportError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Boundary = Literal["closed", "advective_outflow"]
LossIntegration = Literal["exponential", "explicit"]
VelocityProvider = Callable[[float], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True)
class KernelResult:
    snapshots: tuple[FloatArray, ...]
    initial_mass: float
    final_mass: float
    minimum_concentration: float
    maximum_concentration: float
    maximum_advective_cfl: float
    maximum_diffusive_cfl: float
    number_of_time_steps: int
    clipped_negative_cell_count: int


def _validate_arrays(c: FloatArray, u: FloatArray, v: FloatArray, land: BoolArray) -> None:
    if c.ndim != 2 or u.shape != c.shape or v.shape != c.shape or land.shape != c.shape:
        raise EulerianTransportError(
            "concentration, velocity, and land arrays must share a 2-D shape"
        )
    if not np.isfinite(c).all() or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise EulerianTransportError("concentration and velocity values must be finite")


def stable_time_step(
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    *,
    dx_metres: float,
    dy_metres: float,
    diffusivity_m2_s: float,
    maximum_dt_seconds: float,
    maximum_advective_cfl: float,
    maximum_diffusive_cfl: float,
    loss_rate_s: float | None = None,
    loss_integration: LossIntegration = "exponential",
) -> float:
    scalars = (
        dx_metres,
        dy_metres,
        diffusivity_m2_s,
        maximum_dt_seconds,
        maximum_advective_cfl,
        maximum_diffusive_cfl,
        *((loss_rate_s,) if loss_rate_s is not None else ()),
    )
    if not all(math.isfinite(value) for value in scalars):
        raise EulerianStabilityError("time-step inputs must be finite")
    if dx_metres <= 0 or dy_metres <= 0 or diffusivity_m2_s < 0 or maximum_dt_seconds <= 0:
        raise EulerianStabilityError("grid spacing, time step, and diffusivity are nonsensical")
    if not 0 < maximum_advective_cfl <= 1 or not 0 < maximum_diffusive_cfl <= 0.5:
        raise EulerianStabilityError("configured CFL limits exceed explicit-scheme bounds")
    if loss_rate_s is not None and loss_rate_s < 0:
        raise EulerianStabilityError("loss rate cannot be negative")
    if velocity_x.shape != velocity_y.shape or not (
        np.isfinite(velocity_x).all() and np.isfinite(velocity_y).all()
    ):
        raise EulerianStabilityError("velocity arrays must have equal shape and finite values")
    advective_rate = (
        float(np.max(np.abs(velocity_x))) / dx_metres
        + float(np.max(np.abs(velocity_y))) / dy_metres
    )
    diffusive_rate = diffusivity_m2_s * (1 / dx_metres**2 + 1 / dy_metres**2)
    candidates = [maximum_dt_seconds]
    if advective_rate > 0:
        candidates.append(maximum_advective_cfl / advective_rate)
    if diffusive_rate > 0:
        candidates.append(maximum_diffusive_cfl / diffusive_rate)
    if loss_integration == "explicit" and loss_rate_s is not None and loss_rate_s > 0:
        candidates.append(1 / loss_rate_s)
    step = min(candidates)
    if not math.isfinite(step) or step <= 0:
        raise EulerianStabilityError("could not construct a stable positive time step")
    return step


def finite_volume_step(
    concentration: FloatArray,
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    land_mask: BoolArray,
    *,
    dx_metres: float,
    dy_metres: float,
    diffusivity_m2_s: float,
    dt_seconds: float,
    loss_rate_s: float | None,
    boundary: Boundary,
    loss_integration: LossIntegration,
) -> tuple[FloatArray, int]:
    c = np.asarray(concentration, dtype=np.float64)
    u = np.asarray(velocity_x, dtype=np.float64)
    v = np.asarray(velocity_y, dtype=np.float64)
    land = np.asarray(land_mask, dtype=np.bool_)
    _validate_arrays(c, u, v, land)
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise EulerianTransportError("finite-volume step requires a finite positive dt")
    water = ~land
    c = np.where(water, c, 0.0)
    height, width = c.shape
    fx = np.zeros((height, width + 1), dtype=np.float64)
    fy = np.zeros((height + 1, width), dtype=np.float64)
    if width > 1:
        open_x = water[:, :-1] & water[:, 1:]
        face_u = 0.5 * (u[:, :-1] + u[:, 1:])
        adv = face_u * np.where(face_u >= 0, c[:, :-1], c[:, 1:])
        diff = -diffusivity_m2_s * (c[:, 1:] - c[:, :-1]) / dx_metres
        fx[:, 1:-1] = np.where(open_x, adv + diff, 0.0)
    if height > 1:
        open_y = water[:-1, :] & water[1:, :]
        face_v = 0.5 * (v[:-1, :] + v[1:, :])
        adv = face_v * np.where(face_v >= 0, c[1:, :], c[:-1, :])
        diff = -diffusivity_m2_s * (c[:-1, :] - c[1:, :]) / dy_metres
        fy[1:-1, :] = np.where(open_y, adv + diff, 0.0)
    if boundary == "advective_outflow":
        fx[:, 0] = np.where(water[:, 0] & (u[:, 0] < 0), u[:, 0] * c[:, 0], 0.0)
        fx[:, -1] = np.where(water[:, -1] & (u[:, -1] > 0), u[:, -1] * c[:, -1], 0.0)
        fy[0, :] = np.where(water[0, :] & (v[0, :] > 0), v[0, :] * c[0, :], 0.0)
        fy[-1, :] = np.where(water[-1, :] & (v[-1, :] < 0), v[-1, :] * c[-1, :], 0.0)
    divergence = (fx[:, 1:] - fx[:, :-1]) / dx_metres + (fy[:-1] - fy[1:]) / dy_metres
    updated = c - dt_seconds * divergence
    if loss_rate_s is not None and loss_rate_s > 0:
        factor = (
            math.exp(-loss_rate_s * dt_seconds)
            if loss_integration == "exponential"
            else 1 - loss_rate_s * dt_seconds
        )
        updated *= factor
    negative = (updated < 0) & water
    return np.where(water, np.maximum(updated, 0), 0.0), int(np.count_nonzero(negative))


def integrate_transport(
    initial_concentration: FloatArray,
    land_mask: BoolArray,
    velocity_at_seconds: VelocityProvider,
    snapshot_seconds: tuple[float, ...],
    *,
    dx_metres: float,
    dy_metres: float,
    diffusivity_m2_s: float,
    loss_rate_s: float | None,
    maximum_dt_seconds: float,
    maximum_advective_cfl: float,
    maximum_diffusive_cfl: float,
    boundary: Boundary,
    loss_integration: LossIntegration,
) -> KernelResult:
    if not snapshot_seconds or tuple(sorted(snapshot_seconds)) != snapshot_seconds:
        raise EulerianTransportError("snapshot offsets must be non-empty and chronological")
    if len(set(snapshot_seconds)) != len(snapshot_seconds) or snapshot_seconds[0] <= 0:
        raise EulerianTransportError("snapshot offsets must be unique and positive")
    c = np.asarray(initial_concentration, dtype=np.float64).copy()
    land = np.asarray(land_mask, dtype=np.bool_)
    zeros = np.zeros_like(c)
    _validate_arrays(c, zeros, zeros, land)
    if np.any(c < 0):
        raise EulerianTransportError("initial concentration must be non-negative")
    c[land] = 0
    area = dx_metres * dy_metres
    initial_mass = float(c.sum() * area)
    if initial_mass <= 0:
        raise EulerianTransportError("initial concentration contains no water-cell mass")
    outputs: list[FloatArray] = []
    elapsed = 0.0
    steps = clipped = 0
    max_adv = max_diff = 0.0
    minimum, maximum = float(c.min()), float(c.max())
    for target in snapshot_seconds:
        while elapsed < target:
            remaining = target - elapsed
            if remaining <= max(1.0, target) * 1e-12:
                elapsed = target
                break
            u0, v0 = velocity_at_seconds(elapsed)
            stable = stable_time_step(
                u0,
                v0,
                dx_metres=dx_metres,
                dy_metres=dy_metres,
                diffusivity_m2_s=diffusivity_m2_s,
                maximum_dt_seconds=maximum_dt_seconds,
                maximum_advective_cfl=maximum_advective_cfl,
                maximum_diffusive_cfl=maximum_diffusive_cfl,
                loss_rate_s=loss_rate_s,
                loss_integration=loss_integration,
            )
            dt = min(stable, remaining)
            u, v = velocity_at_seconds(elapsed + dt / 2)
            adv_cfl = dt * (
                float(np.max(np.abs(u))) / dx_metres + float(np.max(np.abs(v))) / dy_metres
            )
            diff_cfl = dt * diffusivity_m2_s * (1 / dx_metres**2 + 1 / dy_metres**2)
            if adv_cfl > maximum_advective_cfl + 1e-12 or diff_cfl > maximum_diffusive_cfl + 1e-12:
                raise EulerianStabilityError("interpolated forcing would violate a CFL limit")
            c, clipped_now = finite_volume_step(
                c,
                u,
                v,
                land,
                dx_metres=dx_metres,
                dy_metres=dy_metres,
                diffusivity_m2_s=diffusivity_m2_s,
                dt_seconds=dt,
                loss_rate_s=loss_rate_s,
                boundary=boundary,
                loss_integration=loss_integration,
            )
            elapsed += dt
            steps += 1
            clipped += clipped_now
            max_adv, max_diff = max(max_adv, adv_cfl), max(max_diff, diff_cfl)
            minimum, maximum = min(minimum, float(c.min())), max(maximum, float(c.max()))
        outputs.append(c.copy())
    return KernelResult(
        tuple(outputs),
        initial_mass,
        float(c.sum() * area),
        minimum,
        maximum,
        max_adv,
        max_diff,
        steps,
        clipped,
    )
