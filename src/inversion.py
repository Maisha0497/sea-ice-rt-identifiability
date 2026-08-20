from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .common import CHANNELS, PhysicalState, noise_vector, vector_from_prediction
from .forward_smrt import run_forward


@dataclass
class InversionResult:
    state: PhysicalState
    cost: float
    success: bool
    message: str
    residual_whitened: np.ndarray
    predicted_db: np.ndarray
    nfev: int


def _state_from_log_vector(
    log_values: np.ndarray,
    template: PhysicalState,
    parameters: Sequence[str],
) -> PhysicalState:
    updates = {
        parameter: float(np.exp(value))
        for parameter, value in zip(parameters, log_values, strict=True)
    }
    return template.replace(**updates)


def invert_observation(
    observed_db: np.ndarray,
    cfg: Mapping,
    initial_state: PhysicalState,
    channels: Sequence[str] = CHANNELS,
    parameters: Sequence[str] | None = None,
) -> InversionResult:
    parameters = tuple(parameters or cfg["inversion"]["parameters"])
    observed = np.asarray(observed_db, dtype=float)
    sigma = noise_vector(cfg, channels)

    if observed.shape != (len(channels),):
        raise ValueError(
            f"Observed vector must have shape {(len(channels),)}, got {observed.shape}."
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed vector contains non-finite values.")

    bounds_cfg = cfg["inversion"]["bounds"]
    lower_raw = np.asarray([float(bounds_cfg[p][0]) for p in parameters], dtype=float)
    upper_raw = np.asarray([float(bounds_cfg[p][1]) for p in parameters], dtype=float)

    if np.any(lower_raw <= 0) or np.any(upper_raw <= lower_raw):
        raise ValueError("Log-space bounds must be positive and ordered.")

    x0 = np.log([initial_state.as_dict()[p] for p in parameters])
    lower = np.log(lower_raw)
    upper = np.log(upper_raw)
    x0 = np.clip(x0, lower, upper)

    def residual(log_values: np.ndarray) -> np.ndarray:
        state = _state_from_log_vector(log_values, initial_state, parameters)
        predicted = vector_from_prediction(run_forward(state, cfg), channels)
        return (predicted - observed) / sigma

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
    )

    fitted_state = _state_from_log_vector(result.x, initial_state, parameters)
    predicted = vector_from_prediction(run_forward(fitted_state, cfg), channels)
    residual_w = (predicted - observed) / sigma

    return InversionResult(
        state=fitted_state,
        cost=float(0.5 * np.dot(residual_w, residual_w)),
        success=bool(result.success),
        message=str(result.message),
        residual_whitened=residual_w,
        predicted_db=predicted,
        nfev=int(result.nfev),
    )


def multistart_inversion(
    observed_db: np.ndarray,
    cfg: Mapping,
    template_state: PhysicalState,
    channels: Sequence[str] = CHANNELS,
    parameters: Sequence[str] | None = None,
) -> list[InversionResult]:
    parameters = tuple(parameters or cfg["inversion"]["parameters"])
    bounds_cfg = cfg["inversion"]["bounds"]
    count = int(cfg["inversion"].get("multistart_count", 8))
    seed = int(cfg["inversion"].get("random_seed", 42))
    rng = np.random.default_rng(seed)

    results: list[InversionResult] = []

    # Include the configured baseline as one deterministic start.
    results.append(
        invert_observation(
            observed_db,
            cfg,
            initial_state=template_state,
            channels=channels,
            parameters=parameters,
        )
    )

    for _ in range(max(0, count - 1)):
        updates = {}
        for parameter in parameters:
            low, high = map(float, bounds_cfg[parameter])
            updates[parameter] = float(np.exp(rng.uniform(np.log(low), np.log(high))))

        initial = template_state.replace(**updates)
        results.append(
            invert_observation(
                observed_db,
                cfg,
                initial_state=initial,
                channels=channels,
                parameters=parameters,
            )
        )

    return sorted(results, key=lambda item: item.cost)
