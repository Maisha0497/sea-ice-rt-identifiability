from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .common import CHANNELS, PhysicalState, noise_vector, vector_from_prediction


ForwardFunction = Callable[[PhysicalState, Mapping], Mapping[str, float]]


def log_parameter_jacobian(
    state: PhysicalState,
    cfg: Mapping,
    forward: ForwardFunction,
    parameters: Sequence[str],
    channels: Sequence[str] = CHANNELS,
    log_step: float | None = None,
) -> np.ndarray:
    """
    Central finite-difference Jacobian with respect to x_j = log(m_j).

    J_ij = d y_i / d log(m_j)

    A step +/-h in log space multiplies the raw parameter by exp(+/-h).
    """
    h = float(log_step if log_step is not None else cfg["inversion"]["log_step"])
    if h <= 0:
        raise ValueError("log_step must be positive.")

    base = state.as_dict()
    jacobian = np.empty((len(channels), len(parameters)), dtype=float)

    for j, parameter in enumerate(parameters):
        value = float(base[parameter])
        if value <= 0:
            raise ValueError(
                f"{parameter}={value} cannot be perturbed in log space; it must be positive."
            )

        plus = state.replace(**{parameter: value * np.exp(h)})
        minus = state.replace(**{parameter: value * np.exp(-h)})

        y_plus = vector_from_prediction(forward(plus, cfg), channels)
        y_minus = vector_from_prediction(forward(minus, cfg), channels)

        jacobian[:, j] = (y_plus - y_minus) / (2.0 * h)

    return jacobian


def whiten_jacobian(
    jacobian: np.ndarray,
    cfg: Mapping,
    channels: Sequence[str] = CHANNELS,
) -> np.ndarray:
    sigma = noise_vector(cfg, channels)
    return jacobian / sigma[:, None]


def svd_diagnostics(jacobian_whitened: np.ndarray):
    u, singular_values, vt = np.linalg.svd(jacobian_whitened, full_matrices=False)

    if singular_values.size == 0:
        condition_number = np.inf
    elif singular_values[-1] <= np.finfo(float).eps:
        condition_number = np.inf
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    return u, singular_values, vt, condition_number
