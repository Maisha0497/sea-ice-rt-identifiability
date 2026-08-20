from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from .common import PhysicalState


def _scalar(value, label: str) -> float:
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.size != 1:
        raise RuntimeError(
            f"Expected one value for {label}, but SMRT returned shape {np.asarray(value).shape}."
        )
    result = float(arr)
    if not np.isfinite(result):
        raise RuntimeError(f"SMRT returned a non-finite value for {label}.")
    return result


def smrt_metadata() -> dict[str, str]:
    import smrt
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("smrt")
    except PackageNotFoundError:
        installed_version = "unknown"

    return {
        "version": installed_version,
        "module_file": str(Path(smrt.__file__).resolve()),
    }


def _build_surface(state: PhysicalState, cfg: Mapping):
    from smrt.core.interface import make_interface

    model_cfg = cfg["model"]
    fixed = cfg["fixed_scene"]

    return make_interface(
        model_cfg["surface_model"],
        roughness_rms=float(state.roughness_rms_m),
        corr_length=float(fixed["surface_corr_length_m"]),
        autocorrelation_function=model_cfg["surface_autocorrelation"],

        # For IEM-Fung-1992, return NaN rather than merely printing a
        # validity warning when ks < 3 or ks*kl < sqrt(eps_r) is violated.
        # The calling code rejects non-finite outputs, so invalid surface
        # states cannot silently enter sweeps, Jacobians, or inversions.
        warning_handling="nan",
    )


def _build_ice(state: PhysicalState, cfg: Mapping):
    from smrt import PSU, make_ice_column

    model_cfg = cfg["model"]
    fixed = cfg["fixed_scene"]
    surface = _build_surface(state, cfg)

    return make_ice_column(
        ice_type=model_cfg["ice_type"],
        thickness=[float(state.ice_thickness_m)],
        temperature=[float(fixed["temperature_k"])],
        microstructure_model=model_cfg["microstructure_model"],
        salinity=[float(state.salinity_ppt) * PSU],
        corr_length=[float(fixed["ice_corr_length_m"])],
        brine_inclusion_shape=model_cfg["brine_inclusion_shape"],
        add_water_substrate=True,
        water_temperature=float(fixed["water_temperature_k"]),
        water_salinity=float(fixed["water_salinity_ppt"]) * PSU,
        surface=surface,
    )


@lru_cache(maxsize=4)
def _model(emmodel: str, rtsolver: str):
    from smrt import make_model

    return make_model(emmodel, rtsolver)


def _extract_channel(result, tx: str, rx: str, label: str) -> float:
    """
    Extract one polarization channel from a result produced for one frequency.

    Because each L/S simulation uses a separate single-frequency sensor,
    the returned Result may not retain ``frequency`` as a selectable
    coordinate. Selecting only the polarization coordinates is therefore
    correct and avoids an xarray KeyError.
    """
    value = result.sigma_dB(
        polarization_inc=tx,
        polarization=rx,
    )
    return _scalar(value, label)


def run_forward(state: PhysicalState, cfg: Mapping) -> dict[str, float]:
    """
    Run one SMRT state and return six radar backscatter channels in dB.

    Channel convention:
      HH: transmit H, receive H
      HV: transmit H, receive V
      VV: transmit V, receive V
    """
    from smrt import sensor_list

    angle = float(cfg["sensor"]["incidence_angle_deg"])
    frequencies = cfg["sensor"]["frequencies_hz"]
    model_cfg = cfg["model"]

    medium = _build_ice(state, cfg)
    model = _model(model_cfg["emmodel"], model_cfg["rtsolver"])

    prediction: dict[str, float] = {}

    for band in ("L", "S"):
        frequency = float(frequencies[band])
        sensor = sensor_list.active(
            frequency=frequency,
            theta_inc=angle,
            polarization_inc=["H", "V"],
            polarization=["H", "V"],
            name=f"{band}_band",
        )

        result = model.run(sensor, medium)

        prediction[f"{band}_HH"] = _extract_channel(result, "H", "H", f"{band}_HH")
        prediction[f"{band}_HV"] = _extract_channel(result, "H", "V", f"{band}_HV")
        prediction[f"{band}_VV"] = _extract_channel(result, "V", "V", f"{band}_VV")

    return prediction
