from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml


CHANNELS = ("L_HH", "L_HV", "L_VV", "S_HH", "S_HV", "S_VV")


@dataclass(frozen=True)
class PhysicalState:
    roughness_rms_m: float
    salinity_ppt: float
    ice_thickness_m: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "PhysicalState":
        return cls(
            roughness_rms_m=float(values["roughness_rms_m"]),
            salinity_ppt=float(values["salinity_ppt"]),
            ice_thickness_m=float(values["ice_thickness_m"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "roughness_rms_m": self.roughness_rms_m,
            "salinity_ppt": self.salinity_ppt,
            "ice_thickness_m": self.ice_thickness_m,
        }

    def replace(self, **updates: float) -> "PhysicalState":
        values = self.as_dict()
        values.update({key: float(value) for key, value in updates.items()})
        return PhysicalState.from_mapping(values)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]



def external_asar_root() -> Path:
    """Return the user-supplied ASAR/UAVSAR data root.

    Set ``RTE_PINN_ASAR_ROOT`` to the directory that contains the original
    ``Originals`` and ``ASAR+RCM Processed`` folders.  If the variable is not
    set, the code looks under ``data_external/ASAR`` inside the project.
    Raw satellite data are intentionally not distributed with this repository.
    """
    value = os.environ.get("RTE_PINN_ASAR_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return project_root() / "data_external" / "ASAR"


def derived_data_dir(*parts: str) -> Path:
    """Project-local directory for regenerated observation products.

    ``data/derived`` is git-ignored because these files are generated from
    external UAVSAR/ROI inputs.
    """
    path = project_root() / "data" / "derived"
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else project_root() / "config" / "base.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping) -> None:
    missing = []
    for key in ("sensor", "model", "fixed_scene", "baseline_state", "inversion", "noise_std_db"):
        if key not in cfg:
            missing.append(key)
    if missing:
        raise KeyError(f"Missing configuration section(s): {missing}")

    for channel in CHANNELS:
        if channel not in cfg["noise_std_db"]:
            raise KeyError(f"Missing noise standard deviation for {channel}")
        if float(cfg["noise_std_db"][channel]) <= 0:
            raise ValueError(f"Noise standard deviation for {channel} must be positive.")


def baseline_state(cfg: Mapping) -> PhysicalState:
    return PhysicalState.from_mapping(cfg["baseline_state"])


def vector_from_prediction(
    prediction: Mapping[str, float],
    channels: Iterable[str] = CHANNELS,
) -> np.ndarray:
    return np.asarray([float(prediction[ch]) for ch in channels], dtype=float)


def noise_vector(
    cfg: Mapping,
    channels: Iterable[str] = CHANNELS,
) -> np.ndarray:
    return np.asarray([float(cfg["noise_std_db"][ch]) for ch in channels], dtype=float)


def ensure_results_dir() -> Path:
    path = project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
