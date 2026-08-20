#!/usr/bin/env python3
from __future__ import annotations

"""
10_myi_multiyear_reachability.py

Purpose
-------
Correct one structural flaw in the earlier reachability tests:

    the MYI observations were being compared against SMRT with
    ice_type="firstyear".

This experiment tests ONLY the MYI incidence bins using the SMRT
"multiyear" ice formulation.

SMRT's multiyear representation uses air bubbles as the scattering
inclusions in a saline-ice background, so porosity must be included as
an explicit model variable. The existing 0.5 mm correlation length is
kept fixed for this first structural test.

This is an EXISTENCE / STRUCTURE diagnostic, not a physical retrieval.

Important
---------
The porosity range used here is deliberately a broad diagnostic range,
not a literature prior. Do not report the retrieved porosity as a
measured or validated MYI property.

Inputs
------
data/derived/SMRT_observation_audit/incidence_binned/
    08_incidence_binned_observations.csv

Optional comparison input:
results/09b_incidence_aware_safe_refinement.csv

Outputs
-------
results/10_myi_multiyear_coarse_grid.csv
results/10_myi_multiyear_reachability.csv
results/10_myi_multiyear_reachability.json
results/10_myi_firstyear_vs_multiyear.csv

Run from project root:
    conda activate geo
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python experiments/10_myi_multiyear_reachability.py
"""

from copy import deepcopy
from pathlib import Path
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config
from src.forward_smrt import _build_surface, _extract_channel, _model


# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------

OBSERVATION_CSV = (
    ROOT / "data" / "derived" / "SMRT_observation_audit"
    / "incidence_binned" / "08_incidence_binned_observations.csv"
)

OLD_FIRSTYEAR_RESULTS = (
    ROOT / "results" / "09b_incidence_aware_safe_refinement.csv"
)

ALL_CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")
SUBSETS = {
    "L_only": ("L_HH", "L_VV"),
    "S_only": ("S_HH", "S_VV"),
    "L_plus_S": ALL_CHANNELS,
}


# ---------------------------------------------------------------------
# DIAGNOSTIC PARAMETER DOMAIN
# ---------------------------------------------------------------------
# Roughness / salinity / thickness match the expanded existence domain
# used in 06b / 09 / 09b.
# ---------------------------------------------------------------------

ROUGHNESS_MIN_M = 0.00025
ROUGHNESS_MAX_M = 0.00300

SALINITY_MIN_PPT = 0.5
SALINITY_MAX_PPT = 15.0

THICKNESS_MIN_M = 0.20
THICKNESS_MAX_M = 6.00

# Diagnostic only. This is NOT yet a literature prior.
POROSITY_MIN = 0.0
POROSITY_MAX = 0.30


# ---------------------------------------------------------------------
# COARSE GRID
# ---------------------------------------------------------------------
# 7 x 6 x 5 x 6 = 1260 states per incidence bin.
# There are currently 4 MYI bins, so ~5040 forward states.
# ---------------------------------------------------------------------

N_ROUGHNESS = 7
N_SALINITY = 6
N_THICKNESS = 5

POROSITY_VALUES = np.array(
    [0.00, 0.02, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)


# ---------------------------------------------------------------------
# SAFE DERIVATIVE-FREE REFINEMENT
# ---------------------------------------------------------------------

N_REFINEMENT_STARTS = 5
MAXITER = 120
INVALID_PENALTY = 1e6

# Optimization coordinates:
# z = [log(roughness), salinity, thickness, porosity]
Z_BOUNDS = [
    (math.log(ROUGHNESS_MIN_M), math.log(ROUGHNESS_MAX_M)),
    (SALINITY_MIN_PPT, SALINITY_MAX_PPT),
    (THICKNESS_MIN_M, THICKNESS_MAX_M),
    (POROSITY_MIN, POROSITY_MAX),
]


def status_from_rms(value: float) -> str:
    if value <= 2.0:
        return "near/reachable"
    if value <= 5.0:
        return "marginal"
    return "clearly outside"


def cfg_at_angle(cfg_base: dict, angle_deg: float) -> dict:
    cfg = deepcopy(cfg_base)
    cfg["sensor"]["incidence_angle_deg"] = float(angle_deg)

    # Critical structural correction.
    cfg["model"]["ice_type"] = "multiyear"

    return cfg


def x_to_z(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            math.log(float(x[0])),
            float(x[1]),
            float(x[2]),
            float(x[3]),
        ],
        dtype=float,
    )


def z_to_x(z: np.ndarray) -> np.ndarray:
    return np.array(
        [
            math.exp(float(z[0])),
            float(z[1]),
            float(z[2]),
            float(z[3]),
        ],
        dtype=float,
    )


def subset_indices(channels: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [ALL_CHANNELS.index(ch) for ch in channels],
        dtype=int,
    )


def at_bounds(x: np.ndarray) -> list[bool]:
    lower = np.array(
        [
            ROUGHNESS_MIN_M,
            SALINITY_MIN_PPT,
            THICKNESS_MIN_M,
            POROSITY_MIN,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            ROUGHNESS_MAX_M,
            SALINITY_MAX_PPT,
            THICKNESS_MAX_M,
            POROSITY_MAX,
        ],
        dtype=float,
    )

    tol = np.maximum((upper - lower) * 1e-3, 1e-12)

    return [
        bool(
            abs(x[i] - lower[i]) <= tol[i]
            or abs(x[i] - upper[i]) <= tol[i]
        )
        for i in range(4)
    ]


def run_forward_multiyear(
    state,
    porosity: float,
    cfg: dict,
) -> dict[str, float]:
    """
    Same L/S active-radar machinery as the existing project, but construct
    the ice column explicitly as SMRT 'multiyear' and pass porosity.
    """
    from smrt import PSU, make_ice_column, sensor_list

    model_cfg = cfg["model"]
    fixed = cfg["fixed_scene"]

    surface = _build_surface(state, cfg)

    medium = make_ice_column(
        ice_type="multiyear",
        thickness=[float(state.ice_thickness_m)],
        temperature=[float(fixed["temperature_k"])],
        microstructure_model=model_cfg["microstructure_model"],
        salinity=[float(state.salinity_ppt) * PSU],
        porosity=[float(porosity)],
        corr_length=[float(fixed["ice_corr_length_m"])],
        brine_inclusion_shape=model_cfg["brine_inclusion_shape"],
        add_water_substrate=True,
        water_temperature=float(fixed["water_temperature_k"]),
        water_salinity=float(fixed["water_salinity_ppt"]) * PSU,
        surface=surface,
    )

    model = _model(
        model_cfg["emmodel"],
        model_cfg["rtsolver"],
    )

    angle = float(cfg["sensor"]["incidence_angle_deg"])
    output: dict[str, float] = {}

    for band in ("L", "S"):
        frequency = float(cfg["sensor"]["frequencies_hz"][band])

        sensor = sensor_list.active(
            frequency=frequency,
            theta_inc=angle,
            polarization_inc=["H", "V"],
            polarization=["H", "V"],
            name=f"{band}_band",
        )

        result = model.run(sensor, medium)

        output[f"{band}_HH"] = _extract_channel(
            result, "H", "H", f"{band}_HH"
        )
        output[f"{band}_VV"] = _extract_channel(
            result, "V", "V", f"{band}_VV"
        )

    if not all(
        np.isfinite(output[ch])
        for ch in ALL_CHANNELS
    ):
        raise ValueError(f"Non-finite output: {output}")

    return output


def make_safe_predictor(base_state, cfg):
    cache: dict[tuple[float, ...], np.ndarray | None] = {}
    stats = {
        "runs": 0,
        "cache_hits": 0,
        "invalid": 0,
    }

    def safe_predict(x: np.ndarray) -> np.ndarray | None:
        key = tuple(round(float(v), 11) for v in x)

        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]

        stats["runs"] += 1

        try:
            state = base_state.replace(
                roughness_rms_m=float(x[0]),
                salinity_ppt=float(x[1]),
                ice_thickness_m=float(x[2]),
            )

            pred = run_forward_multiyear(
                state=state,
                porosity=float(x[3]),
                cfg=cfg,
            )

            y = np.array(
                [pred[ch] for ch in ALL_CHANNELS],
                dtype=float,
            )

            if np.any(~np.isfinite(y)):
                raise ValueError(f"Invalid prediction: {y}")

            cache[key] = y

        except Exception:
            stats["invalid"] += 1
            cache[key] = None

        return cache[key]

    return safe_predict, stats


def build_coarse_grid(
    base_state,
    cfg: dict,
    ice_type_label: str,
    incidence_bin: str,
    angle_deg: float,
) -> pd.DataFrame:
    roughness_values = np.geomspace(
        ROUGHNESS_MIN_M,
        ROUGHNESS_MAX_M,
        N_ROUGHNESS,
    )
    salinity_values = np.linspace(
        SALINITY_MIN_PPT,
        SALINITY_MAX_PPT,
        N_SALINITY,
    )
    thickness_values = np.linspace(
        THICKNESS_MIN_M,
        THICKNESS_MAX_M,
        N_THICKNESS,
    )

    rows: list[dict[str, object]] = []

    total = (
        len(roughness_values)
        * len(salinity_values)
        * len(thickness_values)
        * len(POROSITY_VALUES)
    )
    count = 0

    for porosity in POROSITY_VALUES:
        for thickness in thickness_values:
            for salinity in salinity_values:
                for roughness in roughness_values:
                    count += 1

                    x = np.array(
                        [
                            roughness,
                            salinity,
                            thickness,
                            porosity,
                        ],
                        dtype=float,
                    )

                    row: dict[str, object] = {
                        "ice_type_label": ice_type_label,
                        "smrt_ice_type": "multiyear",
                        "incidence_bin": incidence_bin,
                        "incidence_angle_deg": angle_deg,
                        "roughness_rms_m": float(roughness),
                        "salinity_ppt": float(salinity),
                        "ice_thickness_m": float(thickness),
                        "porosity": float(porosity),
                        "valid": False,
                        "warning": "",
                        "error": "",
                    }

                    try:
                        state = base_state.replace(
                            roughness_rms_m=float(roughness),
                            salinity_ppt=float(salinity),
                            ice_thickness_m=float(thickness),
                        )

                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            pred = run_forward_multiyear(
                                state,
                                float(porosity),
                                cfg,
                            )

                        row["valid"] = True

                        for ch in ALL_CHANNELS:
                            row[f"pred_{ch}_db"] = float(pred[ch])

                        if caught:
                            row["warning"] = " | ".join(
                                str(w.message) for w in caught
                            )

                    except Exception as exc:
                        row["error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                    rows.append(row)

                    if count % max(1, total // 10) == 0:
                        print(
                            f"    coarse {count}/{total} "
                            f"({100*count/total:.0f}%)",
                            flush=True,
                        )

    return pd.DataFrame(rows)


def refine_subset(
    coarse_valid: pd.DataFrame,
    observed_all: np.ndarray,
    channels: tuple[str, ...],
    safe_predict,
) -> dict[str, object]:
    indices = subset_indices(channels)
    obs = observed_all[indices]

    pred_cols = [f"pred_{ch}_db" for ch in channels]
    pred_matrix = coarse_valid[pred_cols].to_numpy(float)

    coarse_rms_all = np.sqrt(
        np.mean(
            (pred_matrix - obs[None, :]) ** 2,
            axis=1,
        )
    )
    order = np.argsort(coarse_rms_all)

    param_cols = [
        "roughness_rms_m",
        "salinity_ppt",
        "ice_thickness_m",
        "porosity",
    ]
    all_pred_cols = [
        f"pred_{ch}_db" for ch in ALL_CHANNELS
    ]

    best_pos = int(order[0])
    best_row = coarse_valid.iloc[best_pos]

    coarse_x = best_row[param_cols].to_numpy(float)
    coarse_y = best_row[all_pred_cols].to_numpy(float)
    coarse_rms = float(coarse_rms_all[best_pos])

    candidates: list[dict[str, object]] = []

    def objective(z: np.ndarray) -> float:
        x = z_to_x(z)
        y = safe_predict(x)

        if y is None:
            return INVALID_PENALTY

        r = y[indices] - obs
        return float(np.mean(r**2))

    for pos in order[:N_REFINEMENT_STARTS]:
        x0 = coarse_valid.iloc[int(pos)][param_cols].to_numpy(float)

        try:
            result = minimize(
                objective,
                x0=x_to_z(x0),
                method="Powell",
                bounds=Z_BOUNDS,
                options={
                    "maxiter": MAXITER,
                    "xtol": 1e-5,
                    "ftol": 1e-7,
                    "disp": False,
                },
            )

            x = z_to_x(result.x)
            y = safe_predict(x)

            if y is None:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        (y[indices] - obs) ** 2
                    )
                )
            )

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "rms": rms,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nfev": int(getattr(result, "nfev", 0)),
                }
            )

        except Exception:
            continue

    # Guaranteed valid fallback.
    candidates.append(
        {
            "x": coarse_x,
            "y": coarse_y,
            "rms": coarse_rms,
            "success": False,
            "message": "coarse-grid fallback",
            "nfev": 0,
        }
    )

    best = min(candidates, key=lambda item: item["rms"])

    x = np.asarray(best["x"], dtype=float)
    y = np.asarray(best["y"], dtype=float)

    residual_all = y - observed_all
    flags = at_bounds(x)

    output: dict[str, object] = {
        "coarse_rms_mismatch_db": coarse_rms,
        "refined_rms_mismatch_db": float(best["rms"]),
        "improvement_db": float(
            coarse_rms - float(best["rms"])
        ),
        "status": status_from_rms(float(best["rms"])),
        "best_roughness_rms_m": float(x[0]),
        "best_salinity_ppt": float(x[1]),
        "best_ice_thickness_m": float(x[2]),
        "best_porosity": float(x[3]),
        "roughness_at_bound": flags[0],
        "salinity_at_bound": flags[1],
        "thickness_at_bound": flags[2],
        "porosity_at_bound": flags[3],
        "refinement_selected": bool(
            best["rms"] < coarse_rms - 1e-8
        ),
        "optimizer_success": bool(best["success"]),
        "optimizer_message": str(best["message"]),
        "optimizer_nfev": int(best["nfev"]),
    }

    for i, ch in enumerate(ALL_CHANNELS):
        output[f"obs_{ch}_db"] = float(observed_all[i])
        output[f"fit_{ch}_db"] = float(y[i])
        output[f"residual_{ch}_db"] = float(
            residual_all[i]
        )

    return output


def main() -> None:
    if not OBSERVATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing observations:\n{OBSERVATION_CSV}"
        )

    observations = pd.read_csv(OBSERVATION_CSV)
    observations = observations[
        observations["ice_type"].astype(str) == "MYI"
    ].copy()

    if observations.empty:
        raise RuntimeError(
            "No MYI rows found in the incidence-binned observation table."
        )

    cfg_base = load_config()
    base_state = baseline_state(cfg_base)
    results_dir = ensure_results_dir()

    print("=" * 78)
    print("MYI STRUCTURAL TEST: FIRSTYEAR -> MULTIYEAR SMRT")
    print("=" * 78)
    print(
        "This experiment corrects ONLY the MYI ice-type structure.\n"
        "SMRT ice_type is forced to 'multiyear'.\n"
        "Air porosity is added as a diagnostic variable.\n"
        f"Ice correlation length stays fixed at "
        f"{float(cfg_base['fixed_scene']['ice_corr_length_m'])*1000:.3f} mm.\n"
        "Porosity range is diagnostic, NOT a literature prior."
    )

    all_coarse: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}

    for idx, rec in observations.iterrows():
        incidence_bin = str(rec["incidence_bin"])
        angle = float(rec["incidence_mean_deg"])

        observed_all = np.array(
            [
                float(rec[f"{ch}_mean_db"])
                for ch in ALL_CHANNELS
            ],
            dtype=float,
        )

        cfg = cfg_at_angle(cfg_base, angle)

        print("\n" + "-" * 78)
        print(
            f"MYI | {incidence_bin} deg | "
            f"theta={angle:.3f} deg"
        )
        print("-" * 78)

        coarse = build_coarse_grid(
            base_state=base_state,
            cfg=cfg,
            ice_type_label="MYI",
            incidence_bin=incidence_bin,
            angle_deg=angle,
        )
        all_coarse.append(coarse)

        valid = coarse[
            coarse["valid"] == True
        ].copy()  # noqa: E712

        print(
            f"  valid coarse states: "
            f"{len(valid):,}/{len(coarse):,}"
        )

        if valid.empty:
            raise RuntimeError(
                f"No valid MYI multiyear states for {incidence_bin}."
            )

        safe_predict, stats = make_safe_predictor(
            base_state,
            cfg,
        )

        key = f"MYI_{incidence_bin}"
        details[key] = {
            "incidence_angle_deg": angle,
            "subsets": {},
        }

        for subset_name, channels in SUBSETS.items():
            result = refine_subset(
                coarse_valid=valid,
                observed_all=observed_all,
                channels=channels,
                safe_predict=safe_predict,
            )

            row = {
                "ice_type": "MYI",
                "smrt_ice_type": "multiyear",
                "incidence_bin": incidence_bin,
                "incidence_angle_deg": angle,
                "n_observation_pixels": int(rec["n_pixels"]),
                "subset": subset_name,
                "channels": ",".join(channels),
                **result,
            }
            rows.append(row)
            details[key]["subsets"][subset_name] = result

            print(
                f"  {subset_name:<8} | "
                f"coarse={result['coarse_rms_mismatch_db']:6.3f} | "
                f"refined={result['refined_rms_mismatch_db']:6.3f} | "
                f"gain={result['improvement_db']:+6.3f} dB | "
                f"{result['status']} | "
                f"porosity={result['best_porosity']:.4f}"
            )

        details[key]["safe_refinement_stats"] = stats

        print(
            f"  refinement SMRT runs={stats['runs']}, "
            f"invalid_trials={stats['invalid']}"
        )

    coarse_df = pd.concat(
        all_coarse,
        ignore_index=True,
    )
    result_df = pd.DataFrame(rows)

    coarse_csv = (
        results_dir / "10_myi_multiyear_coarse_grid.csv"
    )
    result_csv = (
        results_dir / "10_myi_multiyear_reachability.csv"
    )
    result_json = (
        results_dir / "10_myi_multiyear_reachability.json"
    )

    coarse_df.to_csv(coarse_csv, index=False)
    result_df.to_csv(result_csv, index=False)

    result_json.write_text(
        json.dumps(
            {
                "purpose": (
                    "MYI reachability using SMRT multiyear structure "
                    "instead of the earlier firstyear structure."
                ),
                "important_note": (
                    "Porosity is a broad diagnostic variable here, "
                    "not a literature-constrained prior."
                ),
                "fixed_ice_corr_length_m": float(
                    cfg_base["fixed_scene"]["ice_corr_length_m"]
                ),
                "porosity_domain": [
                    POROSITY_MIN,
                    POROSITY_MAX,
                ],
                "details": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL MYI MULTIYEAR RESULTS")
    print("=" * 78)

    display_cols = [
        "ice_type",
        "incidence_bin",
        "incidence_angle_deg",
        "subset",
        "refined_rms_mismatch_db",
        "status",
        "best_roughness_rms_m",
        "best_salinity_ppt",
        "best_ice_thickness_m",
        "best_porosity",
        "porosity_at_bound",
    ]

    print(
        result_df[display_cols]
        .sort_values(
            ["incidence_angle_deg", "subset"]
        )
        .to_string(index=False)
    )

    # Optional direct comparison with the previous firstyear result.
    comparison_csv = (
        results_dir
        / "10_myi_firstyear_vs_multiyear.csv"
    )

    if OLD_FIRSTYEAR_RESULTS.exists():
        old = pd.read_csv(OLD_FIRSTYEAR_RESULTS)
        old = old[
            old["ice_type"].astype(str) == "MYI"
        ].copy()

        old = old[
            [
                "incidence_bin",
                "subset",
                "refined_rms_mismatch_db",
            ]
        ].rename(
            columns={
                "refined_rms_mismatch_db":
                    "firstyear_rms_db"
            }
        )

        new = result_df[
            [
                "incidence_bin",
                "subset",
                "refined_rms_mismatch_db",
                "best_porosity",
            ]
        ].rename(
            columns={
                "refined_rms_mismatch_db":
                    "multiyear_rms_db"
            }
        )

        comparison = old.merge(
            new,
            on=["incidence_bin", "subset"],
            how="inner",
        )

        comparison["improvement_multiyear_vs_firstyear_db"] = (
            comparison["firstyear_rms_db"]
            - comparison["multiyear_rms_db"]
        )

        comparison.to_csv(
            comparison_csv,
            index=False,
        )

        print("\nFIRSTYEAR vs MULTIYEAR:")
        print(
            comparison.sort_values(
                ["incidence_bin", "subset"]
            ).to_string(index=False)
        )

    print("\nSaved:")
    print(f"  {coarse_csv}")
    print(f"  {result_csv}")
    print(f"  {result_json}")

    if OLD_FIRSTYEAR_RESULTS.exists():
        print(f"  {comparison_csv}")

    print(
        "\nSTOP HERE. Do not add snow yet. "
        "First decide whether correcting MYI from firstyear to multiyear "
        "materially changes reachability."
    )


if __name__ == "__main__":
    main()
