#!/usr/bin/env python3
from __future__ import annotations

"""
09_incidence_aware_reachability.py

Purpose
-------
Test whether the CURRENT bare-ice SMRT configuration can reproduce the
incidence-binned UAVSAR observations prepared by experiment 08.

For every retained MYI / NI / TI incidence bin:
    1. set SMRT incidence angle to that bin's actual mean incidence angle;
    2. evaluate a coarse expanded physical-state grid;
    3. refine the best coarse candidates with bounded least_squares;
    4. report reachability separately for:
         - L_only  = L_HH, L_VV
         - S_only  = S_HH, S_VV
         - L_plus_S = L_HH, L_VV, S_HH, S_VV

This is a FORWARD-MODEL EXISTENCE / REACHABILITY diagnostic.
It is NOT the final physical inversion.

The parameter bounds intentionally match experiment 06b:
    roughness RMS: 0.00025 to 0.00300 m
    salinity:      0.5 to 15.0 ppt
    thickness:     0.20 to 6.00 m

The coarse grid here is intentionally smaller than 06b because the full grid
must be repeated at many incidence angles. The best states are then refined
continuously.

Input
-----
data/derived/SMRT_observation_audit/incidence_binned/
    08_incidence_binned_observations.csv

Outputs
-------
results/
    09_incidence_aware_coarse_grid.csv
    09_incidence_aware_reachability.csv
    09_incidence_aware_reachability.json
    09_incidence_aware_reachability_MYI.png
    09_incidence_aware_reachability_NI.png
    09_incidence_aware_reachability_TI.png

Run from project root:
    conda activate geo
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python experiments/09_incidence_aware_reachability.py
"""

from copy import deepcopy
from pathlib import Path
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    baseline_state,
    ensure_results_dir,
    load_config,
    vector_from_prediction,
)
from src.forward_smrt import run_forward


# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------

OBSERVATION_CSV = (
    ROOT / "data" / "derived" / "SMRT_observation_audit"
    / "incidence_binned" / "08_incidence_binned_observations.csv"
)

ALL_CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")

SUBSETS = {
    "L_only": ("L_HH", "L_VV"),
    "S_only": ("S_HH", "S_VV"),
    "L_plus_S": ALL_CHANNELS,
}


# ---------------------------------------------------------------------
# EXPANDED PHYSICAL DOMAIN — MATCHES 06b
# ---------------------------------------------------------------------

ROUGHNESS_MIN_M = 0.00025
ROUGHNESS_MAX_M = 0.00300

SALINITY_MIN_PPT = 0.5
SALINITY_MAX_PPT = 15.0

THICKNESS_MIN_M = 0.20
THICKNESS_MAX_M = 6.00


# ---------------------------------------------------------------------
# COMPUTATIONAL SETTINGS
# ---------------------------------------------------------------------
#
# 9 x 9 x 7 = 567 coarse states per incidence bin.
# Experiment 08 currently produced ~14 retained bins, so the coarse stage is
# ~8,000 forward states rather than repeating the full 06b 11,875-state grid
# for every angle.
#
# The best candidates are then refined continuously.
# ---------------------------------------------------------------------

N_ROUGHNESS = 9
N_SALINITY = 9
N_THICKNESS = 7

N_REFINEMENT_STARTS = 5
MAX_NFEV = 250

# Equal 1-dB scaling is used ONLY as a common geometric distance diagnostic,
# exactly in the spirit of 06/06b. This is not yet the final observation
# covariance.
DISTANCE_SCALE_DB = 1.0


def status_from_rms(value: float) -> str:
    if value <= 2.0:
        return "near/reachable"
    if value <= 5.0:
        return "marginal"
    return "clearly outside"


def rms_db(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [ROUGHNESS_MIN_M, SALINITY_MIN_PPT, THICKNESS_MIN_M],
        dtype=float,
    )
    upper = np.array(
        [ROUGHNESS_MAX_M, SALINITY_MAX_PPT, THICKNESS_MAX_M],
        dtype=float,
    )
    return lower, upper


def at_bounds(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[bool]:
    tol = np.maximum((upper - lower) * 1e-3, 1e-12)
    return [
        bool(
            abs(x[i] - lower[i]) <= tol[i]
            or abs(x[i] - upper[i]) <= tol[i]
        )
        for i in range(len(x))
    ]


def cfg_at_angle(cfg_base: dict, angle_deg: float) -> dict:
    cfg = deepcopy(cfg_base)
    cfg["sensor"]["incidence_angle_deg"] = float(angle_deg)
    return cfg


def predict_all(
    base_state,
    cfg: dict,
    x: np.ndarray,
) -> np.ndarray:
    state = base_state.replace(
        roughness_rms_m=float(x[0]),
        salinity_ppt=float(x[1]),
        ice_thickness_m=float(x[2]),
    )

    prediction = run_forward(state, cfg)
    y = vector_from_prediction(prediction, ALL_CHANNELS).astype(float)

    if y.shape != (4,) or np.any(~np.isfinite(y)):
        raise ValueError(f"Invalid forward prediction: {y}")

    return y


def subset_indices(channels: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [ALL_CHANNELS.index(ch) for ch in channels],
        dtype=int,
    )


def residual_subset(
    base_state,
    cfg: dict,
    x: np.ndarray,
    observed_subset: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    prediction = predict_all(base_state, cfg, x)
    return (
        prediction[indices] - observed_subset
    ) / DISTANCE_SCALE_DB


def build_coarse_grid_for_angle(
    cfg_base: dict,
    base_state,
    angle_deg: float,
    ice_type: str,
    incidence_bin: str,
) -> pd.DataFrame:
    cfg = cfg_at_angle(cfg_base, angle_deg)

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

    for thickness in thickness_values:
        for salinity in salinity_values:
            for roughness in roughness_values:
                x = np.array(
                    [roughness, salinity, thickness],
                    dtype=float,
                )

                row: dict[str, object] = {
                    "ice_type": ice_type,
                    "incidence_bin": incidence_bin,
                    "incidence_angle_deg": float(angle_deg),
                    "roughness_rms_m": float(roughness),
                    "salinity_ppt": float(salinity),
                    "ice_thickness_m": float(thickness),
                    "valid": False,
                    "warning": "",
                    "error": "",
                }

                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        y = predict_all(base_state, cfg, x)

                    row["valid"] = True

                    for ch, value in zip(ALL_CHANNELS, y):
                        row[f"pred_{ch}_db"] = float(value)

                    if caught:
                        row["warning"] = " | ".join(
                            str(item.message) for item in caught
                        )

                except Exception as exc:
                    row["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                rows.append(row)

    return pd.DataFrame(rows)


def refine_one_subset(
    cfg_base: dict,
    base_state,
    angle_deg: float,
    coarse_valid: pd.DataFrame,
    observed_all: np.ndarray,
    subset_name: str,
    channels: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    cfg = cfg_at_angle(cfg_base, angle_deg)
    indices = subset_indices(channels)

    observed_subset = observed_all[indices]

    pred_cols = [f"pred_{ch}_db" for ch in channels]
    pred_matrix = coarse_valid[pred_cols].to_numpy(dtype=float)

    residual_matrix = pred_matrix - observed_subset[None, :]
    rms_values = np.sqrt(np.mean(residual_matrix**2, axis=1))
    order = np.argsort(rms_values)

    param_cols = [
        "roughness_rms_m",
        "salinity_ppt",
        "ice_thickness_m",
    ]

    refined: list[dict[str, object]] = []

    for position in order[:N_REFINEMENT_STARTS]:
        x0 = coarse_valid.iloc[int(position)][param_cols].to_numpy(
            dtype=float
        )

        try:
            result = least_squares(
                lambda x: residual_subset(
                    base_state,
                    cfg,
                    x,
                    observed_subset,
                    indices,
                ),
                x0=x0,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                max_nfev=MAX_NFEV,
            )

            y_all = predict_all(base_state, cfg, result.x)
            y_subset = y_all[indices]
            r_subset = y_subset - observed_subset

            refined.append(
                {
                    "x": np.asarray(result.x, dtype=float),
                    "prediction_all": y_all,
                    "prediction_subset": y_subset,
                    "residual_subset": r_subset,
                    "rms": rms_db(r_subset),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nfev": int(result.nfev),
                }
            )

        except Exception:
            # This mirrors the earlier reachability scripts:
            # an invalid intermediate state can kill one refinement start,
            # but other starts and the coarse fallback remain available.
            continue

    if refined:
        best = min(refined, key=lambda item: float(item["rms"]))

        x_best = np.asarray(best["x"], dtype=float)
        prediction_all = np.asarray(
            best["prediction_all"],
            dtype=float,
        )
        prediction_subset = np.asarray(
            best["prediction_subset"],
            dtype=float,
        )
        residual = np.asarray(
            best["residual_subset"],
            dtype=float,
        )

        optimizer_success = bool(best["success"])
        optimizer_message = str(best["message"])
        optimizer_nfev = int(best["nfev"])
        used_refinement = True

    else:
        best_position = int(order[0])
        row = coarse_valid.iloc[best_position]

        x_best = row[param_cols].to_numpy(dtype=float)
        prediction_all = row[
            [f"pred_{ch}_db" for ch in ALL_CHANNELS]
        ].to_numpy(dtype=float)
        prediction_subset = prediction_all[indices]
        residual = prediction_subset - observed_subset

        optimizer_success = False
        optimizer_message = (
            "All local refinements failed; coarse-grid minimum used."
        )
        optimizer_nfev = 0
        used_refinement = False

    mismatch = rms_db(residual)
    flags = at_bounds(x_best, lower, upper)

    output: dict[str, object] = {
        "subset": subset_name,
        "channels": ",".join(channels),
        "best_rms_mismatch_db": mismatch,
        "status": status_from_rms(mismatch),
        "best_roughness_rms_m": float(x_best[0]),
        "best_salinity_ppt": float(x_best[1]),
        "best_ice_thickness_m": float(x_best[2]),
        "roughness_at_bound": flags[0],
        "salinity_at_bound": flags[1],
        "thickness_at_bound": flags[2],
        "used_continuous_refinement": used_refinement,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "optimizer_nfev": optimizer_nfev,
    }

    for i, ch in enumerate(channels):
        output[f"obs_{ch}_db"] = float(observed_subset[i])
        output[f"fit_{ch}_db"] = float(prediction_subset[i])
        output[f"residual_{ch}_db"] = float(residual[i])

    # Save all four predicted channels at the selected state even when
    # only L or only S was fitted. This is useful for diagnosing
    # cross-band incompatibility.
    for i, ch in enumerate(ALL_CHANNELS):
        output[f"selected_state_pred_{ch}_db"] = float(
            prediction_all[i]
        )

    return output


def main() -> None:
    if not OBSERVATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing observation file:\n{OBSERVATION_CSV}\n"
            "Run experiment 08 first."
        )

    observations = pd.read_csv(OBSERVATION_CSV)

    required = {
        "ice_type",
        "incidence_bin",
        "incidence_mean_deg",
        *(f"{ch}_mean_db" for ch in ALL_CHANNELS),
    }
    missing = required.difference(observations.columns)

    if missing:
        raise ValueError(
            f"Observation table missing columns: {sorted(missing)}"
        )

    cfg_base = load_config()
    base_state = baseline_state(cfg_base)
    results_dir = ensure_results_dir()
    lower, upper = parameter_bounds()

    coarse_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    details: dict[str, dict] = {}

    n_states_per_bin = (
        N_ROUGHNESS * N_SALINITY * N_THICKNESS
    )

    print("=" * 78)
    print("INCIDENCE-AWARE SMRT REACHABILITY TEST")
    print("=" * 78)
    print(f"Observation bins: {len(observations)}")
    print(f"Coarse states per bin: {n_states_per_bin:,}")
    print(
        "Expanded bounds:\n"
        f"  roughness = {ROUGHNESS_MIN_M:g} to "
        f"{ROUGHNESS_MAX_M:g} m\n"
        f"  salinity  = {SALINITY_MIN_PPT:g} to "
        f"{SALINITY_MAX_PPT:g} ppt\n"
        f"  thickness = {THICKNESS_MIN_M:g} to "
        f"{THICKNESS_MAX_M:g} m"
    )
    print(
        "\nImportant: each bin is modeled at its ACTUAL mean incidence angle."
    )

    for obs_index, record in observations.iterrows():
        ice_type = str(record["ice_type"])
        incidence_bin = str(record["incidence_bin"])
        angle_deg = float(record["incidence_mean_deg"])

        observed_all = np.array(
            [float(record[f"{ch}_mean_db"]) for ch in ALL_CHANNELS],
            dtype=float,
        )

        print("\n" + "-" * 78)
        print(
            f"[{obs_index + 1}/{len(observations)}] "
            f"{ice_type} | {incidence_bin} deg | "
            f"SMRT theta={angle_deg:.3f} deg"
        )
        print("-" * 78)

        coarse = build_coarse_grid_for_angle(
            cfg_base=cfg_base,
            base_state=base_state,
            angle_deg=angle_deg,
            ice_type=ice_type,
            incidence_bin=incidence_bin,
        )
        coarse_all.append(coarse)

        valid = coarse[coarse["valid"] == True].copy()  # noqa: E712

        if valid.empty:
            print("  No valid SMRT coarse states.")
            for subset_name, channels in SUBSETS.items():
                row = {
                    "ice_type": ice_type,
                    "incidence_bin": incidence_bin,
                    "incidence_angle_deg": angle_deg,
                    "subset": subset_name,
                    "channels": ",".join(channels),
                    "best_rms_mismatch_db": np.nan,
                    "status": "no valid SMRT states",
                }
                summary_rows.append(row)
            continue

        print(
            f"  valid coarse states: {len(valid):,} / {len(coarse):,}"
        )

        bin_key = f"{ice_type}_{incidence_bin}"
        details[bin_key] = {
            "ice_type": ice_type,
            "incidence_bin": incidence_bin,
            "incidence_angle_deg": angle_deg,
            "n_observation_pixels": int(record["n_pixels"]),
            "observed_db": {
                ch: float(record[f"{ch}_mean_db"])
                for ch in ALL_CHANNELS
            },
            "subsets": {},
        }

        for subset_name, channels in SUBSETS.items():
            result = refine_one_subset(
                cfg_base=cfg_base,
                base_state=base_state,
                angle_deg=angle_deg,
                coarse_valid=valid,
                observed_all=observed_all,
                subset_name=subset_name,
                channels=channels,
                lower=lower,
                upper=upper,
            )

            row = {
                "ice_type": ice_type,
                "incidence_bin": incidence_bin,
                "incidence_angle_deg": angle_deg,
                "n_observation_pixels": int(record["n_pixels"]),
                **result,
            }
            summary_rows.append(row)

            details[bin_key]["subsets"][subset_name] = result

            print(
                f"  {subset_name:<8} | "
                f"RMS={result['best_rms_mismatch_db']:6.3f} dB | "
                f"{result['status']}"
            )

    coarse_df = pd.concat(coarse_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    coarse_csv = (
        results_dir / "09_incidence_aware_coarse_grid.csv"
    )
    summary_csv = (
        results_dir / "09_incidence_aware_reachability.csv"
    )
    summary_json = (
        results_dir / "09_incidence_aware_reachability.json"
    )

    coarse_df.to_csv(coarse_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "purpose": (
            "Forward-model reachability at actual UAVSAR incidence angles "
            "without empirical normalization to 35 degrees."
        ),
        "observation_file": str(OBSERVATION_CSV),
        "channels": list(ALL_CHANNELS),
        "subsets": {
            key: list(value) for key, value in SUBSETS.items()
        },
        "note": (
            "Equal 1 dB scaling is a common geometric distance only. "
            "It is not the final observation covariance."
        ),
        "bounds": {
            "roughness_rms_m": [
                ROUGHNESS_MIN_M,
                ROUGHNESS_MAX_M,
            ],
            "salinity_ppt": [
                SALINITY_MIN_PPT,
                SALINITY_MAX_PPT,
            ],
            "ice_thickness_m": [
                THICKNESS_MIN_M,
                THICKNESS_MAX_M,
            ],
        },
        "coarse_grid": {
            "n_roughness": N_ROUGHNESS,
            "n_salinity": N_SALINITY,
            "n_thickness": N_THICKNESS,
            "states_per_incidence_bin": n_states_per_bin,
        },
        "details": details,
    }

    summary_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    # One simple figure per ice type. No subplot panels.
    for ice_type in summary_df["ice_type"].dropna().unique():
        ice_df = summary_df[
            (summary_df["ice_type"] == ice_type)
            & summary_df["best_rms_mismatch_db"].notna()
        ].copy()

        if ice_df.empty:
            continue

        plt.figure(figsize=(8.5, 5.5))

        for subset_name in SUBSETS:
            sub = ice_df[ice_df["subset"] == subset_name].sort_values(
                "incidence_angle_deg"
            )

            if sub.empty:
                continue

            plt.plot(
                sub["incidence_angle_deg"],
                sub["best_rms_mismatch_db"],
                marker="o",
                label=subset_name.replace("_", " "),
            )

        plt.axhline(
            2.0,
            linestyle="--",
            linewidth=1,
            label="2 dB diagnostic line",
        )
        plt.axhline(
            5.0,
            linestyle=":",
            linewidth=1,
            label="5 dB diagnostic line",
        )
        plt.xlabel("Actual mean incidence angle (deg)")
        plt.ylabel("Minimum RMS model-observation mismatch (dB)")
        plt.title(
            f"{ice_type}: incidence-aware SMRT reachability"
        )
        plt.legend()
        plt.tight_layout()

        figure_path = (
            results_dir
            / f"09_incidence_aware_reachability_{ice_type}.png"
        )
        plt.savefig(figure_path, dpi=200)
        plt.close()

    print("\n" + "=" * 78)
    print("FINAL INCIDENCE-AWARE RESULTS")
    print("=" * 78)

    display_cols = [
        "ice_type",
        "incidence_bin",
        "incidence_angle_deg",
        "subset",
        "best_rms_mismatch_db",
        "status",
    ]

    print(
        summary_df[display_cols]
        .sort_values(
            ["ice_type", "incidence_angle_deg", "subset"]
        )
        .to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {coarse_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")

    for ice_type in summary_df["ice_type"].dropna().unique():
        print(
            "  "
            + str(
                results_dir
                / f"09_incidence_aware_reachability_{ice_type}.png"
            )
        )

    print(
        "\nSTOP HERE. Do not add snow yet. "
        "Interpret whether the bare-ice mismatch survives after "
        "matching SMRT to the actual incidence angle."
    )


if __name__ == "__main__":
    main()
