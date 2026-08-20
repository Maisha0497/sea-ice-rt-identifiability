#!/usr/bin/env python3
from __future__ import annotations

"""
09b_incidence_aware_safe_refinement.py

Fix the failed continuous refinement from experiment 09 without rerunning the
coarse SMRT grids.

Reason:
least_squares perturbs parameters to estimate derivatives. Some perturbations
cross the hard SMRT/IEM validity boundary and raise exceptions. Experiment 09
then discarded that whole refinement start.

This script:
- reuses results/09_incidence_aware_coarse_grid.csv
- uses bounded derivative-free Powell minimization
- optimizes log(roughness), salinity, thickness
- gives invalid SMRT/IEM trial states a large finite penalty instead of failing
- compares coarse vs refined RMS for L-only, S-only, and L+S

This is still a reachability diagnostic, not a final inversion.
"""

from copy import deepcopy
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config, vector_from_prediction
from src.forward_smrt import run_forward


OBSERVATION_CSV = (
    ROOT / "data" / "derived" / "SMRT_observation_audit"
    / "incidence_binned" / "08_incidence_binned_observations.csv"
)
COARSE_GRID_CSV = ROOT / "results" / "09_incidence_aware_coarse_grid.csv"

ALL_CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")
SUBSETS = {
    "L_only": ("L_HH", "L_VV"),
    "S_only": ("S_HH", "S_VV"),
    "L_plus_S": ALL_CHANNELS,
}

ROUGHNESS_MIN_M = 0.00025
ROUGHNESS_MAX_M = 0.00300
SALINITY_MIN_PPT = 0.5
SALINITY_MAX_PPT = 15.0
THICKNESS_MIN_M = 0.20
THICKNESS_MAX_M = 6.00

N_STARTS = 8
MAXITER = 160
INVALID_PENALTY = 1e6

Z_BOUNDS = [
    (math.log(ROUGHNESS_MIN_M), math.log(ROUGHNESS_MAX_M)),
    (SALINITY_MIN_PPT, SALINITY_MAX_PPT),
    (THICKNESS_MIN_M, THICKNESS_MAX_M),
]


def status_from_rms(rms):
    if rms <= 2:
        return "near/reachable"
    if rms <= 5:
        return "marginal"
    return "clearly outside"


def cfg_at_angle(cfg_base, angle_deg):
    cfg = deepcopy(cfg_base)
    cfg["sensor"]["incidence_angle_deg"] = float(angle_deg)
    return cfg


def x_to_z(x):
    return np.array([math.log(float(x[0])), float(x[1]), float(x[2])])


def z_to_x(z):
    return np.array([math.exp(float(z[0])), float(z[1]), float(z[2])])


def at_bounds(x):
    lo = np.array([ROUGHNESS_MIN_M, SALINITY_MIN_PPT, THICKNESS_MIN_M])
    hi = np.array([ROUGHNESS_MAX_M, SALINITY_MAX_PPT, THICKNESS_MAX_M])
    tol = np.maximum((hi - lo) * 1e-3, 1e-12)
    return [
        bool(abs(x[i] - lo[i]) <= tol[i] or abs(x[i] - hi[i]) <= tol[i])
        for i in range(3)
    ]


def subset_indices(channels):
    return np.array([ALL_CHANNELS.index(ch) for ch in channels], dtype=int)


def make_safe_predictor(base_state, cfg):
    cache = {}
    stats = {"runs": 0, "cache_hits": 0, "invalid": 0}

    def safe_predict(x):
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
            pred = run_forward(state, cfg)
            y = vector_from_prediction(pred, ALL_CHANNELS).astype(float)
            if y.shape != (4,) or np.any(~np.isfinite(y)):
                raise ValueError(f"Invalid prediction: {y}")
            cache[key] = y
        except Exception:
            stats["invalid"] += 1
            cache[key] = None

        return cache[key]

    return safe_predict, stats


def refine_subset(coarse_valid, observed_all, channels, safe_predict):
    indices = subset_indices(channels)
    obs = observed_all[indices]

    pred_cols = [f"pred_{ch}_db" for ch in channels]
    pred_matrix = coarse_valid[pred_cols].to_numpy(float)
    coarse_rms_all = np.sqrt(np.mean((pred_matrix - obs[None, :]) ** 2, axis=1))
    order = np.argsort(coarse_rms_all)

    param_cols = ["roughness_rms_m", "salinity_ppt", "ice_thickness_m"]
    all_pred_cols = [f"pred_{ch}_db" for ch in ALL_CHANNELS]

    best_pos = int(order[0])
    coarse_row = coarse_valid.iloc[best_pos]
    coarse_x = coarse_row[param_cols].to_numpy(float)
    coarse_y = coarse_row[all_pred_cols].to_numpy(float)
    coarse_rms = float(coarse_rms_all[best_pos])

    candidates = []

    def objective(z):
        x = z_to_x(z)
        y = safe_predict(x)
        if y is None:
            return INVALID_PENALTY
        r = y[indices] - obs
        return float(np.mean(r ** 2))

    for pos in order[:N_STARTS]:
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
            rms = float(np.sqrt(np.mean((y[indices] - obs) ** 2)))
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

    best = min(candidates, key=lambda d: d["rms"])
    x = np.asarray(best["x"], float)
    y = np.asarray(best["y"], float)
    residual_all = y - observed_all
    flags = at_bounds(x)

    out = {
        "coarse_rms_mismatch_db": coarse_rms,
        "refined_rms_mismatch_db": float(best["rms"]),
        "improvement_db": float(coarse_rms - best["rms"]),
        "status": status_from_rms(float(best["rms"])),
        "best_roughness_rms_m": float(x[0]),
        "best_salinity_ppt": float(x[1]),
        "best_ice_thickness_m": float(x[2]),
        "roughness_at_bound": flags[0],
        "salinity_at_bound": flags[1],
        "thickness_at_bound": flags[2],
        "refinement_selected": bool(best["rms"] < coarse_rms - 1e-8),
        "optimizer_success": bool(best["success"]),
        "optimizer_message": str(best["message"]),
        "optimizer_nfev": int(best["nfev"]),
    }

    for i, ch in enumerate(ALL_CHANNELS):
        out[f"obs_{ch}_db"] = float(observed_all[i])
        out[f"fit_{ch}_db"] = float(y[i])
        out[f"residual_{ch}_db"] = float(residual_all[i])

    return out


def main():
    if not OBSERVATION_CSV.exists():
        raise FileNotFoundError(f"Missing:\n{OBSERVATION_CSV}")
    if not COARSE_GRID_CSV.exists():
        raise FileNotFoundError(f"Missing:\n{COARSE_GRID_CSV}\nRun experiment 09 first.")

    observations = pd.read_csv(OBSERVATION_CSV)
    coarse = pd.read_csv(COARSE_GRID_CSV)

    cfg_base = load_config()
    base_state = baseline_state(cfg_base)
    results_dir = ensure_results_dir()

    rows = []
    details = {}

    print("=" * 78)
    print("09b SAFE DERIVATIVE-FREE REFINEMENT")
    print("=" * 78)
    print("Reusing experiment 09 coarse grids; no coarse grid rerun.")
    print("Invalid IEM/SMRT trial points get a finite penalty instead of aborting.")

    for idx, rec in observations.iterrows():
        ice = str(rec["ice_type"])
        inc_bin = str(rec["incidence_bin"])
        angle = float(rec["incidence_mean_deg"])

        observed_all = np.array(
            [float(rec[f"{ch}_mean_db"]) for ch in ALL_CHANNELS]
        )

        valid = coarse[
            (coarse["ice_type"].astype(str) == ice)
            & (coarse["incidence_bin"].astype(str) == inc_bin)
            & (coarse["valid"] == True)
        ].copy()

        if valid.empty:
            raise RuntimeError(f"No valid coarse states for {ice} {inc_bin}")

        safe_predict, stats = make_safe_predictor(
            base_state,
            cfg_at_angle(cfg_base, angle),
        )

        print(f"\n[{idx+1}/{len(observations)}] {ice} | {inc_bin} deg | theta={angle:.3f}")

        key = f"{ice}_{inc_bin}"
        details[key] = {"subsets": {}}

        for subset_name, channels in SUBSETS.items():
            result = refine_subset(valid, observed_all, channels, safe_predict)
            rows.append(
                {
                    "ice_type": ice,
                    "incidence_bin": inc_bin,
                    "incidence_angle_deg": angle,
                    "n_observation_pixels": int(rec["n_pixels"]),
                    "subset": subset_name,
                    "channels": ",".join(channels),
                    **result,
                }
            )
            details[key]["subsets"][subset_name] = result

            print(
                f"  {subset_name:<8} | "
                f"coarse={result['coarse_rms_mismatch_db']:6.3f} | "
                f"refined={result['refined_rms_mismatch_db']:6.3f} | "
                f"gain={result['improvement_db']:+6.3f} dB | "
                f"{result['status']}"
            )

        details[key]["eval_stats"] = stats
        print(
            f"  SMRT runs={stats['runs']}, "
            f"cache_hits={stats['cache_hits']}, "
            f"invalid_trials={stats['invalid']}"
        )

    df = pd.DataFrame(rows)

    csv_path = results_dir / "09b_incidence_aware_safe_refinement.csv"
    json_path = results_dir / "09b_incidence_aware_safe_refinement.json"
    compare_path = results_dir / "09b_coarse_vs_refined.csv"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "method": (
                    "bounded derivative-free Powell in log(roughness), "
                    "salinity, thickness; invalid SMRT/IEM states receive "
                    "finite penalty"
                ),
                "details": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    compare_cols = [
        "ice_type",
        "incidence_bin",
        "incidence_angle_deg",
        "subset",
        "coarse_rms_mismatch_db",
        "refined_rms_mismatch_db",
        "improvement_db",
        "status",
        "best_roughness_rms_m",
        "best_salinity_ppt",
        "best_ice_thickness_m",
        "roughness_at_bound",
        "salinity_at_bound",
        "thickness_at_bound",
        "refinement_selected",
    ]
    df[compare_cols].to_csv(compare_path, index=False)

    print("\n" + "=" * 78)
    print("FINAL SAFE-REFINEMENT RESULTS")
    print("=" * 78)

    print(
        df[
            [
                "ice_type",
                "incidence_bin",
                "incidence_angle_deg",
                "subset",
                "coarse_rms_mismatch_db",
                "refined_rms_mismatch_db",
                "improvement_db",
                "status",
            ]
        ]
        .sort_values(["ice_type", "incidence_angle_deg", "subset"])
        .to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {compare_path}")

    print(
        "\nSTOP HERE. Use refined_rms_mismatch_db, not the old experiment-09 "
        "coarse values, for the bare-ice reachability conclusion."
    )


if __name__ == "__main__":
    main()
