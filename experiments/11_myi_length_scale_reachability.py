#!/usr/bin/env python3
from __future__ import annotations

"""
11_myi_length_scale_reachability.py

Tests whether the remaining MYI mismatch is caused by two length scales that
were fixed in experiment 10:

  surface_corr_length_m  -> IEM air/ice interface
  ice_corr_length_m      -> MYI air-bubble microstructure

The script keeps the old bounds for roughness, salinity, thickness and
porosity. It does NOT add snow and does NOT expand those old bounds.

For each MYI incidence bin it:
1. uses the actual mean incidence angle from experiment 08;
2. uses SMRT ice_type="multiyear";
3. builds a coarse grid over roughness, surface corr length, porosity and ice
   corr length, with salinity=0.5 ppt and thickness=6 m for the coarse stage;
4. safely refines the best candidates with bounded Powell over all six
   variables;
5. reports L-only, S-only and joint L+S RMS.

Inputs:
  data/derived/SMRT_observation_audit/incidence_binned/
      08_incidence_binned_observations.csv
  results/10_myi_multiyear_reachability.csv

Outputs:
  results/11_myi_length_scale_coarse_grid.csv
  results/11_myi_length_scale_reachability.csv
  results/11_myi_length_scale_reachability.json
  results/11_myi_length_scale_vs_experiment10.csv
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

OBSERVATION_CSV = (
    ROOT / "data" / "derived" / "SMRT_observation_audit"
    / "incidence_binned" / "08_incidence_binned_observations.csv"
)
EXPERIMENT10_CSV = ROOT / "results" / "10_myi_multiyear_reachability.csv"

CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")
SUBSETS = {
    "L_only": ("L_HH", "L_VV"),
    "L_plus_S": CHANNELS,
}

# Existing bounds: unchanged.
R_MIN, R_MAX = 0.00025, 0.00300
S_MIN, S_MAX = 0.5, 15.0
D_MIN, D_MAX = 0.20, 6.00
P_MIN, P_MAX = 0.0, 0.30

# Length-scale domains already used in earlier project diagnostics.
LSURF_MIN, LSURF_MAX = 0.005, 0.050
LICE_MIN, LICE_MAX = 0.0005, 0.0120

ROUGHNESS_VALUES = np.array([0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030])
SURFACE_CORR_VALUES = np.array([0.005, 0.010, 0.020, 0.030, 0.050])
POROSITY_VALUES = np.array([0.10, 0.20, 0.30])
ICE_CORR_VALUES = np.array([0.0005, 0.0010, 0.0020, 0.0030, 0.0050, 0.0080, 0.0120])

COARSE_SALINITY = 0.5
COARSE_THICKNESS = 6.0

N_STARTS = 4
MAXITER = 140
INVALID_PENALTY = 1e6

# z = [log r, log lsurf, salinity, thickness, porosity, log lice]
Z_BOUNDS = [
    (math.log(R_MIN), math.log(R_MAX)),
    (math.log(LSURF_MIN), math.log(LSURF_MAX)),
    (S_MIN, S_MAX),
    (D_MIN, D_MAX),
    (P_MIN, P_MAX),
    (math.log(LICE_MIN), math.log(LICE_MAX)),
]


def status(rms):
    if rms <= 2:
        return "near/reachable"
    if rms <= 5:
        return "marginal"
    return "clearly outside"


def x_to_z(x):
    return np.array([
        math.log(float(x[0])),
        math.log(float(x[1])),
        float(x[2]),
        float(x[3]),
        float(x[4]),
        math.log(float(x[5])),
    ])


def z_to_x(z):
    return np.array([
        math.exp(float(z[0])),
        math.exp(float(z[1])),
        float(z[2]),
        float(z[3]),
        float(z[4]),
        math.exp(float(z[5])),
    ])


def indices(chs):
    return np.array([CHANNELS.index(c) for c in chs], dtype=int)


def flags(x):
    lo = np.array([R_MIN, LSURF_MIN, S_MIN, D_MIN, P_MIN, LICE_MIN])
    hi = np.array([R_MAX, LSURF_MAX, S_MAX, D_MAX, P_MAX, LICE_MAX])
    tol = np.maximum((hi - lo) * 1e-3, 1e-12)
    return [
        bool(abs(x[i] - lo[i]) <= tol[i] or abs(x[i] - hi[i]) <= tol[i])
        for i in range(6)
    ]


def run_multiyear(base_state, cfg_base, angle_deg, x):
    from smrt import PSU, make_ice_column, sensor_list

    roughness, surface_corr, salinity, thickness, porosity, ice_corr = map(float, x)

    cfg = deepcopy(cfg_base)
    cfg["sensor"]["incidence_angle_deg"] = float(angle_deg)
    cfg["model"]["ice_type"] = "multiyear"
    cfg["fixed_scene"]["surface_corr_length_m"] = surface_corr
    cfg["fixed_scene"]["ice_corr_length_m"] = ice_corr

    state = base_state.replace(
        roughness_rms_m=roughness,
        salinity_ppt=salinity,
        ice_thickness_m=thickness,
    )

    surface = _build_surface(state, cfg)
    mc = cfg["model"]
    fixed = cfg["fixed_scene"]

    medium = make_ice_column(
        ice_type="multiyear",
        thickness=[thickness],
        temperature=[float(fixed["temperature_k"])],
        microstructure_model=mc["microstructure_model"],
        salinity=[salinity * PSU],
        porosity=[porosity],
        corr_length=[ice_corr],
        brine_inclusion_shape=mc["brine_inclusion_shape"],
        add_water_substrate=True,
        water_temperature=float(fixed["water_temperature_k"]),
        water_salinity=float(fixed["water_salinity_ppt"]) * PSU,
        surface=surface,
    )

    model = _model(mc["emmodel"], mc["rtsolver"])
    out = {}

    for band in ("L", "S"):
        sensor = sensor_list.active(
            frequency=float(cfg["sensor"]["frequencies_hz"][band]),
            theta_inc=float(angle_deg),
            polarization_inc=["H", "V"],
            polarization=["H", "V"],
            name=f"{band}_band",
        )
        result = model.run(sensor, medium)
        out[f"{band}_HH"] = _extract_channel(result, "H", "H", f"{band}_HH")
        out[f"{band}_VV"] = _extract_channel(result, "V", "V", f"{band}_VV")

    y = np.array([out[c] for c in CHANNELS], dtype=float)
    if y.shape != (4,) or np.any(~np.isfinite(y)):
        raise ValueError(f"Invalid prediction: {y}")
    return y


def safe_predictor(base_state, cfg_base, angle):
    cache = {}
    stats = {"runs": 0, "cache_hits": 0, "invalid": 0}

    def predict(x):
        key = tuple(round(float(v), 11) for v in x)
        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]
        stats["runs"] += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = run_multiyear(base_state, cfg_base, angle, x)
            cache[key] = y
        except Exception:
            stats["invalid"] += 1
            cache[key] = None
        return cache[key]

    return predict, stats


def rms_triplet(y, obs):
    return {
        "L_only_rms_db": float(np.sqrt(np.mean((y[:2] - obs[:2]) ** 2))),
        "S_only_rms_db": float(np.sqrt(np.mean((y[2:] - obs[2:]) ** 2))),
        "L_plus_S_rms_db": float(np.sqrt(np.mean((y - obs) ** 2))),
    }


def coarse_grid(base_state, cfg_base, angle, inc_bin, obs):
    rows = []
    total = len(ROUGHNESS_VALUES) * len(SURFACE_CORR_VALUES) * len(POROSITY_VALUES) * len(ICE_CORR_VALUES)
    n = 0

    for lice in ICE_CORR_VALUES:
        for p in POROSITY_VALUES:
            for lsurf in SURFACE_CORR_VALUES:
                for r in ROUGHNESS_VALUES:
                    n += 1
                    x = np.array([r, lsurf, COARSE_SALINITY, COARSE_THICKNESS, p, lice])
                    row = {
                        "ice_type": "MYI",
                        "incidence_bin": inc_bin,
                        "incidence_angle_deg": angle,
                        "roughness_rms_m": r,
                        "surface_corr_length_m": lsurf,
                        "salinity_ppt": COARSE_SALINITY,
                        "ice_thickness_m": COARSE_THICKNESS,
                        "porosity": p,
                        "ice_corr_length_m": lice,
                        "valid": False,
                        "error": "",
                    }
                    try:
                        y = run_multiyear(base_state, cfg_base, angle, x)
                        row["valid"] = True
                        for i, ch in enumerate(CHANNELS):
                            row[f"pred_{ch}_db"] = float(y[i])
                        row.update(rms_triplet(y, obs))
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)

                    if n % max(1, total // 10) == 0:
                        print(f"    coarse {n}/{total} ({100*n/total:.0f}%)", flush=True)

    return pd.DataFrame(rows)


def refine(valid, obs, chs, predict):
    idx = indices(chs)
    obs_sub = obs[idx]
    metric_col = "L_plus_S_rms_db" if chs == CHANNELS else "L_only_rms_db"
    metric = valid[metric_col].to_numpy(float)
    order = np.argsort(metric)

    param_cols = [
        "roughness_rms_m",
        "surface_corr_length_m",
        "salinity_ppt",
        "ice_thickness_m",
        "porosity",
        "ice_corr_length_m",
    ]
    pred_cols = [f"pred_{ch}_db" for ch in CHANNELS]

    p0 = int(order[0])
    coarse_x = valid.iloc[p0][param_cols].to_numpy(float)
    coarse_y = valid.iloc[p0][pred_cols].to_numpy(float)
    coarse_rms = float(metric[p0])

    candidates = []

    def objective(z):
        x = z_to_x(z)
        y = predict(x)
        if y is None:
            return INVALID_PENALTY
        return float(np.mean((y[idx] - obs_sub) ** 2))

    for pos in order[:N_STARTS]:
        x0 = valid.iloc[int(pos)][param_cols].to_numpy(float)
        try:
            result = minimize(
                objective,
                x0=x_to_z(x0),
                method="Powell",
                bounds=Z_BOUNDS,
                options={"maxiter": MAXITER, "xtol": 1e-5, "ftol": 1e-7, "disp": False},
            )
            x = z_to_x(result.x)
            y = predict(x)
            if y is None:
                continue
            target_rms = float(np.sqrt(np.mean((y[idx] - obs_sub) ** 2)))
            candidates.append({
                "x": x,
                "y": y,
                "target_rms": target_rms,
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(getattr(result, "nfev", 0)),
            })
        except Exception:
            continue

    candidates.append({
        "x": coarse_x,
        "y": coarse_y,
        "target_rms": coarse_rms,
        "success": False,
        "message": "coarse-grid fallback",
        "nfev": 0,
    })

    best = min(candidates, key=lambda d: d["target_rms"])
    x = np.asarray(best["x"], float)
    y = np.asarray(best["y"], float)
    f = flags(x)
    rr = rms_triplet(y, obs)

    out = {
        "coarse_target_rms_db": coarse_rms,
        "refined_target_rms_db": float(best["target_rms"]),
        "target_improvement_db": float(coarse_rms - best["target_rms"]),
        **rr,
        "joint_status": status(rr["L_plus_S_rms_db"]),
        "best_roughness_rms_m": float(x[0]),
        "best_surface_corr_length_m": float(x[1]),
        "best_salinity_ppt": float(x[2]),
        "best_ice_thickness_m": float(x[3]),
        "best_porosity": float(x[4]),
        "best_ice_corr_length_m": float(x[5]),
        "roughness_at_bound": f[0],
        "surface_corr_at_bound": f[1],
        "salinity_at_bound": f[2],
        "thickness_at_bound": f[3],
        "porosity_at_bound": f[4],
        "ice_corr_at_bound": f[5],
        "optimizer_success": bool(best["success"]),
        "optimizer_message": str(best["message"]),
        "optimizer_nfev": int(best["nfev"]),
    }

    residual = y - obs
    for i, ch in enumerate(CHANNELS):
        out[f"obs_{ch}_db"] = float(obs[i])
        out[f"fit_{ch}_db"] = float(y[i])
        out[f"residual_{ch}_db"] = float(residual[i])

    return out


def main():
    if not OBSERVATION_CSV.exists():
        raise FileNotFoundError(f"Missing:\n{OBSERVATION_CSV}")
    if not EXPERIMENT10_CSV.exists():
        raise FileNotFoundError(f"Missing:\n{EXPERIMENT10_CSV}")

    obs_df = pd.read_csv(OBSERVATION_CSV)
    obs_df = obs_df[obs_df["ice_type"].astype(str) == "MYI"].copy()
    exp10 = pd.read_csv(EXPERIMENT10_CSV)

    cfg = load_config()
    base = baseline_state(cfg)
    results_dir = ensure_results_dir()

    print("=" * 78)
    print("MYI TWO-LENGTH-SCALE REACHABILITY TEST")
    print("=" * 78)
    print("SMRT multiyear; actual mean incidence angle per bin.")
    print("surface corr length: 0.5-5.0 cm")
    print("ice corr length: 0.5-12 mm")
    print("salinity/thickness/porosity bounds unchanged.")

    coarse_all = []
    rows = []
    details = {}

    for k, (_, rec) in enumerate(obs_df.iterrows(), start=1):
        inc_bin = str(rec["incidence_bin"])
        angle = float(rec["incidence_mean_deg"])
        obs = np.array([float(rec[f"{ch}_mean_db"]) for ch in CHANNELS])

        print("\n" + "-" * 78)
        print(f"[{k}/{len(obs_df)}] MYI | {inc_bin} deg | theta={angle:.3f}")
        print("-" * 78)

        coarse = coarse_grid(base, cfg, angle, inc_bin, obs)
        coarse_all.append(coarse)
        valid = coarse[coarse["valid"] == True].copy()  # noqa: E712
        print(f"  valid coarse states: {len(valid):,}/{len(coarse):,}")
        if valid.empty:
            raise RuntimeError(f"No valid states for MYI {inc_bin}")

        predict, stats = safe_predictor(base, cfg, angle)
        details[inc_bin] = {"subsets": {}}

        for subset_name, chs in SUBSETS.items():
            result = refine(valid, obs, chs, predict)
            rows.append({
                "ice_type": "MYI",
                "incidence_bin": inc_bin,
                "incidence_angle_deg": angle,
                "optimized_subset": subset_name,
                **result,
            })
            details[inc_bin]["subsets"][subset_name] = result

            print(
                f"  optimize {subset_name:<8} | "
                f"target={result['refined_target_rms_db']:6.3f} | "
                f"joint={result['L_plus_S_rms_db']:6.3f} | "
                f"L={result['L_only_rms_db']:6.3f} | "
                f"S={result['S_only_rms_db']:6.3f} | "
                f"{result['joint_status']}"
            )

        details[inc_bin]["evaluation_stats"] = stats
        print(f"  refinement SMRT runs={stats['runs']}, invalid_trials={stats['invalid']}")

    coarse_df = pd.concat(coarse_all, ignore_index=True)
    result_df = pd.DataFrame(rows)

    coarse_csv = results_dir / "11_myi_length_scale_coarse_grid.csv"
    result_csv = results_dir / "11_myi_length_scale_reachability.csv"
    result_json = results_dir / "11_myi_length_scale_reachability.json"
    compare_csv = results_dir / "11_myi_length_scale_vs_experiment10.csv"

    coarse_df.to_csv(coarse_csv, index=False)
    result_df.to_csv(result_csv, index=False)
    result_json.write_text(json.dumps({
        "purpose": "MYI reachability with free surface and ice microstructure correlation lengths",
        "surface_corr_length_domain_m": [LSURF_MIN, LSURF_MAX],
        "ice_corr_length_domain_m": [LICE_MIN, LICE_MAX],
        "details": details,
    }, indent=2), encoding="utf-8")

    new_joint = result_df[result_df["optimized_subset"] == "L_plus_S"][
        [
            "incidence_bin",
            "L_plus_S_rms_db",
            "L_only_rms_db",
            "S_only_rms_db",
            "best_roughness_rms_m",
            "best_surface_corr_length_m",
            "best_salinity_ppt",
            "best_ice_thickness_m",
            "best_porosity",
            "best_ice_corr_length_m",
        ]
    ].rename(columns={"L_plus_S_rms_db": "experiment11_joint_rms_db"})

    old_joint = exp10[exp10["subset"].astype(str) == "L_plus_S"][
        ["incidence_bin", "refined_rms_mismatch_db"]
    ].rename(columns={"refined_rms_mismatch_db": "experiment10_joint_rms_db"})

    comparison = old_joint.merge(new_joint, on="incidence_bin", how="inner")
    comparison["improvement_vs_experiment10_db"] = (
        comparison["experiment10_joint_rms_db"] - comparison["experiment11_joint_rms_db"]
    )
    comparison.to_csv(compare_csv, index=False)

    final = result_df[result_df["optimized_subset"] == "L_plus_S"].copy()

    print("\n" + "=" * 78)
    print("FINAL TWO-LENGTH-SCALE RESULTS")
    print("=" * 78)

    cols = [
        "incidence_bin",
        "incidence_angle_deg",
        "L_only_rms_db",
        "S_only_rms_db",
        "L_plus_S_rms_db",
        "joint_status",
        "best_roughness_rms_m",
        "best_surface_corr_length_m",
        "best_salinity_ppt",
        "best_ice_thickness_m",
        "best_porosity",
        "best_ice_corr_length_m",
        "surface_corr_at_bound",
        "ice_corr_at_bound",
        "salinity_at_bound",
        "thickness_at_bound",
        "porosity_at_bound",
    ]
    print(final[cols].sort_values("incidence_angle_deg").to_string(index=False))

    print("\nEXPERIMENT 10 vs 11:")
    print(comparison.sort_values("incidence_bin").to_string(index=False))

    print("\nSaved:")
    print(f"  {coarse_csv}")
    print(f"  {result_csv}")
    print(f"  {result_json}")
    print(f"  {compare_csv}")

    print(
        "\nSTOP HERE. If joint reachability improves strongly, constrain these "
        "two length scales from literature/field data next. If L remains several "
        "dB too dark, move to snow/interface/deformation physics rather than "
        "blindly expanding salinity, thickness or porosity bounds."
    )


if __name__ == "__main__":
    main()
