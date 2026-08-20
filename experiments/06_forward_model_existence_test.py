from __future__ import annotations

"""
06_forward_model_existence_test.py

Tests whether the current SMRT + interface model can reproduce the real mean
winter observations for MYI, NI, and TI using L_HH, L_VV, S_HH, and S_VV.

This is a forward-model reachability test, not the final real-data inversion.

Run from the project root:
    conda activate geo
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python experiments/06_forward_model_existence_test.py
"""

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

from src.common import baseline_state, ensure_results_dir, load_config, vector_from_prediction
from src.forward_smrt import run_forward

OBSERVATION_JSON = (
    ROOT / "data" / "derived" / "real_observations_for_inversion"
    / "real_observation_summary.json"
)

CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")

N_ROUGHNESS = 21
N_SALINITY = 21
N_THICKNESS = 15

THICKNESS_MIN_M = 0.20
THICKNESS_MAX_M = 3.00

DISTANCE_SCALE_DB = np.ones(4, dtype=float)
N_REFINEMENT_STARTS = 8


def load_observations(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Observation file not found:\n{path}\n"
            "Generate it first with experiments/prepare_real_observations.py."
        )

    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"ice_type", *(f"{ch}_mean_db" for ch in CHANNELS)}

    for i, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Record {i} is missing: {sorted(missing)}")

    return records


def get_bounds(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = cfg["inversion"]["bounds"]

    rmin, rmax = map(float, bounds["roughness_rms_m"])
    smin, smax = map(float, bounds["salinity_ppt"])

    if "ice_thickness_m" in bounds:
        tmin, tmax = map(float, bounds["ice_thickness_m"])
    else:
        tmin, tmax = THICKNESS_MIN_M, THICKNESS_MAX_M

    lower = np.array([rmin, smin, tmin], dtype=float)
    upper = np.array([rmax, smax, tmax], dtype=float)

    if np.any(lower >= upper):
        raise ValueError(f"Invalid bounds: lower={lower}, upper={upper}")

    return lower, upper


def predict(base, cfg: dict, x: np.ndarray) -> np.ndarray:
    state = base.replace(
        roughness_rms_m=float(x[0]),
        salinity_ppt=float(x[1]),
        ice_thickness_m=float(x[2]),
    )

    y = vector_from_prediction(run_forward(state, cfg), CHANNELS).astype(float)

    if y.shape != (4,) or np.any(~np.isfinite(y)):
        raise ValueError(f"Invalid prediction: {y}")

    return y


def residual(base, cfg: dict, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return (predict(base, cfg, x) - obs) / DISTANCE_SCALE_DB


def rms_db(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r**2)))


def status_from_rms(value: float) -> str:
    if value <= 2.0:
        return "near/reachable"
    if value <= 5.0:
        return "marginal"
    return "clearly outside"


def bound_flags(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[bool]:
    tol = np.maximum((upper - lower) * 1e-3, 1e-12)
    return [
        bool(abs(x[i] - lower[i]) <= tol[i] or abs(x[i] - upper[i]) <= tol[i])
        for i in range(3)
    ]


def main() -> None:
    cfg = load_config()
    base = baseline_state(cfg)
    results_dir = ensure_results_dir()

    observations = load_observations(OBSERVATION_JSON)
    lower, upper = get_bounds(cfg)

    roughness_values = np.geomspace(lower[0], upper[0], N_ROUGHNESS)
    salinity_values = np.linspace(lower[1], upper[1], N_SALINITY)
    thickness_values = np.linspace(lower[2], upper[2], N_THICKNESS)

    total = N_ROUGHNESS * N_SALINITY * N_THICKNESS

    print("Forward-model existence test")
    print(f"Total coarse states: {total:,}")
    print(
        f"Bounds: roughness={lower[0]:.6g}..{upper[0]:.6g} m, "
        f"salinity={lower[1]:.6g}..{upper[1]:.6g} ppt, "
        f"thickness={lower[2]:.6g}..{upper[2]:.6g} m"
    )

    rows = []
    count = 0

    for thickness in thickness_values:
        for salinity in salinity_values:
            for roughness in roughness_values:
                count += 1
                x = np.array([roughness, salinity, thickness], dtype=float)

                row = {
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
                        y = predict(base, cfg, x)

                    row["valid"] = True
                    for ch, value in zip(CHANNELS, y):
                        row[f"pred_{ch}_db"] = float(value)

                    if caught:
                        row["warning"] = " | ".join(str(w.message) for w in caught)

                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"

                rows.append(row)

                if count % max(1, total // 20) == 0 or count == total:
                    print(f"  {count:,}/{total:,} ({100 * count / total:.0f}%)")

    grid = pd.DataFrame(rows)
    grid_path = results_dir / "06_forward_model_reachable_grid.csv"
    grid.to_csv(grid_path, index=False)

    valid = grid[grid["valid"]].copy()
    if valid.empty:
        raise RuntimeError(f"No valid states. Inspect {grid_path}")

    pred_cols = [f"pred_{ch}_db" for ch in CHANNELS]
    param_cols = ["roughness_rms_m", "salinity_ppt", "ice_thickness_m"]

    model_ranges = {
        ch: {
            "min_db": float(valid[f"pred_{ch}_db"].min()),
            "max_db": float(valid[f"pred_{ch}_db"].max()),
        }
        for ch in CHANNELS
    }

    pred_matrix = valid[pred_cols].to_numpy(float)

    summary_rows = []
    details = {}

    for record in observations:
        ice_type = str(record["ice_type"])
        obs = np.array([record[f"{ch}_mean_db"] for ch in CHANNELS], dtype=float)

        coarse_residuals = pred_matrix - obs[None, :]
        coarse_scores = np.sum((coarse_residuals / DISTANCE_SCALE_DB) ** 2, axis=1)
        order = np.argsort(coarse_scores)

        refined = []

        for pos in order[:N_REFINEMENT_STARTS]:
            x0 = valid.iloc[int(pos)][param_cols].to_numpy(float)

            try:
                result = least_squares(
                    lambda x: residual(base, cfg, x, obs),
                    x0=x0,
                    bounds=(lower, upper),
                    method="trf",
                    x_scale="jac",
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=300,
                )

                y_fit = predict(base, cfg, result.x)
                r = y_fit - obs

                refined.append(
                    {
                        "result": result,
                        "prediction": y_fit,
                        "residual": r,
                        "score": float(np.sum((r / DISTANCE_SCALE_DB) ** 2)),
                    }
                )

            except Exception as exc:
                print(f"  Refinement warning for {ice_type}: {exc}")

        if refined:
            best = min(refined, key=lambda item: item["score"])
            result = best["result"]
            x_best = np.asarray(result.x, dtype=float)
            y_best = best["prediction"]
            r_best = best["residual"]
            success = bool(result.success)
            message = str(result.message)
            nfev = int(result.nfev)
        else:
            row0 = valid.iloc[int(order[0])]
            x_best = row0[param_cols].to_numpy(float)
            y_best = row0[pred_cols].to_numpy(float)
            r_best = y_best - obs
            success = False
            message = "All refinements failed; coarse-grid minimum used."
            nfev = 0

        rms = rms_db(r_best)
        flags = bound_flags(x_best, lower, upper)

        scalar_inside = {
            ch: bool(model_ranges[ch]["min_db"] <= obs[i] <= model_ranges[ch]["max_db"])
            for i, ch in enumerate(CHANNELS)
        }

        row = {
            "ice_type": ice_type,
            "status": status_from_rms(rms),
            "best_rms_mismatch_db": rms,
            "best_half_sum_squared_1db": 0.5 * float(np.sum(r_best**2)),
            "best_roughness_rms_m": float(x_best[0]),
            "best_salinity_ppt": float(x_best[1]),
            "best_ice_thickness_m": float(x_best[2]),
            "roughness_at_bound": flags[0],
            "salinity_at_bound": flags[1],
            "thickness_at_bound": flags[2],
            "optimizer_success": success,
            "optimizer_message": message,
            "optimizer_nfev": nfev,
        }

        for i, ch in enumerate(CHANNELS):
            row[f"obs_{ch}_db"] = float(obs[i])
            row[f"fit_{ch}_db"] = float(y_best[i])
            row[f"residual_{ch}_db"] = float(r_best[i])
            row[f"obs_{ch}_inside_scalar_range"] = scalar_inside[ch]

        summary_rows.append(row)

        details[ice_type] = {
            "observation_db": {ch: float(obs[i]) for i, ch in enumerate(CHANNELS)},
            "best_state": {
                "roughness_rms_m": float(x_best[0]),
                "salinity_ppt": float(x_best[1]),
                "ice_thickness_m": float(x_best[2]),
            },
            "best_prediction_db": {ch: float(y_best[i]) for i, ch in enumerate(CHANNELS)},
            "residual_db_model_minus_observation": {
                ch: float(r_best[i]) for i, ch in enumerate(CHANNELS)
            },
            "rms_mismatch_db": rms,
            "status": status_from_rms(rms),
            "parameter_at_bound": {
                "roughness_rms_m": flags[0],
                "salinity_ppt": flags[1],
                "ice_thickness_m": flags[2],
            },
            "observed_channel_inside_global_scalar_range": scalar_inside,
            "optimizer_success": success,
            "optimizer_message": message,
        }

        print(f"\n{ice_type}")
        print(f"  status: {status_from_rms(rms)}")
        print(f"  RMS mismatch: {rms:.3f} dB")
        print(
            f"  state: roughness={x_best[0]:.8g} m, "
            f"salinity={x_best[1]:.5g} ppt, "
            f"thickness={x_best[2]:.5g} m"
        )
        for ch, o, f, r in zip(CHANNELS, obs, y_best, r_best):
            print(f"  {ch}: obs={o:.3f}, fit={f:.3f}, residual={r:+.3f} dB")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = results_dir / "06_forward_model_existence_summary.csv"
    summary_json = results_dir / "06_forward_model_existence_summary.json"

    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "channels": list(CHANNELS),
        "observation_file": str(OBSERVATION_JSON),
        "note": (
            "Equal 1 dB scaling is only a common numerical distance. "
            "It is not the final real observation covariance."
        ),
        "parameter_bounds": {
            "roughness_rms_m": [float(lower[0]), float(upper[0])],
            "salinity_ppt": [float(lower[1]), float(upper[1])],
            "ice_thickness_m": [float(lower[2]), float(upper[2])],
        },
        "coarse_grid": {
            "roughness_count": N_ROUGHNESS,
            "salinity_count": N_SALINITY,
            "thickness_count": N_THICKNESS,
            "total_states": int(total),
            "valid_states": int(len(valid)),
        },
        "global_modeled_channel_ranges_db": model_ranges,
        "classes": details,
    }

    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for row in summary_rows:
        ice_type = row["ice_type"]
        obs = np.array([row[f"obs_{ch}_db"] for ch in CHANNELS])
        fit = np.array([row[f"fit_{ch}_db"] for ch in CHANNELS])

        positions = np.arange(len(CHANNELS))
        width = 0.36

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.bar(positions - width / 2, obs, width=width, label="Observed mean")
        ax.bar(positions + width / 2, fit, width=width, label="Closest modeled")
        ax.set_xticks(positions)
        ax.set_xticklabels(CHANNELS)
        ax.set_ylabel("Backscatter (dB)")
        ax.set_title(
            f"{ice_type}: observed vs closest modeled "
            f"(RMS mismatch {row['best_rms_mismatch_db']:.2f} dB)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            results_dir / f"06_existence_{ice_type}_observed_vs_modeled.png",
            dpi=200,
        )
        plt.close(fig)

    cloud = valid.sample(n=min(15000, len(valid)), random_state=42)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(cloud["pred_L_HH_db"], cloud["pred_L_VV_db"], s=8, alpha=0.25)
    for record in observations:
        ax.scatter(
            record["L_HH_mean_db"],
            record["L_VV_mean_db"],
            s=100,
            marker="*",
            label=f"{record['ice_type']} observed",
        )
    ax.set_xlabel("L_HH (dB)")
    ax.set_ylabel("L_VV (dB)")
    ax.set_title("L-band reachable model cloud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "06_reachable_cloud_L_band.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(cloud["pred_S_HH_db"], cloud["pred_S_VV_db"], s=8, alpha=0.25)
    for record in observations:
        ax.scatter(
            record["S_HH_mean_db"],
            record["S_VV_mean_db"],
            s=100,
            marker="*",
            label=f"{record['ice_type']} observed",
        )
    ax.set_xlabel("S_HH (dB)")
    ax.set_ylabel("S_VV (dB)")
    ax.set_title("S-band reachable model cloud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "06_reachable_cloud_S_band.png", dpi=200)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {grid_path}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {results_dir / '06_reachable_cloud_L_band.png'}")
    print(f"  {results_dir / '06_reachable_cloud_S_band.png'}")


if __name__ == "__main__":
    main()
