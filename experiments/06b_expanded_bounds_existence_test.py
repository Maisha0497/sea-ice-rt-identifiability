from __future__ import annotations

"""
06b_expanded_bounds_existence_test.py

Repeat the real-observation forward-model existence test using wider parameter
bounds without editing config/base.yaml.

Run from project root:
    conda activate geo
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python experiments/06b_expanded_bounds_existence_test.py
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

ROUGHNESS_MIN_M = 0.00025
ROUGHNESS_MAX_M = 0.00300
SALINITY_MIN_PPT = 0.5
SALINITY_MAX_PPT = 15.0
THICKNESS_MIN_M = 0.20
THICKNESS_MAX_M = 6.00

N_ROUGHNESS = 25
N_SALINITY = 25
N_THICKNESS = 19
N_REFINEMENT_STARTS = 10
DISTANCE_SCALE_DB = np.ones(4, dtype=float)


def load_observations(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Observation file not found:\n{path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"ice_type", *(f"{ch}_mean_db" for ch in CHANNELS)}
    for i, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Record {i} missing fields: {sorted(missing)}")
    return records


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


def rms_db(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def at_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[bool]:
    tolerance = np.maximum((upper - lower) * 1e-3, 1e-12)
    return [
        bool(abs(x[i] - lower[i]) <= tolerance[i] or abs(x[i] - upper[i]) <= tolerance[i])
        for i in range(3)
    ]


def classify(rms: float) -> str:
    if rms <= 2.0:
        return "near/reachable"
    if rms <= 5.0:
        return "marginal"
    return "clearly outside"


def main() -> None:
    cfg = load_config()
    base = baseline_state(cfg)
    results_dir = ensure_results_dir()
    observations = load_observations(OBSERVATION_JSON)

    lower = np.array([ROUGHNESS_MIN_M, SALINITY_MIN_PPT, THICKNESS_MIN_M], dtype=float)
    upper = np.array([ROUGHNESS_MAX_M, SALINITY_MAX_PPT, THICKNESS_MAX_M], dtype=float)

    roughness_values = np.geomspace(ROUGHNESS_MIN_M, ROUGHNESS_MAX_M, N_ROUGHNESS)
    salinity_values = np.linspace(SALINITY_MIN_PPT, SALINITY_MAX_PPT, N_SALINITY)
    thickness_values = np.linspace(THICKNESS_MIN_M, THICKNESS_MAX_M, N_THICKNESS)

    total = N_ROUGHNESS * N_SALINITY * N_THICKNESS
    print("Expanded-bounds forward-model existence test")
    print(f"Total coarse states: {total:,}")
    print(
        f"Roughness: {ROUGHNESS_MIN_M:g} to {ROUGHNESS_MAX_M:g} m\n"
        f"Salinity:  {SALINITY_MIN_PPT:g} to {SALINITY_MAX_PPT:g} ppt\n"
        f"Thickness: {THICKNESS_MIN_M:g} to {THICKNESS_MAX_M:g} m"
    )

    rows: list[dict[str, object]] = []
    count = 0
    for thickness in thickness_values:
        for salinity in salinity_values:
            for roughness in roughness_values:
                count += 1
                x = np.array([roughness, salinity, thickness], dtype=float)
                row: dict[str, object] = {
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
    grid_path = results_dir / "06b_expanded_reachable_grid.csv"
    grid.to_csv(grid_path, index=False)

    valid = grid[grid["valid"]].copy()
    if valid.empty:
        raise RuntimeError(f"No valid states. Inspect {grid_path}")

    pred_cols = [f"pred_{ch}_db" for ch in CHANNELS]
    param_cols = ["roughness_rms_m", "salinity_ppt", "ice_thickness_m"]
    pred_matrix = valid[pred_cols].to_numpy(dtype=float)

    modeled_ranges = {
        ch: {
            "min_db": float(valid[f"pred_{ch}_db"].min()),
            "max_db": float(valid[f"pred_{ch}_db"].max()),
        }
        for ch in CHANNELS
    }

    output_rows: list[dict[str, object]] = []
    details: dict[str, dict] = {}

    for record in observations:
        ice_type = str(record["ice_type"])
        obs = np.array([record[f"{ch}_mean_db"] for ch in CHANNELS], dtype=float)
        coarse_residual = pred_matrix - obs[None, :]
        score = np.sum((coarse_residual / DISTANCE_SCALE_DB) ** 2, axis=1)
        order = np.argsort(score)

        candidates = []
        for pos in order[:N_REFINEMENT_STARTS]:
            x0 = valid.iloc[int(pos)][param_cols].to_numpy(dtype=float)
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
                    max_nfev=400,
                )
                fit = predict(base, cfg, result.x)
                r = fit - obs
                candidates.append({
                    "result": result,
                    "fit": fit,
                    "residual": r,
                    "score": float(np.sum(r**2)),
                })
            except Exception as exc:
                print(f"  Refinement warning for {ice_type}: {exc}")

        if candidates:
            best = min(candidates, key=lambda item: item["score"])
            result = best["result"]
            x_best = np.asarray(result.x, dtype=float)
            fit = best["fit"]
            r = best["residual"]
            success = bool(result.success)
            message = str(result.message)
        else:
            coarse = valid.iloc[int(order[0])]
            x_best = coarse[param_cols].to_numpy(dtype=float)
            fit = coarse[pred_cols].to_numpy(dtype=float)
            r = fit - obs
            success = False
            message = "All refinements failed; coarse minimum used."

        flags = at_bounds(x_best, lower, upper)
        mismatch = rms_db(r)
        scalar_inside = {
            ch: bool(modeled_ranges[ch]["min_db"] <= obs[i] <= modeled_ranges[ch]["max_db"])
            for i, ch in enumerate(CHANNELS)
        }

        row: dict[str, object] = {
            "ice_type": ice_type,
            "status": classify(mismatch),
            "rms_mismatch_db": mismatch,
            "best_roughness_rms_m": float(x_best[0]),
            "best_salinity_ppt": float(x_best[1]),
            "best_ice_thickness_m": float(x_best[2]),
            "roughness_at_bound": flags[0],
            "salinity_at_bound": flags[1],
            "thickness_at_bound": flags[2],
            "optimizer_success": success,
            "optimizer_message": message,
        }
        for i, ch in enumerate(CHANNELS):
            row[f"obs_{ch}_db"] = float(obs[i])
            row[f"fit_{ch}_db"] = float(fit[i])
            row[f"residual_{ch}_db"] = float(r[i])
            row[f"obs_{ch}_inside_scalar_range"] = scalar_inside[ch]
        output_rows.append(row)

        details[ice_type] = {
            "observation_db": {ch: float(obs[i]) for i, ch in enumerate(CHANNELS)},
            "best_state": {
                "roughness_rms_m": float(x_best[0]),
                "salinity_ppt": float(x_best[1]),
                "ice_thickness_m": float(x_best[2]),
            },
            "best_prediction_db": {ch: float(fit[i]) for i, ch in enumerate(CHANNELS)},
            "residual_db_model_minus_observation": {ch: float(r[i]) for i, ch in enumerate(CHANNELS)},
            "rms_mismatch_db": mismatch,
            "status": classify(mismatch),
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
        print(f"  status: {classify(mismatch)}")
        print(f"  RMS mismatch: {mismatch:.3f} dB")
        print(
            f"  state: roughness={x_best[0]:.8g} m, "
            f"salinity={x_best[1]:.5g} ppt, "
            f"thickness={x_best[2]:.5g} m"
        )
        for ch, observed, modeled, residual_value in zip(CHANNELS, obs, fit, r):
            print(
                f"  {ch}: obs={observed:.3f}, fit={modeled:.3f}, "
                f"residual={residual_value:+.3f} dB"
            )

    summary_df = pd.DataFrame(output_rows)
    summary_csv = results_dir / "06b_expanded_existence_summary.csv"
    summary_json = results_dir / "06b_expanded_existence_summary.json"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "channels": list(CHANNELS),
        "observation_file": str(OBSERVATION_JSON),
        "note": (
            "This is an expanded-bound diagnostic. Equal 1 dB scaling is only "
            "a common distance, not the final real covariance."
        ),
        "expanded_bounds": {
            "roughness_rms_m": [ROUGHNESS_MIN_M, ROUGHNESS_MAX_M],
            "salinity_ppt": [SALINITY_MIN_PPT, SALINITY_MAX_PPT],
            "ice_thickness_m": [THICKNESS_MIN_M, THICKNESS_MAX_M],
        },
        "coarse_grid": {
            "roughness_count": N_ROUGHNESS,
            "salinity_count": N_SALINITY,
            "thickness_count": N_THICKNESS,
            "total_states": total,
            "valid_states": int(len(valid)),
        },
        "global_modeled_channel_ranges_db": modeled_ranges,
        "classes": details,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cloud = valid.sample(n=min(20000, len(valid)), random_state=42)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(cloud["pred_L_HH_db"], cloud["pred_L_VV_db"], s=8, alpha=0.25, label="Expanded reachable cloud")
    for record in observations:
        ax.scatter(record["L_HH_mean_db"], record["L_VV_mean_db"], s=100, marker="*", label=f"{record['ice_type']} observed")
    ax.set_xlabel("L_HH (dB)")
    ax.set_ylabel("L_VV (dB)")
    ax.set_title("Expanded L-band reachable model cloud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "06b_expanded_cloud_L_band.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(cloud["pred_S_HH_db"], cloud["pred_S_VV_db"], s=8, alpha=0.25, label="Expanded reachable cloud")
    for record in observations:
        ax.scatter(record["S_HH_mean_db"], record["S_VV_mean_db"], s=100, marker="*", label=f"{record['ice_type']} observed")
    ax.set_xlabel("S_HH (dB)")
    ax.set_ylabel("S_VV (dB)")
    ax.set_title("Expanded S-band reachable model cloud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "06b_expanded_cloud_S_band.png", dpi=200)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {grid_path}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {results_dir / '06b_expanded_cloud_L_band.png'}")
    print(f"  {results_dir / '06b_expanded_cloud_S_band.png'}")


if __name__ == "__main__":
    main()
