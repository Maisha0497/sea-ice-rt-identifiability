from __future__ import annotations

"""
05b_synthetic_cost_surface.py

Diagnose the roughness-salinity synthetic inversion by evaluating the same
weighted least-squares objective on a 2-D grid.

It computes TWO surfaces:
1. noisy-data cost: explains why the noisy inversion moved away from truth;
2. clean-data cost: checks whether the forward/inverse implementation can
   recover the exact synthetic truth without noise.

Run from the project root:
    python experiments/05b_synthetic_cost_surface.py

Required existing file:
    results/05_synthetic_summary.json
"""

from pathlib import Path
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    baseline_state,
    ensure_results_dir,
    load_config,
    noise_vector,
    vector_from_prediction,
)
from src.forward_smrt import run_forward


CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")

# Resolution can be increased later. 41 x 41 = 1,681 forward runs.
N_ROUGHNESS = 41
N_SALINITY = 41


def weighted_cost(
    predicted_db: np.ndarray,
    observed_db: np.ndarray,
    sigma_db: np.ndarray,
) -> float:
    """Match scipy.optimize.least_squares: cost = 0.5 * sum(r_whitened**2)."""
    residual_whitened = (predicted_db - observed_db) / sigma_db
    return 0.5 * float(np.dot(residual_whitened, residual_whitened))


def nearest_grid_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def main() -> None:
    cfg = load_config()
    base = baseline_state(cfg)
    results_dir = ensure_results_dir()

    summary_path = results_dir / "05_synthetic_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run experiments/05_synthetic_inversion.py first."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    summary_channels = tuple(summary.get("channels", CHANNELS))
    if summary_channels != CHANNELS:
        raise ValueError(
            f"Expected channels {CHANNELS}, but summary contains {summary_channels}."
        )

    truth = summary["truth"]
    retrieved = summary["best_retrieval"]

    clean_obs = np.array(
        [summary["clean_observation_db"][ch] for ch in CHANNELS],
        dtype=float,
    )
    noisy_obs = np.array(
        [summary["noisy_observation_db"][ch] for ch in CHANNELS],
        dtype=float,
    )
    sigma = noise_vector(cfg, CHANNELS).astype(float)

    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError(f"All noise standard deviations must be finite and > 0. Got {sigma}")

    bounds = cfg["inversion"]["bounds"]
    r_min, r_max = map(float, bounds["roughness_rms_m"])
    s_min, s_max = map(float, bounds["salinity_ppt"])

    # Roughness is sampled logarithmically because the inversion uses positive
    # physical parameters spanning more than one order of magnitude.
    roughness_values = np.geomspace(r_min, r_max, N_ROUGHNESS)
    salinity_values = np.linspace(s_min, s_max, N_SALINITY)

    rows: list[dict[str, float | bool | str]] = []
    total = N_ROUGHNESS * N_SALINITY
    counter = 0

    print(f"Evaluating {total} forward states...")
    print(f"Channels: {CHANNELS}")
    print(f"Roughness range: {r_min:g} to {r_max:g} m")
    print(f"Salinity range: {s_min:g} to {s_max:g} ppt")

    for salinity in salinity_values:
        for roughness in roughness_values:
            counter += 1
            state = base.replace(
                roughness_rms_m=float(roughness),
                salinity_ppt=float(salinity),
                # Thickness remains fixed exactly as in the synthetic inversion.
                ice_thickness_m=float(truth["ice_thickness_m"]),
            )

            row: dict[str, float | bool | str] = {
                "roughness_rms_m": float(roughness),
                "salinity_ppt": float(salinity),
                "ice_thickness_m": float(truth["ice_thickness_m"]),
                "valid": False,
                "error": "",
                "cost_clean": np.nan,
                "cost_noisy": np.nan,
            }

            try:
                # Preserve warnings in the output rather than silently ignoring them.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pred = vector_from_prediction(
                        run_forward(state, cfg),
                        CHANNELS,
                    ).astype(float)

                if np.any(~np.isfinite(pred)):
                    raise ValueError(f"Non-finite prediction: {pred}")

                row.update(
                    {
                        "valid": True,
                        "cost_clean": weighted_cost(pred, clean_obs, sigma),
                        "cost_noisy": weighted_cost(pred, noisy_obs, sigma),
                    }
                )
                for ch, value in zip(CHANNELS, pred):
                    row[f"pred_{ch}_db"] = float(value)

                if caught:
                    row["error"] = " | ".join(str(w.message) for w in caught)

            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

            rows.append(row)

            if counter % max(1, total // 20) == 0 or counter == total:
                print(f"  {counter}/{total} ({100.0 * counter / total:.0f}%)")

    df = pd.DataFrame(rows)
    csv_path = results_dir / "05b_synthetic_cost_surface.csv"
    df.to_csv(csv_path, index=False)

    valid = df[df["valid"]].copy()
    if valid.empty:
        raise RuntimeError(
            "No valid forward states were produced. Inspect the error column in "
            f"{csv_path}."
        )

    clean_best = valid.loc[valid["cost_clean"].idxmin()]
    noisy_best = valid.loc[valid["cost_noisy"].idxmin()]

    # Evaluate exact truth and exact optimizer retrieval, not merely nearest grid nodes.
    def evaluate_exact(roughness: float, salinity: float) -> dict[str, float]:
        state = base.replace(
            roughness_rms_m=float(roughness),
            salinity_ppt=float(salinity),
            ice_thickness_m=float(truth["ice_thickness_m"]),
        )
        pred = vector_from_prediction(run_forward(state, cfg), CHANNELS).astype(float)
        return {
            "cost_clean": weighted_cost(pred, clean_obs, sigma),
            "cost_noisy": weighted_cost(pred, noisy_obs, sigma),
        }

    truth_cost = evaluate_exact(
        float(truth["roughness_rms_m"]),
        float(truth["salinity_ppt"]),
    )
    retrieval_cost = evaluate_exact(
        float(retrieved["roughness_rms_m"]),
        float(retrieved["salinity_ppt"]),
    )

    payload = {
        "channels": list(CHANNELS),
        "grid": {
            "roughness_sampling": "geometric",
            "roughness_min_m": r_min,
            "roughness_max_m": r_max,
            "roughness_count": N_ROUGHNESS,
            "salinity_sampling": "linear",
            "salinity_min_ppt": s_min,
            "salinity_max_ppt": s_max,
            "salinity_count": N_SALINITY,
            "valid_states": int(valid.shape[0]),
            "total_states": int(df.shape[0]),
        },
        "truth": truth,
        "optimizer_retrieval": retrieved,
        "exact_truth_cost": truth_cost,
        "exact_optimizer_retrieval_cost": retrieval_cost,
        "clean_grid_minimum": {
            "roughness_rms_m": float(clean_best["roughness_rms_m"]),
            "salinity_ppt": float(clean_best["salinity_ppt"]),
            "cost": float(clean_best["cost_clean"]),
        },
        "noisy_grid_minimum": {
            "roughness_rms_m": float(noisy_best["roughness_rms_m"]),
            "salinity_ppt": float(noisy_best["salinity_ppt"]),
            "cost": float(noisy_best["cost_noisy"]),
        },
    }

    json_path = results_dir / "05b_synthetic_cost_surface_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build matrices for plotting.
    clean_matrix = (
        df.pivot(
            index="salinity_ppt",
            columns="roughness_rms_m",
            values="cost_clean",
        )
        .reindex(index=salinity_values, columns=roughness_values)
        .to_numpy(dtype=float)
    )
    noisy_matrix = (
        df.pivot(
            index="salinity_ppt",
            columns="roughness_rms_m",
            values="cost_noisy",
        )
        .reindex(index=salinity_values, columns=roughness_values)
        .to_numpy(dtype=float)
    )

    def plot_surface(
        matrix: np.ndarray,
        title: str,
        filename: str,
        grid_best: pd.Series,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

        # log10(1 + cost) keeps both the minimum and broad high-cost region visible.
        display = np.log10(1.0 + matrix)
        mesh = ax.pcolormesh(
            roughness_values * 1000.0,
            salinity_values,
            display,
            shading="auto",
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"$\log_{10}(1+\mathrm{cost})$")

        finite_cost = matrix[np.isfinite(matrix)]
        if finite_cost.size:
            levels = np.unique(
                np.quantile(finite_cost, [0.02, 0.05, 0.10, 0.25, 0.50])
            )
            levels = levels[np.isfinite(levels)]
            if levels.size >= 2:
                ax.contour(
                    roughness_values * 1000.0,
                    salinity_values,
                    matrix,
                    levels=levels,
                    linewidths=0.8,
                )

        ax.scatter(
            float(truth["roughness_rms_m"]) * 1000.0,
            float(truth["salinity_ppt"]),
            marker="*",
            s=180,
            label="Synthetic truth",
        )
        ax.scatter(
            float(retrieved["roughness_rms_m"]) * 1000.0,
            float(retrieved["salinity_ppt"]),
            marker="x",
            s=100,
            label="Optimizer retrieval",
        )
        ax.scatter(
            float(grid_best["roughness_rms_m"]) * 1000.0,
            float(grid_best["salinity_ppt"]),
            marker="o",
            facecolors="none",
            s=100,
            label="Grid minimum",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Surface RMS roughness (mm)")
        ax.set_ylabel("Ice salinity (ppt)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=200)
        plt.close(fig)

    plot_surface(
        clean_matrix,
        "Synthetic cost surface — clean observations",
        "05b_cost_surface_clean.png",
        clean_best,
    )
    plot_surface(
        noisy_matrix,
        "Synthetic cost surface — noisy observations",
        "05b_cost_surface_noisy.png",
        noisy_best,
    )

    print("\nExact-state costs")
    print(
        f"  Truth:     clean={truth_cost['cost_clean']:.6g}, "
        f"noisy={truth_cost['cost_noisy']:.6g}"
    )
    print(
        f"  Retrieval: clean={retrieval_cost['cost_clean']:.6g}, "
        f"noisy={retrieval_cost['cost_noisy']:.6g}"
    )

    print("\nGrid minima")
    print(
        "  Clean: "
        f"roughness={clean_best['roughness_rms_m']:.9g} m, "
        f"salinity={clean_best['salinity_ppt']:.6g} ppt, "
        f"cost={clean_best['cost_clean']:.6g}"
    )
    print(
        "  Noisy: "
        f"roughness={noisy_best['roughness_rms_m']:.9g} m, "
        f"salinity={noisy_best['salinity_ppt']:.6g} ppt, "
        f"cost={noisy_best['cost_noisy']:.6g}"
    )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {results_dir / '05b_cost_surface_clean.png'}")
    print(f"  {results_dir / '05b_cost_surface_noisy.png'}")


if __name__ == "__main__":
    main()
