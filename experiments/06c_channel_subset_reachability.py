from __future__ import annotations

"""
06c_channel_subset_reachability.py

Compare forward-model reachability for:
    1. L-only: L_HH, L_VV
    2. S-only: S_HH, S_VV
    3. Joint L+S: L_HH, L_VV, S_HH, S_VV

This script reuses the expanded forward grid already produced by:
    experiments/06b_expanded_bounds_existence_test.py

It does not rerun SMRT.

Run from the project root:
    conda activate geo
    python experiments/06c_channel_subset_reachability.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_CSV = ROOT / "results" / "06b_expanded_reachable_grid.csv"
OBSERVATION_JSON = (
    ROOT / "data" / "derived" / "real_observations_for_inversion"
    / "real_observation_summary.json"
)
OUTPUT_DIR = ROOT / "results"

SUBSETS = {
    "L_only": ("L_HH", "L_VV"),
    "S_only": ("S_HH", "S_VV"),
    "L_plus_S": ("L_HH", "L_VV", "S_HH", "S_VV"),
}


def rms_db(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def diagnostic_label(value: float) -> str:
    if value <= 2.0:
        return "near/reachable"
    if value <= 5.0:
        return "marginal"
    return "clearly outside"


def main() -> None:
    if not GRID_CSV.exists():
        raise FileNotFoundError(
            f"Missing expanded grid:\n{GRID_CSV}\n"
            "Run experiments/06b_expanded_bounds_existence_test.py first."
        )
    if not OBSERVATION_JSON.exists():
        raise FileNotFoundError(f"Missing observation summary:\n{OBSERVATION_JSON}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(GRID_CSV)
    grid = grid[grid["valid"] == True].copy()  # noqa: E712
    if grid.empty:
        raise RuntimeError("The expanded grid contains no valid forward states.")

    observations = json.loads(OBSERVATION_JSON.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, object]] = []
    detailed: dict[str, dict] = {}

    for record in observations:
        ice_type = str(record["ice_type"])
        detailed[ice_type] = {}

        for subset_name, channels in SUBSETS.items():
            pred_columns = [f"pred_{channel}_db" for channel in channels]
            prediction_matrix = grid[pred_columns].to_numpy(dtype=float)
            observation = np.array(
                [record[f"{channel}_mean_db"] for channel in channels],
                dtype=float,
            )

            residual_matrix = prediction_matrix - observation[None, :]
            rms_values = np.sqrt(np.mean(np.square(residual_matrix), axis=1))

            best_position = int(np.argmin(rms_values))
            best_row = grid.iloc[best_position]
            best_prediction = best_row[pred_columns].to_numpy(dtype=float)
            best_residual = best_prediction - observation
            best_rms = rms_db(best_residual)

            result_row: dict[str, object] = {
                "ice_type": ice_type,
                "subset": subset_name,
                "channels": ",".join(channels),
                "best_rms_mismatch_db": best_rms,
                "status": diagnostic_label(best_rms),
                "best_roughness_rms_m": float(best_row["roughness_rms_m"]),
                "best_salinity_ppt": float(best_row["salinity_ppt"]),
                "best_ice_thickness_m": float(best_row["ice_thickness_m"]),
            }

            for index, channel in enumerate(channels):
                result_row[f"obs_{channel}_db"] = float(observation[index])
                result_row[f"fit_{channel}_db"] = float(best_prediction[index])
                result_row[f"residual_{channel}_db"] = float(best_residual[index])

            summary_rows.append(result_row)

            detailed[ice_type][subset_name] = {
                "channels": list(channels),
                "best_rms_mismatch_db": best_rms,
                "status": diagnostic_label(best_rms),
                "best_state": {
                    "roughness_rms_m": float(best_row["roughness_rms_m"]),
                    "salinity_ppt": float(best_row["salinity_ppt"]),
                    "ice_thickness_m": float(best_row["ice_thickness_m"]),
                },
                "observation_db": {
                    channel: float(observation[index])
                    for index, channel in enumerate(channels)
                },
                "best_prediction_db": {
                    channel: float(best_prediction[index])
                    for index, channel in enumerate(channels)
                },
                "residual_db_model_minus_observation": {
                    channel: float(best_residual[index])
                    for index, channel in enumerate(channels)
                },
            }

            print(
                f"{ice_type:>3} | {subset_name:<8} | "
                f"RMS={best_rms:6.3f} dB | {diagnostic_label(best_rms)}"
            )

    summary = pd.DataFrame(summary_rows)

    csv_path = OUTPUT_DIR / "06c_channel_subset_reachability.csv"
    json_path = OUTPUT_DIR / "06c_channel_subset_reachability.json"
    figure_path = OUTPUT_DIR / "06c_channel_subset_reachability.png"

    summary.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "source_grid": str(GRID_CSV),
                "observation_file": str(OBSERVATION_JSON),
                "important_note": (
                    "This is a nearest-state comparison on the existing expanded "
                    "coarse grid. It is intended to decide whether L-only, S-only, "
                    "or joint L+S is the most defensible next diagnostic."
                ),
                "results": detailed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ice_types = list(dict.fromkeys(summary["ice_type"]))
    subset_names = list(SUBSETS)
    positions = np.arange(len(ice_types))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for index, subset_name in enumerate(subset_names):
        values = []
        for ice_type in ice_types:
            value = summary.loc[
                (summary["ice_type"] == ice_type)
                & (summary["subset"] == subset_name),
                "best_rms_mismatch_db",
            ].iloc[0]
            values.append(float(value))

        offset = (index - 1) * width
        ax.bar(
            positions + offset,
            values,
            width=width,
            label=subset_name.replace("_", " "),
        )

    ax.axhline(2.0, linestyle="--", linewidth=1, label="2 dB diagnostic line")
    ax.axhline(5.0, linestyle=":", linewidth=1, label="5 dB diagnostic line")
    ax.set_xticks(positions)
    ax.set_xticklabels(ice_types)
    ax.set_ylabel("Minimum RMS model-observation mismatch (dB)")
    ax.set_title("Reachability comparison by channel subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {figure_path}")


if __name__ == "__main__":
    main()
