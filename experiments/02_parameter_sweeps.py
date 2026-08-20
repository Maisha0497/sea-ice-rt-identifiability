from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import CHANNELS, baseline_state, ensure_results_dir, load_config
from src.forward_smrt import run_forward


def sweep_values(spec: dict) -> np.ndarray:
    return np.linspace(
        float(spec["start"]),
        float(spec["stop"]),
        int(spec["count"]),
    )


def run_sweep(parameter: str, cfg: dict) -> pd.DataFrame:
    """
    Run one-at-a-time parameter sweeps.

    Invalid SMRT/IEM states are retained in the CSV with:
      valid = False
      channel values = NaN
      error = exception message

    This prevents one invalid state from terminating the entire sweep.
    """
    base = baseline_state(cfg)
    rows = []

    for index, value in enumerate(sweep_values(cfg["sweeps"][parameter]), start=1):
        value = float(value)
        state = base.replace(**{parameter: value})

        row = {
            parameter: value,
            "valid": False,
            "error": "",
        }

        try:
            prediction = run_forward(state, cfg)

            if not all(np.isfinite(prediction[channel]) for channel in CHANNELS):
                raise RuntimeError("Forward model returned a non-finite channel value.")

            row.update(prediction)
            row["valid"] = True
            print(
                f"[{index:02d}] {parameter}={value:.8g}: valid",
                flush=True,
            )

        except Exception as exc:
            row.update({channel: np.nan for channel in CHANNELS})
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index:02d}] {parameter}={value:.8g}: INVALID — {row['error']}",
                flush=True,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, parameter: str, output: Path) -> None:
    valid_df = df.loc[df["valid"]].copy()

    plt.figure(figsize=(8, 5))

    if valid_df.empty:
        plt.text(
            0.5,
            0.5,
            "No valid SMRT states in this sweep",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        for channel in CHANNELS:
            plt.plot(
                valid_df[parameter],
                valid_df[channel],
                marker="o",
                markersize=3,
                label=channel,
            )

    invalid_df = df.loc[~df["valid"]]
    if not invalid_df.empty:
        ymin, ymax = plt.ylim()
        y_marker = ymin + 0.03 * (ymax - ymin if ymax > ymin else 1.0)
        plt.scatter(
            invalid_df[parameter],
            np.full(len(invalid_df), y_marker),
            marker="x",
            label="Invalid IEM/SMRT state",
        )

    plt.xlabel(parameter)
    plt.ylabel(r"$\sigma^0$ (dB)")
    plt.title(f"SMRT sensitivity to {parameter}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main():
    cfg = load_config()
    results_dir = ensure_results_dir()

    for parameter in (
    "roughness_rms_m",
    "salinity_ppt",
    "ice_thickness_m",
):
        df = run_sweep(parameter, cfg)

        csv_path = results_dir / f"02_sweep_{parameter}.csv"
        png_path = results_dir / f"02_sweep_{parameter}.png"

        df.to_csv(csv_path, index=False)
        plot_sweep(df, parameter, png_path)

        valid_count = int(df["valid"].sum())
        invalid_count = int((~df["valid"]).sum())

        print(f"\nSaved: {csv_path}")
        print(f"Saved: {png_path}")
        print(
            f"{parameter}: {valid_count} valid, "
            f"{invalid_count} invalid out of {len(df)}"
        )

        if invalid_count:
            print("Invalid values:")
            print(
                df.loc[~df["valid"], [parameter, "error"]]
                .to_string(index=False)
            )
        print()


if __name__ == "__main__":
    main()
