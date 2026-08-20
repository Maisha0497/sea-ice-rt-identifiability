#!/usr/bin/env python3
"""
01b_iem_validity_grid.py

Purpose
-------
Test whether combinations of RMS roughness and surface correlation length
produce SMRT IEM-Fung-1992 validity warnings at L-band and S-band.

This is a model-domain diagnostic, not an inversion and not a fit to SAR data.

Outputs
-------
results/01b_iem_validity_grid.csv
results/01b_iem_validity_grid.png

Run from the project root:
    python experiments/01b_iem_validity_grid.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config
from src.forward_smrt import _build_ice, _extract_channel, _model

try:
    from smrt.error import SMRTWarning
except ImportError:
    # Fallback only if a future SMRT version moves the warning class.
    SMRTWarning = Warning


# Diagnostic grid only. These are candidate values, not accepted physical priors.
ROUGHNESS_VALUES_M = np.array(
    [0.00025, 0.00050, 0.00100, 0.00200, 0.00300],
    dtype=float,
)

CORRELATION_LENGTH_VALUES_M = np.array(
    [0.005, 0.010, 0.020, 0.030, 0.050],
    dtype=float,
)


def run_one_band(state, cfg, band: str):
    """Run one band and return validity, warning text, and co-pol predictions."""
    from smrt import sensor_list

    frequency = float(cfg["sensor"]["frequencies_hz"][band])
    angle = float(cfg["sensor"]["incidence_angle_deg"])
    model_cfg = cfg["model"]

    # The IEM warning can be emitted while the interface/medium is constructed,
    # not only while model.run() is executed. Therefore the warning-capture
    # context must enclose the entire SMRT construction and simulation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        medium = _build_ice(state, cfg)
        model = _model(model_cfg["emmodel"], model_cfg["rtsolver"])

        sensor = sensor_list.active(
            frequency=frequency,
            theta_inc=angle,
            polarization_inc=["H", "V"],
            polarization=["H", "V"],
            name=f"{band}_band",
        )

        result = model.run(sensor, medium)

        hh = _extract_channel(result, "H", "H", f"{band}_HH")
        vv = _extract_channel(result, "V", "V", f"{band}_VV")

    smrt_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, SMRTWarning)
        or "roughness" in str(item.message).lower()
        or "correlation_length" in str(item.message).lower()
        or "ks*kl" in str(item.message).lower()
    ]

    warning_text = " | ".join(smrt_messages)
    finite = np.isfinite(hh) and np.isfinite(vv)
    valid = bool((len(smrt_messages) == 0) and finite)

    return valid, warning_text, float(hh), float(vv)


def main():
    cfg_base = load_config()
    base_state = baseline_state(cfg_base)
    rows = []

    total = len(ROUGHNESS_VALUES_M) * len(CORRELATION_LENGTH_VALUES_M)
    counter = 0

    for roughness_m in ROUGHNESS_VALUES_M:
        for corr_length_m in CORRELATION_LENGTH_VALUES_M:
            counter += 1
            print(
                f"[{counter:02d}/{total}] "
                f"roughness={roughness_m * 1000:.2f} mm, "
                f"corr_length={corr_length_m * 100:.2f} cm"
            )

            cfg = deepcopy(cfg_base)
            cfg["fixed_scene"]["surface_corr_length_m"] = float(corr_length_m)
            state = base_state.replace(roughness_rms_m=float(roughness_m))

            row = {
                "roughness_rms_m": roughness_m,
                "roughness_rms_mm": roughness_m * 1000.0,
                "surface_corr_length_m": corr_length_m,
                "surface_corr_length_cm": corr_length_m * 100.0,
            }

            for band in ("L", "S"):
                try:
                    valid, warning_text, hh, vv = run_one_band(
                        state=state,
                        cfg=cfg,
                        band=band,
                    )
                    row[f"{band}_valid"] = valid
                    row[f"{band}_warning"] = warning_text
                    row[f"{band}_HH_dB"] = hh
                    row[f"{band}_VV_dB"] = vv
                except Exception as exc:
                    row[f"{band}_valid"] = False
                    row[f"{band}_warning"] = f"ERROR: {type(exc).__name__}: {exc}"
                    row[f"{band}_HH_dB"] = np.nan
                    row[f"{band}_VV_dB"] = np.nan

            row["both_valid"] = bool(row["L_valid"] and row["S_valid"])
            rows.append(row)

    df = pd.DataFrame(rows)
    results_dir = ensure_results_dir()

    csv_path = results_dir / "01b_iem_validity_grid.csv"
    df.to_csv(csv_path, index=False)

    pivot = (
        df.pivot(
            index="roughness_rms_mm",
            columns="surface_corr_length_cm",
            values="both_valid",
        )
        .sort_index(ascending=True)
        .astype(int)
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.xticks(
        np.arange(len(pivot.columns)),
        [f"{value:g}" for value in pivot.columns],
    )
    plt.yticks(
        np.arange(len(pivot.index)),
        [f"{value:g}" for value in pivot.index],
    )
    plt.xlabel("Surface correlation length (cm)")
    plt.ylabel("RMS roughness (mm)")
    plt.title("IEM-Fung validity at both L and S bands\n1 = valid, 0 = invalid/warning")
    plt.colorbar(label="Both bands valid")
    plt.tight_layout()

    png_path = results_dir / "01b_iem_validity_grid.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    valid_df = df[df["both_valid"]].copy()

    print("\n=== IEM VALIDITY GRID COMPLETE ===")
    print(f"Tested combinations: {len(df)}")
    print(f"Valid at both L and S: {len(valid_df)}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {png_path}")

    if valid_df.empty:
        print(
            "\nNo tested combination was valid at both bands. "
            "Do not continue to sensitivity or inversion. Expand the grid "
            "toward smaller roughness/correlation length or reconsider the "
            "interface model."
        )
        return

    current_r = float(cfg_base["baseline_state"]["roughness_rms_m"])
    current_l = float(cfg_base["fixed_scene"]["surface_corr_length_m"])

    valid_df["distance_from_current_log"] = np.sqrt(
        np.log(valid_df["roughness_rms_m"] / current_r) ** 2
        + np.log(valid_df["surface_corr_length_m"] / current_l) ** 2
    )

    closest = valid_df.sort_values("distance_from_current_log").iloc[0]

    print("\nClosest valid candidate to the current configuration:")
    print(
        f"  roughness_rms_m: "
        f"{closest['roughness_rms_m']:.6f} "
        f"({closest['roughness_rms_mm']:.3f} mm)"
    )
    print(
        f"  surface_corr_length_m: "
        f"{closest['surface_corr_length_m']:.6f} "
        f"({closest['surface_corr_length_cm']:.3f} cm)"
    )
    print(
        "\nThis is only a numerically valid candidate. "
        "Do not automatically treat it as the true sea-ice roughness."
    )


if __name__ == "__main__":
    main()
