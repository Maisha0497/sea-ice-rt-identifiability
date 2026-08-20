from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import CHANNELS, baseline_state, ensure_results_dir, load_config
from src.forward_smrt import run_forward


def main():
    cfg = load_config()
    base = baseline_state(cfg)
    results_dir = ensure_results_dir()

    # Use the currently tested one-at-a-time ranges.
    roughness_values = np.linspace(0.00025, 0.00100, 16)
    salinity_values = np.linspace(0.5, 10.5625, 22)

    rows = []
    total = len(roughness_values) * len(salinity_values)
    counter = 0

    for roughness in roughness_values:
        for salinity in salinity_values:
            counter += 1
            state = base.replace(
                roughness_rms_m=float(roughness),
                salinity_ppt=float(salinity),
            )

            valid = False
            error = ""

            try:
                prediction = run_forward(state, cfg)
                valid = all(
                    np.isfinite(prediction[channel])
                    for channel in CHANNELS
                )
                if not valid:
                    error = "Non-finite forward-model output."
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "roughness_rms_m": float(roughness),
                    "roughness_rms_mm": float(roughness * 1000.0),
                    "salinity_ppt": float(salinity),
                    "ice_thickness_m": float(base.ice_thickness_m),
                    "valid": bool(valid),
                    "error": error,
                }
            )

            status = "valid" if valid else "INVALID"
            print(
                f"[{counter:03d}/{total}] "
                f"roughness={roughness * 1000:.3f} mm, "
                f"salinity={salinity:.4f} ppt: {status}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    csv_path = results_dir / "04b_joint_roughness_salinity_validity.csv"
    png_path = results_dir / "04b_joint_roughness_salinity_validity.png"
    df.to_csv(csv_path, index=False)

    grid = (
        df.pivot(
            index="salinity_ppt",
            columns="roughness_rms_mm",
            values="valid",
        )
        .astype(int)
        .sort_index(ascending=True)
    )

    plt.figure(figsize=(9, 6))
    image = plt.imshow(
        grid.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            grid.columns.min(),
            grid.columns.max(),
            grid.index.min(),
            grid.index.max(),
        ],
        vmin=0,
        vmax=1,
    )
    plt.colorbar(image, ticks=[0, 1], label="Validity: 0 = invalid, 1 = valid")
    plt.xlabel("RMS roughness (mm)")
    plt.ylabel("Ice salinity (ppt)")
    plt.title(
        "Joint IEM/SMRT validity at fixed "
        f"thickness = {base.ice_thickness_m:.2f} m"
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    valid_count = int(df["valid"].sum())
    invalid_count = int((~df["valid"]).sum())

    print("\n=== JOINT VALIDITY GRID COMPLETE ===")
    print(f"Tested combinations: {len(df)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {png_path}")

    if invalid_count:
        print("\nInvalid combinations:")
        print(
            df.loc[
                ~df["valid"],
                ["roughness_rms_mm", "salinity_ppt"],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
