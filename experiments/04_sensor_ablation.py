from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config
from src.forward_smrt import run_forward
from src.sensitivity import log_parameter_jacobian, svd_diagnostics, whiten_jacobian


CHANNEL_SETS = {
    "L_only": ("L_HH", "L_HV", "L_VV"),
    "S_only": ("S_HH", "S_HV", "S_VV"),
    "LS_copol": ("L_HH", "L_VV", "S_HH", "S_VV"),
    "LS_full": ("L_HH", "L_HV", "L_VV", "S_HH", "S_HV", "S_VV"),
}


def main():
    cfg = load_config()
    state = baseline_state(cfg)
    parameters = tuple(cfg["inversion"]["parameters"])
    rows = []

    for name, channels in CHANNEL_SETS.items():
        jacobian = log_parameter_jacobian(
            state=state,
            cfg=cfg,
            forward=run_forward,
            parameters=parameters,
            channels=channels,
        )
        jacobian_w = whiten_jacobian(jacobian, cfg, channels)
        _, singular_values, _, condition_number = svd_diagnostics(jacobian_w)

        row = {
            "experiment": name,
            "n_channels": len(channels),
            "condition_number": condition_number,
        }
        for i, value in enumerate(singular_values, start=1):
            row[f"s{i}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    output = ensure_results_dir() / "04_sensor_ablation.csv"
    df.to_csv(output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
