from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import CHANNELS, baseline_state, ensure_results_dir, load_config
from src.forward_smrt import run_forward
from src.sensitivity import log_parameter_jacobian, svd_diagnostics, whiten_jacobian


def main():
    cfg = load_config()
    state = baseline_state(cfg)
    parameters = tuple(cfg["inversion"]["parameters"])

    jacobian = log_parameter_jacobian(
        state=state,
        cfg=cfg,
        forward=run_forward,
        parameters=parameters,
        channels=CHANNELS,
    )
    jacobian_w = whiten_jacobian(jacobian, cfg, CHANNELS)
    u, singular_values, vt, condition_number = svd_diagnostics(jacobian_w)

    results_dir = ensure_results_dir()

    pd.DataFrame(
        jacobian,
        index=CHANNELS,
        columns=parameters,
    ).to_csv(results_dir / "03_jacobian_log_db.csv")

    pd.DataFrame(
        jacobian_w,
        index=CHANNELS,
        columns=parameters,
    ).to_csv(results_dir / "03_jacobian_whitened.csv")

    pd.DataFrame(
        vt.T,
        index=parameters,
        columns=[f"mode_{i+1}" for i in range(vt.shape[0])],
    ).to_csv(results_dir / "03_right_singular_vectors.csv")

    summary = {
        "parameters": list(parameters),
        "channels": list(CHANNELS),
        "singular_values": singular_values.tolist(),
        "condition_number": condition_number,
    }
    (results_dir / "03_svd_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Whitened singular values:")
    for i, value in enumerate(singular_values, start=1):
        print(f"  s{i}: {value:.6g}")

    print(f"Condition number: {condition_number:.6g}")
    print("\nRight singular vectors (parameter combinations):")
    print(pd.DataFrame(vt.T, index=parameters))


if __name__ == "__main__":
    main()
