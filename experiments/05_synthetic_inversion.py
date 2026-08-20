from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config, noise_vector, vector_from_prediction
from src.forward_smrt import run_forward
from src.inversion import multistart_inversion


def main():
    cfg = load_config()
    base = baseline_state(cfg)
    rng = np.random.default_rng(int(cfg["inversion"]["random_seed"]))
    channels = ("L_HH", "L_VV", "S_HH", "S_VV")

    # Truth lies safely inside the tested joint-validity domain.
    truth = base.replace(
        roughness_rms_m=0.00075,
        salinity_ppt=4.0,
    )

    clean = vector_from_prediction(run_forward(truth, cfg), channels)
    sigma = noise_vector(cfg, channels)
    noisy = clean + rng.normal(0.0, sigma)

    fits = multistart_inversion(
        observed_db=noisy,
        cfg=cfg,
        template_state=base,
        channels=channels,
    )

    rows = []
    for rank, fit in enumerate(fits, start=1):
        row = {
            "rank": rank,
            "cost": fit.cost,
            "success": fit.success,
            "nfev": fit.nfev,
            **fit.state.as_dict(),
        }
        rows.append(row)

    results_dir = ensure_results_dir()
    pd.DataFrame(rows).to_csv(
        results_dir / "05_synthetic_multistart.csv",
        index=False,
    )

    best = fits[0]
    payload = {
        "channels": list(channels),
        "noise_std_db": dict(zip(channels, sigma.tolist())),
        "truth": truth.as_dict(),
        "clean_observation_db": dict(zip(channels, clean.tolist())),
        "noisy_observation_db": dict(zip(channels, noisy.tolist())),
        "best_retrieval": best.state.as_dict(),
        "best_cost": best.cost,
        "best_success": best.success,
        "best_message": best.message,
        "best_whitened_residual": dict(
            zip(channels, best.residual_whitened.tolist())
        ),
    }
    (results_dir / "05_synthetic_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("Truth:")
    print(truth.as_dict())
    print("\nBest retrieval:")
    print(best.state.as_dict())
    print(f"\nBest cost: {best.cost:.6f}")
    print(f"Successful: {best.success}")
    print("\nAll multistart solutions:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
