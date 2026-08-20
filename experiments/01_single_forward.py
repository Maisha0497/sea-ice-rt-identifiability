from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config
from src.forward_smrt import run_forward, smrt_metadata


def main():
    cfg = load_config()
    state = baseline_state(cfg)
    prediction = run_forward(state, cfg)

    payload = {
        "smrt": smrt_metadata(),
        "state": state.as_dict(),
        "incidence_angle_deg": cfg["sensor"]["incidence_angle_deg"],
        "prediction_db": prediction,
    }

    output = ensure_results_dir() / "01_single_forward.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("State:")
    for key, value in state.as_dict().items():
        print(f"  {key}: {value}")

    print("\nPredicted sigma0 (dB):")
    for channel, value in prediction.items():
        print(f"  {channel}: {value:.6f}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
