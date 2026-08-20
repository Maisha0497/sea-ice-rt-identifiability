#!/usr/bin/env python3
from __future__ import annotations

"""
13_myi_one_layer_snow_test.py

Purpose
-------
First snow-physics test after the bare-ice reachability stage.

This is NOT another bare-ice reachability optimization.

For each MYI incidence bin:
1. read the best semi-constrained MYI joint state from experiment 12;
2. keep ALL ice/interface parameters fixed at that state;
3. place ONE dry/fresh snow layer above the multiyear ice;
4. sweep only snow depth, density, and exponential correlation length;
5. compare snow-covered predictions against the exact bare-ice experiment-12
   prediction.

The question is intentionally narrow:

    Does a physically reasonable one-layer MYI snow cover reduce the remaining
    L-band mismatch without destroying the already-good S-band agreement?

Snow setup
----------
This first test uses fresh/dry snow:
    salinity = 0 PSU
    temperature = 260 K

Parameter grid:
    depth       = 0.10, 0.20, 0.30, 0.40 m
    density     = 250, 300, 350 kg m^-3
    corr_length = 0.10, 0.20, 0.30 mm

These are deliberately simple literature-scale values, not UAVSAR scene
measurements.

Important geometry
------------------
The experiment-12 IEM roughness/correlation length is retained, but once snow
is added it becomes the SNOW-ICE interface rather than the air-ice surface.

SMRT stack:
    flat snow-air surface
    one exponential snow layer
    rough IEM snow-ice interface
    multiyear ice layer
    water substrate

SMRT Snowpack addition preserves the upper snow surface, the ice-column first
interface, and the ice-column substrate.

Inputs
------
results/12_myi_semi_constrained_reachability.csv

Outputs
-------
results/13_myi_one_layer_snow_grid.csv
results/13_myi_one_layer_snow_best.csv
results/13_myi_one_layer_snow_best.json
results/13_myi_snow_vs_bare.csv

Run
---
cd <repository-root>
conda activate geo
export PYTHONPATH="$PWD:$PYTHONPATH"
python experiments/13_myi_one_layer_snow_test.py
"""

from copy import deepcopy
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import baseline_state, ensure_results_dir, load_config
from src.forward_smrt import _build_surface, _extract_channel, _model


EXPERIMENT12_CSV = (
    ROOT / "results" / "12_myi_semi_constrained_reachability.csv"
)

CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")

SNOW_DEPTHS_M = np.array([0.10, 0.20, 0.30, 0.40], dtype=float)
SNOW_DENSITIES_KGM3 = np.array([250.0, 300.0, 350.0], dtype=float)
SNOW_CORR_LENGTHS_M = np.array([0.00010, 0.00020, 0.00030], dtype=float)

SNOW_TEMPERATURE_K = 260.0
SNOW_SALINITY_PPT = 0.0
SNOW_MICROSTRUCTURE = "exponential"


def rms(values):
    a = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(a**2)))


def channel_rms(pred, obs):
    residual = pred - obs
    return {
        "L_only_rms_db": rms(residual[:2]),
        "S_only_rms_db": rms(residual[2:]),
        "L_plus_S_rms_db": rms(residual),
    }


def run_snow_covered(
    base_state,
    cfg_base,
    angle_deg,
    ice_params,
    snow_depth_m,
    snow_density_kgm3,
    snow_corr_length_m,
):
    """
    Build:
        snow-air Flat
        snow layer
        experiment-12 IEM interface
        MYI layer
        water substrate
    """
    from smrt import PSU, make_ice_column, make_snowpack, sensor_list

    cfg = deepcopy(cfg_base)
    cfg["sensor"]["incidence_angle_deg"] = float(angle_deg)
    cfg["model"]["ice_type"] = "multiyear"

    cfg["fixed_scene"]["surface_corr_length_m"] = float(
        ice_params["surface_corr_length_m"]
    )
    cfg["fixed_scene"]["ice_corr_length_m"] = float(
        ice_params["ice_corr_length_m"]
    )

    state = base_state.replace(
        roughness_rms_m=float(ice_params["roughness_rms_m"]),
        salinity_ppt=float(ice_params["salinity_ppt"]),
        ice_thickness_m=float(ice_params["ice_thickness_m"]),
    )

    # In the bare model this object was the air-ice surface.
    # After snow + ice stacking, it becomes the interface at the top
    # of the ice layer, i.e. the snow-ice interface.
    snow_ice_interface = _build_surface(state, cfg)

    mc = cfg["model"]
    fixed = cfg["fixed_scene"]

    ice = make_ice_column(
        ice_type="multiyear",
        thickness=[float(ice_params["ice_thickness_m"])],
        temperature=[float(fixed["temperature_k"])],
        microstructure_model=mc["microstructure_model"],
        salinity=[float(ice_params["salinity_ppt"]) * PSU],
        porosity=[float(ice_params["porosity"])],
        corr_length=[float(ice_params["ice_corr_length_m"])],
        brine_inclusion_shape=mc["brine_inclusion_shape"],
        add_water_substrate=True,
        water_temperature=float(fixed["water_temperature_k"]),
        water_salinity=float(fixed["water_salinity_ppt"]) * PSU,
        surface=snow_ice_interface,
    )

    snow = make_snowpack(
        thickness=[float(snow_depth_m)],
        microstructure_model=SNOW_MICROSTRUCTURE,
        density=[float(snow_density_kgm3)],
        temperature=[SNOW_TEMPERATURE_K],
        salinity=[SNOW_SALINITY_PPT * PSU],
        corr_length=[float(snow_corr_length_m)],
    )

    # SMRT stacks the first snowpack on top of the second.
    # snow has no substrate; ice retains the water substrate.
    medium = snow + ice

    model = _model(mc["emmodel"], mc["rtsolver"])

    output = {}

    for band in ("L", "S"):
        sensor = sensor_list.active(
            frequency=float(cfg["sensor"]["frequencies_hz"][band]),
            theta_inc=float(angle_deg),
            polarization_inc=["H", "V"],
            polarization=["H", "V"],
            name=f"{band}_band",
        )

        result = model.run(sensor, medium)

        output[f"{band}_HH"] = _extract_channel(
            result, "H", "H", f"{band}_HH"
        )
        output[f"{band}_VV"] = _extract_channel(
            result, "V", "V", f"{band}_VV"
        )

    pred = np.array([output[ch] for ch in CHANNELS], dtype=float)

    if pred.shape != (4,) or np.any(~np.isfinite(pred)):
        raise ValueError(f"Invalid/non-finite snow prediction: {pred}")

    return pred


def main():
    if not EXPERIMENT12_CSV.exists():
        raise FileNotFoundError(
            f"Missing experiment-12 result:\n{EXPERIMENT12_CSV}"
        )

    exp12 = pd.read_csv(EXPERIMENT12_CSV)
    joint = exp12[
        exp12["optimized_subset"].astype(str) == "L_plus_S"
    ].copy()

    if joint.empty:
        raise RuntimeError(
            "No L_plus_S rows found in experiment 12."
        )

    cfg = load_config()
    base_state = baseline_state(cfg)
    results_dir = ensure_results_dir()

    print("=" * 78)
    print("MYI ONE-LAYER SNOW TEST")
    print("=" * 78)
    print("This is NOT another bare-ice reachability test.")
    print("Ice state: fixed to experiment-12 joint solution in each incidence bin.")
    print("Snow:")
    print("  depth       = 0.10, 0.20, 0.30, 0.40 m")
    print("  density     = 250, 300, 350 kg/m3")
    print("  corr length = 0.10, 0.20, 0.30 mm")
    print("  temperature = 260 K")
    print("  salinity    = 0 ppt")
    print("")
    print("Goal: improve remaining L-band mismatch without ruining S-band.")

    grid_rows = []
    best_rows = []

    total_snow_states = (
        len(SNOW_DEPTHS_M)
        * len(SNOW_DENSITIES_KGM3)
        * len(SNOW_CORR_LENGTHS_M)
    )

    for run_i, (_, rec) in enumerate(joint.iterrows(), start=1):
        inc_bin = str(rec["incidence_bin"])
        angle = float(rec["incidence_angle_deg"])

        obs = np.array(
            [float(rec[f"obs_{ch}_db"]) for ch in CHANNELS],
            dtype=float,
        )
        bare = np.array(
            [float(rec[f"fit_{ch}_db"]) for ch in CHANNELS],
            dtype=float,
        )

        bare_rms = channel_rms(bare, obs)

        ice_params = {
            "roughness_rms_m": float(rec["best_roughness_rms_m"]),
            "surface_corr_length_m": float(rec["best_surface_corr_length_m"]),
            "salinity_ppt": float(rec["best_salinity_ppt"]),
            "ice_thickness_m": float(rec["best_ice_thickness_m"]),
            "porosity": float(rec["best_porosity"]),
            "ice_corr_length_m": float(rec["best_ice_corr_length_m"]),
        }

        print("\n" + "-" * 78)
        print(
            f"[{run_i}/{len(joint)}] MYI | {inc_bin} deg | theta={angle:.3f}"
        )
        print("-" * 78)
        print(
            f"  bare: L={bare_rms['L_only_rms_db']:.3f} dB | "
            f"S={bare_rms['S_only_rms_db']:.3f} dB | "
            f"joint={bare_rms['L_plus_S_rms_db']:.3f} dB"
        )

        local_rows = []
        n = 0

        for snow_depth in SNOW_DEPTHS_M:
            for density in SNOW_DENSITIES_KGM3:
                for snow_corr in SNOW_CORR_LENGTHS_M:
                    n += 1

                    row = {
                        "ice_type": "MYI",
                        "incidence_bin": inc_bin,
                        "incidence_angle_deg": angle,
                        "snow_depth_m": float(snow_depth),
                        "snow_density_kgm3": float(density),
                        "snow_corr_length_m": float(snow_corr),
                        "snow_temperature_k": SNOW_TEMPERATURE_K,
                        "snow_salinity_ppt": SNOW_SALINITY_PPT,
                        "valid": False,
                        "error": "",
                    }

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pred = run_snow_covered(
                                base_state=base_state,
                                cfg_base=cfg,
                                angle_deg=angle,
                                ice_params=ice_params,
                                snow_depth_m=snow_depth,
                                snow_density_kgm3=density,
                                snow_corr_length_m=snow_corr,
                            )

                        rr = channel_rms(pred, obs)

                        row["valid"] = True
                        row.update(rr)

                        row["delta_L_vs_bare_db"] = (
                            rr["L_only_rms_db"]
                            - bare_rms["L_only_rms_db"]
                        )
                        row["delta_S_vs_bare_db"] = (
                            rr["S_only_rms_db"]
                            - bare_rms["S_only_rms_db"]
                        )
                        row["delta_joint_vs_bare_db"] = (
                            rr["L_plus_S_rms_db"]
                            - bare_rms["L_plus_S_rms_db"]
                        )

                        # Positive means snow made the model prediction brighter.
                        for i, ch in enumerate(CHANNELS):
                            row[f"obs_{ch}_db"] = float(obs[i])
                            row[f"bare_{ch}_db"] = float(bare[i])
                            row[f"snow_{ch}_db"] = float(pred[i])
                            row[f"snow_minus_bare_{ch}_db"] = float(
                                pred[i] - bare[i]
                            )
                            row[f"residual_{ch}_db"] = float(
                                pred[i] - obs[i]
                            )

                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"

                    grid_rows.append(row)
                    local_rows.append(row)

                    if n % 12 == 0 or n == total_snow_states:
                        print(
                            f"    snow states {n}/{total_snow_states}",
                            flush=True,
                        )

        local = pd.DataFrame(local_rows)
        valid = local[local["valid"] == True].copy()  # noqa: E712

        if valid.empty:
            print("  ERROR: no valid snow-covered states.")
            continue

        # Primary selection: minimum joint RMS.
        best = valid.sort_values(
            ["L_plus_S_rms_db", "L_only_rms_db", "S_only_rms_db"]
        ).iloc[0].to_dict()

        best["bare_L_only_rms_db"] = bare_rms["L_only_rms_db"]
        best["bare_S_only_rms_db"] = bare_rms["S_only_rms_db"]
        best["bare_L_plus_S_rms_db"] = bare_rms["L_plus_S_rms_db"]

        # Useful diagnostic: does snow improve L by >=0.5 dB while not
        # worsening S by more than 1 dB?
        helpful = valid[
            (valid["delta_L_vs_bare_db"] <= -0.5)
            & (valid["delta_S_vs_bare_db"] <= 1.0)
        ].copy()

        best["n_helpful_snow_states"] = int(len(helpful))
        best["snow_helpful_for_L_without_large_S_penalty"] = bool(
            len(helpful) > 0
        )

        best_rows.append(best)

        print(
            f"  best snow: depth={best['snow_depth_m']:.2f} m | "
            f"density={best['snow_density_kgm3']:.0f} kg/m3 | "
            f"lc={best['snow_corr_length_m']*1000:.2f} mm"
        )
        print(
            f"  snow: L={best['L_only_rms_db']:.3f} dB | "
            f"S={best['S_only_rms_db']:.3f} dB | "
            f"joint={best['L_plus_S_rms_db']:.3f} dB"
        )
        print(
            f"  change vs bare: "
            f"L={best['delta_L_vs_bare_db']:+.3f} dB | "
            f"S={best['delta_S_vs_bare_db']:+.3f} dB | "
            f"joint={best['delta_joint_vs_bare_db']:+.3f} dB"
        )
        print(
            f"  helpful states (L improves >=0.5 dB, "
            f"S worsens <=1 dB): {len(helpful)}"
        )

    grid_df = pd.DataFrame(grid_rows)
    best_df = pd.DataFrame(best_rows)

    grid_csv = results_dir / "13_myi_one_layer_snow_grid.csv"
    best_csv = results_dir / "13_myi_one_layer_snow_best.csv"
    best_json = results_dir / "13_myi_one_layer_snow_best.json"
    compare_csv = results_dir / "13_myi_snow_vs_bare.csv"

    grid_df.to_csv(grid_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    best_json.write_text(
        json.dumps(
            {
                "purpose": (
                    "Test one-layer fresh/dry MYI snow over the fixed "
                    "experiment-12 semi-constrained ice states."
                ),
                "snow_grid": {
                    "depth_m": SNOW_DEPTHS_M.tolist(),
                    "density_kgm3": SNOW_DENSITIES_KGM3.tolist(),
                    "corr_length_m": SNOW_CORR_LENGTHS_M.tolist(),
                    "temperature_k": SNOW_TEMPERATURE_K,
                    "salinity_ppt": SNOW_SALINITY_PPT,
                },
                "best_by_incidence_bin": best_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison_cols = [
        "incidence_bin",
        "incidence_angle_deg",
        "bare_L_only_rms_db",
        "L_only_rms_db",
        "delta_L_vs_bare_db",
        "bare_S_only_rms_db",
        "S_only_rms_db",
        "delta_S_vs_bare_db",
        "bare_L_plus_S_rms_db",
        "L_plus_S_rms_db",
        "delta_joint_vs_bare_db",
        "snow_depth_m",
        "snow_density_kgm3",
        "snow_corr_length_m",
        "n_helpful_snow_states",
        "snow_helpful_for_L_without_large_S_penalty",
    ]

    if not best_df.empty:
        best_df[comparison_cols].to_csv(compare_csv, index=False)

    print("\n" + "=" * 78)
    print("FINAL ONE-LAYER SNOW RESULTS")
    print("=" * 78)

    if best_df.empty:
        print("No valid snow-covered states were produced.")
    else:
        print(
            best_df[comparison_cols]
            .sort_values("incidence_angle_deg")
            .to_string(index=False)
        )

    print("\nSaved:")
    print(f"  {grid_csv}")
    print(f"  {best_csv}")
    print(f"  {best_json}")
    if not best_df.empty:
        print(f"  {compare_csv}")

    print(
        "\nSTOP HERE. "
        "Do not optimize the ice and snow together yet. "
        "First decide whether this simple fresh/dry snow layer helps the "
        "remaining L-band mismatch while preserving S-band."
    )


if __name__ == "__main__":
    main()
