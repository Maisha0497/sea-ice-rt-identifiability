#!/usr/bin/env python3
from __future__ import annotations

"""
07_prepare_no_incidence_normalization.py

Purpose
-------
Create a clean winter UAVSAR co-pol observation set for the SMRT incidence-angle
audit WITHOUT the empirical normalization to 35 degrees.

For each of:
    LHH, LVV, SHH, SVV

the script does exactly:

    Level-2 DN/amplitude
        -> noise-corrected sigma0 in dB
        -> NO empirical incidence normalization
        -> Lee-Sigma filter in LINEAR intensity
        -> dB
        -> garbage mask

Calibration equation copied from the existing UAVSAR processing notebook:

    sigma0_dB =
        10*log10(max(DN^2 - N, 1e-10))
        + 10*log10(sin(theta))
        - 42

Important
---------
The sin(theta) term is part of the sigma0 calibration. It is NOT the empirical
35-degree normalization that we are removing.

This script deliberately does NOT:
- apply sigma0_norm = sigma0 - theta_d*(theta - 35 deg)
- overwrite any of the existing processed TIFFs
- change the RTE-PINN config
- run SMRT or inversion

Outputs
-------
A new audit folder containing, for each channel:
    *_sigma0_no_inc_norm_unfiltered.tif
    *_sigma0_no_inc_norm_leeSigma.tif

and:
    07_no_incidence_processing_summary.csv
    07_no_incidence_processing_manifest.json

Run from the RTE-PINN project root:
    conda activate geo
    python experiments/07_prepare_no_incidence_normalization.py
"""

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import derived_data_dir, external_asar_root


# =====================================================================
# EXTERNAL INPUTS AND PROJECT-LOCAL DERIVED OUTPUTS
# =====================================================================

ASAR_ROOT = external_asar_root()

L_DIR = (
    ASAR_ROOT / "Originals" / "Winter" / "L_tb_extracted_all files"
    / "ASAR_L_JOINT_FP_ID01118_LINE01_RUN01_141219_V1.3.B"
)

S_DIR = (
    ASAR_ROOT / "Originals" / "Winter" / "S"
    / "ASAR_S_JOINT_FP_ID01118_LINE01_RUN01_141219_V1.3.B"
)

INCIDENCE_TIF = (
    S_DIR
    / "ASAR_L_S_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_INC_MAP_V1.3.B.tif"
)

INPUTS = {
    "LHH": (
        L_DIR
        / "ASAR_L_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_HH_V1.3.B.tif"
    ),
    "LVV": (
        L_DIR
        / "ASAR_L_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_VV_V1.3.B.tif"
    ),
    "SHH": (
        S_DIR
        / "ASAR_S_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_HH_V1.3.B.tif"
    ),
    "SVV": (
        S_DIR
        / "ASAR_S_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_VV_V1.3.B.tif"
    ),
}

OUTPUT_DIR = derived_data_dir(
    "SMRT_observation_audit", "no_incidence_normalization"
)

# Existing winter noise values from the uploaded processing notebook.
NOISE_VALUES = {
    "LHH": 2.791368,
    "LVV": 1.411524,
    "SHH": 24.16259,
    "SVV": 27.50169,
}

# Existing calibration constant.
K_DB = 42.0

# Same Lee-Sigma settings as the existing notebook.
WINDOW_SIZE = 5
LEE_SIGMA_V = {
    "L": 0.10,
    "S": 0.05,
}
LEE_THRESHOLD = 0.5

# Values below this are treated as garbage, as in the existing notebook.
GARBAGE_THRESHOLD_DB = -140.0

# Processing tile size. 2048 matches the notebook's chunked filtering approach.
CHUNK_SIZE = 2048

# Keep the unfiltered calibrated sigma0 file for provenance/audit.
KEEP_UNFILTERED = True

EPS = 1e-10


# =====================================================================
# PROCESSING FUNCTIONS
# =====================================================================

def require_inputs() -> None:
    missing = []

    if not INCIDENCE_TIF.exists():
        missing.append(f"INCIDENCE_TIF:\n  {INCIDENCE_TIF}")

    for channel, path in INPUTS.items():
        if not path.exists():
            missing.append(f"{channel}:\n  {path}")

    if missing:
        raise FileNotFoundError(
            "\n\nOne or more input paths do not exist.\n"
            "Edit ONLY the path section near the top of this file.\n\n"
            + "\n\n".join(missing)
        )


def profile_for_output(src: rasterio.io.DatasetReader) -> dict:
    profile = src.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=np.nan,
    )
    return profile


def open_incidence_on_source_grid(
    incidence_src: rasterio.io.DatasetReader,
    source_src: rasterio.io.DatasetReader,
) -> WarpedVRT:
    """
    Make the incidence map readable on exactly the current channel's grid.

    If the incidence raster already matches the source raster, WarpedVRT is
    effectively just a consistent grid view.
    """
    return WarpedVRT(
        incidence_src,
        crs=source_src.crs,
        transform=source_src.transform,
        width=source_src.width,
        height=source_src.height,
        resampling=Resampling.bilinear,
        nodata=incidence_src.nodata,
    )


def calibrate_sigma0_no_empirical_normalization(
    input_path: Path,
    incidence_path: Path,
    channel: str,
    output_path: Path,
) -> dict:
    """
    Calibrate one channel to noise-corrected sigma0 dB.

    IMPORTANT:
    The empirical 35-degree incidence normalization is NOT applied.
    """
    noise = float(NOISE_VALUES[channel])

    total_valid = 0
    total_sum = 0.0
    global_min = math.inf
    global_max = -math.inf

    with rasterio.open(input_path) as src, rasterio.open(incidence_path) as inc_src:
        if src.crs is None:
            raise ValueError(f"{channel}: source raster has no CRS: {input_path}")

        profile = profile_for_output(src)

        with open_incidence_on_source_grid(inc_src, src) as inc_vrt:
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.update_tags(
                    quantity="sigma0_dB",
                    empirical_incidence_normalization="NOT_APPLIED",
                    calibration=(
                        "10log10(max(DN^2-N,1e-10))"
                        "+10log10(sin(theta))-42"
                    ),
                    noise_value=str(noise),
                    K_dB=str(K_DB),
                    incidence_source=str(incidence_path),
                    source=str(input_path),
                )

                for _, window in src.block_windows(1):
                    dn_ma = src.read(1, window=window, masked=True).astype(np.float64)
                    inc_ma = inc_vrt.read(1, window=window, masked=True).astype(np.float64)

                    dn = np.asarray(dn_ma.data, dtype=np.float64)
                    theta = np.asarray(inc_ma.data, dtype=np.float64)

                    invalid = (
                        np.ma.getmaskarray(dn_ma)
                        | np.ma.getmaskarray(inc_ma)
                        | ~np.isfinite(dn)
                        | ~np.isfinite(theta)
                        | (theta <= 0.0)
                        | (theta >= 90.0)
                    )

                    dn2_minus_n = np.maximum(dn**2 - noise, EPS)
                    sin_theta = np.maximum(np.sin(np.deg2rad(theta)), EPS)

                    sigma0_db = (
                        10.0 * np.log10(dn2_minus_n)
                        + 10.0 * np.log10(sin_theta)
                        - K_DB
                    )

                    invalid |= ~np.isfinite(sigma0_db)
                    invalid |= sigma0_db < GARBAGE_THRESHOLD_DB
                    sigma0_db[invalid] = np.nan

                    valid = np.isfinite(sigma0_db)
                    if np.any(valid):
                        values = sigma0_db[valid]
                        total_valid += int(values.size)
                        total_sum += float(values.sum())
                        global_min = min(global_min, float(values.min()))
                        global_max = max(global_max, float(values.max()))

                    dst.write(sigma0_db.astype(np.float32), 1, window=window)

    return {
        "calibrated_valid_pixels": total_valid,
        "calibrated_mean_db_QA_only": (
            total_sum / total_valid if total_valid else np.nan
        ),
        "calibrated_min_db": global_min if total_valid else np.nan,
        "calibrated_max_db": global_max if total_valid else np.nan,
    }


def db_to_lin(data_db: np.ndarray) -> np.ndarray:
    return np.maximum(10.0 ** (data_db / 10.0), EPS)


def lin_to_db(data_lin: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(data_lin, EPS))


def lee_sigma_filter_fast_linear(
    data_lin: np.ndarray,
    window_size: int,
    sigma_v: float,
    threshold: float,
) -> np.ndarray:
    """
    Same Lee-Sigma logic as the uploaded processing notebook.
    """
    nan_mask = ~np.isfinite(data_lin)

    if np.all(nan_mask):
        return data_lin.copy()

    finite = ~nan_mask
    work = data_lin.copy()

    # NaN-safe local inpainting, matching the notebook logic.
    if np.any(nan_mask):
        local_sum = uniform_filter(
            np.where(finite, work, 0.0),
            size=window_size,
            mode="reflect",
        )
        local_count = uniform_filter(
            finite.astype(float),
            size=window_size,
            mode="reflect",
        )
        local_mean = np.divide(
            local_sum,
            local_count,
            out=np.zeros_like(local_sum),
            where=local_count > 0,
        )
        work = np.where(finite, work, local_mean)

    mu = uniform_filter(work, size=window_size, mode="reflect")
    mu2 = uniform_filter(work**2, size=window_size, mode="reflect")
    var = np.maximum(mu2 - mu**2, 1e-12)

    cv_local = np.sqrt(var) / np.maximum(mu, 1e-12)
    cv_noise = float(sigma_v)

    alpha = np.zeros_like(cv_local, dtype=work.dtype)

    homogeneous = cv_local <= threshold
    alpha[homogeneous] = np.maximum(
        0.0,
        1.0
        - (cv_noise**2)
        / np.maximum(cv_local[homogeneous] ** 2, 1e-12),
    )

    heterogeneous = ~homogeneous
    alpha[heterogeneous] = 0.5 * np.maximum(
        0.0,
        1.0
        - (cv_noise**2)
        / np.maximum(cv_local[heterogeneous] ** 2, 1e-12),
    )

    filtered = alpha * work + (1.0 - alpha) * mu
    filtered[nan_mask] = np.nan

    return filtered


def clamp_window(
    row_off: int,
    col_off: int,
    height: int,
    width: int,
    raster_height: int,
    raster_width: int,
) -> Window:
    row0 = max(0, row_off)
    col0 = max(0, col_off)
    row1 = min(raster_height, row_off + height)
    col1 = min(raster_width, col_off + width)

    return Window(
        col_off=col0,
        row_off=row0,
        width=col1 - col0,
        height=row1 - row0,
    )


def lee_sigma_filter_tiled(
    calibrated_path: Path,
    output_path: Path,
    band: str,
) -> dict:
    """
    Apply Lee-Sigma in linear intensity using overlapping tiles.

    The halo is WINDOW_SIZE//2, matching the neighborhood needed by the
    5x5 uniform filters.
    """
    sigma_v = float(LEE_SIGMA_V[band])
    overlap = WINDOW_SIZE // 2

    total_valid = 0
    total_sum = 0.0
    global_min = math.inf
    global_max = -math.inf

    with rasterio.open(calibrated_path) as src:
        profile = profile_for_output(src)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.update_tags(
                quantity="sigma0_dB",
                empirical_incidence_normalization="NOT_APPLIED",
                speckle_filter="Lee-Sigma in linear intensity",
                lee_window_size=str(WINDOW_SIZE),
                lee_sigma_v=str(sigma_v),
                lee_threshold=str(LEE_THRESHOLD),
                source=str(calibrated_path),
            )

            for row_off in range(0, src.height, CHUNK_SIZE):
                for col_off in range(0, src.width, CHUNK_SIZE):
                    core_height = min(CHUNK_SIZE, src.height - row_off)
                    core_width = min(CHUNK_SIZE, src.width - col_off)

                    expanded = clamp_window(
                        row_off=row_off - overlap,
                        col_off=col_off - overlap,
                        height=core_height + 2 * overlap,
                        width=core_width + 2 * overlap,
                        raster_height=src.height,
                        raster_width=src.width,
                    )

                    data_ma = src.read(
                        1,
                        window=expanded,
                        masked=True,
                    ).astype(np.float64)

                    data_db = np.asarray(data_ma.data, dtype=np.float64)
                    data_db[np.ma.getmaskarray(data_ma)] = np.nan
                    data_db[~np.isfinite(data_db)] = np.nan

                    data_lin = db_to_lin(data_db)

                    filtered_lin = lee_sigma_filter_fast_linear(
                        data_lin=data_lin,
                        window_size=WINDOW_SIZE,
                        sigma_v=sigma_v,
                        threshold=LEE_THRESHOLD,
                    )

                    filtered_db = lin_to_db(filtered_lin)
                    filtered_db[~np.isfinite(data_db)] = np.nan
                    filtered_db[filtered_db < GARBAGE_THRESHOLD_DB] = np.nan

                    # Crop the expanded tile back to the core.
                    core_row0 = row_off - int(expanded.row_off)
                    core_col0 = col_off - int(expanded.col_off)

                    core = filtered_db[
                        core_row0 : core_row0 + core_height,
                        core_col0 : core_col0 + core_width,
                    ]

                    valid = np.isfinite(core)
                    if np.any(valid):
                        values = core[valid]
                        total_valid += int(values.size)
                        total_sum += float(values.sum())
                        global_min = min(global_min, float(values.min()))
                        global_max = max(global_max, float(values.max()))

                    dst.write(
                        core.astype(np.float32),
                        1,
                        window=Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=core_width,
                            height=core_height,
                        ),
                    )

    return {
        "filtered_valid_pixels": total_valid,
        "filtered_mean_db_QA_only": (
            total_sum / total_valid if total_valid else np.nan
        ),
        "filtered_min_db": global_min if total_valid else np.nan,
        "filtered_max_db": global_max if total_valid else np.nan,
    }


def describe_raster(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
            "pixel_size_x": float(abs(src.transform.a)),
            "pixel_size_y": float(abs(src.transform.e)),
            "transform": tuple(src.transform),
        }


def main() -> None:
    require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("WINTER UAVSAR SIGMA0 AUDIT — NO EMPIRICAL INCIDENCE NORMALIZATION")
    print("=" * 78)
    print(f"Output folder:\n  {OUTPUT_DIR}\n")
    print(f"Incidence map:\n  {INCIDENCE_TIF}\n")

    incidence_meta = describe_raster(INCIDENCE_TIF)
    print("Incidence map metadata:")
    print(
        f"  CRS={incidence_meta['crs']}, "
        f"shape=({incidence_meta['height']}, {incidence_meta['width']}), "
        f"pixel={incidence_meta['pixel_size_x']:.6g} x "
        f"{incidence_meta['pixel_size_y']:.6g}"
    )

    summary_rows = []
    output_files = {}

    for channel, input_path in INPUTS.items():
        band = channel[0]

        print("\n" + "-" * 78)
        print(f"PROCESSING {channel}")
        print("-" * 78)
        print(f"Input:\n  {input_path}")

        src_meta = describe_raster(input_path)
        print(
            f"Input metadata: CRS={src_meta['crs']}, "
            f"shape=({src_meta['height']}, {src_meta['width']}), "
            f"pixel={src_meta['pixel_size_x']:.6g} x "
            f"{src_meta['pixel_size_y']:.6g}"
        )

        unfiltered_path = (
            OUTPUT_DIR / f"{channel}_sigma0_no_inc_norm_unfiltered.tif"
        )
        filtered_path = (
            OUTPUT_DIR / f"{channel}_sigma0_no_inc_norm_leeSigma.tif"
        )

        print("\n1/2 Calibrating to noise-corrected sigma0 dB...")
        calibration_stats = calibrate_sigma0_no_empirical_normalization(
            input_path=input_path,
            incidence_path=INCIDENCE_TIF,
            channel=channel,
            output_path=unfiltered_path,
        )

        print(
            "    calibration QA: "
            f"valid={calibration_stats['calibrated_valid_pixels']:,}, "
            f"range={calibration_stats['calibrated_min_db']:.3f} to "
            f"{calibration_stats['calibrated_max_db']:.3f} dB"
        )

        print("\n2/2 Applying Lee-Sigma in LINEAR intensity...")
        filter_stats = lee_sigma_filter_tiled(
            calibrated_path=unfiltered_path,
            output_path=filtered_path,
            band=band,
        )

        print(
            "    filtered QA: "
            f"valid={filter_stats['filtered_valid_pixels']:,}, "
            f"range={filter_stats['filtered_min_db']:.3f} to "
            f"{filter_stats['filtered_max_db']:.3f} dB"
        )

        final_meta = describe_raster(filtered_path)

        if final_meta["crs"] != "EPSG:32607":
            print(
                "    NOTE: final raster is not EPSG:32607. "
                "That is not fatal; the next observation script can align grids."
            )

        if not KEEP_UNFILTERED:
            unfiltered_path.unlink(missing_ok=True)

        output_files[channel] = str(filtered_path)

        summary_rows.append(
            {
                "channel": channel,
                "input_file": str(input_path),
                "incidence_file": str(INCIDENCE_TIF),
                "noise_value": NOISE_VALUES[channel],
                "K_dB": K_DB,
                "empirical_incidence_normalization": False,
                "lee_window_size": WINDOW_SIZE,
                "lee_sigma_v": LEE_SIGMA_V[band],
                "lee_threshold": LEE_THRESHOLD,
                "final_file": str(filtered_path),
                **src_meta,
                **calibration_stats,
                **filter_stats,
            }
        )

        print(f"\n✓ {channel} final file:\n  {filtered_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "07_no_incidence_processing_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest = {
        "purpose": (
            "Create winter L/S co-pol sigma0 observations without the "
            "empirical normalization to 35 degrees."
        ),
        "calibration_equation": (
            "sigma0_dB = 10*log10(max(DN^2-N,1e-10)) "
            "+ 10*log10(sin(theta)) - 42"
        ),
        "important_note": (
            "The sin(theta) term is part of sigma0 calibration. "
            "The empirical theta_d*(theta-35) normalization is not applied."
        ),
        "season": "winter",
        "noise_values": NOISE_VALUES,
        "K_dB": K_DB,
        "garbage_threshold_db": GARBAGE_THRESHOLD_DB,
        "lee_sigma": {
            "window_size": WINDOW_SIZE,
            "sigma_v": LEE_SIGMA_V,
            "threshold": LEE_THRESHOLD,
            "domain": "linear intensity",
        },
        "incidence_tif": str(INCIDENCE_TIF),
        "outputs": output_files,
    }

    manifest_json = OUTPUT_DIR / "07_no_incidence_processing_manifest.json"
    manifest_json.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("No empirical incidence normalization was applied.")
    print("\nFinal filtered files:")
    for channel, path in output_files.items():
        print(f"  {channel}: {path}")

    print(f"\nSummary CSV:\n  {summary_csv}")
    print(f"Manifest JSON:\n  {manifest_json}")

    print(
        "\nSTOP HERE after this script finishes. "
        "The next step is to extract the exact same MYI/NI/TI pixels "
        "together with their incidence angles."
    )


if __name__ == "__main__":
    main()
