#!/usr/bin/env python3
from __future__ import annotations

"""
08_prepare_incidence_binned_observations.py

Extract matched winter MYI/NI/TI pixels from the four no-incidence-normalization
sigma0 rasters produced by experiment 07, attach the Level-2 incidence angle,
and summarize them in 3-degree incidence bins.

Representative backscatter means use:
    dB -> linear power -> arithmetic mean -> dB

Outputs
-------
08_pixels_MYI.csv
08_pixels_NI.csv
08_pixels_TI.csv
08_class_overall_observations.csv
08_incidence_binned_observations.csv
08_incidence_binned_observations.json
08_processing_manifest.json
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import derived_data_dir, external_asar_root


# ---------------------------------------------------------------------
# EXTERNAL INPUTS AND PROJECT-LOCAL DERIVED OUTPUTS
# ---------------------------------------------------------------------

ASAR_ROOT = external_asar_root()

INPUT_DIR = derived_data_dir(
    "SMRT_observation_audit", "no_incidence_normalization"
)

RASTERS = {
    "L_HH": INPUT_DIR / "LHH_sigma0_no_inc_norm_leeSigma.tif",
    "L_VV": INPUT_DIR / "LVV_sigma0_no_inc_norm_leeSigma.tif",
    "S_HH": INPUT_DIR / "SHH_sigma0_no_inc_norm_leeSigma.tif",
    "S_VV": INPUT_DIR / "SVV_sigma0_no_inc_norm_leeSigma.tif",
}

INCIDENCE_TIF = (
    ASAR_ROOT / "Originals" / "Winter" / "S"
    / "ASAR_S_JOINT_FP_ID01118_LINE01_RUN01_141219_V1.3.B"
    / "ASAR_L_S_JOINT_FP_ID01118_LINE01_RUN01_141219_LEVEL2_INC_MAP_V1.3.B.tif"
)

SHAPE_DIR = (
    ASAR_ROOT / "ASAR+RCM Processed" / "shapefiles"
    / "FID shapes" / "All merged" / "latest"
)

SHAPEFILES = {
    "MYI": SHAPE_DIR / "ENVELOPE_MYI.shp",
    "NI": SHAPE_DIR / "ENVELOPE_NI.shp",
    "TI": SHAPE_DIR / "ENVELOPE_TI_merged.shp",
}

OUTPUT_DIR = derived_data_dir(
    "SMRT_observation_audit", "incidence_binned"
)

CHANNELS = ("L_HH", "L_VV", "S_HH", "S_VV")

MIN_DB = -140.0
MAX_DB = 20.0
MIN_INCIDENCE_DEG = 20.0
MAX_INCIDENCE_DEG = 60.0

# 3-degree bins: [30,33), [33,36), ..., [51,54)
BIN_EDGES_DEG = np.arange(30.0, 54.0 + 0.001, 3.0)
MIN_PIXELS_PER_BIN = 100


def require_files(paths: dict[str, Path], label: str) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} file(s):\n" + "\n".join(missing)
        )


def clamp_window(window: Window, width: int, height: int) -> Window:
    window = window.round_offsets().round_lengths()
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(width, int(window.col_off + window.width))
    row_end = min(height, int(window.row_off + window.height))

    if col_end <= col_off or row_end <= row_off:
        raise ValueError("Shapefile does not overlap raster.")

    return Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )


def build_geometry_mask(gdf, reference):
    if gdf.empty:
        raise ValueError("Shapefile contains no features.")
    if gdf.crs is None or reference.crs is None:
        raise ValueError("Missing CRS.")

    gdf_ref = gdf.to_crs(reference.crs)
    geometries = [
        geom.__geo_interface__
        for geom in gdf_ref.geometry
        if geom is not None and not geom.is_empty
    ]

    minx, miny, maxx, maxy = gdf_ref.total_bounds
    window = clamp_window(
        from_bounds(minx, miny, maxx, maxy, transform=reference.transform),
        reference.width,
        reference.height,
    )
    transform = reference.window_transform(window)

    inside = geometry_mask(
        geometries,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        invert=True,
        all_touched=False,
    )
    return inside, window, transform


def read_on_reference_grid(dataset, reference, window):
    with WarpedVRT(
        dataset,
        crs=reference.crs,
        transform=reference.transform,
        width=reference.width,
        height=reference.height,
        resampling=Resampling.bilinear,
        nodata=dataset.nodata,
    ) as vrt:
        return vrt.read(1, window=window, masked=True).astype(np.float64)


def power_mean_db(values_db: np.ndarray) -> float:
    values_db = np.asarray(values_db, dtype=float)
    linear = 10.0 ** (values_db / 10.0)
    return float(10.0 * np.log10(np.mean(linear)))


def summarize(values_db: np.ndarray, prefix: str) -> dict[str, float]:
    values_db = np.asarray(values_db, dtype=float)
    linear_mean_db = power_mean_db(values_db)
    db_mean = float(np.mean(values_db))
    return {
        f"{prefix}_mean_db": linear_mean_db,
        f"{prefix}_mean_of_db": db_mean,
        f"{prefix}_linear_vs_db_mean_difference_db": linear_mean_db - db_mean,
        f"{prefix}_median_db": float(np.median(values_db)),
        f"{prefix}_std_db": float(np.std(values_db, ddof=1)) if len(values_db) > 1 else 0.0,
        f"{prefix}_p05_db": float(np.percentile(values_db, 5)),
        f"{prefix}_p95_db": float(np.percentile(values_db, 95)),
    }


def main() -> None:
    require_files(RASTERS, "backscatter")
    require_files(SHAPEFILES, "shapefile")

    if not INCIDENCE_TIF.exists():
        raise FileNotFoundError(f"Missing incidence map:\n{INCIDENCE_TIF}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        f"{BIN_EDGES_DEG[i]:.0f}-{BIN_EDGES_DEG[i+1]:.0f}"
        for i in range(len(BIN_EDGES_DEG) - 1)
    ]

    overall_rows = []
    binned_rows = []

    print("=" * 78)
    print("INCIDENCE-BINNED UAVSAR OBSERVATION PREPARATION")
    print("=" * 78)
    print("Input: no empirical 35-degree normalization")
    print("Mean: dB -> linear power -> mean -> dB")
    print(f"Bins: {', '.join(labels)} deg")

    with rasterio.open(RASTERS["L_HH"]) as reference:
        handles = {name: rasterio.open(path) for name, path in RASTERS.items()}
        inc_handle = rasterio.open(INCIDENCE_TIF)

        try:
            for ice_type, shp_path in SHAPEFILES.items():
                print(f"\n--- {ice_type} ---")

                gdf = gpd.read_file(shp_path)
                inside, window, transform = build_geometry_mask(gdf, reference)

                arrays = {
                    name: read_on_reference_grid(ds, reference, window)
                    for name, ds in handles.items()
                }
                incidence = read_on_reference_grid(
                    inc_handle, reference, window
                )

                common_valid = inside.copy()

                for name, arr in arrays.items():
                    data = np.asarray(arr.data, dtype=float)
                    common_valid &= ~np.ma.getmaskarray(arr)
                    common_valid &= np.isfinite(data)
                    common_valid &= data > MIN_DB
                    common_valid &= data < MAX_DB

                inc_data = np.asarray(incidence.data, dtype=float)
                common_valid &= ~np.ma.getmaskarray(incidence)
                common_valid &= np.isfinite(inc_data)
                common_valid &= inc_data >= MIN_INCIDENCE_DEG
                common_valid &= inc_data <= MAX_INCIDENCE_DEG

                n_polygon = int(np.count_nonzero(inside))
                n_common = int(np.count_nonzero(common_valid))

                if n_common == 0:
                    raise RuntimeError(
                        f"No common valid pixels for {ice_type}."
                    )

                rr, cc = np.where(common_valid)
                xs, ys = rasterio.transform.xy(
                    transform, rr, cc, offset="center"
                )

                pixel_df = pd.DataFrame(
                    {
                        "ice_type": ice_type,
                        "row_in_window": rr,
                        "col_in_window": cc,
                        "x": np.asarray(xs, dtype=float),
                        "y": np.asarray(ys, dtype=float),
                        "incidence_angle_deg": inc_data[common_valid],
                        "L_HH": arrays["L_HH"].data[common_valid],
                        "L_VV": arrays["L_VV"].data[common_valid],
                        "S_HH": arrays["S_HH"].data[common_valid],
                        "S_VV": arrays["S_VV"].data[common_valid],
                    }
                )

                pixel_df["incidence_bin"] = pd.cut(
                    pixel_df["incidence_angle_deg"],
                    bins=BIN_EDGES_DEG,
                    labels=labels,
                    right=False,
                    include_lowest=True,
                ).astype("string")

                pixel_path = OUTPUT_DIR / f"08_pixels_{ice_type}.csv"
                pixel_df.to_csv(pixel_path, index=False)

                inc_all = pixel_df["incidence_angle_deg"].to_numpy(float)

                overall = {
                    "ice_type": ice_type,
                    "polygon_pixel_count": n_polygon,
                    "common_valid_pixel_count": n_common,
                    "incidence_mean_deg": float(np.mean(inc_all)),
                    "incidence_median_deg": float(np.median(inc_all)),
                    "incidence_p05_deg": float(np.percentile(inc_all, 5)),
                    "incidence_p95_deg": float(np.percentile(inc_all, 95)),
                }

                for ch in CHANNELS:
                    overall.update(
                        summarize(pixel_df[ch].to_numpy(float), ch)
                    )

                overall_rows.append(overall)

                print(f"Common valid pixels: {n_common:,}")
                print(
                    f"Incidence mean={overall['incidence_mean_deg']:.3f} deg, "
                    f"p05-p95={overall['incidence_p05_deg']:.3f}-"
                    f"{overall['incidence_p95_deg']:.3f} deg"
                )

                for i, label in enumerate(labels):
                    sub = pixel_df[pixel_df["incidence_bin"] == label]
                    n = len(sub)

                    if n < MIN_PIXELS_PER_BIN:
                        if n:
                            print(
                                f"  {label:>5} deg: {n:,} pixels "
                                f"SKIPPED (<{MIN_PIXELS_PER_BIN})"
                            )
                        continue

                    inc = sub["incidence_angle_deg"].to_numpy(float)

                    row = {
                        "ice_type": ice_type,
                        "incidence_bin": label,
                        "bin_lower_deg": float(BIN_EDGES_DEG[i]),
                        "bin_upper_deg": float(BIN_EDGES_DEG[i + 1]),
                        "n_pixels": int(n),
                        "incidence_mean_deg": float(np.mean(inc)),
                        "incidence_median_deg": float(np.median(inc)),
                        "incidence_std_deg": float(np.std(inc, ddof=1)) if n > 1 else 0.0,
                        "incidence_p05_deg": float(np.percentile(inc, 5)),
                        "incidence_p95_deg": float(np.percentile(inc, 95)),
                    }

                    for ch in CHANNELS:
                        row.update(
                            summarize(sub[ch].to_numpy(float), ch)
                        )

                    binned_rows.append(row)

                    print(
                        f"  {label:>5} deg | n={n:6,d} | "
                        f"theta={row['incidence_mean_deg']:.3f} | "
                        f"LHH={row['L_HH_mean_db']:.3f}, "
                        f"LVV={row['L_VV_mean_db']:.3f}, "
                        f"SHH={row['S_HH_mean_db']:.3f}, "
                        f"SVV={row['S_VV_mean_db']:.3f} dB"
                    )

                print(f"Saved pixels: {pixel_path}")

        finally:
            for ds in handles.values():
                ds.close()
            inc_handle.close()

    overall_df = pd.DataFrame(overall_rows)
    binned_df = pd.DataFrame(binned_rows)

    if binned_df.empty:
        raise RuntimeError("No incidence bins were retained.")

    overall_csv = OUTPUT_DIR / "08_class_overall_observations.csv"
    bins_csv = OUTPUT_DIR / "08_incidence_binned_observations.csv"
    bins_json = OUTPUT_DIR / "08_incidence_binned_observations.json"
    manifest_json = OUTPUT_DIR / "08_processing_manifest.json"

    overall_df.to_csv(overall_csv, index=False)
    binned_df.to_csv(bins_csv, index=False)
    bins_json.write_text(
        json.dumps(binned_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "backscatter_rasters": {k: str(v) for k, v in RASTERS.items()},
        "incidence_tif": str(INCIDENCE_TIF),
        "shapefiles": {k: str(v) for k, v in SHAPEFILES.items()},
        "mean_convention": "dB -> linear power -> mean -> dB",
        "common_pixel_requirement": (
            "inside polygon and valid in L_HH, L_VV, S_HH, S_VV, incidence"
        ),
        "incidence_bin_edges_deg": BIN_EDGES_DEG.tolist(),
        "minimum_pixels_per_bin": MIN_PIXELS_PER_BIN,
    }
    manifest_json.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Overall summary:\n  {overall_csv}")
    print(f"Binned summary:\n  {bins_csv}")
    print(f"Binned JSON:\n  {bins_json}")
    print(f"Manifest:\n  {manifest_json}")

    print("\nOverall no-normalization class means:")
    print(
        overall_df[
            [
                "ice_type",
                "incidence_mean_deg",
                "L_HH_mean_db",
                "L_VV_mean_db",
                "S_HH_mean_db",
                "S_VV_mean_db",
                "common_valid_pixel_count",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSTOP HERE. Next step: run SMRT at each retained bin's "
        "actual mean incidence angle."
    )


if __name__ == "__main__":
    main()
