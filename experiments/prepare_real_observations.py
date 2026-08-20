from __future__ import annotations

"""
prepare_real_observations.py

Extract matched winter L/S co-polarized pixels under MYI, NI, and TI shapefiles.

Outputs:
    observations/real_pixels_MYI.csv
    observations/real_pixels_NI.csv
    observations/real_pixels_TI.csv
    observations/real_observation_summary.csv
    observations/real_observation_summary.json

Important:
- All four rasters are sampled on the LHH reference grid.
- The same geometry mask and the same valid-pixel mask are used for every channel.
- Pixels are kept only where all four channels are finite and valid.
- Input raster values are assumed to already be in dB.
"""

from pathlib import Path
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import derived_data_dir, external_asar_root


# ---------------------------------------------------------------------
# EXTERNAL INPUTS AND PROJECT-LOCAL DERIVED OUTPUTS
# ---------------------------------------------------------------------

ASAR_ROOT = external_asar_root()

RASTER_DIR = (
    ASAR_ROOT / "ASAR+RCM Processed" / "Processed TIFFs"
    / "Winter Params tiffs processed" / "ALL TIFFS"
)

SHAPE_DIR = (
    ASAR_ROOT / "ASAR+RCM Processed" / "shapefiles"
    / "FID shapes" / "All merged" / "latest"
)

OUTPUT_DIR = derived_data_dir("real_observations_for_inversion")

RASTERS = {
    "L_HH": RASTER_DIR / "LHH_Level2_complete_winter.tif",
    "L_VV": RASTER_DIR / "LVV_Level2_complete_winter.tif",
    "S_HH": RASTER_DIR / "SHH_Level2_complete_winter.tif",
    "S_VV": RASTER_DIR / "SVV_Level2_complete_winter.tif",
}

SHAPEFILES = {
    "MYI": SHAPE_DIR / "ENVELOPE_MYI.shp",
    "NI": SHAPE_DIR / "ENVELOPE_NI.shp",
    "TI": SHAPE_DIR / "ENVELOPE_TI_merged.shp",
}

# Remove physically impossible / garbage values.
# Change these only if your calibrated TIFF range requires it.
MIN_DB = -140.0
MAX_DB = 20.0

# Set to an integer, e.g. 200000, to randomly limit saved pixels per class.
# None saves every matched valid pixel.
MAX_SAVED_PIXELS_PER_CLASS: int | None = None

RANDOM_SEED = 42


def require_files(paths: dict[str, Path], label: str) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(f"Missing {label} file(s):\n{message}")


def clamp_window(window: Window, width: int, height: int) -> Window:
    """Round a geometry-derived window and clamp it to raster bounds."""
    window = window.round_offsets().round_lengths()

    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(width, int(window.col_off + window.width))
    row_end = min(height, int(window.row_off + window.height))

    if col_end <= col_off or row_end <= row_off:
        raise ValueError("Shapefile does not overlap the raster grid.")

    return Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )


def build_geometry_mask(
    gdf: gpd.GeoDataFrame,
    reference: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, Window, rasterio.Affine]:
    """Create one mask on the reference grid for use by all four channels."""
    if gdf.empty:
        raise ValueError("Shapefile contains no features.")

    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS.")

    if reference.crs is None:
        raise ValueError("Reference raster has no CRS.")

    gdf_ref = gdf.to_crs(reference.crs)
    geometries = [
        geom.__geo_interface__
        for geom in gdf_ref.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geometries:
        raise ValueError("Shapefile contains no usable geometries.")

    minx, miny, maxx, maxy = gdf_ref.total_bounds

    window = clamp_window(
        from_bounds(minx, miny, maxx, maxy, transform=reference.transform),
        width=reference.width,
        height=reference.height,
    )

    window_transform = reference.window_transform(window)

    # True means the pixel center is inside at least one polygon.
    inside = geometry_mask(
        geometries,
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform,
        invert=True,
        all_touched=False,
    )

    return inside, window, window_transform


def read_on_reference_grid(
    dataset: rasterio.io.DatasetReader,
    reference: rasterio.io.DatasetReader,
    window: Window,
) -> np.ma.MaskedArray:
    """
    Read one channel on the exact reference CRS, transform, width, and height.

    WarpedVRT becomes a no-op when the raster is already aligned. Otherwise,
    it resamples onto the LHH grid using bilinear interpolation.
    """
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


def summarize_channel(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)

    # Existing method: arithmetic mean of dB values
    mean_of_db = float(np.mean(values))

    # Power-domain mean:
    # dB -> linear power -> average -> dB
    linear_power = 10.0 ** (values / 10.0)
    linear_mean_db = float(
        10.0 * np.log10(np.mean(linear_power))
    )

    return {
        # Use the linear-power average as the representative mean
        # consumed by the reachability scripts.
        f"{prefix}_mean_db": linear_mean_db,

        # Keep the old calculation so we can audit the difference.
        f"{prefix}_mean_of_db": mean_of_db,

        f"{prefix}_linear_vs_db_mean_difference_db":
            linear_mean_db - mean_of_db,

        f"{prefix}_median_db": float(np.median(values)),
        f"{prefix}_std_db": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
        f"{prefix}_p05_db": float(np.percentile(values, 5)),
        f"{prefix}_p25_db": float(np.percentile(values, 25)),
        f"{prefix}_p75_db": float(np.percentile(values, 75)),
        f"{prefix}_p95_db": float(np.percentile(values, 95)),
        f"{prefix}_min_db": float(np.min(values)),
        f"{prefix}_max_db": float(np.max(values)),
    }

def main() -> None:
    require_files(RASTERS, "raster")
    require_files(SHAPEFILES, "shapefile")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    summary_rows: list[dict[str, object]] = []

    with rasterio.open(RASTERS["L_HH"]) as reference:
        raster_handles = {
            name: rasterio.open(path)
            for name, path in RASTERS.items()
        }

        try:
            for ice_type, shp_path in SHAPEFILES.items():
                print(f"\nProcessing {ice_type}: {shp_path.name}")

                gdf = gpd.read_file(shp_path)
                inside, window, window_transform = build_geometry_mask(
                    gdf,
                    reference,
                )

                channel_arrays: dict[str, np.ma.MaskedArray] = {}

                for channel, dataset in raster_handles.items():
                    arr = read_on_reference_grid(
                        dataset=dataset,
                        reference=reference,
                        window=window,
                    )

                    if arr.shape != inside.shape:
                        raise RuntimeError(
                            f"{channel} shape {arr.shape} does not match "
                            f"mask shape {inside.shape}."
                        )

                    channel_arrays[channel] = arr

                # One common valid mask for all channels.
                common_valid = inside.copy()

                for channel, arr in channel_arrays.items():
                    data = np.asarray(arr.data, dtype=np.float64)
                    arr_mask = np.ma.getmaskarray(arr)

                    common_valid &= ~arr_mask
                    common_valid &= np.isfinite(data)
                    common_valid &= data > MIN_DB
                    common_valid &= data < MAX_DB

                n_polygon_pixels = int(np.count_nonzero(inside))
                n_valid_pixels = int(np.count_nonzero(common_valid))

                if n_valid_pixels == 0:
                    raise RuntimeError(
                        f"No common valid four-channel pixels found for {ice_type}."
                    )

                rows, cols = np.where(common_valid)
                xs, ys = rasterio.transform.xy(
                    window_transform,
                    rows,
                    cols,
                    offset="center",
                )

                pixel_df = pd.DataFrame(
                    {
                        "ice_type": ice_type,
                        "row_in_window": rows,
                        "col_in_window": cols,
                        "x": np.asarray(xs, dtype=float),
                        "y": np.asarray(ys, dtype=float),
                        "L_HH": channel_arrays["L_HH"].data[common_valid],
                        "L_VV": channel_arrays["L_VV"].data[common_valid],
                        "S_HH": channel_arrays["S_HH"].data[common_valid],
                        "S_VV": channel_arrays["S_VV"].data[common_valid],
                    }
                )

                original_count = len(pixel_df)

                if (
                    MAX_SAVED_PIXELS_PER_CLASS is not None
                    and original_count > MAX_SAVED_PIXELS_PER_CLASS
                ):
                    selected = rng.choice(
                        original_count,
                        size=MAX_SAVED_PIXELS_PER_CLASS,
                        replace=False,
                    )
                    pixel_df = (
                        pixel_df.iloc[np.sort(selected)]
                        .reset_index(drop=True)
                    )

                pixel_path = OUTPUT_DIR / f"real_pixels_{ice_type}.csv"
                pixel_df.to_csv(pixel_path, index=False)

                summary: dict[str, object] = {
                    "ice_type": ice_type,
                    "shapefile": str(shp_path),
                    "reference_raster": str(RASTERS["L_HH"]),
                    "crs": str(reference.crs),
                    "polygon_pixel_count": n_polygon_pixels,
                    "common_valid_pixel_count": n_valid_pixels,
                    "saved_pixel_count": int(len(pixel_df)),
                    "valid_fraction_within_polygon": (
                        n_valid_pixels / n_polygon_pixels
                        if n_polygon_pixels > 0
                        else np.nan
                    ),
                }

                for channel in RASTERS:
                    values = np.asarray(
                        channel_arrays[channel].data[common_valid],
                        dtype=float,
                    )
                    summary.update(summarize_channel(values, channel))

                summary_rows.append(summary)

                print(f"  Polygon pixels: {n_polygon_pixels:,}")
                print(f"  Common valid pixels: {n_valid_pixels:,}")
                print(f"  Saved pixels: {len(pixel_df):,}")
                print(f"  Output: {pixel_path}")

        finally:
            for dataset in raster_handles.values():
                dataset.close()

    summary_df = pd.DataFrame(summary_rows)

    # Put the key inversion columns first.
    first_columns = [
        "ice_type",
        "common_valid_pixel_count",
        "saved_pixel_count",
        "L_HH_mean_db",
        "L_VV_mean_db",
        "S_HH_mean_db",
        "S_VV_mean_db",
        "L_HH_std_db",
        "L_VV_std_db",
        "S_HH_std_db",
        "S_VV_std_db",
    ]
    remaining_columns = [
        column
        for column in summary_df.columns
        if column not in first_columns
    ]
    summary_df = summary_df[first_columns + remaining_columns]

    summary_csv = OUTPUT_DIR / "real_observation_summary.csv"
    summary_json = OUTPUT_DIR / "real_observation_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            summary_df.to_dict(orient="records"),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nCompleted.")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("\nMean dB observations:")
    print(
        summary_df[
            [
                "ice_type",
                "L_HH_mean_db",
                "L_VV_mean_db",
                "S_HH_mean_db",
                "S_VV_mean_db",
                "common_valid_pixel_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
