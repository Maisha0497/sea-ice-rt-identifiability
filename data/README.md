# Data notes

Raw UAVSAR data, processed GeoTIFFs and ROI shapefiles are not redistributed in this repository.

## External root

Set:

```bash
export RTE_PINN_ASAR_ROOT=/path/to/ASAR
```

The public scripts expect this root to contain the same broad structure used in the pilot study:

```text
ASAR/
├── Originals/
│   └── Winter/
│       ├── L_tb_extracted_all files/
│       │   └── ASAR_L_JOINT_FP_ID01118_LINE01_RUN01_141219_V1.3.B/
│       └── S/
│           └── ASAR_S_JOINT_FP_ID01118_LINE01_RUN01_141219_V1.3.B/
│
└── ASAR+RCM Processed/
    ├── Processed TIFFs/
    │   └── Winter Params tiffs processed/
    │       └── ALL TIFFS/
    └── shapefiles/
        └── FID shapes/
            └── All merged/
                └── latest/
```

The scripts check for the exact raster and shapefile names they require and will raise an informative `FileNotFoundError` if an input is absent.

## Derived products

Regenerated observation products are written under:

```text
data/derived/
```

This directory is intentionally git-ignored. It may contain:

```text
data/derived/
├── real_observations_for_inversion/
└── SMRT_observation_audit/
    ├── no_incidence_normalization/
    └── incidence_binned/
```

The later reachability experiments read the incidence-binned summary from this project-local derived directory.

## `observations_template.csv`

The template documents the observation-vector convention used by the early inversion utilities. It is not a substitute for the UAVSAR observation-preparation workflow.
