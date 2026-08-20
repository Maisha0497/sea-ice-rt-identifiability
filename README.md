# Sea-ice radiative-transfer identifiability with L/S-band SAR

Preliminary SMRT forward-model and identifiability study for multi-frequency sea-ice SAR.

## Project status

This repository is a **pilot / feasibility study**, not a finished sea-ice retrieval product. The immediate objective is to determine whether a physically defensible radiative-transfer forward model can reproduce the observations, and whether the physical parameters are identifiable enough to justify a later inverse model (e.g. a physics-constrained neural network).

The project deliberately follows this order:

```text
forward-model construction
        -> model-domain / IEM validity
        -> parameter sensitivity
        -> Jacobian + SVD identifiability
        -> synthetic inversion sanity check
        -> real-observation reachability
        -> observation / incidence-angle audit
        -> MYI structural-model correction
        -> physical-constraint tests
        -> first snow-physics test
        -> future inverse model
```

The real-data PINN inversion is **not implemented as a final result here**. That is intentional: an inverse model is only meaningful if the forward model can first represent the observations over a physically defensible parameter domain.

## Scientific question

Let the physical state be \(m\), the forward radiative-transfer model be \(G\), and the observed SAR vector be \(y\):

\[
y = G(m) + \eta.
\]

Before learning an inverse map \(y \rightarrow \hat m\), this project asks:

1. **Existence / reachability:** is there a physically allowed \(m\) for which \(G(m) \approx y\)?
2. **Identifiability:** do different physical states produce distinguishable SAR signatures?
3. **Stability:** are weak parameter directions strongly amplified by measurement noise?
4. **Model structure:** do assumptions such as FYI vs MYI structure, interface length scales, and snow representation materially change reachability?

## Forward model

The baseline experiments use:

- [SMRT](https://www.smrt-model.science/) for microwave radiative transfer;
- IBA electromagnetic model;
- DORT radiative-transfer solver;
- IEM-Fung-1992 rough interface;
- active L band (1.257 GHz) and S band (3.200 GHz);
- HH/HV/VV simulations in the early sensitivity experiments;
- L-HH, L-VV, S-HH and S-VV for the real-data reachability sequence.

`config/base.yaml` intentionally retains the original **first-year-ice baseline** used in experiments 00-09. Experiments 10-13 explicitly override this with SMRT's multiyear-ice structure when testing MYI observations.

## What has been demonstrated so far

### 1. Two-parameter sensitivity / SVD baseline

For the initial roughness-salinity diagnostic using all six L/S polarimetric channels, the whitened Jacobian had singular values approximately **9.25** and **2.73**, with condition number **3.39**. Sensor ablation showed substantially poorer local conditioning for L-only (**15.80**) than S-only (**2.08**) in this specific two-parameter baseline.

This result should **not** be interpreted as proving full sea-ice identifiability: it only diagnoses the two parameters included in that experiment.

### 2. Synthetic inversion sanity check

Synthetic observations generated from the same forward model were inverted successfully. With added 1 dB channel noise, a truth of

- roughness = 0.75 mm,
- salinity = 4.0 ppt

produced a best retrieval of approximately

- roughness = 0.799 mm,
- salinity = 2.29 ppt.

The observation reconstruction was good, while the parameter shift illustrates why low reconstruction loss is not by itself proof of unique physical recovery.

### 3. Initial real-observation reachability

Using the original observation-preparation route, the baseline bare-ice model showed substantial mismatch. Expanding the parameter bounds did not remove the discrepancy. Channel-subset tests also indicated frequency-dependent behavior (for example, S-only NI was much closer to the model manifold than L-only NI).

The committed 06 and 06b summary CSVs are explicitly labelled **historical pre-mean-audit** because they were generated before the representative class mean was changed to linear-power averaging. They are retained to document the debugging sequence, not for quantitative comparison with the later incidence-aware results. Experiment 06c reflects the later power-mean observation summary but still predates the no-incidence-normalization audit. The observation treatment was then rebuilt in experiments 07-08 before further physical interpretation.

### 4. Observation / incidence-angle audit

Experiments 07-09 removed the empirical normalization to 35 degrees, recomputed class observations with linear-power averaging, binned observations by actual incidence angle, and evaluated SMRT at each bin's measured mean angle.

The large MYI mismatch persisted under the first-year-ice structural model, so the discrepancy could not be explained simply by dB averaging or fixed 35-degree geometry.

### 5. Correcting the MYI structural model was decisive

The earlier MYI comparisons had inherited `ice_type: firstyear` from the baseline configuration. Experiment 10 explicitly used SMRT's `multiyear` representation and introduced air porosity.

For the four retained MYI incidence bins, joint L+S RMS mismatch changed from approximately:

| Incidence bin | FYI structure (dB) | MYI structure (dB) | Improvement (dB) |
|---|---:|---:|---:|
| 30-33° | 11.56 | 5.01 | 6.55 |
| 45-48° | 12.87 | 5.07 | 7.79 |
| 48-51° | 14.25 | 6.22 | 8.02 |
| 51-54° | 13.65 | 5.53 | 8.12 |

This shows that **forward-model structure, not only parameter tuning, strongly controls reachability**.

### 6. Separating surface and internal MYI length scales improved mathematical reachability

Experiment 11 allowed the IEM surface correlation length and the MYI internal microstructure/bubble correlation length to vary independently. Joint RMS mismatch fell to approximately **1.05-2.97 dB** across the four incidence bins.

However, some best-fitting states pushed salinity, porosity, thickness or correlation-length parameters toward broad diagnostic limits. Therefore these fits are treated as evidence of **model flexibility / parameter compensation**, not validated geophysical retrievals.

### 7. Physically tighter MYI bounds exposed a remaining L-band problem

Experiment 12 restricted internal MYI properties to tighter ranges:

- salinity: 1-4 ppt;
- thickness: 1-3 m;
- ice microstructure correlation length: 0.2-0.8 mm.

The joint mismatch increased to about **3.24-4.03 dB**. At the joint solutions, S-band RMS remained approximately **0.76-2.66 dB**, while L-band RMS remained approximately **4.27-5.65 dB**.

Several parameters again reached their allowed bounds. This is a central current result: an excellent unconstrained SAR fit does not imply a well-identified physical state.

### 8. A simple fresh/dry snow layer did not resolve the discrepancy

Experiment 13 froze the experiment-12 ice/interface state and added one fresh/dry snow layer while sweeping only snow depth, density and snow correlation length.

The best one-layer snow state **worsened** joint L+S RMS by approximately **0.23-0.30 dB** in every MYI incidence bin. No tested snow state improved L-band RMS by at least 0.5 dB while preserving S-band within the specified tolerance.

This does not show that snow is unimportant. It shows only that this simple one-layer fresh/dry representation is not sufficient to explain the remaining mismatch.

## Current interpretation

The current pilot supports the following research direction:

> **How can multi-frequency SAR be inverted with a radiative-transfer model while maintaining physical plausibility, identifiability and uncertainty awareness, rather than obtaining good fits through parameter compensation?**

The next physical questions include snow stratigraphy, saline/basal snow, snow-ice interface physics, deformation/heterogeneity and independent measurements needed to constrain the inverse problem. These are **future research questions**, not completed results in this repository.

## Repository layout

```text
config/        baseline sensor/model configuration
src/           reusable SMRT forward, inversion and sensitivity utilities
experiments/   numbered scientific diagnostics
results/       selected summary tables and figures from the pilot study
data/          templates and documentation; raw UAVSAR data are not distributed
```

Intermediate coarse grids and raw per-pixel observation tables are intentionally excluded from the repository. They can be regenerated from the scripts when the required source data are available.

## Installation

Create/activate a Python environment, install the Python dependencies, and install SMRT separately:

```bash
python -m pip install -r requirements.txt
```

SMRT was developed/tested in this project from a local checkout. Follow the current SMRT installation instructions for your environment and verify it with:

```bash
python experiments/00_check_smrt.py
```

## External data

Raw UAVSAR / processed raster inputs and ROI shapefiles are **not redistributed** in this repository.

Set the external data root before running the real-observation preparation scripts:

```bash
export RTE_PINN_ASAR_ROOT=/path/to/ASAR
```

The expected directory layout is documented in [`data/README.md`](data/README.md). Derived observation products are written to `data/derived/`, which is git-ignored.

## Suggested experiment order

The experiments are chronological rather than a single production pipeline. A reasonable reading/rerun order is:

```text
00 -> 01 -> 01b -> 02 -> 03 -> 04 -> 04b -> 05 -> 05b
prepare_real_observations -> 06 -> 06b -> 06c
07 -> 08 -> 09 -> 09b -> 10 -> 11 -> 12 -> 13
```

Experiments 06-06c use the earlier observation-preparation path; experiments 07 onward perform the incidence/averaging audit and should be used for the later MYI interpretation.

## Limitations and claim discipline

This repository does **not** establish:

- a validated retrieval of true sea-ice salinity, thickness, roughness or porosity;
- uniqueness of the full physical inverse problem;
- a completed real-data PINN/KAN inversion;
- validation of the simple snow representation against coincident field measurements;
- that the remaining L-band discrepancy is caused by any single missing process.

The defensible claim is narrower: the experiments diagnose forward-model adequacy, sensitivity and parameter compensation, and identify where additional physical constraints and measurements are required before a final inverse model is justified.

## Reference

SMRT: Picard, G., Sandells, M., & Löwe, H. (2018), *SMRT: An active-passive microwave radiative transfer model for snow with multiple microstructure and scattering formulations*, Geoscientific Model Development, 11, 2763-2788. https://doi.org/10.5194/gmd-11-2763-2018
