# Sea-ice radiative-transfer identifiability with L/S-band SAR

**Pilot study using SMRT to test forward-model adequacy, identifiability, and the feasibility of a later physics-constrained inverse model for multi-frequency sea-ice SAR.**

> **Current status:** this is a feasibility / methods study, not a validated geophysical retrieval product. The real-data PINN inversion is intentionally not presented as a final result because the forward model and parameter identifiability are being tested first.

## At a glance

```mermaid
flowchart LR
    A[Observed L/S UAVSAR] --> B[SMRT forward model]
    B --> C[Initial FYI structure used for MYI]
    C --> D[Joint RMS ~11-14 dB]
    D --> E[Correct MYI structure + porosity]
    E --> F[Joint RMS ~5-6 dB]
    F --> G[Separate surface and internal length scales]
    G --> H[Joint RMS ~1-3 dB]
    H --> I[But broad parameters compensate]
    I --> J[Physically tighter MYI bounds]
    J --> K[Joint RMS ~3-4 dB\nremaining mismatch mainly L-band]
    K --> L[Add simple fresh/dry snow layer]
    L --> M[No improvement]
    M --> N[Future PhD question:\nphysically constrained + identifiable inversion]
```

### Main findings

- **Forward-model structure matters strongly.** Correcting the MYI representation reduced joint L+S mismatch by roughly **6.5-8.1 dB** across the retained incidence bins.
- **A very good SAR fit does not automatically imply a meaningful physical retrieval.** Allowing broad surface and internal length-scale freedom produced joint RMS values of about **1.05-2.97 dB**, but several parameters moved toward broad diagnostic limits, indicating parameter compensation.
- **Physical constraints expose the remaining problem.** With tighter MYI ranges, joint RMS increased to about **3.24-4.03 dB**; S-band remained comparatively close while L-band retained about **4.27-5.65 dB RMS** mismatch.
- **A simple homogeneous fresh/dry snow layer did not solve the residual.** It slightly worsened the best joint fit in every tested incidence bin.

## Why this project starts with the forward problem

Let the sea-ice physical state be \(m\), the forward radiative-transfer model be \(G\), and the observed SAR vector be \(y\):

\[
y = G(m) + \eta.
\]

The eventual goal is an inverse mapping

\[
y \rightarrow \hat m,
\]

potentially using optimization or a physics-constrained neural network. Before doing that, this project asks four prior questions:

1. **Existence / reachability:** is there a physically allowed \(m\) for which \(G(m) \approx y\)?
2. **Identifiability:** can different physical states be distinguished from their SAR signatures?
3. **Stability:** do weak parameter directions amplify observation noise strongly?
4. **Model structure:** do assumptions such as FYI vs MYI structure, interface length scales, and snow representation materially change reachability?

If the forward model cannot reproduce the observations over a physically defensible parameter space, an inverse model cannot recover physically meaningful parameters merely by changing the optimizer or neural-network architecture.

## Forward model

The baseline experiments use:

- [SMRT](https://www.smrt-model.science/) for microwave radiative transfer;
- IBA electromagnetic model;
- DORT radiative-transfer solver;
- IEM-Fung-1992 rough-interface representation;
- active L band at approximately **1.257 GHz** and S band at approximately **3.200 GHz**;
- HH/HV/VV simulations in the early sensitivity experiments;
- L-HH, L-VV, S-HH and S-VV for the real-data reachability sequence.

`config/base.yaml` retains the original first-year-ice baseline used in experiments 00-09. Experiments 10-13 explicitly override this with SMRT's multiyear-ice structure for the MYI tests.

## Experiment sequence

### 1. Forward-model construction and IEM validity

Experiments 00-01 established a reproducible active L/S SMRT forward simulation. Experiment 01b mapped where the IEM rough-surface configuration was numerically valid before using that model inside parameter searches.

![IEM validity diagnostic](results/01b_iem_validity_grid.png)

### 2. Parameter sensitivity

Experiment 02 varied ice thickness, salinity and RMS roughness one at a time to establish how the simulated SAR responds to each parameter.

<p align="center">
  <img src="results/02_sweep_roughness_rms_m.png" width="32%" />
  <img src="results/02_sweep_salinity_ppt.png" width="32%" />
  <img src="results/02_sweep_ice_thickness_m.png" width="32%" />
</p>

### 3. Local identifiability with Jacobian + SVD

Experiment 03 built a finite-difference Jacobian and decomposed it using SVD. In the initial two-parameter roughness-salinity diagnostic, the whitened Jacobian had singular values of approximately **9.25** and **2.73**, with condition number **3.39**.

The interpretation is local: the singular vectors identify parameter combinations seen by the SAR, while the singular values indicate how strongly those combinations are observed. This does **not** establish identifiability of the full sea-ice parameter set.

Experiment 04 then compared L-only, S-only and L+S information content. In that specific two-parameter baseline, L-only conditioning was poorer (**15.80**) than S-only (**2.08**), illustrating that frequency combinations change the local information content.

### 4. Synthetic inversion sanity check

Experiments 05 and 05b generated synthetic observations from the same forward model and inverted them. This tests whether the inversion machinery can recover a forward-model-consistent solution before using real observations.

With 1 dB synthetic channel noise, a truth of roughness **0.75 mm** and salinity **4.0 ppt** yielded a best solution of approximately roughness **0.799 mm** and salinity **2.29 ppt**. The observation reconstruction remained good even though the parameter estimate shifted.

This illustrates a central inverse-problem point: **low reconstruction loss does not by itself prove unique physical recovery.**

![Noisy synthetic cost surface](results/05b_cost_surface_noisy.png)

### 5. Initial reachability with real UAVSAR-derived observations

Experiment 06 asked whether any allowed bare-ice state could reproduce the real class-level observations. Expanding the bounds in 06b did not remove the mismatch. Experiment 06c separated L-only, S-only and joint reachability and showed clear frequency-dependent behavior.

The 06 and 06b summary tables are retained as **historical pre-mean-audit diagnostics**. They were generated before the representative class mean was changed to linear-power averaging and therefore should not be quantitatively compared with the later incidence-aware results.

### 6. Observation and incidence-angle audit

Experiments 07-09 rebuilt the comparison more carefully:

- removed the empirical normalization to 35°;
- averaged SAR in linear power before converting back to dB;
- binned the observations by actual incidence angle;
- ran SMRT at each bin's measured mean angle.

The large MYI mismatch remained while the model still used the first-year-ice structural representation. Therefore the discrepancy was not explained simply by dB averaging or fixed 35° geometry.

![Incidence-aware MYI reachability before structural correction](results/09_incidence_aware_reachability_MYI.png)

Experiment 09b replaced a refinement that could enter invalid IEM regions with a safe derivative-free refinement. The refined search did not improve the coarse-grid solution, confirming that the remaining mismatch was not simply a local optimizer failure for that model structure.

### 7. Correcting the MYI structural model

A key debugging step was recognizing that the earlier MYI comparisons had inherited `ice_type: firstyear` from the baseline configuration. Experiment 10 used SMRT's multiyear representation and introduced air porosity.

| Incidence bin | FYI structure joint RMS (dB) | MYI structure joint RMS (dB) | Improvement (dB) |
|---|---:|---:|---:|
| 30-33° | 11.56 | 5.01 | 6.55 |
| 45-48° | 12.87 | 5.07 | 7.79 |
| 48-51° | 14.25 | 6.22 | 8.02 |
| 51-54° | 13.65 | 5.53 | 8.12 |

**Interpretation:** forward-model structure, not only parameter tuning, strongly controls whether the observations are reachable.

### 8. Separating surface and internal MYI length scales

Experiment 11 allowed two physically distinct scales to vary independently:

- IEM surface correlation length;
- internal MYI microstructure / bubble correlation length.

Joint RMS fell to approximately **1.05-2.97 dB** across the retained incidence bins.

However, several broad-search solutions pushed salinity, porosity, thickness or correlation-length parameters toward diagnostic limits. Therefore these solutions are interpreted as evidence of **model flexibility and parameter compensation**, not validated geophysical retrievals.

### 9. Physically tighter MYI bounds

Experiment 12 restricted internal MYI properties to tighter ranges:

- salinity: **1-4 ppt**;
- thickness: **1-3 m**;
- internal ice correlation length: **0.2-0.8 mm**.

The joint mismatch increased to approximately **3.24-4.03 dB**. At the joint solutions:

- S-band RMS was approximately **0.76-2.66 dB**;
- L-band RMS was approximately **4.27-5.65 dB**.

Several parameters again reached their allowed bounds. This is the central current result: **a mathematically excellent SAR fit can be produced through parameter compensation, while physically tighter assumptions expose a remaining mainly L-band discrepancy.**

### 10. First snow-physics test

Experiment 13 froze the experiment-12 ice/interface state and added a single fresh/dry snow layer while varying only snow depth, density and snow correlation length.

The best snow state worsened joint L+S RMS by approximately **0.23-0.30 dB** in every retained MYI incidence bin. No tested snow state improved L-band RMS by at least 0.5 dB while keeping S-band within the diagnostic tolerance.

This does **not** show that snow is unimportant. It shows that this simple one-layer fresh/dry representation is not sufficient to explain the remaining mismatch.

## Current research direction

The pilot motivates the following broader question:

> **How can multi-frequency SAR be inverted with a radiative-transfer model while maintaining physical plausibility, identifiability and uncertainty awareness, rather than obtaining good fits through parameter compensation?**

Potential next questions include:

- snow stratigraphy and vertical layering;
- saline or basal snow;
- snow-ice interface properties;
- deformed / heterogeneous MYI;
- improved microstructure representations;
- independent field measurements needed to constrain the inverse problem;
- observation covariance, priors and posterior uncertainty;
- eventual real-data physics-constrained inversion.

These are future research directions, not completed claims in this repository.

## Repository layout

```text
config/        baseline sensor/model configuration
src/           reusable SMRT forward, inversion and sensitivity utilities
experiments/   numbered scientific diagnostics
results/       selected summary tables and figures
data/          templates and documentation; raw UAVSAR data are not distributed
```

Intermediate coarse grids and raw per-pixel observation tables are intentionally excluded from the public repository.

## Data and provenance

The real-observation experiments use **derived L/S-band SAR observations originating from the UAVSAR datasets used in the underlying MSc sea-ice work**. Raw UAVSAR products, processed rasters, ROI shapefiles and per-pixel thesis datasets are **not redistributed** in this repository.

To run the real-observation preparation scripts with an authorized local copy of the source data:

```bash
export RTE_PINN_ASAR_ROOT=/path/to/ASAR
```

The expected directory layout is documented in [`data/README.md`](data/README.md). Derived observation products are written to `data/derived/`, which is git-ignored.

## Suggested experiment order

```text
00 -> 01 -> 01b -> 02 -> 03 -> 04 -> 04b -> 05 -> 05b
prepare_real_observations -> 06 -> 06b -> 06c
07 -> 08 -> 09 -> 09b -> 10 -> 11 -> 12 -> 13
```

The experiments are chronological diagnostics rather than a single production pipeline.

## Installation

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install SMRT separately using the current SMRT installation instructions and verify the environment with:

```bash
python experiments/00_check_smrt.py
```

## Limitations and claim discipline

This repository does **not** establish:

- a validated retrieval of true sea-ice salinity, thickness, roughness or porosity;
- uniqueness of the full physical inverse problem;
- a completed real-data PINN/KAN inversion;
- validation of the simple snow representation against coincident field measurements;
- that the remaining L-band discrepancy is caused by any single missing process.

The defensible conclusion is narrower: **the experiments diagnose forward-model adequacy, local sensitivity, model-structure dependence and parameter compensation, and identify where additional physics and independent constraints are required before a final inverse model is justified.**

## Reference

Picard, G., Sandells, M., & Löwe, H. (2018). *SMRT: An active-passive microwave radiative transfer model for snow with multiple microstructure and scattering formulations*. Geoscientific Model Development, 11, 2763-2788. https://doi.org/10.5194/gmd-11-2763-2018
