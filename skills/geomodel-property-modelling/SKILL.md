---
name: geomodel-property-modelling
description: "Use when populating continuous reservoir properties: trend removal and residual simulation, porosity workflow, permeability from porosity, kh checks against well tests, Kv/Kh assignment, water saturation from saturation-height functions, random seeds, conditioning to seismic (with its acceptance rule), thin barriers and thief zones, and single-realisation QC products. Primary user: the property agent."
---

# Property (petrophysical) modelling

**Used by:** property agent (owner); well agent (verifier: histograms, leave-one-out wells, variogram reproduction).

## 1. Principles

- Purpose (usually flow simulation) drives choices. **Permeability contrasts** — barriers, baffles, correlatable high-permeability beds — matter more than exact absolute values.
- **Trend first ("trendology"):** extract predictable trends by data analysis (porosity–depth compaction, lateral trends, cement differences between water and oil legs), remove them, simulate only the residual, add the trends back. Do not expect the algorithm to discover trends.
- Start simple. Usually only porosity is simulated; **k = f(φ, facies)** and **Sw = f(φ, k, height above FWL)**, using the petrophysicist's functions.
- A good model honours concept and wells without copying input statistics: well statistics are rarely representative averages and their variance is always too high. Check output statistics after every run, **including the hydrocarbon column only**.
- Build conceptual average property maps per zone before modelling; aim to reproduce them in 3D while honouring wells.
- Prefer Gaussian random-function simulation (fast, reproduces distributions) over sequential Gaussian; use kriging only for QC or where heterogeneity does not matter.

## 2. Porosity workflow

1. Filter by depositional facies.
2. Remove bad data; restore water-leg cemented samples to their uncemented state.
3. Datum to a fixed depth to remove the compaction trend.
4. Remove the lateral trend (map); the residual should be roughly normal and centred on zero (e.g. ±0.03).
5. Simulate the residual; add back the map average, depth trend and water-leg cement effect.

## 3. Permeability

- Fit a poro-perm regression on clean core plugs (exclude measurement errors and cemented plugs) and apply it to modelled porosity — preferred: keeps contrasts, gives values between arithmetic and geometric upscaled permeability, adds no extra noise. Alternative: co-krige upscaled log permeability with porosity (log-normal).
- Any superimposed scatter must be far smaller than plug scatter.
- **kh check:** compare model kh with log-derived kh and well-test kh; extract kh vs radius around tested wells; find geological explanations for boundaries the tests see (pinch-out, fault). Well tests are hard data.

## 4. Kv/Kh

1. Check how values were derived and that data and averaging are representative.
2. Assign **per zone**, varying with facies and properties and reflecting the concept — never one global value.
3. Kv usually comes from a Kh relationship, not upscaled logs; check it with zone and facies crossplots.
4. Use dynamic indicators (pressure breaks between zones) as well as rock data.
5. Ranges: lower in strongly layered systems with pressure breaks; higher in amalgamated sands; laminated, micaceous shoreface lower than channel sand. Typical tested values **0.01–0.6** (thin, extensive silty bed tops 0.01–0.1; injectites Kv ≈ Kh).
6. Plugs are micro-scale and biased to good rock; the model value must reflect sub-cell laminations.
7. Fine layering plus explicit shales can double-count barriers and exaggerate them.
8. Validate against well observations and simulated vertical communication. Kv/Kh is a key history-matching parameter.

## 5. Water saturation

- Use the **saturation-height function** (height above FWL, porosity, permeability — geometric permeability), not a generic simulation algorithm. Clip between Swirr and 1.
- Height above FWL: "centre of the part of the cell above the contact" when volumetrics use fractional cell volumes; "centre of whole cell" when full-cell volumes are used. Sw must be consistent with the GRV used for STOIIP.
- Compare with log Sw; plot Sw vs height above FWL coloured by permeability or facies (Sw rises as porosity falls). Agree whether the datum is OWC or FWL.
- Near the contact the model under-estimates Sw as J becomes asymptotic; it matters with thick cells, large transition zones or steep dips — then upscale Sw from a fine contact-parallel model.
- Agree with the reservoir engineer whether Sw is recomputed in the simulator; check in the static-vs-dynamic in-place comparison.

## 6. Random seeds

Realisations from the same inputs differ and are equally probable. **Set seeds explicitly** (fixed, or driven by the uncertainty workflow) so the base case is reproducible; fix the seed while testing other parameters.

## 7. Conditioning to seismic — acceptance rule

- Seismic response mixes fluid, thickness, lithology, quality, over- and underburden, imaging and noise; isolate the target property (fluid substitution, forward modelling, inversion).
- **Condition only when a relationship is demonstrated by crossplots** (e.g. impedance vs porosity, zone-average porosity vs attribute). **If the correlation coefficient is below 0.5, do not condition.**
- Absence of an expected signal is information (thin or absent reservoir, low net) unless imaging explains it.
- Remove artefacts first: low amplitudes near faults, shallow-gas effects, tuning (including tuning between contacts and top/base).
- Validate by forward modelling the 3D model to seismic; do not set low porosities to zero (unrealistic velocities).

## 8. Thin, flow-critical layers

High-permeability streaks and tight extensive baffles thinner than the grid must be represented explicitly (own zone or layers) or as transmissibility barriers, and must survive into the dynamic model.

## 9. Single-realisation QC products

- **Sections** (strike and dip through wells): porosity, permeability and Sw along wells vs logs and core; consistency with vertical proportion curves.
- **Maps per zone:** average net/facies, porosity, permeability, saturation; cross-check them (in a single-porosity system high porosity should coincide with good facies and high permeability); look for bull's-eyes and unexplained jumps.
- **Statistics:** plausible min/mean/max; expected histogram shape; compared with raw and upscaled logs.
- **Crossplots:** porosity vs depth (trend captured), poro-perm by facies or rock type, saturation-height, NTG vs porosity.
- **NTG** agrees with wells and concept.
- Before handover: porosity, NTG, Sw and PVT consistent, realistic and aligned with contacts and geology.

## 10. Ensemble expectations

Publish an **ensemble**, never a single realisation. Checklist: porosity distributions by facies match upscaled wells; mean and P10/P50/P90 within expected ranges; no local anomalies; poro-perm relationship preserved; permeability spans the uncertainty; Kv/Kh realistic.

## In this project (Norne)

- Delivered properties: PORO, NTG, PERMX (FACIES optional) in `grid.grdecl` / `props.grdecl`.
- Benchmark checks (`docs/NORNE_BENCHMARK.md` §5): along blind-well paths porosity MAE ≤ 0.02, log₁₀ k MAE ≤ 0.3, NTG MAE ≤ 0.05; R² of zone-average porosity maps ≥ 0.5. Water saturation is GIVEN (saturation-height per zone) for volumes.
- Libraries: gstools, geostatspy, gstatsim.
