---
name: geomodel-well-log-upscaling
description: "Use when blocking well logs onto grid cells: averaging method per log type, biasing to facies or net, treatment of horizontal wells, shoulder effects and their remedies, permeability and vertical-permeability averaging, upscaling with fractional net-to-gross, and raw-vs-upscaled QC. Primary user: the property agent (verified by the well agent)."
---

# Well-log upscaling (blocking)

**Used by:** property agent (owner); well agent (verifier: upscaled vs fine-log histograms).

Each cell holds one value. Logs are blocked onto the cells the wells penetrate ("informed cells"), which then condition property modelling. Method choice matters little when cells are finer than the features of interest and a lot when cells are much thicker than the heterogeneity.

## 1. General rules

- Upscale **parent properties first** (net flag, facies) and use them to bias later ones.
- Treat logs **as lines** (always for discrete logs, usually for continuous).
- Horizontal wells are not representative for statistics but strongly affect flow: upscale twice — vertical wells only (for data analysis) and all wells (for modelling). If vertical wells are too few, add horizontals that penetrate whole zones.
- Software reports log statistics over whole upscaled cells, so "porosity by facies" includes fractions of other facies. Use **facies-specific logs** (e.g. porosity defined only where facies = 1) for clean statistics.
- Always compare upscaled vs raw statistics, filtered by net and facies.

## 2. Default settings (test alternatives)

| Log | Type | Averaging | Bias to | Note |
|---|---|---|---|---|
| Net flag | discrete | most-of | none | compare upscaled vs raw statistics |
| Facies | discrete | most-of | none or net flag | compare upscaled vs raw statistics |
| Porosity | continuous | arithmetic | facies (or net flag) | |
| Horizontal permeability | continuous | arithmetic or geometric | facies | question whether to upscale at all |

Use vertical wells "through cell" and horizontal wells "simple"; normally no extra weighting. "Most-of" loses a little fidelity for discrete logs but is generally adequate.

## 3. Shoulder effects

- Cells straddling facies boundaries mix values: the best reservoir is pulled down, medium reservoir spreads wider, shale porosity is inflated. Watch for long low or high tails in upscaled histograms.
- Remedies, in increasing effectiveness:
  1. Bias to facies (works when beds are thick relative to layers).
  2. Facies-specific logs.
  3. Remove transitional samples at every facies boundary before upscaling ("cleaned" log).
- Worse in deviated and high-angle wells crossing boundaries obliquely — flag and edit there.
- Adjusting the layering also reduces shoulder effects in thin sands.

## 4. Permeability

- Preferably **do not upscale permeability**: derive it in the model from porosity (and facies).
- If upscaled: **arithmetic** for layer-cake reservoirs, **geometric** for discontinuous or disorganised ones.
- Vertical permeability is normally not upscaled (plug Kv is not representative at cell scale; one thin shale makes Kv ≈ 0). If it is: **harmonic** for layer-cake, geometric otherwise; unweighted by net.
- Permeability used in saturation-height functions: **geometric mean**.
- Upscaled permeability loses the tight plug poro-perm correlation (r ≈ 0.65–0.85 at cell scale) — that scatter is a scale effect, not plug scatter.

## 5. With a separate net input

- Whenever a separate net input (net flag or fractional net-to-gross) is used, weight porosity and permeability averages by it; weight saturation by net and porosity.
- When net/non-net is carried only by facies, block "model" porosity and permeability (already multiplied by net fraction) by plain arithmetic mean.
- In thin-bed (fractional net) workflows, logs set to *undefined* (not zero) where the net flag is 0 strongly reduce shoulder problems.

## 6. Other data

Point sets with attributes (e.g. deep azimuthal resistivity inversions) can be upscaled like logs to inform cells away from the well. Confidence falls with distance; delete unwanted points. Such data separate reservoir from shale well but are ambiguous for facies — upscale as net, then distribute rock types stochastically.

## 7. QC

1. Histograms of raw logs vs upscaled cells for facies and continuous properties, per zone and per zone/facies — distributions should match.
2. Well sections of upscaled vs raw logs; check the fluid transition zone.
3. Random-pick methods can legitimately disagree with log statistics — interpret with care.
4. Porosity per facies in the model matches the upscaled wells.

## In this project (Norne)

- Logs: ~46 Norne wellbores in LAS; the eight blind wellbores are excluded from any upscaling or statistics.
- Petrophysical properties are scored along blind-well paths against the reference cells (porosity MAE ≤ 0.02, log₁₀ k MAE ≤ 0.3, NTG MAE ≤ 0.05) — `docs/NORNE_BENCHMARK.md` §5.
