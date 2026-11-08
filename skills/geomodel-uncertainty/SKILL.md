---
name: geomodel-uncertainty
description: "Use when framing, running or QC-ing uncertainty: concept scenarios vs parameter ranges, sensitivity and tornado analysis, stress cases, structural/GRV uncertainty setup, multi-realisation QC and discard rules, and post-history-match ensemble diagnostics (parameter spread, scalar updates, ranking, incremental analysis). Primary users: the orchestrator (uncertainty ledger) and the volumetrics agent."
---

# Uncertainty and ensembles

**Used by:** orchestrator (owns the uncertainty ledger); volumetrics agent; property and structural agents for their own ranges.

## 1. Framing

- Start from the **decision** the model supports and a geological framing of the problem — not from arbitrary ranges.
- The main uncertainties are usually **conceptual**: structure, depositional architecture, connectivity, rock properties. Represent alternative concepts as **separate scenarios**, with parameter variability inside each.
- Classic failure: varying parameters only around one preferred concept → over-optimistic P50 and too narrow a P10–P90. Multiple concepts widen the range and shift P50; many projects fall outside their own range because concept uncertainty was missing.
- Each discipline maps its data uncertainty and interpretation scenarios objectively; rank uncertainties by impact during the test phase; then carry the important ones. Challenge "unknown unknowns".
- Analogues constrain structure, architecture, connectivity, NTG, porosity, permeability and contacts.

## 2. Analysis tools

- **Sensitivity / tornado / Pareto** plots rank what drives STOIIP, reserves and forecasts.
- **Histograms** for volume ranges; **crossplots** for parameter correlations.
- **Stress (extreme) cases** are not truth but bound plausible outcomes and test pre-identified risks (e.g. extreme barrier variograms, no vs strong aquifer); they can reveal contingencies or more robust well paths.
- Fix or variable-drive **random seeds** so sensitivities are reproducible.
- Ensembles: start small, grow with maturity.

## 3. Structural / GRV uncertainty (setup)

Per horizon: depth and isochore standard deviation (constant or mapped) with a short common perturbation variogram (< 500 m); contact ranges from pressure, petrophysics and structure; run either "all at once" (total range) or "one at a time" (tornado). Store variables, not grids.

## 4. Multi-realisation QC (static)

- An ensemble needs the same QC as a single model. Quick: parameter overview, bull's-eyes at wells, consistency with the framework.
- Detailed:
  - every parameter sampled across its full range, no clustering or gaps;
  - GRV, NTG, φ, k, HCPV and STOIIP distributions reviewed;
  - variability without geologically unrealistic members;
  - low/mid/high cases and outliers consistent with the concept;
  - spread covers volume, connectivity, quality and flow;
  - extreme cases compared at well level.
- **Discard rule:** discard only realisations clearly outside the stated bounds (range 100–500 m channel width: keep ~100 m cases; a 10 m case means an input error — find it). Remove members that artificially stretch or squeeze the range. **No duplicate or near-identical realisations.**
- Ensemble checklist: spans the uncertainty space (not too narrow, not too wide); scenarios implemented and their impact understood; outliers understood; parameter–result correlations make sense; input uncertainty preserved after conditioning.

## 5. Parameters over project life

Use sensitivity analysis to find what matters; track the number, ranges and impact of uncertain parameters over successive studies to show learning and uncertainty reduction.

## 6. After history matching (ensemble diagnostics)

- **Parameter spread:** data-informed parameters narrow; weakly informed ones stay wide; means shift where a systematic update was needed. Over-collapse = conditioning too strong, poor prior or too few members. Too little narrowing = weak data sensitivity or poor parameterisation. Always judge with fit metrics.
- **Scalar updates:** prior vs posterior histogram per parameter. Moderate shift = acceptable; **shift to the edge of the prior** = missing physics, incomplete parameterisation or too-narrow prior; strong narrowing = collapse. Combine with simulated-vs-observed curves and parameter-vs-response crossplots.
- **Sensitivity before the update:** correlate parameters with objectives over time to find drivers, non-influential parameters (remove), conflicts (one parameter pushing two objectives opposite ways) and spurious correlations (localise).
- **Spatial changes:** compute deltas **per matched realisation pair and per cell**, then statistics of the deltas (differencing ensemble means hides cancelling changes); use log space for permeability; normalise by the prior range. Changes should be localised near data and geologically explainable.
- **Ranking:** distance per member d = √Σₜ(simₜ − obsₜ)² at field and well level; combined rank = field rank + well rank. Forecast with the whole ensemble; if three cases are unavoidable, select them transparently near P10/P50/P90 and state how the three-case range differs from the full range. Statistical P10/P50/P90 maps are smoother or more extreme than any real member — pick physical realisations near those percentiles.
- **Incremental analysis:** increment = alternative − base **per matched realisation pair**; rank options on expected value and downside. If an infill well's own production far exceeds the field increment, the difference is acceleration, not new recovery.
- **Principle:** a lower misfit only shows that conditioning happened. Accept a match only with a physical explanation; a case is usable for forecasting only if it is both dynamically acceptable **and** geologically credible.

## In this project (Norne)

- Unresolved producer/verifier disagreements become named scenarios in the **uncertainty ledger** (`docs/SWARM_ARCHITECTURE.md` §4).
- The base case is scored with the benchmark metrics; the ensemble is scored for **calibration** — does the true STOIIP (≈ 156.3 × 10⁶ Sm³) fall inside the swarm's own P10–P90.
- The 120 Norne realizations share one structural grid and contain only 10 distinct facies models: they constrain facies proportions, not structural or volume uncertainty.
