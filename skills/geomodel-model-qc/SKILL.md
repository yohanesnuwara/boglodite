---
name: geomodel-model-qc
description: "Use when reviewing or verifying any part of a reservoir model: the master QC order, the 30-item static QC checklist, step-specific QC reminders, the pre-simulation sanity check, the minimum 'fit for decision' ensemble checklist, integrated static + reservoir-engineering QC, and rules for automated and AI-assisted QC. Primary users: every verifier agent and the orchestrator at gates."
---

# Model QC

**Used by:** all verifier agents (each takes its section); orchestrator (gate reviews).

QC is continuous, documented and reviewable. Every finding cites the check, the numbers and the evidence (plot, table, script).

## 1. Master QC order

1. **Pre-modelling:** concept, analogues, petrophysics, seismic input, dynamic input, data-readiness gate (`geomodel-data-readiness`, `geomodel-workflow`).
2. **Pre-simulation static:** structure → grid, layering and upscaling → 3D population → Kv/Kh → basic dynamic check → uncertainty range → multi-realisation → review displays → comparison with previous model.
3. **Simulation:** early dynamic interaction, fault transmissibility, parameter count (`geomodel-dynamic-handover`).
4. **Post-history-match ensemble:** parameter spread, 4D, concept preservation, ensemble analytics (`geomodel-uncertainty`).

## 2. Static QC checklist (30 items — before looking at individual realisations)

**Input data**
1. Wells, logs, core, seismic and surfaces are consistent and final.
2. Facies, rock types, saturation regions and zonation are aligned across disciplines.
3. Input uncertainty ranges are documented and justified.
4. Trends, variograms, correlations and conceptual assumptions are stored and traceable.

**Structure**
5. Tops and faults honour the seismic interpretation.
6. Realisations cover the intended structural uncertainty range.
7. Compartments and juxtapositions are geologically realistic.
8. Connected volumes agree with the structural concept.
9. GRV distributions are sensible.

**Facies**
10. Facies proportions reproduce well observations.
11. Facies fractions per zone follow the expected distributions.
12. Connectivity and continuity are geologically realistic.
13. No artefacts (bull's-eyes, isolated patches, striping).
14. P10/P50/P90 facies maps reviewed.

**Porosity**
15. Porosity distributions by facies match the upscaled wells.
16. Mean, P10, P50, P90 within expected ranges.
17. No unrealistic local anomalies.

**Permeability**
18. Poro-perm relationship preserved.
19. Permeability distributions cover the uncertainty range.
20. Kv/Kh distributions and trends are realistic.

**Volumetrics**
21. STOIIP/GIIP distributions are plausible.
22. Volumes per segment and zone make geological sense.
23. Volume changes are explained by known uncertainties.
24. Both ensemble statistics and individual outliers checked.

**Ensemble**
25. The ensemble spans the uncertainty space — neither too narrow nor too wide.
26. Scenarios are implemented and their impact is understood.
27. No duplicate or near-identical realisations.
28. Outliers identified and understood.
29. Parameter–result correlations make sense.
30. Input uncertainty is preserved after conditioning.

## 3. Step-specific reminders

| Step | Checks |
|---|---|
| Structure | No negative cell volumes in the area of interest; inside-out and cell-angle checks; test-simulate before property modelling; no bull's-eyes at wells; horizons checked along fault planes and against depth seismic |
| Zonation | Model isochores vs input and well thicknesses; map edges and sparse areas; intersection playback |
| Layering | Raw vs upscaled log statistics; intersection playback in all directions |
| Upscaling | Upscaled vs raw statistics per facies, filtered by net; shoulder-effect tails |
| Facies | Wells invisible in the model; targets met (model and hydrocarbon column); facies fraction maps with well annotations; trends visible |
| Properties | Output statistics each run incl. hydrocarbon column; kh vs log and well test; Sw vs logs and Sw–height plot; poro-perm plot |
| Saturation height | Sw–J plot shape (no banana or S shape) |
| Volumes | Spreadsheet check; previous vintages; static vs dynamic; difference maps |

## 4. Minimum review package (every gate)

Concept vs model sections · isochore, average-property and in-place maps per zone/region · segment volume tables (bulk, net, PV, HCPV, STOIIP) · model-vs-seismic sections at regular spacing · histograms and crossplots of model vs wells (whole model and hydrocarbon column) · key conceptual uncertainties captured · differences from the previous model quantified and explained · simulation at geogrid scale if upscaled · comparison with analogue fields · forward seismic modelling where fluids are visible.

## 5. Pre-simulation sanity check (static → dynamic handover)

- **Grid and structure:** no negative or twisted cells; faults and geometry honoured; resolution suits flow and runtime.
- **Properties and fluids:** porosity, NTG, Sw and PVT consistent and aligned with contacts and geology.
- **Consistency:** volumetrics, contacts and connectivity tell one story; plausible in simple visual or dynamic checks.
- Optional: streamlines (flow paths), connected-volume QC, well-test check (test kh vs logs, core and geology).

## 6. Minimum ensemble QC — "fit for decision"

1. Structural uncertainty covered.
2. Facies proportions honour data and concept.
3. Porosity and permeability distributions validated.
4. Connected volumes reviewed.
5. STOIIP/GIIP distributions validated.
6. All realisations run.
7. Prior envelopes cover the observations.
8. History-match updates remain geologically plausible.
9. No ensemble collapse.
10. Key uncertainty drivers identified and communicated.

## 7. Integrated static + reservoir-engineering QC (most important)

Static uncertainties should explain the dynamic uncertainty; dynamic updates must be geologically defensible; connected volumes, facies connectivity, fault transmissibility, aquifer strength and pressure communication must tell one consistent story; judge the whole ensemble, not a few members.

## 8. Automated and AI-assisted QC

- Script facies, porosity and permeability checks; dashboards for ensemble metrics; embed QC in the workflow.
- Standard QC package with every model; version iterations and assumptions.
- AI may summarise, compare versions, review images for inconsistencies, suggest checks and write scripts. **Every decision-relevant finding must be reproducible by a script or data check that runs outside the AI**, plus a verifier sign-off. Answers vary with phrasing; responsibility stays with the modeller and reviewer.

## In this project (Norne swarm)

- Verifier agents run these checks on INPUT data only; they never see the benchmark TRUTH (`docs/SWARM_ARCHITECTURE.md` §1, §5).
- A verdict = check name + numbers + evidence file + pass/fail; after two failed rounds the disagreement goes to the uncertainty ledger.
- The benchmark later measures **verifier quality**: whether rejections were justified by the true error and whether true errors were caught.
