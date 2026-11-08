---
name: geomodel-workflow
description: "Use when planning, sequencing or reviewing a reservoir static-modelling study: model objectives, conceptual model, model design, the ready-for-modelling gate, scripted workflows, review displays, comparison with the previous model and documentation. Primary user: the orchestrator agent."
---

# Reservoir modelling workflow and planning

**Used by:** orchestrator (owner); every agent when it needs to know where its task sits in the study.

## Philosophy

- Build the **simplest model that meets the objectives** — "as simple as possible, but no simpler". Keep in mind whether it may later be needed for harder questions.
- Models exist mostly to predict **fluid flow**. Required detail grows with heterogeneity and with lower fluid mobility: viscous or heavy oil needs far more heterogeneity than a permeable gas reservoir, which is mostly structure-controlled.
- Three phases:
  1. **Planning and preparation** — objectives, concept, input QC, data analysis, learnings from earlier models, design tests, written plan, ready-for-modelling review.
  2. **Execution** — structural framework and grid → property modelling → volumetrics → (optional) upscaling to a simulation grid → initialisation and verification in the simulator → final QC.
  3. **Documentation** — any model used at a decision gate is archived with a written report.

## 1. Objectives (write them down first)

- The business decision the model supports.
- The modelling objectives, and **explicitly what the model must not be used for**.
- Scope follows objective: a fine sector model may answer an infill question better than a full-field model.

## 2. Conceptual model — the most critical step

- A wrong or poorly integrated concept is the **largest single cause of wrong in-place and recoverable volumes**, most often by missing the full range of scenarios the data allow.
- Mitigations: one geoscientist owns the integrated overview (core, sedimentology vs seismic, raw vs interpreted logs, flow implications); every discipline owns the concept and its uncertainties; invite outside challenge.
- Inputs integrated into the concept: biostratigraphy, core sedimentology, analogues and regional studies, petrography, geochemistry and PVT, core analysis and petrophysics, early static models and early simulation feedback.
- **Deliverables before property modelling** (the concept is a reviewed artefact, not an idea):
  - tectonic event chart
  - reservoir zonation and facies logs
  - isochore, facies-distribution and property (NTG, porosity) maps per zone
  - description of proven and potential barriers and baffles
  - compartment maps and cross-sections
  - uncertainty description per element, plus alternative concepts
  - field-scale sketches (maps and sections) shared by the team
- Dynamic data constrain the concept: production, DSTs, formation-pressure surveys, PLT, tracers, produced-fluid variation, 4D. They inform kh, property trends, zonation, distance to barriers (DST), minimum connected volume (DST), minimum produced volume, swept area (4D).

### Analogues
Choose analogues that match on lithology (rock type, porosity, permeability, grain size), structure, stratigraphy and sedimentology, processes (tectonics, deposition, erosion, diagenesis), temporal and spatial scale, and **fluid type**. Use them for property screening (poro-depth, poro-perm), fault-seal behaviour, and as a reality check on recovery factor.

## 3. Learning from previous models

Record: history-match learnings (e.g. compartments), why a new model is needed instead of an update, what worked and what did not.

## 4. Model design (kick-off with all disciplines and the model users)

Decide and document:
- Area of interest and how much aquifer to include.
- Geological elements that control flow and front movement (depends on objective, geology, viscosity).
- Zones hard-wired in the framework vs "pseudo zones" carried as a property.
- Which faults: segment-bounding and flow-affecting ones. Could a simpler unfaulted grid work? Model-bounding faults without cross-flow can often be left out of the pillar grid.
- How the framework will be built given seismic quality.
- Whether facies modelling is needed — **zone by zone**, from data showing petrophysical contrast between facies.
- Thin thief zones or barriers that must be represented.
- Fit-for-purpose resolution: smallest feature to capture vs runtime; build the static model at simulation scale whenever possible.
- Petrophysical approach: classic bulk input by default; thin-bed (fractional NTG) input only with a clear reason (see `geomodel-petrophysics`).
- Whether seismic can condition properties — only after the correlation is demonstrated.
- Key uncertainties to test and how; whether an ensemble is needed (start small, grow it).
- How the model will be updated as data arrive.

## 5. Plan

- Timeline of data availability and subtasks; parallelise (early tests on preliminary frameworks).
- **All technology tested before modelling starts.** New code during modelling is schedule and quality risk; if accepted, add contingency and a fallback.
- **Data cut-off date.** Late data are a common cause of delay; include them only after a risk-based decision.
- Work in short increments ("sprints") with a defined deliverable each, inside a test phase; build test models to learn what works.

## 6. Ready-for-modelling gate (G1)

A short formal review. If items are open, weigh the cost of delay against the risk of proceeding.

| Area | Must be true |
|---|---|
| Input | Objectives documented; data list complete; all data available, QC'd, stored; dynamic input documented; previous learnings documented; **coordinate system verified**; cut-off date set |
| Concept | Documented; correlation panels, maps and sections available |
| Design and plan | Design meeting held; key elements, structural plan, property plan, uncertainty plan, update plan and schedule documented |

"Ready" means data are **technically validated** (QC'd, depth-matched, consistent), **geologically coherent** (fit the structural and depositional frameworks) and **traceable** (versioned, shared). Keep data and interpretation separate; make uncertainty explicit — data can support several interpretations.

## 7. Workflows

- Build every model through a **scripted, commented workflow**. It is part of the documentation, gives repeatability and makes updates fast. Deliver it with the model.
- Fix random **seeds** explicitly (or drive them from the uncertainty workflow) so the base case is reproducible.

## 8. Early dynamic interaction

Send early versions to simulation for quick runs (stability, runtime, basic flow). Feed results back into cell size, orientation and the uncertainties to carry.

## 9. Review displays (for every gate)

- Concept sketches beside static and dynamic model sections.
- Well sections: model logs vs measured logs for a near-P50 realisation plus P10 and P90.
- Per-zone isochore, average-property and in-place maps (mean, P10, P50, P90).
- Segment volume tables: bulk, net, pore volume, HCPV, STOIIP.
- Model-vs-seismic sections at regular spacing.
- Histograms and crossplots of model vs wells, for the whole model **and the hydrocarbon column**.
- Evidence that the key conceptual uncertainties are captured.
- Comparison with the previous model, differences quantified and explained.

## 10. Comparison with the previous model

Always compare STOIIP and its components with the previous build — even a justified rebuild can hide a large unexplained change. Show a **waterfall of the change by component** (GRV, NTG, porosity, Sw, formation volume factor).

## 11. Documentation (report contents)

Objectives · input database · design referencing objectives and concept · model description (framework and grid; properties with conditioning, targets, variograms; volumetrics; upscaling; dynamic initialisation and verification) · QC · learnings. Archived workflows must run. Give the model a short name and prefix all its input objects with it.

## 12. Use of AI in QC

AI may summarise documents, compare versions, review maps and sections, suggest checks and write scripts. Every decision-relevant finding must be **independently reproducible**: attach a script or data check that runs in approved software, plus a verifier sign-off. AI supports judgement; responsibility stays with the modeller and reviewer.

## In this project

- Gates G1–G5 and the decision-card policy are defined in `docs/SWARM_ARCHITECTURE.md` §4. The G1 table above is the orchestrator's checklist for G1.
- Objectives, zonation (Garn, Not, Ile, Tofte, Tilje), contacts and target grid dimensions are GIVEN by the benchmark (`docs/NORNE_BENCHMARK.md` §2.1, §3).
- Every published object records its tool, parameters and seed in the project store — this is the "scripted workflow" requirement.
