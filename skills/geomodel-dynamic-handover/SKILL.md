---
name: geomodel-dynamic-handover
description: "Use when preparing a static model for flow simulation or checking it there: PVT and SCAL review, initialisation (equilibration regions, saturation functions, end-point scaling, aquifer, wells, observed data), first-run verification, upscaling to a simulation grid, Kv/Kh and transmissibility multipliers, fault transmissibility via shale gouge ratio, and history-match ensemble checks including 4D. Primary user: the handover agent (deferred in Norne v1) and the volumetrics agent for static-vs-dynamic checks."
---

# Dynamic handover and simulation QC

**Used by:** handover agent (deferred for Norne v1); volumetrics agent (static vs dynamic in-place check); orchestrator (gate G5).

## 1. Fluids (PVT)

- Build from representative samples and laboratory work used to tune an equation of state; use analogue PVT when none exists.
- Black-oil when sufficient; tables must span the expected pressure range (extrapolation causes artefacts and convergence problems). Decide early: two- vs three-phase, black-oil vs compositional.
- Checks: samples uncontaminated; Bo/Bg, viscosity, Rs and compressibility benchmarked against analogues; lab data agree with pressure gradients; contacts validated with pressure, wells and structure; fluid + SCAL give realistic mobility ratios.

## 2. Special core analysis (SCAL)

- Review before building the dynamic model; plot on normalised saturation Sw* = (Sw − Swi)/(1 − Swi) to judge whether end-point scaling is appropriate.
- Scale relative permeability so 1 = permeability at 100 % brine at reservoir conditions (Krw(Sw = 1) = 1; Kro(Swi) may be < 1).
- Use primary-drainage capillary pressure to build a SCAL-based saturation-height function and compare with log Sw. Imbibition Pc is much sharper and can often be approximated as zero.
- Checks: samples represent rock types; curves fit environment and rock typing; Pc honours transition zones; end points physical; consistent with the petrophysical saturation model.
- Parameterise relative permeability with **continuous distributions on the underlying parameters** (e.g. LET per rock type) rather than picking from discrete curve sets.

## 3. Initialisation

- Components: grid with NTG, porosity, permeability; PVT with initial contacts and pressure; rock compressibility; saturation functions; aquifer; wells with completions; observed data. Keep it as simple and transparent as possible.
- Prefer **equilibration** with equilibration regions; use non-equilibrium 3D arrays only for tilted contacts and similar cases.
- Saturation functions: smooth (check derivatives), enough points (e.g. 21), enough significant digits. Saturation regions must rest on measured or analogue data.
- End-point scaling: per-cell Pc scaling to honour geomodel Sw (SWATINIT-type); critical-water scaling when imbibition Pc ≈ 0. Avoid over-complicated combinations; visualise results.
- Aquifer: analytical or numerical at the model edges; size and properties consistent with geology; strength benchmarked against pressure support.
- Wells: official trajectories, completions affecting the match, skin from tests.

## 4. First-run verification

1. No wells, a few years of equilibration: field and region pressure and saturations must stay stable.
2. With wells: realistic pressure limits; default numerics; a **minimum pore-volume cut-off** removes tiny cells.
3. Convergence problems: isolate (does it start with one well?), map problem cells, remove partly isolated cells.
4. In-place volume per region vs the static model (differences understood); Sw in all wells vs static model, logs and core.
5. The first history-match attempt on the geomodel is key feedback to the geomodeller; record every modification as a learning.

## 5. Upscaling to a simulation grid

- Prefer building the geomodel at simulation scale (saves a step and its QC).
- If needed: identical grid origin; coarse cells an integer multiple of fine cells. Rule of thumb: runtime grows roughly with the **square** of the active-cell factor; an efficient model is a few hundred thousand cells.
- Thin oil rims: contact-parallel grids, fine in the oil zone — at the cost of fault handling and thin tight layers that may open (restore with region transmissibility multipliers).
- Property averaging: NTG unweighted; porosity weighted by NTG; saturations weighted by porosity × NTG; permeability by flow-based or arithmetic–harmonic methods chosen by matching fine-grid behaviour.
- QC: visual fine vs coarse; upscaled k and φ vs core and geomodel in every well; per-well HCPV checks; **static vs dynamic STOIIP**.

## 6. Vertical communication and transmissibility

- PermZ = Kh × (Kv/Kh); effective simulation-scale Kv is lower than core suggests; model Kv/Kh as spatially correlated and facies-dependent; decide whether vertical multipliers are needed for extensive shales.
- Transmissibility multipliers represent faults and sub-grid barriers; they must be backed by geology or fault-seal analysis, calibrated to dynamic data and distributed spatially. **Red flag:** multipliers used to hide problems elsewhere.

### Fault transmissibility workflow (build scenarios — the impact is large)

1. 3D shale/clay volume property.
2. Throw per fault (face).
3. **Shale gouge ratio** per fault: SGR = Σ(Vshᵢ · Δzᵢ) / throw × 100 % over the beds that slipped past each point.
4. Fault permeability from a fault-perm vs SGR (or clay) relationship.
5. Transmissibility multiplier per fault.
6. Effective cross-fault transmissibility; test scenarios from sealing to open. A gas-bearing fault is not necessarily open.

## 7. History-match ensemble checks (summary)

- Simulation readiness: all members run, stable, acceptable runtime, sensible material balance.
- Observations sit inside the **prior** envelope generously in every vector (BHP, water cut, GOR, rates, RFT); compare pressure–depth profiles zone by zone.
- Posterior converges and narrows without collapse; updates stay geologically plausible; no compensation (wrong geology, right production).
- Relative-permeability envelopes per region stay inside optimistic/pessimistic bounds and remain smooth.
- 4D (when used): observed vs mean simulated maps for prior, production-only and production+4D posteriors; coverage map (fraction of members within 2σ of the tolerance); residual (sim − obs)/tolerance — ≤ 0.5 excellent, 0.5–1 acceptable, 1–2 moderate bias, > 2 investigate; expect objective-function convergence in 3–4 iterations.

## In this project (Norne)

- Deferred for v1 (`docs/SWARM_ARCHITECTURE.md` §2.1). The ATW 2013 Eclipse deck (PVT, SCAL, EQUIL, FAULTS, multipliers) is hidden TRUTH; agents must not read it.
- When this stage is added, an OPM Flow initialisation run can check the static connectivity metric and the static-vs-dynamic STOIIP.
