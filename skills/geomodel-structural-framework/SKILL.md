---
name: geomodel-structural-framework
description: "Use when building or QC-ing a structural framework: choosing pillar vs volume-based modelling, fault modelling rules, horizon modelling and horizon types, final well-top adjustment, ordered structural QC (3D, sections, seismic overlay, displacement, isochores, cell geometry) and structural/GRV uncertainty. Primary user: the structural framework agent."
---

# Structural framework

**Used by:** structural framework agent (owner); fault agent (verifies throw and truncations); gridding agent (consumer).

Often the most time-consuming step. The grid must be fine enough for the geological controls on flow but coarse enough for acceptable runtime — **test-simulate the framework early**.

## 1. Choose the approach

- Agree in a cross-disciplinary review **which interpreted faults to include** (depends on purpose, full-field vs sector).
- Faults without displacement on the surfaces are normally left out of the structural model and represented later as fault lines (barriers) in the simulation grid.
- **Corner-point (pillar) gridding is the default** — mature, with the best layering and workflow options.
- Consider volume-based modelling when faults are very numerous (e.g. > 100) or geometry is complex (many truncations, faults not cutting the whole column). Its limits: no depth conversion inside the tool, no structural-uncertainty workflow, and it must be converted to a corner-point grid with stair-step faults for simulation — do that conversion early, then zone and populate classically.

## 2. Fault modelling rules

- Keep fault geometry as simple as possible: **linear (two-point) pillars** usually suffice through the reservoir interval.
- Do not extend fault models far above or below the reservoir.
- Listric (three-point) and S-shaped (five-point) pillars work statically but can break the simulation grid; stair-step faults may be better in complex settings.
- Avoid **vertically truncating faults** (Y or λ junctions) in pillar grids: model the antithetic fault as stair-stepped or simplify the faulting.
- Make every fault connection a **clean cut**; adjust neighbouring pillar angles so pillars do not twist at junctions; inspect junctions in 3D from several angles.
- Faults that bound the model and carry no cross-flow can often be omitted from the pillar grid.

## 3. Horizon modelling

- Honour the geophysical input as closely as possible and QC against depth seismic. Reject overly messy input, especially horizon–fault relationships; small edits are easier after converting surfaces to points.
- Keep the fault cut-off distance (how far horizons are truncated/extrapolated from a fault) minimal with good interpretation; check both sides separately; QC horizons along fault planes in 3D.
- Large mismatches between fault model and horizons: edit pillars after consulting the interpreter; horizon–fault lines can improve the fit.
- Set the **horizon type** correctly: conformable, erosional, base (onlap/discontinuous). It is crucial for erosional unconformities; if the eroded surface and the unconformity have similar dip, restored horizons may be needed to get a clean truncation.
- **Final well-top adjustment** even if depth conversion already tied: influence radius ≈ 200 m for small residuals, otherwise about the average well spacing (an undefined radius is infinite). No well may show as a bull's-eye.
- The final framework is reviewed jointly by interpreter and modeller.

## 4. Ordered structural QC

1. **Pillar skeleton** (before layering): check top and base skeleton surfaces separately for folded cells; inspect fault tips, relay ramps, branch lines and intersections; pillars should follow the dominant structural trend. A skeleton can look fine in map view and be wrong in section — always check both.
2. **3D inspection:** rotate; switch zones off; compare with the original fault pattern and concept sketches; decide whether a defect comes from the data or the modelling.
3. **Cross-sections in at least two directions** (across and along faults): no spikes, no crossing horizons, consistent displacement.
4. **Seismic overlay:** faults and horizons line up with the seismic.
5. **Displacement:** throw on fault planes or juxtaposition diagrams; throw changes smoothly along strike unless truncated; watch for "displacement curling" that creates false juxtapositions; check throw–distance profiles.
6. **Isochores:** near-constant in a stable basin; thickening toward faults with syn-depositional faulting; any trend must fit the concept. Investigate abrupt changes.
7. **Cell geometry:** no negative-volume cells inside the area of interest, no inside-out (twisted) cells, acceptable cell angles. Orthogonality drives simulation accuracy and runtime.

Structural checklist items: tops and faults honour the interpretation; realisations cover the intended structural range; compartments and juxtapositions are geologically realistic; connected volumes agree with the structural concept; GRV distributions are sensible.

## 5. Structural and GRV uncertainty

- Sources: interpretation, depth conversion, isochores, contacts. In a depth-built model, interpretation and depth-conversion uncertainty can be combined.
- Approaches: discrete scenarios, a stochastic workflow, or both.
- Stochastic setup: per horizon, a depth (and isochore) standard deviation — constant or mapped — and a common perturbation variogram. **Keep that variogram short (< 500 m)**: a long one combined with well adjustment cancels the intended uncertainty.
- Contacts: a range around the base value from pressure data, petrophysics and structural reasoning.
- Run modes: full uncertainty (everything varies together) vs one-at-a-time sensitivity (loops = n × variables) for tornado plots of what drives bulk rock volume.
- Store variables and results, not grids; regenerate any case on demand.

## 6. Verification tests

- On volumetrics: contacts lie within closure and above spill; compartments match the framework.
- Verified by the fault agent: framework throw equals interpreted throw; truncation rules consistent with timing and throw.
- Verified by the well agent: tops never invert stratigraphic order; isochores non-negative; conformable horizons do not cross.

## In this project (Norne)

- Zonation GIVEN: Garn (K1–3), Not (K4, inactive shale), Ile (K5–11), Tofte (K12–18), Tilje (K19–22).
- Benchmark structural checks (`docs/NORNE_BENCHMARK.md` §5): zone-top surface MAE ≤ 10 m, blind-well pick MAE ≤ 10 m, **fault throw MAE ≤ 10 m** (missing or invented faults count their full throw).
- Libraries: LoopStructural or GemPy for implicit horizon/fault modelling; the corner-point pillar grid is built by the gridding agent from this framework.
