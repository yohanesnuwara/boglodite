---
name: geomodel-gridding-layering
description: "Use when designing or QC-ing a 3D reservoir grid: model extent, orientation, lateral cell size, zonation and isochores (TVT/TST, proportional correction), layering methods and layer thickness, minimum cell thickness, thin flow-critical layers and grid handover checks. Primary user: the gridding agent."
---

# Gridding, zonation and layering

**Used by:** gridding agent (owner); fault agent and well agent (verifiers); property agent (consumer).

## 1. Extent, orientation, lateral resolution

- **Extent:** the maximum hydrocarbon accumulation plus enough surrounding geology to reproduce the aquifer response (agree with the reservoir engineer).
- **Orientation:** along the main flow direction, structural grain, depositional direction and injector–producer pattern; in steep structures align with dip.
- **Orthogonality:** use zig-zag faults where a fault cannot follow an I/J line; assign I or J trend to a few major faults; add trend lines if helpful.
- **Cell size** is a compromise:

| | Smaller cells | Larger cells |
|---|---|---|
| Gains | resolution, dip continuity, accurate flow | speed, simpler workflow |
| Costs | data volume, runtime | features averaged out |

- Run a sensitivity on lateral cell size and grid direction: displacement and numerical dispersion, areal size, near-well pressure (critical in low-permeability rock). Typical areal cells are around 200 × 200 m; vertical permeability ratios depend on cell size.
- Ideally static and dynamic grids share the same resolution. Test early versions for runtime and grid-related dynamic problems.

## 2. Zonation

- **Zones are flow units**; layering is defined per zone and **layers cannot cross zones**, so choose zones carefully. Boundaries reflect barriers, property or fluid changes, contacts, pressure, bedding dip, depositional setting or age.
- Zone surfaces: some from seismic, others from gridded well isochores; all tied to well tops.
- Flow units that cannot be explicit zones may be carried as zone properties ("pseudo zones"), at the cost of independent layering and statistics.
- **Isochores from well tops:** for deviated wells use true vertical thickness (TVT) or true stratigraphic thickness (TST) — needs dip and azimuth per zone and well. Include conceptual trends, onlap and truncation.
- If zones are evenly distributed, well tops alone can be enough (conformable algorithm).
- **Isochore closure:** stacking isochores from top or base rarely matches the seismic envelope. Default: **proportional correction** (error spread in proportion to zone thickness). With erosion at the top: build from the base without correction (the residual goes to the top zone).
- Intermediate horizons are also tied to tops, with a relatively small radius (< 1000 m).
- QC: compute isochores from the model and compare with input and with well thicknesses; inspect map edges and sparse-well areas; play intersections through the model.

## 3. Layering

- Goal: capture the main vertical heterogeneity at acceptable runtime.
- Methods:
  - **Proportional** — conformable to top and base.
  - **Follow top / follow base** with fixed cell thickness — handles onlap, truncation and pinch-out; adds layers and reduces orthogonality.
  - **Fractional** — varying resolution inside the zone (e.g. finer in the middle).
- Typical static layer thickness **0.5–2 m**; heterogeneous reservoirs need finer layers than homogeneous ones.
- **Rule from a vertical variogram:** compute the vertical variogram of a key property (e.g. porosity), read its range, set layer thickness **≤ half the range** (3 m range → ~1.5 m layers as a start).
- Flatter (contact-parallel) layering helps thin oil columns but can break up shales; layers can be coarser in the aquifer. Check layering suits saturation modelling (vertically refined test grids help).
- **Minimum cell thickness** option: collapse very thin cells to cut pinch-out cells, improve grid quality and runtime.
- Iterate: upscale facies and petrophysical logs, compare raw vs upscaled statistics, adjust layering until acceptable (see `geomodel-well-log-upscaling`).

## 4. Thin, flow-critical layers

High-permeability streaks (e.g. basal channel lags causing water under-running) and laterally extensive tight shales thinner than the grid must be represented explicitly (own zone or layers) **or** as transmissibility barriers — and must survive into the simulation grid.

## 5. Grid QC and handover

- Play intersections in all directions.
- Geometry properties: cell volume (no negatives inside the area of interest), inside-out cells, cell angles.
- Handover checks: no negative or twisted cells; faults honoured; zone thickness ≥ 0; no horizon crossing after gridding; resolution suits flow and runtime. Hand the grid to the reservoir engineer early and iterate.

## 6. Verification received

- Fault agent: throw and juxtaposition across grid fault faces equal the framework's.
- Well agent: zone tops at non-blind wells within tolerance; zone thickness ≥ 0; no horizon crossing.
- On failure: change increments, stair-step the fault, or re-layer the zone.

## In this project (Norne)

- Target grid GIVEN: corner-point **46 × 112 × 22**, absolute ED50/UTM32N coordinates, no map-axes transform, layer K → zone per the GIVEN zonation; delivered as `grid.grdecl` (COORD/ZCORN/ACTNUM) — `docs/NORNE_BENCHMARK.md` §5.0.
- Benchmark checks touched by this agent: zone-top MAE, fault throw MAE, zone thickness MAE (≤ 5 m), zone consistency (≥ 99 % of columns in order, no negative cells), pinch-out agreement (≥ 90 %), and static connectivity (compartments from face overlaps across faults).
- The fault-aware pillar gridder is the main build risk of milestone M2 (`docs/SWARM_ARCHITECTURE.md` §8).
