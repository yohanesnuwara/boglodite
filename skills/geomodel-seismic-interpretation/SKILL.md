---
name: geomodel-seismic-interpretation
description: "Use when interpreting or QC-ing seismic horizons and faults for a reservoir model: interpretation order, fault confidence categories, horizon–fault consistency, ordered interpretation QC, plausibility rules such as throw profiles and fault length vs throw, and 4D input QC. Primary users: the horizon agent and the fault agent."
---

# Seismic horizon and fault interpretation

**Used by:** horizon agent, fault agent (owners); structural framework agent (consumer).

Horizons and faults are the main input to the structural framework. **Interpret them together**, not independently, and fix problems during interpretation rather than during modelling.

## 1. Principles

- Detail must match the model's purpose and cell size: faults picked every 50 m cannot be honoured in a 200 m grid.
- Main pitfall: the modeller uses horizons and faults without understanding their basis, simplifications and uncertainty. The modeller should interpret part of the data itself, especially difficult areas.
- If depth seismic is interpreted, apply a depth-to-depth well-tie correction before modelling and keep a corrected volume for final QC.
- Seismic may also inform reservoir and fluid distribution (amplitude, inversion) — see `geomodel-property-modelling` for when that is allowed.

## 2. Interpretation order and display

- Display: high-contrast colour bar, minimal vertical exaggeration (ideally 1:1), arbitrary lines perpendicular to strike, several orientations, work in 3D.
- **Faults first, then horizons.** Auto-track horizons with "stop at visible faults", then extend manually up to the faults.
- Spend time on **branch lines** (fault–fault intersections); use horizon attributes to find them.
- Iterate: build a quick structural framework from the picks, look for busts, fix, rebuild.

## 3. Fault confidence categories (record in the fault name)

| Category | Meaning |
|---|---|
| 1 | Offset of both top and base reservoir is certain |
| 2 | Offset of top **or** base is uncertain |
| 3 | Both offsets uncertain |

- Use a consistent naming scheme; prefix model faults with the model name.
- Large faults with clear offset go into the grid. Small faults and discontinuities are kept as a separate set for possible use as simulation barriers (e.g. if needed in history matching).
- Faults are straightened in modelling, so straighten them during interpretation: **planar faults within a thin reservoir interval** are adequate and the simplification is better checked in the seismic domain.

## 4. Fault plausibility checks

- Throw sense consistent (normal or reverse) along each fault.
- Throw–distance profiles vary smoothly and die out toward the tips; interacting faults show linked profiles.
- **Fault length ≈ 20 × maximum throw** as a rough plausibility check.
- Inspect cut-off lines and heave polygons; investigate abrupt isochore changes across faults.
- A stick must coincide with a coherence or curvature anomaly along most of its length, and must not run parallel to the acquisition footprint (possible artefact).
- Fault-position uncertainty is hard to represent except through alternative grids; it matters for wells and volumes.

## 5. Horizon QC and horizon–fault consistency

- Typical problem: amplitudes change near faults, so horizons stop short of them. Fix by merging fault cut-off points with the picks and gridding to fill the gap — in the seismic domain.
- No horizon pick may cross a fault without an offset consistent with that fault's throw.
- Small throws need not be honoured exactly in the surfaces (the framework handles them), but know the uncertainty near faults; a hanging-wall surface extrapolated to the fault can be requested.
- Remove systematic noise (e.g. acquisition footprint) before depth conversion; horizons will be smoothed anyway.
- Tuning thickens thin intervals; correct it when it matters (e.g. with a seismic net-pay method).

### Ordered interpretation QC
1. QC the **raw, ungridded picks** first; use gridded surfaces only afterwards.
2. Review faults and horizons together in 3D for misalignment and gaps.
3. 2D restoration and flattening to check thickness correlation and structural consistency.
4. Curvature attributes (e.g. maximum curvature) for subtle fault trends.
5. Surface tools — isochore, anomaly, gradient, edge detection, boundary polygons — to catch picking errors.

## 6. Output to the framework

- Deliver fault sticks and a fault framework; both are QC'd in 3D with depth seismic.
- Convert framework surfaces and faults to regular grids for depth conversion, volumetrics and modelling.
- Minimum horizon input: the interpreted surfaces closest to top and base reservoir.
- The final model's surfaces must be checked against the seismic by the interpreter.

## 7. 4D seismic input (when used)

- Interpreter and reservoir engineer must agree on what the 4D response means and how uncertain it is.
- Three domains: reservoir (pressure/saturation change), rock physics (elastic), seismic (amplitude). Compare in one common domain for matching; cross-domain comparison is qualitative only.
- Quick QC: observed 4D attribute maps beside simulated pressure- and saturation-change maps for the same interval.
- Pitfall: a 4D signal dominated by gas coming out of solution while the simulator ignores critical gas saturation.

## In this project (Norne)

- Seismic: 4D vintages 2001, 2003, 2004, 2006; IL 970–1290, XL 1300–2300, 4 ms. The 2001 vintage is the proposed structural input.
- Reference interpretation (hidden TRUTH, never available to agents): 2006 fault sticks and the 2006 Not 2 Fm top. The benchmark scores faults by fault-level and point-level F1 (100 m) and the horizon by time MAE (≤ 15 ms) — `docs/NORNE_BENCHMARK.md` §5.
- 62 reference faults cross the active reservoir; some 2006 faults sit 180–265 m from the reference grid's faults, so interpretation and grid do not fully agree even in the truth.
