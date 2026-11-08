---
name: geomodel-well-tie-depth-conversion
description: "Use when tying seismic to wells, choosing wavelets and seismic tops, building velocity models and depth-converting horizons and faults, or estimating depth uncertainty. Primary user: the well agent (tie, velocity and depth conversion)."
---

# Well ties and depth conversion

**Used by:** well agent (owner); horizon agent (verifies picks against tops); structural framework agent (consumer of depth surfaces).

## 1. Well ties

- Tie every mapped horizon to wells to establish its **geological meaning**.
- Data choice: near-offset data often tie best when quality is good; full-offset data usually have better signal-to-noise. Where AVO effects are expected, build synthetics from extended elastic impedance at the angle that matches the stack used.
- Wavelet: extract statistically for QC, then deterministically from the wells to decide which geological pick matches the mapped event.
- Where AVO matters, fluid-substitute the logs to see how the tie changes with and without hydrocarbons (polarity can flip downflank); substitute porosity too if reservoir quality varies.
- Polarity must be consistent with the impedance contrast at the tie.

## 2. Seismic tops vs geological tops

- A mapped seismic event often does not coincide with the geological surface (tuning, weak impedance contrast).
- Identify the seismic pick in the well logs and communicate it, so interpreter and modeller know what each surface represents.
- If the acoustic event is far from the reservoir top (e.g. an underlying coal), depth-convert the acoustic event and shift to the geological top in the geomodel; if the difference is small, correct during depth conversion.

## 3. Velocity model and depth conversion

- Depth conversion is owned by the geophysicist and done with the team so everyone uses the same surfaces.
- For development decisions, **test at least three velocity models**: wells only; seismic velocities only; combined.
- Typical simple forms: layer-cake interval velocities, or V0 + kZ per zone; calibrate to checkshots and time–depth tables.
- QC: depth-thickness maps between depth surfaces; residuals at every tie well; store the velocity model under the model name.
- If the model supports well delivery, depth-convert the overburden horizons accurately too.
- Depth conversion inside the modelling tool (time model first): tight link to interpretation, easy updates, velocity uncertainty can enter the uncertainty workflow; but it can distort pillar geometry, rules out volume-based frameworks, and works only for pillar or simple unfaulted grids.

## 4. Depth uncertainty

- Typical depth-conversion uncertainty ≈ **1–2 standard deviations of the raw well-tie residuals**.
- Alternative surfaces (alternative picks, alternative velocity models) show the uncertainty and can drive alternative structural scenarios.
- Communicate the uncertainty explicitly to the framework and volumetrics agents.

## 5. Final well-top adjustment (hand-off to the framework)

Even after depth conversion ties to tops, the framework makes a final exact tie: small influence radius (≈ 200 m) where residuals are small, otherwise about the average well spacing; check for bull's-eyes at wells.

## 6. Verification tests this agent runs on others

- Horizon picks: pick-to-top residual at every non-blind well after time–depth conversion within tolerance; polarity consistent with log reflectivity. If one horizon fails, check neighbouring horizons to decide whether the pick, the tie or the top is wrong.

## In this project (Norne)

- Checkshots exist for exploration wells **6608/10-2 and 6608/10-4**; production wells have time–depth tables; interval-velocity models exist but are TRUTH (hidden).
- The official depth OWC surfaces have time twins — a free calibration check for any velocity model (compare converted time twins with official depths), allowed only on non-scored surfaces.
- Blind wells (B-2, D-3, D-4, F-3 families) must never be used for ties, velocity fitting or residual checks — they are the benchmark's depth test (blind-well pick MAE ≤ 10 m).
- The reference grid itself misses the blind picks by ≈ 5–7 m RMS: that is the realistic floor.
