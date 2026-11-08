---
name: geomodel-data-readiness
description: "Use when loading, auditing or accepting input data for a reservoir model: coordinate systems and conventions, required-data lists, consistency checks between disciplines, petrophysical data hierarchy, core-vs-log QC, well representativeness, fluid contacts from pressure data. Primary user: the data-steward agent (which verifies every other agent's objects)."
---

# Data readiness and input QC

**Used by:** data steward (owner and universal verifier); well agent for core/log checks.

Late or inconsistent input is the most common cause of delay and of models that must be rebuilt. Fix problems **in the input**, not during modelling.

## 1. Project conventions (check before anything else)

- **Coordinate reference system** identical across all disciplines and objects.
- Units, datums (seismic reference datum, mean sea level), **time vs depth** tagging, depth sign convention (TVDSS positive down), grid index order.
- These are never converted automatically by modelling software: a wrong tag produces a valid-looking object in the wrong place (a *silent failure*).
- Required-data list for static and dynamic input; all items stored in the agreed format; a **data cut-off date** recorded.

## 2. Consistency checks between disciplines

- Sedimentological facies logs vs petrophysical logs.
- Zone tops vs seismic horizons; time thickness between horizons plausible; misties removed.
- Net flag vs facies in every well: the net flag must be 0 in every non-net facies.
- Well picks: on a correlation panel (one consistent picking strategy?), in a table, and through per-zone isochore maps looking for anomalies.
- Facies, rock types, saturation regions and zonation **aligned across disciplines**.
- Input uncertainty ranges documented and justified; trends, variograms and conceptual assumptions stored and traceable.
- Data measured at very different scales can disagree; resolving that is part of preparation, not of modelling.

## 3. Petrophysical data hierarchy (most → least trusted)

1. **Conventional core** — overburden-corrected porosity, permeability, saturation, grain density. Direct, fewest assumptions, same reference as special core analysis (SCAL). Weak in poorly consolidated rock.
2. **Modern wireline logs** — near in-situ and continuous, but model-dependent.
3. **SCAL results** (saturation and cementation exponents) — controlled but often biased (dead oil, oxidation, non-reservoir conditions).
4. **Modern LWD logs** — continuous; noisier sensors.
5. **Vintage wireline** (pre-1990) — more borehole effects, often stored uncorrected; stronger shoulder and thin-bed effects.
6. **Cuttings and mud gas** — poor depth control; qualitative only.

Consequence: model input logs must broadly agree with core where core exists.

## 4. Core vs log (first data QC)

- Crossplot core against log-derived properties; calibrate logs to core.
- Core porosity must be **overburden-corrected**.
- Poro-depth and poro-perm must plot consistently; if not, decide whether it is a data error or a real effect (diagenesis, cement).
- Plot routine and special core analysis poro-perm together; check plugs are not biased to the best sand.
- The poro-perm trend should agree with overburden-corrected data and with mobility from pressure tests.
- Display: well section with interpreted logs and core points.

## 5. Representativeness and statistics

- Wells are often placed in favourable spots (strong amplitudes, crests): their statistics over-represent good rock.
- **Horizontal wells** are not representative for statistics (they sample few layers). Exclude them from data analysis; keep them for conditioning.
- Log statistics are sample-count weighted: denser sampling biases them. Resample to a fixed interval if needed.
- Bad-hole intervals: usually non-net; decide explicitly.
- Compaction-sensitive reservoirs: consider restoring properties to initial (time-zero) conditions so depletion is not mistaken for a spatial trend.

## 6. Fluids and contacts

- Pressure–depth plots constrain the free-water level; compare gradients with laboratory fluid densities.
- Validate oil–water, gas–oil contacts and free-water level against pressure, well observations and structure; document the uncertainty.

## 7. Verification role (data steward)

Reject any object whose CRS, unit, domain (time/depth), index order or sign convention disagrees with the project, stating the mismatch. Typical traps:
- coordinates stored with a map-axes transform (e.g. Y negated);
- grids exported with a reversed index axis;
- a time surface whose values look like metres;
- depth sign conventions (negative TVDSS);
- mixed pick generations or definitions for the "same" marker;
- truncated or partial files.

## In this project (Norne)

The Norne data contain each of the traps above (`docs/NORNE_BENCHMARK.md` §10–§11):
- realization grids R2–R120 have **Y negated and J reversed** relative to the reference ordering;
- the "Not 2 Fm … input_geomodel" horizon is **two-way time** despite depth-like values;
- the picks file stores TVDSS as **negative** numbers; use the 2002 pick generation (the 2006 "Not" top is a different definition);
- `Top_Not_2_2003_Time.txt` is **partial** (IL 821–992 only);
- the CSV picks export is malformed — use the tab-separated file.
CRS for all objects: ED50 / UTM zone 32N. Depth: TVDSS metres, positive down. Time: TWT ms.
