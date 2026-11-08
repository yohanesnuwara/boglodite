---
name: geomodel-facies-modelling
description: "Use when defining facies schemes, choosing variograms and facies algorithms (SIS, truncated Gaussian, object-based, MPS), hierarchical and net modelling, setting targets and trends, and QC-ing a facies model against the concept and wells. Primary user: the property agent."
---

# Facies modelling

**Used by:** property agent (owner); well agent (verifies against wells).

Facies bin petrophysical data into geological entities with comparable properties, so they can be distributed according to the concept.

## 1. Facies scheme

- Facies must be **resolvable by the grid**. Typically **2–5 facies per zone**: the sedimentologist's scheme, a simplification, or petrophysical rock types (grain size, sorting, diagenesis — may represent flow better than environment labels).
- If the core scheme is far finer than the grid, or has more than ~3–5 facies per zone, lump it into a compromise with sedimentologist and petrophysicist.
- Use poro-perm crossplots coloured by facies: at equal porosity, permeability can differ by an order of magnitude between facies — that is the reason to model facies at all.
- Rock typing: a hybrid of grain size and depositional energy as proxies for rock quality is usually best (pure sedimentological schemes have too many trends; pure property groupings lose predictive geology).
- Limit the count: many simultaneous facies create target and variogram conflicts. Minority facies that matter for flow (e.g. mudstone barriers) are better modelled as objects or in a separate step; a ~5 % facies distributed stochastically rarely affects flow and is often missed by the algorithm.
- **Targets:** do not copy vertical-well proportions blindly — correct for well-placement bias and for trends.

## 2. Variograms

- Horizontal ranges are much larger than vertical: vertical typically a few metres; horizontal facies ranges ~100–10,000 m. Lateral ranges are rarely measurable from wells — use analogues (lateral correlation is typically tens to hundreds of times the observed vertical thickness).
- Shape and orientation follow the environment (channel lags long along axis, short across; shorefaces long along the shoreline). Take orientation from regional understanding or seismic; complex patterns via digitised flow lines converted to azimuth properties.
- A variogram cannot exceed the body that contains it: in hierarchical modelling, ranges shrink at each lower level.
- Spherical is the most common type. Keep the **nugget near zero** for sharp boundaries (nugget is mostly cosmetic and limited by cell size). Sill = total variance.
- QC: test several types and ranges; inspect zone-average maps and arbitrary sections. Upscaled well cells must not be identifiable. If targets are missed, long variograms may conflict with wells — try shorter.

## 3. Algorithms

| Method | Strengths | Cautions |
|---|---|---|
| Sequential indicator simulation | Most used; honours proportions | Seed-sensitive; weak at shapes; the input range acts as a maximum, so set a larger range than the desired average; best with trend maps and vertical proportion curves |
| Truncated Gaussian (with trends) | Ordered facies (progradation, retrogradation), transitions, environments and facies associations | Controls which facies may touch |
| Object-based | Channels, lobes, ellipses; training images | Bimodal flow behaviour; strongly realisation-dependent; use as main method only with excellent analogues or dense wells |
| Multipoint statistics | Object-like shapes with proportion honouring | Needs a good 3D training image; usually impractical |
| Adaptive pluri-Gaussian | Used in ensemble/assisted history matching | Needs per-facies probability volumes and two latent fields |

Methods can be combined within one model.

## 4. Typical discrete workflow

1. Split gross rock into net vs non-net (background shale), with trend lines steering variograms (e.g. axis / off-axis / margin / background associations).
2. Inside net, populate sub-facies.
3. Add cemented facies last with short variograms so they do not control flow.
4. Populate petrophysics per facies; cement and shale set to zero.

Diagenetic facies (cement, fracture zones) cut across depositional boundaries and usually go last; check whether cement volume differs above and below the free-water level.

## 5. Hierarchical modelling

When one zone contains several large-scale environments: model environment → architectural element → facies or rock type within each. Parents may share facies with different proportions and variograms. Two levels are usually enough.

## 6. Net distribution

- Net-to-gross is usually best as **discrete 0/1** (captures flow), often derived from facies with explicit non-net facies; continuous NTG is acceptable only where non-net hardly affects flow.
- Net target maps: average probabilities consistent with the environment, honouring wells, smooth transitions between maps.
- Vertical trends must have the same average as the horizontal trends.

## 7. QC

- Targets honoured in the final grid — whole model **and** hydrocarbon column. Facies proportions plausible in every realisation.
- Concept visible on I, J, K and arbitrary sections; wells do not stand out; trends and variogram anisotropy visible. At least one strike and one dip section through wells.
- Per-zone facies fraction maps (summing to 1) with well fractions annotated; the concept is clearest in the extreme facies (non-net, best reservoir).
- Compare with steering inputs: concept map vs model, average maps vs guiding maps, vertical proportion curve vs realised proportions.
- Model-average NTG map plus a **net thickness** map (NTG alone says nothing about sand thickness).
- No artefacts: bull's-eyes, isolated patches, striping. Bull's-eyes or jumps near wells need a technical explanation.
- Ensemble: P10/P50/P90 facies maps reviewed; facies fractions per zone follow the expected distribution; connectivity and continuity realistic.

## In this project (Norne)

- Facies codes in the Norne realizations: **0, 1, 3** (legend still open — `docs/NORNE_BENCHMARK.md` §11). Only 10 distinct facies models exist among the 120 realizations.
- Benchmark check: per-zone facies proportions must fall inside the realizations' P10–P90 (mean shortfall ≤ 0.05). Cell-by-cell facies agreement is **not** scored, because the reference is one stochastic realization.
- Libraries: gstools, geostatspy, gstatsim (SIS/SGS); MPS deferred.
