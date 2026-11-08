---
name: geomodel-volumetrics
description: "Use when computing, reporting or QC-ing in-place volumes: the STOIIP/GIIP equation, fractional vs full-cell volumes and height-to-contact conventions, steep-layer errors, sanity checks, reporting by zone and segment, comparison with previous models, and ensemble volume analysis. Primary user: the contacts & volumetrics agent."
---

# Volumetrics

**Used by:** contacts & volumetrics agent (owner); structural framework agent (verifies closure, spill and compartments); well agent (verifies Sw).

## 1. Equations

- **STOIIP = GRV × NTG × φ × (1 − Sw) / Bo** (gas: GIIP with Bg).
- GRV comes from the framework (cell bulk volume). Net volume = GRV × NTG; PV = net volume × φ; HCPV = PV × (1 − Sw).
- The volumetric process uses **fractional cell volumes** for cells cut by a contact; simple property calculators do not restrict to above the contact.

## 2. Height to contact

- "Centre of the part of the cell above the contact" pairs with fractional GRV; "centre of the whole cell" pairs with full-cell GRV (common in simulators). Discuss, compare, and keep Sw consistent with the GRV convention — especially if the simulator takes Sw from the static model.

## 3. Steep layers and contacts

- On flanks and near contacts, steep layers distort volumes and Sw. Estimate the error with a copy using very fine horizontal or contact-parallel layers (0.25–0.5 m), compute Sw there and, optionally, upscale arithmetically to test reproducibility on the simulation grid.
- Large steep cells with high horizontal permeability can act as artificial conduits for water and gas.

## 4. Contacts

- Contacts from pressure gradients, log evidence and structural reasoning (spill points); give a base value and a range.
- Contacts must lie within closure and be consistent with spill points; compartments must be defined by the framework. Unresolved contact questions become **scenarios** (single vs multiple contacts), not a forced choice.

## 5. Sanity checks

- Compare with expectations and resource densities from earlier vintages; for new fields do an independent spreadsheet estimate.
- Compare static and dynamic in-place volumes (mandatory once a simulation model exists).
- Compare with the previous model revision: explain differences and make difference maps; show a **waterfall of the STOIIP change by component** (GRV, NTG, φ, Sw, Bo).

## 6. Reporting

- Volumes by zone, region/segment, depositional environment and facies; bulk, net, PV, HCPV, STOIIP tables per segment; pie charts of relative contributions.
- STOIIP and HCPV maps per zone and in total. **Do not plan wells on a single stochastic realisation's map.**
- Review **connected volumes** (volume reachable from wells), not only totals.

## 7. Ensemble volumes

- Box plots and histograms of STOIIP per ensemble (P10–P90, P50, min–max) with a statistics table; histograms reveal bimodality and skew.
- Check the prior range against independent volumetric or material-balance estimates.
- Analyse by zone and region as well as field — zone changes can cancel at field level.
- After history matching: a change of **more than about ±5 % of field volume** needs a technical explanation (e.g. part of the prior could not match history, or a pore-volume multiplier forced the match); combining several properties' extreme cases can create stacked, very unlikely outcomes. A healthy update: narrower posterior, similar centre.
- Checklist: distributions plausible; volumes per segment and zone make geological sense; changes explained by known uncertainties; both statistics and individual outliers checked.

## In this project (Norne)

- GIVEN (`docs/NORNE_BENCHMARK.md` §2.1, §7): five equilibration regions with their GOC and WOC; Bo per region (1.30–1.35); a saturation-height function per zone; volumes reported per zone × EQLNUM.
- Reference STOIIP ≈ **156.3 × 10⁶ Sm³**; Ile + Tofte in region 5 hold ~85 % of it. Structural sensitivity ≈ 0.6 % of STOIIP per metre of depth shift.
- Benchmark checks: pore-volume, HCPV and STOIIP error within ±5 %; ensemble calibration — does the true STOIIP fall inside the swarm's own P10–P90 (`docs/SWARM_ARCHITECTURE.md` §7.3).
- Compartments for the connectivity check: hydrocarbon cells linked through overlapping faces, including across faults (`docs/NORNE_BENCHMARK.md` §5.3).
