---
name: geomodel-petrophysics
description: "Use when interpreting or choosing petrophysical inputs for a geomodel: classic total-porosity vs thin-bed (Thomas-Stieber / fractional net) methods, rock categories, curve definitions, Archie and Waxman-Smits saturation, net and pay cut-offs, overburden and brine permeability corrections, saturation-height (J-function) fitting and QC, vertical permeability, and modelling paths. Primary user: the well agent."
---

# Petrophysics for geomodelling

**Used by:** well agent (owner); property agent (consumer); data steward (core-vs-log checks).

Goal: porosity, permeability, water saturation and shale volume that are geologically meaningful, internally consistent and fit for simulation.

## 1. Definitions

- **Free-water level (FWL):** depth where capillary pressure between oil (or gas) and water is zero.
- **Net:** rock porous and permeable enough to hold hydrocarbons above the FWL (gas and oil definitions may differ).
- **Net flag:** 1 for rock that would hold hydrocarbons above FWL; its average is net-to-gross (NG).
- **PV:** pore volume in net. **HCPV:** pore volume occupied by hydrocarbons initially. **STOIIP:** stock-tank oil initially in place.

## 2. Curves typically delivered

- Flags: bad hole, coal, gas, calcite-cemented, pay, net.
- Discrete: depositional facies; rock type (from poro-perm trends and/or mercury-injection data).
- Classic continuous: total porosity (usually density-based), total water saturation, shale volume, log permeability (brine, overburden-corrected; usually a poro-perm power law per facies).
- Thin-bed continuous: fractional net (FNTG), sand porosity, sand saturation, sand permeability, and "model" curves = sand value × FNTG.
- Legacy: effective porosity PHIE = PHIT − Vsh × PHIsh.

## 3. Choosing the method — rock categories

| Category | Rock | Method |
|---|---|---|
| A | Thick net and non-net bodies (e.g. shallow marine) | Classic: total porosity + log permeability + net flag. Preferred whenever possible; porosity and permeability **net-weighted** when upscaled |
| B | Low–medium heterolithic; non-net below log resolution inside net | Net flag misses it; porosity and permeability biased; because permeability is a log function of porosity, small porosity errors give large permeability errors and too-high modelled Sw. Classic may still work if mild; when in doubt use fractional net + sand porosity |
| C | Highly heterogeneous, thin-bedded | Must use fractional net / sand porosity / sand permeability; conventional methods fail (logs look like shale while core shows ~50 % good net). Load model curves directly, arithmetic upscaling, facies controls 3D distribution; apply a pore-volume cut-off to set very-low-HCPV cells to non-net |

- Only one log set per modelling path; hybrids per zone or facies are allowed and often best (classic for the best facies, thin-bed for the worst).
- The net flag must be synchronised with facies (0 in non-net facies).

## 4. Classic interpretation

- Total porosity from density; matrix density ≈ 2.645 g/cc for quartz, or from core grain density, or as a function of shale volume. Fluid density depends on flushed-zone saturation — iterate porosity and saturation to convergence.
- Shale volume preferably from density–neutron separation (better resolution than gamma ray) in oil or water zones.
- **Archie:** Swⁿ = (a · Rw) / (φᵐ · Rt).
- **Waxman–Smits** (when core data exist and formation water is moderately saline, ~30,000–90,000 ppm; accounts for conductive clay): 1/Rt = φ^m* · Sw^n* / (a · Rw) + φ^m* · B · Qv · Sw^(n*−1) / a.
- Net flag from porosity and shale-volume cut-offs (e.g. shale volume > 0.5 → non-net).

## 5. Thin-bed (Thomas–Stieber) interpretation

- Plot bulk total porosity against bulk shale volume inside the triangle defined by shale porosity and the cleanest expected sand; the position gives the sand fraction and sand porosity. Distinguishes laminated, dispersed and structural shale (laminated shale is less harmful than the same volume dispersed in pores).
- Gamma ray and deep resistivity smear thin beds most; density/neutron degrade below ~30 cm beds.
- **Pitfall:** sand fraction ≠ net. Apply cut-offs to convert sand fraction to fractional net (clay-clogged clean sand is not net; very dispersed sand in laminated rock is not net).
- Weight sand porosity by fractional net.

## 6. Corrections

- Correct all interpreted logs to downhole stress and match to a depth-shifted, overburden-corrected core dataset.
- Convert permeability from low-stress gas permeability to **brine permeability at reservoir conditions** — relative permeability end points and the J-function use brine permeability.
- Poro-perm transforms per facies or rock type, with their correlation coefficient. If a model-porosity-based permeability transform correlates below ~0.8, challenge the petrophysics.

## 7. Saturation height (J-function)

- Leverett J: **J = Pc / (σ · cos θ) · √(k / φ)**.
- In the model Pc = Δρ · g · h, with h = height above FWL: J = (Δρ · g · h) / (σ · cos θ) · √(k / φ).
- Fit **Sw = a · J^b** by regression of log Sw on log J.
- Convert laboratory Pc to the reservoir fluid system first; use reservoir-condition interfacial tension and contact angle.
- k and φ must be brine permeability and the matching porosity (classic: log permeability / total porosity).
- Normalised alternative: Sw* = (Sw − Swirr)/(1 − Swirr); Sw = Sw*(J) · (1 − Swirr) + Swirr, with Swirr a power function of brine permeability from special core analysis — include only plugs that truly reached irreducible saturation.
- **QC:** a banana- or S-shaped Sw–J plot cannot be used without major error. Compare modelled Sw with log Sw; plot Sw vs height above FWL coloured by permeability or facies.
- Permeability input to the saturation-height function: **geometric mean**.

## 8. Vertical permeability

Start from horizontal permeability × core Kv/Kh in net; it is plug-scale, so upscale harmonically and unweighted; large uncertainty; a team decision (see `geomodel-property-modelling` for Kv/Kh ranges).

## 9. Modelling paths (per zone)

1. **Classic:** model porosity; assign permeability and Sw by functions. Preferred when logs describe the reservoir well.
2. **Model porosity (porosity × net fraction):** less stochastic, but permeability is hard — needs a model-porosity transform or co-kriging.
3. **Fractional net + sand porosity:** keeps the permeability link, but fractional net must be distributed (use its trend if well averages follow the concept; otherwise short variograms or the mean).

Keep properties internally consistent along one path and involve the reservoir engineer.

## 10. QC

- Well section comparing interpreted logs with core — the single most effective display.
- Lithology QC: log lithology, core, spectroscopy and core-image facies in one plot; show a simpler rock-type model still matches poro-perm.
- Poro-perm and saturation-height functions plotted across the model extent.

## In this project (Norne)

- LAS logs exist for ~46 Norne wellbores (blind wellbores excluded from every fit); core and well reports are PDFs in `Geological_and_Petrophysical_Reports/` (INPUT unless they describe the reference model).
- The benchmark GIVES a saturation-height function per zone (`docs/NORNE_BENCHMARK.md` §7) for scoring volumes; the well agent's own saturation model is used for internal QC.
