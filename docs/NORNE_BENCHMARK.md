# Norne Static-Modelling Benchmark — Specification (draft v0.2)

Status: **draft** · Owner: Boglodite · Created 2026-11-06 · v0.2 adds baseline results (§10) and corrections from the baseline run

This document defines how the Boglodite geomodelling swarm is evaluated against
the Norne field dataset. It fixes *what agents may see*, *what they are scored
against*, *how each stage is scored*, and *how the whole swarm is scored*. It is
the contract that every agent, tool, and experiment must respect. Nothing here
depends on a particular LLM, framework, or agent topology — those are the
variables under test.

Code: `norne_benchmark/` (readers in `io.py`, baseline in `baseline.py`).
Viewer: `sandbox/norne_viewer.py` → `tmp/norne_3d.html`.

---

## 1. Goal

Given raw subsurface data for Norne (seismic, wells), an agentic system must
build a reservoir static model — faults, horizons, velocity model, corner-point
grid, properties, volumetrics — and that model is scored by its closeness to the
Norne reference model. The benchmark answers three questions:

1. **Stage skill** — how good is each specialist agent when fed correct inputs?
2. **System skill** — how good is the full swarm end-to-end, and where does
   error propagate or get corrected by feedback loops?
3. **Process efficiency** — what does it cost (tokens, wall-clock, review
   rounds, human interventions) to reach that quality?

## 2. Data inventory

Root (read-only mount): `NORNE_ROOT=/mnt/t7/Dell Inspiron/Local Disk D/Full-Norne`.
Raw data is never copied into the repository; tools read from `NORNE_ROOT`.

| Folder | Content | Format / notes |
|---|---|---|
| `seismics/` | 4D vintages 2001, 2003, 2004, 2006 (full stack) | SEG-Y, IBM float, IL 970–1290 (321), XL 1300–2300 (1001), 0–4000 ms @ 4 ms, coord scalar −100 |
| `FAULTS/kseim_r06_faults_depth` | Fault sticks, 2006 interpretation, depth (79 faults, 1,872 sticks) | ASCII `X Y Z code seg <pt><name> ... SSTVD meters`; a new stick starts at pt = 1 |
| `FAULTS/tral_r06_faults` | Fault sticks, 2006 interpretation, time | same layout, Z in TWT ms; also contains a copy of the kseim depth faults |
| `horisons/Top_Aare_2003_Time.txt`, `Top_Not_2_2003_Time.txt` | Structural horizons, time | `IL XL X Y Z [flag]`; IL/XL **identical** to SEG-Y numbering; surfaces extend beyond the cube. `Top_Not_2_2003_Time.txt` is **truncated** on this copy (see §11) |
| `horisons/tral+lmig32r06_fxy+Not_2_Fm_Top+...input_geomodel` | Not 2 Fm top used as geomodel input (2006) | same layout; **Z is TWT ms** (values look like metres but are not — verified, §10.3) |
| `horisons/bosd-4D_OWC-{2001,2003,2004,2006}_OFFICIAL_DEPTH` | 4D-derived OWC surfaces, depth | same layout, metres |
| `horisons/bosd_*-time` (3 files) | Time twins of the official OWC surfaces | TWT ms; usable as depth-conversion calibration |
| `time-depth-conversion_tdq/` | 6-layer interval-velocity model; 2003 depth-conversion velocity model | `.avf` |
| `wells/well_logs/` | LAS logs, ~46 Norne wellbores | LAS 2 |
| `wells/position - Original/` | Deviation logs, 46 files | OpenWorks export, two layouts (absolute X/Y, or dX/dY offsets from the well header) |
| `wells/well_header/` | Wellhead coordinates | OpenWorks export |
| `wells/time-depth/` | Time–depth tables per well | OpenWorks export |
| `wells/Wells with chek-shots/` | Checkshots for exploration wells 6608/10-2 and 6608/10-4 | XLS + LAS |
| `wells/picks/Final Well Picks.txt` | Formation tops, 45 wellbores, 2002 and 2006 generations | TSV `well, x, y, marker, MD, TVDKB, TVDSS`; the CSV variant is malformed — do not use |
| `reservoir_models/Norne_ATW_2013/` | Eclipse deck: grid `IRAP_1005.GRDECL` (46×112×22), ACTNUM, PORO, NTG, PERMX, SWATINIT, EQLNUM, FIPNUM, FAULTS, EQUIL, PVT | Eclipse ASCII |
| `reservoir_models/Norne_Full_field_realization_2013/` | 120 Petrel realizations `FullNorne_R*.GRDECL` (FACIES, SWAT, GR, NEUTRON, ACTNUM) | Same geometry as ATW (§10.6); R2–R120 exported with Y negated and J reversed |
| `GEOLINK North Sea wells/` | ~200 LAS from **other** North Sea fields | Not Norne — training material only |
| `Reports_and_Useful_Information/`, `Geological_and_Petrophysical_Reports/` | Processing report, loading parameters, simulation-model reports, well reports | PDF — knowledge source for skills |

### 2.1 Canonical conventions

- **CRS:** ED50 / UTM zone 32N (`ST_ED50_UTM32N_CM9NE`). All exchange artifacts carry X/Y in this CRS.
- **Depth:** TVDSS, metres, **positive down**. (Source files mix signs; readers normalise.)
- **Time:** TWT, milliseconds, positive down.
- **Lateral georeferencing:** X/Y is canonical. IL/XL is identical between horizon files and SEG-Y (affine fit residual 0.00 lines), so IL/XL may be used as a secondary key.
- **Grid indexing:** Eclipse I/J/K, 1-based in reports, 0-based in code; K increases downward. Realization files must be canonicalised to ATW ordering before comparison.
- **Cell volume:** `xtgeo.Grid.get_bulk_volume()` for both reference and candidates, so both are measured with the same algorithm. (A custom 24-tetrahedron estimate differed by 0.22 % in total and up to 22 % on twisted cells.)
- **Zonation** (from the ATW 2013 deck):

| Zone | Layers (K) | Sub-layers |
|---|---|---|
| Garn | 1–3 | Garn 3, Garn 2, Garn 1 |
| Not | 4 | Not (shale, inactive) |
| Ile | 5–11 | Ile 2.2, 2.1.3, 2.1.2, 2.1.1, 1.3, 1.2, 1.1 |
| Tofte | 12–18 | Tofte 2.2, 2.1.3, 2.1.2, 2.1.1, 1.2.2, 1.2.1, 1.1 |
| Tilje | 19–22 | Tilje 4, 3, 2, 1 |

- **Equilibration regions** (EQUIL in `E3.prop`; contacts TVDSS m; B<sub>o</sub> = saturated PVTO value at the region's GOC pressure):

| EQLNUM | Segment / zone | GOC | WOC | P at GOC (bar) | B<sub>o</sub> |
|---|---|---|---|---|---|
| 1 | C + D, Garn | 2582.0 | 2692.0 | 268.56 | 1.3447 |
| 2 | G, Garn | 2500.0 | 2585.5 | 263.41 | 1.3379 |
| 3 | E, Garn | 2582.0 | 2618.0 | 269.46 | 1.3459 |
| 4 | G, Ile–Tilje | 2200.0 | 2400.0 | 236.92 | 1.3044 |
| 5 | C + D + E, Ile–Tilje | 2585.0 | 2693.3 | 268.77 | 1.3450 |

## 3. Visibility tiers

Every file is assigned exactly one tier. The tier table is the leakage contract.

| Tier | Meaning | Content |
|---|---|---|
| **INPUT** | Agents may read | Seismic (all vintages), LAS logs, deviation logs, well headers, checkshots, well time–depth tables, picks of *non-blind* wells (2002 generation), GEOLINK LAS, reports **except** the model-describing reports listed under TRUTH |
| **GIVEN** | Supplied as fixed parameters, not scored | Fluid contacts and B<sub>o</sub> per region (§2.1), PVT, zonation scheme (zone names and layers per zone), CRS, target grid dimensions 46×112×22 |
| **TRUTH** | Hidden; read only by the scorer | Fault sticks (both files), all horizon files, TDQ velocity models, both reservoir-model folders, picks of *blind* wells, `NorneReservoirSimulationModelUpdatedReferenceCase2005.pdf`, `Norne model for atw2013.pdf`, `IO_workshop_2008_-_Updating_the_Norne_Reservoir_Model_with_4D_Data.pdf` |

Rationale for GIVEN: the benchmark measures geomodelling skill. Fluid
characterisation is a separate discipline; giving contacts and B<sub>o</sub>
removes it as a confounder, so volumetric error is attributable to geometry and
properties. A later "hard mode" may move contacts to TRUTH (agents derive them
from logs and 4D).

### 3.1 Blind wells

Selected deterministically by `norne_benchmark.baseline` (seed 42, ≈ 20 % of
well **families** per equilibration region — a family is a parent wellbore and
its sidetracks, which share the upper path) and frozen in
`evaluation/norne/blind_wells.json`:

- Families: **6608/10-B-2, D-3, D-4, F-3**
- Wellbores: B-2 H, D-3 H, D-3 AH, D-3 BY1HT2, D-3 BY2H, D-4 H, D-4 AH, F-3 H

Their picks are TRUTH. Their logs are INPUT only in tier D2 (§4).
Region 2 (G Garn, 2 families) has no blind well; this is accepted for v0.2.

### 3.2 Contamination controls

The ATW-2013 Norne deck is public (OPM `opm-data`), so models may have
memorised Norne facts. Controls:

- **Recall probe:** before scored runs, each model is asked closed-book for
  Norne volumes, contacts and layer counts; answers are logged. Accuracy gains
  concentrated in memorisable numbers (e.g. STOIIP) relative to geometric
  metrics flag contamination.
- **Provenance requirement:** every scored artifact must have a provenance
  chain to INPUT files via tool calls (recorded by the project store). A
  number with no provenance scores zero.
- **Filesystem isolation:** agent workspaces never mount TRUTH paths; the
  scorer runs in a separate process.

## 4. Difficulty tiers

| Tier | Upstream inputs provided | Purpose |
|---|---|---|
| **D0 – oracle** | TRUTH artifacts of every upstream stage | Isolated stage skill (§6.1) |
| **D1 – assisted** | INPUT + all non-blind picks + target zonation | Realistic interpreter setting |
| **D2 – raw** | INPUT only; blind-well logs included, no picks for them | Full autonomy |

## 5. Stages, artifact contracts, and metrics

Agents exchange **typed artifacts** through the project store (each has id,
version, producer, provenance, CRS, domain). The scorer reads final versions
and every per-round snapshot.

| # | Stage / agent | Output artifact | Reference (TRUTH) |
|---|---|---|---|
| S1 | Seismic conditioning | conditioned SEG-Y cube | — (indirect, via downstream stages) |
| S2 | Fault interpretation | `FaultSet` — named fault sticks in time and/or depth | `kseim_r06_faults_depth`, `tral_r06_faults` |
| S3 | Horizon interpretation | `Surface` per horizon, time | `tral+lmig32r06…input_geomodel` (Not 2, 2006), `Top_Aare_2003_Time` |
| S4 | Well tie & depth conversion | `WellTie` per well; `VelocityModel`; depth `Surface`s / `FaultSet` | checkshots, TD tables, blind-well picks, OWC time/depth twins, TDQ models |
| S5 | Structural framework & gridding | `CornerPointGrid` (COORD/ZCORN/ACTNUM) + `FaultFaces` | ATW `IRAP_1005.GRDECL` + `FAULT_JUN_05.INC` |
| S6 | Property modelling | `GridProperty` PORO, NTG, PERMX (FACIES optional) | ATW PETRO; facies from the 10 distinct realizations |
| S7 | Volumetrics | `VolumeTable` per zone × EQLNUM | ATW reference volumes (§7, §10.5) |

### 5.1 Metrics and calibrated tolerances

τ values are calibrated from the baseline (§10): each is set at the level of
disagreement that already exists between reference sources, so an agent is not
penalised for error that the truth itself contains. They are frozen before any
scored run.

| Metric family | Primary metric | τ | Calibration source |
|---|---|---|---|
| Depth (S4, S5) | RMS of candidate vs. blind-well 2002 picks | **10 m** | Reference grid vs. 2002 picks: RMS 4.9–6.7 m, P90 8–11 m (§10.2) |
| Faults (S2, S5) | Symmetric fault-surface distance; recall/precision @ τ | **100 m** | ≈ 1 cell; reference grid faults vs. 2006 sticks: median 48–65 m, 76 % / 68 % within 100 m (§10.4) |
| Horizons, time (S3) | Median \|ΔT\| on common nodes | **15 ms** | 2006 vs. 2003 interpretation of the same horizon: median \|ΔT\| 11 ms (§10.3) |
| Volumes (S7) | Relative STOIIP / HCPV error | **5 %** | A 10 m structural shift changes STOIIP by −5.7 % / +0.3 % (§10.5) |
| Properties (S6) | Per-zone Wasserstein distance of PORO, NTG, log k | to be set | Spread of the 10 facies realizations (§10.6) |

Detailed metrics:

**S2 Faults** (sticks densified to 10 m; per-fault matching by Hungarian assignment on mean distance)
- `fault_recall@τ`, `fault_precision@τ`, `fault_mean_dist`, `fault_hd95`, `fault_count_err` (faults intersecting the active reservoir; reference: 62).
- Robust statistics (median, P90) are primary: the distance distributions are heavy-tailed (§10.4).

**S3 Horizons** (resampled to the SEG-Y IL/XL lattice inside the reference outline)
- `hz_median_abs`, `hz_p90_abs`, `hz_coverage`, `hz_fault_gap_ok`.

**S4 Well tie / depth conversion**
- `tie_corr` — normalised cross-correlation synthetic vs. seismic in the reservoir window.
- `tdr_rms_ms` — candidate time–depth relation vs. checkshots / TD tables.
- `blind_pick_rms` — depth-converted surfaces at blind-well pick XY vs. blind picks (**key depth metric**).
- `owc_twin_rms` — depth-converted OWC time twins vs. their official depth surfaces.

**S5 Grid**
- `zone_top_rms` per zone vs. reference ZCORN interfaces; `isochore_rms` per zone.
- `brv_err` per zone × EQLNUM.
- `fault_face_recall@τ` vs. reference internal fault faces.
- `grid_quality` — negative/zero thickness, inverted pillars, twisted cells (non-zero ⇒ overt failure).
- `pick_misfit_rms` vs. blind-well picks.

**S6 Properties**
- `poro_wasserstein`, `ntg_wasserstein`, `logk_wasserstein` per zone; `poro_well_mae` at blind-well cells; `variogram_range_err`.
- `facies_rank` — rank of candidate zone facies proportions among the 10 distinct realizations.

**S7 Volumetrics** (headline)
- Relative error of BRV, NRV, PV, HCPV, STOIIP per zone × EQLNUM and field total vs. ATW.

### 5.2 Outcome classes (carried over from the F3 rubric)

- **correct** — artifact valid and all primary metrics within τ.
- **silent_failure** — artifact valid-looking but a primary metric outside 3τ (e.g. wrong CRS, J-axis reversal, sign flip, time/depth confusion — all of which occur in the source data itself, §10).
- **overt_failure** — artifact missing, unreadable, or fails validity checks.

## 6. Scoring modes

### 6.1 Isolated (stage skill)
Each stage runs in tier D0 with TRUTH upstream artifacts. Score = stage metrics.

### 6.2 Chained (system skill)
The full swarm runs in tier D1 or D2. Every stage is scored on its *final*
artifact. The **propagation gap** `Δ_s = score_isolated(s) − score_chained(s)`
locates where error accumulates.

### 6.3 Per-round trajectory (feedback loops)
The project store snapshots artifacts at every review round. The scorer
evaluates each snapshot, producing score-vs-round curves. This measures whether
cross-agent review (e.g. the fault agent reviewing the grid) improves the model
or only churns.

### 6.4 Composite score
For each primary metric `m` with tolerance `τ_m`: `s_m = exp(−e_m / τ_m)` ∈ (0, 1].
Stage score = mean of its primary `s_m`; composite = weighted mean of stage
scores with weights S2 0.15, S3 0.10, S4 0.15, S5 0.20, S6 0.15, S7 0.25.
The composite is reported **alongside** stage scores, never instead of them.

## 7. Reference volumetrics definition

From ATW 2013, per active cell (ACTNUM = 1):

- `BRV` = xtgeo bulk volume
- `NRV = BRV · NTG`
- `PV = NRV · PORO`
- `HCPV_oil = PV · (1 − SWATINIT)` for cells whose centre depth is below GOC and at or above WOC of their EQLNUM; `HCPV_gas` likewise above GOC
- `STOIIP = HCPV_oil / B_o(EQLNUM)`

SWATINIT is defined for 99.6 % of oil-zone cells (0.4 % are zero and counted
as fully oil-saturated). Eclipse's `MINPV 500` deactivation is not applied.
Volumes are reported per zone × EQLNUM and as field totals.

## 8. Baseline ("ceiling") computation

`uv run python -m norne_benchmark.baseline [--skip-ensemble]` checks the TRUTH
data for internal consistency and calibrates τ. Outputs go to
`outputs/norne_baseline/` (git-ignored, agent-inaccessible in scored runs).

1. **Georeferencing** — affine fits of SEG-Y and horizon IL/XL → X/Y; grid and active-cell footprint inside the seismic.
2. **Grid vs. picks** — reference zone tops (ZCORN) vs. 2002 and 2006 picks at the pick XY (from deviation logs).
3. **Not 2 horizon** — 2006 geomodel-input vs. 2003 interpretation (time domain).
4. **Grid vs. fault sticks** — distances between depth sticks over the active reservoir and grid fault faces (ZCORN throw > 0.5 m); FAULTS-keyword consistency.
5. **Reference volumetrics** — §7.
6. **Blind-well selection** — §3.1.
7. **Ensemble** — 120 realizations canonicalised to ATW ordering; geometry identity, ACTNUM, facies proportions per zone.

## 9. Swarm process metrics

Recorded per run from the orchestrator event log (Logfire traces):

- tokens in/out per agent and total; wall-clock; parallelism factor (Σ agent-time / wall-clock)
- review rounds per artifact; issues opened / resolved / rejected; **false-alarm rate** (issues filed against artifacts already within τ)
- human interventions (count, stage, reason)
- tool-call failures and retries

Planned ablations: sequential vs. parallel vs. parallel + feedback; reviewers
on/off; skills on/off; LLM family; local vs. cloud.

## 10. Baseline results (2026-11-06)

### 10.1 Georeferencing
- SEG-Y IL/XL → X/Y is an exact affine map (RMS 0.45 m, the coordinate rounding).
- Horizon IL/XL maps to SEG-Y IL/XL as the identity (residual 0.00 lines) — an earlier assumption that the numbering differed was wrong. Horizons simply extend beyond the cube: only 4 % of the regional Top Not 2 (2003) nodes, 90 % of the Not 2 geomodel input and 99 % of Top Åre fall inside it.
- **98.8 % of active grid cells lie inside the seismic cube** (78.6 % of all grid pillars; the inactive margin extends to IL 780–1308, XL 1150–2403).

### 10.2 Reference grid vs. well picks
Grid zone tops vs. picks at the deviation-log XY:

| Surface | 2002 pick generation: n · RMS · bias · P90 \|r\| (m) | 2006 generation: n · RMS · bias (m) |
|---|---|---|
| Garn top (`Garn Fm.`) | 37 · 5.2 · +1.6 · 9.0 | — (`FANGST GP. 2006`: 48 · 23.4 · +9.3) |
| Not top | 37 · 6.1 · +1.5 · 10.2 | 47 · **46.1 · +40.8** |
| Ile top | 37 · 5.6 · +1.1 · 9.7 | 44 · 6.5 · −0.3 |
| Tofte top | 21 · 5.7 · +2.1 · 10.9 | 25 · 6.7 · +0.9 |
| Tilje top | 16 · 4.9 · +2.1 · 8.8 | 17 · 5.4 · +2.6 |
| Tilje base | 16 · 6.7 · +1.3 · 8.1 | 17 · 15.1 · +5.1 |

The ATW grid honours the **2002 pick generation** at 5–7 m RMS; the 2006 Not
top is a different definition (≈ 41 m deeper). Scoring uses 2002 picks.
Garn-top picks disagree with their own deviation logs by 17–31 m TVDSS RMS in
some wells (Ile and deeper: 0 m) — see §11.

### 10.3 Horizons
The "Not 2 Fm Top … input_geomodel" file is in **TWT ms**: node-for-node it
matches the 2003 Not 2 time horizon (17,132 common nodes, mean +16.4 ms,
median |ΔT| 11.1 ms, P90 |ΔT| 93.5 ms, std 53.6 ms). Depth comparison against
the grid therefore requires a velocity model and is part of the S4 oracle test.
The distribution is heavy-tailed (P90 ≫ median), so S3 uses median
statistics. Where the tail sits (near faults, at the survey edge) is not yet
analysed.

### 10.4 Faults
- Grid fault faces (ZCORN throw > 0.5 m): 11,520 I-faces + 2,922 J-faces. FAULTS keyword: 61 faults, 19,866 faces, of which 74 % carry throw (the rest are zero-throw transmissibility barriers or boundary faces).
- 2006 depth sticks over the active reservoir: 62 faults, 32,447 points (10 m densified).
- Grid face → nearest stick: median 48 m, P75 98 m; **76 % within 100 m**.
- Stick → nearest grid face: median 65 m, P75 123 m; **68 % within 100 m**. 40 of 62 faults have a median distance under 100 m; faults such as `D_01`, `main_west`, `C_31` and `BC` sit 180–265 m from the nearest grid fault, i.e. the 2006 interpretation contains faults or fault positions the ATW grid does not represent.

### 10.5 Reference volumetrics (ATW 2013)

Field totals: 44,927 active cells · BRV 3,382.7 · NRV 2,731.0 · PV 674.6 ·
HCPV oil 210.1 · HCPV gas 46.3 (all 10⁶ m³) · **STOIIP 156.2 × 10⁶ Sm³**.

| Zone | EQLNUM | Cells | BRV | PV | HCPV oil | STOIIP |
|---|---|---|---|---|---|---|
| Garn | 1 | 4,224 | 317.7 | 59.5 | 11.3 | 8.4 |
| Garn | 2 | 1,227 | 75.8 | 14.8 | 6.7 | 5.0 |
| Garn | 3 | 1,296 | 127.3 | 22.6 | 3.0 | 2.3 |
| Ile | 4 | 2,329 | 42.8 | 9.4 | 0 | 0 |
| Ile | 5 | 12,679 | 575.6 | 134.1 | 83.8 | 62.3 |
| Tofte | 4 | 2,961 | 149.7 | 31.9 | 0 | 0 |
| Tofte | 5 | 12,880 | 751.5 | 188.0 | 95.5 | 71.0 |
| Tilje | 4 | 501 | 52.9 | 10.4 | 0 | 0 |
| Tilje | 5 | 6,830 | 1,289.4 | 204.0 | 9.7 | 7.2 |

(Volumes 10⁶ m³, STOIIP 10⁶ Sm³.) Ile + Tofte in region 5 hold 85 % of the oil.

**Sensitivity:** shifting the structure relative to the contacts changes
STOIIP by −5.7 % (10 m shallower), −2.7 % (5 m shallower), +0.3 % (5 m deeper)
and −0.5 % (10 m deeper). Shallower structure moves oil into the gas cap,
deeper structure moves oil-leg base into water while the gas cap shrinks. This
asymmetry is why τ<sub>vol</sub> = 5 % pairs with τ<sub>depth</sub> = 10 m.

### 10.6 Realization ensemble
- All 120 realizations share the ATW geometry exactly (ZCORN and COORD within 5 mm) once canonicalised: R1 is in ATW ordering; R2–R120 are exported with Y negated and the J axis reversed.
- ACTNUM is identical across realizations (104,970 active cells; R1: 104,939) and is a superset of ATW's 44,927 active cells (99.98 % overlap) — the realizations keep water-leg and margin cells that ATW deactivates.
- Only **10 distinct facies models** exist among the 120 (one is shared by 78 realizations).
- Consequence: the ensemble constrains **facies proportions only** — not structure, not bulk volume. Facies P10/P50/P90 on ATW-active cells (codes 0/1/3) per zone are in `outputs/norne_baseline/ensemble_envelope.csv`; e.g. Ile facies 0: 0.51 / 0.54 / 0.54.

## 11. Open issues

- **Truncated file:** `horisons/Top_Not_2_2003_Time.txt` on the T7 copy is exactly 26 MiB with a cut-off last record; inlines below 821 are missing. Outside the seismic cube and the active reservoir, so the impact is small — but re-download it from the source release if possible.
- **Garn-top pick depths:** for some wells the Garn-top pick TVDSS disagrees with the deviation-log TVDSS at the same MD by 17–31 m RMS (Ile and deeper agree exactly). Find out which is right before Garn picks are used as blind truth.
- **Checkshots** exist only for two exploration wells; production wells have TD tables. Confirm TD-table provenance before using them as TRUTH vs INPUT.
- **Canonical seismic vintage:** proposal is 2001 (pre-production baseline) for structure; other vintages for a later 4D stage.
- **Facies codes** (0/1/3) need a geological legend before S6 facies scoring.
- **Region 2** (G Garn) has no blind well.
- **STOIIP check:** 156.2 × 10⁶ Sm³ is close to commonly quoted Norne figures, but has not yet been checked against the 2005 reference-case report on the drive (no PDF reader installed).
