# Norne Static-Modelling Benchmark — Specification (draft v0.2)

Status: **draft** · Owner: Boglodite · Created 2026-11-06 · v0.4 (2026-11-07) replaces the stage metrics with geoscience metrics and adds connectivity (§5) · v0.2 adds baseline results (§10) · v0.3 (2026-11-07) adds the scorer (§5.0, §6.5), the saturation-height GIVEN (§7) and corrects the volume sensitivity (§10.5) and the 2003-horizon coverage (§10.3, §11)

This document defines how the Boglodite geomodelling swarm is evaluated against
the Norne field dataset. It fixes *what agents may see*, *what they are scored
against*, *how each stage is scored*, and *how the whole swarm is scored*. It is
the contract that every agent, tool, and experiment must respect. Nothing here
depends on a particular LLM, framework, or agent topology — those are the
variables under test.

Code: `norne_benchmark/` (readers/writer in `io.py`, baseline in `baseline.py`, scorer in `score.py`, scorer self-test in `selftest.py`).
Viewer: `sandbox/norne_viewer.py` → `tmp/norne_3d.html`.
Swarm architecture (agents, project store, verification pairs, run layout): [`SWARM_ARCHITECTURE.md`](./SWARM_ARCHITECTURE.md).

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
| `horisons/Top_Aare_2003_Time.txt`, `Top_Not_2_2003_Time.txt` | Structural horizons, time | `IL XL X Y Z [flag]`; IL/XL **identical** to SEG-Y numbering; surfaces extend beyond the cube. `Top_Not_2_2003_Time.txt` is **partial** on this copy — only IL 821–992 and 1549–1558 (see §11) |
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
| **GIVEN** | Supplied as fixed parameters, not scored | Fluid contacts and B<sub>o</sub> per region (§2.1), saturation-height function per zone (§7), PVT, zonation scheme (zone names and layers per zone), CRS, target grid dimensions 46×112×22 |
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

### 5.0 Candidate contract (what the scorer reads)

A scored run delivers one directory; every file is optional and each stage is
scored only if its inputs exist (`norne_benchmark/score.py`):

| File | Content | Stages |
|---|---|---|
| `grid.grdecl` | `SPECGRID 46 112 22`, `COORD`, `ZCORN`, `ACTNUM` in absolute ED50/UTM32N (no MAPAXES); `PORO`, `NTG`, `PERMX`, `FACIES` (codes 0/1/3) may be included | structural, stratigraphic, facies, petrophysical, evaluation, connectivity |
| `props.grdecl` | `PORO`, `NTG`, `PERMX`, `FACIES` if not in `grid.grdecl` | facies, petrophysical, evaluation |
| `faults.csv` | depth fault sticks: `x, y, z, stick, fault` (TVDSS m, positive down) | fault interpretation |
| `horizon_not2_time.csv` | Not 2 Fm top in time: `il, xl, twt` (any IL/XL lattice; interpolated onto reference nodes, gaps > 4 lines left uncovered) | seismic horizons |
| `manifest.json` | run metadata (model, agent pattern, tokens, wall-clock) — copied into the score | — |

Layer *K* of the candidate grid belongs to the zone given by the GIVEN zonation
(§2.1). Cell ordering is free (e.g. J reversed is fine) — every comparison is in
X/Y/Z. Candidate cells are assigned to an EQLNUM region by the nearest active
reference cell in the same layer (≤ 300 m; otherwise no region and no
hydrocarbons).

Agents exchange **typed artifacts** through the project store (each has id,
version, producer, provenance, CRS, domain). The scorer reads final versions
and every per-round snapshot.

Stages are scored in the geoscientist's terms. Every metric is a familiar
quantity in domain units (MAE / RMSE in m or ms, F1, %, R²), judged against a
stated tolerance and reported as pass ✓ / fail ✗. The 0–1 score derived from it
exists only to rank runs. Stage weight follows importance (very high = 3, high = 2).

| Stage | Importance | Primary metrics (tolerance) | Compared against (TRUTH) |
|---|---|---|---|
| **Seismic horizons** | very high | Not 2 Fm top time **MAE ≤ 15 ms**; reservoir area **covered ≥ 90 %** | 2006 Not 2 Fm geomodel-input horizon (TWT) |
| **Fault interpretation** | very high | **Fault-level F1 ≥ 0.75** (one-to-one matching; a fault matches when ≥ 50 % of each lies within 100 m of the other); **point-level F1 ≥ 0.75** (within 100 m) | 2006 depth fault sticks over the active reservoir (faults ≥ 300 m of stick) |
| **Structural modelling** | very high | Zone-top **surface MAE ≤ 10 m**; blind-well **pick MAE ≤ 10 m**; **fault throw MAE ≤ 10 m** (every reference and candidate fault face; a missing or spurious fault counts its full throw) | ATW zone-top interfaces; 2002 picks in blind wells; ATW internal fault faces |
| **Stratigraphic modelling** | high | Zone **thickness MAE ≤ 5 m** (mean of Garn, Ile, Tofte, Tilje); **zone consistency ≥ 99 %** (columns with interfaces in order and no non-positive cells); **pinch-out agreement ≥ 90 %** (zone present/absent where the reference has it) | ATW zone isochores |
| **Facies modelling** | high | Mean facies **proportion outside the realization P10–P90 ≤ 0.05** per zone and code | 10 distinct facies models among the 120 realizations |
| **Petrophysical modelling** | very high | Along the blind-well paths: **porosity MAE ≤ 0.02**, **log₁₀ k MAE ≤ 0.3**, **NTG MAE ≤ 0.05**; **R² of zone-average porosity maps ≥ 0.5** | ATW cells penetrated by blind wells (conditioned to their logs); ATW zone-average maps |
| **Reservoir evaluation** | very high | **Pore-volume, HCPV and STOIIP error within ±5 %** (GIVEN contacts, B<sub>o</sub>, saturation-height) | ATW volumes (§7) |
| **Reservoir connectivity** | very high | **Compartment agreement ARI ≥ 0.8** (adjusted Rand index of hydrocarbon compartments); **well-pair agreement ≥ 90 %** (each pair of wells connected/separated as in the reference) | ATW static compartments (§5.3) |

Secondary metrics (reported, not scored): RMSE and bias for every MAE, fault
precision and recall, median fault distance, gross-rock-volume error, HCPV error
per zone × region, porosity R² along wells, compartment counts.

**Why not cell-by-cell F1 or R² for facies and properties against the reference
model?** The reference properties and facies are one stochastic realization; an
equally valid realization would score poorly cell by cell (the 120 Norne
realizations differ from each other that way). Cell-wise comparison is fair only
where the truth is data — along the blind wells — and otherwise properties are
compared as smoothed maps and distributions. Facies macro-F1 / balanced accuracy
/ IoU at the blind wells will be added once facies logs and a code legend are
established (§11).

### 5.1 Tolerances

Tolerances are set from how much independent truth sources already disagree, so
an agent is not penalised for error that the truth itself contains: depth 10 m
(reference grid vs. 2002 picks: RMS 4.9–6.7 m, P90 8–11 m), time 15 ms (2006 vs.
2003 Not 2 interpretation: median |ΔT| 11 ms), volume 5 % (≈ 8–9 m of structural
error, §10.5), fault distance 100 m (≈ one cell). Fault F1, thickness, facies,
property and connectivity tolerances are **provisional** until calibrated.

### 5.2 Outcome classes

- **correct** — every primary check passes.
- **degraded** — no primary metric worse than 3× its tolerance, but at least one check fails.
- **silent failure** — a primary metric worse than 3× its tolerance, or measured on under 50 % of the reference (wrong CRS, Y sign flip, time/depth confusion…). For "≥" metrics, 3× means the shortfall from 1 is three times the allowed shortfall (e.g. F1 < 0.25).
- **overt failure** — artifact missing, unreadable, or wrong dimensions.

### 5.3 Static connectivity

Hydrocarbon cells are active cells above their region's WOC. Two cells are
connected vertically when adjacent in the same column, and laterally when their
shared face intervals overlap by more than 0.5 m — which includes juxtaposition of
different layers across a fault. Compartments are the connected components.
Fault seal (transmissibility multipliers) is not modelled in v0.4; a flow-based
check with OPM Flow is a later option.

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
Each primary metric gets a score: exp(−value/tol) for "≤" metrics, exp(−(1−value)/(1−tol))
for "≥" metrics, so a metric exactly at its tolerance scores e⁻¹ ≈ 0.37. Stage score =
mean of its primary metric scores; composite = importance-weighted mean of the stages
submitted. Reports lead with the checks passed; the composite only ranks runs.

### 6.5 Scorer self-test

`uv run --no-sync python -m norne_benchmark.score --selftest` builds eight
candidates from TRUTH with one controlled defect each and asserts the scorer's
response (19 checks, all must pass before any agent run is scored):

| Case | Expected | Observed (2026-11-07) |
|---|---|---|
| reference itself | S2, S3, S5, S6, S7 = 1.00; S4 = truth-vs-truth ceiling | 1.00 ×5; S4 0.52 (blind-pick RMS 6.6 m); composite 0.928 |
| structure 10 m deeper | zone-top RMS 10 m, STOIIP ≈ −6 %, BRV unchanged | 10.00 m, −5.95 %, ±0 |
| Y negated (MAPAXES trap) | silent failure on S4, S5, S7 | silent failure ×3 |
| J axis reversed, same geometry | identical to reference | identical |
| porosity −10 % | STOIIP −10 %, S6 degraded, geometry perfect | −10.00 %, poro W = 0.024, S5 1.00 |
| 10 largest faults removed | recall drops, precision stays | recall 0.61, precision 1.00 |
| 2003 Not 2 interpretation | median ≈ 11 ms on covered nodes; silent failure on coverage | 10.7 ms; coverage 0.05 → silent failure |
| malformed files | overt failures | overt ×5 |

Note the composite of the perfect reference is 0.928, not 1.0: S4 compares the
grid to blind-well picks, and the reference grid itself misses them by 6.6 m.
That is the ceiling any agent can reach on S4.

### 6.6 Visual reports

`norne_benchmark/report.py` turns a scored run into a self-contained
`report.html` (scorecards; 3D candidate grid coloured by depth error; zone-top
error maps; blind-well residual dot plot; fault map and per-fault recall table;
time-horizon ΔT map; per-zone property histograms; volume error bars), and a
directory of runs into an `index.html` leaderboard (stage-score heatmap with
outcome icons, sortable). The self-test writes both for its eight cases under
`outputs/norne_scores/selftest/`. Reports embed reference-derived data and must
stay outside agent workspaces.

    uv run --no-sync python -m norne_benchmark.report --candidate DIR
    uv run --no-sync python -m norne_benchmark.report --index outputs/norne_scores/selftest

## 7. Reference volumetrics definition

From ATW 2013, per active cell (ACTNUM = 1):

- `BRV` = xtgeo bulk volume
- `NRV = BRV · NTG`
- `PV = NRV · PORO`
- `HCPV_oil = PV · (1 − S_w(zone, h))` for cells whose centre depth is below GOC and at or above WOC of their EQLNUM; `HCPV_gas` likewise above GOC
- `STOIIP = HCPV_oil / B_o(EQLNUM)`

**S<sub>w</sub> is a GIVEN saturation-height function**, not a per-cell property:
for each zone, the PV-weighted mean of the reference SWATINIT in 5 m bins of
height *h* above the cell's WOC (gaps filled from the nearest bin; S<sub>w</sub> = 1
below WOC). It is applied identically to reference and candidate, so a candidate
grid needs no saturation property, and S<sub>w</sub> responds to structural error
the way a real model does (a cell that moves toward the WOC gets wetter).

| h above WOC (m) | 0–5 | 10–15 | 25–30 | 50–55 | 100–105 | 200–205 |
|---|---|---|---|---|---|---|
| Garn | 0.64 | 0.34 | 0.25 | 0.18 | 0.18 | 0.16 |
| Ile | 0.68 | 0.34 | 0.23 | 0.17 | 0.16 | 0.17 |
| Tofte | 0.60 | 0.30 | 0.21 | 0.15 | 0.19 | 0.19 |
| Tilje | 0.66 | 0.32 | 0.23 | 0.21 | 0.21 | 0.21 |

With this function the reference STOIIP is **156.31 × 10⁶ Sm³** (cell-by-cell
SWATINIT: 156.23, a 0.05 % difference). The scorer uses the saturation-height
value throughout. Eclipse's `MINPV 500` deactivation is not applied. Volumes are
reported per zone × EQLNUM and as field totals.

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
statistics. **Caveat:** the 2003 file on this copy covers only IL 821–992 (and
1549–1558), so these common nodes all lie in the strip IL 899–992 at the
north-western end of the reservoir; the 15 ms tolerance rests on that strip. Where the tail sits (near faults, at the survey edge) is not yet
analysed.

### 10.3b Input-to-truth floor for the time horizon (2026-11-08)

Not 2 NO 2006 Top picks converted to time with the INPUT time–depth tables differ from the
reference 2006 Not 2 horizon by **MAE 22.2 ms** (median: ties 11 ms deeper; E-4 area 60–90 ms).
The reference sits on a **trough** at 81 % of locations (peak 15 %, lateral dimming/reversal around
IL 1080–1160 × XL 2000–2100). The reference was depth/time-converted with a different model (TDQ),
not the well time–depth tables. A horizon tied perfectly to the INPUT checkshot times would therefore
score ≈ 22 ms; synthetic-seismogram ties recover part of the offset (median −10.6 ms). The tolerance
stays at 15 ms; the floor is reported alongside (score info `input_floor_ms`), like the 6.6 m blind-
well floor in depth.

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

**Sensitivity** (corrected in v0.3): shifting the whole structure relative to
the fixed contacts, with S<sub>w</sub> from the saturation-height function, changes
STOIIP by **+5.6 % (10 m shallower), +3.0 % (5 m shallower), −3.1 % (5 m deeper)
and −6.0 % (10 m deeper)** — about 0.6 % per metre. (v0.2 reported an asymmetric
−5.7 % / −0.5 %; that calculation kept each cell's SWATINIT attached to the cell
as it moved, carrying the transition zone with the structure, which is not
physical.) A 5 % volume tolerance therefore corresponds to an 8–9 m structural
error, consistent with τ<sub>depth</sub> = 10 m.

### 10.6 Realization ensemble
- All 120 realizations share the ATW geometry exactly (ZCORN and COORD within 5 mm) once canonicalised: R1 is in ATW ordering; R2–R120 are exported with Y negated and the J axis reversed.
- ACTNUM is identical across realizations (104,970 active cells; R1: 104,939) and is a superset of ATW's 44,927 active cells (99.98 % overlap) — the realizations keep water-leg and margin cells that ATW deactivates.
- Only **10 distinct facies models** exist among the 120 (one is shared by 78 realizations).
- Consequence: the ensemble constrains **facies proportions only** — not structure, not bulk volume. Facies P10/P50/P90 on ATW-active cells (codes 0/1/3) per zone are in `outputs/norne_baseline/ensemble_envelope.csv`; e.g. Ile facies 0: 0.51 / 0.54 / 0.54.

## 11. Open issues

- **Partial file:** `horisons/Top_Not_2_2003_Time.txt` on the T7 copy is exactly 26 MiB with a cut-off last record, and contains only IL 821–992 and 1549–1558 — about 5 % of the reservoir footprint. (v0.2 wrongly said only IL < 821 was lost.) It is not used as S3 truth (the 2006 horizon is complete), but the S3 tolerance calibration rests on the IL 899–992 strip. Re-download it from the source release if possible.
- **Garn-top pick depths:** for some wells the Garn-top pick TVDSS disagrees with the deviation-log TVDSS at the same MD by 17–31 m RMS (Ile and deeper agree exactly). Find out which is right before Garn picks are used as blind truth.
- **Checkshots** exist only for two exploration wells; production wells have TD tables. Confirm TD-table provenance before using them as TRUTH vs INPUT.
- **Canonical seismic vintage:** proposal is 2001 (pre-production baseline) for structure; other vintages for a later 4D stage.
- **Facies codes** (0/1/3) need a geological legend before S6 facies scoring.
- **Region 2** (G Garn) has no blind well.
- **Property tolerances** (S6) are provisional (poro 0.02, NTG 0.05, log k 0.3); calibrate from the facies realizations or log-vs-cell comparisons.
- **STOIIP check:** 156.2 × 10⁶ Sm³ is close to commonly quoted Norne figures, but has not yet been checked against the 2005 reference-case report on the drive (no PDF reader installed).
