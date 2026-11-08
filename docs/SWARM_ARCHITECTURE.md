# Boglodite Geomodelling Swarm — Architecture (draft v0.1)

Status: **draft** · Created 2026-11-08 · Companion to [`NORNE_BENCHMARK.md`](./NORNE_BENCHMARK.md)

This document fixes how the swarm of geoscience agents is built and run, so that
it can be scored by the Norne benchmark. It adapts the *Strata Swarm* design
proposal (blackboard, producer/verifier pairs, uncertainty ledger, human gates)
to the Norne data and to the needs of a controlled experiment.

---

## 1. Principles

1. **The LLM decides; deterministic tools compute.** An agent reads its task and
   the objects it depends on, chooses parameters, calls a versioned tool, reads the
   diagnostics, and writes a verdict citing them. It never computes a semblance,
   a variogram or a volume itself.
2. **A blackboard, not a pipeline.** Agents publish typed, versioned objects to a
   shared project store and subscribe to the objects they depend on. Work flows
   forward, backward (rejections) and in parallel.
3. **The producer never grades itself.** Every published object has at least one
   verifier from another domain — chosen because *its own work breaks* if the
   object is wrong, so it has both an incentive and a test.
4. **Confidence travels with the data.** Picks carry per-trace confidence,
   faults a support count, properties a realization index. Verifiers read
   confidence, not only geometry.
5. **Disagreement is not an error.** If producer and verifier do not converge in
   two rounds, both versions become named scenarios in the uncertainty ledger and
   one decision card for a human (or, in autonomous runs, for the orchestrator's
   documented default).
6. **Humans freeze, agents iterate.** Between gates the swarm may rework anything;
   after a gate, only a human reopens.
7. **Agents never see TRUTH.** Verifiers use INPUT data only (benchmark §3). The
   scorer runs in a separate process after the run. The two kinds of check are
   kept apart on purpose — and comparing them is itself a result (§7.3).

## 2. Agent roster — Norne v1

Eight specialists plus the orchestrator. The full 14-agent design remains the
target; deferred agents are listed in §2.1.

| Agent | Owns / publishes | Verifies (test) | Tools (open source) |
|---|---|---|---|
| **Data steward** | Project settings (CRS, units, datums, time/depth tags); loaded `Cube`, `WellSet`, `Tops`, `TDTable` | **Everyone**: rejects any object whose CRS, unit, domain or index convention disagrees with the project (catches Norne's MAPAXES Y-flip, J-reversed exports, time-labelled-as-depth horizon, TVDSS sign, 2002 vs 2006 picks) | segyio, lasio, wellpathpy, pyproj, xtgeo |
| **Horizon agent** | `HorizonPicks` (time) with confidence and phase; `Surface` (time) | Fault agent's sticks: every horizon crossed shows consistent displacement; throw dies toward tips | scipy (cross-correlation tracker), scikit-image, xtgeo RegularSurface, verde / pykrige |
| **Fault agent** | `FaultSticks`, `FaultSet` (time and depth), fault families | Horizon picks: none cross a fault without offset. Gridding: grid throw and juxtaposition equal interpreted values | d2geo coherence, scikit-image (ridges/Hough); FaultSeg (Boglodite MCP) only where coherence fails |
| **Well agent** (log QC + correlation + tie + velocity) | `LogSet` (QC'd, Vsh/φ/NTG), `Tops` (reconciled), `WellTie`, `Wavelet`, `TDTable`, `VelocityModel`; depth-converted `Surface`s and `FaultSet` | Horizon picks: pick-to-top residual at every non-blind well after T–D; polarity vs. impedance. Property ensemble: upscaled vs. fine-log histograms | lasio, welly, bruges, striplog, dtaidistance, scipy.signal |
| **Structural framework agent** | `FaultFramework` (fault surfaces, truncations), `HorizonModel` (depth) | Volumetrics: contacts vs. closure and spill; compartment identity | LoopStructural, GemPy, pyvista, xtgeo |
| **Gridding agent** | `Grid` (corner-point, 46 × 112 × 22, zones per GIVEN zonation) | — (verified by fault + well agents) | **own pillar gridder** (fault-aware corner-point; main build risk), xtgeo Grid, resqpy |
| **Property agent** | `Property` ensembles: FACIES, PORO, NTG, PERMX — never a single realization | — (verified by well agent) | gstools, geostatspy, gstatsim |
| **Contacts & volumetrics agent** | `Volumes` per zone × region × realization; `Compartments` | Grid: no negative or zero cells in HC zones; contacts honour GIVEN | xtgeo, numpy, SALib |
| **Orchestrator** | Task graph, assignments, gates, ledger, human inbox; the final candidate export | Enforces the two-round rule and gate freezes | Pydantic AI + own event scheduler (§4) |

### 2.1 Deferred agents (from the full design)

| Agent | Why deferred for Norne v1 | Revisit when |
|---|---|---|
| Conditioning | Norne 4D cubes are already processed | horizon/fault confidence is limited by noise |
| Attribute | Folded into the fault agent's tool set | attribute parameterisation needs its own owner |
| Inversion & rock physics | Well control is dense (≈ 46 wellbores) | property agent needs a secondary variable |
| Handover | OPM Flow initialisation check comes later | connectivity metric is validated against flow |

## 3. Project store (blackboard)

Every object is a Pydantic model (metadata) plus a payload file in a standard
format. Objects are immutable; a change is a new version.

```
ObjectMeta
  id, type, version            e.g. "Surface:IleTop", 3
  producer                     agent name
  inputs                       [object_id@version, …]          ← provenance
  tool_calls                   [{tool, version, params, diagnostics}]
  crs, domain, unit            "ED50/UTM32N", "depth" | "time", "m" | "ms"
  confidence                   object-level summary; per-element in payload
  status                       draft → verified | rejected → frozen
  verdicts                     [{verifier, pass, findings, evidence, round}]
  scenario                     None | ledger scenario name
  payload                      path (GRDECL, xtgeo .gri/.roff, CSV, zarr)
```

| Type | Payload | Producer |
|---|---|---|
| `Cube` | SEG-Y reference (read-only path) + geometry | data steward |
| `WellSet`, `LogSet`, `Tops`, `TDTable` | CSV / LAS | data steward → well agent |
| `HorizonPicks`, `Surface` | CSV (il, xl, twt, conf) / xtgeo surface | horizon agent; depth versions: well agent |
| `FaultSticks`, `FaultSet`, `FaultFramework` | CSV (x, y, z, stick, fault, support) | fault agent; framework agent |
| `VelocityModel`, `WellTie`, `Wavelet` | CSV / JSON | well agent |
| `Grid` | GRDECL (COORD/ZCORN/ACTNUM) | gridding agent |
| `Property` | GRDECL keyword per realization | property agent |
| `Volumes`, `Compartments` | CSV / JSON | volumetrics agent |
| `Scenario`, `Decision` | JSON | orchestrator / human |

Store layout per run: `runs/<run_id>/store/<type>/<name>/v<k>/{meta.json, payload}`
plus an append-only `events.jsonl` (publish, subscribe, verdict, task, gate,
human decision, token usage). Agents reach the store only through store tools.

## 4. Orchestration

- **Event-driven scheduler** (not a fixed graph). Events: object published,
  verdict posted, task finished, gate passed. A new version marks every dependent
  object **stale**, which re-queues its producer and its verifiers.
- **Task graph** from the study brief: for each target object, a producer task and
  one verifier task per verification pair (§5). Independent tasks run in parallel
  (e.g. horizon, fault and well agents start together once the data steward has
  published).
- **Two-round rule:** producer → verifier → (rework → re-verify) at most twice; then
  both versions go to the ledger as scenarios and one decision card is raised.
- **Gates** G1 data accepted · G2 interpretation frozen · G3 framework frozen ·
  G4 property model accepted · G5 candidate exported.
- **Autonomy modes** (a benchmark variable):

| Mode | Decision cards and gates answered by |
|---|---|
| `autonomous` | orchestrator, using a documented default rule; logged as such |
| `gates_only` | a human at G1–G5; the swarm decides everything else |
| `every_decision` | a human at every card |

- **Framework:** each agent is a Pydantic AI agent with typed outputs (objects,
  verdicts). Deterministic tools are exposed as MCP servers (§6), extending the
  existing `boglodite-seismic` server. Logfire traces every call.

## 5. Verification pairs on Norne

Only pairs with a deterministic test that runs on INPUT data are built.

| Producer → object | Verifier | Test | If it fails |
|---|---|---|---|
| Any → any object | Data steward | CRS, unit, domain, index order and sign conventions match the project | Reject with the mismatch |
| Horizon → picks | Fault agent | No pick crosses a fault without an offset consistent with the fault's throw | Re-track with the fault as barrier |
| Horizon → picks | Well agent | Pick-to-top residual at every non-blind well after T–D within tolerance; polarity consistent with log reflectivity | Decide whether pick, tie or top is wrong using neighbouring horizons |
| Fault → sticks | Horizon agent | Every horizon crossed is displaced; throw varies smoothly and dies at the tips | Demote to lineament |
| Fault → sticks | Fault agent's attribute check (separate tool, same agent) | Stick lies on a coherence low along ≥ 80 % of its length | Flag as possible artefact |
| Well → T–D / velocity | Data steward | Checkshot residuals (6608/10-2, 10-4) and TD tables; depth-converted horizon hits non-blind tops | Re-fit velocity per zone |
| Well → tops | Framework agent | Tops never invert stratigraphic order; isochores non-negative | Move top or reclassify surface |
| Framework → fault framework | Fault agent | Framework throw equals interpreted throw; truncations consistent | Swap truncation, re-weld |
| Gridding → grid | Fault agent | Throw and juxtaposition across grid fault faces equal the framework | Change increments or stair-step the fault |
| Gridding → grid | Well agent | Zone tops at non-blind wells; zone thickness ≥ 0; no horizon crossing | Re-layer zone |
| Property → ensemble | Well agent | Upscaled vs. fine-log histograms; leave-one-out well RMSE (non-blind wells only); variogram reproduction | Re-fit variogram / trend |
| Volumetrics → volumes | Framework agent | Contacts within closure; compartments defined by the framework | Open a contact scenario |

The blind wells (benchmark §3.1) are **never** used by any verifier.

## 6. Tool servers (MCP)

| Server | v1 tools (each deterministic, versioned, JSON-schema in/out) |
|---|---|
| `boglodite-data` | inspect / load SEG-Y, LAS, deviation, tops, TD tables; CRS and convention audit |
| `boglodite-seismic` (exists) | FaultSeg, MalenoV; + coherence, horizon tracker, throw profile, offset-at-fault |
| `boglodite-well` | log QC, Vsh/φ/NTG, top reconciliation, synthetic + tie, wavelet, velocity fit, depth conversion |
| `boglodite-structure` | fault surfaces from sticks, implicit horizon model, truncation rules, framework QC |
| `boglodite-grid` | pillar gridder (fault-aware), zonation/layering, cell QC, juxtaposition check, GRDECL export |
| `boglodite-property` | upscaling, variogram fit, SGS/SIS realizations, histogram / blind-well QC |
| `boglodite-volume` | volumes with GIVEN contacts, B<sub>o</sub>, saturation-height; compartments |
| `boglodite-store` | publish, read, subscribe, post verdict, open scenario, raise decision card |

## 7. Benchmark integration

### 7.1 Run layout and isolation

```
runs/<run_id>/
  manifest.json      model, autonomy mode, tier (D0/D1/D2), seed, agent roster, tool versions
  store/             project store (§3)
  events.jsonl       every event, token usage, timings
  candidate/         exported per benchmark §5.0 (grid.grdecl, faults.csv, horizon_not2_time.csv)
```

Agent processes see INPUT paths and `runs/<run_id>/` only. After the run, the
scorer (separate process) reads `candidate/` and TRUTH and writes the score and
report under `outputs/norne_scores/runs/<run_id>/`.

### 7.2 Difficulty tiers as store contents

- **D0 (oracle):** the upstream objects of the stage under test are injected as
  *frozen* objects built from TRUTH; only that stage's agent(s) run.
- **D1 (assisted):** non-blind picks are injected as frozen `Tops`; everything else
  is produced by the swarm.
- **D2 (raw):** only INPUT is loaded by the data steward.

### 7.3 What is measured besides the model

| Quantity | From |
|---|---|
| Stage and composite scores, per final object and **per round** | scorer on each store snapshot |
| Tokens, wall-clock, parallelism (Σ agent-time / wall-clock) | events.jsonl |
| Review rounds per object; issues opened / resolved / sent to the ledger | events.jsonl |
| **Verifier quality**: precision (rejections that were justified by the true error) and recall (true errors that were caught) | verdicts vs. scorer |
| **Ensemble calibration**: does the true value (e.g. STOIIP 156.3 × 10⁶ Sm³) fall inside the swarm's own P10–P90? | volumes ensemble vs. TRUTH |
| Human decisions: count, gate, time to answer | events.jsonl (`Decision`) |

### 7.4 Planned comparisons

Sequential chain vs. blackboard with verifiers vs. blackboard without verifiers;
autonomy modes; LLM family (cloud vs. local); skills on/off; D0 vs. D1 vs. D2
(propagation gap, benchmark §6.2).

## 8. Build order

| Milestone | Content | First score it produces |
|---|---|---|
| **M2** | Deterministic slice, no LLM: zone tops from non-blind 2002 picks + TRUE fault sticks → structural framework → **fault-aware pillar gridder** → volumes. Store and events skeleton. | Structural, stratigraphic, evaluation, connectivity (tier D0/D1) |
| **M3** | Wrap M2 tools as MCP servers; Pydantic AI agents for data steward, framework, gridding, volumetrics; orchestrator with verification pairs and ledger | Same stages, now by agents; first process metrics |
| **M4** | Well agent (tie, velocity, depth conversion), horizon agent, fault agent; replace TRUE inputs one by one | Horizons, faults; propagation gaps |
| **M5** | Property agent ensembles; full swarm in parallel with feedback; autonomy modes; comparisons (§7.4) | All eight stages; calibration; verifier quality |

## 9. Open questions

- LLM backend(s) for M3: Claude via API, other providers, local (LM Studio / Ornith).
- Pillar gridder design: build ZCORN from the framework per fault block, or adapt
  an existing library — decided in M2.
- Facies code legend (0/1/3) and facies logs for blind-well facies metrics.
- Default rules the orchestrator uses for decision cards in `autonomous` mode.
- How the reservoir-modelling and ensemble-QC guideline PDFs in `tmp/` feed the
  agents' skills (QC checklists as verifier tests).
