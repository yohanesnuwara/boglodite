# Strata dev notes — horizon slice (2026-11-08)

## Well ties (Not 2 NO 2006 Top, non-blind wells)
- TD tables: depth column is TVD below KB, not MD (header is wrong). As TVD: tables agree to median 6.9 ms;
  as MD: median 60 ms, up to 670 ms. Ricker synthetic tables are unreliable (excluded by default).
- With own tables + nearest-neighbour fallback: 36/36 wellbores tied (13 own, 23 neighbour);
  spread between tables median 9.4 ms, max 23.8 ms; no outlier tables after the TVD fix.

## Tracking trials (strata.tools.horizon.track, 2001 cube 1700–3300 ms) → tracking_trials.json
| case | coverage | jumps | well MAE | well max |
|---|---|---|---|---|
| trough, 3 seeds | 0.987 | 20788 | 71.1 | 324.7 |
| trough, all seeds | 0.986 | 22716 | 38.8 | 214.0 |
| trough, all, corr 0.85 step 6 | 0.968 | 19725 | 34.5 | 177.6 |
| trough, all, corr 0.9 step 4 | 0.882 | 9905 | 19.3 | 138.1 |
| **peak, all seeds** | 0.986 | 20661 | **10.2** | **33.0** |
- Tied tops sit closer to the peak (median offset 7.2 ms) than the trough (9.2 ms); the first polarity
  heuristic over-weighted trough amplitude → fixed (offset consistency dominates).
- ~20k jumps (>12 ms between neighbours) remain even in the best case → faults; next: fault agent / barriers.
- Tracking a full cube takes 12–17 s (321 × 1001 traces).

## First Claude run (run 20261108-175254-abd5, anthropic:claude-sonnet-5-5, autonomous) — 2026-11-08
- Agents behaved sensibly: steward audit; well agent chose 'Not 2 NO 2006 Top' with reasoning (2002 set has no Not 2);
  horizon agent chose peak, +5 ms bulk shift, iterated 4–6 tracking parameter sets per round (strict min_corr 0.92–0.95,
  max_step 3–4 ms); verifier wrote cited findings. Verifier rejected v1/v2/v3 (well MAE 25.8 → 35.6 → 15.0 ms) → ledger → default accept.
- Benchmark: Not 2 top time MAE **39.0 ms ✗** (bias +25.5 ms), coverage 97.5 % ✓ — worse than scripted baseline (34.0 ms, coverage 98.5 %).
- Tokens: ≈ 361k input / 19k output (horizon agent ≈ 292k input over 3 rounds).

## Benchmark-side check: reference horizon vs well ties (TRUTH; never shown to agents) → reference_vs_ties.csv
- The reference 2006 Not 2 Fm top sits on a **trough** (median 2.0 ms from nearest trough; 15.4 ms from nearest peak).
- Well ties (Not 2 NO 2006 Top + TD tables) are **deeper than the reference by median 11 ms, MAE 22 ms**; E-4 area 60–90 ms
  (those wells are neighbour-tied 2.1–2.4 km away). → a horizon fitting the ties perfectly would score ~22 ms MAE (> 15 ms tolerance):
  the reference was converted with a different time–depth relation (TDQ model), not the TD tables. This is a truth-vs-INPUT floor,
  analogous to the 6.6 m blind-well floor in depth.
- Geophysically, Not 2 top (shale-dominated Not Fm) could be expected as a trough; an agent can only know that from a
  synthetic seismogram built from the logs — a missing tool.

## Improvement round (2026-11-08 evening)
### Diagnosis
- Reference Not 2 top: trough at 81 % of locations, peak 15 % (genuine lateral dimming/reversal around IL 1080–1160, XL 2000–2100)
  → reference_horizon_polarity.csv. Both earlier runs tracked the peak below.
- Greedy tracker error is bimodal: ~40 % of the reservoir area within 8 ms, the rest one cycle off (±25–40 ms) or on another
  reflector (> 50 ms) across large-throw faults. Fixing polarity + shift alone did not help (35.9 ms); naive fault barriers hurt (48.5 ms).
### Well–seismic tie (strata/tools/welltie.py)
- synthetic_tie (Ricker 25 Hz, bulk lag): trough at top 7/8 wells, median lag −11.5 ms, corr 0.55–0.92 → synthetic_ties.csv
- optimise_tie (differential evolution: bulk shift ±40 ms, phase ±90°, Ricker 15–45 Hz; then regularised stretch & squeeze = cubic-spline
  lag, 6 knots ±8 ms, penalties on roughness/size): corr median 0.798 → 0.817, stretch ±2 ms, lag at top median −10.6 ms,
  phase ~+18–24°, wavelet 22–27 Hz, trough 6/8; C-4AH outlier (+37 ms, neighbour-tied) → optimised_ties.csv, lag curves json.
- Method follows wtie (Tschannen et al. 2022, GPL-3: Bayesian opt + windowed-xcorr stretch/squeeze), re-implemented, not copied.
  tools/dlseis_well_tie and tools/automatic_well_tie cloned for reference (not installed: py3.7/torch-cu10.2 env).
### Tracking (strata/tools/horizon.py) — scored on the reservoir footprint (benchmark HZ metric)
| case | time MAE | coverage | bias |
|---|---|---|---|
| greedy, peak, all seeds (old) | 34.0 | 0.985 | +9.3 |
| greedy, trough, −11.5 ms | 35.9 | 0.985 | +22.9 |
| greedy, trough, −11.5, barrier p92 | 48.5 | 0.983 | +46.3 |
| greedy, trough, tied, guide ±15 | 21.7 | 0.951 | +4.4 |
| viterbi, tied, ±25 | 21.9 | 0.985 | +1.3 |
| **viterbi, tied, ±35, fault-aware** | **20.4** | 0.985 | +2.1 |
- Fault map: discontinuity (1 − best lagged NCC with neighbours, 100 ms slabs 2300–2700 ms), 1.6 s; 92nd percentile = 7 % of traces.
### Scripted swarm run with all improvements: run 20261108-185720-0ae3 → HZ time MAE 22.3 ms, coverage 98.5 %, bias +0.5
  (well verifier MAE 6.9 ms; fault verifier rejects: off-fault jumps 6.2 % of picked > 3 % guide).

## Opus 5.5 run with improved tools: run 20261108-185917-1e55 (autonomous)
- Agents: steward; well agent ‖ fault agent in parallel; well agent excluded Ricker tables, ran optimise_well_tie, chose trough;
  fault agent tried p92/p95/p93 → published p93; horizon agent: fault-aware Viterbi, trough, guide 16 → 12 ms over reworks.
- Verdicts: well agent rejected v1 narrowly (MAE 6.72 ms, one well), PASSED v2 (5.34 ms) and v3 (5.32 ms); fault agent rejected all
  (off-fault jumps 4.5 → 3.1 → 2.7 %, but "≈90 % of fault crossings without offset").
- Benchmark (final v3): HZ time MAE 24.1 ms ✗, coverage 97.8 % ✓, bias +0.8. Tokens ≈ 416k in / 22k out.
### Per-version scores (per_version_scores.json)
| run | v1 | v2 | v3 |
|---|---|---|---|
| Opus 1e55 | 23.2 | 23.5 | 24.1 |
| scripted 0ae3 | 20.9 | 22.3 | 22.3 |
→ verifier-driven reworks made horizons slightly WORSE in both runs. The fault verifier's "crossings without offset" criterion is
miscalibrated (discontinuity map includes small-throw faults/noise → ~90 % flat crossings even for good horizons) → false alarms.
This is the "verifier quality" effect (SWARM_ARCHITECTURE §7.3). Next: calibrate the fault verifier (strong/long faults only,
throw-weighted) or drop that criterion; keep the best-scoring-by-verifier version rather than the latest.

## Fix round "no cheating" (2026-11-08 night) — all checks/thresholds input-only, truth used only for reporting
- Fault verifier: throw consistency — seismic throw estimated across long fault segments by lagged NCC,
  kept only if forward/reverse/1.5× window agree (self-consistency check measured: reverse agreement 64 %, window 80 % before filter).
  Rule fixed a priori: off-fault jumps ≤ 3 % AND throw mismatch ≤ 30 %. Old "crossing without offset" (~90 % everywhere) dropped.
- Viterbi fault_mode="skip": skip fault-zone traces, cross with the reliable seismic throw (input-only eval: mismatch 0.40 → 0.31).
- Orchestrator: keeps best version by verifiers only (passes, then Σ metric/tolerance); reworks get full version history.
- Scorer/spec: input-to-truth floor 22.2 ms reported (§10.3b); tolerance unchanged 15 ms.
| run | v1 | v2 | v3 | exported |
|---|---|---|---|---|
| scripted 623a | 20.6 | 20.6 | 20.6 | 20.6 |
| Opus 7076 (selection by well MAE) | 23.3 | 24.0 | 23.5 | v2 24.0 |
| Opus 2632 (+trade-off guidance, normalised selection) | 20.4 | 21.3 | 20.4 | v2 21.3 |
Opus: 39.0 (Sonnet, old tools) → 24.1 → 21.3 ms. Verifiers still pick within ~1 ms noise of truth-best; remaining error is near the 22 ms floor.
Tokens per Opus run ≈ 380–470k in / 22k out.

## Deviation repair (user idea, 2026-11-08 late) — strata/tools/repair.py
User: where the pick starts deviating, follow the trend of the trusted picks (same high amplitude/same value),
put seeds on it and re-solve with Viterbi (reference tmp/viterbi.py: template DP, anchors hard, max jump).
Input-only diagnosis IL1120: before XL1707 pick on trough −2.1…−3.5×RMS; after the 25 ms jump amplitude ≈ −0.5…+0.3
(not even a trough); bright trough continues at 2412→2404 ms (−1.9…−2.8×RMS); no fault in the map.
Attempts (thresholds fixed a priori; truth only for reporting, scored once per attempt):
1. per-line repair applied directly, 2 passes both directions → cascade, 30–45 % traces changed, climbed onto
   brighter events. INPUT checks worse (off-fault jumps 2.1→4.4 %, 7.0→9.9 %). TRUTH 21.3→33.5, 20.6→35.8 ms. Rejected.
   (also fixed acceptance to "same amplitude" 0.6–1.6× trusted, per the user's criterion)
2. per-line repair gated by 3D jump decrease → only 7/563 and 147/4704 accepted (deviation is a 3D region). No effect.
3. ADOPTED: consensus seeds = inline AND crossline re-tracks agree ≤ 8 ms (96 % / 78 % agreement where both
   propose; mean shift −7 ms), every 5th trace → anchors with the wells; guide = horizon with consensus times;
   Viterbi re-solve (±35 ms, fault skip).
   | horizon | well MAE | off-fault jumps | throw mismatch | event amp | TRUTH MAE |
   | Opus 2632 v2 before/after | 5.81/6.44 | 0.021/0.071 | 0.47/0.33 | 1.23/1.53 | 21.3/20.7 |
   | scripted 623a before/after | 6.54/6.54 | 0.070/0.066 | 0.29/0.30 | 1.49/1.77 | 20.6/19.4 |
   IL1120 MAE 14.1 → 6.2 ms (inline1120_true_vs_agents_repair.png). Remaining: two short dips (XL~1752, ~1812)
   and east of XL1815 a large fault the fault map misses — seeds continue flat there (input can't see it).
Wired as horizon-agent tool repair_deviations (scripted plan runs it). Scripted end-to-end run 20261108-200629-9025: 19.2 ms.
Files: retracked_*.npz, retrack_seeds_*.json, retrack_input_eval.json, repair_eval.json, plots/plot_inline_true_vs_agents.py
Opus 5.5 run with repair_deviations: 20261108-201137-9b63 → HZ time MAE 18.8 ms (best so far), coverage 98.5 %, bias +2.2;
used repair_deviations every round; selected v3; no version passed both verifiers. Tokens 317k in / 22k out.
IL1120: 7.7 ms (scripted+repair 6.0 ms) — Opus v3 has a dip at XL 1727–1752 that the scripted run doesn't.
Plots: repair_error_maps.png, inline{1080,1120,1160}_true_vs_agents_repair.png, inline1120_true_vs_latest.png (plots/plot_inline_simple.py).

## Research: NCS seismic foundation model (2026-11-09) — for a horizon "probability" guide
Source: https://huggingface.co/NorskRegnesentralSTI/NCS-v1-3d-base · code https://github.com/NorskRegnesentral/NCS_models
(Apache-2.0, pushed 2026-10-28) · paper arXiv 2603.23211 (Ordoñez et al., NR).
- ViT-MAE (85 % masking), 88 M params, 12 layers, hidden 768; input 224³ crops, 16³ patches → 2,744 patch tokens;
  LieRE positional encoding (resolution-flexible). Siblings: NCS-v1-2d-base, NCS-v1-2.5d-base (best average in paper).
- Pretrained on 829 full-stack time+depth cubes (27 TB) from DISKOS (Norwegian shelf); 4D vintages excluded
  (Norne may or may not be in the corpus — seismic only, no labels, so no truth leakage either way).
- Output: features (inline, xline, time, 768) at patch spacing 16 samples (= 64 ms here); --densify 2/4/8/16 → 8/4/2/1-sample
  spacing at large compute cost (3D: densify^3 passes). Patch support 16 samples ⇒ paper's own limitation: thin bodies /
  closely spaced reflectors hard to separate. ⇒ NOT a 4-ms horizon picker; good for "which package / which level".
- Paper evaluation: frozen backbone + kNN / linear probe, 100 labels/class: package segmentation (key horizons → packages),
  salt, injectites, flatspot, Penobscot. Similarity search (mean embedding of example clicks · dot product) shown for
  faults, flatspots, truncations. Horizon tracking itself is not evaluated.
- Cost: 3D FP16 peak 578 MB/crop, 242 GMac/crop; 1024³ cube in ~12 s on an RTX 4000 (20 GB).
- Here: GPU RTX PRO 500 Blackwell laptop, 6 GB (needs cu128 torch); RAM 15 GB; cube cache 321×1001×401 (1700–3300 ms).
  Rough: ~50 crops at densify 1, ~3,000 at densify 4 (minutes). Deps pin torch~=2.4 (2.x), transformers~=4.50.2 → keep in
  its own uv env under tools/NCS_models, export features to .npy for Strata.
### NCS implementation (2026-11-09) — strata/tools/foundation.py
- torch 2.11+cu128 + NCS added to main env (explicit pytorch-cu128 index for torch/torchvision only). TF 2.21 lost GPU by
  default (libcusolver.so.11 not found; cusolver 11.7.5→11.7.3) — works with LD_LIBRARY_PATH=.venv/.../nvidia/*/lib.
- Features: NCS-v1-3d-base on cached cube, densify t4/xy2 → grid 40×125×99×768 (8 traces, 16 ms), 1,728 crops, 425 s,
  0.14 s/crop GPU (CPU-bound merge). Cache data/strata_cache/ncs3d_t4_xy2.npy (760 MB).
- Linear probe (torch L-BFGS, C=0.05) above/below tied Not 2 top, 36 non-blind wells, |Δt| 16–120 ms (468 samples).
  Leave-one-well-out boundary error (input only): NCS MAE 15.9 / median 12.2 ms vs interpolation of other wells 9.4 / 5.8.
  NCS fixes C-3H, F-2H; worse at E-4AH*, F-4H. Field P volume 2.6 s; cache ncs3d_not2_prob.npz.
- Sections (ncs_prob_sections.png): boundary steps DOWN east of XL~1965 (IL1080) and ~1820 (IL1120) where the agents stay up.
- Hybrid level check (a priori: |hz − NCS| > 40 ms, contrast ≥ 0.5, area ≥ 200, Viterbi guide = NCS boundary ±25 ms
  [LOO 80th pct 23.8]): flags 14–15 % of picked traces (33–34 regions), mean shift +30–32 ms (deeper).
  INPUT checks: well MAE unchanged; off-fault jumps 5.2→8.5 % (Opus) / 6.8→9.5 % (scripted); throw mismatch +0.02–0.03.
  (These checks rely on the fault map, which misses the large faults in question → they penalise splices there.)
  Truth score pending: reference needs the T7 drive.
- TRUTH (scored once, T7 mounted): Opus 9b63 v3 18.79 → 15.66 ms (RMSE 34.7 → 28.9); scripted 9025 19.23 → 15.90 ms.
  NCS boundary surface alone: 18.66 ms (bias +3.6). Within ±15 ms: 71 % → 73 %; better 8 % / worse 2 % of reservoir area.
  Big wins: SE block IL1215–1260 × XL1830–1960, IL1050–1080 × XL~1990. Still wrong: SE block XL1960–2150 (+60, not flagged);
  worse at reservoir edges (west edge XL1430–1470 on IL1080/1120/1160 pulled down ~80 ms; small patches IL~1050 XL1600).
  Sections: re-tracked spans inside flagged areas are spiky (alternate between levels) — needs smoothing / excursion fix.
- Fault probability maps (fault_probability_maps.png): (a) discontinuity percentile — noisy; (b) NCS probe trained on
  discontinuity ≥97th vs ≤50th pct at the Not 2 level — smooth, marks several reference fault zones (truth faults overlaid,
  display only) but blurry (8-trace grid, 64-ms patches) and lights up large non-fault areas outside the reservoir.
  Cache: data/strata_cache/ncs3d_fault_prob_not2.npz. Plots script: plots/plot_ncs_results.py.
### NCS + deviation repair (repair re-run on the NCS-corrected horizon, agent-tool defaults, window 35)
| | NCS | NCS+repair |
| Opus 9b63 TRUTH MAE/RMSE | 15.66/28.9 | 14.53/28.1 ✓ |
| scripted 9025 | 15.90/29.0 | 14.82/28.3 ✓ |
INPUT: well MAE 6.53→6.88 / 6.54→6.80; throw mismatch 0.43→0.36 / 0.32→0.33; off-fault jumps 8.5→9.0 % / 9.5→8.7 %.
Sections (ncs_repair_sections.png): inline MAE IL1080 8.7→11.3, IL1120 7.9→6.6, IL1160 9.5→11.0 — the field-wide gain comes
from elsewhere; in down-thrown blocks the repair (amplitude/continuity) partly pulls picks back up to the bright upper event,
and spikes inside flagged areas remain. Idea: keep the NCS level as a soft guide inside the repair's Viterbi in flagged areas.

## Agent tool choice with NCS (2026-11-09) — horizon agent tools: track, repair, ncs_level_check, compare_candidates, select_candidate
- Scripted baseline (track→repair→NCS→repair→fill): run 20261109-193635-2683 → 14.67 ms ✓ (HZ checks 2/2), cov 98.4 %.
- Opus 5.5 free choice: run 20261109-200906-a55c → exported v3 19.36 ms. Per version: v1 17.38 (base Viterbi → NCS → repair,
  chose the NCS candidate over the repaired one), v2 23.84 (greedy, "following the fault agent's advice to tighten"),
  v3 19.36 (greedy, min_corr 0.92). Verifiers: v1 rejected by both; v2 well ✓ fault ✗; v3 both ✓ → selected.
  Tokens 937k in / 26.5k out (≈ $4.3 at $4/$20 per M) — round 3 alone ran ~20 tracking trials.
- Lesson: the fault verifier (fault map misses large faults) pushes the agent away from NCS level corrections; the agent
  satisfies the verifiers, not the truth. Verifier quality is now the bottleneck again.
- Map comparison (v3_vs_ncs_maps.png, gridded on reference nodes; MAE slightly differs from scorer's interpolation):
  Opus v3 18.6 ms, ±15 ms 73 %, unpicked 8.5 % | scripted NCS 14.6 ms, 75 %, unpicked 3.0 %. NCS better on 16 % of area
  (SE up-thrown block IL1210–1260 × XL1850–1980; IL1040–1080 × XL1720–1800; IL~1130 × XL1740–1800), v3 better on 7 %
  (north edge IL990–1050 × XL1900–2100, small patches). Both still fail: SE block east XL1980–2150 (agents ~60 ms too
  deep; truth is an UP-thrown high at 2350–2400 ms) and the narrow down-thrown strip IL1130–1190 × XL1830–1870 (−60).

## Trend-following (dip-guided) horizon (2026-11-09) — strata/tools/dip.py, tool dip_guided_horizon
Diagnosis IL1224 (il1224_diagnosis.png): true Not 2 follows the strong trough over the SE high (~2370 ms, XL1850–2150),
ABOVE the ±35 ms window of the well-interpolated guide → every tracker was boxed into a dim zone → sawtooth.
Method: structure-tensor slopes (σ_deriv 1, smoothing (4,4,2), window 2250–2700 ms; 9 s, cached
slopes_2250_2700.npz; median planarity 0.87) → least-squares surface matching slopes, anchors = wells (w 100),
fault map relaxes slopes (×0.02), optional NCS level soft pull (w 0.05 × contrast weight), 4 re-sampling iterations
(13 s) → Viterbi snap ±10 ms; where no event, keep the surface (offset-median fill tested, worse on dip misfit, rejected).
| variant | TRUTH MAE | RMSE | bias |
| surface, wells only | 19.99 | 30.1 | −3.4 |
| snapped, wells only | 19.43 | 30.7 | −4.0 |
| surface, + NCS level | 16.85 | 23.7 | +2.7 |
| snapped, + NCS level | 15.16 | 23.3 | +3.0 |  (best RMSE so far; scripted NCS pipeline 14.67 / 28.2)
INPUT dip misfit (mean ms / frac > 4 ms): dip+NCS snapped 1.13–1.34 / 0.06–0.08; Opus v3 0.99 / 0.02; scripted NCS 2.06 / 0.068.
Sections (dip_sections.png): IL1224 follows the high's shape but ~20–30 ms (one cycle) BELOW the truth (no well /
confident NCS pin there); IL1080 8.4 ms (best), IL1120 8.6 (worse than 5.1; box excursions where snap/fill alternate),
IL1160 12.1. Sawtooth gone; remaining: cycle-level offset far from wells, box excursions between snapped/fill areas.
Smoke test as agent tool: 10.6 s; compare_candidates now includes dip_misfit_ms / dip_misfit_frac.

## Why the scripted pipeline stays best (2026-11-09) — proxy_vs_truth.csv (48 horizons, all runs + dev variants)
Spearman rank correlation of INPUT checks with TRUTH MAE (+ = check agrees with truth):
| check | all 48 | comparable 40 (trough era, excl. failed cascade) |
| well_mae | +0.14 | −0.27 |
| off_fault_jumps | −0.42 | −0.47 |
| throw_mismatch | −0.27 | −0.10 |
| dip_misfit | −0.28 | −0.52 |
| event_amp (higher=brighter) | −0.45 | −0.42 |
| ncs_disagree | +0.63 | +0.36 |
| coverage | −0.21 | −0.26 |
⇒ Smoothness/continuity/brightness checks are ANTI-correlated with accuracy in this heavily faulted field (wrong-level
horizons that stay on one reflector across unmapped faults look smooth, bright and well-tied). Only NCS-level agreement
points the right way. Agents optimise these proxies (Opus a55c: ~20 trials) → worse (Goodhart); the script ignores feedback.
Also: the scripted RECIPE was composed by me after truth-scoring each step (tool thresholds a priori, composition not) →
it is a researcher-tuned baseline, not a blind one. Example of the trap: dip+NCS offset-fill (rejected on dip misfit)
scores 14.39 ms — best of all; NOT adopted (would be choosing by truth).
Next: (1) re-design checks around level (NCS agreement, wells) not smoothness; (2) validate any check redesign on a
second, untouched target to avoid truth-tuning; (3) label the scripted recipe "researcher-tuned"; repeat runs for variance.

## Reference workflow + playbook + judging tools (2026-11-09 late)
- "scripted" renamed "reference" (alias kept). Playbook skill skills/geomodel-horizon-playbook (loaded by horizon agent).
- strata/tools/workflow.py: shared steps (track/repair/ncs/dip/fill), replay with held-out wells, method agreement.
  Tools: holdout_wells_check, method_agreement_check; candidates carry their step chain.
- Test (playbook_tooltest.json; 26 spatial well groups, 3 folds; interpolation baseline 12.1 ms at hidden wells):
| candidate | hold-out (hidden wells) | TRUTH MAE / RMSE (reporting only) |
| track | 7.9 | 20.45 / 35.3 |
| track→repair | 8.8 | 18.97 / 34.8 |
| +ncs | 11.3 | 15.90 / 29.4 |
| +repair | 9.7 | 15.07 / 28.9 |
| +fill (reference chain) | 9.7 | 15.08 / 29.0 |
| dip + NCS | 15.8 | 15.63 / 23.4 |
| dip, wells only | 12.4 | 19.36 / 30.5 |
  Spearman hold-out vs truth −0.27 → hold-out at wells is a NEAR-WELL check; the big errors are far from wells.
  Method agreement: methods agree (≤15 ms) on 22 % of picks; largest disagreement regions include survey edges.
  Playbook §3 updated accordingly (hold-out = near-well safety check, not an accuracy judge).

## User's H01 2D horizon agent (strata-agent-work.zip, 03_h01_horizon_agent) run on every inline (2026-11-11)
Code unchanged; driver outputs/h01_fullfield/run_h01_fullfield.py. Honesty changes vs the zip's eval: seeds from WELL TIES
(88 inlines within ±2 of a tie; others carried from the neighbouring inline's confident pick), never the reference.
Note: the zip's eval_h01 seeds from the reference time and uses tral_r06_faults, which is in FAULTS/ (truth tier).
Variant A = fault traces from our input fault map (vertical per section); B = tral_r06 (reference faults, not benchmark-valid).
Runtime 92 s (A) / 128 s (B); inline 1121 loops forever in variant B (30 s cap → left empty).
| variant | result | scorer MAE | coverage | map: MAE where picked / ±15 ms / area picked |
| A | DP one seed | 39.5 | 0.985 | 39.5 / 67 % / 100 % |
| A | agent confident | 36.7 | 0.926 | 35.0 / 74 % / 80 % |
| A | agent all | 37.1 | 0.983 | 34.3 / 74 % / 92 % |
| B | DP one seed | 39.9 | 0.985 | 40.0 / 67 % / 100 % |
| B | agent confident | 25.2 | 0.943 | 20.2 / 82 % / 66 % |
| B | agent all | 27.5 | 0.971 | 21.3 / 81 % / 80 % |
Maps h01_{A,B}_maps.png: strong inline striping — each inline solved independently; a wrong seed/fault decision puts
the whole line on another cycle, nothing ties neighbouring lines. Good in the well-tied west; SE block still wrong.
Fault quality matters a lot (A 37 vs B 27 ms). Idea: use H01 confident picks as anchors in a 3D solve, or run H01 on
inlines AND crosslines and keep picks where the two agree (loop closure).
