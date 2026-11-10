"""Horizon agent: seeds from well ties, chooses polarity and tracking parameters, tracks,
fills gaps and publishes HorizonPicks with per-trace confidence."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from strata.agents.common import RunContext_, df_brief, make_model, plan, skill, tool_return
from strata.tools import horizon as H


class HorizonResult(BaseModel):
    horizon_object: str
    summary: str
    polarity: str
    seed_wells: list[str]
    params: dict = {}


INSTRUCTIONS = """You are the HORIZON AGENT of a reservoir-modelling agent team on the Norne field.
You interpret one seismic horizon in TWT (ms) on the 2001 3D cube (IL 970–1290, XL 1300–2300, 4 ms),
seeded from the well agent's ties of the requested formation top. You cannot see the well agent's
verification until it is published; you will be told its findings if it rejects your horizon.

Tools (you choose which to use, in which order, and how often):
- seed_report: for each tied well, the nearest peak and trough to the (seismic-tied) top time, a polarity
  suggestion, and the EXPECTED EVENT from the well agent's synthetic seismograms. The synthetic-tie event
  outranks amplitude heuristics. Decide the polarity first.
- track_horizon: the base pick. method='viterbi' (global, passes through the tied tops, follows fault
  throws when the fault map is available) is usually more robust than 'greedy' region growing, which can
  leak onto another reflector across faults. guide_window_ms bounds how far the horizon may depart from
  the surface through the tied tops: narrow (≤ 16 ms) gives fewer jumps but cannot follow fault throws;
  wide (~35 ms) follows throws better.
- repair_deviations: CONTINUITY repair. Finds where picks leave their event (a jump away from faults after
  which the event amplitude fades), re-tracks the same event from the last trusted pick, keeps seeds
  where inline and crossline re-tracks agree, and re-solves with Viterbi. Good for local cycle skips and
  leaks. Risk: it trusts amplitude and continuity, so in a down-thrown fault block it can pull the pick
  back up onto a bright, continuous reflector of the upper block.
- ncs_level_check: LEVEL check with a seismic foundation model (NCS, pretrained on Norwegian-shelf seismic).
  A probe trained on the tied well tops gives P(below the top) in 3D; its above→below boundary marks the
  stratigraphic level from rock character, so it can step down across large faults that the fault map
  misses. Where the horizon and that boundary disagree by > disagree_ms and the boundary is confident,
  the horizon is re-tracked with the boundary as guide; elsewhere it is kept. It reports its accuracy at
  held-out wells (leave-one-well-out) next to simple interpolation between wells: it is LESS precise than
  the wells (coarse: 64-ms patches, 16-ms grid), so use it for the level, not for precision. Risks: it is
  weakest at the edges of the field far from wells, and re-tracked areas can be spiky (alternating between
  two levels) — a repair_deviations pass afterwards can clean that, but see the repair's risk above.
- dip_guided_horizon: TREND-FOLLOWING pick. Measures the reflector dip everywhere and solves for the
  surface that follows it through the tied tops (relaxed at mapped faults, softly pulled to the NCS level
  where confident), then snaps it to the event. Use it where picks cut across dipping reflectors or zig-zag
  between cycles (sawtooth), typically far from wells. Risks: away from wells and without a confident NCS
  level it can sit one cycle off, parallel to the right event; it does not create fault throws by itself.
  It can also serve as a better base than track_horizon, followed by the level check or repair.
- fill_and_smooth: fill small gaps (do not fill large areas).
- compare_candidates: every step above keeps the previous horizon as a named candidate. This scores all
  candidates with input-only checks: well residual, off-fault jumps, throw mismatch at reliable fault
  crossings, mean event amplitude, coverage, the fraction that disagrees with the NCS level, and DIP
  CONFORMITY (how well the picks follow the local reflector dip; sawtooth or reflector-crossing picks
  score badly).
  select_candidate makes one of them the working horizon again.
- publish_horizon with a short honest summary of what you did and why, then return your result.

How to choose: there is no single best recipe. Base pick first, then use the repair and the level check
where your diagnostics show the problem they address, and compare candidates before publishing. No check
is perfect: the fault-based checks rely on the fault map, which misses some large faults, so a real step
down at such a fault can look like an "off-fault jump"; the well residual only sees well locations;
event amplitude rewards bright reflectors even when they are the wrong level. Weigh them together,
explain your choice, and prefer the candidate that is consistent with the wells AND the NCS level unless
you have a specific reason. You never see the benchmark reference; do not ask for it.
Parameter meaning (greedy): search_ms (how far to look for the next extremum), window_ms (waveform length
compared), min_corr (0–1 acceptance threshold), max_step_ms (largest allowed change between traces).

Domain guidance:
""" + skill("geomodel-seismic-interpretation")


def _set_working(ctx: RunContext_, hz, label: str) -> str:
    """Make hz the working horizon and keep it as a named candidate."""
    cands = ctx.state.setdefault("candidates", {})
    key = f"{len(cands) + 1}_{label}"
    cands[key] = hz
    ctx.state["working_horizon"] = hz
    ctx.state["working_label"] = key
    return key


def _ties(ctx: RunContext_):
    return ctx.state["ties"][ctx.state["marker"]]


async def seed_report(ctx: RunContext[RunContext_], search_ms: float = 16.0) -> dict:
    """Nearest peak/trough to the tied time at every tied well inside the cube, and a polarity report."""
    snapped = H.snap_seeds(ctx.deps.cube, _ties(ctx.deps), search_ms)
    ctx.deps.state["snapped"] = snapped
    rep = H.polarity_report(snapped)
    d = {"polarity": rep, "expected_event_from_well_tie": ctx.deps.state.get("expected_event"),
         "wells": df_brief(snapped)}
    return ctx.deps.record("seed_report", {"search_ms": search_ms}, d)


async def track_horizon(ctx: RunContext[RunContext_], polarity: str, method: str = "viterbi",
                        seed_wells: list[str] | None = None, guide_window_ms: float = 30.0,
                        use_fault_map: bool = True, search_ms: float = 8.0, window_ms: float = 24.0,
                        min_corr: float = 0.75, max_step_ms: float = 8.0, bulk_shift_ms: float = 0.0) -> dict:
    """Track the horizon from the tied well tops (all tied wells if seed_wells is omitted), at the tied
    time + bulk_shift_ms, on events of `polarity` ('peak'|'trough').
    method='viterbi': global optimal path along every inline then crossline through candidate events
      within ±guide_window_ms of a smooth surface through the tied tops; passes through the tied points.
      With use_fault_map and a published FaultMap it skips fault zones and crosses each fault using
      the seismic throw estimated across it (when that estimate is self-consistent); fault-zone traces
      stay unpicked.
    method='greedy': seeded region growing by trace-to-trace correlation (search_ms, window_ms, min_corr,
      max_step_ms), also restricted to ±guide_window_ms of the tied-top surface (0 = unguided)."""
    ties = _ties(ctx.deps)
    ties = ties[ties.in_cube & ties.twt.notna()]
    if seed_wells:
        ties = ties[ties.well.isin(seed_wells)]
    if ties.empty:
        return {"error": "no usable seed wells"}
    seeds = [(r.il, r.xl, r.twt + bulk_shift_ms) for r in ties.itertuples()]
    cube = ctx.deps.cube
    guide = H.guide_surface(cube, seeds) if guide_window_ms > 0 else None
    fmask = ctx.deps.state.get("fault_mask") if use_fault_map else None
    if method == "viterbi":
        hz, rep = H.viterbi_track(cube, guide if guide is not None else H.guide_surface(cube, seeds), polarity,
                                  window_ms=guide_window_ms or 30.0, fault=fmask, anchors=seeds,
                                  fault_mode="skip" if fmask is not None else "relax")
    else:
        hz, rep = H.track(cube, seeds, polarity, search_ms, window_ms, min_corr, max_step_ms,
                          guide=guide, guide_window_ms=guide_window_ms)
    rep["candidate"] = _set_working(ctx.deps, hz, f"track_{method}")
    ctx.deps.state["working_params"] = {**rep["params"], "bulk_shift_ms": bulk_shift_ms,
                                        "seed_wells": ties.well.tolist(), "method": method,
                                        "fault_map_used": fmask is not None}
    rep["seed_wells"] = ties.well.tolist()
    return ctx.deps.record("track_horizon", ctx.deps.state["working_params"], rep)


async def repair_deviations(ctx: RunContext[RunContext_], window_ms: float = 35.0) -> dict:
    """Find where the working horizon leaves its event and re-track it. A deviation = a > 12 ms jump away
    from faults after which the event amplitude drops below half of the trusted picks before it. From the
    last trusted pick the same event (same waveform, same amplitude) is re-tracked along inlines and along
    crosslines; where both directions agree within 8 ms the re-tracked times become SEEDS, which are
    added to the tied well points as anchors, and the whole horizon is re-solved with Viterbi (±window_ms
    around the corrected surface). Compare the verifier numbers before/after; keep the better version."""
    from strata.tools import repair as RP
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    wp = ctx.deps.state.get("working_params", {})
    ties = _ties(ctx.deps)
    ties = ties[ties.in_cube & ties.twt.notna()]
    wells = [(r.il, r.xl, r.twt + wp.get("bulk_shift_ms", 0.0)) for r in ties.itertuples()]
    new, rep = RP.retrack_with_seeds(ctx.deps.cube, hz, wells, wp.get("polarity", "trough"),
                                     fault=ctx.deps.state.get("fault_mask"), window_ms=window_ms)
    cand = _set_working(ctx.deps, new, "repair")
    ctx.deps.state["repair_seeds"] = rep.pop("seed_points")
    ctx.deps.state["working_params"] = {**wp, "deviation_repair": True, "repair_window_ms": window_ms}
    out = {"deviation_repair": rep["deviation_repair"], "coverage": rep.get("coverage"),
           "jumps_frac": rep.get("jumps_frac"), "seeds_used": len(ctx.deps.state["repair_seeds"]), "candidate": cand}
    return ctx.deps.record("repair_deviations", {"window_ms": window_ms}, out)


def _tie_points(ctx: RunContext_) -> list[tuple[float, float, float]]:
    ties = _ties(ctx)
    ties = ties[ties.in_cube & ties.twt.notna()]
    shift = ctx.state.get("working_params", {}).get("bulk_shift_ms", 0.0)
    return [(r.il, r.xl, r.twt + shift) for r in ties.itertuples()], ties


def _ncs_model(ctx: RunContext_) -> dict:
    """NCS features (cached on disk), probe trained on this run's tied tops, LOO stats — cached in state."""
    if "ncs" in ctx.state:
        return ctx.state["ncs"]
    import pandas as pd
    from strata.tools import foundation as FM
    cube = ctx.cube
    feats, co = FM.extract_features(cube)
    pts, ties = _tie_points(ctx)
    wells = [(w, a, b, t) for w, (a, b, t) in zip(ties.well, pts)]
    guide = H.guide_surface(cube, pts)
    P, B, surface, info = FM.probability_and_surface(feats, co, cube, wells, guide)
    loo = pd.DataFrame(FM.leave_one_well_out(feats, co, cube, wells))
    stats = {"wells": len(wells), "ncs_mae_ms": round(float(loo.fm_err_ms.abs().mean()), 1),
             "ncs_median_ms": round(float(loo.fm_err_ms.abs().median()), 1),
             "interpolation_mae_ms": round(float(loo.interp_err_ms.abs().mean()), 1),
             "interpolation_median_ms": round(float(loo.interp_err_ms.abs().median()), 1),
             "ncs_80th_pct_ms": round(float(loo.fm_err_ms.abs().quantile(0.8)), 1),
             "wells_where_ncs_better": loo.loc[loo.fm_err_ms.abs() < loo.interp_err_ms.abs() - 5, "well"].tolist()}
    ctx.state["ncs"] = {"surface": surface, "contrast": info["contrast_grid"], "coords": co, "loo": stats,
                        "train_accuracy": info["train_accuracy"], "median_contrast": info["median_contrast"]}
    return ctx.state["ncs"]


async def ncs_level_check(ctx: RunContext[RunContext_], disagree_ms: float = 40.0, min_contrast: float = 0.5,
                          window_ms: float = 25.0) -> dict:
    """Foundation-model LEVEL check on the working horizon. Trains P(below the top) on the tied well tops
    (NCS seismic foundation-model features, cached), finds the above→below boundary, and re-tracks with
    Viterbi (boundary as guide, ±window_ms, wells as anchors) only where the horizon and the boundary
    differ by > disagree_ms and the boundary contrast ≥ min_contrast. Reports leave-one-well-out accuracy
    of the boundary vs interpolation between wells, the flagged fraction and the mean shift."""
    from strata.tools import foundation as FM
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    try:
        m = _ncs_model(ctx.deps)
    except Exception as e:                                    # no GPU / no NCS package and no cache
        return {"error": f"NCS model unavailable: {type(e).__name__}: {e}"}
    pts, _ = _tie_points(ctx.deps)
    pol = ctx.deps.state.get("working_params", {}).get("polarity", "trough")
    new, info, flag = FM.level_check_retrack(ctx.deps.cube, hz, m["surface"], m["contrast"], m["coords"], pts,
                                             fault=ctx.deps.state.get("fault_mask"), polarity=pol,
                                             disagree_ms=disagree_ms, min_contrast=min_contrast, window_ms=window_ms)
    info["candidate"] = _set_working(ctx.deps, new, "ncs")
    ctx.deps.state["ncs_flag"] = flag
    out = {**info, "leave_one_well_out": m["loo"], "probe_train_accuracy": m["train_accuracy"],
           "median_boundary_contrast": m["median_contrast"]}
    return ctx.deps.record("ncs_level_check", {"disagree_ms": disagree_ms, "min_contrast": min_contrast,
                                               "window_ms": window_ms}, out)


async def dip_guided_horizon(ctx: RunContext[RunContext_], polarity: str = "trough", use_ncs_level: bool = True,
                             snap_window_ms: float = 10.0) -> dict:
    """TREND-FOLLOWING horizon. Measures the local reflector dip everywhere (structure tensor of the
    seismic) and solves for the surface whose slopes follow that dip, passing through the tied well tops;
    slope constraints are relaxed at mapped faults so the surface can break there; with use_ncs_level
    it is softly pulled toward the NCS level where that is confident. The surface is then snapped to the
    nearest `polarity` event within ±snap_window_ms by Viterbi; where no event is found the surface
    itself is kept. Follows the structure away from wells instead of cutting across dipping reflectors,
    so it removes sawtooth/cycle-hopping picks; it can sit one cycle off where no well or confident NCS
    level pins it, and it never creates a fault throw by itself."""
    from strata.tools import dip as DP
    cube = ctx.deps.cube
    pts, _ = _tie_points(ctx.deps)
    fm = ctx.deps.state.get("fault_mask")
    level = lw = None
    if use_ncs_level:
        try:
            m = _ncs_model(ctx.deps)
            co = m["coords"]
            ia = np.clip(np.searchsorted(co["il_sample"], np.arange(cube.shape[0])), 0, len(co["il_sample"]) - 1)
            ja = np.clip(np.searchsorted(co["xl_sample"], np.arange(cube.shape[1])), 0, len(co["xl_sample"]) - 1)
            level, lw = m["surface"], np.clip((m["contrast"][np.ix_(ia, ja)] - 0.5) / 0.5, 0, 1)
        except Exception as e:
            return {"error": f"NCS level unavailable ({type(e).__name__}); call with use_ncs_level=false"}
    surf, info = DP.lsq_horizon(cube, H.guide_surface(cube, pts), pts, fault=fm, level=level, level_weight=lw)
    snap, rep = H.viterbi_track(cube, surf, polarity, window_ms=snap_window_ms, fault=fm, anchors=pts,
                                fault_mode="skip" if fm is not None else "relax", second_window_ms=min(8.0, snap_window_ms))
    hz = H.Horizon(np.where(np.isfinite(snap.twt), snap.twt, surf).astype(np.float32),
                   np.where(np.isfinite(snap.twt), snap.conf, 0.5).astype(np.float32))
    info["snapped_frac_of_cube"] = round(float(np.isfinite(snap.twt).mean()), 3)
    info["dip_conformity"] = DP.dip_conformity(cube, hz.twt, fm)
    info["candidate"] = _set_working(ctx.deps, hz, "dip")
    ctx.deps.state.setdefault("working_params", {}).update({"polarity": polarity, "method": "dip_lsq"})
    return ctx.deps.record("dip_guided_horizon", {"polarity": polarity, "use_ncs_level": use_ncs_level,
                                                  "snap_window_ms": snap_window_ms}, info)


async def compare_candidates(ctx: RunContext[RunContext_]) -> dict:
    """Score every candidate horizon of this task with input-only checks (lower is better unless noted):
    well_mae_ms, off_fault_jumps (fraction of picked traces), throw_mismatch (fraction of reliable fault
    crossings), event_amp (mean amplitude of the chosen polarity at the picks, ×RMS, higher = brighter),
    coverage, ncs_disagree (fraction of picks > 40 ms from a confident NCS boundary; only after
    ncs_level_check has run)."""
    from strata.tools import faults as F
    cands = ctx.deps.state.get("candidates", {})
    if not cands:
        return {"error": "no candidates yet"}
    cube = ctx.deps.cube
    _, ties = _tie_points(ctx.deps)
    fm = ctx.deps.state.get("fault_mask")
    pol = ctx.deps.state.get("working_params", {}).get("polarity", "trough")
    sign = 1 if pol == "peak" else -1
    ncs = ctx.deps.state.get("ncs")
    rows = {}
    for key, hz in cands.items():
        t = hz.twt; ok = np.isfinite(t)
        ii, jj = np.nonzero(ok)
        kk = np.clip(np.round((t[ok] - cube.t0) / cube.dt).astype(int), 0, cube.shape[2] - 1)
        r = {"well_mae_ms": round(float(H.well_residuals(cube, hz, ties).residual_ms.abs().mean()), 2),
             "event_amp": round(float(np.mean(sign * cube.data[ii, jj, kk]) / cube.rms), 3),
             "coverage": round(float(ok.mean()), 3)}
        from strata.tools import dip as DP
        dc = DP.dip_conformity(cube, t, fm)
        r["dip_misfit_ms"] = dc["dip_misfit_mean_ms"]
        r["dip_misfit_frac"] = dc["dip_misfit_frac_over_tol"]
        if fm is not None:
            j = F.horizon_vs_faults(t, fm)
            r["off_fault_jumps"] = round(j["jumps_off_faults"] / max(int(ok.sum()), 1), 4)
            r["throw_mismatch"] = F.throw_consistency(cube, t, fm)["mismatch_frac"]
        if ncs is not None:
            co = ncs["coords"]
            ia = np.clip(np.searchsorted(co["il_sample"], np.arange(cube.shape[0])), 0, len(co["il_sample"]) - 1)
            ja = np.clip(np.searchsorted(co["xl_sample"], np.arange(cube.shape[1])), 0, len(co["xl_sample"]) - 1)
            con = ncs["contrast"][np.ix_(ia, ja)]
            d = np.abs(t - ncs["surface"])
            r["ncs_disagree"] = round(float(np.mean((d > 40) & (con >= 0.5) & ok) / max(ok.mean(), 1e-9)), 4)
        rows[key] = r
    return ctx.deps.record("compare_candidates", {}, {"working": ctx.deps.state.get("working_label"),
                                                     "candidates": rows})


async def select_candidate(ctx: RunContext[RunContext_], candidate: str) -> dict:
    """Make the named candidate (from compare_candidates) the working horizon."""
    cands = ctx.deps.state.get("candidates", {})
    if candidate not in cands:
        return {"error": f"unknown candidate; available: {list(cands)}"}
    ctx.deps.state["working_horizon"] = cands[candidate]
    ctx.deps.state["working_label"] = candidate
    return ctx.deps.record("select_candidate", {"candidate": candidate}, {"working": candidate})


async def fill_and_smooth(ctx: RunContext[RunContext_], max_gap_traces: int = 4, smooth: int = 0) -> dict:
    """Fill holes up to max_gap_traces wide in the working horizon; optional median smoothing (odd size)."""
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    hz2, rep = H.fill_gaps(hz, max_gap_traces, smooth)
    rep["candidate"] = _set_working(ctx.deps, hz2, "fill")
    return ctx.deps.record("fill_and_smooth", {"max_gap_traces": max_gap_traces, "smooth": smooth}, rep)


async def publish_horizon(ctx: RunContext[RunContext_], name: str, summary: str) -> dict:
    """Publish the working horizon as HorizonPicks:<name> (new version)."""
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "nothing to publish"}
    cube = ctx.deps.cube
    q = H.qc(hz)

    def write(d):
        H.save(hz, d / "horizon.npz")
        H.to_table(cube, hz).to_csv(d / "picks.csv", index=False)
        return ["horizon.npz", "picks.csv"]
    m = ctx.deps.store.publish("HorizonPicks", name, ctx.deps.agent, write, domain="time", unit="ms",
                               inputs=[ctx.deps.state.get("ties_ref", "")], confidence=q["conf_median"],
                               summary=summary[:300], tool_calls=ctx.deps.take_calls())
    ctx.deps.state["horizon_name"] = name
    return {"published": m.ref, "qc": q}


def _spread_seeds(snapped, k: int = 3) -> list[str]:
    """k wells far apart (greedy farthest-point) — the scripted seeding policy."""
    if snapped is None or snapped.empty:
        return []
    pts = snapped[["il", "xl"]].values.astype(float)
    chosen = [int(np.argmin(pts[:, 0] + pts[:, 1]))]
    while len(chosen) < min(k, len(pts)):
        d = np.min([np.hypot(*(pts - pts[c]).T) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    return snapped.well.iloc[chosen].tolist()


def build(model_spec: str, name: str, rework: bool = False) -> Agent:
    def track_args(R):
        sr = tool_return(R, "seed_report") or {}
        pol = sr.get("expected_event_from_well_tie") or sr.get("polarity", {}).get("suggested", "peak")
        return {"polarity": pol, "method": "viterbi", "guide_window_ms": 35.0}

    p = plan([("seed_report", {}), ("track_horizon", track_args), ("repair_deviations", {}),
              ("ncs_level_check", {}), ("repair_deviations", {}), ("fill_and_smooth", {"max_gap_traces": 4}),
              ("compare_candidates", {}),
              ("publish_horizon", {"name": name, "summary": "Scripted track: suggested polarity, default parameters."})],
             lambda R: {"horizon_object": (tool_return(R, "publish_horizon") or {}).get("published", ""),
                        "summary": "scripted", "polarity": track_args(R)["polarity"], "seed_wells": [],
                        "params": {}})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=HorizonResult,
                 instructions=INSTRUCTIONS, tools=[seed_report, track_horizon, dip_guided_horizon, repair_deviations, ncs_level_check, compare_candidates,
                        select_candidate, fill_and_smooth, publish_horizon],
                 name="horizon_agent")
