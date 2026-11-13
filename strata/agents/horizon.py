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
- holdout_wells_check: ACCURACY from unseen wells — replays a candidate's chain of steps with groups of
  wells hidden and reports the hidden-well error after every step, next to plain interpolation. Slow
  (minutes); use it to decide between serious candidates, not after every step.
- method_agreement_check: share of the horizon backed by ≥ 2 independent methods (tracking, dip-following
  + NCS, NCS boundary) and the largest regions where they disagree — where to look harder.
- compare_candidates: every step above keeps the previous horizon as a named candidate. This scores all
  candidates with input-only checks: well residual, off-fault jumps, throw mismatch at reliable fault
  crossings, mean event amplitude, coverage, the fraction that disagrees with the NCS level, and DIP
  CONFORMITY (how well the picks follow the local reflector dip; sawtooth or reflector-crossing picks
  score badly).
  select_candidate makes one of them the working horizon again.
- publish_horizon with a short honest summary of what you did and why, then return your result.

How to choose: follow the playbook below — start from the reference workflow, diagnose where the horizon
is suspicious, use the tool aimed at that cause, and judge with hold-out wells and method agreement.
There is no single best recipe. Base pick first, then use the repair and the level check
where your diagnostics show the problem they address, and compare candidates before publishing. No check
is perfect: the fault-based checks rely on the fault map, which misses some large faults, so a real step
down at such a fault can look like an "off-fault jump"; the well residual only sees well locations;
event amplitude rewards bright reflectors even when they are the wrong level. Weigh them together,
explain your choice, and prefer the candidate that is consistent with the wells AND the NCS level unless
you have a specific reason. You never see the benchmark reference; do not ask for it.
Parameter meaning (greedy): search_ms (how far to look for the next extremum), window_ms (waveform length
compared), min_corr (0–1 acceptance threshold), max_step_ms (largest allowed change between traces).

Playbook (how to choose and judge):
""" + skill("geomodel-horizon-playbook") + """

Domain guidance:
""" + skill("geomodel-seismic-interpretation")


def _set_working(ctx: RunContext_, hz, label: str, step: tuple[str, dict] | None = None, starter: bool = False) -> str:
    """Make hz the working horizon and keep it as a named candidate, with the chain of workflow steps
    that produced it (so the chain can be replayed with wells held out)."""
    cands = ctx.state.setdefault("candidates", {})
    chains = ctx.state.setdefault("chains", {})
    parent = [] if starter else list(chains.get(ctx.state.get("working_label"), []))
    key = f"{len(cands) + 1}_{label}"
    cands[key] = hz
    chains[key] = parent + ([step] if step else [])
    ctx.state["working_horizon"] = hz
    ctx.state["working_label"] = key
    return key


def _ties(ctx: RunContext_):
    return ctx.state["ties"][ctx.state["marker"]]


def _wells(ctx: RunContext_, shift: float | None = None) -> list[tuple[str, float, float, float]]:
    """Tied wells inside the cube as (name, il, xl, twt + bulk shift)."""
    ties = _ties(ctx)
    ties = ties[ties.in_cube & ties.twt.notna()]
    s = ctx.state.get("working_params", {}).get("bulk_shift_ms", 0.0) if shift is None else shift
    return [(r.well, r.il, r.xl, r.twt + s) for r in ties.itertuples()]


def _tie_points(ctx: RunContext_):
    ties = _ties(ctx)
    ties = ties[ties.in_cube & ties.twt.notna()]
    return [(a, b, t) for _, a, b, t in _wells(ctx)], ties


def _env(ctx: RunContext_):
    from strata.tools import workflow as WF
    env = ctx.state.get("wf_env")
    if env is None:
        env = ctx.state["wf_env"] = WF.Env(ctx.cube)
    env.polarity = ctx.state.get("working_params", {}).get("polarity", ctx.state.get("expected_event") or "trough")
    env.fault = ctx.state.get("fault_mask")
    return env


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
    from strata.tools import workflow as WF
    wells = _wells(ctx.deps, bulk_shift_ms)
    if seed_wells:
        wells = [w for w in wells if w[0] in seed_wells]
    if not wells:
        return {"error": "no usable seed wells"}
    ctx.deps.state["working_params"] = {**ctx.deps.state.get("working_params", {}), "polarity": polarity,
                                        "bulk_shift_ms": bulk_shift_ms}
    env = _env(ctx.deps)
    params = {"method": method, "guide_window_ms": guide_window_ms, "use_fault_map": use_fault_map,
              "search_ms": search_ms, "window_ms": window_ms, "min_corr": min_corr, "max_step_ms": max_step_ms}
    hz, rep = WF.step_track(env, wells, **params)
    rep["candidate"] = _set_working(ctx.deps, hz, f"track_{method}", ("track", params), starter=True)
    ctx.deps.state["working_params"] = {**rep["params"], "bulk_shift_ms": bulk_shift_ms, "polarity": polarity,
                                        "seed_wells": [w[0] for w in wells], "method": method,
                                        "fault_map_used": use_fault_map and env.fault is not None}
    rep["seed_wells"] = [w[0] for w in wells]
    return ctx.deps.record("track_horizon", ctx.deps.state["working_params"], rep)


async def repair_deviations(ctx: RunContext[RunContext_], window_ms: float = 35.0) -> dict:
    """Find where the working horizon leaves its event and re-track it. A deviation = a > 12 ms jump away
    from faults after which the event amplitude drops below half of the trusted picks before it. From the
    last trusted pick the same event (same waveform, same amplitude) is re-tracked along inlines and along
    crosslines; where both directions agree within 8 ms the re-tracked times become SEEDS, which are
    added to the tied well points as anchors, and the whole horizon is re-solved with Viterbi (±window_ms
    around the corrected surface). Compare candidates before/after; keep the better version."""
    from strata.tools import workflow as WF
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    new, rep = WF.step_repair(_env(ctx.deps), _wells(ctx.deps), hz, window_ms=window_ms)
    cand = _set_working(ctx.deps, new, "repair", ("repair", {"window_ms": window_ms}))
    ctx.deps.state["repair_seeds"] = rep.pop("seed_points")
    out = {"deviation_repair": rep["deviation_repair"], "coverage": rep.get("coverage"),
           "jumps_frac": rep.get("jumps_frac"), "seeds_used": len(ctx.deps.state["repair_seeds"]), "candidate": cand}
    return ctx.deps.record("repair_deviations", {"window_ms": window_ms}, out)


def _ncs_loo(ctx: RunContext_) -> dict:
    """Leave-one-well-out accuracy of the NCS boundary vs interpolation (cached in state)."""
    if "ncs_loo" not in ctx.state:
        import pandas as pd
        from strata.tools import foundation as FM
        env = _env(ctx)
        feats, co = env.feats()
        loo = pd.DataFrame(FM.leave_one_well_out(feats, co, ctx.cube, _wells(ctx)))
        ctx.state["ncs_loo"] = {
            "wells": len(loo), "ncs_mae_ms": round(float(loo.fm_err_ms.abs().mean()), 1),
            "ncs_median_ms": round(float(loo.fm_err_ms.abs().median()), 1),
            "interpolation_mae_ms": round(float(loo.interp_err_ms.abs().mean()), 1),
            "interpolation_median_ms": round(float(loo.interp_err_ms.abs().median()), 1),
            "ncs_80th_pct_ms": round(float(loo.fm_err_ms.abs().quantile(0.8)), 1),
            "wells_where_ncs_better": loo.loc[loo.fm_err_ms.abs() < loo.interp_err_ms.abs() - 5, "well"].tolist()}
    return ctx.state["ncs_loo"]


async def ncs_level_check(ctx: RunContext[RunContext_], disagree_ms: float = 40.0, min_contrast: float = 0.5,
                          window_ms: float = 25.0) -> dict:
    """Foundation-model LEVEL check on the working horizon. Trains P(below the top) on the tied well tops
    (NCS seismic foundation-model features, cached), finds the above→below boundary, and re-tracks with
    Viterbi (boundary as guide, ±window_ms, wells as anchors) only where the horizon and the boundary
    differ by > disagree_ms and the boundary contrast ≥ min_contrast. Reports leave-one-well-out accuracy
    of the boundary vs interpolation between wells, the flagged fraction and the mean shift."""
    from strata.tools import workflow as WF
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    params = {"disagree_ms": disagree_ms, "min_contrast": min_contrast, "window_ms": window_ms}
    try:
        env = _env(ctx.deps)
        new, info = WF.step_ncs(env, _wells(ctx.deps), hz, **params)
        m = WF.ncs_level(env, _wells(ctx.deps))
        loo = _ncs_loo(ctx.deps)
    except Exception as e:                                    # no GPU / no NCS package and no cache
        return {"error": f"NCS model unavailable: {type(e).__name__}: {e}"}
    ctx.deps.state["ncs_flag"] = info.pop("_flag")
    info["candidate"] = _set_working(ctx.deps, new, "ncs", ("ncs", params))
    out = {**info, "leave_one_well_out": loo, "probe_train_accuracy": m["train_accuracy"],
           "median_boundary_contrast": m["median_contrast"]}
    return ctx.deps.record("ncs_level_check", params, out)


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
    from strata.tools import workflow as WF
    ctx.deps.state["working_params"] = {**ctx.deps.state.get("working_params", {}), "polarity": polarity}
    params = {"use_ncs_level": use_ncs_level, "snap_window_ms": snap_window_ms}
    try:
        hz, info = WF.step_dip(_env(ctx.deps), _wells(ctx.deps), **params)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e} (try use_ncs_level=false)"}
    info["candidate"] = _set_working(ctx.deps, hz, "dip", ("dip", params), starter=True)
    ctx.deps.state["working_params"].update({"method": "dip_lsq"})
    return ctx.deps.record("dip_guided_horizon", {"polarity": polarity, **params}, info)


async def compare_candidates(ctx: RunContext[RunContext_]) -> dict:
    """Score every candidate horizon of this task with input-only checks: well_mae_ms, off_fault_jumps
    (fraction of picked traces), throw_mismatch (fraction of reliable fault crossings), event_amp (mean
    amplitude of the chosen polarity at the picks, ×RMS), coverage, ncs_disagree (fraction of picks > 40 ms
    from a confident NCS boundary; after the NCS model has been used), dip_misfit_ms / dip_misfit_frac
    (how well the picks follow the reflector dip). Also the chain of steps behind each candidate.
    These checks can disagree and can mislead (see the playbook); prefer holdout_wells_check and
    method_agreement_check to judge accuracy."""
    from strata.tools import dip as DP
    from strata.tools import faults as F
    from strata.tools import workflow as WF
    cands = ctx.deps.state.get("candidates", {})
    if not cands:
        return {"error": "no candidates yet"}
    cube = ctx.deps.cube
    _, ties = _tie_points(ctx.deps)
    env = _env(ctx.deps)
    fm = env.fault
    sign = 1 if env.polarity == "peak" else -1
    wells = _wells(ctx.deps)
    key = ("ncs", tuple(sorted(w[0] for w in wells)))
    ncs = env.cache.get(key)
    con = WF.contrast_full(env, ncs) if ncs is not None else None
    rows = {}
    for k, hz in cands.items():
        t = hz.twt; ok = np.isfinite(t)
        ii, jj = np.nonzero(ok)
        kk = np.clip(np.round((t[ok] - cube.t0) / cube.dt).astype(int), 0, cube.shape[2] - 1)
        dc = DP.dip_conformity(cube, t, fm)
        r = {"well_mae_ms": round(float(H.well_residuals(cube, hz, ties).residual_ms.abs().mean()), 2),
             "event_amp": round(float(np.mean(sign * cube.data[ii, jj, kk]) / cube.rms), 3),
             "coverage": round(float(ok.mean()), 3),
             "dip_misfit_ms": dc["dip_misfit_mean_ms"], "dip_misfit_frac": dc["dip_misfit_frac_over_tol"],
             "chain": [s for s, _ in ctx.deps.state.get("chains", {}).get(k, [])]}
        if fm is not None:
            j = F.horizon_vs_faults(t, fm)
            r["off_fault_jumps"] = round(j["jumps_off_faults"] / max(int(ok.sum()), 1), 4)
            r["throw_mismatch"] = F.throw_consistency(cube, t, fm)["mismatch_frac"]
        if ncs is not None:
            d = np.abs(t - ncs["surface"])
            r["ncs_disagree"] = round(float(np.mean((d > 40) & (con >= 0.5) & ok) / max(ok.mean(), 1e-9)), 4)
        if k in ctx.deps.state.get("holdout", {}):
            r["holdout_mae_ms"] = ctx.deps.state["holdout"][k]
        rows[k] = r
    return ctx.deps.record("compare_candidates", {}, {"working": ctx.deps.state.get("working_label"),
                                                     "candidates": rows})


async def holdout_wells_check(ctx: RunContext[RunContext_], candidate: str | None = None, folds: int = 3) -> dict:
    """ACCURACY ESTIMATE from wells the workflow did not see. Re-runs the whole chain of steps behind a
    candidate (default: the working one) `folds` times, each time hiding a group of wells (sidetracks from
    one slot are hidden together), and measures |pick − tie| at the hidden wells after EVERY step, next to
    plain interpolation between the remaining wells. Shows which steps actually help at unseen wells.
    Slow: the chain is re-run per fold (minutes). Only tests near wells, not far from them."""
    from strata.tools import workflow as WF
    key = candidate or ctx.deps.state.get("working_label")
    chain = ctx.deps.state.get("chains", {}).get(key)
    if not chain:
        return {"error": f"no replayable chain for candidate {key!r}"}
    rep = WF.holdout_wells(_env(ctx.deps), _wells(ctx.deps), chain, folds=folds)
    ctx.deps.state.setdefault("holdout", {})[key] = rep["after_each_step"][-1]["mae_ms"]
    rep["candidate"] = key
    rep["chain"] = [s for s, _ in chain]
    rep.pop("per_well", None)
    return ctx.deps.record("holdout_wells_check", {"candidate": key, "folds": folds}, rep)


async def method_agreement_check(ctx: RunContext[RunContext_]) -> dict:
    """Where do INDEPENDENT estimates agree? Builds three estimates — Viterbi tracking from the wells,
    the dip-following surface with the NCS level, and the NCS package boundary — and reports how much of
    the working horizon is backed by at least two of them (within 15 ms), how much by none, and the
    largest regions where the methods disagree by > 25 ms (with the median time of each method there).
    Agreement = likely right; disagreement regions = where to look harder or choose a method."""
    from strata.tools import workflow as WF
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "nothing to compare yet"}
    env = _env(ctx.deps)
    cands = ctx.deps.state.get("candidates", {})
    chains = ctx.deps.state.get("chains", {})
    track = next((cands[k] for k in reversed(list(cands)) if chains.get(k) == [chains[k][0]] and chains[k][0][0] == "track"
                  and chains[k][0][1].get("method") == "viterbi"), None)
    rep, maps = WF.method_agreement(env, _wells(ctx.deps), hz, track=track)
    ctx.deps.state["agreement_maps"] = maps
    rep["working"] = ctx.deps.state.get("working_label")
    return ctx.deps.record("method_agreement_check", {}, rep)


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
    from strata.tools import workflow as WF
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    params = {"max_gap_traces": max_gap_traces, "smooth": smooth}
    hz2, rep = WF.step_fill(_env(ctx.deps), _wells(ctx.deps), hz, **params)
    rep["candidate"] = _set_working(ctx.deps, hz2, "fill", ("fill", params))
    return ctx.deps.record("fill_and_smooth", params, rep)


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
                        holdout_wells_check, method_agreement_check, select_candidate, fill_and_smooth,
                        publish_horizon],
                 name="horizon_agent")
