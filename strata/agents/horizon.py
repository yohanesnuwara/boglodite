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

Workflow:
1. seed_report: for each tied well, the nearest peak and trough to the (seismic-tied) top time, a
   polarity suggestion, and the EXPECTED EVENT published by the well agent from synthetic seismograms.
   The synthetic-tie event outranks amplitude heuristics. Decide the polarity.
2. track_horizon. method='viterbi' (global, passes through the tied tops, follows fault throws when the
   fault map is available) is usually more robust than 'greedy' region growing, which can leak onto
   another reflector across faults. guide_window_ms bounds how far the horizon may depart from the
   surface through the tied tops. Read the QC: coverage, confidence and JUMPS (neighbouring picks
   > 12 ms apart = cycle skips or faults). Compare alternatives if useful.
3. fill_and_smooth small gaps (do not fill large areas).
4. publish_horizon with a short honest summary, then return your result.
Trade-off to know (measured on this data with input-only checks): with the fault map, Viterbi crosses
faults using the seismic throw, but only if the true event across the fault lies inside the candidate
window — a narrow guide_window_ms (≤ 16 ms) keeps the horizon on the smooth well-guide surface, which
reduces off-fault jumps yet prevents following fault throws (the fault verifier's second test). In a
comparison, ±35 ms gave better throw consistency (0.31 vs 0.33 mismatch) and a lower well residual
(6.1 vs 6.7 ms) than ±25 ms. Balance both fault tests and the well residual; do not optimise one alone.
Parameter meaning: search_ms (how far to look for the next extremum), window_ms (waveform length
compared), min_corr (0–1 acceptance threshold), max_step_ms (largest allowed change between traces).
Stricter thresholds give fewer jumps but lower coverage.

Domain guidance:
""" + skill("geomodel-seismic-interpretation")


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
    ctx.deps.state["working_horizon"] = hz
    ctx.deps.state["working_params"] = {**rep["params"], "bulk_shift_ms": bulk_shift_ms,
                                        "seed_wells": ties.well.tolist(), "method": method,
                                        "fault_map_used": fmask is not None}
    rep["seed_wells"] = ties.well.tolist()
    return ctx.deps.record("track_horizon", ctx.deps.state["working_params"], rep)


async def fill_and_smooth(ctx: RunContext[RunContext_], max_gap_traces: int = 4, smooth: int = 0) -> dict:
    """Fill holes up to max_gap_traces wide in the working horizon; optional median smoothing (odd size)."""
    hz = ctx.deps.state.get("working_horizon")
    if hz is None:
        return {"error": "track first"}
    hz2, rep = H.fill_gaps(hz, max_gap_traces, smooth)
    ctx.deps.state["working_horizon"] = hz2
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
    def seeds(R):
        rep = tool_return(R, "seed_report") or {}
        wells = [w["well"] for w in rep.get("wells", [])]
        if rework:
            return wells                                    # scripted fix: seed from every tied well
        import pandas as pd
        return _spread_seeds(pd.DataFrame(rep.get("wells", [])), 3)

    def track_args(R):
        sr = tool_return(R, "seed_report") or {}
        pol = sr.get("expected_event_from_well_tie") or sr.get("polarity", {}).get("suggested", "peak")
        return {"polarity": pol, "method": "viterbi", "guide_window_ms": 35.0}

    p = plan([("seed_report", {}), ("track_horizon", track_args), ("fill_and_smooth", {"max_gap_traces": 4}),
              ("publish_horizon", {"name": name, "summary": "Scripted track: suggested polarity, default parameters."})],
             lambda R: {"horizon_object": (tool_return(R, "publish_horizon") or {}).get("published", ""),
                        "summary": "scripted", "polarity": track_args(R)["polarity"], "seed_wells": [],
                        "params": {}})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=HorizonResult,
                 instructions=INSTRUCTIONS, tools=[seed_report, track_horizon, fill_and_smooth, publish_horizon],
                 name="horizon_agent")
