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
1. seed_report: for each tied well, the nearest peak and trough to the tied time; a polarity suggestion.
   Decide the polarity and whether a bulk shift between tie time and the reflector is needed.
2. track_horizon with seed wells of your choice (at least 3, spread across the field; using all tied
   wells is allowed) and tracking parameters. Read the QC: coverage, confidence, and JUMPS (neighbouring
   picks > 12 ms apart = cycle skips or crossing unrecognised faults). Iterate parameters if needed.
3. fill_and_smooth small gaps (do not fill large areas).
4. publish_horizon with a short honest summary, then return your result.
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
    d = {"polarity": rep, "wells": df_brief(snapped)}
    return ctx.deps.record("seed_report", {"search_ms": search_ms}, d)


async def track_horizon(ctx: RunContext[RunContext_], polarity: str, seed_wells: list[str] | None = None,
                        search_ms: float = 8.0, window_ms: float = 24.0, min_corr: float = 0.75,
                        max_step_ms: float = 8.0, bulk_shift_ms: float = 0.0) -> dict:
    """Track the horizon from the chosen seed wells (all tied wells if omitted). Seeds sit at the
    tied time + bulk_shift_ms and snap to the nearest extremum of `polarity` ('peak'|'trough')."""
    ties = _ties(ctx.deps)
    ties = ties[ties.in_cube & ties.twt.notna()]
    if seed_wells:
        ties = ties[ties.well.isin(seed_wells)]
    if ties.empty:
        return {"error": "no usable seed wells"}
    seeds = [(r.il, r.xl, r.twt + bulk_shift_ms) for r in ties.itertuples()]
    hz, rep = H.track(ctx.deps.cube, seeds, polarity, search_ms, window_ms, min_corr, max_step_ms)
    ctx.deps.state["working_horizon"] = hz
    ctx.deps.state["working_params"] = {**rep["params"], "bulk_shift_ms": bulk_shift_ms,
                                        "seed_wells": ties.well.tolist()}
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
        rep = (tool_return(R, "seed_report") or {}).get("polarity", {})
        return {"polarity": rep.get("suggested", "peak"), "seed_wells": seeds(R)}

    p = plan([("seed_report", {}), ("track_horizon", track_args), ("fill_and_smooth", {"max_gap_traces": 4}),
              ("publish_horizon", {"name": name, "summary": "Scripted track: suggested polarity, default parameters."})],
             lambda R: {"horizon_object": (tool_return(R, "publish_horizon") or {}).get("published", ""),
                        "summary": "scripted", "polarity": track_args(R)["polarity"], "seed_wells": seeds(R),
                        "params": {}})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=HorizonResult,
                 instructions=INSTRUCTIONS, tools=[seed_report, track_horizon, fill_and_smooth, publish_horizon],
                 name="horizon_agent")
