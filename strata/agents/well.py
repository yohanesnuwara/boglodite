"""Well agent: ties formation tops to seismic time (producer) and verifies horizon picks
against those ties (verifier)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from strata.agents.common import RunContext_, df_brief, make_model, plan, skill, tool_return
from strata.tools import horizon as H
from strata.tools import welltie as WT


class WellTieResult(BaseModel):
    marker: str
    ties_object: str
    wells_tied: int
    summary: str
    concerns: list[str] = []


class HorizonVerdict(BaseModel):
    passed: bool
    summary: str
    findings: list[str] = []
    bad_wells: list[str] = []
    suggested_fix: str = ""


ROLE = """You are the WELL AGENT of a reservoir-modelling agent team on the Norne field (ED50/UTM32N,
TVDSS m positive down, TWT ms). Blind wells have been withheld from you; never ask for them.
Time–depth tables are given per well with a 'source' name (td_vsp = VSP, TIME-CKS = checkshots,
td_par / test:td_par = calibrated time–depth pairs, *_prop = proposals, Ricker synthetics…).
Several tables per well give different times; the spread between them is your tie uncertainty."""

TIE_INSTRUCTIONS = ROLE + """

TASK: tie the formation top requested in the study brief to seismic time at every well.
1. Use list_markers to find the marker(s) that correspond to the requested surface. Picks exist in a
   2002 and a 2006 generation with different definitions; choose deliberately and say why.
2. qc_td_tables to find inconsistent time–depth tables; exclude real outliers in the tie.
3. tie_marker for the chosen marker (checkshot / time–depth conversion of the top).
4. optimise_well_tie: synthetic seismograms at wells with sonic + density — this is the real well–
   seismic tie. It gives the time correction at the top, the polarity and the EVENT (peak or trough)
   the top falls on. Judge the ties: low correlation or a lag far from the others (e.g. a well tied
   through a neighbour's table) is untrustworthy.
5. publish_ties with apply_seismic_tie=True, your untrusted_tie_wells and the expected_event, then
   return the result. Note concerns (neighbour-tied wells, low correlations, polarity disagreements).

Domain guidance:
""" + skill("geomodel-well-tie-depth-conversion")

VERIFY_INSTRUCTIONS = ROLE + """

TASK: VERIFY a horizon published by the horizon agent. You did not produce it; judge it only with your
well ties. Call check_horizon_at_wells and decide pass/fail.
Guidance: one seismic sample is 4 ms. The tie itself is uncertain: 'tie_uncertainty_ms' is the median
spread between time–depth tables. A good horizon has mean absolute residual ≤ max(8 ms, tie uncertainty)
and no well beyond 3× that, unless the well's own table spread explains it. A consistent bias (all
residuals one sign) suggests the horizon follows a neighbouring loop or the tie is shifted. Unpicked wells inside the cube
are a coverage problem. In findings, cite the numbers; list offending wells in bad_wells; in
suggested_fix tell the horizon agent concretely what to change (seed wells, polarity, bulk shift,
search window, correlation threshold).

Domain guidance:
""" + skill("geomodel-model-qc").split("## 3.")[0]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


# ── producer tools ───────────────────────────────────────────────────────────
async def list_markers(ctx: RunContext[RunContext_], pattern: str = "") -> list[dict]:
    """Formation markers in the well picks (optionally filtered by a substring), with the number
    of wells and their TVDSS range, ordered by depth."""
    df = ctx.deps.wells.marker_summary(pattern)
    ctx.deps.record("list_markers", {"pattern": pattern}, {"n_markers": len(df)})
    return df_brief(df, 80)


async def qc_td_tables(ctx: RunContext[RunContext_], ref_tvdss: float = 2550.0, outlier_ms: float = 40.0) -> dict:
    """Compare every time–depth table with all other wells' tables at a common depth (its deepest
    point, capped at ref_tvdss). Large deviations reveal time shifts or datum errors in a table."""
    q = ctx.deps.wells.td_table_qc(ref_tvdss, outlier_ms)
    ctx.deps.state["td_qc"] = q
    d = {"tables": len(q), "outliers": q.loc[q.outlier, "table"].tolist(),
         "median_abs_deviation_ms": round(float(q.deviation_ms.abs().median()), 2),
         "worst": df_brief(q[["table", "eval_tvdss", "twt", "others_median", "n_others", "deviation_ms", "outlier"]], 12),
         "note": "depth column of the TD files is TVD below KB (TVDSS = depth − KB), not MD as the header claims"}
    return ctx.deps.record("qc_td_tables", {"ref_tvdss": ref_tvdss, "outlier_ms": outlier_ms}, d)


async def tie_marker(ctx: RunContext[RunContext_], marker: str, td_preference: list[str] | None = None,
                     exclude_tables: list[str] | None = None) -> dict:
    """Convert `marker` to seismic time at every well: TVDSS → TWT with the well's own time–depth
    tables, or (if none covers the top) with the nearest wells' tables ('neighbour' method).
    exclude_tables: 'well:source' keys to ignore (e.g. outliers from qc_td_tables).
    Returns per-well results (method, source, TWT, spread between tables) and statistics."""
    ties = ctx.deps.wells.tie_marker(marker, ctx.deps.cube, td_preference, exclude_tables=tuple(exclude_tables or ()))
    if ties.empty:
        return ctx.deps.record("tie_marker", {"marker": marker}, {"error": "no wells with this marker and a TD table"})
    ctx.deps.state.setdefault("ties", {})[marker] = ties
    ok = ties[ties.twt.notna()]
    d = {"marker": marker, "wells_with_marker": len(ties), "tied": int(ties.twt.notna().sum()),
         "in_seismic": int(ok.in_cube.sum()), "td_sources_used": ok.td_source.value_counts().to_dict(),
         "twt_range_ms": [round(float(ok.twt.min()), 1), round(float(ok.twt.max()), 1)] if len(ok) else None,
         "table_spread_ms": {"median": round(float(ok.table_spread_ms.median()), 1),
                             "max": round(float(ok.table_spread_ms.max()), 1)} if len(ok) else None,
         "methods": ties.method.value_counts().to_dict(),
         "wells": df_brief(ties[["well", "il", "xl", "tvdss", "twt", "method", "td_source", "n_tables", "table_spread_ms", "in_cube"]])}
    return ctx.deps.record("tie_marker", {"marker": marker, "td_preference": td_preference,
                                          "exclude_tables": exclude_tables}, d)


async def optimise_well_tie(ctx: RunContext[RunContext_], marker: str, stretch_max_ms: float = 8.0,
                            roughness_weight: float = 0.02, half_window_ms: float = 150.0) -> dict:
    """Well–seismic tie at every well with sonic + density: synthetic seismogram, global optimisation
    of bulk shift / phase / wavelet frequency, then regularised stretch & squeeze. Returns per well the
    correlation (bulk and stretched), polarity, the time correction at the top (lag_at_top_ms) and the
    event (peak/trough) the top falls on."""
    from strata.data import read_logs
    ties = ctx.deps.state.get("ties", {}).get(marker)
    if ties is None:
        return {"error": "call tie_marker first"}
    res, curves = WT.optimise_tie(ctx.deps.cube, ctx.deps.wells, read_logs(), ties, half_window_ms=half_window_ms,
                                  stretch_max_ms=stretch_max_ms, roughness_weight=roughness_weight)
    ctx.deps.state["opt_ties"] = res
    ctx.deps.state["lag_curves"] = curves
    d = {"wells_tied_to_seismic": len(res), "median_lag_at_top_ms": float(res.lag_at_top_ms.median()) if len(res) else None,
         "corr_bulk_median": float(res.corr_bulk.median()) if len(res) else None,
         "corr_stretched_median": float(res.corr_stretched.median()) if len(res) else None,
         "event_at_top_votes": res.event_at_top.value_counts().to_dict() if len(res) else {},
         "polarity_votes": res.polarity.value_counts().to_dict() if len(res) else {},
         "wells": df_brief(res.drop(columns=["td_relation"])) if len(res) else []}
    return ctx.deps.record("optimise_well_tie", {"marker": marker, "stretch_max_ms": stretch_max_ms,
                                                 "roughness_weight": roughness_weight, "half_window_ms": half_window_ms}, d)


async def publish_ties(ctx: RunContext[RunContext_], marker: str, notes: str, apply_seismic_tie: bool = True,
                       untrusted_tie_wells: list[str] | None = None, expected_event: str | None = None) -> dict:
    """Publish the well ties of `marker`. With apply_seismic_tie, each top time is corrected by its
    well's lag_at_top_ms from optimise_well_tie; wells without a seismic tie (or listed in
    untrusted_tie_wells) get the median lag. expected_event ('peak'|'trough') is published for the
    horizon agent."""
    ties = ctx.deps.state.get("ties", {}).get(marker)
    if ties is None:
        return {"error": "call tie_marker first"}
    ties = ties.copy()
    opt = ctx.deps.state.get("opt_ties")
    ties["seismic_lag_ms"] = 0.0
    ties["tie_corr"] = np.nan
    if apply_seismic_tie and opt is not None and len(opt):
        bad = set(untrusted_tie_wells or [])
        good = opt[~opt.well.isin(bad)]
        med = float(good.lag_at_top_ms.median()) if len(good) else 0.0
        lag = dict(zip(good.well, good.lag_at_top_ms))
        corr = dict(zip(opt.well, opt.corr_stretched))
        ties["seismic_lag_ms"] = [lag.get(w, med) for w in ties.well]
        ties["tie_corr"] = [corr.get(w, np.nan) for w in ties.well]
        ties["twt_checkshot"] = ties.twt
        ties["twt"] = ties.twt + ties.seismic_lag_ms
    ties["expected_event"] = expected_event
    ctx.deps.state["ties"][marker] = ties
    ctx.deps.state["expected_event"] = expected_event

    def write(d):
        ties.drop(columns=["all_tables"]).to_csv(d / "ties.csv", index=False)
        ties[["well", "all_tables"]].to_json(d / "all_tables.json", orient="records")
        return ["ties.csv", "all_tables.json"]
    m = ctx.deps.store.publish("WellTies", _slug(marker), ctx.deps.agent, write, domain="time", unit="ms",
                               summary=f"{marker}: {int(ties.twt.notna().sum())} wells tied. {notes}"[:300],
                               tool_calls=ctx.deps.take_calls())
    ctx.deps.state["ties_ref"] = m.ref
    ctx.deps.state["marker"] = marker
    return {"published": m.ref}


# ── verifier tool ────────────────────────────────────────────────────────────
async def check_horizon_at_wells(ctx: RunContext[RunContext_], horizon: str) -> dict:
    """Residuals (horizon time − tied top time, ms) of the latest version of HorizonPicks:<horizon>
    at every tied well."""
    s = ctx.deps.store
    m = s.get("HorizonPicks", horizon)
    hz = H.load(s.path(m, "horizon.npz"))
    ties = ctx.deps.state["ties"][ctx.deps.state["marker"]]
    res = H.well_residuals(ctx.deps.cube, hz, ties)
    r = res.residual_ms.dropna()
    d = {"horizon": m.ref, "wells_checked": len(res), "unpicked_at_wells": int(res.residual_ms.isna().sum()),
         "tie_uncertainty_ms": round(float(ties.table_spread_ms.median()), 2),
         "mae_ms": round(float(r.abs().mean()), 2) if len(r) else None,
         "bias_ms": round(float(r.mean()), 2) if len(r) else None,
         "max_abs_ms": round(float(r.abs().max()), 2) if len(r) else None,
         "wells_over_20ms": res.loc[res.residual_ms.abs() > 20, "well"].tolist(),
         "residuals": df_brief(res.sort_values("residual_ms", key=lambda c: -c.abs()))}
    ctx.deps.state["last_residuals"] = res
    return ctx.deps.record("check_horizon_at_wells", {"horizon": m.ref}, d)


# ── scripted (deterministic baseline) policies ───────────────────────────────
def _scripted_marker(returns, target: str) -> dict:
    rows = tool_return(returns, "list_markers") or []
    cand = [r for r in rows if "top" in r["marker"].lower()] or rows
    best = max(cand, key=lambda r: r["wells"]) if cand else {"marker": target}
    return {"marker": best["marker"]}


def build_tie(model_spec: str, target: str) -> Agent:
    def tie_args(R):
        q = tool_return(R, "qc_td_tables") or {}
        return {**_scripted_marker(R, target), "exclude_tables": q.get("outliers", [])}

    def publish_args(R):
        o = tool_return(R, "optimise_well_tie") or {}
        med = o.get("median_lag_at_top_ms") or 0.0
        bad = [w["well"] for w in o.get("wells", []) if abs(w["lag_at_top_ms"] - med) > 20 or w["corr_stretched"] < 0.5]
        votes = o.get("event_at_top_votes") or {}
        return {"marker": _scripted_marker(R, target)["marker"], "notes": "scripted", "apply_seismic_tie": True,
                "untrusted_tie_wells": bad, "expected_event": max(votes, key=votes.get) if votes else None}

    p = plan([("list_markers", {"pattern": target}), ("qc_td_tables", {}),
              ("tie_marker", tie_args),
              ("optimise_well_tie", lambda R: _scripted_marker(R, target)),
              ("publish_ties", publish_args)],
             lambda R: {"marker": _scripted_marker(R, target)["marker"], "ties_object": "", "wells_tied": 0,
                        "summary": "Scripted tie: marker with most wells matching the target.", "concerns": []})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=WellTieResult,
                 instructions=TIE_INSTRUCTIONS, tools=[list_markers, qc_td_tables, tie_marker, optimise_well_tie, publish_ties],
                 name="well_agent")


def _scripted_verdict(returns) -> dict:
    d = tool_return(returns, "check_horizon_at_wells") or {}
    mae, mx = d.get("mae_ms"), d.get("max_abs_ms")
    tol = max(8.0, d.get("tie_uncertainty_ms") or 0.0)
    bad = [r["well"] for r in d.get("residuals", []) if r.get("residual_ms") is not None and abs(r["residual_ms"]) > 3 * tol]
    ok = mae is not None and mae <= tol and not bad and d.get("unpicked_at_wells", 1) == 0
    return {"passed": ok, "summary": f"MAE {mae} ms (tolerance {tol:.1f}), max {mx} ms, {d.get('unpicked_at_wells')} wells unpicked.",
            "findings": [f"wells beyond {3 * tol:.0f} ms: {bad}"] if bad else ([] if ok else [f"MAE {mae} ms exceeds {tol:.1f} ms"]),
            "bad_wells": bad,
            "suggested_fix": "" if ok else "Seed from all tied wells and re-track."}


def build_verify(model_spec: str, horizon: str) -> Agent:
    p = plan([("check_horizon_at_wells", {"horizon": horizon})], _scripted_verdict)
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=HorizonVerdict,
                 instructions=VERIFY_INSTRUCTIONS, tools=[check_horizon_at_wells], name="well_agent_verifier")
