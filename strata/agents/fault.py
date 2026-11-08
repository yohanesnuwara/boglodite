"""Fault agent: discontinuity-based fault map (producer) and horizon–fault consistency (verifier)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from strata.agents.common import RunContext_, make_model, plan, skill, tool_return
from strata.agents.well import HorizonVerdict
from strata.tools import faults as F
from strata.tools import horizon as H


class FaultMapResult(BaseModel):
    fault_map_object: str
    summary: str
    params: dict = {}


ROLE = """You are the FAULT AGENT of a reservoir-modelling agent team on the Norne field (2001 3D cube,
IL 970–1290, XL 1300–2300, 4 ms). Norne is heavily faulted with throws up to >100 ms."""

PRODUCE = ROLE + """

TASK: build a fault map for the reservoir interval (about 2300–2700 ms TWT) from the seismic, to be used
by the horizon agent (tracking may follow throws at faults) and by you to verify horizons.
compute_discontinuity, then make_fault_map with a percentile threshold (higher = fewer, stronger
faults), inspect the statistics, and publish_fault_map. Return a short result.

Domain guidance:
""" + skill("geomodel-seismic-interpretation")

VERIFY = ROLE + """

TASK: VERIFY a horizon published by the horizon agent against your fault map, with
check_horizon_vs_faults. Two tests, both from the seismic only:
1. Off-fault jumps (neighbouring picks > 12 ms apart away from faults) are cycle skips or leaks onto
   another reflector. Guide: ≤ 3 % of picked traces.
2. Throw consistency: at each crossing of a long fault, the seismic offset is estimated by correlating
   the traces on both sides; only self-consistent estimates are used (forward, reverse and a longer
   window agree — reflections are periodic, single estimates can lock onto the wrong cycle). The
   horizon's offset should match within 8 ms. Guide: ≤ 30 % of reliable crossings mismatched.
Many map "faults" are small or noise — do not penalise flat crossings where the seismic shows no throw.
Pass when both guides are met; cite the numbers and give a concrete suggested_fix.
"""


async def compute_discontinuity(ctx: RunContext[RunContext_], t_from_ms: float = 2300.0, t_to_ms: float = 2700.0,
                                slab_ms: float = 100.0) -> dict:
    """Trace-to-trace discontinuity (1 − best lagged correlation with neighbours) in time slabs."""
    d = F.discontinuity(ctx.deps.cube, t_from_ms, t_to_ms, slab_ms)
    ctx.deps.state["discontinuity"] = d
    return ctx.deps.record("compute_discontinuity", {"t_from_ms": t_from_ms, "t_to_ms": t_to_ms, "slab_ms": slab_ms},
                           {"percentiles": {p: round(float(np.percentile(d, p)), 3) for p in (50, 80, 90, 95, 99)}})


async def make_fault_map(ctx: RunContext[RunContext_], percentile: float = 92.0, min_size: int = 30) -> dict:
    """Threshold the discontinuity at `percentile` and clean it into a fault map."""
    d = ctx.deps.state.get("discontinuity")
    if d is None:
        return {"error": "compute_discontinuity first"}
    m, info = F.fault_mask(d, percentile, min_size)
    ctx.deps.state["fault_mask_working"] = m
    return ctx.deps.record("make_fault_map", {"percentile": percentile, "min_size": min_size}, info)


async def publish_fault_map(ctx: RunContext[RunContext_], summary: str) -> dict:
    """Publish the working fault map as FaultMap:reservoir."""
    m = ctx.deps.state.get("fault_mask_working")
    if m is None:
        return {"error": "make_fault_map first"}
    d = ctx.deps.state["discontinuity"]

    def write(p):
        np.savez_compressed(p / "faults.npz", mask=m, discontinuity=d.astype(np.float16))
        return ["faults.npz"]
    obj = ctx.deps.store.publish("FaultMap", "reservoir", ctx.deps.agent, write, domain="time",
                                 summary=summary[:300], tool_calls=ctx.deps.take_calls())
    ctx.deps.state["fault_mask"] = m
    ctx.deps.state["fault_ref"] = obj.ref
    return {"published": obj.ref, "fault_fraction": round(float(m.mean()), 4)}


async def check_horizon_vs_faults(ctx: RunContext[RunContext_], horizon: str) -> dict:
    """Jumps on/off faults and fault crossings without offset for the latest HorizonPicks:<horizon>."""
    s = ctx.deps.store
    mobj = s.get("HorizonPicks", horizon)
    hz = H.load(s.path(mobj, "horizon.npz"))
    m = ctx.deps.state.get("fault_mask")
    if m is None:
        return {"error": "no fault map published"}
    r0 = F.horizon_vs_faults(hz.twt, m)
    picked = int(np.isfinite(hz.twt).sum())
    r = {"picked_traces": picked, "jumps_on_faults": r0["jumps_on_faults"], "jumps_off_faults": r0["jumps_off_faults"],
         "off_fault_jumps_per_picked": round(r0["jumps_off_faults"] / max(picked, 1), 4),
         "throw_consistency": F.throw_consistency(ctx.deps.cube, hz.twt, m)}
    ctx.deps.state["last_fault_check"] = {"version": mobj.version, **r}
    return ctx.deps.record("check_horizon_vs_faults", {"horizon": mobj.ref}, r)


def build_produce(model_spec: str) -> Agent:
    p = plan([("compute_discontinuity", {}), ("make_fault_map", {"percentile": 92.0}),
              ("publish_fault_map", {"summary": "Scripted: discontinuity 2300–2700 ms, 92nd percentile."})],
             lambda R: {"fault_map_object": (tool_return(R, "publish_fault_map") or {}).get("published", ""),
                        "summary": "scripted", "params": {"percentile": 92.0}})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=FaultMapResult, instructions=PRODUCE,
                 tools=[compute_discontinuity, make_fault_map, publish_fault_map], name="fault_agent")


def _scripted_verdict(R) -> dict:
    d = tool_return(R, "check_horizon_vs_faults") or {}
    frac = d.get("off_fault_jumps_per_picked")
    tc = d.get("throw_consistency") or {}
    mm = tc.get("mismatch_frac")
    ok_j = frac is not None and frac <= 0.03
    ok_t = mm is None or mm <= 0.30
    ok = ok_j and ok_t
    return {"passed": ok, "summary": f"off-fault jumps {frac} of picked (≤0.03), throw mismatch {mm} of "
                                     f"{tc.get('reliable_throw_estimates')} reliable crossings (≤0.30)",
            "findings": ([] if ok_j else ["too many jumps away from faults"]) + ([] if ok_t else ["horizon offsets disagree with seismic throws"]),
            "bad_wells": [],
            "suggested_fix": "" if ok else "Use Viterbi tracking with the fault map and a tighter guide window."}


def build_verify(model_spec: str, horizon: str) -> Agent:
    p = plan([("check_horizon_vs_faults", {"horizon": horizon})], _scripted_verdict)
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=HorizonVerdict, instructions=VERIFY,
                 tools=[check_horizon_vs_faults], name="fault_agent_verifier")
