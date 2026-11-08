"""Data steward agent: loads and audits INPUT data, fixes project conventions (gate G1)."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from strata.agents.common import RunContext_, make_model, plan, skill
from strata.config import GIVEN, SEISMIC_FILE


class DataAudit(BaseModel):
    accepted: bool
    summary: str
    issues: list[str] = []


INSTRUCTIONS = f"""You are the DATA STEWARD of a reservoir-modelling agent team working on the Norne field.
Your job: inspect the INPUT data you are given, confirm the project conventions, record any data issues,
and publish the project settings that every other agent must follow. You never interpret geology.

Project conventions (GIVEN, not negotiable): {json.dumps(GIVEN)}.
Call inspect_seismic and inspect_wells, then publish_project_settings with your notes and issues,
then return your audit. Accept the data unless something makes interpretation impossible; list every
inconsistency you notice (conventions, missing tables, sign conventions, partial coverage) as an issue.

Domain guidance:
{skill('geomodel-data-readiness')}
"""


async def inspect_seismic(ctx: RunContext[RunContext_]) -> dict:
    """Geometry and amplitude statistics of the structural seismic cube (2001 vintage)."""
    c = ctx.deps.cube
    xy = c.ilxl_to_xy([c.ilines[0], c.ilines[-1], c.ilines[0]], [c.xlines[0], c.xlines[0], c.xlines[-1]])
    d = {"file": SEISMIC_FILE, "inlines": [int(c.ilines[0]), int(c.ilines[-1]), len(c.ilines)],
         "xlines": [int(c.xlines[0]), int(c.xlines[-1]), len(c.xlines)], "sample_rate_ms": c.dt,
         "window_loaded_ms": [c.t0, round(c.t0 + c.dt * (c.shape[2] - 1), 1)],
         "bin_spacing_m": {"inline_step": round(float(np.hypot(*(xy[1] - xy[0]) / (len(c.ilines) - 1))), 2),
                           "xline_step": round(float(np.hypot(*(xy[2] - xy[0]) / (len(c.xlines) - 1))), 2)},
         "corner_xy": np.round(xy, 1).tolist(), "amplitude_rms": round(c.rms, 4),
         "geometry": "IL/XL → X/Y affine fitted from trace headers (coordinate scalar applied)"}
    return ctx.deps.record("inspect_seismic", {}, d)


async def inspect_wells(ctx: RunContext[RunContext_]) -> dict:
    """Counts and conventions of the non-blind wells: paths, time–depth tables, picks."""
    w = ctx.deps.wells
    src = {}
    for tables in w.td.values():
        for t in tables:
            src[t["source"]] = src.get(t["source"], 0) + 1
    gens = w.picks.marker.str.extract(r"(20\d\d)")[0].value_counts().to_dict()
    d = {"wellbores_with_paths": len(w.paths), "wellbores_with_td_tables": len(w.td),
         "wellbores_with_picks": int(w.picks.wkey.nunique()), "blind_wellbores_withheld": len(w.blind),
         "td_table_sources": dict(sorted(src.items(), key=lambda kv: -kv[1])),
         "pick_rows": len(w.picks), "pick_generations_in_marker_names": gens,
         "tvdss_convention": "source stores TVDSS negative; loaded as positive down",
         "wells_without_td_table": sorted(set(w.picks.wkey) - set(w.td))[:20]}
    return ctx.deps.record("inspect_wells", {}, d)


async def publish_project_settings(ctx: RunContext[RunContext_], notes: str, issues: list[str]) -> dict:
    """Publish the project settings (conventions + audit findings) to the project store."""
    settings = {**GIVEN, "notes": notes, "issues": issues}

    def write(d):
        (d / "settings.json").write_text(json.dumps(settings, indent=2))
        return ["settings.json"]
    m = ctx.deps.store.publish("ProjectSettings", "norne", ctx.deps.agent, write, summary=notes[:300],
                               tool_calls=ctx.deps.take_calls())
    return {"published": m.ref}


def build(model_spec: str) -> Agent:
    p = plan([("inspect_seismic", {}), ("inspect_wells", {}),
              ("publish_project_settings", {"notes": "Scripted audit: conventions per GIVEN; blind wells withheld.",
                                            "issues": []})],
             {"accepted": True, "summary": "Scripted data audit completed.", "issues": []})
    return Agent(make_model(model_spec, p), deps_type=RunContext_, output_type=DataAudit,
                 instructions=INSTRUCTIONS, tools=[inspect_seismic, inspect_wells, publish_project_settings],
                 name="data_steward")
