"""FastAPI backend of the Strata workbench.

Runs execute in this process (one asyncio task per run); every run is fully recorded in
runs/<run_id>/ and past runs are re-opened from disk. Benchmark scores are produced by
the scorer in a separate process and only shown after a run has ended.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from strata.agents.common import DEFAULT_MODEL
from strata.config import REPO, RUNS_DIR, data_available
from strata.orchestrator import Orchestrator, RunConfig
from strata.store import Store

STATIC = Path(__file__).parent / "static"
SCORES = REPO / "outputs" / "norne_scores" / "runs"
app = FastAPI(title="Strata workbench")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
SCORES.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=SCORES), name="reports")
ACTIVE: dict[str, Orchestrator] = {}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _store(run_id: str) -> Store:
    if run_id in ACTIVE:
        return ACTIVE[run_id].store
    if not (RUNS_DIR / run_id).exists():
        raise HTTPException(404, f"run {run_id} not found")
    return Store(run_id)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"default_model": DEFAULT_MODEL, "data_available": data_available(),
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "models": ["anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5", "anthropic:claude-haiku-5-5", "reference"],
            "autonomy": ["autonomous", "gates_only", "every_decision"],
            "default_brief": RunConfig().brief}


# ── runs ─────────────────────────────────────────────────────────────────────
class StartRun(BaseModel):
    brief: str | None = None
    target: str | None = None
    model: str | None = None
    autonomy: str = "gates_only"


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS_DIR.glob("*"), reverse=True) if RUNS_DIR.exists() else []:
        if (d / "manifest.json").exists():
            m = json.loads((d / "manifest.json").read_text())
            m["active"] = d.name in ACTIVE and ACTIVE[d.name].status in ("running", "waiting")
            out.append(m)
    return out


@app.post("/api/runs")
async def start_run(req: StartRun):
    cfg = RunConfig()
    for k in ("brief", "target", "model"):
        if getattr(req, k):
            setattr(cfg, k, getattr(req, k))
    cfg.autonomy = req.autonomy
    orch = Orchestrator(cfg)
    ACTIVE[orch.store.run_id] = orch

    async def go():
        try:
            await orch.run()
        except Exception:
            pass                                    # recorded as run_error in events.jsonl

    asyncio.create_task(go())
    return {"run_id": orch.store.run_id}


@app.get("/api/runs/{run_id}")
def run_info(run_id: str):
    s = _store(run_id)
    o = ACTIVE.get(run_id)
    cards = [{"id": c.id, "title": c.title, "body": c.body, "options": c.options, "default": c.default, "gate": c.gate}
             for c in (o.cards.values() if o else []) if c.future is not None and not c.future.done()]
    score = None
    sp = SCORES / run_id / "score.json"
    if sp.exists():
        score = json.loads(sp.read_text())
    return {"manifest": s.manifest(), "status": o.status if o else s.manifest().get("status", "unknown"),
            "tree": s.tree(), "cards": cards, "score": score,
            "report": f"/reports/{run_id}/report.html" if (SCORES / run_id / "report.html").exists() else None}


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str, since: int = 0):
    return [e for e in _store(run_id).events() if e["seq"] > since]


class Answer(BaseModel):
    choice: str


@app.post("/api/runs/{run_id}/cards/{card_id}")
def answer(run_id: str, card_id: str, a: Answer):
    o = ACTIVE.get(run_id)
    if not o or not o.answer(card_id, a.choice):
        raise HTTPException(400, "card not pending or invalid choice")
    return {"ok": True}


@app.websocket("/api/runs/{run_id}/ws")
async def run_ws(ws: WebSocket, run_id: str):
    await ws.accept()
    o = ACTIVE.get(run_id)
    if not o:
        await ws.close()
        return
    q = o.store.subscribe()
    try:
        while True:
            ev = await q.get()
            await ws.send_text(json.dumps(ev, default=str))
    except WebSocketDisconnect:
        pass
    finally:
        o.store.unsubscribe(q)


# ── data for the viewers ─────────────────────────────────────────────────────
@app.get("/api/seismic/meta")
def seismic_meta():
    if not data_available():
        raise HTTPException(503, "Norne data drive not mounted")
    from strata.data import load_cube
    c = load_cube()
    return {"ilines": [int(c.ilines[0]), int(c.ilines[-1])], "xlines": [int(c.xlines[0]), int(c.xlines[-1])],
            "t0": c.t0, "dt": c.dt, "nt": int(c.shape[2]), "rms": c.rms}


@app.get("/api/seismic/section")
def seismic_section(kind: str = "inline", idx: int = 1100):
    from strata.data import load_cube
    c = load_cube()
    if kind == "inline":
        a = np.asarray(c.data[int(np.clip(idx - c.ilines[0], 0, len(c.ilines) - 1))]).T      # (nt, nxl)
    elif kind == "xline":
        a = np.asarray(c.data[:, int(np.clip(idx - c.xlines[0], 0, len(c.xlines) - 1))]).T   # (nt, nil)
    else:
        k = int(np.clip(round((idx - c.t0) / c.dt), 0, c.shape[2] - 1))
        a = np.asarray(c.data[:, :, k])                                                       # (nil, nxl)
    q = np.clip(np.round(a / (3 * c.rms) * 127), -127, 127).astype(np.int8)
    return {"h": int(q.shape[0]), "w": int(q.shape[1]), "data": _b64(q)}


@app.get("/api/runs/{run_id}/horizon")
def run_horizon(run_id: str, name: str = "Not2_top", version: int | None = None):
    s = _store(run_id)
    try:
        m = s.get("HorizonPicks", name, version)
    except KeyError:
        raise HTTPException(404, "no horizon yet")
    z = np.load(s.path(m, "horizon.npz"))
    return {"version": m.version, "status": m.status, "shape": list(z["twt"].shape),
            "twt": _b64(z["twt"].astype(np.float32)), "conf": _b64(z["conf"].astype(np.float32)),
            "versions": s.versions("HorizonPicks", name)}


@app.get("/api/runs/{run_id}/ties")
def run_ties(run_id: str):
    s = _store(run_id)
    tree = [o for o in s.tree() if o["type"] == "WellTies"]
    if not tree:
        return []
    o = tree[-1]
    m = s.get("WellTies", o["name"], o["version"])
    import pandas as pd
    df = pd.read_csv(s.path(m, "ties.csv"))
    return json.loads(df[["well", "il", "xl", "twt", "td_source", "table_spread_ms", "in_cube"]].to_json(orient="records"))


@app.get("/api/runs/{run_id}/object")
def run_object(run_id: str, id: str, version: int | None = None):
    t, n = id.split(":", 1)
    m = _store(run_id).get(t, n, version)
    return JSONResponse(json.loads(m.model_dump_json()))


def main(port: int = 8270, open_browser: bool = True):
    import threading
    import webbrowser

    import uvicorn
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
