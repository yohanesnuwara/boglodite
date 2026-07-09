"""Boglodite HITL console server.

Run with `boglodite` (installed script), `uv run boglodite`, or
`uv run python main.py`. Serves:

    GET  /                      the console (single-page, no build step)
    GET  /api/meta              SEGY survey ranges + environment status
    GET  /api/slice             inline/xline/time slice rendered as PNG
    GET  /api/outputs           listing of outputs/ (newest first)
    GET  /api/output/view       a result file rendered/served as an image
    GET  /api/skills            skills/*/SKILL.md registry (frontmatter parsed)
    GET  /api/models            models from the BYOK provider, if reachable
    WS   /ws                    chat <-> Copilot CLI event stream
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import urllib.request
import webbrowser

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response

from .agent import AgentSession
from .config import CONFIG
from .seismic import SegyVolume, render_npy_png, render_seismic_png

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Boglodite Console")

_volume: SegyVolume | None = None
_volume_err: str | None = None
_session = AgentSession(copilot_bin=CONFIG.copilot_bin, cwd=CONFIG.repo_root)


def get_volume() -> SegyVolume | None:
    global _volume, _volume_err
    if _volume is None and _volume_err is None:
        path = CONFIG.segy_path
        if path is None:
            _volume_err = ("No SEGY volume found under data/. Run the "
                           "initiate-boglodite skill (downloads the F3 dataset) "
                           "or set BOGLODITE_SEGY.")
        else:
            try:
                _volume = SegyVolume(path)
            except Exception as e:  # noqa: BLE001
                _volume_err = f"Failed to open {path}: {e}"
    return _volume


# ── pages ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/static/{name}")
def static_file(name: str):
    path = os.path.normpath(os.path.join(STATIC_DIR, name))
    if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


# ── seismic APIs ─────────────────────────────────────────────────────────────
@app.get("/api/meta")
def meta():
    vol = get_volume()
    return {
        "segy": vol.meta() if vol else None,
        "segy_error": _volume_err,
        "repo_root": CONFIG.repo_root,
        "copilot_found": _session.resolve_bin() is not None,
        "copilot_bin": CONFIG.copilot_bin,
        "session_id": _session.session_id,
        "provider_base_url": os.environ.get("COPILOT_PROVIDER_BASE_URL"),
        "default_model": os.environ.get("COPILOT_MODEL"),
    }


@app.get("/api/slice")
def api_slice(axis: str = Query(..., pattern="^(inline|xline|time)$"),
              value: int = Query(...)):
    vol = get_volume()
    if vol is None:
        raise HTTPException(503, _volume_err or "SEGY volume unavailable")
    data, info = vol.slice(axis, value)
    png = render_seismic_png(data, info)
    return Response(png, media_type="image/png",
                    headers={"X-Slice-Value": str(info["value"])})


# ── outputs APIs ─────────────────────────────────────────────────────────────
_VIEWABLE = (".png", ".jpg", ".jpeg", ".npy", ".gif", ".svg")


@app.get("/api/outputs")
def api_outputs():
    out = CONFIG.outputs_dir
    files = []
    if os.path.isdir(out):
        for name in os.listdir(out):
            p = os.path.join(out, name)
            if os.path.isfile(p) and name.lower().endswith(_VIEWABLE):
                st = os.stat(p)
                files.append({"name": name, "mtime": st.st_mtime, "size": st.st_size})
    files.sort(key=lambda f: -f["mtime"])
    return {"files": files}


@app.get("/api/output/view")
def api_output_view(name: str):
    path = os.path.normpath(os.path.join(CONFIG.outputs_dir, name))
    if not path.startswith(os.path.normpath(CONFIG.outputs_dir)):
        raise HTTPException(400, "Invalid path")
    if not os.path.isfile(path):
        raise HTTPException(404, f"{name} not found in outputs/")
    ext = os.path.splitext(name)[1].lower()
    if ext == ".npy":
        try:
            return Response(render_npy_png(path), media_type="image/png")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"Cannot render {name}: {e}") from e
    return FileResponse(path)


# ── skills / models APIs ─────────────────────────────────────────────────────
def _parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@app.get("/api/skills")
def api_skills():
    skills = []
    root = CONFIG.skills_dir
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d, "SKILL.md")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        fm = _parse_frontmatter(f.read(4000))
                except OSError:
                    fm = {}
                skills.append({
                    "dir": d,
                    "name": fm.get("name", d),
                    "description": fm.get("description", "")[:220],
                    "path": os.path.relpath(p, CONFIG.repo_root),
                })
    return {"skills": skills}


@app.get("/api/models")
def api_models():
    """List models from the BYOK provider (e.g. LM Studio) when configured."""
    base = os.environ.get("COPILOT_PROVIDER_BASE_URL", "").rstrip("/")
    default = os.environ.get("COPILOT_MODEL")
    models: list[str] = []
    if base:
        try:
            with urllib.request.urlopen(f"{base}/models", timeout=3) as r:
                payload = json.load(r)
            models = [m.get("id") for m in payload.get("data", []) if m.get("id")]
        except Exception:  # noqa: BLE001 — provider offline is fine
            pass
    if default and default not in models:
        models.insert(0, default)
    return {"models": models, "default": default}


# ── chat websocket ───────────────────────────────────────────────────────────
def _build_prompt(text: str, skill_paths: list[str]) -> str:
    if not skill_paths:
        return text
    lines = ["Before answering, load and follow these skill files "
             "(relative to the repo root):"]
    lines += [f"- {p}" for p in skill_paths]
    lines += ["", "User request:", text]
    return "\n".join(lines)


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "status", "state": "idle",
                        "session_id": _session.session_id})
    turn_task: asyncio.Task | None = None

    async def run_turn(payload: dict):
        text = payload.get("text", "").strip()
        if not text:
            return
        _session.model = payload.get("model") or None
        prompt = _build_prompt(text, payload.get("skills") or [])
        before = {f["name"] for f in api_outputs()["files"]}
        await ws.send_json({"type": "status", "state": "running"})
        try:
            async for ev in _session.run_turn(prompt):
                await ws.send_json(ev)
        finally:
            after = api_outputs()["files"]
            new = [f["name"] for f in after if f["name"] not in before]
            await ws.send_json({"type": "outputs_changed", "new": new})
            await ws.send_json({"type": "status", "state": "idle"})

    try:
        while True:
            payload = await ws.receive_json()
            mtype = payload.get("type")
            if mtype == "chat":
                if _session.running:
                    await ws.send_json({"type": "info",
                                        "text": "Agent is busy — stop it first."})
                    continue
                turn_task = asyncio.create_task(run_turn(payload))
            elif mtype == "stop":
                await _session.stop()
                await ws.send_json({"type": "info", "text": "Run stopped."})
            elif mtype == "new_session":
                await _session.stop()
                _session.new_session()
                await ws.send_json({"type": "status", "state": "idle",
                                    "session_id": _session.session_id,
                                    "reset": True})
    except WebSocketDisconnect:
        if turn_task and not turn_task.done():
            turn_task.cancel()


# ── launcher ─────────────────────────────────────────────────────────────────
def _register_skills_dir():
    """Best-effort: register skills/ with Copilot CLI (idempotent)."""
    import subprocess

    if _session.resolve_bin() is None or not os.path.isdir(CONFIG.skills_dir):
        return
    try:
        subprocess.run(
            [CONFIG.copilot_bin, "skill", "add", CONFIG.skills_dir],
            cwd=CONFIG.repo_root, capture_output=True, timeout=15, check=False,
        )
    except Exception:  # noqa: BLE001
        pass


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="boglodite",
                                     description="Boglodite HITL console")
    parser.add_argument("--port", type=int, default=CONFIG.port)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--segy", help="path to SEGY volume to display")
    parser.add_argument("--copilot-bin", help="copilot executable")
    args = parser.parse_args()

    if args.segy:
        os.environ["BOGLODITE_SEGY"] = args.segy
    if args.copilot_bin:
        CONFIG.copilot_bin = args.copilot_bin
        _session.copilot_bin = args.copilot_bin

    os.makedirs(CONFIG.outputs_dir, exist_ok=True)
    _register_skills_dir()

    url = f"http://{args.host}:{args.port}"
    print(f"\n  ⛰  Boglodite console → {url}\n")
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True
        ).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
