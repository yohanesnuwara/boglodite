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

from fastapi import (Body, FastAPI, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
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
# ── hot reload ───────────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(__file__)


@app.get("/api/devstamp")
def devstamp():
    """Latest mtime across the UI source tree.

    The frontend polls this; when the stamp changes (you edited a .py, .html,
    .js, or .css file) the browser reloads itself. Combine with `boglodite
    --dev` for backend auto-restart too.
    """
    stamp = 0.0
    for root, _, files in os.walk(_PKG_DIR):
        if "__pycache__" in root:
            continue
        for name in files:
            try:
                stamp = max(stamp, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return {"stamp": stamp}


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


# ── local provider (Run Locally / BYOK) ─────────────────────────────────────
_PROVIDER_VARS = ("COPILOT_PROVIDER_BASE_URL", "COPILOT_MODEL", "COPILOT_OFFLINE")
_ENV_SH = os.path.join(CONFIG.repo_root, "set-copilot-env.sh")


def _parse_env_sh() -> dict:
    """Read defaults from set-copilot-env.sh (export VAR="value")."""
    out = {}
    if os.path.isfile(_ENV_SH):
        try:
            with open(_ENV_SH, encoding="utf-8") as f:
                for m in re.finditer(r'export\s+(\w+)="?([^"\n]*)"?', f.read()):
                    out[m.group(1)] = m.group(2)
        except OSError:
            pass
    return out


def _initial_provider() -> dict:
    sh = _parse_env_sh()
    base = os.environ.get("COPILOT_PROVIDER_BASE_URL") or \
        sh.get("COPILOT_PROVIDER_BASE_URL") or "http://192.168.56.1:1234/v1"
    model = os.environ.get("COPILOT_MODEL") or sh.get("COPILOT_MODEL") or ""
    # ON if the launching shell already sourced the provider env.
    return {"enabled": bool(os.environ.get("COPILOT_PROVIDER_BASE_URL")),
            "host": "LM Studio", "base_url": base, "model": model}


_provider = _initial_provider()


def _apply_provider():
    """Reflect _provider into the next copilot spawn's environment.

    No restart needed: each chat turn spawns copilot with a freshly built
    environment (agent.build_env), so changes apply on the very next turn.
    """
    if _provider["enabled"] and _provider["base_url"]:
        _session.env_overrides = {
            "COPILOT_PROVIDER_BASE_URL": _provider["base_url"],
            "COPILOT_MODEL": _provider["model"] or "",
            "COPILOT_OFFLINE": "true",
        }
        _session.env_unset = []
    else:
        _session.env_overrides = {}
        _session.env_unset = list(_PROVIDER_VARS)


def _write_env_sh():
    """Keep set-copilot-env.sh in sync so terminal copilot use matches the UI."""
    try:
        with open(_ENV_SH, "w", encoding="utf-8") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                f'export COPILOT_PROVIDER_BASE_URL="{_provider["base_url"]}"\n'
                f'export COPILOT_MODEL="{_provider["model"]}"\n'
                'export COPILOT_OFFLINE="true"\n'
                'echo "Copilot env vars loaded."\n'
            )
    except OSError:
        pass


_apply_provider()


@app.get("/api/provider")
def api_provider_get():
    return dict(_provider)


@app.get("/api/provider/models")
def api_provider_models(base_url: str):
    """Fetch model ids from an OpenAI-compatible endpoint (e.g. LM Studio).

    Done server-side because the browser can't call LM Studio directly
    (cross-origin, no CORS headers on local servers).
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            payload = json.load(r)
        models = [m.get("id") for m in payload.get("data", []) if m.get("id")]
        return {"models": models, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": f"Cannot reach {url}: {e}"}


@app.post("/api/provider")
def api_provider_set(payload: dict = Body(...)):
    _provider["enabled"] = bool(payload.get("enabled"))
    if payload.get("base_url"):
        _provider["base_url"] = str(payload["base_url"]).strip()
    if payload.get("model") is not None:
        _provider["model"] = str(payload["model"]).strip()
    if payload.get("host"):
        _provider["host"] = str(payload["host"])
    _apply_provider()
    if _provider["enabled"]:
        _write_env_sh()
    return dict(_provider)



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
    if _provider["enabled"]:
        base = (_provider["base_url"] or "").rstrip("/")
        default = _provider["model"] or None
    else:
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
_SKILL_PREAMBLE = ("Load and follow these skill files (relative to the repo "
                   "root); keep them in mind for the whole session:")
# Skill paths already injected, keyed by session id — re-sending the full
# skill preamble every turn makes the agent re-read every SKILL.md and
# balloons the prompt (very slow on local models), so inject once per
# session and only re-send skills that were newly selected.
_skills_sent: dict[str, set] = {}


def _build_prompt(text: str, skill_paths: list[str]) -> str:
    sent = _skills_sent.setdefault(_session.session_id, set())
    new = [p for p in skill_paths if p not in sent]
    if not new:
        return text
    sent.update(new)
    lines = [_SKILL_PREAMBLE] + [f"- {p}" for p in new] + ["", "User request:", text]
    return "\n".join(lines)


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "status", "state": "idle",
                        "session_id": _session.session_id})
    turn_task: asyncio.Task | None = None

    async def tail_run_log(stop: asyncio.Event):
        """Stream lines appended to outputs/run.log while the agent works.

        Sandbox scripts pipe long-running output through
        `... 2>&1 | tee -a outputs/run.log` (see copilot-instructions.md),
        which lands here live — TensorFlow GPU logs, epoch progress, etc.
        """
        path = os.path.join(CONFIG.outputs_dir, "run.log")
        offset = os.path.getsize(path) if os.path.isfile(path) else 0
        while not stop.is_set():
            try:
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    if size < offset:      # file was truncated/recreated
                        offset = 0
                    if size > offset:
                        with open(path, "r", encoding="utf-8",
                                  errors="replace") as f:
                            f.seek(offset)
                            chunk = f.read(size - offset)
                            offset = size
                        for line in chunk.splitlines():
                            if line.strip():
                                await ws.send_json({"type": "proc", "text": line})
            except OSError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.7)
            except asyncio.TimeoutError:
                pass

    async def run_turn(payload: dict):
        text = payload.get("text", "").strip()
        if not text:
            return
        _session.model = payload.get("model") or None
        prompt = _build_prompt(text, payload.get("skills") or [])
        # Snapshot name->mtime so overwritten files (agents often rewrite the
        # same filename, e.g. F3_fault_inline_150.png) count as new results.
        before = {f["name"]: f["mtime"] for f in api_outputs()["files"]}
        await ws.send_json({"type": "status", "state": "running"})
        stop_tail = asyncio.Event()
        tail_task = asyncio.create_task(tail_run_log(stop_tail))
        try:
            async for ev in _session.run_turn(prompt):
                # Some CLI versions echo the submitted prompt back as a
                # message event without a role — never show our own prompt
                # (or the skill preamble) as an agent bubble.
                if ev.get("type") == "assistant":
                    t = ev.get("text", "").strip()
                    if t == prompt.strip() or t == text.strip() or \
                            t.startswith(_SKILL_PREAMBLE):
                        continue
                await ws.send_json(ev)
        finally:
            stop_tail.set()
            await tail_task
            after = api_outputs()["files"]          # already newest-first
            new = [f["name"] for f in after
                   if f["name"] not in before or f["mtime"] > before[f["name"]]]
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
                _skills_sent.pop(_session.session_id, None)
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
    parser.add_argument("--dev", action="store_true",
                        help="auto-restart the server when boglodite_ui/ "
                             "source files change (the browser reloads "
                             "itself in all modes)")
    parser.add_argument("--segy", help="path to SEGY volume to display")
    parser.add_argument("--copilot-bin", help="copilot executable")
    args = parser.parse_args()

    if args.segy:
        os.environ["BOGLODITE_SEGY"] = args.segy
    if args.copilot_bin:
        os.environ["BOGLODITE_COPILOT_BIN"] = args.copilot_bin
        CONFIG.copilot_bin = args.copilot_bin
        _session.copilot_bin = args.copilot_bin

    os.makedirs(CONFIG.outputs_dir, exist_ok=True)
    _register_skills_dir()

    url = f"http://{args.host}:{args.port}"
    print(f"\n  ⛰  Boglodite console → {url}"
          + ("   [dev mode: auto-reload]" if args.dev else "") + "\n")
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True
        ).start()
    if args.dev:
        # Reload mode needs an import string; module state resets on reload,
        # which is fine during development.
        uvicorn.run("boglodite_ui.app:app", host=args.host, port=args.port,
                    log_level="warning", reload=True,
                    reload_dirs=[os.path.dirname(__file__)])
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
