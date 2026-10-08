"""Deterministic adapters exposed by the Boglodite MCP server.

These functions deliberately keep the MCP protocol wiring separate from the
scientific execution adapters.  The adapters call the existing validated
Boglodite prediction scripts in subprocesses, so the model code, data
conditioning, output naming, and GPU behaviour remain single-sourced.

The MCP server itself uses stdio.  For that reason, child process stdout/stderr
is captured and returned as tool output rather than inherited by the server.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Dutch F3 seismic data" / "Dutch Government_F3_entire_8bit seismic.segy"
FAULTSEG_MODEL = REPO_ROOT / "models" / "faultSeg_model" / "model" / "fseg-60.hdf5"
MALENOV_MODEL = REPO_ROOT / "models" / "F3_multiclass_model.h5"
FAULTSEG_SCRIPT = REPO_ROOT / "sandbox" / "FaultSeg" / "predict_only_fault.py"
MALENOV_SCRIPT = REPO_ROOT / "sandbox" / "MalenoV" / "predict_only_facies_stable.py"
OUTPUT_DIR = REPO_ROOT / "outputs"

SliceType = Literal["inline", "xline", "timeslice"]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _exists(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "exists": path.exists()}


def readiness() -> dict[str, Any]:
    """Return readiness of the local seismic data, models, and adapters."""
    faultseg = {
        "ready": DATA_PATH.exists() and FAULTSEG_MODEL.exists() and FAULTSEG_SCRIPT.exists(),
        "data": _exists(DATA_PATH),
        "model": _exists(FAULTSEG_MODEL),
        "adapter": _exists(FAULTSEG_SCRIPT),
    }
    malenov = {
        "ready": DATA_PATH.exists() and MALENOV_MODEL.exists() and MALENOV_SCRIPT.exists(),
        "data": _exists(DATA_PATH),
        "model": _exists(MALENOV_MODEL),
        "adapter": _exists(MALENOV_SCRIPT),
    }
    return {
        "repo_root": str(REPO_ROOT),
        "faultseg": faultseg,
        "malenov": malenov,
        "ready": faultseg["ready"] and malenov["ready"],
    }


def inspect_f3_volume() -> dict[str, Any]:
    """Inspect the configured F3 SEG-Y volume without loading the full cube."""
    if not DATA_PATH.exists():
        return {
            "status": "not_ready",
            "message": f"SEG-Y data not found at {_rel(DATA_PATH)}",
            "path": _rel(DATA_PATH),
        }

    # Lazy import keeps MCP startup lightweight and allows readiness checks even
    # before the full scientific environment is installed.
    import segyio

    with segyio.open(str(DATA_PATH), "r", strict=False) as f:
        f.mmap()
        return {
            "status": "ok",
            "path": _rel(DATA_PATH),
            "shape": [len(f.ilines), len(f.xlines), len(f.samples)],
            "axis_order": ["inline", "crossline", "sample"],
            "inline": {
                "start": int(f.ilines[0]),
                "end": int(f.ilines[-1]),
                "step": int(f.ilines[1] - f.ilines[0]) if len(f.ilines) > 1 else None,
            },
            "crossline": {
                "start": int(f.xlines[0]),
                "end": int(f.xlines[-1]),
                "step": int(f.xlines[1] - f.xlines[0]) if len(f.xlines) > 1 else None,
            },
            "time_ms": {
                "start": float(f.samples[0]),
                "end": float(f.samples[-1]),
                "step": float(f.samples[1] - f.samples[0]) if len(f.samples) > 1 else None,
            },
        }


async def _run_adapter(
    *,
    name: str,
    command: list[str],
    expected_outputs: list[Path],
    prerequisites: list[Path],
) -> dict[str, Any]:
    missing = [p for p in prerequisites if not p.exists()]
    if missing:
        return {
            "status": "not_ready",
            "tool": name,
            "message": "Required local files are missing.",
            "missing": [_rel(p) for p in missing],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = float(os.environ.get("BOGLODITE_MCP_TIMEOUT", "7200"))
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "status": "timeout",
            "tool": name,
            "timeout_seconds": timeout_s,
            "command": command,
        }

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    outputs = [_rel(p) for p in expected_outputs if p.exists()]

    result = {
        "status": "ok" if proc.returncode == 0 else "error",
        "tool": name,
        "exit_code": proc.returncode,
        "command": command,
        "outputs": outputs,
        # Keep model context bounded while retaining the most useful execution
        # evidence. Full files remain in outputs/ and the regular Boglodite log.
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-6000:],
    }
    if proc.returncode == 0 and not outputs:
        result["status"] = "error"
        result["message"] = "Adapter exited successfully but expected outputs were not created."
    return result


async def run_faultseg(slice_type: SliceType, coordinate: int) -> dict[str, Any]:
    """Run the canonical FaultSeg adapter on an inline, crossline, or time slice."""
    tag = {"inline": "inline", "xline": "xline", "timeslice": "timeslice"}[slice_type]
    npy = OUTPUT_DIR / f"F3_fault_{tag}_{coordinate}.npy"
    png = OUTPUT_DIR / f"F3_fault_{tag}_{coordinate}.png"
    command = [
        sys.executable,
        str(FAULTSEG_SCRIPT),
        "--orientation",
        slice_type,
        "--value",
        str(coordinate),
    ]
    result = await _run_adapter(
        name="FaultSeg",
        command=command,
        expected_outputs=[npy, png],
        prerequisites=[DATA_PATH, FAULTSEG_MODEL, FAULTSEG_SCRIPT],
    )
    result["request"] = {"slice_type": slice_type, "coordinate": coordinate}
    return result


async def run_malenov(inline: int) -> dict[str, Any]:
    """Run the canonical MalenoV facies-classification adapter on one inline."""
    prob = OUTPUT_DIR / f"F3_multi_prob_{inline}.npy"
    cls = OUTPUT_DIR / f"F3_multi_class_{inline}.npy"
    png = OUTPUT_DIR / f"F3_multi_inline_{inline}.png"
    command = [sys.executable, str(MALENOV_SCRIPT), "--inline", str(inline)]
    result = await _run_adapter(
        name="MalenoV",
        command=command,
        expected_outputs=[prob, cls, png],
        prerequisites=[DATA_PATH, MALENOV_MODEL, MALENOV_SCRIPT],
    )
    result["request"] = {"inline": inline}
    return result
