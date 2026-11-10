"""Paths and the INPUT allow-list.

Agents never get file-system tools. Everything they can read goes through the data
layer, which resolves paths only inside INPUT_DIRS below — TRUTH folders (horizons,
fault sticks, velocity models, reservoir models) are unreachable by construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from the repo's .env (git-ignored) without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(REPO / ".env")
NORNE_ROOT = Path(os.environ.get("NORNE_ROOT", "/mnt/t7/Dell Inspiron/Local Disk D/Full-Norne"))
RUNS_DIR = Path(os.environ.get("STRATA_RUNS", REPO / "runs"))
CACHE_DIR = REPO / "data" / "strata_cache"            # derived INPUT arrays (git-ignored via data/*)
BLIND_WELLS_FILE = REPO / "evaluation" / "norne" / "blind_wells.json"

# Benchmark visibility tier INPUT (docs/NORNE_BENCHMARK.md §3). Relative to NORNE_ROOT.
INPUT_DIRS = (
    "seismics",
    "wells/well_logs",
    "wells/position - Original",
    "wells/well_header",
    "wells/time-depth",
    "wells/picks",
    "wells/Wells with chek-shots",
)
SEISMIC_FILE = "seismics/norne4d_2001-full.sgy"        # 2001 baseline = structural cube

# GIVEN parameters (not scored; docs/NORNE_BENCHMARK.md §2.1)
GIVEN = {
    "crs": "ED50 / UTM zone 32N",
    "depth": "TVDSS metres, positive down",
    "time": "TWT milliseconds, positive down",
    "zonation": {"Garn": [1, 3], "Not": [4, 4], "Ile": [5, 11], "Tofte": [12, 18], "Tilje": [19, 22]},
    "target_grid": [46, 112, 22],
}


def input_path(rel: str) -> Path:
    """Resolve a path under NORNE_ROOT, refusing anything outside the INPUT tier."""
    rel = rel.strip("/")
    if not any(rel == d or rel.startswith(d + "/") for d in INPUT_DIRS):
        raise PermissionError(f"'{rel}' is not INPUT data (allowed: {', '.join(INPUT_DIRS)})")
    p = (NORNE_ROOT / rel).resolve()
    if NORNE_ROOT.resolve() not in p.parents and p != NORNE_ROOT.resolve():
        raise PermissionError(f"'{rel}' escapes the data root")
    return p


def blind_wells() -> set[str]:
    """Normalised keys of the blind wellbores — excluded from everything agents see."""
    return set(json.loads(BLIND_WELLS_FILE.read_text())["blind_wells"])


def data_available() -> bool:
    return (NORNE_ROOT / SEISMIC_FILE).exists()
