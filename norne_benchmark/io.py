"""Read-only readers for the Norne dataset.

Every function only opens files for reading. Paths resolve under NORNE_ROOT,
overridable via the environment variable of the same name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NORNE_ROOT = Path(os.environ.get(
    "NORNE_ROOT", "/mnt/t7/Dell Inspiron/Local Disk D/Full-Norne"))

ATW_DIR = NORNE_ROOT / "reservoir_models/Norne_ATW_2013"
ENS_DIR = NORNE_ROOT / "reservoir_models/Norne_Full_field_realization_2013"
NX, NY, NZ = 46, 112, 22

# Zonation from the ATW-2013 deck (1-based layers, inclusive).
ZONES = {"Garn": (1, 3), "Not": (4, 4), "Ile": (5, 11), "Tofte": (12, 18), "Tilje": (19, 22)}


def zone_of_layer() -> np.ndarray:
    """Array (NZ,) of zone names indexed by 0-based layer."""
    out = np.empty(NZ, dtype=object)
    for name, (k1, k2) in ZONES.items():
        out[k1 - 1:k2] = name
    return out


# ── GRDECL keyword parsing ───────────────────────────────────────────────────
_COMMENT = re.compile(r"--[^\n]*")


def _expand(tokens: list[str]) -> np.ndarray:
    vals: list[float] = []
    for t in tokens:
        if "*" in t:
            n, v = t.split("*", 1)
            vals.extend([float(v) if v else np.nan] * int(n))
        else:
            vals.append(float(t))
    return np.asarray(vals)


def read_grdecl_keywords(path: Path, keywords: set[str]) -> dict[str, np.ndarray]:
    """Return {keyword: flat float array} for the requested keywords.

    Handles Eclipse repeat syntax (``n*value``) and ``--`` comments.
    Arrays are in Eclipse order (I fastest, then J, then K).
    """
    text = _COMMENT.sub("", Path(path).read_text(errors="replace"))
    out: dict[str, np.ndarray] = {}
    for kw in keywords:
        m = re.search(rf"(?m)^\s*{kw}\s*$", text) or re.search(rf"(?m)^\s*{kw}\b", text)
        if not m:
            continue
        end = text.index("/", m.end())
        body = text[m.end():end]
        if "*" in body:
            out[kw] = _expand(body.split())
        else:
            out[kw] = np.array(body.split(), dtype=float)
    return out


def to_ijk(flat: np.ndarray, nx: int = NX, ny: int = NY, nz: int = NZ) -> np.ndarray:
    """Eclipse-ordered flat array → (nx, ny, nz)."""
    return flat.reshape(nz, ny, nx).transpose(2, 1, 0)


# ── Raw corner-point geometry ────────────────────────────────────────────────
@dataclass
class CornerPoint:
    coord: np.ndarray   # (ny+1, nx+1, 6)  x1 y1 z1 x2 y2 z2
    zcorn: np.ndarray   # (nz, 2, ny, 2, nx, 2)  [k][top/bot][j][jc][i][ic]

    def corner_xyz(self) -> np.ndarray:
        """All 8 corners per cell: (nx, ny, nz, 8, 3), corner order
        (top: i0j0, i1j0, i0j1, i1j1; bottom: same)."""
        nz, _, ny, _, nx, _ = self.zcorn.shape
        out = np.empty((nx, ny, nz, 8, 3))
        for tb in (0, 1):
            for jc in (0, 1):
                for ic in (0, 1):
                    z = self.zcorn[:, tb, :, jc, :, ic].transpose(2, 1, 0)  # (nx, ny, nz)
                    p = self.coord[jc:jc + ny, ic:ic + nx]                 # (ny, nx, 6)
                    p = p.transpose(1, 0, 2)[:, :, None, :]                 # (nx, ny, 1, 6)
                    dz = p[..., 5] - p[..., 2]
                    t = np.where(np.abs(dz) > 1e-9, (z - p[..., 2]) / np.where(dz == 0, 1, dz), 0.0)
                    idx = tb * 4 + jc * 2 + ic
                    out[..., idx, 0] = p[..., 0] + t * (p[..., 3] - p[..., 0])
                    out[..., idx, 1] = p[..., 1] + t * (p[..., 4] - p[..., 1])
                    out[..., idx, 2] = z
        return out


def read_cornerpoint(path: Path, flip_y: bool = False) -> tuple[CornerPoint, dict[str, np.ndarray]]:
    kws = read_grdecl_keywords(path, {"COORD", "ZCORN", "ACTNUM", "FACIES", "SWAT"})
    coord = kws["COORD"].reshape(NY + 1, NX + 1, 6).copy()
    if flip_y:  # Petrel MAPAXES (0,-1000,0,0,1000,0) stores Y negated
        coord[..., 1] *= -1
        coord[..., 4] *= -1
    zcorn = kws["ZCORN"].reshape(NZ, 2, NY, 2, NX, 2)
    return CornerPoint(coord, zcorn), kws


# ── Wells ────────────────────────────────────────────────────────────────────
def norm_well(name: str) -> str:
    """'NO 6608/10-B-1 AHT2' / '6608/10-B-1 H' → '6608/10-B-1AHT2' / '6608/10-B-1H'."""
    return re.sub(r"\s+", "", name.replace("NO ", "", 1)).upper()


def read_picks() -> pd.DataFrame:
    df = pd.read_csv(NORNE_ROOT / "wells/picks/Final Well Picks.txt", sep="\t", skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Well name": "well", "surface x": "sx", "surface y": "sy",
                            "Marker name": "marker", "MD": "md", "TVD KB": "tvdkb", "TVD SS": "tvdss"})
    df["marker"] = df["marker"].str.strip()
    df["well"] = df["well"].str.strip()
    df["wkey"] = df["well"].map(norm_well)
    for c in ("sx", "sy", "md", "tvdkb", "tvdss"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tvdss"] = df["tvdss"].abs()  # normalise: positive down
    return df


def read_wellheads() -> dict[str, tuple[float, float]]:
    """{well key: (x, y)} from the OpenWorks well-header export."""
    out = {}
    txt = (NORNE_ROOT / "wells/well_header/norne-ntntu46-WellHeaders-only.asc").read_text(errors="replace")
    for m in re.finditer(r"WELL:\s+(NO\s+\S+\s+\S+)\s+([\d.]+)\s+([\d.]+)", txt):
        out[norm_well(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def read_position_logs() -> dict[str, pd.DataFrame]:
    """{well key: DataFrame(md, tvdss, x, y)} from OpenWorks PosLog exports.

    Two layouts exist: tab-separated with absolute X/Y, and space-separated
    with only dX/dY offsets (absolute = wellhead + offset).
    """
    heads = read_wellheads()
    out: dict[str, pd.DataFrame] = {}
    for f in sorted((NORNE_ROOT / "wells/position - Original").glob("*PosLog.asc")):
        name, rows = None, []
        for line in f.read_text(errors="replace").splitlines():
            parts = line.replace("\t", " ").split()
            if not parts:
                continue
            if parts[0] == "PLG1":
                name = re.sub(r"\s+", " ", line.replace("\t", "").split("UWI:")[-1]).strip()
            elif parts[0] == "PLG4":
                vals = [float(p) for p in parts[1:]]
                md, _tvd, tvdss, dx, dy = vals[:5]
                if len(vals) >= 7:
                    x, y = vals[5], vals[6]
                else:
                    hx, hy = heads.get(norm_well(name or ""), (np.nan, np.nan))
                    x, y = hx + dx, hy + dy
                rows.append((md, abs(tvdss), x, y))
        if rows:
            # file names are cleaner than the PLG1 headers: norne-660810-B-1_AHT2-PosLog.asc
            m = re.match(r"norne-6608(\d\d)-(.+)-PosLog", f.name)
            key = norm_well(f"6608/{m.group(1)}-{m.group(2).replace('_', '')}") if m else norm_well(name or f.stem)
            out[key] = pd.DataFrame(rows, columns=["md", "tvdss", "x", "y"])
    return out


# ── Point sets (faults, horizons) ────────────────────────────────────────────
def read_fault_sticks(path: Path) -> pd.DataFrame:
    """Columns: x, y, z, code, seg, pt, name. One row per stick vertex.
    A new stick starts where pt == 1."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        code, seg = int(p[3]), int(p[4])
        m = re.match(r"(\d)(.*)", p[5])
        pt, name = int(m.group(1)), m.group(2)
        rows.append((x, y, z, code, seg, pt, name))
    df = pd.DataFrame(rows, columns=["x", "y", "z", "code", "seg", "pt", "name"])
    df["stick"] = (df["pt"] == 1).cumsum()
    return df


def read_horizon_points(path: Path) -> pd.DataFrame:
    """Columns: il, xl, x, y, z (and flag)."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                rows.append(tuple(float(v) for v in p[:5]))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["il", "xl", "x", "y", "z"])


def eclipse_faults(path: Path) -> pd.DataFrame:
    """FAULTS keyword: name, i1, i2, j1, j2, k1, k2, face (1-based)."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("--")[0].strip()
        m = re.match(r"'([^']+)'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+'?([XYZI\-]+)'?", line)
        if m:
            rows.append((m.group(1), *map(int, m.groups()[1:7]), m.group(8)))
    return pd.DataFrame(rows, columns=["name", "i1", "i2", "j1", "j2", "k1", "k2", "face"])


# ── Writing ──────────────────────────────────────────────────────────────────
def _fmt_block(vals: np.ndarray, fmt: str, per_line: int = 6) -> str:
    vals = np.asarray(vals).ravel()
    s = [fmt % v for v in vals]
    return "\n".join(" ".join(s[i:i + per_line]) for i in range(0, len(s), per_line))


def write_grdecl(path: Path, cp: "CornerPoint", actnum: np.ndarray,
                 props: dict[str, np.ndarray] | None = None) -> None:
    """Write SPECGRID/COORD/ZCORN/ACTNUM (+ optional properties) as Eclipse ASCII.

    ``actnum`` and properties are (nx, ny, nz) arrays; COORD/ZCORN come from ``cp``.
    No MAPAXES is written — coordinates are absolute ED50/UTM32N.
    """
    ny1, nx1, _ = cp.coord.shape
    nz = cp.zcorn.shape[0]
    nx, ny = nx1 - 1, ny1 - 1
    ecl = lambda a: np.asarray(a).transpose(2, 1, 0).ravel()
    parts = [f"SPECGRID\n {nx} {ny} {nz} 1 F /\n",
             "COORD\n" + _fmt_block(cp.coord, "%.3f") + "\n/\n",
             "ZCORN\n" + _fmt_block(cp.zcorn, "%.3f") + "\n/\n",
             "ACTNUM\n" + _fmt_block(ecl(actnum).astype(int), "%d", 20) + "\n/\n"]
    for name, arr in (props or {}).items():
        parts.append(f"{name}\n" + _fmt_block(ecl(arr), "%.6g") + "\n/\n")
    Path(path).write_text("".join(parts))
