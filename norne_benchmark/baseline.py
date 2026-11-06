"""Baseline ("ceiling") computation for the Norne benchmark — spec §8.

Checks the TRUTH data for internal consistency and computes reference
volumetrics and the realization-ensemble envelope. Reads NORNE_ROOT only;
writes to outputs/norne_baseline/ and evaluation/norne/blind_wells.json.

    uv run python -m norne_benchmark.baseline [--skip-ensemble] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from norne_benchmark import io

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/norne_baseline"
BLIND_JSON = REPO / "evaluation/norne/blind_wells.json"
SEGY = io.NORNE_ROOT / "seismics/norne4d_2001-full.sgy"
GRID_FILE = io.ATW_DIR / "INCLUDE/GRID/IRAP_1005.GRDECL"

# Grid layer interfaces (0-based: interface n = top of layer n; 22 = base of layer 21)
INTERFACES = {"Garn top": 0, "Not top": 3, "Ile top": 4, "Tofte top": 11, "Tilje top": 18, "Tilje base": 22}
PICK_MARKERS = {
    "Garn top": ["Garn Fm. NO 2002 Top", "Garn 3 NO 2002 Top", "FANGST GP. NO 2006 Top", "FANGST GP. NO 2002 Top"],
    "Not top": ["Not Fm. NO 2006 Top", "Not Fm. NO 2002 Top"],
    "Ile top": ["Ile Fm. NO 2006 Top", "Ile Fm. NO 2002 Top"],
    "Tofte top": ["Tofte Fm. NO 2006 Top", "Tofte Fm. NO 2002 Top"],
    "Tilje top": ["Tilje Fm. NO 2006 Top", "Tilje Fm. NO 2002 Top"],
    "Tilje base": ["Tilje Fm. NO 2006 Base", "Tilje Fm. NO 2002 Base"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── geometry helpers ─────────────────────────────────────────────────────────
def bulk_volume_xtgeo() -> np.ndarray:
    """Cell bulk volume (nx, ny, nz) via xtgeo — the canonical implementation used by
    both the reference and the scorer, so candidate and reference are measured alike."""
    import xtgeo
    g = xtgeo.grid_from_file(GRID_FILE, fformat="grdecl")
    return np.asarray(g.get_bulk_volume(asmasked=False).values, dtype=float)


def interface_points(corners: np.ndarray, n: int) -> np.ndarray:
    """(nx*ny, 3) face-centre points of interface n (top of layer n / base of last)."""
    face = corners[:, :, n, 0:4, :] if n < io.NZ else corners[:, :, io.NZ - 1, 4:8, :]
    return face.mean(-2).reshape(-1, 3)


class Interfaces:
    """Linear interpolators Z(x, y) for each grid layer interface."""

    def __init__(self, corners: np.ndarray):
        self.pts = {n: interface_points(corners, n) for n in range(io.NZ + 1)}
        xy = self.pts[0][:, :2]
        self.tree = cKDTree(xy)
        self.interp = {n: LinearNDInterpolator(p[:, :2], p[:, 2]) for n, p in self.pts.items()}

    def z(self, n: int, x, y, max_dist: float = 150.0) -> np.ndarray:
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        d, _ = self.tree.query(np.c_[x, y])
        z = self.interp[n](x, y)
        return np.where(d <= max_dist, z, np.nan)


# ── 1. georeferencing ────────────────────────────────────────────────────────
def _affine(src: np.ndarray, dst: np.ndarray):
    A = np.c_[src, np.ones(len(src))]
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
    res = np.linalg.norm(A @ coef - dst, axis=1)
    return coef, float(np.sqrt((res ** 2).mean())), float(res.max())


def check_georef(atw_corners: np.ndarray, act: np.ndarray) -> dict:
    import segyio
    out: dict = {}
    with segyio.open(SEGY, "r") as s:
        n = s.tracecount
        idx = sorted(set(range(0, n, 211)) | {0, n - 1, len(s.xlines) - 1, n - len(s.xlines)})
        F = segyio.TraceField
        rows = []
        for i in idx:
            h = s.header[i]
            sc = h[F.SourceGroupScalar]
            f = (1 / abs(sc)) if sc < 0 else (sc or 1)
            rows.append((h[F.INLINE_3D], h[F.CROSSLINE_3D], h[F.CDP_X] * f, h[F.CDP_Y] * f))
        out["segy"] = {"file": SEGY.name, "ilines": [int(s.ilines[0]), int(s.ilines[-1])],
                       "xlines": [int(s.xlines[0]), int(s.xlines[-1])],
                       "samples_ms": [float(s.samples[0]), float(s.samples[-1]), float(s.samples[1] - s.samples[0])]}
    seis = np.array(rows, float)
    c_s, rms, mx = _affine(seis[:, :2], seis[:, 2:])
    out["segy"]["affine_ilxl_to_xy"] = c_s.round(4).tolist()
    out["segy"]["affine_rms_m"], out["segy"]["affine_max_m"] = round(rms, 3), round(mx, 3)
    # inverse: XY -> seismic IL/XL
    c_inv, *_ = _affine(seis[:, 2:], seis[:, :2])

    def xy_to_seis(xy):
        return np.c_[xy, np.ones(len(xy))] @ c_inv

    out["horizons"] = {}
    for name in ["Top_Not_2_2003_Time.txt", "Top_Aare_2003_Time.txt",
                 "tral+lmig32r06_fxy+Not_2_Fm_Top+TIME_TDQ+2006+input_geomodel", "bosd-4D_OWC-2006_OFFICIAL_DEPTH"]:
        h = io.read_horizon_points(io.NORNE_ROOT / "horisons" / name)
        sub = h.iloc[:: max(1, len(h) // 5000)]
        c_h, rms_h, _ = _affine(sub[["il", "xl"]].values, sub[["x", "y"]].values)
        s_ilxl = xy_to_seis(sub[["x", "y"]].values)
        c_map, rms_map, _ = _affine(sub[["il", "xl"]].values, s_ilxl)
        inside = ((s_ilxl[:, 0] >= 970) & (s_ilxl[:, 0] <= 1290) & (s_ilxl[:, 1] >= 1300) & (s_ilxl[:, 1] <= 2300)).mean()
        out["horizons"][name] = {
            "n_points": len(h), "z_range": [round(h.z.min(), 2), round(h.z.max(), 2)],
            "il_range": [h.il.min(), h.il.max()], "xl_range": [h.xl.min(), h.xl.max()],
            "affine_rms_m": round(rms_h, 3),
            "horizon_ilxl_to_segy_ilxl": c_map.round(4).tolist(), "mapping_rms_lines": round(rms_map, 3),
            "fraction_inside_seismic": round(float(inside), 4),
        }
    pill = atw_corners[..., 0:4, :2].reshape(-1, 2)
    g = xy_to_seis(pill)
    out["grid_inside_seismic"] = round(float(((g[:, 0] >= 970) & (g[:, 0] <= 1290) &
                                              (g[:, 1] >= 1300) & (g[:, 1] <= 2300)).mean()), 4)
    ca = xy_to_seis(atw_corners[act][:, 0:4, :2].mean(1))
    out["active_cells_inside_seismic"] = round(float(((ca[:, 0] >= 970) & (ca[:, 0] <= 1290) &
                                                      (ca[:, 1] >= 1300) & (ca[:, 1] <= 2300)).mean()), 4)
    out["grid_seis_il_range"] = [round(g[:, 0].min(), 1), round(g[:, 0].max(), 1)]
    out["grid_seis_xl_range"] = [round(g[:, 1].min(), 1), round(g[:, 1].max(), 1)]
    return out


# ── 2. grid vs picks ─────────────────────────────────────────────────────────
def pick_positions(picks: pd.DataFrame, poslogs: dict) -> pd.DataFrame:
    rows = []
    for r in picks.itertuples():
        pl = poslogs.get(r.wkey)
        if pl is None or not np.isfinite(r.md):
            continue
        x = np.interp(r.md, pl.md, pl.x)
        y = np.interp(r.md, pl.md, pl.y)
        tvd_pl = np.interp(r.md, pl.md, pl.tvdss)
        rows.append((r.well, r.wkey, r.marker, r.md, r.tvdss, x, y, tvd_pl))
    return pd.DataFrame(rows, columns=["well", "wkey", "marker", "md", "tvdss", "x", "y", "tvdss_poslog"])


def check_picks(ifc: Interfaces, pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    for surf, markers in PICK_MARKERS.items():
        n = INTERFACES[surf]
        for mk in markers:
            sub = pp[pp.marker == mk]
            if sub.empty:
                continue
            zg = ifc.z(n, sub.x.values, sub.y.values)
            for r, z in zip(sub.itertuples(), zg):
                recs.append((surf, mk, r.well, r.wkey, r.x, r.y, r.tvdss, r.tvdss_poslog, z, z - r.tvdss))
    df = pd.DataFrame(recs, columns=["surface", "marker", "well", "wkey", "x", "y", "pick_tvdss",
                                     "poslog_tvdss", "grid_tvdss", "residual_m"])
    inside = df.dropna(subset=["residual_m"])
    summ = (inside.groupby(["surface", "marker"]).residual_m
            .agg(n="count", bias_m="mean", rms_m=lambda v: float(np.sqrt((v ** 2).mean())),
                 p50_abs_m=lambda v: float(np.median(np.abs(v))), p90_abs_m=lambda v: float(np.percentile(np.abs(v), 90)))
            .round(2).reset_index())
    summ["pick_vs_poslog_tvd_rms_m"] = (inside.groupby(["surface", "marker"])
                                        .apply(lambda g: float(np.sqrt(((g.pick_tvdss - g.poslog_tvdss) ** 2).mean())),
                                               include_groups=False).round(2).values)
    return df, summ


# ── 3. grid vs horizon ───────────────────────────────────────────────────────
def check_horizon() -> dict:
    """The 2006 'Not 2 geomodel input' horizon is in TWT (ms), not depth — verified by
    matching it node-by-node (same IL/XL) to the 2003 Not 2 time horizon. Their
    difference = inter-interpretation variability, which calibrates the S3 tolerance.
    Grid comparison in depth needs a velocity model and belongs to the S4 oracle test."""
    a = io.read_horizon_points(io.NORNE_ROOT / "horisons/tral+lmig32r06_fxy+Not_2_Fm_Top+TIME_TDQ+2006+input_geomodel")
    b = io.read_horizon_points(io.NORNE_ROOT / "horisons/Top_Not_2_2003_Time.txt")
    m = a.merge(b, on=["il", "xl"], suffixes=("_2006", "_2003"))
    d = (m.z_2006 - m.z_2003).values
    return {"domain": "TWT ms (verified)", "matched_nodes": int(len(m)),
            "diff_2006_minus_2003_ms": {"mean": round(float(d.mean()), 2), "std": round(float(d.std()), 2),
                                        "median_abs": round(float(np.median(np.abs(d))), 2),
                                        "p90_abs": round(float(np.percentile(np.abs(d), 90)), 2)}}


# ── 4. grid vs fault sticks ──────────────────────────────────────────────────
def grid_fault_faces(cp: io.CornerPoint, corners: np.ndarray, act: np.ndarray, thr: float = 0.5):
    """Face centres where ZCORN is discontinuous between neighbours (I and J)."""
    z = cp.zcorn  # (nz, 2, ny, 2, nx, 2)
    zi = np.abs(z[..., :-1, 1] - z[..., 1:, 0]).max(axis=(1, 3))          # (nz, ny, nx-1)
    zj = np.abs(z[:, :, :-1, 1, :, :] - z[:, :, 1:, 0, :, :]).max(axis=(1, 4))  # (nz, ny-1, nx)
    fi = zi.transpose(2, 1, 0) > thr                                         # (nx-1, ny, nz)
    fj = zj.transpose(2, 1, 0) > thr                                         # (nx, ny-1, nz)
    fi &= act[:-1] | act[1:]
    fj &= act[:, :-1] | act[:, 1:]
    ci = corners[:-1, :, :, [1, 3, 5, 7], :].mean(-2)[fi]
    cj = corners[:, :-1, :, [2, 3, 6, 7], :].mean(-2)[fj]
    return np.vstack([ci, cj]), int(fi.sum()), int(fj.sum()), fi, fj


def keyword_fault_faces(corners: np.ndarray) -> np.ndarray:
    ef = io.eclipse_faults(io.ATW_DIR / "INCLUDE/FAULT/FAULT_JUN_05.INC")
    pts = []
    for r in ef.itertuples():
        for i in range(r.i1 - 1, r.i2):
            for j in range(r.j1 - 1, r.j2):
                for k in range(r.k1 - 1, r.k2):
                    idx = [1, 3, 5, 7] if r.face.startswith("X") else [2, 3, 6, 7]
                    pts.append(corners[i, j, k, idx, :].mean(0))
    return np.array(pts), ef


def densify_sticks(fs: pd.DataFrame, step: float = 10.0) -> pd.DataFrame:
    out = []
    for (sid, name), g in fs.groupby(["stick", "name"], sort=False):
        p = g[["x", "y", "z"]].values
        if len(p) == 1:
            out.append((name, *p[0]))
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s[-1] + 1e-6, step)
        for c in zip(*(np.interp(t, s, p[:, d]) for d in range(3))):
            out.append((name, *c))
    return pd.DataFrame(out, columns=["name", "x", "y", "z"])


def check_faults(cp, corners, act, ifc: Interfaces) -> tuple[pd.DataFrame, dict, np.ndarray, pd.DataFrame]:
    faces, n_i, n_j, _, _ = grid_fault_faces(cp, corners, act)
    kw_faces, ef = keyword_fault_faces(corners)
    sticks = densify_sticks(io.read_fault_sticks(io.NORNE_ROOT / "FAULTS/kseim_r06_faults_depth"))
    top = ifc.z(0, sticks.x.values, sticks.y.values, max_dist=300)
    base = ifc.z(io.NZ, sticks.x.values, sticks.y.values, max_dist=300)
    # keep stick points inside the reservoir interval AND over active grid columns
    col_act = act.any(axis=2).ravel()
    act_tree = cKDTree(ifc.pts[0][col_act, :2])
    d_act, _ = act_tree.query(sticks[["x", "y"]].values)
    res = sticks[(sticks.z >= top - 50) & (sticks.z <= base + 50) & (d_act <= 100)].copy()

    t_faces, t_kw = cKDTree(faces), cKDTree(kw_faces)
    res["d_grid_m"], _ = t_faces.query(res[["x", "y", "z"]].values)
    res["d_keyword_m"], _ = t_kw.query(res[["x", "y", "z"]].values)
    per = (res.groupby("name").agg(n_pts=("x", "size"), d_grid_median=("d_grid_m", "median"),
                                   d_grid_p90=("d_grid_m", lambda v: np.percentile(v, 90)),
                                   d_kw_median=("d_keyword_m", "median"))
           .round(1).sort_values("n_pts", ascending=False).reset_index())

    t_st = cKDTree(res[["x", "y", "z"]].values)
    d_face_to_stick, _ = t_st.query(faces)
    d_kw_to_disc, _ = t_faces.query(kw_faces)
    summary = {
        "grid_fault_faces_I": n_i, "grid_fault_faces_J": n_j,
        "keyword_faults": int(ef.name.nunique()), "keyword_fault_faces": int(len(kw_faces)),
        "keyword_faces_on_zcorn_discontinuity_frac@1m": round(float((d_kw_to_disc < 1).mean()), 3),
        "stick_points_in_reservoir": int(len(res)), "stick_faults_in_reservoir": int(res.name.nunique()),
        "stick_to_gridface_dist_m": {q: round(float(np.percentile(res.d_grid_m, p)), 1)
                                     for q, p in [("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95)]},
        "gridface_to_stick_dist_m": {q: round(float(np.percentile(d_face_to_stick, p)), 1)
                                     for q, p in [("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95)]},
        "gridfaces_within_100m_of_stick_frac": round(float((d_face_to_stick < 100).mean()), 3),
        "stickpoints_within_100m_of_gridface_frac": round(float((res.d_grid_m < 100).mean()), 3),
    }
    return per, summary, faces, res


# ── 5. reference volumetrics ─────────────────────────────────────────────────
def read_equil() -> pd.DataFrame:
    # Records are one per line, terminated by '/', followed by an uncommented label
    # ("/   C+D: Garn") — so parse line by line and keep only the part before '/'.
    txt = re.sub(r"--[^\n]*", "", (io.ATW_DIR / "INCLUDE/PETRO/E3.prop").read_text(errors="replace"))
    rows = []
    for line in txt[txt.index("EQUIL") + 5:].splitlines():
        if re.match(r"\s*[A-Z]{3,}", line):
            break  # next keyword
        if "/" not in line:
            continue
        v = line.split("/")[0].split()
        if len(v) >= 5:
            datum, p, woc, _, goc = (float(x) for x in v[:5])
            rows.append((len(rows) + 1, datum, p, woc, goc))
    return pd.DataFrame(rows, columns=["eqlnum", "datum", "p_datum_bar", "woc", "goc"])


def read_bo_saturated() -> pd.DataFrame:
    """Saturated (Rs, Psat, Bo) from the first PVTO table."""
    txt = re.sub(r"--[^\n]*", "", (io.ATW_DIR / "INCLUDE/PVT/PVT-WET-GAS.INC").read_text(errors="replace"))
    body = txt[txt.index("PVTO") + 4:]
    rows = []
    for rec in body.split("/"):
        v = rec.split()
        if not v:
            break  # empty record terminates table 1
        if len(v) >= 4:
            rows.append((float(v[0]), float(v[1]), float(v[2])))
    return pd.DataFrame(rows, columns=["rs", "psat_bar", "bo"])


def volumetrics(corners, act, props) -> tuple[pd.DataFrame, dict]:
    brv = bulk_volume_xtgeo()
    zc = corners[..., 2].mean(-1)
    eq = read_equil()
    bo_tab = read_bo_saturated()
    eql = props["EQLNUM"].astype(int)
    goc = np.full(eql.shape, np.nan)
    woc = np.full(eql.shape, np.nan)
    bo = np.full(eql.shape, np.nan)
    bo_by_region = {}
    for r in eq.itertuples():
        m = eql == r.eqlnum
        goc[m], woc[m] = r.goc, r.woc
        # contacts at GOC are at saturation pressure: Psat = P at datum (= GOC in this deck)
        b = float(np.interp(r.p_datum_bar, bo_tab.psat_bar, bo_tab.bo))
        bo[m] = b
        bo_by_region[int(r.eqlnum)] = round(b, 4)
    a = act.astype(bool)
    ntg, poro, sw = props["NTG"], props["PORO"], props["SWATINIT"]
    nrv = brv * ntg
    pv = nrv * poro
    oil = a & (zc > goc) & (zc <= woc)
    gas = a & (zc <= goc)
    hc_oil = np.where(oil, pv * (1 - sw), 0.0)
    hc_gas = np.where(gas, pv * (1 - sw), 0.0)
    stoiip = np.where(oil, hc_oil / bo, 0.0)
    zone = np.broadcast_to(io.zone_of_layer()[None, None, :], act.shape)

    df = pd.DataFrame({"zone": zone[a], "eqlnum": eql[a], "brv": brv[a], "nrv": nrv[a], "pv": pv[a],
                       "hcpv_oil": hc_oil[a], "hcpv_gas": hc_gas[a], "stoiip": stoiip[a]})
    tab = df.groupby(["zone", "eqlnum"]).agg(n_cells=("brv", "size"), brv=("brv", "sum"), nrv=("nrv", "sum"),
                                             pv=("pv", "sum"), hcpv_oil=("hcpv_oil", "sum"),
                                             hcpv_gas=("hcpv_gas", "sum"), stoiip=("stoiip", "sum")).reset_index()
    order = {z: i for i, z in enumerate(io.ZONES)}
    tab = tab.sort_values(["zone", "eqlnum"], key=lambda s: s.map(order) if s.name == "zone" else s)
    tot = df[["brv", "nrv", "pv", "hcpv_oil", "hcpv_gas", "stoiip"]].sum()
    info = {"equil": eq.to_dict("records"), "bo_by_eqlnum": bo_by_region,
            "active_cells": int(a.sum()), "swatinit_zero_in_oilzone_frac": round(float((sw[oil] == 0).mean()), 4),
            "totals_1e6_m3": {k: round(float(v) / 1e6, 3) for k, v in tot.items()}}
    return tab, info, brv


# ── 6. ensemble ──────────────────────────────────────────────────────────────
def _canonicalise(cp: io.CornerPoint, ref: io.CornerPoint) -> tuple[io.CornerPoint, bool]:
    """Petrel exports R2..R120 with Y negated (MAPAXES) and the J axis reversed;
    R1 is exported in the ATW ordering. Return the variant that matches the
    reference geometry, and whether J was reversed."""
    best = (np.inf, cp, False)
    for flip in (False, True):
        c = cp.coord.copy()
        if flip:
            c[..., [1, 4]] *= -1
            c, z = c[::-1], cp.zcorn[:, :, ::-1, ::-1]
        else:
            z = cp.zcorn
        d = float(np.abs(c[..., [0, 1, 3, 4]] - ref.coord[..., [0, 1, 3, 4]]).max())
        if d < best[0]:
            best = (d, io.CornerPoint(c, z), flip)
    return best[1], best[2]


def _ens_one(args):
    path, ref, atw_act, brv, zone = args
    cp_raw, kws = io.read_cornerpoint(path)
    cp, rev_j = _canonicalise(cp_raw, ref)

    def prop(kw):
        a = io.to_ijk(kws[kw])
        return a[:, ::-1, :] if rev_j else a

    act = prop("ACTNUM").astype(bool)
    fac = prop("FACIES") if "FACIES" in kws else None
    rec = {"realization": int(re.search(r"R(\d+)", path.name).group(1)), "j_reversed": rev_j,
           "coord_xy_maxdiff_m": float(np.abs(cp.coord[..., [0, 1, 3, 4]] - ref.coord[..., [0, 1, 3, 4]]).max()),
           "zcorn_maxdiff_m": float(np.abs(cp.zcorn - ref.zcorn).max()),
           "active": int(act.sum()), "atw_active_subset_frac": float(act[atw_act].mean()),
           "actnum_agree_atw": float((act == atw_act).mean())}
    for zn in io.ZONES:
        m_own = act & (zone == zn)
        m_atw = atw_act & (zone == zn)       # facies stats on the ATW-active reservoir
        rec[f"brv_{zn}"] = float(brv[m_own].sum())
        if fac is not None and m_atw.any():
            vals, cnt = np.unique(fac[m_atw], return_counts=True)
            for v, c in zip(vals, cnt):
                rec[f"facies{int(v)}_{zn}"] = float(c / m_atw.sum())
    return rec


def ensemble(cp_atw, act_atw, brv, workers: int) -> pd.DataFrame:
    files = sorted(io.ENS_DIR.glob("FullNorne_R*.GRDECL"), key=lambda p: int(re.search(r"R(\d+)", p.name).group(1)))
    zone = np.broadcast_to(io.zone_of_layer()[None, None, :], act_atw.shape)
    args = [(f, cp_atw, act_atw, brv, zone) for f in files]
    recs = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_ens_one, args), 1):
            recs.append(r)
            if i % 10 == 0:
                log(f"  ensemble {i}/{len(files)}")
    return pd.DataFrame(recs).sort_values("realization")


# ── 7. blind wells ───────────────────────────────────────────────────────────
def select_blind_wells(picks_df: pd.DataFrame, props, ifc: Interfaces, seed: int = 42, frac: float = 0.2) -> dict:
    g = picks_df[(picks_df.surface == "Garn top") & picks_df.residual_m.notna()].drop_duplicates("wkey")
    i_tree = cKDTree(ifc.pts[0][:, :2])
    _, idx = i_tree.query(g[["x", "y"]].values)
    ii, jj = np.unravel_index(idx, (io.NX, io.NY))
    g = g.assign(eqlnum=props["EQLNUM"][ii, jj, 0].astype(int), fipnum=props["FIPNUM"][ii, jj, 0].astype(int),
                 family=g.wkey.str.extract(r"^(6608/10-[A-Z]-\d)")[0])
    fam = g.groupby("family").agg(eqlnum=("eqlnum", lambda s: s.mode().iloc[0]), wells=("wkey", list)).reset_index()
    rng = np.random.default_rng(seed)
    chosen = []
    for reg, sub in fam.groupby("eqlnum"):
        k = int(round(frac * len(sub)))
        if len(sub) >= 3:
            k = max(k, 1)
        chosen += list(rng.choice(sub.family.values, size=k, replace=False))
    blind = fam[fam.family.isin(chosen)]
    return {"seed": seed, "fraction": frac, "unit": "well family (parent + sidetracks share upper path)",
            "families_by_eqlnum": fam.groupby("eqlnum").family.apply(list).to_dict(),
            "blind_families": sorted(chosen),
            "blind_wells": sorted(w for ws in blind.wells for w in ws)}


# ── QC figure ────────────────────────────────────────────────────────────────
def qc_figure(corners, act, faces, sticks_res, picks_df, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    top = corners[:, :, 0, [0, 1, 3, 2], :]
    a0 = act.any(axis=2)
    polys = top[..., :2][a0]
    z = top[..., 2].mean(-1)[a0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax in axes:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Easting (m, ED50 UTM32N)", fontsize=9)
    ax = axes[0]
    pc = PolyCollection(polys, array=z, cmap="Blues", edgecolors="none")
    ax.add_collection(pc)
    ax.autoscale()
    fig.colorbar(pc, ax=ax, shrink=0.6, label="Garn top, grid (TVDSS m)")
    ax.scatter(sticks_res.x, sticks_res.y, s=0.5, c="#e8590c", alpha=0.35, label="fault sticks over active grid (2006, depth)")
    ax.scatter(faces[:, 0], faces[:, 1], s=0.8, c="#212529", label="grid fault faces (ZCORN throw > 0.5 m)")
    ax.legend(loc="lower right", fontsize=8, markerscale=6)
    ax.set_title("Reference grid vs. fault sticks", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=9)

    ax = axes[1]
    ax.add_collection(PolyCollection(polys, facecolors="#e9ecef", edgecolors="none"))
    ax.autoscale()
    gp = picks_df[(picks_df.marker == "Garn Fm. NO 2002 Top") & picks_df.residual_m.notna()]
    lim = max(5.0, float(np.percentile(np.abs(gp.residual_m), 95))) if len(gp) else 10
    sc = ax.scatter(gp.x, gp.y, c=gp.residual_m, cmap="RdBu_r", vmin=-lim, vmax=lim, s=40,
                    edgecolors="#222222", linewidths=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="grid − pick (m), Garn top")
    for r in gp.itertuples():
        ax.annotate(r.wkey.replace("6608/10-", ""), (r.x, r.y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Garn top: reference grid vs. well picks", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"norne_root": str(io.NORNE_ROOT), "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    log("loading ATW 2013 reference grid + properties")
    cp, _ = io.read_cornerpoint(GRID_FILE)
    corners = cp.corner_xyz()
    props = {}
    for f, kw in [("GRID/ACTNUM_0704.prop", "ACTNUM"), ("PETRO/PORO_0704.prop", "PORO"), ("PETRO/NTG_0704.prop", "NTG"),
                  ("PETRO/PERM_0704.prop", "PERMX"), ("PETRO/SWINITIAL.INC", "SWATINIT"),
                  ("PETRO/EQLNUM_0704.prop", "EQLNUM"), ("PETRO/FIPNUM_0704.prop", "FIPNUM")]:
        props[kw] = io.to_ijk(io.read_grdecl_keywords(io.ATW_DIR / "INCLUDE" / f, {kw})[kw])
    act = props["ACTNUM"].astype(bool)
    ifc = Interfaces(corners)

    log("1/7 georeferencing")
    summary["georef"] = check_georef(corners, act)

    log("2/7 grid vs well picks")
    pp = pick_positions(io.read_picks(), io.read_position_logs())
    picks_df, picks_summ = check_picks(ifc, pp)
    picks_df.to_csv(OUT / "picks_vs_grid.csv", index=False)
    picks_summ.to_csv(OUT / "picks_vs_grid_summary.csv", index=False)
    summary["picks_vs_grid"] = picks_summ.to_dict("records")

    log("3/7 Not 2 horizon: 2006 geomodel input vs 2003 interpretation")
    summary["horizon_not2"] = check_horizon()

    log("4/7 grid vs fault sticks")
    per_fault, fsum, faces, sticks_res = check_faults(cp, corners, act, ifc)
    per_fault.to_csv(OUT / "faults_vs_grid.csv", index=False)
    summary["faults_vs_grid"] = fsum

    log("5/7 reference volumetrics")
    vol, vinfo, brv = volumetrics(corners, act, props)
    vol.to_csv(OUT / "volumetrics_atw.csv", index=False)
    summary["volumetrics_atw"] = vinfo

    log("6/7 blind-well selection")
    blind = select_blind_wells(picks_df, props, ifc)
    BLIND_JSON.parent.mkdir(parents=True, exist_ok=True)
    BLIND_JSON.write_text(json.dumps(blind, indent=2))
    summary["blind_wells"] = blind

    log("QC figure")
    qc_figure(corners, act, faces, sticks_res, picks_df, OUT / "qc_map.png")

    ens = None
    if not args.skip_ensemble:
        log("7/7 ensemble of 120 realizations (reads ~1.7 GB from the drive)")
        ens = ensemble(cp, act, brv, args.workers)
        ens.to_csv(OUT / "ensemble.csv", index=False)
    elif (OUT / "ensemble.csv").exists():
        log("7/7 ensemble: reusing cached ensemble.csv")
        ens = pd.read_csv(OUT / "ensemble.csv")
    if ens is not None:
        num = ens.drop(columns=["realization", "j_reversed"], errors="ignore")
        env = num.quantile([0.1, 0.5, 0.9]).T
        env.columns = ["p10", "p50", "p90"]
        env.to_csv(OUT / "ensemble_envelope.csv")
        atw_brv = {f"brv_{z}": float(brv[act & (np.broadcast_to(io.zone_of_layer()[None, None, :], act.shape) == z)].sum())
                   for z in io.ZONES}
        summary["ensemble"] = {
            "n": len(ens),
            "geometry_identical_to_atw": bool((ens.zcorn_maxdiff_m < 0.01).all() and (ens.coord_xy_maxdiff_m < 0.01).all()),
            "zcorn_maxdiff_m_max": round(float(ens.zcorn_maxdiff_m.max()), 3),
            "coord_xy_maxdiff_m_max": round(float(ens.coord_xy_maxdiff_m.max()), 3),
            "active_p10_p50_p90": env.loc["active"].round(0).tolist(),
            "actnum_agree_atw_mean": round(float(ens.actnum_agree_atw.mean()), 4),
            "atw_brv_by_zone_1e6": {k: round(v / 1e6, 2) for k, v in atw_brv.items()},
            "brv_envelope_own_actnum_1e6": {k: (env.loc[k] / 1e6).round(2).tolist() for k in atw_brv},
            "atw_active_subset_frac_min": round(float(ens.atw_active_subset_frac.min()), 4),
            "facies_envelope_on_atw_active": {k: env.loc[k].round(4).tolist() for k in env.index if k.startswith("facies")},
        }

    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done → {OUT}")


if __name__ == "__main__":
    main()
