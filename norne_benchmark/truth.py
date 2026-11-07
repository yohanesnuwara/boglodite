"""TRUTH access, candidate grids and shared geometry for the Norne benchmark.

Everything that reads the Norne reference (NORNE_ROOT, read-only) lives here;
`score.py` turns these into metrics. Nothing here writes anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from norne_benchmark import baseline as B
from norne_benchmark import io

ZONE_IFACES = B.INTERFACES                                  # surface name → interface index
BLIND_MARKERS = {"Garn top": "Garn Fm. NO 2002 Top", "Not top": "Not Fm. NO 2002 Top",
                 "Ile top": "Ile Fm. NO 2002 Top", "Tofte top": "Tofte Fm. NO 2002 Top",
                 "Tilje top": "Tilje Fm. NO 2002 Top", "Tilje base": "Tilje Fm. NO 2002 Base"}
RESERVOIR_ZONES = ["Garn", "Ile", "Tofte", "Tilje"]        # Not is inactive shale
FACIES_CODES = [0, 1, 3]


# ── geometry helpers ─────────────────────────────────────────────────────────
def fault_face_centres(g: "Grid") -> np.ndarray:
    """Centres of fault faces (ZCORN throw > 0.5 m), as the midpoint of the two cells'
    face centres — so the result does not depend on I/J ordering of the grid."""
    _, _, _, fi, fj = B.grid_fault_faces(g.cp, g.corners, g.act)
    c = g.corners
    ci = 0.5 * (c[:-1, :, :, [1, 3, 5, 7], :].mean(-2) + c[1:, :, :, [0, 2, 4, 6], :].mean(-2))[fi]
    cj = 0.5 * (c[:, :-1, :, [2, 3, 6, 7], :].mean(-2) + c[:, 1:, :, [0, 1, 4, 5], :].mean(-2))[fj]
    return np.vstack([ci, cj])


@dataclass
class Grid:
    cp: io.CornerPoint
    act: np.ndarray                      # (nx, ny, nz) bool
    props: dict[str, np.ndarray]         # (nx, ny, nz)
    path: Path

    @cached_property
    def corners(self) -> np.ndarray:
        return self.cp.corner_xyz()

    @cached_property
    def ifc(self) -> B.Interfaces:
        return B.Interfaces(self.corners)

    @cached_property
    def brv(self) -> np.ndarray:
        import xtgeo
        g = xtgeo.grid_from_file(self.path, fformat="grdecl")
        return np.asarray(g.get_bulk_volume(asmasked=False).values, dtype=float)

    @cached_property
    def centres(self) -> np.ndarray:
        return self.corners.mean(-2)


def load_grid(path: Path, props_path: Path | None = None) -> Grid:
    cp, kws = io.read_cornerpoint(path)
    props = {k: io.to_ijk(v) for k, v in io.read_grdecl_keywords(path, {"PORO", "NTG", "PERMX", "FACIES"}).items()}
    if props_path and props_path.exists():
        more = io.read_grdecl_keywords(props_path, {"PORO", "NTG", "PERMX", "FACIES"})
        props.update({k: io.to_ijk(v) for k, v in more.items()})
    act = io.to_ijk(kws["ACTNUM"]).astype(bool) if "ACTNUM" in kws else np.ones((io.NX, io.NY, io.NZ), bool)
    return Grid(cp, act, props, Path(path))


def grid_validity(g: Grid) -> dict:
    """Fraction of active cells with non-positive thickness, inverted pillars, or non-finite corners."""
    z = g.corners[..., 2]
    thick = z[..., 4:8].mean(-1) - z[..., 0:4].mean(-1)
    bad = g.act & ((thick <= 0) | ~np.isfinite(g.corners).all(axis=(-1, -2)))
    inverted = int((g.cp.coord[..., 5] < g.cp.coord[..., 2]).sum())
    return {"bad_cell_frac": float(bad.sum() / max(g.act.sum(), 1)), "inverted_pillars": inverted,
            "active_cells": int(g.act.sum())}


# ── reference (TRUTH) ────────────────────────────────────────────────────────
class Reference:
    """Lazily loaded TRUTH: ATW grid + properties, faults, horizons, blind picks, GIVENs."""

    def __init__(self):
        self.grid = load_grid(B.GRID_FILE)
        for f, kw in [("GRID/ACTNUM_0704.prop", "ACTNUM"), ("PETRO/PORO_0704.prop", "PORO"),
                      ("PETRO/NTG_0704.prop", "NTG"), ("PETRO/PERM_0704.prop", "PERMX"),
                      ("PETRO/SWINITIAL.INC", "SWATINIT"), ("PETRO/EQLNUM_0704.prop", "EQLNUM")]:
            a = io.to_ijk(io.read_grdecl_keywords(io.ATW_DIR / "INCLUDE" / f, {kw})[kw])
            if kw == "ACTNUM":
                self.grid.act = a.astype(bool)
            else:
                self.grid.props[kw] = a
        self.eqlnum = self.grid.props.pop("EQLNUM").astype(int)
        self.equil = B.read_equil().set_index("eqlnum")
        bo = B.read_bo_saturated()
        self.bo = {int(r): float(np.interp(self.equil.loc[r, "p_datum_bar"], bo.psat_bar, bo.bo)) for r in self.equil.index}
        self.blind = json.loads(B.BLIND_JSON.read_text())

    # columns over the active reservoir, used to restrict comparisons
    @cached_property
    def active_tree(self) -> cKDTree:
        col = self.grid.act.any(axis=2).ravel()
        return cKDTree(self.grid.ifc.pts[0][col, :2])

    def over_reservoir(self, xy: np.ndarray, d: float = 100.0) -> np.ndarray:
        return self.active_tree.query(xy)[0] <= d

    @cached_property
    def iface_samples(self) -> dict[int, np.ndarray]:
        """Reference interface points (x, y, z) over active columns, per interface index."""
        col = self.grid.act.any(axis=2).ravel()
        return {n: self.grid.ifc.pts[n][col] for n in set(ZONE_IFACES.values())}

    @cached_property
    def fault_points(self) -> pd.DataFrame:
        g = self.grid
        _, _, _, res = B.check_faults(g.cp, g.corners, g.act, g.ifc)
        return res[["name", "x", "y", "z"]].reset_index(drop=True)

    @cached_property
    def fault_faces(self) -> np.ndarray:
        return fault_face_centres(self.grid)

    @cached_property
    def horizon(self) -> pd.DataFrame:
        h = io.read_horizon_points(io.NORNE_ROOT / "horisons/tral+lmig32r06_fxy+Not_2_Fm_Top+TIME_TDQ+2006+input_geomodel")
        h = h[self.over_reservoir(h[["x", "y"]].values, 150)]
        return h.rename(columns={"z": "twt"})[["il", "xl", "x", "y", "twt"]].reset_index(drop=True)

    @cached_property
    def blind_picks(self) -> pd.DataFrame:
        pp = B.pick_positions(io.read_picks(), io.read_position_logs())
        pp = pp[pp.wkey.isin(set(self.blind["blind_wells"]))]
        rows = [(s, r.wkey, r.x, r.y, r.tvdss) for s, mk in BLIND_MARKERS.items()
                for r in pp[pp.marker == mk].itertuples()]
        return pd.DataFrame(rows, columns=["surface", "well", "x", "y", "tvdss"])

    @cached_property
    def eql_trees(self) -> dict[int, tuple[cKDTree, np.ndarray]]:
        """Per layer: KD-tree of active reference cell centres → EQLNUM (for candidate region assignment)."""
        out = {}
        for k in range(io.NZ):
            m = self.grid.act[:, :, k]
            if m.any():
                out[k] = (cKDTree(self.grid.centres[:, :, k][m][:, :2]), self.eqlnum[:, :, k][m])
        return out

    SH_BIN_M = 5.0          # saturation-height table resolution
    SH_MAX_M = 300.0

    @cached_property
    def sw_height(self) -> dict[str, np.ndarray]:
        """GIVEN saturation-height function per zone: PV-weighted mean SWATINIT of the
        reference in 5 m bins of height above the cell's WOC (bins without data are
        filled from the nearest populated bin). Applied identically to reference and
        candidate, so S_w responds to structural error the way a real model does."""
        g = self.grid
        zc = g.centres[..., 2]
        pv = g.brv * g.props["NTG"] * g.props["PORO"]
        woc = np.zeros(g.act.shape)
        for r, e in self.equil.iterrows():
            woc[self.eqlnum == r] = e.woc
        h = woc - zc
        zone = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
        nb = int(self.SH_MAX_M / self.SH_BIN_M)
        out = {}
        for zn in io.ZONES:
            m = g.act & (zone == zn) & (self.eqlnum > 0) & (h > 0) & (pv > 0)
            if not m.any():
                continue
            b = np.clip((h[m] / self.SH_BIN_M).astype(int), 0, nb - 1)
            num = np.bincount(b, pv[m] * g.props["SWATINIT"][m], nb)
            den = np.bincount(b, pv[m], nb)
            tab = np.full(nb, np.nan)
            tab[den > 0] = num[den > 0] / den[den > 0]
            idx = np.where(den > 0)[0]
            tab = np.interp(np.arange(nb), idx, tab[idx])     # fill gaps / extrapolate flat
            out[zn] = tab
        return out

    def sw(self, zone: str, height_above_woc: np.ndarray) -> np.ndarray:
        tab = self.sw_height.get(zone)
        if tab is None:
            return np.ones_like(height_above_woc)
        b = np.clip((height_above_woc / self.SH_BIN_M).astype(int), 0, len(tab) - 1)
        return np.where(height_above_woc > 0, tab[b], 1.0)

    @cached_property
    def faces(self) -> tuple[np.ndarray, np.ndarray]:
        return fault_face_table(self.grid)

    @cached_property
    def blind_paths(self) -> pd.DataFrame:
        return well_paths(set(self.blind["blind_wells"]))

    @cached_property
    def all_paths(self) -> pd.DataFrame:
        return well_paths(None)

    @cached_property
    def comp(self) -> np.ndarray:
        return compartments(self, self.grid, self.eqlnum)

    @cached_property
    def facies_envelope(self) -> pd.DataFrame:
        """P10/P50/P90 facies proportions per zone over the realization ensemble
        (baseline output, computed on ATW-active cells)."""
        env = pd.read_csv(B.OUT / "ensemble_envelope.csv", index_col=0)
        return env[env.index.str.startswith("facies")]

    @cached_property
    def layer_cell_trees(self) -> dict[int, tuple[cKDTree, np.ndarray]]:
        """Per layer: KD-tree of active reference cell centres (x, y) → flat cell index."""
        g = self.grid
        flat = np.arange(g.act.size).reshape(g.act.shape)
        out = {}
        for k in range(io.NZ):
            m = g.act[:, :, k]
            if m.any():
                out[k] = (cKDTree(g.centres[:, :, k][m][:, :2]), flat[:, :, k][m])
        return out

    def map_cells(self, g: "Grid", max_d: float = 100.0) -> np.ndarray:
        """For every candidate cell, the flat index of the nearest active reference cell in
        the same layer (−1 if none within max_d)."""
        out = np.full(g.act.shape, -1, dtype=np.int64)
        for k, (tree, flat) in self.layer_cell_trees.items():
            if k >= g.act.shape[2]:
                continue
            d, i = tree.query(g.centres[:, :, k, :2].reshape(-1, 2))
            out[:, :, k] = np.where(d <= max_d, flat[i], -1).reshape(g.act.shape[:2])
        return out

    @cached_property
    def volumes(self) -> pd.DataFrame:
        return volume_table(self, self.grid, self.eqlnum)


# ── volumetrics (GIVEN contacts, B_o and S_w) ────────────────────────────────
def assign_eqlnum(ref: Reference, g: Grid) -> np.ndarray:
    out = np.zeros(g.act.shape, int)
    for k, (tree, val) in ref.eql_trees.items():
        if k >= g.act.shape[2]:
            continue
        xy = g.centres[:, :, k, :2].reshape(-1, 2)
        d, idx = tree.query(xy)
        e = np.where(d <= 300, val[idx], 0).reshape(g.act.shape[:2])
        out[:, :, k] = e
    return out


def volume_table(ref: Reference, g: Grid, eqlnum: np.ndarray) -> pd.DataFrame:
    zc = g.centres[..., 2]
    zone = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
    nrv = g.brv * g.props["NTG"]
    pv = nrv * g.props["PORO"]
    rows = []
    for zn in io.ZONES:
        for r in sorted(set(np.unique(eqlnum[g.act & (zone == zn)])) - {0}):
            m = g.act & (zone == zn) & (eqlnum == r)
            e = ref.equil.loc[r]
            oil = m & (zc > e.goc) & (zc <= e.woc)
            gas = m & (zc <= e.goc)
            hc_oil = (pv[oil] * (1 - ref.sw(zn, e.woc - zc[oil]))).sum()
            hc_gas = (pv[gas] * (1 - ref.sw(zn, e.woc - zc[gas]))).sum()
            rows.append({"zone": zn, "eqlnum": int(r), "brv": g.brv[m].sum(), "nrv": nrv[m].sum(), "pv": pv[m].sum(),
                         "hcpv_oil": hc_oil, "hcpv_gas": hc_gas, "stoiip": hc_oil / ref.bo[int(r)]})
    return pd.DataFrame(rows, columns=["zone", "eqlnum", "brv", "nrv", "pv", "hcpv_oil", "hcpv_gas", "stoiip"])


def candidate_fault_points(ref: Reference, df: pd.DataFrame) -> pd.DataFrame:
    """Densify candidate sticks and keep the points in the reservoir interval over the
    active reservoir — the same filter applied to the reference sticks."""
    cand = B.densify_sticks(df.rename(columns={"fault": "name"}))
    g = ref.grid
    top = g.ifc.z(0, cand.x.values, cand.y.values, max_dist=300)
    base = g.ifc.z(io.NZ, cand.x.values, cand.y.values, max_dist=300)
    keep = (cand.z >= top - 50) & (cand.z <= base + 50) & ref.over_reservoir(cand[["x", "y"]].values)
    return cand[keep.values].reset_index(drop=True)


def horizon_on_ref_nodes(ref: Reference, df: pd.DataFrame) -> np.ndarray:
    """Interpolate a candidate time surface (any IL/XL lattice) onto the reference
    nodes; nodes farther than 4 lines from any candidate node stay NaN (uncovered)."""
    from scipy.interpolate import LinearNDInterpolator
    df = df[["il", "xl", "twt"]].dropna()
    rn = ref.horizon[["il", "xl"]].values.astype(float)
    if len(df) < 3:
        return np.full(len(rn), np.nan)
    twt_c = LinearNDInterpolator(df[["il", "xl"]].values.astype(float), df.twt.values)(rn)
    near = cKDTree(df[["il", "xl"]].values.astype(float)).query(rn)[0] <= 4
    return np.where(near, twt_c, np.nan)


def _interp_cov(g: Grid, n: int, xy: np.ndarray) -> tuple[np.ndarray, float]:
    z = g.ifc.z(n, xy[:, 0], xy[:, 1])
    return z, float(np.isfinite(z).mean())




# ── new geometry for scorer v2 ───────────────────────────────────────────────
def cell_thickness(g: Grid) -> np.ndarray:
    z = g.corners[..., 2]
    return z[..., 4:8].mean(-1) - z[..., 0:4].mean(-1)


def locate(g: Grid, xyz: np.ndarray) -> np.ndarray:
    """(N, 3) points → (N, 3) integer (i, j, k) of the containing cell, −1 where outside.
    Layer from the interpolated layer interfaces at the point's XY; column = nearest."""
    xyz = np.atleast_2d(xyz)
    nx, ny, nz = g.act.shape
    Z = np.vstack([g.ifc.z(n, xyz[:, 0], xyz[:, 1]) for n in range(nz + 1)])     # (nz+1, N)
    inside = (xyz[None, :, 2] >= Z[:-1]) & (xyz[None, :, 2] < Z[1:])
    k = np.where(inside.any(0), inside.argmax(0), -1)
    d, idx = g.ifc.tree.query(xyz[:, :2])
    i, j = np.unravel_index(idx, (nx, ny))
    ok = (k >= 0) & (d <= 150)
    return np.where(ok[:, None], np.c_[i, j, k], -1)


def fault_face_table(g: Grid, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Internal fault faces (both cells active, mean |ΔZ| over the four corner pairs > thr):
    returns (centres (N, 3), throw (N,)). Centre = midpoint of the two cells' faces."""
    z, act, c = g.cp.zcorn, g.act, g.corners
    thr_i = np.abs(z[..., :-1, 1] - z[..., 1:, 0]).mean(axis=(1, 3)).transpose(2, 1, 0)                # (nx-1, ny, nz)
    thr_j = np.abs(z[:, :, :-1, 1, :, :] - z[:, :, 1:, 0, :, :]).mean(axis=(1, 4)).transpose(2, 1, 0)  # (nx, ny-1, nz)
    mi = act[:-1] & act[1:] & (thr_i > thr)
    mj = act[:, :-1] & act[:, 1:] & (thr_j > thr)
    ci = 0.5 * (c[:-1, :, :, [1, 3, 5, 7], :].mean(-2) + c[1:, :, :, [0, 2, 4, 6], :].mean(-2))[mi]
    cj = 0.5 * (c[:, :-1, :, [2, 3, 6, 7], :].mean(-2) + c[:, 1:, :, [0, 1, 4, 5], :].mean(-2))[mj]
    return np.vstack([ci, cj]), np.r_[thr_i[mi], thr_j[mj]]


def compartments(ref: "Reference", g: Grid, eqlnum: np.ndarray, min_overlap: float = 0.5) -> np.ndarray:
    """Hydrocarbon compartments: connected components of active cells above their region's
    WOC, linked vertically and laterally wherever two cells' shared face intervals overlap by
    more than min_overlap metres (juxtaposition across faults included). −1 = not HC."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    act, zc = g.act, g.centres[..., 2]
    nx, ny, nz = act.shape
    woc = np.full(act.shape, -np.inf)
    for r, e in ref.equil.iterrows():
        woc[eqlnum == r] = e.woc
    hc = act & (eqlnum > 0) & (zc <= woc)
    idx = np.arange(act.size).reshape(act.shape)
    rows, cols = [], []
    m = hc[:, :, :-1] & hc[:, :, 1:]
    rows.append(idx[:, :, :-1][m]); cols.append(idx[:, :, 1:][m])
    cz = g.corners[..., 2]
    sides = [  # (A slice, B slice, A face corners top/bot, B face corners top/bot)
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), ([1, 3], [5, 7]), ([0, 2], [4, 6])),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), ([2, 3], [6, 7]), ([0, 1], [4, 5])),
    ]
    for sa, sb, (at, ab), (bt, bb) in sides:
        At, Ab = cz[sa][..., at].mean(-1), cz[sa][..., ab].mean(-1)
        Bt, Bb = cz[sb][..., bt].mean(-1), cz[sb][..., bb].mean(-1)
        hA, hB, iA, iB = hc[sa], hc[sb], idx[sa], idx[sb]
        for sft in range(-(nz - 1), nz):
            ka = slice(max(0, -sft), nz - max(0, sft))
            kb = slice(max(0, sft), nz - max(0, -sft))
            ov = np.minimum(Ab[..., ka], Bb[..., kb]) - np.maximum(At[..., ka], Bt[..., kb])
            m = (ov > min_overlap) & hA[..., ka] & hB[..., kb]
            rows.append(iA[..., ka][m]); cols.append(iB[..., kb][m])
    r, c = np.concatenate(rows), np.concatenate(cols)
    adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(act.size, act.size))
    _, lab = connected_components(adj, directed=False)
    return np.where(hc.ravel(), lab, -1).reshape(act.shape)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings of the same items (1 = identical partitions)."""
    if len(a) == 0:
        return float("nan")
    ct = pd.crosstab(a, b).values.astype(float)
    comb = lambda n: n * (n - 1) / 2
    s_ij, s_a, s_b, tot = comb(ct).sum(), comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum(), comb(ct.sum())
    exp = s_a * s_b / tot if tot else 0.0
    mx = 0.5 * (s_a + s_b)
    return 1.0 if np.isclose(mx, exp) else float((s_ij - exp) / (mx - exp))


def well_paths(keys: set[str] | None, step: float = 5.0) -> pd.DataFrame:
    """Well trajectories sampled every `step` m MD below 2300 m TVDSS: well, x, y, z."""
    rows = []
    for key, d in io.read_position_logs().items():
        if keys is not None and key not in keys:
            continue
        md = np.arange(d.md.min(), d.md.max(), step)
        z = np.interp(md, d.md, d.tvdss)
        m = z >= 2300
        rows.append(pd.DataFrame({"well": key, "x": np.interp(md, d.md, d.x)[m], "y": np.interp(md, d.md, d.y)[m], "z": z[m]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["well", "x", "y", "z"])
