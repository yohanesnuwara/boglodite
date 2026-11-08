"""Build a self-contained 3D cell viewer of the Norne ATW-2013 reference model.

Packs the corner-point grid (COORD/ZCORN/ACTNUM), reference properties, the
Eclipse FAULTS keyword faces, 2006 depth fault sticks, wells, well picks and the
baseline benchmark results into tmp/norne_3d.html (three.js, renders cells with
edges and fault faces). Reads NORNE_ROOT read-only.

    PYTHONPATH=. uv run python sandbox/norne_viewer.py
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from norne_benchmark import baseline as b
from norne_benchmark import io

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp/norne_3d.html"
TEMPLATE = REPO / "sandbox/norne_viewer_template.html"
LAYER_NAMES = ["Garn 3", "Garn 2", "Garn 1", "Not", "Ile 2.2", "Ile 2.1.3", "Ile 2.1.2", "Ile 2.1.1",
               "Ile 1.3", "Ile 1.2", "Ile 1.1", "Tofte 2.2", "Tofte 2.1.3", "Tofte 2.1.2", "Tofte 2.1.1",
               "Tofte 1.2.2", "Tofte 1.2.1", "Tofte 1.1", "Tilje 4", "Tilje 3", "Tilje 2", "Tilje 1"]
PICK_MARKERS = ["Garn Fm. NO 2002 Top", "Not Fm. NO 2002 Top", "Ile Fm. NO 2002 Top",
                "Tofte Fm. NO 2002 Top", "Tilje Fm. NO 2002 Top", "Tilje Fm. NO 2002 Base"]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def ecl(a: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) → flat Eclipse order (I fastest, then J, then K)."""
    return a.transpose(2, 1, 0).ravel()


def main() -> None:
    cp, _ = io.read_cornerpoint(b.GRID_FILE)
    c = cp.corner_xyz()
    P = {kw: io.to_ijk(io.read_grdecl_keywords(io.ATW_DIR / "INCLUDE" / f, {kw})[kw])
         for f, kw in [("GRID/ACTNUM_0704.prop", "ACTNUM"), ("PETRO/PORO_0704.prop", "PORO"),
                       ("PETRO/NTG_0704.prop", "NTG"), ("PETRO/PERM_0704.prop", "PERMX"),
                       ("PETRO/SWINITIAL.INC", "SWATINIT"), ("PETRO/EQLNUM_0704.prop", "EQLNUM"),
                       ("PETRO/FIPNUM_0704.prop", "FIPNUM")]}
    act = P["ACTNUM"].astype(bool)
    cx = float(np.round(c[act][..., 0].mean()))
    cy = float(np.round(c[act][..., 1].mean()))

    coord = cp.coord.copy()
    coord[..., [0, 3]] -= cx
    coord[..., [1, 4]] -= cy
    z0 = float(np.floor(cp.zcorn.min()))
    zc = np.round((cp.zcorn - z0) * 10).astype(np.uint16)          # decimetres above z0

    # Eclipse FAULTS keyword → per-cell +I / +J face fault index (1-based, 0 = none)
    ef = io.eclipse_faults(io.ATW_DIR / "INCLUDE/FAULT/FAULT_JUN_05.INC")
    names = sorted(ef.name.unique())
    fidx = {n: i + 1 for i, n in enumerate(names)}
    fI = np.zeros(act.shape, np.uint8)
    fJ = np.zeros(act.shape, np.uint8)
    for r in ef.itertuples():
        tgt = fI if r.face.startswith("X") else fJ
        tgt[r.i1 - 1:r.i2, r.j1 - 1:r.j2, r.k1 - 1:r.k2] = fidx[r.name]

    logk = np.log10(np.clip(P["PERMX"], 0.01, None))
    grid = {
        "NX": io.NX, "NY": io.NY, "NZ": io.NZ, "z0": z0, "cx": cx, "cy": cy,
        "coord": b64(coord.astype(np.float32).ravel()), "zcorn": b64(zc.ravel()),
        "act": b64(ecl(act).astype(np.uint8)),
        "fI": b64(ecl(fI)), "fJ": b64(ecl(fJ)), "faultNames": names,
        "props": {
            "PORO": b64(np.round(ecl(P["PORO"]) * 10000).astype(np.uint16)),
            "NTG": b64(np.round(ecl(P["NTG"]) * 10000).astype(np.uint16)),
            "LOGK": b64(np.round((ecl(logk) + 2) * 1000).astype(np.uint16)),
            "SWAT": b64(np.round(ecl(P["SWATINIT"]) * 10000).astype(np.uint16)),
            "EQLNUM": b64(ecl(P["EQLNUM"]).astype(np.uint8)),
            "FIPNUM": b64(ecl(P["FIPNUM"]).astype(np.uint8)),
        },
        "layerNames": LAYER_NAMES,
        "equil": b.read_equil().to_dict("records"),
    }

    # 2006 depth fault sticks over the active reservoir, as polylines grouped by fault
    fs = io.read_fault_sticks(io.NORNE_ROOT / "FAULTS/kseim_r06_faults_depth")
    ifc = b.Interfaces(c)
    col_act = act.any(axis=2).ravel()
    from scipy.spatial import cKDTree
    act_tree = cKDTree(ifc.pts[0][col_act, :2])
    d_act, _ = act_tree.query(fs[["x", "y"]].values)
    fs = fs.assign(near=d_act <= 150)
    sticks = {}
    for (sid, name), g in fs.groupby(["stick", "name"], sort=False):
        if not g.near.any() or g.z.max() < 2300 or g.z.min() > 3200:
            continue
        short = re.sub(r"^DepthTDQ_kseim_|_r06_RR$", "", name)
        sticks.setdefault(short, []).append(np.round(np.c_[g.x - cx, g.y - cy, g.z], 1).ravel().tolist())

    # fault surfaces: ribbons between neighbouring sticks of each fault, clipped to the grid depth span
    zlo, zhi = float(c[act][..., 2].min()) - 40, float(c[act][..., 2].max()) + 40
    fvg = pd.read_csv(b.OUT / "faults_vs_grid.csv")
    fvg["short"] = fvg.name.str.replace(r"^DepthTDQ_kseim_|_r06_RR$", "", regex=True)
    fdist = fvg.set_index("short")[["d_grid_median", "d_grid_p90"]].to_dict("index")
    NS = 10
    fsurf = []
    for name, sl in sticks.items():
        rows = []
        for s in sl:
            p = np.array(s).reshape(-1, 3)
            p = p[np.argsort(p[:, 2])]
            mx, my = p[:, 0].mean() + cx, p[:, 1].mean() + cy
            if act_tree.query([mx, my])[0] > 150:          # stick not over the active reservoir
                rows.append(None)
                continue
            # clip to the local reservoir interval (top of grid − 30 m … base + 30 m at this XY)
            top = float(ifc.z(0, mx, my, max_dist=400)[0])
            bot = float(ifc.z(io.NZ, mx, my, max_dist=400)[0])
            if not (np.isfinite(top) and np.isfinite(bot)):
                rows.append(None)
                continue
            a, bb = max(p[0, 2], top - 30, zlo), min(p[-1, 2], bot + 30, zhi)
            if bb - a < 5:
                rows.append(None)
                continue
            zz = np.linspace(a, bb, NS)
            rows.append(np.c_[np.interp(zz, p[:, 2], p[:, 0]), np.interp(zz, p[:, 2], p[:, 1]), zz])
        verts, tris = [], []
        for r0, r1 in zip(rows[:-1], rows[1:]):
            if r0 is None or r1 is None:
                continue
            if np.hypot(*(r0.mean(0)[:2] - r1.mean(0)[:2])) > 600:   # not neighbouring sticks
                continue
            base = len(verts)
            verts += r0.tolist() + r1.tolist()
            for q in range(NS - 1):
                a0, a1, b0, b1 = base + q, base + q + 1, base + NS + q, base + NS + q + 1
                tris += [a0, b0, a1, a1, b0, b1]
        if tris:
            d = fdist.get(name, {})
            fsurf.append({"name": name, "v": np.round(np.array(verts), 1).ravel().tolist(), "t": tris,
                          "d50": d.get("d_grid_median"), "d90": d.get("d_grid_p90")})

    # OWC surfaces (official depth), triangulated on their IL/XL lattice
    owcs = []
    for yr in (2001, 2003, 2004, 2006):
        h = io.read_horizon_points(io.NORNE_ROOT / f"horisons/bosd-4D_OWC-{yr}_OFFICIAL_DEPTH")
        st = 3
        h = h[((h.il - h.il.min()) % st == 0) & ((h.xl - h.xl.min()) % st == 0)].reset_index(drop=True)
        key = {(int(r.il), int(r.xl)): n for n, r in enumerate(h.itertuples())}
        tris = []
        for (il, xl), n in key.items():
            a, bq, cq = key.get((il + st, xl)), key.get((il, xl + st)), key.get((il + st, xl + st))
            if a is not None and bq is not None and cq is not None:
                tris += [n, a, cq, n, cq, bq]
        owcs.append({"label": f"OWC {yr}", "year": yr, "median": round(float(h.z.median()), 1),
                     "v": np.round(np.c_[h.x - cx, h.y - cy, h.z], 1).ravel().tolist(), "t": tris})

    # wells (reservoir section)
    blind = set(json.loads(b.BLIND_JSON.read_text())["blind_wells"])
    wells = []
    for key, d in sorted(io.read_position_logs().items()):
        d = d[d.tvdss >= 2150]
        if len(d) < 2:
            continue
        d = d.iloc[:: max(1, len(d) // 80)]
        wells.append({"name": key.replace("6608/10-", ""), "blind": key in blind,
                      "p": np.round(np.c_[d.x - cx, d.y - cy, d.tvdss], 1).ravel().tolist()})

    # 2002 picks with grid residuals (baseline output)
    pk = pd.read_csv(b.OUT / "picks_vs_grid.csv")
    pk = pk[pk.marker.isin(PICK_MARKERS) & pk.residual_m.notna()]
    picks = [{"w": r.wkey.replace("6608/10-", ""), "m": r.marker.replace(" NO 2002", ""),
              "p": [round(r.x - cx, 1), round(r.y - cy, 1), round(r.pick_tvdss, 1)],
              "g": round(r.grid_tvdss, 1), "r": round(r.residual_m, 1)} for r in pk.itertuples()]

    summ = json.loads((b.OUT / "baseline_summary.json").read_text())
    ens = pd.read_csv(b.OUT / "ensemble.csv")
    n_distinct = int(ens.filter(like="facies").round(6).drop_duplicates().shape[0])
    bench = {
        "vol": summ["volumetrics_atw"]["totals_1e6_m3"],
        "picks": [p for p in summ["picks_vs_grid"] if p["marker"] in PICK_MARKERS],
        "faults": summ["faults_vs_grid"],
        "georef": summ["georef"]["active_cells_inside_seismic"],
        "ens_n": int(len(ens)), "ens_distinct": n_distinct,
        "blind": json.loads(b.BLIND_JSON.read_text())["blind_families"],
        "horizon": summ["horizon_not2"],
    }
    data = {"grid": grid, "sticks": sticks, "fsurf": fsurf, "owc": owcs, "wells": wells, "picks": picks, "bench": bench}
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
