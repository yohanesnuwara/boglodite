"""Visual benchmark reports: one self-contained report.html per scored run, and an
index.html leaderboard over all runs under a directory.

Reports contain TRUTH (difference maps reveal the reference), so they are written
under outputs/norne_scores/ and must never be readable by agents during a run.

    uv run --no-sync python -m norne_benchmark.report --candidate DIR      # one run (scores it if needed)
    uv run --no-sync python -m norne_benchmark.report --index ROOT         # leaderboard over ROOT/**/score.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from norne_benchmark import io
from norne_benchmark import score as S

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _ecl(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).transpose(2, 1, 0).ravel()


def _r(a, nd=1):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


class Ctx:
    """Shared reference frame and per-layer reference trees for one report session."""

    def __init__(self, ref: S.Reference):
        self.ref = ref
        g = ref.grid
        c = g.centres[g.act]
        self.cx, self.cy = float(np.round(c[:, 0].mean())), float(np.round(c[:, 1].mean()))
        self.layer_trees = {}
        for k in range(io.NZ):
            m = g.act[:, :, k]
            if m.any():
                cc = g.centres[:, :, k][m]
                self.layer_trees[k] = (cKDTree(cc[:, :2]), cc[:, 2])
        col = g.act.any(axis=2)
        top = g.corners[:, :, 0][:, :, [0, 1, 3, 2], :2][col]           # (n, 4, 2) column outlines
        self.col_polys = (top - [self.cx, self.cy]).round(0)


def cell_depth_error(ctx: Ctx, g: S.Grid) -> np.ndarray:
    """Candidate cell-centre depth minus the nearest active reference cell in the same
    layer (≤ 100 m away); NaN where there is no reference cell nearby."""
    err = np.full(g.act.shape, np.nan)
    for k, (tree, z) in ctx.layer_trees.items():
        if k >= g.act.shape[2]:
            continue
        m = g.act[:, :, k]
        if not m.any():
            continue
        cc = g.centres[:, :, k][m]
        d, idx = tree.query(cc[:, :2])
        e = np.where(d <= 100, cc[:, 2] - z[idx], np.nan)
        layer = np.full(m.shape, np.nan)
        layer[m] = e
        err[:, :, k] = layer
    return err


def grid_payload(ctx: Ctx, g: S.Grid) -> dict:
    act = g.act
    cen = g.centres[act]
    off = float(np.hypot(cen[:, 0].mean() - ctx.cx, cen[:, 1].mean() - ctx.cy)) if act.any() else None
    # frame the 3D view on the candidate itself (it may be far from the reference)
    ox, oy = (float(cen[:, 0].mean()), float(cen[:, 1].mean())) if act.any() else (ctx.cx, ctx.cy)
    coord = g.cp.coord.copy()
    coord[..., [0, 3]] -= ox
    coord[..., [1, 4]] -= oy
    z0 = float(np.floor(np.nanmin(g.cp.zcorn)))
    zc = np.clip(np.round((g.cp.zcorn - z0) * 10), 0, 65535).astype(np.uint16)
    err = cell_depth_error(ctx, g)
    e16 = np.where(np.isfinite(err), np.clip(np.round(err * 10), -32000, 32000), -32768).astype(np.int16)
    nx, ny, nz = act.shape
    return {"NX": nx, "NY": ny, "NZ": nz, "z0": z0, "ox": round(ox - ctx.cx, 1), "oy": round(oy - ctx.cy, 1),
            "coord": _b64(coord.astype(np.float32).ravel()), "zcorn": _b64(zc.ravel()),
            "act": _b64(_ecl(act).astype(np.uint8)), "err": _b64(_ecl(e16)),
            "offset_m": None if off is None else round(off, 0),
            "err_coverage": round(float(np.isfinite(err[act]).mean()), 4) if act.any() else 0.0}


def zone_top_maps(ctx: Ctx, g: S.Grid) -> dict:
    out = {}
    for s, n in S.ZONE_IFACES.items():
        p = ctx.ref.iface_samples[n]
        z = g.ifc.z(n, p[:, 0], p[:, 1])
        out[s] = _r(z - p[:, 2], 1)
    thick = {}
    for zn in S.RESERVOIR_ZONES:
        k1, k2 = io.ZONES[zn]
        p = ctx.ref.iface_samples[k1 - 1]
        t_ref = ctx.ref.grid.ifc.z(k2, p[:, 0], p[:, 1]) - p[:, 2]
        t_c = g.ifc.z(k2, p[:, 0], p[:, 1]) - g.ifc.z(k1 - 1, p[:, 0], p[:, 1])
        thick[zn] = _r(t_c - t_ref, 1)
    return {"polys": ctx.col_polys.ravel().tolist(), "diff": out, "thick": thick}


def fault_payload(ctx: Ctx, df: pd.DataFrame) -> dict:
    ref = ctx.ref
    R = ref.fault_points
    C = S.candidate_fault_points(ref, df) if {"x", "y", "z", "stick", "fault"} <= set(df.columns) else pd.DataFrame(columns=["name", "x", "y", "z"])
    rng = np.random.default_rng(0)
    if len(C):
        d_rc = cKDTree(C[["x", "y", "z"]].values).query(R[["x", "y", "z"]].values)[0]
        d_cr = cKDTree(R[["x", "y", "z"]].values).query(C[["x", "y", "z"]].values)[0]
    else:
        d_rc, d_cr = np.full(len(R), np.inf), np.array([])
    tau = S.FAULT_M
    per = (pd.DataFrame({"name": R.name, "d": d_rc}).groupby("name")
           .agg(n=("d", "size"), recall=("d", lambda v: float((v <= tau).mean())), median=("d", "median"))
           .sort_values("n", ascending=False).reset_index())
    per["name"] = per.name.str.replace(r"^DepthTDQ_kseim_|_r06_RR$", "", regex=True)

    def sub(df_, d, n):
        i = rng.choice(len(df_), size=min(n, len(df_)), replace=False) if len(df_) else np.array([], int)
        x = df_.x.values[i] - ctx.cx
        y = df_.y.values[i] - ctx.cy
        return np.c_[x, y, df_.z.values[i], np.minimum(d[i], 9999)].round(1).ravel().tolist()

    return {"ref": sub(R, d_rc, 7000), "cand": sub(C, d_cr, 7000),
            "per_fault": [{"name": r.name, "n": int(r.n), "recall": round(r.recall, 3),
                           "median": None if not np.isfinite(r.median) else round(float(r.median), 1)}
                          for r in per.itertuples()]}


def horizon_payload(ctx: Ctx, df: pd.DataFrame) -> dict:
    ref = ctx.ref
    h = ref.horizon
    twt_c = S.horizon_on_ref_nodes(ref, df)
    keep = ((h.il % 2) == 0) & ((h.xl % 2) == 0)
    return {"x": _r(h.x[keep] - ctx.cx, 0), "y": _r(h.y[keep] - ctx.cy, 0),
            "d": _r((twt_c - h.twt.values)[keep.values], 1), "cell": 25.0}


def prop_hists(ctx: Ctx, g: S.Grid) -> dict:
    ref = ctx.ref
    if any(k not in g.props for k in ("PORO", "NTG", "PERMX")):
        return {}
    spec = {"PORO": (0, 0.4, lambda a: a), "NTG": (0, 1, lambda a: a),
            "LOGK": (-2, 4, lambda a: np.log10(np.clip(a, 0.01, None)))}
    zg = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
    zr = np.broadcast_to(io.zone_of_layer()[None, None, :], ref.grid.act.shape)
    out = {}
    for key, (lo, hi, f) in spec.items():
        src = "PERMX" if key == "LOGK" else key
        bins = np.linspace(lo, hi, 41)
        out[key] = {"lo": lo, "hi": hi, "zones": {}}
        for zn in S.RESERVOIR_ZONES:
            a = f(g.props[src][g.act & (zg == zn)])
            b = f(ref.grid.props[src][ref.grid.act & (zr == zn)])
            hc = np.histogram(np.clip(a, lo, hi), bins)[0]
            hr = np.histogram(np.clip(b, lo, hi), bins)[0]
            out[key]["zones"][zn] = {"cand": np.round(hc / max(hc.sum(), 1), 4).tolist(),
                                     "ref": np.round(hr / max(hr.sum(), 1), 4).tolist()}
    return out


def build_report(cand_dir: Path, ctx: Ctx, res: dict | None = None, out: Path | None = None) -> Path:
    cand_dir = Path(cand_dir)
    if res is None:
        res = json.loads((cand_dir / "score.json").read_text()) if (cand_dir / "score.json").exists() \
            else S.score_candidate(cand_dir, ctx.ref)
    data = {"run": cand_dir.name, "res": res, "tau": {"depth_m": 10.0, "fault_m": S.FAULT_M, "time_ms": 15.0, "thick_m": 5.0}}
    if (cand_dir / "grid.grdecl").exists():
        try:
            g = S.load_grid(cand_dir / "grid.grdecl", cand_dir / "props.grdecl")
            data["grid"] = grid_payload(ctx, g)
            data["maps"] = zone_top_maps(ctx, g)
            data["hist"] = prop_hists(ctx, g)
        except Exception as e:      # unreadable grid: the scorecard already says overt failure
            data["grid_error"] = str(e)
    if (cand_dir / "faults.csv").exists():
        data["faults"] = fault_payload(ctx, pd.read_csv(cand_dir / "faults.csv"))
    if (cand_dir / "horizon_not2_time.csv").exists():
        hz = pd.read_csv(cand_dir / "horizon_not2_time.csv")
        if {"il", "xl", "twt"} <= set(hz.columns):
            data["horizon"] = horizon_payload(ctx, hz)
    out = out or (cand_dir / "report.html")
    html = (TEMPLATES / "report.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out.write_text(html.replace("__TITLE__", f"Norne run · {cand_dir.name}"))
    return out


def build_index(root: Path, title: str | None = None) -> Path:
    runs = []
    for f in sorted(Path(root).rglob("score.json")):
        r = json.loads(f.read_text())
        rel = f.parent.relative_to(root)
        runs.append({"run": str(rel), "report": str(rel / "report.html") if (f.parent / "report.html").exists() else None,
                     "composite": r["composite"], "coverage": r.get("stage_coverage"), "manifest": r.get("manifest", {}),
                     "stages": {s: {"score": d["score"], "outcome": d["outcome"], "name": d.get("name", s),
                                    "passed": d.get("passed"), "n": d.get("n_primary")} for s, d in r["stages"].items()},
                     "checks": [r.get("checks_passed"), r.get("checks_total")],
                     "headline": {"stoiip": (r["stages"].get("VOL", {}).get("info", {}) or {}).get("stoiip_1e6_sm3")}})
    out = Path(root) / "index.html"
    html = (TEMPLATES / "index.html").read_text().replace("__DATA__", json.dumps({"root": str(root), "title": title, "runs": runs}))
    out.write_text(html)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--index", type=Path)
    a = ap.parse_args()
    if a.candidate:
        print("→", build_report(a.candidate, Ctx(S.Reference())))
    if a.index:
        print("→", build_index(a.index))
    if not (a.candidate or a.index):
        ap.error("--candidate DIR and/or --index ROOT")


if __name__ == "__main__":
    main()
