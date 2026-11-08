"""Scorer v2 for the Norne static-modelling benchmark — spec docs/NORNE_BENCHMARK.md §5–§7.

Every metric is reported in geoscience terms (MAE in m or ms, F1, %, R²) against a
stated tolerance, with pass/fail. A 0–1 score per metric exists only to rank runs.

A *candidate* is a directory produced by an agent run (or by hand):

    grid.grdecl            SPECGRID 46 112 22, COORD, ZCORN, ACTNUM  (+ PORO, NTG, PERMX, FACIES optional)
    props.grdecl           optional; PORO / NTG / PERMX / FACIES if not in grid.grdecl
    faults.csv             depth fault sticks: x, y, z, stick, fault        (TVDSS m, positive down)
    horizon_not2_time.csv  Not 2 Fm top in time: il, xl, twt                (TWT ms)
    manifest.json          optional run metadata (model, pattern, tokens, …)

    uv run --no-sync python -m norne_benchmark.score --candidate DIR [--out score.json]
    uv run --no-sync python -m norne_benchmark.score --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from norne_benchmark import io
from norne_benchmark.truth import (  # noqa: F401  (re-exported for report.py / selftest.py)
    BLIND_MARKERS, FACIES_CODES, RESERVOIR_ZONES, ZONE_IFACES, Grid, Reference, _interp_cov, adjusted_rand,
    assign_eqlnum, candidate_fault_points, cell_thickness, compartments, fault_face_table, horizon_on_ref_nodes,
    load_grid, locate, volume_table)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/norne_scores"

# Stage keys, display names and weights (Very high = 3, High = 2)
STAGES = {
    "HZ": ("Seismic horizons", 3), "FLT": ("Fault interpretation", 3), "STRUCT": ("Structural modelling", 3),
    "STRAT": ("Stratigraphic modelling", 2), "FAC": ("Facies modelling", 2), "PETRO": ("Petrophysical modelling", 3),
    "VOL": ("Reservoir evaluation", 3), "CONN": ("Reservoir connectivity", 3),
}
FAULT_M = 100.0          # distance within which a fault point counts as found
MIN_FAULT_PTS = 30       # reference faults shorter than this (×10 m of stick) are ignored
MIN_COVERAGE = 0.5       # below this, a metric is treated as failed (silent failure)


# ── metric model ─────────────────────────────────────────────────────────────
@dataclass
class Metric:
    """A metric in domain units judged against a tolerance.

    sense "le": good when value ≤ tol (errors: MAE, %…); score = exp(−value/tol).
    sense "ge": good when value ≥ tol (F1, R², coverage); score = exp(−(1−value)/(1−tol)).
    """
    key: str
    label: str
    unit: str
    value: float | None
    tol: float
    sense: str = "le"
    primary: bool = True
    n: int | None = None
    note: str = ""

    @property
    def finite(self) -> bool:
        return self.value is not None and math.isfinite(self.value)

    @property
    def err(self) -> float:          # error normalised so that err = 1 at the tolerance
        if not self.finite:
            return math.inf
        if self.sense == "le":
            return max(abs(self.value), 0.0) / self.tol
        return max(1.0 - self.value, 0.0) / max(1.0 - self.tol, 1e-9)

    @property
    def passed(self) -> bool:
        return self.err <= 1.0

    @property
    def score(self) -> float:
        return 0.0 if not math.isfinite(self.err) else math.exp(-self.err)

    def as_dict(self):
        v = round(float(self.value), 4) if self.finite else None
        return {"label": self.label, "unit": self.unit, "value": v, "tol": self.tol, "sense": self.sense,
                "pass": self.passed, "score": round(self.score, 4), "primary": self.primary, "n": self.n,
                "note": self.note}


@dataclass
class StageResult:
    stage: str
    metrics: list[Metric] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    overt: str | None = None

    @property
    def primary(self) -> list[Metric]:
        return [m for m in self.metrics if m.primary]

    @property
    def score(self) -> float:
        if self.overt or not self.primary:
            return 0.0
        return float(np.mean([m.score for m in self.primary]))

    @property
    def outcome(self) -> str:
        if self.overt:
            return "overt_failure"
        if any(m.err > 3 for m in self.primary):
            return "silent_failure"
        return "correct" if all(m.passed for m in self.primary) else "degraded"

    def as_dict(self):
        return {"name": STAGES[self.stage][0], "weight": STAGES[self.stage][1], "score": round(self.score, 4),
                "outcome": self.outcome, "overt": self.overt, "passed": sum(m.passed for m in self.primary),
                "n_primary": len(self.primary), "metrics": {m.key: m.as_dict() for m in self.metrics}, "info": self.info}


def mae(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.mean(np.abs(a))) if len(a) else math.nan


def rmse(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else math.nan


def r2(pred, obs):
    m = np.isfinite(pred) & np.isfinite(obs)
    if m.sum() < 3 or np.var(obs[m]) == 0:
        return math.nan
    return float(1 - np.sum((pred[m] - obs[m]) ** 2) / np.sum((obs[m] - obs[m].mean()) ** 2))


def gated(value, coverage):
    """A value measured on too little of the reference is not a result."""
    return value if coverage >= MIN_COVERAGE else math.nan


# ── 1. Seismic horizons ──────────────────────────────────────────────────────
def score_horizons(ref: Reference, df: pd.DataFrame) -> StageResult:
    st = StageResult("HZ")
    if not {"il", "xl", "twt"} <= set(df.columns):
        st.overt = "horizon_not2_time.csv needs columns il, xl, twt"
        return st
    twt = horizon_on_ref_nodes(ref, df)
    ok = np.isfinite(twt)
    cov = float(ok.mean())
    d = twt[ok] - ref.horizon.twt.values[ok]
    st.metrics = [
        Metric("time_mae", "Not 2 Fm top · time MAE", "ms", gated(mae(d), cov), 15, n=int(ok.sum())),
        Metric("coverage", "Reservoir area covered", "frac", cov, 0.90, "ge"),
        Metric("time_rmse", "Not 2 Fm top · time RMSE", "ms", gated(rmse(d), cov), 20, primary=False),
        Metric("time_bias", "Not 2 Fm top · mean bias", "ms", float(d.mean()) if len(d) else None, 10, primary=False),
    ]
    st.info = {"reference_nodes": len(ref.horizon), "reference": "Not 2 Fm top, 2006 geomodel input (TWT)",
               "input_floor_ms": 22.2,
               "floor_note": "Checkshot/TD-table times of the Not 2 well tops differ from the reference horizon by "
                             "MAE 22.2 ms (median +11 ms; reference on a trough). The reference was converted with a "
                             "different time–depth model; spec §10.7. Tolerance stays 15 ms."}
    return st


# ── 2. Fault interpretation ──────────────────────────────────────────────────
def score_faults(ref: Reference, df: pd.DataFrame) -> StageResult:
    st = StageResult("FLT")
    need = {"x", "y", "z", "stick", "fault"}
    if not need <= set(df.columns):
        st.overt = f"faults.csv missing columns {sorted(need - set(df.columns))}"
        return st
    C = candidate_fault_points(ref, df)
    R = ref.fault_points
    rsize = R.name.value_counts()
    rnames = list(rsize[rsize >= MIN_FAULT_PTS].index)
    R = R[R.name.isin(rnames)].reset_index(drop=True)
    csize = C.name.value_counts()
    cnames = list(csize[csize >= MIN_FAULT_PTS].index)
    C = C[C.name.isin(cnames)].reset_index(drop=True)
    if C.empty:
        st.metrics = [Metric("fault_f1", "Fault F1 (fault-level)", "", 0.0, 0.75, "ge", n=0),
                      Metric("point_f1", "Fault F1 (point-level)", "", 0.0, 0.75, "ge")]
        st.info = {"reference_faults": len(rnames), "candidate_faults": 0}
        return st
    # point level
    d_rc = cKDTree(C[["x", "y", "z"]].values).query(R[["x", "y", "z"]].values)[0]
    d_cr = cKDTree(R[["x", "y", "z"]].values).query(C[["x", "y", "z"]].values)[0]
    p_rec, p_prec = float((d_rc <= FAULT_M).mean()), float((d_cr <= FAULT_M).mean())
    p_f1 = 2 * p_rec * p_prec / (p_rec + p_prec) if p_rec + p_prec else 0.0
    # fault level: coverage of each reference fault by each candidate fault, and vice versa
    rpts = {r: R.loc[R.name == r, ["x", "y", "z"]].values for r in rnames}
    cpts = {c: C.loc[C.name == c, ["x", "y", "z"]].values for c in cnames}
    ctree = {c: cKDTree(p) for c, p in cpts.items()}
    rtree = {r: cKDTree(p) for r, p in rpts.items()}
    cov_rc = np.array([[(ctree[c].query(rpts[r])[0] <= FAULT_M).mean() for c in cnames] for r in rnames])
    cov_cr = np.array([[(rtree[r].query(cpts[c])[0] <= FAULT_M).mean() for c in cnames] for r in rnames])
    ri, ci = linear_sum_assignment(-(cov_rc + cov_cr))
    tp_pairs = [(rnames[a], cnames[b]) for a, b in zip(ri, ci) if cov_rc[a, b] >= 0.5 and cov_cr[a, b] >= 0.5]
    tp = len(tp_pairs)
    prec, rec = tp / len(cnames), tp / len(rnames)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    matched = {r for r, _ in tp_pairs}
    st.metrics = [
        Metric("fault_f1", "Fault F1 (fault-level, one-to-one matched)", "", f1, 0.75, "ge", n=len(rnames)),
        Metric("point_f1", "Fault F1 (point-level, within 100 m)", "", p_f1, 0.75, "ge", n=len(R)),
        Metric("fault_precision", "Fault precision", "", prec, 0.75, "ge", primary=False),
        Metric("fault_recall", "Fault recall", "", rec, 0.75, "ge", primary=False),
        Metric("median_distance", "Median distance to nearest fault", "m",
               0.5 * (np.median(d_rc) + np.median(d_cr)), FAULT_M, primary=False),
    ]
    st.info = {"reference_faults": len(rnames), "candidate_faults": len(cnames), "matched": tp,
               "missed": sorted(set(rnames) - matched)[:40],
               "rule": "a fault matches when ≥ 50 % of each lies within 100 m of the other (one-to-one)"}
    return st


# ── 3. Structural modelling ──────────────────────────────────────────────────
def score_structure(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("STRUCT")
    per, covs, allr = {}, [], []
    for s, n in ZONE_IFACES.items():
        p = ref.iface_samples[n]
        z, cov = _interp_cov(g, n, p[:, :2])
        covs.append(cov)
        r = z - p[:, 2]
        allr.append(r)
        per[s] = {"mae_m": round(mae(r), 2), "rmse_m": round(rmse(r), 2), "bias_m": round(float(np.nanmean(r)), 2) if cov else None}
    cov = float(np.mean(covs))
    surf_mae = gated(float(np.nanmean([v["mae_m"] for v in per.values()])), cov)
    # blind wells
    bp = ref.blind_picks
    res, picks = [], []
    for s, n in ZONE_IFACES.items():
        sub = bp[bp.surface == s]
        if sub.empty:
            continue
        z, _ = _interp_cov(g, n, sub[["x", "y"]].values)
        r = z - sub.tvdss.values
        res.append(r)
        picks += [{"surface": s, "well": w.replace("6608/10-", ""), "pick": round(float(t), 2),
                   "residual": None if not np.isfinite(v) else round(float(v), 2)} for w, t, v in zip(sub.well, sub.tvdss, r)]
    r = np.concatenate(res) if res else np.array([])
    well_cov = float(np.isfinite(r).mean()) if len(r) else 0.0
    # fault throw: every reference and candidate fault face, matched within 100 m; unmatched → error = its throw
    rf, rt = ref.faces
    cf, ct = fault_face_table(g)
    if len(cf):
        d_r, i_r = cKDTree(cf).query(rf)
        e_r = np.where(d_r <= FAULT_M, np.abs(ct[i_r] - rt), rt)
        d_c, i_c = cKDTree(rf).query(cf)
        e_c = np.where(d_c <= FAULT_M, np.abs(rt[i_c] - ct), ct)
    else:
        e_r, e_c = rt.copy(), np.array([])
    thr_mae = float(np.mean(np.r_[e_r, e_c]))
    st.metrics = [
        Metric("surface_mae", "Zone-top surfaces · depth MAE", "m", surf_mae, 10, n=int(sum(len(ref.iface_samples[n]) for n in ZONE_IFACES.values()))),
        Metric("well_mae", "Blind-well picks · depth MAE", "m", gated(mae(r), well_cov), 10, n=int(len(r)),
               note="the reference grid itself misses these by ~5 m"),
        Metric("throw_mae", "Fault throw MAE (missing/extra faults count fully)", "m", thr_mae, 10, n=int(len(rf))),
        Metric("surface_rmse", "Zone-top surfaces · depth RMSE", "m", gated(rmse(np.concatenate(allr)), cov), 15, primary=False),
        Metric("well_rmse", "Blind-well picks · depth RMSE", "m", gated(rmse(r), well_cov), 15, primary=False),
        Metric("coverage", "Reservoir area covered", "frac", cov, 0.90, "ge", primary=False),
    ]
    st.info = {"per_surface": per, "picks": picks, "fault_faces": {"reference": int(len(rf)), "candidate": int(len(cf))},
               "reference_throw_m": {"median": round(float(np.median(rt)), 1), "p90": round(float(np.percentile(rt, 90)), 1)},
               "note": "Garn-top picks have an open TVD issue (spec §11)"}
    return st


# ── 4. Stratigraphic modelling ───────────────────────────────────────────────
def score_strat(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("STRAT")
    per, pres_ok, n_pres, covs = {}, 0, 0, []
    for zn in RESERVOIR_ZONES:
        k1, k2 = io.ZONES[zn]
        p = ref.iface_samples[k1 - 1]
        t_ref = ref.grid.ifc.z(k2, p[:, 0], p[:, 1]) - p[:, 2]
        zt, cov = _interp_cov(g, k1 - 1, p[:, :2])
        zb, _ = _interp_cov(g, k2, p[:, :2])
        t_c = zb - zt
        covs.append(cov)
        m = np.isfinite(t_c) & np.isfinite(t_ref)
        per[zn] = {"mae_m": round(mae(t_c - t_ref), 2), "ref_mean_m": round(float(np.nanmean(t_ref)), 1),
                   "cand_mean_m": round(float(np.nanmean(t_c)), 1) if m.any() else None}
        pres_ok += int(((t_c[m] > 0.5) == (t_ref[m] > 0.5)).sum())
        n_pres += int(m.sum())
    cov = float(np.mean(covs))
    th_mae = gated(float(np.nanmean([v["mae_m"] for v in per.values()])), cov)
    # consistency: per active column, interfaces in order and no non-positive cell
    thick = cell_thickness(g)
    col_act = g.act.any(axis=2)
    zi = np.stack([g.ifc.pts[n][:, 2].reshape(g.act.shape[:2]) for n in range(g.act.shape[2] + 1)], -1)
    ordered = (np.diff(zi, axis=-1) >= -0.01).all(-1)
    no_neg = ~(g.act & ~(thick > 0)).any(-1)
    cons = float((ordered & no_neg)[col_act].mean()) if col_act.any() else 0.0
    st.metrics = [
        Metric("thickness_mae", "Zone thickness MAE (mean of Garn, Ile, Tofte, Tilje)", "m", th_mae, 5),
        Metric("consistency", "Columns with zones in order and no negative cells", "frac", cons, 0.99, "ge"),
        Metric("presence", "Zone present/absent agreement (pinch-outs)", "frac", gated(pres_ok / max(n_pres, 1), cov), 0.90, "ge", n=n_pres),
    ]
    st.info = {"per_zone": per}
    return st


# ── 5. Facies modelling ──────────────────────────────────────────────────────
def score_facies(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("FAC")
    if "FACIES" not in g.props:
        st.overt = "no FACIES property submitted"
        return st
    env = ref.facies_envelope
    zone = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
    outside, rows = [], []
    for zn in RESERVOIR_ZONES:
        m = g.act & (zone == zn)
        f = np.round(g.props["FACIES"][m]).astype(int)
        for code in FACIES_CODES:
            key = f"facies{code}_{zn}"
            if key not in env.index:
                continue
            prop = float((f == code).mean()) if len(f) else 0.0
            lo, mid, hi = env.loc[key, ["p10", "p50", "p90"]]
            o = max(lo - prop, 0.0, prop - hi)
            outside.append(o)
            rows.append({"zone": zn, "code": code, "candidate": round(prop, 4), "p10": round(lo, 4), "p50": round(mid, 4),
                         "p90": round(hi, 4), "outside": round(o, 4)})
    st.metrics = [Metric("proportion_outside", "Facies proportion outside realization P10–P90 (mean)", "frac",
                         float(np.mean(outside)) if outside else None, 0.05, n=len(outside))]
    st.info = {"table": rows, "pending": "macro-F1 / balanced accuracy at blind wells need facies logs and a code legend (spec §11)"}
    return st


# ── 6. Petrophysical modelling ───────────────────────────────────────────────
def _cell_values(g: Grid, ijk: np.ndarray, prop: str) -> np.ndarray:
    ok = (ijk >= 0).all(1)
    out = np.full(len(ijk), np.nan)
    i, j, k = ijk[ok].T
    a = g.act[i, j, k]
    v = np.where(a, g.props[prop][i, j, k], np.nan)
    out[ok] = v
    return out


def score_petro(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("PETRO")
    miss = [k for k in ("PORO", "NTG", "PERMX") if k not in g.props]
    if miss:
        st.overt = f"missing properties {miss}"
        return st
    P = ref.blind_paths
    xyz = P[["x", "y", "z"]].values
    ijk_r, ijk_c = locate(ref.grid, xyz), locate(g, xyz)
    tr = {k: _cell_values(ref.grid, ijk_r, k) for k in ("PORO", "NTG", "PERMX")}
    cd = {k: _cell_values(g, ijk_c, k) for k in ("PORO", "NTG", "PERMX")}
    inres = np.isfinite(tr["PORO"])
    tr = {k: v[inres] for k, v in tr.items()}
    cd = {k: v[inres] for k, v in cd.items()}
    cov = float(np.isfinite(cd["PORO"]).mean()) if inres.any() else 0.0
    lk = lambda a: np.log10(np.clip(a, 0.01, None))
    dp, dn, dk = cd["PORO"] - tr["PORO"], cd["NTG"] - tr["NTG"], lk(cd["PERMX"]) - lk(tr["PERMX"])
    # zone-average porosity maps (thickness-weighted) at reference columns
    zone_r = np.broadcast_to(io.zone_of_layer()[None, None, :], ref.grid.act.shape)
    zone_c = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
    th_r, th_c = np.clip(cell_thickness(ref.grid), 0, None), np.clip(cell_thickness(g), 0, None)
    col_xy = ref.grid.ifc.pts[0][:, :2].reshape(*ref.grid.act.shape[:2], 2)
    d, idx = g.ifc.tree.query(col_xy.reshape(-1, 2))
    ci, cj = np.unravel_index(idx, g.act.shape[:2])
    ci, cj, near = ci.reshape(ref.grid.act.shape[:2]), cj.reshape(ref.grid.act.shape[:2]), (d <= 150).reshape(ref.grid.act.shape[:2])
    map_r, map_c = [], []
    for zn in RESERVOIR_ZONES:
        wr = np.where(ref.grid.act & (zone_r == zn), th_r, 0)
        mr = (wr * ref.grid.props["PORO"]).sum(-1) / np.where(wr.sum(-1) > 0, wr.sum(-1), np.nan)
        wc = np.where(g.act & (zone_c == zn), th_c, 0)
        mc = (wc * g.props["PORO"]).sum(-1) / np.where(wc.sum(-1) > 0, wc.sum(-1), np.nan)
        mc_at = np.where(near, mc[ci, cj], np.nan)
        m = np.isfinite(mr)
        map_r.append(mr[m]); map_c.append(mc_at[m])
    map_r, map_c = np.concatenate(map_r), np.concatenate(map_c)
    st.metrics = [
        Metric("poro_mae", "Porosity MAE along blind wells", "frac", gated(mae(dp), cov), 0.02, n=int(inres.sum())),
        Metric("logk_mae", "log₁₀ permeability MAE along blind wells", "log mD", gated(mae(dk), cov), 0.30),
        Metric("ntg_mae", "Net-to-gross MAE along blind wells", "frac", gated(mae(dn), cov), 0.05),
        Metric("poro_map_r2", "R² of zone-average porosity maps", "", r2(map_c, map_r), 0.50, "ge", n=int(np.isfinite(map_c).sum())),
        Metric("poro_rmse", "Porosity RMSE along blind wells", "frac", gated(rmse(dp), cov), 0.03, primary=False),
        Metric("poro_r2_wells", "Porosity R² along blind wells", "", r2(cd["PORO"], tr["PORO"]), 0.5, "ge", primary=False),
        Metric("logk_r2_wells", "log₁₀ k R² along blind wells", "", r2(lk(cd["PERMX"]), lk(tr["PERMX"])), 0.5, "ge", primary=False),
    ]
    zone = np.broadcast_to(io.zone_of_layer()[None, None, :], g.act.shape)
    st.info = {"truth": "reference-model cells along the blind-well paths (conditioned to those wells' logs)",
               "wasserstein_poro": {zn: round(float(wasserstein_distance(g.props["PORO"][g.act & (zone == zn)],
                                                                        ref.grid.props["PORO"][ref.grid.act & (zone_r == zn)])), 4)
                                    for zn in RESERVOIR_ZONES if (g.act & (zone == zn)).any()},
               "well_samples": {"x": P.x.values[inres].round(0).tolist(), "y": P.y.values[inres].round(0).tolist(),
                                "z": P.z.values[inres].round(1).tolist(), "ref": np.round(tr["PORO"], 4).tolist(),
                                "cand": [None if not np.isfinite(v) else round(float(v), 4) for v in cd["PORO"]]}}
    return st


# ── 7. Reservoir evaluation ──────────────────────────────────────────────────
def score_volumes(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("VOL")
    if any(k not in g.props for k in ("PORO", "NTG")):
        st.overt = "PORO and NTG are required to compute volumes"
        return st
    vc = volume_table(ref, g, assign_eqlnum(ref, g))
    vr = ref.volumes
    rel = lambda c, r: (c / r - 1) if r else math.nan
    m = vr.merge(vc, on=["zone", "eqlnum"], how="left", suffixes=("_ref", "_c")).fillna(0.0)
    w = m.hcpv_oil_ref
    hc_w = float((np.abs(m.hcpv_oil_c / m.hcpv_oil_ref.where(w > 0) - 1) * w).sum() / w.sum())
    pct = lambda x: 100 * x
    st.metrics = [
        Metric("pv_err", "Pore-volume error", "%", pct(rel(vc.pv.sum(), vr.pv.sum())), 5),
        Metric("hcpv_err", "HCPV (oil) error", "%", pct(rel(vc.hcpv_oil.sum(), vr.hcpv_oil.sum())), 5),
        Metric("stoiip_err", "STOIIP error", "%", pct(rel(vc.stoiip.sum(), vr.stoiip.sum())), 5),
        Metric("grv_err", "Gross rock volume error", "%", pct(rel(vc.brv.sum(), vr.brv.sum())), 5, primary=False),
        Metric("hcpv_zone_err", "HCPV error per zone × region (HCPV-weighted mean |error|)", "%", pct(hc_w), 10, primary=False),
    ]
    st.info = {"stoiip_1e6_sm3": {"candidate": round(vc.stoiip.sum() / 1e6, 2), "reference": round(vr.stoiip.sum() / 1e6, 2)},
               "table": m.assign(**{c: m[c] / 1e6 for c in m.columns if c.endswith(("_ref", "_c"))}).round(3).to_dict("records")}
    return st


# ── 8. Reservoir connectivity ────────────────────────────────────────────────
def _well_comps(g: Grid, comp: np.ndarray, paths: pd.DataFrame) -> dict[str, set]:
    ijk = locate(g, paths[["x", "y", "z"]].values)
    ok = (ijk >= 0).all(1)
    lab = np.full(len(paths), -1)
    lab[ok] = comp[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    out = {}
    for w, l in zip(paths.well.values, lab):
        if l >= 0:
            out.setdefault(w, set()).add(int(l))
    return out


def score_connectivity(ref: Reference, g: Grid) -> StageResult:
    st = StageResult("CONN")
    eq = assign_eqlnum(ref, g)
    cc = compartments(ref, g, eq)
    rc = ref.comp
    # compartment agreement on cells that are hydrocarbon-bearing in both models
    mp = ref.map_cells(g)
    both = (cc >= 0) & (mp >= 0)
    r_lab = rc.ravel()[mp[both]]
    both_idx = np.where(both)
    keep = r_lab >= 0
    ari = adjusted_rand(cc[both_idx][keep], r_lab[keep]) if keep.any() else math.nan
    hc_cov = float(keep.sum() / max((rc >= 0).sum(), 1))
    # well-pair connectivity
    P = ref.all_paths
    wr, wc = _well_comps(ref.grid, rc, P), _well_comps(g, cc, P)
    wells = sorted(set(P.well))
    agree = tot = disc_ref = 0
    for a in range(len(wells)):
        for b in range(a + 1, len(wells)):
            x, y = wells[a], wells[b]
            cr = bool(wr.get(x, set()) & wr.get(y, set()))
            cn = bool(wc.get(x, set()) & wc.get(y, set()))
            agree += cr == cn
            tot += 1
            disc_ref += not cr
    pv = g.brv * g.props.get("NTG", 1) * g.props.get("PORO", 1)
    big = lambda lab, v: int(sum(v[lab == l].sum() > 0.01 * v[lab >= 0].sum() for l in np.unique(lab[lab >= 0])))
    pvr = ref.grid.brv * ref.grid.props["NTG"] * ref.grid.props["PORO"]
    st.metrics = [
        Metric("compartment_ari", "Compartment agreement (adjusted Rand index)", "", gated(ari, hc_cov), 0.80, "ge",
               n=int(keep.sum())),
        Metric("well_pair_agreement", "Well pairs with the same connected/separated status", "frac",
               agree / tot if tot else None, 0.90, "ge", n=tot),
        Metric("hc_overlap", "Reference HC cells also HC in candidate", "frac", hc_cov, 0.90, "ge", primary=False),
    ]
    st.info = {"compartments_over_1pct_pv": {"reference": big(rc, pvr), "candidate": big(cc, pv)},
               "well_pairs": tot, "reference_separated_pairs": disc_ref,
               "rule": "static connectivity: hydrocarbon cells linked where cell faces overlap > 0.5 m (juxtaposition); fault seal not modelled"}
    return st


# ── driver ───────────────────────────────────────────────────────────────────
def score_candidate(cand_dir: Path, ref: Reference | None = None) -> dict:
    ref = ref or Reference()
    cand_dir = Path(cand_dir)
    stages: dict[str, StageResult] = {}
    t0 = time.time()
    if (cand_dir / "horizon_not2_time.csv").exists():
        stages["HZ"] = score_horizons(ref, pd.read_csv(cand_dir / "horizon_not2_time.csv"))
    if (cand_dir / "faults.csv").exists():
        stages["FLT"] = score_faults(ref, pd.read_csv(cand_dir / "faults.csv"))
    if (cand_dir / "grid.grdecl").exists():
        try:
            g = load_grid(cand_dir / "grid.grdecl", cand_dir / "props.grdecl")
            if g.cp.zcorn.shape != ref.grid.cp.zcorn.shape:
                raise ValueError(f"grid dimensions {g.act.shape} ≠ target (46, 112, 22)")
        except Exception as e:
            for s in ("STRUCT", "STRAT", "FAC", "PETRO", "VOL", "CONN"):
                stages[s] = StageResult(s, overt=f"grid.grdecl unusable: {e}")
        else:
            for s, fn in (("STRUCT", score_structure), ("STRAT", score_strat), ("FAC", score_facies),
                          ("PETRO", score_petro), ("VOL", score_volumes), ("CONN", score_connectivity)):
                stages[s] = fn(ref, g)
    w = {s: STAGES[s][1] for s in stages}
    composite = sum(w[s] * stages[s].score for s in stages) / sum(w.values()) if w else 0.0
    manifest = json.loads((cand_dir / "manifest.json").read_text()) if (cand_dir / "manifest.json").exists() else {}
    passed = sum(sum(m.passed for m in r.primary) for r in stages.values() if not r.overt)
    n_prim = sum(len(r.primary) for r in stages.values())
    return {"scorer": "v2", "candidate": str(cand_dir), "manifest": manifest, "composite": round(composite, 4),
            "checks_passed": passed, "checks_total": n_prim,
            "stages_scored": [s for s in STAGES if s in stages],
            "stage_coverage": round(sum(w.values()) / sum(v[1] for v in STAGES.values()), 3),
            "stages": {s: stages[s].as_dict() for s in STAGES if s in stages}, "seconds": round(time.time() - t0, 1)}


def summary_line(res: dict) -> str:
    parts = [f"{s} {d['passed']}/{d['n_primary']}" + ("✓" if d["outcome"] == "correct" else f" ({d['outcome']})")
             for s, d in res["stages"].items()]
    return f"{res['checks_passed']}/{res['checks_total']} checks · composite {res['composite']:.3f} | " + " · ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true", help="build perturbed reference candidates and score them")
    a = ap.parse_args()
    if a.selftest:
        from norne_benchmark import selftest
        raise SystemExit(0 if selftest.run() else 1)
    if not a.candidate:
        ap.error("--candidate DIR or --selftest required")
    res = score_candidate(a.candidate)
    out = a.out or (a.candidate / "score.json")
    out.write_text(json.dumps(res, indent=2))
    print(summary_line(res))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
