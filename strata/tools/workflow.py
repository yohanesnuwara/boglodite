"""Horizon workflow steps as plain functions, so a workflow can be REPLAYED with a subset of wells.

Used by the horizon agent's tools (one implementation for both) and by
  - holdout_wells: k-fold, spatially grouped well hold-out — re-runs a candidate's whole chain of
    steps with some wells hidden and measures the error at the hidden wells after every step
    (an input-only accuracy estimate: the closest thing to a truth score the agents can have);
  - method_agreement: where independent estimates (Viterbi tracking, dip-following + NCS surface,
    NCS package boundary) agree with each other and with the working horizon.

A well = (name, il, xl, twt) with twt already including any bulk shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from strata.data import Cube
from strata.tools import horizon as H


@dataclass
class Env:
    cube: Cube
    polarity: str = "trough"
    fault: np.ndarray | None = None
    cache: dict = field(default_factory=dict)

    def feats(self):
        if "feats" not in self.cache:
            from strata.tools import foundation as FM
            self.cache["feats"] = FM.extract_features(self.cube)
        return self.cache["feats"]


def pts(wells) -> list[tuple[float, float, float]]:
    return [(il, xl, t) for _, il, xl, t in wells]


# ── steps ────────────────────────────────────────────────────────────────────
def step_track(env: Env, wells, method: str = "viterbi", guide_window_ms: float = 30.0, use_fault_map: bool = True,
               search_ms: float = 8.0, window_ms: float = 24.0, min_corr: float = 0.75, max_step_ms: float = 8.0, **_):
    seeds = pts(wells)
    guide = H.guide_surface(env.cube, seeds)
    fm = env.fault if use_fault_map else None
    if method == "viterbi":
        return H.viterbi_track(env.cube, guide, env.polarity, window_ms=guide_window_ms or 30.0, fault=fm, anchors=seeds,
                               fault_mode="skip" if fm is not None else "relax")
    return H.track(env.cube, seeds, env.polarity, search_ms, window_ms, min_corr, max_step_ms,
                   guide=guide if guide_window_ms > 0 else None, guide_window_ms=guide_window_ms)


def step_repair(env: Env, wells, hz, window_ms: float = 35.0, **_):
    from strata.tools import repair as RP
    return RP.retrack_with_seeds(env.cube, hz, pts(wells), env.polarity, fault=env.fault, window_ms=window_ms)


def ncs_level(env: Env, wells) -> dict:
    """NCS probe trained on these wells → boundary surface, contrast grid, coords (cached per well set)."""
    key = ("ncs", tuple(sorted(w[0] for w in wells)))
    if key not in env.cache:
        from strata.tools import foundation as FM
        feats, co = env.feats()
        guide = H.guide_surface(env.cube, pts(wells))
        _, _, surface, info = FM.probability_and_surface(feats, co, env.cube, list(wells), guide)
        env.cache[key] = {"surface": surface, "contrast": info["contrast_grid"], "coords": co,
                          "train_accuracy": info["train_accuracy"], "median_contrast": info["median_contrast"]}
    return env.cache[key]


def contrast_full(env: Env, m: dict) -> np.ndarray:
    co, (ni, nj) = m["coords"], env.cube.shape[:2]
    ia = np.clip(np.searchsorted(co["il_sample"], np.arange(ni)), 0, len(co["il_sample"]) - 1)
    ja = np.clip(np.searchsorted(co["xl_sample"], np.arange(nj)), 0, len(co["xl_sample"]) - 1)
    return m["contrast"][np.ix_(ia, ja)]


def step_ncs(env: Env, wells, hz, disagree_ms: float = 40.0, min_contrast: float = 0.5, window_ms: float = 25.0, **_):
    from strata.tools import foundation as FM
    m = ncs_level(env, wells)
    new, info, flag = FM.level_check_retrack(env.cube, hz, m["surface"], m["contrast"], m["coords"], pts(wells),
                                             fault=env.fault, polarity=env.polarity, disagree_ms=disagree_ms,
                                             min_contrast=min_contrast, window_ms=window_ms)
    info["_flag"] = flag
    return new, info


def dip_surface(env: Env, wells, use_ncs_level: bool = True):
    from strata.tools import dip as DP
    level = lw = None
    if use_ncs_level:
        m = ncs_level(env, wells)
        level, lw = m["surface"], np.clip((contrast_full(env, m) - 0.5) / 0.5, 0, 1)
    return DP.lsq_horizon(env.cube, H.guide_surface(env.cube, pts(wells)), pts(wells), fault=env.fault,
                          level=level, level_weight=lw)


def step_dip(env: Env, wells, use_ncs_level: bool = True, snap_window_ms: float = 10.0, **_):
    from strata.tools import dip as DP
    surf, info = dip_surface(env, wells, use_ncs_level)
    snap, _ = H.viterbi_track(env.cube, surf, env.polarity, window_ms=snap_window_ms, fault=env.fault, anchors=pts(wells),
                              fault_mode="skip" if env.fault is not None else "relax", second_window_ms=min(8.0, snap_window_ms))
    hz = H.Horizon(np.where(np.isfinite(snap.twt), snap.twt, surf).astype(np.float32),
                   np.where(np.isfinite(snap.twt), snap.conf, 0.5).astype(np.float32))
    info["snapped_frac_of_cube"] = round(float(np.isfinite(snap.twt).mean()), 3)
    info["dip_conformity"] = DP.dip_conformity(env.cube, hz.twt, env.fault)
    return hz, info


def step_fill(env: Env, wells, hz, max_gap_traces: int = 4, smooth: int = 0, **_):
    return H.fill_gaps(hz, max_gap_traces, smooth)


STEPS = {"track": step_track, "repair": step_repair, "ncs": step_ncs, "dip": step_dip, "fill": step_fill}
STARTERS = {"track", "dip"}                                  # steps that build a horizon from scratch


def run_chain(env: Env, wells, chain: list[tuple[str, dict]], after_step=None):
    hz = None
    for k, (name, params) in enumerate(chain):
        fn = STEPS[name]
        hz, _ = fn(env, wells, **params) if name in STARTERS else fn(env, wells, hz, **params)
        if after_step:
            after_step(k, name, hz)
    return hz


# ── input-only validation ─────────────────────────────────────────────────────
def well_groups(wells, radius_traces: float = 15.0) -> list[list[int]]:
    """Spatial clusters of wells (sidetracks from one slot are near-duplicates and must be held out together)."""
    xy = np.array([(w[1], w[2]) for w in wells], float)
    n = len(wells)
    lab = -np.ones(n, int); g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = g
        while stack:
            a = stack.pop()
            for b in np.where((lab < 0) & (np.hypot(*(xy - xy[a]).T) <= radius_traces))[0]:
                lab[b] = g; stack.append(b)
        g += 1
    return [list(np.where(lab == k)[0]) for k in range(g)]


def _at(cube: Cube, hz, il, xl) -> float:
    i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
    return float(hz.twt[i, j]) if hz is not None and 0 <= i < hz.twt.shape[0] and 0 <= j < hz.twt.shape[1] else np.nan


def holdout_wells(env: Env, wells, chain: list[tuple[str, dict]], folds: int = 3, log=None) -> dict:
    """Spatially grouped k-fold hold-out: for each fold re-run `chain` without its wells and measure
    |pick − tie| at the hidden wells after every step; also the error of plain interpolation between
    the remaining wells. Returns per-step MAE over all hidden wells."""
    groups = sorted(well_groups(wells), key=lambda g: (np.mean([wells[i][1] for i in g]), np.mean([wells[i][2] for i in g])))
    fold_of = {}
    for k, g in enumerate(groups):
        for i in g:
            fold_of[i] = k % folds
    errs = {k: [] for k in range(len(chain))}
    interp = []
    per_well = []
    for f in range(folds):
        train = [w for i, w in enumerate(wells) if fold_of[i] != f]
        test = [w for i, w in enumerate(wells) if fold_of[i] == f]
        if not test or len(train) < 3:
            continue
        g = H.guide_surface(env.cube, pts(train))
        for w in test:
            i, j = int(round(w[1] - env.cube.ilines[0])), int(round(w[2] - env.cube.xlines[0]))
            interp.append(abs(g[i, j] - w[3]))

        def after(k, name, hz):
            for w in test:
                errs[k].append(abs(_at(env.cube, hz, w[1], w[2]) - w[3]))
        hz = run_chain(env, train, chain, after)
        per_well += [{"well": w[0], "fold": f, "err_ms": round(_at(env.cube, hz, w[1], w[2]) - w[3], 1)} for w in test]
        if log:
            log(f"holdout fold {f + 1}/{folds}: {len(test)} hidden wells")

    def mae(v):
        v = np.array(v, float); v = v[np.isfinite(v)]
        return (round(float(v.mean()), 2), round(float(np.median(v)), 2), int(len(v))) if len(v) else (None, None, 0)
    steps = [{"step": f"{k + 1}_{name}", "mae_ms": mae(errs[k])[0], "median_ms": mae(errs[k])[1], "n": mae(errs[k])[2]}
             for k, (name, _) in enumerate(chain)]
    return {"folds": folds, "well_groups": len(groups), "hidden_wells": len(per_well),
            "interpolation_between_training_wells": {"mae_ms": mae(interp)[0], "median_ms": mae(interp)[1]},
            "after_each_step": steps, "per_well": per_well,
            "note": "error at wells NOT used by the workflow; only tests near wells, not far from them"}


def method_agreement(env: Env, wells, hz, tol_ms: float = 15.0, region_ms: float = 25.0, min_area: int = 300,
                     track=None, dip=None) -> tuple[dict, dict]:
    """Three independent estimates — Viterbi tracking, dip-following + NCS surface, NCS package boundary —
    compared with each other and with the working horizon. Returns (report, maps)."""
    if track is None:
        track, _ = step_track(env, wells, guide_window_ms=35.0)
    if dip is None:
        dip, _ = dip_surface(env, wells, True)
    ncs = ncs_level(env, wells)["surface"]
    E = np.stack([track.twt, dip, ncs]).astype(float)
    spread = np.nanmax(E, 0) - np.nanmin(E, 0)
    t = hz.twt
    ok = np.isfinite(t)
    close = (np.abs(E - t[None]) <= tol_ms).sum(0)
    regions = []
    bad = ndimage.binary_opening(np.isfinite(spread) & (spread > region_ms), iterations=2)
    lab, n = ndimage.label(bad)
    if n:
        sizes = ndimage.sum(bad, lab, range(1, n + 1))
        for k in np.argsort(sizes)[::-1][:8]:
            if sizes[k] < min_area:
                break
            ii, jj = np.nonzero(lab == k + 1)
            regions.append({"il": int(env.cube.ilines[int(np.median(ii))]), "xl": int(env.cube.xlines[int(np.median(jj))]),
                            "traces": int(sizes[k]),
                            "il_range": [int(env.cube.ilines[ii.min()]), int(env.cube.ilines[ii.max()])],
                            "xl_range": [int(env.cube.xlines[jj.min()]), int(env.cube.xlines[jj.max()])],
                            "median_ms": {"track": round(float(np.nanmedian(E[0][ii, jj])), 1),
                                          "dip_ncs": round(float(np.nanmedian(E[1][ii, jj])), 1),
                                          "ncs_boundary": round(float(np.nanmedian(E[2][ii, jj])), 1),
                                          "working": round(float(np.nanmedian(t[ii, jj])), 1) if np.isfinite(t[ii, jj]).any() else None}})
    rep = {"tol_ms": tol_ms,
           "methods_agree_frac": round(float(np.mean(spread[ok] <= tol_ms)), 3),
           "working_backed_by_2plus_methods_frac": round(float(np.mean(close[ok] >= 2)), 3),
           "working_backed_by_none_frac": round(float(np.mean(close[ok] == 0)), 3),
           "largest_disagreement_regions": regions}
    return rep, {"track": track.twt, "dip": dip, "ncs": ncs, "spread": spread, "close": close}
