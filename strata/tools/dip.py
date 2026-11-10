"""Reflector-dip field and dip-following ("least-squares from local slopes") horizons.

1. local_slopes: inline/crossline reflector slope p, q (ms per trace) and a planarity confidence,
   from the gradient structure tensor of the seismic (plane-wave model u(x,t) = f(t − p·x) ⇒
   p = −<u_x u_t>/<u_t u_t>). Cached to disk.
2. lsq_horizon: the surface t(il, xl) whose slopes match the local reflector slopes everywhere,
   passing through anchor points (well ties), with slope equations down-weighted where the
   reflector is not planar or a fault is likely, and an optional soft pull to a level surface
   (NCS boundary). Solved as one sparse least-squares problem, re-sampling slopes at the current
   surface a few times (after Wu & Fomel 2018, Geophysics 83(4), "Least-squares horizons with
   local slopes and multigrid correlations").
3. dip_conformity: how well a picked horizon follows the reflector dip (sawtooth / reflector-
   crossing detector).

Input data only. All weights are fixed a priori below.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from strata.config import CACHE_DIR
from strata.data import Cube

TOOL_VERSION = "dip-0.1"
T_FROM, T_TO = 2250.0, 2700.0
SIG_DERIV, SIG_SMOOTH = 1.0, (4.0, 4.0, 2.0)        # gradient and tensor-smoothing sigmas (il, xl, t samples)
MAX_SLOPE = 3.0                                      # samples per trace
W_ANCHOR, W_LEVEL, W_TIKH = 100.0, 0.05, 1e-3
FAULT_DOWNWEIGHT = 0.02


def local_slopes(cube: Cube, t_from: float = T_FROM, t_to: float = T_TO):
    """(p, q, cp, cq, k0): slopes along inline-index and crossline-index axes in ms/trace, planarity
    confidences 0–1, and the first sample index of the window. Cached in data/strata_cache."""
    path = CACHE_DIR / f"slopes_{int(t_from)}_{int(t_to)}.npz"
    if path.exists():
        z = np.load(path)
        return z["p"], z["q"], z["cp"], z["cq"], int(z["k0"])
    k0, k1 = int((t_from - cube.t0) / cube.dt), int((t_to - cube.t0) / cube.dt) + 1
    D = np.asarray(cube.data[:, :, k0:k1], np.float32)
    ux = ndimage.gaussian_filter(D, SIG_DERIV, order=(1, 0, 0))
    uy = ndimage.gaussian_filter(D, SIG_DERIV, order=(0, 1, 0))
    ut = ndimage.gaussian_filter(D, SIG_DERIV, order=(0, 0, 1))
    del D
    g = lambda a: ndimage.gaussian_filter(a, SIG_SMOOTH)
    Ttt = g(ut * ut) + 1e-12
    out = []
    for u in (ux, uy):
        Tut = g(u * ut); Tuu = g(u * u)
        s = np.clip(-Tut / Ttt, -MAX_SLOPE, MAX_SLOPE)
        res = np.maximum(Tuu - Tut ** 2 / Ttt, 0)                    # energy not explained by the plane wave
        conf = np.clip(2 * Ttt / (Ttt + res) - 1, 0, 1)               # 1 = planar, 0 = isotropic noise
        out += [(s * cube.dt).astype(np.float32), conf.astype(np.float16)]
    p, cp, q, cq = out
    np.savez(path, p=p, q=q, cp=cp, cq=cq, k0=k0)
    return p, q, cp, cq, k0


def _sample(vol, k0, t, t0, dt):
    k = np.clip(np.round((t - t0) / dt).astype(int) - k0, 0, vol.shape[2] - 1)
    ii, jj = np.indices(t.shape)
    return vol[ii, jj, k].astype(np.float64)


def lsq_horizon(cube: Cube, init: np.ndarray, anchors: list[tuple[float, float, float]],
                fault: np.ndarray | None = None, level: np.ndarray | None = None,
                level_weight: np.ndarray | None = None, iters: int = 4, log=None) -> tuple[np.ndarray, dict]:
    """Dip-following surface. init: starting surface (e.g. well-guide); anchors: (il, xl, twt);
    fault: bool map where slope equations are relaxed; level/level_weight: soft target surface and
    0–1 weights (e.g. NCS boundary × contrast)."""
    p, q, cp, cq, k0 = local_slopes(cube)
    ni, nj = init.shape
    n = ni * nj
    idx = np.arange(n).reshape(ni, nj)
    fz = np.zeros((ni, nj), bool) if fault is None else ndimage.binary_dilation(fault, iterations=1)
    A = []
    for il, xl, t in anchors:
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        if 0 <= i < ni and 0 <= j < nj:
            A.append((idx[i, j], t))
    a_rows = np.array([a for a, _ in A]); a_val = np.array([v for _, v in A])
    t = np.clip(np.where(np.isfinite(init), init, np.nanmedian(init)), T_FROM, T_TO).astype(np.float64)
    t0 = time.time()
    hist = []
    for it in range(iters):
        P, Q = _sample(p, k0, t, cube.t0, cube.dt), _sample(q, k0, t, cube.t0, cube.dt)
        CP, CQ = _sample(cp, k0, t, cube.t0, cube.dt), _sample(cq, k0, t, cube.t0, cube.dt)
        # inline-direction differences: t[i+1,j] − t[i,j] = mean slope of the two traces
        wx = 0.5 * (CP[1:] + CP[:-1]) * np.where(fz[1:] | fz[:-1], FAULT_DOWNWEIGHT, 1.0)
        bx = 0.5 * (P[1:] + P[:-1])
        wy = 0.5 * (CQ[:, 1:] + CQ[:, :-1]) * np.where(fz[:, 1:] | fz[:, :-1], FAULT_DOWNWEIGHT, 1.0)
        by = 0.5 * (Q[:, 1:] + Q[:, :-1])
        rows, cols, vals, rhs, w = [], [], [], [], []
        r = 0
        for (a, b, ww, bb) in ((idx[1:], idx[:-1], wx, bx), (idx[:, 1:], idx[:, :-1], wy, by)):
            m = a.size
            rr = np.arange(r, r + m)
            rows += [rr, rr]; cols += [a.ravel(), b.ravel()]; vals += [np.ones(m), -np.ones(m)]
            rhs.append(bb.ravel()); w.append(ww.ravel()); r += m
        m = len(a_rows)
        if m:
            rr = np.arange(r, r + m); rows.append(rr); cols.append(a_rows); vals.append(np.ones(m))
            rhs.append(a_val); w.append(np.full(m, W_ANCHOR)); r += m
        if level is not None:
            lw = (W_LEVEL * (level_weight if level_weight is not None else np.ones_like(level))).ravel()
            ok = np.isfinite(level.ravel()) & (lw > 0)
            rr = np.arange(r, r + ok.sum()); rows.append(rr); cols.append(np.arange(n)[ok]); vals.append(np.ones(ok.sum()))
            rhs.append(level.ravel()[ok]); w.append(lw[ok]); r += int(ok.sum())
        rr = np.arange(r, r + n); rows.append(rr); cols.append(np.arange(n)); vals.append(np.ones(n))
        rhs.append(t.ravel()); w.append(np.full(n, W_TIKH)); r += n
        M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n))
        Wd = sparse.diags(np.concatenate(w))
        N = (M.T @ Wd @ M).tocsc()
        y = M.T @ (np.concatenate(w) * np.concatenate(rhs))
        new = spsolve(N, y).reshape(ni, nj)
        new = np.clip(new, T_FROM, T_TO)
        change = float(np.mean(np.abs(new - t)))
        t = new
        hist.append(round(change, 2))
        if log:
            log(f"lsq_horizon iter {it + 1}: mean change {change:.2f} ms, {time.time() - t0:.0f} s")
    info = {"iterations": iters, "mean_change_per_iter_ms": hist, "anchors": int(len(a_rows)),
            "seconds": round(time.time() - t0, 1),
            "weights": {"anchor": W_ANCHOR, "level": W_LEVEL, "fault_downweight": FAULT_DOWNWEIGHT}}
    return t.astype(np.float32), info


def dip_conformity(cube: Cube, twt: np.ndarray, fault: np.ndarray | None = None, tol_ms: float = 4.0) -> dict:
    """Mismatch between the picked horizon's step to the next trace and the local reflector slope,
    away from faults, where the reflector is planar (confidence ≥ 0.5). A horizon that follows its
    reflector has small misfit; sawtooth picks and reflector-crossing picks have large misfit."""
    p, q, cp, cq, k0 = local_slopes(cube)
    t = np.where(np.isfinite(twt), twt, T_FROM)
    P, Q = _sample(p, k0, t, cube.t0, cube.dt), _sample(q, k0, t, cube.t0, cube.dt)
    CP, CQ = _sample(cp, k0, t, cube.t0, cube.dt), _sample(cq, k0, t, cube.t0, cube.dt)
    fz = np.zeros(twt.shape, bool) if fault is None else ndimage.binary_dilation(fault, iterations=2)
    errs = []
    for d, s, c, f in ((np.diff(twt, axis=0), 0.5 * (P[1:] + P[:-1]), np.minimum(CP[1:], CP[:-1]), fz[1:] | fz[:-1]),
                       (np.diff(twt, axis=1), 0.5 * (Q[:, 1:] + Q[:, :-1]), np.minimum(CQ[:, 1:], CQ[:, :-1]), fz[:, 1:] | fz[:, :-1])):
        ok = np.isfinite(d) & (c >= 0.5) & ~f
        errs.append(np.abs(d[ok] - s[ok]))
    e = np.concatenate(errs)
    return {"dip_misfit_mean_ms": round(float(e.mean()), 3) if e.size else None,
            "dip_misfit_frac_over_tol": round(float((e > tol_ms).mean()), 4) if e.size else None,
            "pairs": int(e.size), "tol_ms": tol_ms}


def snap_to_event(cube: Cube, surface: np.ndarray, anchors, fault=None, polarity: str = "trough",
                  window_ms: float = 10.0, offset_window: int = 21):
    """Snap a dip-following surface to the nearest event of `polarity` with Viterbi (±window_ms).
    Where no event is found, fill with the surface shifted by the local median (snapped − surface)
    offset over offset_window traces, so the fill sits on the same event as its neighbours instead of
    stepping back to the raw surface (which avoids box-shaped excursions)."""
    from strata.tools.horizon import Horizon, viterbi_track
    hz, rep = viterbi_track(cube, surface, polarity, window_ms=window_ms, fault=fault, anchors=anchors,
                            fault_mode="skip" if fault is not None else "relax", second_window_ms=min(8.0, window_ms))
    off = np.where(np.isfinite(hz.twt), hz.twt - surface, np.nan)
    pad = offset_window // 2
    from numpy.lib.stride_tricks import sliding_window_view
    o = np.pad(off, pad, mode="constant", constant_values=np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            local = np.nanmedian(sliding_window_view(o, (offset_window, offset_window))[:, :, ::4, ::4].reshape(*off.shape, -1), axis=-1)
    local = np.where(np.isfinite(local), local, 0.0)
    twt = np.where(np.isfinite(hz.twt), hz.twt, surface + local).astype(np.float32)
    conf = np.where(np.isfinite(hz.twt), hz.conf, 0.5).astype(np.float32)
    return Horizon(twt, conf), {"snapped_frac": round(float(np.isfinite(hz.twt).mean()), 3),
                                "median_offset_ms": round(float(np.nanmedian(off)), 2)}
