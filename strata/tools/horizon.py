"""Horizon interpretation tools: seed snapping, polarity analysis, region-growing
tracking with confidence, gap filling, QC and well residuals.

All times are TWT ms. A horizon is two (n_il, n_xl) arrays: `twt` (NaN = not picked)
and `conf` (0–1, normalised cross-correlation with the trace it was tracked from).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from strata.data import Cube

TOOL_VERSION = "horizon-0.1"


@dataclass
class Horizon:
    twt: np.ndarray
    conf: np.ndarray

    def picked(self) -> np.ndarray:
        return np.isfinite(self.twt)


# ── helpers ──────────────────────────────────────────────────────────────────
def _extrema(tr: np.ndarray, sign: int) -> np.ndarray:
    s = sign * tr
    return np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1


def _refine(tr: np.ndarray, k: int) -> float:
    """Parabolic sub-sample position of an extremum at integer index k."""
    if k <= 0 or k >= len(tr) - 1:
        return float(k)
    a, b, c = tr[k - 1], tr[k], tr[k + 1]
    den = a - 2 * b + c
    return float(k + 0.5 * (a - c) / den) if den != 0 else float(k)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# ── seeds and polarity ───────────────────────────────────────────────────────
def snap_seeds(cube: Cube, ties: pd.DataFrame, search_ms: float = 16.0) -> pd.DataFrame:
    """For each tied well top: the nearest peak and nearest trough within ±search_ms of the
    tied time, with their offsets and normalised amplitudes."""
    rows = []
    t_ax = cube.t_axis()
    for r in ties[ties.in_cube & ties.twt.notna()].itertuples():
        i, j = cube.index(r.il, r.xl)
        tr = np.asarray(cube.data[i, j], float)
        k0 = (r.twt - cube.t0) / cube.dt
        if not 2 <= k0 < len(tr) - 2:
            continue
        w = int(np.ceil(search_ms / cube.dt))
        row = {"well": r.well, "il": int(cube.ilines[i]), "xl": int(cube.xlines[j]), "tie_twt": r.twt}
        for name, sign in (("peak", 1), ("trough", -1)):
            ex = _extrema(tr, sign)
            ex = ex[np.abs(ex - k0) <= w]
            if len(ex):
                k = int(ex[np.argmin(np.abs(ex - k0))])
                kk = _refine(tr, k)
                row[f"{name}_twt"] = float(np.interp(kk, np.arange(len(t_ax)), t_ax))
                row[f"{name}_offset_ms"] = row[f"{name}_twt"] - r.twt
                row[f"{name}_amp"] = float(sign * tr[k] / cube.rms)
            else:
                row[f"{name}_twt"] = row[f"{name}_offset_ms"] = row[f"{name}_amp"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def polarity_report(snapped: pd.DataFrame) -> dict:
    """Which extremum type sits consistently on the tied top? Smaller, more consistent
    offsets and stronger amplitude favour that polarity."""
    out = {}
    for p in ("peak", "trough"):
        off = snapped[f"{p}_offset_ms"].dropna()
        amp = snapped[f"{p}_amp"].dropna()
        out[p] = {"wells": int(len(off)), "median_abs_offset_ms": round(float(off.abs().median()), 2) if len(off) else None,
                  "offset_std_ms": round(float(off.std()), 2) if len(off) > 1 else None,
                  "median_amp_rms": round(float(amp.median()), 2) if len(amp) else None}
    # offset consistency dominates; amplitude only breaks near-ties (a strong reflector that
    # sits consistently away from the tied tops is the wrong event)
    score = {p: (out[p]["median_abs_offset_ms"] or 99) + 0.5 * (out[p]["offset_std_ms"] or 99) - 0.5 * (out[p]["median_amp_rms"] or 0)
             for p in ("peak", "trough")}
    out["suggested"] = min(score, key=score.get)
    out["bulk_shift_ms"] = round(float(snapped[f"{out['suggested']}_offset_ms"].median()), 2)
    return out


# ── tracking ─────────────────────────────────────────────────────────────────
def track(cube: Cube, seeds: list[tuple[int, int, float]], polarity: str = "peak", search_ms: float = 8.0,
          window_ms: float = 24.0, min_corr: float = 0.75, max_step_ms: float = 8.0,
          barrier: np.ndarray | None = None, max_traces: int | None = None,
          guide: np.ndarray | None = None, guide_window_ms: float = 30.0) -> tuple[Horizon, dict]:
    """Region-growing tracker from seeds (il, xl, twt). Each step moves to a 4-neighbour,
    searches ±search_ms around the current time for an extremum of the chosen polarity,
    accepts the best normalised cross-correlation (≥ min_corr) of a ±window_ms waveform
    within max_step_ms. Traces with higher correlation are expanded first.
    `barrier` (n_il, n_xl) bool marks traces the tracker may not enter (e.g. faults).
    `guide` (n_il, n_xl) TWT surface (e.g. gridded well ties): candidates further than
    guide_window_ms from it are rejected — stops leaking onto other reflectors across faults."""
    sign = 1 if polarity == "peak" else -1
    D = cube.data
    ni, nj, nt = D.shape
    twt = np.full((ni, nj), np.nan, np.float32)
    conf = np.zeros((ni, nj), np.float32)
    hw = max(2, int(round(window_ms / 2 / cube.dt)))
    sw = max(1, int(round(search_ms / cube.dt)))
    ms = max_step_ms / cube.dt
    heap: list = []
    for il, xl, t in seeds:
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        if not (0 <= i < ni and 0 <= j < nj):
            continue
        tr = np.asarray(D[i, j], float)
        k0 = (t - cube.t0) / cube.dt
        ex = _extrema(tr, sign)
        ex = ex[np.abs(ex - k0) <= sw]
        if not len(ex):
            continue
        k = _refine(tr, int(ex[np.argmin(np.abs(ex - k0))]))
        twt[i, j] = cube.t0 + k * cube.dt
        conf[i, j] = 1.0
        heapq.heappush(heap, (-1.0, i, j))
    n = 0
    while heap:
        _, i, j = heapq.heappop(heap)
        n += 1
        if max_traces and n > max_traces:
            break
        k = (twt[i, j] - cube.t0) / cube.dt
        ki = int(round(k))
        if ki - hw < 0 or ki + hw + 1 > nt:
            continue
        ref = np.asarray(D[i, j, ki - hw:ki + hw + 1], float)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < ni and 0 <= b < nj) or np.isfinite(twt[a, b]):
                continue
            if barrier is not None and barrier[a, b]:
                continue
            lo, hi = max(hw, ki - sw - 1), min(nt - hw - 1, ki + sw + 2)
            if hi <= lo:
                continue
            tr = np.asarray(D[a, b, lo - 1:hi + 1], float)
            ex = _extrema(tr, sign) + lo - 1
            best, best_c = None, -1.0
            for e in ex:
                if abs(e - k) > ms or e - hw < 0 or e + hw + 1 > nt:
                    continue
                if guide is not None and np.isfinite(guide[a, b]) and abs(cube.t0 + e * cube.dt - guide[a, b]) > guide_window_ms:
                    continue
                c = _ncc(ref, np.asarray(D[a, b, e - hw:e + hw + 1], float))
                if c > best_c + 1e-6 or (abs(c - best_c) <= 1e-6 and best is not None and abs(e - k) < abs(best - k)):
                    best, best_c = e, c
            if best is None or best_c < min_corr:
                continue
            full = np.asarray(D[a, b], float)
            twt[a, b] = cube.t0 + _refine(full, int(best)) * cube.dt
            conf[a, b] = best_c
            heapq.heappush(heap, (-best_c, a, b))
    hz = Horizon(twt, conf)
    params = dict(polarity=polarity, search_ms=search_ms, window_ms=window_ms, min_corr=min_corr,
                  max_step_ms=max_step_ms, n_seeds=len(seeds), guided=guide is not None,
                  guide_window_ms=guide_window_ms if guide is not None else None, barrier=barrier is not None)
    return hz, {"params": params, **qc(hz)}


def fill_gaps(hz: Horizon, max_gap_traces: int = 4, smooth: int = 0) -> tuple[Horizon, dict]:
    """Fill holes no wider than max_gap_traces by iterative neighbour averaging (filled
    traces get confidence 0); optional median smoothing of size `smooth`."""
    twt = hz.twt.astype(float).copy()
    conf = hz.conf.copy()
    filled = 0
    for _ in range(max_gap_traces):
        m = np.isfinite(twt)
        if m.all():
            break
        s = ndimage.uniform_filter(np.where(m, twt, 0.0), 3, mode="nearest")
        c = ndimage.uniform_filter(m.astype(float), 3, mode="nearest")
        new = (~m) & (c > 0.34)
        twt[new] = s[new] / c[new]
        filled += int(new.sum())
    if smooth and smooth > 1:
        m = np.isfinite(twt)
        med = ndimage.median_filter(np.where(m, twt, np.nanmean(twt)), size=smooth)
        twt = np.where(m, med, np.nan)
    out = Horizon(twt.astype(np.float32), conf)
    return out, {"filled_traces": filled, "smooth": smooth, **qc(out)}


# ── QC ───────────────────────────────────────────────────────────────────────
def qc(hz: Horizon, jump_ms: float = 12.0) -> dict:
    """Coverage, confidence, and 'jumps' — neighbouring picks differing by more than
    jump_ms, a symptom of cycle skips or of tracking across an unrecognised fault."""
    m = hz.picked()
    dI = np.abs(np.diff(hz.twt, axis=0))
    dJ = np.abs(np.diff(hz.twt, axis=1))
    jumps = int(np.nansum(dI > jump_ms) + np.nansum(dJ > jump_ms))
    c = hz.conf[m & (hz.conf > 0)]
    return {"coverage": round(float(m.mean()), 4), "picked_traces": int(m.sum()),
            "twt_range_ms": [round(float(np.nanmin(hz.twt)), 1), round(float(np.nanmax(hz.twt)), 1)] if m.any() else None,
            "conf_median": round(float(np.median(c)), 3) if len(c) else None,
            "conf_p10": round(float(np.percentile(c, 10)), 3) if len(c) else None,
            "jumps": jumps, "jump_threshold_ms": jump_ms}


def well_residuals(cube: Cube, hz: Horizon, ties: pd.DataFrame) -> pd.DataFrame:
    """Horizon time at each tied well minus the tied top time (ms)."""
    rows = []
    for r in ties[ties.in_cube & ties.twt.notna()].itertuples():
        i, j = cube.index(r.il, r.xl)
        t = float(hz.twt[i, j]) if np.isfinite(hz.twt[i, j]) else np.nan
        rows.append({"well": r.well, "il": int(cube.ilines[i]), "xl": int(cube.xlines[j]), "tie_twt": round(r.twt, 2),
                     "horizon_twt": round(t, 2) if np.isfinite(t) else None,
                     "residual_ms": round(t - r.twt, 2) if np.isfinite(t) else None,
                     "conf": round(float(hz.conf[i, j]), 3), "td_source": r.td_source,
                     "table_spread_ms": round(r.table_spread_ms, 1)})
    return pd.DataFrame(rows)


def to_table(cube: Cube, hz: Horizon) -> pd.DataFrame:
    i, j = np.where(hz.picked())
    return pd.DataFrame({"il": cube.ilines[i], "xl": cube.xlines[j], "twt": np.round(hz.twt[i, j], 2),
                         "conf": np.round(hz.conf[i, j], 3)})


def save(hz: Horizon, path) -> None:
    np.savez_compressed(path, twt=hz.twt, conf=hz.conf)


def load(path) -> Horizon:
    z = np.load(path)
    return Horizon(z["twt"], z["conf"])


def guide_surface(cube: Cube, points: list[tuple[float, float, float]], smooth_traces: int = 15) -> np.ndarray:
    """A smooth TWT trend surface through (il, xl, twt) well points: linear interpolation
    inside the wells' hull, nearest-value outside, then Gaussian smoothing."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    p = np.array([(il - cube.ilines[0], xl - cube.xlines[0]) for il, xl, _ in points], float)
    v = np.array([t for *_, t in points], float)
    gi, gj = np.mgrid[0:cube.shape[0], 0:cube.shape[1]]
    q = np.c_[gi.ravel(), gj.ravel()]
    g = LinearNDInterpolator(p, v)(q) if len(p) >= 3 else np.full(len(q), np.nan)
    nn = NearestNDInterpolator(p, v)(q)
    g = np.where(np.isfinite(g), g, nn).reshape(cube.shape[:2])
    return ndimage.gaussian_filter(g, smooth_traces) if smooth_traces else g


# ── global tracking: Viterbi / dynamic programming along lines ───────────────
def _reliable_throw(D: np.ndarray, t0: float, dt: float, a: int, b: int, t: float, max_throw_ms: float,
                    win_ms: float = 80.0, tol_ms: float = 8.0, min_ncc: float = 0.6) -> float | None:
    """Seismic time offset from trace a to trace b (rows of a line array) at time t, kept only if
    forward, reverse and 1.5× window estimates agree within tol_ms."""
    L = int(max_throw_ms / dt)

    def est(i, j, tt, w):
        hw = int(w / 2 / dt); k = int(round((tt - t0) / dt))
        if k - hw - L < 0 or k + hw + L >= D.shape[1]:
            return None
        x = D[i, k - hw:k + hw + 1].astype(float); x -= x.mean()
        best = (-1.0, 0.0)
        for lag in range(-L, L + 1):
            y = D[j, k - hw + lag:k + hw + 1 + lag].astype(float); y = y - y.mean()
            c = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))
            if c > best[0]:
                best = (c, lag * dt)
        return best
    f = est(a, b, t, win_ms)
    if not f or f[0] < min_ncc:
        return None
    r = est(b, a, t + f[1], win_ms)
    g = est(a, b, t, win_ms * 1.5)
    if not (r and g) or abs(f[1] + r[1]) > tol_ms or abs(f[1] - g[1]) > tol_ms:
        return None
    return f[1]


def _line_viterbi(D: np.ndarray, t0: float, dt: float, guide: np.ndarray, sign: int, window_ms: float, hw: int,
                  w_jump: float, w_wave: float, w_guide: float, fault: np.ndarray | None,
                  anchors: dict[int, float], w_anchor: float, fault_mode: str = "relax",
                  max_throw_ms: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Optimal event path along one line (n_tr, nt) through candidate extrema."""
    n, nt = D.shape
    cands, waves = [], []
    fz = np.zeros(n, bool) if fault is None else np.asarray(fault, bool)
    near_fault = fz | np.r_[fz[1:], False] | np.r_[False, fz[:-1]] if fault_mode == "skip" else np.zeros(n, bool)
    if fault_mode == "skip" and fault is not None:
        near_fault = ndimage.binary_dilation(fz, iterations=2) & ~fz
    for k in range(n):
        g = guide[k]
        if not np.isfinite(g):
            cands.append(np.array([])); waves.append(None); continue
        wk = window_ms + (max_throw_ms if near_fault[k] else 0.0)
        lo, hi = int(max(hw + 1, (g - wk - t0) / dt)), int(min(nt - hw - 2, (g + wk - t0) / dt))
        tr = D[k]
        ex = np.array([e for e in _extrema(tr[lo - 1:hi + 2], sign) + lo - 1 if hw <= e < nt - hw - 1], int)
        if not len(ex):
            cands.append(np.array([])); waves.append(None); continue
        tt = np.array([t0 + _refine(tr, int(e)) * dt for e in ex])
        W = np.stack([tr[e - hw:e + hw + 1] for e in ex]).astype(float)
        W = W - W.mean(1, keepdims=True)
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
        cands.append(tt); waves.append(W)
    twt = np.full(n, np.nan, np.float32)
    conf = np.zeros(n, np.float32)
    # Nodes = traces with candidates outside fault zones. Consecutive nodes are linked; a gap made by a
    # fault zone (≤ max_fault_gap traces) is crossed directly: the waveforms on both sides are compared
    # and the time-jump penalty is nearly removed, so the path can follow the throw. Fault-zone traces
    # stay unpicked. Larger gaps start a new, independent path.
    infault = np.zeros(n, bool) if fault is None else np.asarray(fault, bool)
    skip = fault_mode == "skip"
    nodes = [k for k in range(n) if len(cands[k]) and not (skip and infault[k])]
    max_fault_gap = 10
    segs, cur = [], []
    for k in nodes:
        if cur and k - cur[-1] > max_fault_gap + 1:
            segs.append(cur); cur = []
        cur.append(k)
    if cur:
        segs.append(cur)
    for seg in segs:
        cost, back = [None] * len(seg), [None] * len(seg)
        for q, idx in enumerate(seg):
            u = w_guide * ((cands[idx] - guide[idx]) / window_ms) ** 2
            if idx in anchors:
                u = u + w_anchor * ((cands[idx] - anchors[idx]) / 8.0) ** 2
            if q == 0:
                cost[q] = u; continue
            prev = seg[q - 1]
            gap = idx - prev
            crosses_fault = (gap > 1 and infault[prev + 1:idx].any()) if skip else bool(infault[idx] or infault[prev])
            dtj = np.abs(cands[idx][:, None] - cands[prev][None, :])
            ncc = waves[idx] @ waves[prev].T
            if crosses_fault and skip:
                # reliable seismic throw estimate across the zone (forward/reverse/longer window agree)
                est = _reliable_throw(D, t0, dt, prev, idx, float(np.nanmedian(cands[prev])), max_throw_ms)
                if est is not None:
                    dth = (cands[idx][:, None] - cands[prev][None, :]) - est
                    trans = w_jump * (dth / 8.0) ** 2 + w_wave * (1 - ncc)
                else:
                    trans = w_jump * 0.02 * (dtj / 8.0) ** 2 + 1.5 * w_wave * (1 - ncc)
            else:
                jump_w = w_jump * 0.15 if crosses_fault else w_jump / gap
                trans = jump_w * (dtj / 8.0) ** 2 + w_wave * (1 - ncc)
            tot = trans + cost[q - 1][None, :]
            back[q] = np.argmin(tot, axis=1)
            cost[q] = u + tot[np.arange(len(cands[idx])), back[q]]
        j = int(np.argmin(cost[-1]))
        for q in range(len(seg) - 1, -1, -1):
            idx = seg[q]
            twt[idx] = cands[idx][j]
            if q > 0:
                jp = back[q][j]
                conf[idx] = float(waves[idx][j] @ waves[seg[q - 1]][jp])
                j = jp
            else:
                conf[idx] = 1.0
    return twt, np.clip(conf, 0, 1)


def viterbi_track(cube: Cube, guide: np.ndarray, polarity: str = "trough", window_ms: float = 25.0,
                  waveform_ms: float = 24.0, w_jump: float = 1.0, w_wave: float = 2.0, w_guide: float = 0.5,
                  fault: np.ndarray | None = None, anchors: list[tuple[float, float, float]] | None = None,
                  w_anchor: float = 5.0, second_window_ms: float = 12.0, fault_mode: str = "relax") -> tuple[Horizon, dict]:
    """Well-constrained global tracking. Pass 1: Viterbi along every inline through candidate
    events within ±window_ms of `guide`. Pass 2: Viterbi along every crossline using the
    smoothed pass-1 result as guide (±second_window_ms) for consistency between lines.
    `fault` (bool map) relaxes the time-jump penalty so paths can follow throws;
    `anchors` (il, xl, twt) pull the path through tied well points."""
    sign = 1 if polarity == "peak" else -1
    D = cube.data
    ni, nj, _ = D.shape
    hw = max(2, int(round(waveform_ms / 2 / cube.dt)))
    A = {}
    for il, xl, t in anchors or []:
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        if 0 <= i < ni and 0 <= j < nj:
            A[(i, j)] = t
    p1 = np.full((ni, nj), np.nan, np.float32)
    for i in range(ni):
        anc = {j: t for (a, j), t in A.items() if abs(a - i) <= 1}
        p1[i], _ = _line_viterbi(np.asarray(D[i], float), cube.t0, cube.dt, guide[i], sign, window_ms, hw,
                                 w_jump, w_wave, w_guide, None if fault is None else fault[i], anc, w_anchor, fault_mode)
    g2 = ndimage.median_filter(np.where(np.isfinite(p1), p1, guide), size=5)
    twt = np.full((ni, nj), np.nan, np.float32)
    conf = np.zeros((ni, nj), np.float32)
    for j in range(nj):
        anc = {i: t for (i, b), t in A.items() if abs(b - j) <= 1}
        twt[:, j], conf[:, j] = _line_viterbi(np.asarray(D[:, j], float), cube.t0, cube.dt, g2[:, j], sign,
                                              second_window_ms, hw, w_jump, w_wave, w_guide,
                                              None if fault is None else fault[:, j], anc, w_anchor, fault_mode)
    hz = Horizon(twt, conf)
    params = dict(method="viterbi", polarity=polarity, window_ms=window_ms, waveform_ms=waveform_ms, w_jump=w_jump,
                  w_wave=w_wave, w_guide=w_guide, fault_aware=fault is not None, fault_mode=fault_mode, anchors=len(A),
                  second_window_ms=second_window_ms)
    return hz, {"params": params, **qc(hz)}
