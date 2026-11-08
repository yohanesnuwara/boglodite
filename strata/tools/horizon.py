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
          barrier: np.ndarray | None = None, max_traces: int | None = None) -> tuple[Horizon, dict]:
    """Region-growing tracker from seeds (il, xl, twt). Each step moves to a 4-neighbour,
    searches ±search_ms around the current time for an extremum of the chosen polarity,
    accepts the best normalised cross-correlation (≥ min_corr) of a ±window_ms waveform
    within max_step_ms. Traces with higher correlation are expanded first.
    `barrier` (n_il, n_xl) bool marks traces the tracker may not enter (e.g. faults)."""
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
                  max_step_ms=max_step_ms, n_seeds=len(seeds))
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
