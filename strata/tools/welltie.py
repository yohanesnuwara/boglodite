"""Synthetic-seismogram well ties: polarity convention, bulk time shift and the event
(peak or trough) that a formation top falls on.

At each well with sonic (DT) and density (RHOB): acoustic impedance → reflectivity in TWT
(through the well's time–depth relation) → convolution with a zero-phase Ricker wavelet →
normalised cross-correlation with the seismic trace at the well, for both signs and for
lags within ±max_lag_ms. With a zero-phase wavelet an impedance increase gives a positive
synthetic peak, so:

    sign = +1  → the data follow SEG normal polarity (impedance increase = peak)
    sign = −1  → reversed / European polarity (impedance increase = trough)
    lag (ms)   → seismic event time = log time + lag (add to the time–depth tie)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strata.data import Cube, Wells

TOOL_VERSION = "welltie-0.1"


def conv_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Convolution centred on x, always len(x) long (np.convolve 'same' returns the longer length)."""
    full = np.convolve(x, w)
    k = (len(w) - 1) // 2
    return full[k:k + len(x)]


def ricker(f_hz: float, dt_ms: float = 1.0, length_ms: float = 128.0) -> np.ndarray:
    t = np.arange(-length_ms / 2, length_ms / 2 + dt_ms, dt_ms) / 1000.0
    a = (np.pi * f_hz * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _td_curve(wells: Wells, key: str, x: float, y: float, exclude: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, str] | None:
    """TVDSS→TWT relation for a well: its own preferred table, else the nearest well's."""
    from strata.data import TD_PREFERENCE
    rank = {s: i for i, s in enumerate(TD_PREFERENCE)}
    own = [t for t in wells.td_tvdss if t["well"] == key and t["source"] not in exclude and "icker" not in t["source"]]
    if own:
        t = min(own, key=lambda t: rank.get(t["source"], 99))
        return t["tvdss"], t["twt"], f"own:{t['source']}"
    cand = [t for t in wells.td_tvdss if t["source"] not in exclude and "icker" not in t["source"]]
    if not cand:
        return None
    t = min(cand, key=lambda t: (np.hypot(t["x"] - x, t["y"] - y), rank.get(t["source"], 99)))
    return t["tvdss"], t["twt"], f"neighbour:{t['well']}:{t['source']}"


def synthetic_tie(cube: Cube, wells: Wells, logs: dict[str, pd.DataFrame], ties: pd.DataFrame,
                  freq_hz: float = 25.0, max_lag_ms: float = 32.0, half_window_ms: float = 120.0,
                  exclude_tables: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per well with DT + RHOB and a tied top: correlation, polarity sign, lag and the event
    the top falls on in the data's polarity."""
    w1 = ricker(freq_hz)
    rows = []
    t_ax = cube.t_axis()
    for r in ties[ties.in_cube & ties.twt.notna()].drop_duplicates("well").itertuples():
        lg = logs.get(r.well)
        if lg is None or not {"DT", "RHOB", "TVDSS"} <= set(lg.columns):
            continue
        lg = lg.dropna(subset=["DT", "RHOB", "TVDSS"]).copy()
        lg["TVDSS"] = lg["TVDSS"].abs()
        lg = lg[(lg.DT > 30) & (lg.DT < 250) & (lg.RHOB > 1.5) & (lg.RHOB < 3.2)].sort_values("TVDSS")
        if len(lg) < 100:
            continue
        td = _td_curve(wells, r.well, r.x, r.y, exclude_tables)
        if td is None:
            continue
        z_td, t_td, td_src = td
        twt_log = np.interp(lg.TVDSS.values, z_td, t_td)
        ai = (304800.0 / lg.DT.values) * lg.RHOB.values
        order = np.argsort(twt_log)
        twt_log, ai = twt_log[order], ai[order]
        keep = np.r_[True, np.diff(twt_log) > 0]
        twt_log, ai = twt_log[keep], ai[keep]
        t1 = np.arange(np.ceil(twt_log[0]), np.floor(twt_log[-1]), 1.0)
        if len(t1) < 60:
            continue
        ai1 = np.interp(t1, twt_log, ai)
        rc = np.r_[0.0, np.diff(ai1) / (ai1[1:] + ai1[:-1])]
        syn = conv_same(rc, w1)
        i, j = cube.index(r.il, r.xl)
        tr = np.asarray(cube.data[i, j], float)
        seis1 = np.interp(t1, t_ax, tr)
        win = np.abs(t1 - r.twt) <= half_window_ms
        best = (0.0, 0, 1)
        L = int(max_lag_ms)
        for lag in range(-L, L + 1):
            idx = np.arange(len(t1)) - lag                    # seismic(t) vs synthetic(t − lag)
            m = win & (idx >= 0) & (idx < len(t1))
            if m.sum() < 40:
                continue
            a, b = seis1[m] - seis1[m].mean(), syn[idx[m]] - syn[idx[m]].mean()
            c = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
            if abs(c) > abs(best[0]):
                best = (c, lag, 1 if c > 0 else -1)
        corr, lag, sign = best
        # event at the top: synthetic around the (log-time) top, in the data's polarity
        k = int(np.argmin(np.abs(t1 - r.twt)))
        seg = sign * syn[max(0, k - 6):k + 7]
        ev = "peak" if seg[np.argmax(np.abs(seg))] > 0 else "trough"
        rows.append({"well": r.well, "td_relation": td_src, "corr": round(abs(corr), 3), "polarity_sign": sign,
                     "lag_ms": lag, "tie_twt": round(r.twt, 1), "corrected_twt": round(r.twt + lag, 1),
                     "event_at_top": ev, "log_samples": len(lg)})
    return pd.DataFrame(rows)


def summarise(st: pd.DataFrame, min_corr: float = 0.3) -> dict:
    good = st[st["corr"] >= min_corr]
    if good.empty:
        return {"wells": len(st), "usable": 0}
    sign_votes = good.polarity_sign.value_counts().to_dict()
    ev = good.event_at_top.value_counts().to_dict()
    return {"wells": len(st), "usable": len(good), "min_corr": min_corr,
            "polarity": "SEG normal (AI increase = peak)" if good.polarity_sign.median() > 0 else "reversed (AI increase = trough)",
            "polarity_votes": {("normal" if k > 0 else "reversed"): v for k, v in sign_votes.items()},
            "median_lag_ms": float(good.lag_ms.median()), "lag_spread_ms": [int(good.lag_ms.min()), int(good.lag_ms.max())],
            "event_at_top_votes": ev, "suggested_event": max(ev, key=ev.get)}


# ── optimised tie: global parameters + regularised stretch & squeeze ─────────
def _rotate(x: np.ndarray, deg: float) -> np.ndarray:
    """Constant phase rotation via the analytic signal."""
    from scipy.signal import hilbert
    if deg == 0:
        return x
    a = hilbert(x)
    return np.real(a * np.exp(1j * np.deg2rad(deg)))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def _well_series(cube: Cube, wells: Wells, logs: dict, r, exclude_tables=()):
    """Reflectivity on a 1 ms log-time axis and the seismic trace on the same axis."""
    lg = logs.get(r.well)
    if lg is None or not {"DT", "RHOB", "TVDSS"} <= set(lg.columns):
        return None
    lg = lg.dropna(subset=["DT", "RHOB", "TVDSS"]).copy()
    lg["TVDSS"] = lg["TVDSS"].abs()
    lg = lg[(lg.DT > 30) & (lg.DT < 250) & (lg.RHOB > 1.5) & (lg.RHOB < 3.2)].sort_values("TVDSS")
    if len(lg) < 100:
        return None
    td = _td_curve(wells, r.well, r.x, r.y, exclude_tables)
    if td is None:
        return None
    twt_log = np.interp(lg.TVDSS.values, td[0], td[1])
    ai = (304800.0 / lg.DT.values) * lg.RHOB.values
    o = np.argsort(twt_log); twt_log, ai = twt_log[o], ai[o]
    keep = np.r_[True, np.diff(twt_log) > 0]; twt_log, ai = twt_log[keep], ai[keep]
    t1 = np.arange(np.ceil(twt_log[0]), np.floor(twt_log[-1]), 1.0)
    if len(t1) < 80:
        return None
    ai1 = np.interp(t1, twt_log, ai)
    rc = np.r_[0.0, np.diff(ai1) / (ai1[1:] + ai1[:-1])]
    i, j = cube.index(r.il, r.xl)
    seis = np.interp(t1, cube.t_axis(), np.asarray(cube.data[i, j], float))
    return t1, rc, seis, td[2]


def optimise_tie(cube: Cube, wells: Wells, logs: dict, ties: pd.DataFrame, half_window_ms: float = 150.0,
                 max_shift_ms: float = 40.0, freq_bounds=(15.0, 45.0), stretch_max_ms: float = 8.0,
                 n_knots: int = 6, roughness_weight: float = 0.02, size_weight: float = 0.002,
                 seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per well with DT + RHOB:
    1. global search (differential evolution) over bulk shift, phase rotation and Ricker
       peak frequency maximising the correlation between synthetic and seismic in the window
       around the top;
    2. stretch & squeeze as a regularised optimisation: a smooth lag curve (cubic spline
       through n_knots, each within ±stretch_max_ms) maximising correlation − roughness −
       size penalties.
    Returns per-well results and the lag curves (for provenance / plots)."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import differential_evolution
    rows, curves = [], {}
    for r in ties[ties.in_cube & ties.twt.notna()].drop_duplicates("well").itertuples():
        s = _well_series(cube, wells, logs, r)
        if s is None:
            continue
        t1, rc, seis, td_src = s
        win = np.abs(t1 - r.twt) <= half_window_ms
        if win.sum() < 80:
            continue
        tw = t1[win]

        def synth(freq, phase):
            return _rotate(conv_same(rc, ricker(freq)), phase)

        def shifted(sy, lag):                    # synthetic at (t − lag): events move later by lag
            return np.interp(tw - lag, t1, sy)

        def f_global(x):
            shift, phase, freq = x
            return -abs(_ncc(seis[win], shifted(synth(freq, phase), shift)))

        g = differential_evolution(f_global, [(-max_shift_ms, max_shift_ms), (-90, 90), freq_bounds],
                                   seed=seed, tol=1e-6, maxiter=60, popsize=12, polish=True)
        shift, phase, freq = g.x
        sy = synth(freq, phase)
        c_bulk = _ncc(seis[win], shifted(sy, shift))
        sign = 1 if c_bulk > 0 else -1
        knots = np.linspace(tw[0], tw[-1], n_knots)

        def lag_curve(kv):
            return CubicSpline(knots, kv, bc_type="natural")(tw)

        def f_stretch(kv):
            lag = shift + lag_curve(kv)
            syw = np.interp(tw - lag, t1, sy)
            rough = np.mean(np.diff(kv) ** 2)
            return -sign * _ncc(seis[win], syw) + roughness_weight * rough + size_weight * np.mean(kv ** 2)

        h = differential_evolution(f_stretch, [(-stretch_max_ms, stretch_max_ms)] * n_knots, seed=seed,
                                   tol=1e-7, maxiter=80, popsize=10, polish=True)
        lag = shift + lag_curve(h.x)
        c_final = _ncc(seis[win], np.interp(tw - lag, t1, sy))
        lag_top = float(np.interp(r.twt, tw, lag))
        # event at the top in the data's polarity, after the tie
        syw = sign * np.interp(t1 - np.interp(t1, tw, lag), t1, sy)
        k = int(np.argmin(np.abs(t1 - (r.twt + lag_top))))
        seg = syw[max(0, k - 6):k + 7]
        ev = "peak" if seg[np.argmax(np.abs(seg))] > 0 else "trough"
        rows.append({"well": r.well, "td_relation": td_src, "bulk_shift_ms": round(float(shift), 1),
                     "phase_deg": round(float(phase), 1), "wavelet_hz": round(float(freq), 1),
                     "polarity": "normal" if sign > 0 else "reversed",
                     "corr_bulk": round(abs(c_bulk), 3), "corr_stretched": round(abs(c_final), 3),
                     "stretch_range_ms": [round(float((lag - shift).min()), 1), round(float((lag - shift).max()), 1)],
                     "lag_at_top_ms": round(lag_top, 1), "tie_twt": round(r.twt, 1),
                     "tied_twt": round(r.twt + lag_top, 1), "event_at_top": ev})
        curves[r.well] = {"t": tw[::4].round(1).tolist(), "lag": lag[::4].round(2).tolist()}
    return pd.DataFrame(rows), curves
