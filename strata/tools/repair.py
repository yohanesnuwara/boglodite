"""Deviation repair: find where a tracked horizon leaves its event and re-track the event's trend.

Idea (interpreter's habit): where a picked horizon suddenly jumps and the event it sits on loses
its amplitude, the picks have probably left the reflector. Look at the trusted picks just before
the jump (their waveform and amplitude = the "trend"), put a seed on the last trusted pick and
re-track the same event onward with dynamic programming (Viterbi), scoring candidates by
similarity to a template taken from the trusted picks — after the template DP tracker of
tmp/viterbi.py (anchors as hard constraints, waveform template, continuity penalty, max jump).

Input data only (seismic, fault map, tied well points). Every threshold is fixed a priori:

  deviation  : |Δt| > JUMP_MS between neighbouring picks, no fault-map trace within ±2 traces,
               trusted side median amplitude ≥ MIN_AMP × RMS, and the other side's median
               amplitude < DROP_RATIO × trusted amplitude (the event "fades" after the jump).
  re-track   : candidates = extrema of the horizon polarity within ±SEARCH_MS of the seed time;
               cost = W_WAVE·(1 − NCC to template) + W_AMP·min(|1 − amp/amp_ref|, 2)
                      + W_STEP·(Δt / dt)²,  |Δt| ≤ MAX_STEP_MS per trace.
  stop       : at the first fault-map trace; when the re-tracked event fades (amp < 0.3·amp_ref
               for 5 traces); or when it rejoins the original picks (≤ 4 ms for 3 traces).
  accept     : same event — mean amplitude within 0.6–1.6 × the trusted amplitude — and stronger
               than the original picks (≥ max(0.8·RMS, 1.5 × original)), mean NCC ≥ 0.7, ≥ 3 traces,
               and every tied well point in the span still honoured within 10 ms.
"""

from __future__ import annotations

import numpy as np

from strata.data import Cube
from strata.tools.horizon import Horizon, _extrema, _refine

TOOL_VERSION = "repair-0.1"
JUMP_MS, MIN_AMP, DROP_RATIO = 12.0, 1.0, 0.5
SEARCH_MS, MAX_STEP_MS, MAX_SPAN = 60.0, 4.0, 200
W_WAVE, W_AMP, W_STEP = 2.0, 1.0, 0.3
HALF_WAVE_MS, CONTEXT = 12.0, 8
AMP_BAND = (0.6, 1.6)


def _k(t, t0, dt):
    return int(round((t - t0) / dt))


def _wave(Dl, j, t, t0, dt, hw):
    k = _k(t, t0, dt)
    if k - hw < 0 or k + hw >= Dl.shape[1]:
        return None
    w = Dl[j, k - hw:k + hw + 1].astype(float)
    w = w - w.mean()
    n = np.linalg.norm(w)
    return w / n if n > 0 else None


def _fid(Dl, j, t, t0, dt, sign, rms):
    if not np.isfinite(t):
        return 0.0
    k = _k(t, t0, dt)
    return float(sign * Dl[j, k] / rms) if 0 <= k < Dl.shape[1] else 0.0


def _retrack(Dl, t0, dt, j0, seed_t, tpl, amp_ref, sign, rms, hw, n_end, anchors):
    """Template DP from the seed at trace j0 to n_end (exclusive). Returns times, amps, nccs for j0+1.."""
    max_step = MAX_STEP_MS
    lo_t, hi_t = seed_t - SEARCH_MS, seed_t + SEARCH_MS
    cand, unary, amps, nccs = [], [], [], []
    for j in range(j0 + 1, n_end):
        tr = Dl[j].astype(float)
        ks = _extrema(tr, sign)
        ks = ks[(ks >= _k(lo_t, t0, dt)) & (ks <= _k(hi_t, t0, dt)) & (ks - hw >= 0) & (ks + hw < len(tr))]
        ts = np.array([t0 + _refine(tr, int(k)) * dt for k in ks])
        a = np.array([sign * tr[k] / rms for k in ks])
        c = np.array([float(_wave(Dl, j, t, t0, dt, hw) @ tpl) for t in ts]) if len(ts) else np.array([])
        u = W_WAVE * (1 - c) + W_AMP * np.minimum(np.abs(1 - a / amp_ref), 2.0)   # same event = same amplitude
        if j in anchors and len(ts):
            u = u + np.where(np.abs(ts - anchors[j]) <= 10.0, 0.0, 1e3)
        cand.append(ts); unary.append(u); amps.append(a); nccs.append(c)
    if not cand or not len(cand[0]):
        return None
    prev_t, prev_c = np.array([seed_t]), np.array([0.0])
    backs, costs = [], []
    for q in range(len(cand)):
        ts = cand[q]
        if not len(ts):
            break
        d = ts[:, None] - prev_t[None, :]
        tot = np.where(np.abs(d) <= max_step, W_STEP * (d / dt) ** 2, np.inf) + prev_c[None, :]
        b = np.argmin(tot, axis=1)
        best = tot[np.arange(len(ts)), b]
        if not np.isfinite(best).any():
            break
        costs.append(best + unary[q]); backs.append(b)
        prev_t, prev_c = ts, costs[-1]
    if not costs:
        return None
    L = len(costs)
    j = int(np.argmin(costs[-1]))
    out_t, out_a, out_c = np.empty(L), np.empty(L), np.empty(L)
    for q in range(L - 1, -1, -1):
        out_t[q], out_a[q], out_c[q] = cand[q][j], amps[q][j], nccs[q][j]
        j = int(backs[q][j])
    return out_t, out_a, out_c


def repair_line(Dl: np.ndarray, t0: float, dt: float, picks: np.ndarray, sign: int, rms: float,
                fault: np.ndarray | None = None, anchors: dict[int, float] | None = None) -> tuple[np.ndarray, list[dict]]:
    """Repair deviations along one line in the forward direction. Dl (n, nt), picks (n,) ms."""
    n = len(picks)
    out = picks.astype(float).copy()
    fault = np.zeros(n, bool) if fault is None else np.asarray(fault, bool)
    anchors = anchors or {}
    hw = max(2, int(round(HALF_WAVE_MS / dt)))
    repairs = []
    j = CONTEXT
    while j < n - 2:
        a, b = out[j], out[j + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b - a) <= JUMP_MS or fault[max(0, j - 2):j + 4].any():
            j += 1; continue
        before = [_fid(Dl, q, out[q], t0, dt, sign, rms) for q in range(j - CONTEXT + 1, j + 1)]
        after = [_fid(Dl, q, out[q], t0, dt, sign, rms) for q in range(j + 1, min(n, j + 1 + CONTEXT))]
        amp_ref = float(np.median(before))
        if amp_ref < MIN_AMP or float(np.median(after)) >= DROP_RATIO * amp_ref:
            j += 1; continue
        ws = [w for q in range(j - CONTEXT + 1, j + 1)
              if np.isfinite(out[q]) and (w := _wave(Dl, q, out[q], t0, dt, hw)) is not None]
        if len(ws) < 3:
            j += 1; continue
        tpl = np.mean(ws, axis=0); tpl /= np.linalg.norm(tpl)
        end = j + 1
        while end < min(n, j + 1 + MAX_SPAN) and not fault[end]:
            end += 1
        r = _retrack(Dl, t0, dt, j, a, tpl, amp_ref, sign, rms, hw, end, anchors)
        if r is None:
            j += 1; continue
        rt, ra, rc = r
        L = len(rt)
        stop, weak, close = L, 0, 0                       # stop rules along the re-tracked path
        for q in range(L):
            jj = j + 1 + q
            weak = weak + 1 if ra[q] < 0.3 * amp_ref else 0
            close = close + 1 if (np.isfinite(out[jj]) and abs(rt[q] - out[jj]) <= 4.0) else 0
            if weak >= 5:
                stop = q - 4; break
            if close >= 3:
                stop = q - 2; break
        if stop < 3:
            j += 1; continue
        span = range(j + 1, j + 1 + stop)
        orig_amp = np.mean([_fid(Dl, q, out[q], t0, dt, sign, rms) for q in span])
        new_amp, new_ncc = float(ra[:stop].mean()), float(rc[:stop].mean())
        wells_ok = all(abs(rt[q - j - 1] - anchors[q]) <= 10.0 for q in span if q in anchors)
        same_event = AMP_BAND[0] * amp_ref <= new_amp <= AMP_BAND[1] * amp_ref
        if same_event and new_amp >= max(0.8, 1.5 * orig_amp) and new_ncc >= 0.7 and wells_ok:
            out[j + 1:j + 1 + stop] = rt[:stop]
            repairs.append({"seed_trace": j, "seed_twt": round(float(a), 1), "trusted_amp": round(amp_ref, 2), "first": j + 1, "last": j + stop,
                            "amp_before": round(float(orig_amp), 2), "amp_after": round(new_amp, 2),
                            "ncc": round(new_ncc, 2), "mean_shift_ms": round(float(np.nanmean(rt[:stop] - picks[span])), 1),
                            "times": rt[:stop].copy()})
            j = j + stop
        else:
            j += 1
    return out, repairs


def _line_candidates(Dl, t0, dt, picks, sign, rms, fault, anchors):
    """Repairs proposed from the ORIGINAL picks of one line, forward and backward (not applied)."""
    n = len(picks)
    _, r1 = repair_line(Dl, t0, dt, picks, sign, rms, fault, anchors)
    _, r2 = repair_line(Dl[::-1], t0, dt, picks[::-1].copy(), sign, rms, None if fault is None else fault[::-1],
                        {n - 1 - k: v for k, v in anchors.items()})
    for r in r2:
        r["seed_trace"], r["first"], r["last"] = n - 1 - r["seed_trace"], n - 1 - r["last"], n - 1 - r["first"]
        r["times"] = r["times"][::-1].copy()
    return r1 + r2


def _jumps(twt: np.ndarray, cells: tuple[np.ndarray, np.ndarray], jump_ms: float) -> int:
    """Number of 4-neighbour pairs touching `cells` whose time difference exceeds jump_ms."""
    ii, jj = cells
    ni, nj = twt.shape
    n = 0
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = ii + di, jj + dj
        ok = (a >= 0) & (a < ni) & (b >= 0) & (b < nj)
        d = np.abs(twt[ii[ok], jj[ok]] - twt[a[ok], b[ok]])
        n += int(np.sum(np.isfinite(d) & (d > jump_ms)))
    return n


def repair_horizon(cube: Cube, hz: Horizon, polarity: str = "trough", fault: np.ndarray | None = None,
                   anchors: list[tuple[float, float, float]] | None = None) -> tuple[Horizon, dict]:
    """Propose repairs along every inline and crossline from the ORIGINAL horizon (no cascading), then
    accept each one only if it makes the surface more continuous in 3D: the number of > JUMP_MS jumps
    between the repaired traces and all their neighbours (along and across the line) must drop.
    Returns the repaired horizon and a report with the seeds (line, trace, twt) of accepted repairs."""
    sign = 1 if polarity == "peak" else -1
    D = cube.data
    ni, nj, _ = D.shape
    A = {}
    for il, xl, t in anchors or []:
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        if 0 <= i < ni and 0 <= j < nj:
            A[(i, j)] = t
    orig = hz.twt.astype(float)
    cands = []
    for i in range(ni):
        for r in _line_candidates(np.asarray(D[i]), cube.t0, cube.dt, orig[i], sign, cube.rms,
                                  None if fault is None else fault[i], {j: t for (a, j), t in A.items() if a == i}):
            k = np.arange(r["first"], r["last"] + 1)
            cands.append((np.full(len(k), i), k, r, "inline"))
    for j in range(nj):
        for r in _line_candidates(np.asarray(D[:, j]), cube.t0, cube.dt, orig[:, j], sign, cube.rms,
                                  None if fault is None else fault[:, j], {i: t for (i, b), t in A.items() if b == j}):
            k = np.arange(r["first"], r["last"] + 1)
            cands.append((k, np.full(len(k), j), r, "crossline"))
    twt = orig.copy()
    accepted, rejected = [], 0
    for ii, jj, r, kind in sorted(cands, key=lambda c: -(c[2]["amp_after"] - c[2]["amp_before"])):
        before = _jumps(twt, (ii, jj), JUMP_MS)
        old = twt[ii, jj].copy()
        twt[ii, jj] = r["times"]
        after = _jumps(twt, (ii, jj), JUMP_MS)
        if after < before:
            line = int(cube.ilines[ii[0]]) if kind == "inline" else int(cube.xlines[jj[0]])
            seed = int(cube.xlines[r["seed_trace"]]) if kind == "inline" else int(cube.ilines[r["seed_trace"]])
            accepted.append({"direction": kind, "line": line, "seed_trace": seed, "seed_twt": r["seed_twt"],
                             "traces": len(ii), "jumps_before": before, "jumps_after": after,
                             **{k: r[k] for k in ("trusted_amp", "amp_before", "amp_after", "ncc", "mean_shift_ms")}})
        else:
            twt[ii, jj] = old
            rejected += 1
    changed = np.isfinite(twt) & np.isfinite(orig) & (np.abs(twt - orig) > 0.5)
    conf = np.where(changed, np.minimum(hz.conf, 0.8), hz.conf).astype(np.float32)

    def mean_amp(t):
        ok = np.isfinite(t)
        i2, j2 = np.nonzero(ok)
        kk = np.clip(np.round((t[ok] - cube.t0) / cube.dt).astype(int), 0, D.shape[2] - 1)
        return float(np.mean(sign * D[i2, j2, kk]) / cube.rms)
    rep = {"proposed": len(cands), "accepted": len(accepted), "rejected_not_more_continuous": rejected,
           "traces_changed": int(changed.sum()),
           "traces_changed_frac": round(float(changed.sum() / max(np.isfinite(twt).sum(), 1)), 4),
           "mean_event_amp_before": round(mean_amp(orig), 3), "mean_event_amp_after": round(mean_amp(twt), 3),
           "params": {"jump_ms": JUMP_MS, "min_amp": MIN_AMP, "drop_ratio": DROP_RATIO, "search_ms": SEARCH_MS,
                      "max_step_ms": MAX_STEP_MS, "amp_band": AMP_BAND, "accept": "3D jumps must decrease"}}
    return Horizon(twt.astype(np.float32), conf), {**rep, "seeds": accepted}


def consensus_seeds(cube: Cube, hz: Horizon, polarity: str = "trough", fault: np.ndarray | None = None,
                    anchors: list[tuple[float, float, float]] | None = None, agree_ms: float = 8.0,
                    every: int = 5) -> tuple[list[tuple[float, float, float]], np.ndarray, dict]:
    """Seeds for re-tracking deviated regions. Re-track proposals are made independently along
    inlines and along crosslines from the ORIGINAL horizon; a trace becomes a seed candidate only
    where both directions propose a repair and agree within agree_ms (two orthogonal re-tracks
    from different seeds landing on the same event). Returned seeds are subsampled every `every`
    traces in both directions. Also returns the consensus time map (NaN elsewhere)."""
    sign = 1 if polarity == "peak" else -1
    D = cube.data
    ni, nj, _ = D.shape
    A = {}
    for il, xl, t in anchors or []:
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        if 0 <= i < ni and 0 <= j < nj:
            A[(i, j)] = t
    orig = hz.twt.astype(float)
    P_in = np.full((ni, nj), np.nan)
    P_x = np.full((ni, nj), np.nan)
    n_in = n_x = 0
    for i in range(ni):
        for r in _line_candidates(np.asarray(D[i]), cube.t0, cube.dt, orig[i], sign, cube.rms,
                                  None if fault is None else fault[i], {j: t for (a, j), t in A.items() if a == i}):
            P_in[i, r["first"]:r["last"] + 1] = r["times"]; n_in += 1
    for j in range(nj):
        for r in _line_candidates(np.asarray(D[:, j]), cube.t0, cube.dt, orig[:, j], sign, cube.rms,
                                  None if fault is None else fault[:, j], {i: t for (i, b), t in A.items() if b == j}):
            P_x[r["first"]:r["last"] + 1, j] = r["times"]; n_x += 1
    both = np.isfinite(P_in) & np.isfinite(P_x)
    agree = both & (np.abs(P_in - P_x) <= agree_ms)
    C = np.where(agree, 0.5 * (P_in + P_x), np.nan)
    ii, jj = np.nonzero(agree & (np.arange(ni)[:, None] % every == 0) & (np.arange(nj)[None, :] % every == 0))
    seeds = [(float(cube.ilines[i]), float(cube.xlines[j]), float(C[i, j])) for i, j in zip(ii, jj)]
    info = {"inline_proposals": n_in, "crossline_proposals": n_x,
            "traces_proposed_inline": int(np.isfinite(P_in).sum()), "traces_proposed_crossline": int(np.isfinite(P_x).sum()),
            "traces_both_directions": int(both.sum()), "traces_agree": int(agree.sum()),
            "mean_shift_at_agreeing_ms": round(float(np.nanmean((C - orig)[agree])), 1) if agree.any() else None,
            "seeds": len(seeds), "agree_ms": agree_ms, "every": every}
    return seeds, C, info


def retrack_with_seeds(cube: Cube, hz: Horizon, wells: list[tuple[float, float, float]], polarity: str = "trough",
                       fault: np.ndarray | None = None, window_ms: float = 35.0, agree_ms: float = 8.0,
                       every: int = 5) -> tuple[Horizon, dict]:
    """Deviation repair end to end: consensus seeds (see consensus_seeds) are added to the tied well
    points as anchors, the guide surface is the current horizon with the consensus times put in
    (median-smoothed; gaps from the well-guide surface), and the global Viterbi tracker re-solves
    the whole horizon in 3D (inline pass then crossline pass, fault-aware)."""
    from scipy import ndimage
    from strata.tools.horizon import guide_surface, viterbi_track
    seeds, C, info = consensus_seeds(cube, hz, polarity, fault, wells, agree_ms, every)
    g = np.where(np.isfinite(C), C, hz.twt.astype(float))
    gw = guide_surface(cube, wells)
    g = np.where(np.isfinite(g), g, gw)
    g = ndimage.median_filter(g, size=5)
    new, rep = viterbi_track(cube, g, polarity, window_ms=window_ms, fault=fault, anchors=list(wells) + seeds,
                             fault_mode="skip" if fault is not None else "relax")
    rep["deviation_repair"] = info
    rep["seed_points"] = [{"il": a, "xl": b, "twt": round(t, 1)} for a, b, t in seeds]
    return new, rep
