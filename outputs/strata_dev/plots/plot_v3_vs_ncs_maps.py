"""Map view: Opus v3 (greedy, both verifiers pass) vs scripted NCS pipeline (reporting only)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H
c = load_cube(); rh = S.Reference().horizon; i0, j0 = int(c.ilines[0]), int(c.xlines[0])
T = np.full(c.shape[:2], np.nan); ii = (rh.il.values - i0).astype(int); jj = (rh.xl.values - j0).astype(int)
ok = (ii >= 0) & (ii < T.shape[0]) & (jj >= 0) & (jj < T.shape[1]); T[ii[ok], jj[ok]] = rh.twt.values[ok]
v3 = H.load("runs/20261109-200906-a55c/store/HorizonPicks/Not2_top/v3/horizon.npz")
ncs = H.load("runs/20261109-193635-2683/store/HorizonPicks/Not2_top/v1/horizon.npz")
rows = np.where(np.isfinite(T).any(1))[0]; cols = np.where(np.isfinite(T).any(0))[0]
r0, r1 = max(rows.min() - 5, 0), min(rows.max() + 5, T.shape[0]); c0, c1 = max(cols.min() - 5, 0), min(cols.max() + 5, T.shape[1])
ext = [j0 + c0, j0 + c1, i0 + r1, i0 + r0]; S_ = np.s_[r0:r1, c0:c1]
inside = np.isfinite(T)
def stats(h):
    e = (h - T)[inside]; f = np.isfinite(e)
    return f"MAE {np.mean(np.abs(e[f])):.1f} ms · ±15 ms: {np.mean(np.abs(e[f]) <= 15):.0%} · unpicked {np.mean(~f):.1%}"
fig, ax = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
vmin, vmax = np.nanpercentile(T, 1), np.nanpercentile(T, 99)
for a, (h, title) in zip(ax[0], ((T, "True Not 2 top (reference)"), (v3.twt, "Opus v3 — greedy, min_corr 0.92 (verifiers ✓✓)"),
                                 (ncs.twt, "Scripted — Viterbi + repair + NCS + repair"))):
    im = a.imshow(np.where(inside, h, np.nan)[S_], cmap="viridis_r", vmin=vmin, vmax=vmax, extent=ext, aspect="auto", interpolation="nearest")
    a.set_title(title + ("" if h is T else "\n" + stats(h))); a.set_xlabel("Crossline"); a.set_ylabel("Inline")
fig.colorbar(im, ax=ax[0], shrink=.8, label="TWT (ms)")
for a, (h, title) in zip(ax[1][:2], ((v3.twt, "Opus v3: agent − truth"), (ncs.twt, "Scripted NCS: agent − truth"))):
    e = np.where(inside, h - T, np.nan)
    im2 = a.imshow(e[S_], cmap="RdBu_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
    gap = inside & ~np.isfinite(h)
    a.imshow(np.where(gap, 1.0, np.nan)[S_], cmap="Greys", vmin=0, vmax=1.5, extent=ext, aspect="auto", interpolation="nearest")
    a.set_title(title + "  (grey = unpicked)"); a.set_xlabel("Crossline"); a.set_ylabel("Inline")
fig.colorbar(im2, ax=ax[1][:2], shrink=.8, label="agent − truth (ms)")
d = np.where(inside, np.abs(v3.twt - T) - np.abs(ncs.twt - T), np.nan)
im3 = ax[1][2].imshow(d[S_], cmap="PiYG_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
dd = d[np.isfinite(d)]
ax[1][2].set_title(f"|v3 error| − |NCS error|  (green = v3 better, pink = NCS better)\nv3 better on {np.mean(dd < -4):.0%}, NCS better on {np.mean(dd > 4):.0%} of the area")
ax[1][2].set_xlabel("Crossline"); ax[1][2].set_ylabel("Inline"); fig.colorbar(im3, ax=ax[1][2], shrink=.8, label="ms")
plt.savefig("outputs/strata_dev/v3_vs_ncs_maps.png", dpi=95); print("ok")
