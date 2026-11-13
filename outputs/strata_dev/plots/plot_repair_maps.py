"""Map of horizon error vs reference (plotting/reporting only) before and after deviation repair, plus seeds."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H

RUN = "20261108-191746-623a"
c = load_cube(); ref = S.Reference().horizon
i0, j0 = int(c.ilines[0]), int(c.xlines[0])
T = np.full(c.shape[:2], np.nan)
ii = (ref.il.values - i0).astype(int); jj = (ref.xl.values - j0).astype(int)
ok = (ii >= 0) & (ii < T.shape[0]) & (jj >= 0) & (jj < T.shape[1]); T[ii[ok], jj[ok]] = ref.twt.values[ok]
before = H.load(f"runs/{RUN}/store/HorizonPicks/Not2_top/v3/horizon.npz").twt
after = H.load(f"outputs/strata_dev/retracked_{RUN}_v3.npz").twt
seeds = json.load(open(f"outputs/strata_dev/retrack_seeds_{RUN}_v3.json"))
fm = np.load(f"runs/{RUN}/store/FaultMap/reservoir/v1/faults.npz")["mask"]
rows = np.where(np.isfinite(T).any(1))[0]; cols = np.where(np.isfinite(T).any(0))[0]
r0, r1, c0, c1 = max(rows.min() - 5, 0), min(rows.max() + 5, T.shape[0]), max(cols.min() - 5, 0), min(cols.max() + 5, T.shape[1])
ext = [j0 + c0, j0 + c1, i0 + r1, i0 + r0]
fig, ax = plt.subplots(1, 3, figsize=(19, 7.2), constrained_layout=True)
for a, (title, h) in zip(ax[:2], (("Before repair (scripted, 20.6 ms)", before), ("After deviation repair (19.4 ms)", after))):
    e = (h - T)[r0:r1, c0:c1]
    im = a.imshow(e, cmap="RdBu_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
    a.contour(np.arange(j0 + c0, j0 + c1), np.arange(i0 + r0, i0 + r1), fm[r0:r1, c0:c1].astype(float), [0.5], colors="k", linewidths=.4)
    good = np.mean(np.abs(e[np.isfinite(e)]) <= 15)
    a.set_title(f"{title}\nagent − truth; within ±15 ms: {good:.0%}"); a.set_xlabel("Crossline"); a.set_ylabel("Inline")
    a.axhline(1120, color="#ffd400", lw=1, ls="--")
ax[1].scatter([s["xl"] for s in seeds], [s["il"] for s in seeds], s=6, c="#ff2d2d", label=f"repair seeds ({len(seeds)})")
ax[1].legend(loc="lower left", fontsize=8)
fig.colorbar(im, ax=ax[:2], shrink=.8, label="agent − truth (ms)")
d = (np.abs(after - T) - np.abs(before - T))[r0:r1, c0:c1]
im2 = ax[2].imshow(d, cmap="PiYG_r", vmin=-40, vmax=40, extent=ext, aspect="auto", interpolation="nearest")
ax[2].set_title(f"Change in |error| (green = better)\nbetter {np.mean(d[np.isfinite(d)] < -4):.0%} · worse {np.mean(d[np.isfinite(d)] > 4):.0%} of reservoir area (|Δ| > 4 ms)")
ax[2].set_xlabel("Crossline"); ax[2].set_ylabel("Inline"); ax[2].axhline(1120, color="#ffd400", lw=1, ls="--")
fig.colorbar(im2, ax=ax[2], shrink=.8, label="|after| − |before| (ms)")
out = "outputs/strata_dev/repair_error_maps.png"; plt.savefig(out, dpi=105); print(out)
